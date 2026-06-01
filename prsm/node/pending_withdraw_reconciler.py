"""Sprint 916 — pending-withdraw reconciler.

Closes the dead-end exposed by sp914: a withdraw debits the off-chain wallet
BEFORE broadcasting the on-chain ERC-20 transfer. sp914 correctly made a
broadcast-but-unconfirmed tx return ``status="pending"`` WITHOUT refunding (the
tx is in the mempool and will likely confirm — refunding then would double-pay).
But if that pending tx later REVERTS on-chain, nothing refunds the off-chain
debit → the user permanently loses the FTNS. The only prior reconciliation
(``OnChainFTNSLedger._reconcile_pending_transactions``) runs at startup and only
updates tx *status*; it takes no corrective action.

This module records each pending withdraw (``job_id → wallet_id, amount,
tx_hash``) in a small bounded, persisted store, and a background reconciler
polls the receipt:

  * confirmed → resolve, no refund (the on-chain transfer succeeded);
  * reverted  → refund the off-chain debit, then resolve;
  * unconfirmed → leave for the next tick.

The refund is IDEMPOTENT: it atomically claims ``withdraw-refund:{job_id}`` via
the local ledger's ``record_nonce`` (the sp898/sp911 primitive) BEFORE crediting,
so a reconciler restart / double-run — or a store that lost its resolved mark in
a crash — credits the wallet at most once.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Local import kept lazy/local where the enum is used to avoid a heavy import
# chain at module load; the tx_type string is stable.
_REFUND_TX_TYPE_VALUE = "bridge_withdraw"


@dataclass
class WithdrawIntent:
    """A pending withdraw whose on-chain tx may still revert."""
    job_id: str
    wallet_id: str
    amount: float
    to_addr: str
    tx_hash: str
    recorded_at: float = field(default_factory=time.time)
    resolved: bool = False
    outcome: Optional[str] = None   # "confirmed" | "refunded"


class PendingWithdrawStore:
    """Bounded, optionally-persisted record of pending withdraws.

    persist_dir=None → in-memory only (tests / ephemeral). Otherwise a single
    JSON file under persist_dir survives restart so a reconciler can refund a
    revert that lands while the daemon was down. Resolved entries are pruned
    once the store exceeds max_entries (sp897 unbounded-disk discipline).
    """

    _FILE = "pending_withdraws.json"

    def __init__(self, persist_dir: Optional[str] = None, max_entries: int = 10_000):
        self._max = max(1, int(max_entries))
        self._path: Optional[Path] = None
        if persist_dir:
            d = Path(persist_dir)
            d.mkdir(parents=True, exist_ok=True)
            self._path = d / self._FILE
        # job_id → WithdrawIntent, insertion-ordered.
        self._intents: Dict[str, WithdrawIntent] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for d in raw.get("intents", []):
                try:
                    self._intents[d["job_id"]] = WithdrawIntent(**d)
                except (TypeError, KeyError):
                    continue
        except (OSError, ValueError) as exc:
            logger.warning("PendingWithdrawStore: load failed (%s); starting empty", exc)

    def _save(self) -> None:
        if not self._path:
            return
        try:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"intents": [asdict(i) for i in self._intents.values()]}
            ))
            tmp.replace(self._path)   # atomic
        except OSError as exc:
            logger.warning("PendingWithdrawStore: save failed (%s)", exc)

    def _prune(self) -> None:
        if len(self._intents) <= self._max:
            return
        # Drop oldest RESOLVED entries first; never drop unresolved (those
        # still owe a reconciliation decision).
        resolved = [j for j, i in self._intents.items() if i.resolved]
        for job_id in resolved:
            if len(self._intents) <= self._max:
                break
            del self._intents[job_id]

    # ── api ──────────────────────────────────────────────────────
    def record(self, *, job_id: str, wallet_id: str, amount: float,
               to_addr: str, tx_hash: str) -> None:
        if job_id in self._intents:
            return   # idempotent — a withdraw records its intent once
        self._intents[job_id] = WithdrawIntent(
            job_id=job_id, wallet_id=wallet_id, amount=float(amount),
            to_addr=to_addr, tx_hash=tx_hash,
        )
        self._prune()
        self._save()

    def unresolved(self) -> List[WithdrawIntent]:
        return [i for i in self._intents.values() if not i.resolved]

    def all(self) -> List[WithdrawIntent]:
        return list(self._intents.values())

    def mark_resolved(self, job_id: str, outcome: str) -> None:
        intent = self._intents.get(job_id)
        if intent is None:
            return
        intent.resolved = True
        intent.outcome = outcome
        self._prune()
        self._save()


async def reconcile_pending_withdraws(
    store: PendingWithdrawStore,
    *,
    get_receipt_status: Callable[[str], Awaitable[str]],
    refund: Callable[[WithdrawIntent], Awaitable[bool]],
) -> Dict[str, int]:
    """Resolve every unresolved withdraw intent.

    ``get_receipt_status(tx_hash)`` → "confirmed" | "reverted" | "pending"
    (anything else is treated as still-pending — leave for the next tick).
    ``refund(intent)`` performs the idempotent off-chain refund.
    """
    confirmed = refunded = still_pending = 0
    for intent in list(store.unresolved()):
        try:
            status = await get_receipt_status(intent.tx_hash)
        except Exception as exc:  # noqa: BLE001 — transient RPC; retry next tick
            logger.debug("reconcile: receipt poll for %s failed: %s",
                         intent.job_id, exc)
            still_pending += 1
            continue
        if status == "confirmed":
            store.mark_resolved(intent.job_id, "confirmed")
            confirmed += 1
        elif status == "reverted":
            try:
                await refund(intent)
            except Exception as exc:  # noqa: BLE001 — leave unresolved, retry
                logger.error("reconcile: refund for %s FAILED (will retry): %s",
                             intent.job_id, exc)
                still_pending += 1
                continue
            store.mark_resolved(intent.job_id, "refunded")
            refunded += 1
        else:
            still_pending += 1
    if confirmed or refunded:
        logger.info(
            "pending-withdraw reconcile: %d confirmed, %d refunded, %d still pending",
            confirmed, refunded, still_pending,
        )
    return {"confirmed": confirmed, "refunded": refunded,
            "still_pending": still_pending}


def resolve_pending_withdraw_reconciler_config_from_env() -> tuple[bool, float]:
    """(enabled, interval_seconds). Enabled by DEFAULT — this is a money-safety
    net and is cheap when idle (it no-ops on an empty store). Disable with
    ``PRSM_PENDING_WITHDRAW_RECONCILER_ENABLED=0``. Interval defaults to 300s,
    clamped to a 60s floor.
    """
    enabled_raw = os.environ.get(
        "PRSM_PENDING_WITHDRAW_RECONCILER_ENABLED", "1",
    ).strip().lower()
    enabled = enabled_raw not in ("0", "false", "no", "off")

    interval = 300.0
    interval_raw = os.environ.get(
        "PRSM_PENDING_WITHDRAW_RECONCILER_INTERVAL_S", "",
    ).strip()
    if interval_raw:
        try:
            interval = float(interval_raw)
        except ValueError:
            logger.warning(
                "PRSM_PENDING_WITHDRAW_RECONCILER_INTERVAL_S=%r invalid; "
                "defaulting to 300s", interval_raw,
            )
    if interval < 60.0:
        interval = 60.0
    return enabled, interval


class PendingWithdrawReconciler:
    """Background worker that polls pending withdraw receipts and refunds
    reverts. Co-locates the on-chain receipt poll (via the FTNS ledger's web3)
    with the off-chain refund (via the local ledger), keeping the pure
    reconcile loop testable in isolation.
    """

    def __init__(
        self,
        *,
        store: PendingWithdrawStore,
        ftns_ledger,
        local_ledger,
        interval_seconds: float = 300.0,
    ):
        self._store = store
        self._ftns_ledger = ftns_ledger
        self._local_ledger = local_ledger
        self.interval_seconds = max(60.0, float(interval_seconds))
        self._running = False
        self.confirmed_total = 0
        self.refunded_total = 0

    async def _get_receipt_status(self, tx_hash: str) -> str:
        """Poll the chain for a single tx receipt → confirmed/reverted/pending."""
        w3 = getattr(self._ftns_ledger, "w3", None)
        if w3 is None or not tx_hash:
            return "pending"
        h = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash
        loop = asyncio.get_running_loop()
        receipt = await loop.run_in_executor(
            None, lambda: w3.eth.get_transaction_receipt(h),
        )
        if receipt is None:
            return "pending"
        return "confirmed" if receipt.get("status") == 1 else "reverted"

    async def _refund(self, intent: WithdrawIntent) -> bool:
        """Refund the off-chain debit, gated by an atomic per-job nonce claim
        so it credits at most once across restarts / double-runs. Returns True
        if it credited, False if the refund was already claimed."""
        nonce = f"withdraw-refund:{intent.job_id}"
        won = await self._local_ledger.record_nonce(nonce, "reconciler")
        if not won:
            logger.info("reconcile: refund for %s already claimed — skipping",
                        intent.job_id)
            return False
        # Resolve the tx_type enum lazily to avoid import-cycle surprises.
        from prsm.node.local_ledger import TransactionType
        await self._local_ledger.credit(
            wallet_id=intent.wallet_id,
            amount=intent.amount,
            tx_type=TransactionType.BRIDGE_WITHDRAW,
            description=(
                f"bridge withdraw REFUND (reconciler: on-chain tx reverted) "
                f"job={intent.job_id} tx={intent.tx_hash}"
            ),
        )
        logger.warning(
            "reconcile: REFUNDED %s FTNS to %s — withdraw %s reverted on-chain",
            intent.amount, intent.wallet_id, intent.job_id,
        )
        return True

    async def reconcile_once(self) -> Dict[str, int]:
        out = await reconcile_pending_withdraws(
            self._store,
            get_receipt_status=self._get_receipt_status,
            refund=self._refund,
        )
        self.confirmed_total += out["confirmed"]
        self.refunded_total += out["refunded"]
        return out

    async def run_forever(self) -> None:
        self._running = True
        logger.info("PendingWithdrawReconciler launched (interval=%.0fs)",
                    self.interval_seconds)
        while self._running:
            try:
                await asyncio.sleep(self.interval_seconds)
                if self._store.unresolved():
                    await self.reconcile_once()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 — never crash the daemon
                logger.warning("PendingWithdrawReconciler tick failed: %s", exc)

    async def stop(self) -> None:
        self._running = False
