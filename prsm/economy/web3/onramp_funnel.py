"""Sprint 857 — Coinbase Pay onramp conversion funnel tracker.

Tracks user-onboarding conversions WITHOUT depending on Coinbase
webhook delivery. The chain is the source of truth — every
``/wallet/onramp/execute`` call records the intent (user_id, address,
expected USD), and periodic balance sweeps via sp862's
WalletBalanceReader detect when USDC actually arrived.

This is intentionally architecturally cleaner than a webhook
receiver: it doesn't trust Coinbase to deliver webhooks reliably,
and works even if the webhook is silently dropped (sandbox tier
has known delivery gaps). Webhooks can be added later as an
optimization, but they're not load-bearing.

State machine:

  INTENT_RECORDED  → user clicked through to Coinbase Pay
  ↓
  PENDING_SETTLEMENT  → balance check shows 0; user may have
                        abandoned OR Coinbase is still settling
  ↓
  CONFIRMED  → USDC delta >= expected_usdc * 0.95 (1.5% Coinbase
               fee + 1% buffer)
  ↓
  EXPIRED  → no USDC arrival after 24h; presumed abandoned

Each intent record persists to ~/.prsm/onramp-funnel/<intent_id>.json
so cross-restart funnel observability survives daemon restarts (same
pattern as sp860's WaaS + KYC stores).
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Conversion is "confirmed" once on-chain USDC delta >=
# expected * THRESHOLD. Coinbase typically takes ~1.5% in fees;
# we allow a 5% slop so transient pricing fluctuations don't
# leave conversions stuck in PENDING_SETTLEMENT.
_CONVERSION_THRESHOLD = 0.95

# 24h after intent recording, mark as EXPIRED if no USDC arrived.
_EXPIRY_SECONDS = 86_400

# Statuses (canonical).
STATUS_INTENT_RECORDED = "INTENT_RECORDED"
STATUS_PENDING_SETTLEMENT = "PENDING_SETTLEMENT"
STATUS_CONFIRMED = "CONFIRMED"
STATUS_EXPIRED = "EXPIRED"


@dataclass
class OnrampIntent:
    intent_id: str
    user_id: Optional[str]
    destination_address: str
    expected_usd: float
    session_token: str  # the Coinbase session token (audit signal)
    created_at: float
    status: str = STATUS_INTENT_RECORDED
    confirmed_at: float = 0.0
    usdc_received: float = 0.0
    usdc_received_units: int = 0
    expired_at: float = 0.0
    # Sp871 — when funnel sweep CONFIRMS USDC arrival, the
    # auto-swap orchestrator stores the prepared Aerodrome swap
    # envelope here (router address + amountIn + amountOutMin +
    # deadline + routes). User retrieves via
    # GET /wallet/onramp/funnel/{intent_id} to execute the swap
    # once the pool ceremony closes. None until CONFIRMED.
    swap_envelope: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OnrampIntent":
        return cls(
            intent_id=d["intent_id"],
            user_id=d.get("user_id"),
            destination_address=d["destination_address"],
            expected_usd=d["expected_usd"],
            session_token=d.get("session_token", ""),
            created_at=d.get("created_at", 0.0),
            status=d.get("status", STATUS_INTENT_RECORDED),
            confirmed_at=d.get("confirmed_at", 0.0),
            usdc_received=d.get("usdc_received", 0.0),
            usdc_received_units=d.get("usdc_received_units", 0),
            expired_at=d.get("expired_at", 0.0),
            swap_envelope=d.get("swap_envelope"),
        )


class OnrampFunnel:
    """Persistent funnel tracker. Default persist dir
    ``~/.prsm/onramp-funnel/``. Opt out via env=":memory:".
    """

    def __init__(
        self,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        self._records: Dict[str, OnrampIntent] = {}
        if persist_dir is None:
            persist_raw = os.environ.get("PRSM_ONRAMP_FUNNEL_DIR")
            if persist_raw == ":memory:":
                self._persist_dir = None
            elif persist_raw:
                self._persist_dir = Path(persist_raw)
            else:
                self._persist_dir = (
                    Path.home() / ".prsm" / "onramp-funnel"
                )
        else:
            self._persist_dir = persist_dir
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self._persist_dir is None:
            return
        for path in self._persist_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                rec = OnrampIntent.from_dict(d)
                self._records[rec.intent_id] = rec
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                logger.warning(
                    "OnrampFunnel: bad record %s: %s", path, exc,
                )

    def _persist(self, rec: OnrampIntent) -> None:
        if self._persist_dir is None:
            return
        path = self._persist_dir / f"{rec.intent_id}.json"
        path.write_text(json.dumps(rec.to_dict(), indent=2))

    def record_intent(
        self,
        *,
        user_id: Optional[str],
        destination_address: str,
        expected_usd: float,
        session_token: str,
    ) -> OnrampIntent:
        """Called from ``/wallet/onramp/execute`` after a session
        token is minted. Records the funnel-entry event."""
        intent_id = f"onramp_{secrets.token_hex(8)}"
        rec = OnrampIntent(
            intent_id=intent_id,
            user_id=user_id,
            destination_address=destination_address,
            expected_usd=expected_usd,
            session_token=session_token,
            created_at=time.time(),
        )
        self._records[intent_id] = rec
        self._persist(rec)
        return rec

    def list_intents(
        self, *, status: Optional[str] = None,
    ) -> List[OnrampIntent]:
        """All intents, newest first. Optionally filter by status."""
        out = list(self._records.values())
        if status:
            out = [r for r in out if r.status == status]
        out.sort(key=lambda r: r.created_at, reverse=True)
        return out

    def get_intent(self, intent_id: str) -> Optional[OnrampIntent]:
        return self._records.get(intent_id)

    def sweep(
        self, *, balance_reader: Any,
        on_confirmed: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Periodic sweep: check on-chain USDC balance against
        each open intent. Transitions:

          INTENT_RECORDED  → PENDING_SETTLEMENT  (after first sweep
                              even if no USDC yet)
          PENDING_SETTLEMENT → CONFIRMED  (USDC arrived above
                                          threshold)
          INTENT_RECORDED / PENDING_SETTLEMENT → EXPIRED
                            (no USDC after 24h)

        Returns summary {checked, confirmed_new, expired_new}.
        """
        now = time.time()
        checked = 0
        confirmed_new = 0
        expired_new = 0
        for rec in list(self._records.values()):
            if rec.status in {STATUS_CONFIRMED, STATUS_EXPIRED}:
                continue
            checked += 1
            try:
                bal = balance_reader.get_balances(
                    rec.destination_address,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OnrampFunnel sweep: balance read failed "
                    "for intent %s: %s", rec.intent_id, exc,
                )
                continue

            threshold_usd = rec.expected_usd * _CONVERSION_THRESHOLD
            if bal.usdc >= threshold_usd:
                rec.status = STATUS_CONFIRMED
                rec.confirmed_at = now
                rec.usdc_received = bal.usdc
                rec.usdc_received_units = bal.usdc_units
                confirmed_new += 1
                # Sp891 — PERSIST the CONFIRMED transition BEFORE
                # firing the callback. The callback (sp885 compliance
                # record + sp874 webhook) must fire AT MOST ONCE per
                # intent. Persisting first means any concurrent
                # sweeper or post-crash reload observes CONFIRMED and
                # skips this intent (guard above) rather than
                # re-confirming + double-firing — which would
                # double-count settled volume against the sp884 tier
                # limit and duplicate the downstream webhook.
                self._persist(rec)
                # Sp871 — fire optional callback on confirm.
                # Fail-soft: a misbehaving callback (e.g., pool
                # not yet seeded → swap-envelope build raises)
                # must NOT undo the CONFIRMED transition.
                if on_confirmed is not None:
                    try:
                        on_confirmed(rec)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "OnrampFunnel sweep: on_confirmed "
                            "callback raised for intent %s: %s",
                            rec.intent_id, exc,
                        )
                    else:
                        # Persist again so any callback-applied
                        # mutations (e.g. sp871 swap_envelope) land.
                        self._persist(rec)
                # Already persisted above; skip the tail persist.
                continue
            elif (
                now - rec.created_at > _EXPIRY_SECONDS
            ):
                rec.status = STATUS_EXPIRED
                rec.expired_at = now
                expired_new += 1
            else:
                # First sweep after recording — leave at
                # PENDING_SETTLEMENT for operator visibility.
                if rec.status == STATUS_INTENT_RECORDED:
                    rec.status = STATUS_PENDING_SETTLEMENT
            self._persist(rec)
        return {
            "checked": checked,
            "confirmed_new": confirmed_new,
            "expired_new": expired_new,
        }

    def summary(self) -> Dict[str, Any]:
        """Aggregate funnel counts for the dashboard."""
        counts = {
            STATUS_INTENT_RECORDED: 0,
            STATUS_PENDING_SETTLEMENT: 0,
            STATUS_CONFIRMED: 0,
            STATUS_EXPIRED: 0,
        }
        total_expected = 0.0
        total_confirmed = 0.0
        for r in self._records.values():
            counts[r.status] = counts.get(r.status, 0) + 1
            total_expected += r.expected_usd
            if r.status == STATUS_CONFIRMED:
                total_confirmed += r.usdc_received
        total = len(self._records)
        rate = (
            counts[STATUS_CONFIRMED] / total
            if total > 0 else 0.0
        )
        return {
            "total_intents": total,
            "status_counts": counts,
            "total_expected_usd": total_expected,
            "total_confirmed_usdc": total_confirmed,
            "conversion_rate": rate,
        }
