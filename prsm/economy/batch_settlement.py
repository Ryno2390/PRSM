"""
Batch Settlement Manager
========================

Queues on-chain FTNS transfers and settles them in batches instead of
broadcasting every individual transaction to Base mainnet.

Why batch?
  - Base L2 costs ~$0.001-0.01 per tx.  At 1,000 jobs/day that's $1-10/day.
  - Netting opposing transfers (A→B 5, B→A 3 = settle 2) cuts volume further.
  - Single-tx settlement amortizes gas across many local operations.
  - The local DAG ledger remains the source of truth for instant balance checks.

Architecture:
  Local DAG Ledger  (instant, free, authoritative)
       |
       v
  BatchSettlementManager._queue   (pending on-chain transfers)
       |
       v  [periodic flush OR manual trigger OR threshold]
  OnChainFTNSLedger.transfer()    (real Base mainnet ERC-20 transfer)

Settlement modes:
  1. PERIODIC  — flush every N seconds (default: 600s / 10 min)
  2. THRESHOLD — flush when pending value exceeds N FTNS (default: 1.0)
  3. MANUAL    — flush on explicit API call (prsm ftns settle)
  4. ON_WITHDRAW — flush only when a user withdraws to external wallet
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SettlementMode(str, Enum):
    """When to flush the settlement queue to on-chain."""
    PERIODIC = "periodic"         # Flush every N seconds
    THRESHOLD = "threshold"       # Flush when pending value exceeds limit
    MANUAL = "manual"             # Only flush on explicit trigger
    ON_WITHDRAW = "on_withdraw"   # Only when tokens leave the network


@dataclass
class PendingTransfer:
    """A queued on-chain transfer waiting for batch settlement."""
    tx_id: str
    from_wallet: str
    to_wallet: str            # resolved 0x address
    amount: float
    job_id: str
    queued_at: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class SettlementResult:
    """Outcome of a batch settlement flush."""
    settled_count: int = 0
    total_amount: float = 0.0
    net_transfers: int = 0     # after netting
    tx_hashes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class BatchSettlementManager:
    """Queues on-chain FTNS transfers and settles in batches.

    Drop-in replacement for direct _on_chain_ftns_transfer calls.
    The local DAG ledger has already committed — this only controls
    WHEN the on-chain mirror broadcast happens.
    """

    def __init__(
        self,
        ftns_ledger,                       # OnChainFTNSLedger instance
        node_id: str,
        connected_address: Optional[str] = None,
        mode: SettlementMode = SettlementMode.PERIODIC,
        flush_interval: float = 600.0,     # 10 minutes
        flush_threshold: float = 1.0,      # 1.0 FTNS
        max_queue_size: int = 1000,
    ):
        self._ftns_ledger = ftns_ledger
        self._node_id = node_id
        self._connected_address = connected_address
        self.mode = mode
        self.flush_interval = flush_interval
        self.flush_threshold = flush_threshold
        self.max_queue_size = max_queue_size

        # Transfer queue
        self._queue: List[PendingTransfer] = []
        self._lock = asyncio.Lock()

        # Deduplication
        self._settled_ids: Set[str] = set()

        # Stats
        self._total_settled = 0
        self._total_gas_saved = 0  # number of transfers avoided by netting
        self._last_flush_at: float = 0.0
        self._settlement_history: List[SettlementResult] = []

        # Background task
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the periodic flush task (if mode is PERIODIC)."""
        if self._running:
            return
        self._running = True
        if self.mode == SettlementMode.PERIODIC:
            self._flush_task = asyncio.create_task(self._periodic_flush_loop())
            logger.info(
                f"BatchSettlement started: mode={self.mode.value}, "
                f"interval={self.flush_interval}s, threshold={self.flush_threshold} FTNS"
            )
        else:
            logger.info(f"BatchSettlement started: mode={self.mode.value}")

    async def stop(self) -> None:
        """Stop the flush task and settle any remaining queue."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush on shutdown
        if self._queue:
            logger.info(f"BatchSettlement stopping: flushing {len(self._queue)} pending transfers")
            await self.flush()

    # ── Enqueue ────────────────────────────────────────────────

    async def enqueue(self, transaction) -> bool:
        """Queue a transaction for batch settlement.

        This is the drop-in replacement for _on_chain_ftns_transfer.
        Called by PaymentEscrow.broadcast_tx after local ledger commit.

        Returns True if queued, False if skipped (dedup, wrong type, etc.)
        """
        if not self._ftns_ledger or not self._ftns_ledger._is_initialized:
            return False

        # Extract fields
        tx_id = getattr(transaction, "tx_id", "")
        from_wallet = getattr(transaction, "from_wallet", "")
        to_wallet = getattr(transaction, "to_wallet", "")
        amount = float(getattr(transaction, "amount", 0))

        if amount <= 0:
            return False

        # Dedup
        if tx_id in self._settled_ids:
            return False

        # Resolve to_wallet to 0x address
        target_address = self._resolve_address(to_wallet)
        if not target_address:
            logger.debug(
                f"BatchSettlement: skipping non-chain wallet {to_wallet[:20]}…"
            )
            return False

        # Queue it
        pending = PendingTransfer(
            tx_id=tx_id,
            from_wallet=from_wallet,
            to_wallet=target_address,
            amount=amount,
            job_id=tx_id,
            description=getattr(transaction, "description", ""),
        )

        async with self._lock:
            self._queue.append(pending)
            self._settled_ids.add(tx_id)

        logger.debug(
            f"BatchSettlement: queued {amount:.6f} FTNS → {target_address[:12]}… "
            f"(queue: {len(self._queue)})"
        )

        # Auto-flush on threshold
        if self.mode in (SettlementMode.PERIODIC, SettlementMode.THRESHOLD):
            pending_total = sum(p.amount for p in self._queue)
            if pending_total >= self.flush_threshold:
                logger.info(
                    f"BatchSettlement: threshold reached ({pending_total:.4f} >= "
                    f"{self.flush_threshold}), flushing"
                )
                asyncio.create_task(self.flush())

        # Auto-flush on queue size
        if len(self._queue) >= self.max_queue_size:
            logger.info(f"BatchSettlement: max queue size reached, flushing")
            asyncio.create_task(self.flush())

        return True

    # ── Settlement / Flush ─────────────────────────────────────

    async def flush(self) -> SettlementResult:
        """Settle all pending transfers in a batch.

        Nets opposing transfers to minimize on-chain transactions:
          A→B 5.0, B→A 3.0 → net: A→B 2.0 (1 tx instead of 2)
        """
        start = time.time()
        result = SettlementResult()

        async with self._lock:
            if not self._queue:
                return result

            pending = list(self._queue)
            self._queue.clear()

        result.settled_count = len(pending)
        result.total_amount = sum(p.amount for p in pending)

        # Net transfers by (from_addr, to_addr) pair
        net_amounts = self._net_transfers(pending)
        result.net_transfers = len(net_amounts)
        self._total_gas_saved += result.settled_count - result.net_transfers

        logger.info(
            f"BatchSettlement: flushing {result.settled_count} transfers "
            f"→ {result.net_transfers} net on-chain txs "
            f"(saved {result.settled_count - result.net_transfers} gas txs)"
        )

        # Execute net transfers
        for (from_addr, to_addr), net_amount in net_amounts.items():
            if net_amount <= 0:
                continue
            try:
                tx_record = await self._ftns_ledger.transfer(
                    job_id=f"batch-{int(time.time())}",
                    to_address=to_addr,
                    amount_ftns=net_amount,
                )
                if tx_record and tx_record.status == "confirmed":
                    result.tx_hashes.append(tx_record.tx_hash)
                    logger.info(
                        f"BatchSettlement: {net_amount:.6f} FTNS → {to_addr[:12]}… "
                        f"confirmed (tx: {tx_record.tx_hash[:16]}…)"
                    )
                elif tx_record and tx_record.status == "rejected":
                    result.errors.append(
                        f"Rejected: {net_amount:.6f} → {to_addr[:12]}"
                    )
                else:
                    result.errors.append(
                        f"No receipt: {net_amount:.6f} → {to_addr[:12]}"
                    )
            except Exception as e:
                result.errors.append(f"{to_addr[:12]}: {e}")
                logger.error(f"BatchSettlement: transfer to {to_addr[:12]}… failed: {e}")

        result.duration_seconds = time.time() - start
        self._last_flush_at = time.time()
        self._total_settled += result.settled_count
        self._settlement_history.append(result)

        # Keep only last 100 settlement records
        if len(self._settlement_history) > 100:
            self._settlement_history = self._settlement_history[-100:]

        return result

    # ── Netting ────────────────────────────────────────────────

    def _net_transfers(
        self, pending: List[PendingTransfer]
    ) -> Dict[Tuple[str, str], float]:
        """Net opposing transfers to minimize on-chain transactions.

        For self-compute (provider == requester), all transfers to the node's
        own on-chain address are aggregated into one transfer.

        For multi-node, opposing transfers between the same pair are netted:
          A→B 5.0 and B→A 3.0 becomes A→B 2.0
        """
        # Aggregate by to_address (most common case: all transfers
        # go from the node's wallet to the node's wallet for self-compute,
        # or to specific provider addresses)
        aggregated: Dict[str, float] = defaultdict(float)
        from_addr = self._connected_address or ""

        for p in pending:
            aggregated[p.to_wallet] += p.amount

        # Convert to (from, to) → amount format
        net: Dict[Tuple[str, str], float] = {}
        for to_addr, amount in aggregated.items():
            if amount > 0:
                net[(from_addr, to_addr)] = amount

        return net

    # ── Address Resolution ─────────────────────────────────────

    def _resolve_address(self, wallet_id: str) -> Optional[str]:
        """Resolve a wallet ID to an on-chain 0x address.

        Returns None if the wallet is internal-only (escrow, named wallets).
        """
        if wallet_id.startswith("0x") and len(wallet_id) >= 40:
            return wallet_id
        if wallet_id == self._node_id and self._connected_address:
            return self._connected_address
        # Internal wallets (escrow-xxx, system, etc.) → skip
        return None

    # ── Background Loop ────────────────────────────────────────

    async def _periodic_flush_loop(self) -> None:
        """Background task: flush the queue periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                if self._queue:
                    await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"BatchSettlement: periodic flush error: {e}")

    # ── Stats & Monitoring ─────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return settlement queue stats for monitoring."""
        pending_total = sum(p.amount for p in self._queue)
        return {
            "mode": self.mode.value,
            "queue_size": len(self._queue),
            "pending_amount": round(pending_total, 6),
            "flush_interval": self.flush_interval,
            "flush_threshold": self.flush_threshold,
            "total_settled": self._total_settled,
            "gas_txs_saved": self._total_gas_saved,
            "last_flush_at": self._last_flush_at,
            "last_flush_ago": (
                round(time.time() - self._last_flush_at, 1)
                if self._last_flush_at else None
            ),
            "settlement_history_count": len(self._settlement_history),
        }

    def get_pending(self) -> List[Dict[str, Any]]:
        """Return pending transfers for inspection."""
        return [
            {
                "tx_id": p.tx_id[:12],
                "to": p.to_wallet[:12] + "…",
                "amount": round(p.amount, 6),
                "queued_at": p.queued_at,
                "age_seconds": round(time.time() - p.queued_at, 1),
            }
            for p in self._queue
        ]

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent settlement history."""
        return [
            {
                "settled_count": r.settled_count,
                "total_amount": round(r.total_amount, 6),
                "net_transfers": r.net_transfers,
                "tx_hashes": [h[:16] + "…" for h in r.tx_hashes],
                "errors": r.errors,
                "duration_seconds": round(r.duration_seconds, 2),
            }
            for r in self._settlement_history[-limit:]
        ]
