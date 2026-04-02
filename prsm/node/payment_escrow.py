"""
Payment Escrow
==============

Escrow system for FTNS payments on compute jobs.

Flow:
1. Requester creates escrow by locking FTNS from their wallet
2. Providers execute the job
3. When consensus is reached, escrow distributes payment to winning provider(s)
4. If consensus fails or job times out, escrow refunds the requester

This ensures providers get paid for work and requesters only pay for
verified results.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from prsm.node.local_ledger import LocalLedger, Transaction, TransactionType

logger = logging.getLogger(__name__)


class EscrowStatus(str, Enum):
    PENDING = "pending"           # Waiting for results
    RELEASED = "released"        # Payment distributed
    REFUNDED = "refunded"        # Money returned to requester
    DISPUTED = "disputed"        # Under dispute resolution


@dataclass
class EscrowEntry:
    """A single escrow for a compute job."""
    escrow_id: str
    job_id: str
    requester_id: str
    amount: float
    status: EscrowStatus = EscrowStatus.PENDING
    provider_winner: Optional[str] = None  # Who earned the payment
    tx_lock: Optional[str] = None          # Transaction that locked the funds
    tx_release: Optional[str] = None       # Transaction that released payment
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PaymentEscrow:
    """Manages escrow accounts for compute job payments.

    The escrow system ensures:
    - Requesters pre-commit FTNS before jobs run
    - Providers are guaranteed payment if they deliver valid results
    - Failed jobs refund the requester
    - Disputed results can trigger partial refunds
    """

    def __init__(
        self,
        ledger: LocalLedger,
        node_id: str,
        broadcast_transaction: Optional[Callable] = None,
    ):
        self.ledger = ledger
        self.node_id = node_id
        self.broadcast_tx = broadcast_transaction  # async func(tx)
        self._escrows: Dict[str, EscrowEntry] = {}
        self._tasks: List[asyncio.Task] = []
        self._running = False

        # Timeout for unreleased escrows (default 1 hour)
        self.default_timeout = 3600.0

    async def create_escrow(
        self,
        job_id: str,
        amount: float,
        requester_id: Optional[str] = None,
    ) -> Optional[EscrowEntry]:
        """Lock FTNS in escrow for a compute job.

        Returns the escrow entry, or None if insufficient balance.
        """
        requester = requester_id or self.node_id
        balance = await self.ledger.get_balance(requester)

        if balance < amount:
            logger.warning(
                f"Escrow rejected: {requester[:12]}... has {balance:.6f} < {amount:.6f}"
            )
            return None

        # Create escrow record
        escrow = EscrowEntry(
            escrow_id=str(uuid.uuid4()),
            job_id=job_id,
            requester_id=requester,
            amount=amount,
        )
        self._escrows[escrow.escrow_id] = escrow

        # Lock funds: transfer from requester to escrow wallet
        escrow_wallet = f"escrow-{escrow.escrow_id}"
        try:
            tx = await self.ledger.transfer(
                from_wallet=requester,
                to_wallet=escrow_wallet,
                amount=amount,
                tx_type=TransactionType.COMPUTE_PAYMENT,
                description=f"Escrow for job {job_id[:8]}",
            )
            escrow.tx_lock = tx.tx_id
            logger.info(
                f"Escrow created: {escrow.escrow_id[:8]}... "
                f"locked {amount:.6f} FTNS for job {job_id[:8]}..."
            )
            # Broadcast escrow creation to network
            if self.broadcast_tx:
                try:
                    await self.broadcast_tx(tx)
                except Exception:
                    pass
            return escrow
        except ValueError as e:
            logger.warning(f"Escrow transfer failed: {e}")
            del self._escrows[escrow.escrow_id]
            return None

    async def release_escrow(
        self,
        job_id: str,
        provider_id: str,
        consensus_reached: bool = True,
        partial_amount: Optional[float] = None,
    ) -> Optional[Transaction]:
        """Release escrow payment to the winning provider.

        If consensus was reached, full payment goes to provider.
        If consensus failed but partial work was done, can specify partial_amount.
        """
        # Find the escrow for this job
        escrow = None
        for e in self._escrows.values():
            if e.job_id == job_id and e.status == EscrowStatus.PENDING:
                escrow = e
                break

        if not escrow:
            logger.warning(f"No pending escrow found for job {job_id[:8]}...")
            return None

        escrow_wallet = f"escrow-{escrow.escrow_id}"
        amount = partial_amount if partial_amount is not None else escrow.amount

        # Get escrow balance
        escrow_balance = await self.ledger.get_balance(escrow_wallet)
        if escrow_balance < amount:
            logger.warning(f"Escrow wallet has {escrow_balance:.6f}, trying to release {amount:.6f}")
            return None

        # Pay the provider
        try:
            tx = await self.ledger.transfer(
                from_wallet=escrow_wallet,
                to_wallet=provider_id,
                amount=amount,
                tx_type=TransactionType.COMPUTE_PAYMENT,
                description=f"Payment for job {job_id[:8]} (consensus={'yes' if consensus_reached else 'partial'})",
            )
            escrow.provider_winner = provider_id
            escrow.tx_release = tx.tx_id
            escrow.status = EscrowStatus.RELEASED
            escrow.completed_at = time.time()

            # Broadcast payment release to network
            if self.broadcast_tx:
                try:
                    await self.broadcast_tx(tx)
                except Exception:
                    pass

            # Refund remainder to requester if partial
            remainder = escrow_balance - amount
            if remainder > 0:
                refund_tx = await self.ledger.transfer(
                    from_wallet=escrow_wallet,
                    to_wallet=escrow.requester_id,
                    amount=remainder,
                    tx_type=TransactionType.TRANSFER,
                    description=f"Escrow refund for job {job_id[:8]}",
                )
                logger.info(
                    f"Refunded {remainder:.6f} FTNS to requester {escrow.requester_id[:12]}..."
                )

            logger.info(
                f"Escrow released: {amount:.6f} FTNS -> {provider_id[:12]}... "
                f"for job {job_id[:8]}..."
            )
            return tx
        except ValueError as e:
            logger.warning(f"Escrow release failed: {e}")
            return None

    async def refund_escrow(self, job_id: str, reason: str = "") -> bool:
        """Refund escrow to the requester (job failed or cancelled)."""
        escrow = None
        for e in self._escrows.values():
            if e.job_id == job_id and e.status == EscrowStatus.PENDING:
                escrow = e
                break

        if not escrow:
            return False

        escrow_wallet = f"escrow-{escrow.escrow_id}"
        balance = await self.ledger.get_balance(escrow_wallet)

        if balance > 0:
            try:
                await self.ledger.transfer(
                    from_wallet=escrow_wallet,
                    to_wallet=escrow.requester_id,
                    amount=balance,
                    tx_type=TransactionType.TRANSFER,
                    description=f"Escrow refund: {reason}",
                )
                logger.info(
                    f"Escrow refunded: {balance:.6f} FTNS -> {escrow.requester_id[:12]}... "
                    f"({reason})"
                )
            except ValueError:
                return False

        escrow.status = EscrowStatus.REFUNDED
        escrow.completed_at = time.time()
        return True

    async def cleanup_expired_escrows(self) -> int:
        """Refund any escrows that have exceeded the timeout."""
        now = time.time()
        cleaned = 0
        for escrow in list(self._escrows.values()):
            if (
                escrow.status == EscrowStatus.PENDING
                and now - escrow.created_at > self.default_timeout
            ):
                await self.refund_escrow(escrow.job_id, reason="Escrow timed out")
                cleaned += 1
        return cleaned

    async def periodic_cleanup(self) -> None:
        """Run cleanup every 10 minutes."""
        self._running = True
        while self._running:
            await asyncio.sleep(600)
            try:
                cleaned = await self.cleanup_expired_escrows()
                if cleaned:
                    logger.info(f"Cleaned up {cleaned} expired escrows")
            except Exception as e:
                logger.error(f"Escrow cleanup error: {e}")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    def get_escrow(self, job_id: str) -> Optional[EscrowEntry]:
        for e in self._escrows.values():
            if e.job_id == job_id:
                return e
        return None

    def get_stats(self) -> Dict[str, Any]:
        statuses = {}
        for e in self._escrows.values():
            statuses[e.status.value] = statuses.get(e.status.value, 0) + 1

        total_locked = sum(
            e.amount for e in self._escrows.values() if e.status == EscrowStatus.PENDING
        )

        return {
            "total_escrows": len(self._escrows),
            "by_status": statuses,
            "total_locked_ftns": round(total_locked, 6),
        }
