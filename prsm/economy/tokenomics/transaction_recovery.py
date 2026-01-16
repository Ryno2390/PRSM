"""
FTNS Transaction Recovery Service
=================================

Handles failed transaction recovery, orphaned lock cleanup, and ledger reconciliation.

This service provides automated recovery for:
1. Stuck transactions (pending for too long)
2. Orphaned locks (locks without corresponding transactions)
3. Balance discrepancies (earned/spent vs actual balance)
4. Negative balance detection and correction

Usage:
    from prsm.economy.tokenomics.transaction_recovery import TransactionRecoveryService

    recovery = TransactionRecoveryService()
    await recovery.initialize()

    # Run recovery
    result = await recovery.recover_stuck_transactions()

    # Run full reconciliation
    health = await recovery.reconcile_ledger()
"""

import asyncio
import structlog
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class RecoveryResult:
    """Result of a recovery operation"""
    success: bool
    recovered_count: int = 0
    failed_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ReconciliationResult:
    """Result of ledger reconciliation"""
    is_healthy: bool
    total_supply: Decimal
    circulating_supply: Decimal
    supply_discrepancy: Decimal
    user_discrepancies: int
    negative_balances: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TransactionRecoveryService:
    """
    Automated recovery service for FTNS transactions.

    Responsibilities:
    1. Detect and recover stuck transactions
    2. Clean up orphaned locks
    3. Reconcile balance discrepancies
    4. Generate audit reports
    5. Handle transaction rollbacks
    """

    def __init__(self, database_service=None):
        """
        Initialize recovery service.

        Args:
            database_service: Database service for session management.
                            If None, will be initialized lazily.
        """
        self._db_service = database_service
        self._initialized = False

        # Configuration
        self.stuck_threshold = timedelta(minutes=5)
        self.lock_timeout = timedelta(minutes=10)
        self.max_recovery_batch = 100
        self.auto_fix_minor_discrepancies = True
        self.minor_discrepancy_threshold = Decimal("0.00000001")

    async def initialize(self):
        """Initialize the service and database connection."""
        if self._initialized:
            return

        if self._db_service is None:
            from prsm.core.database_service import get_database_service
            self._db_service = get_database_service()

        self._initialized = True
        logger.info("TransactionRecoveryService initialized")

    async def _get_session(self) -> AsyncSession:
        """Get database session."""
        if not self._initialized:
            await self.initialize()
        return self._db_service.get_session()

    # =========================================================================
    # Transaction Recovery
    # =========================================================================

    async def recover_stuck_transactions(self) -> RecoveryResult:
        """
        Find and recover transactions stuck in 'pending' state.

        Recovery strategy:
        1. Identify transactions pending > threshold
        2. Check if balances were actually modified
        3. Either complete or rollback based on state
        """
        recovered = []
        failed = []

        async with await self._get_session() as session:
            try:
                # Find stuck transactions
                stuck_query = text("""
                    SELECT
                        id, from_user_id, to_user_id, amount, transaction_type,
                        balance_before_sender, balance_after_sender,
                        balance_before_receiver, balance_after_receiver,
                        created_at
                    FROM ftns_transactions
                    WHERE status = 'pending'
                    AND created_at < NOW() - INTERVAL ':threshold_minutes minutes'
                    ORDER BY created_at ASC
                    LIMIT :max_batch
                    FOR UPDATE SKIP LOCKED
                """)

                result = await session.execute(stuck_query, {
                    "threshold_minutes": int(self.stuck_threshold.total_seconds() / 60),
                    "max_batch": self.max_recovery_batch
                })

                rows = result.fetchall()
                logger.info(f"Found {len(rows)} stuck transactions")

                for row in rows:
                    try:
                        action = await self._recover_single_transaction(session, row)
                        recovered.append({
                            "transaction_id": str(row.id),
                            "action": action,
                            "from_user": row.from_user_id,
                            "amount": str(row.amount)
                        })
                    except Exception as e:
                        failed.append({
                            "transaction_id": str(row.id),
                            "error": str(e)
                        })
                        logger.error(f"Failed to recover transaction {row.id}: {e}")

                await session.commit()

            except Exception as e:
                await session.rollback()
                logger.error(f"Transaction recovery failed: {e}")
                return RecoveryResult(
                    success=False,
                    details={"error": str(e)}
                )

        result = RecoveryResult(
            success=True,
            recovered_count=len(recovered),
            failed_count=len(failed),
            details={
                "recovered_transactions": recovered,
                "failed_transactions": failed
            }
        )

        if recovered or failed:
            logger.info(
                "Transaction recovery completed",
                recovered=len(recovered),
                failed=len(failed)
            )

        return result

    async def _recover_single_transaction(
        self,
        session: AsyncSession,
        tx_row
    ) -> str:
        """
        Attempt recovery of a single stuck transaction.

        Returns:
            Action taken: 'completed', 'rolled_back', or 'marked_failed'
        """
        # Get current sender balance to check if deduction happened
        if tx_row.from_user_id:
            balance_query = text("""
                SELECT balance FROM ftns_balances WHERE user_id = :user_id
            """)
            balance_result = await session.execute(
                balance_query, {"user_id": tx_row.from_user_id}
            )
            current_balance = balance_result.scalar()

            if current_balance is None:
                # Sender doesn't exist - mark as failed
                await self._mark_transaction_failed(
                    session, tx_row.id, "Sender account not found"
                )
                return "marked_failed"

            # Check if deduction completed
            if tx_row.balance_after_sender is not None:
                current_balance = Decimal(str(current_balance))
                expected_after = Decimal(str(tx_row.balance_after_sender))

                if abs(current_balance - expected_after) < self.minor_discrepancy_threshold:
                    # Deduction completed, check receiver and mark complete
                    await self._complete_stuck_transaction(session, tx_row)
                    return "completed"
                else:
                    # Inconsistent state - rollback
                    await self._rollback_transaction(session, tx_row)
                    return "rolled_back"

        # Transaction never started - mark as failed
        await self._mark_transaction_failed(
            session, tx_row.id, "Transaction timeout - never started"
        )
        return "marked_failed"

    async def _mark_transaction_failed(
        self,
        session: AsyncSession,
        transaction_id: str,
        error_message: str
    ):
        """Mark transaction as failed."""
        await session.execute(text("""
            UPDATE ftns_transactions
            SET status = 'failed',
                error_message = :error_message,
                updated_at = NOW()
            WHERE id = :id
        """), {
            "id": str(transaction_id),
            "error_message": error_message
        })

    async def _complete_stuck_transaction(
        self,
        session: AsyncSession,
        tx_row
    ):
        """Complete a stuck transaction that was partially processed."""
        # Mark as completed
        await session.execute(text("""
            UPDATE ftns_transactions
            SET status = 'completed',
                updated_at = NOW()
            WHERE id = :id
        """), {"id": str(tx_row.id)})

        logger.info(f"Completed stuck transaction {tx_row.id}")

    async def _rollback_transaction(self, session: AsyncSession, tx_row):
        """Rollback a partially completed transaction."""
        amount = Decimal(str(tx_row.amount))

        # Restore sender balance
        if tx_row.from_user_id and tx_row.balance_before_sender is not None:
            before_balance = Decimal(str(tx_row.balance_before_sender))
            await session.execute(text("""
                UPDATE ftns_balances
                SET balance = :balance,
                    total_spent = total_spent - :amount,
                    version = version + 1,
                    updated_at = NOW()
                WHERE user_id = :user_id
            """), {
                "balance": str(before_balance),
                "amount": str(amount),
                "user_id": tx_row.from_user_id
            })

        # Reverse receiver credit if applicable
        if tx_row.to_user_id and tx_row.balance_before_receiver is not None:
            before_balance = Decimal(str(tx_row.balance_before_receiver))
            await session.execute(text("""
                UPDATE ftns_balances
                SET balance = :balance,
                    total_earned = total_earned - :amount,
                    version = version + 1,
                    updated_at = NOW()
                WHERE user_id = :user_id
            """), {
                "balance": str(before_balance),
                "amount": str(amount),
                "user_id": tx_row.to_user_id
            })

        # Mark transaction as rolled back
        await session.execute(text("""
            UPDATE ftns_transactions
            SET status = 'rolled_back',
                error_message = 'Automatic rollback due to timeout and inconsistent state',
                updated_at = NOW()
            WHERE id = :id
        """), {"id": str(tx_row.id)})

        logger.warning(f"Transaction {tx_row.id} rolled back")

    # =========================================================================
    # Lock Cleanup
    # =========================================================================

    async def cleanup_orphaned_locks(self) -> RecoveryResult:
        """
        Clean up orphaned transaction locks.

        Orphaned locks can occur when:
        - Process crashes during transaction
        - Database connection lost
        - Application restart during transaction
        """
        cleaned = 0

        async with await self._get_session() as session:
            try:
                # Delete expired locks
                result = await session.execute(text("""
                    DELETE FROM ftns_transaction_locks
                    WHERE expires_at < NOW()
                    RETURNING lock_key
                """))

                deleted_rows = result.fetchall()
                cleaned = len(deleted_rows)

                await session.commit()

                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} orphaned locks")

            except Exception as e:
                await session.rollback()
                logger.error(f"Lock cleanup failed: {e}")
                return RecoveryResult(
                    success=False,
                    details={"error": str(e)}
                )

        return RecoveryResult(
            success=True,
            recovered_count=cleaned,
            details={"orphaned_locks_removed": cleaned}
        )

    # =========================================================================
    # Ledger Reconciliation
    # =========================================================================

    async def reconcile_ledger(self) -> ReconciliationResult:
        """
        Full ledger reconciliation.

        Verifies that:
        1. Sum of all balances equals (minted - burned)
        2. Each user's balance = earned - spent
        3. No negative balances exist
        """
        async with await self._get_session() as session:
            try:
                # Calculate expected total from transactions
                totals_query = text("""
                    SELECT
                        COALESCE(SUM(CASE WHEN transaction_type = 'mint' THEN amount ELSE 0 END), 0) as total_minted,
                        COALESCE(SUM(CASE WHEN transaction_type = 'burn' THEN amount ELSE 0 END), 0) as total_burned
                    FROM ftns_transactions
                    WHERE status = 'completed'
                """)

                totals_result = await session.execute(totals_query)
                totals = totals_result.fetchone()

                total_minted = Decimal(str(totals.total_minted or 0))
                total_burned = Decimal(str(totals.total_burned or 0))
                expected_supply = total_minted - total_burned

                # Calculate actual total from balances
                actual_query = text("""
                    SELECT COALESCE(SUM(balance), 0) as total
                    FROM ftns_balances
                """)
                actual_result = await session.execute(actual_query)
                actual_supply = Decimal(str(actual_result.scalar() or 0))

                # Calculate circulating (non-system accounts)
                circulating_query = text("""
                    SELECT COALESCE(SUM(balance), 0) as total
                    FROM ftns_balances
                    WHERE account_type = 'user'
                """)
                circulating_result = await session.execute(circulating_query)
                circulating_supply = Decimal(str(circulating_result.scalar() or 0))

                supply_discrepancy = actual_supply - expected_supply

                # Find individual user discrepancies
                discrepancy_query = text("""
                    SELECT
                        user_id,
                        balance as recorded_balance,
                        total_earned - total_spent as calculated_balance,
                        balance - (total_earned - total_spent) as discrepancy
                    FROM ftns_balances
                    WHERE ABS(balance - (total_earned - total_spent)) > 0.00000001
                """)

                discrepancy_result = await session.execute(discrepancy_query)
                discrepancies = discrepancy_result.fetchall()

                # Check for negative balances
                negative_query = text("""
                    SELECT user_id, balance
                    FROM ftns_balances
                    WHERE balance < 0
                """)

                negative_result = await session.execute(negative_query)
                negative_balances = negative_result.fetchall()

                # Build result
                is_healthy = (
                    abs(supply_discrepancy) < self.minor_discrepancy_threshold and
                    len(discrepancies) == 0 and
                    len(negative_balances) == 0
                )

                details = {
                    "total_minted": str(total_minted),
                    "total_burned": str(total_burned),
                    "user_discrepancies": [
                        {
                            "user_id": d.user_id,
                            "recorded": str(d.recorded_balance),
                            "calculated": str(d.calculated_balance),
                            "discrepancy": str(d.discrepancy)
                        }
                        for d in discrepancies
                    ],
                    "negative_balance_users": [
                        {"user_id": n.user_id, "balance": str(n.balance)}
                        for n in negative_balances
                    ]
                }

                result = ReconciliationResult(
                    is_healthy=is_healthy,
                    total_supply=actual_supply,
                    circulating_supply=circulating_supply,
                    supply_discrepancy=supply_discrepancy,
                    user_discrepancies=len(discrepancies),
                    negative_balances=len(negative_balances),
                    details=details
                )

                if not is_healthy:
                    logger.warning(
                        "Ledger reconciliation found issues",
                        supply_discrepancy=str(supply_discrepancy),
                        user_discrepancies=len(discrepancies),
                        negative_balances=len(negative_balances)
                    )
                else:
                    logger.info("Ledger reconciliation passed - ledger is healthy")

                return result

            except Exception as e:
                logger.error(f"Ledger reconciliation failed: {e}")
                return ReconciliationResult(
                    is_healthy=False,
                    total_supply=Decimal("0"),
                    circulating_supply=Decimal("0"),
                    supply_discrepancy=Decimal("0"),
                    user_discrepancies=-1,
                    negative_balances=-1,
                    details={"error": str(e)}
                )

    async def fix_user_balance(self, user_id: str, reason: str = "Reconciliation fix") -> bool:
        """
        Recalculate and fix a user's balance based on transaction history.

        Args:
            user_id: User to fix
            reason: Reason for the fix (for audit log)

        Returns:
            True if fix was applied
        """
        async with await self._get_session() as session:
            try:
                # Calculate correct values from transactions
                calc_query = text("""
                    SELECT
                        COALESCE(SUM(CASE
                            WHEN to_user_id = :user_id AND transaction_type IN ('mint', 'transfer', 'reward')
                            THEN amount ELSE 0 END), 0) as total_earned,
                        COALESCE(SUM(CASE
                            WHEN from_user_id = :user_id AND transaction_type IN ('burn', 'transfer', 'fee', 'deduction')
                            THEN amount ELSE 0 END), 0) as total_spent
                    FROM ftns_transactions
                    WHERE status = 'completed'
                    AND (from_user_id = :user_id OR to_user_id = :user_id)
                """)

                calc_result = await session.execute(calc_query, {"user_id": user_id})
                calc = calc_result.fetchone()

                total_earned = Decimal(str(calc.total_earned or 0))
                total_spent = Decimal(str(calc.total_spent or 0))
                correct_balance = total_earned - total_spent

                # Update the balance
                await session.execute(text("""
                    UPDATE ftns_balances
                    SET balance = :balance,
                        total_earned = :total_earned,
                        total_spent = :total_spent,
                        version = version + 1,
                        updated_at = NOW()
                    WHERE user_id = :user_id
                """), {
                    "balance": str(correct_balance),
                    "total_earned": str(total_earned),
                    "total_spent": str(total_spent),
                    "user_id": user_id
                })

                await session.commit()

                logger.info(
                    f"Fixed balance for user {user_id}",
                    new_balance=str(correct_balance),
                    reason=reason
                )

                return True

            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to fix balance for {user_id}: {e}")
                return False

    # =========================================================================
    # Background Job
    # =========================================================================

    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run all maintenance tasks.

        Returns:
            Summary of maintenance results
        """
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tasks": {}
        }

        # 1. Recover stuck transactions
        tx_result = await self.recover_stuck_transactions()
        results["tasks"]["stuck_transactions"] = {
            "success": tx_result.success,
            "recovered": tx_result.recovered_count,
            "failed": tx_result.failed_count
        }

        # 2. Clean up orphaned locks
        lock_result = await self.cleanup_orphaned_locks()
        results["tasks"]["orphaned_locks"] = {
            "success": lock_result.success,
            "cleaned": lock_result.recovered_count
        }

        # 3. Reconcile ledger
        recon_result = await self.reconcile_ledger()
        results["tasks"]["reconciliation"] = {
            "healthy": recon_result.is_healthy,
            "supply_discrepancy": str(recon_result.supply_discrepancy),
            "user_discrepancies": recon_result.user_discrepancies,
            "negative_balances": recon_result.negative_balances
        }

        results["overall_healthy"] = (
            tx_result.success and
            lock_result.success and
            recon_result.is_healthy
        )

        logger.info("Maintenance completed", **results)

        return results


# Background job runner
async def start_recovery_background_job(interval_seconds: int = 300):
    """
    Start background job for periodic transaction recovery.

    Args:
        interval_seconds: How often to run recovery (default 5 minutes)
    """
    recovery_service = TransactionRecoveryService()
    await recovery_service.initialize()

    logger.info(f"Starting recovery background job (interval: {interval_seconds}s)")

    while True:
        try:
            results = await recovery_service.run_maintenance()

            if not results["overall_healthy"]:
                logger.warning("Maintenance detected issues", results=results)

        except Exception as e:
            logger.error(f"Recovery background job failed: {e}")

        await asyncio.sleep(interval_seconds)


# Global instance
_recovery_service: Optional[TransactionRecoveryService] = None


async def get_recovery_service() -> TransactionRecoveryService:
    """Get or create global recovery service instance."""
    global _recovery_service
    if _recovery_service is None:
        _recovery_service = TransactionRecoveryService()
        await _recovery_service.initialize()
    return _recovery_service
