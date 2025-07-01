"""
Production FTNS Ledger with Database Persistence
==============================================

Production-grade FTNS token ledger that addresses Gemini's critical audit finding:
"The core ledger for balances and transactions is an in-memory Python dictionary, 
which would be wiped on every server restart."

This implementation provides:
- PostgreSQL-backed persistent storage
- ACID transaction guarantees  
- Real balance and transaction tracking
- Blockchain oracle preparation
- High-performance database optimizations
- Comprehensive audit trails
- Real-time balance validation
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
import structlog

from sqlalchemy import text, select, insert, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from prsm.core.database_service import get_database_service
from prsm.core.models import FTNSTransaction, FTNSBalance, User
from prsm.core.config import get_settings

# Set precision for financial calculations
getcontext().prec = 28

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class TransactionRequest:
    """Request for FTNS transaction"""
    from_user_id: str
    to_user_id: Optional[str]  # None for mint/burn operations
    amount: Decimal
    transaction_type: str  # transfer, mint, burn, reward, fee
    description: str
    metadata: Dict[str, Any]
    reference_id: Optional[str] = None  # External reference (API call, marketplace transaction, etc.)


@dataclass
class BalanceSnapshot:
    """Point-in-time balance snapshot for analytics"""
    user_id: str
    balance: Decimal
    locked_balance: Decimal
    pending_incoming: Decimal
    pending_outgoing: Decimal
    last_transaction_id: Optional[str]
    timestamp: datetime


@dataclass
class LedgerStats:
    """Ledger statistics for monitoring"""
    total_supply: Decimal
    circulating_supply: Decimal
    locked_supply: Decimal
    total_accounts: int
    active_accounts_24h: int
    transaction_volume_24h: Decimal
    transaction_count_24h: int
    largest_balance: Decimal
    smallest_positive_balance: Decimal


class ProductionFTNSLedger:
    """
    Production-grade FTNS ledger with PostgreSQL persistence.
    
    Addresses Gemini audit findings:
    - Replaces in-memory Python dictionary with persistent database
    - Provides ACID transaction guarantees
    - Enables real marketplace value transfer
    - Supports blockchain oracle integration
    - Maintains comprehensive audit trails
    """
    
    def __init__(self):
        self.database_service = get_database_service()
        
        # Transaction lock to prevent race conditions
        self._transaction_locks: Dict[str, asyncio.Lock] = {}
        
        # Cache for frequently accessed balances (with TTL)
        self._balance_cache: Dict[str, Tuple[Decimal, datetime]] = {}
        self._cache_ttl = timedelta(seconds=30)  # 30 second cache
        
        # System accounts
        self.SYSTEM_MINT_ACCOUNT = "system_mint"
        self.SYSTEM_BURN_ACCOUNT = "system_burn"
        self.SYSTEM_REWARDS_ACCOUNT = "system_rewards"
        self.SYSTEM_FEES_ACCOUNT = "system_fees"
        
        # Initialize system accounts on startup
        asyncio.create_task(self._initialize_system_accounts())
    
    async def _initialize_system_accounts(self):
        """Initialize system accounts for mint/burn operations"""
        try:
            system_accounts = [
                self.SYSTEM_MINT_ACCOUNT,
                self.SYSTEM_BURN_ACCOUNT, 
                self.SYSTEM_REWARDS_ACCOUNT,
                self.SYSTEM_FEES_ACCOUNT
            ]
            
            for account_id in system_accounts:
                await self.ensure_account_exists(account_id, is_system_account=True)
                
            logger.info("✅ System accounts initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize system accounts: {e}")
    
    async def ensure_account_exists(self, user_id: str, is_system_account: bool = False) -> bool:
        """Ensure user account exists in the ledger"""
        try:
            async with self.database_service.get_session() as session:
                # Check if account exists
                query = text("""
                    SELECT COUNT(*) FROM ftns_balances 
                    WHERE user_id = :user_id
                """)
                
                result = await session.execute(query, {"user_id": user_id})
                exists = result.scalar() > 0
                
                if not exists:
                    # Create new account
                    insert_query = text("""
                        INSERT INTO ftns_balances 
                        (user_id, balance, locked_balance, total_earned, total_spent, 
                         account_type, created_at, updated_at)
                        VALUES 
                        (:user_id, 0, 0, 0, 0, :account_type, NOW(), NOW())
                    """)
                    
                    await session.execute(insert_query, {
                        "user_id": user_id,
                        "account_type": "system" if is_system_account else "user"
                    })
                    
                    await session.commit()
                    
                    logger.info(f"Created FTNS account for user: {user_id}")
                    return True
                
                return exists
                
        except Exception as e:
            logger.error(f"Failed to ensure account exists for {user_id}: {e}")
            return False
    
    async def get_balance(self, user_id: str) -> BalanceSnapshot:
        """Get current balance for user with caching"""
        try:
            # Check cache first
            if user_id in self._balance_cache:
                cached_balance, cached_time = self._balance_cache[user_id]
                if datetime.now(timezone.utc) - cached_time < self._cache_ttl:
                    return BalanceSnapshot(
                        user_id=user_id,
                        balance=cached_balance,
                        locked_balance=Decimal('0'),
                        pending_incoming=Decimal('0'),
                        pending_outgoing=Decimal('0'),
                        last_transaction_id=None,
                        timestamp=cached_time
                    )
            
            async with self.database_service.get_session() as session:
                # Get current balance
                balance_query = text("""
                    SELECT 
                        balance,
                        locked_balance,
                        total_earned,
                        total_spent,
                        updated_at
                    FROM ftns_balances 
                    WHERE user_id = :user_id
                """)
                
                balance_result = await session.execute(balance_query, {"user_id": user_id})
                balance_row = balance_result.fetchone()
                
                if not balance_row:
                    # Create account if it doesn't exist
                    await self.ensure_account_exists(user_id)
                    return BalanceSnapshot(
                        user_id=user_id,
                        balance=Decimal('0'),
                        locked_balance=Decimal('0'),
                        pending_incoming=Decimal('0'),
                        pending_outgoing=Decimal('0'),
                        last_transaction_id=None,
                        timestamp=datetime.now(timezone.utc)
                    )
                
                # Get pending transactions
                pending_query = text("""
                    SELECT 
                        SUM(CASE WHEN to_user_id = :user_id AND status = 'pending' THEN amount ELSE 0 END) as pending_incoming,
                        SUM(CASE WHEN from_user_id = :user_id AND status = 'pending' THEN amount ELSE 0 END) as pending_outgoing
                    FROM ftns_transactions 
                    WHERE (to_user_id = :user_id OR from_user_id = :user_id) 
                    AND status = 'pending'
                """)
                
                pending_result = await session.execute(pending_query, {"user_id": user_id})
                pending_row = pending_result.fetchone()
                
                # Get last transaction
                last_tx_query = text("""
                    SELECT id FROM ftns_transactions 
                    WHERE from_user_id = :user_id OR to_user_id = :user_id
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                
                last_tx_result = await session.execute(last_tx_query, {"user_id": user_id})
                last_tx_row = last_tx_result.fetchone()
                
                balance = Decimal(str(balance_row.balance))
                locked_balance = Decimal(str(balance_row.locked_balance))
                pending_incoming = Decimal(str(pending_row.pending_incoming or 0))
                pending_outgoing = Decimal(str(pending_row.pending_outgoing or 0))
                
                # Update cache
                self._balance_cache[user_id] = (balance, datetime.now(timezone.utc))
                
                return BalanceSnapshot(
                    user_id=user_id,
                    balance=balance,
                    locked_balance=locked_balance,
                    pending_incoming=pending_incoming,
                    pending_outgoing=pending_outgoing,
                    last_transaction_id=last_tx_row.id if last_tx_row else None,
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            logger.error(f"Failed to get balance for {user_id}: {e}")
            raise
    
    async def execute_transaction(self, request: TransactionRequest) -> str:
        """Execute FTNS transaction with ACID guarantees"""
        transaction_id = str(uuid4())
        
        try:
            # Get lock for transaction participants
            lock_key = self._get_transaction_lock_key(request.from_user_id, request.to_user_id)
            if lock_key not in self._transaction_locks:
                self._transaction_locks[lock_key] = asyncio.Lock()
            
            async with self._transaction_locks[lock_key]:
                async with self.database_service.get_session() as session:
                    # Start database transaction
                    async with session.begin():
                        # Validate transaction
                        await self._validate_transaction(session, request, transaction_id)
                        
                        # Create transaction record
                        await self._create_transaction_record(session, request, transaction_id)
                        
                        # Update balances
                        await self._update_balances(session, request, transaction_id)
                        
                        # Clear cache for affected users
                        self._clear_balance_cache([request.from_user_id, request.to_user_id])
                        
                        # Commit transaction
                        await session.commit()
                        
                        logger.info(f"✅ Transaction completed: {transaction_id}")
                        return transaction_id
                        
        except Exception as e:
            logger.error(f"Transaction failed {transaction_id}: {e}")
            # Mark transaction as failed
            await self._mark_transaction_failed(transaction_id, str(e))
            raise
    
    async def _validate_transaction(self, session: AsyncSession, request: TransactionRequest, transaction_id: str):
        """Validate transaction requirements"""
        
        # Ensure accounts exist
        await self.ensure_account_exists(request.from_user_id)
        if request.to_user_id:
            await self.ensure_account_exists(request.to_user_id)
        
        # Validate amount
        if request.amount <= 0:
            raise ValueError(f"Transaction amount must be positive: {request.amount}")
        
        # Check balance for non-mint operations
        if request.transaction_type != "mint":
            balance_query = text("""
                SELECT balance, locked_balance 
                FROM ftns_balances 
                WHERE user_id = :user_id
            """)
            
            result = await session.execute(balance_query, {"user_id": request.from_user_id})
            balance_row = result.fetchone()
            
            if not balance_row:
                raise ValueError(f"Account not found: {request.from_user_id}")
            
            available_balance = Decimal(str(balance_row.balance)) - Decimal(str(balance_row.locked_balance))
            
            if available_balance < request.amount:
                raise ValueError(f"Insufficient balance: {available_balance} < {request.amount}")
    
    async def _create_transaction_record(self, session: AsyncSession, request: TransactionRequest, transaction_id: str):
        """Create transaction record in database"""
        insert_query = text("""
            INSERT INTO ftns_transactions 
            (id, from_user_id, to_user_id, amount, transaction_type, description, 
             status, metadata, reference_id, created_at)
            VALUES 
            (:id, :from_user_id, :to_user_id, :amount, :transaction_type, :description,
             'completed', :metadata, :reference_id, NOW())
        """)
        
        await session.execute(insert_query, {
            "id": transaction_id,
            "from_user_id": request.from_user_id,
            "to_user_id": request.to_user_id,
            "amount": str(request.amount),
            "transaction_type": request.transaction_type,
            "description": request.description,
            "metadata": json.dumps(request.metadata),
            "reference_id": request.reference_id
        })
    
    async def _update_balances(self, session: AsyncSession, request: TransactionRequest, transaction_id: str):
        """Update user balances atomically"""
        
        # Update sender balance (except for mint operations)
        if request.transaction_type != "mint":
            sender_update = text("""
                UPDATE ftns_balances 
                SET 
                    balance = balance - :amount,
                    total_spent = total_spent + :amount,
                    last_transaction_id = :transaction_id,
                    updated_at = NOW()
                WHERE user_id = :user_id
            """)
            
            await session.execute(sender_update, {
                "amount": str(request.amount),
                "transaction_id": transaction_id,
                "user_id": request.from_user_id
            })
        
        # Update receiver balance (except for burn operations)
        if request.transaction_type != "burn" and request.to_user_id:
            receiver_update = text("""
                UPDATE ftns_balances 
                SET 
                    balance = balance + :amount,
                    total_earned = total_earned + :amount,
                    last_transaction_id = :transaction_id,
                    updated_at = NOW()
                WHERE user_id = :user_id
            """)
            
            await session.execute(receiver_update, {
                "amount": str(request.amount),
                "transaction_id": transaction_id,
                "user_id": request.to_user_id
            })
    
    async def _mark_transaction_failed(self, transaction_id: str, error_message: str):
        """Mark transaction as failed"""
        try:
            async with self.database_service.get_session() as session:
                update_query = text("""
                    UPDATE ftns_transactions 
                    SET status = 'failed', error_message = :error_message, updated_at = NOW()
                    WHERE id = :transaction_id
                """)
                
                await session.execute(update_query, {
                    "transaction_id": transaction_id,
                    "error_message": error_message
                })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to mark transaction as failed: {e}")
    
    def _get_transaction_lock_key(self, from_user_id: str, to_user_id: Optional[str]) -> str:
        """Generate lock key for transaction participants"""
        participants = [from_user_id]
        if to_user_id:
            participants.append(to_user_id)
        return ":".join(sorted(participants))
    
    def _clear_balance_cache(self, user_ids: List[Optional[str]]):
        """Clear balance cache for specified users"""
        for user_id in user_ids:
            if user_id and user_id in self._balance_cache:
                del self._balance_cache[user_id]
    
    # ==========================================
    # High-level Operations
    # ==========================================
    
    async def mint_tokens(self, to_user_id: str, amount: Decimal, description: str, metadata: Dict[str, Any] = None) -> str:
        """Mint new FTNS tokens to user account"""
        request = TransactionRequest(
            from_user_id=self.SYSTEM_MINT_ACCOUNT,
            to_user_id=to_user_id,
            amount=amount,
            transaction_type="mint",
            description=description,
            metadata=metadata or {}
        )
        
        return await self.execute_transaction(request)
    
    async def burn_tokens(self, from_user_id: str, amount: Decimal, description: str, metadata: Dict[str, Any] = None) -> str:
        """Burn FTNS tokens from user account"""
        request = TransactionRequest(
            from_user_id=from_user_id,
            to_user_id=self.SYSTEM_BURN_ACCOUNT,
            amount=amount,
            transaction_type="burn",
            description=description,
            metadata=metadata or {}
        )
        
        return await self.execute_transaction(request)
    
    async def transfer_tokens(
        self, 
        from_user_id: str, 
        to_user_id: str, 
        amount: Decimal, 
        description: str,
        metadata: Dict[str, Any] = None,
        reference_id: Optional[str] = None
    ) -> str:
        """Transfer FTNS tokens between users"""
        request = TransactionRequest(
            from_user_id=from_user_id,
            to_user_id=to_user_id,
            amount=amount,
            transaction_type="transfer",
            description=description,
            metadata=metadata or {},
            reference_id=reference_id
        )
        
        return await self.execute_transaction(request)
    
    async def reward_user(
        self, 
        user_id: str, 
        amount: Decimal, 
        description: str,
        reward_type: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Reward user with FTNS tokens"""
        request = TransactionRequest(
            from_user_id=self.SYSTEM_REWARDS_ACCOUNT,
            to_user_id=user_id,
            amount=amount,
            transaction_type="reward",
            description=description,
            metadata={
                "reward_type": reward_type,
                **(metadata or {})
            }
        )
        
        return await self.execute_transaction(request)
    
    async def charge_fee(
        self, 
        from_user_id: str, 
        amount: Decimal, 
        description: str,
        fee_type: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Charge fee in FTNS tokens"""
        request = TransactionRequest(
            from_user_id=from_user_id,
            to_user_id=self.SYSTEM_FEES_ACCOUNT,
            amount=amount,
            transaction_type="fee",
            description=description,
            metadata={
                "fee_type": fee_type,
                **(metadata or {})
            }
        )
        
        return await self.execute_transaction(request)
    
    # ==========================================
    # Analytics and Reporting
    # ==========================================
    
    async def get_ledger_stats(self) -> LedgerStats:
        """Get comprehensive ledger statistics"""
        try:
            async with self.database_service.get_session() as session:
                # Total supply calculation
                supply_query = text("""
                    SELECT 
                        SUM(balance) as total_supply,
                        SUM(CASE WHEN account_type = 'user' THEN balance ELSE 0 END) as circulating_supply,
                        SUM(locked_balance) as locked_supply,
                        COUNT(*) as total_accounts,
                        MAX(balance) as largest_balance,
                        MIN(CASE WHEN balance > 0 THEN balance ELSE NULL END) as smallest_positive_balance
                    FROM ftns_balances
                """)
                
                supply_result = await session.execute(supply_query)
                supply_row = supply_result.fetchone()
                
                # Activity statistics (24h)
                activity_query = text("""
                    SELECT 
                        COUNT(DISTINCT from_user_id) + COUNT(DISTINCT to_user_id) as active_accounts,
                        SUM(amount) as transaction_volume,
                        COUNT(*) as transaction_count
                    FROM ftns_transactions 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    AND status = 'completed'
                """)
                
                activity_result = await session.execute(activity_query)
                activity_row = activity_result.fetchone()
                
                return LedgerStats(
                    total_supply=Decimal(str(supply_row.total_supply or 0)),
                    circulating_supply=Decimal(str(supply_row.circulating_supply or 0)),
                    locked_supply=Decimal(str(supply_row.locked_supply or 0)),
                    total_accounts=supply_row.total_accounts or 0,
                    active_accounts_24h=activity_row.active_accounts or 0,
                    transaction_volume_24h=Decimal(str(activity_row.transaction_volume or 0)),
                    transaction_count_24h=activity_row.transaction_count or 0,
                    largest_balance=Decimal(str(supply_row.largest_balance or 0)),
                    smallest_positive_balance=Decimal(str(supply_row.smallest_positive_balance or 0))
                )
                
        except Exception as e:
            logger.error(f"Failed to get ledger stats: {e}")
            raise
    
    async def get_transaction_history(
        self, 
        user_id: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get transaction history for user"""
        try:
            async with self.database_service.get_session() as session:
                query = text("""
                    SELECT 
                        id,
                        from_user_id,
                        to_user_id,
                        amount,
                        transaction_type,
                        description,
                        status,
                        metadata,
                        reference_id,
                        created_at
                    FROM ftns_transactions 
                    WHERE from_user_id = :user_id OR to_user_id = :user_id
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """)
                
                result = await session.execute(query, {
                    "user_id": user_id,
                    "limit": limit,
                    "offset": offset
                })
                
                transactions = []
                for row in result.fetchall():
                    transactions.append({
                        "id": row.id,
                        "from_user_id": row.from_user_id,
                        "to_user_id": row.to_user_id,
                        "amount": str(row.amount),
                        "transaction_type": row.transaction_type,
                        "description": row.description,
                        "status": row.status,
                        "metadata": json.loads(row.metadata) if row.metadata else {},
                        "reference_id": row.reference_id,
                        "created_at": row.created_at.isoformat(),
                        "direction": "incoming" if row.to_user_id == user_id else "outgoing"
                    })
                
                return transactions
                
        except Exception as e:
            logger.error(f"Failed to get transaction history for {user_id}: {e}")
            raise
    
    async def validate_ledger_integrity(self) -> Dict[str, Any]:
        """Validate ledger integrity and consistency"""
        try:
            async with self.database_service.get_session() as session:
                # Check balance consistency
                balance_check = text("""
                    SELECT 
                        user_id,
                        balance,
                        total_earned,
                        total_spent,
                        (total_earned - total_spent) as calculated_balance
                    FROM ftns_balances 
                    WHERE ABS(balance - (total_earned - total_spent)) > 0.000001
                """)
                
                balance_result = await session.execute(balance_check)
                balance_inconsistencies = balance_result.fetchall()
                
                # Check transaction totals
                transaction_check = text("""
                    SELECT 
                        SUM(CASE WHEN transaction_type = 'mint' THEN amount ELSE 0 END) as total_minted,
                        SUM(CASE WHEN transaction_type = 'burn' THEN amount ELSE 0 END) as total_burned,
                        SUM(CASE WHEN transaction_type = 'transfer' THEN amount ELSE 0 END) as total_transferred
                    FROM ftns_transactions 
                    WHERE status = 'completed'
                """)
                
                tx_result = await session.execute(transaction_check)
                tx_row = tx_result.fetchone()
                
                # Get total supply
                supply_query = text("SELECT SUM(balance) as total_supply FROM ftns_balances")
                supply_result = await session.execute(supply_query)
                supply_row = supply_result.fetchone()
                
                return {
                    "integrity_status": "valid" if len(balance_inconsistencies) == 0 else "invalid",
                    "balance_inconsistencies": len(balance_inconsistencies),
                    "total_supply": str(supply_row.total_supply or 0),
                    "total_minted": str(tx_row.total_minted or 0),
                    "total_burned": str(tx_row.total_burned or 0),
                    "total_transferred": str(tx_row.total_transferred or 0),
                    "validation_timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to validate ledger integrity: {e}")
            raise
    
    async def prepare_blockchain_sync_data(self, since_timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Prepare data for blockchain oracle synchronization"""
        try:
            async with self.database_service.get_session() as session:
                # Get recent transactions for blockchain sync
                since_clause = "AND created_at > :since_timestamp" if since_timestamp else ""
                
                query = text(f"""
                    SELECT 
                        id,
                        from_user_id,
                        to_user_id,
                        amount,
                        transaction_type,
                        created_at,
                        reference_id
                    FROM ftns_transactions 
                    WHERE status = 'completed'
                    {since_clause}
                    ORDER BY created_at ASC
                """)
                
                params = {"since_timestamp": since_timestamp} if since_timestamp else {}
                result = await session.execute(query, params)
                
                transactions = []
                for row in result.fetchall():
                    transactions.append({
                        "id": row.id,
                        "from_user_id": row.from_user_id,
                        "to_user_id": row.to_user_id,
                        "amount": str(row.amount),
                        "transaction_type": row.transaction_type,
                        "timestamp": row.created_at.isoformat(),
                        "reference_id": row.reference_id
                    })
                
                # Get current balances for verification
                balance_query = text("""
                    SELECT user_id, balance 
                    FROM ftns_balances 
                    WHERE account_type = 'user' AND balance > 0
                """)
                
                balance_result = await session.execute(balance_query)
                balances = {
                    row.user_id: str(row.balance) 
                    for row in balance_result.fetchall()
                }
                
                return {
                    "transactions": transactions,
                    "balances": balances,
                    "sync_timestamp": datetime.now(timezone.utc).isoformat(),
                    "transaction_count": len(transactions)
                }
                
        except Exception as e:
            logger.error(f"Failed to prepare blockchain sync data: {e}")
            raise


# Global instance
_production_ledger = None

async def get_production_ledger() -> ProductionFTNSLedger:
    """Get the global production ledger instance"""
    global _production_ledger
    if _production_ledger is None:
        _production_ledger = ProductionFTNSLedger()
    return _production_ledger