#!/usr/bin/env python3
"""
FTNS (Fine-Tuning Network Service) Token Management Service

⚠️ DEPRECATION NOTICE ⚠️
========================
This in-memory FTNSService is DEPRECATED and should NOT be used for production.

Race Condition Vulnerabilities:
- Uses in-memory Python dictionaries (no persistence)
- No transaction isolation between operations
- No locking mechanism for concurrent access
- No idempotency protection

Replacement Options:
1. AtomicFTNSService (prsm.economy.tokenomics.atomic_ftns_service)
   - SELECT FOR UPDATE row-level locking
   - Optimistic concurrency control via version column
   - Idempotency key support
   
2. FTNSQueries (prsm.core.database)
   - execute_atomic_transfer() for transfers
   - execute_atomic_deduct() for deductions
   - Uses PostgreSQL stored procedures for atomicity

Migration Guide:
    # OLD (deprecated, vulnerable to race conditions):
    from prsm.economy.tokenomics.ftns_service import get_ftns_service
    service = get_ftns_service()
    service.deduct_tokens(user_id, amount)
    
    # NEW (recommended, atomic with race condition protection):
    from prsm.core.database import FTNSQueries
    result = await FTNSQueries.execute_atomic_deduct(
        user_id=user_id,
        amount=amount,
        idempotency_key=f"deduct:{user_id}:{context_id}",
        description="Token usage"
    )

This module will be removed in a future version.
========================

Original Purpose:
- Token distribution for AI training contributions
- Reward calculation for model performance improvements
- Collaborative reasoning incentive management
- Training data contribution rewards
"""

import structlog
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum

logger = structlog.get_logger(__name__)

# Module-level constants for external imports
INITIAL_BALANCE = Decimal('1000.0')  # Welcome grant for new users
MIN_TRANSACTION_AMOUNT = Decimal('0.01')  # Minimum FTNS per transaction
MAX_TRANSACTION_AMOUNT = Decimal('1000000.0')  # Maximum single transaction
DEFAULT_REWARD_MULTIPLIER = Decimal('1.0')
STAKING_REWARD_RATE = Decimal('0.05')  # 5% annual staking rewards

# Export these constants
__all__ = [
    'FTNSService',
    'FTNSBalance',
    'FTNSTransaction',
    'FTNSTransactionType',
    'LaunchGuardrails',
    'get_ftns_service',
    'award_training_tokens',
    'get_user_ftns_balance',
    'ftns_service',
    'INITIAL_BALANCE',
    'MIN_TRANSACTION_AMOUNT',
    'MAX_TRANSACTION_AMOUNT',
    'DEFAULT_REWARD_MULTIPLIER',
    'STAKING_REWARD_RATE',
]


class FTNSTransactionType(Enum):
    """Types of FTNS token transactions"""
    TRAINING_REWARD = "training_reward"
    MODEL_IMPROVEMENT = "model_improvement"
    DATA_CONTRIBUTION = "data_contribution"
    COLLABORATIVE_REASONING = "collaborative_reasoning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    PIPELINE_EXECUTION = "pipeline_execution"
    SYSTEM_USAGE = "system_usage"


@dataclass
class FTNSBalance:
    """FTNS token balance for a user"""
    user_id: str
    balance: Decimal
    last_updated: datetime
    total_earned: Decimal = Decimal('0')
    total_spent: Decimal = Decimal('0')


@dataclass
class FTNSTransaction:
    """FTNS token transaction record"""
    transaction_id: str
    user_id: str
    amount: Decimal
    transaction_type: FTNSTransactionType
    description: str
    timestamp: datetime
    metadata: Dict[str, Any]
    balance_after: Decimal


@dataclass
class LaunchGuardrails:
    """Capped Mainnet: Limits exposure during the first 90 days"""
    start_time: datetime
    global_stake_cap: Decimal = Decimal("1000000.0") # 1M FTNS total
    user_stake_cap: Decimal = Decimal("10000.0")    # 10k FTNS per user
    
    def get_current_caps(self) -> Tuple[Decimal, Decimal]:
        """Progressive Cap Schedule: Increases every 30 days"""
        days_active = (datetime.now(timezone.utc) - self.start_time).days
        
        if days_active < 30:
            return self.global_stake_cap, self.user_stake_cap
        elif days_active < 60:
            return self.global_stake_cap * 5, self.user_stake_cap * 5
        elif days_active < 90:
            return self.global_stake_cap * 10, self.user_stake_cap * 10
        
        return Decimal("Infinity"), Decimal("Infinity")

import warnings

class FTNSService:
    """
    FTNS Token Management Service
    
    .. deprecated::
        This service is DEPRECATED due to race condition vulnerabilities.
        Use AtomicFTNSService instead for all new code.
        
        Migration:
            from prsm.economy.tokenomics.atomic_ftns_service import get_atomic_ftns_service
            ftns = await get_atomic_ftns_service()
    
    Handles token distribution, rewards, and usage tracking for the
    PRSM AI training and collaboration ecosystem.
    """
    
    def __init__(self):
        """Initialize FTNS service"""
        warnings.warn(
            "FTNSService is deprecated due to race condition vulnerabilities. "
            "Use AtomicFTNSService from prsm.economy.tokenomics.atomic_ftns_service instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.user_balances: Dict[str, FTNSBalance] = {}
        self.transactions: List[FTNSTransaction] = []
        self.total_staked = Decimal('0')
        self.guardrails = LaunchGuardrails(start_time=datetime.now(timezone.utc))
        self.reward_rates = {
            FTNSTransactionType.TRAINING_REWARD: Decimal('10.0'),
            FTNSTransactionType.MODEL_IMPROVEMENT: Decimal('25.0'),
            FTNSTransactionType.DATA_CONTRIBUTION: Decimal('5.0'),
            FTNSTransactionType.COLLABORATIVE_REASONING: Decimal('15.0'),
            FTNSTransactionType.KNOWLEDGE_DISTILLATION: Decimal('20.0'),
            FTNSTransactionType.PIPELINE_EXECUTION: Decimal('2.0')
        }
        
        logger.warning("FTNSService initialized (DEPRECATED - use AtomicFTNSService instead)")

    def stake_tokens(self, user_id: str, amount: Decimal) -> bool:
        """Stakes tokens while enforcing Launch Guardrails (Capped Mainnet)"""
        global_cap, user_cap = self.guardrails.get_current_caps()
        
        # 1. Check User Cap
        if amount > user_cap:
            logger.warning(f"Stake denied: User cap exceeded. Max: {user_cap}")
            return False
            
        # 2. Check Global Cap
        if self.total_staked + amount > global_cap:
            logger.warning(f"Stake denied: Global network cap reached. Max: {global_cap}")
            return False
            
        # 3. Process Stake
        if self.deduct_tokens(user_id, amount, description="Governance Stake"):
            self.total_staked += amount
            logger.info(f"✅ Successful stake: {amount} FTNS for {user_id}")
            return True
        return False

    def burn_tokens(self, user_id: str, amount: Decimal, reason: str = "System Fee"):
        """Permanently removes tokens from circulation (Deflationary mechanism)"""
        if self.deduct_tokens(user_id, amount, description=f"BURN: {reason}"):
            # In a real blockchain, this would send to a dead address (0x0...)
            logger.info(f"🔥 BURNED {amount} FTNS for reason: {reason}")
            return True
        return False
    
    async def get_user_balance(self, user_id: str) -> "FTNSBalance":
        """Get current FTNS balance for user (async for test compatibility)"""
        if user_id not in self.user_balances:
            self.user_balances[user_id] = FTNSBalance(
                user_id=user_id,
                balance=Decimal('0'),
                last_updated=datetime.now(timezone.utc)
            )
        
        return self.user_balances[user_id]
    
    def get_user_balance_sync(self, user_id: str) -> "FTNSBalance":
        """Get current FTNS balance (sync version)"""
        if user_id not in self.user_balances:
            self.user_balances[user_id] = FTNSBalance(
                user_id=user_id,
                balance=Decimal('0'),
                last_updated=datetime.now(timezone.utc)
            )
        
        return self.user_balances[user_id]
    
    def get_user_balance_decimal(self, user_id: str) -> Decimal:
        """Get current FTNS balance as Decimal (sync version for internal use)"""
        return self.get_user_balance_sync(user_id).balance
    
    def award_tokens(self, user_id: str, 
                    transaction_type: FTNSTransactionType,
                    base_amount: Optional[Decimal] = None,
                    multiplier: Decimal = Decimal('1.0'),
                    description: str = "",
                    metadata: Optional[Dict[str, Any]] = None) -> FTNSTransaction:
        """Award FTNS tokens to user
        
        ⚠️ DEPRECATED: This method is NOT thread-safe and has no persistence.
        Use FTNSQueries.execute_atomic_deduct() with negative amount for rewards.
        """
        logger.warning(
            "DEPRECATED: FTNSService.award_tokens() is deprecated. "
            "Use FTNSQueries.execute_atomic_deduct() instead."
        )
        try:
            # Calculate reward amount
            if base_amount is None:
                base_amount = self.reward_rates.get(transaction_type, Decimal('1.0'))
            
            award_amount = base_amount * multiplier
            
            # Update user balance
            balance_obj = self.get_user_balance_sync(user_id)
            current_balance = balance_obj.balance
            new_balance = current_balance + award_amount
            
            # Update balance record
            if user_id in self.user_balances:
                balance_record = self.user_balances[user_id]
                balance_record.balance = new_balance
                balance_record.total_earned += award_amount
                balance_record.last_updated = datetime.now(timezone.utc)
            else:
                balance_record = FTNSBalance(
                    user_id=user_id,
                    balance=new_balance,
                    last_updated=datetime.now(timezone.utc),
                    total_earned=award_amount
                )
                self.user_balances[user_id] = balance_record
            
            # Create transaction record
            tx_type_val = transaction_type.value if transaction_type else "unknown"
            transaction = FTNSTransaction(
                transaction_id=f"ftns_{len(self.transactions) + 1:06d}",
                user_id=user_id,
                amount=award_amount,
                transaction_type=transaction_type,
                description=description or f"Reward for {tx_type_val}",
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {},
                balance_after=new_balance
            )
            
            self.transactions.append(transaction)
            
            logger.info("FTNS tokens awarded",
                       user_id=user_id,
                       amount=float(award_amount),
                       new_balance=float(new_balance),
                       transaction_type=transaction_type.value)
            
            return transaction
            
        except Exception as e:
            logger.error(f"Failed to award FTNS tokens: {e}", user_id=user_id)
            raise
    
    def deduct_tokens(self, user_id: str,
                     amount: Decimal,
                     description: str = "",
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Deduct FTNS tokens from user balance
        
        ⚠️ DEPRECATED: This method has race condition vulnerabilities.
        Use FTNSQueries.execute_atomic_deduct() for thread-safe operations.
        """
        logger.warning(
            "DEPRECATED: FTNSService.deduct_tokens() is deprecated. "
            "Use FTNSQueries.execute_atomic_deduct() instead."
        )
        try:
            current_balance = self.get_user_balance_decimal(user_id)
            
            if current_balance < amount:
                logger.warning("Insufficient FTNS balance",
                              user_id=user_id,
                              requested=float(amount),
                              available=float(current_balance))
                return False
            
            # Update balance
            new_balance = current_balance - amount
            balance_record = self.user_balances[user_id]
            balance_record.balance = new_balance
            balance_record.total_spent += amount
            balance_record.last_updated = datetime.now(timezone.utc)
            
            # Create transaction record
            transaction = FTNSTransaction(
                transaction_id=f"ftns_{len(self.transactions) + 1:06d}",
                user_id=user_id,
                amount=-amount,  # Negative for deduction
                transaction_type=FTNSTransactionType.SYSTEM_USAGE,
                description=description or "Token usage",
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {},
                balance_after=new_balance
            )
            
            self.transactions.append(transaction)
            
            logger.info("FTNS tokens deducted",
                        user_id=user_id,
                        amount=float(amount),
                        new_balance=float(new_balance))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deduct FTNS tokens: {e}", user_id=user_id)
            return False
    
    async def reward_contribution(self, user_id: str, contribution_type: str, amount: float) -> bool:
        """Async method for rewarding contributions (wrapper for award_tokens)."""
        try:
            type_map = {
                "data": FTNSTransactionType.DATA_CONTRIBUTION,
                "training": FTNSTransactionType.TRAINING_REWARD,
                "model": FTNSTransactionType.MODEL_IMPROVEMENT,
                "collaboration": FTNSTransactionType.COLLABORATIVE_REASONING,
                "knowledge": FTNSTransactionType.KNOWLEDGE_DISTILLATION,
                "pipeline": FTNSTransactionType.PIPELINE_EXECUTION,
            }
            tx_type = type_map.get(contribution_type, FTNSTransactionType.DATA_CONTRIBUTION)
            
            self.award_tokens(
                user_id=user_id,
                transaction_type=tx_type,
                base_amount=Decimal(str(amount)),
                description=f"Contribution reward: {contribution_type}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to reward contribution: {e}")
            return False
    
    def get_user_transaction_history(self, user_id: str,
                                   limit: int = 50) -> List[FTNSTransaction]:
        """Get transaction history for user"""
        user_transactions = [
            tx for tx in self.transactions 
            if tx.user_id == user_id
        ]
        
        # Sort by timestamp, most recent first
        user_transactions.sort(key=lambda x: x.timestamp, reverse=True)
        
        return user_transactions[:limit]
    
    def calculate_training_reward(self, 
                                performance_improvement: float,
                                training_time_hours: float,
                                model_complexity: str = "medium") -> Decimal:
        """Calculate reward for AI training contributions"""
        base_reward = self.reward_rates[FTNSTransactionType.TRAINING_REWARD]
        
        # Performance multiplier (0.5x to 3.0x based on improvement)
        perf_multiplier = Decimal(str(max(0.5, min(3.0, 1.0 + performance_improvement))))
        
        # Time multiplier (small bonus for longer training)
        time_multiplier = Decimal(str(1.0 + min(0.5, training_time_hours / 10.0)))
        
        # Complexity multiplier
        complexity_multipliers = {
            "simple": Decimal('0.8'),
            "medium": Decimal('1.0'),
            "complex": Decimal('1.5'),
            "expert": Decimal('2.0')
        }
        complexity_multiplier = complexity_multipliers.get(model_complexity, Decimal('1.0'))
        
        total_reward = base_reward * perf_multiplier * time_multiplier * complexity_multiplier
        
        return total_reward
    
    def record_nwtn_pipeline_usage(self, user_id: str,
                                  candidates_processed: int,
                                  operations_count: int,
                                  processing_time: float) -> FTNSTransaction:
        """Record FTNS usage for NWTN pipeline processing"""
        # Calculate cost based on computational resources used
        base_cost = Decimal('1.0')  # Base cost per pipeline run
        
        # Scale with computational complexity
        complexity_multiplier = Decimal(str(min(5.0, operations_count / 1_000_000)))  # Scale by millions of operations
        candidate_multiplier = Decimal(str(min(2.0, candidates_processed / 1000)))  # Scale by thousands of candidates
        time_multiplier = Decimal(str(min(3.0, processing_time / 60)))  # Scale by minutes
        
        total_cost = base_cost * (complexity_multiplier + candidate_multiplier + time_multiplier)
        
        # Award tokens instead of deduct (incentivize usage during development)
        return self.award_tokens(
            user_id=user_id,
            transaction_type=FTNSTransactionType.PIPELINE_EXECUTION,
            base_amount=total_cost,
            description=f"NWTN pipeline execution: {candidates_processed} candidates, {operations_count:,} operations",
            metadata={
                "candidates_processed": candidates_processed,
                "operations_count": operations_count,
                "processing_time_seconds": processing_time,
                "complexity_score": float(complexity_multiplier),
                "pipeline_type": "nwtn_enhanced"
            }
        )
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get FTNS service usage statistics"""
        total_users = len(self.user_balances)
        total_transactions = len(self.transactions)
        total_tokens_distributed = sum(
            tx.amount for tx in self.transactions 
            if tx.amount > 0
        )
        total_tokens_used = sum(
            abs(tx.amount) for tx in self.transactions 
            if tx.amount < 0
        )
        
        return {
            "total_users": total_users,
            "total_transactions": total_transactions,
            "total_tokens_distributed": float(total_tokens_distributed),
            "total_tokens_used": float(total_tokens_used),
            "active_token_supply": float(total_tokens_distributed - total_tokens_used),
            "average_balance": float(total_tokens_distributed / max(total_users, 1)),
            "transaction_types": {
                tx_type.value: len([tx for tx in self.transactions if tx.transaction_type == tx_type])
                for tx_type in FTNSTransactionType
            }
        }


# Global FTNS service instance
_global_ftns_service: Optional[FTNSService] = None


def get_ftns_service() -> FTNSService:
    """Get global FTNS service instance"""
    global _global_ftns_service
    if _global_ftns_service is None:
        _global_ftns_service = FTNSService()
    return _global_ftns_service


# Convenience functions
def award_training_tokens(user_id: str, performance_improvement: float,
                         training_time_hours: float, model_complexity: str = "medium") -> FTNSTransaction:
    """Award tokens for AI training contribution"""
    service = get_ftns_service()
    reward_amount = service.calculate_training_reward(performance_improvement, training_time_hours, model_complexity)
    
    return service.award_tokens(
        user_id=user_id,
        transaction_type=FTNSTransactionType.TRAINING_REWARD,
        base_amount=reward_amount,
        description=f"AI training reward: {performance_improvement:.1%} improvement, {training_time_hours:.1f}h training",
        metadata={
            "performance_improvement": performance_improvement,
            "training_time_hours": training_time_hours,
            "model_complexity": model_complexity,
            "reward_calculation": "performance_based"
        }
    )


def get_user_ftns_balance(user_id: str) -> Decimal:
    """Get user's current FTNS token balance"""
    return get_ftns_service().get_user_balance(user_id)


# Module-level singleton for direct import
ftns_service = get_ftns_service()
