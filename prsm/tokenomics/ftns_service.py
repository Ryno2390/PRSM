"""
PRSM FTNS Token Service
Enhanced from Co-Lab's tokenomics service with PRSM-specific features

The FTNS (Fungible Tokens for Node Support) Service manages the complete
token economy that powers PRSM's decentralized operations.

Core Economic Functions:

1. Context Allocation & Pricing:
   - Dynamic pricing based on supply/demand and query complexity
   - Context unit allocation for NWTN computational resources
   - User tier-based discounts and premium pricing
   - Real-time cost calculation with microsecond precision

2. Contribution Rewards:
   - Data contribution rewards (per MB uploaded)
   - Model contribution bounties for specialist AIs
   - Research publication rewards with citation bonuses
   - Teaching success bonuses for improved student models
   - Governance participation incentives

3. Revenue Distribution:
   - Quarterly dividend distributions to token holders
   - Royalty payments for content creators based on usage
   - Impact-based rewards for highly cited research
   - Performance bonuses for high-quality teaching models

4. Economic Sustainability:
   - Token circulation management preventing inflation
   - Burn mechanisms for maintaining scarcity
   - Staking rewards for long-term holders
   - Market-making for price stability

FTNS Use Cases:

Resource Access:
- NWTN context allocation for complex queries
- Priority processing for time-sensitive research
- Advanced feature access and early releases
- Model marketplace rentals and subscriptions

Earning Opportunities:
- Upload datasets, models, or research papers
- Provide computational resources (model hosting)
- Train high-performing teacher models
- Participate in governance voting
- Host P2P federation nodes

Economic Mechanisms:

Dynamic Pricing:
- Base context cost adjusted by complexity multipliers
- User tier discounts for frequent contributors
- Peak/off-peak pricing for load balancing
- Quality bonuses for highly-rated content

Reward Calculation:
- Contribution value assessment with quality metrics
- Citation tracking for research impact measurement
- Student improvement tracking for teaching rewards
- Governance participation scoring

Token Circulation:
- Minting for system rewards and contributions
- Burning for deflationary pressure
- Dividend pools from platform revenue
- Marketplace transaction fees

Integration Points:
- NWTN Orchestrator: Context cost calculation and charging
- Agent Framework: Computational resource allocation
- Teacher Models: Performance-based reward distribution
- Governance System: Voting weight and participation rewards
- P2P Federation: Node operation incentives
- Safety Infrastructure: Economic penalties for violations

The FTNS system ensures sustainable economics while incentivizing
quality contributions and democratic participation in PRSM governance.
"""

import asyncio
import math
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4

# Set precision for financial calculations
getcontext().prec = 18

from ..core.config import settings
from ..core.models import (
    FTNSTransaction, FTNSBalance, ProvenanceRecord,
    PRSMSession, ArchitectTask, TeacherModel
)

# === FTNS Cost Parameters ===

# Base NWTN costs
BASE_NWTN_FEE = float(getattr(settings, "FTNS_BASE_NWTN_FEE", 1.0))
CONTEXT_UNIT_COST = float(getattr(settings, "FTNS_CONTEXT_UNIT_COST", 0.1))
ARCHITECT_DECOMPOSITION_COST = float(getattr(settings, "FTNS_ARCHITECT_COST", 5.0))
COMPILER_SYNTHESIS_COST = float(getattr(settings, "FTNS_COMPILER_COST", 10.0))

# Agent-specific costs
AGENT_COSTS = {
    "architect": float(getattr(settings, "FTNS_ARCHITECT_COST", 5.0)),
    "prompter": float(getattr(settings, "FTNS_PROMPTER_COST", 2.0)),
    "router": float(getattr(settings, "FTNS_ROUTER_COST", 1.0)),
    "executor": float(getattr(settings, "FTNS_EXECUTOR_COST", 8.0)),
    "compiler": float(getattr(settings, "FTNS_COMPILER_COST", 10.0)),
}

# Teacher model costs
TEACHER_TRAINING_COST = float(getattr(settings, "FTNS_TEACHER_TRAINING_COST", 50.0))
RLVR_VALIDATION_COST = float(getattr(settings, "FTNS_RLVR_COST", 25.0))

# === FTNS Reward Parameters ===

# Data contribution rewards
REWARD_PER_MB = float(getattr(settings, "FTNS_REWARD_PER_MB", 0.05))
MODEL_CONTRIBUTION_REWARD = float(getattr(settings, "FTNS_MODEL_REWARD", 100.0))
RESEARCH_PUBLICATION_REWARD = float(getattr(settings, "FTNS_RESEARCH_REWARD", 500.0))

# Teaching rewards
SUCCESSFUL_TEACHING_REWARD = float(getattr(settings, "FTNS_TEACHING_REWARD", 20.0))
STUDENT_IMPROVEMENT_MULTIPLIER = float(getattr(settings, "FTNS_IMPROVEMENT_MULTIPLIER", 2.0))

# Governance participation
GOVERNANCE_PARTICIPATION_REWARD = float(getattr(settings, "FTNS_GOVERNANCE_REWARD", 5.0))

# === FTNS Service Class ===

class FTNSService:
    """
    Enhanced FTNS token service for PRSM
    Manages context allocation, rewards, and token economy
    """
    
    def __init__(self):
        self.balances: Dict[str, FTNSBalance] = {}
        self.transactions: List[FTNSTransaction] = []
        self.provenance_records: Dict[str, ProvenanceRecord] = {}
    
    # === Context Management ===
    
    async def calculate_context_cost(self, session: PRSMSession, context_units: int) -> float:
        """
        Calculate cost for context allocation in NWTN
        
        Args:
            session: PRSM session requesting context
            context_units: Number of context units requested
            
        Returns:
            Cost in FTNS tokens
        """
        base_cost = context_units * CONTEXT_UNIT_COST
        
        # Apply user tier discounts (if implemented)
        user_multiplier = await self._get_user_tier_multiplier(session.user_id)
        
        # Apply complexity-based pricing
        complexity_multiplier = 1.0
        if hasattr(session, 'complexity_estimate'):
            complexity_multiplier = 1.0 + (session.complexity_estimate * 0.5)
        
        total_cost = base_cost * user_multiplier * complexity_multiplier
        
        return round(total_cost, 8)
    
    async def charge_context_access(self, user_id: str, context_units: int) -> bool:
        """
        Charge user for context access
        
        Args:
            user_id: User requesting context
            context_units: Number of context units to charge for
            
        Returns:
            True if charge successful, False if insufficient funds
        """
        if context_units <= 0:
            return True
        
        # Calculate cost
        session = PRSMSession(user_id=user_id)  # Minimal session for cost calc
        cost = await self.calculate_context_cost(session, context_units)
        
        # Check balance
        if not await self._has_sufficient_balance(user_id, cost):
            return False
        
        # Create transaction
        transaction = FTNSTransaction(
            from_user=user_id,
            to_user="system",
            amount=cost,
            transaction_type="charge",
            description=f"Context access charge for {context_units} units",
            context_units=context_units
        )
        
        # Update balance and record transaction
        await self._update_balance(user_id, -cost)
        await self._record_transaction(transaction)
        
        return True
    
    async def allocate_context(self, session: PRSMSession, required_context: int) -> bool:
        """
        Allocate context for a PRSM session
        
        Args:
            session: PRSM session
            required_context: Required context units
            
        Returns:
            True if allocation successful
        """
        # Check if user has pre-allocated enough context
        if session.nwtn_context_allocation >= required_context:
            return True
        
        # Calculate additional context needed
        additional_context = required_context - session.nwtn_context_allocation
        
        # Charge for additional context
        success = await self.charge_context_access(session.user_id, additional_context)
        
        if success:
            session.nwtn_context_allocation = required_context
        
        return success
    
    # === Reward Distribution ===
    
    async def reward_contribution(self, user_id: str, contribution_type: str, value: float, 
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Reward user for various types of contributions
        
        Args:
            user_id: User to reward
            contribution_type: Type of contribution (data, model, research, teaching)
            value: Value metric for reward calculation
            metadata: Additional metadata for reward calculation
            
        Returns:
            True if reward successful
        """
        reward_amount = await self._calculate_contribution_reward(
            contribution_type, value, metadata
        )
        
        if reward_amount <= 0:
            return True  # No reward to give
        
        # Create reward transaction
        transaction = FTNSTransaction(
            from_user=None,  # System mint
            to_user=user_id,
            amount=reward_amount,
            transaction_type="reward",
            description=f"Contribution reward: {contribution_type}",
            ipfs_cid=metadata.get('cid') if metadata else None
        )
        
        # Update balance and record transaction
        await self._update_balance(user_id, reward_amount)
        await self._record_transaction(transaction)
        
        return True
    
    async def calculate_royalties(self, content_hash: str, access_count: int) -> float:
        """
        Calculate royalties for content usage
        
        Args:
            content_hash: IPFS hash of accessed content
            access_count: Number of times content was accessed
            
        Returns:
            Royalty amount in FTNS tokens
        """
        if content_hash not in self.provenance_records:
            return 0.0
        
        record = self.provenance_records[content_hash]
        
        # Calculate royalties based on access count and content type
        base_royalty = access_count * 0.01  # 0.01 FTNS per access
        
        # Apply content quality multiplier
        quality_multiplier = 1.0
        if record.access_count > 100:  # Popular content gets bonus
            quality_multiplier = 1.5
        
        total_royalty = base_royalty * quality_multiplier
        
        return round(total_royalty, 8)
    
    async def distribute_dividends(self, holders: List[str], pool: float) -> Dict[str, float]:
        """
        Distribute quarterly dividends to FTNS holders
        
        Args:
            holders: List of FTNS holders
            pool: Total dividend pool to distribute
            
        Returns:
            Dictionary mapping user_id to dividend amount
        """
        if pool <= 0 or not holders:
            return {}
        
        # Calculate total holdings
        total_holdings = 0.0
        holder_balances = {}
        
        for holder in holders:
            balance = await self._get_balance(holder)
            if balance > 0:
                holder_balances[holder] = balance
                total_holdings += balance
        
        if total_holdings <= 0:
            return {}
        
        # Distribute proportionally
        distributions = {}
        for holder, balance in holder_balances.items():
            share = balance / total_holdings
            dividend = pool * share
            
            if dividend > 0:
                distributions[holder] = round(dividend, 8)
                
                # Create dividend transaction
                transaction = FTNSTransaction(
                    from_user=None,  # System distribution
                    to_user=holder,
                    amount=dividend,
                    transaction_type="dividend",
                    description="Quarterly dividend distribution"
                )
                
                await self._update_balance(holder, dividend)
                await self._record_transaction(transaction)
        
        return distributions
    
    async def reward_teaching_success(self, teacher_id: str, student_improvement: float) -> float:
        """
        Reward teacher models for successful student improvement
        
        Args:
            teacher_id: ID of the teacher model
            student_improvement: Improvement score (0.0 to 1.0)
            
        Returns:
            Reward amount given
        """
        base_reward = SUCCESSFUL_TEACHING_REWARD
        improvement_bonus = student_improvement * STUDENT_IMPROVEMENT_MULTIPLIER * base_reward
        
        total_reward = base_reward + improvement_bonus
        
        # Find teacher model owner for reward
        # In real implementation, this would query the teacher model registry
        teacher_owner = f"teacher_owner_{teacher_id}"  # Placeholder
        
        await self.reward_contribution(
            teacher_owner, 
            "teaching", 
            total_reward,
            {"teacher_id": teacher_id, "improvement": student_improvement}
        )
        
        return total_reward
    
    # === Private Helper Methods ===
    
    async def _calculate_contribution_reward(self, contribution_type: str, value: float, 
                                           metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate reward amount based on contribution type and value"""
        
        if contribution_type == "data":
            # Reward based on data size
            return value * REWARD_PER_MB
        
        elif contribution_type == "model":
            # Fixed reward for model contributions
            return MODEL_CONTRIBUTION_REWARD
        
        elif contribution_type == "research":
            # Reward for research publications
            base_reward = RESEARCH_PUBLICATION_REWARD
            
            # Apply quality multipliers if available
            if metadata and 'citations' in metadata:
                citation_bonus = min(metadata['citations'] * 10, 500)  # Cap at 500 FTNS
                base_reward += citation_bonus
            
            return base_reward
        
        elif contribution_type == "teaching":
            # Teaching rewards handled separately
            return value
        
        elif contribution_type == "governance":
            # Governance participation reward
            return GOVERNANCE_PARTICIPATION_REWARD
        
        else:
            return 0.0
    
    async def _get_user_tier_multiplier(self, user_id: str) -> float:
        """Get pricing multiplier based on user tier"""
        # Placeholder - in real implementation, check user tier
        return 1.0
    
    async def _has_sufficient_balance(self, user_id: str, amount: float) -> bool:
        """Check if user has sufficient balance"""
        balance = await self._get_balance(user_id)
        return balance >= amount
    
    async def _get_balance(self, user_id: str) -> float:
        """Get user's current FTNS balance"""
        if user_id not in self.balances:
            self.balances[user_id] = FTNSBalance(user_id=user_id, balance=0.0)
        
        return self.balances[user_id].balance
    
    async def _update_balance(self, user_id: str, amount: float) -> None:
        """Update user's balance by amount (positive for credit, negative for debit)"""
        if user_id not in self.balances:
            self.balances[user_id] = FTNSBalance(user_id=user_id, balance=0.0)
        
        self.balances[user_id].balance += amount
        self.balances[user_id].updated_at = datetime.now(timezone.utc)
    
    async def _record_transaction(self, transaction: FTNSTransaction) -> None:
        """Record a transaction in the ledger"""
        self.transactions.append(transaction)
    
    # === Public Query Methods ===
    
    async def get_user_balance(self, user_id: str) -> FTNSBalance:
        """Get user's complete balance information"""
        if user_id not in self.balances:
            self.balances[user_id] = FTNSBalance(user_id=user_id, balance=0.0)
        
        return self.balances[user_id]
    
    async def get_transaction_history(self, user_id: str, limit: int = 100) -> List[FTNSTransaction]:
        """Get user's transaction history"""
        user_transactions = [
            tx for tx in self.transactions 
            if tx.from_user == user_id or tx.to_user == user_id
        ]
        
        # Sort by creation time (most recent first)
        user_transactions.sort(key=lambda x: x.created_at, reverse=True)
        
        return user_transactions[:limit]
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide FTNS statistics"""
        total_supply = sum(balance.balance for balance in self.balances.values())
        total_transactions = len(self.transactions)
        
        transaction_types = {}
        for tx in self.transactions:
            tx_type = tx.transaction_type
            transaction_types[tx_type] = transaction_types.get(tx_type, 0) + 1
        
        return {
            "total_supply": total_supply,
            "total_holders": len(self.balances),
            "total_transactions": total_transactions,
            "transaction_types": transaction_types,
            "active_sessions": 0  # Placeholder
        }


# === Global Service Instance ===
ftns_service = FTNSService()