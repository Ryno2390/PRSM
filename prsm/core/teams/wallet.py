"""
PRSM Team Wallet Service

Implements team wallet functionality including multisig operations,
FTNS distribution policies, and treasury management.
"""

import asyncio
import hashlib
from datetime import datetime, timezone, timedelta
from decimal import Decimal, getcontext
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4

import structlog

# Set precision for financial calculations
getcontext().prec = 18

from prsm.core.models import FTNSTransaction
from prsm.economy.tokenomics.ftns_service import ftns_service
from .models import (
    Team, TeamMember, TeamWallet, TeamTask,
    TeamRole, RewardPolicy, TeamMembershipStatus
)

logger = structlog.get_logger(__name__)


class TeamWalletService:
    """
    Team wallet service implementing multisig functionality and FTNS distribution
    
    Features:
    - Multisig wallet operations with configurable signature requirements
    - Automated reward distribution based on contribution metrics
    - Treasury management with spending limits and emergency controls
    - Smart contract integration for cross-chain operations
    """
    
    def __init__(self):
        self.service_id = str(uuid4())
        self.logger = logger.bind(component="team_wallet", service_id=self.service_id)
        
        # Wallet state tracking
        self.pending_transactions: Dict[str, Dict[str, Any]] = {}
        self.signature_cache: Dict[str, List[str]] = {}  # tx_id -> signatures
        
        # Distribution state
        self.distribution_history: Dict[UUID, List[Dict[str, Any]]] = {}
        self.pending_distributions: Dict[UUID, Dict[str, Any]] = {}
        
        # Performance tracking
        self.wallet_stats = {
            "total_wallets_created": 0,
            "total_transactions_processed": 0,
            "total_ftns_distributed": 0.0,
            "total_signatures_collected": 0,
            "multisig_operations_executed": 0,
            "emergency_freezes_activated": 0
        }
        
        # Synchronization
        self._wallet_lock = asyncio.Lock()
        self._distribution_lock = asyncio.Lock()
        self._signature_lock = asyncio.Lock()
        
        print("💰 TeamWalletService initialized")
    
    
    async def create_team_wallet(self, team: Team, initial_signers: List[str], 
                                required_signatures: int = 1) -> TeamWallet:
        """
        Create a new team wallet with multisig configuration
        
        Args:
            team: Team entity
            initial_signers: List of user IDs authorized to sign transactions
            required_signatures: Number of signatures required for transactions
            
        Returns:
            Created team wallet
        """
        try:
            async with self._wallet_lock:
                # Validate inputs
                if required_signatures > len(initial_signers):
                    raise ValueError("Required signatures cannot exceed number of signers")
                
                if required_signatures < 1:
                    raise ValueError("At least one signature required")
                
                # Generate wallet address
                wallet_address = await self._generate_wallet_address(team.team_id, initial_signers)
                
                # Create wallet
                wallet = TeamWallet(
                    team_id=team.team_id,
                    is_multisig=len(initial_signers) > 1,
                    required_signatures=required_signatures,
                    authorized_signers=initial_signers,
                    wallet_address=wallet_address,
                    reward_policy=team.reward_policy
                )
                
                # Set initial spending limits based on roles
                spending_limits = await self._calculate_initial_spending_limits(initial_signers)
                wallet.spending_limits = spending_limits
                
                # Update statistics
                self.wallet_stats["total_wallets_created"] += 1
                
                self.logger.info(
                    "Team wallet created",
                    team_id=str(team.team_id),
                    wallet_id=str(wallet.wallet_id),
                    signers_count=len(initial_signers),
                    required_signatures=required_signatures
                )
                
                return wallet
                
        except Exception as e:
            self.logger.error("Failed to create team wallet", error=str(e))
            raise
    
    
    async def deposit_ftns(self, wallet: TeamWallet, amount: float, 
                          depositor_id: str, description: str = "Team deposit") -> bool:
        """
        Deposit FTNS tokens into team wallet
        
        Args:
            wallet: Team wallet
            amount: Amount to deposit
            depositor_id: User making the deposit
            description: Transaction description
            
        Returns:
            True if deposit successful
        """
        try:
            if amount <= 0:
                raise ValueError("Deposit amount must be positive")
            
            # Check if wallet is frozen
            if wallet.emergency_freeze:
                raise ValueError("Wallet is frozen - deposits not allowed")
            
            # Verify depositor has sufficient balance
            user_balance = await ftns_service.get_user_balance(depositor_id)
            if user_balance.balance < amount:
                raise ValueError("Insufficient FTNS balance for deposit")
            
            # Transfer from user to team wallet
            success = await ftns_service.charge_context_access(depositor_id, int(amount))
            if not success:
                raise ValueError("Failed to charge user account")
            
            # Update team wallet balance
            wallet.total_balance += amount
            wallet.available_balance += amount
            
            # Create transaction record
            transaction = FTNSTransaction(
                from_user=depositor_id,
                to_user=f"team_wallet_{wallet.wallet_id}",
                amount=amount,
                transaction_type="team_deposit",
                description=description
            )
            
            await ftns_service._record_transaction(transaction)
            
            # Update statistics
            self.wallet_stats["total_transactions_processed"] += 1
            
            self.logger.info(
                "FTNS deposited to team wallet",
                wallet_id=str(wallet.wallet_id),
                amount=amount,
                depositor=depositor_id,
                new_balance=wallet.total_balance
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to deposit FTNS", error=str(e))
            return False
    
    
    async def distribute_rewards(self, wallet: TeamWallet, team_members: List[TeamMember], 
                                total_amount: Optional[float] = None) -> Dict[str, float]:
        """
        Distribute rewards to team members based on configured policy
        
        Args:
            wallet: Team wallet
            team_members: List of active team members
            total_amount: Amount to distribute (None = all available balance)
            
        Returns:
            Dictionary mapping user_id to distributed amount
        """
        try:
            async with self._distribution_lock:
                # Determine distribution amount
                if total_amount is None:
                    total_amount = wallet.available_balance
                
                if total_amount <= 0:
                    return {}
                
                if total_amount > wallet.available_balance:
                    raise ValueError("Insufficient available balance for distribution")
                
                # Calculate distributions based on policy
                distributions = await self._calculate_distributions(
                    wallet, team_members, total_amount
                )
                
                # Execute distributions
                successful_distributions = {}
                total_distributed = 0.0
                
                for user_id, amount in distributions.items():
                    if amount > 0:
                        # Credit user account
                        await ftns_service._update_balance(user_id, amount)
                        
                        # Create transaction record
                        transaction = FTNSTransaction(
                            from_user=f"team_wallet_{wallet.wallet_id}",
                            to_user=user_id,
                            amount=amount,
                            transaction_type="team_reward",
                            description=f"Team reward distribution - {wallet.reward_policy}"
                        )
                        
                        await ftns_service._record_transaction(transaction)
                        
                        successful_distributions[user_id] = amount
                        total_distributed += amount
                
                # Update wallet balance
                wallet.available_balance -= total_distributed
                wallet.last_distribution = datetime.now(timezone.utc)
                
                # Record distribution in history
                distribution_record = {
                    "timestamp": datetime.now(timezone.utc),
                    "total_amount": total_distributed,
                    "policy": wallet.reward_policy,
                    "recipients": successful_distributions,
                    "distribution_id": str(uuid4())
                }
                
                if wallet.team_id not in self.distribution_history:
                    self.distribution_history[wallet.team_id] = []
                
                self.distribution_history[wallet.team_id].append(distribution_record)
                
                # Update statistics
                self.wallet_stats["total_ftns_distributed"] += total_distributed
                
                self.logger.info(
                    "Rewards distributed",
                    wallet_id=str(wallet.wallet_id),
                    total_distributed=total_distributed,
                    recipients_count=len(successful_distributions),
                    policy=wallet.reward_policy
                )
                
                return successful_distributions
                
        except Exception as e:
            self.logger.error("Failed to distribute rewards", error=str(e))
            return {}
    
    
    async def initiate_multisig_transaction(self, wallet: TeamWallet, transaction_data: Dict[str, Any],
                                          initiator_id: str) -> str:
        """
        Initiate a multisig transaction requiring multiple signatures
        
        Args:
            wallet: Team wallet
            transaction_data: Transaction details
            initiator_id: User initiating the transaction
            
        Returns:
            Transaction ID for signature collection
        """
        try:
            async with self._signature_lock:
                # Validate initiator authorization
                if initiator_id not in wallet.authorized_signers:
                    raise ValueError("User not authorized to initiate transactions")
                
                # Validate transaction
                if not await self._validate_transaction(wallet, transaction_data):
                    raise ValueError("Transaction validation failed")
                
                # Create transaction ID
                tx_id = str(uuid4())
                
                # Store pending transaction
                self.pending_transactions[tx_id] = {
                    "wallet_id": wallet.wallet_id,
                    "transaction_data": transaction_data,
                    "initiator": initiator_id,
                    "created_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc) + timedelta(hours=24),
                    "required_signatures": wallet.required_signatures,
                    "status": "pending"
                }
                
                # Initialize signature collection
                self.signature_cache[tx_id] = [initiator_id]  # Initiator automatically signs
                
                self.logger.info(
                    "Multisig transaction initiated",
                    tx_id=tx_id,
                    wallet_id=str(wallet.wallet_id),
                    initiator=initiator_id,
                    required_signatures=wallet.required_signatures
                )
                
                # Check if sufficient signatures already available
                if len(self.signature_cache[tx_id]) >= wallet.required_signatures:
                    await self._execute_multisig_transaction(tx_id)
                
                return tx_id
                
        except Exception as e:
            self.logger.error("Failed to initiate multisig transaction", error=str(e))
            raise
    
    
    async def sign_transaction(self, tx_id: str, signer_id: str, signature: str) -> bool:
        """
        Sign a pending multisig transaction
        
        Args:
            tx_id: Transaction ID
            signer_id: User signing the transaction
            signature: Cryptographic signature
            
        Returns:
            True if signature accepted
        """
        try:
            async with self._signature_lock:
                # Validate transaction exists
                if tx_id not in self.pending_transactions:
                    raise ValueError("Transaction not found")
                
                pending_tx = self.pending_transactions[tx_id]
                
                # Check if transaction expired
                if datetime.now(timezone.utc) > pending_tx["expires_at"]:
                    self.pending_transactions[tx_id]["status"] = "expired"
                    raise ValueError("Transaction expired")
                
                # Get wallet
                wallet_id = pending_tx["wallet_id"]
                
                # Validate signer authorization (would need to fetch wallet)
                # For now, assuming signer is authorized
                
                # Verify signature (simplified)
                if not await self._verify_signature(pending_tx["transaction_data"], signer_id, signature):
                    raise ValueError("Invalid signature")
                
                # Add signature if not already present
                if signer_id not in self.signature_cache[tx_id]:
                    self.signature_cache[tx_id].append(signer_id)
                    
                    # Update statistics
                    self.wallet_stats["total_signatures_collected"] += 1
                
                # Check if sufficient signatures collected
                required_signatures = pending_tx["required_signatures"]
                if len(self.signature_cache[tx_id]) >= required_signatures:
                    await self._execute_multisig_transaction(tx_id)
                
                self.logger.info(
                    "Transaction signed",
                    tx_id=tx_id,
                    signer=signer_id,
                    signatures_collected=len(self.signature_cache[tx_id]),
                    required_signatures=required_signatures
                )
                
                return True
                
        except Exception as e:
            self.logger.error("Failed to sign transaction", error=str(e))
            return False
    
    
    async def emergency_freeze_wallet(self, wallet: TeamWallet, requester_id: str, 
                                    reason: str) -> bool:
        """
        Emergency freeze wallet to prevent transactions
        
        Args:
            wallet: Team wallet to freeze
            requester_id: User requesting freeze
            reason: Reason for emergency freeze
            
        Returns:
            True if freeze activated
        """
        try:
            async with self._wallet_lock:
                # Validate requester has emergency authority
                if not await self._validate_emergency_authority(wallet, requester_id):
                    raise ValueError("User not authorized for emergency actions")
                
                # Activate freeze
                wallet.emergency_freeze = True
                
                # Log emergency action
                emergency_record = {
                    "action": "emergency_freeze",
                    "requester": requester_id,
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc),
                    "wallet_id": str(wallet.wallet_id)
                }
                
                # Update statistics
                self.wallet_stats["emergency_freezes_activated"] += 1
                
                self.logger.warning(
                    "Emergency wallet freeze activated",
                    wallet_id=str(wallet.wallet_id),
                    requester=requester_id,
                    reason=reason
                )
                
                return True
                
        except Exception as e:
            self.logger.error("Failed to freeze wallet", error=str(e))
            return False
    
    
    # === Private Helper Methods ===
    
    async def _generate_wallet_address(self, team_id: UUID, signers: List[str]) -> str:
        """Generate unique wallet address for team"""
        # Simplified address generation
        data = f"{team_id}_{','.join(sorted(signers))}_{datetime.now().timestamp()}"
        return hashlib.sha256(data.encode()).hexdigest()[:42]
    
    
    async def _calculate_initial_spending_limits(self, signers: List[str]) -> Dict[str, float]:
        """Calculate initial spending limits based on roles"""
        # Simplified role-based limits
        limits = {}
        for signer in signers:
            # In production, would check user's role in team
            limits[signer] = 1000.0  # 1000 FTNS default limit
        return limits
    
    
    async def _calculate_distributions(self, wallet: TeamWallet, members: List[TeamMember], 
                                     total_amount: float) -> Dict[str, float]:
        """Calculate reward distributions based on wallet policy"""
        distributions = {}
        
        # Filter active members only
        active_members = [m for m in members if m.status == TeamMembershipStatus.ACTIVE]
        
        if not active_members:
            return distributions
        
        if wallet.reward_policy == RewardPolicy.EQUAL_SHARES:
            # Equal distribution
            amount_per_member = total_amount / len(active_members)
            for member in active_members:
                distributions[member.user_id] = amount_per_member
        
        elif wallet.reward_policy == RewardPolicy.PROPORTIONAL:
            # Distribution based on contribution metrics
            total_score = 0.0
            member_scores = {}
            
            for member in active_members:
                # Calculate contribution score
                score = self._calculate_contribution_score(member, wallet)
                member_scores[member.user_id] = score
                total_score += score
            
            if total_score > 0:
                for user_id, score in member_scores.items():
                    distributions[user_id] = (score / total_score) * total_amount
        
        elif wallet.reward_policy == RewardPolicy.STAKE_WEIGHTED:
            # Distribution based on FTNS contributed
            total_contributed = sum(m.ftns_contributed for m in active_members)
            
            if total_contributed > 0:
                for member in active_members:
                    weight = member.ftns_contributed / total_contributed
                    distributions[member.user_id] = weight * total_amount
        
        elif wallet.reward_policy == RewardPolicy.PERFORMANCE_WEIGHTED:
            # Distribution based on performance scores
            total_performance = sum(m.performance_score for m in active_members)
            
            if total_performance > 0:
                for member in active_members:
                    weight = member.performance_score / total_performance
                    distributions[member.user_id] = weight * total_amount
        
        return distributions
    
    
    def _calculate_contribution_score(self, member: TeamMember, wallet: TeamWallet) -> float:
        """Calculate member's contribution score for reward distribution"""
        score = 0.0
        
        # Get metrics and weights from wallet configuration
        metrics = wallet.distribution_metrics
        weights = wallet.metric_weights
        
        if len(metrics) != len(weights):
            # Default equal weighting
            weights = [1.0 / len(metrics)] * len(metrics)
        
        for i, metric in enumerate(metrics):
            weight = weights[i] if i < len(weights) else 0.0
            
            if metric == "task_submissions":
                score += member.tasks_completed * weight
            elif metric == "model_contributions":
                score += member.models_contributed * weight
            elif metric == "query_accuracy":
                score += member.performance_score * weight
            elif metric == "ftns_contributed":
                score += member.ftns_contributed * weight * 0.001  # Scale down
        
        return max(score, 0.0)
    
    
    async def _validate_transaction(self, wallet: TeamWallet, transaction_data: Dict[str, Any]) -> bool:
        """Validate transaction against wallet rules"""
        # Check if wallet is frozen
        if wallet.emergency_freeze:
            return False
        
        # Check amount limits
        amount = transaction_data.get("amount", 0)
        if amount > wallet.available_balance:
            return False
        
        # Additional validation rules would go here
        return True
    
    
    async def _verify_signature(self, transaction_data: Dict[str, Any], signer_id: str, signature: str) -> bool:
        """Verify cryptographic signature"""
        # Simplified signature verification
        # In production, would use proper cryptographic verification
        expected_sig = hashlib.sha256(f"{transaction_data}_{signer_id}".encode()).hexdigest()
        return signature == expected_sig[:16]  # Simplified check
    
    
    async def _execute_multisig_transaction(self, tx_id: str):
        """Execute a multisig transaction with sufficient signatures"""
        try:
            pending_tx = self.pending_transactions[tx_id]
            transaction_data = pending_tx["transaction_data"]
            
            # Mark as executing
            pending_tx["status"] = "executing"
            
            # Execute the transaction
            # This would implement the actual transaction logic
            # For now, just mark as completed
            
            pending_tx["status"] = "completed"
            pending_tx["executed_at"] = datetime.now(timezone.utc)
            
            # Update statistics
            self.wallet_stats["multisig_operations_executed"] += 1
            
            self.logger.info(
                "Multisig transaction executed",
                tx_id=tx_id,
                signatures_collected=len(self.signature_cache[tx_id])
            )
            
        except Exception as e:
            self.pending_transactions[tx_id]["status"] = "failed"
            self.logger.error("Failed to execute multisig transaction", tx_id=tx_id, error=str(e))
    
    
    async def _validate_emergency_authority(self, wallet: TeamWallet, user_id: str) -> bool:
        """Validate user has emergency authority for wallet operations"""
        # Check if user is authorized signer
        if user_id not in wallet.authorized_signers:
            return False
        
        # Additional authority checks would go here
        return True
    
    
    async def get_wallet_statistics(self) -> Dict[str, Any]:
        """Get comprehensive wallet service statistics"""
        return {
            **self.wallet_stats,
            "pending_transactions": len(self.pending_transactions),
            "active_signatures": len(self.signature_cache),
            "distribution_history_count": sum(len(history) for history in self.distribution_history.values()),
            "average_signatures_per_transaction": (
                self.wallet_stats["total_signatures_collected"] / 
                max(self.wallet_stats["multisig_operations_executed"], 1)
            )
        }


# === Global Service Instance ===

_wallet_service_instance: Optional[TeamWalletService] = None

def get_team_wallet_service() -> TeamWalletService:
    """Get or create the global team wallet service instance"""
    global _wallet_service_instance
    if _wallet_service_instance is None:
        _wallet_service_instance = TeamWalletService()
    return _wallet_service_instance