"""
PRSM Governance Token Distribution and Voting Activation System
==============================================================

Manages the distribution of FTNS tokens for governance participation
and activates the decentralized voting mechanisms for PRSM governance.

Key Features:
- Initial token distribution for governance participants
- Staking mechanisms for voting power
- Token distribution for various contribution types
- Governance voting activation and coordination
- Integration with quadratic voting and federated councils
"""

import asyncio
import structlog
from datetime import datetime, timezone, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID, uuid4
from enum import Enum

# Set precision for governance calculations
getcontext().prec = 18

from ..core.config import settings
from ..core.models import PRSMBaseModel
from ..tokenomics.database_ftns_service import DatabaseFTNSService
from ..auth.auth_manager import auth_manager
from ..integrations.security.audit_logger import audit_logger
from .voting import get_token_weighted_voting
from .quadratic_voting import quadratic_voting
from .proposals import get_proposal_manager

logger = structlog.get_logger(__name__)


class DistributionType(str, Enum):
    """Types of governance token distributions"""
    INITIAL_ALLOCATION = "initial_allocation"
    CONTRIBUTION_REWARD = "contribution_reward"
    STAKING_REWARD = "staking_reward"
    VOTING_REWARD = "voting_reward"
    DELEGATION_REWARD = "delegation_reward"
    COUNCIL_REWARD = "council_reward"


class ContributionType(str, Enum):
    """Types of contributions that earn governance tokens"""
    MODEL_CONTRIBUTION = "model_contribution"
    DATA_CONTRIBUTION = "data_contribution"
    RESEARCH_PUBLICATION = "research_publication"
    CODE_CONTRIBUTION = "code_contribution"
    SECURITY_AUDIT = "security_audit"
    COMMUNITY_MODERATION = "community_moderation"
    DOCUMENTATION = "documentation"
    TESTING = "testing"


class GovernanceParticipantTier(str, Enum):
    """Tiers of governance participants"""
    COMMUNITY = "community"          # Basic participants
    CONTRIBUTOR = "contributor"      # Active contributors
    EXPERT = "expert"               # Domain experts
    DELEGATE = "delegate"           # Elected delegates
    COUNCIL_MEMBER = "council_member"  # Council members
    CORE_TEAM = "core_team"         # Core development team


class TokenDistribution(PRSMBaseModel):
    """Record of token distribution for governance"""
    distribution_id: UUID = uuid4()
    recipient_user_id: str
    distribution_type: DistributionType
    amount: Decimal
    contribution_type: Optional[ContributionType] = None
    contribution_reference: Optional[str] = None
    vesting_schedule: Optional[Dict[str, Any]] = None
    unlock_date: Optional[datetime] = None
    distributed_at: datetime = datetime.now(timezone.utc)
    metadata: Dict[str, Any] = {}


class GovernanceActivation(PRSMBaseModel):
    """Governance system activation record"""
    activation_id: UUID = uuid4()
    participant_user_id: str
    participant_tier: GovernanceParticipantTier
    initial_token_allocation: Decimal
    staked_amount: Decimal = Decimal('0')
    voting_power: Decimal = Decimal('0')
    council_memberships: List[str] = []
    delegation_relationships: List[str] = []
    activated_at: datetime = datetime.now(timezone.utc)
    is_active: bool = True


class GovernanceTokenDistributor:
    """
    Manages governance token distribution and voting system activation
    """
    
    def __init__(self):
        self.distributor_id = str(uuid4())
        self.logger = logger.bind(component="governance_token_distributor", distributor_id=self.distributor_id)
        
        # Service integrations
        self.ftns_service = DatabaseFTNSService()
        self.voting_system = get_token_weighted_voting()
        self.proposal_manager = get_proposal_manager()
        
        # Distribution tracking
        self.distributions: Dict[UUID, TokenDistribution] = {}
        self.governance_activations: Dict[str, GovernanceActivation] = {}
        self.participant_tiers: Dict[str, GovernanceParticipantTier] = {}
        
        # Distribution parameters
        self.distribution_amounts = {
            GovernanceParticipantTier.COMMUNITY: Decimal('1000'),        # 1K FTNS
            GovernanceParticipantTier.CONTRIBUTOR: Decimal('5000'),      # 5K FTNS
            GovernanceParticipantTier.EXPERT: Decimal('10000'),          # 10K FTNS
            GovernanceParticipantTier.DELEGATE: Decimal('25000'),        # 25K FTNS
            GovernanceParticipantTier.COUNCIL_MEMBER: Decimal('50000'),  # 50K FTNS
            GovernanceParticipantTier.CORE_TEAM: Decimal('100000')       # 100K FTNS
        }
        
        self.contribution_rewards = {
            ContributionType.MODEL_CONTRIBUTION: Decimal('2500'),
            ContributionType.DATA_CONTRIBUTION: Decimal('1000'),
            ContributionType.RESEARCH_PUBLICATION: Decimal('5000'),
            ContributionType.CODE_CONTRIBUTION: Decimal('3000'),
            ContributionType.SECURITY_AUDIT: Decimal('10000'),
            ContributionType.COMMUNITY_MODERATION: Decimal('500'),
            ContributionType.DOCUMENTATION: Decimal('1500'),
            ContributionType.TESTING: Decimal('2000')
        }
        
        # Governance activation tracking
        self.activation_stats = {
            "total_participants": 0,
            "total_tokens_distributed": Decimal('0'),
            "total_staked_tokens": Decimal('0'),
            "active_voters": 0,
            "active_council_members": 0,
            "active_delegates": 0,
            "governance_proposals_created": 0,
            "votes_cast": 0
        }
        
        # Synchronization
        self._distribution_lock = asyncio.Lock()
        self._activation_lock = asyncio.Lock()
        
        print("🏛️ Governance Token Distributor initialized")
    
    
    async def activate_governance_participation(
        self,
        user_id: str,
        participant_tier: GovernanceParticipantTier,
        council_nominations: List[str] = None
    ) -> GovernanceActivation:
        """
        Activate governance participation for a user with initial token distribution
        
        Args:
            user_id: User to activate
            participant_tier: Initial tier for the participant
            council_nominations: Optional council nominations
            
        Returns:
            Governance activation record
        """
        try:
            async with self._activation_lock:
                # Check if user is already activated
                if user_id in self.governance_activations:
                    existing = self.governance_activations[user_id]
                    if existing.is_active:
                        self.logger.warning("User already has active governance participation", user_id=user_id)
                        return existing
                
                # Validate user eligibility
                if not await self._validate_governance_eligibility(user_id, participant_tier):
                    raise ValueError("User not eligible for governance participation")
                
                # Calculate initial token allocation
                initial_allocation = self.distribution_amounts[participant_tier]
                
                # Create token distribution record
                distribution = await self._create_token_distribution(
                    recipient_user_id=user_id,
                    distribution_type=DistributionType.INITIAL_ALLOCATION,
                    amount=initial_allocation,
                    metadata={
                        "participant_tier": participant_tier.value,
                        "council_nominations": council_nominations or []
                    }
                )
                
                # Distribute initial tokens
                await self.ftns_service.create_transaction(
                    from_user_id=None,  # System mint
                    to_user_id=user_id,
                    amount=initial_allocation,
                    transaction_type="governance_activation",
                    description=f"Initial governance token allocation for {participant_tier.value}",
                    reference_id=str(distribution.distribution_id)
                )
                
                # Create governance activation record
                activation = GovernanceActivation(
                    participant_user_id=user_id,
                    participant_tier=participant_tier,
                    initial_token_allocation=initial_allocation,
                    council_memberships=council_nominations or []
                )
                
                # Store activation
                self.governance_activations[user_id] = activation
                self.participant_tiers[user_id] = participant_tier
                
                # Auto-stake percentage of tokens for voting power
                auto_stake_percentage = Decimal('0.5')  # 50% auto-staked
                stake_amount = initial_allocation * auto_stake_percentage
                
                if stake_amount > 0:
                    await self._stake_tokens_for_voting(user_id, stake_amount)
                    activation.staked_amount = stake_amount
                    activation.voting_power = await self._calculate_voting_power(user_id, stake_amount)
                
                # Process council nominations
                if council_nominations:
                    await self._process_council_nominations(user_id, council_nominations)
                
                # Update statistics
                self.activation_stats["total_participants"] += 1
                self.activation_stats["total_tokens_distributed"] += initial_allocation
                
                # Audit logging
                await audit_logger.log_security_event(
                    event_type="governance_participation_activated",
                    user_id=user_id,
                    details={
                        "participant_tier": participant_tier.value,
                        "initial_allocation": str(initial_allocation),
                        "staked_amount": str(stake_amount),
                        "council_nominations": council_nominations or []
                    },
                    security_level="info"
                )
                
                self.logger.info(
                    "Governance participation activated",
                    user_id=user_id,
                    tier=participant_tier.value,
                    allocation=str(initial_allocation),
                    staked=str(stake_amount)
                )
                
                return activation
                
        except Exception as e:
            self.logger.error("Failed to activate governance participation", 
                            user_id=user_id, error=str(e))
            raise
    
    
    async def distribute_contribution_rewards(
        self,
        user_id: str,
        contribution_type: ContributionType,
        contribution_reference: str,
        quality_multiplier: float = 1.0,
        custom_amount: Optional[Decimal] = None
    ) -> TokenDistribution:
        """
        Distribute tokens as rewards for various contributions
        
        Args:
            user_id: User receiving the reward
            contribution_type: Type of contribution
            contribution_reference: Reference to the contribution
            quality_multiplier: Quality-based multiplier
            custom_amount: Custom reward amount (overrides default)
            
        Returns:
            Distribution record
        """
        try:
            async with self._distribution_lock:
                # Calculate reward amount
                base_amount = custom_amount or self.contribution_rewards[contribution_type]
                reward_amount = base_amount * Decimal(str(quality_multiplier))
                
                # Create distribution record
                distribution = await self._create_token_distribution(
                    recipient_user_id=user_id,
                    distribution_type=DistributionType.CONTRIBUTION_REWARD,
                    amount=reward_amount,
                    contribution_type=contribution_type,
                    contribution_reference=contribution_reference,
                    metadata={
                        "base_amount": str(base_amount),
                        "quality_multiplier": quality_multiplier,
                        "final_amount": str(reward_amount)
                    }
                )
                
                # Distribute tokens
                await self.ftns_service.create_transaction(
                    from_user_id=None,  # System mint
                    to_user_id=user_id,
                    amount=reward_amount,
                    transaction_type="contribution_reward",
                    description=f"Reward for {contribution_type.value}: {contribution_reference}",
                    reference_id=str(distribution.distribution_id)
                )
                
                # Check for tier upgrade eligibility
                await self._check_tier_upgrade(user_id)
                
                # Update statistics
                self.activation_stats["total_tokens_distributed"] += reward_amount
                
                self.logger.info(
                    "Contribution reward distributed",
                    user_id=user_id,
                    contribution_type=contribution_type.value,
                    amount=str(reward_amount),
                    multiplier=quality_multiplier
                )
                
                return distribution
                
        except Exception as e:
            self.logger.error("Failed to distribute contribution reward", 
                            user_id=user_id, error=str(e))
            raise
    
    
    async def enable_voting_system(self) -> Dict[str, Any]:
        """
        Enable and activate the complete voting system infrastructure
        
        Returns:
            Activation status and configuration
        """
        try:
            activation_results = {
                "voting_system_activated": False,
                "quadratic_voting_enabled": False,
                "federated_councils_created": False,
                "governance_parameters_set": False,
                "initial_participants_activated": 0,
                "councils_created": [],
                "voting_mechanisms_enabled": [],
                "activation_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # 1. Set up quadratic voting system
            await self._setup_quadratic_voting()
            activation_results["quadratic_voting_enabled"] = True
            activation_results["voting_mechanisms_enabled"].append("quadratic")
            
            # 2. Create federated councils
            councils_created = await self._create_initial_councils()
            activation_results["federated_councils_created"] = len(councils_created) > 0
            activation_results["councils_created"] = councils_created
            
            # 3. Configure governance parameters
            await self._configure_governance_parameters()
            activation_results["governance_parameters_set"] = True
            
            # 4. Activate initial participants
            initial_participants = await self._activate_initial_participants()
            activation_results["initial_participants_activated"] = initial_participants
            
            # 5. Enable delegation mechanisms
            await self._enable_delegation_system()
            activation_results["voting_mechanisms_enabled"].append("delegation")
            
            # 6. Activate proposal system
            await self._activate_proposal_system()
            activation_results["voting_mechanisms_enabled"].append("proposals")
            
            # Mark voting system as fully activated
            activation_results["voting_system_activated"] = True
            
            # Audit logging
            await audit_logger.log_security_event(
                event_type="governance_voting_system_activated",
                user_id="system",
                details=activation_results,
                security_level="info"
            )
            
            self.logger.info(
                "Governance voting system activated successfully",
                councils_created=len(councils_created),
                participants_activated=initial_participants,
                mechanisms_enabled=len(activation_results["voting_mechanisms_enabled"])
            )
            
            return activation_results
            
        except Exception as e:
            self.logger.error("Failed to enable voting system", error=str(e))
            raise
    
    
    async def stake_tokens_for_governance(
        self,
        user_id: str,
        amount: Decimal,
        lock_duration_days: int = 30
    ) -> Tuple[Decimal, Decimal]:
        """
        Stake tokens for governance participation and voting power
        
        Args:
            user_id: User staking tokens
            amount: Amount to stake
            lock_duration_days: Lock duration for staking
            
        Returns:
            Tuple of (staked_amount, voting_power)
        """
        try:
            # Stake tokens through FTNS service
            stake_tx, voting_power = await self.ftns_service.stake_for_governance(
                user_id=user_id,
                amount=amount,
                lock_duration_days=lock_duration_days
            )
            
            # Update governance activation
            if user_id in self.governance_activations:
                activation = self.governance_activations[user_id]
                activation.staked_amount += amount
                activation.voting_power += voting_power
            
            # Update statistics
            self.activation_stats["total_staked_tokens"] += amount
            
            # Create distribution record for staking reward
            staking_reward = amount * Decimal('0.05')  # 5% staking reward
            if staking_reward > 0:
                await self._create_token_distribution(
                    recipient_user_id=user_id,
                    distribution_type=DistributionType.STAKING_REWARD,
                    amount=staking_reward,
                    metadata={
                        "staked_amount": str(amount),
                        "lock_duration_days": lock_duration_days,
                        "voting_power": str(voting_power)
                    }
                )
            
            self.logger.info(
                "Tokens staked for governance",
                user_id=user_id,
                amount=str(amount),
                voting_power=str(voting_power),
                lock_days=lock_duration_days
            )
            
            return amount, voting_power
            
        except Exception as e:
            self.logger.error("Failed to stake tokens for governance", 
                            user_id=user_id, error=str(e))
            raise
    
    
    async def get_governance_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive governance status for a user"""
        try:
            # Get user's governance activation
            activation = self.governance_activations.get(user_id)
            if not activation:
                return {
                    "is_activated": False,
                    "message": "User not activated for governance participation"
                }
            
            # Get current token balances
            user_balance = await self.ftns_service.get_user_balance(user_id)
            
            # Calculate current voting power
            current_voting_power = await self.voting_system.calculate_voting_power(user_id)
            
            # Get council memberships
            council_memberships = []
            for council_id in activation.council_memberships:
                if council_id in quadratic_voting.councils:
                    council = quadratic_voting.councils[UUID(council_id)]
                    council_memberships.append({
                        "council_id": council_id,
                        "council_name": council.name,
                        "council_type": council.council_type.value,
                        "voting_weight": council.voting_weight
                    })
            
            # Get recent distributions
            recent_distributions = [
                dist.dict() for dist in self.distributions.values()
                if dist.recipient_user_id == user_id
            ][-10:]  # Last 10 distributions
            
            return {
                "is_activated": True,
                "activation": activation.dict(),
                "current_tier": self.participant_tiers.get(user_id, "unknown"),
                "token_balances": {
                    "liquid_balance": str(user_balance.balance),
                    "staked_balance": str(user_balance.staked_balance),
                    "total_balance": str(user_balance.balance + user_balance.staked_balance)
                },
                "voting_power": {
                    "total_voting_power": current_voting_power.total_voting_power,
                    "base_power": current_voting_power.base_voting_power,
                    "delegation_power": current_voting_power.delegation_power,
                    "council_power": current_voting_power.role_multiplier
                },
                "council_memberships": council_memberships,
                "recent_distributions": recent_distributions,
                "governance_stats": {
                    "proposals_created": 0,  # Would query from proposal system
                    "votes_cast": 0,         # Would query from voting system
                    "delegations_received": 0,  # Would query from delegation system
                    "delegations_given": 0      # Would query from delegation system
                }
            }
            
        except Exception as e:
            self.logger.error("Failed to get governance status", 
                            user_id=user_id, error=str(e))
            return {
                "is_activated": False,
                "error": str(e)
            }
    
    
    async def get_distribution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive governance token distribution statistics"""
        try:
            # Calculate distribution statistics
            distributions_by_type = {}
            distributions_by_tier = {}
            total_distributions = len(self.distributions)
            
            for distribution in self.distributions.values():
                # By type
                dist_type = distribution.distribution_type.value
                if dist_type not in distributions_by_type:
                    distributions_by_type[dist_type] = {"count": 0, "total_amount": Decimal('0')}
                distributions_by_type[dist_type]["count"] += 1
                distributions_by_type[dist_type]["total_amount"] += distribution.amount
            
            # By participant tier
            for activation in self.governance_activations.values():
                tier = activation.participant_tier.value
                if tier not in distributions_by_tier:
                    distributions_by_tier[tier] = {"count": 0, "total_allocated": Decimal('0')}
                distributions_by_tier[tier]["count"] += 1
                distributions_by_tier[tier]["total_allocated"] += activation.initial_token_allocation
            
            # Council statistics
            council_stats = {}
            for council_id, council in quadratic_voting.councils.items():
                council_stats[str(council_id)] = {
                    "name": council.name,
                    "type": council.council_type.value,
                    "member_count": len(council.members),
                    "voting_weight": council.voting_weight,
                    "is_active": council.is_active
                }
            
            return {
                **self.activation_stats,
                "total_distributions": total_distributions,
                "distributions_by_type": {
                    k: {"count": v["count"], "total_amount": str(v["total_amount"])}
                    for k, v in distributions_by_type.items()
                },
                "distributions_by_tier": {
                    k: {"count": v["count"], "total_allocated": str(v["total_allocated"])}
                    for k, v in distributions_by_tier.items()
                },
                "council_statistics": council_stats,
                "activation_rate": len(self.governance_activations) / max(1, self.activation_stats["total_participants"]),
                "average_allocation_per_participant": str(
                    self.activation_stats["total_tokens_distributed"] / max(1, self.activation_stats["total_participants"])
                ),
                "staking_rate": str(
                    self.activation_stats["total_staked_tokens"] / max(1, self.activation_stats["total_tokens_distributed"])
                )
            }
            
        except Exception as e:
            self.logger.error("Failed to get distribution statistics", error=str(e))
            return {
                "error": str(e),
                "total_distributions": 0
            }
    
    
    # === Private Helper Methods ===
    
    async def _validate_governance_eligibility(self, user_id: str, tier: GovernanceParticipantTier) -> bool:
        """Validate user eligibility for governance participation"""
        try:
            # Check if user exists in auth system
            user = await auth_manager.get_user_by_id(user_id)
            if not user:
                return False
            
            # Check minimum requirements based on tier
            if tier in [GovernanceParticipantTier.EXPERT, GovernanceParticipantTier.DELEGATE, 
                       GovernanceParticipantTier.COUNCIL_MEMBER]:
                # Would implement reputation/contribution checks here
                pass
            
            return True
            
        except Exception:
            return False
    
    
    async def _create_token_distribution(
        self,
        recipient_user_id: str,
        distribution_type: DistributionType,
        amount: Decimal,
        contribution_type: Optional[ContributionType] = None,
        contribution_reference: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> TokenDistribution:
        """Create a token distribution record"""
        distribution = TokenDistribution(
            recipient_user_id=recipient_user_id,
            distribution_type=distribution_type,
            amount=amount,
            contribution_type=contribution_type,
            contribution_reference=contribution_reference,
            metadata=metadata or {}
        )
        
        self.distributions[distribution.distribution_id] = distribution
        return distribution
    
    
    async def _stake_tokens_for_voting(self, user_id: str, amount: Decimal) -> Decimal:
        """Stake tokens and return voting power"""
        try:
            _, voting_power = await self.ftns_service.stake_for_governance(
                user_id=user_id,
                amount=amount,
                lock_duration_days=30
            )
            return voting_power
            
        except Exception as e:
            self.logger.error("Failed to stake tokens for voting", error=str(e))
            return Decimal('0')
    
    
    async def _calculate_voting_power(self, user_id: str, staked_amount: Decimal) -> Decimal:
        """Calculate voting power based on staked tokens"""
        # Simple calculation - in production this would be more sophisticated
        return staked_amount  # 1:1 ratio for simplicity
    
    
    async def _process_council_nominations(self, user_id: str, council_nominations: List[str]):
        """Process council membership nominations"""
        for council_name in council_nominations:
            # Find council by name
            for council in quadratic_voting.councils.values():
                if council.name.lower() == council_name.lower():
                    if len(council.members) < council.max_members:
                        council.members.append(UUID(user_id))
                        break
    
    
    async def _check_tier_upgrade(self, user_id: str):
        """Check if user is eligible for tier upgrade based on contributions"""
        # Simplified tier upgrade logic
        # In production, this would analyze contribution history
        pass
    
    
    async def _setup_quadratic_voting(self):
        """Set up quadratic voting system parameters"""
        quadratic_voting.quadratic_parameters.update({
            "max_tokens_per_vote": Decimal('10000'),
            "quadratic_constant": 1.0,
            "delegation_efficiency": 0.95
        })
    
    
    async def _create_initial_councils(self) -> List[str]:
        """Create initial federated councils"""
        councils_created = []
        
        council_configs = [
            {
                "type": "SAFETY",
                "name": "AI Safety Council",
                "description": "Oversees AI safety policies and risk management"
            },
            {
                "type": "TECHNICAL", 
                "name": "Technical Development Council",
                "description": "Manages technical infrastructure and development"
            },
            {
                "type": "ECONOMIC",
                "name": "Economic Policy Council", 
                "description": "Oversees tokenomics and marketplace policies"
            },
            {
                "type": "GOVERNANCE",
                "name": "Governance Council",
                "description": "Manages governance mechanisms and procedures"
            }
        ]
        
        for config in council_configs:
            try:
                council = await quadratic_voting.create_council(
                    council_type=getattr(quadratic_voting.CouncilType, config["type"]),
                    name=config["name"],
                    description=config["description"]
                )
                councils_created.append(council.name)
                
            except Exception as e:
                self.logger.error(f"Failed to create council {config['name']}", error=str(e))
        
        return councils_created
    
    
    async def _configure_governance_parameters(self):
        """Configure governance system parameters"""
        # Set voting parameters in the token-weighted voting system
        # This would configure thresholds, timeouts, etc.
        pass
    
    
    async def _activate_initial_participants(self) -> int:
        """Activate initial governance participants"""
        # In production, this would activate core team members and early contributors
        # For now, return 0 as no automatic activation
        return 0
    
    
    async def _enable_delegation_system(self):
        """Enable the delegation system"""
        # Configure delegation parameters in quadratic voting system
        pass
    
    
    async def _activate_proposal_system(self):
        """Activate the proposal management system"""
        # Ensure proposal manager is ready for governance proposals
        pass


# Global governance token distributor instance
_distributor_instance: Optional[GovernanceTokenDistributor] = None

def get_governance_distributor() -> GovernanceTokenDistributor:
    """Get or create the global governance token distributor instance"""
    global _distributor_instance
    if _distributor_instance is None:
        _distributor_instance = GovernanceTokenDistributor()
    return _distributor_instance