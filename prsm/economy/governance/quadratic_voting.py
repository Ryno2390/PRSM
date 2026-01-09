"""
Quadratic Voting and Federated Councils
=======================================

Advanced governance mechanisms to prevent governance capture while enabling
efficient decision-making across diverse stakeholder groups.

Key Features:
- Quadratic voting to dampen influence of large token holders
- Federated councils representing different domains and regions
- Rotational delegation based on contribution types, not just stake
- Multi-sig safety nets and specialized governance zones
- Democratic participation while preventing plutocracy
"""

import asyncio
import math
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, Field


class VotingMechanism(str, Enum):
    """Types of voting mechanisms available"""
    QUADRATIC = "quadratic"              # Square root of tokens
    LINEAR = "linear"                    # One token one vote
    DELEGATED = "delegated"              # Delegation to representatives
    COUNCIL_WEIGHTED = "council_weighted"  # Council-based with weights
    HYBRID = "hybrid"                    # Combination of mechanisms


class CouncilType(str, Enum):
    """Types of federated councils"""
    DOMAIN_EXPERT = "domain_expert"      # Scientific domain specialists
    GEOGRAPHIC = "geographic"            # Regional representation
    INSTITUTIONAL_TIER = "institutional_tier"  # Hobbyist/startup/enterprise/frontier
    TECHNICAL = "technical"              # Technical infrastructure and development
    SAFETY = "safety"                    # AI safety and risk management
    ECONOMIC = "economic"                # Tokenomics and marketplace
    GOVERNANCE = "governance"            # Governance and policy


class ProposalCategory(str, Enum):
    """Categories of governance proposals"""
    PROTOCOL_UPGRADE = "protocol_upgrade"      # Core protocol changes
    ECONOMIC_POLICY = "economic_policy"        # Tokenomics adjustments
    SAFETY_POLICY = "safety_policy"           # Safety and security policies
    GOVERNANCE_RULE = "governance_rule"        # Governance mechanism changes
    NETWORK_PARAMETER = "network_parameter"    # Network configuration
    EMERGENCY_ACTION = "emergency_action"      # Emergency interventions
    COUNCIL_FORMATION = "council_formation"    # Creating/modifying councils


@dataclass
class VotingPower:
    """Calculated voting power for different mechanisms"""
    linear_power: float         # Raw token-based power
    quadratic_power: float      # Quadratic (sqrt) adjusted power
    delegated_power: float      # Power from delegation
    council_power: float        # Power from council membership
    total_effective_power: float  # Final combined power


class Council(BaseModel):
    """Federated council representing specific stakeholder groups"""
    council_id: UUID = Field(default_factory=uuid4)
    council_type: CouncilType
    name: str
    description: str
    
    # Membership
    members: List[UUID] = Field(default_factory=list)
    max_members: int = 21  # Odd number for tie-breaking
    
    # Governance parameters
    voting_weight: float = Field(ge=0.0, le=1.0)  # Weight in overall governance
    proposal_categories: List[ProposalCategory] = Field(default_factory=list)
    required_supermajority: float = Field(ge=0.5, le=1.0, default=0.67)
    
    # Terms and rotation
    term_length_days: int = 365  # 1 year terms
    max_consecutive_terms: int = 2
    rotation_schedule: Dict[str, datetime] = Field(default_factory=dict)
    
    # Creation and status
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


class DelegationRecord(BaseModel):
    """Record of voting power delegation"""
    delegation_id: UUID = Field(default_factory=uuid4)
    delegator_id: UUID  # Who is delegating
    delegate_id: UUID   # Who receives the delegation
    
    # Delegation scope
    proposal_categories: List[ProposalCategory] = Field(default_factory=list)  # Empty = all
    councils: List[UUID] = Field(default_factory=list)  # Empty = all councils
    
    # Power delegation
    delegated_power_percentage: float = Field(ge=0.0, le=100.0)
    
    # Temporal aspects
    delegation_start: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delegation_end: Optional[datetime] = None
    is_active: bool = True


class QuadraticVote(BaseModel):
    """Individual vote using quadratic voting mechanism"""
    vote_id: UUID = Field(default_factory=uuid4)
    voter_id: UUID
    proposal_id: UUID
    
    # Vote details
    choice: str  # "approve", "reject", "abstain", or multiple choice option
    tokens_committed: Decimal  # Tokens committed to this vote
    quadratic_power: float     # Calculated quadratic voting power
    
    # Context
    voting_mechanism: VotingMechanism
    council_id: Optional[UUID] = None  # If voted through council
    delegation_source: Optional[UUID] = None  # If vote via delegation
    
    # Metadata
    reasoning: Optional[str] = None
    cast_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QuadraticVotingSystem:
    """
    Advanced governance system using quadratic voting and federated councils
    to ensure democratic participation while preventing plutocratic capture.
    """
    
    def __init__(self):
        # Council management
        self.councils: Dict[UUID, Council] = {}
        self.council_memberships: Dict[UUID, Set[UUID]] = {}  # user_id -> set of council_ids
        
        # Delegation system
        self.active_delegations: Dict[UUID, List[DelegationRecord]] = {}  # delegator_id -> delegations
        self.delegation_chains: Dict[UUID, Dict[UUID, float]] = {}  # user_id -> delegate_id -> power
        
        # Voting records
        self.votes: Dict[UUID, List[QuadraticVote]] = {}  # proposal_id -> votes
        self.voting_power_cache: Dict[UUID, VotingPower] = {}  # user_id -> current power
        
        # System parameters
        self.quadratic_parameters = {
            "max_tokens_per_vote": Decimal('10000'),  # Maximum tokens committable to single vote
            "quadratic_constant": 1.0,                # Multiplier for quadratic calculation
            "delegation_efficiency": 0.95,            # 5% loss in delegation to prevent infinite chains
        }
        
        # Default councils configuration
        self.default_councils_config = {
            CouncilType.SAFETY: {"weight": 0.25, "max_members": 15},
            CouncilType.TECHNICAL: {"weight": 0.20, "max_members": 21},
            CouncilType.ECONOMIC: {"weight": 0.20, "max_members": 17},
            CouncilType.DOMAIN_EXPERT: {"weight": 0.15, "max_members": 25},
            CouncilType.GEOGRAPHIC: {"weight": 0.10, "max_members": 19},
            CouncilType.GOVERNANCE: {"weight": 0.10, "max_members": 13}
        }
        
        print("🗳️ Quadratic Voting System initialized")
        print("   - Federated councils framework active")
        print("   - Anti-plutocracy mechanisms enabled")
        print("   - Delegation and rotation systems ready")
    
    async def create_council(self,
                           council_type: CouncilType,
                           name: str,
                           description: str,
                           initial_members: List[UUID] = None,
                           custom_parameters: Dict[str, Any] = None) -> Council:
        """
        Create a new federated council for specialized governance.
        """
        
        # Get default parameters for council type
        defaults = self.default_councils_config.get(council_type, {})
        
        # Apply custom parameters
        voting_weight = custom_parameters.get("voting_weight", defaults.get("weight", 0.1))
        max_members = custom_parameters.get("max_members", defaults.get("max_members", 21))
        
        # Create council
        council = Council(
            council_type=council_type,
            name=name,
            description=description,
            members=initial_members or [],
            max_members=max_members,
            voting_weight=voting_weight,
            proposal_categories=custom_parameters.get("proposal_categories", []),
            required_supermajority=custom_parameters.get("supermajority", 0.67)
        )
        
        # Register council
        self.councils[council.council_id] = council
        
        # Update memberships
        for member_id in council.members:
            if member_id not in self.council_memberships:
                self.council_memberships[member_id] = set()
            self.council_memberships[member_id].add(council.council_id)
        
        print(f"🏛️ Council created: {name}")
        print(f"   - Type: {council_type}")
        print(f"   - Members: {len(council.members)}")
        print(f"   - Voting weight: {voting_weight:.3f}")
        
        return council
    
    async def delegate_voting_power(self,
                                  delegator_id: UUID,
                                  delegate_id: UUID,
                                  power_percentage: float,
                                  scope_categories: List[ProposalCategory] = None,
                                  scope_councils: List[UUID] = None,
                                  duration_days: Optional[int] = None) -> DelegationRecord:
        """
        Delegate voting power to another participant with optional scope and duration.
        """
        
        # Validate delegation
        if delegator_id == delegate_id:
            raise ValueError("Cannot delegate to self")
        
        if power_percentage <= 0 or power_percentage > 100:
            raise ValueError("Power percentage must be between 0 and 100")
        
        # Check for circular delegation
        if await self._would_create_circular_delegation(delegator_id, delegate_id):
            raise ValueError("Delegation would create circular dependency")
        
        # Create delegation record
        delegation_end = None
        if duration_days:
            delegation_end = datetime.now(timezone.utc) + timedelta(days=duration_days)
        
        delegation = DelegationRecord(
            delegator_id=delegator_id,
            delegate_id=delegate_id,
            proposal_categories=scope_categories or [],
            councils=scope_councils or [],
            delegated_power_percentage=power_percentage,
            delegation_end=delegation_end
        )
        
        # Store delegation
        if delegator_id not in self.active_delegations:
            self.active_delegations[delegator_id] = []
        
        self.active_delegations[delegator_id].append(delegation)
        
        # Update delegation chains
        await self._update_delegation_chains()
        
        print(f"🔄 Voting power delegated: {delegator_id} → {delegate_id}")
        print(f"   - Power: {power_percentage}%")
        print(f"   - Scope: {len(scope_categories) if scope_categories else 'all'} categories")
        
        return delegation
    
    async def calculate_voting_power(self,
                                   user_id: UUID,
                                   token_balance: Decimal,
                                   proposal_category: ProposalCategory,
                                   council_context: Optional[UUID] = None) -> VotingPower:
        """
        Calculate comprehensive voting power using multiple mechanisms.
        """
        
        # Linear power (direct token ownership)
        linear_power = float(token_balance)
        
        # Quadratic power (square root of tokens)
        quadratic_power = math.sqrt(linear_power) * self.quadratic_parameters["quadratic_constant"]
        
        # Delegated power (power received from others)
        delegated_power = await self._calculate_delegated_power(
            user_id, proposal_category, council_context
        )
        
        # Council power (additional power from council membership)
        council_power = await self._calculate_council_power(
            user_id, proposal_category, council_context
        )
        
        # Calculate total effective power
        # Use quadratic as base, add delegated and council power
        total_effective_power = quadratic_power + delegated_power + council_power
        
        voting_power = VotingPower(
            linear_power=linear_power,
            quadratic_power=quadratic_power,
            delegated_power=delegated_power,
            council_power=council_power,
            total_effective_power=total_effective_power
        )
        
        # Cache for performance
        cache_key = f"{user_id}_{proposal_category}_{council_context}"
        self.voting_power_cache[cache_key] = voting_power
        
        return voting_power
    
    async def cast_quadratic_vote(self,
                                voter_id: UUID,
                                proposal_id: UUID,
                                choice: str,
                                tokens_committed: Decimal,
                                voting_context: Dict[str, Any] = None) -> QuadraticVote:
        """
        Cast a vote using quadratic voting mechanism.
        """
        
        # Validate token commitment
        if tokens_committed > self.quadratic_parameters["max_tokens_per_vote"]:
            raise ValueError(f"Cannot commit more than {self.quadratic_parameters['max_tokens_per_vote']} tokens")
        
        # Calculate quadratic voting power
        quadratic_power = math.sqrt(float(tokens_committed)) * self.quadratic_parameters["quadratic_constant"]
        
        # Create vote record
        vote = QuadraticVote(
            voter_id=voter_id,
            proposal_id=proposal_id,
            choice=choice,
            tokens_committed=tokens_committed,
            quadratic_power=quadratic_power,
            voting_mechanism=VotingMechanism.QUADRATIC,
            council_id=voting_context.get("council_id") if voting_context else None,
            delegation_source=voting_context.get("delegation_source") if voting_context else None,
            reasoning=voting_context.get("reasoning") if voting_context else None
        )
        
        # Store vote
        if proposal_id not in self.votes:
            self.votes[proposal_id] = []
        
        self.votes[proposal_id].append(vote)
        
        print(f"🗳️ Quadratic vote cast: {voter_id}")
        print(f"   - Proposal: {proposal_id}")
        print(f"   - Choice: {choice}")
        print(f"   - Tokens committed: {tokens_committed}")
        print(f"   - Quadratic power: {quadratic_power:.3f}")
        
        return vote
    
    async def tally_votes(self,
                         proposal_id: UUID,
                         proposal_category: ProposalCategory) -> Dict[str, Any]:
        """
        Tally votes for a proposal using quadratic and council-weighted mechanisms.
        """
        
        proposal_votes = self.votes.get(proposal_id, [])
        
        if not proposal_votes:
            return {
                "total_votes": 0,
                "vote_distribution": {},
                "quadratic_results": {},
                "council_results": {},
                "final_result": "no_votes"
            }
        
        # Tally quadratic votes
        quadratic_tally = {}
        total_quadratic_power = 0
        
        for vote in proposal_votes:
            choice = vote.choice
            power = vote.quadratic_power
            
            quadratic_tally[choice] = quadratic_tally.get(choice, 0) + power
            total_quadratic_power += power
        
        # Calculate quadratic percentages
        quadratic_percentages = {
            choice: (power / total_quadratic_power) * 100
            for choice, power in quadratic_tally.items()
        }
        
        # Tally council-weighted votes
        council_tally = await self._tally_council_votes(proposal_votes, proposal_category)
        
        # Determine final result
        final_result = max(quadratic_percentages.items(), key=lambda x: x[1])[0]
        
        # Check for supermajority requirements
        winning_percentage = quadratic_percentages[final_result]
        requires_supermajority = proposal_category in [
            ProposalCategory.PROTOCOL_UPGRADE,
            ProposalCategory.GOVERNANCE_RULE,
            ProposalCategory.EMERGENCY_ACTION
        ]
        
        if requires_supermajority and winning_percentage < 67:
            final_result = "insufficient_supermajority"
        
        return {
            "total_votes": len(proposal_votes),
            "total_quadratic_power": total_quadratic_power,
            "vote_distribution": quadratic_tally,
            "quadratic_percentages": quadratic_percentages,
            "council_results": council_tally,
            "final_result": final_result,
            "winning_percentage": winning_percentage,
            "supermajority_required": requires_supermajority,
            "supermajority_achieved": winning_percentage >= 67 if requires_supermajority else True
        }
    
    async def rotate_council_membership(self,
                                      council_id: UUID,
                                      rotation_percentage: float = 0.33) -> Dict[str, Any]:
        """
        Rotate council membership to prevent entrenchment and ensure fresh perspectives.
        """
        
        if council_id not in self.councils:
            raise ValueError(f"Council {council_id} not found")
        
        council = self.councils[council_id]
        current_members = council.members.copy()
        
        # Determine how many members to rotate
        rotation_count = max(1, int(len(current_members) * rotation_percentage))
        
        # Select members for rotation (longest serving first)
        # In a real implementation, this would check service terms
        members_to_rotate = current_members[:rotation_count]
        
        # Remove rotating members
        for member_id in members_to_rotate:
            council.members.remove(member_id)
            self.council_memberships[member_id].discard(council_id)
        
        # Add rotation timestamp
        council.rotation_schedule[datetime.now(timezone.utc).isoformat()] = {
            "rotated_out": members_to_rotate,
            "rotation_count": rotation_count
        }
        
        print(f"🔄 Council rotation completed: {council.name}")
        print(f"   - Members rotated: {rotation_count}")
        print(f"   - New size: {len(council.members)}")
        
        return {
            "council_id": council_id,
            "members_rotated_out": members_to_rotate,
            "current_member_count": len(council.members),
            "available_positions": council.max_members - len(council.members),
            "next_rotation_due": datetime.now(timezone.utc) + timedelta(days=council.term_length_days)
        }
    
    async def _would_create_circular_delegation(self,
                                               delegator_id: UUID,
                                               delegate_id: UUID) -> bool:
        """Check if delegation would create circular dependency"""
        
        # Traverse delegation chain from delegate
        visited = set()
        current = delegate_id
        
        while current and current not in visited:
            visited.add(current)
            
            # If we reach the original delegator, we have a circle
            if current == delegator_id:
                return True
            
            # Find next in chain
            next_delegate = None
            for delegation in self.active_delegations.get(current, []):
                if delegation.is_active:
                    next_delegate = delegation.delegate_id
                    break
            
            current = next_delegate
        
        return False
    
    async def _update_delegation_chains(self):
        """Update delegation chain calculations for efficient power computation"""
        
        # Clear existing chains
        self.delegation_chains.clear()
        
        # Build chains for each user
        for delegator_id, delegations in self.active_delegations.items():
            for delegation in delegations:
                if not delegation.is_active:
                    continue
                
                # Skip expired delegations
                if delegation.delegation_end and delegation.delegation_end < datetime.now(timezone.utc):
                    delegation.is_active = False
                    continue
                
                # Add to chains
                if delegator_id not in self.delegation_chains:
                    self.delegation_chains[delegator_id] = {}
                
                delegate_id = delegation.delegate_id
                power_fraction = delegation.delegated_power_percentage / 100.0
                
                # Apply delegation efficiency loss
                effective_power = power_fraction * self.quadratic_parameters["delegation_efficiency"]
                
                self.delegation_chains[delegator_id][delegate_id] = effective_power
    
    async def _calculate_delegated_power(self,
                                       user_id: UUID,
                                       proposal_category: ProposalCategory,
                                       council_context: Optional[UUID]) -> float:
        """Calculate power delegated to this user"""
        
        delegated_power = 0.0
        
        # Find all delegations to this user
        for delegator_id, delegations in self.active_delegations.items():
            for delegation in delegations:
                if delegation.delegate_id != user_id or not delegation.is_active:
                    continue
                
                # Check if delegation applies to this context
                if delegation.proposal_categories and proposal_category not in delegation.proposal_categories:
                    continue
                
                if delegation.councils and council_context not in delegation.councils:
                    continue
                
                # Add delegated power (would need delegator's token balance in real implementation)
                # For now, use a placeholder calculation
                delegated_power += delegation.delegated_power_percentage * 0.1  # Placeholder
        
        return delegated_power
    
    async def _calculate_council_power(self,
                                     user_id: UUID,
                                     proposal_category: ProposalCategory,
                                     council_context: Optional[UUID]) -> float:
        """Calculate additional power from council membership"""
        
        council_power = 0.0
        
        # Get user's council memberships
        user_councils = self.council_memberships.get(user_id, set())
        
        for council_id in user_councils:
            council = self.councils.get(council_id)
            if not council or not council.is_active:
                continue
            
            # Check if council has authority over this proposal category
            if council.proposal_categories and proposal_category not in council.proposal_categories:
                continue
            
            # If specific council context, only count that council
            if council_context and council_id != council_context:
                continue
            
            # Add council voting weight divided by number of members
            member_power = council.voting_weight / len(council.members) if council.members else 0
            council_power += member_power * 100  # Scale appropriately
        
        return council_power
    
    async def _tally_council_votes(self,
                                 votes: List[QuadraticVote],
                                 proposal_category: ProposalCategory) -> Dict[str, Any]:
        """Tally votes with council weighting"""
        
        council_tally = {}
        
        # Group votes by council
        council_votes = {}
        for vote in votes:
            if vote.council_id:
                if vote.council_id not in council_votes:
                    council_votes[vote.council_id] = []
                council_votes[vote.council_id].append(vote)
        
        # Tally each council separately
        for council_id, council_vote_list in council_votes.items():
            council = self.councils.get(council_id)
            if not council:
                continue
            
            # Tally votes within council
            council_choice_tally = {}
            for vote in council_vote_list:
                choice = vote.choice
                council_choice_tally[choice] = council_choice_tally.get(choice, 0) + vote.quadratic_power
            
            # Determine council's majority choice
            if council_choice_tally:
                council_majority = max(council_choice_tally.items(), key=lambda x: x[1])[0]
                council_tally[council_id] = {
                    "choice": council_majority,
                    "weight": council.voting_weight,
                    "member_votes": len(council_vote_list),
                    "vote_distribution": council_choice_tally
                }
        
        return council_tally


# Global quadratic voting system instance
quadratic_voting = QuadraticVotingSystem()