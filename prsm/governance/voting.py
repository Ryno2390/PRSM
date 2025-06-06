"""
PRSM Token-Weighted Voting System

Implements comprehensive token-weighted voting infrastructure including
proposal creation, voting power calculation, and governance coordination.

Based on execution_plan.md Phase 3, Week 17-18 requirements.
"""

import asyncio
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from enum import Enum
import structlog

# Set precision for voting calculations
getcontext().prec = 18

from ..core.config import settings
from ..core.models import GovernanceProposal, Vote, PRSMBaseModel
from ..tokenomics.ftns_service import ftns_service
from ..safety.monitor import SafetyMonitor

logger = structlog.get_logger()

# === Governance Configuration ===

# Voting power parameters
MIN_VOTING_BALANCE = float(getattr(settings, "GOVERNANCE_MIN_VOTING_BALANCE", 100.0))  # 100 FTNS minimum
VOTING_POWER_CURVE = float(getattr(settings, "GOVERNANCE_VOTING_POWER_CURVE", 0.5))  # Square root curve
MAX_VOTING_POWER_RATIO = float(getattr(settings, "GOVERNANCE_MAX_VOTING_POWER_RATIO", 100.0))  # Max 100x difference

# Proposal parameters
PROPOSAL_SUBMISSION_FEE = float(getattr(settings, "GOVERNANCE_PROPOSAL_FEE", 1000.0))  # 1000 FTNS
MIN_PROPOSAL_SUPPORT = int(getattr(settings, "GOVERNANCE_MIN_PROPOSAL_SUPPORT", 10))  # 10 supporters
PROPOSAL_VALIDITY_PERIOD = int(getattr(settings, "GOVERNANCE_PROPOSAL_VALIDITY_DAYS", 14))  # 14 days

# Voting thresholds
QUORUM_PERCENTAGE = float(getattr(settings, "GOVERNANCE_QUORUM_PERCENTAGE", 0.2))  # 20% participation
APPROVAL_THRESHOLD = float(getattr(settings, "GOVERNANCE_APPROVAL_THRESHOLD", 0.6))  # 60% approval
SUPERMAJORITY_THRESHOLD = float(getattr(settings, "GOVERNANCE_SUPERMAJORITY_THRESHOLD", 0.75))  # 75% for critical proposals

# Term limits
GOVERNANCE_TERM_LENGTH_DAYS = int(getattr(settings, "GOVERNANCE_TERM_LENGTH_DAYS", 365))  # 1 year terms
MAX_CONSECUTIVE_TERMS = int(getattr(settings, "GOVERNANCE_MAX_CONSECUTIVE_TERMS", 2))  # 2 consecutive terms max


class ProposalCategory(str, Enum):
    """Categories of governance proposals"""
    CONSTITUTIONAL = "constitutional"  # Core system changes
    ECONOMIC = "economic"  # Token economics, fees, rewards
    TECHNICAL = "technical"  # System upgrades, features
    SAFETY = "safety"  # Safety policies, security measures
    OPERATIONAL = "operational"  # Day-to-day operations
    COMMUNITY = "community"  # Community programs, initiatives
    EMERGENCY = "emergency"  # Emergency measures


class VotingPeriod(str, Enum):
    """Voting period types"""
    STANDARD = "standard"  # 7 days
    EXTENDED = "extended"  # 14 days
    EXPEDITED = "expedited"  # 3 days
    EMERGENCY = "emergency"  # 24 hours


class GovernanceRole(str, Enum):
    """Governance roles"""
    DELEGATE = "delegate"
    COMMITTEE_MEMBER = "committee_member"
    PROPOSAL_REVIEWER = "proposal_reviewer"
    EMERGENCY_RESPONDER = "emergency_responder"


class VotingPowerCalculation(PRSMBaseModel):
    """Voting power calculation result"""
    voter_id: str
    token_balance: float
    base_voting_power: float
    role_multiplier: float
    reputation_bonus: float
    delegation_power: float
    total_voting_power: float
    calculation_time: datetime = datetime.now(timezone.utc)


class DelegationRecord(PRSMBaseModel):
    """Vote delegation record"""
    delegation_id: UUID = uuid4()
    delegator_id: str
    delegate_id: str
    scope: str  # "all", "category", "specific"
    category_filter: Optional[ProposalCategory] = None
    proposal_filter: Optional[UUID] = None
    delegation_power: float
    created_at: datetime = datetime.now(timezone.utc)
    expires_at: Optional[datetime] = None
    is_active: bool = True


class GovernanceTerm(PRSMBaseModel):
    """Governance role term"""
    term_id: UUID = uuid4()
    holder_id: str
    role: GovernanceRole
    start_date: datetime
    end_date: datetime
    term_number: int  # Consecutive term count
    performance_metrics: Dict[str, Any] = {}
    is_active: bool = True


class VotingResults(PRSMBaseModel):
    """Comprehensive voting results"""
    proposal_id: UUID
    total_eligible_voting_power: float
    total_votes_cast: int
    total_voting_power_used: float
    participation_rate: float
    
    # Vote breakdown
    votes_for: int = 0
    votes_against: int = 0
    votes_abstain: int = 0
    
    # Power breakdown
    voting_power_for: float = 0.0
    voting_power_against: float = 0.0
    voting_power_abstain: float = 0.0
    
    # Results
    approval_percentage: float
    meets_quorum: bool
    proposal_approved: bool
    result_type: str  # "approved", "rejected", "failed_quorum"
    
    # Metadata
    voting_concluded_at: datetime = datetime.now(timezone.utc)
    final_status: str = "completed"


class TokenWeightedVoting:
    """
    Token-weighted voting system for PRSM governance
    Handles proposal creation, voting power calculation, delegation, and term management
    """
    
    def __init__(self):
        self.voting_id = str(uuid4())
        self.logger = logger.bind(component="token_weighted_voting", voting_id=self.voting_id)
        
        # Voting state
        self.proposals: Dict[UUID, GovernanceProposal] = {}
        self.votes: Dict[UUID, List[Vote]] = {}
        self.voting_results: Dict[UUID, VotingResults] = {}
        
        # Delegation management
        self.delegations: Dict[str, List[DelegationRecord]] = defaultdict(list)  # delegator_id -> delegations
        self.delegation_pool: Dict[str, float] = defaultdict(float)  # delegate_id -> total delegated power
        
        # Governance roles and terms
        self.governance_roles: Dict[str, List[GovernanceRole]] = defaultdict(list)
        self.active_terms: Dict[str, List[GovernanceTerm]] = defaultdict(list)
        self.term_history: Dict[str, List[GovernanceTerm]] = defaultdict(list)
        
        # Voting power cache
        self.voting_power_cache: Dict[str, VotingPowerCalculation] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        
        # Safety integration
        self.safety_monitor = SafetyMonitor()
        
        # Performance statistics
        self.governance_stats = {
            "total_proposals_created": 0,
            "total_votes_cast": 0,
            "total_voting_power_allocated": 0.0,
            "active_delegations": 0,
            "governance_roles_assigned": 0,
            "proposals_approved": 0,
            "proposals_rejected": 0,
            "average_participation_rate": 0.0
        }
        
        # Synchronization
        self._voting_lock = asyncio.Lock()
        self._delegation_lock = asyncio.Lock()
        self._role_lock = asyncio.Lock()
        
        print("🗳️ TokenWeightedVoting initialized")
    
    
    async def create_proposal(self, proposer_id: str, proposal: GovernanceProposal) -> UUID:
        """
        Create a new governance proposal with token-weighted voting
        
        Args:
            proposer_id: ID of the user creating the proposal
            proposal: Governance proposal to create
            
        Returns:
            UUID of the created proposal
        """
        try:
            async with self._voting_lock:
                # Validate proposer eligibility
                if not await self._validate_proposer_eligibility(proposer_id):
                    raise ValueError("Proposer not eligible to create proposals")
                
                # Charge proposal submission fee
                proposal_fee_paid = await ftns_service.charge_context_access(proposer_id, int(PROPOSAL_SUBMISSION_FEE))
                if not proposal_fee_paid:
                    raise ValueError("Insufficient FTNS balance for proposal submission fee")
                
                # Safety validation
                safety_check = await self.safety_monitor.validate_model_output(
                    {"proposal": proposal.dict()},
                    ["no_malicious_content", "content_appropriateness"]
                )
                
                if not safety_check:
                    raise ValueError("Proposal failed safety validation")
                
                # Set proposal metadata
                proposal.proposer_id = proposer_id
                proposal.created_at = datetime.now(timezone.utc)
                proposal.status = "active"
                
                # Determine voting period based on category
                voting_period = await self._determine_voting_period(proposal.proposal_type)
                proposal.voting_starts = datetime.now(timezone.utc)
                proposal.voting_ends = proposal.voting_starts + timedelta(days=voting_period)
                
                # Store proposal
                self.proposals[proposal.proposal_id] = proposal
                self.votes[proposal.proposal_id] = []
                
                # Update statistics
                self.governance_stats["total_proposals_created"] += 1
                
                self.logger.info(
                    "Governance proposal created",
                    proposal_id=str(proposal.proposal_id),
                    category=proposal.proposal_type,
                    voting_ends=proposal.voting_ends.isoformat()
                )
                
                return proposal.proposal_id
                
        except Exception as e:
            self.logger.error("Failed to create proposal", error=str(e))
            raise
    
    
    async def cast_vote(self, voter_id: str, proposal_id: UUID, vote: bool, 
                       rationale: Optional[str] = None) -> bool:
        """
        Cast a vote on a governance proposal with token-weighted power
        
        Args:
            voter_id: ID of the voter
            proposal_id: ID of the proposal to vote on
            vote: True for yes/approve, False for no/reject
            rationale: Optional reasoning for the vote
            
        Returns:
            True if vote was cast successfully
        """
        try:
            async with self._voting_lock:
                # Validate vote
                if not await self._validate_vote_eligibility(voter_id, proposal_id):
                    return False
                
                proposal = self.proposals[proposal_id]
                
                # Check voting period
                current_time = datetime.now(timezone.utc)
                if current_time < proposal.voting_starts or current_time > proposal.voting_ends:
                    self.logger.warning("Voting period not active", proposal_id=str(proposal_id))
                    return False
                
                # Check for existing vote
                existing_votes = [v for v in self.votes[proposal_id] if v.voter_id == voter_id]
                if existing_votes:
                    self.logger.warning("Voter has already voted", voter_id=voter_id)
                    return False
                
                # Calculate voting power
                voting_power_calc = await self.calculate_voting_power(voter_id)
                
                # Create vote record
                vote_record = Vote(
                    proposal_id=proposal_id,
                    voter_id=voter_id,
                    vote=vote,
                    voting_power=voting_power_calc.total_voting_power,
                    rationale=rationale,
                    created_at=current_time
                )
                
                # Record vote
                self.votes[proposal_id].append(vote_record)
                
                # Update proposal vote counts
                if vote:
                    proposal.votes_for += 1
                else:
                    proposal.votes_against += 1
                
                proposal.total_voting_power += voting_power_calc.total_voting_power
                
                # Check if voting should conclude early
                if await self._should_conclude_voting_early(proposal_id):
                    await self._conclude_voting(proposal_id)
                
                # Update statistics
                self.governance_stats["total_votes_cast"] += 1
                self.governance_stats["total_voting_power_allocated"] += voting_power_calc.total_voting_power
                
                self.logger.info(
                    "Vote cast successfully",
                    voter_id=voter_id,
                    proposal_id=str(proposal_id),
                    vote=vote,
                    voting_power=voting_power_calc.total_voting_power
                )
                
                return True
                
        except Exception as e:
            self.logger.error("Failed to cast vote", error=str(e))
            return False
    
    
    async def calculate_voting_power(self, voter_id: str) -> VotingPowerCalculation:
        """
        Calculate comprehensive voting power for a user
        
        Args:
            voter_id: ID of the voter
            
        Returns:
            Detailed voting power calculation
        """
        try:
            # Check cache first
            if (voter_id in self.voting_power_cache and 
                voter_id in self.cache_expiry and 
                datetime.now(timezone.utc) < self.cache_expiry[voter_id]):
                return self.voting_power_cache[voter_id]
            
            # Get user's FTNS balance
            user_balance = await ftns_service.get_user_balance(voter_id)
            token_balance = user_balance.balance
            
            # Base voting power calculation
            if token_balance < MIN_VOTING_BALANCE:
                base_voting_power = 0.0
            else:
                # Apply power curve to reduce whale dominance
                normalized_balance = token_balance / MIN_VOTING_BALANCE
                base_voting_power = math.pow(normalized_balance, VOTING_POWER_CURVE)
            
            # Role multiplier
            role_multiplier = await self._calculate_role_multiplier(voter_id)
            
            # Reputation bonus (simplified)
            reputation_bonus = await self._calculate_reputation_bonus(voter_id)
            
            # Delegation power (power delegated to this user)
            delegation_power = self.delegation_pool.get(voter_id, 0.0)
            
            # Calculate total voting power
            total_voting_power = (base_voting_power * role_multiplier * (1 + reputation_bonus)) + delegation_power
            
            # Apply maximum voting power ratio
            max_allowed_power = MIN_VOTING_BALANCE * MAX_VOTING_POWER_RATIO
            total_voting_power = min(total_voting_power, max_allowed_power)
            
            # Create calculation record
            calculation = VotingPowerCalculation(
                voter_id=voter_id,
                token_balance=token_balance,
                base_voting_power=base_voting_power,
                role_multiplier=role_multiplier,
                reputation_bonus=reputation_bonus,
                delegation_power=delegation_power,
                total_voting_power=total_voting_power
            )
            
            # Cache result for 1 hour
            self.voting_power_cache[voter_id] = calculation
            self.cache_expiry[voter_id] = datetime.now(timezone.utc) + timedelta(hours=1)
            
            return calculation
            
        except Exception as e:
            self.logger.error("Failed to calculate voting power", voter_id=voter_id, error=str(e))
            return VotingPowerCalculation(
                voter_id=voter_id,
                token_balance=0.0,
                base_voting_power=0.0,
                role_multiplier=1.0,
                reputation_bonus=0.0,
                delegation_power=0.0,
                total_voting_power=0.0
            )
    
    
    async def implement_term_limits(self, governance_roles: List[str]) -> Dict[str, Any]:
        """
        Implement and enforce term limits for governance roles
        
        Args:
            governance_roles: List of role holders to check
            
        Returns:
            Term limit enforcement results
        """
        try:
            async with self._role_lock:
                enforcement_results = {
                    "roles_checked": len(governance_roles),
                    "terms_expired": 0,
                    "terms_extended": 0,
                    "new_elections_required": [],
                    "enforcement_actions": []
                }
                
                current_time = datetime.now(timezone.utc)
                
                for role_holder_id in governance_roles:
                    # Check active terms for this role holder
                    active_terms = self.active_terms.get(role_holder_id, [])
                    
                    for term in active_terms:
                        # Check if term has expired
                        if current_time > term.end_date:
                            # Expire the term
                            term.is_active = False
                            enforcement_results["terms_expired"] += 1
                            
                            # Move to history
                            self.term_history[role_holder_id].append(term)
                            self.active_terms[role_holder_id].remove(term)
                            
                            enforcement_results["enforcement_actions"].append({
                                "action": "term_expired",
                                "role_holder": role_holder_id,
                                "role": term.role,
                                "term_id": str(term.term_id)
                            })
                            
                            # Check if new election is needed
                            enforcement_results["new_elections_required"].append({
                                "role": term.role,
                                "previous_holder": role_holder_id,
                                "election_type": "regular"
                            })
                        
                        # Check consecutive term limits
                        elif term.term_number >= MAX_CONSECUTIVE_TERMS:
                            # Force term end for consecutive term limit
                            term.end_date = current_time
                            term.is_active = False
                            enforcement_results["terms_expired"] += 1
                            
                            enforcement_results["enforcement_actions"].append({
                                "action": "consecutive_term_limit",
                                "role_holder": role_holder_id,
                                "role": term.role,
                                "consecutive_terms": term.term_number
                            })
                            
                            enforcement_results["new_elections_required"].append({
                                "role": term.role,
                                "previous_holder": role_holder_id,
                                "election_type": "term_limit_replacement"
                            })
                
                self.logger.info(
                    "Term limits enforced",
                    roles_checked=enforcement_results["roles_checked"],
                    terms_expired=enforcement_results["terms_expired"],
                    new_elections=len(enforcement_results["new_elections_required"])
                )
                
                return enforcement_results
                
        except Exception as e:
            self.logger.error("Failed to implement term limits", error=str(e))
            return {
                "roles_checked": 0,
                "terms_expired": 0,
                "terms_extended": 0,
                "new_elections_required": [],
                "enforcement_actions": [],
                "error": str(e)
            }
    
    
    async def delegate_voting_power(self, delegator_id: str, delegate_id: str, 
                                  scope: str = "all", category: Optional[ProposalCategory] = None,
                                  proposal_id: Optional[UUID] = None) -> bool:
        """Delegate voting power to another user"""
        try:
            async with self._delegation_lock:
                # Validate delegation
                if not await self._validate_delegation(delegator_id, delegate_id, scope):
                    return False
                
                # Calculate delegator's voting power
                voting_power_calc = await self.calculate_voting_power(delegator_id)
                delegation_power = voting_power_calc.total_voting_power
                
                # Create delegation record
                delegation = DelegationRecord(
                    delegator_id=delegator_id,
                    delegate_id=delegate_id,
                    scope=scope,
                    category_filter=category,
                    proposal_filter=proposal_id,
                    delegation_power=delegation_power
                )
                
                # Store delegation
                self.delegations[delegator_id].append(delegation)
                self.delegation_pool[delegate_id] += delegation_power
                
                # Update statistics
                self.governance_stats["active_delegations"] += 1
                
                self.logger.info(
                    "Voting power delegated",
                    delegator=delegator_id,
                    delegate=delegate_id,
                    power=delegation_power,
                    scope=scope
                )
                
                return True
                
        except Exception as e:
            self.logger.error("Failed to delegate voting power", error=str(e))
            return False
    
    
    async def assign_governance_role(self, user_id: str, role: GovernanceRole, 
                                   term_length_days: Optional[int] = None) -> bool:
        """Assign a governance role to a user"""
        try:
            async with self._role_lock:
                # Validate role assignment
                if not await self._validate_role_assignment(user_id, role):
                    return False
                
                # Determine term length
                if term_length_days is None:
                    term_length_days = GOVERNANCE_TERM_LENGTH_DAYS
                
                # Calculate consecutive term number
                previous_terms = [t for t in self.term_history.get(user_id, []) if t.role == role]
                consecutive_terms = len([t for t in previous_terms if t.end_date > datetime.now(timezone.utc) - timedelta(days=30)])
                
                # Check term limits
                if consecutive_terms >= MAX_CONSECUTIVE_TERMS:
                    self.logger.warning("User has reached consecutive term limit", user_id=user_id, role=role)
                    return False
                
                # Create new term
                term = GovernanceTerm(
                    holder_id=user_id,
                    role=role,
                    start_date=datetime.now(timezone.utc),
                    end_date=datetime.now(timezone.utc) + timedelta(days=term_length_days),
                    term_number=consecutive_terms + 1
                )
                
                # Store term
                self.active_terms[user_id].append(term)
                self.governance_roles[user_id].append(role)
                
                # Update statistics
                self.governance_stats["governance_roles_assigned"] += 1
                
                self.logger.info(
                    "Governance role assigned",
                    user_id=user_id,
                    role=role,
                    term_id=str(term.term_id),
                    end_date=term.end_date.isoformat()
                )
                
                return True
                
        except Exception as e:
            self.logger.error("Failed to assign governance role", error=str(e))
            return False
    
    
    async def get_proposal_results(self, proposal_id: UUID) -> Optional[VotingResults]:
        """Get comprehensive voting results for a proposal"""
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        votes = self.votes[proposal_id]
        
        # Calculate total eligible voting power
        total_eligible_power = await self._calculate_total_eligible_voting_power()
        
        # Calculate vote breakdowns
        votes_for = sum(1 for v in votes if v.vote)
        votes_against = sum(1 for v in votes if not v.vote)
        votes_abstain = 0  # Not implemented in current vote model
        
        voting_power_for = sum(v.voting_power for v in votes if v.vote)
        voting_power_against = sum(v.voting_power for v in votes if not v.vote)
        voting_power_used = voting_power_for + voting_power_against
        
        # Calculate results
        participation_rate = voting_power_used / total_eligible_power if total_eligible_power > 0 else 0
        approval_percentage = voting_power_for / voting_power_used if voting_power_used > 0 else 0
        
        meets_quorum = participation_rate >= QUORUM_PERCENTAGE
        
        # Determine approval based on proposal category
        required_threshold = await self._get_approval_threshold(proposal.proposal_type)
        proposal_approved = meets_quorum and approval_percentage >= required_threshold
        
        # Determine result type
        if not meets_quorum:
            result_type = "failed_quorum"
        elif proposal_approved:
            result_type = "approved"
        else:
            result_type = "rejected"
        
        return VotingResults(
            proposal_id=proposal_id,
            total_eligible_voting_power=total_eligible_power,
            total_votes_cast=len(votes),
            total_voting_power_used=voting_power_used,
            participation_rate=participation_rate,
            votes_for=votes_for,
            votes_against=votes_against,
            votes_abstain=votes_abstain,
            voting_power_for=voting_power_for,
            voting_power_against=voting_power_against,
            voting_power_abstain=0.0,
            approval_percentage=approval_percentage,
            meets_quorum=meets_quorum,
            proposal_approved=proposal_approved,
            result_type=result_type
        )
    
    
    async def get_governance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive governance statistics"""
        # Calculate average participation rate
        if self.voting_results:
            avg_participation = statistics.mean([r.participation_rate for r in self.voting_results.values()])
            self.governance_stats["average_participation_rate"] = avg_participation
        
        return {
            **self.governance_stats,
            "active_proposals": len([p for p in self.proposals.values() if p.status == "active"]),
            "total_active_delegations": sum(len(delegations) for delegations in self.delegations.values()),
            "total_delegation_power": sum(self.delegation_pool.values()),
            "active_governance_roles": sum(len(roles) for roles in self.governance_roles.values()),
            "current_voting_sessions": len([
                p for p in self.proposals.values() 
                if p.status == "active" and 
                   p.voting_starts <= datetime.now(timezone.utc) <= p.voting_ends
            ]),
            "governance_configuration": {
                "min_voting_balance": MIN_VOTING_BALANCE,
                "proposal_submission_fee": PROPOSAL_SUBMISSION_FEE,
                "quorum_percentage": QUORUM_PERCENTAGE,
                "approval_threshold": APPROVAL_THRESHOLD,
                "term_length_days": GOVERNANCE_TERM_LENGTH_DAYS
            }
        }
    
    
    # === Private Helper Methods ===
    
    async def _validate_proposer_eligibility(self, proposer_id: str) -> bool:
        """Validate if user is eligible to create proposals"""
        # Check minimum FTNS balance
        user_balance = await ftns_service.get_user_balance(proposer_id)
        if user_balance.balance < PROPOSAL_SUBMISSION_FEE:
            return False
        
        # Check for governance role or sufficient token balance
        if proposer_id in self.governance_roles:
            return True
        
        if user_balance.balance >= MIN_VOTING_BALANCE * 10:  # 10x minimum for proposals
            return True
        
        return False
    
    
    async def _determine_voting_period(self, proposal_category: str) -> int:
        """Determine voting period in days based on proposal category"""
        if proposal_category == "emergency":
            return 1  # 1 day for emergency
        elif proposal_category == "constitutional":
            return 14  # 14 days for constitutional changes
        elif proposal_category in ["safety", "economic"]:
            return 10  # 10 days for critical categories
        else:
            return 7  # 7 days standard
    
    
    async def _validate_vote_eligibility(self, voter_id: str, proposal_id: UUID) -> bool:
        """Validate if user is eligible to vote"""
        if proposal_id not in self.proposals:
            return False
        
        # Check minimum voting power
        voting_power_calc = await self.calculate_voting_power(voter_id)
        return voting_power_calc.total_voting_power > 0
    
    
    async def _calculate_role_multiplier(self, user_id: str) -> float:
        """Calculate voting power multiplier based on governance roles"""
        if user_id not in self.governance_roles:
            return 1.0
        
        roles = self.governance_roles[user_id]
        multiplier = 1.0
        
        for role in roles:
            if role == GovernanceRole.DELEGATE:
                multiplier *= 1.2
            elif role == GovernanceRole.COMMITTEE_MEMBER:
                multiplier *= 1.1
            elif role == GovernanceRole.PROPOSAL_REVIEWER:
                multiplier *= 1.05
        
        return min(multiplier, 1.5)  # Cap at 1.5x
    
    
    async def _calculate_reputation_bonus(self, user_id: str) -> float:
        """Calculate reputation-based voting power bonus"""
        # Simplified reputation calculation
        # In production, this would integrate with reputation system
        return 0.1  # 10% bonus for active participants
    
    
    async def _validate_delegation(self, delegator_id: str, delegate_id: str, scope: str) -> bool:
        """Validate delegation parameters"""
        if delegator_id == delegate_id:
            return False
        
        # Check if delegator has voting power
        voting_power_calc = await self.calculate_voting_power(delegator_id)
        if voting_power_calc.total_voting_power <= 0:
            return False
        
        # Check for circular delegation
        if await self._has_circular_delegation(delegator_id, delegate_id):
            return False
        
        return True
    
    
    async def _has_circular_delegation(self, delegator_id: str, delegate_id: str) -> bool:
        """Check for circular delegation patterns"""
        visited = set()
        current = delegate_id
        
        while current and current not in visited:
            visited.add(current)
            # Find if current user has delegated to someone else
            delegations = self.delegations.get(current, [])
            if delegations:
                current = delegations[0].delegate_id  # Simplified - check first delegation
            else:
                current = None
            
            if current == delegator_id:
                return True
        
        return False
    
    
    async def _validate_role_assignment(self, user_id: str, role: GovernanceRole) -> bool:
        """Validate governance role assignment"""
        # Check minimum voting power for role
        voting_power_calc = await self.calculate_voting_power(user_id)
        min_power_required = MIN_VOTING_BALANCE * 5  # 5x minimum for governance roles
        
        return voting_power_calc.total_voting_power >= min_power_required
    
    
    async def _should_conclude_voting_early(self, proposal_id: UUID) -> bool:
        """Check if voting should conclude early"""
        proposal = self.proposals[proposal_id]
        votes = self.votes[proposal_id]
        
        if len(votes) < 10:  # Minimum votes for early conclusion
            return False
        
        # Check for overwhelming majority
        total_power = sum(v.voting_power for v in votes)
        power_for = sum(v.voting_power for v in votes if v.vote)
        
        if total_power > 0:
            approval_rate = power_for / total_power
            if approval_rate >= 0.9 or approval_rate <= 0.1:
                return True
        
        return False
    
    
    async def _conclude_voting(self, proposal_id: UUID):
        """Conclude voting and update proposal status"""
        results = await self.get_proposal_results(proposal_id)
        if results:
            proposal = self.proposals[proposal_id]
            
            if results.proposal_approved:
                proposal.status = "approved"
                self.governance_stats["proposals_approved"] += 1
            else:
                proposal.status = "rejected"
                self.governance_stats["proposals_rejected"] += 1
            
            self.voting_results[proposal_id] = results
            
            self.logger.info(
                "Voting concluded",
                proposal_id=str(proposal_id),
                result=results.result_type,
                participation_rate=results.participation_rate,
                approval_percentage=results.approval_percentage
            )
    
    
    async def _calculate_total_eligible_voting_power(self) -> float:
        """Calculate total eligible voting power in the system"""
        # This is a simplified calculation
        # In production, this would query all token holders
        return 1000000.0  # Assume 1M FTNS total eligible voting power
    
    
    async def _get_approval_threshold(self, proposal_category: str) -> float:
        """Get approval threshold based on proposal category"""
        if proposal_category == "constitutional":
            return SUPERMAJORITY_THRESHOLD
        elif proposal_category in ["safety", "emergency"]:
            return APPROVAL_THRESHOLD
        else:
            return APPROVAL_THRESHOLD


# === Global Token-Weighted Voting Instance ===

_voting_instance: Optional[TokenWeightedVoting] = None

def get_token_weighted_voting() -> TokenWeightedVoting:
    """Get or create the global token-weighted voting instance"""
    global _voting_instance
    if _voting_instance is None:
        _voting_instance = TokenWeightedVoting()
    return _voting_instance