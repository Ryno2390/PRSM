"""
PRSM Safety Governance

Implements safety governance system for submitting safety proposals,
voting on safety measures, and implementing approved safety policies.

Based on execution_plan.md Week 9-10 requirements.

Integration with Governance Execution (Phase 3.3):
- Connects approved proposals to GovernanceExecutor
- Supports timelocked execution for sensitive changes
- Provides execution transparency and audit trails
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from enum import Enum
import structlog

from pydantic import Field, BaseModel, validator
from prsm.core.models import SafetyLevel, SafetyFlag, GovernanceProposal, PRSMBaseModel

# Import governance execution system
from prsm.governance.execution import (
    GovernanceExecutor,
    GovernanceAction,
    GovernanceActionStatus,
    ExecutionResult,
    ProposalType as ExecutionProposalType,
    TimelockController,
    GovernableParameterRegistry,
    get_governance_executor
)

logger = structlog.get_logger()


class ProposalType(str, Enum):
    """Types of safety proposals"""
    SAFETY_POLICY = "safety_policy"
    CIRCUIT_BREAKER_CONFIG = "circuit_breaker_config"
    MODEL_RESTRICTIONS = "model_restrictions"
    NETWORK_PROTOCOL = "network_protocol"
    EMERGENCY_PROCEDURE = "emergency_procedure"
    MONITORING_ENHANCEMENT = "monitoring_enhancement"
    GOVERNANCE_RULE = "governance_rule"


class ProposalStatus(str, Enum):
    """Status of safety proposals"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    VOTING = "voting"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    EXPIRED = "expired"


class VoteType(str, Enum):
    """Types of votes"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DELEGATE = "delegate"


class UrgencyLevel(str, Enum):
    """Urgency levels for proposals"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# === Helper Functions for Enum Handling ===

def _proposal_status_value(proposal_status):
    """Helper function to get the value of a proposal status, whether it's an enum or string"""
    if isinstance(proposal_status, ProposalStatus):
        return proposal_status.value
    elif isinstance(proposal_status, str):
        return proposal_status
    else:
        raise ValueError(f"Invalid proposal_status type: {type(proposal_status)}")


def _proposal_status_name(proposal_status):
    """Helper function to get the name of a proposal status, whether it's an enum or string"""
    if isinstance(proposal_status, ProposalStatus):
        return proposal_status.name
    elif isinstance(proposal_status, str):
        return ProposalStatus(proposal_status).name
    else:
        raise ValueError(f"Invalid proposal_status type: {type(proposal_status)}")


def _proposal_status_enum(proposal_status):
    """Helper function to convert a proposal status to an enum"""
    if isinstance(proposal_status, ProposalStatus):
        return proposal_status
    elif isinstance(proposal_status, str):
        return ProposalStatus(proposal_status)
    else:
        raise ValueError(f"Invalid proposal_status type: {type(proposal_status)}")


def _proposal_type_value(proposal_type):
    """Helper function to get the value of a proposal type, whether it's an enum or string"""
    if isinstance(proposal_type, ProposalType):
        return proposal_type.value
    elif isinstance(proposal_type, str):
        return proposal_type
    else:
        raise ValueError(f"Invalid proposal_type type: {type(proposal_type)}")


def _proposal_type_name(proposal_type):
    """Helper function to get the name of a proposal type, whether it's an enum or string"""
    if isinstance(proposal_type, ProposalType):
        return proposal_type.name
    elif isinstance(proposal_type, str):
        return ProposalType(proposal_type).name
    else:
        raise ValueError(f"Invalid proposal_type type: {type(proposal_type)}")


def _proposal_type_enum(proposal_type):
    """Helper function to convert a proposal type to an enum"""
    if isinstance(proposal_type, ProposalType):
        return proposal_type
    elif isinstance(proposal_type, str):
        return ProposalType(proposal_type)
    else:
        raise ValueError(f"Invalid proposal_type type: {type(proposal_type)}")


def _vote_type_value(vote_type):
    """Helper function to get the value of a vote type, whether it's an enum or string"""
    if isinstance(vote_type, VoteType):
        return vote_type.value
    elif isinstance(vote_type, str):
        return vote_type
    else:
        raise ValueError(f"Invalid vote_type type: {type(vote_type)}")


def _vote_type_name(vote_type):
    """Helper function to get the name of a vote type, whether it's an enum or string"""
    if isinstance(vote_type, VoteType):
        return vote_type.name
    elif isinstance(vote_type, str):
        return VoteType(vote_type).name
    else:
        raise ValueError(f"Invalid vote_type type: {type(vote_type)}")


def _vote_type_enum(vote_type):
    """Helper function to convert a vote type to an enum"""
    if isinstance(vote_type, VoteType):
        return vote_type
    elif isinstance(vote_type, str):
        return VoteType(vote_type)
    else:
        raise ValueError(f"Invalid vote_type type: {type(vote_type)}")


def _urgency_level_value(urgency_level):
    """Helper function to get the value of an urgency level, whether it's an enum or string"""
    if isinstance(urgency_level, UrgencyLevel):
        return urgency_level.value
    elif isinstance(urgency_level, str):
        return urgency_level
    else:
        raise ValueError(f"Invalid urgency_level type: {type(urgency_level)}")


def _urgency_level_name(urgency_level):
    """Helper function to get the name of an urgency level, whether it's an enum or string"""
    if isinstance(urgency_level, UrgencyLevel):
        return urgency_level.name
    elif isinstance(urgency_level, str):
        return UrgencyLevel(urgency_level).name
    else:
        raise ValueError(f"Invalid urgency_level type: {type(urgency_level)}")


def _urgency_level_enum(urgency_level):
    """Helper function to convert an urgency level to an enum"""
    if isinstance(urgency_level, UrgencyLevel):
        return urgency_level
    elif isinstance(urgency_level, str):
        return UrgencyLevel(urgency_level)
    else:
        raise ValueError(f"Invalid urgency_level type: {type(urgency_level)}")


class SafetyProposal(PRSMBaseModel):
    """Safety governance proposal"""
    proposal_id: UUID = Field(default_factory=uuid4)
    proposer_id: str
    proposal_type: ProposalType
    title: str
    description: str
    urgency_level: UrgencyLevel = UrgencyLevel.NORMAL
    affected_components: List[str] = Field(default_factory=list)
    implementation_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Voting configuration
    required_approval_percentage: float = Field(ge=0.5, le=1.0, default=0.6)
    minimum_participation: float = Field(ge=0.1, le=1.0, default=0.3)
    voting_deadline: datetime
    
    # Status tracking
    status: ProposalStatus = ProposalStatus.DRAFT
    submission_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    review_notes: List[str] = Field(default_factory=list)
    
    # Implementation
    implementation_plan: Optional[str] = None
    estimated_impact: Optional[str] = None
    rollback_plan: Optional[str] = None
    
    @validator('voting_deadline')
    def voting_deadline_must_be_future(cls, v):
        if v <= datetime.now(timezone.utc):
            raise ValueError('Voting deadline must be in the future')
        return v


class SafetyVote(PRSMBaseModel):
    """Vote on a safety proposal"""
    vote_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID
    voter_id: str
    vote_type: VoteType
    voting_power: float = Field(ge=0.0, le=1.0, default=1.0)
    reasoning: Optional[str] = None
    vote_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_anonymous: bool = False
    
    # Delegation (if applicable)
    delegated_from: Optional[str] = None
    delegation_scope: Optional[str] = None


class ProposalReview(PRSMBaseModel):
    """Review of a safety proposal"""
    review_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID
    reviewer_id: str
    review_type: str  # "technical", "safety", "legal", "community"
    recommendation: str  # "approve", "reject", "modify", "defer"
    review_notes: str
    concerns: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    review_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ImplementationResult(PRSMBaseModel):
    """Result of proposal implementation"""
    implementation_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID
    implemented_by: str
    implementation_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool
    implementation_notes: str
    verification_steps: List[str] = Field(default_factory=list)
    rollback_available: bool = True
    monitoring_metrics: Dict[str, Any] = Field(default_factory=dict)


class VotingResults(PRSMBaseModel):
    """Results of proposal voting"""
    proposal_id: UUID
    total_eligible_voters: int
    total_votes_cast: int
    participation_rate: float
    
    votes_by_type: Dict[VoteType, int] = Field(default_factory=dict)
    approval_percentage: float
    rejection_percentage: float
    abstention_percentage: float
    
    voting_concluded: bool
    result: str  # "approved", "rejected", "inconclusive"
    conclusion_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SafetyGovernance:
    """
    Safety governance system that manages safety proposals, voting,
    and implementation of approved safety measures.
    
    Integration with Governance Execution (Phase 3.3):
    - Uses GovernanceExecutor for actual parameter changes
    - Supports timelocked execution for sensitive changes
    - Provides execution transparency and audit trails
    """
    
    def __init__(self, governance_id: str = None, governance_executor: GovernanceExecutor = None):
        self.governance_id = governance_id or str(uuid4())
        self.logger = logger.bind(component="safety_governance", governance_id=self.governance_id)
        
        # Proposal management
        self.proposals: Dict[UUID, SafetyProposal] = {}
        self.votes: Dict[UUID, List[SafetyVote]] = {}
        self.reviews: Dict[UUID, List[ProposalReview]] = {}
        self.implementations: Dict[UUID, ImplementationResult] = {}
        
        # Governance configuration
        self.eligible_voters: Set[str] = set()
        self.reviewer_roles: Dict[str, List[str]] = {}  # reviewer_id -> roles
        self.implementation_authorities: Set[str] = set()
        
        # Voting configuration
        self.default_voting_period = timedelta(days=7)
        self.emergency_voting_period = timedelta(hours=24)
        self.critical_voting_period = timedelta(days=3)
        
        # Implementation tracking
        self.active_implementations: Dict[UUID, Any] = {}
        self.implementation_queue: List[UUID] = []
        
        # Governance executor integration (Phase 3.3)
        self.governance_executor = governance_executor or get_governance_executor()
        
        # Execution tracking
        self.execution_results: Dict[UUID, ExecutionResult] = {}
        self.pending_executions: Dict[UUID, str] = {}  # proposal_id -> action_id
        
    async def submit_safety_proposal(self, proposal: SafetyProposal) -> UUID:
        """
        Submit a safety proposal for review and voting.
        
        Args:
            proposal: Safety proposal to submit
            
        Returns:
            UUID of the submitted proposal
        """
        self.logger.info(
            "Submitting safety proposal",
            proposal_type=_proposal_type_value(proposal.proposal_type),
            urgency=_urgency_level_value(proposal.urgency_level),
            proposer=proposal.proposer_id
        )
        
        try:
            # Validate proposal
            if not await self._validate_proposal(proposal):
                raise ValueError("Invalid proposal")
            
            # Set voting deadline based on urgency
            if not proposal.voting_deadline:
                proposal.voting_deadline = await self._calculate_voting_deadline(proposal.urgency_level)
            
            # Update status
            proposal.status = ProposalStatus.SUBMITTED
            proposal.submission_time = datetime.now(timezone.utc)
            
            # Store proposal
            self.proposals[proposal.proposal_id] = proposal
            self.votes[proposal.proposal_id] = []
            self.reviews[proposal.proposal_id] = []
            
            # Initiate review process
            await self._initiate_review_process(proposal)
            
            self.logger.info(
                "Safety proposal submitted successfully",
                proposal_id=str(proposal.proposal_id),
                voting_deadline=proposal.voting_deadline.isoformat()
            )
            
            return proposal.proposal_id
            
        except Exception as e:
            self.logger.error("Failed to submit safety proposal", error=str(e))
            raise
    
    async def vote_on_safety_measure(self, voter_id: str, proposal_id: UUID, vote: VoteType, 
                                   reasoning: Optional[str] = None) -> bool:
        """
        Cast a vote on a safety proposal.
        
        Args:
            voter_id: ID of the voter
            proposal_id: ID of the proposal to vote on
            vote: Type of vote (approve/reject/abstain)
            reasoning: Optional reasoning for the vote
            
        Returns:
            True if vote was cast successfully
        """
        self.logger.info(
            "Casting vote on safety proposal",
            voter_id=voter_id,
            proposal_id=str(proposal_id),
            vote_type=_vote_type_value(vote)
        )
        
        try:
            # Validate vote
            if not await self._validate_vote(voter_id, proposal_id, vote):
                return False
            
            proposal = self.proposals[proposal_id]
            
            # Check if voting is still open
            if datetime.now(timezone.utc) > proposal.voting_deadline:
                self.logger.warning("Voting period has ended", proposal_id=str(proposal_id))
                return False
            
            # Check if voter is eligible
            if voter_id not in self.eligible_voters:
                self.logger.warning("Voter not eligible", voter_id=voter_id)
                return False
            
            # Check for existing vote
            existing_votes = [v for v in self.votes[proposal_id] if v.voter_id == voter_id]
            if existing_votes:
                self.logger.warning("Voter has already voted", voter_id=voter_id)
                return False
            
            # Calculate voting power
            voting_power = await self._calculate_voting_power(voter_id, proposal)
            
            # Create vote
            safety_vote = SafetyVote(
                proposal_id=proposal_id,
                voter_id=voter_id,
                vote_type=vote,
                voting_power=voting_power,
                reasoning=reasoning
            )
            
            # Record vote
            self.votes[proposal_id].append(safety_vote)
            
            # Check if voting is complete
            if await self._is_voting_complete(proposal_id):
                await self._conclude_voting(proposal_id)
            
            self.logger.info(
                "Vote cast successfully",
                voter_id=voter_id,
                proposal_id=str(proposal_id),
                voting_power=voting_power
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to cast vote", error=str(e))
            return False
    
    async def implement_approved_measures(self, proposal_id: UUID) -> bool:
        """
        Implement approved safety measures.
        
        Args:
            proposal_id: ID of the approved proposal to implement
            
        Returns:
            True if implementation was successful
        """
        self.logger.info("Implementing approved safety measures", proposal_id=str(proposal_id))
        
        try:
            if proposal_id not in self.proposals:
                self.logger.error("Proposal not found", proposal_id=str(proposal_id))
                return False
            
            proposal = self.proposals[proposal_id]
            
            # Verify proposal is approved
            if proposal.status != ProposalStatus.APPROVED:
                self.logger.error("Proposal not approved", status=proposal.status)
                return False
            
            # Check if already implemented
            if proposal_id in self.implementations:
                self.logger.warning("Proposal already implemented", proposal_id=str(proposal_id))
                return True
            
            # Execute implementation
            implementation_result = await self._execute_implementation(proposal)
            
            # Record implementation
            self.implementations[proposal_id] = implementation_result
            
            # Update proposal status
            if implementation_result.success:
                proposal.status = ProposalStatus.IMPLEMENTED
                self.logger.info(
                    "Safety measures implemented successfully",
                    proposal_id=str(proposal_id),
                    implementation_id=str(implementation_result.implementation_id)
                )
            else:
                self.logger.error(
                    "Implementation failed",
                    proposal_id=str(proposal_id),
                    notes=implementation_result.implementation_notes
                )
            
            return implementation_result.success
            
        except Exception as e:
            self.logger.error("Failed to implement safety measures", error=str(e))
            return False
    
    async def get_governance_status(self) -> Dict[str, Any]:
        """Get comprehensive governance status"""
        current_time = datetime.now(timezone.utc)
        
        # Proposal statistics
        proposals_by_status = {}
        for status in ProposalStatus:
            proposals_by_status[_proposal_status_value(status)] = len([
                p for p in self.proposals.values() if _proposal_status_value(p.status) == _proposal_status_value(status)
            ])
        
        # Active voting
        active_voting = [
            p for p in self.proposals.values() 
            if p.status == ProposalStatus.VOTING and p.voting_deadline > current_time
        ]
        
        # Recent implementations
        recent_implementations = [
            impl for impl in self.implementations.values()
            if (current_time - impl.implementation_time).days <= 30
        ]
        
        # Voter participation
        total_votes = sum(len(votes) for votes in self.votes.values())
        
        return {
            'governance_id': self.governance_id,
            'proposal_summary': {
                'total_proposals': len(self.proposals),
                'by_status': proposals_by_status,
                'active_voting': len(active_voting),
                'pending_implementation': len([
                    p for p in self.proposals.values() 
                    if p.status == ProposalStatus.APPROVED
                ])
            },
            'voting_summary': {
                'eligible_voters': len(self.eligible_voters),
                'total_votes_cast': total_votes,
                'active_voting_sessions': len(active_voting)
            },
            'implementation_summary': {
                'total_implementations': len(self.implementations),
                'recent_implementations': len(recent_implementations),
                'successful_implementations': len([
                    impl for impl in self.implementations.values() if impl.success
                ]),
                'pending_queue': len(self.implementation_queue)
            }
        }
    
    async def add_eligible_voter(self, voter_id: str) -> bool:
        """Add an eligible voter"""
        self.eligible_voters.add(voter_id)
        self.logger.info("Added eligible voter", voter_id=voter_id)
        return True
    
    async def remove_eligible_voter(self, voter_id: str) -> bool:
        """Remove an eligible voter"""
        self.eligible_voters.discard(voter_id)
        self.logger.info("Removed eligible voter", voter_id=voter_id)
        return True
    
    # === Private Helper Methods ===
    
    async def _validate_proposal(self, proposal: SafetyProposal) -> bool:
        """Validate a safety proposal"""
        # Check required fields
        if not proposal.title or not proposal.description:
            return False
        
        # Check proposer eligibility (for now, all proposers are eligible)
        # In production, this would check governance permissions
        
        # Validate affected components
        if not proposal.affected_components:
            proposal.review_notes.append("No affected components specified")
        
        return True
    
    async def _calculate_voting_deadline(self, urgency: UrgencyLevel) -> datetime:
        """Calculate voting deadline based on urgency"""
        current_time = datetime.now(timezone.utc)
        
        if urgency == UrgencyLevel.EMERGENCY:
            return current_time + timedelta(hours=6)
        elif urgency == UrgencyLevel.CRITICAL:
            return current_time + self.critical_voting_period
        elif urgency == UrgencyLevel.HIGH:
            return current_time + timedelta(days=2)
        else:
            return current_time + self.default_voting_period
    
    async def _initiate_review_process(self, proposal: SafetyProposal):
        """Initiate the review process for a proposal"""
        proposal.status = ProposalStatus.UNDER_REVIEW
        
        # Auto-transition to voting for low-complexity proposals
        if proposal.urgency_level in [UrgencyLevel.LOW, UrgencyLevel.NORMAL]:
            # Add a delay to simulate review time
            await asyncio.sleep(0.1)
            proposal.status = ProposalStatus.VOTING
    
    async def _validate_vote(self, voter_id: str, proposal_id: UUID, vote: VoteType) -> bool:
        """Validate a vote"""
        if proposal_id not in self.proposals:
            return False
        
        if voter_id not in self.eligible_voters:
            return False
        
        # Check for duplicate votes
        existing_votes = [v for v in self.votes[proposal_id] if v.voter_id == voter_id]
        if existing_votes:
            return False
        
        return True
    
    async def _calculate_voting_power(self, voter_id: str, proposal: SafetyProposal) -> float:
        """Calculate voting power for a voter"""
        # For now, all voters have equal power
        # In production, this could be based on stake, reputation, etc.
        return 1.0
    
    async def _is_voting_complete(self, proposal_id: UUID) -> bool:
        """Check if voting is complete"""
        proposal = self.proposals[proposal_id]
        votes = self.votes[proposal_id]
        
        # Check if deadline passed
        if datetime.now(timezone.utc) > proposal.voting_deadline:
            return True
        
        # Check if minimum participation reached and all eligible voters voted
        participation_rate = len(votes) / len(self.eligible_voters) if self.eligible_voters else 0
        
        # Early conclusion if unanimous or overwhelming majority
        if len(votes) >= 3:  # Minimum for early conclusion
            approve_votes = len([v for v in votes if v.vote_type == VoteType.APPROVE])
            total_votes = len(votes)
            
            if approve_votes / total_votes >= 0.9 or approve_votes / total_votes <= 0.1:
                return True
        
        return False
    
    async def _conclude_voting(self, proposal_id: UUID):
        """Conclude voting and determine result"""
        proposal = self.proposals[proposal_id]
        votes = self.votes[proposal_id]
        
        # Calculate results
        total_eligible = len(self.eligible_voters)
        total_votes = len(votes)
        participation_rate = total_votes / total_eligible if total_eligible > 0 else 0
        
        votes_by_type = {vote_type: 0 for vote_type in VoteType}
        total_voting_power = 0
        
        for vote in votes:
            votes_by_type[_vote_type_enum(vote.vote_type)] += 1
            total_voting_power += vote.voting_power
        
        # Calculate percentages
        approve_votes = votes_by_type[VoteType.APPROVE]
        reject_votes = votes_by_type[VoteType.REJECT]
        
        if total_votes > 0:
            approval_percentage = approve_votes / total_votes
            rejection_percentage = reject_votes / total_votes
            abstention_percentage = votes_by_type[VoteType.ABSTAIN] / total_votes
        else:
            approval_percentage = 0.0
            rejection_percentage = 0.0
            abstention_percentage = 0.0
        
        # Determine result
        voting_result = "inconclusive"
        
        if participation_rate >= proposal.minimum_participation:
            if approval_percentage >= proposal.required_approval_percentage:
                voting_result = "approved"
                proposal.status = ProposalStatus.APPROVED
            elif rejection_percentage > (1 - proposal.required_approval_percentage):
                voting_result = "rejected"
                proposal.status = ProposalStatus.REJECTED
        
        # Create voting results
        results = VotingResults(
            proposal_id=proposal_id,
            total_eligible_voters=total_eligible,
            total_votes_cast=total_votes,
            participation_rate=participation_rate,
            votes_by_type=votes_by_type,
            approval_percentage=approval_percentage,
            rejection_percentage=rejection_percentage,
            abstention_percentage=abstention_percentage,
            voting_concluded=True,
            result=voting_result
        )
        
        self.logger.info(
            "Voting concluded",
            proposal_id=str(proposal_id),
            result=voting_result,
            participation_rate=participation_rate,
            approval_percentage=approval_percentage
        )
    
    async def _execute_implementation(self, proposal: SafetyProposal) -> ImplementationResult:
        """
        Execute implementation of approved proposal.
        
        Integration with GovernanceExecutor (Phase 3.3):
        - Creates governance action from proposal
        - Routes through timelock controller
        - Executes actual configuration changes
        """
        implementation_notes = []
        verification_steps = []
        success = True
        
        try:
            # Map proposal type to execution proposal type
            execution_type = self._map_proposal_type_to_execution_type(proposal.proposal_type)
            
            # Build proposal data for execution
            proposal_data = {
                "target_module": proposal.implementation_details.get("target_module", "system"),
                "target_parameter": proposal.implementation_details.get("target_parameter", ""),
                "current_value": proposal.implementation_details.get("current_value"),
                "proposed_value": proposal.implementation_details.get("proposed_value"),
                "proposer_id": proposal.proposer_id,
                "urgency": _urgency_level_value(proposal.urgency_level),
                "metadata": proposal.implementation_details.get("metadata", {
                    "title": proposal.title,
                    "description": proposal.description,
                    "affected_components": proposal.affected_components,
                })
            }
            
            # Execute through governance executor
            execution_result = await self.governance_executor.execute_proposal(
                proposal_id=str(proposal.proposal_id),
                proposal_type=execution_type.value,
                proposal_data=proposal_data
            )
            
            # Track execution
            self.execution_results[proposal.proposal_id] = execution_result
            self.pending_executions[proposal.proposal_id] = execution_result.action_id
            
            if execution_result.success:
                implementation_notes.append(f"Governance action executed: {execution_result.message}")
                verification_steps.extend(execution_result.verification_steps)
                
                # If action was scheduled (not immediately executed), note the timelock
                if "scheduled" in execution_result.message.lower():
                    implementation_notes.append(f"Action ID: {execution_result.action_id}")
                    verification_steps.append("Action scheduled with timelock")
            else:
                success = False
                implementation_notes.append(f"Execution failed: {execution_result.error_details or execution_result.message}")
                
            # Add to implementation queue for monitoring
            self.implementation_queue.append(proposal.proposal_id)
            
        except Exception as e:
            success = False
            implementation_notes.append(f"Implementation failed: {str(e)}")
            self.logger.error(
                "Governance execution failed",
                proposal_id=str(proposal.proposal_id),
                error=str(e)
            )
        
        return ImplementationResult(
            proposal_id=proposal.proposal_id,
            implemented_by=self.governance_id,
            success=success,
            implementation_notes="; ".join(implementation_notes),
            verification_steps=verification_steps,
            monitoring_metrics={
                "implementation_duration": 0.1,  # Will be updated by executor
                "verification_passed": success,
                "action_id": self.pending_executions.get(proposal.proposal_id, "")
            }
        )
    
    def _map_proposal_type_to_execution_type(self, proposal_type: ProposalType) -> ExecutionProposalType:
        """Map safety proposal type to execution proposal type."""
        mapping = {
            ProposalType.SAFETY_POLICY: ExecutionProposalType.PARAMETER_CHANGE,
            ProposalType.CIRCUIT_BREAKER_CONFIG: ExecutionProposalType.CIRCUIT_BREAKER,
            ProposalType.MODEL_RESTRICTIONS: ExecutionProposalType.ACCESS_CONTROL,
            ProposalType.EMERGENCY_PROCEDURE: ExecutionProposalType.EMERGENCY_ACTION,
            ProposalType.NETWORK_PROTOCOL: ExecutionProposalType.PROTOCOL_UPGRADE,
            ProposalType.GOVERNANCE_RULE: ExecutionProposalType.PARAMETER_CHANGE,
            ProposalType.MONITORING_ENHANCEMENT: ExecutionProposalType.PARAMETER_CHANGE,
        }
        return mapping.get(proposal_type, ExecutionProposalType.PARAMETER_CHANGE)
    
    async def get_execution_status(self, proposal_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get the execution status of a proposal.
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            Execution status dict or None if not found
        """
        if proposal_id not in self.proposals:
            return None
        
        # Check if we have an execution result
        if proposal_id in self.execution_results:
            result = self.execution_results[proposal_id]
            return {
                "proposal_id": str(proposal_id),
                "action_id": result.action_id,
                "success": result.success,
                "message": result.message,
                "execution_time": result.execution_time.isoformat(),
                "verification_steps": result.verification_steps,
                "affected_resources": result.affected_resources,
            }
        
        # Check if there's a pending execution
        if proposal_id in self.pending_executions:
            action_id = self.pending_executions[proposal_id]
            record = await self.governance_executor.timelock.get_action_record(action_id)
            if record:
                return {
                    "proposal_id": str(proposal_id),
                    "action_id": action_id,
                    "status": record.status.value,
                    "scheduled_at": record.scheduled_at.isoformat(),
                    "execute_at": record.execute_at.isoformat(),
                    "is_ready": record.is_ready,
                    "time_remaining_seconds": record.time_remaining.total_seconds(),
                }
        
        return None
    
    async def process_ready_executions(self) -> List[Dict[str, Any]]:
        """
        Process all executions that are ready.
        
        This should be called periodically by a background task.
        
        Returns:
            List of execution results
        """
        results = await self.governance_executor.process_ready_executions()
        
        # Update tracking
        for result in results:
            for proposal_id, action_id in list(self.pending_executions.items()):
                if action_id == result.action_id:
                    self.execution_results[proposal_id] = result
                    break
        
        return [r.to_dict() for r in results]
    
    async def cancel_execution(self, proposal_id: UUID, reason: str) -> bool:
        """
        Cancel a pending execution.
        
        Args:
            proposal_id: ID of the proposal
            reason: Reason for cancellation
            
        Returns:
            True if cancelled successfully
        """
        if proposal_id not in self.pending_executions:
            self.logger.warning("No pending execution for proposal", proposal_id=str(proposal_id))
            return False
        
        action_id = self.pending_executions[proposal_id]
        success = await self.governance_executor.cancel_execution(action_id, reason)
        
        if success:
            del self.pending_executions[proposal_id]
            self.logger.info(
                "Execution cancelled",
                proposal_id=str(proposal_id),
                action_id=action_id,
                reason=reason
            )
        
        return success
    
    def set_execution_services(
        self,
        ftns_service=None,
        staking_manager=None,
        network_manager=None
    ):
        """
        Set service references for governance execution.
        
        Args:
            ftns_service: FTNS service instance
            staking_manager: Staking manager instance
            network_manager: Network manager instance
        """
        if ftns_service:
            self.governance_executor.set_ftns_service(ftns_service)
        if staking_manager:
            self.governance_executor.set_staking_manager(staking_manager)
        if network_manager:
            self.governance_executor.set_network_manager(network_manager)