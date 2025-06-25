"""
PRSM Team Governance Service

Implements team-level governance including proposal creation, voting,
and decision-making based on configurable governance models.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

import structlog

from ..governance.voting import get_token_weighted_voting
from .models import (
    Team, TeamMember, TeamGovernance, TeamProposal, TeamVote,
    GovernanceModel, TeamRole, TeamMembershipStatus
)

logger = structlog.get_logger(__name__)


class TeamGovernanceService:
    """
    Team governance service implementing configurable governance models
    
    Governance Models:
    - Autocratic: Founder/owner has full control
    - Democratic: One member = one vote
    - Meritocratic: FTNS-weighted votes + contribution scoring
    - DAO Hybrid: Custom smart contract constitution
    """
    
    def __init__(self):
        self.service_id = str(uuid4())
        self.logger = logger.bind(component="team_governance", service_id=self.service_id)
        
        # Governance state
        self.team_governance: Dict[UUID, TeamGovernance] = {}
        self.proposals: Dict[UUID, TeamProposal] = {}
        self.votes: Dict[UUID, List[TeamVote]] = {}  # proposal_id -> votes
        
        # Integration with main governance system
        self.main_governance = get_token_weighted_voting()
        
        # Performance statistics
        self.governance_stats = {
            "total_governance_configs": 0,
            "total_proposals_created": 0,
            "total_votes_cast": 0,
            "proposals_passed": 0,
            "proposals_failed": 0,
            "average_participation_rate": 0.0
        }
        
        # Synchronization
        self._governance_lock = asyncio.Lock()
        self._proposal_lock = asyncio.Lock()
        self._voting_lock = asyncio.Lock()
        
        print("🗳️ TeamGovernanceService initialized")
    
    
    async def create_team_governance(self, team: Team, constitution: Optional[Dict[str, Any]] = None) -> TeamGovernance:
        """
        Create governance configuration for a team
        
        Args:
            team: Team to create governance for
            constitution: Custom constitutional rules
            
        Returns:
            Created team governance configuration
        """
        try:
            async with self._governance_lock:
                # Create default constitution if none provided
                if constitution is None:
                    constitution = await self._create_default_constitution(team.governance_model)
                
                # Create governance configuration
                governance = TeamGovernance(
                    team_id=team.team_id,
                    model=team.governance_model,
                    constitution=constitution,
                    voting_period_days=constitution.get("voting_period_days", 7),
                    quorum_percentage=constitution.get("quorum_percentage", 0.5),
                    approval_threshold=constitution.get("approval_threshold", 0.6),
                    proposal_types=constitution.get("proposal_types", [
                        "membership", "treasury", "governance", "technical"
                    ]),
                    max_owner_power=constitution.get("max_owner_power", 0.4),
                    member_protection_threshold=constitution.get("member_protection_threshold", 0.25)
                )
                
                # Store governance configuration
                self.team_governance[team.team_id] = governance
                
                # Update statistics
                self.governance_stats["total_governance_configs"] += 1
                
                self.logger.info(
                    "Team governance created",
                    team_id=str(team.team_id),
                    model=governance.model,
                    voting_period=governance.voting_period_days
                )
                
                return governance
                
        except Exception as e:
            self.logger.error("Failed to create team governance", error=str(e))
            raise
    
    
    async def create_proposal(self, team_id: UUID, proposer_id: str, 
                             proposal_data: Dict[str, Any]) -> TeamProposal:
        """
        Create a governance proposal for the team
        
        Args:
            team_id: Team to create proposal for
            proposer_id: Member creating the proposal
            proposal_data: Proposal details
            
        Returns:
            Created proposal
        """
        try:
            async with self._proposal_lock:
                # Validate governance exists
                if team_id not in self.team_governance:
                    raise ValueError("Team governance not configured")
                
                governance = self.team_governance[team_id]
                
                # Validate proposer eligibility
                if not await self._can_create_proposals(team_id, proposer_id):
                    raise ValueError("User not authorized to create proposals")
                
                # Validate proposal data
                if not await self._validate_proposal_data(proposal_data):
                    raise ValueError("Invalid proposal data")
                
                # Determine voting period
                voting_period = timedelta(days=governance.voting_period_days)
                
                # Create proposal
                proposal = TeamProposal(
                    team_id=team_id,
                    title=proposal_data["title"],
                    description=proposal_data["description"],
                    proposal_type=proposal_data.get("proposal_type", "general"),
                    proposed_by=proposer_id,
                    voting_ends=datetime.now(timezone.utc) + voting_period,
                    required_quorum=governance.quorum_percentage,
                    required_approval=await self._get_approval_threshold(governance, proposal_data.get("proposal_type")),
                    proposed_changes=proposal_data.get("proposed_changes", {}),
                    implementation_plan=proposal_data.get("implementation_plan"),
                    estimated_cost=proposal_data.get("estimated_cost")
                )
                
                # Store proposal
                self.proposals[proposal.proposal_id] = proposal
                self.votes[proposal.proposal_id] = []
                
                # Add to active proposals
                governance.active_proposals.append(proposal.proposal_id)
                governance.total_proposals += 1
                
                # Update statistics
                self.governance_stats["total_proposals_created"] += 1
                
                self.logger.info(
                    "Team proposal created",
                    team_id=str(team_id),
                    proposal_id=str(proposal.proposal_id),
                    proposer=proposer_id,
                    title=proposal.title,
                    voting_ends=proposal.voting_ends.isoformat()
                )
                
                return proposal
                
        except Exception as e:
            self.logger.error("Failed to create team proposal", error=str(e))
            raise
    
    
    async def cast_vote(self, proposal_id: UUID, voter_id: str, vote: str, 
                       rationale: Optional[str] = None) -> bool:
        """
        Cast a vote on a team proposal
        
        Args:
            proposal_id: Proposal to vote on
            voter_id: Member casting the vote
            vote: Vote choice ("for", "against", "abstain")
            rationale: Optional reasoning for the vote
            
        Returns:
            True if vote cast successfully
        """
        try:
            async with self._voting_lock:
                # Validate proposal exists
                if proposal_id not in self.proposals:
                    raise ValueError("Proposal not found")
                
                proposal = self.proposals[proposal_id]
                
                # Validate voting period
                current_time = datetime.now(timezone.utc)
                if current_time < proposal.voting_starts or current_time > proposal.voting_ends:
                    raise ValueError("Voting period not active")
                
                # Validate voter eligibility
                if not await self._can_vote(proposal.team_id, voter_id):
                    raise ValueError("User not authorized to vote")
                
                # Check for existing vote
                existing_votes = [v for v in self.votes[proposal_id] if v.voter_id == voter_id]
                if existing_votes:
                    raise ValueError("User has already voted on this proposal")
                
                # Calculate voting power based on governance model
                voting_power = await self._calculate_voting_power(proposal.team_id, voter_id)
                
                # Create vote record
                vote_record = TeamVote(
                    proposal_id=proposal_id,
                    team_id=proposal.team_id,
                    voter_id=voter_id,
                    vote=vote,
                    voting_power=voting_power,
                    rationale=rationale
                )
                
                # Store vote
                self.votes[proposal_id].append(vote_record)
                
                # Update proposal vote counts
                if vote == "for":
                    proposal.votes_for += 1
                    proposal.power_for += voting_power
                elif vote == "against":
                    proposal.votes_against += 1
                    proposal.power_against += voting_power
                elif vote == "abstain":
                    proposal.votes_abstain += 1
                    proposal.power_abstain += voting_power
                
                # Check if proposal should conclude
                if await self._should_conclude_voting(proposal_id):
                    await self._conclude_proposal(proposal_id)
                
                # Update statistics
                self.governance_stats["total_votes_cast"] += 1
                
                self.logger.info(
                    "Team vote cast",
                    proposal_id=str(proposal_id),
                    voter=voter_id,
                    vote=vote,
                    voting_power=voting_power
                )
                
                return True
                
        except Exception as e:
            self.logger.error("Failed to cast team vote", error=str(e))
            return False
    
    
    async def get_proposal_results(self, proposal_id: UUID) -> Optional[Dict[str, Any]]:
        """Get voting results for a proposal"""
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        votes = self.votes[proposal_id]
        
        # Calculate participation
        team_governance = self.team_governance[proposal.team_id]
        eligible_voters = await self._count_eligible_voters(proposal.team_id)
        participation_rate = len(votes) / max(eligible_voters, 1)
        
        # Calculate approval percentage
        total_power = proposal.power_for + proposal.power_against + proposal.power_abstain
        approval_percentage = proposal.power_for / max(total_power, 1)
        
        # Determine if proposal passes
        meets_quorum = participation_rate >= proposal.required_quorum
        meets_approval = approval_percentage >= proposal.required_approval
        proposal_passes = meets_quorum and meets_approval
        
        current_time = datetime.now(timezone.utc)
        
        return {
            "proposal_id": proposal_id,
            "total_votes": len(votes),
            "eligible_voters": eligible_voters,
            "participation_rate": participation_rate,
            "votes_for": proposal.votes_for,
            "votes_against": proposal.votes_against,
            "votes_abstain": proposal.votes_abstain,
            "power_for": proposal.power_for,
            "power_against": proposal.power_against,
            "power_abstain": proposal.power_abstain,
            "approval_percentage": approval_percentage,
            "meets_quorum": meets_quorum,
            "meets_approval": meets_approval,
            "proposal_passes": proposal_passes,
            "status": proposal.status,
            "voting_concluded": current_time > proposal.voting_ends
        }
    
    
    # === Private Helper Methods ===
    
    async def _create_default_constitution(self, governance_model: GovernanceModel) -> Dict[str, Any]:
        """Create default constitutional rules based on governance model"""
        
        base_constitution = {
            "voting_period_days": 7,
            "proposal_types": ["membership", "treasury", "governance", "technical"],
            "emergency_procedures": {
                "emergency_voting_period_hours": 24,
                "emergency_threshold": 0.75
            },
            "member_protection": {
                "min_member_protection_threshold": 0.25,
                "removal_requires_supermajority": True
            }
        }
        
        if governance_model == GovernanceModel.AUTOCRATIC:
            base_constitution.update({
                "quorum_percentage": 0.1,  # Low quorum for owner decisions
                "approval_threshold": 0.5,
                "max_owner_power": 1.0,    # Owner has unlimited power
                "owner_veto_power": True
            })
        
        elif governance_model == GovernanceModel.DEMOCRATIC:
            base_constitution.update({
                "quorum_percentage": 0.5,  # Majority participation required
                "approval_threshold": 0.6, # 60% approval needed
                "max_owner_power": 0.4,    # Owner limited to 40% power
                "equal_voting_rights": True
            })
        
        elif governance_model == GovernanceModel.MERITOCRATIC:
            base_constitution.update({
                "quorum_percentage": 0.4,  # 40% participation required
                "approval_threshold": 0.65, # 65% approval needed
                "max_owner_power": 0.5,    # Owner limited to 50% power
                "contribution_weighting": True,
                "ftns_weight": 0.4,
                "contribution_weight": 0.6
            })
        
        elif governance_model == GovernanceModel.DAO_HYBRID:
            base_constitution.update({
                "quorum_percentage": 0.3,  # 30% participation required
                "approval_threshold": 0.7, # 70% approval needed
                "max_owner_power": 0.3,    # Owner limited to 30% power
                "quadratic_voting": True,
                "delegation_allowed": True,
                "proposal_bond_required": True
            })
        
        return base_constitution
    
    
    async def _can_create_proposals(self, team_id: UUID, user_id: str) -> bool:
        """Check if user can create proposals for the team"""
        # Check if user is team member
        # This would integrate with the main team service
        # For now, assume all active members can create proposals
        return True
    
    
    async def _can_vote(self, team_id: UUID, user_id: str) -> bool:
        """Check if user can vote on team proposals"""
        # Check if user is active team member with voting rights
        # This would integrate with the main team service
        return True
    
    
    async def _validate_proposal_data(self, proposal_data: Dict[str, Any]) -> bool:
        """Validate proposal data"""
        required_fields = ["title", "description"]
        
        for field in required_fields:
            if not proposal_data.get(field):
                return False
        
        # Validate title and description length
        if len(proposal_data["title"]) < 5 or len(proposal_data["title"]) > 200:
            return False
        
        if len(proposal_data["description"]) < 10 or len(proposal_data["description"]) > 2000:
            return False
        
        return True
    
    
    async def _get_approval_threshold(self, governance: TeamGovernance, proposal_type: Optional[str]) -> float:
        """Get approval threshold based on proposal type"""
        # Check for type-specific thresholds
        if proposal_type and proposal_type in governance.type_thresholds:
            return governance.type_thresholds[proposal_type]
        
        # Return default threshold
        return governance.approval_threshold
    
    
    async def _calculate_voting_power(self, team_id: UUID, user_id: str) -> float:
        """Calculate voting power based on governance model"""
        governance = self.team_governance[team_id]
        
        if governance.model == GovernanceModel.AUTOCRATIC:
            # Check if user is owner
            # If owner, gets majority power; others get minimal power
            return 1.0  # Simplified
        
        elif governance.model == GovernanceModel.DEMOCRATIC:
            # Equal voting power for all members
            return 1.0
        
        elif governance.model == GovernanceModel.MERITOCRATIC:
            # Power based on FTNS contribution and performance
            # This would integrate with member data
            return 1.0  # Simplified
        
        elif governance.model == GovernanceModel.DAO_HYBRID:
            # Custom calculation based on constitution
            return 1.0  # Simplified
        
        return 1.0
    
    
    async def _count_eligible_voters(self, team_id: UUID) -> int:
        """Count eligible voters for the team"""
        # This would integrate with the main team service
        # For now, return a placeholder count
        return 10
    
    
    async def _should_conclude_voting(self, proposal_id: UUID) -> bool:
        """Check if voting should conclude early"""
        proposal = self.proposals[proposal_id]
        
        # Check if voting period ended
        if datetime.now(timezone.utc) > proposal.voting_ends:
            return True
        
        # Check for overwhelming majority
        total_power = proposal.power_for + proposal.power_against
        if total_power > 0:
            approval_rate = proposal.power_for / total_power
            if approval_rate >= 0.9 or approval_rate <= 0.1:
                return True
        
        return False
    
    
    async def _conclude_proposal(self, proposal_id: UUID):
        """Conclude voting and update proposal status"""
        proposal = self.proposals[proposal_id]
        results = await self.get_proposal_results(proposal_id)
        
        if results and results["proposal_passes"]:
            proposal.status = "passed"
            self.governance_stats["proposals_passed"] += 1
        else:
            proposal.status = "failed"
            self.governance_stats["proposals_failed"] += 1
        
        # Remove from active proposals
        governance = self.team_governance[proposal.team_id]
        if proposal_id in governance.active_proposals:
            governance.active_proposals.remove(proposal_id)
        
        # Update governance statistics
        governance.proposals_passed += (1 if proposal.status == "passed" else 0)
        
        # Update participation rate
        if results:
            current_avg = governance.average_participation
            total_proposals = governance.total_proposals
            governance.average_participation = (
                (current_avg * (total_proposals - 1) + results["participation_rate"]) / total_proposals
            )
        
        governance.last_vote = datetime.now(timezone.utc)
        
        self.logger.info(
            "Team proposal concluded",
            proposal_id=str(proposal_id),
            status=proposal.status,
            participation_rate=results["participation_rate"] if results else 0,
            approval_percentage=results["approval_percentage"] if results else 0
        )
    
    
    async def get_governance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive governance statistics"""
        # Calculate average participation across all teams
        if self.team_governance:
            avg_participation = sum(g.average_participation for g in self.team_governance.values()) / len(self.team_governance)
            self.governance_stats["average_participation_rate"] = avg_participation
        
        return {
            **self.governance_stats,
            "active_proposals": sum(len(g.active_proposals) for g in self.team_governance.values()),
            "governance_models_distribution": {
                model: sum(1 for g in self.team_governance.values() if g.model == model)
                for model in GovernanceModel
            },
            "total_active_governance_configs": len(self.team_governance)
        }


# === Global Service Instance ===

_governance_service_instance: Optional[TeamGovernanceService] = None

def get_team_governance_service() -> TeamGovernanceService:
    """Get or create the global team governance service instance"""
    global _governance_service_instance
    if _governance_service_instance is None:
        _governance_service_instance = TeamGovernanceService()
    return _governance_service_instance