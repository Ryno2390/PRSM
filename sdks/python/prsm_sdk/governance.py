"""
PRSM SDK Governance Client
Participate in PRSM governance and voting
"""

import structlog
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from .exceptions import PRSMError

logger = structlog.get_logger(__name__)


class ProposalStatus(str, Enum):
    """Status of a governance proposal"""
    DRAFT = "draft"
    ACTIVE = "active"
    VOTING = "voting"
    PASSED = "passed"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"


class ProposalType(str, Enum):
    """Types of governance proposals"""
    PARAMETER_CHANGE = "parameter_change"
    PROTOCOL_UPGRADE = "protocol_upgrade"
    TREASURY_SPEND = "treasury_spend"
    MODEL_ADDITION = "model_addition"
    MODEL_REMOVAL = "model_removal"
    FEE_ADJUSTMENT = "fee_adjustment"
    GOVERNANCE_CHANGE = "governance_change"
    OTHER = "other"


class VoteChoice(str, Enum):
    """Vote options"""
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


class Proposal(BaseModel):
    """Governance proposal"""
    proposal_id: str = Field(..., description="Unique proposal ID")
    title: str = Field(..., description="Proposal title")
    description: str = Field(..., description="Proposal description")
    proposal_type: ProposalType = Field(..., description="Type of proposal")
    status: ProposalStatus = Field(..., description="Current status")
    proposer: str = Field(..., description="Proposer address")
    created_at: datetime = Field(..., description="Creation timestamp")
    voting_starts: datetime = Field(..., description="Voting start time")
    voting_ends: datetime = Field(..., description="Voting end time")
    quorum: float = Field(..., description="Required quorum percentage")
    threshold: float = Field(..., description="Required approval threshold")
    votes_yes: float = Field(0, description="Yes votes (FTNS)")
    votes_no: float = Field(0, description="No votes (FTNS)")
    votes_abstain: float = Field(0, description="Abstain votes (FTNS)")
    total_voters: int = Field(0, description="Total voters")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Proposal parameters")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ProposalCreate(BaseModel):
    """Request to create a proposal"""
    title: str = Field(..., min_length=10, max_length=200, description="Proposal title")
    description: str = Field(..., min_length=50, description="Proposal description")
    proposal_type: ProposalType = Field(..., description="Type of proposal")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Proposal parameters")
    duration_days: int = Field(7, ge=1, le=30, description="Voting duration in days")
    quorum: float = Field(0.1, ge=0.01, le=1, description="Required quorum")
    threshold: float = Field(0.5, ge=0.1, le=1, description="Approval threshold")


class Vote(BaseModel):
    """Vote record"""
    vote_id: str = Field(..., description="Vote ID")
    proposal_id: str = Field(..., description="Proposal ID")
    voter: str = Field(..., description="Voter address")
    choice: VoteChoice = Field(..., description="Vote choice")
    voting_power: float = Field(..., description="Voting power (FTNS)")
    timestamp: datetime = Field(..., description="Vote timestamp")
    reason: Optional[str] = Field(None, description="Vote reason")


class VoteRequest(BaseModel):
    """Request to cast a vote"""
    proposal_id: str = Field(..., description="Proposal to vote on")
    choice: VoteChoice = Field(..., description="Vote choice")
    reason: Optional[str] = Field(None, max_length=500, description="Vote reason")


class GovernanceStats(BaseModel):
    """Governance statistics"""
    total_proposals: int = Field(..., description="Total proposals")
    active_proposals: int = Field(..., description="Active proposals")
    total_votes: int = Field(..., description="Total votes cast")
    total_voting_power: float = Field(..., description="Total voting power")
    participation_rate: float = Field(..., description="Participation rate")
    quorum_met_rate: float = Field(..., description="Rate of quorum met")


class DelegationInfo(BaseModel):
    """Voting delegation information"""
    delegator: str = Field(..., description="Delegator address")
    delegate: str = Field(..., description="Delegate address")
    voting_power: float = Field(..., description="Delegated voting power")
    created_at: datetime = Field(..., description="Delegation timestamp")


class GovernanceClient:
    """
    Client for governance operations
    
    Provides methods for:
    - Creating proposals
    - Voting on proposals
    - Delegating votes
    - Viewing governance state
    """
    
    def __init__(self, client):
        """
        Initialize governance client
        
        Args:
            client: PRSMClient instance for making API requests
        """
        self._client = client
    
    async def create_proposal(
        self,
        title: str,
        description: str,
        proposal_type: ProposalType,
        parameters: Optional[Dict[str, Any]] = None,
        duration_days: int = 7,
        quorum: float = 0.1,
        threshold: float = 0.5
    ) -> Proposal:
        """
        Create a new governance proposal
        
        Args:
            title: Proposal title
            description: Proposal description
            proposal_type: Type of proposal
            parameters: Proposal parameters
            duration_days: Voting duration in days
            quorum: Required quorum percentage
            threshold: Required approval threshold
            
        Returns:
            Created Proposal
            
        Example:
            proposal = await client.governance.create_proposal(
                title="Increase FTNS staking rewards",
                description="Proposal to increase staking APY from 5% to 7%",
                proposal_type=ProposalType.PARAMETER_CHANGE,
                parameters={"staking_apy": 0.07}
            )
            print(f"Created proposal: {proposal.proposal_id}")
        """
        request = ProposalCreate(
            title=title,
            description=description,
            proposal_type=proposal_type,
            parameters=parameters or {},
            duration_days=duration_days,
            quorum=quorum,
            threshold=threshold
        )
        
        response = await self._client._request(
            "POST",
            "/governance/proposals",
            json_data=request.model_dump()
        )
        
        return Proposal(**response)
    
    async def get_proposal(self, proposal_id: str) -> Proposal:
        """
        Get proposal details
        
        Args:
            proposal_id: Proposal identifier
            
        Returns:
            Proposal details
            
        Example:
            proposal = await client.governance.get_proposal("prop_123")
            print(f"Status: {proposal.status}")
            print(f"Yes votes: {proposal.votes_yes}")
        """
        response = await self._client._request(
            "GET",
            f"/governance/proposals/{proposal_id}"
        )
        
        return Proposal(**response)
    
    async def list_proposals(
        self,
        status: Optional[ProposalStatus] = None,
        proposal_type: Optional[ProposalType] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Proposal]:
        """
        List governance proposals
        
        Args:
            status: Filter by status
            proposal_type: Filter by type
            limit: Maximum results
            offset: Offset for pagination
            
        Returns:
            List of Proposal objects
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status.value
        if proposal_type:
            params["type"] = proposal_type.value
        
        response = await self._client._request(
            "GET",
            "/governance/proposals",
            params=params
        )
        
        return [Proposal(**p) for p in response.get("proposals", [])]
    
    async def vote(
        self,
        proposal_id: str,
        choice: VoteChoice,
        reason: Optional[str] = None
    ) -> Vote:
        """
        Cast a vote on a proposal
        
        Args:
            proposal_id: Proposal to vote on
            choice: Vote choice (yes/no/abstain)
            reason: Optional reason for vote
            
        Returns:
            Vote record
            
        Example:
            vote = await client.governance.vote(
                proposal_id="prop_123",
                choice=VoteChoice.YES,
                reason="This proposal benefits the network"
            )
            print(f"Voted with {vote.voting_power} voting power")
        """
        request = VoteRequest(
            proposal_id=proposal_id,
            choice=choice,
            reason=reason
        )
        
        response = await self._client._request(
            "POST",
            f"/governance/proposals/{proposal_id}/vote",
            json_data=request.model_dump()
        )
        
        return Vote(**response)
    
    async def get_vote(self, proposal_id: str, voter: Optional[str] = None) -> Optional[Vote]:
        """
        Get vote details
        
        Args:
            proposal_id: Proposal identifier
            voter: Voter address (defaults to current user)
            
        Returns:
            Vote record or None if not voted
        """
        params = {}
        if voter:
            params["voter"] = voter
        
        try:
            response = await self._client._request(
                "GET",
                f"/governance/proposals/{proposal_id}/vote",
                params=params
            )
            return Vote(**response)
        except PRSMError:
            return None
    
    async def get_proposal_votes(
        self,
        proposal_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Vote]:
        """
        Get all votes for a proposal
        
        Args:
            proposal_id: Proposal identifier
            limit: Maximum results
            offset: Offset for pagination
            
        Returns:
            List of Vote objects
        """
        response = await self._client._request(
            "GET",
            f"/governance/proposals/{proposal_id}/votes",
            params={"limit": limit, "offset": offset}
        )
        
        return [Vote(**v) for v in response.get("votes", [])]
    
    async def cancel_proposal(self, proposal_id: str) -> bool:
        """
        Cancel a proposal (proposer only)
        
        Args:
            proposal_id: Proposal to cancel
            
        Returns:
            True if cancelled successfully
        """
        response = await self._client._request(
            "POST",
            f"/governance/proposals/{proposal_id}/cancel"
        )
        
        return response.get("cancelled", False)
    
    async def execute_proposal(self, proposal_id: str) -> bool:
        """
        Execute a passed proposal
        
        Args:
            proposal_id: Proposal to execute
            
        Returns:
            True if executed successfully
        """
        response = await self._client._request(
            "POST",
            f"/governance/proposals/{proposal_id}/execute"
        )
        
        return response.get("executed", False)
    
    async def delegate(
        self,
        delegate_address: str,
        amount: Optional[float] = None
    ) -> DelegationInfo:
        """
        Delegate voting power to another address
        
        Args:
            delegate_address: Address to delegate to
            amount: Amount to delegate (None for all)
            
        Returns:
            DelegationInfo
            
        Example:
            delegation = await client.governance.delegate(
                "0xabc...",
                amount=1000
            )
            print(f"Delegated {delegation.voting_power} voting power")
        """
        data = {"delegate": delegate_address}
        if amount is not None:
            data["amount"] = amount
        
        response = await self._client._request(
            "POST",
            "/governance/delegate",
            json_data=data
        )
        
        return DelegationInfo(**response)
    
    async def undelegate(self, delegate_address: str) -> bool:
        """
        Remove delegation
        
        Args:
            delegate_address: Address to undelegate from
            
        Returns:
            True if undelegated successfully
        """
        response = await self._client._request(
            "POST",
            "/governance/undelegate",
            json_data={"delegate": delegate_address}
        )
        
        return response.get("undelegated", False)
    
    async def get_delegations(self) -> List[DelegationInfo]:
        """
        Get user's delegations
        
        Returns:
            List of DelegationInfo objects
        """
        response = await self._client._request(
            "GET",
            "/governance/delegations"
        )
        
        return [DelegationInfo(**d) for d in response.get("delegations", [])]
    
    async def get_voting_power(self) -> float:
        """
        Get user's voting power
        
        Returns:
            Voting power in FTNS
        """
        response = await self._client._request(
            "GET",
            "/governance/voting-power"
        )
        
        return response.get("voting_power", 0.0)
    
    async def get_stats(self) -> GovernanceStats:
        """
        Get governance statistics
        
        Returns:
            GovernanceStats with overall statistics
        """
        response = await self._client._request(
            "GET",
            "/governance/stats"
        )
        
        return GovernanceStats(**response)
    
    async def get_active_proposals(self) -> List[Proposal]:
        """
        Get all active proposals
        
        Returns:
            List of active Proposal objects
        """
        return await self.list_proposals(status=ProposalStatus.ACTIVE)
    
    async def get_proposal_comments(
        self,
        proposal_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get comments on a proposal
        
        Args:
            proposal_id: Proposal identifier
            limit: Maximum results
            
        Returns:
            List of comment dictionaries
        """
        response = await self._client._request(
            "GET",
            f"/governance/proposals/{proposal_id}/comments",
            params={"limit": limit}
        )
        
        return response.get("comments", [])
    
    async def add_comment(
        self,
        proposal_id: str,
        comment: str
    ) -> Dict[str, Any]:
        """
        Add a comment to a proposal
        
        Args:
            proposal_id: Proposal identifier
            comment: Comment text
            
        Returns:
            Created comment
        """
        response = await self._client._request(
            "POST",
            f"/governance/proposals/{proposal_id}/comments",
            json_data={"comment": comment}
        )
        
        return response