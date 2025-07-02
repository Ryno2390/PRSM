"""
PRSM Teams API Endpoints

RESTful API endpoints for team management functionality including
team creation, membership, governance, and collaborative operations.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..teams.service import get_team_service
from ..teams.wallet import get_team_wallet_service
from ..teams.governance import get_team_governance_service
from ..teams.models import (
    Team, TeamMember, TeamTask, TeamProposal,
    TeamRole, TeamType, GovernanceModel, RewardPolicy
)

# Initialize router
router = APIRouter(prefix="/api/v1/teams", tags=["teams"])

# Service dependencies
team_service = get_team_service()
wallet_service = get_team_wallet_service()
governance_service = get_team_governance_service()


# === Request/Response Models ===

class CreateTeamRequest(BaseModel):
    """Request model for creating a team"""
    name: str = Field(..., min_length=3, max_length=100)
    description: str = Field(..., min_length=10, max_length=1000)
    team_type: TeamType = TeamType.RESEARCH
    governance_model: GovernanceModel = GovernanceModel.DEMOCRATIC
    reward_policy: RewardPolicy = RewardPolicy.PROPORTIONAL
    is_public: bool = True
    max_members: Optional[int] = None
    entry_stake_required: float = Field(default=0.0, ge=0.0)
    research_domains: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    external_links: Dict[str, str] = Field(default_factory=dict)
    contact_info: Dict[str, str] = Field(default_factory=dict)


class TeamResponse(BaseModel):
    """Response model for team data"""
    team_id: str
    name: str
    description: str
    team_type: str
    governance_model: str
    reward_policy: str
    is_public: bool
    max_members: Optional[int]
    member_count: int
    total_ftns_earned: float
    impact_score: float
    is_active: bool
    founding_date: datetime
    research_domains: List[str]
    keywords: List[str]


class InviteMemberRequest(BaseModel):
    """Request model for inviting team members"""
    invitee_id: str
    role: TeamRole = TeamRole.MEMBER
    message: Optional[str] = None


class CreateTaskRequest(BaseModel):
    """Request model for creating team tasks"""
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10, max_length=2000)
    task_type: str = "research"
    priority: str = "medium"
    assigned_to: List[str] = Field(default_factory=list)
    ftns_budget: float = Field(default=0.0, ge=0.0)
    due_date: Optional[datetime] = None
    estimated_hours: Optional[float] = None
    requires_consensus: bool = False
    consensus_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)


class CreateProposalRequest(BaseModel):
    """Request model for creating governance proposals"""
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10, max_length=2000)
    proposal_type: str = "general"
    proposed_changes: Dict[str, Any] = Field(default_factory=dict)
    implementation_plan: Optional[str] = None
    estimated_cost: Optional[float] = None


class CastVoteRequest(BaseModel):
    """Request model for casting votes"""
    vote: str = Field(..., pattern="^(for|against|abstain)$")
    rationale: Optional[str] = None


class DepositFTNSRequest(BaseModel):
    """Request model for depositing FTNS to team wallet"""
    amount: float = Field(..., gt=0.0)
    description: str = "Team deposit"


class DistributeRewardsRequest(BaseModel):
    """Request model for distributing team rewards"""
    amount: Optional[float] = None  # None = distribute all available


# === Helper Functions ===

def get_current_user_id() -> str:
    """Get current user ID from authentication context"""
    # Placeholder - in production, this would extract from JWT token or session
    return "current_user_placeholder"


async def validate_team_access(team_id: UUID, user_id: str, required_role: Optional[TeamRole] = None) -> Team:
    """Validate user has access to team with optional role requirement"""
    team = await team_service.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    members = await team_service.get_team_members(team_id)
    user_member = next((m for m in members if m.user_id == user_id), None)
    
    if not user_member:
        raise HTTPException(status_code=403, detail="Not a team member")
    
    if required_role and user_member.role != required_role:
        if required_role == TeamRole.OWNER and user_member.role not in [TeamRole.OWNER]:
            raise HTTPException(status_code=403, detail="Owner role required")
        elif required_role == TeamRole.ADMIN and user_member.role not in [TeamRole.OWNER, TeamRole.ADMIN]:
            raise HTTPException(status_code=403, detail="Admin role required")
    
    return team


# === Team Management Endpoints ===

@router.post("/create", response_model=TeamResponse)
async def create_team(request: CreateTeamRequest, user_id: str = Depends(get_current_user_id)):
    """Create a new team"""
    try:
        team = await team_service.create_team(user_id, request.dict())
        
        return TeamResponse(
            team_id=str(team.team_id),
            name=team.name,
            description=team.description,
            team_type=team.team_type,
            governance_model=team.governance_model,
            reward_policy=team.reward_policy,
            is_public=team.is_public,
            max_members=team.max_members,
            member_count=team.member_count,
            total_ftns_earned=team.total_ftns_earned,
            impact_score=team.impact_score,
            is_active=team.is_active,
            founding_date=team.founding_date,
            research_domains=team.research_domains,
            keywords=team.keywords
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create team")


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(team_id: UUID):
    """Get team details"""
    team = await team_service.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    return TeamResponse(
        team_id=str(team.team_id),
        name=team.name,
        description=team.description,
        team_type=team.team_type,
        governance_model=team.governance_model,
        reward_policy=team.reward_policy,
        is_public=team.is_public,
        max_members=team.max_members,
        member_count=team.member_count,
        total_ftns_earned=team.total_ftns_earned,
        impact_score=team.impact_score,
        is_active=team.is_active,
        founding_date=team.founding_date,
        research_domains=team.research_domains,
        keywords=team.keywords
    )


@router.get("/", response_model=List[TeamResponse])
async def search_teams(
    query: str = Query("", description="Search query"),
    team_type: Optional[TeamType] = Query(None, description="Filter by team type"),
    research_domains: Optional[List[str]] = Query(None, description="Filter by research domains"),
    min_members: Optional[int] = Query(None, description="Minimum number of members"),
    max_members: Optional[int] = Query(None, description="Maximum number of members"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return")
):
    """Search and filter teams"""
    filters = {}
    if team_type:
        filters["team_type"] = team_type
    if research_domains:
        filters["research_domains"] = research_domains
    if min_members is not None:
        filters["min_members"] = min_members
    if max_members is not None:
        filters["max_members"] = max_members
    
    teams = await team_service.search_teams(query, filters)
    
    # Apply limit
    teams = teams[:limit]
    
    return [
        TeamResponse(
            team_id=str(team.team_id),
            name=team.name,
            description=team.description,
            team_type=team.team_type,
            governance_model=team.governance_model,
            reward_policy=team.reward_policy,
            is_public=team.is_public,
            max_members=team.max_members,
            member_count=team.member_count,
            total_ftns_earned=team.total_ftns_earned,
            impact_score=team.impact_score,
            is_active=team.is_active,
            founding_date=team.founding_date,
            research_domains=team.research_domains,
            keywords=team.keywords
        )
        for team in teams
    ]


@router.get("/user/{user_id}", response_model=List[TeamResponse])
async def get_user_teams(user_id: str):
    """Get all teams a user is a member of"""
    teams = await team_service.get_user_teams(user_id)
    
    return [
        TeamResponse(
            team_id=str(team.team_id),
            name=team.name,
            description=team.description,
            team_type=team.team_type,
            governance_model=team.governance_model,
            reward_policy=team.reward_policy,
            is_public=team.is_public,
            max_members=team.max_members,
            member_count=team.member_count,
            total_ftns_earned=team.total_ftns_earned,
            impact_score=team.impact_score,
            is_active=team.is_active,
            founding_date=team.founding_date,
            research_domains=team.research_domains,
            keywords=team.keywords
        )
        for team in teams
    ]


# === Membership Management Endpoints ===

@router.post("/{team_id}/invite")
async def invite_member(
    team_id: UUID, 
    request: InviteMemberRequest, 
    user_id: str = Depends(get_current_user_id)
):
    """Invite a user to join the team"""
    try:
        await validate_team_access(team_id, user_id)
        
        invitation = await team_service.invite_member(
            team_id, user_id, request.invitee_id, request.role, request.message
        )
        
        return {
            "invitation_id": str(invitation.invitation_id),
            "invited_user": invitation.invited_user,
            "role": invitation.role,
            "expires_at": invitation.expires_at,
            "status": "sent"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/invitations/{invitation_id}/accept")
async def accept_invitation(invitation_id: UUID, user_id: str = Depends(get_current_user_id)):
    """Accept a team invitation"""
    try:
        success = await team_service.accept_invitation(invitation_id, user_id)
        
        if success:
            return {"status": "accepted", "message": "Successfully joined team"}
        else:
            raise HTTPException(status_code=400, detail="Failed to accept invitation")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{team_id}/leave")
async def leave_team(team_id: UUID, user_id: str = Depends(get_current_user_id)):
    """Leave a team"""
    try:
        success = await team_service.leave_team(team_id, user_id)
        
        if success:
            return {"status": "left", "message": "Successfully left team"}
        else:
            raise HTTPException(status_code=400, detail="Failed to leave team")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{team_id}/members")
async def get_team_members(team_id: UUID):
    """Get team members list"""
    team = await team_service.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    members = await team_service.get_team_members(team_id)
    
    return [
        {
            "user_id": member.user_id,
            "role": member.role,
            "status": member.status,
            "joined_at": member.joined_at,
            "performance_score": member.performance_score,
            "tasks_completed": member.tasks_completed,
            "ftns_contributed": member.ftns_contributed
        }
        for member in members
    ]


# === Task Management Endpoints ===

@router.post("/{team_id}/tasks", response_model=dict)
async def create_task(
    team_id: UUID, 
    request: CreateTaskRequest, 
    user_id: str = Depends(get_current_user_id)
):
    """Create a new team task"""
    try:
        await validate_team_access(team_id, user_id)
        
        task = await team_service.create_team_task(team_id, user_id, request.dict())
        
        return {
            "task_id": str(task.task_id),
            "title": task.title,
            "description": task.description,
            "status": task.status,
            "created_by": task.created_by,
            "ftns_budget": task.ftns_budget,
            "due_date": task.due_date
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{team_id}/tasks")
async def get_team_tasks(team_id: UUID, status: Optional[str] = Query(None)):
    """Get team tasks with optional status filter"""
    team = await team_service.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # In a full implementation, this would query tasks from the database
    # For now, return placeholder
    return {"message": "Task listing not yet implemented"}


# === Wallet Management Endpoints ===

@router.post("/{team_id}/wallet/deposit")
async def deposit_ftns(
    team_id: UUID, 
    request: DepositFTNSRequest, 
    user_id: str = Depends(get_current_user_id)
):
    """Deposit FTNS tokens to team wallet"""
    try:
        team = await validate_team_access(team_id, user_id)
        
        # Get team wallet (would need to implement wallet retrieval)
        # For now, create a mock wallet
        from ..teams.models import TeamWallet
        wallet = TeamWallet(team_id=team_id)
        
        success = await wallet_service.deposit_ftns(
            wallet, request.amount, user_id, request.description
        )
        
        if success:
            return {
                "status": "success",
                "amount_deposited": request.amount,
                "new_balance": wallet.total_balance
            }
        else:
            raise HTTPException(status_code=400, detail="Deposit failed")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{team_id}/wallet/distribute")
async def distribute_rewards(
    team_id: UUID, 
    request: DistributeRewardsRequest, 
    user_id: str = Depends(get_current_user_id)
):
    """Distribute rewards to team members"""
    try:
        team = await validate_team_access(team_id, user_id, TeamRole.TREASURER)
        
        # Get team wallet and members
        from ..teams.models import TeamWallet
        wallet = TeamWallet(team_id=team_id)
        members = await team_service.get_team_members(team_id)
        
        distributions = await wallet_service.distribute_rewards(
            wallet, members, request.amount
        )
        
        return {
            "status": "success",
            "total_distributed": sum(distributions.values()),
            "distributions": distributions
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{team_id}/wallet")
async def get_wallet_info(team_id: UUID, user_id: str = Depends(get_current_user_id)):
    """Get team wallet information"""
    await validate_team_access(team_id, user_id)
    
    # In a full implementation, this would retrieve actual wallet data
    return {
        "team_id": str(team_id),
        "total_balance": 0.0,
        "available_balance": 0.0,
        "locked_balance": 0.0,
        "reward_policy": "proportional",
        "last_distribution": None
    }


# === Governance Endpoints ===

@router.post("/{team_id}/proposals", response_model=dict)
async def create_proposal(
    team_id: UUID, 
    request: CreateProposalRequest, 
    user_id: str = Depends(get_current_user_id)
):
    """Create a governance proposal"""
    try:
        await validate_team_access(team_id, user_id)
        
        proposal = await governance_service.create_proposal(team_id, user_id, request.dict())
        
        return {
            "proposal_id": str(proposal.proposal_id),
            "title": proposal.title,
            "proposal_type": proposal.proposal_type,
            "voting_ends": proposal.voting_ends,
            "status": proposal.status
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{team_id}/proposals/{proposal_id}/vote")
async def cast_vote(
    team_id: UUID, 
    proposal_id: UUID, 
    request: CastVoteRequest, 
    user_id: str = Depends(get_current_user_id)
):
    """Cast a vote on a proposal"""
    try:
        await validate_team_access(team_id, user_id)
        
        success = await governance_service.cast_vote(
            proposal_id, user_id, request.vote, request.rationale
        )
        
        if success:
            return {"status": "vote_cast", "vote": request.vote}
        else:
            raise HTTPException(status_code=400, detail="Failed to cast vote")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{team_id}/proposals/{proposal_id}/results")
async def get_proposal_results(team_id: UUID, proposal_id: UUID):
    """Get voting results for a proposal"""
    team = await team_service.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    results = await governance_service.get_proposal_results(proposal_id)
    if not results:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    return results


@router.get("/{team_id}/proposals")
async def get_team_proposals(team_id: UUID, status: Optional[str] = Query(None)):
    """Get team proposals with optional status filter"""
    team = await team_service.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # In a full implementation, this would query proposals from the governance service
    return {"message": "Proposal listing not yet implemented"}


# === Statistics Endpoints ===

@router.get("/{team_id}/statistics")
async def get_team_statistics(team_id: UUID):
    """Get comprehensive team statistics"""
    team = await team_service.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    return {
        "team_id": str(team_id),
        "member_count": team.member_count,
        "total_ftns_earned": team.total_ftns_earned,
        "total_tasks_completed": team.total_tasks_completed,
        "impact_score": team.impact_score,
        "founding_date": team.founding_date,
        "active_since_days": (datetime.now(timezone.utc) - team.founding_date).days
    }


@router.get("/system/statistics")
async def get_system_statistics():
    """Get system-wide team statistics"""
    team_stats = await team_service.get_service_statistics()
    wallet_stats = await wallet_service.get_wallet_statistics()
    governance_stats = await governance_service.get_governance_statistics()
    
    return {
        "teams": team_stats,
        "wallets": wallet_stats,
        "governance": governance_stats,
        "timestamp": datetime.now(timezone.utc)
    }