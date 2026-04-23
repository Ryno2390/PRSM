"""
Governance API
==============

REST API endpoints for PRSM governance system including token distribution,
voting activation, and governance participation management.
"""

import structlog
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field

from prsm.core.auth import get_current_user
from prsm.core.auth.models import UserRole
from prsm.core.auth.auth_manager import auth_manager
from prsm.economy.governance.token_distribution import (
    get_governance_distributor, 
    GovernanceParticipantTier, 
    ContributionType
)
from prsm.economy.governance.voting import get_token_weighted_voting
from prsm.economy.governance.quadratic_voting import quadratic_voting
from prsm.core.database import GovernanceQueries

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/governance", tags=["governance"])


# === Request/Response Models ===

class ActivateGovernanceRequest(BaseModel):
    """Request to activate governance participation"""
    participant_tier: GovernanceParticipantTier = GovernanceParticipantTier.COMMUNITY
    council_nominations: List[str] = Field(default_factory=list, description="List of council names to nominate for")
    auto_stake_percentage: float = Field(default=0.5, ge=0.0, le=1.0, description="Percentage of tokens to auto-stake")


class DistributeRewardRequest(BaseModel):
    """Request to distribute contribution rewards"""
    user_id: str
    contribution_type: ContributionType
    contribution_reference: str
    quality_multiplier: float = Field(default=1.0, ge=0.1, le=5.0)
    custom_amount: Optional[Decimal] = None


class StakeTokensRequest(BaseModel):
    """Request to stake tokens for governance"""
    amount: Decimal = Field(gt=0, description="Amount of FTNS tokens to stake")
    lock_duration_days: int = Field(default=30, ge=1, le=365, description="Lock duration in days")


class CastVoteRequest(BaseModel):
    """Request to cast a governance vote"""
    proposal_id: str
    vote_choice: bool = Field(description="True for approve, False for reject")
    voting_power: Optional[Decimal] = None
    rationale: Optional[str] = None


class CreateProposalRequest(BaseModel):
    """Request to create a governance proposal"""
    title: str = Field(min_length=10, max_length=200)
    description: str = Field(min_length=100, max_length=5000)
    proposal_type: str
    implementation_details: Optional[Dict[str, Any]] = None
    budget_impact: Optional[Dict[str, Any]] = None


class GovernanceResponse(BaseModel):
    """Standard governance response"""
    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# === Governance Activation ===

@router.post("/activate", response_model=GovernanceResponse)
async def activate_governance_participation(
    request: ActivateGovernanceRequest,
    current_user: str = Depends(get_current_user)
) -> GovernanceResponse:
    """
    Activate governance participation for the current user
    
    🏛️ GOVERNANCE ACTIVATION:
    - Distributes initial FTNS tokens based on participant tier
    - Sets up voting power through token staking
    - Processes council membership nominations
    - Enables participation in governance proposals and voting
    
    📊 PARTICIPANT TIERS:
    - Community: 1,000 FTNS initial allocation
    - Contributor: 5,000 FTNS initial allocation  
    - Expert: 10,000 FTNS initial allocation
    - Delegate: 25,000 FTNS initial allocation
    - Council Member: 50,000 FTNS initial allocation
    - Core Team: 100,000 FTNS initial allocation
    """
    try:
        distributor = get_governance_distributor()
        
        # Activate governance participation
        activation = await distributor.activate_governance_participation(
            user_id=current_user,
            participant_tier=request.participant_tier,
            council_nominations=request.council_nominations
        )
        
        return GovernanceResponse(
            success=True,
            message=f"✅ Governance participation activated at {request.participant_tier.value} tier",
            data={
                "activation": {
                    "activation_id": str(activation.activation_id),
                    "participant_tier": activation.participant_tier.value,
                    "initial_allocation": str(activation.initial_token_allocation),
                    "staked_amount": str(activation.staked_amount),
                    "voting_power": str(activation.voting_power),
                    "council_memberships": activation.council_memberships,
                    "activated_at": activation.activated_at.isoformat()
                }
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to activate governance participation",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate governance participation"
        )


@router.get("/status", response_model=GovernanceResponse)
async def get_governance_status(
    current_user: str = Depends(get_current_user)
) -> GovernanceResponse:
    """
    Get comprehensive governance status for the current user
    
    📊 STATUS INFORMATION:
    - Governance activation status and tier
    - Current token balances (liquid and staked)
    - Voting power calculation breakdown
    - Council memberships and roles
    - Recent token distributions and rewards
    - Participation statistics
    """
    try:
        distributor = get_governance_distributor()
        
        status_info = await distributor.get_governance_status(current_user)
        
        return GovernanceResponse(
            success=True,
            message="Governance status retrieved successfully",
            data={"governance_status": status_info}
        )
        
    except Exception as e:
        logger.error("Failed to get governance status",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve governance status"
        )


# === Token Distribution ===

@router.post("/rewards/distribute", response_model=GovernanceResponse)
async def distribute_contribution_reward(
    request: DistributeRewardRequest,
    current_user: str = Depends(get_current_user)
) -> GovernanceResponse:
    """
    Distribute token rewards for contributions (Admin/Moderator only)
    
    🎁 CONTRIBUTION REWARDS:
    - Model Contribution: 2,500 FTNS base reward
    - Research Publication: 5,000 FTNS base reward
    - Security Audit: 10,000 FTNS base reward
    - Code Contribution: 3,000 FTNS base reward
    - Documentation: 1,500 FTNS base reward
    - Quality multiplier: 0.1x to 5.0x based on contribution quality
    """
    try:
        # Check admin permissions
        user = await auth_manager.get_user_by_id(current_user)
        if not user or user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required to distribute rewards"
            )
        
        distributor = get_governance_distributor()
        
        # Distribute the reward
        distribution = await distributor.distribute_contribution_rewards(
            user_id=request.user_id,
            contribution_type=request.contribution_type,
            contribution_reference=request.contribution_reference,
            quality_multiplier=request.quality_multiplier,
            custom_amount=request.custom_amount
        )
        
        return GovernanceResponse(
            success=True,
            message=f"🎁 Contribution reward distributed for {request.contribution_type.value}",
            data={
                "distribution": {
                    "distribution_id": str(distribution.distribution_id),
                    "recipient": request.user_id,
                    "amount": str(distribution.amount),
                    "contribution_type": distribution.contribution_type.value,
                    "contribution_reference": distribution.contribution_reference,
                    "distributed_at": distribution.distributed_at.isoformat()
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to distribute contribution reward",
                    admin_user=current_user,
                    target_user=request.user_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to distribute contribution reward"
        )


@router.post("/stake", response_model=GovernanceResponse)
async def stake_tokens_for_governance(
    request: StakeTokensRequest,
    current_user: str = Depends(get_current_user)
) -> GovernanceResponse:
    """
    Stake FTNS tokens for governance voting power
    
    🗳️ STAKING BENEFITS:
    - Increased voting power based on stake amount and lock duration
    - Staking rewards (5% of staked amount)
    - Longer lock periods provide voting power multipliers (up to 4x)
    - Participation in governance decisions and proposals
    """
    try:
        distributor = get_governance_distributor()
        
        # Stake tokens
        staked_amount, voting_power = await distributor.stake_tokens_for_governance(
            user_id=current_user,
            amount=request.amount,
            lock_duration_days=request.lock_duration_days
        )
        
        return GovernanceResponse(
            success=True,
            message=f"✅ {staked_amount} FTNS tokens staked for governance",
            data={
                "staking": {
                    "staked_amount": str(staked_amount),
                    "voting_power": str(voting_power),
                    "lock_duration_days": request.lock_duration_days,
                    "unlock_date": (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + 
                                  timedelta(days=request.lock_duration_days)).isoformat()
                }
            }
        )
        
    except Exception as e:
        logger.error("Failed to stake tokens for governance",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stake tokens for governance"
        )


# === Voting System ===

@router.post("/voting/enable", response_model=GovernanceResponse)
async def enable_voting_system(
    current_user: str = Depends(get_current_user)
) -> GovernanceResponse:
    """
    Enable the complete governance voting system (Admin only)
    
    🏛️ VOTING SYSTEM ACTIVATION:
    - Activates quadratic voting mechanisms
    - Creates federated councils (Safety, Technical, Economic, Governance)
    - Sets up delegation systems
    - Configures proposal management
    - Enables token-weighted voting
    """
    try:
        # Check admin permissions
        user = await auth_manager.get_user_by_id(current_user)
        if not user or user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required to enable voting system"
            )
        
        distributor = get_governance_distributor()
        
        # Enable the complete voting system
        activation_results = await distributor.enable_voting_system()
        
        return GovernanceResponse(
            success=activation_results["voting_system_activated"],
            message="🗳️ Governance voting system activated successfully!",
            data={"activation_results": activation_results}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to enable voting system",
                    admin_user=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enable voting system"
        )


@router.post("/vote", response_model=GovernanceResponse)
async def cast_governance_vote(
    request: CastVoteRequest,
    current_user: str = Depends(get_current_user)
) -> GovernanceResponse:
    """
    Cast a vote on a governance proposal
    
    🗳️ VOTING PROCESS:
    - Uses token-weighted voting power based on staked tokens
    - Supports quadratic voting to reduce whale influence
    - Records vote with optional rationale
    - Rewards governance participation
    - Applies vote to proposal outcome calculation
    """
    try:
        voting_system = get_token_weighted_voting()
        
        # Cast the vote
        vote_success = await voting_system.cast_vote(
            voter_id=current_user,
            proposal_id=UUID(request.proposal_id),
            vote=request.vote_choice,
            rationale=request.rationale
        )
        
        if not vote_success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to cast vote - check eligibility and proposal status"
            )
        
        return GovernanceResponse(
            success=True,
            message=f"✅ Vote cast {'for' if request.vote_choice else 'against'} proposal",
            data={
                "vote": {
                    "proposal_id": request.proposal_id,
                    "vote_choice": request.vote_choice,
                    "rationale": request.rationale,
                    "cast_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to cast governance vote",
                    user_id=current_user,
                    proposal_id=request.proposal_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cast governance vote"
        )


# === Proposals ===

@router.post("/proposals", response_model=GovernanceResponse)
async def create_governance_proposal(
    request: CreateProposalRequest,
    current_user: str = Depends(get_current_user)
) -> GovernanceResponse:
    """
    Create a new governance proposal
    
    📋 PROPOSAL CREATION:
    - Requires minimum FTNS balance for proposal submission fee
    - Validates proposal content and safety
    - Sets up voting timeline based on proposal category
    - Initiates community review and discussion period
    - Enables voting once validation is complete
    """
    try:
        voting_system = get_token_weighted_voting()
        
        # Create the proposal (simplified - would need full GovernanceProposal object)
        from prsm.core.models import GovernanceProposal
        
        proposal = GovernanceProposal(
            title=request.title,
            description=request.description,
            proposal_type=request.proposal_type,
            proposer_id=current_user,
            status="draft"
        )
        
        # Create proposal through voting system
        proposal_id = await voting_system.create_proposal(current_user, proposal)
        
        return GovernanceResponse(
            success=True,
            message="📋 Governance proposal created successfully",
            data={
                "proposal": {
                    "proposal_id": str(proposal_id),
                    "title": request.title,
                    "proposal_type": request.proposal_type,
                    "status": "active",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to create governance proposal",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create governance proposal"
        )


@router.get("/proposals", response_model=GovernanceResponse)
async def list_governance_proposals(
    status_filter: Optional[str] = Query(
        None,
        description="Filter by status: active, draft, approved, rejected, executed"
    ),
    current_user: str = Depends(get_current_user),
) -> GovernanceResponse:
    """
    List governance proposals, optionally filtered by status.

    Returns proposals from the active voting system. Proposals are
    currently in-memory (DB persistence is Phase 2). The list is
    accurate for the lifetime of the current process.
    """
    try:
        voting_system = get_token_weighted_voting()

        proposals = list(voting_system.proposals.values())
        if status_filter:
            proposals = [p for p in proposals if p.status == status_filter]

        proposal_list = [
            {
                "proposal_id": str(p.proposal_id),
                "title": p.title,
                "description": (
                    p.description[:200] + "..."
                    if len(p.description) > 200
                    else p.description
                ),
                "proposal_type": p.proposal_type,
                "status": p.status,
                "proposer_id": p.proposer_id,
                "votes_for": p.votes_for,
                "votes_against": p.votes_against,
                "voting_starts": p.voting_starts.isoformat() if p.voting_starts else None,
                "voting_ends": p.voting_ends.isoformat() if p.voting_ends else None,
            }
            for p in proposals
        ]

        return GovernanceResponse(
            success=True,
            message=f"Found {len(proposal_list)} proposal(s)",
            data={
                "proposals": proposal_list,
                "total_count": len(proposal_list),
                "filter": status_filter or "all",
            },
        )

    except Exception as e:
        logger.error("Failed to list governance proposals", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Governance system temporarily unavailable",
        )


@router.get("/proposals/{proposal_id}", response_model=GovernanceResponse)
async def get_governance_proposal(
    proposal_id: str,
    current_user: str = Depends(get_current_user),
) -> GovernanceResponse:
    """
    Get a specific governance proposal by ID.

    First checks in-memory proposals, then falls back to database lookup.
    Returns full proposal details including voting status and metadata.
    """
    try:
        voting_system = get_token_weighted_voting()
        
        # Try to find in memory first
        proposal_uuid = UUID(proposal_id)
        proposal = voting_system.proposals.get(proposal_uuid)
        
        # If not in memory, try database
        if proposal is None:
            proposal_data = await GovernanceQueries.get_proposal(proposal_id)
            if proposal_data is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Proposal {proposal_id} not found"
                )
            
            # Return database record directly
            return GovernanceResponse(
                success=True,
                message="Proposal retrieved from database",
                data={
                    "proposal_id": proposal_data["proposal_id"],
                    "title": proposal_data["title"],
                    "description": proposal_data["description"],
                    "proposal_type": proposal_data["proposal_type"],
                    "status": proposal_data["status"],
                    "proposer_id": proposal_data["proposer_id"],
                    "votes_for": proposal_data.get("votes_for", 0),
                    "votes_against": proposal_data.get("votes_against", 0),
                    "total_voting_power": proposal_data.get("total_voting_power", 0.0),
                    "voting_starts": proposal_data.get("voting_starts"),
                    "voting_ends": proposal_data.get("voting_ends"),
                    "created_at": proposal_data.get("created_at"),
                    "updated_at": proposal_data.get("updated_at"),
                },
            )
        
        # Return in-memory proposal
        return GovernanceResponse(
            success=True,
            message="Proposal retrieved successfully",
            data={
                "proposal_id": str(proposal.proposal_id),
                "title": proposal.title,
                "description": proposal.description,
                "proposal_type": proposal.proposal_type,
                "status": proposal.status,
                "proposer_id": proposal.proposer_id,
                "votes_for": proposal.votes_for,
                "votes_against": proposal.votes_against,
                "total_voting_power": proposal.total_voting_power,
                "voting_starts": proposal.voting_starts.isoformat() if proposal.voting_starts else None,
                "voting_ends": proposal.voting_ends.isoformat() if proposal.voting_ends else None,
                "created_at": proposal.created_at.isoformat() if proposal.created_at else None,
            },
        )

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid proposal ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get governance proposal", error=str(e), proposal_id=proposal_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve proposal"
        )


# === Statistics and Information ===

@router.get("/statistics", response_model=GovernanceResponse)
async def get_governance_statistics(
    current_user: str = Depends(get_current_user)
) -> GovernanceResponse:
    """
    Get comprehensive governance system statistics
    
    📊 GOVERNANCE METRICS:
    - Total participants and activation rates
    - Token distribution by type and tier
    - Voting system participation rates
    - Council membership and activity
    - Proposal creation and approval rates
    - Staking and delegation statistics
    """
    try:
        distributor = get_governance_distributor()
        voting_system = get_token_weighted_voting()
        
        # Get distribution statistics
        distribution_stats = await distributor.get_distribution_statistics()
        
        # Get voting statistics
        voting_stats = await voting_system.get_governance_statistics()
        
        return GovernanceResponse(
            success=True,
            message="Governance statistics retrieved successfully",
            data={
                "distribution_statistics": distribution_stats,
                "voting_statistics": voting_stats,
                "system_status": {
                    "governance_enabled": True,
                    "quadratic_voting_active": len(quadratic_voting.councils) > 0,
                    "active_councils": len(quadratic_voting.councils),
                    "total_participants": distribution_stats.get("total_participants", 0)
                }
            }
        )
        
    except Exception as e:
        logger.error("Failed to get governance statistics",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve governance statistics"
        )


@router.get("/councils", response_model=GovernanceResponse)
async def get_governance_councils(
    current_user: str = Depends(get_current_user)
) -> GovernanceResponse:
    """
    Get information about all governance councils
    
    🏛️ COUNCIL INFORMATION:
    - List of all active councils with member counts
    - Council types and voting weights
    - Membership requirements and application processes
    - Recent council activities and decisions
    """
    try:
        councils_info = []
        
        for council_id, council in quadratic_voting.councils.items():
            councils_info.append({
                "council_id": str(council_id),
                "name": council.name,
                "type": council.council_type.value,
                "description": council.description,
                "member_count": len(council.members),
                "max_members": council.max_members,
                "voting_weight": council.voting_weight,
                "is_active": council.is_active,
                "created_at": council.created_at.isoformat()
            })
        
        return GovernanceResponse(
            success=True,
            message="Governance councils retrieved successfully",
            data={
                "councils": councils_info,
                "total_councils": len(councils_info)
            }
        )
        
    except Exception as e:
        logger.error("Failed to get governance councils",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve governance councils"
        )