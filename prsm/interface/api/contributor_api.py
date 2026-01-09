"""
FTNS Contributor API
REST API endpoints for contributor status and proof-of-contribution

This module provides HTTP endpoints for:
- Submitting contribution proofs for verification
- Checking contributor status and eligibility
- Retrieving contribution history and metrics
- Managing contribution rewards and multipliers

All endpoints require authentication and integrate with the ContributorManager
for proof verification and status management.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field, validator
import structlog

from prsm.core.auth import get_current_user
from prsm.core.database import get_db_session
from prsm.economy.tokenomics.contributor_manager import ContributorManager, ContributionProofRequest
from prsm.economy.tokenomics.models import ContributorTier, ContributionType, ProofStatus
from prsm.core.ipfs_client import get_ipfs_client

logger = structlog.get_logger(__name__)
router = APIRouter()


# === REQUEST/RESPONSE MODELS ===

class ContributionProofSubmission(BaseModel):
    """Request model for submitting contribution proofs"""
    contribution_type: str = Field(..., description="Type of contribution being proved")
    proof_data: Dict[str, Any] = Field(..., description="Proof data and metadata")
    expected_value: Optional[Decimal] = Field(None, description="Expected contribution value")
    quality_claim: Optional[float] = Field(None, ge=0.0, le=1.0, description="Claimed quality score")
    
    @validator('contribution_type')
    def validate_contribution_type(cls, v):
        valid_types = [ct.value for ct in ContributionType]
        if v not in valid_types:
            raise ValueError(f"Invalid contribution type. Must be one of: {valid_types}")
        return v


class ContributionProofResponse(BaseModel):
    """Response model for successful proof submission"""
    proof_id: UUID
    contribution_type: str
    contribution_value: Decimal
    quality_score: float
    verification_confidence: float
    new_contributor_status: str
    earning_multiplier: float
    verification_timestamp: datetime


class ContributorStatusResponse(BaseModel):
    """Response model for contributor status queries"""
    user_id: str
    contributor_tier: str
    contribution_score: Decimal
    last_contribution_date: Optional[datetime]
    can_earn_ftns: bool
    earning_multiplier: float
    grace_period_expires: Optional[datetime]
    total_proofs_submitted: int
    verified_proofs: int
    contribution_breakdown: Dict[str, int]


class ContributionHistoryResponse(BaseModel):
    """Response model for contribution history"""
    total_count: int
    page: int
    page_size: int
    proofs: List[Dict[str, Any]]


class ContributionMetricsResponse(BaseModel):
    """Response model for contribution metrics"""
    user_id: str
    period_type: str
    metrics: Dict[str, Any]
    performance_trends: Dict[str, float]
    tier_progression: List[Dict[str, Any]]


# === API ENDPOINTS ===

@router.post("/submit-proof", response_model=ContributionProofResponse)
async def submit_contribution_proof(
    proof_submission: ContributionProofSubmission,
    current_user=Depends(get_current_user),
    db=Depends(get_db_session),
    ipfs_client=Depends(get_ipfs_client)
):
    """
    Submit proof of contribution for verification
    
    This endpoint allows users to submit cryptographic proofs of their
    contributions to the PRSM network. Successful verification updates
    their contributor status and earning eligibility.
    """
    
    try:
        # Initialize contributor manager
        contributor_manager = ContributorManager(db, ipfs_client)
        
        # Create proof request
        proof_request = ContributionProofRequest(
            contribution_type=proof_submission.contribution_type,
            proof_data=proof_submission.proof_data,
            expected_value=proof_submission.expected_value,
            quality_claim=proof_submission.quality_claim
        )
        
        # Verify and store proof
        proof = await contributor_manager.verify_contribution_proof(
            current_user.user_id,
            proof_request
        )
        
        # Get updated status
        status = await contributor_manager.get_contributor_status(current_user.user_id)
        multiplier = await contributor_manager.get_contribution_multiplier(current_user.user_id)
        
        await logger.ainfo(
            "Contribution proof submitted successfully",
            user_id=current_user.user_id,
            proof_id=str(proof.proof_id),
            contribution_type=proof_submission.contribution_type,
            value=float(proof.contribution_value)
        )
        
        return ContributionProofResponse(
            proof_id=proof.proof_id,
            contribution_type=proof.contribution_type,
            contribution_value=proof.contribution_value,
            quality_score=proof.quality_score,
            verification_confidence=proof.verification_confidence,
            new_contributor_status=status.status if status else ContributorTier.NONE.value,
            earning_multiplier=multiplier,
            verification_timestamp=proof.verification_timestamp
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        await logger.aerror(
            "Failed to submit contribution proof",
            user_id=current_user.user_id,
            error=str(e),
            contribution_type=proof_submission.contribution_type
        )
        raise HTTPException(status_code=500, detail="Internal server error during proof submission")


@router.get("/status", response_model=ContributorStatusResponse)
async def get_contributor_status(
    current_user=Depends(get_current_user),
    db=Depends(get_db_session)
):
    """
    Get current contributor status and eligibility
    
    Returns comprehensive information about the user's contributor status,
    earning eligibility, and contribution metrics.
    """
    
    try:
        contributor_manager = ContributorManager(db)
        
        # Get contributor status
        status = await contributor_manager.get_contributor_status(current_user.user_id)
        can_earn = await contributor_manager.can_earn_ftns(current_user.user_id)
        multiplier = await contributor_manager.get_contribution_multiplier(current_user.user_id)
        
        # Get contribution counts
        recent_proofs = await contributor_manager._get_recent_proofs(current_user.user_id, days=30)
        
        # Calculate breakdown by contribution type
        contribution_breakdown = {}
        for contrib_type in ContributionType:
            count = sum(1 for proof in recent_proofs if proof.contribution_type == contrib_type.value)
            if count > 0:
                contribution_breakdown[contrib_type.value] = count
        
        # Get total proof counts
        all_proofs = await contributor_manager._get_recent_proofs(current_user.user_id, days=365)
        verified_proofs = len([p for p in all_proofs if p.verification_status == ProofStatus.VERIFIED.value])
        
        return ContributorStatusResponse(
            user_id=current_user.user_id,
            contributor_tier=status.status if status else ContributorTier.NONE.value,
            contribution_score=status.contribution_score if status else Decimal('0.0'),
            last_contribution_date=status.last_contribution_date if status else None,
            can_earn_ftns=can_earn,
            earning_multiplier=multiplier,
            grace_period_expires=status.grace_period_expires if status else None,
            total_proofs_submitted=len(all_proofs),
            verified_proofs=verified_proofs,
            contribution_breakdown=contribution_breakdown
        )
        
    except Exception as e:
        await logger.aerror(
            "Failed to get contributor status",
            user_id=current_user.user_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error retrieving status")


@router.get("/history", response_model=ContributionHistoryResponse)
async def get_contribution_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    contribution_type: Optional[str] = Query(None, description="Filter by contribution type"),
    status: Optional[str] = Query(None, description="Filter by verification status"),
    days: Optional[int] = Query(None, ge=1, le=365, description="Number of days to look back"),
    current_user=Depends(get_current_user),
    db=Depends(get_db_session)
):
    """
    Get contribution history with filtering and pagination
    
    Returns a paginated list of contribution proofs submitted by the user,
    with optional filtering by type, status, and time period.
    """
    
    try:
        contributor_manager = ContributorManager(db)
        
        # Build query filters
        days_back = days or 90  # Default to 90 days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        # Get filtered proofs (simplified - would use proper SQLAlchemy query in production)
        all_proofs = await contributor_manager._get_recent_proofs(current_user.user_id, days=days_back)
        
        # Apply filters
        filtered_proofs = all_proofs
        if contribution_type:
            filtered_proofs = [p for p in filtered_proofs if p.contribution_type == contribution_type]
        if status:
            filtered_proofs = [p for p in filtered_proofs if p.verification_status == status]
        
        # Apply pagination
        total_count = len(filtered_proofs)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_proofs = filtered_proofs[start_idx:end_idx]
        
        # Convert to response format
        proof_data = []
        for proof in page_proofs:
            proof_data.append({
                "proof_id": str(proof.proof_id),
                "contribution_type": proof.contribution_type,
                "contribution_value": float(proof.contribution_value),
                "quality_score": proof.quality_score,
                "verification_status": proof.verification_status,
                "verification_confidence": proof.verification_confidence,
                "verification_timestamp": proof.verification_timestamp,
                "submitted_at": proof.submitted_at,
                "proof_hash": proof.proof_hash
            })
        
        return ContributionHistoryResponse(
            total_count=total_count,
            page=page,
            page_size=page_size,
            proofs=proof_data
        )
        
    except Exception as e:
        await logger.aerror(
            "Failed to get contribution history",
            user_id=current_user.user_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error retrieving history")


@router.get("/metrics", response_model=ContributionMetricsResponse)
async def get_contribution_metrics(
    period: str = Query("monthly", regex="^(daily|weekly|monthly|quarterly)$"),
    periods: int = Query(6, ge=1, le=24, description="Number of periods to include"),
    current_user=Depends(get_current_user),
    db=Depends(get_db_session)
):
    """
    Get contribution metrics and trends
    
    Returns aggregated metrics showing contribution patterns, quality trends,
    and tier progression over time.
    """
    
    try:
        contributor_manager = ContributorManager(db)
        
        # Calculate time periods
        now = datetime.now(timezone.utc)
        if period == "daily":
            period_delta = timedelta(days=1)
        elif period == "weekly":
            period_delta = timedelta(weeks=1)
        elif period == "monthly":
            period_delta = timedelta(days=30)
        else:  # quarterly
            period_delta = timedelta(days=90)
        
        # Get contribution data for each period
        metrics_data = {}
        performance_trends = {}
        tier_progression = []
        
        for i in range(periods):
            period_end = now - (period_delta * i)
            period_start = period_end - period_delta
            
            period_proofs = await contributor_manager._get_recent_proofs(
                current_user.user_id, 
                days=int((now - period_start).days)
            )
            
            # Filter to this specific period
            period_proofs = [
                p for p in period_proofs 
                if period_start <= p.verification_timestamp <= period_end
            ]
            
            # Calculate metrics for this period
            if period_proofs:
                total_value = sum(float(p.contribution_value) for p in period_proofs)
                avg_quality = sum(p.quality_score for p in period_proofs) / len(period_proofs)
                contribution_types = len(set(p.contribution_type for p in period_proofs))
                
                period_key = period_start.strftime("%Y-%m-%d")
                metrics_data[period_key] = {
                    "total_contributions": len(period_proofs),
                    "total_value": total_value,
                    "average_quality": avg_quality,
                    "unique_types": contribution_types,
                    "period_start": period_start,
                    "period_end": period_end
                }
        
        # Calculate performance trends
        if len(metrics_data) >= 2:
            recent_periods = sorted(metrics_data.items())[-2:]
            if len(recent_periods) == 2:
                prev_data = recent_periods[0][1]
                curr_data = recent_periods[1][1]
                
                performance_trends = {
                    "contribution_count_trend": (
                        (curr_data["total_contributions"] - prev_data["total_contributions"]) /
                        max(prev_data["total_contributions"], 1) * 100
                    ),
                    "quality_trend": (
                        (curr_data["average_quality"] - prev_data["average_quality"]) * 100
                    ),
                    "value_trend": (
                        (curr_data["total_value"] - prev_data["total_value"]) /
                        max(prev_data["total_value"], 1) * 100
                    )
                }
        
        # Get tier progression (simplified)
        current_status = await contributor_manager.get_contributor_status(current_user.user_id)
        if current_status:
            tier_progression.append({
                "date": current_status.last_status_update,
                "tier": current_status.status,
                "score": float(current_status.contribution_score)
            })
        
        return ContributionMetricsResponse(
            user_id=current_user.user_id,
            period_type=period,
            metrics=metrics_data,
            performance_trends=performance_trends,
            tier_progression=tier_progression
        )
        
    except Exception as e:
        await logger.aerror(
            "Failed to get contribution metrics",
            user_id=current_user.user_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error retrieving metrics")


@router.post("/update-status")
async def force_status_update(
    current_user=Depends(get_current_user),
    db=Depends(get_db_session)
):
    """
    Force an immediate update of contributor status
    
    Manually triggers recalculation of contributor tier based on recent
    contributions. Useful after submitting multiple proofs.
    """
    
    try:
        contributor_manager = ContributorManager(db)
        
        new_tier = await contributor_manager.update_contributor_status(current_user.user_id)
        multiplier = await contributor_manager.get_contribution_multiplier(current_user.user_id)
        
        await logger.ainfo(
            "Contributor status updated",
            user_id=current_user.user_id,
            new_tier=new_tier.value
        )
        
        return {
            "success": True,
            "new_tier": new_tier.value,
            "earning_multiplier": multiplier,
            "updated_at": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        await logger.aerror(
            "Failed to update contributor status",
            user_id=current_user.user_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error updating status")


@router.get("/validation-criteria/{contribution_type}")
async def get_validation_criteria(
    contribution_type: str = Path(..., description="Contribution type to get criteria for")
):
    """
    Get validation criteria for a specific contribution type
    
    Returns the requirements and validation rules for submitting
    proofs of a particular contribution type.
    """
    
    try:
        # Validate contribution type
        valid_types = [ct.value for ct in ContributionType]
        if contribution_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid contribution type. Must be one of: {valid_types}")
        
        contributor_manager = ContributorManager(None)  # No DB session needed for criteria
        criteria = contributor_manager._get_validation_criteria(contribution_type)
        
        # Add additional helpful information
        type_info = {
            ContributionType.STORAGE.value: {
                "description": "Provide distributed storage for PRSM network data",
                "required_fields": ["ipfs_hashes", "storage_duration_hours", "redundancy_factor"],
                "example_value_range": "0.01-10.0 FTNS per GB-hour"
            },
            ContributionType.COMPUTE.value: {
                "description": "Provide computational resources for AI processing",
                "required_fields": ["work_units_completed", "computation_hash", "benchmark_score"],
                "example_value_range": "0.05-5.0 FTNS per work unit"
            },
            ContributionType.DATA.value: {
                "description": "Contribute high-quality datasets for AI training",
                "required_fields": ["dataset_hash", "peer_reviews", "quality_metrics"],
                "example_value_range": "10.0-50.0 FTNS per dataset"
            },
            ContributionType.GOVERNANCE.value: {
                "description": "Participate in network governance and decision making",
                "required_fields": ["votes_cast", "proposals_reviewed", "participation_period"],
                "example_value_range": "2.0-20.0 FTNS per participation action"
            },
            ContributionType.DOCUMENTATION.value: {
                "description": "Create documentation and educational content",
                "required_fields": ["documents_contributed", "peer_approvals", "content_hash"],
                "example_value_range": "5.0-30.0 FTNS per document"
            },
            ContributionType.MODEL.value: {
                "description": "Contribute AI models to the network",
                "required_fields": ["model_hash", "performance_metrics", "validation_results"],
                "example_value_range": "50.0-500.0 FTNS per model"
            },
            ContributionType.RESEARCH.value: {
                "description": "Publish research and scientific contributions",
                "required_fields": ["publication_hash", "citations", "peer_review_status"],
                "example_value_range": "100.0-2000.0 FTNS per publication"
            },
            ContributionType.TEACHING.value: {
                "description": "Provide educational content and instruction",
                "required_fields": ["content_hash", "student_feedback", "completion_rates"],
                "example_value_range": "20.0-200.0 FTNS per teaching session"
            }
        }
        
        response = {
            "contribution_type": contribution_type,
            "validation_criteria": criteria,
            "type_info": type_info.get(contribution_type, {}),
            "tier_requirements": {
                "basic": "10+ contribution points with 30%+ average quality",
                "active": "50+ contribution points with 60%+ average quality", 
                "power_user": "150+ contribution points with 80%+ average quality"
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        await logger.aerror(
            "Failed to get validation criteria",
            contribution_type=contribution_type,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error retrieving criteria")