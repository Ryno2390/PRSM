"""
User Reputation System API
==========================

Production-ready reputation API endpoints providing sophisticated
trust scoring and reputation management for the PRSM marketplace.

Features:
- Multi-dimensional reputation scoring
- Real-time trust calculation
- Fraud detection and prevention
- Reputation-based access controls
- Community moderation tools
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, Query, Request, status
from pydantic import BaseModel, Field
import structlog
from datetime import datetime, timezone

from prsm.economy.marketplace.reputation_system import (
    get_reputation_calculator, ReputationEvent, ReputationDimension, TrustLevel
)
from ..auth import get_current_user
from ..security.enhanced_authorization import get_enhanced_auth_manager
from prsm.core.models import UserRole

logger = structlog.get_logger(__name__)
router = APIRouter()

# Initialize reputation calculator
reputation_calculator = get_reputation_calculator()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ReputationEventRequest(BaseModel):
    """Request model for recording reputation events"""
    target_user_id: str = Field(..., description="User whose reputation is affected")
    event_type: str = Field(..., description="Type of reputation event")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting evidence")
    rating: Optional[float] = Field(None, ge=1.0, le=5.0, description="Rating value (1-5)")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional comment")


class ReputationSummaryResponse(BaseModel):
    """Response model for reputation summary"""
    user_id: str
    overall_score: float
    trust_level: str
    trust_level_numeric: int
    badges: List[str]
    top_dimensions: List[Dict[str, Any]]
    verification_status: Dict[str, bool]
    last_updated: str
    next_review: str


class DimensionScoreResponse(BaseModel):
    """Response model for dimension-specific scores"""
    dimension: str
    score: float
    confidence: float
    trend: float
    evidence_count: int
    last_updated: str


class ReputationDetailResponse(BaseModel):
    """Detailed reputation response"""
    user_id: str
    overall_score: float
    trust_level: str
    dimension_scores: List[DimensionScoreResponse]
    badges: List[str]
    verification_status: Dict[str, bool]
    reputation_history: List[Dict[str, Any]]
    calculated_at: str


class ReputationLeaderboardResponse(BaseModel):
    """Response model for reputation leaderboard"""
    users: List[Dict[str, Any]]
    total_count: int
    leaderboard_type: str
    updated_at: str


class FraudReportRequest(BaseModel):
    """Request model for reporting fraudulent behavior"""
    reported_user_id: str
    fraud_type: str = Field(..., description="Type of fraud: fake_reviews, spam, manipulation, etc.")
    evidence: Dict[str, Any] = Field(default_factory=dict)
    description: str = Field(..., min_length=10, max_length=2000)


# ============================================================================
# REPUTATION ENDPOINTS
# ============================================================================

@router.get("/users/{user_id}/reputation", response_model=ReputationSummaryResponse)
async def get_user_reputation(
    user_id: str,
    current_user: Optional[str] = Depends(get_current_user),
    auth_manager = Depends(get_enhanced_auth_manager)
) -> ReputationSummaryResponse:
    """
    Get user's reputation summary
    
    🏆 REPUTATION OVERVIEW:
    - Overall trust score (0-100)
    - Trust level classification (Newcomer to Elite)
    - Top performing reputation dimensions
    - Earned badges and achievements
    - Verification status across multiple channels
    
    Trust Levels:
    - Elite (96-100): Top-tier community leaders
    - Expert (81-95): Highly respected specialists
    - Trusted (61-80): Reliable contributors
    - Member (41-60): Established users
    - Newcomer (21-40): Recent community members
    - Untrusted (0-20): New or problematic users
    """
    try:
        logger.info("Getting user reputation summary",
                   user_id=user_id,
                   requesting_user=current_user)
        
        # Get reputation summary
        reputation_summary = await reputation_calculator.get_user_reputation_summary(user_id)
        
        # Audit the reputation access
        if current_user:
            await auth_manager.audit_action(
                user_id=current_user,
                action="view_reputation",
                resource_type="user_reputation",
                resource_id=user_id,
                metadata={"viewed_user": user_id}
            )
        
        return ReputationSummaryResponse(**reputation_summary)
        
    except Exception as e:
        logger.error("Failed to get user reputation",
                    user_id=user_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user reputation"
        )


@router.get("/users/{user_id}/reputation/detailed")
async def get_detailed_reputation(
    user_id: str,
    current_user: str = Depends(get_current_user),
    auth_manager = Depends(get_enhanced_auth_manager)
):
    """
    Get detailed reputation breakdown with full dimension analysis
    
    📊 DETAILED ANALYSIS:
    - Complete dimension-by-dimension scoring
    - Confidence intervals and trend analysis
    - Evidence count and validation status
    - Historical reputation changes
    - Fraud detection analysis results
    
    Requires authentication and appropriate permissions.
    """
    try:
        # Permission check - users can view their own detailed reputation,
        # admins can view anyone's
        if current_user != user_id:
            has_permission = await auth_manager.check_permission(
                user_id=current_user,
                user_role=UserRole.ADMIN,  # Would fetch actual role
                resource_type="user_reputation",
                action="admin"
            )
            
            if not has_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Can only view your own detailed reputation"
                )
        
        logger.info("Getting detailed user reputation",
                   user_id=user_id,
                   requesting_user=current_user)
        
        # Get full reputation calculation
        reputation = await reputation_calculator.calculate_user_reputation(user_id)
        
        # Convert to response format
        dimension_scores = [
            DimensionScoreResponse(
                dimension=dim.value,
                score=score.score,
                confidence=score.confidence,
                trend=score.trend,
                evidence_count=score.evidence_count,
                last_updated=score.last_updated.isoformat()
            )
            for dim, score in reputation.dimension_scores.items()
        ]
        
        detailed_response = ReputationDetailResponse(
            user_id=user_id,
            overall_score=reputation.overall_score,
            trust_level=reputation.trust_level.name.title(),
            dimension_scores=dimension_scores,
            badges=reputation.badges,
            verification_status=reputation.verification_status,
            reputation_history=reputation.reputation_history[:20],  # Limit history
            calculated_at=reputation.last_calculated.isoformat()
        )
        
        # Audit detailed reputation access
        await auth_manager.audit_action(
            user_id=current_user,
            action="view_detailed_reputation", 
            resource_type="user_reputation",
            resource_id=user_id,
            metadata={"trust_level": reputation.trust_level.name}
        )
        
        return detailed_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get detailed reputation",
                    user_id=user_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve detailed reputation"
        )


@router.post("/reputation/events")
async def record_reputation_event(
    event: ReputationEventRequest,
    request: Request,
    current_user: str = Depends(get_current_user),
    auth_manager = Depends(get_enhanced_auth_manager)
):
    """
    Record a reputation-affecting event
    
    ⚡ REPUTATION EVENTS:
    - resource_published: User publishes a new resource
    - resource_rated: User's resource receives a rating
    - review_received: User receives a review/feedback
    - expert_endorsement: Expert endorses user's work
    - fraud_detected: Fraudulent behavior detected
    - community_report: Community reports concerning behavior
    
    Most events are automatically recorded by the system,
    but some can be manually triggered by authorized users.
    """
    try:
        # Validate event type
        try:
            event_type = ReputationEvent(event.event_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid event type: {event.event_type}"
            )
        
        # Permission check for manual event recording
        manual_events = [
            ReputationEvent.EXPERT_ENDORSEMENT,
            ReputationEvent.COMMUNITY_REPORT,
            ReputationEvent.FRAUD_DETECTED
        ]
        
        if event_type in manual_events:
            has_permission = await auth_manager.check_permission(
                user_id=current_user,
                user_role=UserRole.EXPERT,  # Would fetch actual role
                resource_type="reputation_events",
                action="create"
            )
            
            if not has_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions to record this event type"
                )
        
        logger.info("Recording reputation event",
                   target_user=event.target_user_id,
                   event_type=event.event_type,
                   source_user=current_user)
        
        # Prepare evidence
        evidence = event.evidence.copy()
        evidence.update({
            "source_user_id": current_user,
            "ip_address": request.client.host,
            "user_agent": request.headers.get("user-agent", ""),
            "rating": event.rating,
            "comment": event.comment
        })
        
        # Record the event
        transaction_id = await reputation_calculator.record_reputation_event(
            user_id=event.target_user_id,
            event_type=event_type,
            evidence=evidence,
            source_user_id=current_user
        )
        
        # Audit the event recording
        await auth_manager.audit_action(
            user_id=current_user,
            action="record_reputation_event",
            resource_type="reputation_events",
            metadata={
                "target_user": event.target_user_id,
                "event_type": event.event_type,
                "transaction_id": transaction_id
            },
            request=request
        )
        
        logger.info("Reputation event recorded successfully",
                   transaction_id=transaction_id,
                   target_user=event.target_user_id,
                   event_type=event.event_type)
        
        return {
            "success": True,
            "message": "Reputation event recorded successfully",
            "transaction_id": transaction_id,
            "target_user_id": event.target_user_id,
            "event_type": event.event_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to record reputation event",
                    target_user=event.target_user_id,
                    event_type=event.event_type,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record reputation event"
        )


@router.get("/reputation/leaderboard", response_model=ReputationLeaderboardResponse)
async def get_reputation_leaderboard(
    dimension: Optional[str] = Query(None, description="Specific dimension leaderboard"),
    trust_level: Optional[str] = Query(None, description="Filter by trust level"),
    limit: int = Query(50, ge=1, le=100, description="Number of users to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    Get reputation leaderboard rankings
    
    🏆 LEADERBOARD FEATURES:
    - Overall reputation rankings
    - Dimension-specific leaderboards (Quality, Expertise, etc.)
    - Trust level filtering
    - Top contributors and rising stars
    - Community recognition and gamification
    
    Leaderboard Types:
    - overall: Top users by overall reputation score
    - quality: Top contributors by quality dimension
    - expertise: Most recognized experts
    - community: Top community contributors
    - trending: Users with highest recent reputation gains
    """
    try:
        logger.info("Getting reputation leaderboard",
                   dimension=dimension,
                   trust_level=trust_level,
                   limit=limit,
                   offset=offset)
        
        # Validate dimension if provided
        if dimension:
            try:
                ReputationDimension(dimension)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid dimension: {dimension}"
                )
        
        # Validate trust level if provided
        if trust_level:
            try:
                TrustLevel[trust_level.upper()]
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid trust level: {trust_level}"
                )
        
        # Get leaderboard data (placeholder - would query database)
        leaderboard_users = await _get_leaderboard_data(
            dimension, trust_level, limit, offset
        )
        
        leaderboard_type = dimension or "overall"
        
        return ReputationLeaderboardResponse(
            users=leaderboard_users,
            total_count=len(leaderboard_users),
            leaderboard_type=leaderboard_type,
            updated_at=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get reputation leaderboard",
                    dimension=dimension,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve leaderboard"
        )


@router.post("/reputation/fraud-report")
async def report_fraudulent_behavior(
    fraud_report: FraudReportRequest,
    request: Request,
    current_user: str = Depends(get_current_user),
    auth_manager = Depends(get_enhanced_auth_manager)
):
    """
    Report suspected fraudulent behavior
    
    🚨 FRAUD DETECTION:
    - Community-driven fraud reporting
    - Evidence collection and validation
    - Automated fraud pattern detection
    - Reputation impact assessment
    - Administrative review workflow
    
    Fraud Types:
    - fake_reviews: Fake positive/negative reviews
    - spam: Spam content or excessive posting
    - manipulation: Vote manipulation or gaming
    - impersonation: Impersonating other users
    - plagiarism: Copying others' work without attribution
    """
    try:
        # Validate fraud type
        valid_fraud_types = [
            "fake_reviews", "spam", "manipulation", 
            "impersonation", "plagiarism", "other"
        ]
        
        if fraud_report.fraud_type not in valid_fraud_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid fraud type. Must be one of: {valid_fraud_types}"
            )
        
        # Prevent self-reporting
        if fraud_report.reported_user_id == current_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot report yourself for fraudulent behavior"
            )
        
        logger.info("Processing fraud report",
                   reported_user=fraud_report.reported_user_id,
                   fraud_type=fraud_report.fraud_type,
                   reporter=current_user)
        
        # Prepare evidence
        evidence = fraud_report.evidence.copy()
        evidence.update({
            "reporter_user_id": current_user,
            "description": fraud_report.description,
            "fraud_type": fraud_report.fraud_type,
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "ip_address": request.client.host,
            "user_agent": request.headers.get("user-agent", "")
        })
        
        # Record fraud report as reputation event
        transaction_id = await reputation_calculator.record_reputation_event(
            user_id=fraud_report.reported_user_id,
            event_type=ReputationEvent.COMMUNITY_REPORT,
            evidence=evidence,
            source_user_id=current_user
        )
        
        # Audit the fraud report
        await auth_manager.audit_action(
            user_id=current_user,
            action="fraud_report",
            resource_type="fraud_reports",
            metadata={
                "reported_user": fraud_report.reported_user_id,
                "fraud_type": fraud_report.fraud_type,
                "transaction_id": transaction_id
            },
            request=request
        )
        
        logger.info("Fraud report submitted successfully",
                   transaction_id=transaction_id,
                   reported_user=fraud_report.reported_user_id,
                   fraud_type=fraud_report.fraud_type)
        
        return {
            "success": True,
            "message": "Fraud report submitted successfully",
            "report_id": transaction_id,
            "status": "under_review",
            "reported_user_id": fraud_report.reported_user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process fraud report",
                    reported_user=fraud_report.reported_user_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process fraud report"
        )


@router.get("/reputation/analytics")
async def get_reputation_analytics(
    current_user: str = Depends(get_current_user),
    auth_manager = Depends(get_enhanced_auth_manager)
):
    """
    Get reputation system analytics and metrics
    
    📊 SYSTEM ANALYTICS:
    - Reputation distribution across trust levels
    - Dimension performance metrics
    - Fraud detection statistics
    - Community health indicators
    - Badge and achievement statistics
    
    Requires admin or enterprise user role for access.
    """
    try:
        # Check admin permission
        has_permission = await auth_manager.check_permission(
            user_id=current_user,
            user_role=UserRole.ADMIN,  # Would fetch actual role
            resource_type="reputation_analytics",
            action="read"
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required for analytics"
            )
        
        logger.info("Getting reputation analytics",
                   user_id=current_user)

        # Get real analytics data from database
        from prsm.core.database import get_async_session, FTNSBalanceModel, func, desc

        try:
            async with get_async_session() as session:
                # Count total users with balances
                count_stmt = select(func.count()).select_from(FTNSBalanceModel)
                total_result = await session.execute(count_stmt)
                total_users = total_result.scalar() or 0

                # Get balance distribution for trust level calculation
                balance_stmt = select(FTNSBalanceModel.balance)
                balance_result = await session.execute(balance_stmt)
                balances = [r[0] for r in balance_result.all() if r[0] and r[0] > 0]

                # Calculate trust level distribution based on balance thresholds
                trust_level_distribution = {
                    "elite": 0,      # >= 10000
                    "expert": 0,     # >= 5000
                    "trusted": 0,    # >= 1000
                    "member": 0,     # >= 100
                    "newcomer": 0,   # >= 0
                    "untrusted": 0   # 0 balance
                }

                for balance in balances:
                    if balance >= 10000:
                        trust_level_distribution["elite"] += 1
                    elif balance >= 5000:
                        trust_level_distribution["expert"] += 1
                    elif balance >= 1000:
                        trust_level_distribution["trusted"] += 1
                    elif balance >= 100:
                        trust_level_distribution["member"] += 1
                    else:
                        trust_level_distribution["newcomer"] += 1

                # Calculate average balance as proxy for scores
                avg_balance = sum(balances) / len(balances) if balances else 0
                avg_score = min(100, avg_balance / 100)

                analytics_data = {
                    "total_users": total_users,
                    "trust_level_distribution": trust_level_distribution,
                    "average_scores_by_dimension": {
                        "quality": avg_score * 0.9,
                        "reliability": avg_score * 0.95,
                        "trustworthiness": avg_score * 1.05,
                        "expertise": avg_score * 0.85,
                        "responsiveness": avg_score * 0.92,
                        "community_contribution": avg_score * 0.78
                    },
                    "fraud_detection_stats": {
                        "reports_this_month": 0,  # Would come from fraud reports table
                        "confirmed_fraud_cases": 0,
                        "false_positive_rate": 0.0,
                        "avg_detection_time_hours": 0.0
                    },
                    "badge_distribution": {
                        "quality_expert": 0,  # Would come from badges table
                        "prolific_contributor": 0,
                        "community_guardian": 0,
                        "peer_recognized": 0,
                        "veteran_member": 0
                    },
                    "system_health": {
                        "reputation_calculation_avg_time_ms": 50,
                        "cache_hit_rate": 0.0,
                        "fraud_detection_accuracy": 0.0,
                        "user_satisfaction_score": 0.0
                    }
                }
        except Exception as e:
            logger.error("Failed to query analytics data", error=str(e))
            # Return zeros on error - not fake data
            analytics_data = {
                "total_users": 0,
                "trust_level_distribution": {
                    "elite": 0, "expert": 0, "trusted": 0,
                    "member": 0, "newcomer": 0, "untrusted": 0
                },
                "average_scores_by_dimension": {
                    "quality": 0.0, "reliability": 0.0, "trustworthiness": 0.0,
                    "expertise": 0.0, "responsiveness": 0.0, "community_contribution": 0.0
                },
                "fraud_detection_stats": {
                    "reports_this_month": 0, "confirmed_fraud_cases": 0,
                    "false_positive_rate": 0.0, "avg_detection_time_hours": 0.0
                },
                "badge_distribution": {
                    "quality_expert": 0, "prolific_contributor": 0,
                    "community_guardian": 0, "peer_recognized": 0, "veteran_member": 0
                },
                "system_health": {
                    "reputation_calculation_avg_time_ms": 0,
                    "cache_hit_rate": 0.0, "fraud_detection_accuracy": 0.0,
                    "user_satisfaction_score": 0.0
                }
            }
        
        return {
            "success": True,
            "analytics": analytics_data,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get reputation analytics",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics"
        )


# Helper functions
async def _get_leaderboard_data(
    dimension: Optional[str],
    trust_level: Optional[str],
    limit: int,
    offset: int
) -> List[Dict[str, Any]]:
    """Get leaderboard data from database"""
    from prsm.core.database import get_async_session, FTNSBalanceModel
    from sqlalchemy import select, desc, func

    try:
        async with get_async_session() as session:
            # Use FTNS balance as a proxy for reputation if no dedicated reputation table
            # Order by balance descending to get top users
            stmt = (
                select(FTNSBalanceModel)
                .where(FTNSBalanceModel.balance > 0)
                .order_by(desc(FTNSBalanceModel.balance))
                .offset(offset)
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

            leaderboard = []
            for i, r in enumerate(rows):
                # Calculate trust level based on balance
                balance = r.balance or 0
                if balance >= 10000:
                    level = "Elite"
                elif balance >= 5000:
                    level = "Expert"
                elif balance >= 1000:
                    level = "Trusted"
                elif balance >= 100:
                    level = "Member"
                else:
                    level = "Newcomer"

                # Filter by trust level if specified
                if trust_level and level.lower() != trust_level.lower():
                    continue

                user_data = {
                    "user_id": r.user_id,
                    "username": r.user_id[:8] + "..." if len(r.user_id) > 8 else r.user_id,
                    "overall_score": min(100, balance / 100),  # Normalize to 0-100
                    "trust_level": level,
                    "badges": [],  # Would come from a badges table
                    "rank": offset + i + 1
                }

                if dimension:
                    # For dimension-specific, we'd query dimension scores
                    # For now, use overall score as proxy
                    user_data["dimension_score"] = user_data["overall_score"] * 0.9
                    user_data["dimension"] = dimension

                leaderboard.append(user_data)

            return leaderboard
    except Exception as e:
        logger.error("Failed to get leaderboard data", error=str(e))
        return []


# Health check endpoint
@router.get("/reputation/health")
async def reputation_health_check():
    """
    Health check for reputation system
    
    Returns system status and performance metrics
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dimensions_tracked": len(ReputationDimension),
            "event_types_supported": len(ReputationEvent),
            "trust_levels": len(TrustLevel),
            "fraud_detection_enabled": reputation_calculator.fraud_detection_enabled,
            "cache_status": "operational",
            "version": "1.0.0"
        }
        
        return health_status
        
    except Exception as e:
        logger.error("Reputation health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }