#!/usr/bin/env python3
"""
PRSM Alpha Program API
=====================

RESTful API endpoints for the alpha testing program, including:
- User registration and authentication
- Usage tracking and analytics
- Feedback collection and management
- Community features and collaboration

🎯 ALPHA API FEATURES:
✅ User registration with validation
✅ API key authentication
✅ Real-time usage tracking
✅ Comprehensive feedback system
✅ Analytics and reporting
✅ Community collaboration tools
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..alpha.user_management import (
    AlphaUserManager, AlphaUser, UserType, 
    get_alpha_manager, FeedbackEntry
)
from prsm.core.models import UserInput, PRSMResponse

router = APIRouter(prefix="/alpha", tags=["Alpha Program"])
security = HTTPBearer()


# Request/Response Models
class AlphaRegistrationRequest(BaseModel):
    """Alpha user registration request"""
    email: EmailStr
    name: str
    organization: str
    user_type: str
    use_case: str
    technical_background: str
    experience_level: str = "intermediate"
    collaboration_consent: bool = False
    marketing_consent: bool = False
    
    @validator('user_type')
    def validate_user_type(cls, v):
        valid_types = [ut.value for ut in UserType]
        if v.lower() not in valid_types:
            raise ValueError(f"User type must be one of: {valid_types}")
        return v.lower()
    
    @validator('use_case')
    def validate_use_case(cls, v):
        if len(v) < 10:
            raise ValueError("Use case description must be at least 10 characters")
        return v


class AlphaRegistrationResponse(BaseModel):
    """Alpha user registration response"""
    success: bool
    message: str
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    ftns_balance: Optional[float] = None
    onboarding_url: Optional[str] = None
    documentation_url: Optional[str] = None
    community_discord: Optional[str] = None
    waitlist_position: Optional[int] = None


class FeedbackRequest(BaseModel):
    """Feedback submission request"""
    category: str  # bug_report|feature_request|general_feedback|quality_rating
    title: str
    description: str
    priority: str = "medium"  # low|medium|high|critical
    query_id: Optional[str] = None
    rating: Optional[int] = None
    tags: List[str] = []
    
    @validator('category')
    def validate_category(cls, v):
        valid_categories = ["bug_report", "feature_request", "general_feedback", "quality_rating"]
        if v not in valid_categories:
            raise ValueError(f"Category must be one of: {valid_categories}")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ["low", "medium", "high", "critical"]
        if v not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")
        return v
    
    @validator('rating')
    def validate_rating(cls, v):
        if v is not None and (v < 1 or v > 5):
            raise ValueError("Rating must be between 1 and 5")
        return v


class UsageTrackingRequest(BaseModel):
    """Usage tracking request"""
    query: str
    response_success: bool
    latency: float
    cost: float
    routing_strategy: str
    region: str
    quality_score: Optional[float] = None
    error_message: Optional[str] = None


class UserStatusResponse(BaseModel):
    """User status response"""
    user_id: str
    email: str
    registration_status: str
    ftns_balance: float
    total_queries: int
    feedback_count: int
    onboarding_completed: bool
    last_active: Optional[datetime]
    engagement_score: float


# Authentication Dependency
async def get_current_alpha_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
) -> AlphaUser:
    """Get current authenticated alpha user"""
    
    api_key = credentials.credentials
    user = await alpha_manager.authenticate_user(api_key)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or user not found"
        )
    
    return user


# Registration Endpoints
@router.post("/register", response_model=AlphaRegistrationResponse)
async def register_alpha_user(
    request: AlphaRegistrationRequest,
    background_tasks: BackgroundTasks,
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
):
    """Register a new alpha user"""
    
    try:
        result = await alpha_manager.register_alpha_user(
            email=request.email,
            name=request.name,
            organization=request.organization,
            user_type_str=request.user_type,
            use_case=request.use_case,
            technical_background=request.technical_background,
            experience_level=request.experience_level,
            collaboration_consent=request.collaboration_consent,
            marketing_consent=request.marketing_consent
        )
        
        if result["success"]:
            return AlphaRegistrationResponse(
                success=True,
                message="Alpha registration successful! Check your email for onboarding instructions.",
                user_id=result["user_id"],
                api_key=result["api_key"],
                ftns_balance=result["ftns_balance"],
                onboarding_url=result["onboarding_url"],
                documentation_url=result["documentation_url"],
                community_discord=result["community_discord"]
            )
        else:
            return AlphaRegistrationResponse(
                success=False,
                message=result["error"],
                waitlist_position=result.get("waitlist_position")
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )


@router.get("/status", response_model=UserStatusResponse)
async def get_user_status(
    current_user: AlphaUser = Depends(get_current_alpha_user),
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
):
    """Get current user status and statistics"""
    
    analytics = await alpha_manager.get_user_analytics(current_user.user_id)
    
    if "error" in analytics:
        raise HTTPException(status_code=404, detail=analytics["error"])
    
    return UserStatusResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        registration_status=current_user.registration_status.value,
        ftns_balance=current_user.ftns_balance,
        total_queries=current_user.total_queries,
        feedback_count=current_user.feedback_count,
        onboarding_completed=current_user.onboarding_completed,
        last_active=current_user.last_active,
        engagement_score=analytics["engagement"]["engagement_score"]
    )


# Usage Tracking Endpoints
@router.post("/track-usage")
async def track_query_usage(
    request: UsageTrackingRequest,
    current_user: AlphaUser = Depends(get_current_alpha_user),
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
):
    """Track user query usage for analytics"""
    
    # Create mock response object for tracking
    mock_response = PRSMResponse(
        session_id="",
        user_id=current_user.user_id,
        final_answer="",
        reasoning_trace=[],
        confidence_score=request.quality_score or 0.0,
        context_used=0,
        ftns_charged=request.cost,
        sources=[],
        safety_validated=True,
        success=request.response_success,
        error=request.error_message
    )
    
    await alpha_manager.track_query_usage(
        user_id=current_user.user_id,
        query=request.query,
        response=mock_response,
        latency=request.latency,
        cost=request.cost,
        routing_strategy=request.routing_strategy,
        region=request.region
    )
    
    return {"status": "success", "message": "Usage tracked successfully"}


@router.get("/analytics")
async def get_user_analytics(
    current_user: AlphaUser = Depends(get_current_alpha_user),
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
):
    """Get comprehensive user analytics"""
    
    analytics = await alpha_manager.get_user_analytics(current_user.user_id)
    
    if "error" in analytics:
        raise HTTPException(status_code=404, detail=analytics["error"])
    
    return analytics


# Feedback Endpoints
@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    current_user: AlphaUser = Depends(get_current_alpha_user),
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
):
    """Submit feedback about PRSM"""
    
    feedback_id = await alpha_manager.submit_feedback(
        user_id=current_user.user_id,
        category=request.category,
        title=request.title,
        description=request.description,
        priority=request.priority,
        query_id=request.query_id,
        rating=request.rating,
        tags=request.tags
    )
    
    return {
        "status": "success",
        "message": "Feedback submitted successfully",
        "feedback_id": feedback_id
    }


@router.get("/feedback")
async def get_user_feedback(
    current_user: AlphaUser = Depends(get_current_alpha_user),
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
):
    """Get user's feedback history"""
    
    user_feedback = [
        {
            "feedback_id": f.feedback_id,
            "category": f.category,
            "title": f.title,
            "description": f.description,
            "priority": f.priority,
            "status": f.status,
            "submitted_at": f.submitted_at.isoformat(),
            "rating": f.rating,
            "tags": f.tags
        }
        for f in alpha_manager.feedback_entries.values()
        if f.user_id == current_user.user_id
    ]
    
    return {
        "feedback_entries": user_feedback,
        "total_count": len(user_feedback)
    }


# Program Analytics (Admin)
@router.get("/program-analytics")
async def get_program_analytics(
    current_user: AlphaUser = Depends(get_current_alpha_user),
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
):
    """Get alpha program analytics (admin only)"""
    
    # In production, this would check for admin privileges
    # For alpha, we'll allow all users to see aggregated stats
    
    analytics = await alpha_manager.get_program_analytics()
    
    # Remove sensitive information for non-admin users
    public_analytics = {
        "program_overview": analytics["program_overview"],
        "usage_metrics": analytics["usage_metrics"],
        "user_distribution": analytics["user_distribution"],
        "engagement_metrics": {
            "avg_engagement_score": analytics["engagement_metrics"]["avg_engagement_score"],
            "avg_quality_rating": analytics["engagement_metrics"]["avg_quality_rating"]
        }
    }
    
    return public_analytics


# Completion and Certification
@router.get("/completion-report")
async def get_completion_report(
    current_user: AlphaUser = Depends(get_current_alpha_user),
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
):
    """Get user's alpha program completion report"""
    
    report = await alpha_manager.generate_user_completion_report(current_user.user_id)
    
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])
    
    return report


@router.post("/complete-onboarding")
async def complete_onboarding(
    current_user: AlphaUser = Depends(get_current_alpha_user),
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
):
    """Mark user onboarding as completed"""
    
    current_user.onboarding_completed = True
    current_user.registration_status = current_user.registration_status.ACTIVE
    
    return {
        "status": "success",
        "message": "Onboarding completed successfully",
        "next_steps": [
            "Start making queries to test PRSM capabilities",
            "Join the Discord community",
            "Provide feedback on your experience",
            "Try different routing strategies",
            "Explore multi-modal features"
        ]
    }


# Community Features
@router.get("/community-stats")
async def get_community_stats(
    current_user: AlphaUser = Depends(get_current_alpha_user),
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
):
    """Get community engagement statistics"""
    
    analytics = await alpha_manager.get_program_analytics()
    
    # Calculate community insights
    total_users = analytics["program_overview"]["total_registered_users"]
    active_users = analytics["program_overview"]["active_users_7d"]
    
    # User ranking (based on engagement)
    user_analytics = await alpha_manager.get_user_analytics(current_user.user_id)
    user_engagement = user_analytics["engagement"]["engagement_score"]
    
    # Mock ranking calculation (in production, this would be more sophisticated)
    ranking_percentile = min(int(user_engagement * 100), 95)
    
    return {
        "community_size": total_users,
        "active_this_week": active_users,
        "total_queries_community": analytics["usage_metrics"]["total_queries"],
        "avg_quality_rating": analytics["engagement_metrics"]["avg_quality_rating"],
        "user_ranking": {
            "engagement_percentile": ranking_percentile,
            "queries_vs_average": current_user.total_queries / max(analytics["usage_metrics"]["avg_queries_per_user"], 1),
            "feedback_contributions": current_user.feedback_count
        },
        "community_achievements": {
            "total_feedback": analytics["usage_metrics"]["total_feedback_entries"],
            "bugs_reported": analytics["feedback_analysis"]["bug_reports"],
            "features_suggested": analytics["feedback_analysis"]["feature_requests"]
        }
    }


@router.get("/leaderboard")
async def get_community_leaderboard(
    current_user: AlphaUser = Depends(get_current_alpha_user),
    alpha_manager: AlphaUserManager = Depends(get_alpha_manager)
):
    """Get community leaderboard (anonymized)"""
    
    # Calculate engagement scores for all users
    leaderboard = []
    
    for user in alpha_manager.alpha_users.values():
        if user.user_id == current_user.user_id:
            continue  # Skip current user for anonymity
        
        metrics = alpha_manager.usage_metrics.get(user.user_id)
        if metrics:
            engagement_score = alpha_manager._calculate_engagement_score(user, metrics)
            
            leaderboard.append({
                "anonymous_id": f"User{hash(user.user_id) % 1000:03d}",
                "user_type": user.user_type.value,
                "engagement_score": round(engagement_score, 2),
                "total_queries": user.total_queries,
                "feedback_count": user.feedback_count,
                "days_active": (datetime.now() - user.registration_date).days
            })
    
    # Sort by engagement score
    leaderboard.sort(key=lambda x: x["engagement_score"], reverse=True)
    
    # Add current user's position
    current_metrics = alpha_manager.usage_metrics.get(current_user.user_id)
    current_engagement = alpha_manager._calculate_engagement_score(current_user, current_metrics) if current_metrics else 0
    
    current_position = len([u for u in leaderboard if u["engagement_score"] > current_engagement]) + 1
    
    return {
        "leaderboard": leaderboard[:10],  # Top 10
        "your_position": current_position,
        "total_participants": len(leaderboard) + 1,
        "your_engagement_score": round(current_engagement, 2)
    }


# Health Check
@router.get("/health")
async def alpha_health_check():
    """Alpha program health check"""
    return {
        "status": "healthy",
        "program": "PRSM Alpha Testing",
        "version": "v0.9.0",
        "timestamp": datetime.now().isoformat()
    }