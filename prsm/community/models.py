"""
Community Management Models
===========================

Data models for community onboarding, early adopter programs, and user engagement.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import Column, String, DateTime, Boolean, Enum as SQLEnum, Text, Integer, JSON, Numeric
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from prsm.core.models import TimestampMixin, Base


class OnboardingStage(str, Enum):
    """Stages of the community onboarding process"""
    REGISTRATION = "registration"
    EMAIL_VERIFICATION = "email_verification"
    PROFILE_SETUP = "profile_setup"
    INTERESTS_SELECTION = "interests_selection"
    INITIAL_ALLOCATION = "initial_allocation"
    FIRST_INTERACTION = "first_interaction"
    TUTORIAL_COMPLETION = "tutorial_completion"
    COMMUNITY_INTRODUCTION = "community_introduction"
    COMPLETED = "completed"


class EarlyAdopterTier(str, Enum):
    """Early adopter program tiers"""
    PIONEER = "pioneer"          # First 100 users - 5x bonuses
    EXPLORER = "explorer"        # Users 101-500 - 3x bonuses
    BUILDER = "builder"          # Users 501-2000 - 2x bonuses
    MEMBER = "member"            # Users 2001-10000 - 1.5x bonuses
    COMMUNITY = "community"      # Users 10000+ - 1x bonuses


class UserInterest(str, Enum):
    """User research and contribution interests"""
    MACHINE_LEARNING = "machine_learning"
    NATURAL_LANGUAGE_PROCESSING = "natural_language_processing"
    COMPUTER_VISION = "computer_vision"
    ROBOTICS = "robotics"
    DRUG_DISCOVERY = "drug_discovery"
    CLIMATE_SCIENCE = "climate_science"
    MATERIALS_SCIENCE = "materials_science"
    BIOINFORMATICS = "bioinformatics"
    ASTROPHYSICS = "astrophysics"
    ECONOMICS = "economics"
    EDUCATION = "education"
    GOVERNANCE = "governance"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    COMMUNITY_BUILDING = "community_building"


class EngagementType(str, Enum):
    """Types of community engagement activities"""
    MODEL_UPLOAD = "model_upload"
    DATA_CONTRIBUTION = "data_contribution"
    RESEARCH_SHARING = "research_sharing"
    PEER_REVIEW = "peer_review"
    TUTORIAL_CREATION = "tutorial_creation"
    COMMUNITY_SUPPORT = "community_support"
    GOVERNANCE_PARTICIPATION = "governance_participation"
    MARKETPLACE_ACTIVITY = "marketplace_activity"
    COLLABORATION = "collaboration"
    FEEDBACK_PROVISION = "feedback_provision"


class OnboardingProgress(Base):
    """User onboarding progress tracking"""
    __tablename__ = "onboarding_progress"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    current_stage = Column(SQLEnum(OnboardingStage), nullable=False, default=OnboardingStage.REGISTRATION)
    completed_stages = Column(JSON, nullable=False, default=list)
    stage_completion_times = Column(JSON, nullable=False, default=dict)
    
    # User profile data
    research_interests = Column(JSON, nullable=True)  # List of UserInterest values
    experience_level = Column(String(50), nullable=True)  # beginner, intermediate, advanced, expert
    institution_affiliation = Column(String(500), nullable=True)
    goals = Column(Text, nullable=True)
    
    # Onboarding metadata
    invitation_code = Column(String(100), nullable=True, index=True)
    referral_user_id = Column(String(255), nullable=True, index=True)
    onboarding_version = Column(String(20), nullable=False, default="v1.0")
    total_completion_time_hours = Column(Numeric(10, 2), nullable=True)
    
    # Status tracking
    is_completed = Column(Boolean, default=False, nullable=False)
    completion_date = Column(DateTime(timezone=True), nullable=True)
    last_activity = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)


class EarlyAdopterProfile(Base):
    """Early adopter program participant profile"""
    __tablename__ = "early_adopter_profiles"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    adopter_tier = Column(SQLEnum(EarlyAdopterTier), nullable=False)
    join_number = Column(Integer, nullable=False)  # 1st, 2nd, 3rd user, etc.
    
    # Program benefits
    ftns_bonus_multiplier = Column(Numeric(4, 2), nullable=False, default=Decimal('1.0'))
    governance_weight_bonus = Column(Numeric(4, 2), nullable=False, default=Decimal('0.0'))
    priority_access_level = Column(Integer, nullable=False, default=1)
    exclusive_features_access = Column(JSON, nullable=False, default=list)
    
    # Program participation
    program_activities = Column(JSON, nullable=False, default=list)
    milestone_achievements = Column(JSON, nullable=False, default=list)
    contribution_scores = Column(JSON, nullable=False, default=dict)
    referral_count = Column(Integer, nullable=False, default=0)
    
    # Recognition and rewards
    recognition_badges = Column(JSON, nullable=False, default=list)
    lifetime_bonus_earned = Column(Numeric(18, 8), nullable=False, default=Decimal('0'))
    special_privileges = Column(JSON, nullable=False, default=list)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    graduated_to_tier = Column(SQLEnum(EarlyAdopterTier), nullable=True)
    graduation_date = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)


class CommunityEngagementRecord(Base):
    """Community engagement activity tracking"""
    __tablename__ = "community_engagement_records"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    engagement_type = Column(SQLEnum(EngagementType), nullable=False)
    
    # Activity details
    activity_description = Column(Text, nullable=False)
    activity_metadata = Column(JSON, nullable=False, default=dict)
    quality_score = Column(Numeric(3, 2), nullable=True)  # 0.0 to 1.0
    impact_score = Column(Numeric(3, 2), nullable=True)   # 0.0 to 1.0
    
    # Rewards and recognition
    ftns_reward = Column(Numeric(18, 8), nullable=False, default=Decimal('0'))
    reputation_points = Column(Integer, nullable=False, default=0)
    badges_earned = Column(JSON, nullable=False, default=list)
    
    # Peer interaction
    collaborators = Column(JSON, nullable=False, default=list)  # List of user_ids
    peer_reviews = Column(JSON, nullable=False, default=list)
    community_feedback = Column(Text, nullable=True)
    
    # Context
    related_resources = Column(JSON, nullable=False, default=list)
    platform_context = Column(String(100), nullable=True)  # web, api, cli, etc.
    
    # Timestamps
    activity_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


class CommunityMilestone(Base):
    """Community achievement milestones"""
    __tablename__ = "community_milestones"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    milestone_name = Column(String(255), nullable=False, unique=True)
    milestone_description = Column(Text, nullable=False)
    milestone_type = Column(String(100), nullable=False)  # onboarding, contribution, collaboration, governance
    
    # Achievement criteria
    criteria = Column(JSON, nullable=False)  # Flexible criteria definition
    reward_ftns = Column(Numeric(18, 8), nullable=False, default=Decimal('0'))
    reward_badges = Column(JSON, nullable=False, default=list)
    reward_privileges = Column(JSON, nullable=False, default=list)
    
    # Milestone metadata
    difficulty_level = Column(String(20), nullable=False, default="beginner")  # beginner, intermediate, advanced, expert
    estimated_time_hours = Column(Numeric(6, 2), nullable=True)
    prerequisite_milestones = Column(JSON, nullable=False, default=list)
    category = Column(String(100), nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_repeatable = Column(Boolean, default=False, nullable=False)
    max_achievements = Column(Integer, nullable=True)  # null = unlimited
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)


class UserMilestoneAchievement(Base):
    """User milestone achievement records"""
    __tablename__ = "user_milestone_achievements"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    milestone_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    
    # Achievement details
    achievement_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    achievement_context = Column(JSON, nullable=False, default=dict)
    completion_time_hours = Column(Numeric(6, 2), nullable=True)
    
    # Rewards received
    ftns_rewarded = Column(Numeric(18, 8), nullable=False, default=Decimal('0'))
    badges_received = Column(JSON, nullable=False, default=list)
    privileges_granted = Column(JSON, nullable=False, default=list)
    
    # Achievement metadata
    achievement_rank = Column(Integer, nullable=True)  # Position among all achievers
    quality_score = Column(Numeric(3, 2), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


# Pydantic models for API

class OnboardingProgressResponse(BaseModel):
    """Onboarding progress response model"""
    user_id: str
    current_stage: OnboardingStage
    completed_stages: List[OnboardingStage]
    completion_percentage: float
    estimated_time_remaining_hours: Optional[float] = None
    next_steps: List[str]
    
    class Config:
        from_attributes = True


class OnboardingStageUpdate(BaseModel):
    """Onboarding stage completion update"""
    stage: OnboardingStage
    completion_data: Dict[str, Any] = {}
    user_inputs: Dict[str, Any] = {}


class EarlyAdopterRegistration(BaseModel):
    """Early adopter program registration"""
    invitation_code: Optional[str] = None
    referral_code: Optional[str] = None
    research_interests: List[UserInterest]
    experience_level: str = Field(..., pattern="^(beginner|intermediate|advanced|expert)$")
    institution_affiliation: Optional[str] = None
    goals: Optional[str] = None


class EarlyAdopterStatusResponse(BaseModel):
    """Early adopter status response"""
    user_id: str
    adopter_tier: EarlyAdopterTier
    join_number: int
    ftns_bonus_multiplier: Decimal
    governance_weight_bonus: Decimal
    priority_access_level: int
    lifetime_bonus_earned: Decimal
    milestone_achievements: List[str]
    recognition_badges: List[str]
    
    class Config:
        from_attributes = True


class EngagementActivity(BaseModel):
    """Community engagement activity submission"""
    engagement_type: EngagementType
    activity_description: str
    activity_metadata: Dict[str, Any] = {}
    collaborators: List[str] = []
    related_resources: List[str] = []


class CommunityStats(BaseModel):
    """Community statistics response"""
    total_users: int
    active_users_30d: int
    early_adopters_by_tier: Dict[str, int]
    onboarding_completion_rate: float
    average_onboarding_time_hours: float
    top_engagement_activities: List[Dict[str, Any]]
    milestone_achievement_stats: Dict[str, int]
    community_growth_rate: float


class WelcomePackage(BaseModel):
    """Welcome package for new users"""
    welcome_message: str
    initial_ftns_grant: Decimal
    recommended_tutorials: List[Dict[str, str]]
    community_links: Dict[str, str]
    getting_started_guide: str
    early_adopter_benefits: List[str]
    milestone_roadmap: List[Dict[str, Any]]