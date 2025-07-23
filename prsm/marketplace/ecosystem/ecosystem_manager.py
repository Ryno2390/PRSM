#!/usr/bin/env python3
"""
Ecosystem Manager
=================

Comprehensive developer ecosystem management with onboarding, verification,
tiers, analytics, and community features for the marketplace platform.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import uuid
from pathlib import Path
import hashlib

from prsm.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class DeveloperTier(Enum):
    """Developer tier levels"""
    COMMUNITY = "community"
    VERIFIED = "verified"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    PARTNER = "partner"


class DeveloperStatus(Enum):
    """Developer status"""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    BANNED = "banned"
    INACTIVE = "inactive"


class ApplicationStatus(Enum):
    """Application status for tier upgrades"""
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_INFO = "requires_info"


@dataclass
class DeveloperMetrics:
    """Developer performance metrics"""
    developer_id: str
    
    # Integration metrics
    total_integrations: int = 0
    published_integrations: int = 0
    featured_integrations: int = 0
    
    # Download and usage metrics
    total_downloads: int = 0
    active_installations: int = 0
    monthly_downloads: int = 0
    
    # Quality metrics
    average_rating: float = 0.0
    total_reviews: int = 0
    support_response_time_hours: float = 24.0
    
    # Revenue metrics
    total_revenue: float = 0.0
    monthly_revenue: float = 0.0
    
    # Community metrics
    forum_posts: int = 0
    forum_reputation: int = 0
    contributions: int = 0
    
    # Compliance metrics
    security_issues: int = 0
    policy_violations: int = 0
    resolved_issues: int = 0
    
    # Time tracking
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score"""
        
        # Base quality components
        rating_score = min(self.average_rating * 20, 100)  # Scale 0-5 to 0-100
        
        # Support responsiveness score
        support_score = max(0, 100 - (self.support_response_time_hours - 24) * 2)
        support_score = min(support_score, 100)
        
        # Compliance score
        if self.published_integrations > 0:
            compliance_rate = max(0, 1 - (self.security_issues + self.policy_violations) / self.published_integrations)
            compliance_score = compliance_rate * 100
        else:
            compliance_score = 100
        
        # Weighted average
        quality_score = (rating_score * 0.4 + support_score * 0.3 + compliance_score * 0.3)
        
        return min(quality_score, 100)
    
    def calculate_popularity_score(self) -> float:
        """Calculate popularity score"""
        
        # Download-based score (logarithmic scaling)
        if self.total_downloads > 0:
            download_score = min(50, 10 * (self.total_downloads ** 0.3))
        else:
            download_score = 0
        
        # Review-based score
        if self.total_reviews > 0:
            review_score = min(30, self.total_reviews * 2)
        else:
            review_score = 0
        
        # Community engagement score
        community_score = min(20, (self.forum_posts * 0.5 + self.contributions * 2))
        
        return download_score + review_score + community_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "developer_id": self.developer_id,
            "total_integrations": self.total_integrations,
            "published_integrations": self.published_integrations,
            "featured_integrations": self.featured_integrations,
            "total_downloads": self.total_downloads,
            "active_installations": self.active_installations,
            "monthly_downloads": self.monthly_downloads,
            "average_rating": self.average_rating,
            "total_reviews": self.total_reviews,
            "support_response_time_hours": self.support_response_time_hours,
            "total_revenue": self.total_revenue,
            "monthly_revenue": self.monthly_revenue,
            "forum_posts": self.forum_posts,
            "forum_reputation": self.forum_reputation,
            "contributions": self.contributions,
            "security_issues": self.security_issues,
            "policy_violations": self.policy_violations,
            "resolved_issues": self.resolved_issues,
            "quality_score": self.calculate_quality_score(),
            "popularity_score": self.calculate_popularity_score(),
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class DeveloperBenefits:
    """Benefits associated with developer tier"""
    tier: DeveloperTier
    
    # Integration limits
    max_integrations: int = 10
    max_versions_per_integration: int = 20
    
    # Revenue sharing
    revenue_share_percent: float = 70.0
    
    # Support and features
    priority_support: bool = False
    beta_features_access: bool = False
    custom_branding: bool = False
    white_label_options: bool = False
    
    # Marketplace features
    featured_listings: int = 0
    promoted_listings: int = 0
    analytics_access: bool = True
    advanced_analytics: bool = False
    
    # Developer tools
    api_rate_limit: int = 1000
    webhook_endpoints: int = 5
    custom_domains: int = 0
    
    # Community features
    verified_badge: bool = False
    forum_moderation: bool = False
    early_access: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tier": self.tier.value,
            "max_integrations": self.max_integrations,
            "max_versions_per_integration": self.max_versions_per_integration,
            "revenue_share_percent": self.revenue_share_percent,
            "priority_support": self.priority_support,
            "beta_features_access": self.beta_features_access,
            "custom_branding": self.custom_branding,
            "white_label_options": self.white_label_options,
            "featured_listings": self.featured_listings,
            "promoted_listings": self.promoted_listings,
            "analytics_access": self.analytics_access,
            "advanced_analytics": self.advanced_analytics,
            "api_rate_limit": self.api_rate_limit,
            "webhook_endpoints": self.webhook_endpoints,
            "custom_domains": self.custom_domains,
            "verified_badge": self.verified_badge,
            "forum_moderation": self.forum_moderation,
            "early_access": self.early_access
        }


@dataclass
class Developer:
    """Developer profile and account information"""
    developer_id: str
    username: str
    email: str
    display_name: str = ""
    
    # Profile information
    bio: str = ""
    company: Optional[str] = None
    website: Optional[str] = None
    github_username: Optional[str] = None
    twitter_username: Optional[str] = None
    avatar_url: Optional[str] = None
    
    # Account status
    status: DeveloperStatus = DeveloperStatus.PENDING
    tier: DeveloperTier = DeveloperTier.COMMUNITY
    verified: bool = False
    
    # Contact and support
    support_email: Optional[str] = None
    phone: Optional[str] = None
    
    # Location and timezone
    country: Optional[str] = None
    timezone: str = "UTC"
    
    # Preferences
    email_notifications: bool = True
    marketing_emails: bool = False
    public_profile: bool = True
    
    # API and security
    api_key: Optional[str] = None
    api_key_created_at: Optional[datetime] = None
    two_factor_enabled: bool = False
    
    # Metrics and performance
    metrics: DeveloperMetrics = field(default_factory=lambda: DeveloperMetrics(""))
    
    # Payment information
    payment_email: Optional[str] = None
    tax_id: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    
    def __post_init__(self):
        if self.metrics.developer_id == "":
            self.metrics.developer_id = self.developer_id
    
    def generate_api_key(self) -> str:
        """Generate new API key"""
        timestamp = str(int(datetime.now().timestamp()))
        key_data = f"{self.developer_id}:{timestamp}:{uuid.uuid4().hex}"
        api_key = hashlib.sha256(key_data.encode()).hexdigest()
        
        self.api_key = f"prm_{api_key[:32]}"
        self.api_key_created_at = datetime.now(timezone.utc)
        
        return self.api_key
    
    def is_eligible_for_tier(self, target_tier: DeveloperTier) -> Tuple[bool, List[str]]:
        """Check if developer is eligible for tier upgrade"""
        
        requirements_met = []
        requirements_failed = []
        
        if target_tier == DeveloperTier.VERIFIED:
            # Verified tier requirements
            if self.metrics.published_integrations >= 1:
                requirements_met.append("Has published integrations")
            else:
                requirements_failed.append("Need at least 1 published integration")
            
            if self.metrics.average_rating >= 3.5:
                requirements_met.append("Good average rating")
            else:
                requirements_failed.append("Need average rating of 3.5+")
            
            if self.verified:
                requirements_met.append("Email verified")
            else:
                requirements_failed.append("Need to verify email")
        
        elif target_tier == DeveloperTier.PROFESSIONAL:
            # Professional tier requirements
            if self.metrics.published_integrations >= 3:
                requirements_met.append("Has multiple integrations")
            else:
                requirements_failed.append("Need at least 3 published integrations")
            
            if self.metrics.total_downloads >= 1000:
                requirements_met.append("Sufficient downloads")
            else:
                requirements_failed.append("Need at least 1,000 total downloads")
            
            if self.metrics.average_rating >= 4.0:
                requirements_met.append("High average rating")
            else:
                requirements_failed.append("Need average rating of 4.0+")
        
        elif target_tier == DeveloperTier.ENTERPRISE:
            # Enterprise tier requirements
            if self.metrics.published_integrations >= 5:
                requirements_met.append("Has many integrations")
            else:
                requirements_failed.append("Need at least 5 published integrations")
            
            if self.metrics.total_downloads >= 10000:
                requirements_met.append("High download count")
            else:
                requirements_failed.append("Need at least 10,000 total downloads")
            
            if self.metrics.monthly_revenue >= 1000:
                requirements_met.append("Significant revenue")
            else:
                requirements_failed.append("Need at least $1,000 monthly revenue")
        
        is_eligible = len(requirements_failed) == 0
        return is_eligible, requirements_failed
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "developer_id": self.developer_id,
            "username": self.username,
            "display_name": self.display_name,
            "bio": self.bio,
            "company": self.company,
            "website": self.website,
            "github_username": self.github_username,
            "twitter_username": self.twitter_username,
            "avatar_url": self.avatar_url,
            "status": self.status.value,
            "tier": self.tier.value,
            "verified": self.verified,
            "country": self.country,
            "timezone": self.timezone,
            "public_profile": self.public_profile,
            "two_factor_enabled": self.two_factor_enabled,
            "metrics": self.metrics.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None
        }
        
        if include_sensitive:
            data.update({
                "email": self.email,
                "support_email": self.support_email,
                "phone": self.phone,
                "email_notifications": self.email_notifications,
                "marketing_emails": self.marketing_emails,
                "payment_email": self.payment_email,
                "tax_id": self.tax_id,
                "api_key": self.api_key,
                "api_key_created_at": self.api_key_created_at.isoformat() if self.api_key_created_at else None
            })
        
        return data


@dataclass
class DeveloperApplication:
    """Application for tier upgrade or special programs"""
    application_id: str
    developer_id: str
    application_type: str  # tier_upgrade, partner_program, etc.
    target_tier: Optional[DeveloperTier] = None
    
    # Application data
    application_data: Dict[str, Any] = field(default_factory=dict)
    business_justification: str = ""
    
    # Status and review
    status: ApplicationStatus = ApplicationStatus.SUBMITTED
    reviewer_id: Optional[str] = None
    review_notes: str = ""
    
    # Timeline
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reviewed_at: Optional[datetime] = None
    decision_at: Optional[datetime] = None
    
    # Supporting documents
    documents: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "application_id": self.application_id,
            "developer_id": self.developer_id,
            "application_type": self.application_type,
            "target_tier": self.target_tier.value if self.target_tier else None,
            "application_data": self.application_data,
            "business_justification": self.business_justification,
            "status": self.status.value,
            "reviewer_id": self.reviewer_id,
            "review_notes": self.review_notes,
            "submitted_at": self.submitted_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "decision_at": self.decision_at.isoformat() if self.decision_at else None,
            "documents": self.documents
        }


class DeveloperTierManager:
    """Manager for developer tiers and benefits"""
    
    def __init__(self):
        self.tier_benefits = {
            DeveloperTier.COMMUNITY: DeveloperBenefits(
                tier=DeveloperTier.COMMUNITY,
                max_integrations=3,
                max_versions_per_integration=10,
                revenue_share_percent=70.0,
                api_rate_limit=500,
                webhook_endpoints=2
            ),
            
            DeveloperTier.VERIFIED: DeveloperBenefits(
                tier=DeveloperTier.VERIFIED,
                max_integrations=10,
                max_versions_per_integration=20,
                revenue_share_percent=75.0,
                verified_badge=True,
                api_rate_limit=2000,
                webhook_endpoints=5
            ),
            
            DeveloperTier.PROFESSIONAL: DeveloperBenefits(
                tier=DeveloperTier.PROFESSIONAL,
                max_integrations=25,
                max_versions_per_integration=50,
                revenue_share_percent=80.0,
                priority_support=True,
                beta_features_access=True,
                featured_listings=2,
                promoted_listings=5,
                advanced_analytics=True,
                verified_badge=True,
                api_rate_limit=5000,
                webhook_endpoints=10,
                custom_domains=1
            ),
            
            DeveloperTier.ENTERPRISE: DeveloperBenefits(
                tier=DeveloperTier.ENTERPRISE,
                max_integrations=100,
                max_versions_per_integration=100,
                revenue_share_percent=85.0,
                priority_support=True,
                beta_features_access=True,
                custom_branding=True,
                white_label_options=True,
                featured_listings=5,
                promoted_listings=10,
                advanced_analytics=True,
                verified_badge=True,
                api_rate_limit=20000,
                webhook_endpoints=25,
                custom_domains=5,
                early_access=True
            ),
            
            DeveloperTier.PARTNER: DeveloperBenefits(
                tier=DeveloperTier.PARTNER,
                max_integrations=500,
                max_versions_per_integration=200,
                revenue_share_percent=90.0,
                priority_support=True,
                beta_features_access=True,
                custom_branding=True,
                white_label_options=True,
                featured_listings=10,
                promoted_listings=20,
                advanced_analytics=True,
                verified_badge=True,
                forum_moderation=True,
                api_rate_limit=100000,
                webhook_endpoints=100,
                custom_domains=20,
                early_access=True
            )
        }
    
    def get_benefits(self, tier: DeveloperTier) -> DeveloperBenefits:
        """Get benefits for a tier"""
        return self.tier_benefits[tier]
    
    def get_all_tiers(self) -> Dict[str, Dict[str, Any]]:
        """Get all tiers with their benefits"""
        return {
            tier.value: benefits.to_dict() 
            for tier, benefits in self.tier_benefits.items()
        }
    
    def can_upgrade_to_tier(self, developer: Developer, target_tier: DeveloperTier) -> Tuple[bool, List[str]]:
        """Check if developer can upgrade to target tier"""
        return developer.is_eligible_for_tier(target_tier)


class DeveloperOnboarding:
    """Developer onboarding and verification system"""
    
    def __init__(self):
        self.onboarding_steps = [
            "email_verification",
            "profile_completion",
            "terms_acceptance",
            "first_integration_created",
            "documentation_reviewed",
            "community_guidelines_read"
        ]
        
        self.verification_requirements = {
            DeveloperTier.VERIFIED: [
                "email_verified",
                "identity_verified",
                "phone_verified"
            ],
            DeveloperTier.PROFESSIONAL: [
                "business_verification",
                "tax_information_provided",
                "payment_method_added"
            ],
            DeveloperTier.ENTERPRISE: [
                "company_verification",
                "legal_agreements_signed",
                "security_assessment_passed"
            ]
        }
    
    def get_onboarding_progress(self, developer: Developer) -> Dict[str, Any]:
        """Get onboarding progress for developer"""
        
        # This would check actual completion status
        # For now, return mock progress
        completed_steps = []
        pending_steps = []
        
        for step in self.onboarding_steps:
            if self._is_step_completed(developer, step):
                completed_steps.append(step)
            else:
                pending_steps.append(step)
        
        progress_percent = (len(completed_steps) / len(self.onboarding_steps)) * 100
        
        return {
            "total_steps": len(self.onboarding_steps),
            "completed_steps": completed_steps,
            "pending_steps": pending_steps,
            "progress_percent": progress_percent,
            "is_complete": len(pending_steps) == 0
        }
    
    def get_verification_requirements(self, tier: DeveloperTier) -> List[str]:
        """Get verification requirements for tier"""
        return self.verification_requirements.get(tier, [])
    
    def _is_step_completed(self, developer: Developer, step: str) -> bool:
        """Check if onboarding step is completed"""
        
        if step == "email_verification":
            return developer.verified
        elif step == "profile_completion":
            return bool(developer.display_name and developer.bio)
        elif step == "terms_acceptance":
            return True  # Would check actual acceptance
        elif step == "first_integration_created":
            return developer.metrics.total_integrations > 0
        else:
            return False  # Default to not completed


class EcosystemAnalytics:
    """Analytics and insights for the developer ecosystem"""
    
    def __init__(self):
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_ttl_seconds = 3600  # 1 hour
    
    async def get_ecosystem_overview(self) -> Dict[str, Any]:
        """Get comprehensive ecosystem overview"""
        
        cache_key = "ecosystem_overview"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]["data"]
        
        # Calculate metrics (would query actual data in production)
        overview = {
            "total_developers": 1250,
            "active_developers": 890,
            "new_developers_this_month": 45,
            "developer_tier_distribution": {
                "community": 650,
                "verified": 420,
                "professional": 150,
                "enterprise": 25,
                "partner": 5
            },
            "top_countries": [
                {"country": "United States", "count": 280},
                {"country": "United Kingdom", "count": 150},
                {"country": "Germany", "count": 120},
                {"country": "Canada", "count": 95},
                {"country": "India", "count": 85}
            ],
            "average_integrations_per_developer": 2.8,
            "total_revenue_last_month": 125000.50,
            "top_performing_developers": await self._get_top_developers(),
            "developer_satisfaction_score": 4.2,
            "support_response_time_avg": 18.5
        }
        
        # Cache result
        self._cache_result(cache_key, overview)
        
        return overview
    
    async def get_developer_analytics(self, developer_id: str) -> Dict[str, Any]:
        """Get detailed analytics for specific developer"""
        
        cache_key = f"developer_analytics_{developer_id}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]["data"]
        
        # Calculate analytics (would query actual data)
        analytics = {
            "performance_trends": {
                "downloads_last_30_days": [120, 135, 98, 145, 167, 189, 201],
                "revenue_last_30_days": [450.20, 520.30, 380.50, 610.75, 720.40, 890.25, 980.60],
                "ratings_trend": [4.1, 4.2, 4.1, 4.3, 4.4, 4.3, 4.5]
            },
            "integration_performance": {
                "top_performing": ["integration_1", "integration_3"],
                "needs_attention": ["integration_2"],
                "performance_by_category": {
                    "AI Models": {"downloads": 1200, "rating": 4.5},
                    "Plugins": {"downloads": 800, "rating": 4.2}
                }
            },
            "user_engagement": {
                "total_users": 450,
                "active_users": 320,
                "user_retention_rate": 0.71,
                "average_session_duration_minutes": 25.5
            },
            "revenue_breakdown": {
                "subscription_revenue": 1250.75,
                "one_time_purchases": 340.20,
                "usage_based_revenue": 890.45,
                "revenue_by_integration": {
                    "integration_1": 950.30,
                    "integration_2": 280.60,
                    "integration_3": 1250.50
                }
            },
            "support_metrics": {
                "tickets_opened": 12,
                "tickets_resolved": 10,
                "average_resolution_time_hours": 8.5,
                "customer_satisfaction": 4.3
            }
        }
        
        # Cache result
        self._cache_result(cache_key, analytics)
        
        return analytics
    
    async def get_marketplace_insights(self) -> Dict[str, Any]:
        """Get marketplace-wide insights"""
        
        cache_key = "marketplace_insights"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]["data"]
        
        insights = {
            "growth_trends": {
                "developer_growth_rate": 0.18,  # 18% monthly growth
                "integration_growth_rate": 0.22,
                "revenue_growth_rate": 0.15
            },
            "market_opportunities": [
                {
                    "category": "AI Models",
                    "opportunity_score": 85,
                    "demand_level": "High",
                    "competition_level": "Medium",
                    "suggested_focus": "Specialized domain models"
                },
                {
                    "category": "Data Connectors",
                    "opportunity_score": 78,
                    "demand_level": "High",
                    "competition_level": "Low",
                    "suggested_focus": "Enterprise database connectors"
                }
            ],
            "ecosystem_health": {
                "diversity_score": 0.73,  # How diverse the ecosystem is
                "sustainability_score": 0.81,  # Long-term sustainability
                "innovation_index": 0.69,  # Rate of innovation
                "quality_index": 0.85  # Overall quality score
            },
            "developer_segments": {
                "hobbyists": {"count": 400, "contribution_percent": 15},
                "professionals": {"count": 300, "contribution_percent": 45},
                "enterprises": {"count": 50, "contribution_percent": 40}
            }
        }
        
        # Cache result
        self._cache_result(cache_key, insights)
        
        return insights
    
    async def _get_top_developers(self) -> List[Dict[str, Any]]:
        """Get top performing developers"""
        
        # Mock data - would query actual developers
        return [
            {
                "developer_id": "dev_001",
                "username": "ai_innovator",
                "total_downloads": 15000,
                "monthly_revenue": 2500.75,
                "average_rating": 4.8,
                "tier": "enterprise"
            },
            {
                "developer_id": "dev_002", 
                "username": "data_wizard",
                "total_downloads": 12000,
                "monthly_revenue": 1800.50,
                "average_rating": 4.6,
                "tier": "professional"
            }
        ]
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        
        if cache_key not in self.metrics_cache:
            return False
        
        cached_entry = self.metrics_cache[cache_key]
        age = (datetime.now(timezone.utc) - cached_entry["cached_at"]).total_seconds()
        
        return age < self.cache_ttl_seconds
    
    def _cache_result(self, cache_key: str, data: Any):
        """Cache analytics result"""
        
        self.metrics_cache[cache_key] = {
            "data": data,
            "cached_at": datetime.now(timezone.utc)
        }


class EcosystemManager:
    """Main ecosystem management system"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./ecosystem_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Core components
        self.developers: Dict[str, Developer] = {}
        self.applications: Dict[str, DeveloperApplication] = {}
        self.tier_manager = DeveloperTierManager()
        self.onboarding = DeveloperOnboarding()
        self.analytics = EcosystemAnalytics()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Any]] = {}
        
        # Statistics
        self.stats = {
            "total_developers": 0,
            "active_developers": 0,
            "pending_applications": 0,
            "tier_distribution": {},
            "monthly_new_developers": 0
        }
        
        logger.info("Ecosystem Manager initialized")
    
    def register_developer(self, developer: Developer) -> bool:
        """Register a new developer"""
        
        try:
            # Validate developer data
            if not self._validate_developer(developer):
                return False
            
            # Generate API key
            developer.generate_api_key()
            
            # Store developer
            self.developers[developer.developer_id] = developer
            
            # Update statistics
            self.stats["total_developers"] += 1
            if developer.status == DeveloperStatus.ACTIVE:
                self.stats["active_developers"] += 1
            
            self._update_tier_distribution()
            
            logger.info(f"Registered developer: {developer.username}")
            
            # Emit event
            asyncio.create_task(self._emit_event("developer_registered", developer.to_dict()))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register developer {developer.username}: {e}")
            return False
    
    def update_developer(self, developer_id: str, updates: Dict[str, Any]) -> bool:
        """Update developer information"""
        
        if developer_id not in self.developers:
            logger.error(f"Developer not found: {developer_id}")
            return False
        
        try:
            developer = self.developers[developer_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(developer, key):
                    setattr(developer, key, value)
            
            developer.updated_at = datetime.now(timezone.utc)
            
            logger.info(f"Updated developer: {developer.username}")
            
            # Emit event
            asyncio.create_task(self._emit_event("developer_updated", developer.to_dict()))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update developer {developer_id}: {e}")
            return False
    
    def get_developer(self, developer_id: str) -> Optional[Developer]:
        """Get developer by ID"""
        return self.developers.get(developer_id)
    
    def get_developer_by_username(self, username: str) -> Optional[Developer]:
        """Get developer by username"""
        for developer in self.developers.values():
            if developer.username == username:
                return developer
        return None
    
    def submit_tier_application(self, developer_id: str, target_tier: DeveloperTier,
                               business_justification: str = "") -> Optional[str]:
        """Submit application for tier upgrade"""
        
        developer = self.get_developer(developer_id)
        if not developer:
            logger.error(f"Developer not found: {developer_id}")
            return None
        
        # Check eligibility
        eligible, requirements = developer.is_eligible_for_tier(target_tier)
        
        if not eligible:
            logger.warning(f"Developer {developer_id} not eligible for {target_tier.value}: {requirements}")
            # Still allow application but note requirements not met
        
        try:
            application_id = f"app_{uuid.uuid4().hex[:8]}"
            
            application = DeveloperApplication(
                application_id=application_id,
                developer_id=developer_id,
                application_type="tier_upgrade",
                target_tier=target_tier,
                business_justification=business_justification,
                application_data={
                    "current_tier": developer.tier.value,
                    "current_metrics": developer.metrics.to_dict(),
                    "eligibility_check": {
                        "eligible": eligible,
                        "missing_requirements": requirements
                    }
                }
            )
            
            self.applications[application_id] = application
            self.stats["pending_applications"] += 1
            
            logger.info(f"Tier application submitted: {application_id} for {developer.username}")
            
            # Emit event
            asyncio.create_task(self._emit_event("application_submitted", application.to_dict()))
            
            return application_id
            
        except Exception as e:
            logger.error(f"Failed to submit application for {developer_id}: {e}")
            return None
    
    def review_application(self, application_id: str, reviewer_id: str,
                          decision: ApplicationStatus, notes: str = "") -> bool:
        """Review and decide on application"""
        
        if application_id not in self.applications:
            logger.error(f"Application not found: {application_id}")
            return False
        
        try:
            application = self.applications[application_id]
            application.status = decision
            application.reviewer_id = reviewer_id
            application.review_notes = notes
            application.reviewed_at = datetime.now(timezone.utc)
            
            if decision in [ApplicationStatus.APPROVED, ApplicationStatus.REJECTED]:
                application.decision_at = datetime.now(timezone.utc)
                self.stats["pending_applications"] -= 1
                
                # If approved and it's a tier upgrade, update developer tier
                if decision == ApplicationStatus.APPROVED and application.application_type == "tier_upgrade":
                    developer = self.get_developer(application.developer_id)
                    if developer and application.target_tier:
                        old_tier = developer.tier
                        developer.tier = application.target_tier
                        developer.updated_at = datetime.now(timezone.utc)
                        
                        self._update_tier_distribution()
                        
                        logger.info(f"Upgraded developer {developer.username} from {old_tier.value} to {application.target_tier.value}")
                        
                        # Emit event
                        asyncio.create_task(self._emit_event("tier_upgraded", {
                            "developer_id": developer.developer_id,
                            "old_tier": old_tier.value,
                            "new_tier": application.target_tier.value
                        }))
            
            logger.info(f"Reviewed application: {application_id} - {decision.value}")
            
            # Emit event
            asyncio.create_task(self._emit_event("application_reviewed", application.to_dict()))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to review application {application_id}: {e}")
            return False
    
    def get_developer_benefits(self, developer_id: str) -> Optional[Dict[str, Any]]:
        """Get benefits for developer based on their tier"""
        
        developer = self.get_developer(developer_id)
        if not developer:
            return None
        
        benefits = self.tier_manager.get_benefits(developer.tier)
        return benefits.to_dict()
    
    def get_onboarding_progress(self, developer_id: str) -> Optional[Dict[str, Any]]:
        """Get onboarding progress for developer"""
        
        developer = self.get_developer(developer_id)
        if not developer:
            return None
        
        return self.onboarding.get_onboarding_progress(developer)
    
    async def get_developer_analytics(self, developer_id: str) -> Optional[Dict[str, Any]]:
        """Get analytics for developer"""
        
        developer = self.get_developer(developer_id)
        if not developer:
            return None
        
        return await self.analytics.get_developer_analytics(developer_id)
    
    async def get_ecosystem_overview(self) -> Dict[str, Any]:
        """Get ecosystem overview and insights"""
        return await self.analytics.get_ecosystem_overview()
    
    def list_developers(self, filters: Optional[Dict[str, Any]] = None,
                       sort_by: str = "created_at", limit: int = 50) -> List[Dict[str, Any]]:
        """List developers with filtering and sorting"""
        
        developers = list(self.developers.values())
        
        # Apply filters
        if filters:
            developers = self._apply_developer_filters(developers, filters)
        
        # Sort developers
        developers = self._sort_developers(developers, sort_by)
        
        # Limit results
        developers = developers[:limit]
        
        return [dev.to_dict() for dev in developers]
    
    def list_applications(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List applications with optional filtering"""
        
        applications = list(self.applications.values())
        
        # Apply filters
        if filters:
            if "status" in filters:
                applications = [app for app in applications if app.status.value == filters["status"]]
            
            if "application_type" in filters:
                applications = [app for app in applications if app.application_type == filters["application_type"]]
            
            if "target_tier" in filters:
                applications = [app for app in applications 
                              if app.target_tier and app.target_tier.value == filters["target_tier"]]
        
        # Sort by submission date (newest first)
        applications.sort(key=lambda x: x.submitted_at, reverse=True)
        
        return [app.to_dict() for app in applications]
    
    def _validate_developer(self, developer: Developer) -> bool:
        """Validate developer data"""
        
        # Basic validation
        if not developer.username or not developer.email or not developer.developer_id:
            logger.error("Developer missing required fields")
            return False
        
        # Check for duplicate username
        if self.get_developer_by_username(developer.username):
            logger.error(f"Username already exists: {developer.username}")
            return False
        
        # Check for duplicate ID
        if developer.developer_id in self.developers:
            logger.error(f"Developer ID already exists: {developer.developer_id}")
            return False
        
        # Email format validation (basic)
        if "@" not in developer.email:
            logger.error(f"Invalid email format: {developer.email}")
            return False
        
        return True
    
    def _apply_developer_filters(self, developers: List[Developer], 
                                filters: Dict[str, Any]) -> List[Developer]:
        """Apply filters to developer list"""
        
        filtered = developers
        
        if "status" in filters:
            filtered = [d for d in filtered if d.status.value == filters["status"]]
        
        if "tier" in filters:
            filtered = [d for d in filtered if d.tier.value == filters["tier"]]
        
        if "verified" in filters:
            filtered = [d for d in filtered if d.verified == filters["verified"]]
        
        if "country" in filters:
            filtered = [d for d in filtered if d.country == filters["country"]]
        
        if "min_integrations" in filters:
            filtered = [d for d in filtered if d.metrics.published_integrations >= filters["min_integrations"]]
        
        if "min_rating" in filters:
            filtered = [d for d in filtered if d.metrics.average_rating >= filters["min_rating"]]
        
        return filtered
    
    def _sort_developers(self, developers: List[Developer], sort_by: str) -> List[Developer]:
        """Sort developers by specified criteria"""
        
        if sort_by == "created_at":
            return sorted(developers, key=lambda x: x.created_at, reverse=True)
        elif sort_by == "popularity":
            return sorted(developers, key=lambda x: x.metrics.calculate_popularity_score(), reverse=True)
        elif sort_by == "quality":
            return sorted(developers, key=lambda x: x.metrics.calculate_quality_score(), reverse=True)
        elif sort_by == "downloads":
            return sorted(developers, key=lambda x: x.metrics.total_downloads, reverse=True)
        elif sort_by == "revenue":
            return sorted(developers, key=lambda x: x.metrics.total_revenue, reverse=True)
        elif sort_by == "username":
            return sorted(developers, key=lambda x: x.username.lower())
        else:
            return developers
    
    def _update_tier_distribution(self):
        """Update tier distribution statistics"""
        
        distribution = {}
        for tier in DeveloperTier:
            distribution[tier.value] = 0
        
        for developer in self.developers.values():
            distribution[developer.tier.value] += 1
        
        self.stats["tier_distribution"] = distribution
    
    def add_event_handler(self, event_type: str, handler):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Added event handler for {event_type}")
    
    async def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit ecosystem event"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(event_data)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")
    
    def get_ecosystem_stats(self) -> Dict[str, Any]:
        """Get comprehensive ecosystem statistics"""
        
        # Update dynamic stats
        self.stats["active_developers"] = len([
            d for d in self.developers.values() 
            if d.status == DeveloperStatus.ACTIVE
        ])
        
        return {
            "ecosystem_statistics": self.stats,
            "tier_benefits": self.tier_manager.get_all_tiers(),
            "top_developers": [
                {
                    "developer_id": dev.developer_id,
                    "username": dev.username,
                    "tier": dev.tier.value,
                    "quality_score": dev.metrics.calculate_quality_score(),
                    "popularity_score": dev.metrics.calculate_popularity_score()
                }
                for dev in sorted(
                    self.developers.values(),
                    key=lambda x: x.metrics.calculate_popularity_score(),
                    reverse=True
                )[:10]
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Export main classes
__all__ = [
    'DeveloperTier',
    'DeveloperStatus',
    'ApplicationStatus',
    'DeveloperMetrics',
    'DeveloperBenefits',
    'Developer',
    'DeveloperApplication',
    'DeveloperTierManager',
    'DeveloperOnboarding',
    'EcosystemAnalytics',
    'EcosystemManager'
]