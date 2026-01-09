#!/usr/bin/env python3
"""
PRSM Alpha User Management System
================================

Manages the alpha testing program with 100+ technical users, including:
- User registration and onboarding
- API key generation and management
- Usage tracking and analytics
- Feedback collection and analysis
- Community engagement features

🎯 ALPHA PROGRAM GOALS:
✅ Recruit 100+ technical users from AI/ML community
✅ Provide comprehensive onboarding experience
✅ Track usage patterns and performance metrics
✅ Collect detailed feedback for product improvement
✅ Build community around PRSM platform
"""

import asyncio
import json
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import structlog

from ..core.database import db_manager
from ..tokenomics.ftns_service import FTNSService
from ..core.models import UserInput, PRSMResponse

logger = structlog.get_logger(__name__)


class UserType(Enum):
    """Alpha user types based on technical background"""
    AI_RESEARCHER = "ai_researcher"
    SOFTWARE_ENGINEER = "software_engineer"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    ACADEMIC = "academic"
    STARTUP_FOUNDER = "startup_founder"
    ENTERPRISE_DEVELOPER = "enterprise_developer"
    OTHER = "other"


class RegistrationStatus(Enum):
    """Alpha registration status"""
    PENDING = "pending"
    APPROVED = "approved"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"


@dataclass
class AlphaUser:
    """Alpha user profile"""
    user_id: str
    email: str
    name: str
    organization: str
    user_type: UserType
    registration_status: RegistrationStatus
    api_key: str
    ftns_balance: float
    registration_date: datetime
    last_active: Optional[datetime] = None
    total_queries: int = 0
    feedback_count: int = 0
    onboarding_completed: bool = False
    preferred_region: Optional[str] = None
    use_cases: List[str] = field(default_factory=list)
    technical_interests: List[str] = field(default_factory=list)
    collaboration_consent: bool = False
    marketing_consent: bool = False


@dataclass
class UsageMetrics:
    """User usage analytics"""
    user_id: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_query_latency: float
    total_cost: float
    avg_cost_per_query: float
    most_used_features: List[str]
    preferred_routing_strategy: str
    regional_usage: Dict[str, int]
    daily_usage_pattern: Dict[int, int]  # hour -> query_count
    quality_ratings: Dict[str, float]  # aspect -> avg_rating


@dataclass
class FeedbackEntry:
    """User feedback entry"""
    feedback_id: str
    user_id: str
    category: str  # bug_report|feature_request|general_feedback|quality_rating
    priority: str  # low|medium|high|critical
    title: str
    description: str
    query_id: Optional[str] = None
    rating: Optional[int] = None  # 1-5 scale
    status: str = "open"  # open|in_progress|resolved|closed
    submitted_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlphaUserManager:
    """Manages alpha user registration, onboarding, and engagement"""
    
    def __init__(self, ftns_service: Optional[FTNSService] = None):
        self.ftns_service = ftns_service or FTNSService()
        self.alpha_users: Dict[str, AlphaUser] = {}
        self.usage_metrics: Dict[str, UsageMetrics] = {}
        self.feedback_entries: Dict[str, FeedbackEntry] = {}
        
        # Alpha program configuration
        self.max_alpha_users = 150  # Target 100+, allow buffer
        self.initial_ftns_grant = 1000.0  # Initial FTNS tokens
        self.bonus_ftns_threshold = 50  # Bonus after 50 queries
        self.bonus_ftns_amount = 500.0
        self.required_feedback_count = 10  # For program completion
        
    async def register_alpha_user(self, 
                                email: str,
                                name: str,
                                organization: str,
                                user_type_str: str,
                                use_case: str,
                                technical_background: str,
                                experience_level: str = "intermediate",
                                collaboration_consent: bool = False,
                                marketing_consent: bool = False) -> Dict[str, Any]:
        """Register a new alpha user"""
        
        # Check if registration is still open
        active_users = len([u for u in self.alpha_users.values() 
                           if u.registration_status in [RegistrationStatus.ACTIVE, RegistrationStatus.APPROVED]])
        
        if active_users >= self.max_alpha_users:
            return {
                "success": False,
                "error": "Alpha program is currently at capacity",
                "waitlist_position": active_users - self.max_alpha_users + 1
            }
        
        # Check for duplicate email
        if any(user.email == email for user in self.alpha_users.values()):
            return {
                "success": False,
                "error": "Email already registered for alpha program"
            }
        
        # Generate user ID and API key
        user_id = f"alpha_{secrets.token_hex(8)}"
        api_key = f"prsm_alpha_{secrets.token_hex(16)}"
        
        # Parse user type
        try:
            user_type = UserType(user_type_str.lower())
        except ValueError:
            user_type = UserType.OTHER
        
        # Create alpha user
        alpha_user = AlphaUser(
            user_id=user_id,
            email=email,
            name=name,
            organization=organization,
            user_type=user_type,
            registration_status=RegistrationStatus.APPROVED,  # Auto-approve for alpha
            api_key=api_key,
            ftns_balance=self.initial_ftns_grant,
            registration_date=datetime.now(),
            use_cases=[use_case],
            technical_interests=[technical_background, experience_level],
            collaboration_consent=collaboration_consent,
            marketing_consent=marketing_consent
        )
        
        # Initialize FTNS balance
        await self.ftns_service.create_user_account(user_id, self.initial_ftns_grant)
        
        # Store user
        self.alpha_users[user_id] = alpha_user
        
        # Initialize usage metrics
        self.usage_metrics[user_id] = UsageMetrics(
            user_id=user_id,
            total_queries=0,
            successful_queries=0,
            failed_queries=0,
            avg_query_latency=0.0,
            total_cost=0.0,
            avg_cost_per_query=0.0,
            most_used_features=[],
            preferred_routing_strategy="",
            regional_usage={},
            daily_usage_pattern={},
            quality_ratings={}
        )
        
        # Send welcome email and onboarding materials
        await self._send_welcome_email(alpha_user)
        
        logger.info("Alpha user registered",
                   user_id=user_id,
                   email=email,
                   organization=organization,
                   user_type=user_type.value)
        
        return {
            "success": True,
            "user_id": user_id,
            "api_key": api_key,
            "ftns_balance": self.initial_ftns_grant,
            "onboarding_url": f"https://alpha.prsm.network/onboard/{user_id}",
            "documentation_url": "https://docs.prsm.network/alpha",
            "community_discord": "https://discord.gg/prsm-alpha"
        }
    
    async def authenticate_user(self, api_key: str) -> Optional[AlphaUser]:
        """Authenticate user by API key"""
        for user in self.alpha_users.values():
            if user.api_key == api_key and user.registration_status == RegistrationStatus.ACTIVE:
                user.last_active = datetime.now()
                return user
        return None
    
    async def track_query_usage(self, 
                              user_id: str,
                              query: str,
                              response: PRSMResponse,
                              latency: float,
                              cost: float,
                              routing_strategy: str,
                              region: str) -> None:
        """Track user query for analytics"""
        
        if user_id not in self.usage_metrics:
            logger.warning("User not found in usage metrics", user_id=user_id)
            return
        
        metrics = self.usage_metrics[user_id]
        user = self.alpha_users.get(user_id)
        
        # Update query counts
        metrics.total_queries += 1
        if response.success:
            metrics.successful_queries += 1
        else:
            metrics.failed_queries += 1
        
        # Update user total
        if user:
            user.total_queries += 1
        
        # Update latency (running average)
        if metrics.total_queries == 1:
            metrics.avg_query_latency = latency
        else:
            metrics.avg_query_latency = (
                (metrics.avg_query_latency * (metrics.total_queries - 1) + latency) 
                / metrics.total_queries
            )
        
        # Update cost tracking
        metrics.total_cost += cost
        metrics.avg_cost_per_query = metrics.total_cost / metrics.total_queries
        
        # Update regional usage
        if region not in metrics.regional_usage:
            metrics.regional_usage[region] = 0
        metrics.regional_usage[region] += 1
        
        # Update daily usage pattern
        current_hour = datetime.now().hour
        if current_hour not in metrics.daily_usage_pattern:
            metrics.daily_usage_pattern[current_hour] = 0
        metrics.daily_usage_pattern[current_hour] += 1
        
        # Update preferred routing strategy
        metrics.preferred_routing_strategy = routing_strategy
        
        # Check for milestones and rewards
        await self._check_usage_milestones(user_id, metrics)
        
        logger.info("Query usage tracked",
                   user_id=user_id,
                   total_queries=metrics.total_queries,
                   avg_latency=metrics.avg_query_latency,
                   total_cost=metrics.total_cost)
    
    async def submit_feedback(self,
                            user_id: str,
                            category: str,
                            title: str,
                            description: str,
                            priority: str = "medium",
                            query_id: Optional[str] = None,
                            rating: Optional[int] = None,
                            tags: List[str] = None) -> str:
        """Submit user feedback"""
        
        feedback_id = f"feedback_{secrets.token_hex(8)}"
        
        feedback = FeedbackEntry(
            feedback_id=feedback_id,
            user_id=user_id,
            category=category,
            priority=priority,
            title=title,
            description=description,
            query_id=query_id,
            rating=rating,
            tags=tags or [],
            metadata={
                "user_agent": "alpha_program",
                "program_version": "v0.9.0"
            }
        )
        
        self.feedback_entries[feedback_id] = feedback
        
        # Update user feedback count
        if user_id in self.alpha_users:
            self.alpha_users[user_id].feedback_count += 1
        
        # Auto-categorize and prioritize based on content
        await self._process_feedback(feedback)
        
        logger.info("Feedback submitted",
                   feedback_id=feedback_id,
                   user_id=user_id,
                   category=category,
                   priority=priority)
        
        return feedback_id
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user analytics"""
        
        if user_id not in self.alpha_users:
            return {"error": "User not found"}
        
        user = self.alpha_users[user_id]
        metrics = self.usage_metrics.get(user_id)
        
        if not metrics:
            return {"error": "No usage data found"}
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(user, metrics)
        
        # Get feedback summary
        user_feedback = [f for f in self.feedback_entries.values() if f.user_id == user_id]
        
        return {
            "user_profile": {
                "user_id": user.user_id,
                "email": user.email,
                "organization": user.organization,
                "user_type": user.user_type.value,
                "registration_date": user.registration_date.isoformat(),
                "days_active": (datetime.now() - user.registration_date).days,
                "onboarding_completed": user.onboarding_completed
            },
            "usage_statistics": {
                "total_queries": metrics.total_queries,
                "success_rate": metrics.successful_queries / max(metrics.total_queries, 1),
                "avg_query_latency": metrics.avg_query_latency,
                "total_cost": metrics.total_cost,
                "cost_per_query": metrics.avg_cost_per_query,
                "preferred_routing": metrics.preferred_routing_strategy,
                "regional_distribution": metrics.regional_usage,
                "usage_by_hour": metrics.daily_usage_pattern
            },
            "engagement": {
                "engagement_score": engagement_score,
                "feedback_submissions": len(user_feedback),
                "collaboration_participation": user.collaboration_consent,
                "last_active": user.last_active.isoformat() if user.last_active else None
            },
            "feedback_summary": {
                "total_feedback": len(user_feedback),
                "bug_reports": len([f for f in user_feedback if f.category == "bug_report"]),
                "feature_requests": len([f for f in user_feedback if f.category == "feature_request"]),
                "avg_rating": sum(f.rating for f in user_feedback if f.rating) / max(len([f for f in user_feedback if f.rating]), 1)
            }
        }
    
    async def get_program_analytics(self) -> Dict[str, Any]:
        """Get overall alpha program analytics"""
        
        total_users = len(self.alpha_users)
        active_users = len([u for u in self.alpha_users.values() 
                           if u.last_active and u.last_active > datetime.now() - timedelta(days=7)])
        
        total_queries = sum(m.total_queries for m in self.usage_metrics.values())
        total_feedback = len(self.feedback_entries)
        
        # User type distribution
        user_types = {}
        for user in self.alpha_users.values():
            user_type = user.user_type.value
            user_types[user_type] = user_types.get(user_type, 0) + 1
        
        # Regional usage distribution
        regional_usage = {}
        for metrics in self.usage_metrics.values():
            for region, count in metrics.regional_usage.items():
                regional_usage[region] = regional_usage.get(region, 0) + count
        
        # Engagement metrics
        engagement_scores = [
            self._calculate_engagement_score(user, self.usage_metrics.get(user.user_id))
            for user in self.alpha_users.values()
            if self.usage_metrics.get(user.user_id)
        ]
        
        avg_engagement = sum(engagement_scores) / max(len(engagement_scores), 1)
        
        # Quality metrics
        quality_ratings = []
        for feedback in self.feedback_entries.values():
            if feedback.rating:
                quality_ratings.append(feedback.rating)
        
        avg_quality_rating = sum(quality_ratings) / max(len(quality_ratings), 1)
        
        return {
            "program_overview": {
                "total_registered_users": total_users,
                "active_users_7d": active_users,
                "target_users": self.max_alpha_users,
                "completion_rate": total_users / self.max_alpha_users,
                "program_start_date": min(u.registration_date for u in self.alpha_users.values()).isoformat() if self.alpha_users else None
            },
            "usage_metrics": {
                "total_queries": total_queries,
                "avg_queries_per_user": total_queries / max(total_users, 1),
                "total_feedback_entries": total_feedback,
                "feedback_per_user": total_feedback / max(total_users, 1)
            },
            "user_distribution": {
                "by_type": user_types,
                "by_region": regional_usage
            },
            "engagement_metrics": {
                "avg_engagement_score": avg_engagement,
                "high_engagement_users": len([s for s in engagement_scores if s > 0.7]),
                "avg_quality_rating": avg_quality_rating,
                "completion_candidates": len([u for u in self.alpha_users.values() 
                                            if u.feedback_count >= self.required_feedback_count])
            },
            "feedback_analysis": {
                "bug_reports": len([f for f in self.feedback_entries.values() if f.category == "bug_report"]),
                "feature_requests": len([f for f in self.feedback_entries.values() if f.category == "feature_request"]),
                "critical_issues": len([f for f in self.feedback_entries.values() if f.priority == "critical"]),
                "resolved_feedback": len([f for f in self.feedback_entries.values() if f.status == "resolved"])
            }
        }
    
    async def generate_user_completion_report(self, user_id: str) -> Dict[str, Any]:
        """Generate completion report for alpha user"""
        
        if user_id not in self.alpha_users:
            return {"error": "User not found"}
        
        user = self.alpha_users[user_id]
        metrics = self.usage_metrics.get(user_id)
        user_feedback = [f for f in self.feedback_entries.values() if f.user_id == user_id]
        
        # Check completion criteria
        completion_criteria = {
            "min_queries": 25,
            "min_feedback": self.required_feedback_count,
            "min_engagement_score": 0.5,
            "testing_duration_days": 14
        }
        
        days_active = (datetime.now() - user.registration_date).days
        engagement_score = self._calculate_engagement_score(user, metrics)
        
        criteria_met = {
            "queries_completed": metrics.total_queries >= completion_criteria["min_queries"] if metrics else False,
            "feedback_submitted": len(user_feedback) >= completion_criteria["min_feedback"],
            "engagement_sufficient": engagement_score >= completion_criteria["min_engagement_score"],
            "testing_duration_met": days_active >= completion_criteria["testing_duration_days"]
        }
        
        completion_eligible = all(criteria_met.values())
        
        return {
            "user_id": user_id,
            "email": user.email,
            "completion_eligible": completion_eligible,
            "completion_criteria": completion_criteria,
            "criteria_status": criteria_met,
            "performance_summary": {
                "total_queries": metrics.total_queries if metrics else 0,
                "feedback_count": len(user_feedback),
                "engagement_score": engagement_score,
                "days_active": days_active,
                "success_rate": metrics.successful_queries / max(metrics.total_queries, 1) if metrics else 0,
                "avg_cost_efficiency": metrics.avg_cost_per_query if metrics else 0
            },
            "feedback_quality": {
                "detailed_feedback": len([f for f in user_feedback if len(f.description) > 100]),
                "bug_reports": len([f for f in user_feedback if f.category == "bug_report"]),
                "feature_suggestions": len([f for f in user_feedback if f.category == "feature_request"])
            },
            "certificate_eligible": completion_eligible,
            "beta_program_eligible": completion_eligible and engagement_score > 0.7
        }
    
    def _calculate_engagement_score(self, user: AlphaUser, metrics: Optional[UsageMetrics]) -> float:
        """Calculate user engagement score (0.0 to 1.0)"""
        
        if not metrics:
            return 0.0
        
        # Base score from query activity
        query_score = min(metrics.total_queries / 50, 1.0)  # Max at 50 queries
        
        # Feedback engagement
        feedback_score = min(user.feedback_count / 10, 1.0)  # Max at 10 feedback items
        
        # Time-based consistency
        days_active = (datetime.now() - user.registration_date).days
        if days_active > 0:
            daily_average = metrics.total_queries / days_active
            consistency_score = min(daily_average / 3, 1.0)  # Max at 3 queries/day
        else:
            consistency_score = 0.0
        
        # Success rate component
        success_score = metrics.successful_queries / max(metrics.total_queries, 1)
        
        # Weighted combination
        engagement_score = (
            query_score * 0.3 +
            feedback_score * 0.3 +
            consistency_score * 0.2 +
            success_score * 0.2
        )
        
        return min(engagement_score, 1.0)
    
    async def _check_usage_milestones(self, user_id: str, metrics: UsageMetrics) -> None:
        """Check and reward usage milestones"""
        
        # Bonus FTNS tokens after significant usage
        if (metrics.total_queries == self.bonus_ftns_threshold and 
            user_id in self.alpha_users):
            
            await self.ftns_service.add_tokens(user_id, self.bonus_ftns_amount)
            
            logger.info("Milestone bonus awarded",
                       user_id=user_id,
                       milestone="50_queries",
                       bonus_amount=self.bonus_ftns_amount)
        
        # Additional milestones can be added here
        milestone_rewards = {
            100: 1000.0,   # 100 queries
            250: 2000.0,   # 250 queries
            500: 5000.0    # 500 queries
        }
        
        for milestone, reward in milestone_rewards.items():
            if metrics.total_queries == milestone:
                await self.ftns_service.add_tokens(user_id, reward)
                logger.info("Major milestone achieved",
                           user_id=user_id,
                           milestone=f"{milestone}_queries",
                           reward=reward)
    
    async def _process_feedback(self, feedback: FeedbackEntry) -> None:
        """Process and categorize feedback automatically"""
        
        # Auto-prioritize based on keywords
        critical_keywords = ["crash", "data loss", "security", "urgent", "broken"]
        high_keywords = ["slow", "timeout", "error", "fail", "bug"]
        
        description_lower = feedback.description.lower()
        
        if any(keyword in description_lower for keyword in critical_keywords):
            feedback.priority = "critical"
        elif any(keyword in description_lower for keyword in high_keywords):
            feedback.priority = "high"
        
        # Auto-tag based on content
        tag_mapping = {
            "performance": ["slow", "timeout", "latency", "speed"],
            "accuracy": ["wrong", "incorrect", "inaccurate", "quality"],
            "cost": ["expensive", "tokens", "pricing", "billing"],
            "routing": ["routing", "region", "local", "cloud"],
            "ui_ux": ["interface", "usability", "confusing", "difficult"]
        }
        
        for tag, keywords in tag_mapping.items():
            if any(keyword in description_lower for keyword in keywords):
                if tag not in feedback.tags:
                    feedback.tags.append(tag)
        
        # Notify team for critical issues
        if feedback.priority == "critical":
            await self._notify_critical_feedback(feedback)
    
    async def _notify_critical_feedback(self, feedback: FeedbackEntry) -> None:
        """Notify team about critical feedback"""
        
        logger.critical("Critical feedback received",
                       feedback_id=feedback.feedback_id,
                       user_id=feedback.user_id,
                       title=feedback.title,
                       description=feedback.description[:200])
    
    async def _send_welcome_email(self, user: AlphaUser) -> None:
        """Send welcome email to new alpha user"""
        
        # In a real implementation, this would send an actual email
        logger.info("Welcome email sent",
                   user_id=user.user_id,
                   email=user.email,
                   api_key=user.api_key[:20] + "...")
    
    async def export_analytics(self, format: str = "json") -> str:
        """Export program analytics for reporting"""
        
        analytics = await self.get_program_analytics()
        
        if format == "json":
            return json.dumps(analytics, indent=2, default=str)
        elif format == "csv":
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write program summary
            writer.writerow(["Metric", "Value"])
            for key, value in analytics["program_overview"].items():
                writer.writerow([key, value])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global alpha user manager instance
alpha_manager = None

def get_alpha_manager() -> AlphaUserManager:
    """Get or create global alpha user manager"""
    global alpha_manager
    if alpha_manager is None:
        alpha_manager = AlphaUserManager()
    return alpha_manager