"""
Early Adopter Program
=====================

Comprehensive early adopter program with tiered benefits, milestone tracking,
and community building features for PRSM's production launch.
"""

import asyncio
import structlog
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from prsm.core.config import settings
from prsm.economy.tokenomics.database_ftns_service import DatabaseFTNSService
from prsm.economy.governance.token_distribution import get_governance_distributor, GovernanceParticipantTier
from ..integrations.security.audit_logger import audit_logger
from .models import (
    EarlyAdopterTier, EarlyAdopterProfile, EarlyAdopterRegistration,
    EarlyAdopterStatusResponse, UserInterest
)

logger = structlog.get_logger(__name__)


class EarlyAdopterProgram:
    """
    Manages the early adopter program with tiered benefits, exclusive access,
    and community building features for PRSM's production launch.
    """
    
    def __init__(self):
        self.program_id = str(uuid4())
        self.logger = logger.bind(component="early_adopter_program", program_id=self.program_id)
        
        # Service integrations
        self.ftns_service = DatabaseFTNSService()
        self.governance_distributor = get_governance_distributor()
        
        # Program tracking
        self.early_adopters: Dict[str, EarlyAdopterProfile] = {}
        self.adoption_milestones = self._initialize_adoption_milestones()
        
        # Tier configurations
        self.tier_configs = {
            EarlyAdopterTier.PIONEER: {
                "max_users": 100,
                "join_number_range": (1, 100),
                "ftns_multiplier": Decimal('5.0'),
                "governance_bonus": Decimal('2.0'),
                "priority_level": 5,
                "exclusive_features": [
                    "direct_dev_access", "feature_preview", "governance_council_nomination",
                    "pioneer_badge", "lifetime_vip_status", "custom_integration_support"
                ],
                "recognition_title": "PRSM Pioneer",
                "special_privileges": [
                    "Priority support queue", "Beta feature access", "Advisory board invitation",
                    "Custom model hosting", "White-glove onboarding", "Direct founder access"
                ]
            },
            EarlyAdopterTier.EXPLORER: {
                "max_users": 400,
                "join_number_range": (101, 500),
                "ftns_multiplier": Decimal('3.0'),
                "governance_bonus": Decimal('1.5'),
                "priority_level": 4,
                "exclusive_features": [
                    "early_access", "explorer_badge", "community_leadership",
                    "advanced_analytics", "priority_model_access"
                ],
                "recognition_title": "PRSM Explorer",
                "special_privileges": [
                    "Feature preview access", "Community moderator privileges", 
                    "Advanced API access", "Priority model hosting", "Monthly dev calls"
                ]
            },
            EarlyAdopterTier.BUILDER: {
                "max_users": 1500,
                "join_number_range": (501, 2000),
                "ftns_multiplier": Decimal('2.0'),
                "governance_bonus": Decimal('1.0'),
                "priority_level": 3,
                "exclusive_features": [
                    "builder_badge", "collaboration_tools", "enhanced_marketplace_access",
                    "community_recognition", "builder_forum_access"
                ],
                "recognition_title": "PRSM Builder",
                "special_privileges": [
                    "Enhanced collaboration tools", "Builder community access",
                    "Marketplace fee reductions", "Priority customer support"
                ]
            },
            EarlyAdopterTier.MEMBER: {
                "max_users": 8000,
                "join_number_range": (2001, 10000),
                "ftns_multiplier": Decimal('1.5'),
                "governance_bonus": Decimal('0.5'),
                "priority_level": 2,
                "exclusive_features": [
                    "member_badge", "community_access", "standard_benefits",
                    "member_newsletter", "basic_analytics"
                ],
                "recognition_title": "PRSM Member",
                "special_privileges": [
                    "Community forum access", "Member-only events",
                    "Standard customer support", "Basic marketplace benefits"
                ]
            },
            EarlyAdopterTier.COMMUNITY: {
                "max_users": None,  # Unlimited
                "join_number_range": (10001, float('inf')),
                "ftns_multiplier": Decimal('1.0'),
                "governance_bonus": Decimal('0.0'),
                "priority_level": 1,
                "exclusive_features": [
                    "community_badge", "basic_access", "community_forum"
                ],
                "recognition_title": "PRSM Community Member",
                "special_privileges": [
                    "Community forum access", "Basic documentation access"
                ]
            }
        }
        
        # Program statistics
        self.program_stats = {
            "total_early_adopters": 0,
            "adopters_by_tier": {tier.value: 0 for tier in EarlyAdopterTier},
            "total_bonuses_distributed": Decimal('0'),
            "average_referral_rate": 0.0,
            "milestone_completion_rates": {},
            "community_growth_rate": 0.0,
            "program_launch_date": datetime.now(timezone.utc)
        }
        
        # Current join number tracker
        self.current_join_number = 1
        
        print("🌟 Early Adopter Program initialized")
        print(f"   - {len(self.tier_configs)} adopter tiers configured")
        print(f"   - Milestone and recognition system active")
    
    async def register_early_adopter(
        self,
        user_id: str,
        registration: EarlyAdopterRegistration
    ) -> EarlyAdopterProfile:
        """
        Register a user in the early adopter program
        
        Args:
            user_id: User to register
            registration: Registration details
            
        Returns:
            Early adopter profile
        """
        try:
            # Check if user is already registered
            if user_id in self.early_adopters:
                existing_profile = self.early_adopters[user_id]
                if existing_profile.is_active:
                    self.logger.info("User already registered as early adopter", user_id=user_id)
                    return existing_profile
            
            # Determine adopter tier based on join number
            join_number = self.current_join_number
            adopter_tier = self._determine_adopter_tier(join_number)
            tier_config = self.tier_configs[adopter_tier]
            
            # Check if tier is full
            if tier_config["max_users"] and join_number > tier_config["join_number_range"][1]:
                # Move to next tier
                adopter_tier = self._get_next_available_tier(join_number)
                tier_config = self.tier_configs[adopter_tier]
            
            # Create early adopter profile
            profile = EarlyAdopterProfile(
                user_id=user_id,
                adopter_tier=adopter_tier,
                join_number=join_number,
                ftns_bonus_multiplier=tier_config["ftns_multiplier"],
                governance_weight_bonus=tier_config["governance_bonus"],
                priority_access_level=tier_config["priority_level"],
                exclusive_features_access=tier_config["exclusive_features"].copy(),
                recognition_badges=[f"{adopter_tier.value}_badge"],
                special_privileges=tier_config["special_privileges"].copy()
            )
            
            # Store profile
            self.early_adopters[user_id] = profile
            self.current_join_number += 1
            
            # Award initial early adopter bonus
            initial_bonus = await self._calculate_initial_bonus(adopter_tier, registration)
            if initial_bonus > 0:
                await self._award_early_adopter_bonus(user_id, initial_bonus, "initial_early_adopter_bonus")
                profile.lifetime_bonus_earned += initial_bonus
            
            # Process referral if applicable
            if registration.referral_code:
                await self._process_referral(user_id, registration.referral_code, adopter_tier)
            
            # Activate governance participation with bonus
            if tier_config["governance_bonus"] > 0:
                governance_tier = self._map_to_governance_tier(adopter_tier)
                await self.governance_distributor.activate_governance_participation(
                    user_id=user_id,
                    participant_tier=governance_tier
                )
            
            # Update program statistics
            self.program_stats["total_early_adopters"] += 1
            self.program_stats["adopters_by_tier"][adopter_tier.value] += 1
            self.program_stats["total_bonuses_distributed"] += initial_bonus
            
            # Audit logging
            await audit_logger.log_security_event(
                event_type="early_adopter_registered",
                user_id=user_id,
                details={
                    "adopter_tier": adopter_tier.value,
                    "join_number": join_number,
                    "ftns_multiplier": str(tier_config["ftns_multiplier"]),
                    "initial_bonus": str(initial_bonus),
                    "research_interests": [interest.value for interest in registration.research_interests]
                },
                security_level="info"
            )
            
            self.logger.info(
                "Early adopter registered",
                user_id=user_id,
                tier=adopter_tier.value,
                join_number=join_number,
                initial_bonus=str(initial_bonus)
            )
            
            return profile
            
        except Exception as e:
            self.logger.error("Failed to register early adopter", user_id=user_id, error=str(e))
            raise
    
    async def track_milestone_achievement(
        self,
        user_id: str,
        milestone_type: str,
        milestone_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Track and reward milestone achievements for early adopters
        
        Args:
            user_id: User achieving milestone
            milestone_type: Type of milestone achieved
            milestone_data: Milestone achievement data
            
        Returns:
            Milestone tracking result with rewards
        """
        try:
            profile = self.early_adopters.get(user_id)
            if not profile or not profile.is_active:
                return {"error": "User not registered as early adopter"}
            
            # Check if milestone is defined
            if milestone_type not in self.adoption_milestones:
                return {"error": f"Unknown milestone type: {milestone_type}"}
            
            milestone_config = self.adoption_milestones[milestone_type]
            
            # Check if milestone already achieved
            if milestone_type in profile.milestone_achievements:
                return {"message": "Milestone already achieved", "rewards": {}}
            
            # Validate milestone achievement
            achievement_valid = await self._validate_milestone_achievement(
                user_id, milestone_type, milestone_data
            )
            
            if not achievement_valid:
                return {"error": "Milestone achievement validation failed"}
            
            # Calculate milestone rewards with early adopter multipliers
            base_reward = milestone_config["base_ftns_reward"]
            multiplied_reward = base_reward * profile.ftns_bonus_multiplier
            
            # Award milestone reward
            await self._award_early_adopter_bonus(
                user_id, multiplied_reward, f"milestone_{milestone_type}"
            )
            
            # Update profile
            profile.milestone_achievements.append(milestone_type)
            profile.lifetime_bonus_earned += multiplied_reward
            
            # Award any special badges
            milestone_badges = milestone_config.get("badges", [])
            for badge in milestone_badges:
                if badge not in profile.recognition_badges:
                    profile.recognition_badges.append(badge)
            
            # Check for tier graduation
            graduation_result = await self._check_tier_graduation(user_id, profile)
            
            # Update statistics
            self.program_stats["total_bonuses_distributed"] += multiplied_reward
            
            result = {
                "milestone_achieved": milestone_type,
                "ftns_reward": str(multiplied_reward),
                "badges_earned": milestone_badges,
                "total_lifetime_bonus": str(profile.lifetime_bonus_earned),
                "tier_graduation": graduation_result
            }
            
            self.logger.info(
                "Early adopter milestone achieved",
                user_id=user_id,
                milestone=milestone_type,
                reward=str(multiplied_reward),
                tier=profile.adopter_tier.value
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Failed to track milestone achievement", 
                            user_id=user_id, milestone=milestone_type, error=str(e))
            raise
    
    async def get_early_adopter_status(self, user_id: str) -> EarlyAdopterStatusResponse:
        """Get comprehensive early adopter status"""
        try:
            profile = self.early_adopters.get(user_id)
            if not profile:
                raise ValueError(f"User {user_id} not registered as early adopter")
            
            return EarlyAdopterStatusResponse(
                user_id=user_id,
                adopter_tier=profile.adopter_tier,
                join_number=profile.join_number,
                ftns_bonus_multiplier=profile.ftns_bonus_multiplier,
                governance_weight_bonus=profile.governance_weight_bonus,
                priority_access_level=profile.priority_access_level,
                lifetime_bonus_earned=profile.lifetime_bonus_earned,
                milestone_achievements=profile.milestone_achievements,
                recognition_badges=profile.recognition_badges
            )
            
        except Exception as e:
            self.logger.error("Failed to get early adopter status", user_id=user_id, error=str(e))
            raise
    
    async def get_program_statistics(self) -> Dict[str, Any]:
        """Get comprehensive program statistics"""
        try:
            # Calculate additional metrics
            active_adopters = len([p for p in self.early_adopters.values() if p.is_active])
            
            # Milestone completion rates
            milestone_stats = {}
            for milestone_type in self.adoption_milestones:
                completed_count = len([
                    p for p in self.early_adopters.values()
                    if milestone_type in p.milestone_achievements
                ])
                milestone_stats[milestone_type] = {
                    "completed_count": completed_count,
                    "completion_rate": completed_count / max(1, active_adopters)
                }
            
            # Average referral rate
            total_referrals = sum(p.referral_count for p in self.early_adopters.values())
            avg_referral_rate = total_referrals / max(1, active_adopters)
            
            # Growth rate calculation
            days_since_launch = (datetime.now(timezone.utc) - self.program_stats["program_launch_date"]).days
            daily_growth_rate = active_adopters / max(1, days_since_launch)
            
            return {
                **self.program_stats,
                "active_early_adopters": active_adopters,
                "milestone_completion_rates": milestone_stats,
                "average_referral_rate": avg_referral_rate,
                "daily_growth_rate": daily_growth_rate,
                "tier_distribution": self._calculate_tier_distribution(),
                "program_health_score": self._calculate_program_health_score(),
                "top_contributors": await self._get_top_contributors(),
                "recent_achievements": await self._get_recent_achievements()
            }
            
        except Exception as e:
            self.logger.error("Failed to get program statistics", error=str(e))
            return {"error": str(e)}
    
    async def generate_invitation_codes(
        self,
        count: int,
        tier_restriction: Optional[EarlyAdopterTier] = None,
        expiry_days: int = 30
    ) -> List[str]:
        """Generate invitation codes for the early adopter program"""
        try:
            invitation_codes = []
            
            for _ in range(count):
                code = f"PRSM-{uuid4().hex[:8].upper()}"
                
                # In production, this would store invitation codes in database
                # with expiry dates and tier restrictions
                invitation_codes.append(code)
            
            self.logger.info(
                "Invitation codes generated",
                count=count,
                tier_restriction=tier_restriction.value if tier_restriction else None,
                expiry_days=expiry_days
            )
            
            return invitation_codes
            
        except Exception as e:
            self.logger.error("Failed to generate invitation codes", error=str(e))
            raise
    
    # === Private Helper Methods ===
    
    def _initialize_adoption_milestones(self) -> Dict[str, Dict[str, Any]]:
        """Initialize adoption milestone configurations"""
        return {
            "first_model_upload": {
                "name": "First Model Upload",
                "description": "Upload your first AI model to PRSM",
                "base_ftns_reward": Decimal('500'),
                "badges": ["model_contributor"],
                "difficulty": "beginner",
                "category": "contribution"
            },
            "first_collaboration": {
                "name": "First Collaboration",
                "description": "Participate in your first research collaboration",
                "base_ftns_reward": Decimal('1000'),
                "badges": ["collaborator"],
                "difficulty": "intermediate",
                "category": "community"
            },
            "governance_participation": {
                "name": "Governance Participation",
                "description": "Cast your first governance vote",
                "base_ftns_reward": Decimal('300'),
                "badges": ["governance_participant"],
                "difficulty": "beginner",
                "category": "governance"
            },
            "research_publication": {
                "name": "Research Publication",
                "description": "Publish research findings using PRSM",
                "base_ftns_reward": Decimal('2000'),
                "badges": ["researcher", "published_author"],
                "difficulty": "advanced",
                "category": "research"
            },
            "community_builder": {
                "name": "Community Builder",
                "description": "Refer 10 new users to PRSM",
                "base_ftns_reward": Decimal('1500'),
                "badges": ["community_builder", "referral_champion"],
                "difficulty": "intermediate",
                "category": "growth"
            },
            "innovation_award": {
                "name": "Innovation Award",
                "description": "Create an innovative use case or improvement",
                "base_ftns_reward": Decimal('3000'),
                "badges": ["innovator", "thought_leader"],
                "difficulty": "expert",
                "category": "innovation"
            }
        }
    
    def _determine_adopter_tier(self, join_number: int) -> EarlyAdopterTier:
        """Determine adopter tier based on join number"""
        for tier, config in self.tier_configs.items():
            min_join, max_join = config["join_number_range"]
            if min_join <= join_number <= max_join:
                return tier
        
        return EarlyAdopterTier.COMMUNITY  # Default to community tier
    
    def _get_next_available_tier(self, join_number: int) -> EarlyAdopterTier:
        """Get next available tier if current is full"""
        all_tiers = list(EarlyAdopterTier)
        
        for tier in all_tiers:
            config = self.tier_configs[tier]
            if config["max_users"] is None or join_number <= config["join_number_range"][1]:
                return tier
        
        return EarlyAdopterTier.COMMUNITY
    
    async def _calculate_initial_bonus(
        self, 
        adopter_tier: EarlyAdopterTier, 
        registration: EarlyAdopterRegistration
    ) -> Decimal:
        """Calculate initial bonus for early adopter registration"""
        base_bonus = Decimal('1000')  # Base bonus for all early adopters
        tier_config = self.tier_configs[adopter_tier]
        
        # Apply tier multiplier
        tier_bonus = base_bonus * tier_config["ftns_multiplier"]
        
        # Additional bonuses
        bonus_multiplier = Decimal('1.0')
        
        # Research interests bonus
        if len(registration.research_interests) >= 3:
            bonus_multiplier += Decimal('0.2')  # 20% bonus for diverse interests
        
        # Experience level bonus
        experience_bonuses = {
            "expert": Decimal('0.5'),
            "advanced": Decimal('0.3'),
            "intermediate": Decimal('0.1'),
            "beginner": Decimal('0.0')
        }
        bonus_multiplier += experience_bonuses.get(registration.experience_level, Decimal('0.0'))
        
        return tier_bonus * bonus_multiplier
    
    async def _award_early_adopter_bonus(
        self, 
        user_id: str, 
        amount: Decimal, 
        bonus_type: str
    ):
        """Award FTNS bonus to early adopter"""
        if amount > 0:
            await self.ftns_service.create_transaction(
                from_user_id=None,  # System mint
                to_user_id=user_id,
                amount=amount,
                transaction_type="early_adopter_bonus",
                description=f"Early adopter bonus: {bonus_type}",
                reference_id=f"early_adopter_{bonus_type}_{user_id}"
            )
    
    async def _process_referral(
        self, 
        new_user_id: str, 
        referral_code: str, 
        new_user_tier: EarlyAdopterTier
    ):
        """Process referral bonus for existing user"""
        # Find referring user (simplified - in production would decode referral code)
        referrer_id = None  # Would decode from referral_code
        
        if referrer_id and referrer_id in self.early_adopters:
            referrer_profile = self.early_adopters[referrer_id]
            
            # Award referral bonus
            referral_bonus = Decimal('200') * referrer_profile.ftns_bonus_multiplier
            await self._award_early_adopter_bonus(referrer_id, referral_bonus, "referral_bonus")
            
            # Update referrer profile
            referrer_profile.referral_count += 1
            referrer_profile.lifetime_bonus_earned += referral_bonus
            
            self.logger.info(
                "Referral processed",
                referrer_id=referrer_id,
                new_user_id=new_user_id,
                bonus=str(referral_bonus)
            )
    
    def _map_to_governance_tier(self, adopter_tier: EarlyAdopterTier) -> GovernanceParticipantTier:
        """Map early adopter tier to governance tier"""
        mapping = {
            EarlyAdopterTier.PIONEER: GovernanceParticipantTier.EXPERT,
            EarlyAdopterTier.EXPLORER: GovernanceParticipantTier.CONTRIBUTOR,
            EarlyAdopterTier.BUILDER: GovernanceParticipantTier.CONTRIBUTOR,
            EarlyAdopterTier.MEMBER: GovernanceParticipantTier.COMMUNITY,
            EarlyAdopterTier.COMMUNITY: GovernanceParticipantTier.COMMUNITY
        }
        return mapping.get(adopter_tier, GovernanceParticipantTier.COMMUNITY)
    
    async def _validate_milestone_achievement(
        self, 
        user_id: str, 
        milestone_type: str, 
        milestone_data: Dict[str, Any]
    ) -> bool:
        """Validate milestone achievement"""
        # Simplified validation - in production would have comprehensive checks
        if milestone_type == "first_model_upload":
            return milestone_data.get("model_uploaded", False)
        elif milestone_type == "governance_participation":
            return milestone_data.get("vote_cast", False)
        elif milestone_type == "first_collaboration":
            return milestone_data.get("collaboration_started", False)
        
        return True  # Default to valid for other milestones
    
    async def _check_tier_graduation(
        self, 
        user_id: str, 
        profile: EarlyAdopterProfile
    ) -> Dict[str, Any]:
        """Check if user qualifies for tier graduation"""
        # Simplified graduation logic
        # In production would have comprehensive criteria
        
        milestone_count = len(profile.milestone_achievements)
        current_tier_index = list(EarlyAdopterTier).index(profile.adopter_tier)
        
        # Example graduation criteria
        graduation_thresholds = {
            EarlyAdopterTier.COMMUNITY: 2,  # 2 milestones to graduate from community
            EarlyAdopterTier.MEMBER: 4,     # 4 milestones for next tier
            EarlyAdopterTier.BUILDER: 6,    # 6 milestones for next tier
            EarlyAdopterTier.EXPLORER: 8    # 8 milestones for pioneer
        }
        
        threshold = graduation_thresholds.get(profile.adopter_tier, float('inf'))
        
        if milestone_count >= threshold and current_tier_index > 0:
            new_tier = list(EarlyAdopterTier)[current_tier_index - 1]
            return {
                "graduated": True,
                "new_tier": new_tier.value,
                "graduation_bonus": "1000 FTNS"
            }
        
        return {"graduated": False}
    
    def _calculate_tier_distribution(self) -> Dict[str, Dict[str, Any]]:
        """Calculate distribution of users across tiers"""
        distribution = {}
        total_adopters = len(self.early_adopters)
        
        for tier in EarlyAdopterTier:
            count = len([p for p in self.early_adopters.values() if p.adopter_tier == tier])
            percentage = (count / total_adopters * 100) if total_adopters > 0 else 0
            
            distribution[tier.value] = {
                "count": count,
                "percentage": percentage,
                "max_capacity": self.tier_configs[tier]["max_users"],
                "capacity_utilized": (count / self.tier_configs[tier]["max_users"] * 100) 
                                   if self.tier_configs[tier]["max_users"] else 0
            }
        
        return distribution
    
    def _calculate_program_health_score(self) -> float:
        """Calculate overall program health score"""
        # Simplified health calculation
        total_adopters = len(self.early_adopters)
        active_adopters = len([p for p in self.early_adopters.values() if p.is_active])
        
        if total_adopters == 0:
            return 0.0
        
        # Factors: activity rate, tier distribution, milestone completion
        activity_score = active_adopters / total_adopters
        
        # Tier distribution score (favor higher tiers)
        tier_score = 0.0
        for tier, profile_list in [(t, [p for p in self.early_adopters.values() if p.adopter_tier == t]) for t in EarlyAdopterTier]:
            tier_weight = {
                EarlyAdopterTier.PIONEER: 5.0,
                EarlyAdopterTier.EXPLORER: 4.0,
                EarlyAdopterTier.BUILDER: 3.0,
                EarlyAdopterTier.MEMBER: 2.0,
                EarlyAdopterTier.COMMUNITY: 1.0
            }[tier]
            tier_score += len(profile_list) * tier_weight
        
        tier_score = tier_score / (total_adopters * 5.0)  # Normalize
        
        # Combined health score
        health_score = (activity_score + tier_score) / 2.0
        return min(1.0, health_score)
    
    async def _get_top_contributors(self) -> List[Dict[str, Any]]:
        """Get top contributing early adopters"""
        contributors = []
        
        for user_id, profile in self.early_adopters.items():
            if profile.is_active:
                score = (
                    len(profile.milestone_achievements) * 10 +
                    profile.referral_count * 5 +
                    float(profile.lifetime_bonus_earned) / 1000
                )
                
                contributors.append({
                    "user_id": user_id,
                    "tier": profile.adopter_tier.value,
                    "milestones": len(profile.milestone_achievements),
                    "referrals": profile.referral_count,
                    "lifetime_bonus": str(profile.lifetime_bonus_earned),
                    "contribution_score": score
                })
        
        # Sort by contribution score and return top 10
        contributors.sort(key=lambda x: x["contribution_score"], reverse=True)
        return contributors[:10]
    
    async def _get_recent_achievements(self) -> List[Dict[str, Any]]:
        """Get recent milestone achievements"""
        # In production, this would query recent achievements from database
        # For now, return empty list
        return []


# Global early adopter program instance
_early_adopter_program_instance: Optional[EarlyAdopterProgram] = None

def get_early_adopter_program() -> EarlyAdopterProgram:
    """Get or create the global early adopter program instance"""
    global _early_adopter_program_instance
    if _early_adopter_program_instance is None:
        _early_adopter_program_instance = EarlyAdopterProgram()
    return _early_adopter_program_instance