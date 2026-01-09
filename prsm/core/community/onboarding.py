"""
Community Onboarding Service
============================

Comprehensive onboarding system for new PRSM users with staged progression,
personalized guidance, and integration with early adopter programs.
"""

import asyncio
import structlog
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from ..core.config import settings
from ..auth.auth_manager import auth_manager
from ..tokenomics.database_ftns_service import DatabaseFTNSService
from ..governance.token_distribution import get_governance_distributor, GovernanceParticipantTier
from ..integrations.security.audit_logger import audit_logger
from .models import (
    OnboardingStage, OnboardingProgress, OnboardingProgressResponse,
    OnboardingStageUpdate, UserInterest, WelcomePackage
)

logger = structlog.get_logger(__name__)


class CommunityOnboardingService:
    """
    Manages the complete user onboarding journey from registration to
    full community participation with personalized guidance and rewards.
    """
    
    def __init__(self):
        self.service_id = str(uuid4())
        self.logger = logger.bind(component="community_onboarding", service_id=self.service_id)
        
        # Service integrations
        self.ftns_service = DatabaseFTNSService()
        self.governance_distributor = get_governance_distributor()
        
        # Onboarding tracking
        self.active_onboardings: Dict[str, OnboardingProgress] = {}
        self.onboarding_templates = self._initialize_onboarding_templates()
        
        # Stage configurations
        self.stage_configs = {
            OnboardingStage.REGISTRATION: {
                "title": "Welcome to PRSM",
                "description": "Complete your account registration",
                "estimated_time_minutes": 5,
                "required_actions": ["email_verification", "terms_acceptance"],
                "rewards": {"ftns": Decimal('100'), "badges": ["early_bird"]}
            },
            OnboardingStage.EMAIL_VERIFICATION: {
                "title": "Verify Your Email",
                "description": "Confirm your email address to secure your account",
                "estimated_time_minutes": 2,
                "required_actions": ["email_confirmation"],
                "rewards": {"ftns": Decimal('50'), "badges": ["verified_user"]}
            },
            OnboardingStage.PROFILE_SETUP: {
                "title": "Set Up Your Profile",
                "description": "Tell us about yourself and your research interests",
                "estimated_time_minutes": 10,
                "required_actions": ["basic_info", "avatar_upload"],
                "rewards": {"ftns": Decimal('200'), "badges": ["profile_complete"]}
            },
            OnboardingStage.INTERESTS_SELECTION: {
                "title": "Research Interests",
                "description": "Select your areas of research and contribution",
                "estimated_time_minutes": 5,
                "required_actions": ["interests_selection", "expertise_level"],
                "rewards": {"ftns": Decimal('150'), "badges": ["research_focused"]}
            },
            OnboardingStage.INITIAL_ALLOCATION: {
                "title": "Token Allocation",
                "description": "Receive your initial FTNS tokens and learn the economy",
                "estimated_time_minutes": 8,
                "required_actions": ["wallet_setup", "allocation_acceptance"],
                "rewards": {"ftns": Decimal('1000'), "badges": ["token_holder"]}
            },
            OnboardingStage.FIRST_INTERACTION: {
                "title": "First Interaction",
                "description": "Make your first query or contribution to PRSM",
                "estimated_time_minutes": 15,
                "required_actions": ["first_query", "result_feedback"],
                "rewards": {"ftns": Decimal('500'), "badges": ["first_steps"]}
            },
            OnboardingStage.TUTORIAL_COMPLETION: {
                "title": "Tutorial Mastery",
                "description": "Complete the interactive PRSM tutorial",
                "estimated_time_minutes": 30,
                "required_actions": ["tutorial_modules", "quiz_completion"],
                "rewards": {"ftns": Decimal('800'), "badges": ["tutorial_master"]}
            },
            OnboardingStage.COMMUNITY_INTRODUCTION: {
                "title": "Community Introduction",
                "description": "Introduce yourself to the community",
                "estimated_time_minutes": 10,
                "required_actions": ["introduction_post", "community_interaction"],
                "rewards": {"ftns": Decimal('300'), "badges": ["community_member"]}
            }
        }
        
        # Welcome packages by user type
        self.welcome_packages = self._initialize_welcome_packages()
        
        # Statistics tracking
        self.onboarding_stats = {
            "total_started": 0,
            "total_completed": 0,
            "average_completion_time_hours": 0.0,
            "stage_completion_rates": {},
            "drop_off_points": {},
            "user_satisfaction_scores": []
        }
        
        print("🚀 Community Onboarding Service initialized")
        print(f"   - {len(self.stage_configs)} onboarding stages configured")
        print(f"   - Personalized guidance and rewards active")
    
    async def start_onboarding(
        self,
        user_id: str,
        invitation_code: Optional[str] = None,
        referral_user_id: Optional[str] = None,
        user_type: str = "researcher"
    ) -> Tuple[OnboardingProgress, WelcomePackage]:
        """
        Start the onboarding process for a new user
        
        Args:
            user_id: User starting onboarding
            invitation_code: Optional invitation code
            referral_user_id: Optional referring user
            user_type: Type of user (researcher, developer, institution)
            
        Returns:
            Tuple of onboarding progress and welcome package
        """
        try:
            # Check if user already has onboarding in progress
            existing_progress = await self._get_onboarding_progress(user_id)
            if existing_progress and not existing_progress.is_completed:
                self.logger.info("Resuming existing onboarding", user_id=user_id)
                welcome_package = self.welcome_packages.get(user_type, self.welcome_packages["researcher"])
                return existing_progress, welcome_package
            
            # Create new onboarding progress
            onboarding = OnboardingProgress(
                user_id=user_id,
                current_stage=OnboardingStage.REGISTRATION,
                completed_stages=[],
                stage_completion_times={},
                invitation_code=invitation_code,
                referral_user_id=referral_user_id,
                onboarding_version="v1.0"
            )
            
            # Store in tracking
            self.active_onboardings[user_id] = onboarding
            
            # Generate personalized welcome package
            welcome_package = await self._generate_welcome_package(user_id, user_type, invitation_code)
            
            # Award initial welcome bonus
            await self._award_welcome_bonus(user_id, welcome_package.initial_ftns_grant)
            
            # Update statistics
            self.onboarding_stats["total_started"] += 1
            
            # Audit logging
            await audit_logger.log_security_event(
                event_type="onboarding_started",
                user_id=user_id,
                details={
                    "user_type": user_type,
                    "invitation_code": invitation_code,
                    "referral_user_id": referral_user_id,
                    "welcome_bonus": str(welcome_package.initial_ftns_grant)
                },
                security_level="info"
            )
            
            self.logger.info(
                "Onboarding started",
                user_id=user_id,
                user_type=user_type,
                welcome_bonus=str(welcome_package.initial_ftns_grant)
            )
            
            return onboarding, welcome_package
            
        except Exception as e:
            self.logger.error("Failed to start onboarding", user_id=user_id, error=str(e))
            raise
    
    async def complete_stage(
        self,
        user_id: str,
        stage_update: OnboardingStageUpdate
    ) -> OnboardingProgressResponse:
        """
        Complete an onboarding stage and advance to the next
        
        Args:
            user_id: User completing the stage
            stage_update: Stage completion data
            
        Returns:
            Updated onboarding progress
        """
        try:
            # Get current onboarding progress
            progress = await self._get_onboarding_progress(user_id)
            if not progress:
                raise ValueError(f"No onboarding found for user {user_id}")
            
            if progress.is_completed:
                raise ValueError("Onboarding already completed")
            
            # Validate stage completion
            stage_config = self.stage_configs[stage_update.stage]
            completion_valid = await self._validate_stage_completion(
                user_id, stage_update.stage, stage_update.completion_data
            )
            
            if not completion_valid:
                raise ValueError(f"Stage {stage_update.stage} completion validation failed")
            
            # Mark stage as completed
            if stage_update.stage not in progress.completed_stages:
                progress.completed_stages.append(stage_update.stage)
                progress.stage_completion_times[stage_update.stage.value] = datetime.now(timezone.utc).isoformat()
            
            # Process user inputs
            await self._process_stage_inputs(user_id, stage_update.stage, stage_update.user_inputs)
            
            # Award stage rewards
            await self._award_stage_rewards(user_id, stage_update.stage, stage_config["rewards"])
            
            # Determine next stage
            next_stage = await self._determine_next_stage(progress)
            
            if next_stage:
                progress.current_stage = next_stage
                progress.last_activity = datetime.now(timezone.utc)
            else:
                # Onboarding completed
                await self._complete_onboarding(user_id, progress)
            
            # Update tracking
            self.active_onboardings[user_id] = progress
            
            # Generate progress response
            response = await self._generate_progress_response(user_id, progress)
            
            self.logger.info(
                "Onboarding stage completed",
                user_id=user_id,
                completed_stage=stage_update.stage.value,
                next_stage=next_stage.value if next_stage else "completed"
            )
            
            return response
            
        except Exception as e:
            self.logger.error("Failed to complete onboarding stage", 
                            user_id=user_id, stage=stage_update.stage, error=str(e))
            raise
    
    async def get_onboarding_status(self, user_id: str) -> OnboardingProgressResponse:
        """Get current onboarding status for a user"""
        try:
            progress = await self._get_onboarding_progress(user_id)
            if not progress:
                raise ValueError(f"No onboarding found for user {user_id}")
            
            return await self._generate_progress_response(user_id, progress)
            
        except Exception as e:
            self.logger.error("Failed to get onboarding status", user_id=user_id, error=str(e))
            raise
    
    async def get_personalized_guidance(self, user_id: str) -> Dict[str, Any]:
        """Get personalized guidance for the user's current stage"""
        try:
            progress = await self._get_onboarding_progress(user_id)
            if not progress or progress.is_completed:
                return {"message": "Onboarding completed", "guidance": []}
            
            current_stage_config = self.stage_configs[progress.current_stage]
            
            # Generate personalized guidance based on user profile
            user_profile = await self._get_user_profile(user_id)
            guidance = await self._generate_personalized_guidance(
                progress.current_stage, user_profile, progress
            )
            
            return {
                "current_stage": progress.current_stage.value,
                "stage_info": current_stage_config,
                "personalized_guidance": guidance,
                "completion_percentage": len(progress.completed_stages) / len(self.stage_configs) * 100,
                "estimated_time_remaining": await self._estimate_time_remaining(progress)
            }
            
        except Exception as e:
            self.logger.error("Failed to get personalized guidance", user_id=user_id, error=str(e))
            raise
    
    async def get_onboarding_statistics(self) -> Dict[str, Any]:
        """Get comprehensive onboarding statistics"""
        try:
            # Calculate completion rates by stage
            stage_stats = {}
            for stage in OnboardingStage:
                completed_count = sum(
                    1 for progress in self.active_onboardings.values()
                    if stage in progress.completed_stages
                )
                stage_stats[stage.value] = {
                    "completed_count": completed_count,
                    "completion_rate": completed_count / max(1, self.onboarding_stats["total_started"])
                }
            
            # Calculate average completion times
            completed_onboardings = [
                progress for progress in self.active_onboardings.values()
                if progress.is_completed and progress.total_completion_time_hours
            ]
            
            avg_completion_time = 0.0
            if completed_onboardings:
                avg_completion_time = sum(
                    float(progress.total_completion_time_hours) 
                    for progress in completed_onboardings
                ) / len(completed_onboardings)
            
            return {
                **self.onboarding_stats,
                "stage_statistics": stage_stats,
                "average_completion_time_hours": avg_completion_time,
                "total_completion_rate": self.onboarding_stats["total_completed"] / max(1, self.onboarding_stats["total_started"]),
                "active_onboardings": len([p for p in self.active_onboardings.values() if not p.is_completed]),
                "user_distribution_by_stage": self._calculate_stage_distribution()
            }
            
        except Exception as e:
            self.logger.error("Failed to get onboarding statistics", error=str(e))
            return {"error": str(e)}
    
    # === Private Helper Methods ===
    
    def _initialize_onboarding_templates(self) -> Dict[str, Any]:
        """Initialize onboarding templates for different user types"""
        return {
            "researcher": {
                "welcome_message": "Welcome to PRSM! Join thousands of researchers accelerating scientific discovery through collaborative AI.",
                "focus_areas": ["model_training", "data_sharing", "research_collaboration"],
                "recommended_first_steps": ["upload_model", "explore_marketplace", "join_research_group"]
            },
            "developer": {
                "welcome_message": "Welcome to PRSM! Build the future of decentralized AI with our comprehensive developer tools.",
                "focus_areas": ["api_integration", "model_development", "infrastructure_contribution"],
                "recommended_first_steps": ["api_setup", "sdk_installation", "first_deployment"]
            },
            "institution": {
                "welcome_message": "Welcome to PRSM! Transform your institution's AI capabilities through our enterprise platform.",
                "focus_areas": ["institutional_integration", "governance_participation", "strategic_partnerships"],
                "recommended_first_steps": ["institutional_setup", "team_onboarding", "governance_activation"]
            }
        }
    
    def _initialize_welcome_packages(self) -> Dict[str, WelcomePackage]:
        """Initialize welcome packages for different user types"""
        return {
            "researcher": WelcomePackage(
                welcome_message="Welcome to the future of collaborative AI research!",
                initial_ftns_grant=Decimal('2000'),
                recommended_tutorials=[
                    {"title": "Getting Started with PRSM", "url": "/tutorials/getting-started"},
                    {"title": "Training Your First Model", "url": "/tutorials/model-training"},
                    {"title": "Research Collaboration", "url": "/tutorials/collaboration"}
                ],
                community_links={
                    "research_forum": "/community/research",
                    "collaboration_board": "/community/collaborations",
                    "help_center": "/help"
                },
                getting_started_guide="/guides/researcher-quickstart",
                early_adopter_benefits=[
                    "5x FTNS bonus for early contributions",
                    "Priority access to new features",
                    "Direct feedback channel to development team",
                    "Exclusive research collaboration opportunities"
                ],
                milestone_roadmap=[
                    {"milestone": "First Model Upload", "reward": "500 FTNS", "difficulty": "beginner"},
                    {"milestone": "Research Paper Publication", "reward": "2000 FTNS", "difficulty": "intermediate"},
                    {"milestone": "Community Collaboration", "reward": "1000 FTNS", "difficulty": "intermediate"}
                ]
            ),
            "developer": WelcomePackage(
                welcome_message="Welcome to the developer-friendly AI platform!",
                initial_ftns_grant=Decimal('1500'),
                recommended_tutorials=[
                    {"title": "API Integration Guide", "url": "/tutorials/api-integration"},
                    {"title": "SDK Development", "url": "/tutorials/sdk-development"},
                    {"title": "Model Deployment", "url": "/tutorials/deployment"}
                ],
                community_links={
                    "developer_forum": "/community/developers",
                    "technical_docs": "/docs",
                    "github_repo": "https://github.com/PRSM-AI/PRSM"
                },
                getting_started_guide="/guides/developer-quickstart",
                early_adopter_benefits=[
                    "3x FTNS bonus for infrastructure contributions",
                    "Early access to new APIs",
                    "Technical advisory board participation",
                    "Open source contribution rewards"
                ],
                milestone_roadmap=[
                    {"milestone": "First API Integration", "reward": "300 FTNS", "difficulty": "beginner"},
                    {"milestone": "SDK Contribution", "reward": "1500 FTNS", "difficulty": "advanced"},
                    {"milestone": "Infrastructure Improvement", "reward": "2500 FTNS", "difficulty": "expert"}
                ]
            ),
            "institution": WelcomePackage(
                welcome_message="Welcome to enterprise-grade AI collaboration!",
                initial_ftns_grant=Decimal('10000'),
                recommended_tutorials=[
                    {"title": "Institutional Setup", "url": "/tutorials/institutional-setup"},
                    {"title": "Team Management", "url": "/tutorials/team-management"},
                    {"title": "Governance Participation", "url": "/tutorials/governance"}
                ],
                community_links={
                    "institutional_forum": "/community/institutions",
                    "partnership_portal": "/partnerships",
                    "governance_hub": "/governance"
                },
                getting_started_guide="/guides/institutional-quickstart",
                early_adopter_benefits=[
                    "Strategic partnership opportunities",
                    "Governance voting weight bonuses",
                    "Dedicated enterprise support",
                    "Custom integration assistance"
                ],
                milestone_roadmap=[
                    {"milestone": "Team Onboarding", "reward": "5000 FTNS", "difficulty": "intermediate"},
                    {"milestone": "Governance Participation", "reward": "10000 FTNS", "difficulty": "intermediate"},
                    {"milestone": "Strategic Partnership", "reward": "25000 FTNS", "difficulty": "advanced"}
                ]
            )
        }
    
    async def _get_onboarding_progress(self, user_id: str) -> Optional[OnboardingProgress]:
        """Get onboarding progress for a user"""
        # Check in-memory cache first
        if user_id in self.active_onboardings:
            return self.active_onboardings[user_id]
        
        # In production, this would query the database
        # For now, return None if not in cache
        return None
    
    async def _validate_stage_completion(
        self, 
        user_id: str, 
        stage: OnboardingStage, 
        completion_data: Dict[str, Any]
    ) -> bool:
        """Validate that a stage has been properly completed"""
        stage_config = self.stage_configs[stage]
        required_actions = stage_config["required_actions"]
        
        # Check that all required actions have been completed
        for action in required_actions:
            if action not in completion_data or not completion_data[action]:
                return False
        
        # Stage-specific validation
        if stage == OnboardingStage.EMAIL_VERIFICATION:
            return completion_data.get("email_verified", False)
        elif stage == OnboardingStage.INTERESTS_SELECTION:
            interests = completion_data.get("research_interests", [])
            return len(interests) >= 1
        elif stage == OnboardingStage.FIRST_INTERACTION:
            return completion_data.get("query_completed", False)
        
        return True
    
    async def _process_stage_inputs(
        self, 
        user_id: str, 
        stage: OnboardingStage, 
        user_inputs: Dict[str, Any]
    ):
        """Process user inputs from stage completion"""
        progress = self.active_onboardings[user_id]
        
        if stage == OnboardingStage.INTERESTS_SELECTION:
            progress.research_interests = user_inputs.get("research_interests", [])
            progress.experience_level = user_inputs.get("experience_level", "beginner")
        elif stage == OnboardingStage.PROFILE_SETUP:
            progress.institution_affiliation = user_inputs.get("institution_affiliation")
            progress.goals = user_inputs.get("goals")
    
    async def _award_stage_rewards(
        self, 
        user_id: str, 
        stage: OnboardingStage, 
        rewards: Dict[str, Any]
    ):
        """Award rewards for completing a stage"""
        if "ftns" in rewards and rewards["ftns"] > 0:
            await self.ftns_service.create_transaction(
                from_user_id=None,  # System mint
                to_user_id=user_id,
                amount=rewards["ftns"],
                transaction_type="onboarding_reward",
                description=f"Onboarding stage completion: {stage.value}",
                reference_id=f"onboarding_{stage.value}_{user_id}"
            )
        
        # Record badge awards (would integrate with badge system)
        if "badges" in rewards:
            self.logger.info("Badges awarded", user_id=user_id, stage=stage.value, badges=rewards["badges"])
    
    async def _award_welcome_bonus(self, user_id: str, amount: Decimal):
        """Award welcome bonus to new user"""
        if amount > 0:
            await self.ftns_service.create_transaction(
                from_user_id=None,  # System mint
                to_user_id=user_id,
                amount=amount,
                transaction_type="welcome_bonus",
                description="Welcome bonus for joining PRSM",
                reference_id=f"welcome_{user_id}"
            )
    
    async def _determine_next_stage(self, progress: OnboardingProgress) -> Optional[OnboardingStage]:
        """Determine the next onboarding stage"""
        all_stages = list(OnboardingStage)
        current_index = all_stages.index(progress.current_stage)
        
        if current_index < len(all_stages) - 1:
            return all_stages[current_index + 1]
        else:
            return None  # Onboarding complete
    
    async def _complete_onboarding(self, user_id: str, progress: OnboardingProgress):
        """Complete the onboarding process"""
        progress.is_completed = True
        progress.completion_date = datetime.now(timezone.utc)
        
        # Calculate total completion time
        if progress.stage_completion_times:
            start_time = min(
                datetime.fromisoformat(time_str.replace('Z', '+00:00')) 
                for time_str in progress.stage_completion_times.values()
            )
            completion_time = progress.completion_date - start_time
            progress.total_completion_time_hours = completion_time.total_seconds() / 3600
        
        # Award completion bonus
        completion_bonus = Decimal('1000')
        await self.ftns_service.create_transaction(
            from_user_id=None,
            to_user_id=user_id,
            amount=completion_bonus,
            transaction_type="onboarding_completion",
            description="Onboarding completion bonus",
            reference_id=f"completion_{user_id}"
        )
        
        # Activate governance participation
        await self.governance_distributor.activate_governance_participation(
            user_id=user_id,
            participant_tier=GovernanceParticipantTier.COMMUNITY
        )
        
        # Update statistics
        self.onboarding_stats["total_completed"] += 1
        
        self.logger.info("Onboarding completed", user_id=user_id, 
                        completion_time_hours=progress.total_completion_time_hours)
    
    async def _generate_progress_response(
        self, 
        user_id: str, 
        progress: OnboardingProgress
    ) -> OnboardingProgressResponse:
        """Generate onboarding progress response"""
        completion_percentage = len(progress.completed_stages) / len(self.stage_configs) * 100
        
        # Generate next steps
        next_steps = []
        if not progress.is_completed:
            current_config = self.stage_configs[progress.current_stage]
            next_steps = [
                f"Complete {action.replace('_', ' ')}" 
                for action in current_config["required_actions"]
            ]
        
        # Estimate time remaining
        estimated_time_remaining = None
        if not progress.is_completed:
            remaining_stages = len(self.stage_configs) - len(progress.completed_stages)
            avg_time_per_stage = 15  # minutes
            estimated_time_remaining = (remaining_stages * avg_time_per_stage) / 60  # hours
        
        return OnboardingProgressResponse(
            user_id=user_id,
            current_stage=progress.current_stage,
            completed_stages=progress.completed_stages,
            completion_percentage=completion_percentage,
            estimated_time_remaining_hours=estimated_time_remaining,
            next_steps=next_steps
        )
    
    async def _generate_welcome_package(
        self, 
        user_id: str, 
        user_type: str, 
        invitation_code: Optional[str]
    ) -> WelcomePackage:
        """Generate personalized welcome package"""
        base_package = self.welcome_packages.get(user_type, self.welcome_packages["researcher"])
        
        # Adjust initial grant based on invitation code or early adopter status
        initial_grant = base_package.initial_ftns_grant
        if invitation_code:
            initial_grant *= Decimal('1.5')  # 50% bonus for invited users
        
        return WelcomePackage(
            welcome_message=base_package.welcome_message,
            initial_ftns_grant=initial_grant,
            recommended_tutorials=base_package.recommended_tutorials,
            community_links=base_package.community_links,
            getting_started_guide=base_package.getting_started_guide,
            early_adopter_benefits=base_package.early_adopter_benefits,
            milestone_roadmap=base_package.milestone_roadmap
        )
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile information"""
        # In production, this would query user profile from auth system
        return {"user_id": user_id, "type": "researcher"}
    
    async def _generate_personalized_guidance(
        self, 
        current_stage: OnboardingStage, 
        user_profile: Dict[str, Any], 
        progress: OnboardingProgress
    ) -> List[str]:
        """Generate personalized guidance for current stage"""
        base_guidance = [
            f"Complete the {current_stage.value.replace('_', ' ')} stage",
            "Follow the step-by-step instructions",
            "Ask for help in the community forum if needed"
        ]
        
        # Add personalized tips based on user profile and progress
        if current_stage == OnboardingStage.INTERESTS_SELECTION:
            base_guidance.append("Select at least 3 research interests for better matching")
        elif current_stage == OnboardingStage.FIRST_INTERACTION:
            base_guidance.append("Try asking about your research domain for personalized results")
        
        return base_guidance
    
    async def _estimate_time_remaining(self, progress: OnboardingProgress) -> float:
        """Estimate time remaining for onboarding completion"""
        if progress.is_completed:
            return 0.0
        
        remaining_stages = []
        all_stages = list(OnboardingStage)
        current_index = all_stages.index(progress.current_stage)
        
        for i in range(current_index, len(all_stages)):
            stage = all_stages[i]
            if stage not in progress.completed_stages:
                remaining_stages.append(stage)
        
        total_time_minutes = sum(
            self.stage_configs[stage]["estimated_time_minutes"] 
            for stage in remaining_stages
        )
        
        return total_time_minutes / 60  # Convert to hours
    
    def _calculate_stage_distribution(self) -> Dict[str, int]:
        """Calculate distribution of users across onboarding stages"""
        distribution = {}
        for stage in OnboardingStage:
            distribution[stage.value] = len([
                p for p in self.active_onboardings.values()
                if p.current_stage == stage and not p.is_completed
            ])
        return distribution


# Global onboarding service instance
_onboarding_service_instance: Optional[CommunityOnboardingService] = None

def get_onboarding_service() -> CommunityOnboardingService:
    """Get or create the global onboarding service instance"""
    global _onboarding_service_instance
    if _onboarding_service_instance is None:
        _onboarding_service_instance = CommunityOnboardingService()
    return _onboarding_service_instance