"""
PRSM RLVR (Reinforcement Learning with Verifiable Rewards) Engine
Enhanced with SEAL (Self-Adapting Language Models) Integration

Implements verifiable reward calculation and teacher weight optimization
for the distilled teacher model system, now enhanced with SEAL's 
self-adapting capabilities for autonomous improvement.

🎯 SEAL INTEGRATION:
- Self-edit reward tracking for RL optimization
- Adaptive reward calculation based on self-generated training data
- Meta-learning rewards for discovering optimal teaching strategies
- Reinforcement learning for self-edit policy optimization

Based on execution_plan.md Week 7-8 requirements + MIT SEAL paper integration.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
import structlog
import math

from pydantic import Field
from prsm.core.models import TeacherModel, LearningSession, PRSMBaseModel
from .teacher_model import TeachingOutcome, TeachingStrategy

logger = structlog.get_logger()


class VerifiableReward(PRSMBaseModel):
    """
    Cryptographically verifiable reward for teacher performance
    """
    reward_id: UUID = Field(default_factory=uuid4)
    teacher_id: UUID
    session_id: UUID
    curriculum_id: Optional[UUID] = None
    base_reward: float
    performance_multiplier: float
    innovation_bonus: float
    consistency_factor: float
    total_reward: float
    verification_hash: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verified: bool = False
    
    
class TeacherWeights(PRSMBaseModel):
    """
    Neural network-style weights for teacher optimization
    """
    teacher_id: UUID
    curriculum_weights: Dict[str, float] = {}
    strategy_weights: Dict[str, float] = {}
    domain_expertise: Dict[str, float] = {}
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    

class RewardCalculation(PRSMBaseModel):
    """
    Detailed breakdown of reward calculation
    """
    base_metrics: Dict[str, float]
    performance_factors: Dict[str, float]
    innovation_indicators: Dict[str, float]
    verification_proofs: Dict[str, str]
    confidence_interval: Tuple[float, float]
    calculation_method: str


class SEALEditReward(PRSMBaseModel):
    """
    SEAL-specific reward for self-edit performance
    
    🎯 SEAL REWARD COMPONENTS:
    - Downstream performance improvement from self-edit application
    - Synthetic data quality metrics
    - Optimization parameter effectiveness
    - Meta-learning contribution to teacher improvement
    """
    edit_id: UUID
    teacher_id: UUID
    edit_type: str  # knowledge_incorporation, curriculum_optimization, student_adaptation
    downstream_improvement: float  # Performance gain from applying this edit
    data_quality_score: float  # Quality of generated synthetic data
    parameter_effectiveness: float  # How well optimization params worked
    meta_learning_contribution: float  # Contribution to overall teacher improvement
    baseline_performance: Dict[str, float]
    updated_performance: Dict[str, float]
    reward_signal: float  # Final reward for RL training
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verified: bool = False


class SEALPolicyState(PRSMBaseModel):
    """
    Current state of SEAL self-edit generation policy
    
    🧠 POLICY TRACKING:
    - Success rates for different edit types
    - Optimal parameter distributions
    - Adaptation patterns and preferences
    - Performance trends over time
    """
    teacher_id: UUID
    edit_type_success_rates: Dict[str, float] = {}
    optimal_parameter_ranges: Dict[str, Tuple[float, float]] = {}
    successful_format_preferences: Dict[str, int] = {}
    adaptation_patterns: Dict[str, Any] = {}
    total_edits_generated: int = 0
    successful_edits: int = 0
    average_reward: float = 0.0
    improvement_rate: float = 0.0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RLVREngine:
    """
    Reinforcement Learning with Verifiable Rewards Engine
    Enhanced with SEAL Self-Adapting Capabilities
    
    Implements cryptographically verifiable reward calculation and
    adaptive teacher weight optimization based on teaching effectiveness,
    now enhanced with SEAL's self-adapting learning mechanisms.
    
    🎯 SEAL ENHANCEMENTS:
    - Self-edit reward tracking and optimization
    - Adaptive policy state management
    - Meta-learning reward calculation
    - ReSTEM-style policy updates
    """
    
    def __init__(self):
        self.logger = logger.bind(component="rlvr_engine")
        self.teacher_weights: Dict[UUID, TeacherWeights] = {}
        self.reward_history: List[VerifiableReward] = []
        self.verification_threshold = 0.95
        
        # SEAL-specific components
        self.seal_edit_rewards: List[SEALEditReward] = []
        self.seal_policy_states: Dict[UUID, SEALPolicyState] = {}
        self.edit_performance_baselines: Dict[str, Dict[str, float]] = {}
        self.meta_learning_history: List[Dict[str, Any]] = []
        
    async def calculate_verifiable_reward(self, teaching_outcome: TeachingOutcome) -> VerifiableReward:
        """
        Calculate cryptographically verifiable reward for teaching performance.
        
        Args:
            teaching_outcome: Results from a completed teaching session
            
        Returns:
            Verifiable reward with cryptographic proof
        """
        self.logger.info(
            "Calculating verifiable reward",
            session_id=str(teaching_outcome.session_id),
            teacher_id=str(teaching_outcome.teacher_id)
        )
        
        # Base reward calculation
        base_reward = await self._calculate_base_reward(teaching_outcome)
        
        # Performance multiplier based on learning gain
        performance_multiplier = await self._calculate_performance_multiplier(teaching_outcome)
        
        # Innovation bonus for novel teaching approaches
        innovation_bonus = await self._calculate_innovation_bonus(teaching_outcome)
        
        # Consistency factor based on historical performance
        consistency_factor = await self._calculate_consistency_factor(teaching_outcome.teacher_id)
        
        # Total reward calculation
        total_reward = (base_reward * performance_multiplier + innovation_bonus) * consistency_factor
        
        # Generate verification hash
        verification_data = {
            'teacher_id': str(teaching_outcome.teacher_id),
            'session_id': str(teaching_outcome.session_id),
            'learning_gain': teaching_outcome.learning_gain,
            'success_rate': teaching_outcome.success_rate,
            'engagement_score': teaching_outcome.engagement_score,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        verification_hash = await self._generate_verification_hash(verification_data)
        
        reward = VerifiableReward(
            teacher_id=teaching_outcome.teacher_id,
            session_id=teaching_outcome.session_id,
            curriculum_id=teaching_outcome.curriculum_id,
            base_reward=base_reward,
            performance_multiplier=performance_multiplier,
            innovation_bonus=innovation_bonus,
            consistency_factor=consistency_factor,
            total_reward=total_reward,
            verification_hash=verification_hash
        )
        
        self.reward_history.append(reward)
        
        self.logger.info(
            "Verifiable reward calculated",
            reward_id=str(reward.reward_id),
            total_reward=total_reward,
            verification_hash=verification_hash[:16] + "..."
        )
        
        return reward
    
    async def update_teacher_weights(self, teacher_id: UUID, reward: VerifiableReward) -> bool:
        """
        Update teacher neural weights based on verifiable reward.
        
        Args:
            teacher_id: Teacher model identifier
            reward: Verified reward from teaching performance
            
        Returns:
            True if weights updated successfully
        """
        self.logger.info("Updating teacher weights", teacher_id=str(teacher_id))
        
        try:
            # Verify reward authenticity
            if not await self._verify_reward_authenticity(reward):
                self.logger.warning("Reward verification failed", reward_id=str(reward.reward_id))
                return False
            
            # Get or create teacher weights
            if teacher_id not in self.teacher_weights:
                self.teacher_weights[teacher_id] = TeacherWeights(teacher_id=teacher_id)
            
            weights = self.teacher_weights[teacher_id]
            
            # Calculate weight gradients based on reward
            gradients = await self._calculate_weight_gradients(reward, weights)
            
            # Apply gradient updates with momentum and weight decay
            await self._apply_weight_updates(weights, gradients)
            
            # Update timestamp
            weights.last_update = datetime.now(timezone.utc)
            
            self.logger.info(
                "Teacher weights updated successfully",
                teacher_id=str(teacher_id),
                total_reward=reward.total_reward,
                gradient_norm=sum(abs(g) for g in gradients.values())
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Error updating teacher weights", error=str(e))
            return False
    
    async def validate_teaching_effectiveness(self, curriculum_id: UUID) -> bool:
        """
        Validate overall teaching effectiveness for a curriculum.
        
        Args:
            curriculum_id: Curriculum to validate
            
        Returns:
            True if teaching effectiveness meets standards
        """
        self.logger.info("Validating teaching effectiveness", curriculum_id=str(curriculum_id))
        
        # Get rewards for this curriculum
        curriculum_rewards = [
            r for r in self.reward_history 
            if r.curriculum_id == curriculum_id  # Note: Need to add curriculum_id to VerifiableReward
        ]
        
        if not curriculum_rewards:
            self.logger.warning("No rewards found for curriculum", curriculum_id=str(curriculum_id))
            return False
        
        # Calculate effectiveness metrics
        avg_reward = sum(r.total_reward for r in curriculum_rewards) / len(curriculum_rewards)
        consistency_score = await self._calculate_curriculum_consistency(curriculum_rewards)
        improvement_trend = await self._calculate_improvement_trend(curriculum_rewards)
        
        # Effectiveness thresholds
        min_reward_threshold = 0.6
        min_consistency_threshold = 0.7
        min_improvement_threshold = -0.1  # Allow slight decline
        
        effectiveness_validated = (
            avg_reward >= min_reward_threshold and
            consistency_score >= min_consistency_threshold and
            improvement_trend >= min_improvement_threshold
        )
        
        validation_details = {
            'curriculum_id': str(curriculum_id),
            'avg_reward': avg_reward,
            'consistency_score': consistency_score,
            'improvement_trend': improvement_trend,
            'validated': effectiveness_validated,
            'session_count': len(curriculum_rewards)
        }
        
        self.logger.info("Teaching effectiveness validation completed", **validation_details)
        
        return effectiveness_validated
    
    async def get_teacher_performance_summary(self, teacher_id: UUID) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for a teacher.
        
        Args:
            teacher_id: Teacher model identifier
            
        Returns:
            Performance summary with metrics and recommendations
        """
        teacher_rewards = [r for r in self.reward_history if r.teacher_id == teacher_id]
        
        if not teacher_rewards:
            return {
                'teacher_id': str(teacher_id),
                'status': 'no_data',
                'total_sessions': 0,
                'avg_reward': 0.0,
                'performance_trend': 0.0,
                'recommendations': ['Complete initial teaching sessions']
            }
        
        # Calculate performance metrics
        total_sessions = len(teacher_rewards)
        avg_reward = sum(r.total_reward for r in teacher_rewards) / total_sessions
        recent_rewards = teacher_rewards[-10:]  # Last 10 sessions
        avg_recent_reward = sum(r.total_reward for r in recent_rewards) / len(recent_rewards)
        
        performance_trend = await self._calculate_performance_trend(teacher_rewards)
        consistency_score = await self._calculate_teacher_consistency(teacher_rewards)
        
        # Generate recommendations
        recommendations = await self._generate_performance_recommendations(
            avg_reward, performance_trend, consistency_score
        )
        
        # Get current weights
        weights = self.teacher_weights.get(teacher_id, None)
        weight_summary = {}
        if weights:
            weight_summary = {
                'curriculum_weights_count': len(weights.curriculum_weights),
                'strategy_weights_count': len(weights.strategy_weights),
                'domain_expertise_count': len(weights.domain_expertise),
                'learning_rate': weights.learning_rate
            }
        
        return {
            'teacher_id': str(teacher_id),
            'status': 'active',
            'total_sessions': total_sessions,
            'avg_reward': avg_reward,
            'avg_recent_reward': avg_recent_reward,
            'performance_trend': performance_trend,
            'consistency_score': consistency_score,
            'weight_summary': weight_summary,
            'recommendations': recommendations,
            'last_session': teacher_rewards[-1].timestamp.isoformat() if teacher_rewards else None
        }
    
    # === Private Helper Methods ===
    
    async def _calculate_base_reward(self, outcome: TeachingOutcome) -> float:
        """Calculate base reward from teaching outcome metrics"""
        # Weighted combination of key metrics
        learning_gain_weight = 0.4
        success_rate_weight = 0.3
        engagement_weight = 0.2
        efficiency_weight = 0.1
        
        # Normalize time efficiency (shorter is better for same results)
        efficiency_score = max(0.1, 1.0 - (outcome.time_spent_minutes - 30) / 120)
        
        base_reward = (
            outcome.learning_gain * learning_gain_weight +
            outcome.success_rate * success_rate_weight +
            outcome.engagement_score * engagement_weight +
            efficiency_score * efficiency_weight
        )
        
        return max(0.0, min(1.0, base_reward))
    
    async def _calculate_performance_multiplier(self, outcome: TeachingOutcome) -> float:
        """Calculate performance multiplier based on exceptional results"""
        if outcome.learning_gain > 0.8:
            return 1.5  # Exceptional learning gain
        elif outcome.learning_gain > 0.6:
            return 1.2  # Good learning gain
        elif outcome.learning_gain > 0.4:
            return 1.0  # Average learning gain
        else:
            return 0.8  # Below average learning gain
    
    async def _calculate_innovation_bonus(self, outcome: TeachingOutcome) -> float:
        """Calculate bonus for innovative teaching approaches"""
        # Check for novel improvements or creative solutions
        innovation_indicators = len(outcome.areas_improved)
        engagement_bonus = max(0, outcome.engagement_score - 0.7)
        
        innovation_bonus = (innovation_indicators * 0.05) + (engagement_bonus * 0.1)
        return min(0.2, innovation_bonus)  # Cap at 0.2
    
    async def _calculate_consistency_factor(self, teacher_id: UUID) -> float:
        """Calculate consistency factor based on historical performance"""
        teacher_rewards = [r for r in self.reward_history if r.teacher_id == teacher_id]
        
        if len(teacher_rewards) < 3:
            return 1.0  # No penalty for new teachers
        
        recent_rewards = teacher_rewards[-10:]
        rewards_values = [r.total_reward for r in recent_rewards]
        
        # Calculate coefficient of variation (std/mean)
        mean_reward = sum(rewards_values) / len(rewards_values)
        variance = sum((r - mean_reward) ** 2 for r in rewards_values) / len(rewards_values)
        std_dev = math.sqrt(variance)
        
        if mean_reward > 0:
            cv = std_dev / mean_reward
            # Higher consistency (lower CV) gets bonus up to 1.1, lower consistency gets penalty down to 0.9
            consistency_factor = max(0.9, min(1.1, 1.1 - cv))
        else:
            consistency_factor = 0.9
        
        return consistency_factor
    
    async def _generate_verification_hash(self, data: Dict[str, Any]) -> str:
        """Generate cryptographic hash for reward verification"""
        # Sort keys for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        
        # Add salt for additional security
        salt = f"PRSM_RLVR_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        salted_data = f"{sorted_data}:{salt}"
        
        return hashlib.sha256(salted_data.encode()).hexdigest()
    
    async def _verify_reward_authenticity(self, reward: VerifiableReward) -> bool:
        """Verify the authenticity of a reward"""
        # In production, this would involve more sophisticated verification
        # For now, check basic integrity
        return (
            reward.total_reward >= 0 and
            reward.verification_hash and
            len(reward.verification_hash) == 64  # SHA256 length
        )
    
    async def _calculate_weight_gradients(self, reward: VerifiableReward, weights: TeacherWeights) -> Dict[str, float]:
        """Calculate gradients for weight updates"""
        # Simplified gradient calculation based on reward signal
        reward_error = reward.total_reward - 0.5  # Target average reward of 0.5
        
        gradients = {}
        
        # Update curriculum weights
        for domain in weights.curriculum_weights:
            gradients[f"curriculum_{domain}"] = reward_error * weights.learning_rate
        
        # Update strategy weights
        for strategy in weights.strategy_weights:
            gradients[f"strategy_{strategy}"] = reward_error * weights.learning_rate * 0.5
        
        # Update domain expertise
        for domain in weights.domain_expertise:
            gradients[f"expertise_{domain}"] = reward_error * weights.learning_rate * 0.3
        
        return gradients
    
    async def _apply_weight_updates(self, weights: TeacherWeights, gradients: Dict[str, float]):
        """Apply gradient updates with momentum and weight decay"""
        # Apply updates to curriculum weights
        for key, gradient in gradients.items():
            if key.startswith("curriculum_"):
                domain = key.replace("curriculum_", "")
                if domain not in weights.curriculum_weights:
                    weights.curriculum_weights[domain] = 0.5
                
                # Apply momentum and weight decay
                old_weight = weights.curriculum_weights[domain]
                weight_update = gradient - (weights.weight_decay * old_weight)
                weights.curriculum_weights[domain] += weight_update
                
                # Clamp weights to reasonable range
                weights.curriculum_weights[domain] = max(-2.0, min(2.0, weights.curriculum_weights[domain]))
        
        # Similar updates for strategy and expertise weights
        for key, gradient in gradients.items():
            if key.startswith("strategy_"):
                strategy = key.replace("strategy_", "")
                if strategy not in weights.strategy_weights:
                    weights.strategy_weights[strategy] = 0.5
                
                old_weight = weights.strategy_weights[strategy]
                weight_update = gradient - (weights.weight_decay * old_weight)
                weights.strategy_weights[strategy] += weight_update
                weights.strategy_weights[strategy] = max(-2.0, min(2.0, weights.strategy_weights[strategy]))
            
            elif key.startswith("expertise_"):
                domain = key.replace("expertise_", "")
                if domain not in weights.domain_expertise:
                    weights.domain_expertise[domain] = 0.5
                
                old_weight = weights.domain_expertise[domain]
                weight_update = gradient - (weights.weight_decay * old_weight)
                weights.domain_expertise[domain] += weight_update
                weights.domain_expertise[domain] = max(0.0, min(1.0, weights.domain_expertise[domain]))
    
    async def _calculate_curriculum_consistency(self, rewards: List[VerifiableReward]) -> float:
        """Calculate consistency score for curriculum performance"""
        if len(rewards) < 2:
            return 1.0
        
        rewards_values = [r.total_reward for r in rewards]
        mean_reward = sum(rewards_values) / len(rewards_values)
        variance = sum((r - mean_reward) ** 2 for r in rewards_values) / len(rewards_values)
        std_dev = math.sqrt(variance)
        
        # Convert to consistency score (0-1, higher is better)
        if mean_reward > 0:
            cv = std_dev / mean_reward
            consistency = max(0.0, 1.0 - cv)
        else:
            consistency = 0.0
        
        return consistency
    
    async def _calculate_improvement_trend(self, rewards: List[VerifiableReward]) -> float:
        """Calculate improvement trend over time"""
        if len(rewards) < 3:
            return 0.0
        
        # Sort by timestamp
        sorted_rewards = sorted(rewards, key=lambda r: r.timestamp)
        rewards_values = [r.total_reward for r in sorted_rewards]
        
        # Simple linear regression slope
        n = len(rewards_values)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(rewards_values)
        sum_xy = sum(x * y for x, y in zip(x_values, rewards_values))
        sum_x2 = sum(x * x for x in x_values)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    async def _calculate_performance_trend(self, rewards: List[VerifiableReward]) -> float:
        """Calculate teacher performance trend"""
        return await self._calculate_improvement_trend(rewards)
    
    async def _calculate_teacher_consistency(self, rewards: List[VerifiableReward]) -> float:
        """Calculate teacher consistency score"""
        return await self._calculate_curriculum_consistency(rewards)
    
    async def _generate_performance_recommendations(
        self, 
        avg_reward: float, 
        trend: float, 
        consistency: float
    ) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if avg_reward < 0.5:
            recommendations.append("Focus on improving student learning outcomes")
            recommendations.append("Review curriculum difficulty and pacing")
        
        if trend < -0.05:
            recommendations.append("Address declining performance trend")
            recommendations.append("Analyze recent teaching sessions for issues")
        
        if consistency < 0.7:
            recommendations.append("Work on consistent teaching quality")
            recommendations.append("Standardize successful teaching approaches")
        
        if avg_reward > 0.8 and trend > 0.05:
            recommendations.append("Excellent performance - consider mentoring other teachers")
            recommendations.append("Share successful strategies with the community")
        
        if not recommendations:
            recommendations.append("Maintain current performance level")
            recommendations.append("Continue monitoring student engagement")
        
        return recommendations
    
    # === SEAL Enhancement Methods ===
    
    async def calculate_seal_edit_reward(
        self,
        edit_id: UUID,
        teacher_id: UUID,
        edit_type: str,
        baseline_performance: Dict[str, float],
        updated_performance: Dict[str, float],
        edit_data: Dict[str, Any]
    ) -> SEALEditReward:
        """
        Calculate SEAL-specific reward for self-edit performance
        
        🎯 SEAL REWARD CALCULATION:
        Based on MIT SEAL paper methodology - rewards self-edits that
        improve downstream task performance after model updates.
        """
        
        self.logger.info("Calculating SEAL edit reward",
                        edit_id=str(edit_id),
                        edit_type=edit_type)
        
        try:
            # Calculate downstream performance improvement
            downstream_improvement = await self._calculate_downstream_improvement(
                baseline_performance, updated_performance
            )
            
            # Assess synthetic data quality
            data_quality_score = await self._assess_synthetic_data_quality(
                edit_data.get("generated_content", "")
            )
            
            # Evaluate optimization parameter effectiveness
            parameter_effectiveness = await self._evaluate_parameter_effectiveness(
                edit_data.get("optimization_params", {}),
                baseline_performance,
                updated_performance
            )
            
            # Calculate meta-learning contribution
            meta_learning_contribution = await self._calculate_meta_learning_contribution(
                teacher_id, edit_type, downstream_improvement
            )
            
            # Compute final reward signal (SEAL's binary + continuous approach)
            reward_signal = await self._compute_seal_reward_signal(
                downstream_improvement,
                data_quality_score,
                parameter_effectiveness,
                meta_learning_contribution
            )
            
            # Create SEAL edit reward
            seal_reward = SEALEditReward(
                edit_id=edit_id,
                teacher_id=teacher_id,
                edit_type=edit_type,
                downstream_improvement=downstream_improvement,
                data_quality_score=data_quality_score,
                parameter_effectiveness=parameter_effectiveness,
                meta_learning_contribution=meta_learning_contribution,
                baseline_performance=baseline_performance,
                updated_performance=updated_performance,
                reward_signal=reward_signal
            )
            
            # Store reward for policy updates
            self.seal_edit_rewards.append(seal_reward)
            
            # Update SEAL policy state
            await self._update_seal_policy_state(teacher_id, seal_reward)
            
            self.logger.info("SEAL edit reward calculated",
                           edit_id=str(edit_id),
                           reward_signal=reward_signal,
                           downstream_improvement=downstream_improvement)
            
            return seal_reward
            
        except Exception as e:
            self.logger.error("SEAL edit reward calculation failed",
                            edit_id=str(edit_id),
                            error=str(e))
            raise
    
    async def update_seal_policy_with_restem(
        self,
        teacher_id: UUID,
        training_iteration: int
    ) -> Dict[str, Any]:
        """
        Update SEAL policy using ReSTEM methodology from the paper
        
        🔄 ReSTEM PROCESS:
        1. Sample self-edits from current policy
        2. Evaluate downstream performance
        3. Filter good edits (reward > threshold)
        4. Supervised fine-tuning on good edits only
        """
        
        self.logger.info("Updating SEAL policy with ReSTEM",
                        teacher_id=str(teacher_id),
                        iteration=training_iteration)
        
        try:
            # Get recent edit rewards for this teacher
            recent_rewards = [
                r for r in self.seal_edit_rewards[-50:]  # Last 50 edits
                if r.teacher_id == teacher_id
            ]
            
            if len(recent_rewards) < 5:
                self.logger.warning("Insufficient edit data for ReSTEM update")
                return {"success": False, "reason": "insufficient_data"}
            
            # Filter good edits (ReSTEM threshold approach)
            reward_threshold = 0.6  # Based on SEAL paper
            good_edits = [r for r in recent_rewards if r.reward_signal >= reward_threshold]
            
            if not good_edits:
                self.logger.warning("No good edits found for ReSTEM update")
                return {"success": False, "reason": "no_good_edits"}
            
            # Extract successful patterns for policy improvement
            successful_patterns = await self._extract_successful_patterns(good_edits)
            
            # Update policy state based on successful patterns
            policy_state = self.seal_policy_states.get(teacher_id)
            if not policy_state:
                policy_state = SEALPolicyState(teacher_id=teacher_id)
                self.seal_policy_states[teacher_id] = policy_state
            
            # Update success rates
            for edit_type in successful_patterns["edit_types"]:
                if edit_type not in policy_state.edit_type_success_rates:
                    policy_state.edit_type_success_rates[edit_type] = 0.0
                
                # Exponential moving average update
                alpha = 0.1
                current_rate = policy_state.edit_type_success_rates[edit_type]
                new_rate = successful_patterns["edit_types"][edit_type]
                policy_state.edit_type_success_rates[edit_type] = (
                    alpha * new_rate + (1 - alpha) * current_rate
                )
            
            # Update optimal parameter ranges
            for param, values in successful_patterns["parameters"].items():
                if values:
                    param_min, param_max = min(values), max(values)
                    policy_state.optimal_parameter_ranges[param] = (param_min, param_max)
            
            # Update format preferences
            for format_type, count in successful_patterns["formats"].items():
                if format_type not in policy_state.successful_format_preferences:
                    policy_state.successful_format_preferences[format_type] = 0
                policy_state.successful_format_preferences[format_type] += count
            
            # Update overall metrics
            policy_state.total_edits_generated = len(recent_rewards)
            policy_state.successful_edits = len(good_edits)
            policy_state.average_reward = sum(r.reward_signal for r in recent_rewards) / len(recent_rewards)
            policy_state.improvement_rate = await self._calculate_policy_improvement_rate(teacher_id)
            policy_state.last_updated = datetime.now(timezone.utc)
            
            # Record meta-learning progress
            meta_learning_record = {
                "teacher_id": str(teacher_id),
                "iteration": training_iteration,
                "good_edits_count": len(good_edits),
                "total_edits_count": len(recent_rewards),
                "success_rate": len(good_edits) / len(recent_rewards),
                "average_reward": policy_state.average_reward,
                "patterns_extracted": len(successful_patterns["edit_types"]),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.meta_learning_history.append(meta_learning_record)
            
            self.logger.info("SEAL policy updated with ReSTEM",
                           teacher_id=str(teacher_id),
                           good_edits=len(good_edits),
                           total_edits=len(recent_rewards),
                           success_rate=len(good_edits) / len(recent_rewards))
            
            return {
                "success": True,
                "good_edits_count": len(good_edits),
                "total_edits_count": len(recent_rewards),
                "success_rate": len(good_edits) / len(recent_rewards),
                "patterns_extracted": successful_patterns,
                "policy_state": policy_state
            }
            
        except Exception as e:
            self.logger.error("ReSTEM policy update failed",
                            teacher_id=str(teacher_id),
                            error=str(e))
            return {"success": False, "error": str(e)}
    
    async def get_seal_performance_metrics(self, teacher_id: UUID) -> Dict[str, Any]:
        """Get comprehensive SEAL performance metrics for a teacher"""
        
        teacher_edits = [r for r in self.seal_edit_rewards if r.teacher_id == teacher_id]
        policy_state = self.seal_policy_states.get(teacher_id)
        
        if not teacher_edits:
            return {
                "teacher_id": str(teacher_id),
                "status": "no_seal_data",
                "total_edits": 0,
                "success_rate": 0.0,
                "average_reward": 0.0
            }
        
        # Calculate metrics
        total_edits = len(teacher_edits)
        successful_edits = len([e for e in teacher_edits if e.reward_signal >= 0.6])
        success_rate = successful_edits / total_edits
        average_reward = sum(e.reward_signal for e in teacher_edits) / total_edits
        
        # Recent performance (last 10 edits)
        recent_edits = teacher_edits[-10:]
        recent_avg_reward = sum(e.reward_signal for e in recent_edits) / len(recent_edits)
        
        # Edit type breakdown
        edit_type_performance = {}
        for edit_type in ["knowledge_incorporation", "curriculum_optimization", "student_adaptation"]:
            type_edits = [e for e in teacher_edits if e.edit_type == edit_type]
            if type_edits:
                edit_type_performance[edit_type] = {
                    "count": len(type_edits),
                    "success_rate": len([e for e in type_edits if e.reward_signal >= 0.6]) / len(type_edits),
                    "average_reward": sum(e.reward_signal for e in type_edits) / len(type_edits)
                }
        
        return {
            "teacher_id": str(teacher_id),
            "status": "active",
            "total_edits": total_edits,
            "successful_edits": successful_edits,
            "success_rate": success_rate,
            "average_reward": average_reward,
            "recent_average_reward": recent_avg_reward,
            "edit_type_performance": edit_type_performance,
            "policy_state": policy_state.dict() if policy_state else None,
            "meta_learning_progress": len([m for m in self.meta_learning_history if m["teacher_id"] == str(teacher_id)])
        }
    
    # === SEAL Private Helper Methods ===
    
    async def _calculate_downstream_improvement(
        self,
        baseline: Dict[str, float],
        updated: Dict[str, float]
    ) -> float:
        """Calculate improvement in downstream task performance"""
        
        if not baseline or not updated:
            return 0.0
        
        improvements = []
        for metric in baseline:
            if metric in updated:
                improvement = updated[metric] - baseline[metric]
                improvements.append(improvement)
        
        if not improvements:
            return 0.0
        
        # Return average improvement, clamped to reasonable range
        avg_improvement = sum(improvements) / len(improvements)
        return max(-1.0, min(1.0, avg_improvement))
    
    async def _assess_synthetic_data_quality(self, generated_content: str) -> float:
        """Assess quality of SEAL-generated synthetic data"""
        
        if not generated_content or len(generated_content.strip()) < 10:
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Length appropriateness (similar to SEAL paper evaluation)
        word_count = len(generated_content.split())
        if 50 <= word_count <= 300:
            quality_score += 0.15
        
        # Structure indicators
        structure_markers = ["1.", "2.", "Q:", "A:", "Example:", "Step"]
        if any(marker in generated_content for marker in structure_markers):
            quality_score += 0.15
        
        # Diversity indicators
        unique_words = len(set(generated_content.lower().split()))
        diversity_ratio = unique_words / max(1, word_count)
        quality_score += min(0.2, diversity_ratio * 0.4)
        
        return min(1.0, max(0.0, quality_score))
    
    async def _evaluate_parameter_effectiveness(
        self,
        params: Dict[str, Any],
        baseline: Dict[str, float],
        updated: Dict[str, float]
    ) -> float:
        """Evaluate effectiveness of optimization parameters"""
        
        if not params:
            return 0.5
        
        effectiveness_score = 0.5
        
        # Learning rate effectiveness
        lr = params.get("learning_rate", 3e-5)
        if isinstance(lr, (int, float)):
            if 1e-5 <= lr <= 1e-4:
                effectiveness_score += 0.1
        
        # Epoch count effectiveness
        epochs = params.get("epochs", 3)
        if isinstance(epochs, (int, float)):
            if 2 <= epochs <= 5:
                effectiveness_score += 0.1
        
        # Performance correlation
        if baseline and updated:
            improvement = await self._calculate_downstream_improvement(baseline, updated)
            if improvement > 0.1:
                effectiveness_score += 0.2
        
        return min(1.0, max(0.0, effectiveness_score))
    
    async def _calculate_meta_learning_contribution(
        self,
        teacher_id: UUID,
        edit_type: str,
        improvement: float
    ) -> float:
        """Calculate contribution to overall meta-learning progress"""
        
        # Get historical performance for this edit type
        historical_edits = [
            r for r in self.seal_edit_rewards
            if r.teacher_id == teacher_id and r.edit_type == edit_type
        ]
        
        if len(historical_edits) < 2:
            return improvement  # First attempts get full credit
        
        # Calculate improvement trend
        recent_improvements = [r.downstream_improvement for r in historical_edits[-5:]]
        avg_recent = sum(recent_improvements) / len(recent_improvements)
        
        # Meta-learning contribution is improvement relative to recent average
        meta_contribution = improvement - avg_recent
        
        return max(0.0, min(1.0, meta_contribution + 0.5))
    
    async def _compute_seal_reward_signal(
        self,
        downstream_improvement: float,
        data_quality: float,
        parameter_effectiveness: float,
        meta_learning: float
    ) -> float:
        """Compute final SEAL reward signal using weighted combination"""
        
        # Weights based on SEAL paper priorities
        improvement_weight = 0.4  # Primary signal from downstream performance
        quality_weight = 0.3      # Quality of generated content
        params_weight = 0.2       # Parameter optimization effectiveness
        meta_weight = 0.1         # Meta-learning contribution
        
        weighted_reward = (
            downstream_improvement * improvement_weight +
            data_quality * quality_weight +
            parameter_effectiveness * params_weight +
            meta_learning * meta_weight
        )
        
        # Apply SEAL's binary threshold approach
        threshold = 0.1  # Minimum improvement required
        if downstream_improvement >= threshold:
            # Boost successful edits
            return min(1.0, max(0.0, weighted_reward + 0.2))
        else:
            # Binary penalty for insufficient improvement
            return 0.0
    
    async def _update_seal_policy_state(self, teacher_id: UUID, reward: SEALEditReward):
        """Update SEAL policy state with new reward information"""
        
        if teacher_id not in self.seal_policy_states:
            self.seal_policy_states[teacher_id] = SEALPolicyState(teacher_id=teacher_id)
        
        policy_state = self.seal_policy_states[teacher_id]
        
        # Update counters
        policy_state.total_edits_generated += 1
        if reward.reward_signal >= 0.6:
            policy_state.successful_edits += 1
        
        # Update running average reward
        alpha = 0.1  # Learning rate for exponential moving average
        if policy_state.average_reward == 0.0:
            policy_state.average_reward = reward.reward_signal
        else:
            policy_state.average_reward = (
                alpha * reward.reward_signal + (1 - alpha) * policy_state.average_reward
            )
        
        policy_state.last_updated = datetime.now(timezone.utc)
    
    async def _extract_successful_patterns(
        self,
        good_edits: List[SEALEditReward]
    ) -> Dict[str, Any]:
        """Extract patterns from successful edits for policy improvement"""
        
        patterns = {
            "edit_types": {},
            "parameters": {},
            "formats": {}
        }
        
        # Analyze edit type success rates
        for edit in good_edits:
            edit_type = edit.edit_type
            if edit_type not in patterns["edit_types"]:
                patterns["edit_types"][edit_type] = 0
            patterns["edit_types"][edit_type] += 1
        
        # Normalize edit type counts to rates
        total_good_edits = len(good_edits)
        for edit_type in patterns["edit_types"]:
            patterns["edit_types"][edit_type] /= total_good_edits
        
        # This would be expanded to extract more patterns from edit data
        # For now, return basic structure
        
        return patterns
    
    async def _calculate_policy_improvement_rate(self, teacher_id: UUID) -> float:
        """Calculate rate of policy improvement over time"""
        
        teacher_edits = [r for r in self.seal_edit_rewards if r.teacher_id == teacher_id]
        
        if len(teacher_edits) < 10:
            return 0.0
        
        # Compare recent performance to earlier performance
        recent_rewards = [r.reward_signal for r in teacher_edits[-10:]]
        earlier_rewards = [r.reward_signal for r in teacher_edits[-20:-10]] if len(teacher_edits) >= 20 else teacher_edits[:-10]
        
        if not earlier_rewards:
            return 0.0
        
        recent_avg = sum(recent_rewards) / len(recent_rewards)
        earlier_avg = sum(earlier_rewards) / len(earlier_rewards)
        
        improvement_rate = recent_avg - earlier_avg
        return max(-1.0, min(1.0, improvement_rate))