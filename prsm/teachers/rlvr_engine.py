"""
PRSM RLVR (Reinforcement Learning with Verifiable Rewards) Engine

Implements verifiable reward calculation and teacher weight optimization
for the distilled teacher model system.

Based on execution_plan.md Week 7-8 requirements.
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
from ..core.models import TeacherModel, LearningSession, PRSMBaseModel
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


class RLVREngine:
    """
    Reinforcement Learning with Verifiable Rewards Engine
    
    Implements cryptographically verifiable reward calculation and
    adaptive teacher weight optimization based on teaching effectiveness.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="rlvr_engine")
        self.teacher_weights: Dict[UUID, TeacherWeights] = {}
        self.reward_history: List[VerifiableReward] = []
        self.verification_threshold = 0.95
        
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