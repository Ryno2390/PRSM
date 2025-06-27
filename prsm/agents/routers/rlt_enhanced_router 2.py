"""
RLT-Enhanced Model Router

Extends the ModelRouter with RLT (Reinforcement Learning Teachers) teacher selection
and quality-based routing capabilities for optimal student-teacher pairing.

Key Features:
- RLT teacher discovery and selection
- Domain-specific teacher specialization
- Student capability matching
- Explanation quality tracking
- Dense reward optimization routing
"""

import asyncio
import time
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .model_router import (
    ModelRouter, RoutingStrategy, ModelCandidate, RoutingDecision, 
    ModelSource, TeacherSelection
)
from ...teachers.seal_rlt_enhanced_teacher import SEALRLTEnhancedTeacher, SEALRLTConfig
from ...teachers.rlt.dense_reward_trainer import RLTTrainingConfig
from ...teachers.rlt.student_comprehension_evaluator import ComprehensionMetrics
from ...core.models import TeacherModel, ModelType, AgentType
from ...federation.model_registry import ModelRegistry

logger = structlog.get_logger(__name__)


class RLTRoutingStrategy(str, Enum):
    """RLT-specific routing strategies"""
    EXPLANATION_QUALITY = "explanation_quality"
    STUDENT_COMPREHENSION = "student_comprehension"
    DENSE_REWARD_OPTIMIZED = "dense_reward_optimized"
    DOMAIN_SPECIALIZED = "domain_specialized"
    PROGRESSIVE_DIFFICULTY = "progressive_difficulty"


class RLTTeacherCandidate(ModelCandidate):
    """Enhanced teacher candidate with RLT-specific metrics"""
    
    # RLT-specific attributes
    explanation_quality_score: float = Field(ge=0.0, le=1.0, default=0.5)
    dense_reward_effectiveness: float = Field(ge=0.0, le=1.0, default=0.5)
    student_comprehension_score: float = Field(ge=0.0, le=1.0, default=0.5)
    domain_expertise_level: float = Field(ge=0.0, le=1.0, default=0.5)
    adaptive_teaching_ability: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # Historical performance metrics
    avg_rlt_reward: float = 0.0
    successful_distillations: int = 0
    student_improvement_rate: float = 0.0
    
    # Quality tracking
    explanation_coherence: float = Field(ge=0.0, le=1.0, default=0.5)
    logical_continuity: float = Field(ge=0.0, le=1.0, default=0.5)
    think_token_quality: float = Field(ge=0.0, le=1.0, default=0.5)
    
    def calculate_rlt_score(self, strategy: RLTRoutingStrategy, student_capability: float = 0.5) -> float:
        """Calculate RLT-specific routing score"""
        
        if strategy == RLTRoutingStrategy.EXPLANATION_QUALITY:
            score = (
                self.explanation_quality_score * 0.4 +
                self.explanation_coherence * 0.3 +
                self.logical_continuity * 0.2 +
                self.think_token_quality * 0.1
            )
        
        elif strategy == RLTRoutingStrategy.STUDENT_COMPREHENSION:
            score = (
                self.student_comprehension_score * 0.5 +
                self.adaptive_teaching_ability * 0.3 +
                self.student_improvement_rate * 0.2
            )
        
        elif strategy == RLTRoutingStrategy.DENSE_REWARD_OPTIMIZED:
            score = (
                self.dense_reward_effectiveness * 0.4 +
                self.avg_rlt_reward * 0.3 +
                self.explanation_quality_score * 0.2 +
                self.student_comprehension_score * 0.1
            )
        
        elif strategy == RLTRoutingStrategy.DOMAIN_SPECIALIZED:
            score = (
                self.domain_expertise_level * 0.5 +
                self.specialization_match_score * 0.3 +
                self.teaching_effectiveness * 0.2
            )
        
        elif strategy == RLTRoutingStrategy.PROGRESSIVE_DIFFICULTY:
            # Match teacher capability to student level
            capability_match = 1.0 - abs(self.domain_expertise_level - student_capability)
            score = (
                capability_match * 0.4 +
                self.adaptive_teaching_ability * 0.3 +
                self.explanation_quality_score * 0.2 +
                self.student_comprehension_score * 0.1
            )
        
        else:
            # Default to balanced RLT scoring
            score = (
                self.explanation_quality_score * 0.3 +
                self.dense_reward_effectiveness * 0.25 +
                self.student_comprehension_score * 0.25 +
                self.domain_expertise_level * 0.2
            )
        
        return min(score, 1.0)
    
    @property
    def specialization_match_score(self) -> float:
        """Calculate specialization matching score"""
        return self.compatibility_score  # Use existing compatibility as base


class RLTTeacherSelection(TeacherSelection):
    """Enhanced teacher selection with RLT capabilities"""
    
    rlt_config: Optional[RLTTrainingConfig] = None
    target_explanation_quality: float = 0.8
    target_comprehension_score: float = 0.75
    dense_reward_weight: float = 0.7
    
    # Student assessment
    student_current_capability: float = 0.5
    student_learning_style: str = "adaptive"
    target_improvement_rate: float = 0.2
    
    # Selection results
    selected_rlt_teacher: Optional[RLTTeacherCandidate] = None
    predicted_improvement: float = 0.0
    estimated_training_time: float = 0.0
    quality_confidence: float = 0.0


class RLTEnhancedRouter(ModelRouter):
    """
    RLT-Enhanced Model Router for Teacher Selection
    
    Extends ModelRouter with RLT-specific capabilities:
    - Dense reward-based teacher selection
    - Student comprehension optimization
    - Domain-specialized teacher routing
    - Explanation quality tracking
    - Progressive difficulty matching
    """
    
    def __init__(self, model_registry: Optional[ModelRegistry] = None, agent_id: Optional[str] = None):
        super().__init__(model_registry, agent_id)
        
        # RLT-specific components
        self.rlt_teacher_pool: Dict[str, List[RLTTeacherCandidate]] = {}
        self.teacher_performance_history: Dict[str, List[float]] = {}
        self.student_teacher_pairings: Dict[str, List[Dict[str, Any]]] = {}
        self.explanation_quality_tracker: Dict[str, List[float]] = {}
        self.dense_reward_history: Dict[str, List[Dict[str, float]]] = {}
        
        # RLT configuration
        self.rlt_config = RLTTrainingConfig()
        
        logger.info("RLT-Enhanced Router initialized",
                   agent_id=self.agent_id,
                   rlt_enabled=True)
    
    async def discover_rlt_teachers(self, domain: str, difficulty: float = 0.5) -> List[RLTTeacherCandidate]:
        """
        Discover RLT teachers for specific domain and difficulty level
        
        Args:
            domain: Target domain (e.g., 'mathematics', 'physics', 'chemistry')
            difficulty: Difficulty level (0.0 = beginner, 1.0 = expert)
            
        Returns:
            List of qualified RLT teacher candidates
        """
        logger.info("Discovering RLT teachers",
                   domain=domain,
                   difficulty=difficulty)
        
        # Check cache first
        cache_key = f"{domain}_{difficulty:.1f}"
        if cache_key in self.rlt_teacher_pool:
            cached_teachers = self.rlt_teacher_pool[cache_key]
            if cached_teachers:
                logger.debug("Using cached RLT teachers", count=len(cached_teachers))
                return cached_teachers
        
        # Discover from multiple sources
        candidates = []
        
        # 1. Local RLT teacher registry
        local_candidates = await self._discover_local_rlt_teachers(domain, difficulty)
        candidates.extend(local_candidates)
        
        # 2. Federated RLT teacher network
        federated_candidates = await self._discover_federated_rlt_teachers(domain, difficulty)
        candidates.extend(federated_candidates)
        
        # 3. Marketplace RLT teachers
        marketplace_candidates = await self._discover_marketplace_rlt_teachers(domain, difficulty)
        candidates.extend(marketplace_candidates)
        
        # Filter and rank by RLT capabilities
        qualified_candidates = await self._filter_rlt_teachers(candidates, domain, difficulty)
        
        # Cache results
        self.rlt_teacher_pool[cache_key] = qualified_candidates
        
        logger.info("RLT teacher discovery completed",
                   domain=domain,
                   candidates_found=len(qualified_candidates))
        
        return qualified_candidates
    
    async def route_to_optimal_teacher(
        self,
        question: str,
        solution: str,
        student_model: str,
        student_capability: float = 0.5
    ) -> RLTTeacherSelection:
        """
        Route to optimal RLT teacher for question-solution pair
        
        Args:
            question: The question to be explained
            solution: The solution to be taught
            student_model: ID of the student model
            student_capability: Current capability level of student (0.0-1.0)
            
        Returns:
            Complete teacher selection with RLT optimization
        """
        start_time = time.time()
        
        # Extract domain from question/solution content
        domain = await self._extract_domain_from_content(question, solution)
        difficulty = await self._assess_content_difficulty(question, solution)
        
        logger.info("Routing to optimal RLT teacher",
                   domain=domain,
                   difficulty=difficulty,
                   student_model=student_model,
                   student_capability=student_capability)
        
        # Discover qualified teachers
        candidates = await self.discover_rlt_teachers(domain, difficulty)
        
        if not candidates:
            logger.warning("No RLT teachers found", domain=domain, difficulty=difficulty)
            return self._create_fallback_selection(student_model, domain)
        
        # Assess student requirements
        student_requirements = await self._assess_student_requirements(
            student_model, question, solution, student_capability
        )
        
        # Score and rank candidates
        for candidate in candidates:
            candidate.rlt_score = candidate.calculate_rlt_score(
                RLTRoutingStrategy.DENSE_REWARD_OPTIMIZED,
                student_capability
            )
        
        # Sort by RLT score
        candidates.sort(key=lambda c: c.rlt_score, reverse=True)
        
        # Select best teacher
        best_teacher = candidates[0]
        
        # Predict performance
        predicted_improvement = await self._predict_teaching_effectiveness(
            best_teacher, student_requirements, question, solution
        )
        
        # Create selection result
        selection = RLTTeacherSelection(
            student_model_id=student_model,
            domain=domain,
            teacher_candidates=[c.model_id for c in candidates[:5]],
            selected_teacher=UUID(best_teacher.model_id) if len(best_teacher.model_id) == 36 else uuid4(),
            selected_rlt_teacher=best_teacher,
            predicted_improvement=predicted_improvement,
            quality_confidence=best_teacher.rlt_score,
            target_explanation_quality=self.rlt_config.quality_threshold,
            student_current_capability=student_capability,
            rlt_config=self.rlt_config
        )
        
        # Store pairing for learning
        await self._record_teacher_student_pairing(selection, question, solution)
        
        routing_time = time.time() - start_time
        
        logger.info("Optimal RLT teacher selected",
                   selected_teacher=best_teacher.model_id,
                   predicted_improvement=predicted_improvement,
                   quality_confidence=best_teacher.rlt_score,
                   routing_time=f"{routing_time:.3f}s")
        
        return selection
    
    async def track_explanation_quality_scores(self, teacher_id: str, quality_metrics: Dict[str, float]):
        """Track explanation quality scores for teacher performance monitoring"""
        
        if teacher_id not in self.explanation_quality_tracker:
            self.explanation_quality_tracker[teacher_id] = []
        
        # Extract quality scores
        overall_quality = quality_metrics.get('overall_quality', 0.0)
        coherence = quality_metrics.get('coherence', 0.0)
        logical_continuity = quality_metrics.get('logical_continuity', 0.0)
        
        # Store quality metrics
        self.explanation_quality_tracker[teacher_id].append(overall_quality)
        
        # Update teacher performance if tracked
        if teacher_id in self.teacher_performance_history:
            self.teacher_performance_history[teacher_id].append(overall_quality)
        else:
            self.teacher_performance_history[teacher_id] = [overall_quality]
        
        # Keep only recent history (last 50 scores)
        if len(self.explanation_quality_tracker[teacher_id]) > 50:
            self.explanation_quality_tracker[teacher_id] = \
                self.explanation_quality_tracker[teacher_id][-50:]
        
        if len(self.teacher_performance_history[teacher_id]) > 50:
            self.teacher_performance_history[teacher_id] = \
                self.teacher_performance_history[teacher_id][-50:]
        
        # Calculate running averages
        avg_quality = np.mean(self.explanation_quality_tracker[teacher_id])
        
        logger.info("Teacher quality scores updated",
                   teacher_id=teacher_id,
                   current_quality=overall_quality,
                   avg_quality=avg_quality,
                   score_count=len(self.explanation_quality_tracker[teacher_id]))
        
        # Detect quality degradation
        await self._detect_quality_issues(teacher_id)
    
    async def get_teacher_specialization_insights(self, domain: str) -> Dict[str, Any]:
        """Get insights into teacher specializations for a domain"""
        
        candidates = await self.discover_rlt_teachers(domain)
        
        if not candidates:
            return {"domain": domain, "teachers_available": 0}
        
        # Calculate specialization metrics
        specialization_scores = [c.domain_expertise_level for c in candidates]
        quality_scores = [c.explanation_quality_score for c in candidates]
        reward_scores = [c.dense_reward_effectiveness for c in candidates]
        
        insights = {
            "domain": domain,
            "teachers_available": len(candidates),
            "specialization_distribution": {
                "avg_expertise": np.mean(specialization_scores),
                "min_expertise": np.min(specialization_scores),
                "max_expertise": np.max(specialization_scores),
                "std_expertise": np.std(specialization_scores)
            },
            "quality_metrics": {
                "avg_explanation_quality": np.mean(quality_scores),
                "avg_dense_reward_effectiveness": np.mean(reward_scores),
                "top_performer": max(candidates, key=lambda c: c.rlt_score).model_id
            },
            "recommendations": await self._generate_domain_recommendations(candidates, domain)
        }
        
        return insights
    
    # Private helper methods
    
    async def _discover_local_rlt_teachers(self, domain: str, difficulty: float) -> List[RLTTeacherCandidate]:
        """Discover RLT teachers from local registry"""
        candidates = []
        
        # Get domain specialists from registry
        specialist_ids = await self.model_registry.discover_specialists(domain)
        
        for teacher_id in specialist_ids:
            model_details = await self.model_registry.get_model_details(teacher_id)
            
            if model_details and getattr(model_details, 'model_type', None) == ModelType.TEACHER:
                # Create RLT teacher candidate
                candidate = await self._create_rlt_candidate(teacher_id, model_details, ModelSource.LOCAL_REGISTRY)
                
                # Filter by difficulty compatibility
                if self._is_difficulty_compatible(candidate, difficulty):
                    candidates.append(candidate)
        
        return candidates
    
    async def _discover_federated_rlt_teachers(self, domain: str, difficulty: float) -> List[RLTTeacherCandidate]:
        """Discover RLT teachers from federated network"""
        candidates = []
        
        # Simulate federated discovery (would integrate with actual federation in production)
        federated_teachers = [
            {
                "teacher_id": f"fed_rlt_teacher_{domain}_01",
                "name": f"Federated {domain.title()} RLT Specialist",
                "specialization": domain,
                "explanation_quality": 0.88,
                "dense_reward_effectiveness": 0.85,
                "domain_expertise": 0.92,
                "avg_rlt_reward": 1.2
            },
            {
                "teacher_id": f"fed_rlt_teacher_{domain}_02", 
                "name": f"Advanced {domain.title()} RLT Teacher",
                "specialization": domain,
                "explanation_quality": 0.91,
                "dense_reward_effectiveness": 0.89,
                "domain_expertise": 0.87,
                "avg_rlt_reward": 1.15
            }
        ]
        
        for teacher_data in federated_teachers:
            if self._matches_difficulty_requirement(teacher_data, difficulty):
                candidate = RLTTeacherCandidate(
                    model_id=teacher_data["teacher_id"],
                    name=teacher_data["name"],
                    specialization=teacher_data["specialization"],
                    model_type=ModelType.TEACHER,
                    source=ModelSource.P2P_NETWORK,
                    explanation_quality_score=teacher_data["explanation_quality"],
                    dense_reward_effectiveness=teacher_data["dense_reward_effectiveness"],
                    domain_expertise_level=teacher_data["domain_expertise"],
                    avg_rlt_reward=teacher_data["avg_rlt_reward"],
                    performance_score=teacher_data["explanation_quality"],
                    teaching_effectiveness=teacher_data["explanation_quality"],
                    compatibility_score=0.9
                )
                candidates.append(candidate)
        
        return candidates
    
    async def _discover_marketplace_rlt_teachers(self, domain: str, difficulty: float) -> List[RLTTeacherCandidate]:
        """Discover RLT teachers from marketplace"""
        candidates = []
        
        # Simulate marketplace RLT teacher discovery
        marketplace_teachers = [
            {
                "teacher_id": f"market_rlt_{domain}_premium",
                "name": f"Premium {domain.title()} RLT Teacher",
                "specialization": domain,
                "explanation_quality": 0.94,
                "dense_reward_effectiveness": 0.92,
                "domain_expertise": 0.95,
                "cost_per_explanation": 0.05,
                "provider_reputation": 0.95
            }
        ]
        
        for teacher_data in marketplace_teachers:
            if difficulty >= 0.7:  # Premium teachers for high difficulty
                candidate = RLTTeacherCandidate(
                    model_id=teacher_data["teacher_id"],
                    name=teacher_data["name"],
                    specialization=teacher_data["specialization"],
                    model_type=ModelType.TEACHER,
                    source=ModelSource.MARKETPLACE,
                    explanation_quality_score=teacher_data["explanation_quality"],
                    dense_reward_effectiveness=teacher_data["dense_reward_effectiveness"],
                    domain_expertise_level=teacher_data["domain_expertise"],
                    performance_score=teacher_data["explanation_quality"],
                    teaching_effectiveness=teacher_data["explanation_quality"],
                    compatibility_score=0.95,
                    cost_per_token=teacher_data["cost_per_explanation"],
                    provider_reputation=teacher_data["provider_reputation"]
                )
                candidates.append(candidate)
        
        return candidates
    
    async def _create_rlt_candidate(self, teacher_id: str, model_details: Any, source: ModelSource) -> RLTTeacherCandidate:
        """Create RLT teacher candidate from model details"""
        
        # Get historical performance if available
        quality_history = self.explanation_quality_tracker.get(teacher_id, [])
        reward_history = self.dense_reward_history.get(teacher_id, [])
        
        avg_quality = np.mean(quality_history) if quality_history else 0.7
        avg_reward = np.mean([r.get('total_reward', 0.0) for r in reward_history]) if reward_history else 0.8
        
        return RLTTeacherCandidate(
            model_id=teacher_id,
            name=getattr(model_details, 'name', teacher_id),
            specialization=getattr(model_details, 'specialization', 'general'),
            model_type=ModelType.TEACHER,
            source=source,
            explanation_quality_score=avg_quality,
            dense_reward_effectiveness=min(avg_reward, 1.0),
            domain_expertise_level=getattr(model_details, 'performance_score', 0.8),
            student_comprehension_score=avg_quality * 0.9,  # Correlate with quality
            adaptive_teaching_ability=0.8,  # Default for established teachers
            performance_score=getattr(model_details, 'performance_score', 0.8),
            teaching_effectiveness=getattr(model_details, 'performance_score', 0.8),
            compatibility_score=0.8,
            avg_rlt_reward=avg_reward,
            successful_distillations=len(quality_history),
            explanation_coherence=avg_quality * 0.95,
            logical_continuity=avg_quality * 0.9,
            think_token_quality=avg_quality * 0.85
        )
    
    async def _filter_rlt_teachers(self, candidates: List[RLTTeacherCandidate], 
                                 domain: str, difficulty: float) -> List[RLTTeacherCandidate]:
        """Filter and rank RLT teacher candidates"""
        
        qualified = []
        
        for candidate in candidates:
            # Quality thresholds
            if candidate.explanation_quality_score < 0.6:
                continue
            
            if candidate.dense_reward_effectiveness < 0.5:
                continue
            
            # Domain matching
            if candidate.specialization != domain and candidate.specialization != "general":
                continue
            
            # Difficulty compatibility
            if not self._is_difficulty_compatible(candidate, difficulty):
                continue
            
            qualified.append(candidate)
        
        # Rank by overall RLT capability
        for candidate in qualified:
            candidate.rlt_score = candidate.calculate_rlt_score(RLTRoutingStrategy.DENSE_REWARD_OPTIMIZED)
        
        qualified.sort(key=lambda c: c.rlt_score, reverse=True)
        
        return qualified[:10]  # Top 10 candidates
    
    def _is_difficulty_compatible(self, candidate: RLTTeacherCandidate, difficulty: float) -> bool:
        """Check if teacher is compatible with difficulty level"""
        
        # Teachers should handle difficulty within their expertise range
        expertise = candidate.domain_expertise_level
        
        # Allow some flexibility in matching
        min_difficulty = max(0.0, expertise - 0.3)
        max_difficulty = min(1.0, expertise + 0.2)
        
        return min_difficulty <= difficulty <= max_difficulty
    
    def _matches_difficulty_requirement(self, teacher_data: Dict[str, Any], difficulty: float) -> bool:
        """Check if teacher data matches difficulty requirement"""
        expertise = teacher_data.get("domain_expertise", 0.8)
        return abs(expertise - difficulty) <= 0.3
    
    async def _extract_domain_from_content(self, question: str, solution: str) -> str:
        """Extract domain from question and solution content"""
        content = f"{question} {solution}".lower()
        
        # Domain keywords
        domain_keywords = {
            "mathematics": ["math", "equation", "theorem", "calculus", "algebra", "geometry", "derivative", "integral"],
            "physics": ["physics", "force", "energy", "momentum", "quantum", "electromagnetic", "newton"],
            "chemistry": ["chemistry", "chemical", "molecule", "reaction", "bond", "atom", "ion"],
            "biology": ["biology", "cell", "protein", "gene", "organism", "evolution", "dna"],
            "computer_science": ["algorithm", "programming", "computer", "software", "code", "data structure"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content for keyword in keywords):
                return domain
        
        return "general"
    
    async def _assess_content_difficulty(self, question: str, solution: str) -> float:
        """Assess difficulty level of question-solution pair"""
        
        # Simple heuristic-based difficulty assessment
        difficulty_score = 0.5  # Base difficulty
        
        # Length factors
        question_length = len(question.split())
        solution_length = len(solution.split())
        
        if question_length > 50 or solution_length > 100:
            difficulty_score += 0.2
        
        # Complexity indicators
        complex_terms = ["advanced", "complex", "difficult", "challenging", "prove", "derive", "optimize"]
        content = f"{question} {solution}".lower()
        
        complexity_count = sum(1 for term in complex_terms if term in content)
        difficulty_score += min(0.3, complexity_count * 0.1)
        
        # Mathematical complexity
        math_indicators = ["∫", "∑", "∂", "lim", "theorem", "proof", "≥", "≤", "∞"]
        math_count = sum(1 for indicator in math_indicators if indicator in content)
        difficulty_score += min(0.2, math_count * 0.05)
        
        return min(1.0, difficulty_score)
    
    async def _assess_student_requirements(self, student_model: str, question: str, 
                                         solution: str, capability: float) -> Dict[str, Any]:
        """Assess student learning requirements"""
        
        return {
            "student_model": student_model,
            "current_capability": capability,
            "content_difficulty": await self._assess_content_difficulty(question, solution),
            "domain": await self._extract_domain_from_content(question, solution),
            "learning_style": "adaptive",  # Default
            "target_improvement": 0.2,  # 20% improvement target
            "comprehension_threshold": 0.75
        }
    
    async def _predict_teaching_effectiveness(self, teacher: RLTTeacherCandidate, 
                                            requirements: Dict[str, Any],
                                            question: str, solution: str) -> float:
        """Predict teaching effectiveness for student-teacher pairing"""
        
        # Base effectiveness from teacher capabilities
        base_effectiveness = (
            teacher.explanation_quality_score * 0.3 +
            teacher.dense_reward_effectiveness * 0.3 +
            teacher.student_comprehension_score * 0.2 +
            teacher.adaptive_teaching_ability * 0.2
        )
        
        # Adjust for capability match
        student_capability = requirements.get("current_capability", 0.5)
        content_difficulty = requirements.get("content_difficulty", 0.5)
        
        capability_match = 1.0 - abs(teacher.domain_expertise_level - content_difficulty)
        student_match = 1.0 - abs(teacher.adaptive_teaching_ability - student_capability)
        
        # Combine factors
        predicted_effectiveness = (
            base_effectiveness * 0.6 +
            capability_match * 0.2 +
            student_match * 0.2
        )
        
        return min(1.0, predicted_effectiveness)
    
    async def _record_teacher_student_pairing(self, selection: RLTTeacherSelection, 
                                            question: str, solution: str):
        """Record teacher-student pairing for learning"""
        
        pairing_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "teacher_id": selection.selected_rlt_teacher.model_id if selection.selected_rlt_teacher else None,
            "student_id": selection.student_model_id,
            "domain": selection.domain,
            "question": question[:100],  # Truncate for storage
            "solution": solution[:100],
            "predicted_improvement": selection.predicted_improvement,
            "quality_confidence": selection.quality_confidence
        }
        
        student_id = selection.student_model_id
        if student_id not in self.student_teacher_pairings:
            self.student_teacher_pairings[student_id] = []
        
        self.student_teacher_pairings[student_id].append(pairing_record)
        
        # Keep only recent pairings
        if len(self.student_teacher_pairings[student_id]) > 100:
            self.student_teacher_pairings[student_id] = \
                self.student_teacher_pairings[student_id][-100:]
    
    def _create_fallback_selection(self, student_model: str, domain: str) -> RLTTeacherSelection:
        """Create fallback selection when no RLT teachers available"""
        
        return RLTTeacherSelection(
            student_model_id=student_model,
            domain=domain,
            teacher_candidates=[],
            selected_teacher=None,
            predicted_improvement=0.1,  # Low improvement expected
            quality_confidence=0.3,
            target_explanation_quality=0.6,
            student_current_capability=0.5
        )
    
    async def _detect_quality_issues(self, teacher_id: str):
        """Detect quality degradation in teacher performance"""
        
        if teacher_id not in self.explanation_quality_tracker:
            return
        
        quality_scores = self.explanation_quality_tracker[teacher_id]
        
        if len(quality_scores) < 10:
            return  # Need more data
        
        # Check for declining trend
        recent_scores = quality_scores[-10:]
        older_scores = quality_scores[-20:-10] if len(quality_scores) >= 20 else quality_scores[:-10]
        
        if older_scores:
            recent_avg = np.mean(recent_scores)
            older_avg = np.mean(older_scores)
            
            if recent_avg < older_avg - 0.1:  # 10% degradation
                logger.warning("Quality degradation detected",
                             teacher_id=teacher_id,
                             recent_avg=recent_avg,
                             older_avg=older_avg,
                             degradation=older_avg - recent_avg)
    
    async def _generate_domain_recommendations(self, candidates: List[RLTTeacherCandidate], 
                                             domain: str) -> List[str]:
        """Generate recommendations for domain teacher selection"""
        
        recommendations = []
        
        if not candidates:
            recommendations.append(f"No RLT teachers available for {domain}")
            return recommendations
        
        # Top performer recommendation
        top_teacher = max(candidates, key=lambda c: c.explanation_quality_score)
        recommendations.append(
            f"Best explanation quality: {top_teacher.name} ({top_teacher.explanation_quality_score:.2f})"
        )
        
        # Dense reward recommendation
        best_reward_teacher = max(candidates, key=lambda c: c.dense_reward_effectiveness)
        recommendations.append(
            f"Best dense reward performance: {best_reward_teacher.name} ({best_reward_teacher.dense_reward_effectiveness:.2f})"
        )
        
        # Coverage analysis
        expertise_levels = [c.domain_expertise_level for c in candidates]
        if np.std(expertise_levels) > 0.2:
            recommendations.append("Good coverage across difficulty levels")
        else:
            recommendations.append("Limited difficulty level coverage - consider expanding teacher pool")
        
        return recommendations
    
    # Public interface methods
    
    async def get_rlt_routing_analytics(self) -> Dict[str, Any]:
        """Get RLT-specific routing analytics"""
        
        base_analytics = await self.get_routing_analytics()
        
        rlt_analytics = {
            "total_rlt_teachers": sum(len(teachers) for teachers in self.rlt_teacher_pool.values()),
            "domains_covered": list(self.rlt_teacher_pool.keys()),
            "teacher_quality_tracking": {
                teacher_id: {
                    "avg_quality": np.mean(scores),
                    "score_count": len(scores),
                    "recent_trend": "improving" if len(scores) >= 5 and scores[-3:] > scores[-6:-3] else "stable"
                }
                for teacher_id, scores in self.explanation_quality_tracker.items()
                if scores
            },
            "student_teacher_pairings": len(self.student_teacher_pairings),
            "dense_reward_tracking": len(self.dense_reward_history)
        }
        
        return {**base_analytics, "rlt_metrics": rlt_analytics}


# Factory function
def create_rlt_enhanced_router(model_registry: Optional[ModelRegistry] = None) -> RLTEnhancedRouter:
    """Create an RLT-enhanced model router"""
    return RLTEnhancedRouter(model_registry=model_registry)