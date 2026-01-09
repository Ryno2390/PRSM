"""
PRSM Curriculum Management System

Implements adaptive curriculum generation, learning gap assessment,
and progressive example creation for the distilled teacher model system.

Based on execution_plan.md Week 7-8 requirements.
"""

import asyncio
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID, uuid4
import structlog
import math

from pydantic import Field, BaseModel
from prsm.core.models import Curriculum, TeacherModel, LearningSession, PRSMBaseModel
from .teacher_model import StudentCapabilities, TeachingStrategy

logger = structlog.get_logger()


class LearningGap(PRSMBaseModel):
    """Identified gap in student learning"""
    gap_id: UUID = Field(default_factory=uuid4)
    domain: str
    skill_area: str
    severity: float = Field(ge=0.0, le=1.0)  # 0=minor, 1=critical
    confidence: float = Field(ge=0.0, le=1.0)  # Assessment confidence
    recommended_exercises: List[str] = Field(default_factory=list)
    estimated_time_to_fill: int = 30  # minutes
    prerequisite_gaps: List[UUID] = Field(default_factory=list)


class DifficultyProgression(PRSMBaseModel):
    """Progressive difficulty curve for curriculum"""
    progression_id: UUID = Field(default_factory=uuid4)
    progression_type: str = "adaptive"  # "linear", "exponential", "adaptive", "spiral"
    start_difficulty: float = Field(ge=0.0, le=1.0)
    target_difficulty: float = Field(ge=0.0, le=1.0)
    steps: int = 5
    learning_objectives: List[str] = Field(default_factory=list)
    mastery_thresholds: List[float] = Field(default_factory=list)


class ExerciseTemplate(PRSMBaseModel):
    """Template for generating progressive exercises"""
    template_id: UUID = Field(default_factory=uuid4)
    template_name: str
    domain: str
    exercise_type: str  # "problem_solving", "concept_review", "application", "synthesis"
    difficulty_range: Tuple[float, float] = (0.0, 1.0)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    learning_objectives: List[str] = Field(default_factory=list)
    estimated_time: int = 15  # minutes
    

class CurriculumMetrics(PRSMBaseModel):
    """Metrics for curriculum effectiveness assessment"""
    curriculum_id: UUID
    total_students: int = 0
    avg_completion_rate: float = 0.0
    avg_learning_gain: float = 0.0
    avg_engagement_score: float = 0.0
    difficulty_distribution: Dict[str, int] = Field(default_factory=dict)
    time_efficiency: float = 0.0  # Learning gain per minute
    drop_off_points: List[int] = Field(default_factory=list)
    success_patterns: Dict[str, float] = Field(default_factory=dict)


class CurriculumGenerator:
    """
    Advanced curriculum generator that creates adaptive, personalized
    learning experiences based on student capabilities and learning gaps.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="curriculum_generator")
        self.exercise_templates: Dict[str, List[ExerciseTemplate]] = {}
        self.curriculum_metrics: Dict[UUID, CurriculumMetrics] = {}
        self.learning_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default exercise templates
        asyncio.create_task(self._initialize_exercise_templates())
    
    async def create_adaptive_curriculum(self, student_capabilities: StudentCapabilities) -> Curriculum:
        """
        Create adaptive curriculum tailored to specific student capabilities.
        
        Args:
            student_capabilities: Assessed capabilities of the student
            
        Returns:
            Personalized curriculum with adaptive difficulty progression
        """
        self.logger.info(
            "Creating adaptive curriculum",
            student_id=str(student_capabilities.student_id),
            learning_style=student_capabilities.learning_style
        )
        
        # Assess learning gaps
        learning_gaps = await self.assess_learning_gaps(student_capabilities.current_performance)
        
        # Determine optimal difficulty progression
        difficulty_progression = await self._design_difficulty_progression(
            student_capabilities, learning_gaps
        )
        
        # Generate progressive examples
        training_examples = await self.generate_progressive_examples(
            difficulty_progression.progression_type,
            [difficulty_progression.start_difficulty, difficulty_progression.target_difficulty],
            student_capabilities
        )
        
        # Define evaluation metrics
        evaluation_metrics = await self._design_evaluation_metrics(
            student_capabilities, learning_gaps
        )
        
        # Select primary domain from strongest capability
        primary_domain = max(
            student_capabilities.current_performance.items(),
            key=lambda x: x[1]
        )[0] if student_capabilities.current_performance else "general"
        
        curriculum = Curriculum(
            teacher_id=uuid4(),  # Will be set by calling teacher
            domain=primary_domain,
            difficulty_level=difficulty_progression.start_difficulty,
            training_examples=training_examples,
            evaluation_metrics=evaluation_metrics
        )
        
        # Initialize curriculum metrics tracking
        self.curriculum_metrics[curriculum.curriculum_id] = CurriculumMetrics(
            curriculum_id=curriculum.curriculum_id
        )
        
        self.logger.info(
            "Adaptive curriculum created",
            curriculum_id=str(curriculum.curriculum_id),
            domain=primary_domain,
            examples_count=len(training_examples),
            gaps_addressed=len(learning_gaps)
        )
        
        return curriculum
    
    async def assess_learning_gaps(self, student_performance: Dict[str, float]) -> List[LearningGap]:
        """
        Analyze student performance to identify specific learning gaps.
        
        Args:
            student_performance: Current performance scores by domain
            
        Returns:
            List of identified learning gaps with severity assessment
        """
        self.logger.info("Assessing learning gaps", domains=list(student_performance.keys()))
        
        learning_gaps = []
        
        for domain, performance_score in student_performance.items():
            # Identify gaps based on performance thresholds
            if performance_score < 0.7:  # Below proficient threshold
                gap_severity = 1.0 - performance_score  # Higher score = larger gap
                
                # Determine specific skill areas within domain
                skill_areas = await self._identify_skill_areas(domain, performance_score)
                
                for skill_area, skill_performance in skill_areas.items():
                    if skill_performance < 0.6:  # Below acceptable threshold
                        gap = LearningGap(
                            domain=domain,
                            skill_area=skill_area,
                            severity=1.0 - skill_performance,
                            confidence=0.8,  # Base confidence level
                            recommended_exercises=await self._recommend_gap_exercises(
                                domain, skill_area, skill_performance
                            ),
                            estimated_time_to_fill=int(30 + (1.0 - skill_performance) * 60)
                        )
                        learning_gaps.append(gap)
        
        # Sort gaps by severity (most critical first)
        learning_gaps.sort(key=lambda g: g.severity, reverse=True)
        
        self.logger.info(
            "Learning gaps assessed",
            total_gaps=len(learning_gaps),
            critical_gaps=len([g for g in learning_gaps if g.severity > 0.7])
        )
        
        return learning_gaps
    
    async def generate_progressive_examples(
        self, 
        difficulty_curve: str, 
        difficulty_range: List[float],
        student_capabilities: StudentCapabilities
    ) -> List[Dict[str, Any]]:
        """
        Generate progressive training examples following specified difficulty curve.
        
        Args:
            difficulty_curve: Type of progression ("linear", "exponential", "adaptive")
            difficulty_range: [start_difficulty, end_difficulty]
            student_capabilities: Student's assessed capabilities
            
        Returns:
            List of progressive training examples
        """
        self.logger.info(
            "Generating progressive examples",
            difficulty_curve=difficulty_curve,
            difficulty_range=difficulty_range
        )
        
        start_diff, target_diff = difficulty_range
        examples = []
        
        # Generate difficulty levels based on curve type
        difficulty_levels = await self._generate_difficulty_levels(
            difficulty_curve, start_diff, target_diff, num_levels=8
        )
        
        # Generate examples for each difficulty level
        for i, difficulty in enumerate(difficulty_levels):
            level_examples = await self._generate_examples_for_level(
                difficulty, 
                student_capabilities, 
                examples_per_level=3
            )
            examples.extend(level_examples)
        
        # Shuffle examples within similar difficulty ranges to avoid predictability
        examples = await self._shuffle_within_difficulty_bands(examples)
        
        self.logger.info(
            "Progressive examples generated",
            total_examples=len(examples),
            difficulty_levels=len(difficulty_levels)
        )
        
        return examples
    
    async def optimize_curriculum_based_on_feedback(
        self, 
        curriculum_id: UUID, 
        student_feedback: List[Dict[str, Any]]
    ) -> Curriculum:
        """
        Optimize existing curriculum based on student feedback and performance.
        
        Args:
            curriculum_id: Curriculum to optimize
            student_feedback: Feedback from students who used curriculum
            
        Returns:
            Optimized curriculum
        """
        self.logger.info(
            "Optimizing curriculum",
            curriculum_id=str(curriculum_id),
            feedback_count=len(student_feedback)
        )
        
        # Analyze feedback patterns
        optimization_insights = await self._analyze_feedback_patterns(student_feedback)
        
        # Identify curriculum weaknesses
        weak_points = await self._identify_curriculum_weaknesses(optimization_insights)
        
        # Generate improved examples
        improved_examples = await self._generate_improved_examples(
            curriculum_id, weak_points, optimization_insights
        )
        
        # Update evaluation metrics
        updated_metrics = await self._update_evaluation_metrics(
            curriculum_id, optimization_insights
        )
        
        # Create optimized curriculum
        optimized_curriculum = await self._create_optimized_curriculum(
            curriculum_id, improved_examples, updated_metrics
        )
        
        self.logger.info(
            "Curriculum optimized",
            original_curriculum=str(curriculum_id),
            optimized_curriculum=str(optimized_curriculum.curriculum_id),
            improvements=len(weak_points)
        )
        
        return optimized_curriculum
    
    async def generate_curriculum_analytics(self, curriculum_id: UUID) -> Dict[str, Any]:
        """
        Generate comprehensive analytics for curriculum performance.
        
        Args:
            curriculum_id: Curriculum to analyze
            
        Returns:
            Analytics report with performance insights
        """
        metrics = self.curriculum_metrics.get(curriculum_id)
        if not metrics:
            return {'error': 'No metrics found for curriculum'}
        
        # Calculate performance indicators
        effectiveness_score = (
            metrics.avg_learning_gain * 0.4 +
            metrics.avg_completion_rate * 0.3 +
            metrics.avg_engagement_score * 0.2 +
            metrics.time_efficiency * 0.1
        )
        
        # Identify improvement opportunities
        improvement_opportunities = []
        if metrics.avg_completion_rate < 0.8:
            improvement_opportunities.append("Reduce curriculum difficulty or length")
        if metrics.avg_engagement_score < 0.7:
            improvement_opportunities.append("Increase interactive elements")
        if metrics.time_efficiency < 0.02:  # Learning gain per minute
            improvement_opportunities.append("Optimize content density")
        
        # Generate recommendations
        recommendations = await self._generate_curriculum_recommendations(metrics)
        
        return {
            'curriculum_id': str(curriculum_id),
            'effectiveness_score': effectiveness_score,
            'metrics': {
                'total_students': metrics.total_students,
                'avg_completion_rate': metrics.avg_completion_rate,
                'avg_learning_gain': metrics.avg_learning_gain,
                'avg_engagement_score': metrics.avg_engagement_score,
                'time_efficiency': metrics.time_efficiency
            },
            'difficulty_distribution': metrics.difficulty_distribution,
            'drop_off_points': metrics.drop_off_points,
            'improvement_opportunities': improvement_opportunities,
            'recommendations': recommendations,
            'success_patterns': metrics.success_patterns
        }
    
    # === Private Helper Methods ===
    
    async def _initialize_exercise_templates(self):
        """Initialize default exercise templates for different domains"""
        domains = ["mathematics", "science", "programming", "language", "reasoning"]
        
        for domain in domains:
            self.exercise_templates[domain] = []
            
            # Problem solving templates
            self.exercise_templates[domain].append(ExerciseTemplate(
                template_name=f"{domain}_problem_solving",
                domain=domain,
                exercise_type="problem_solving",
                difficulty_range=(0.2, 0.8),
                parameters={"complexity": "moderate", "guidance": "minimal"},
                learning_objectives=[f"solve_{domain}_problems"],
                estimated_time=20
            ))
            
            # Concept review templates
            self.exercise_templates[domain].append(ExerciseTemplate(
                template_name=f"{domain}_concept_review",
                domain=domain,
                exercise_type="concept_review",
                difficulty_range=(0.1, 0.6),
                parameters={"depth": "comprehensive", "examples": "multiple"},
                learning_objectives=[f"understand_{domain}_concepts"],
                estimated_time=15
            ))
            
            # Application templates
            self.exercise_templates[domain].append(ExerciseTemplate(
                template_name=f"{domain}_application",
                domain=domain,
                exercise_type="application",
                difficulty_range=(0.4, 0.9),
                parameters={"context": "real_world", "creativity": "encouraged"},
                learning_objectives=[f"apply_{domain}_knowledge"],
                estimated_time=25
            ))
    
    async def _design_difficulty_progression(
        self, 
        capabilities: StudentCapabilities, 
        gaps: List[LearningGap]
    ) -> DifficultyProgression:
        """Design optimal difficulty progression for student"""
        # Determine starting difficulty based on current performance
        avg_performance = sum(capabilities.current_performance.values()) / len(capabilities.current_performance)
        start_difficulty = max(0.1, avg_performance - 0.1)  # Start slightly below current level
        
        # Determine target difficulty based on capabilities and gaps
        gap_severity = sum(gap.severity for gap in gaps) / len(gaps) if gaps else 0.0
        target_difficulty = min(0.9, avg_performance + 0.3 - gap_severity * 0.2)
        
        # Choose progression type based on student characteristics
        if capabilities.learning_style == "analytical":
            progression_type = "linear"
        elif capabilities.attention_span > 0.8:
            progression_type = "exponential"
        else:
            progression_type = "adaptive"
        
        return DifficultyProgression(
            progression_type=progression_type,
            start_difficulty=start_difficulty,
            target_difficulty=target_difficulty,
            steps=6,
            learning_objectives=[f"master_{gap.skill_area}" for gap in gaps[:3]],
            mastery_thresholds=[0.7, 0.8, 0.85, 0.9, 0.9, 0.95]
        )
    
    async def _design_evaluation_metrics(
        self, 
        capabilities: StudentCapabilities, 
        gaps: List[LearningGap]
    ) -> Dict[str, float]:
        """Design evaluation metrics based on student needs"""
        base_metrics = {
            'accuracy_threshold': 0.8,
            'speed_target': 1.0,
            'consistency_requirement': 0.75,
            'engagement_minimum': 0.6
        }
        
        # Adjust metrics based on capabilities
        if capabilities.attention_span < 0.5:
            base_metrics['engagement_minimum'] = 0.8  # Higher engagement needed
        
        # Adjust for learning gaps
        if gaps and any(gap.severity > 0.7 for gap in gaps):
            base_metrics['accuracy_threshold'] = 0.7  # Lower initial threshold
        
        return base_metrics
    
    async def _identify_skill_areas(self, domain: str, performance: float) -> Dict[str, float]:
        """Identify specific skill areas within a domain"""
        # Simulate skill area breakdown
        skill_areas = {
            f"{domain}_fundamentals": performance + random.uniform(-0.1, 0.1),
            f"{domain}_application": performance + random.uniform(-0.15, 0.05),
            f"{domain}_analysis": performance + random.uniform(-0.2, 0.1),
            f"{domain}_synthesis": performance + random.uniform(-0.25, 0.15)
        }
        
        # Clamp values to valid range
        for skill in skill_areas:
            skill_areas[skill] = max(0.0, min(1.0, skill_areas[skill]))
        
        return skill_areas
    
    async def _recommend_gap_exercises(self, domain: str, skill_area: str, performance: float) -> List[str]:
        """Recommend exercises to address specific learning gaps"""
        exercises = []
        
        if performance < 0.3:
            exercises.extend([
                f"Basic {skill_area} review",
                f"Guided {skill_area} practice",
                f"Foundational {skill_area} concepts"
            ])
        elif performance < 0.6:
            exercises.extend([
                f"Intermediate {skill_area} practice",
                f"Applied {skill_area} problems",
                f"{skill_area} skill building"
            ])
        else:
            exercises.extend([
                f"Advanced {skill_area} challenges",
                f"{skill_area} mastery exercises"
            ])
        
        return exercises
    
    async def _generate_difficulty_levels(
        self, 
        curve_type: str, 
        start: float, 
        target: float, 
        num_levels: int
    ) -> List[float]:
        """Generate difficulty levels following specified curve"""
        if curve_type == "linear":
            step = (target - start) / (num_levels - 1)
            return [start + i * step for i in range(num_levels)]
        
        elif curve_type == "exponential":
            # Exponential growth curve
            levels = []
            for i in range(num_levels):
                progress = i / (num_levels - 1)
                # Use exponential curve: y = start + (target - start) * progress^2
                difficulty = start + (target - start) * (progress ** 1.5)
                levels.append(difficulty)
            return levels
        
        elif curve_type == "adaptive":
            # Adaptive curve with steeper growth in middle
            levels = []
            for i in range(num_levels):
                progress = i / (num_levels - 1)
                # S-curve for adaptive progression
                s_curve = 1 / (1 + math.exp(-10 * (progress - 0.5)))
                difficulty = start + (target - start) * s_curve
                levels.append(difficulty)
            return levels
        
        else:
            # Default to linear
            return await self._generate_difficulty_levels("linear", start, target, num_levels)
    
    async def _generate_examples_for_level(
        self, 
        difficulty: float, 
        capabilities: StudentCapabilities, 
        examples_per_level: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate examples for specific difficulty level"""
        examples = []
        
        # Get relevant domain
        primary_domain = list(capabilities.current_performance.keys())[0] if capabilities.current_performance else "general"
        
        for i in range(examples_per_level):
            example = {
                'example_id': str(uuid4()),
                'difficulty': difficulty,
                'domain': primary_domain,
                'type': random.choice(["problem_solving", "concept_review", "application"]),
                'content': f"Level {difficulty:.2f} example {i+1} for {primary_domain}",
                'expected_output': f"Solution for difficulty {difficulty:.2f}, example {i+1}",
                'learning_objectives': [f"achieve_{primary_domain}_level_{difficulty:.1f}"],
                'estimated_time_minutes': int(10 + difficulty * 25),
                'personalization_notes': f"Adapted for {capabilities.learning_style} learner",
                'hints_available': difficulty > 0.6,  # Provide hints for harder problems
                'multiple_approaches': capabilities.learning_style == "analytical"
            }
            examples.append(example)
        
        return examples
    
    async def _shuffle_within_difficulty_bands(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Shuffle examples within similar difficulty bands to avoid predictability"""
        # Group by difficulty bands
        bands = {}
        for example in examples:
            difficulty = example['difficulty']
            band = round(difficulty * 4) / 4  # Create 0.25-wide bands
            if band not in bands:
                bands[band] = []
            bands[band].append(example)
        
        # Shuffle within each band
        shuffled_examples = []
        for band in sorted(bands.keys()):
            band_examples = bands[band]
            random.shuffle(band_examples)
            shuffled_examples.extend(band_examples)
        
        return shuffled_examples
    
    async def _analyze_feedback_patterns(self, feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze student feedback to identify patterns"""
        if not feedback:
            return {}
        
        # Aggregate feedback metrics
        avg_difficulty = sum(f.get('perceived_difficulty', 0.5) for f in feedback) / len(feedback)
        avg_engagement = sum(f.get('engagement_score', 0.5) for f in feedback) / len(feedback)
        avg_clarity = sum(f.get('clarity_score', 0.5) for f in feedback) / len(feedback)
        
        # Identify common complaints
        common_issues = {}
        for f in feedback:
            issues = f.get('issues', [])
            for issue in issues:
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        # Identify success patterns
        success_factors = {}
        for f in feedback:
            if f.get('success_rating', 0) > 0.7:
                factors = f.get('success_factors', [])
                for factor in factors:
                    success_factors[factor] = success_factors.get(factor, 0) + 1
        
        return {
            'avg_difficulty': avg_difficulty,
            'avg_engagement': avg_engagement,
            'avg_clarity': avg_clarity,
            'common_issues': common_issues,
            'success_factors': success_factors,
            'total_feedback': len(feedback)
        }
    
    async def _identify_curriculum_weaknesses(self, insights: Dict[str, Any]) -> List[str]:
        """Identify curriculum weaknesses from analysis"""
        weaknesses = []
        
        if insights.get('avg_difficulty', 0.5) > 0.8:
            weaknesses.append("excessive_difficulty")
        
        if insights.get('avg_engagement', 0.5) < 0.6:
            weaknesses.append("low_engagement")
        
        if insights.get('avg_clarity', 0.5) < 0.7:
            weaknesses.append("unclear_instructions")
        
        common_issues = insights.get('common_issues', {})
        if common_issues.get('too_fast_paced', 0) > len(insights.get('total_feedback', 1)) * 0.3:
            weaknesses.append("pacing_issues")
        
        return weaknesses
    
    async def _generate_improved_examples(
        self, 
        curriculum_id: UUID, 
        weak_points: List[str], 
        insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate improved examples addressing identified weaknesses"""
        improved_examples = []
        
        # Address each weakness type
        for weakness in weak_points:
            if weakness == "excessive_difficulty":
                # Generate easier transition examples
                for i in range(3):
                    example = {
                        'example_id': str(uuid4()),
                        'difficulty': 0.4 + i * 0.1,
                        'type': 'transition_support',
                        'content': f"Simplified example {i+1} with step-by-step guidance",
                        'improvement_target': 'difficulty_reduction',
                        'additional_scaffolding': True
                    }
                    improved_examples.append(example)
            
            elif weakness == "low_engagement":
                # Generate more interactive examples
                for i in range(2):
                    example = {
                        'example_id': str(uuid4()),
                        'difficulty': 0.5,
                        'type': 'interactive_engagement',
                        'content': f"Interactive example {i+1} with gamification elements",
                        'improvement_target': 'engagement_boost',
                        'interactive_elements': True,
                        'immediate_feedback': True
                    }
                    improved_examples.append(example)
        
        return improved_examples
    
    async def _update_evaluation_metrics(
        self, 
        curriculum_id: UUID, 
        insights: Dict[str, Any]
    ) -> Dict[str, float]:
        """Update evaluation metrics based on optimization insights"""
        base_metrics = {
            'accuracy_threshold': 0.8,
            'speed_target': 1.0,
            'consistency_requirement': 0.75,
            'engagement_minimum': 0.6
        }
        
        # Adjust based on feedback
        if insights.get('avg_difficulty', 0.5) > 0.8:
            base_metrics['accuracy_threshold'] = 0.7  # Lower threshold for difficult content
        
        if insights.get('avg_engagement', 0.5) < 0.6:
            base_metrics['engagement_minimum'] = 0.8  # Require higher engagement
        
        return base_metrics
    
    async def _create_optimized_curriculum(
        self, 
        original_id: UUID, 
        improved_examples: List[Dict[str, Any]], 
        updated_metrics: Dict[str, float]
    ) -> Curriculum:
        """Create optimized curriculum with improvements"""
        return Curriculum(
            teacher_id=uuid4(),  # Will be set by calling teacher
            domain="optimized_curriculum",
            difficulty_level=0.5,
            training_examples=improved_examples,
            evaluation_metrics=updated_metrics
        )
    
    async def _generate_curriculum_recommendations(self, metrics: CurriculumMetrics) -> List[str]:
        """Generate recommendations for curriculum improvement"""
        recommendations = []
        
        if metrics.avg_completion_rate < 0.7:
            recommendations.append("Consider reducing curriculum length or complexity")
        
        if metrics.avg_engagement_score < 0.6:
            recommendations.append("Add more interactive and varied content types")
        
        if metrics.time_efficiency < 0.015:
            recommendations.append("Optimize content density and pacing")
        
        if metrics.avg_learning_gain < 0.6:
            recommendations.append("Review learning objectives alignment with content")
        
        if not recommendations:
            recommendations.append("Curriculum performing well - maintain current approach")
        
        return recommendations