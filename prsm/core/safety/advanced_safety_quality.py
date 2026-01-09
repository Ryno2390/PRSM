"""
Advanced Safety & Quality Framework for RLT Teachers

Comprehensive safety and quality assurance system for RLT (Reinforcement Learning Teachers)
that ensures safe, ethical, and effective teaching through multi-layered validation,
impact assessment, and continuous monitoring.

Key Features:
- Explanation safety validation with content filtering
- Student learning impact assessment and monitoring
- Quality degradation prevention with early detection
- Ethical teaching guidelines compliance verification
- Real-time safety monitoring and alerting
- Bias detection and mitigation
- Content appropriateness validation
- Learning trajectory analysis
"""

import asyncio
import json
import time
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
from statistics import mean, stdev
import random

import structlog

from prsm.compute.teachers.seal import SEALService
from prsm.compute.teachers.rlt.quality_monitor import QualityMetrics, QualityMonitor
from prsm.core.monitoring.rlt_performance_monitor import RLTPerformanceMonitor, RLTMetrics
from ..benchmarking.rlt_evaluation_benchmark import EvaluationProblem, TeachingEvaluationResult

logger = structlog.get_logger(__name__)


class SafetyViolationType(Enum):
    """Types of safety violations"""
    HARMFUL_CONTENT = "harmful_content"
    INAPPROPRIATE_LANGUAGE = "inappropriate_language"
    BIAS_DETECTED = "bias_detected"
    MISINFORMATION = "misinformation"
    PRIVACY_VIOLATION = "privacy_violation"
    ETHICAL_VIOLATION = "ethical_violation"
    QUALITY_DEGRADATION = "quality_degradation"
    NEGATIVE_IMPACT = "negative_impact"


class SafetySeverity(Enum):
    """Severity levels for safety violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityDegradationType(Enum):
    """Types of quality degradation"""
    EXPLANATION_CLARITY = "explanation_clarity"
    LOGICAL_COHERENCE = "logical_coherence"
    FACTUAL_ACCURACY = "factual_accuracy"
    PEDAGOGICAL_EFFECTIVENESS = "pedagogical_effectiveness"
    STUDENT_ENGAGEMENT = "student_engagement"
    COMPREHENSION_SUPPORT = "comprehension_support"


@dataclass
class SafetyViolation:
    """Safety violation record"""
    violation_id: str
    violation_type: SafetyViolationType
    severity: SafetySeverity
    description: str
    detected_content: str
    context: Dict[str, Any]
    timestamp: datetime
    teacher_id: str
    student_id: Optional[str]
    mitigation_applied: bool
    mitigation_details: Optional[str]


@dataclass
class LearningImpactAssessment:
    """Student learning impact assessment"""
    assessment_id: str
    student_id: str
    teacher_id: str
    assessment_period: Tuple[datetime, datetime]
    
    # Learning progress metrics
    comprehension_improvement: float
    engagement_level: float
    learning_velocity: float
    retention_quality: float
    
    # Safety and wellbeing metrics
    confusion_indicators: float
    frustration_level: float
    confidence_change: float
    motivation_impact: float
    
    # Quality metrics
    explanation_effectiveness: float
    teaching_quality_score: float
    adaptive_response_quality: float
    
    # Risk indicators
    negative_impact_risk: float
    learning_obstacle_score: float
    dependency_risk: float
    
    timestamp: datetime
    assessment_notes: Optional[str]


@dataclass
class QualityDegradationAlert:
    """Quality degradation detection alert"""
    alert_id: str
    degradation_type: QualityDegradationType
    severity: SafetySeverity
    teacher_id: str
    detection_timestamp: datetime
    
    # Degradation metrics
    baseline_quality: float
    current_quality: float
    degradation_percentage: float
    trend_direction: str  # "declining", "stable", "improving"
    
    # Evidence
    evidence_samples: List[Dict[str, Any]]
    statistical_significance: float
    confidence_level: float
    
    # Mitigation
    recommended_actions: List[str]
    auto_mitigation_applied: bool
    mitigation_effectiveness: Optional[float]


@dataclass
class EthicalGuidelinesViolation:
    """Ethical guidelines violation record"""
    violation_id: str
    guideline_category: str
    violation_description: str
    severity: SafetySeverity
    teacher_id: str
    context: Dict[str, Any]
    detection_method: str
    timestamp: datetime
    remediation_required: bool
    remediation_plan: Optional[Dict[str, Any]]


@dataclass
class SafetyConfiguration:
    """Configuration for safety and quality systems"""
    # Content safety thresholds
    harmful_content_threshold: float = 0.1
    bias_detection_threshold: float = 0.3
    misinformation_threshold: float = 0.2
    
    # Quality monitoring
    quality_degradation_threshold: float = 0.15  # 15% degradation triggers alert
    impact_assessment_frequency: int = 24  # hours
    
    # Safety monitoring
    safety_check_frequency: int = 1  # Real-time (every explanation)
    violation_escalation_threshold: int = 3  # violations before escalation
    
    # Ethical guidelines
    bias_categories: List[str] = field(default_factory=lambda: [
        "gender", "race", "age", "religion", "nationality", "socioeconomic"
    ])
    
    # Student protection
    student_privacy_protection: bool = True
    age_appropriate_content: bool = True
    emotional_safety_monitoring: bool = True


class ExplanationSafetyValidator:
    """Validates explanations for safety and appropriateness"""
    
    def __init__(self, config: SafetyConfiguration):
        self.config = config
        self.harmful_patterns = self._load_harmful_patterns()
        self.bias_detectors = self._initialize_bias_detectors()
        self.content_filters = self._initialize_content_filters()
        
    async def validate_explanation_safety(
        self,
        explanation: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, List[SafetyViolation]]:
        """Comprehensive safety validation of explanation content"""
        
        violations = []
        
        # Check for harmful content
        harmful_violations = await self._detect_harmful_content(explanation, context)
        violations.extend(harmful_violations)
        
        # Check for bias
        bias_violations = await self._detect_bias(explanation, context)
        violations.extend(bias_violations)
        
        # Check for misinformation
        misinformation_violations = await self._detect_misinformation(explanation, context)
        violations.extend(misinformation_violations)
        
        # Check age appropriateness
        if self.config.age_appropriate_content:
            age_violations = await self._check_age_appropriateness(explanation, context)
            violations.extend(age_violations)
        
        # Check privacy compliance
        if self.config.student_privacy_protection:
            privacy_violations = await self._check_privacy_compliance(explanation, context)
            violations.extend(privacy_violations)
        
        # Determine overall safety
        is_safe = all(v.severity != SafetySeverity.CRITICAL for v in violations)
        
        return is_safe, violations
    
    async def _detect_harmful_content(
        self,
        explanation: str,
        context: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """Detect harmful content in explanations"""
        violations = []
        
        # Check against harmful patterns
        for pattern_type, patterns in self.harmful_patterns.items():
            for pattern in patterns:
                if re.search(pattern, explanation, re.IGNORECASE):
                    violation = SafetyViolation(
                        violation_id=f"harm_{int(time.time() * 1000)}",
                        violation_type=SafetyViolationType.HARMFUL_CONTENT,
                        severity=self._assess_harm_severity(pattern_type),
                        description=f"Detected {pattern_type} content",
                        detected_content=explanation[:200],
                        context=context,
                        timestamp=datetime.now(timezone.utc),
                        teacher_id=context.get("teacher_id", "unknown"),
                        student_id=context.get("student_id"),
                        mitigation_applied=False,
                        mitigation_details=None
                    )
                    violations.append(violation)
        
        # Advanced content analysis (simplified simulation)
        content_toxicity = await self._analyze_content_toxicity(explanation)
        if content_toxicity > self.config.harmful_content_threshold:
            violation = SafetyViolation(
                violation_id=f"toxic_{int(time.time() * 1000)}",
                violation_type=SafetyViolationType.HARMFUL_CONTENT,
                severity=SafetySeverity.HIGH if content_toxicity > 0.7 else SafetySeverity.MEDIUM,
                description=f"High toxicity score: {content_toxicity:.3f}",
                detected_content=explanation[:200],
                context=context,
                timestamp=datetime.now(timezone.utc),
                teacher_id=context.get("teacher_id", "unknown"),
                student_id=context.get("student_id"),
                mitigation_applied=False,
                mitigation_details=None
            )
            violations.append(violation)
        
        return violations
    
    async def _detect_bias(
        self,
        explanation: str,
        context: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """Detect bias in explanations"""
        violations = []
        
        for bias_category in self.config.bias_categories:
            bias_score = await self._calculate_bias_score(explanation, bias_category)
            
            if bias_score > self.config.bias_detection_threshold:
                violation = SafetyViolation(
                    violation_id=f"bias_{bias_category}_{int(time.time() * 1000)}",
                    violation_type=SafetyViolationType.BIAS_DETECTED,
                    severity=SafetySeverity.HIGH if bias_score > 0.7 else SafetySeverity.MEDIUM,
                    description=f"Detected {bias_category} bias (score: {bias_score:.3f})",
                    detected_content=explanation[:200],
                    context={**context, "bias_category": bias_category, "bias_score": bias_score},
                    timestamp=datetime.now(timezone.utc),
                    teacher_id=context.get("teacher_id", "unknown"),
                    student_id=context.get("student_id"),
                    mitigation_applied=False,
                    mitigation_details=None
                )
                violations.append(violation)
        
        return violations
    
    async def _detect_misinformation(
        self,
        explanation: str,
        context: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """Detect potential misinformation"""
        violations = []
        
        # Factual consistency check
        factual_accuracy = await self._verify_factual_accuracy(explanation, context)
        
        if factual_accuracy < (1.0 - self.config.misinformation_threshold):
            violation = SafetyViolation(
                violation_id=f"misinfo_{int(time.time() * 1000)}",
                violation_type=SafetyViolationType.MISINFORMATION,
                severity=SafetySeverity.HIGH,
                description=f"Low factual accuracy: {factual_accuracy:.3f}",
                detected_content=explanation[:200],
                context={**context, "factual_accuracy": factual_accuracy},
                timestamp=datetime.now(timezone.utc),
                teacher_id=context.get("teacher_id", "unknown"),
                student_id=context.get("student_id"),
                mitigation_applied=False,
                mitigation_details=None
            )
            violations.append(violation)
        
        return violations
    
    async def _check_age_appropriateness(
        self,
        explanation: str,
        context: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """Check if content is age-appropriate"""
        violations = []
        
        student_age = context.get("student_age")
        if student_age:
            appropriateness_score = await self._assess_age_appropriateness(explanation, student_age)
            
            if appropriateness_score < 0.7:  # Below acceptable threshold
                violation = SafetyViolation(
                    violation_id=f"age_{int(time.time() * 1000)}",
                    violation_type=SafetyViolationType.INAPPROPRIATE_LANGUAGE,
                    severity=SafetySeverity.MEDIUM,
                    description=f"Content may not be age-appropriate for {student_age} years old",
                    detected_content=explanation[:200],
                    context={**context, "appropriateness_score": appropriateness_score},
                    timestamp=datetime.now(timezone.utc),
                    teacher_id=context.get("teacher_id", "unknown"),
                    student_id=context.get("student_id"),
                    mitigation_applied=False,
                    mitigation_details=None
                )
                violations.append(violation)
        
        return violations
    
    async def _check_privacy_compliance(
        self,
        explanation: str,
        context: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """Check for privacy violations"""
        violations = []
        
        # Check for personal information exposure
        privacy_risk = await self._detect_privacy_violations(explanation)
        
        if privacy_risk > 0.3:
            violation = SafetyViolation(
                violation_id=f"privacy_{int(time.time() * 1000)}",
                violation_type=SafetyViolationType.PRIVACY_VIOLATION,
                severity=SafetySeverity.HIGH,
                description="Potential privacy information exposure",
                detected_content=explanation[:200],
                context={**context, "privacy_risk": privacy_risk},
                timestamp=datetime.now(timezone.utc),
                teacher_id=context.get("teacher_id", "unknown"),
                student_id=context.get("student_id"),
                mitigation_applied=False,
                mitigation_details=None
            )
            violations.append(violation)
        
        return violations
    
    def _load_harmful_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for harmful content detection"""
        return {
            "violence": [
                r"\b(kill|murder|hurt|harm|attack|violence)\b",
                r"\b(weapon|gun|knife|bomb)\b"
            ],
            "hate_speech": [
                r"\b(hate|despise|inferior|superior)\s+(people|group|race)\b",
                r"\bstupid\s+(people|students|children)\b"
            ],
            "discrimination": [
                r"\b(all|every)\s+(women|men|girls|boys)\s+are\b",
                r"\b(never|always)\s+(trust|believe)\s+(people|person)\b"
            ],
            "inappropriate": [
                r"\b(drugs|alcohol|smoking)\s+(are|is)\s+(good|great|awesome)\b",
                r"\b(cheat|lie|steal)\s+(to|for)\s+(success|win)\b"
            ]
        }
    
    def _initialize_bias_detectors(self) -> Dict[str, Any]:
        """Initialize bias detection systems"""
        return {
            "gender": {"keywords": ["he", "she", "man", "woman", "male", "female"]},
            "race": {"keywords": ["race", "ethnicity", "color", "origin"]},
            "age": {"keywords": ["old", "young", "age", "generation"]},
            "religion": {"keywords": ["religion", "faith", "belief", "church"]},
            "nationality": {"keywords": ["country", "nation", "nationality", "citizenship"]},
            "socioeconomic": {"keywords": ["poor", "rich", "wealth", "class", "income"]}
        }
    
    def _initialize_content_filters(self) -> Dict[str, Any]:
        """Initialize content filtering systems"""
        return {
            "profanity_filter": True,
            "toxicity_detector": True,
            "bias_detector": True,
            "factual_checker": True
        }
    
    def _assess_harm_severity(self, pattern_type: str) -> SafetySeverity:
        """Assess severity of harmful content"""
        severity_map = {
            "violence": SafetySeverity.CRITICAL,
            "hate_speech": SafetySeverity.HIGH,
            "discrimination": SafetySeverity.HIGH,
            "inappropriate": SafetySeverity.MEDIUM
        }
        return severity_map.get(pattern_type, SafetySeverity.LOW)
    
    async def _analyze_content_toxicity(self, explanation: str) -> float:
        """Analyze content toxicity (mock implementation)"""
        # Simplified toxicity analysis
        toxic_indicators = [
            "stupid", "idiot", "worthless", "useless", "failure",
            "never", "always", "impossible", "can't do", "give up"
        ]
        
        toxic_count = sum(1 for indicator in toxic_indicators if indicator in explanation.lower())
        word_count = len(explanation.split())
        
        return min(1.0, toxic_count / max(word_count * 0.1, 1))
    
    async def _calculate_bias_score(self, explanation: str, bias_category: str) -> float:
        """Calculate bias score for a category (mock implementation)"""
        detector = self.bias_detectors.get(bias_category, {})
        keywords = detector.get("keywords", [])
        
        # Check for biased language patterns
        bias_patterns = [
            r"\b(all|every|no)\s+" + keyword + r"\b" for keyword in keywords
        ]
        
        bias_count = 0
        for pattern in bias_patterns:
            if re.search(pattern, explanation, re.IGNORECASE):
                bias_count += 1
        
        # Simple bias scoring
        return min(1.0, bias_count * 0.3)
    
    async def _verify_factual_accuracy(self, explanation: str, context: Dict[str, Any]) -> float:
        """Verify factual accuracy of explanation (mock implementation)"""
        # Mock factual verification
        domain = context.get("domain", "general")
        
        # Check for obviously false statements
        false_indicators = [
            r"2\s*\+\s*2\s*=\s*5",
            r"earth\s+is\s+flat",
            r"sun\s+revolves\s+around\s+earth"
        ]
        
        for indicator in false_indicators:
            if re.search(indicator, explanation, re.IGNORECASE):
                return 0.1  # Very low accuracy
        
        # Simulate domain-specific accuracy
        domain_accuracy = {
            "mathematics": 0.95,
            "physics": 0.92,
            "chemistry": 0.90,
            "biology": 0.88,
            "general": 0.85
        }
        
        return domain_accuracy.get(domain, 0.80)
    
    async def _assess_age_appropriateness(self, explanation: str, student_age: int) -> float:
        """Assess age appropriateness of content"""
        # Age-inappropriate indicators by age group
        if student_age < 12:
            inappropriate = ["complex", "advanced", "sophisticated", "intricate"]
        elif student_age < 16:
            inappropriate = ["extremely complex", "graduate-level", "research-grade"]
        else:
            inappropriate = []  # Most content appropriate for 16+
        
        inappropriate_count = sum(1 for word in inappropriate if word in explanation.lower())
        complexity_penalty = inappropriate_count * 0.2
        
        return max(0.0, 1.0 - complexity_penalty)
    
    async def _detect_privacy_violations(self, explanation: str) -> float:
        """Detect privacy violations in content"""
        # Privacy-sensitive patterns
        privacy_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{3}-\d{3}-\d{4}\b",  # Phone
            r"\b\d+\s+[A-Za-z\s]+\s+(Street|St|Avenue|Ave|Road|Rd)\b"  # Address
        ]
        
        violations = sum(1 for pattern in privacy_patterns 
                        if re.search(pattern, explanation))
        
        return min(1.0, violations * 0.3)


class StudentLearningImpactAssessor:
    """Assesses impact of teaching on student learning and wellbeing"""
    
    def __init__(self, config: SafetyConfiguration):
        self.config = config
        self.student_profiles = {}
        self.impact_history = defaultdict(list)
        
    async def assess_learning_impact(
        self,
        student_id: str,
        teacher_id: str,
        interaction_data: Dict[str, Any],
        assessment_period_hours: int = 24
    ) -> LearningImpactAssessment:
        """Comprehensive assessment of teaching impact on student learning"""
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=assessment_period_hours)
        
        # Collect interaction data for the period
        period_interactions = self._get_period_interactions(
            student_id, teacher_id, start_time, end_time
        )
        
        # Assess learning progress
        progress_metrics = await self._assess_learning_progress(
            student_id, period_interactions
        )
        
        # Assess safety and wellbeing
        wellbeing_metrics = await self._assess_student_wellbeing(
            student_id, period_interactions
        )
        
        # Assess teaching quality
        quality_metrics = await self._assess_teaching_quality(
            teacher_id, period_interactions
        )
        
        # Calculate risk indicators
        risk_metrics = await self._calculate_risk_indicators(
            student_id, teacher_id, period_interactions
        )
        
        assessment = LearningImpactAssessment(
            assessment_id=f"impact_{student_id}_{int(time.time())}",
            student_id=student_id,
            teacher_id=teacher_id,
            assessment_period=(start_time, end_time),
            
            # Learning progress
            comprehension_improvement=progress_metrics["comprehension_improvement"],
            engagement_level=progress_metrics["engagement_level"],
            learning_velocity=progress_metrics["learning_velocity"],
            retention_quality=progress_metrics["retention_quality"],
            
            # Safety and wellbeing
            confusion_indicators=wellbeing_metrics["confusion_indicators"],
            frustration_level=wellbeing_metrics["frustration_level"],
            confidence_change=wellbeing_metrics["confidence_change"],
            motivation_impact=wellbeing_metrics["motivation_impact"],
            
            # Quality metrics
            explanation_effectiveness=quality_metrics["explanation_effectiveness"],
            teaching_quality_score=quality_metrics["teaching_quality_score"],
            adaptive_response_quality=quality_metrics["adaptive_response_quality"],
            
            # Risk indicators
            negative_impact_risk=risk_metrics["negative_impact_risk"],
            learning_obstacle_score=risk_metrics["learning_obstacle_score"],
            dependency_risk=risk_metrics["dependency_risk"],
            
            timestamp=end_time,
            assessment_notes=None
        )
        
        # Store assessment
        self.impact_history[student_id].append(assessment)
        
        return assessment
    
    def _get_period_interactions(
        self,
        student_id: str,
        teacher_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get interactions for assessment period (mock implementation)"""
        # Mock interaction data
        interactions = []
        num_interactions = random.randint(5, 20)
        
        for i in range(num_interactions):
            interaction_time = start_time + timedelta(
                hours=random.uniform(0, (end_time - start_time).total_seconds() / 3600)
            )
            
            interaction = {
                "timestamp": interaction_time,
                "student_id": student_id,
                "teacher_id": teacher_id,
                "question_difficulty": random.uniform(0.3, 0.9),
                "explanation_quality": random.uniform(0.6, 0.95),
                "student_response_quality": random.uniform(0.4, 0.9),
                "engagement_indicators": {
                    "time_spent": random.uniform(60, 300),  # seconds
                    "follow_up_questions": random.randint(0, 3),
                    "completion_status": random.choice(["completed", "partial", "abandoned"])
                },
                "confusion_markers": random.uniform(0.0, 0.3),
                "satisfaction_score": random.uniform(0.6, 1.0)
            }
            interactions.append(interaction)
        
        return sorted(interactions, key=lambda x: x["timestamp"])
    
    async def _assess_learning_progress(
        self,
        student_id: str,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Assess student learning progress"""
        
        if not interactions:
            return {
                "comprehension_improvement": 0.0,
                "engagement_level": 0.0,
                "learning_velocity": 0.0,
                "retention_quality": 0.0
            }
        
        # Calculate comprehension improvement
        early_responses = [i["student_response_quality"] for i in interactions[:len(interactions)//2]]
        later_responses = [i["student_response_quality"] for i in interactions[len(interactions)//2:]]
        
        early_avg = mean(early_responses) if early_responses else 0.5
        later_avg = mean(later_responses) if later_responses else 0.5
        comprehension_improvement = (later_avg - early_avg) / early_avg if early_avg > 0 else 0.0
        
        # Calculate engagement level
        engagement_scores = []
        for interaction in interactions:
            engagement_factors = interaction["engagement_indicators"]
            time_score = min(1.0, engagement_factors["time_spent"] / 180.0)  # Normalize to 3 minutes
            questions_score = min(1.0, engagement_factors["follow_up_questions"] / 2.0)
            completion_score = 1.0 if engagement_factors["completion_status"] == "completed" else 0.5
            
            engagement_score = (time_score + questions_score + completion_score) / 3
            engagement_scores.append(engagement_score)
        
        engagement_level = mean(engagement_scores)
        
        # Calculate learning velocity (improvement rate)
        if len(interactions) > 1:
            quality_progression = [i["student_response_quality"] for i in interactions]
            velocity = (quality_progression[-1] - quality_progression[0]) / len(interactions)
            learning_velocity = max(0.0, velocity)
        else:
            learning_velocity = 0.0
        
        # Estimate retention quality
        satisfaction_scores = [i["satisfaction_score"] for i in interactions]
        confusion_scores = [i["confusion_markers"] for i in interactions]
        
        avg_satisfaction = mean(satisfaction_scores)
        avg_confusion = mean(confusion_scores)
        retention_quality = avg_satisfaction * (1 - avg_confusion)
        
        return {
            "comprehension_improvement": max(-1.0, min(1.0, comprehension_improvement)),
            "engagement_level": max(0.0, min(1.0, engagement_level)),
            "learning_velocity": max(0.0, min(1.0, learning_velocity)),
            "retention_quality": max(0.0, min(1.0, retention_quality))
        }
    
    async def _assess_student_wellbeing(
        self,
        student_id: str,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Assess student wellbeing indicators"""
        
        if not interactions:
            return {
                "confusion_indicators": 0.0,
                "frustration_level": 0.0,
                "confidence_change": 0.0,
                "motivation_impact": 0.0
            }
        
        # Calculate confusion indicators
        confusion_scores = [i["confusion_markers"] for i in interactions]
        confusion_indicators = mean(confusion_scores)
        
        # Estimate frustration level
        abandonment_rate = len([i for i in interactions 
                               if i["engagement_indicators"]["completion_status"] == "abandoned"]) / len(interactions)
        low_satisfaction = len([i for i in interactions if i["satisfaction_score"] < 0.5]) / len(interactions)
        frustration_level = (abandonment_rate + low_satisfaction) / 2
        
        # Calculate confidence change
        early_quality = mean([i["student_response_quality"] for i in interactions[:len(interactions)//3]])
        late_quality = mean([i["student_response_quality"] for i in interactions[-len(interactions)//3:]])
        confidence_change = (late_quality - early_quality) / max(early_quality, 0.1)
        
        # Estimate motivation impact
        engagement_trend = self._calculate_engagement_trend(interactions)
        satisfaction_trend = self._calculate_satisfaction_trend(interactions)
        motivation_impact = (engagement_trend + satisfaction_trend) / 2
        
        return {
            "confusion_indicators": max(0.0, min(1.0, confusion_indicators)),
            "frustration_level": max(0.0, min(1.0, frustration_level)),
            "confidence_change": max(-1.0, min(1.0, confidence_change)),
            "motivation_impact": max(-1.0, min(1.0, motivation_impact))
        }
    
    async def _assess_teaching_quality(
        self,
        teacher_id: str,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Assess teaching quality from interactions"""
        
        if not interactions:
            return {
                "explanation_effectiveness": 0.0,
                "teaching_quality_score": 0.0,
                "adaptive_response_quality": 0.0
            }
        
        # Calculate explanation effectiveness
        explanation_qualities = [i["explanation_quality"] for i in interactions]
        explanation_effectiveness = mean(explanation_qualities)
        
        # Calculate overall teaching quality
        difficulty_adaptation = self._assess_difficulty_adaptation(interactions)
        response_quality = mean([i["student_response_quality"] for i in interactions])
        teaching_quality_score = (explanation_effectiveness + difficulty_adaptation + response_quality) / 3
        
        # Assess adaptive response quality
        adaptive_response_quality = self._assess_adaptive_responses(interactions)
        
        return {
            "explanation_effectiveness": max(0.0, min(1.0, explanation_effectiveness)),
            "teaching_quality_score": max(0.0, min(1.0, teaching_quality_score)),
            "adaptive_response_quality": max(0.0, min(1.0, adaptive_response_quality))
        }
    
    async def _calculate_risk_indicators(
        self,
        student_id: str,
        teacher_id: str,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate risk indicators for student learning"""
        
        if not interactions:
            return {
                "negative_impact_risk": 0.0,
                "learning_obstacle_score": 0.0,
                "dependency_risk": 0.0
            }
        
        # Calculate negative impact risk
        high_confusion = len([i for i in interactions if i["confusion_markers"] > 0.5]) / len(interactions)
        low_satisfaction = len([i for i in interactions if i["satisfaction_score"] < 0.4]) / len(interactions)
        negative_impact_risk = (high_confusion + low_satisfaction) / 2
        
        # Calculate learning obstacle score
        abandonment_rate = len([i for i in interactions 
                               if i["engagement_indicators"]["completion_status"] == "abandoned"]) / len(interactions)
        difficulty_mismatch = self._assess_difficulty_mismatch(interactions)
        learning_obstacle_score = (abandonment_rate + difficulty_mismatch) / 2
        
        # Calculate dependency risk
        consistency_in_help = self._assess_help_consistency(interactions)
        dependency_risk = min(1.0, consistency_in_help * 1.2)  # High consistency might indicate over-dependence
        
        return {
            "negative_impact_risk": max(0.0, min(1.0, negative_impact_risk)),
            "learning_obstacle_score": max(0.0, min(1.0, learning_obstacle_score)),
            "dependency_risk": max(0.0, min(1.0, dependency_risk))
        }
    
    def _calculate_engagement_trend(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate engagement trend over time"""
        if len(interactions) < 2:
            return 0.0
        
        # Calculate engagement scores over time
        engagement_scores = []
        for interaction in interactions:
            factors = interaction["engagement_indicators"]
            score = (
                min(1.0, factors["time_spent"] / 180.0) +
                min(1.0, factors["follow_up_questions"] / 2.0) +
                (1.0 if factors["completion_status"] == "completed" else 0.5)
            ) / 3
            engagement_scores.append(score)
        
        # Calculate trend (simple linear regression slope)
        n = len(engagement_scores)
        x_mean = (n - 1) / 2
        y_mean = mean(engagement_scores)
        
        numerator = sum((i - x_mean) * (score - y_mean) for i, score in enumerate(engagement_scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        trend = numerator / denominator if denominator > 0 else 0.0
        return max(-1.0, min(1.0, trend * 10))  # Normalize to [-1, 1]
    
    def _calculate_satisfaction_trend(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate satisfaction trend over time"""
        if len(interactions) < 2:
            return 0.0
        
        satisfaction_scores = [i["satisfaction_score"] for i in interactions]
        
        # Simple trend calculation
        early_avg = mean(satisfaction_scores[:len(satisfaction_scores)//2])
        late_avg = mean(satisfaction_scores[len(satisfaction_scores)//2:])
        
        trend = (late_avg - early_avg) / max(early_avg, 0.1)
        return max(-1.0, min(1.0, trend))
    
    def _assess_difficulty_adaptation(self, interactions: List[Dict[str, Any]]) -> float:
        """Assess how well difficulty was adapted to student"""
        difficulties = [i["question_difficulty"] for i in interactions]
        qualities = [i["student_response_quality"] for i in interactions]
        
        # Good adaptation means difficulty increases as quality improves
        if len(difficulties) < 2:
            return 0.7  # Default reasonable score
        
        # Check if difficulty adaptation correlates with student improvement
        correlation = self._calculate_correlation(qualities, difficulties)
        return max(0.0, min(1.0, 0.5 + correlation * 0.5))  # Convert correlation to 0-1 scale
    
    def _assess_adaptive_responses(self, interactions: List[Dict[str, Any]]) -> float:
        """Assess quality of adaptive responses"""
        adaptive_scores = []
        
        for i, interaction in enumerate(interactions):
            if i > 0:
                prev_quality = interactions[i-1]["student_response_quality"]
                curr_difficulty = interaction["question_difficulty"]
                
                # Good adaptation: easier questions after poor performance, harder after good
                if prev_quality < 0.5 and curr_difficulty < 0.6:
                    adaptive_scores.append(1.0)  # Good adaptation to difficulty
                elif prev_quality > 0.8 and curr_difficulty > 0.7:
                    adaptive_scores.append(1.0)  # Good progression
                else:
                    adaptive_scores.append(0.7)  # Neutral adaptation
        
        return mean(adaptive_scores) if adaptive_scores else 0.7
    
    def _assess_difficulty_mismatch(self, interactions: List[Dict[str, Any]]) -> float:
        """Assess mismatch between question difficulty and student ability"""
        mismatches = []
        
        for interaction in interactions:
            difficulty = interaction["question_difficulty"]
            quality = interaction["student_response_quality"]
            
            # Mismatch when difficulty is much higher or lower than performance
            mismatch = abs(difficulty - quality)
            mismatches.append(mismatch)
        
        return mean(mismatches) if mismatches else 0.0
    
    def _assess_help_consistency(self, interactions: List[Dict[str, Any]]) -> float:
        """Assess consistency in help provision"""
        # Mock implementation - in real system would analyze help patterns
        help_levels = [i["explanation_quality"] for i in interactions]
        if len(help_levels) < 2:
            return 0.5
        
        # High consistency = low variance
        variance = stdev(help_levels) if len(help_levels) > 1 else 0.0
        consistency = max(0.0, 1.0 - variance)
        return consistency
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        x_mean = mean(x)
        y_mean = mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        x_var = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        y_var = sum((y[i] - y_mean) ** 2 for i in range(len(y)))
        
        denominator = (x_var * y_var) ** 0.5
        
        return numerator / denominator if denominator > 0 else 0.0


class QualityDegradationDetector:
    """Detects and prevents quality degradation in teaching"""
    
    def __init__(self, config: SafetyConfiguration):
        self.config = config
        self.quality_baselines = {}
        self.quality_history = defaultdict(deque)
        self.degradation_alerts = {}
        
    async def monitor_quality_degradation(
        self,
        teacher_id: str,
        current_metrics: Dict[str, float],
        window_size: int = 50
    ) -> List[QualityDegradationAlert]:
        """Monitor for quality degradation in teaching performance"""
        
        # Store current metrics
        for metric_name, value in current_metrics.items():
            self.quality_history[f"{teacher_id}_{metric_name}"].append({
                "value": value,
                "timestamp": datetime.now(timezone.utc)
            })
            
            # Maintain window size
            if len(self.quality_history[f"{teacher_id}_{metric_name}"]) > window_size:
                self.quality_history[f"{teacher_id}_{metric_name}"].popleft()
        
        # Detect degradation
        alerts = []
        
        for metric_name, value in current_metrics.items():
            alert = await self._detect_metric_degradation(
                teacher_id, metric_name, value, window_size
            )
            if alert:
                alerts.append(alert)
        
        return alerts
    
    async def _detect_metric_degradation(
        self,
        teacher_id: str,
        metric_name: str,
        current_value: float,
        window_size: int
    ) -> Optional[QualityDegradationAlert]:
        """Detect degradation in a specific metric"""
        
        history_key = f"{teacher_id}_{metric_name}"
        history = self.quality_history[history_key]
        
        if len(history) < 10:  # Need sufficient history
            return None
        
        # Calculate baseline (early performance average)
        baseline_values = [entry["value"] for entry in list(history)[:window_size//3]]
        baseline = mean(baseline_values)
        
        # Calculate recent performance
        recent_values = [entry["value"] for entry in list(history)[-window_size//3:]]
        recent_avg = mean(recent_values)
        
        # Check for significant degradation
        degradation_percentage = (baseline - recent_avg) / baseline if baseline > 0 else 0
        
        if degradation_percentage > self.config.quality_degradation_threshold:
            # Determine degradation type
            degradation_type = self._classify_degradation_type(metric_name)
            
            # Assess severity
            severity = self._assess_degradation_severity(degradation_percentage)
            
            # Generate evidence
            evidence_samples = self._generate_evidence_samples(history, window_size//6)
            
            # Calculate statistical significance
            statistical_significance = self._calculate_statistical_significance(
                baseline_values, recent_values
            )
            
            # Generate recommendations
            recommendations = self._generate_mitigation_recommendations(
                degradation_type, degradation_percentage
            )
            
            alert = QualityDegradationAlert(
                alert_id=f"deg_{teacher_id}_{metric_name}_{int(time.time())}",
                degradation_type=degradation_type,
                severity=severity,
                teacher_id=teacher_id,
                detection_timestamp=datetime.now(timezone.utc),
                baseline_quality=baseline,
                current_quality=recent_avg,
                degradation_percentage=degradation_percentage,
                trend_direction="declining",
                evidence_samples=evidence_samples,
                statistical_significance=statistical_significance,
                confidence_level=min(0.99, statistical_significance + 0.1),
                recommended_actions=recommendations,
                auto_mitigation_applied=False,
                mitigation_effectiveness=None
            )
            
            return alert
        
        return None
    
    def _classify_degradation_type(self, metric_name: str) -> QualityDegradationType:
        """Classify the type of quality degradation"""
        classification_map = {
            "explanation_quality": QualityDegradationType.EXPLANATION_CLARITY,
            "logical_coherence": QualityDegradationType.LOGICAL_COHERENCE,
            "factual_accuracy": QualityDegradationType.FACTUAL_ACCURACY,
            "teaching_effectiveness": QualityDegradationType.PEDAGOGICAL_EFFECTIVENESS,
            "student_engagement": QualityDegradationType.STUDENT_ENGAGEMENT,
            "comprehension_support": QualityDegradationType.COMPREHENSION_SUPPORT
        }
        
        return classification_map.get(metric_name, QualityDegradationType.EXPLANATION_CLARITY)
    
    def _assess_degradation_severity(self, degradation_percentage: float) -> SafetySeverity:
        """Assess severity of quality degradation"""
        if degradation_percentage > 0.5:  # >50% degradation
            return SafetySeverity.CRITICAL
        elif degradation_percentage > 0.3:  # >30% degradation
            return SafetySeverity.HIGH
        elif degradation_percentage > 0.15:  # >15% degradation
            return SafetySeverity.MEDIUM
        else:
            return SafetySeverity.LOW
    
    def _generate_evidence_samples(
        self,
        history: deque,
        sample_count: int
    ) -> List[Dict[str, Any]]:
        """Generate evidence samples showing degradation"""
        samples = []
        recent_entries = list(history)[-sample_count:]
        
        for entry in recent_entries:
            samples.append({
                "timestamp": entry["timestamp"].isoformat(),
                "value": entry["value"],
                "type": "performance_sample"
            })
        
        return samples
    
    def _calculate_statistical_significance(
        self,
        baseline_values: List[float],
        recent_values: List[float]
    ) -> float:
        """Calculate statistical significance of degradation"""
        if len(baseline_values) < 2 or len(recent_values) < 2:
            return 0.5
        
        # Simplified t-test calculation
        baseline_mean = mean(baseline_values)
        recent_mean = mean(recent_values)
        
        if len(baseline_values) > 1 and len(recent_values) > 1:
            baseline_std = stdev(baseline_values)
            recent_std = stdev(recent_values)
            
            pooled_std = ((baseline_std ** 2 + recent_std ** 2) / 2) ** 0.5
            
            if pooled_std > 0:
                t_stat = abs(baseline_mean - recent_mean) / pooled_std
                # Simplified p-value approximation
                significance = min(0.99, t_stat / 4.0)
                return significance
        
        return 0.6  # Default moderate significance
    
    def _generate_mitigation_recommendations(
        self,
        degradation_type: QualityDegradationType,
        degradation_percentage: float
    ) -> List[str]:
        """Generate mitigation recommendations"""
        
        base_recommendations = [
            "Review recent teaching sessions for quality issues",
            "Analyze student feedback and engagement metrics",
            "Consider retraining on specific problem areas"
        ]
        
        type_specific = {
            QualityDegradationType.EXPLANATION_CLARITY: [
                "Focus on clearer explanation structures",
                "Use more concrete examples and analogies",
                "Break down complex concepts into smaller steps"
            ],
            QualityDegradationType.LOGICAL_COHERENCE: [
                "Review logical flow in explanations",
                "Ensure proper sequencing of concepts",
                "Validate reasoning chains"
            ],
            QualityDegradationType.FACTUAL_ACCURACY: [
                "Verify factual content against reliable sources",
                "Update knowledge base with recent information",
                "Implement additional fact-checking procedures"
            ],
            QualityDegradationType.PEDAGOGICAL_EFFECTIVENESS: [
                "Adapt teaching methods to student learning styles",
                "Increase interactivity and engagement",
                "Provide more personalized feedback"
            ],
            QualityDegradationType.STUDENT_ENGAGEMENT: [
                "Use more engaging examples and scenarios",
                "Increase interactive elements",
                "Adjust pacing to student responses"
            ],
            QualityDegradationType.COMPREHENSION_SUPPORT: [
                "Provide additional scaffolding for difficult concepts",
                "Use multiple explanation approaches",
                "Check for understanding more frequently"
            ]
        }
        
        recommendations = base_recommendations + type_specific.get(degradation_type, [])
        
        if degradation_percentage > 0.3:
            recommendations.extend([
                "Consider temporary teaching suspension pending review",
                "Implement immediate quality oversight",
                "Schedule emergency retraining session"
            ])
        
        return recommendations


class EthicalGuidelinesMonitor:
    """Monitors compliance with ethical teaching guidelines"""
    
    def __init__(self, config: SafetyConfiguration):
        self.config = config
        self.ethical_guidelines = self._load_ethical_guidelines()
        self.violation_history = defaultdict(list)
        
    async def check_ethical_compliance(
        self,
        teacher_id: str,
        teaching_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[EthicalGuidelinesViolation]:
        """Check compliance with ethical teaching guidelines"""
        
        violations = []
        
        # Check each guideline category
        for category, guidelines in self.ethical_guidelines.items():
            category_violations = await self._check_category_compliance(
                teacher_id, teaching_content, context, category, guidelines
            )
            violations.extend(category_violations)
        
        # Store violations
        for violation in violations:
            self.violation_history[teacher_id].append(violation)
        
        return violations
    
    async def _check_category_compliance(
        self,
        teacher_id: str,
        teaching_content: Dict[str, Any],
        context: Dict[str, Any],
        category: str,
        guidelines: Dict[str, Any]
    ) -> List[EthicalGuidelinesViolation]:
        """Check compliance for a specific guideline category"""
        
        violations = []
        
        if category == "fairness_equity":
            violations.extend(await self._check_fairness_equity(
                teacher_id, teaching_content, context
            ))
        elif category == "transparency_explainability":
            violations.extend(await self._check_transparency(
                teacher_id, teaching_content, context
            ))
        elif category == "student_autonomy":
            violations.extend(await self._check_student_autonomy(
                teacher_id, teaching_content, context
            ))
        elif category == "privacy_data_protection":
            violations.extend(await self._check_privacy_protection(
                teacher_id, teaching_content, context
            ))
        elif category == "beneficial_impact":
            violations.extend(await self._check_beneficial_impact(
                teacher_id, teaching_content, context
            ))
        
        return violations
    
    async def _check_fairness_equity(
        self,
        teacher_id: str,
        teaching_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[EthicalGuidelinesViolation]:
        """Check fairness and equity guidelines"""
        violations = []
        
        explanation = teaching_content.get("explanation", "")
        
        # Check for discriminatory language
        discriminatory_patterns = [
            r"\b(boys|girls)\s+are\s+(better|worse)\s+at\b",
            r"\b(only|never)\s+(men|women|boys|girls)\s+can\b",
            r"\bpeople\s+from\s+\w+\s+are\s+(always|never)\b"
        ]
        
        for pattern in discriminatory_patterns:
            if re.search(pattern, explanation, re.IGNORECASE):
                violation = EthicalGuidelinesViolation(
                    violation_id=f"fairness_{int(time.time() * 1000)}",
                    guideline_category="fairness_equity",
                    violation_description="Detected potentially discriminatory language",
                    severity=SafetySeverity.HIGH,
                    teacher_id=teacher_id,
                    context={**context, "matched_pattern": pattern},
                    detection_method="pattern_matching",
                    timestamp=datetime.now(timezone.utc),
                    remediation_required=True,
                    remediation_plan={
                        "action": "review_explanation",
                        "steps": ["Remove discriminatory language", "Provide inclusive alternative"]
                    }
                )
                violations.append(violation)
        
        return violations
    
    async def _check_transparency(
        self,
        teacher_id: str,
        teaching_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[EthicalGuidelinesViolation]:
        """Check transparency and explainability guidelines"""
        violations = []
        
        explanation = teaching_content.get("explanation", "")
        
        # Check if explanation provides reasoning
        reasoning_indicators = ["because", "since", "therefore", "thus", "as a result"]
        has_reasoning = any(indicator in explanation.lower() for indicator in reasoning_indicators)
        
        if not has_reasoning and len(explanation) > 100:  # Only for substantial explanations
            violation = EthicalGuidelinesViolation(
                violation_id=f"transparency_{int(time.time() * 1000)}",
                guideline_category="transparency_explainability",
                violation_description="Explanation lacks clear reasoning or justification",
                severity=SafetySeverity.MEDIUM,
                teacher_id=teacher_id,
                context=context,
                detection_method="reasoning_analysis",
                timestamp=datetime.now(timezone.utc),
                remediation_required=True,
                remediation_plan={
                    "action": "add_reasoning",
                    "steps": ["Include clear reasoning steps", "Explain the 'why' behind concepts"]
                }
            )
            violations.append(violation)
        
        return violations
    
    async def _check_student_autonomy(
        self,
        teacher_id: str,
        teaching_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[EthicalGuidelinesViolation]:
        """Check student autonomy guidelines"""
        violations = []
        
        explanation = teaching_content.get("explanation", "")
        
        # Check for overly prescriptive language
        prescriptive_patterns = [
            r"\byou\s+must\s+(always|never)\b",
            r"\bthe\s+only\s+way\s+is\b",
            r"\bnever\s+question\b",
            r"\bdon't\s+think\s+about\b"
        ]
        
        for pattern in prescriptive_patterns:
            if re.search(pattern, explanation, re.IGNORECASE):
                violation = EthicalGuidelinesViolation(
                    violation_id=f"autonomy_{int(time.time() * 1000)}",
                    guideline_category="student_autonomy",
                    violation_description="Explanation may restrict student thinking autonomy",
                    severity=SafetySeverity.MEDIUM,
                    teacher_id=teacher_id,
                    context={**context, "matched_pattern": pattern},
                    detection_method="pattern_matching",
                    timestamp=datetime.now(timezone.utc),
                    remediation_required=True,
                    remediation_plan={
                        "action": "encourage_exploration",
                        "steps": ["Use more open-ended language", "Encourage student exploration"]
                    }
                )
                violations.append(violation)
        
        return violations
    
    async def _check_privacy_protection(
        self,
        teacher_id: str,
        teaching_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[EthicalGuidelinesViolation]:
        """Check privacy and data protection guidelines"""
        violations = []
        
        # Check if student data is properly handled
        if context.get("student_data_used", False):
            data_protection_score = context.get("data_protection_score", 1.0)
            
            if data_protection_score < 0.8:
                violation = EthicalGuidelinesViolation(
                    violation_id=f"privacy_{int(time.time() * 1000)}",
                    guideline_category="privacy_data_protection",
                    violation_description="Insufficient data protection measures",
                    severity=SafetySeverity.HIGH,
                    teacher_id=teacher_id,
                    context=context,
                    detection_method="data_protection_analysis",
                    timestamp=datetime.now(timezone.utc),
                    remediation_required=True,
                    remediation_plan={
                        "action": "strengthen_data_protection",
                        "steps": ["Review data handling procedures", "Implement stronger protections"]
                    }
                )
                violations.append(violation)
        
        return violations
    
    async def _check_beneficial_impact(
        self,
        teacher_id: str,
        teaching_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[EthicalGuidelinesViolation]:
        """Check beneficial impact guidelines"""
        violations = []
        
        # Check learning impact metrics
        impact_score = context.get("learning_impact_score", 0.8)
        
        if impact_score < 0.5:  # Below acceptable beneficial impact
            violation = EthicalGuidelinesViolation(
                violation_id=f"impact_{int(time.time() * 1000)}",
                guideline_category="beneficial_impact",
                violation_description="Teaching shows insufficient beneficial impact",
                severity=SafetySeverity.MEDIUM,
                teacher_id=teacher_id,
                context=context,
                detection_method="impact_analysis",
                timestamp=datetime.now(timezone.utc),
                remediation_required=True,
                remediation_plan={
                    "action": "improve_teaching_effectiveness",
                    "steps": ["Analyze student responses", "Adjust teaching approach", "Monitor improvement"]
                }
            )
            violations.append(violation)
        
        return violations
    
    def _load_ethical_guidelines(self) -> Dict[str, Dict[str, Any]]:
        """Load ethical guidelines configuration"""
        return {
            "fairness_equity": {
                "description": "Ensure fair and equitable treatment of all students",
                "requirements": [
                    "No discriminatory language or bias",
                    "Equal opportunity for learning",
                    "Inclusive examples and scenarios"
                ]
            },
            "transparency_explainability": {
                "description": "Provide clear, understandable explanations",
                "requirements": [
                    "Clear reasoning for all statements",
                    "Explainable decision-making process",
                    "Open about limitations and uncertainties"
                ]
            },
            "student_autonomy": {
                "description": "Respect student autonomy and critical thinking",
                "requirements": [
                    "Encourage independent thinking",
                    "Avoid overly prescriptive guidance",
                    "Support student exploration and discovery"
                ]
            },
            "privacy_data_protection": {
                "description": "Protect student privacy and data",
                "requirements": [
                    "Secure handling of student information",
                    "No unauthorized data sharing",
                    "Respect for student privacy rights"
                ]
            },
            "beneficial_impact": {
                "description": "Ensure teaching has positive impact on learning",
                "requirements": [
                    "Demonstrable learning improvements",
                    "Positive student wellbeing outcomes",
                    "Long-term educational benefits"
                ]
            }
        }


class AdvancedSafetyQualityFramework:
    """
    Comprehensive Advanced Safety & Quality Framework
    
    Integrates all safety and quality components for comprehensive
    monitoring, assessment, and assurance of RLT teacher behavior.
    """
    
    def __init__(self, config: Optional[SafetyConfiguration] = None):
        self.config = config or SafetyConfiguration()
        
        # Initialize components
        self.safety_validator = ExplanationSafetyValidator(self.config)
        self.impact_assessor = StudentLearningImpactAssessor(self.config)
        self.quality_detector = QualityDegradationDetector(self.config)
        self.ethics_monitor = EthicalGuidelinesMonitor(self.config)
        
        # Monitoring state
        self.safety_violations = defaultdict(list)
        self.impact_assessments = defaultdict(list)
        self.quality_alerts = defaultdict(list)
        self.ethical_violations = defaultdict(list)
        
        # Safety statistics
        self.safety_stats = {
            "total_validations": 0,
            "violations_detected": 0,
            "quality_degradations": 0,
            "ethical_violations": 0,
            "impact_assessments_completed": 0
        }
        
        logger.info("Advanced Safety & Quality Framework initialized")
    
    async def comprehensive_safety_check(
        self,
        teacher_id: str,
        teaching_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive safety and quality check"""
        
        check_id = f"safety_check_{int(time.time() * 1000)}"
        start_time = datetime.now(timezone.utc)
        
        results = {
            "check_id": check_id,
            "teacher_id": teacher_id,
            "timestamp": start_time,
            "overall_safety_status": "unknown",
            "safety_violations": [],
            "quality_alerts": [],
            "ethical_violations": [],
            "impact_assessment": None,
            "recommendations": [],
            "risk_level": "unknown"
        }
        
        try:
            # 1. Explanation Safety Validation
            explanation = teaching_content.get("explanation", "")
            is_safe, safety_violations = await self.safety_validator.validate_explanation_safety(
                explanation, context
            )
            
            results["safety_violations"] = [asdict(v) for v in safety_violations]
            self.safety_violations[teacher_id].extend(safety_violations)
            
            # 2. Quality Degradation Detection
            quality_metrics = teaching_content.get("quality_metrics", {})
            if quality_metrics:
                quality_alerts = await self.quality_detector.monitor_quality_degradation(
                    teacher_id, quality_metrics
                )
                results["quality_alerts"] = [asdict(a) for a in quality_alerts]
                self.quality_alerts[teacher_id].extend(quality_alerts)
            
            # 3. Ethical Guidelines Compliance
            ethical_violations = await self.ethics_monitor.check_ethical_compliance(
                teacher_id, teaching_content, context
            )
            results["ethical_violations"] = [asdict(v) for v in ethical_violations]
            self.ethical_violations[teacher_id].extend(ethical_violations)
            
            # 4. Student Learning Impact Assessment (if student data available)
            student_id = context.get("student_id")
            if student_id:
                impact_assessment = await self.impact_assessor.assess_learning_impact(
                    student_id, teacher_id, {"teaching_content": teaching_content, **context}
                )
                results["impact_assessment"] = asdict(impact_assessment)
                self.impact_assessments[student_id].append(impact_assessment)
                self.safety_stats["impact_assessments_completed"] += 1
            
            # 5. Overall Risk Assessment
            risk_level = self._calculate_overall_risk_level(
                safety_violations, quality_alerts, ethical_violations,
                results.get("impact_assessment")
            )
            results["risk_level"] = risk_level
            
            # 6. Generate Recommendations
            recommendations = self._generate_safety_recommendations(
                safety_violations, quality_alerts, ethical_violations, risk_level
            )
            results["recommendations"] = recommendations
            
            # 7. Determine Overall Safety Status
            results["overall_safety_status"] = self._determine_safety_status(
                is_safe, risk_level, safety_violations, ethical_violations
            )
            
            # Update statistics
            self.safety_stats["total_validations"] += 1
            if safety_violations:
                self.safety_stats["violations_detected"] += len(safety_violations)
            if quality_alerts:
                self.safety_stats["quality_degradations"] += len(quality_alerts)
            if ethical_violations:
                self.safety_stats["ethical_violations"] += len(ethical_violations)
            
            logger.info(f"Safety check completed: {check_id}, status: {results['overall_safety_status']}")
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            results["overall_safety_status"] = "error"
            results["error"] = str(e)
        
        return results
    
    async def generate_safety_report(
        self,
        teacher_id: str,
        report_period_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate comprehensive safety report for a teacher"""
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=report_period_hours)
        
        # Collect data for the period
        period_violations = [
            v for v in self.safety_violations[teacher_id]
            if start_time <= v.timestamp <= end_time
        ]
        
        period_quality_alerts = [
            a for a in self.quality_alerts[teacher_id]
            if start_time <= a.detection_timestamp <= end_time
        ]
        
        period_ethical_violations = [
            v for v in self.ethical_violations[teacher_id]
            if start_time <= v.timestamp <= end_time
        ]
        
        # Generate report
        report = {
            "report_id": f"safety_report_{teacher_id}_{int(time.time())}",
            "teacher_id": teacher_id,
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": report_period_hours
            },
            "summary": {
                "total_safety_violations": len(period_violations),
                "total_quality_alerts": len(period_quality_alerts),
                "total_ethical_violations": len(period_ethical_violations),
                "overall_safety_score": self._calculate_safety_score(
                    period_violations, period_quality_alerts, period_ethical_violations
                )
            },
            "violations_by_type": self._categorize_violations_by_type(period_violations),
            "quality_degradation_summary": self._summarize_quality_alerts(period_quality_alerts),
            "ethical_compliance_summary": self._summarize_ethical_violations(period_ethical_violations),
            "trends_and_patterns": await self._analyze_safety_trends(teacher_id, start_time, end_time),
            "recommendations": self._generate_improvement_recommendations(
                period_violations, period_quality_alerts, period_ethical_violations
            ),
            "generated_at": end_time.isoformat()
        }
        
        return report
    
    async def get_network_safety_statistics(self) -> Dict[str, Any]:
        """Get network-wide safety statistics"""
        
        total_teachers = len(set(
            list(self.safety_violations.keys()) +
            list(self.quality_alerts.keys()) +
            list(self.ethical_violations.keys())
        ))
        
        # Calculate violation rates
        total_violations = sum(len(violations) for violations in self.safety_violations.values())
        total_quality_alerts = sum(len(alerts) for alerts in self.quality_alerts.values())
        total_ethical_violations = sum(len(violations) for violations in self.ethical_violations.values())
        
        # Safety score distribution
        teacher_safety_scores = []
        for teacher_id in set(self.safety_violations.keys()):
            recent_violations = self.safety_violations[teacher_id][-10:]  # Last 10 violations
            recent_quality_alerts = self.quality_alerts[teacher_id][-10:]
            recent_ethical_violations = self.ethical_violations[teacher_id][-10:]
            
            safety_score = self._calculate_safety_score(
                recent_violations, recent_quality_alerts, recent_ethical_violations
            )
            teacher_safety_scores.append(safety_score)
        
        avg_safety_score = mean(teacher_safety_scores) if teacher_safety_scores else 1.0
        
        return {
            "network_overview": {
                "total_teachers_monitored": total_teachers,
                "average_network_safety_score": avg_safety_score,
                "total_validations_performed": self.safety_stats["total_validations"]
            },
            "violation_statistics": {
                "total_safety_violations": total_violations,
                "total_quality_degradations": total_quality_alerts,
                "total_ethical_violations": total_ethical_violations,
                "violation_rate": total_violations / max(self.safety_stats["total_validations"], 1)
            },
            "safety_distribution": {
                "high_safety_teachers": len([s for s in teacher_safety_scores if s > 0.8]),
                "medium_safety_teachers": len([s for s in teacher_safety_scores if 0.6 <= s <= 0.8]),
                "low_safety_teachers": len([s for s in teacher_safety_scores if s < 0.6])
            },
            "system_health": {
                "overall_safety_status": "healthy" if avg_safety_score > 0.8 else "needs_attention",
                "monitoring_effectiveness": min(1.0, self.safety_stats["violations_detected"] / max(total_violations, 1)),
                "assessment_coverage": self.safety_stats["impact_assessments_completed"]
            }
        }
    
    def _calculate_overall_risk_level(
        self,
        safety_violations: List[SafetyViolation],
        quality_alerts: List[QualityDegradationAlert],
        ethical_violations: List[EthicalGuidelinesViolation],
        impact_assessment: Optional[Dict[str, Any]]
    ) -> str:
        """Calculate overall risk level"""
        
        risk_score = 0.0
        
        # Safety violations contribute to risk
        for violation in safety_violations:
            if violation.severity == SafetySeverity.CRITICAL:
                risk_score += 1.0
            elif violation.severity == SafetySeverity.HIGH:
                risk_score += 0.7
            elif violation.severity == SafetySeverity.MEDIUM:
                risk_score += 0.4
            else:
                risk_score += 0.1
        
        # Quality alerts contribute to risk
        for alert in quality_alerts:
            if alert.severity == SafetySeverity.CRITICAL:
                risk_score += 0.8
            elif alert.severity == SafetySeverity.HIGH:
                risk_score += 0.5
            elif alert.severity == SafetySeverity.MEDIUM:
                risk_score += 0.3
            else:
                risk_score += 0.1
        
        # Ethical violations contribute to risk
        for violation in ethical_violations:
            if violation.severity == SafetySeverity.CRITICAL:
                risk_score += 0.9
            elif violation.severity == SafetySeverity.HIGH:
                risk_score += 0.6
            elif violation.severity == SafetySeverity.MEDIUM:
                risk_score += 0.3
            else:
                risk_score += 0.1
        
        # Impact assessment contributes to risk
        if impact_assessment:
            negative_impact_risk = impact_assessment.get("negative_impact_risk", 0.0)
            risk_score += negative_impact_risk * 0.5
        
        # Determine risk level
        if risk_score >= 2.0:
            return "critical"
        elif risk_score >= 1.0:
            return "high"
        elif risk_score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _generate_safety_recommendations(
        self,
        safety_violations: List[SafetyViolation],
        quality_alerts: List[QualityDegradationAlert],
        ethical_violations: List[EthicalGuidelinesViolation],
        risk_level: str
    ) -> List[str]:
        """Generate safety recommendations"""
        
        recommendations = []
        
        # Risk-based recommendations
        if risk_level == "critical":
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Suspend teaching operations pending review",
                "Conduct emergency safety audit",
                "Implement immediate remediation measures"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "Increase monitoring frequency",
                "Review and update safety protocols",
                "Consider additional training"
            ])
        
        # Violation-specific recommendations
        if safety_violations:
            violation_types = {v.violation_type for v in safety_violations}
            if SafetyViolationType.BIAS_DETECTED in violation_types:
                recommendations.append("Implement bias detection and mitigation training")
            if SafetyViolationType.HARMFUL_CONTENT in violation_types:
                recommendations.append("Review content filtering and safety protocols")
        
        # Quality-specific recommendations
        if quality_alerts:
            recommendations.extend([
                "Analyze quality degradation patterns",
                "Implement quality improvement measures",
                "Consider retraining on affected areas"
            ])
        
        # Ethics-specific recommendations
        if ethical_violations:
            recommendations.extend([
                "Review ethical guidelines compliance",
                "Provide additional ethics training",
                "Implement stronger ethical monitoring"
            ])
        
        return recommendations
    
    def _determine_safety_status(
        self,
        is_safe: bool,
        risk_level: str,
        safety_violations: List[SafetyViolation],
        ethical_violations: List[EthicalGuidelinesViolation]
    ) -> str:
        """Determine overall safety status"""
        
        if not is_safe or risk_level == "critical":
            return "unsafe"
        elif risk_level == "high":
            return "caution"
        elif risk_level == "medium" or safety_violations or ethical_violations:
            return "monitored"
        else:
            return "safe"
    
    def _calculate_safety_score(
        self,
        violations: List[SafetyViolation],
        quality_alerts: List[QualityDegradationAlert],
        ethical_violations: List[EthicalGuidelinesViolation]
    ) -> float:
        """Calculate overall safety score (0-1, higher is better)"""
        
        base_score = 1.0
        
        # Deduct for violations
        for violation in violations:
            if violation.severity == SafetySeverity.CRITICAL:
                base_score -= 0.3
            elif violation.severity == SafetySeverity.HIGH:
                base_score -= 0.2
            elif violation.severity == SafetySeverity.MEDIUM:
                base_score -= 0.1
            else:
                base_score -= 0.05
        
        # Deduct for quality alerts
        for alert in quality_alerts:
            if alert.severity == SafetySeverity.CRITICAL:
                base_score -= 0.25
            elif alert.severity == SafetySeverity.HIGH:
                base_score -= 0.15
            elif alert.severity == SafetySeverity.MEDIUM:
                base_score -= 0.08
            else:
                base_score -= 0.03
        
        # Deduct for ethical violations
        for violation in ethical_violations:
            if violation.severity == SafetySeverity.CRITICAL:
                base_score -= 0.2
            elif violation.severity == SafetySeverity.HIGH:
                base_score -= 0.12
            elif violation.severity == SafetySeverity.MEDIUM:
                base_score -= 0.06
            else:
                base_score -= 0.02
        
        return max(0.0, base_score)
    
    def _categorize_violations_by_type(self, violations: List[SafetyViolation]) -> Dict[str, int]:
        """Categorize violations by type"""
        categories = defaultdict(int)
        for violation in violations:
            categories[violation.violation_type.value] += 1
        return dict(categories)
    
    def _summarize_quality_alerts(self, alerts: List[QualityDegradationAlert]) -> Dict[str, Any]:
        """Summarize quality degradation alerts"""
        if not alerts:
            return {"total_alerts": 0}
        
        return {
            "total_alerts": len(alerts),
            "by_type": {
                alert_type.value: len([a for a in alerts if a.degradation_type == alert_type])
                for alert_type in QualityDegradationType
            },
            "average_degradation": mean([a.degradation_percentage for a in alerts]),
            "most_severe": max([a.severity.value for a in alerts])
        }
    
    def _summarize_ethical_violations(self, violations: List[EthicalGuidelinesViolation]) -> Dict[str, Any]:
        """Summarize ethical violations"""
        if not violations:
            return {"total_violations": 0}
        
        categories = defaultdict(int)
        for violation in violations:
            categories[violation.guideline_category] += 1
        
        return {
            "total_violations": len(violations),
            "by_category": dict(categories),
            "remediation_required": len([v for v in violations if v.remediation_required])
        }
    
    async def _analyze_safety_trends(
        self,
        teacher_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Analyze safety trends over time"""
        # Mock trend analysis
        return {
            "violation_trend": "stable",  # "increasing", "decreasing", "stable"
            "quality_trend": "improving",
            "risk_trend": "decreasing",
            "compliance_trend": "stable"
        }
    
    def _generate_improvement_recommendations(
        self,
        violations: List[SafetyViolation],
        quality_alerts: List[QualityDegradationAlert],
        ethical_violations: List[EthicalGuidelinesViolation]
    ) -> List[str]:
        """Generate specific improvement recommendations"""
        recommendations = []
        
        if violations:
            recommendations.append("Review and strengthen content safety protocols")
        
        if quality_alerts:
            recommendations.append("Implement quality monitoring and improvement processes")
        
        if ethical_violations:
            recommendations.append("Enhance ethical guidelines training and compliance")
        
        return recommendations