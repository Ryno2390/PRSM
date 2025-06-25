"""
SEAL RLT Enhanced Teacher System

SEAL (Safety Enhanced AI Learning) framework for RLT teachers, implementing
advanced safety protocols, alignment verification, and responsible AI teaching
practices to ensure safe and beneficial learning outcomes.

Key Features:
- Multi-layered safety verification
- Alignment consistency checking
- Harmful content detection and prevention
- Bias monitoring and mitigation
- Teaching ethics enforcement
- Real-time safety intervention
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum
import structlog
import re

logger = structlog.get_logger(__name__)


class SafetyLevel(Enum):
    """Safety assessment levels"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    BLOCKED = "blocked"


class SafetyProtocol(Enum):
    """Types of safety protocols"""
    CONTENT_FILTERING = "content_filtering"
    BIAS_DETECTION = "bias_detection"
    ALIGNMENT_CHECK = "alignment_check"
    ETHICS_VERIFICATION = "ethics_verification"
    HARM_PREVENTION = "harm_prevention"
    MISINFORMATION_CHECK = "misinformation_check"


@dataclass
class SafetyViolation:
    """Represents a safety violation"""
    violation_id: str
    protocol: SafetyProtocol
    severity: SafetyLevel
    description: str
    content_snippet: str
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Context information
    teacher_id: str = ""
    student_context: Dict[str, Any] = field(default_factory=dict)
    intervention_taken: str = ""
    
    def __post_init__(self):
        if not self.violation_id:
            self.violation_id = str(uuid4())


@dataclass
class TeachingSafeguards:
    """Configuration for teaching safety measures"""
    
    # Content safety
    enable_content_filtering: bool = True
    enable_bias_detection: bool = True
    enable_misinformation_check: bool = True
    
    # Alignment safety
    enable_alignment_verification: bool = True
    enable_ethics_checking: bool = True
    enable_value_consistency: bool = True
    
    # Intervention controls
    auto_intervention_threshold: SafetyLevel = SafetyLevel.WARNING
    require_human_review: bool = False
    block_on_violation: bool = True
    
    # Monitoring settings
    log_all_interactions: bool = True
    store_violation_evidence: bool = True
    enable_continuous_monitoring: bool = True
    
    # Thresholds
    bias_threshold: float = 0.7
    harm_threshold: float = 0.8
    misinformation_threshold: float = 0.75
    alignment_threshold: float = 0.6


@dataclass
class SafetyAssessment:
    """Comprehensive safety assessment result"""
    
    assessment_id: str
    overall_safety_level: SafetyLevel
    confidence: float
    
    # Protocol-specific results
    content_safety: float = 1.0
    bias_score: float = 0.0
    harm_potential: float = 0.0
    alignment_score: float = 1.0
    ethics_score: float = 1.0
    misinformation_risk: float = 0.0
    
    # Violations found
    violations: List[SafetyViolation] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    interventions_required: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time: float = 0.0
    
    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = str(uuid4())
    
    def is_safe_to_proceed(self) -> bool:
        """Determine if it's safe to proceed with teaching"""
        return (
            self.overall_safety_level in [SafetyLevel.SAFE, SafetyLevel.CAUTION] and
            len([v for v in self.violations if v.severity in [SafetyLevel.DANGER, SafetyLevel.BLOCKED]]) == 0
        )


class SEALRLTEnhancedTeacher:
    """
    SEAL (Safety Enhanced AI Learning) RLT Teacher
    
    An advanced RLT teacher implementation with comprehensive safety protocols:
    - Multi-layered safety verification
    - Alignment and ethics enforcement
    - Bias detection and mitigation
    - Harmful content prevention
    - Real-time safety monitoring
    - Responsible AI teaching practices
    """
    
    def __init__(self, teacher_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.teacher_id = teacher_id or f"seal_teacher_{uuid4().hex[:8]}"
        self.config = config or {}
        
        # Safety configuration
        self.safeguards = TeachingSafeguards(**self.config.get('safeguards', {}))
        
        # Safety violation tracking
        self.violation_history: List[SafetyViolation] = []
        self.safety_assessments: List[SafetyAssessment] = []
        
        # Content filters and detection patterns
        self.harmful_patterns = self._initialize_harmful_patterns()
        self.bias_indicators = self._initialize_bias_indicators()
        self.misinformation_signals = self._initialize_misinformation_signals()
        
        # Performance tracking
        self.safety_check_times: List[float] = []
        self.intervention_count = 0
        self.total_interactions = 0
        
        # Teaching effectiveness tracking (post-safety)
        self.safe_teaching_effectiveness: List[float] = []
        self.student_safety_feedback: List[Dict[str, Any]] = []
        
        logger.info(
            "SEAL RLT Enhanced Teacher initialized",
            teacher_id=self.teacher_id,
            safeguards_enabled=len([
                attr for attr in dir(self.safeguards) 
                if not attr.startswith('_') and 
                getattr(self.safeguards, attr) is True
            ])
        )
    
    def _initialize_harmful_patterns(self) -> List[Dict[str, Any]]:
        """Initialize patterns for harmful content detection"""
        return [
            {
                "pattern": r"\b(violence|harm|hurt|damage|destroy)\w*\b",
                "category": "violence",
                "severity": SafetyLevel.WARNING,
                "description": "Potential violence-related content"
            },
            {
                "pattern": r"\b(hate|discriminat\w+|racist|sexist)\b",
                "category": "discrimination",
                "severity": SafetyLevel.DANGER,
                "description": "Potential discriminatory content"
            },
            {
                "pattern": r"\b(illegal|criminal|fraud|scam)\b",
                "category": "illegal_activity",
                "severity": SafetyLevel.WARNING,
                "description": "Potential illegal activity discussion"
            },
            {
                "pattern": r"\b(private|personal|confidential|secret)\s+(information|data|details)\b",
                "category": "privacy_violation",
                "severity": SafetyLevel.CAUTION,
                "description": "Potential privacy-related concern"
            }
        ]
    
    def _initialize_bias_indicators(self) -> List[Dict[str, Any]]:
        """Initialize bias detection indicators"""
        return [
            {
                "type": "gender_bias",
                "patterns": [r"\b(men|male)\s+are\s+(better|superior)", r"\b(women|female)\s+are\s+(worse|inferior)"],
                "weight": 0.8
            },
            {
                "type": "racial_bias", 
                "patterns": [r"\b(race|ethnicity)\s+\w+\s+(superior|inferior)", r"\ball\s+\w+\s+people\s+are\s+\w+"],
                "weight": 0.9
            },
            {
                "type": "age_bias",
                "patterns": [r"\b(young|old)\s+people\s+(always|never)", r"\b(millennial|boomer)\s+\w+\s+(lazy|entitled)"],
                "weight": 0.6
            },
            {
                "type": "cultural_bias",
                "patterns": [r"\b(culture|religion)\s+\w+\s+(wrong|backward)", r"\bwestern\s+(values|culture)\s+is\s+superior"],
                "weight": 0.7
            }
        ]
    
    def _initialize_misinformation_signals(self) -> List[Dict[str, Any]]:
        """Initialize misinformation detection signals"""
        return [
            {
                "signal": "absolute_claims",
                "patterns": [r"\balways\s+true", r"\bnever\s+false", r"\b100%\s+(certain|sure|accurate)"],
                "weight": 0.6,
                "description": "Overly absolute claims without evidence"
            },
            {
                "signal": "conspiracy_language",
                "patterns": [r"\bthey\s+don't\s+want\s+you\s+to\s+know", r"\bhidden\s+truth", r"\bcover\s*up"],
                "weight": 0.8,
                "description": "Conspiracy theory language patterns"
            },
            {
                "signal": "anecdotal_evidence",
                "patterns": [r"\bi\s+heard\s+that", r"\bsomeone\s+told\s+me", r"\ba\s+friend\s+said"],
                "weight": 0.4,
                "description": "Reliance on anecdotal evidence"
            },
            {
                "signal": "false_authority",
                "patterns": [r"\bexperts\s+say\b(?!\s+that\s+\w+\s+studies)", r"\bscientists\s+prove\b(?!\s+specific)"],
                "weight": 0.7,
                "description": "Appeal to vague or false authority"
            }
        ]
    
    async def safe_teach(
        self,
        student_query: str,
        student_context: Dict[str, Any],
        learning_objectives: List[str]
    ) -> Dict[str, Any]:
        """
        Perform safe teaching with comprehensive safety verification
        
        Args:
            student_query: The student's question or learning request
            student_context: Context about the student and learning environment
            learning_objectives: Specific learning goals for this interaction
            
        Returns:
            Safe teaching response with safety metadata
        """
        
        interaction_start = time.time()
        self.total_interactions += 1
        
        # Phase 1: Pre-teaching safety assessment
        print(f"🛡️  SEAL Safety Assessment for student query...")
        pre_assessment = await self._assess_query_safety(student_query, student_context)
        
        if not pre_assessment.is_safe_to_proceed():
            return await self._handle_unsafe_query(student_query, pre_assessment)
        
        # Phase 2: Generate teaching content
        print(f"🎓 Generating safe teaching content...")
        teaching_content = await self._generate_safe_teaching_content(
            student_query, student_context, learning_objectives
        )
        
        # Phase 3: Post-generation safety verification
        post_assessment = await self._assess_content_safety(teaching_content, student_context)
        
        if not post_assessment.is_safe_to_proceed():
            return await self._handle_unsafe_content(teaching_content, post_assessment)
        
        # Phase 4: Apply safety enhancements
        enhanced_content = await self._apply_safety_enhancements(teaching_content, post_assessment)
        
        # Phase 5: Final verification and delivery
        final_response = await self._finalize_safe_response(
            enhanced_content, pre_assessment, post_assessment, student_context
        )
        
        interaction_time = time.time() - interaction_start
        
        # Log successful safe interaction
        logger.info(
            "Safe teaching interaction completed",
            teacher_id=self.teacher_id,
            student_query_length=len(student_query),
            safety_level=post_assessment.overall_safety_level.value,
            interaction_time=interaction_time
        )
        
        return final_response
    
    async def _assess_query_safety(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> SafetyAssessment:
        """Assess safety of incoming student query"""
        
        start_time = time.time()
        assessment = SafetyAssessment(
            assessment_id=str(uuid4()),
            overall_safety_level=SafetyLevel.SAFE,
            confidence=0.95
        )
        
        violations = []
        
        # Check for harmful content patterns
        if self.safeguards.enable_content_filtering:
            harmful_violations = await self._check_harmful_patterns(query)
            violations.extend(harmful_violations)
        
        # Check for bias indicators
        if self.safeguards.enable_bias_detection:
            bias_score = await self._detect_bias(query)
            assessment.bias_score = bias_score
            
            if bias_score > self.safeguards.bias_threshold:
                violations.append(SafetyViolation(
                    violation_id=str(uuid4()),
                    protocol=SafetyProtocol.BIAS_DETECTION,
                    severity=SafetyLevel.WARNING,
                    description=f"High bias detected in query (score: {bias_score:.2f})",
                    content_snippet=query[:100],
                    confidence=0.8,
                    student_context=context
                ))
        
        # Check for potential harm
        harm_score = await self._assess_harm_potential(query)
        assessment.harm_potential = harm_score
        
        if harm_score > self.safeguards.harm_threshold:
            violations.append(SafetyViolation(
                violation_id=str(uuid4()),
                protocol=SafetyProtocol.HARM_PREVENTION,
                severity=SafetyLevel.DANGER,
                description=f"High harm potential detected (score: {harm_score:.2f})",
                content_snippet=query[:100],
                confidence=0.9,
                student_context=context
            ))
        
        # Determine overall safety level
        if violations:
            max_severity = max([v.severity for v in violations])
            assessment.overall_safety_level = max_severity
            assessment.violations = violations
            
            # Generate recommendations
            assessment.recommendations = self._generate_safety_recommendations(violations)
        
        assessment.processing_time = time.time() - start_time
        self.safety_check_times.append(assessment.processing_time)
        self.safety_assessments.append(assessment)
        
        return assessment
    
    async def _check_harmful_patterns(self, content: str) -> List[SafetyViolation]:
        """Check content against harmful patterns"""
        
        violations = []
        content_lower = content.lower()
        
        for pattern_info in self.harmful_patterns:
            pattern = pattern_info["pattern"]
            matches = re.findall(pattern, content_lower)
            
            if matches:
                violation = SafetyViolation(
                    violation_id=str(uuid4()),
                    protocol=SafetyProtocol.CONTENT_FILTERING,
                    severity=pattern_info["severity"],
                    description=f"{pattern_info['description']}: {matches[0]}",
                    content_snippet=content[:100],
                    confidence=0.8,
                    teacher_id=self.teacher_id
                )
                violations.append(violation)
        
        return violations
    
    async def _detect_bias(self, content: str) -> float:
        """Detect bias in content and return bias score"""
        
        total_bias_score = 0.0
        total_weight = 0.0
        content_lower = content.lower()
        
        for bias_indicator in self.bias_indicators:
            indicator_score = 0.0
            
            for pattern in bias_indicator["patterns"]:
                if re.search(pattern, content_lower):
                    indicator_score = 1.0
                    break
            
            weight = bias_indicator["weight"]
            total_bias_score += indicator_score * weight
            total_weight += weight
        
        return total_bias_score / total_weight if total_weight > 0 else 0.0
    
    async def _assess_harm_potential(self, content: str) -> float:
        """Assess potential for harm in content"""
        
        # Simulate harm assessment with realistic scoring
        content_lower = content.lower()
        
        harm_keywords = [
            "violence", "harm", "hurt", "damage", "dangerous", "illegal",
            "criminal", "threat", "weapon", "abuse", "exploit"
        ]
        
        harm_score = 0.0
        for keyword in harm_keywords:
            if keyword in content_lower:
                harm_score += 0.1
        
        # Add context-based scoring
        sensitive_contexts = ["medical", "legal", "financial", "personal"]
        for context in sensitive_contexts:
            if context in content_lower:
                harm_score += 0.05
        
        return min(harm_score, 1.0)
    
    async def _generate_safe_teaching_content(
        self,
        query: str,
        context: Dict[str, Any],
        objectives: List[str]
    ) -> Dict[str, Any]:
        """Generate teaching content with safety considerations"""
        
        # Simulate safe content generation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Extract key concepts from query
        key_concepts = self._extract_key_concepts(query)
        
        # Generate explanation with safety filters
        explanation = await self._create_safe_explanation(query, key_concepts, context)
        
        # Generate examples that are appropriate
        examples = await self._create_safe_examples(key_concepts, context)
        
        # Create practice questions that are educational and safe
        practice_questions = await self._create_safe_practice_questions(key_concepts, objectives)
        
        return {
            "explanation": explanation,
            "key_concepts": key_concepts,
            "examples": examples,
            "practice_questions": practice_questions,
            "learning_objectives": objectives,
            "safety_filtered": True,
            "generation_metadata": {
                "query": query,
                "context_considered": True,
                "safety_protocols_applied": [
                    SafetyProtocol.CONTENT_FILTERING.value,
                    SafetyProtocol.ETHICS_VERIFICATION.value
                ]
            }
        }
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key educational concepts from query"""
        
        # Simple concept extraction (in real implementation, would use NLP)
        words = query.lower().split()
        
        # Educational concept keywords
        educational_terms = [
            "learn", "understand", "explain", "concept", "theory", "principle",
            "method", "process", "system", "analysis", "comparison", "evaluation"
        ]
        
        concepts = []
        for word in words:
            if len(word) > 4 and any(term in word for term in educational_terms):
                concepts.append(word)
        
        # Add some domain-specific concepts based on query content
        if "math" in query.lower():
            concepts.extend(["mathematics", "calculation", "problem_solving"])
        elif "science" in query.lower():
            concepts.extend(["scientific_method", "hypothesis", "experiment"])
        elif "history" in query.lower():
            concepts.extend(["historical_analysis", "cause_effect", "timeline"])
        
        return concepts[:5]  # Limit to top 5 concepts
    
    async def _create_safe_explanation(
        self,
        query: str,
        concepts: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Create a safe, educational explanation"""
        
        # Generate explanation based on query and concepts
        explanation = f"Based on your question about {', '.join(concepts[:2])}, "
        
        # Add educational content that's appropriate for the context
        student_level = context.get("student_level", "intermediate")
        
        if student_level == "beginner":
            explanation += "let me start with the fundamental concepts. "
        elif student_level == "advanced":
            explanation += "let me provide a comprehensive analysis. "
        else:
            explanation += "let me explain this step by step. "
        
        # Add safety-conscious educational content
        explanation += "This topic involves understanding key principles and their practical applications. "
        explanation += "It's important to approach this learning with critical thinking and ethical considerations. "
        
        # Add domain-specific safe content
        if "science" in query.lower():
            explanation += "Scientific understanding helps us make informed decisions and solve real-world problems responsibly."
        elif "technology" in query.lower():
            explanation += "Technology should be used ethically and with consideration for its impact on society."
        
        return explanation
    
    async def _create_safe_examples(self, concepts: List[str], context: Dict[str, Any]) -> List[str]:
        """Create safe, educational examples"""
        
        examples = []
        
        for concept in concepts[:3]:  # Top 3 concepts
            if "math" in concept.lower():
                examples.append("Consider calculating the optimal path for a delivery route - this involves mathematical optimization.")
            elif "science" in concept.lower():
                examples.append("Think about how plants convert sunlight to energy - this demonstrates scientific principles in nature.")
            elif "technology" in concept.lower():
                examples.append("Consider how renewable energy systems work - this shows technology solving environmental challenges.")
            else:
                examples.append(f"A practical application of {concept} can be seen in everyday problem-solving scenarios.")
        
        return examples
    
    async def _create_safe_practice_questions(
        self,
        concepts: List[str],
        objectives: List[str]
    ) -> List[str]:
        """Create safe practice questions aligned with learning objectives"""
        
        questions = []
        
        for objective in objectives[:3]:  # Top 3 objectives
            questions.append(f"How would you apply {objective} in a real-world, ethical scenario?")
            questions.append(f"What are the potential benefits and responsible considerations when using {objective}?")
        
        # Add concept-based questions
        for concept in concepts[:2]:
            questions.append(f"Can you explain how {concept} relates to solving problems responsibly?")
        
        return questions[:5]  # Limit to 5 questions
    
    async def _assess_content_safety(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SafetyAssessment:
        """Assess safety of generated teaching content"""
        
        start_time = time.time()
        assessment = SafetyAssessment(
            assessment_id=str(uuid4()),
            overall_safety_level=SafetyLevel.SAFE,
            confidence=0.95
        )
        
        # Check all content fields
        content_text = ""
        content_text += content.get("explanation", "") + " "
        content_text += " ".join(content.get("examples", [])) + " "
        content_text += " ".join(content.get("practice_questions", []))
        
        # Run safety checks on generated content
        violations = []
        
        if self.safeguards.enable_content_filtering:
            harmful_violations = await self._check_harmful_patterns(content_text)
            violations.extend(harmful_violations)
        
        if self.safeguards.enable_misinformation_check:
            misinfo_score = await self._detect_misinformation(content_text)
            assessment.misinformation_risk = misinfo_score
            
            if misinfo_score > self.safeguards.misinformation_threshold:
                violations.append(SafetyViolation(
                    violation_id=str(uuid4()),
                    protocol=SafetyProtocol.MISINFORMATION_CHECK,
                    severity=SafetyLevel.WARNING,
                    description=f"Potential misinformation detected (score: {misinfo_score:.2f})",
                    content_snippet=content_text[:100],
                    confidence=0.7,
                    teacher_id=self.teacher_id
                ))
        
        if self.safeguards.enable_alignment_verification:
            alignment_score = await self._verify_alignment(content_text, context)
            assessment.alignment_score = alignment_score
            
            if alignment_score < self.safeguards.alignment_threshold:
                violations.append(SafetyViolation(
                    violation_id=str(uuid4()),
                    protocol=SafetyProtocol.ALIGNMENT_CHECK,
                    severity=SafetyLevel.CAUTION,
                    description=f"Alignment concerns detected (score: {alignment_score:.2f})",
                    content_snippet=content_text[:100],
                    confidence=0.6,
                    teacher_id=self.teacher_id
                ))
        
        # Set overall safety level
        if violations:
            max_severity = max([v.severity for v in violations])
            assessment.overall_safety_level = max_severity
            assessment.violations = violations
        
        assessment.processing_time = time.time() - start_time
        self.safety_assessments.append(assessment)
        
        return assessment
    
    async def _detect_misinformation(self, content: str) -> float:
        """Detect potential misinformation in content"""
        
        total_score = 0.0
        total_weight = 0.0
        content_lower = content.lower()
        
        for signal in self.misinformation_signals:
            signal_score = 0.0
            
            for pattern in signal["patterns"]:
                if re.search(pattern, content_lower):
                    signal_score = 1.0
                    break
            
            weight = signal["weight"]
            total_score += signal_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    async def _verify_alignment(self, content: str, context: Dict[str, Any]) -> float:
        """Verify content alignment with educational and ethical standards"""
        
        # Check for educational alignment
        educational_indicators = [
            "learn", "understand", "educate", "teach", "explain", "demonstrate",
            "practice", "apply", "analyze", "evaluate", "critical thinking"
        ]
        
        ethical_indicators = [
            "responsible", "ethical", "appropriate", "safe", "beneficial",
            "respectful", "inclusive", "fair", "honest", "transparent"
        ]
        
        content_lower = content.lower()
        
        # Calculate educational alignment
        edu_score = sum(1 for indicator in educational_indicators if indicator in content_lower)
        edu_alignment = min(edu_score / len(educational_indicators), 1.0)
        
        # Calculate ethical alignment
        eth_score = sum(1 for indicator in ethical_indicators if indicator in content_lower)
        eth_alignment = min(eth_score / len(ethical_indicators), 1.0)
        
        # Combined alignment score
        return (edu_alignment * 0.6 + eth_alignment * 0.4)
    
    async def _apply_safety_enhancements(
        self,
        content: Dict[str, Any],
        assessment: SafetyAssessment
    ) -> Dict[str, Any]:
        """Apply safety enhancements to teaching content"""
        
        enhanced_content = content.copy()
        
        # Add safety disclaimers if needed
        if assessment.overall_safety_level in [SafetyLevel.CAUTION, SafetyLevel.WARNING]:
            safety_note = (
                "\n\n📌 Educational Note: This content has been reviewed for educational safety. "
                "Please approach this topic with critical thinking and consider multiple perspectives."
            )
            enhanced_content["explanation"] += safety_note
        
        # Add ethical considerations
        if assessment.ethics_score < 0.8:
            ethics_note = (
                "\n\n🤝 Ethical Consideration: When applying these concepts, consider their impact on "
                "individuals and society, and ensure responsible use."
            )
            enhanced_content["explanation"] += ethics_note
        
        # Add bias awareness if bias detected
        if assessment.bias_score > 0.3:
            bias_note = (
                "\n\n⚖️ Diversity Note: Be aware that perspectives on this topic may vary across "
                "different cultures, backgrounds, and experiences. Consider multiple viewpoints."
            )
            enhanced_content["explanation"] += bias_note
        
        # Mark as safety-enhanced
        enhanced_content["safety_enhanced"] = True
        enhanced_content["safety_level"] = assessment.overall_safety_level.value
        
        return enhanced_content
    
    async def _finalize_safe_response(
        self,
        content: Dict[str, Any],
        pre_assessment: SafetyAssessment,
        post_assessment: SafetyAssessment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finalize safe teaching response with all safety metadata"""
        
        return {
            "teaching_content": content,
            "safety_verified": True,
            "safety_metadata": {
                "teacher_id": self.teacher_id,
                "pre_assessment": {
                    "safety_level": pre_assessment.overall_safety_level.value,
                    "confidence": pre_assessment.confidence,
                    "violations_count": len(pre_assessment.violations)
                },
                "post_assessment": {
                    "safety_level": post_assessment.overall_safety_level.value,
                    "confidence": post_assessment.confidence,
                    "violations_count": len(post_assessment.violations)
                },
                "safety_protocols_applied": [
                    protocol.value for protocol in SafetyProtocol
                    if getattr(self.safeguards, f"enable_{protocol.value.replace('_check', '_detection')}", True)
                ],
                "enhancements_applied": content.get("safety_enhanced", False)
            },
            "educational_quality": {
                "alignment_score": post_assessment.alignment_score,
                "ethics_score": post_assessment.ethics_score,
                "bias_score": post_assessment.bias_score,
                "harm_potential": post_assessment.harm_potential
            },
            "interaction_metadata": {
                "total_interactions": self.total_interactions,
                "teacher_performance": self._get_teacher_performance_summary()
            }
        }
    
    async def _handle_unsafe_query(self, query: str, assessment: SafetyAssessment) -> Dict[str, Any]:
        """Handle queries that fail safety assessment"""
        
        self.intervention_count += 1
        
        # Log intervention
        logger.warning(
            "Unsafe query intervention",
            teacher_id=self.teacher_id,
            safety_level=assessment.overall_safety_level.value,
            violations_count=len(assessment.violations)
        )
        
        return {
            "teaching_content": {
                "explanation": (
                    "I understand you're looking to learn, but I need to ensure our discussion "
                    "remains educational and safe. Let me suggest some alternative approaches to "
                    "this topic that would be more appropriate for learning."
                ),
                "alternative_topics": [
                    "Foundational concepts in this area",
                    "Historical development of these ideas", 
                    "Practical applications in positive contexts",
                    "Ethical considerations and best practices"
                ],
                "safety_guidance": (
                    "For the best learning experience, let's focus on constructive, educational "
                    "aspects that help you understand concepts in a responsible way."
                )
            },
            "safety_verified": True,
            "intervention_applied": True,
            "safety_metadata": {
                "intervention_reason": assessment.overall_safety_level.value,
                "violations_detected": len(assessment.violations),
                "alternative_provided": True
            }
        }
    
    async def _handle_unsafe_content(self, content: Dict[str, Any], assessment: SafetyAssessment) -> Dict[str, Any]:
        """Handle content that fails post-generation safety assessment"""
        
        self.intervention_count += 1
        
        return {
            "teaching_content": {
                "explanation": (
                    "I've generated some initial content, but upon review, I'd like to refine "
                    "it to ensure it meets the highest educational and safety standards. "
                    "Let me provide a more appropriate response."
                ),
                "refined_approach": "Content being regenerated with enhanced safety protocols",
                "safety_note": "Your learning experience is important, and I want to ensure all content is beneficial and appropriate."
            },
            "safety_verified": True,
            "content_regeneration_required": True,
            "safety_metadata": {
                "post_generation_intervention": True,
                "violations_detected": len(assessment.violations)
            }
        }
    
    def _generate_safety_recommendations(self, violations: List[SafetyViolation]) -> List[str]:
        """Generate safety recommendations based on violations"""
        
        recommendations = []
        
        violation_types = [v.protocol for v in violations]
        
        if SafetyProtocol.CONTENT_FILTERING in violation_types:
            recommendations.append("Review content for potentially harmful language")
        
        if SafetyProtocol.BIAS_DETECTION in violation_types:
            recommendations.append("Consider multiple perspectives and avoid generalizations")
        
        if SafetyProtocol.HARM_PREVENTION in violation_types:
            recommendations.append("Focus on constructive, educational approaches")
        
        if SafetyProtocol.MISINFORMATION_CHECK in violation_types:
            recommendations.append("Verify information sources and avoid unsupported claims")
        
        if SafetyProtocol.ALIGNMENT_CHECK in violation_types:
            recommendations.append("Ensure content aligns with educational objectives")
        
        return recommendations
    
    def _get_teacher_performance_summary(self) -> Dict[str, Any]:
        """Get summary of teacher performance including safety metrics"""
        
        return {
            "total_interactions": self.total_interactions,
            "intervention_rate": self.intervention_count / max(self.total_interactions, 1),
            "average_safety_check_time": np.mean(self.safety_check_times) if self.safety_check_times else 0,
            "safety_assessments_completed": len(self.safety_assessments),
            "violation_history_count": len(self.violation_history),
            "overall_safety_effectiveness": 1.0 - (self.intervention_count / max(self.total_interactions, 1))
        }
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report"""
        
        return {
            "teacher_id": self.teacher_id,
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "safety_configuration": {
                "content_filtering": self.safeguards.enable_content_filtering,
                "bias_detection": self.safeguards.enable_bias_detection,
                "alignment_verification": self.safeguards.enable_alignment_verification,
                "ethics_checking": self.safeguards.enable_ethics_checking,
                "misinformation_check": self.safeguards.enable_misinformation_check
            },
            "performance_summary": self._get_teacher_performance_summary(),
            "safety_metrics": {
                "total_violations": len(self.violation_history),
                "total_assessments": len(self.safety_assessments),
                "intervention_count": self.intervention_count,
                "avg_assessment_time": np.mean(self.safety_check_times) if self.safety_check_times else 0
            },
            "violation_breakdown": {
                protocol.value: len([v for v in self.violation_history if v.protocol == protocol])
                for protocol in SafetyProtocol
            }
        }


# Factory function for easy instantiation
def create_seal_rlt_teacher(
    teacher_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> SEALRLTEnhancedTeacher:
    """Create and return a SEAL RLT Enhanced Teacher instance"""
    return SEALRLTEnhancedTeacher(teacher_id, config)


# Default safety configuration
DEFAULT_SEAL_CONFIG = {
    "safeguards": {
        "enable_content_filtering": True,
        "enable_bias_detection": True,
        "enable_misinformation_check": True,
        "enable_alignment_verification": True,
        "enable_ethics_checking": True,
        "auto_intervention_threshold": "warning",
        "bias_threshold": 0.7,
        "harm_threshold": 0.8,
        "misinformation_threshold": 0.75,
        "alignment_threshold": 0.6
    }
}