"""
Red Team Student Content Filtering Implementation

🛡️ RED TEAM STUDENT CONTENT FILTERING (Item 4.2):
- Age-appropriate content validation with developmental stage awareness
- Harmful content blocking with multi-layer detection systems
- Privacy protection measures with comprehensive data safeguarding
- Safe learning environment maintenance with real-time monitoring
- Parental control integration and oversight capabilities

This module implements comprehensive safety systems for student learning
environments, based on Red Team adversarial testing methodologies to
ensure robust protection against harmful content and privacy violations.
"""

import asyncio
import json
import re
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import (
    PRSMBaseModel, TimestampMixin, SafetyLevel, TaskStatus
)
from prsm.agents.executors.model_executor import ModelExecutor

logger = structlog.get_logger(__name__)


class AgeGroup(str, Enum):
    """Age groups for content appropriateness"""
    EARLY_CHILDHOOD = "early_childhood"  # 3-6 years
    ELEMENTARY = "elementary"  # 7-11 years
    MIDDLE_SCHOOL = "middle_school"  # 12-14 years
    HIGH_SCHOOL = "high_school"  # 15-18 years
    ADULT = "adult"  # 18+ years


class ContentCategory(str, Enum):
    """Content categories for filtering"""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    SOCIAL_INTERACTION = "social_interaction"
    CREATIVE_EXPRESSION = "creative_expression"
    ASSESSMENT = "assessment"
    COMMUNICATION = "communication"
    RESEARCH = "research"
    COLLABORATION = "collaboration"


class HarmfulContentType(str, Enum):
    """Types of harmful content to detect and block"""
    VIOLENCE = "violence"
    SEXUAL_CONTENT = "sexual_content"
    HATE_SPEECH = "hate_speech"
    BULLYING = "bullying"
    SELF_HARM = "self_harm"
    SUBSTANCE_ABUSE = "substance_abuse"
    DANGEROUS_ACTIVITIES = "dangerous_activities"
    INAPPROPRIATE_LANGUAGE = "inappropriate_language"
    MISINFORMATION = "misinformation"
    SCAMS_FRAUD = "scams_fraud"
    PRIVACY_VIOLATIONS = "privacy_violations"
    PREDATORY_BEHAVIOR = "predatory_behavior"
    EXTREMIST_CONTENT = "extremist_content"
    CYBERBULLYING = "cyberbullying"
    DOXXING = "doxxing"


class PrivacyRiskType(str, Enum):
    """Types of privacy risks to detect and prevent"""
    PERSONAL_INFORMATION_SHARING = "personal_information_sharing"
    LOCATION_DISCLOSURE = "location_disclosure"
    CONTACT_INFORMATION_LEAK = "contact_information_leak"
    FAMILY_INFORMATION_EXPOSURE = "family_information_exposure"
    SCHOOL_INFORMATION_LEAK = "school_information_leak"
    FINANCIAL_INFORMATION = "financial_information"
    BIOMETRIC_DATA_EXPOSURE = "biometric_data_exposure"
    SOCIAL_MEDIA_INTEGRATION = "social_media_integration"
    THIRD_PARTY_TRACKING = "third_party_tracking"
    DATA_RETENTION_VIOLATION = "data_retention_violation"


class FilterAction(str, Enum):
    """Actions to take when content issues are detected"""
    ALLOW = "allow"
    BLOCK = "block"
    MODERATE = "moderate"
    FLAG_FOR_REVIEW = "flag_for_review"
    REQUIRE_SUPERVISION = "require_supervision"
    REQUEST_CONSENT = "request_consent"
    ANONYMIZE = "anonymize"
    REDIRECT = "redirect"


class ContentSafetyScore(PRSMBaseModel):
    """Safety score for content evaluation"""
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall safety score (1.0 = completely safe)")
    age_appropriateness: float = Field(ge=0.0, le=1.0)
    educational_value: float = Field(ge=0.0, le=1.0)
    harm_risk_score: float = Field(ge=0.0, le=1.0, description="Risk of harm (0.0 = no risk)")
    privacy_risk_score: float = Field(ge=0.0, le=1.0, description="Privacy risk (0.0 = no risk)")
    toxicity_score: float = Field(ge=0.0, le=1.0, description="Toxicity level (0.0 = not toxic)")
    
    # Detailed risk breakdowns
    violence_risk: float = Field(ge=0.0, le=1.0, default=0.0)
    sexual_content_risk: float = Field(ge=0.0, le=1.0, default=0.0)
    language_appropriateness: float = Field(ge=0.0, le=1.0, default=1.0)
    emotional_safety: float = Field(ge=0.0, le=1.0, default=1.0)


class HarmfulContentDetection(TimestampMixin):
    """Detection result for harmful content"""
    detection_id: UUID = Field(default_factory=uuid4)
    student_id: str
    content_id: str
    content_type: ContentCategory
    
    # Detection Results
    harmful_content_types: List[HarmfulContentType] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    severity_level: SafetyLevel = SafetyLevel.NONE
    
    # Content Analysis
    detected_patterns: List[str] = Field(default_factory=list)
    context_analysis: Dict[str, Any] = Field(default_factory=dict)
    risk_factors: List[str] = Field(default_factory=list)
    
    # Mitigation
    recommended_action: FilterAction = FilterAction.ALLOW
    mitigation_strategies: List[str] = Field(default_factory=list)
    safe_alternatives: List[str] = Field(default_factory=list)
    
    # Additional Metadata
    age_group: AgeGroup
    detection_method: str = "multi_layer_analysis"
    false_positive_likelihood: float = Field(ge=0.0, le=1.0, default=0.1)


class PrivacyProtectionResult(TimestampMixin):
    """Privacy protection analysis result"""
    protection_id: UUID = Field(default_factory=uuid4)
    student_id: str
    content_id: str
    
    # Privacy Risk Assessment
    privacy_risks: List[PrivacyRiskType] = Field(default_factory=list)
    risk_severity: SafetyLevel = SafetyLevel.NONE
    personal_data_detected: List[str] = Field(default_factory=list)
    
    # Data Protection Analysis
    data_collection_concerns: List[str] = Field(default_factory=list)
    third_party_sharing_risks: List[str] = Field(default_factory=list)
    retention_policy_violations: List[str] = Field(default_factory=list)
    
    # Protective Actions
    protection_actions: List[FilterAction] = Field(default_factory=list)
    anonymization_required: bool = False
    parental_consent_required: bool = False
    
    # Compliance
    coppa_compliance: bool = True
    ferpa_compliance: bool = True
    gdpr_compliance: bool = True
    local_privacy_compliance: bool = True


class AgeAppropriatenessAssessment(TimestampMixin):
    """Age appropriateness assessment result"""
    assessment_id: UUID = Field(default_factory=uuid4)
    content_id: str
    target_age_group: AgeGroup
    
    # Appropriateness Analysis
    recommended_age_groups: List[AgeGroup] = Field(default_factory=list)
    inappropriate_elements: List[str] = Field(default_factory=list)
    developmental_considerations: List[str] = Field(default_factory=list)
    
    # Content Adaptation
    adaptation_suggestions: List[str] = Field(default_factory=list)
    complexity_adjustment: Optional[str] = None
    supervision_recommendation: bool = False
    
    # Educational Value
    learning_objectives_alignment: float = Field(ge=0.0, le=1.0, default=0.5)
    cognitive_load_assessment: float = Field(ge=0.0, le=1.0, default=0.5)
    engagement_prediction: float = Field(ge=0.0, le=1.0, default=0.5)


class SafeLearningEnvironmentStatus(TimestampMixin):
    """Status of safe learning environment"""
    status_id: UUID = Field(default_factory=uuid4)
    student_id: str
    environment_id: str
    
    # Safety Metrics
    overall_safety_score: float = Field(ge=0.0, le=1.0)
    content_safety_score: float = Field(ge=0.0, le=1.0)
    interaction_safety_score: float = Field(ge=0.0, le=1.0)
    privacy_protection_score: float = Field(ge=0.0, le=1.0)
    
    # Recent Activity Analysis
    recent_violations: List[Dict[str, Any]] = Field(default_factory=list)
    safety_trends: Dict[str, float] = Field(default_factory=dict)
    intervention_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Environment Configuration
    active_filters: List[str] = Field(default_factory=list)
    parental_controls: Dict[str, Any] = Field(default_factory=dict)
    supervision_level: str = "standard"  # minimal, standard, strict, supervised
    
    # Recommendations
    safety_recommendations: List[str] = Field(default_factory=list)
    environment_adjustments: List[str] = Field(default_factory=list)


class RedTeamStudentSafetyEngine:
    """
    Red Team Student Safety Engine
    
    Comprehensive safety system for student learning environments with:
    - Multi-layer harmful content detection and blocking
    - Age-appropriate content validation with developmental awareness
    - Privacy protection with COPPA, FERPA, and GDPR compliance
    - Real-time safety monitoring and intervention
    - Parental control integration and reporting
    """
    
    def __init__(self, student_id: str, age_group: AgeGroup):
        self.student_id = student_id
        self.age_group = age_group
        self.model_executor = ModelExecutor()
        
        # Safety Databases
        self.harmful_patterns = self._initialize_harmful_patterns()
        self.privacy_patterns = self._initialize_privacy_patterns()
        self.age_appropriateness_rules = self._initialize_age_rules()
        
        # Safety Thresholds by Age Group
        self.safety_thresholds = self._get_age_appropriate_thresholds(age_group)
        
        # Detection History
        self.detection_history: List[HarmfulContentDetection] = []
        self.privacy_assessments: List[PrivacyProtectionResult] = []
        self.age_assessments: List[AgeAppropriatenessAssessment] = []
        
        # Real-time Monitoring
        self.active_monitoring = True
        self.intervention_queue: List[Dict[str, Any]] = []
        
        logger.info("RedTeamStudentSafetyEngine initialized",
                   student_id=student_id,
                   age_group=age_group)
    
    def _initialize_harmful_patterns(self) -> Dict[HarmfulContentType, List[str]]:
        """Initialize harmful content detection patterns"""
        return {
            HarmfulContentType.VIOLENCE: [
                r'\b(kill|murder|violence|assault|attack|fight|hurt|pain|blood|weapon|gun|knife)\b',
                r'\b(stab|shoot|punch|kick|hit|slap|abuse|torture|harm)\b',
                r'\b(war|battle|conflict|aggression|hostile|threatening)\b'
            ],
            HarmfulContentType.SEXUAL_CONTENT: [
                r'\b(sex|sexual|nude|naked|porn|adult|intimate|erotic)\b',
                r'\b(breast|genital|reproduction|anatomy|puberty)\b',  # Context-dependent
                r'\b(dating|relationship|romance|love|attraction)\b'  # Age-dependent
            ],
            HarmfulContentType.HATE_SPEECH: [
                r'\b(hate|discrimination|prejudice|racism|sexism|homophobia)\b',
                r'\b(stereotype|bias|intolerance|exclusion|marginalize)\b',
                r'\b(slur|offensive|derogatory|insulting|demeaning)\b'
            ],
            HarmfulContentType.BULLYING: [
                r'\b(bully|tease|mock|ridicule|humiliate|embarrass)\b',
                r'\b(exclude|isolate|ignore|reject|ostracize)\b',
                r'\b(mean|cruel|nasty|rude|hurtful|unkind)\b'
            ],
            HarmfulContentType.SELF_HARM: [
                r'\b(suicide|self.harm|cutting|depression|anxiety|stress)\b',
                r'\b(hurt.yourself|end.it.all|give.up|hopeless|worthless)\b',
                r'\b(eating.disorder|anorexia|bulimia|body.image)\b'
            ],
            HarmfulContentType.SUBSTANCE_ABUSE: [
                r'\b(drugs|alcohol|smoking|drinking|addiction|substance)\b',
                r'\b(cocaine|marijuana|heroin|meth|pills|prescription)\b',
                r'\b(beer|wine|vodka|whiskey|drunk|high|stoned)\b'
            ],
            HarmfulContentType.DANGEROUS_ACTIVITIES: [
                r'\b(dangerous|risky|unsafe|hazardous|reckless)\b',
                r'\b(fire|explosive|chemical|poison|toxic|radioactive)\b',
                r'\b(climbing|jumping|driving|racing|extreme.sports)\b'
            ],
            HarmfulContentType.INAPPROPRIATE_LANGUAGE: [
                r'\b(profanity|curse|swear|vulgar|obscene|inappropriate)\b',
                r'\b(damn|hell|stupid|idiot|moron|loser|freak)\b',
                r'\b(shut.up|go.away|i.hate.you|you.suck)\b'
            ],
            HarmfulContentType.MISINFORMATION: [
                r'\b(fake.news|conspiracy|hoax|lie|false|untrue)\b',
                r'\b(medical.advice|diagnosis|treatment|cure|miracle)\b',
                r'\b(political|election|voting|government|authority)\b'
            ],
            HarmfulContentType.PRIVACY_VIOLATIONS: [
                r'\b(personal.information|address|phone|email|password)\b',
                r'\b(full.name|birthday|age|school|location|home)\b',
                r'\b(parent|family|sibling|relative|guardian)\b'
            ]
        }
    
    def _initialize_privacy_patterns(self) -> Dict[PrivacyRiskType, List[str]]:
        """Initialize privacy risk detection patterns"""
        return {
            PrivacyRiskType.PERSONAL_INFORMATION_SHARING: [
                r'\b(my.name.is|i.am|call.me|known.as)\b',
                r'\b(first.name|last.name|full.name|real.name)\b',
                r'\b(age|birthday|birth.date|born.on)\b'
            ],
            PrivacyRiskType.LOCATION_DISCLOSURE: [
                r'\b(i.live|my.address|my.house|my.home)\b',
                r'\b(city|state|country|zip.code|postal)\b',
                r'\b(near|close.to|next.to|around|neighborhood)\b'
            ],
            PrivacyRiskType.CONTACT_INFORMATION_LEAK: [
                r'\b(phone.number|cell.phone|telephone|mobile)\b',
                r'\b(email|gmail|yahoo|hotmail|contact.me)\b',
                r'\b(social.media|facebook|instagram|snapchat|tiktok)\b'
            ],
            PrivacyRiskType.FAMILY_INFORMATION_EXPOSURE: [
                r'\b(my.mom|my.dad|my.parent|my.family)\b',
                r'\b(brother|sister|sibling|grandmother|grandfather)\b',
                r'\b(family.member|relative|guardian|caregiver)\b'
            ],
            PrivacyRiskType.SCHOOL_INFORMATION_LEAK: [
                r'\b(my.school|attend|go.to|study.at)\b',
                r'\b(teacher|principal|classroom|grade|class)\b',
                r'\b(school.name|university|college|academy)\b'
            ]
        }
    
    def _initialize_age_rules(self) -> Dict[AgeGroup, Dict[str, Any]]:
        """Initialize age-appropriate content rules"""
        return {
            AgeGroup.EARLY_CHILDHOOD: {
                "vocabulary_level": "basic",
                "concept_complexity": "concrete",
                "attention_span_minutes": 15,
                "supervision_required": True,
                "restricted_topics": [
                    "violence", "death", "complex_emotions", "abstract_concepts",
                    "romantic_relationships", "body_changes", "political_issues"
                ],
                "encouraged_topics": [
                    "colors", "shapes", "animals", "family", "friendship",
                    "basic_numbers", "letters", "simple_stories"
                ]
            },
            AgeGroup.ELEMENTARY: {
                "vocabulary_level": "elementary",
                "concept_complexity": "simple_abstract",
                "attention_span_minutes": 30,
                "supervision_required": False,
                "restricted_topics": [
                    "graphic_violence", "sexual_content", "substance_abuse",
                    "complex_political_issues", "existential_topics"
                ],
                "encouraged_topics": [
                    "science", "nature", "history", "geography", "basic_math",
                    "reading", "creativity", "problem_solving", "friendship"
                ]
            },
            AgeGroup.MIDDLE_SCHOOL: {
                "vocabulary_level": "intermediate",
                "concept_complexity": "moderate_abstract",
                "attention_span_minutes": 45,
                "supervision_required": False,
                "restricted_topics": [
                    "explicit_violence", "sexual_content", "substance_abuse",
                    "extreme_political_views", "self_harm_detailed"
                ],
                "encouraged_topics": [
                    "advanced_science", "literature", "history", "current_events",
                    "social_issues", "identity", "peer_relationships", "ethics"
                ]
            },
            AgeGroup.HIGH_SCHOOL: {
                "vocabulary_level": "advanced",
                "concept_complexity": "complex_abstract",
                "attention_span_minutes": 60,
                "supervision_required": False,
                "restricted_topics": [
                    "graphic_violence", "explicit_sexual_content", "illegal_activities",
                    "dangerous_instructions", "hate_speech"
                ],
                "encouraged_topics": [
                    "critical_thinking", "complex_science", "philosophy",
                    "political_science", "economics", "career_planning",
                    "college_preparation", "social_responsibility"
                ]
            },
            AgeGroup.ADULT: {
                "vocabulary_level": "unrestricted",
                "concept_complexity": "unrestricted",
                "attention_span_minutes": 120,
                "supervision_required": False,
                "restricted_topics": [
                    "illegal_activities", "harmful_instructions", "extreme_hate_speech"
                ],
                "encouraged_topics": ["unrestricted"]
            }
        }
    
    def _get_age_appropriate_thresholds(self, age_group: AgeGroup) -> Dict[str, float]:
        """Get safety thresholds appropriate for age group"""
        thresholds = {
            AgeGroup.EARLY_CHILDHOOD: {
                "violence_tolerance": 0.05,
                "language_tolerance": 0.02,
                "complexity_tolerance": 0.3,
                "supervision_threshold": 0.2
            },
            AgeGroup.ELEMENTARY: {
                "violence_tolerance": 0.1,
                "language_tolerance": 0.05,
                "complexity_tolerance": 0.5,
                "supervision_threshold": 0.3
            },
            AgeGroup.MIDDLE_SCHOOL: {
                "violence_tolerance": 0.2,
                "language_tolerance": 0.1,
                "complexity_tolerance": 0.7,
                "supervision_threshold": 0.4
            },
            AgeGroup.HIGH_SCHOOL: {
                "violence_tolerance": 0.3,
                "language_tolerance": 0.15,
                "complexity_tolerance": 0.9,
                "supervision_threshold": 0.5
            },
            AgeGroup.ADULT: {
                "violence_tolerance": 0.5,
                "language_tolerance": 0.3,
                "complexity_tolerance": 1.0,
                "supervision_threshold": 0.8
            }
        }
        
        return thresholds.get(age_group, thresholds[AgeGroup.ELEMENTARY])
    
    async def analyze_content_safety(
        self,
        content: str,
        content_type: ContentCategory,
        context: Optional[Dict[str, Any]] = None
    ) -> HarmfulContentDetection:
        """
        Comprehensive content safety analysis using multi-layer detection
        """
        logger.debug("Analyzing content safety",
                    student_id=self.student_id,
                    content_length=len(content),
                    content_type=content_type)
        
        # Initialize detection result
        detection = HarmfulContentDetection(
            student_id=self.student_id,
            content_id=str(uuid4()),
            content_type=content_type,
            age_group=self.age_group
        )
        
        # Multi-layer detection
        pattern_results = await self._pattern_based_detection(content)
        ml_results = await self._ml_based_detection(content, context)
        contextual_results = await self._contextual_analysis(content, content_type, context)
        
        # Combine detection results
        all_detected_types = set()
        all_detected_types.update(pattern_results.get("harmful_types", []))
        all_detected_types.update(ml_results.get("harmful_types", []))
        all_detected_types.update(contextual_results.get("harmful_types", []))
        
        detection.harmful_content_types = list(all_detected_types)
        
        # Calculate confidence scores
        detection.confidence_scores = await self._calculate_detection_confidence(
            pattern_results, ml_results, contextual_results
        )
        
        # Determine severity level
        detection.severity_level = await self._determine_severity_level(
            detection.harmful_content_types, detection.confidence_scores
        )
        
        # Extract detected patterns and risk factors
        detection.detected_patterns = pattern_results.get("patterns", [])
        detection.risk_factors = contextual_results.get("risk_factors", [])
        detection.context_analysis = contextual_results.get("context", {})
        
        # Determine recommended action
        detection.recommended_action = await self._determine_filter_action(detection)
        
        # Generate mitigation strategies
        detection.mitigation_strategies = await self._generate_mitigation_strategies(detection)
        detection.safe_alternatives = await self._suggest_safe_alternatives(detection)
        
        # Calculate false positive likelihood
        detection.false_positive_likelihood = await self._calculate_false_positive_risk(
            detection, content, context
        )
        
        # Store detection
        self.detection_history.append(detection)
        
        logger.info("Content safety analysis completed",
                   student_id=self.student_id,
                   severity=detection.severity_level,
                   action=detection.recommended_action,
                   harmful_types=len(detection.harmful_content_types))
        
        return detection
    
    async def _pattern_based_detection(self, content: str) -> Dict[str, Any]:
        """Pattern-based harmful content detection"""
        detected_types = []
        detected_patterns = []
        
        content_lower = content.lower()
        
        for harmful_type, patterns in self.harmful_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    detected_types.append(harmful_type)
                    detected_patterns.append(pattern)
                    break  # One pattern per type is enough
        
        return {
            "harmful_types": detected_types,
            "patterns": detected_patterns,
            "method": "pattern_matching"
        }
    
    async def _ml_based_detection(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Machine learning-based content analysis"""
        try:
            # Use model executor for ML-based analysis
            analysis_prompt = f"""
            Analyze the following content for harmful elements suitable for a {self.age_group.value} student:
            
            Content: {content[:500]}...
            
            Identify any harmful content types from: violence, sexual content, hate speech,
            bullying, self-harm, substance abuse, dangerous activities, inappropriate language,
            misinformation, privacy violations.
            
            Return assessment with confidence scores.
            """
            
            response = await self.model_executor.process({
                "task": analysis_prompt,
                "models": ["gpt-3.5-turbo"],
                "parallel": False
            })
            
            if response and response[0].success:
                # Parse ML response (simplified for implementation)
                return {
                    "harmful_types": [],  # Would parse from actual ML response
                    "confidence_scores": {},
                    "method": "machine_learning"
                }
            
        except Exception as e:
            logger.error("ML-based detection failed", error=str(e))
        
        return {
            "harmful_types": [],
            "confidence_scores": {},
            "method": "machine_learning_fallback"
        }
    
    async def _contextual_analysis(
        self,
        content: str,
        content_type: ContentCategory,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Contextual content analysis considering educational setting"""
        risk_factors = []
        harmful_types = []
        context_info = {}
        
        # Educational context analysis
        if content_type == ContentCategory.EDUCATIONAL:
            # More lenient for educational content
            if "science" in content.lower() and "reproduction" in content.lower():
                if self.age_group in [AgeGroup.EARLY_CHILDHOOD, AgeGroup.ELEMENTARY]:
                    risk_factors.append("age_inappropriate_science_topic")
        
        # Social interaction context
        if content_type == ContentCategory.SOCIAL_INTERACTION:
            # Stricter monitoring for social content
            if "meet" in content.lower() and "person" in content.lower():
                risk_factors.append("potential_stranger_interaction")
        
        # Age-specific context analysis
        age_rules = self.age_appropriateness_rules.get(self.age_group, {})
        restricted_topics = age_rules.get("restricted_topics", [])
        
        for topic in restricted_topics:
            if topic.replace("_", " ") in content.lower():
                risk_factors.append(f"age_restricted_topic_{topic}")
        
        context_info = {
            "content_type": content_type.value,
            "age_group": self.age_group.value,
            "educational_context": content_type == ContentCategory.EDUCATIONAL,
            "social_context": content_type == ContentCategory.SOCIAL_INTERACTION
        }
        
        return {
            "harmful_types": harmful_types,
            "risk_factors": risk_factors,
            "context": context_info,
            "method": "contextual_analysis"
        }
    
    async def _calculate_detection_confidence(
        self,
        pattern_results: Dict[str, Any],
        ml_results: Dict[str, Any],
        contextual_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate confidence scores for detected harmful content types"""
        confidence_scores = {}
        
        # Combine confidence from different detection methods
        all_types = set()
        all_types.update(pattern_results.get("harmful_types", []))
        all_types.update(ml_results.get("harmful_types", []))
        all_types.update(contextual_results.get("harmful_types", []))
        
        for harmful_type in all_types:
            confidence = 0.0
            method_count = 0
            
            # Pattern-based confidence
            if harmful_type in pattern_results.get("harmful_types", []):
                confidence += 0.7  # High confidence for pattern matches
                method_count += 1
            
            # ML-based confidence
            if harmful_type in ml_results.get("harmful_types", []):
                ml_confidence = ml_results.get("confidence_scores", {}).get(harmful_type.value, 0.8)
                confidence += ml_confidence
                method_count += 1
            
            # Contextual confidence
            if harmful_type in contextual_results.get("harmful_types", []):
                confidence += 0.6  # Moderate confidence for contextual detection
                method_count += 1
            
            # Average confidence across methods
            if method_count > 0:
                confidence_scores[harmful_type.value] = min(1.0, confidence / method_count)
        
        return confidence_scores
    
    async def _determine_severity_level(
        self,
        harmful_types: List[HarmfulContentType],
        confidence_scores: Dict[str, float]
    ) -> SafetyLevel:
        """Determine severity level based on detected harmful content"""
        if not harmful_types:
            return SafetyLevel.NONE
        
        # Severity mapping for different harmful content types
        severity_map = {
            HarmfulContentType.VIOLENCE: SafetyLevel.HIGH,
            HarmfulContentType.SEXUAL_CONTENT: SafetyLevel.HIGH,
            HarmfulContentType.HATE_SPEECH: SafetyLevel.HIGH,
            HarmfulContentType.SELF_HARM: SafetyLevel.CRITICAL,
            HarmfulContentType.PREDATORY_BEHAVIOR: SafetyLevel.CRITICAL,
            HarmfulContentType.BULLYING: SafetyLevel.MEDIUM,
            HarmfulContentType.INAPPROPRIATE_LANGUAGE: SafetyLevel.LOW,
            HarmfulContentType.MISINFORMATION: SafetyLevel.MEDIUM,
            HarmfulContentType.PRIVACY_VIOLATIONS: SafetyLevel.HIGH
        }
        
        max_severity = SafetyLevel.NONE
        
        for harmful_type in harmful_types:
            type_severity = severity_map.get(harmful_type, SafetyLevel.LOW)
            confidence = confidence_scores.get(harmful_type.value, 0.0)
            
            # Adjust severity based on confidence
            if confidence > 0.8:
                # High confidence - use full severity
                if type_severity.value == "critical":
                    max_severity = SafetyLevel.CRITICAL
                elif type_severity.value == "high" and max_severity.value != "critical":
                    max_severity = SafetyLevel.HIGH
                elif type_severity.value == "medium" and max_severity.value in ["none", "low"]:
                    max_severity = SafetyLevel.MEDIUM
                elif type_severity.value == "low" and max_severity.value == "none":
                    max_severity = SafetyLevel.LOW
            elif confidence > 0.5:
                # Medium confidence - reduce severity by one level
                if type_severity.value == "critical":
                    max_severity = SafetyLevel.HIGH
                elif type_severity.value == "high" and max_severity.value not in ["critical", "high"]:
                    max_severity = SafetyLevel.MEDIUM
                elif type_severity.value == "medium" and max_severity.value in ["none", "low"]:
                    max_severity = SafetyLevel.LOW
        
        return max_severity
    
    async def _determine_filter_action(self, detection: HarmfulContentDetection) -> FilterAction:
        """Determine appropriate filter action based on detection results"""
        severity = detection.severity_level
        age_group = detection.age_group
        confidence = max(detection.confidence_scores.values()) if detection.confidence_scores else 0.0
        
        # Critical severity - always block
        if severity == SafetyLevel.CRITICAL:
            return FilterAction.BLOCK
        
        # High severity - block or moderate based on age and confidence
        if severity == SafetyLevel.HIGH:
            if age_group in [AgeGroup.EARLY_CHILDHOOD, AgeGroup.ELEMENTARY]:
                return FilterAction.BLOCK
            elif confidence > 0.8:
                return FilterAction.BLOCK
            else:
                return FilterAction.MODERATE
        
        # Medium severity - moderate or flag based on context
        if severity == SafetyLevel.MEDIUM:
            if age_group == AgeGroup.EARLY_CHILDHOOD:
                return FilterAction.BLOCK
            elif age_group == AgeGroup.ELEMENTARY:
                return FilterAction.MODERATE
            else:
                return FilterAction.FLAG_FOR_REVIEW
        
        # Low severity - flag or allow based on age
        if severity == SafetyLevel.LOW:
            if age_group in [AgeGroup.EARLY_CHILDHOOD, AgeGroup.ELEMENTARY]:
                return FilterAction.MODERATE
            else:
                return FilterAction.FLAG_FOR_REVIEW
        
        return FilterAction.ALLOW
    
    async def _generate_mitigation_strategies(
        self,
        detection: HarmfulContentDetection
    ) -> List[str]:
        """Generate mitigation strategies for detected harmful content"""
        strategies = []
        
        for harmful_type in detection.harmful_content_types:
            if harmful_type == HarmfulContentType.VIOLENCE:
                strategies.extend([
                    "Replace violent language with constructive alternatives",
                    "Provide conflict resolution resources",
                    "Redirect to peaceful problem-solving activities"
                ])
            elif harmful_type == HarmfulContentType.BULLYING:
                strategies.extend([
                    "Implement anti-bullying intervention protocols",
                    "Provide empathy and kindness resources",
                    "Connect with school counselor if needed"
                ])
            elif harmful_type == HarmfulContentType.INAPPROPRIATE_LANGUAGE:
                strategies.extend([
                    "Suggest appropriate language alternatives",
                    "Provide communication skills resources",
                    "Model respectful language use"
                ])
            elif harmful_type == HarmfulContentType.PRIVACY_VIOLATIONS:
                strategies.extend([
                    "Educate about personal information safety",
                    "Remove or anonymize personal details",
                    "Implement privacy protection protocols"
                ])
        
        # Age-specific strategies
        if detection.age_group == AgeGroup.EARLY_CHILDHOOD:
            strategies.append("Provide simplified, age-appropriate explanations")
            strategies.append("Engage parent/guardian for guidance")
        
        return list(set(strategies))  # Remove duplicates
    
    async def _suggest_safe_alternatives(
        self,
        detection: HarmfulContentDetection
    ) -> List[str]:
        """Suggest safe alternative content or activities"""
        alternatives = []
        
        # Content type-specific alternatives
        if detection.content_type == ContentCategory.EDUCATIONAL:
            alternatives.extend([
                "Age-appropriate educational videos",
                "Interactive learning games",
                "Guided reading materials",
                "Educational coloring activities"
            ])
        elif detection.content_type == ContentCategory.SOCIAL_INTERACTION:
            alternatives.extend([
                "Moderated group discussions",
                "Collaborative learning projects",
                "Peer mentoring activities",
                "Supervised social skills practice"
            ])
        
        # Age-specific alternatives
        age_rules = self.age_appropriateness_rules.get(detection.age_group, {})
        encouraged_topics = age_rules.get("encouraged_topics", [])
        
        for topic in encouraged_topics[:3]:  # Limit to top 3
            alternatives.append(f"Content about {topic.replace('_', ' ')}")
        
        return alternatives
    
    async def _calculate_false_positive_risk(
        self,
        detection: HarmfulContentDetection,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate likelihood of false positive detection"""
        base_risk = 0.1  # Base 10% false positive risk
        
        # Factors that increase false positive risk
        if detection.content_type == ContentCategory.EDUCATIONAL:
            base_risk += 0.2  # Educational content often has false positives
        
        if len(detection.harmful_content_types) == 1:
            base_risk += 0.1  # Single detection type has higher false positive risk
        
        if max(detection.confidence_scores.values(), default=0.0) < 0.6:
            base_risk += 0.3  # Low confidence increases false positive risk
        
        # Factors that decrease false positive risk
        if len(detection.harmful_content_types) > 2:
            base_risk -= 0.2  # Multiple types reduce false positive risk
        
        if detection.severity_level in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]:
            base_risk -= 0.1  # High severity reduces false positive risk
        
        return max(0.0, min(1.0, base_risk))
    
    async def assess_privacy_protection(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PrivacyProtectionResult:
        """
        Assess privacy protection needs and compliance
        """
        logger.debug("Assessing privacy protection",
                    student_id=self.student_id,
                    content_length=len(content))
        
        # Initialize protection result
        protection = PrivacyProtectionResult(
            student_id=self.student_id,
            content_id=str(uuid4())
        )
        
        # Detect privacy risks
        protection.privacy_risks = await self._detect_privacy_risks(content)
        protection.personal_data_detected = await self._detect_personal_data(content)
        
        # Assess risk severity
        protection.risk_severity = await self._assess_privacy_risk_severity(
            protection.privacy_risks, protection.personal_data_detected
        )
        
        # Analyze data protection concerns
        protection.data_collection_concerns = await self._analyze_data_collection(content, context)
        protection.third_party_sharing_risks = await self._analyze_third_party_risks(content, context)
        protection.retention_policy_violations = await self._check_retention_policies(content, context)
        
        # Determine protection actions
        protection.protection_actions = await self._determine_protection_actions(protection)
        protection.anonymization_required = await self._requires_anonymization(protection)
        protection.parental_consent_required = await self._requires_parental_consent(protection)
        
        # Check compliance
        protection.coppa_compliance = await self._check_coppa_compliance(protection)
        protection.ferpa_compliance = await self._check_ferpa_compliance(protection)
        protection.gdpr_compliance = await self._check_gdpr_compliance(protection)
        protection.local_privacy_compliance = await self._check_local_compliance(protection)
        
        # Store assessment
        self.privacy_assessments.append(protection)
        
        logger.info("Privacy protection assessment completed",
                   student_id=self.student_id,
                   risk_severity=protection.risk_severity,
                   anonymization_required=protection.anonymization_required,
                   consent_required=protection.parental_consent_required)
        
        return protection
    
    async def _detect_privacy_risks(self, content: str) -> List[PrivacyRiskType]:
        """Detect privacy risks in content"""
        detected_risks = []
        content_lower = content.lower()
        
        for risk_type, patterns in self.privacy_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    detected_risks.append(risk_type)
                    break  # One pattern per type is enough
        
        return detected_risks
    
    async def _detect_personal_data(self, content: str) -> List[str]:
        """Detect specific personal data in content"""
        personal_data = []
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        personal_data.extend([f"email: {email}" for email in emails])
        
        # Phone number patterns
        phone_pattern = r'\b(?:\d{3}-\d{3}-\d{4}|\(\d{3}\)\s\d{3}-\d{4}|\d{10})\b'
        phones = re.findall(phone_pattern, content)
        personal_data.extend([f"phone: {phone}" for phone in phones])
        
        # Address patterns (simplified)
        address_pattern = r'\b\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b'
        addresses = re.findall(address_pattern, content, re.IGNORECASE)
        personal_data.extend([f"address: {' '.join(addr)}" for addr in addresses])
        
        return personal_data
    
    async def _assess_privacy_risk_severity(
        self,
        privacy_risks: List[PrivacyRiskType],
        personal_data: List[str]
    ) -> SafetyLevel:
        """Assess severity of privacy risks"""
        if not privacy_risks and not personal_data:
            return SafetyLevel.NONE
        
        # High-risk privacy violations
        high_risk_types = [
            PrivacyRiskType.LOCATION_DISCLOSURE,
            PrivacyRiskType.CONTACT_INFORMATION_LEAK,
            PrivacyRiskType.BIOMETRIC_DATA_EXPOSURE
        ]
        
        if any(risk in high_risk_types for risk in privacy_risks):
            return SafetyLevel.HIGH
        
        # Medium-risk violations
        if len(privacy_risks) > 2 or len(personal_data) > 1:
            return SafetyLevel.MEDIUM
        
        # Low-risk violations
        if privacy_risks or personal_data:
            return SafetyLevel.LOW
        
        return SafetyLevel.NONE
    
    async def _analyze_data_collection(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Analyze data collection concerns"""
        concerns = []
        
        # Check for data collection requests
        if "provide" in content.lower() and any(term in content.lower() for term in ["information", "details", "data"]):
            concerns.append("Requests for personal information")
        
        if "survey" in content.lower() or "questionnaire" in content.lower():
            concerns.append("Data collection through surveys")
        
        return concerns
    
    async def _analyze_third_party_risks(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Analyze third-party sharing risks"""
        risks = []
        
        # Check for third-party services
        third_party_indicators = ["share with", "send to", "external", "third party", "partner"]
        if any(indicator in content.lower() for indicator in third_party_indicators):
            risks.append("Potential third-party data sharing")
        
        return risks
    
    async def _check_retention_policies(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Check for data retention policy violations"""
        violations = []
        
        # Check for permanent storage indicators
        if "forever" in content.lower() or "permanently" in content.lower():
            violations.append("Indicates permanent data retention")
        
        return violations
    
    async def _determine_protection_actions(
        self,
        protection: PrivacyProtectionResult
    ) -> List[FilterAction]:
        """Determine protection actions needed"""
        actions = []
        
        if protection.risk_severity == SafetyLevel.HIGH:
            actions.extend([FilterAction.BLOCK, FilterAction.ANONYMIZE])
        elif protection.risk_severity == SafetyLevel.MEDIUM:
            actions.extend([FilterAction.MODERATE, FilterAction.REQUEST_CONSENT])
        elif protection.risk_severity == SafetyLevel.LOW:
            actions.append(FilterAction.FLAG_FOR_REVIEW)
        
        return actions
    
    async def _requires_anonymization(self, protection: PrivacyProtectionResult) -> bool:
        """Check if content requires anonymization"""
        return (
            protection.risk_severity in [SafetyLevel.HIGH, SafetyLevel.MEDIUM] or
            len(protection.personal_data_detected) > 0
        )
    
    async def _requires_parental_consent(self, protection: PrivacyProtectionResult) -> bool:
        """Check if parental consent is required"""
        return (
            self.age_group in [AgeGroup.EARLY_CHILDHOOD, AgeGroup.ELEMENTARY] and
            protection.risk_severity in [SafetyLevel.HIGH, SafetyLevel.MEDIUM]
        )
    
    async def _check_coppa_compliance(self, protection: PrivacyProtectionResult) -> bool:
        """Check COPPA compliance (Children's Online Privacy Protection Act)"""
        # COPPA applies to children under 13
        if self.age_group in [AgeGroup.EARLY_CHILDHOOD, AgeGroup.ELEMENTARY]:
            # Strict requirements for children under 13
            return (
                not protection.privacy_risks or
                protection.parental_consent_required
            )
        return True
    
    async def _check_ferpa_compliance(self, protection: PrivacyProtectionResult) -> bool:
        """Check FERPA compliance (Family Educational Rights and Privacy Act)"""
        # FERPA applies to educational records
        educational_risks = [
            PrivacyRiskType.SCHOOL_INFORMATION_LEAK,
            PrivacyRiskType.FAMILY_INFORMATION_EXPOSURE
        ]
        
        if any(risk in educational_risks for risk in protection.privacy_risks):
            return protection.anonymization_required
        return True
    
    async def _check_gdpr_compliance(self, protection: PrivacyProtectionResult) -> bool:
        """Check GDPR compliance (General Data Protection Regulation)"""
        # GDPR has strict requirements for personal data
        return (
            not protection.personal_data_detected or
            protection.anonymization_required
        )
    
    async def _check_local_compliance(self, protection: PrivacyProtectionResult) -> bool:
        """Check local privacy law compliance"""
        # Placeholder for local privacy law checks
        return True
    
    async def evaluate_age_appropriateness(
        self,
        content: str,
        target_age_group: AgeGroup,
        context: Optional[Dict[str, Any]] = None
    ) -> AgeAppropriatenessAssessment:
        """
        Evaluate age appropriateness of content with developmental considerations
        """
        logger.debug("Evaluating age appropriateness",
                    content_length=len(content),
                    target_age=target_age_group)
        
        # Initialize assessment
        assessment = AgeAppropriatenessAssessment(
            content_id=str(uuid4()),
            target_age_group=target_age_group
        )
        
        # Get age rules
        age_rules = self.age_appropriateness_rules.get(target_age_group, {})
        
        # Analyze content appropriateness
        assessment.recommended_age_groups = await self._determine_appropriate_ages(
            content, age_rules
        )
        
        assessment.inappropriate_elements = await self._identify_inappropriate_elements(
            content, age_rules
        )
        
        assessment.developmental_considerations = await self._analyze_developmental_fit(
            content, target_age_group, age_rules
        )
        
        # Generate content adaptations
        assessment.adaptation_suggestions = await self._suggest_content_adaptations(
            content, target_age_group, assessment.inappropriate_elements
        )
        
        assessment.complexity_adjustment = await self._suggest_complexity_adjustment(
            content, target_age_group, age_rules
        )
        
        assessment.supervision_recommendation = await self._recommend_supervision(
            content, target_age_group, assessment.inappropriate_elements
        )
        
        # Educational value assessment
        assessment.learning_objectives_alignment = await self._assess_learning_alignment(
            content, target_age_group
        )
        
        assessment.cognitive_load_assessment = await self._assess_cognitive_load(
            content, target_age_group, age_rules
        )
        
        assessment.engagement_prediction = await self._predict_engagement(
            content, target_age_group
        )
        
        # Store assessment
        self.age_assessments.append(assessment)
        
        logger.info("Age appropriateness evaluation completed",
                   target_age=target_age_group,
                   appropriate_ages=len(assessment.recommended_age_groups),
                   inappropriate_elements=len(assessment.inappropriate_elements),
                   supervision_needed=assessment.supervision_recommendation)
        
        return assessment
    
    async def _determine_appropriate_ages(
        self,
        content: str,
        age_rules: Dict[str, Any]
    ) -> List[AgeGroup]:
        """Determine which age groups content is appropriate for"""
        appropriate_ages = []
        
        # Analyze content complexity
        vocabulary_level = await self._analyze_vocabulary_level(content)
        concept_complexity = await self._analyze_concept_complexity(content)
        
        # Check against each age group
        for age_group in AgeGroup:
            group_rules = self.age_appropriateness_rules.get(age_group, {})
            
            # Check vocabulary appropriateness
            vocab_appropriate = await self._check_vocabulary_appropriateness(
                vocabulary_level, group_rules.get("vocabulary_level", "basic")
            )
            
            # Check concept appropriateness
            concept_appropriate = await self._check_concept_appropriateness(
                concept_complexity, group_rules.get("concept_complexity", "concrete")
            )
            
            # Check for restricted topics
            topics_appropriate = await self._check_topic_appropriateness(
                content, group_rules.get("restricted_topics", [])
            )
            
            if vocab_appropriate and concept_appropriate and topics_appropriate:
                appropriate_ages.append(age_group)
        
        return appropriate_ages
    
    async def _identify_inappropriate_elements(
        self,
        content: str,
        age_rules: Dict[str, Any]
    ) -> List[str]:
        """Identify specific inappropriate elements for age group"""
        inappropriate = []
        
        restricted_topics = age_rules.get("restricted_topics", [])
        content_lower = content.lower()
        
        for topic in restricted_topics:
            topic_words = topic.replace("_", " ").split()
            if all(word in content_lower for word in topic_words):
                inappropriate.append(f"Contains {topic.replace('_', ' ')}")
        
        # Check vocabulary complexity
        vocab_level = age_rules.get("vocabulary_level", "basic")
        if vocab_level == "basic" and await self._has_advanced_vocabulary(content):
            inappropriate.append("Advanced vocabulary")
        
        # Check concept complexity
        concept_level = age_rules.get("concept_complexity", "concrete")
        if concept_level == "concrete" and await self._has_abstract_concepts(content):
            inappropriate.append("Abstract concepts")
        
        return inappropriate
    
    async def _analyze_developmental_fit(
        self,
        content: str,
        target_age: AgeGroup,
        age_rules: Dict[str, Any]
    ) -> List[str]:
        """Analyze developmental appropriateness"""
        considerations = []
        
        attention_span = age_rules.get("attention_span_minutes", 30)
        estimated_time = len(content) / 200  # Rough estimate: 200 words per minute
        
        if estimated_time > attention_span:
            considerations.append(f"Content length exceeds typical attention span of {attention_span} minutes")
        
        # Social-emotional considerations
        if target_age == AgeGroup.EARLY_CHILDHOOD:
            if "fear" in content.lower() or "scary" in content.lower():
                considerations.append("May cause anxiety or fear in young children")
        
        if target_age in [AgeGroup.EARLY_CHILDHOOD, AgeGroup.ELEMENTARY]:
            if "competition" in content.lower() and "lose" in content.lower():
                considerations.append("Competitive elements may cause stress")
        
        return considerations
    
    async def _suggest_content_adaptations(
        self,
        content: str,
        target_age: AgeGroup,
        inappropriate_elements: List[str]
    ) -> List[str]:
        """Suggest adaptations to make content age-appropriate"""
        suggestions = []
        
        for element in inappropriate_elements:
            if "advanced vocabulary" in element.lower():
                suggestions.append("Replace complex words with simpler alternatives")
            elif "abstract concepts" in element.lower():
                suggestions.append("Use concrete examples and analogies")
            elif "violence" in element.lower():
                suggestions.append("Replace with peaceful conflict resolution")
            elif "complex emotions" in element.lower():
                suggestions.append("Simplify emotional content with basic feelings")
        
        # Age-specific suggestions
        if target_age == AgeGroup.EARLY_CHILDHOOD:
            suggestions.extend([
                "Add visual aids and illustrations",
                "Use repetition for key concepts",
                "Include interactive elements"
            ])
        elif target_age == AgeGroup.ELEMENTARY:
            suggestions.extend([
                "Add hands-on activities",
                "Include real-world examples",
                "Provide step-by-step instructions"
            ])
        
        return suggestions
    
    async def _suggest_complexity_adjustment(
        self,
        content: str,
        target_age: AgeGroup,
        age_rules: Dict[str, Any]
    ) -> Optional[str]:
        """Suggest complexity adjustments"""
        target_complexity = age_rules.get("concept_complexity", "concrete")
        current_complexity = await self._analyze_concept_complexity(content)
        
        if current_complexity == "complex_abstract" and target_complexity != "unrestricted":
            return "Reduce complexity to concrete examples"
        elif current_complexity == "moderate_abstract" and target_complexity == "concrete":
            return "Simplify to concrete concepts"
        elif current_complexity == "concrete" and target_complexity in ["complex_abstract", "moderate_abstract"]:
            return "Can increase complexity with abstract concepts"
        
        return None
    
    async def _recommend_supervision(
        self,
        content: str,
        target_age: AgeGroup,
        inappropriate_elements: List[str]
    ) -> bool:
        """Recommend if supervision is needed"""
        age_rules = self.age_appropriateness_rules.get(target_age, {})
        
        # Always supervise early childhood
        if target_age == AgeGroup.EARLY_CHILDHOOD:
            return True
        
        # Supervise if required by age rules
        if age_rules.get("supervision_required", False):
            return True
        
        # Supervise if inappropriate elements detected
        if inappropriate_elements:
            return True
        
        return False
    
    async def _assess_learning_alignment(
        self,
        content: str,
        target_age: AgeGroup
    ) -> float:
        """Assess alignment with learning objectives"""
        # Simplified assessment - would be more sophisticated in practice
        age_rules = self.age_appropriateness_rules.get(target_age, {})
        encouraged_topics = age_rules.get("encouraged_topics", [])
        
        content_lower = content.lower()
        alignment_score = 0.0
        
        for topic in encouraged_topics:
            topic_words = topic.replace("_", " ").split()
            if any(word in content_lower for word in topic_words):
                alignment_score += 0.2
        
        return min(1.0, alignment_score)
    
    async def _assess_cognitive_load(
        self,
        content: str,
        target_age: AgeGroup,
        age_rules: Dict[str, Any]
    ) -> float:
        """Assess cognitive load appropriateness"""
        # Factors affecting cognitive load
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Age-appropriate thresholds
        age_thresholds = {
            AgeGroup.EARLY_CHILDHOOD: {"max_words": 100, "max_sentence_length": 5},
            AgeGroup.ELEMENTARY: {"max_words": 300, "max_sentence_length": 10},
            AgeGroup.MIDDLE_SCHOOL: {"max_words": 500, "max_sentence_length": 15},
            AgeGroup.HIGH_SCHOOL: {"max_words": 1000, "max_sentence_length": 20},
            AgeGroup.ADULT: {"max_words": float('inf'), "max_sentence_length": float('inf')}
        }
        
        thresholds = age_thresholds.get(target_age, age_thresholds[AgeGroup.ELEMENTARY])
        
        # Calculate load score (0.0 = too high, 1.0 = appropriate)
        word_score = min(1.0, thresholds["max_words"] / max(word_count, 1))
        sentence_score = min(1.0, thresholds["max_sentence_length"] / max(avg_sentence_length, 1))
        
        return (word_score + sentence_score) / 2
    
    async def _predict_engagement(
        self,
        content: str,
        target_age: AgeGroup
    ) -> float:
        """Predict engagement level for target age group"""
        engagement_factors = []
        content_lower = content.lower()
        
        # Engagement indicators by age
        if target_age == AgeGroup.EARLY_CHILDHOOD:
            indicators = ["story", "animal", "color", "game", "fun", "play"]
        elif target_age == AgeGroup.ELEMENTARY:
            indicators = ["adventure", "mystery", "discovery", "experiment", "explore"]
        elif target_age == AgeGroup.MIDDLE_SCHOOL:
            indicators = ["challenge", "competition", "friendship", "identity", "choice"]
        elif target_age == AgeGroup.HIGH_SCHOOL:
            indicators = ["future", "career", "independence", "relationship", "justice"]
        else:
            indicators = ["goal", "achievement", "impact", "innovation", "leadership"]
        
        engagement_score = 0.0
        for indicator in indicators:
            if indicator in content_lower:
                engagement_score += 0.2
        
        return min(1.0, engagement_score)
    
    # Helper methods for vocabulary and concept analysis
    async def _analyze_vocabulary_level(self, content: str) -> str:
        """Analyze vocabulary complexity level"""
        # Simplified analysis - would use more sophisticated NLP in practice
        words = content.lower().split()
        
        advanced_words = ["sophisticated", "phenomenon", "methodology", "comprehensive", "synthesis"]
        intermediate_words = ["analyze", "compare", "evaluate", "explain", "describe"]
        
        advanced_count = sum(1 for word in words if word in advanced_words)
        intermediate_count = sum(1 for word in words if word in intermediate_words)
        
        if advanced_count > len(words) * 0.1:
            return "advanced"
        elif intermediate_count > len(words) * 0.1:
            return "intermediate"
        else:
            return "basic"
    
    async def _analyze_concept_complexity(self, content: str) -> str:
        """Analyze concept complexity level"""
        # Simplified analysis
        abstract_indicators = ["theory", "philosophy", "concept", "principle", "abstract"]
        moderate_indicators = ["relationship", "pattern", "system", "process", "method"]
        
        content_lower = content.lower()
        
        if any(indicator in content_lower for indicator in abstract_indicators):
            return "complex_abstract"
        elif any(indicator in content_lower for indicator in moderate_indicators):
            return "moderate_abstract"
        else:
            return "concrete"
    
    async def _check_vocabulary_appropriateness(
        self,
        content_vocab_level: str,
        target_vocab_level: str
    ) -> bool:
        """Check if vocabulary level is appropriate"""
        vocab_hierarchy = ["basic", "elementary", "intermediate", "advanced", "unrestricted"]
        
        try:
            content_index = vocab_hierarchy.index(content_vocab_level)
            target_index = vocab_hierarchy.index(target_vocab_level)
            return content_index <= target_index
        except ValueError:
            return True  # Default to appropriate if levels not found
    
    async def _check_concept_appropriateness(
        self,
        content_concept_level: str,
        target_concept_level: str
    ) -> bool:
        """Check if concept complexity is appropriate"""
        concept_hierarchy = ["concrete", "simple_abstract", "moderate_abstract", "complex_abstract", "unrestricted"]
        
        try:
            content_index = concept_hierarchy.index(content_concept_level)
            target_index = concept_hierarchy.index(target_concept_level)
            return content_index <= target_index
        except ValueError:
            return True  # Default to appropriate if levels not found
    
    async def _check_topic_appropriateness(
        self,
        content: str,
        restricted_topics: List[str]
    ) -> bool:
        """Check if content topics are appropriate"""
        content_lower = content.lower()
        
        for topic in restricted_topics:
            topic_words = topic.replace("_", " ").split()
            if all(word in content_lower for word in topic_words):
                return False
        
        return True
    
    async def _has_advanced_vocabulary(self, content: str) -> bool:
        """Check if content has advanced vocabulary"""
        return await self._analyze_vocabulary_level(content) == "advanced"
    
    async def _has_abstract_concepts(self, content: str) -> bool:
        """Check if content has abstract concepts"""
        complexity = await self._analyze_concept_complexity(content)
        return complexity in ["moderate_abstract", "complex_abstract"]
    
    async def monitor_safe_learning_environment(
        self,
        environment_id: str,
        time_window_hours: int = 24
    ) -> SafeLearningEnvironmentStatus:
        """
        Monitor and assess the overall safety of a learning environment
        """
        logger.debug("Monitoring safe learning environment",
                    student_id=self.student_id,
                    environment_id=environment_id,
                    time_window=time_window_hours)
        
        # Initialize status
        status = SafeLearningEnvironmentStatus(
            student_id=self.student_id,
            environment_id=environment_id
        )
        
        # Analyze recent activity within time window
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        
        recent_detections = [
            detection for detection in self.detection_history
            if detection.created_at >= cutoff_time
        ]
        
        recent_privacy = [
            assessment for assessment in self.privacy_assessments
            if assessment.created_at >= cutoff_time
        ]
        
        # Calculate safety scores
        status.content_safety_score = await self._calculate_content_safety_score(recent_detections)
        status.privacy_protection_score = await self._calculate_privacy_score(recent_privacy)
        status.interaction_safety_score = await self._calculate_interaction_safety_score(recent_detections)
        
        # Overall safety score
        status.overall_safety_score = (
            status.content_safety_score * 0.4 +
            status.privacy_protection_score * 0.3 +
            status.interaction_safety_score * 0.3
        )
        
        # Analyze recent violations
        status.recent_violations = await self._analyze_recent_violations(
            recent_detections, recent_privacy
        )
        
        # Calculate safety trends
        status.safety_trends = await self._calculate_safety_trends(
            recent_detections, recent_privacy
        )
        
        # Generate recommendations
        status.safety_recommendations = await self._generate_safety_recommendations(status)
        status.environment_adjustments = await self._suggest_environment_adjustments(status)
        
        # Set active filters and controls
        status.active_filters = await self._get_active_filters()
        status.parental_controls = await self._get_parental_controls()
        status.supervision_level = await self._determine_supervision_level(status)
        
        logger.info("Safe learning environment monitoring completed",
                   student_id=self.student_id,
                   overall_safety=status.overall_safety_score,
                   recent_violations=len(status.recent_violations),
                   supervision_level=status.supervision_level)
        
        return status
    
    async def _calculate_content_safety_score(
        self,
        recent_detections: List[HarmfulContentDetection]
    ) -> float:
        """Calculate content safety score based on recent detections"""
        if not recent_detections:
            return 1.0  # Perfect score if no detections
        
        # Weight detections by severity
        severity_weights = {
            SafetyLevel.NONE: 0.0,
            SafetyLevel.LOW: 0.1,
            SafetyLevel.MEDIUM: 0.3,
            SafetyLevel.HIGH: 0.6,
            SafetyLevel.CRITICAL: 1.0
        }
        
        total_weight = sum(
            severity_weights.get(detection.severity_level, 0.0)
            for detection in recent_detections
        )
        
        # Normalize by number of content interactions
        max_possible_weight = len(recent_detections) * severity_weights[SafetyLevel.CRITICAL]
        
        if max_possible_weight == 0:
            return 1.0
        
        safety_score = 1.0 - (total_weight / max_possible_weight)
        return max(0.0, safety_score)
    
    async def _calculate_privacy_score(
        self,
        recent_privacy: List[PrivacyProtectionResult]
    ) -> float:
        """Calculate privacy protection score"""
        if not recent_privacy:
            return 1.0  # Perfect score if no privacy assessments
        
        # Weight privacy risks by severity
        severity_weights = {
            SafetyLevel.NONE: 0.0,
            SafetyLevel.LOW: 0.1,
            SafetyLevel.MEDIUM: 0.3,
            SafetyLevel.HIGH: 0.6,
            SafetyLevel.CRITICAL: 1.0
        }
        
        total_weight = sum(
            severity_weights.get(assessment.risk_severity, 0.0)
            for assessment in recent_privacy
        )
        
        max_possible_weight = len(recent_privacy) * severity_weights[SafetyLevel.CRITICAL]
        
        if max_possible_weight == 0:
            return 1.0
        
        privacy_score = 1.0 - (total_weight / max_possible_weight)
        return max(0.0, privacy_score)
    
    async def _calculate_interaction_safety_score(
        self,
        recent_detections: List[HarmfulContentDetection]
    ) -> float:
        """Calculate interaction safety score"""
        # Focus on social interaction safety
        social_detections = [
            detection for detection in recent_detections
            if detection.content_type == ContentCategory.SOCIAL_INTERACTION
        ]
        
        if not social_detections:
            return 1.0  # Perfect score if no social interactions
        
        # Analyze social safety patterns
        bullying_count = sum(
            1 for detection in social_detections
            if HarmfulContentType.BULLYING in detection.harmful_content_types
        )
        
        predatory_count = sum(
            1 for detection in social_detections
            if HarmfulContentType.PREDATORY_BEHAVIOR in detection.harmful_content_types
        )
        
        # Higher penalties for social safety violations
        safety_deductions = (bullying_count * 0.3) + (predatory_count * 0.5)
        max_deductions = len(social_detections) * 0.5
        
        if max_deductions == 0:
            return 1.0
        
        interaction_score = 1.0 - min(1.0, safety_deductions / max_deductions)
        return max(0.0, interaction_score)
    
    async def _analyze_recent_violations(
        self,
        recent_detections: List[HarmfulContentDetection],
        recent_privacy: List[PrivacyProtectionResult]
    ) -> List[Dict[str, Any]]:
        """Analyze recent safety violations"""
        violations = []
        
        # Content safety violations
        for detection in recent_detections:
            if detection.severity_level in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]:
                violations.append({
                    "type": "content_safety",
                    "severity": detection.severity_level.value,
                    "harmful_types": [ht.value for ht in detection.harmful_content_types],
                    "timestamp": detection.created_at,
                    "action_taken": detection.recommended_action.value
                })
        
        # Privacy violations
        for assessment in recent_privacy:
            if assessment.risk_severity in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]:
                violations.append({
                    "type": "privacy_violation",
                    "severity": assessment.risk_severity.value,
                    "privacy_risks": [pr.value for pr in assessment.privacy_risks],
                    "timestamp": assessment.created_at,
                    "compliance_issues": {
                        "coppa": not assessment.coppa_compliance,
                        "ferpa": not assessment.ferpa_compliance,
                        "gdpr": not assessment.gdpr_compliance
                    }
                })
        
        return violations
    
    async def _calculate_safety_trends(
        self,
        recent_detections: List[HarmfulContentDetection],
        recent_privacy: List[PrivacyProtectionResult]
    ) -> Dict[str, float]:
        """Calculate safety trends over time"""
        trends = {}
        
        # Calculate detection rate trend
        if len(recent_detections) >= 2:
            early_detections = recent_detections[:len(recent_detections)//2]
            late_detections = recent_detections[len(recent_detections)//2:]
            
            early_rate = len(early_detections) / max(len(early_detections), 1)
            late_rate = len(late_detections) / max(len(late_detections), 1)
            
            trends["detection_rate_change"] = late_rate - early_rate
        else:
            trends["detection_rate_change"] = 0.0
        
        # Calculate severity trend
        if recent_detections:
            severity_values = {
                SafetyLevel.NONE: 0, SafetyLevel.LOW: 1, SafetyLevel.MEDIUM: 2,
                SafetyLevel.HIGH: 3, SafetyLevel.CRITICAL: 4
            }
            
            avg_severity = sum(
                severity_values.get(detection.severity_level, 0)
                for detection in recent_detections
            ) / len(recent_detections)
            
            trends["average_severity"] = avg_severity
        else:
            trends["average_severity"] = 0.0
        
        return trends
    
    async def _generate_safety_recommendations(
        self,
        status: SafeLearningEnvironmentStatus
    ) -> List[str]:
        """Generate safety recommendations based on environment status"""
        recommendations = []
        
        # Low overall safety score
        if status.overall_safety_score < 0.7:
            recommendations.append("Increase content filtering strictness")
            recommendations.append("Implement additional supervision measures")
        
        # Content safety issues
        if status.content_safety_score < 0.8:
            recommendations.append("Review and update harmful content detection rules")
            recommendations.append("Implement pre-screening for all content")
        
        # Privacy protection issues
        if status.privacy_protection_score < 0.8:
            recommendations.append("Enhance privacy protection measures")
            recommendations.append("Review data collection and sharing policies")
        
        # Social interaction safety issues
        if status.interaction_safety_score < 0.8:
            recommendations.append("Implement stricter social interaction monitoring")
            recommendations.append("Provide additional social skills training")
        
        # Age-specific recommendations
        if self.age_group in [AgeGroup.EARLY_CHILDHOOD, AgeGroup.ELEMENTARY]:
            recommendations.append("Ensure constant adult supervision")
            recommendations.append("Limit access to social features")
        
        return recommendations
    
    async def _suggest_environment_adjustments(
        self,
        status: SafeLearningEnvironmentStatus
    ) -> List[str]:
        """Suggest specific environment adjustments"""
        adjustments = []
        
        # Recent violations require immediate action
        if status.recent_violations:
            adjustments.append("Temporarily increase supervision level")
            adjustments.append("Review content access permissions")
        
        # Trending safety issues
        trends = status.safety_trends
        if trends.get("detection_rate_change", 0) > 0.1:
            adjustments.append("Implement proactive content screening")
        
        if trends.get("average_severity", 0) > 2.0:
            adjustments.append("Restrict access to high-risk content categories")
        
        return adjustments
    
    async def _get_active_filters(self) -> List[str]:
        """Get list of currently active content filters"""
        filters = []
        
        # Age-appropriate filters
        if self.age_group == AgeGroup.EARLY_CHILDHOOD:
            filters.extend([
                "violence_filter", "inappropriate_language_filter",
                "stranger_interaction_filter", "complex_content_filter"
            ])
        elif self.age_group == AgeGroup.ELEMENTARY:
            filters.extend([
                "violence_filter", "sexual_content_filter",
                "bullying_detection", "privacy_protection"
            ])
        
        # Universal filters
        filters.extend([
            "harmful_content_detection", "privacy_risk_detection",
            "age_appropriateness_check"
        ])
        
        return filters
    
    async def _get_parental_controls(self) -> Dict[str, Any]:
        """Get current parental control settings"""
        controls = {
            "content_filtering": True,
            "time_limits": True,
            "activity_monitoring": True,
            "social_interactions": "restricted" if self.age_group in [
                AgeGroup.EARLY_CHILDHOOD, AgeGroup.ELEMENTARY
            ] else "monitored"
        }
        
        # Age-specific controls
        if self.age_group == AgeGroup.EARLY_CHILDHOOD:
            controls.update({
                "constant_supervision": True,
                "external_links": "blocked",
                "user_generated_content": "blocked"
            })
        
        return controls
    
    async def _determine_supervision_level(
        self,
        status: SafeLearningEnvironmentStatus
    ) -> str:
        """Determine appropriate supervision level"""
        # Base supervision level on age
        base_levels = {
            AgeGroup.EARLY_CHILDHOOD: "supervised",
            AgeGroup.ELEMENTARY: "strict",
            AgeGroup.MIDDLE_SCHOOL: "standard",
            AgeGroup.HIGH_SCHOOL: "minimal",
            AgeGroup.ADULT: "minimal"
        }
        
        base_level = base_levels.get(self.age_group, "standard")
        
        # Increase supervision based on safety score
        if status.overall_safety_score < 0.5:
            return "supervised"
        elif status.overall_safety_score < 0.7 and base_level != "supervised":
            return "strict"
        
        return base_level


# Factory Functions
def create_red_team_student_safety_engine(
    student_id: str,
    age_group: AgeGroup
) -> RedTeamStudentSafetyEngine:
    """Create a Red Team student safety engine for content filtering"""
    return RedTeamStudentSafetyEngine(student_id, age_group)


def get_age_group_from_age(age: int) -> AgeGroup:
    """Determine age group from numerical age"""
    if age <= 6:
        return AgeGroup.EARLY_CHILDHOOD
    elif age <= 11:
        return AgeGroup.ELEMENTARY
    elif age <= 14:
        return AgeGroup.MIDDLE_SCHOOL
    elif age <= 18:
        return AgeGroup.HIGH_SCHOOL
    else:
        return AgeGroup.ADULT


def create_content_safety_assessment(
    content: str,
    student_age: int,
    content_category: ContentCategory = ContentCategory.EDUCATIONAL
) -> Dict[str, Any]:
    """Quick content safety assessment helper"""
    age_group = get_age_group_from_age(student_age)
    safety_engine = create_red_team_student_safety_engine("assessment_user", age_group)
    
    # This would be async in practice, simplified for helper function
    return {
        "age_group": age_group.value,
        "content_category": content_category.value,
        "assessment_ready": True,
        "safety_engine_id": safety_engine.student_id
    }