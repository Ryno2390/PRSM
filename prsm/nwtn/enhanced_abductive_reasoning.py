#!/usr/bin/env python3
"""
Enhanced Abductive Reasoning Engine for NWTN
===========================================

This module implements a comprehensive abductive reasoning system based on
elemental components derived from cognitive science and philosophy of science.

The system follows the five elemental components of abductive reasoning:
1. Observation of Phenomena
2. Hypothesis Generation
3. Selection of Best Explanation
4. Evaluation of Fit
5. Inference Application

Key Features:
- Comprehensive phenomenon identification and articulation
- Creative hypothesis generation with multiple strategies
- Sophisticated explanation selection with multiple criteria
- Rigorous hypothesis testing and validation
- Practical inference application for decision-making
"""

import asyncio
import numpy as np
import statistics
import time
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from uuid import uuid4
from datetime import datetime, timezone
from collections import defaultdict, Counter
import math
import logging
import re

import structlog

logger = structlog.get_logger(__name__)


class PhenomenonType(Enum):
    """Types of phenomena requiring explanation"""
    ANOMALOUS = "anomalous"                # Unexpected or surprising observations
    PUZZLING = "puzzling"                  # Confusing or unclear observations
    INCOMPLETE = "incomplete"              # Partial or missing information
    CONTRADICTORY = "contradictory"        # Conflicting observations
    NOVEL = "novel"                       # New or unprecedented observations
    SYSTEMATIC = "systematic"              # Recurring patterns needing explanation
    EXCEPTIONAL = "exceptional"            # Outliers or exceptions to rules
    TEMPORAL = "temporal"                 # Time-based phenomena
    CAUSAL = "causal"                     # Cause-effect relationships
    EMERGENT = "emergent"                 # Complex system behaviors


class HypothesisOrigin(Enum):
    """Origins of hypothesis generation"""
    ANALOGICAL = "analogical"              # Based on analogies to known cases
    CAUSAL = "causal"                     # Based on causal mechanisms
    THEORETICAL = "theoretical"           # Based on existing theories
    EMPIRICAL = "empirical"               # Based on empirical patterns
    CREATIVE = "creative"                 # Creative insight or intuition
    ELIMINATIVE = "eliminative"           # Process of elimination
    ABDUCTIVE = "abductive"               # Pure abductive inference
    COLLABORATIVE = "collaborative"       # Combination of multiple approaches


class ExplanationType(Enum):
    """Types of explanations in abductive reasoning"""
    CAUSAL = "causal"                     # Cause-and-effect explanation
    MECHANISTIC = "mechanistic"           # How something works
    FUNCTIONAL = "functional"             # What purpose something serves
    STRUCTURAL = "structural"             # How something is organized
    TELEOLOGICAL = "teleological"         # Why something exists/happens
    STATISTICAL = "statistical"           # Based on probability/statistics
    INTENTIONAL = "intentional"           # Based on goals/intentions
    EMERGENT = "emergent"                 # Complex system behavior
    REDUCTIVE = "reductive"               # Reducing to simpler components
    HOLISTIC = "holistic"                 # Considering whole systems
    THEORETICAL = "theoretical"           # Based on theoretical framework


class ExplanationCriteria(Enum):
    """Criteria for evaluating explanations"""
    SIMPLICITY = "simplicity"             # Occam's razor - simpler is better
    SCOPE = "scope"                       # Explains more phenomena
    PLAUSIBILITY = "plausibility"         # Consistent with known facts
    COHERENCE = "coherence"               # Internal logical consistency
    TESTABILITY = "testability"           # Can generate testable predictions
    NOVELTY = "novelty"                   # Provides new insights
    PRECISION = "precision"               # Specific and detailed
    GENERALITY = "generality"             # Applies broadly
    EXPLANATORY_POWER = "explanatory_power"  # Explains why, not just what
    CONSILIENCE = "consilience"           # Unifies disparate observations


class ConfidenceLevel(Enum):
    """Confidence levels for abductive conclusions"""
    VERY_HIGH = "very_high"    # >90%
    HIGH = "high"              # 70-90%
    MODERATE = "moderate"      # 50-70%
    LOW = "low"                # 30-50%
    VERY_LOW = "very_low"      # <30%


@dataclass
class Phenomenon:
    """A phenomenon requiring explanation"""
    
    # Core identification
    id: str
    description: str
    phenomenon_type: PhenomenonType
    
    # Phenomenon characteristics
    observations: List[str] = field(default_factory=list)
    anomalous_features: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    
    # Context and domain
    domain: str = "general"
    context: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    
    # Relevance and importance
    relevance_score: float = 0.0
    importance_score: float = 0.0
    urgency_score: float = 0.0
    
    # Relationships
    related_phenomena: List[str] = field(default_factory=list)
    prerequisite_knowledge: List[str] = field(default_factory=list)
    
    # Analysis metadata
    articulation_quality: float = 0.0
    completeness_score: float = 0.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_overall_score(self) -> float:
        """Calculate overall phenomenon score"""
        factors = [
            self.relevance_score,
            self.importance_score,
            self.articulation_quality,
            self.completeness_score
        ]
        return sum(factors) / len(factors)


@dataclass
class Hypothesis:
    """A hypothesis explaining phenomena"""
    
    # Core identification
    id: str
    statement: str
    explanation_type: ExplanationType
    origin: HypothesisOrigin
    
    # Hypothesis structure
    premises: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    mechanisms: List[str] = field(default_factory=list)
    
    # Scope and coverage
    explained_phenomena: List[str] = field(default_factory=list)
    scope_domains: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Evaluation scores
    simplicity_score: float = 0.0
    scope_score: float = 0.0
    plausibility_score: float = 0.0
    coherence_score: float = 0.0
    testability_score: float = 0.0
    explanatory_power_score: float = 0.0
    consilience_score: float = 0.0
    overall_score: float = 0.0
    
    # Evidence relationships
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    neutral_evidence: List[str] = field(default_factory=list)
    
    # Validation
    fit_score: float = 0.0
    validation_tests: List[str] = field(default_factory=list)
    validation_results: Dict[str, float] = field(default_factory=dict)
    
    # Practical application
    actionable_insights: List[str] = field(default_factory=list)
    practical_implications: List[str] = field(default_factory=list)
    decision_guidance: List[str] = field(default_factory=list)
    
    # Confidence and uncertainty
    confidence: float = 0.0
    uncertainty_sources: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_overall_score(self) -> float:
        """Calculate overall hypothesis score"""
        # Weighted combination of criteria
        weights = {
            'simplicity': 0.15,
            'scope': 0.20,
            'plausibility': 0.20,
            'coherence': 0.15,
            'testability': 0.10,
            'explanatory_power': 0.15,
            'consilience': 0.05
        }
        
        score = (
            weights['simplicity'] * self.simplicity_score +
            weights['scope'] * self.scope_score +
            weights['plausibility'] * self.plausibility_score +
            weights['coherence'] * self.coherence_score +
            weights['testability'] * self.testability_score +
            weights['explanatory_power'] * self.explanatory_power_score +
            weights['consilience'] * self.consilience_score
        )
        
        # Adjust for evidence support
        if self.supporting_evidence or self.contradicting_evidence:
            total_evidence = len(self.supporting_evidence) + len(self.contradicting_evidence)
            evidence_support = len(self.supporting_evidence) / total_evidence if total_evidence > 0 else 0.5
            score *= (0.5 + 0.5 * evidence_support)
        
        # Adjust for validation results
        if self.validation_results:
            avg_validation = sum(self.validation_results.values()) / len(self.validation_results)
            score *= avg_validation
        
        self.overall_score = score
        return score


@dataclass
class ExplanationEvaluation:
    """Evaluation of explanation fit and validation"""
    
    # Core identification
    id: str
    hypothesis_id: str
    
    # Fit assessment
    phenomenon_fit: float = 0.0
    evidence_consistency: float = 0.0
    prediction_accuracy: float = 0.0
    mechanistic_plausibility: float = 0.0
    
    # Validation tests
    validation_tests: List[str] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    
    # Comparative analysis
    alternative_comparisons: List[str] = field(default_factory=list)
    competitive_advantage: float = 0.0
    
    # Robustness
    robustness_score: float = 0.0
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    
    # Confidence assessment
    confidence_level: ConfidenceLevel = ConfidenceLevel.MODERATE
    confidence_factors: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_overall_fit_score(self) -> float:
        """Calculate overall fit score"""
        factors = [
            self.phenomenon_fit,
            self.evidence_consistency,
            self.prediction_accuracy,
            self.mechanistic_plausibility
        ]
        return sum(factors) / len(factors)


@dataclass
class InferenceApplication:
    """Application of abductive inference"""
    
    # Core identification
    id: str
    hypothesis_id: str
    application_type: str
    
    # Application context
    context: Dict[str, Any] = field(default_factory=dict)
    domain: str = "general"
    stakeholders: List[str] = field(default_factory=list)
    
    # Guidance and recommendations
    action_recommendations: List[str] = field(default_factory=list)
    decision_guidance: List[str] = field(default_factory=list)
    risk_assessments: List[str] = field(default_factory=list)
    
    # Practical implications
    immediate_actions: List[str] = field(default_factory=list)
    long_term_strategies: List[str] = field(default_factory=list)
    monitoring_requirements: List[str] = field(default_factory=list)
    
    # Predictions and forecasts
    predictions: List[str] = field(default_factory=list)
    forecasts: Dict[str, Any] = field(default_factory=dict)
    contingency_plans: List[str] = field(default_factory=list)
    
    # Success metrics
    success_indicators: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)
    
    # Confidence and reliability
    application_confidence: float = 0.0
    reliability_assessment: float = 0.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AbductiveReasoning:
    """Complete abductive reasoning process with all elemental components"""
    
    id: str
    query: str
    
    # Elemental components
    phenomena: List[Phenomenon]
    hypotheses: List[Hypothesis]
    best_explanation: Hypothesis
    evaluation: ExplanationEvaluation
    inference_application: InferenceApplication
    
    # Results
    conclusion: str
    confidence_level: ConfidenceLevel
    
    # Process metadata
    reasoning_quality: float = 0.0
    processing_time: float = 0.0
    certainty_score: float = 0.0
    
    # Alternative explanations
    alternative_explanations: List[Hypothesis] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PhenomenonObservationEngine:
    """Engine for comprehensive phenomenon observation and articulation"""
    
    def __init__(self):
        self.phenomenon_types = self._initialize_phenomenon_types()
        self.articulation_strategies = self._initialize_articulation_strategies()
        self.relevance_assessors = self._initialize_relevance_assessors()
    
    async def observe_phenomena(self, raw_observations: List[str], context: Dict[str, Any] = None) -> List[Phenomenon]:
        """Observe and articulate phenomena requiring explanation"""
        
        logger.info(f"Observing phenomena from {len(raw_observations)} observations")
        
        phenomena = []
        
        for i, observation in enumerate(raw_observations):
            # Identify phenomenon type
            phenomenon_type = await self._identify_phenomenon_type(observation, context)
            
            # Articulate the phenomenon
            phenomenon = await self._articulate_phenomenon(observation, phenomenon_type, i, context)
            
            # Assess relevance and importance
            phenomenon = await self._assess_phenomenon_relevance(phenomenon, context)
            
            # Identify relationships
            phenomenon = await self._identify_phenomenon_relationships(phenomenon, phenomena)
            
            # Validate phenomenon quality
            validated_phenomenon = await self._validate_phenomenon(phenomenon)
            
            phenomena.append(validated_phenomenon)
        
        # Filter and rank phenomena
        filtered_phenomena = await self._filter_and_rank_phenomena(phenomena)
        
        logger.info(f"Identified {len(filtered_phenomena)} significant phenomena")
        
        return filtered_phenomena
    
    async def _identify_phenomenon_type(self, observation: str, context: Dict[str, Any] = None) -> PhenomenonType:
        """Identify the type of phenomenon"""
        
        obs_lower = observation.lower()
        
        # Anomalous indicators
        if any(indicator in obs_lower for indicator in ["unexpected", "surprising", "anomalous", "unusual", "strange"]):
            return PhenomenonType.ANOMALOUS
        
        # Puzzling indicators
        if any(indicator in obs_lower for indicator in ["puzzling", "confusing", "unclear", "mysterious", "baffling"]):
            return PhenomenonType.PUZZLING
        
        # Incomplete indicators
        if any(indicator in obs_lower for indicator in ["partial", "incomplete", "missing", "unknown", "uncertain"]):
            return PhenomenonType.INCOMPLETE
        
        # Contradictory indicators
        if any(indicator in obs_lower for indicator in ["contradictory", "conflicting", "inconsistent", "opposed"]):
            return PhenomenonType.CONTRADICTORY
        
        # Novel indicators
        if any(indicator in obs_lower for indicator in ["new", "novel", "unprecedented", "first time", "never seen"]):
            return PhenomenonType.NOVEL
        
        # Systematic indicators
        if any(indicator in obs_lower for indicator in ["pattern", "systematic", "recurring", "consistent", "regular"]):
            return PhenomenonType.SYSTEMATIC
        
        # Exceptional indicators
        if any(indicator in obs_lower for indicator in ["exception", "outlier", "rare", "unusual case", "special"]):
            return PhenomenonType.EXCEPTIONAL
        
        # Temporal indicators
        if any(indicator in obs_lower for indicator in ["temporal", "time", "sequence", "chronological", "when"]):
            return PhenomenonType.TEMPORAL
        
        # Causal indicators
        if any(indicator in obs_lower for indicator in ["cause", "effect", "because", "leads to", "results in"]):
            return PhenomenonType.CAUSAL
        
        return PhenomenonType.SYSTEMATIC  # Default
    
    async def _articulate_phenomenon(self, observation: str, phenomenon_type: PhenomenonType, index: int, context: Dict[str, Any] = None) -> Phenomenon:
        """Articulate the phenomenon comprehensively"""
        
        # Extract key observations
        observations = await self._extract_observations(observation)
        
        # Identify anomalous features
        anomalous_features = await self._identify_anomalous_features(observation, phenomenon_type)
        
        # Identify missing information
        missing_information = await self._identify_missing_information(observation)
        
        # Identify contradictions
        contradictions = await self._identify_contradictions(observation)
        
        # Determine domain
        domain = await self._determine_domain(observation, context)
        
        # Create phenomenon
        phenomenon = Phenomenon(
            id=f"phenomenon_{index+1}",
            description=observation,
            phenomenon_type=phenomenon_type,
            observations=observations,
            anomalous_features=anomalous_features,
            missing_information=missing_information,
            contradictions=contradictions,
            domain=domain,
            context=context or {}
        )
        
        return phenomenon
    
    async def _extract_observations(self, observation: str) -> List[str]:
        """Extract individual observations from text"""
        
        # Split by common separators
        observation_parts = []
        
        # Split by sentences
        sentences = re.split(r'[.!?]+', observation)
        for sentence in sentences:
            if sentence.strip():
                observation_parts.append(sentence.strip())
        
        # Split by commas and semicolons
        for part in observation_parts.copy():
            if ',' in part or ';' in part:
                sub_parts = re.split(r'[,;]', part)
                observation_parts.extend([p.strip() for p in sub_parts if p.strip()])
        
        return observation_parts[:5]  # Limit to top 5
    
    async def _identify_anomalous_features(self, observation: str, phenomenon_type: PhenomenonType) -> List[str]:
        """Identify anomalous features in the observation"""
        
        anomalous_features = []
        obs_lower = observation.lower()
        
        # Type-specific anomalous features
        if phenomenon_type == PhenomenonType.ANOMALOUS:
            # Look for deviation indicators
            deviation_patterns = [
                r"different from (.+)",
                r"unlike (.+)",
                r"contrary to (.+)",
                r"opposite of (.+)",
                r"not expected (.+)"
            ]
            
            for pattern in deviation_patterns:
                matches = re.findall(pattern, obs_lower)
                anomalous_features.extend(matches)
        
        elif phenomenon_type == PhenomenonType.EXCEPTIONAL:
            # Look for exceptional characteristics
            exceptional_patterns = [
                r"only case (.+)",
                r"exception to (.+)",
                r"rare occurrence (.+)",
                r"unusual for (.+)"
            ]
            
            for pattern in exceptional_patterns:
                matches = re.findall(pattern, obs_lower)
                anomalous_features.extend(matches)
        
        # General anomaly indicators
        general_anomalies = [
            "unexpectedly high",
            "unusually low", 
            "surprising result",
            "contrary to expectations",
            "abnormal behavior"
        ]
        
        for anomaly in general_anomalies:
            if anomaly in obs_lower:
                anomalous_features.append(anomaly)
        
        return anomalous_features
    
    async def _identify_missing_information(self, observation: str) -> List[str]:
        """Identify missing information in the observation"""
        
        missing_info = []
        obs_lower = observation.lower()
        
        # Explicit missing information indicators
        missing_indicators = [
            "unknown",
            "unclear",
            "missing",
            "not specified",
            "to be determined",
            "undetermined",
            "incomplete",
            "partial"
        ]
        
        for indicator in missing_indicators:
            if indicator in obs_lower:
                # Try to identify what's missing
                context_words = obs_lower.split()
                try:
                    index = context_words.index(indicator)
                    # Get surrounding context
                    start = max(0, index - 2)
                    end = min(len(context_words), index + 3)
                    context = " ".join(context_words[start:end])
                    missing_info.append(f"Missing: {context}")
                except ValueError:
                    missing_info.append(f"Missing: {indicator}")
        
        # Question patterns indicating missing info
        question_patterns = [
            r"what (.+)\?",
            r"how (.+)\?",
            r"why (.+)\?",
            r"when (.+)\?",
            r"where (.+)\?"
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, obs_lower)
            for match in matches:
                missing_info.append(f"Unknown: {match}")
        
        return missing_info
    
    async def _identify_contradictions(self, observation: str) -> List[str]:
        """Identify contradictions in the observation"""
        
        contradictions = []
        obs_lower = observation.lower()
        
        # Explicit contradiction indicators
        contradiction_indicators = [
            "contradicts",
            "conflicts with",
            "inconsistent with",
            "opposed to",
            "contrary to",
            "paradox",
            "contradictory"
        ]
        
        for indicator in contradiction_indicators:
            if indicator in obs_lower:
                # Extract contradiction context
                parts = obs_lower.split(indicator)
                if len(parts) >= 2:
                    before = parts[0].strip()[-50:]  # Last 50 chars before
                    after = parts[1].strip()[:50]   # First 50 chars after
                    contradictions.append(f"{before} {indicator} {after}")
        
        # But/however patterns
        contrast_patterns = [
            r"(.+) but (.+)",
            r"(.+) however (.+)",
            r"(.+) although (.+)",
            r"(.+) while (.+)"
        ]
        
        for pattern in contrast_patterns:
            matches = re.findall(pattern, obs_lower)
            for match in matches:
                contradictions.append(f"Contrast: {match[0]} vs {match[1]}")
        
        return contradictions
    
    async def _determine_domain(self, observation: str, context: Dict[str, Any] = None) -> str:
        """Determine the domain of the phenomenon"""
        
        # Check context first
        if context and "domain" in context:
            return context["domain"]
        
        # Domain classification based on keywords
        domain_keywords = {
            "medical": ["patient", "symptom", "diagnosis", "treatment", "disease", "health", "medicine", "clinical"],
            "scientific": ["experiment", "hypothesis", "data", "research", "study", "analysis", "theory", "scientific"],
            "technical": ["system", "component", "failure", "error", "malfunction", "technical", "engineering", "software"],
            "criminal": ["crime", "suspect", "evidence", "witness", "investigation", "police", "legal", "forensic"],
            "business": ["market", "financial", "economic", "business", "trade", "money", "cost", "profit"],
            "social": ["behavior", "society", "culture", "group", "social", "interaction", "community", "people"],
            "environmental": ["climate", "weather", "environmental", "ecology", "nature", "pollution", "ecosystem"],
            "psychological": ["behavior", "mental", "cognitive", "emotional", "psychological", "mind", "consciousness"],
            "educational": ["student", "learning", "education", "academic", "school", "teaching", "knowledge"],
            "biological": ["organism", "biology", "life", "cell", "genetic", "evolution", "species", "biological"]
        }
        
        obs_lower = observation.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in obs_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            if domain_scores[best_domain] > 0:
                return best_domain
        
        return "general"
    
    async def _assess_phenomenon_relevance(self, phenomenon: Phenomenon, context: Dict[str, Any] = None) -> Phenomenon:
        """Assess relevance and importance of phenomenon"""
        
        # Relevance assessment
        relevance_score = 0.7  # Base relevance
        
        # Adjust based on phenomenon type
        type_weights = {
            PhenomenonType.ANOMALOUS: 0.9,
            PhenomenonType.PUZZLING: 0.8,
            PhenomenonType.CONTRADICTORY: 0.9,
            PhenomenonType.NOVEL: 0.8,
            PhenomenonType.EXCEPTIONAL: 0.7,
            PhenomenonType.SYSTEMATIC: 0.6,
            PhenomenonType.INCOMPLETE: 0.5,
            PhenomenonType.TEMPORAL: 0.6,
            PhenomenonType.CAUSAL: 0.8,
            PhenomenonType.EMERGENT: 0.7
        }
        
        relevance_score = type_weights.get(phenomenon.phenomenon_type, 0.7)
        
        # Adjust for anomalous features
        if phenomenon.anomalous_features:
            relevance_score += min(0.2, len(phenomenon.anomalous_features) * 0.05)
        
        # Adjust for missing information (reduces relevance)
        if phenomenon.missing_information:
            relevance_score -= min(0.3, len(phenomenon.missing_information) * 0.05)
        
        phenomenon.relevance_score = max(0.0, min(1.0, relevance_score))
        
        # Importance assessment
        importance_score = 0.6  # Base importance
        
        # Domain-specific importance
        domain_importance = {
            "medical": 0.9,
            "criminal": 0.8,
            "technical": 0.7,
            "scientific": 0.8,
            "business": 0.6,
            "social": 0.5,
            "environmental": 0.7
        }
        
        importance_score = domain_importance.get(phenomenon.domain, 0.6)
        
        # Urgency assessment
        urgency_indicators = ["urgent", "immediate", "critical", "emergency", "time-sensitive"]
        urgency_score = 0.5  # Base urgency
        
        for indicator in urgency_indicators:
            if indicator in phenomenon.description.lower():
                urgency_score += 0.2
        
        phenomenon.importance_score = max(0.0, min(1.0, importance_score))
        phenomenon.urgency_score = max(0.0, min(1.0, urgency_score))
        
        return phenomenon
    
    async def _identify_phenomenon_relationships(self, phenomenon: Phenomenon, existing_phenomena: List[Phenomenon]) -> Phenomenon:
        """Identify relationships between phenomena"""
        
        related_phenomena = []
        
        for existing_phenomenon in existing_phenomena:
            # Check for relationships
            relationship_score = await self._calculate_phenomenon_similarity(phenomenon, existing_phenomenon)
            
            if relationship_score > 0.6:
                related_phenomena.append(existing_phenomenon.id)
        
        phenomenon.related_phenomena = related_phenomena
        
        return phenomenon
    
    async def _calculate_phenomenon_similarity(self, phenomenon1: Phenomenon, phenomenon2: Phenomenon) -> float:
        """Calculate similarity between phenomena"""
        
        # Domain similarity
        domain_similarity = 1.0 if phenomenon1.domain == phenomenon2.domain else 0.0
        
        # Type similarity
        type_similarity = 1.0 if phenomenon1.phenomenon_type == phenomenon2.phenomenon_type else 0.0
        
        # Content similarity (simple word overlap)
        words1 = set(phenomenon1.description.lower().split())
        words2 = set(phenomenon2.description.lower().split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        words1 -= stop_words
        words2 -= stop_words
        
        if words1 and words2:
            content_similarity = len(words1 & words2) / len(words1 | words2)
        else:
            content_similarity = 0.0
        
        # Weighted combination
        similarity = (
            0.3 * domain_similarity +
            0.2 * type_similarity +
            0.5 * content_similarity
        )
        
        return similarity
    
    async def _validate_phenomenon(self, phenomenon: Phenomenon) -> Phenomenon:
        """Validate phenomenon quality"""
        
        # Articulation quality
        articulation_factors = [
            len(phenomenon.description) > 20,  # Sufficient description
            len(phenomenon.observations) > 0,   # Has observations
            phenomenon.domain != "general",    # Specific domain
            len(phenomenon.anomalous_features) > 0 or phenomenon.phenomenon_type != PhenomenonType.SYSTEMATIC  # Has notable features
        ]
        
        phenomenon.articulation_quality = sum(articulation_factors) / len(articulation_factors)
        
        # Completeness score
        completeness_factors = [
            len(phenomenon.observations) > 0,
            len(phenomenon.missing_information) == 0,
            len(phenomenon.contradictions) == 0,
            phenomenon.relevance_score > 0.5
        ]
        
        phenomenon.completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        return phenomenon
    
    async def _filter_and_rank_phenomena(self, phenomena: List[Phenomenon]) -> List[Phenomenon]:
        """Filter and rank phenomena by importance"""
        
        # Filter out low-quality phenomena
        filtered_phenomena = [
            p for p in phenomena
            if p.get_overall_score() > 0.3
        ]
        
        # Sort by overall score
        filtered_phenomena.sort(key=lambda p: p.get_overall_score(), reverse=True)
        
        return filtered_phenomena
    
    def _initialize_phenomenon_types(self) -> Dict[str, Any]:
        """Initialize phenomenon type configurations"""
        return {
            "anomalous": {"weight": 0.9, "indicators": ["unexpected", "surprising", "unusual"]},
            "puzzling": {"weight": 0.8, "indicators": ["confusing", "unclear", "mysterious"]},
            "contradictory": {"weight": 0.9, "indicators": ["contradictory", "conflicting", "inconsistent"]},
            "novel": {"weight": 0.8, "indicators": ["new", "novel", "unprecedented"]},
            "exceptional": {"weight": 0.7, "indicators": ["exception", "outlier", "rare"]},
            "systematic": {"weight": 0.6, "indicators": ["pattern", "systematic", "recurring"]},
            "incomplete": {"weight": 0.5, "indicators": ["partial", "incomplete", "missing"]},
            "temporal": {"weight": 0.6, "indicators": ["temporal", "time", "sequence"]},
            "causal": {"weight": 0.8, "indicators": ["cause", "effect", "because"]},
            "emergent": {"weight": 0.7, "indicators": ["emergent", "complex", "system"]}
        }
    
    def _initialize_articulation_strategies(self) -> Dict[str, Any]:
        """Initialize articulation strategies"""
        return {
            "detailed_description": {"focus": "comprehensive description"},
            "anomaly_identification": {"focus": "anomalous features"},
            "gap_identification": {"focus": "missing information"},
            "contradiction_analysis": {"focus": "contradictions"},
            "relationship_mapping": {"focus": "phenomenon relationships"}
        }
    
    def _initialize_relevance_assessors(self) -> Dict[str, Any]:
        """Initialize relevance assessment methods"""
        return {
            "type_based": {"weight": 0.3},
            "domain_based": {"weight": 0.2},
            "anomaly_based": {"weight": 0.3},
            "context_based": {"weight": 0.2}
        }


class HypothesisGenerationEngine:
    """Engine for comprehensive hypothesis generation"""
    
    def __init__(self):
        self.generation_strategies = self._initialize_generation_strategies()
        self.hypothesis_origins = self._initialize_hypothesis_origins()
        self.creativity_enhancers = self._initialize_creativity_enhancers()
    
    async def generate_hypotheses(self, phenomena: List[Phenomenon], context: Dict[str, Any] = None) -> List[Hypothesis]:
        """Generate comprehensive hypotheses to explain phenomena"""
        
        logger.info(f"Generating hypotheses for {len(phenomena)} phenomena")
        
        all_hypotheses = []
        
        # Multiple generation strategies
        for strategy in self.generation_strategies:
            strategy_hypotheses = await self._apply_generation_strategy(strategy, phenomena, context)
            all_hypotheses.extend(strategy_hypotheses)
        
        # Remove duplicates and enhance
        unique_hypotheses = await self._remove_duplicate_hypotheses(all_hypotheses)
        enhanced_hypotheses = await self._enhance_hypotheses(unique_hypotheses, phenomena)
        
        # Filter and rank
        filtered_hypotheses = await self._filter_and_rank_hypotheses(enhanced_hypotheses)
        
        logger.info(f"Generated {len(filtered_hypotheses)} unique hypotheses")
        
        return filtered_hypotheses
    
    async def _apply_generation_strategy(self, strategy: str, phenomena: List[Phenomenon], context: Dict[str, Any] = None) -> List[Hypothesis]:
        """Apply a specific generation strategy"""
        
        if strategy == "analogical":
            return await self._generate_analogical_hypotheses(phenomena, context)
        elif strategy == "causal":
            return await self._generate_causal_hypotheses(phenomena, context)
        elif strategy == "theoretical":
            return await self._generate_theoretical_hypotheses(phenomena, context)
        elif strategy == "empirical":
            return await self._generate_empirical_hypotheses(phenomena, context)
        elif strategy == "creative":
            return await self._generate_creative_hypotheses(phenomena, context)
        elif strategy == "eliminative":
            return await self._generate_eliminative_hypotheses(phenomena, context)
        else:
            return []
    
    async def _generate_analogical_hypotheses(self, phenomena: List[Phenomenon], context: Dict[str, Any] = None) -> List[Hypothesis]:
        """Generate hypotheses based on analogies"""
        
        hypotheses = []
        
        # Domain-specific analogies
        domain_analogies = {
            "medical": [
                "Similar to known disease pattern",
                "Analogous to immune system response",
                "Like metabolic dysfunction",
                "Similar to infection progression"
            ],
            "technical": [
                "Similar to known system failure",
                "Analogous to component degradation",
                "Like configuration error",
                "Similar to resource exhaustion"
            ],
            "scientific": [
                "Similar to known phenomenon",
                "Analogous to theoretical model",
                "Like experimental artifact",
                "Similar to measurement error"
            ],
            "criminal": [
                "Similar to known crime pattern",
                "Analogous to past investigations",
                "Like staged crime scene",
                "Similar to inside job"
            ]
        }
        
        for phenomenon in phenomena:
            analogies = domain_analogies.get(phenomenon.domain, ["Similar to known pattern"])
            
            for i, analogy in enumerate(analogies):
                hypothesis = Hypothesis(
                    id=f"analogical_{phenomenon.id}_{i}",
                    statement=f"{analogy} explains {phenomenon.description}",
                    explanation_type=ExplanationType.CAUSAL,
                    origin=HypothesisOrigin.ANALOGICAL,
                    explained_phenomena=[phenomenon.id],
                    premises=[f"Analogy: {analogy}"],
                    assumptions=[f"Similar mechanisms apply to {phenomenon.domain}"],
                    predictions=[f"Similar outcomes should follow as in analogous cases"]
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_causal_hypotheses(self, phenomena: List[Phenomenon], context: Dict[str, Any] = None) -> List[Hypothesis]:
        """Generate causal hypotheses"""
        
        hypotheses = []
        
        # Causal patterns
        causal_patterns = [
            "Direct causal mechanism",
            "Indirect causal chain",
            "Common cause explanation",
            "Feedback loop mechanism",
            "Threshold effect",
            "Cumulative effect",
            "Interaction effect",
            "Emergent property"
        ]
        
        for phenomenon in phenomena:
            # Extract potential causes and effects
            causes = await self._extract_potential_causes(phenomenon)
            effects = await self._extract_potential_effects(phenomenon)
            
            for pattern in causal_patterns:
                for cause in causes:
                    hypothesis = Hypothesis(
                        id=f"causal_{phenomenon.id}_{pattern.replace(' ', '_')}",
                        statement=f"{pattern}: {cause} explains {phenomenon.description}",
                        explanation_type=ExplanationType.CAUSAL,
                        origin=HypothesisOrigin.CAUSAL,
                        explained_phenomena=[phenomenon.id],
                        premises=[f"Causal pattern: {pattern}"],
                        assumptions=[f"Causal relationship exists between {cause} and observed phenomenon"],
                        predictions=[f"Manipulating {cause} should affect the phenomenon"],
                        mechanisms=[f"Causal mechanism through {pattern}"]
                    )
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_theoretical_hypotheses(self, phenomena: List[Phenomenon], context: Dict[str, Any] = None) -> List[Hypothesis]:
        """Generate hypotheses based on existing theories"""
        
        hypotheses = []
        
        # Domain-specific theories
        domain_theories = {
            "medical": [
                "Pathophysiological theory",
                "Pharmacological theory",
                "Immunological theory",
                "Genetic theory",
                "Environmental theory"
            ],
            "technical": [
                "Systems theory",
                "Failure theory",
                "Information theory",
                "Control theory",
                "Complexity theory"
            ],
            "scientific": [
                "Physical theory",
                "Chemical theory",
                "Biological theory",
                "Mathematical theory",
                "Computational theory"
            ],
            "social": [
                "Behavioral theory",
                "Social theory",
                "Psychological theory",
                "Economic theory",
                "Cultural theory"
            ]
        }
        
        for phenomenon in phenomena:
            theories = domain_theories.get(phenomenon.domain, ["General theory"])
            
            for theory in theories:
                hypothesis = Hypothesis(
                    id=f"theoretical_{phenomenon.id}_{theory.replace(' ', '_')}",
                    statement=f"{theory} explains {phenomenon.description}",
                    explanation_type=ExplanationType.MECHANISTIC,
                    origin=HypothesisOrigin.THEORETICAL,
                    explained_phenomena=[phenomenon.id],
                    premises=[f"Based on {theory}"],
                    assumptions=[f"{theory} applies to this phenomenon"],
                    predictions=[f"Theoretical predictions from {theory} should hold"],
                    mechanisms=[f"Theoretical mechanism from {theory}"]
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_empirical_hypotheses(self, phenomena: List[Phenomenon], context: Dict[str, Any] = None) -> List[Hypothesis]:
        """Generate hypotheses based on empirical patterns"""
        
        hypotheses = []
        
        # Empirical patterns
        empirical_patterns = [
            "Statistical correlation",
            "Temporal pattern",
            "Spatial pattern",
            "Frequency pattern",
            "Magnitude pattern",
            "Trend pattern",
            "Cyclical pattern",
            "Threshold pattern"
        ]
        
        for phenomenon in phenomena:
            for pattern in empirical_patterns:
                hypothesis = Hypothesis(
                    id=f"empirical_{phenomenon.id}_{pattern.replace(' ', '_')}",
                    statement=f"{pattern} explains {phenomenon.description}",
                    explanation_type=ExplanationType.STATISTICAL,
                    origin=HypothesisOrigin.EMPIRICAL,
                    explained_phenomena=[phenomenon.id],
                    premises=[f"Empirical pattern: {pattern}"],
                    assumptions=[f"Empirical pattern {pattern} is reliable"],
                    predictions=[f"Pattern {pattern} should persist or evolve predictably"],
                    mechanisms=[f"Empirical mechanism through {pattern}"]
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_creative_hypotheses(self, phenomena: List[Phenomenon], context: Dict[str, Any] = None) -> List[Hypothesis]:
        """Generate creative and novel hypotheses"""
        
        hypotheses = []
        
        # Creative approaches
        creative_approaches = [
            "Novel mechanism",
            "Unexpected interaction",
            "Hidden variable",
            "Emergent property",
            "Paradigm shift",
            "Interdisciplinary connection",
            "Counterintuitive explanation",
            "Radical reinterpretation"
        ]
        
        for phenomenon in phenomena:
            for approach in creative_approaches:
                hypothesis = Hypothesis(
                    id=f"creative_{phenomenon.id}_{approach.replace(' ', '_')}",
                    statement=f"{approach} explains {phenomenon.description}",
                    explanation_type=ExplanationType.EMERGENT,
                    origin=HypothesisOrigin.CREATIVE,
                    explained_phenomena=[phenomenon.id],
                    premises=[f"Creative approach: {approach}"],
                    assumptions=[f"Novel explanation through {approach} is possible"],
                    predictions=[f"Creative predictions from {approach} should be testable"],
                    mechanisms=[f"Creative mechanism through {approach}"]
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_eliminative_hypotheses(self, phenomena: List[Phenomenon], context: Dict[str, Any] = None) -> List[Hypothesis]:
        """Generate hypotheses through elimination"""
        
        hypotheses = []
        
        # Common alternative explanations to eliminate
        common_alternatives = [
            "Measurement error",
            "Sampling bias",
            "Confounding variable",
            "Coincidence",
            "Systematic error",
            "Observer bias",
            "Instrument malfunction",
            "Environmental factor"
        ]
        
        for phenomenon in phenomena:
            for alternative in common_alternatives:
                hypothesis = Hypothesis(
                    id=f"eliminative_{phenomenon.id}_{alternative.replace(' ', '_')}",
                    statement=f"After eliminating {alternative}, remaining explanation for {phenomenon.description}",
                    explanation_type=ExplanationType.REDUCTIVE,
                    origin=HypothesisOrigin.ELIMINATIVE,
                    explained_phenomena=[phenomenon.id],
                    premises=[f"Elimination of {alternative}"],
                    assumptions=[f"{alternative} can be ruled out"],
                    predictions=[f"Eliminating {alternative} should clarify the phenomenon"],
                    mechanisms=[f"Eliminative process excluding {alternative}"]
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _extract_potential_causes(self, phenomenon: Phenomenon) -> List[str]:
        """Extract potential causes from phenomenon"""
        
        causes = []
        
        # From description
        causal_patterns = [
            r"because of (.+)",
            r"due to (.+)",
            r"caused by (.+)",
            r"resulting from (.+)",
            r"triggered by (.+)",
            r"following (.+)",
            r"after (.+)",
            r"from (.+)"
        ]
        
        for pattern in causal_patterns:
            matches = re.findall(pattern, phenomenon.description, re.IGNORECASE)
            causes.extend(matches)
        
        # From observations
        for observation in phenomenon.observations:
            for pattern in causal_patterns:
                matches = re.findall(pattern, observation, re.IGNORECASE)
                causes.extend(matches)
        
        # Domain-specific common causes
        domain_causes = {
            "medical": ["infection", "inflammation", "genetic factor", "environmental factor", "medication"],
            "technical": ["hardware failure", "software bug", "configuration error", "resource exhaustion", "external interference"],
            "scientific": ["experimental error", "measurement noise", "environmental factor", "systematic bias", "theoretical limitation"],
            "criminal": ["motive", "opportunity", "means", "planning", "circumstance"]
        }
        
        if phenomenon.domain in domain_causes:
            causes.extend(domain_causes[phenomenon.domain])
        
        return causes[:10]  # Limit to top 10
    
    async def _extract_potential_effects(self, phenomenon: Phenomenon) -> List[str]:
        """Extract potential effects from phenomenon"""
        
        effects = []
        
        # From description
        effect_patterns = [
            r"leads to (.+)",
            r"causes (.+)",
            r"results in (.+)",
            r"produces (.+)",
            r"triggers (.+)",
            r"followed by (.+)",
            r"then (.+)",
            r"consequently (.+)"
        ]
        
        for pattern in effect_patterns:
            matches = re.findall(pattern, phenomenon.description, re.IGNORECASE)
            effects.extend(matches)
        
        # From observations
        for observation in phenomenon.observations:
            for pattern in effect_patterns:
                matches = re.findall(pattern, observation, re.IGNORECASE)
                effects.extend(matches)
        
        # Domain-specific common effects
        domain_effects = {
            "medical": ["symptom", "complication", "recovery", "deterioration", "side effect"],
            "technical": ["system failure", "performance degradation", "error message", "data corruption", "service disruption"],
            "scientific": ["measurement change", "theoretical implication", "experimental result", "prediction", "discovery"],
            "criminal": ["evidence", "witness testimony", "physical trace", "behavioral change", "investigation lead"]
        }
        
        if phenomenon.domain in domain_effects:
            effects.extend(domain_effects[phenomenon.domain])
        
        return effects[:10]  # Limit to top 10
    
    async def _remove_duplicate_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Remove duplicate hypotheses"""
        
        unique_hypotheses = []
        seen_statements = set()
        
        for hypothesis in hypotheses:
            # Create normalized statement for comparison
            normalized = hypothesis.statement.lower().strip()
            
            if normalized not in seen_statements:
                unique_hypotheses.append(hypothesis)
                seen_statements.add(normalized)
        
        return unique_hypotheses
    
    async def _enhance_hypotheses(self, hypotheses: List[Hypothesis], phenomena: List[Phenomenon]) -> List[Hypothesis]:
        """Enhance hypotheses with additional details"""
        
        enhanced_hypotheses = []
        
        for hypothesis in hypotheses:
            # Add scope domains
            hypothesis.scope_domains = list(set(p.domain for p in phenomena if p.id in hypothesis.explained_phenomena))
            
            # Add limitations
            hypothesis.limitations = await self._identify_hypothesis_limitations(hypothesis)
            
            # Add mechanisms if not present
            if not hypothesis.mechanisms:
                hypothesis.mechanisms = await self._infer_mechanisms(hypothesis)
            
            # Add predictions if not present
            if not hypothesis.predictions:
                hypothesis.predictions = await self._generate_predictions(hypothesis)
            
            enhanced_hypotheses.append(hypothesis)
        
        return enhanced_hypotheses
    
    async def _identify_hypothesis_limitations(self, hypothesis: Hypothesis) -> List[str]:
        """Identify limitations of hypothesis"""
        
        limitations = []
        
        # Origin-based limitations
        if hypothesis.origin == HypothesisOrigin.ANALOGICAL:
            limitations.append("Analogy may not be perfect")
        elif hypothesis.origin == HypothesisOrigin.CREATIVE:
            limitations.append("Creative hypothesis requires validation")
        elif hypothesis.origin == HypothesisOrigin.ELIMINATIVE:
            limitations.append("Elimination may be incomplete")
        
        # Type-based limitations
        if hypothesis.explanation_type == ExplanationType.STATISTICAL:
            limitations.append("Statistical explanation does not imply causation")
        elif hypothesis.explanation_type == ExplanationType.THEORETICAL:
            limitations.append("Theory may not apply to this specific case")
        
        # Scope limitations
        if len(hypothesis.scope_domains) == 1:
            limitations.append(f"Limited to {hypothesis.scope_domains[0]} domain")
        
        # Assumption limitations
        if len(hypothesis.assumptions) > 3:
            limitations.append("Many assumptions reduce reliability")
        
        return limitations
    
    async def _infer_mechanisms(self, hypothesis: Hypothesis) -> List[str]:
        """Infer mechanisms for hypothesis"""
        
        mechanisms = []
        
        # Type-based mechanisms
        if hypothesis.explanation_type == ExplanationType.CAUSAL:
            mechanisms.append("Direct causal mechanism")
        elif hypothesis.explanation_type == ExplanationType.MECHANISTIC:
            mechanisms.append("Underlying mechanistic process")
        elif hypothesis.explanation_type == ExplanationType.FUNCTIONAL:
            mechanisms.append("Functional mechanism")
        elif hypothesis.explanation_type == ExplanationType.EMERGENT:
            mechanisms.append("Emergent system property")
        
        # Origin-based mechanisms
        if hypothesis.origin == HypothesisOrigin.ANALOGICAL:
            mechanisms.append("Analogical mechanism transfer")
        elif hypothesis.origin == HypothesisOrigin.THEORETICAL:
            mechanisms.append("Theoretical mechanism application")
        
        return mechanisms
    
    async def _generate_predictions(self, hypothesis: Hypothesis) -> List[str]:
        """Generate predictions for hypothesis"""
        
        predictions = []
        
        # Type-based predictions
        if hypothesis.explanation_type == ExplanationType.CAUSAL:
            predictions.append("Causal intervention should affect outcome")
        elif hypothesis.explanation_type == ExplanationType.MECHANISTIC:
            predictions.append("Mechanism should be observable")
        elif hypothesis.explanation_type == ExplanationType.STATISTICAL:
            predictions.append("Statistical pattern should persist")
        
        # General predictions
        predictions.append("Hypothesis should explain related phenomena")
        predictions.append("Hypothesis should make testable predictions")
        
        return predictions
    
    async def _filter_and_rank_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Filter and rank hypotheses"""
        
        # Filter out very similar hypotheses
        filtered_hypotheses = []
        
        for hypothesis in hypotheses:
            # Check similarity with existing hypotheses
            is_similar = False
            for existing in filtered_hypotheses:
                similarity = await self._calculate_hypothesis_similarity(hypothesis, existing)
                if similarity > 0.8:
                    is_similar = True
                    break
            
            if not is_similar:
                filtered_hypotheses.append(hypothesis)
        
        # Rank by potential (combination of origin and type)
        def rank_hypothesis(h):
            origin_weights = {
                HypothesisOrigin.ANALOGICAL: 0.7,
                HypothesisOrigin.CAUSAL: 0.9,
                HypothesisOrigin.THEORETICAL: 0.8,
                HypothesisOrigin.EMPIRICAL: 0.7,
                HypothesisOrigin.CREATIVE: 0.6,
                HypothesisOrigin.ELIMINATIVE: 0.6
            }
            
            type_weights = {
                ExplanationType.CAUSAL: 0.9,
                ExplanationType.MECHANISTIC: 0.8,
                ExplanationType.FUNCTIONAL: 0.7,
                ExplanationType.STATISTICAL: 0.6,
                ExplanationType.THEORETICAL: 0.7
            }
            
            origin_weight = origin_weights.get(h.origin, 0.5)
            type_weight = type_weights.get(h.explanation_type, 0.5)
            
            return origin_weight * type_weight
        
        filtered_hypotheses.sort(key=rank_hypothesis, reverse=True)
        
        return filtered_hypotheses[:20]  # Limit to top 20
    
    async def _calculate_hypothesis_similarity(self, hypothesis1: Hypothesis, hypothesis2: Hypothesis) -> float:
        """Calculate similarity between hypotheses"""
        
        # Statement similarity
        words1 = set(hypothesis1.statement.lower().split())
        words2 = set(hypothesis2.statement.lower().split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "explains", "through", "via"}
        words1 -= stop_words
        words2 -= stop_words
        
        if words1 and words2:
            statement_similarity = len(words1 & words2) / len(words1 | words2)
        else:
            statement_similarity = 0.0
        
        # Type and origin similarity
        type_similarity = 1.0 if hypothesis1.explanation_type == hypothesis2.explanation_type else 0.0
        origin_similarity = 1.0 if hypothesis1.origin == hypothesis2.origin else 0.0
        
        # Weighted combination
        similarity = (
            0.6 * statement_similarity +
            0.2 * type_similarity +
            0.2 * origin_similarity
        )
        
        return similarity
    
    def _initialize_generation_strategies(self) -> List[str]:
        """Initialize generation strategies"""
        return [
            "analogical",
            "causal",
            "theoretical",
            "empirical",
            "creative",
            "eliminative"
        ]
    
    def _initialize_hypothesis_origins(self) -> Dict[str, Any]:
        """Initialize hypothesis origins configuration"""
        return {
            "analogical": {"weight": 0.7, "creativity": 0.6},
            "causal": {"weight": 0.9, "creativity": 0.5},
            "theoretical": {"weight": 0.8, "creativity": 0.4},
            "empirical": {"weight": 0.7, "creativity": 0.3},
            "creative": {"weight": 0.6, "creativity": 0.9},
            "eliminative": {"weight": 0.6, "creativity": 0.5}
        }
    
    def _initialize_creativity_enhancers(self) -> List[str]:
        """Initialize creativity enhancement techniques"""
        return [
            "lateral_thinking",
            "perspective_shifting",
            "constraint_removal",
            "analogy_extension",
            "synthesis_combination",
            "inversion_thinking"
        ]


class ExplanationSelectionEngine:
    """Engine for selecting the best explanation from hypotheses"""
    
    def __init__(self):
        self.evaluation_criteria = self._initialize_evaluation_criteria()
        self.selection_strategies = self._initialize_selection_strategies()
        self.comparison_methods = self._initialize_comparison_methods()
    
    async def select_best_explanation(self, hypotheses: List[Hypothesis], phenomena: List[Phenomenon], context: Dict[str, Any] = None) -> Hypothesis:
        """Select the best explanation from hypotheses"""
        
        logger.info(f"Selecting best explanation from {len(hypotheses)} hypotheses")
        
        if not hypotheses:
            return await self._create_default_hypothesis(phenomena)
        
        # Evaluate all hypotheses
        evaluated_hypotheses = await self._evaluate_hypotheses(hypotheses, phenomena)
        
        # Apply selection criteria
        scored_hypotheses = await self._apply_selection_criteria(evaluated_hypotheses, phenomena)
        
        # Perform comparative analysis
        ranked_hypotheses = await self._perform_comparative_analysis(scored_hypotheses)
        
        # Select best with confidence assessment
        best_hypothesis = await self._select_with_confidence(ranked_hypotheses, phenomena)
        
        logger.info(f"Selected best explanation: {best_hypothesis.statement}")
        
        return best_hypothesis
    
    async def _evaluate_hypotheses(self, hypotheses: List[Hypothesis], phenomena: List[Phenomenon]) -> List[Hypothesis]:
        """Evaluate hypotheses against all criteria"""
        
        evaluated_hypotheses = []
        
        for hypothesis in hypotheses:
            # Evaluate against each criterion
            hypothesis.simplicity_score = await self._evaluate_simplicity(hypothesis)
            hypothesis.scope_score = await self._evaluate_scope(hypothesis, phenomena)
            hypothesis.plausibility_score = await self._evaluate_plausibility(hypothesis, phenomena)
            hypothesis.coherence_score = await self._evaluate_coherence(hypothesis)
            hypothesis.testability_score = await self._evaluate_testability(hypothesis)
            hypothesis.explanatory_power_score = await self._evaluate_explanatory_power(hypothesis, phenomena)
            hypothesis.consilience_score = await self._evaluate_consilience(hypothesis, phenomena)
            
            # Calculate overall score
            hypothesis.calculate_overall_score()
            
            evaluated_hypotheses.append(hypothesis)
        
        return evaluated_hypotheses
    
    async def _evaluate_simplicity(self, hypothesis: Hypothesis) -> float:
        """Evaluate simplicity of hypothesis (Occam's razor)"""
        
        # Factors affecting simplicity
        assumption_penalty = len(hypothesis.assumptions) * 0.1
        mechanism_penalty = len(hypothesis.mechanisms) * 0.05
        
        # Word count penalty for complex statements
        word_count = len(hypothesis.statement.split())
        complexity_penalty = max(0, (word_count - 15) * 0.01)
        
        # Premise complexity
        premise_complexity = len(hypothesis.premises) * 0.05
        
        # Base simplicity
        simplicity = 1.0 - assumption_penalty - mechanism_penalty - complexity_penalty - premise_complexity
        
        # Bonus for elegant explanations
        elegance_indicators = ["single", "unified", "simple", "direct", "straightforward"]
        for indicator in elegance_indicators:
            if indicator in hypothesis.statement.lower():
                simplicity += 0.1
                break
        
        return max(0.0, min(1.0, simplicity))
    
    async def _evaluate_scope(self, hypothesis: Hypothesis, phenomena: List[Phenomenon]) -> float:
        """Evaluate scope of hypothesis (breadth of explanation)"""
        
        # Phenomena coverage
        explained_phenomena = len(hypothesis.explained_phenomena)
        total_phenomena = len(phenomena)
        
        if total_phenomena > 0:
            coverage_score = explained_phenomena / total_phenomena
        else:
            coverage_score = 0.0
        
        # Domain coverage
        domain_coverage = len(hypothesis.scope_domains)
        max_domains = len(set(p.domain for p in phenomena))
        
        if max_domains > 0:
            domain_score = domain_coverage / max_domains
        else:
            domain_score = 0.0
        
        # Mechanism breadth
        mechanism_breadth = min(1.0, len(hypothesis.mechanisms) * 0.2)
        
        # Prediction breadth
        prediction_breadth = min(1.0, len(hypothesis.predictions) * 0.15)
        
        # Combined scope
        scope = (
            0.4 * coverage_score +
            0.3 * domain_score +
            0.2 * mechanism_breadth +
            0.1 * prediction_breadth
        )
        
        return max(0.0, min(1.0, scope))
    
    async def _evaluate_plausibility(self, hypothesis: Hypothesis, phenomena: List[Phenomenon]) -> float:
        """Evaluate plausibility of hypothesis"""
        
        # Base plausibility
        plausibility = 0.7
        
        # Origin-based plausibility
        origin_plausibility = {
            HypothesisOrigin.ANALOGICAL: 0.7,
            HypothesisOrigin.CAUSAL: 0.8,
            HypothesisOrigin.THEORETICAL: 0.9,
            HypothesisOrigin.EMPIRICAL: 0.8,
            HypothesisOrigin.CREATIVE: 0.5,
            HypothesisOrigin.ELIMINATIVE: 0.7
        }
        
        plausibility *= origin_plausibility.get(hypothesis.origin, 0.7)
        
        # Type-based plausibility
        type_plausibility = {
            ExplanationType.CAUSAL: 0.8,
            ExplanationType.MECHANISTIC: 0.9,
            ExplanationType.FUNCTIONAL: 0.7,
            ExplanationType.STATISTICAL: 0.6,
            ExplanationType.THEORETICAL: 0.8
        }
        
        plausibility *= type_plausibility.get(hypothesis.explanation_type, 0.7)
        
        # Assumption plausibility
        if len(hypothesis.assumptions) > 5:
            plausibility *= 0.8  # Too many assumptions reduce plausibility
        
        # Domain consistency
        domains = set(hypothesis.scope_domains)
        if len(domains) == 1:
            plausibility *= 1.1  # Domain consistency bonus
        elif len(domains) > 3:
            plausibility *= 0.9  # Too many domains reduce plausibility
        
        return max(0.0, min(1.0, plausibility))
    
    async def _evaluate_coherence(self, hypothesis: Hypothesis) -> float:
        """Evaluate internal coherence of hypothesis"""
        
        # Base coherence
        coherence = 0.8
        
        # Check for logical consistency
        statement_words = set(hypothesis.statement.lower().split())
        
        # Check for contradictory terms
        contradictory_pairs = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("cause", "prevent"),
            ("enable", "disable"),
            ("create", "destroy"),
            ("activate", "deactivate")
        ]
        
        for term1, term2 in contradictory_pairs:
            if term1 in statement_words and term2 in statement_words:
                coherence -= 0.2
        
        # Check assumption consistency
        assumption_text = " ".join(hypothesis.assumptions).lower()
        for term1, term2 in contradictory_pairs:
            if term1 in assumption_text and term2 in assumption_text:
                coherence -= 0.1
        
        # Check premise-conclusion consistency
        premise_text = " ".join(hypothesis.premises).lower()
        statement_lower = hypothesis.statement.lower()
        
        # Simple consistency check
        premise_words = set(premise_text.split())
        statement_words = set(statement_lower.split())
        
        if premise_words and statement_words:
            overlap = len(premise_words & statement_words)
            if overlap > 0:
                coherence += 0.1
        
        # Mechanism consistency
        if hypothesis.mechanisms:
            mechanism_text = " ".join(hypothesis.mechanisms).lower()
            if any(word in mechanism_text for word in statement_words):
                coherence += 0.05
        
        return max(0.0, min(1.0, coherence))
    
    async def _evaluate_testability(self, hypothesis: Hypothesis) -> float:
        """Evaluate testability of hypothesis"""
        
        # Base testability
        testability = 0.5
        
        # Prediction-based testability
        prediction_count = len(hypothesis.predictions)
        if prediction_count > 0:
            testability += min(0.3, prediction_count * 0.1)
        
        # Specific testability indicators
        testable_indicators = [
            "measure", "test", "observe", "detect", "quantify",
            "experiment", "verify", "validate", "confirm", "check"
        ]
        
        prediction_text = " ".join(hypothesis.predictions).lower()
        for indicator in testable_indicators:
            if indicator in prediction_text:
                testability += 0.1
                break
        
        # Mechanism testability
        if hypothesis.mechanisms:
            mechanism_text = " ".join(hypothesis.mechanisms).lower()
            for indicator in testable_indicators:
                if indicator in mechanism_text:
                    testability += 0.05
                    break
        
        # Domain-specific testability
        if hypothesis.scope_domains:
            domain_testability = {
                "medical": 0.8,
                "scientific": 0.9,
                "technical": 0.7,
                "criminal": 0.6,
                "social": 0.5
            }
            
            avg_domain_testability = sum(domain_testability.get(domain, 0.5) for domain in hypothesis.scope_domains) / len(hypothesis.scope_domains)
            testability *= avg_domain_testability
        
        return max(0.0, min(1.0, testability))
    
    async def _evaluate_explanatory_power(self, hypothesis: Hypothesis, phenomena: List[Phenomenon]) -> float:
        """Evaluate explanatory power of hypothesis"""
        
        # Base explanatory power
        explanatory_power = 0.6
        
        # Mechanism explanatory power
        if hypothesis.mechanisms:
            explanatory_power += min(0.2, len(hypothesis.mechanisms) * 0.05)
        
        # Depth of explanation
        depth_indicators = ["because", "due to", "mechanism", "process", "why", "how"]
        for indicator in depth_indicators:
            if indicator in hypothesis.statement.lower():
                explanatory_power += 0.05
        
        # Phenomena coverage quality
        covered_phenomena = [p for p in phenomena if p.id in hypothesis.explained_phenomena]
        if covered_phenomena:
            # Quality of coverage
            coverage_quality = sum(p.get_overall_score() for p in covered_phenomena) / len(covered_phenomena)
            explanatory_power *= coverage_quality
        
        # Type-based explanatory power
        type_power = {
            ExplanationType.CAUSAL: 0.9,
            ExplanationType.MECHANISTIC: 1.0,
            ExplanationType.FUNCTIONAL: 0.8,
            ExplanationType.STATISTICAL: 0.6,
            ExplanationType.THEORETICAL: 0.8
        }
        
        explanatory_power *= type_power.get(hypothesis.explanation_type, 0.7)
        
        # Prediction power
        if hypothesis.predictions:
            prediction_power = min(0.2, len(hypothesis.predictions) * 0.03)
            explanatory_power += prediction_power
        
        return max(0.0, min(1.0, explanatory_power))
    
    async def _evaluate_consilience(self, hypothesis: Hypothesis, phenomena: List[Phenomenon]) -> float:
        """Evaluate consilience (unification of diverse observations)"""
        
        # Base consilience
        consilience = 0.5
        
        # Domain diversity
        domains = set(hypothesis.scope_domains)
        if len(domains) > 1:
            consilience += min(0.3, (len(domains) - 1) * 0.1)
        
        # Phenomenon type diversity
        covered_phenomena = [p for p in phenomena if p.id in hypothesis.explained_phenomena]
        if covered_phenomena:
            phenomenon_types = set(p.phenomenon_type for p in covered_phenomena)
            if len(phenomenon_types) > 1:
                consilience += min(0.2, (len(phenomenon_types) - 1) * 0.05)
        
        # Unification indicators
        unification_indicators = ["unified", "connects", "integrates", "combines", "unifies", "links"]
        for indicator in unification_indicators:
            if indicator in hypothesis.statement.lower():
                consilience += 0.1
                break
        
        # Mechanism integration
        if len(hypothesis.mechanisms) > 1:
            consilience += 0.1
        
        return max(0.0, min(1.0, consilience))
    
    async def _apply_selection_criteria(self, hypotheses: List[Hypothesis], phenomena: List[Phenomenon]) -> List[Hypothesis]:
        """Apply selection criteria to rank hypotheses"""
        
        # Sort by overall score
        ranked_hypotheses = sorted(hypotheses, key=lambda h: h.overall_score, reverse=True)
        
        return ranked_hypotheses
    
    async def _perform_comparative_analysis(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Perform comparative analysis between hypotheses"""
        
        if len(hypotheses) < 2:
            return hypotheses
        
        # Compare top hypotheses
        for i, hypothesis in enumerate(hypotheses[:5]):  # Top 5
            # Compare with others
            competitors = [h for h in hypotheses if h != hypothesis]
            
            # Calculate competitive advantage
            if competitors:
                max_competitor_score = max(h.overall_score for h in competitors)
                hypothesis.competitive_advantage = hypothesis.overall_score - max_competitor_score
            else:
                hypothesis.competitive_advantage = 0.0
        
        return hypotheses
    
    async def _select_with_confidence(self, hypotheses: List[Hypothesis], phenomena: List[Phenomenon]) -> Hypothesis:
        """Select best hypothesis with confidence assessment"""
        
        if not hypotheses:
            return await self._create_default_hypothesis(phenomena)
        
        best_hypothesis = hypotheses[0]
        
        # Calculate confidence
        confidence_factors = [
            best_hypothesis.overall_score,
            best_hypothesis.plausibility_score,
            best_hypothesis.coherence_score,
            best_hypothesis.explanatory_power_score
        ]
        
        # Adjust for competitive advantage
        if len(hypotheses) > 1:
            gap_to_next = best_hypothesis.overall_score - hypotheses[1].overall_score
            confidence_factors.append(min(1.0, gap_to_next * 2))
        
        # Calculate confidence
        confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Adjust for uncertainty sources
        uncertainty_sources = []
        
        if best_hypothesis.overall_score < 0.7:
            uncertainty_sources.append("Low overall score")
        
        if len(best_hypothesis.assumptions) > 3:
            uncertainty_sources.append("Many assumptions")
        
        if best_hypothesis.testability_score < 0.5:
            uncertainty_sources.append("Limited testability")
        
        if len(hypotheses) > 1 and hypotheses[1].overall_score > 0.6:
            uncertainty_sources.append("Strong competing hypotheses")
        
        best_hypothesis.confidence = confidence
        best_hypothesis.uncertainty_sources = uncertainty_sources
        
        return best_hypothesis
    
    async def _create_default_hypothesis(self, phenomena: List[Phenomenon]) -> Hypothesis:
        """Create a default hypothesis when none are available"""
        
        return Hypothesis(
            id="default_hypothesis",
            statement="No plausible explanation found for the observed phenomena",
            explanation_type=ExplanationType.MECHANISTIC,
            origin=HypothesisOrigin.ELIMINATIVE,
            explained_phenomena=[p.id for p in phenomena],
            overall_score=0.1,
            confidence=0.1,
            uncertainty_sources=["No viable hypotheses generated"]
        )
    
    def _initialize_evaluation_criteria(self) -> Dict[str, float]:
        """Initialize evaluation criteria weights"""
        return {
            "simplicity": 0.15,
            "scope": 0.20,
            "plausibility": 0.20,
            "coherence": 0.15,
            "testability": 0.10,
            "explanatory_power": 0.15,
            "consilience": 0.05
        }
    
    def _initialize_selection_strategies(self) -> List[str]:
        """Initialize selection strategies"""
        return [
            "best_overall_score",
            "best_explanatory_power",
            "most_testable",
            "most_plausible",
            "most_consilient"
        ]
    
    def _initialize_comparison_methods(self) -> List[str]:
        """Initialize comparison methods"""
        return [
            "pairwise_comparison",
            "multi_criteria_analysis",
            "competitive_analysis",
            "sensitivity_analysis"
        ]


class FitEvaluationEngine:
    """Engine for evaluating explanation fit and validation"""
    
    def __init__(self):
        self.validation_tests = self._initialize_validation_tests()
        self.fit_assessors = self._initialize_fit_assessors()
        self.robustness_tests = self._initialize_robustness_tests()
    
    async def evaluate_explanation_fit(self, hypothesis: Hypothesis, phenomena: List[Phenomenon], context: Dict[str, Any] = None) -> ExplanationEvaluation:
        """Evaluate how well explanation fits the phenomena"""
        
        logger.info(f"Evaluating explanation fit for hypothesis: {hypothesis.id}")
        
        # Create evaluation
        evaluation = ExplanationEvaluation(
            id=f"eval_{hypothesis.id}",
            hypothesis_id=hypothesis.id
        )
        
        # Assess phenomenon fit
        evaluation.phenomenon_fit = await self._assess_phenomenon_fit(hypothesis, phenomena)
        
        # Assess evidence consistency
        evaluation.evidence_consistency = await self._assess_evidence_consistency(hypothesis, phenomena)
        
        # Assess prediction accuracy
        evaluation.prediction_accuracy = await self._assess_prediction_accuracy(hypothesis, phenomena)
        
        # Assess mechanistic plausibility
        evaluation.mechanistic_plausibility = await self._assess_mechanistic_plausibility(hypothesis, phenomena)
        
        # Perform validation tests
        evaluation = await self._perform_validation_tests(evaluation, hypothesis, phenomena)
        
        # Perform comparative analysis
        evaluation = await self._perform_comparative_analysis(evaluation, hypothesis, phenomena)
        
        # Assess robustness
        evaluation.robustness_score = await self._assess_robustness(hypothesis, phenomena)
        
        # Calculate confidence
        evaluation.confidence_level = await self._calculate_confidence_level(evaluation)
        
        # Update hypothesis with fit score
        hypothesis.fit_score = evaluation.get_overall_fit_score()
        
        logger.info(f"Evaluation complete. Fit score: {hypothesis.fit_score:.2f}")
        
        return evaluation
    
    async def _assess_phenomenon_fit(self, hypothesis: Hypothesis, phenomena: List[Phenomenon]) -> float:
        """Assess how well hypothesis fits the phenomena"""
        
        fit_scores = []
        
        for phenomenon in phenomena:
            if phenomenon.id in hypothesis.explained_phenomena:
                # Calculate fit for this phenomenon
                fit = await self._calculate_single_phenomenon_fit(hypothesis, phenomenon)
                fit_scores.append(fit)
        
        if fit_scores:
            return sum(fit_scores) / len(fit_scores)
        else:
            return 0.0
    
    async def _calculate_single_phenomenon_fit(self, hypothesis: Hypothesis, phenomenon: Phenomenon) -> float:
        """Calculate fit for a single phenomenon"""
        
        # Base fit
        fit = 0.6
        
        # Keyword overlap
        hyp_words = set(hypothesis.statement.lower().split())
        phen_words = set(phenomenon.description.lower().split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        hyp_words -= stop_words
        phen_words -= stop_words
        
        if hyp_words and phen_words:
            overlap = len(hyp_words & phen_words)
            fit += min(0.3, overlap * 0.05)
        
        # Domain consistency
        if hypothesis.scope_domains and phenomenon.domain in hypothesis.scope_domains:
            fit += 0.1
        
        # Phenomenon type compatibility
        type_compatibility = {
            (PhenomenonType.ANOMALOUS, ExplanationType.CAUSAL): 0.9,
            (PhenomenonType.PUZZLING, ExplanationType.MECHANISTIC): 0.8,
            (PhenomenonType.CONTRADICTORY, ExplanationType.THEORETICAL): 0.7,
            (PhenomenonType.NOVEL, ExplanationType.EMERGENT): 0.9,
            (PhenomenonType.SYSTEMATIC, ExplanationType.STATISTICAL): 0.8,
            (PhenomenonType.CAUSAL, ExplanationType.CAUSAL): 1.0
        }
        
        compatibility = type_compatibility.get((phenomenon.phenomenon_type, hypothesis.explanation_type), 0.7)
        fit *= compatibility
        
        # Anomalous feature handling
        if phenomenon.anomalous_features:
            # Check if hypothesis addresses anomalous features
            hyp_text = hypothesis.statement.lower()
            addressed_features = sum(1 for feature in phenomenon.anomalous_features if feature.lower() in hyp_text)
            if addressed_features > 0:
                fit += min(0.2, addressed_features * 0.1)
        
        return max(0.0, min(1.0, fit))
    
    async def _assess_evidence_consistency(self, hypothesis: Hypothesis, phenomena: List[Phenomenon]) -> float:
        """Assess consistency with available evidence"""
        
        consistency_scores = []
        
        for phenomenon in phenomena:
            if phenomenon.id in hypothesis.explained_phenomena:
                # Check consistency with observations
                for observation in phenomenon.observations:
                    consistency = await self._calculate_observation_consistency(hypothesis, observation)
                    consistency_scores.append(consistency)
        
        if consistency_scores:
            return sum(consistency_scores) / len(consistency_scores)
        else:
            return 0.7  # Neutral if no observations
    
    async def _calculate_observation_consistency(self, hypothesis: Hypothesis, observation: str) -> float:
        """Calculate consistency between hypothesis and observation"""
        
        # Simple consistency based on contradiction detection
        consistency = 0.8  # Base consistency
        
        # Check for contradictory terms
        hyp_words = set(hypothesis.statement.lower().split())
        obs_words = set(observation.lower().split())
        
        # Contradictory pairs
        contradictory_pairs = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("cause", "prevent"),
            ("present", "absent"),
            ("high", "low"),
            ("active", "inactive")
        ]
        
        for term1, term2 in contradictory_pairs:
            if ((term1 in hyp_words and term2 in obs_words) or 
                (term2 in hyp_words and term1 in obs_words)):
                consistency -= 0.3
        
        # Supportive overlap
        overlap = len(hyp_words & obs_words)
        if overlap > 0:
            consistency += min(0.2, overlap * 0.02)
        
        return max(0.0, min(1.0, consistency))
    
    async def _assess_prediction_accuracy(self, hypothesis: Hypothesis, phenomena: List[Phenomenon]) -> float:
        """Assess accuracy of predictions"""
        
        # For now, assess potential prediction accuracy
        # In a full implementation, would test against actual outcomes
        
        prediction_quality = 0.6  # Base quality
        
        # Prediction specificity
        if hypothesis.predictions:
            specific_indicators = ["specific", "measure", "quantify", "exactly", "precisely"]
            for prediction in hypothesis.predictions:
                for indicator in specific_indicators:
                    if indicator in prediction.lower():
                        prediction_quality += 0.1
                        break
        
        # Prediction testability
        testable_indicators = ["test", "verify", "check", "observe", "measure"]
        for prediction in hypothesis.predictions:
            for indicator in testable_indicators:
                if indicator in prediction.lower():
                    prediction_quality += 0.05
                    break
        
        # Prediction diversity
        if len(hypothesis.predictions) > 2:
            prediction_quality += 0.1
        
        return max(0.0, min(1.0, prediction_quality))
    
    async def _assess_mechanistic_plausibility(self, hypothesis: Hypothesis, phenomena: List[Phenomenon]) -> float:
        """Assess mechanistic plausibility"""
        
        plausibility = 0.7  # Base plausibility
        
        # Mechanism clarity
        if hypothesis.mechanisms:
            clear_mechanisms = sum(1 for mechanism in hypothesis.mechanisms if len(mechanism.split()) > 3)
            plausibility += min(0.2, clear_mechanisms * 0.05)
        
        # Mechanism type appropriateness
        if hypothesis.explanation_type == ExplanationType.MECHANISTIC:
            plausibility += 0.1
        elif hypothesis.explanation_type == ExplanationType.CAUSAL:
            plausibility += 0.05
        
        # Domain-specific plausibility
        domain_plausibility = {
            "medical": 0.8,
            "scientific": 0.9,
            "technical": 0.8,
            "criminal": 0.7,
            "social": 0.6
        }
        
        if hypothesis.scope_domains:
            avg_domain_plausibility = sum(domain_plausibility.get(domain, 0.7) for domain in hypothesis.scope_domains) / len(hypothesis.scope_domains)
            plausibility *= avg_domain_plausibility
        
        return max(0.0, min(1.0, plausibility))
    
    async def _perform_validation_tests(self, evaluation: ExplanationEvaluation, hypothesis: Hypothesis, phenomena: List[Phenomenon]) -> ExplanationEvaluation:
        """Perform validation tests"""
        
        validation_tests = []
        test_results = {}
        
        # Consistency test
        validation_tests.append("consistency_test")
        test_results["consistency_test"] = evaluation.evidence_consistency
        
        # Completeness test
        validation_tests.append("completeness_test")
        completeness = len(hypothesis.explained_phenomena) / len(phenomena) if phenomena else 0.0
        test_results["completeness_test"] = completeness
        
        # Coherence test
        validation_tests.append("coherence_test")
        test_results["coherence_test"] = hypothesis.coherence_score
        
        # Testability test
        validation_tests.append("testability_test")
        test_results["testability_test"] = hypothesis.testability_score
        
        # Plausibility test
        validation_tests.append("plausibility_test")
        test_results["plausibility_test"] = hypothesis.plausibility_score
        
        evaluation.validation_tests = validation_tests
        evaluation.test_results = test_results
        
        return evaluation
    
    async def _perform_comparative_analysis(self, evaluation: ExplanationEvaluation, hypothesis: Hypothesis, phenomena: List[Phenomenon]) -> ExplanationEvaluation:
        """Perform comparative analysis"""
        
        # For now, use competitive advantage from hypothesis
        evaluation.competitive_advantage = getattr(hypothesis, 'competitive_advantage', 0.0)
        
        # Alternative comparisons
        evaluation.alternative_comparisons = [
            "Compared to null hypothesis",
            "Compared to simpler explanations",
            "Compared to domain-specific alternatives"
        ]
        
        return evaluation
    
    async def _assess_robustness(self, hypothesis: Hypothesis, phenomena: List[Phenomenon]) -> float:
        """Assess robustness of explanation"""
        
        robustness = 0.7  # Base robustness
        
        # Assumption robustness
        if len(hypothesis.assumptions) < 3:
            robustness += 0.1
        elif len(hypothesis.assumptions) > 5:
            robustness -= 0.1
        
        # Scope robustness
        if len(hypothesis.scope_domains) > 1:
            robustness += 0.1
        
        # Mechanism robustness
        if hypothesis.mechanisms:
            robustness += 0.1
        
        # Prediction robustness
        if len(hypothesis.predictions) > 2:
            robustness += 0.1
        
        return max(0.0, min(1.0, robustness))
    
    async def _calculate_confidence_level(self, evaluation: ExplanationEvaluation) -> ConfidenceLevel:
        """Calculate confidence level"""
        
        overall_fit = evaluation.get_overall_fit_score()
        
        if overall_fit >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif overall_fit >= 0.7:
            return ConfidenceLevel.HIGH
        elif overall_fit >= 0.5:
            return ConfidenceLevel.MODERATE
        elif overall_fit >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _initialize_validation_tests(self) -> List[str]:
        """Initialize validation tests"""
        return [
            "consistency_test",
            "completeness_test",
            "coherence_test",
            "testability_test",
            "plausibility_test",
            "robustness_test"
        ]
    
    def _initialize_fit_assessors(self) -> List[str]:
        """Initialize fit assessment methods"""
        return [
            "phenomenon_fit",
            "evidence_consistency",
            "prediction_accuracy",
            "mechanistic_plausibility"
        ]
    
    def _initialize_robustness_tests(self) -> List[str]:
        """Initialize robustness tests"""
        return [
            "assumption_variation",
            "scope_extension",
            "mechanism_alternative",
            "prediction_sensitivity"
        ]


class InferenceApplicationEngine:
    """Engine for applying abductive inference to practical contexts"""
    
    def __init__(self):
        self.application_strategies = self._initialize_application_strategies()
        self.domain_adapters = self._initialize_domain_adapters()
        self.decision_frameworks = self._initialize_decision_frameworks()
    
    async def apply_inference(self, hypothesis: Hypothesis, evaluation: ExplanationEvaluation, context: Dict[str, Any] = None) -> InferenceApplication:
        """Apply abductive inference to practical contexts"""
        
        logger.info(f"Applying inference for hypothesis: {hypothesis.id}")
        
        # Create application
        application = InferenceApplication(
            id=f"app_{hypothesis.id}",
            hypothesis_id=hypothesis.id,
            application_type=await self._determine_application_type(hypothesis, context),
            context=context or {},
            domain=hypothesis.scope_domains[0] if hypothesis.scope_domains else "general"
        )
        
        # Generate action recommendations
        application.action_recommendations = await self._generate_action_recommendations(hypothesis, evaluation, context)
        
        # Generate decision guidance
        application.decision_guidance = await self._generate_decision_guidance(hypothesis, evaluation, context)
        
        # Assess risks
        application.risk_assessments = await self._assess_risks(hypothesis, evaluation, context)
        
        # Generate practical implications
        application = await self._generate_practical_implications(application, hypothesis, evaluation)
        
        # Generate predictions and forecasts
        application = await self._generate_predictions_forecasts(application, hypothesis, evaluation)
        
        # Define success metrics
        application.success_indicators = await self._define_success_indicators(hypothesis, evaluation)
        
        # Assess application confidence
        application.application_confidence = await self._assess_application_confidence(hypothesis, evaluation)
        
        logger.info(f"Inference application complete. Confidence: {application.application_confidence:.2f}")
        
        return application
    
    async def _determine_application_type(self, hypothesis: Hypothesis, context: Dict[str, Any] = None) -> str:
        """Determine the type of application"""
        
        if context and "application_type" in context:
            return context["application_type"]
        
        # Determine from hypothesis characteristics
        if hypothesis.scope_domains:
            domain = hypothesis.scope_domains[0]
            
            domain_applications = {
                "medical": "diagnostic",
                "criminal": "investigative",
                "technical": "troubleshooting",
                "scientific": "research",
                "business": "strategic",
                "social": "intervention",
                "educational": "pedagogical"
            }
            
            return domain_applications.get(domain, "general")
        
        return "general"
    
    async def _generate_action_recommendations(self, hypothesis: Hypothesis, evaluation: ExplanationEvaluation, context: Dict[str, Any] = None) -> List[str]:
        """Generate action recommendations"""
        
        recommendations = []
        
        # Domain-specific recommendations
        if hypothesis.scope_domains:
            domain = hypothesis.scope_domains[0]
            
            if domain == "medical":
                recommendations.extend([
                    "Conduct additional diagnostic tests to confirm hypothesis",
                    "Initiate treatment protocol based on suspected condition",
                    "Monitor patient response to treatment",
                    "Consult specialists if needed"
                ])
            
            elif domain == "criminal":
                recommendations.extend([
                    "Collect additional evidence to support hypothesis",
                    "Interview relevant witnesses",
                    "Conduct forensic analysis",
                    "Pursue investigative leads"
                ])
            
            elif domain == "technical":
                recommendations.extend([
                    "Implement diagnostic procedures",
                    "Test suspected components",
                    "Apply corrective measures",
                    "Monitor system performance"
                ])
            
            elif domain == "scientific":
                recommendations.extend([
                    "Design experiments to test hypothesis",
                    "Collect additional data",
                    "Peer review findings",
                    "Publish results"
                ])
            
            elif domain == "business":
                recommendations.extend([
                    "Implement strategic changes",
                    "Monitor key performance indicators",
                    "Adjust resource allocation",
                    "Evaluate market response"
                ])
        
        # General recommendations
        recommendations.extend([
            "Test key predictions of the hypothesis",
            "Gather additional supporting evidence",
            "Monitor for contradictory evidence",
            "Review and update hypothesis as needed"
        ])
        
        # Confidence-based recommendations
        if evaluation.confidence_level == ConfidenceLevel.HIGH or evaluation.confidence_level == ConfidenceLevel.VERY_HIGH:
            recommendations.append("Proceed with high confidence actions")
        elif evaluation.confidence_level == ConfidenceLevel.LOW or evaluation.confidence_level == ConfidenceLevel.VERY_LOW:
            recommendations.append("Proceed with caution and gather more evidence")
        
        return recommendations
    
    async def _generate_decision_guidance(self, hypothesis: Hypothesis, evaluation: ExplanationEvaluation, context: Dict[str, Any] = None) -> List[str]:
        """Generate decision guidance"""
        
        guidance = []
        
        # Confidence-based guidance
        if evaluation.confidence_level == ConfidenceLevel.VERY_HIGH:
            guidance.append("High confidence: Proceed with decisive action")
        elif evaluation.confidence_level == ConfidenceLevel.HIGH:
            guidance.append("Good confidence: Proceed with standard precautions")
        elif evaluation.confidence_level == ConfidenceLevel.MODERATE:
            guidance.append("Moderate confidence: Proceed with careful monitoring")
        elif evaluation.confidence_level == ConfidenceLevel.LOW:
            guidance.append("Low confidence: Proceed with extreme caution")
        else:
            guidance.append("Very low confidence: Seek additional evidence before proceeding")
        
        # Testability-based guidance
        if hypothesis.testability_score > 0.7:
            guidance.append("High testability: Implement testing protocol")
        elif hypothesis.testability_score < 0.3:
            guidance.append("Low testability: Focus on evidence gathering")
        
        # Scope-based guidance
        if hypothesis.scope_score > 0.7:
            guidance.append("Broad scope: Consider system-wide implications")
        elif hypothesis.scope_score < 0.3:
            guidance.append("Limited scope: Focus on specific areas")
        
        # Risk-based guidance
        if evaluation.robustness_score > 0.7:
            guidance.append("Robust explanation: Suitable for high-stakes decisions")
        elif evaluation.robustness_score < 0.3:
            guidance.append("Limited robustness: Avoid high-risk applications")
        
        return guidance
    
    async def _assess_risks(self, hypothesis: Hypothesis, evaluation: ExplanationEvaluation, context: Dict[str, Any] = None) -> List[str]:
        """Assess risks associated with the hypothesis"""
        
        risks = []
        
        # Confidence-based risks
        if evaluation.confidence_level == ConfidenceLevel.LOW or evaluation.confidence_level == ConfidenceLevel.VERY_LOW:
            risks.append("Risk of incorrect explanation leading to wrong actions")
        
        # Assumption-based risks
        if len(hypothesis.assumptions) > 3:
            risks.append("Risk from unvalidated assumptions")
        
        # Testability-based risks
        if hypothesis.testability_score < 0.5:
            risks.append("Risk of unverifiable explanation")
        
        # Scope-based risks
        if hypothesis.scope_score < 0.5:
            risks.append("Risk of incomplete explanation")
        
        # Domain-specific risks
        if hypothesis.scope_domains:
            domain = hypothesis.scope_domains[0]
            
            domain_risks = {
                "medical": ["Risk of misdiagnosis", "Risk of inappropriate treatment"],
                "criminal": ["Risk of wrongful accusation", "Risk of missed evidence"],
                "technical": ["Risk of system failure", "Risk of data loss"],
                "scientific": ["Risk of false discovery", "Risk of research bias"],
                "business": ["Risk of strategic failure", "Risk of financial loss"]
            }
            
            risks.extend(domain_risks.get(domain, []))
        
        # Uncertainty-based risks
        if hypothesis.uncertainty_sources:
            risks.append("Risk from identified uncertainty sources")
        
        return risks
    
    async def _generate_practical_implications(self, application: InferenceApplication, hypothesis: Hypothesis, evaluation: ExplanationEvaluation) -> InferenceApplication:
        """Generate practical implications"""
        
        # Immediate actions
        application.immediate_actions = []
        
        if evaluation.confidence_level == ConfidenceLevel.HIGH or evaluation.confidence_level == ConfidenceLevel.VERY_HIGH:
            application.immediate_actions.append("Implement immediate response based on hypothesis")
        
        if hypothesis.testability_score > 0.7:
            application.immediate_actions.append("Begin testing protocol")
        
        application.immediate_actions.append("Brief stakeholders on findings")
        application.immediate_actions.append("Set up monitoring system")
        
        # Long-term strategies
        application.long_term_strategies = []
        
        if hypothesis.scope_score > 0.7:
            application.long_term_strategies.append("Develop comprehensive strategy based on broad implications")
        
        application.long_term_strategies.append("Establish ongoing evaluation process")
        application.long_term_strategies.append("Build knowledge base for similar cases")
        application.long_term_strategies.append("Train team on new insights")
        
        # Monitoring requirements
        application.monitoring_requirements = []
        
        if evaluation.confidence_level == ConfidenceLevel.MODERATE or evaluation.confidence_level == ConfidenceLevel.LOW:
            application.monitoring_requirements.append("Continuous monitoring of key indicators")
        
        application.monitoring_requirements.extend([
            "Track prediction accuracy",
            "Monitor for contradictory evidence",
            "Assess implementation effectiveness",
            "Review and update hypothesis regularly"
        ])
        
        return application
    
    async def _generate_predictions_forecasts(self, application: InferenceApplication, hypothesis: Hypothesis, evaluation: ExplanationEvaluation) -> InferenceApplication:
        """Generate predictions and forecasts"""
        
        # Predictions
        application.predictions = hypothesis.predictions.copy()
        
        # Add application-specific predictions
        if evaluation.confidence_level == ConfidenceLevel.HIGH or evaluation.confidence_level == ConfidenceLevel.VERY_HIGH:
            application.predictions.append("High success probability for recommended actions")
        
        if hypothesis.scope_score > 0.7:
            application.predictions.append("Broad impact across multiple areas")
        
        # Forecasts
        application.forecasts = {}
        
        # Short-term forecast
        if evaluation.confidence_level == ConfidenceLevel.HIGH or evaluation.confidence_level == ConfidenceLevel.VERY_HIGH:
            application.forecasts["short_term"] = "Positive outcomes expected within 1-3 months"
        else:
            application.forecasts["short_term"] = "Uncertain outcomes, monitoring required"
        
        # Medium-term forecast
        if hypothesis.scope_score > 0.5:
            application.forecasts["medium_term"] = "Systematic improvements expected within 6-12 months"
        else:
            application.forecasts["medium_term"] = "Limited impact expected"
        
        # Long-term forecast
        if hypothesis.explanatory_power_score > 0.7:
            application.forecasts["long_term"] = "Fundamental understanding improvements expected"
        else:
            application.forecasts["long_term"] = "Incremental progress expected"
        
        # Contingency plans
        application.contingency_plans = []
        
        if evaluation.confidence_level == ConfidenceLevel.LOW or evaluation.confidence_level == ConfidenceLevel.VERY_LOW:
            application.contingency_plans.append("Prepare alternative explanations")
        
        if len(hypothesis.assumptions) > 3:
            application.contingency_plans.append("Develop plans for assumption failures")
        
        application.contingency_plans.extend([
            "Prepare for unexpected outcomes",
            "Maintain alternative action plans",
            "Establish early warning systems"
        ])
        
        return application
    
    async def _define_success_indicators(self, hypothesis: Hypothesis, evaluation: ExplanationEvaluation) -> List[str]:
        """Define success indicators"""
        
        indicators = []
        
        # Prediction-based indicators
        if hypothesis.predictions:
            indicators.append("Accurate prediction outcomes")
        
        # Testability-based indicators
        if hypothesis.testability_score > 0.5:
            indicators.append("Successful test results")
        
        # Scope-based indicators
        if hypothesis.scope_score > 0.5:
            indicators.append("Broad applicability confirmed")
        
        # Confidence-based indicators
        if evaluation.confidence_level == ConfidenceLevel.HIGH or evaluation.confidence_level == ConfidenceLevel.VERY_HIGH:
            indicators.append("High-confidence outcomes achieved")
        
        # General indicators
        indicators.extend([
            "Stakeholder satisfaction",
            "Problem resolution",
            "Knowledge advancement",
            "Practical impact"
        ])
        
        return indicators
    
    async def _assess_application_confidence(self, hypothesis: Hypothesis, evaluation: ExplanationEvaluation) -> float:
        """Assess confidence in the application"""
        
        confidence_factors = [
            hypothesis.confidence,
            evaluation.get_overall_fit_score(),
            hypothesis.testability_score,
            hypothesis.plausibility_score,
            evaluation.robustness_score
        ]
        
        # Weight factors
        weights = [0.3, 0.3, 0.2, 0.1, 0.1]
        
        confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
        
        # Adjust for uncertainty sources
        if hypothesis.uncertainty_sources:
            uncertainty_penalty = min(0.3, len(hypothesis.uncertainty_sources) * 0.05)
            confidence -= uncertainty_penalty
        
        # Adjust for assumptions
        if len(hypothesis.assumptions) > 3:
            assumption_penalty = min(0.2, (len(hypothesis.assumptions) - 3) * 0.02)
            confidence -= assumption_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _initialize_application_strategies(self) -> Dict[str, Any]:
        """Initialize application strategies"""
        return {
            "diagnostic": {"focus": "diagnosis and treatment"},
            "investigative": {"focus": "evidence and leads"},
            "troubleshooting": {"focus": "problem solving"},
            "research": {"focus": "knowledge advancement"},
            "strategic": {"focus": "strategic decisions"},
            "intervention": {"focus": "interventions and changes"},
            "pedagogical": {"focus": "learning and teaching"}
        }
    
    def _initialize_domain_adapters(self) -> Dict[str, Any]:
        """Initialize domain-specific adapters"""
        return {
            "medical": {"risk_tolerance": "low", "evidence_standards": "high"},
            "criminal": {"risk_tolerance": "very_low", "evidence_standards": "very_high"},
            "technical": {"risk_tolerance": "medium", "evidence_standards": "medium"},
            "scientific": {"risk_tolerance": "medium", "evidence_standards": "high"},
            "business": {"risk_tolerance": "medium", "evidence_standards": "medium"},
            "social": {"risk_tolerance": "medium", "evidence_standards": "medium"}
        }
    
    def _initialize_decision_frameworks(self) -> List[str]:
        """Initialize decision frameworks"""
        return [
            "confidence_based",
            "risk_based",
            "evidence_based",
            "stakeholder_based",
            "outcome_based"
        ]


class EnhancedAbductiveReasoningEngine:
    """Enhanced abductive reasoning engine with all elemental components"""
    
    def __init__(self):
        # Initialize all component engines
        self.phenomenon_engine = PhenomenonObservationEngine()
        self.hypothesis_engine = HypothesisGenerationEngine()
        self.selection_engine = ExplanationSelectionEngine()
        self.evaluation_engine = FitEvaluationEngine()
        self.application_engine = InferenceApplicationEngine()
        
        # Storage for reasoning processes
        self.reasoning_processes: List[AbductiveReasoning] = []
        
        # Configuration
        self.max_hypotheses = 20
        self.min_confidence_threshold = 0.3
        
        logger.info("Enhanced Abductive Reasoning Engine initialized")
    
    async def perform_abductive_reasoning(self, raw_observations: List[str], query: str, context: Dict[str, Any] = None) -> AbductiveReasoning:
        """Perform comprehensive abductive reasoning"""
        
        start_time = time.time()
        
        logger.info(f"Performing abductive reasoning: {query}")
        logger.info(f"Processing {len(raw_observations)} observations")
        
        # Component 1: Observation of Phenomena
        phenomena = await self.phenomenon_engine.observe_phenomena(raw_observations, context)
        
        # Component 2: Hypothesis Generation
        hypotheses = await self.hypothesis_engine.generate_hypotheses(phenomena, context)
        
        # Component 3: Selection of Best Explanation
        best_explanation = await self.selection_engine.select_best_explanation(hypotheses, phenomena, context)
        
        # Component 4: Evaluation of Fit
        evaluation = await self.evaluation_engine.evaluate_explanation_fit(best_explanation, phenomena, context)
        
        # Component 5: Inference Application
        inference_application = await self.application_engine.apply_inference(best_explanation, evaluation, context)
        
        # Create comprehensive reasoning result
        processing_time = time.time() - start_time
        
        # Calculate overall reasoning quality
        quality_factors = [
            sum(p.get_overall_score() for p in phenomena) / len(phenomena) if phenomena else 0,
            best_explanation.overall_score,
            evaluation.get_overall_fit_score(),
            inference_application.application_confidence
        ]
        
        reasoning_quality = sum(quality_factors) / len(quality_factors)
        
        # Determine confidence level
        confidence_level = evaluation.confidence_level
        
        # Generate conclusion
        conclusion = f"Best explanation: {best_explanation.statement}"
        
        # Get alternative explanations
        alternative_explanations = [h for h in hypotheses if h != best_explanation][:3]
        
        # Create comprehensive reasoning result
        reasoning = AbductiveReasoning(
            id=str(uuid4()),
            query=query,
            phenomena=phenomena,
            hypotheses=hypotheses,
            best_explanation=best_explanation,
            evaluation=evaluation,
            inference_application=inference_application,
            reasoning_quality=reasoning_quality,
            processing_time=processing_time,
            conclusion=conclusion,
            confidence_level=confidence_level,
            certainty_score=best_explanation.confidence,
            alternative_explanations=alternative_explanations
        )
        
        # Store reasoning process
        self.reasoning_processes.append(reasoning)
        
        logger.info(f"Abductive reasoning complete in {processing_time:.2f}s")
        logger.info(f"Quality: {reasoning_quality:.2f}, Confidence: {confidence_level.value}")
        
        return reasoning
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get comprehensive reasoning statistics"""
        
        if not self.reasoning_processes:
            return {"total_processes": 0}
        
        return {
            "total_processes": len(self.reasoning_processes),
            "average_quality": sum(r.reasoning_quality for r in self.reasoning_processes) / len(self.reasoning_processes),
            "average_processing_time": sum(r.processing_time for r in self.reasoning_processes) / len(self.reasoning_processes),
            "confidence_distribution": {
                level.value: sum(1 for r in self.reasoning_processes if r.confidence_level == level)
                for level in ConfidenceLevel
            },
            "phenomena_types": {
                ptype.value: sum(1 for r in self.reasoning_processes for p in r.phenomena if p.phenomenon_type == ptype)
                for ptype in PhenomenonType
            },
            "hypothesis_origins": {
                origin.value: sum(1 for r in self.reasoning_processes for h in r.hypotheses if h.origin == origin)
                for origin in HypothesisOrigin
            },
            "explanation_types": {
                etype.value: sum(1 for r in self.reasoning_processes if r.best_explanation.explanation_type == etype)
                for etype in ExplanationType
            },
            "average_phenomena_per_process": sum(len(r.phenomena) for r in self.reasoning_processes) / len(self.reasoning_processes),
            "average_hypotheses_per_process": sum(len(r.hypotheses) for r in self.reasoning_processes) / len(self.reasoning_processes)
        }