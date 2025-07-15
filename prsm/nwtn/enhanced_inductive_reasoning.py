#!/usr/bin/env python3
"""
Enhanced Inductive Reasoning Engine for NWTN
===========================================

This module implements a comprehensive inductive reasoning system based on
elemental components derived from empirical science and cognitive research.

The system follows the five elemental components of inductive reasoning:
1. Observation Collection
2. Pattern Recognition
3. Generalization
4. Evaluation of Scope and Exceptions
5. Predictive or Explanatory Inference

Key Features:
- Comprehensive observation collection with quality assessment
- Advanced pattern recognition with statistical validation
- Probabilistic generalization with uncertainty quantification
- Systematic evaluation of scope, exceptions, and limitations
- Predictive inference application with confidence intervals
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


class ObservationType(Enum):
    """Types of observations for inductive reasoning"""
    EMPIRICAL = "empirical"              # Direct sensory observations
    EXPERIMENTAL = "experimental"        # Controlled experimental data
    HISTORICAL = "historical"            # Historical records and data
    ANECDOTAL = "anecdotal"             # Anecdotal evidence
    STATISTICAL = "statistical"         # Statistical data points
    TESTIMONIAL = "testimonial"         # Witness testimony
    MEASUREMENT = "measurement"         # Quantitative measurements
    SURVEY = "survey"                   # Survey responses
    BEHAVIORAL = "behavioral"           # Behavioral observations
    NATURAL = "natural"                 # Natural phenomena observations


class PatternType(Enum):
    """Types of patterns in inductive reasoning"""
    SEQUENTIAL = "sequential"           # Temporal sequences
    FREQUENCY = "frequency"             # Frequency patterns
    CORRELATION = "correlation"         # Correlational patterns
    CAUSAL = "causal"                  # Causal patterns
    CYCLICAL = "cyclical"              # Cyclical patterns
    TREND = "trend"                    # Trend patterns
    CLUSTERING = "clustering"          # Clustering patterns
    DISTRIBUTION = "distribution"      # Distribution patterns
    THRESHOLD = "threshold"            # Threshold patterns
    CATEGORICAL = "categorical"        # Categorical patterns


class GeneralizationType(Enum):
    """Types of generalizations in inductive reasoning"""
    UNIVERSAL = "universal"             # Universal generalizations (All A are B)
    STATISTICAL = "statistical"        # Statistical generalizations (X% of A are B)
    PROBABILISTIC = "probabilistic"    # Probabilistic generalizations (A probably B)
    CONDITIONAL = "conditional"        # Conditional generalizations (If A then B)
    CAUSAL = "causal"                  # Causal generalizations (A causes B)
    PREDICTIVE = "predictive"          # Predictive generalizations (A will be B)
    EXPLANATORY = "explanatory"        # Explanatory generalizations (A explains B)
    NORMATIVE = "normative"            # Normative generalizations (A should be B)


class InductiveMethodType(Enum):
    """Methods of inductive reasoning"""
    ENUMERATION = "enumeration"         # Simple enumeration
    STATISTICAL_INFERENCE = "statistical_inference"  # Statistical inference
    MILL_METHODS = "mill_methods"      # Mill's methods
    BAYESIAN_INFERENCE = "bayesian_inference"  # Bayesian inference
    ELIMINATIVE_INDUCTION = "eliminative_induction"  # Eliminative induction
    ANALOGICAL_INDUCTION = "analogical_induction"  # Analogical induction
    ABDUCTIVE_INFERENCE = "abductive_inference"  # Abductive inference
    TREND_EXTRAPOLATION = "trend_extrapolation"  # Trend extrapolation


class ConfidenceLevel(Enum):
    """Confidence levels for inductive conclusions"""
    VERY_HIGH = "very_high"    # >90%
    HIGH = "high"              # 70-90%
    MODERATE = "moderate"      # 50-70%
    LOW = "low"                # 30-50%
    VERY_LOW = "very_low"      # <30%


@dataclass
class Observation:
    """Enhanced observation with comprehensive collection and validation"""
    
    # Core identification
    id: str
    content: str
    observation_type: ObservationType
    
    # Collection metadata
    source: str
    collection_method: str
    collection_timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Data extraction
    entities: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    measurements: Dict[str, float] = field(default_factory=dict)
    
    # Quality assessment
    reliability: float = 1.0
    precision: float = 1.0
    accuracy: float = 1.0
    completeness: float = 1.0
    
    # Sampling information
    sample_size: Optional[int] = None
    sample_bias: float = 0.0
    representativeness: float = 1.0
    
    # Temporal information
    observation_period: Optional[Tuple[datetime, datetime]] = None
    sequence_position: Optional[int] = None
    
    # Validation
    verified: bool = False
    verification_method: Optional[str] = None
    cross_validated: bool = False
    
    def __post_init__(self):
        """Post-processing after initialization"""
        if self.collection_timestamp is None:
            self.collection_timestamp = datetime.now(timezone.utc)
        
        # Calculate overall quality score
        self.quality_score = self._calculate_quality_score()
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score for the observation"""
        factors = [
            self.reliability,
            self.precision,
            self.accuracy,
            self.completeness,
            self.representativeness,
            (1.0 - self.sample_bias)
        ]
        return sum(factors) / len(factors)
    
    def get_quality_score(self) -> float:
        """Get overall quality score"""
        return self.quality_score


@dataclass
class Pattern:
    """Enhanced pattern with comprehensive recognition and validation"""
    
    # Core identification
    id: str
    pattern_type: PatternType
    description: str
    
    # Pattern structure
    pattern_elements: List[str]
    pattern_expression: str
    pattern_conditions: List[str] = field(default_factory=list)
    
    # Statistical properties
    frequency: int = 0
    support: float = 0.0
    confidence: float = 0.0
    lift: float = 0.0
    
    # Evidence base
    supporting_observations: List[Observation] = field(default_factory=list)
    contradicting_observations: List[Observation] = field(default_factory=list)
    
    # Statistical validation
    statistical_significance: float = 0.0
    p_value: float = 1.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    effect_size: float = 0.0
    
    # Robustness testing
    stability_score: float = 0.0
    cross_validation_score: float = 0.0
    
    # Temporal properties
    temporal_stability: float = 0.0
    trend_direction: Optional[str] = None
    
    # Scope and applicability
    domain_coverage: List[str] = field(default_factory=list)
    context_dependencies: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_strength_score(self) -> float:
        """Calculate overall pattern strength"""
        factors = [
            self.support,
            self.confidence,
            self.statistical_significance,
            self.stability_score,
            self.cross_validation_score
        ]
        return sum(factors) / len(factors)


@dataclass
class Generalization:
    """Enhanced generalization with probabilistic assessment"""
    
    # Core identification
    id: str
    generalization_type: GeneralizationType
    statement: str
    
    # Supporting pattern
    primary_pattern: Pattern
    supporting_patterns: List[Pattern] = field(default_factory=list)
    
    # Probabilistic properties
    probability: float = 0.5
    uncertainty: float = 0.5
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    
    # Scope definition
    scope_conditions: List[str] = field(default_factory=list)
    applicable_domains: List[str] = field(default_factory=list)
    population_coverage: float = 1.0
    
    # Evidence base
    total_observations: int = 0
    supporting_observations: int = 0
    contradicting_observations: int = 0
    
    # Validation properties
    cross_validation_accuracy: float = 0.0
    external_validation: bool = False
    replication_studies: int = 0
    
    # Temporal properties
    temporal_validity: float = 1.0
    time_horizon: Optional[str] = None
    
    # Revision tracking
    revision_count: int = 0
    last_revision: Optional[datetime] = None
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_reliability_score(self) -> float:
        """Calculate overall reliability of the generalization"""
        base_reliability = self.probability * (1.0 - self.uncertainty)
        
        # Adjust for validation
        if self.external_validation:
            base_reliability *= 1.2
        
        if self.cross_validation_accuracy > 0:
            base_reliability *= self.cross_validation_accuracy
        
        # Adjust for evidence base
        if self.total_observations > 0:
            evidence_factor = min(1.0, self.supporting_observations / self.total_observations)
            base_reliability *= evidence_factor
        
        return min(1.0, base_reliability)


@dataclass
class ScopeEvaluation:
    """Evaluation of scope and exceptions for inductive reasoning"""
    
    # Scope assessment
    scope_breadth: float = 0.0      # How broad is the scope
    scope_depth: float = 0.0        # How deep is the scope
    scope_precision: float = 0.0    # How precise is the scope
    
    # Exception analysis
    exceptions_found: List[str] = field(default_factory=list)
    exception_rate: float = 0.0
    exception_patterns: List[Pattern] = field(default_factory=list)
    
    # Limitation identification
    sample_limitations: List[str] = field(default_factory=list)
    methodological_limitations: List[str] = field(default_factory=list)
    domain_limitations: List[str] = field(default_factory=list)
    temporal_limitations: List[str] = field(default_factory=list)
    
    # Boundary conditions
    boundary_conditions: List[str] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)
    
    # Robustness assessment
    robustness_score: float = 0.0
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    scope_recommendations: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class InductiveInference:
    """Enhanced inductive inference with predictive and explanatory capabilities"""
    
    # Core identification
    id: str
    inference_type: str  # "predictive" or "explanatory"
    statement: str
    
    # Supporting generalization
    generalization: Generalization
    application_context: Dict[str, Any]
    
    # Predictive properties
    prediction_confidence: float = 0.0
    prediction_horizon: Optional[str] = None
    prediction_accuracy: float = 0.0
    
    # Explanatory properties
    explanatory_power: float = 0.0
    causal_strength: float = 0.0
    mechanism_clarity: float = 0.0
    
    # Uncertainty quantification
    epistemic_uncertainty: float = 0.0    # Model uncertainty
    aleatoric_uncertainty: float = 0.0    # Data uncertainty
    total_uncertainty: float = 0.0
    
    # Validation
    validation_tests: List[str] = field(default_factory=list)
    validation_results: Dict[str, float] = field(default_factory=dict)
    
    # Application tracking
    applications: int = 0
    success_rate: float = 0.0
    failure_modes: List[str] = field(default_factory=list)
    
    # Feedback incorporation
    feedback_received: List[str] = field(default_factory=list)
    adjustments_made: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_utility_score(self) -> float:
        """Calculate overall utility of the inference"""
        if self.inference_type == "predictive":
            return (self.prediction_confidence + self.prediction_accuracy) / 2
        else:  # explanatory
            return (self.explanatory_power + self.causal_strength) / 2


@dataclass
class InductiveReasoning:
    """Complete inductive reasoning process with all elemental components"""
    
    id: str
    query: str
    
    # Elemental components
    observation_collection: List[Observation]
    pattern_recognition: List[Pattern]
    generalization: Generalization
    scope_evaluation: ScopeEvaluation
    inference: InductiveInference
    
    # Process metadata
    method_used: InductiveMethodType
    conclusion: str
    confidence_level: ConfidenceLevel
    
    # Optional fields with defaults
    reasoning_quality: float = 0.0
    processing_time: float = 0.0
    probability: float = 0.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ObservationCollectionEngine:
    """Engine for comprehensive observation collection and validation"""
    
    def __init__(self):
        self.collection_methods = self._initialize_collection_methods()
        self.quality_standards = self._initialize_quality_standards()
        self.bias_detection = self._initialize_bias_detection()
    
    async def collect_observations(self, raw_data: List[str], context: Dict[str, Any] = None) -> List[Observation]:
        """Collect and validate observations from raw data"""
        
        logger.info(f"Collecting observations from {len(raw_data)} data points")
        
        observations = []
        
        for i, data_point in enumerate(raw_data):
            # Create observation
            observation = await self._create_observation(data_point, i, context)
            
            # Validate observation
            validated_observation = await self._validate_observation(observation, context)
            
            # Assess quality
            await self._assess_observation_quality(validated_observation)
            
            observations.append(validated_observation)
        
        # Assess collection quality
        await self._assess_collection_quality(observations)
        
        logger.info(f"Collected {len(observations)} validated observations")
        return observations
    
    async def _create_observation(self, data_point: str, index: int, context: Dict[str, Any]) -> Observation:
        """Create observation from data point"""
        
        # Determine observation type
        obs_type = await self._determine_observation_type(data_point, context)
        
        # Extract entities, attributes, and relationships
        entities = await self._extract_entities(data_point)
        attributes = await self._extract_attributes(data_point)
        relationships = await self._extract_relationships(data_point)
        measurements = await self._extract_measurements(data_point)
        
        # Determine source and collection method
        source = context.get("source", "unknown") if context else "unknown"
        collection_method = context.get("collection_method", "direct") if context else "direct"
        
        observation = Observation(
            id=f"obs_{index + 1}",
            content=data_point,
            observation_type=obs_type,
            source=source,
            collection_method=collection_method,
            collection_timestamp=datetime.now(timezone.utc),
            context=context or {},
            entities=entities,
            attributes=attributes,
            relationships=relationships,
            measurements=measurements,
            sequence_position=index
        )
        
        return observation
    
    async def _determine_observation_type(self, data_point: str, context: Dict[str, Any]) -> ObservationType:
        """Determine the type of observation"""
        
        # Check context hints
        if context:
            if "experiment" in context.get("method", "").lower():
                return ObservationType.EXPERIMENTAL
            elif "survey" in context.get("method", "").lower():
                return ObservationType.SURVEY
            elif "historical" in context.get("source", "").lower():
                return ObservationType.HISTORICAL
        
        # Content-based classification
        data_lower = data_point.lower()
        
        if any(word in data_lower for word in ["measured", "recorded", "observed"]):
            return ObservationType.MEASUREMENT
        elif any(word in data_lower for word in ["experiment", "trial", "test"]):
            return ObservationType.EXPERIMENTAL
        elif any(word in data_lower for word in ["witness", "testimony", "reported"]):
            return ObservationType.TESTIMONIAL
        elif any(word in data_lower for word in ["behavior", "action", "response"]):
            return ObservationType.BEHAVIORAL
        elif any(word in data_lower for word in ["natural", "phenomenon", "occurs"]):
            return ObservationType.NATURAL
        elif re.search(r'\d+(\.\d+)?', data_point):
            return ObservationType.STATISTICAL
        else:
            return ObservationType.EMPIRICAL
    
    async def _extract_entities(self, data_point: str) -> List[str]:
        """Extract entities from observation"""
        
        entities = []
        
        # Extract proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', data_point)
        entities.extend(proper_nouns)
        
        # Extract common entities
        entity_patterns = [
            r'\b(?:the|a|an)\s+([a-z]+(?:\s+[a-z]+)?)\b',
            r'\b([a-z]+)\s+(?:was|were|is|are)\b',
            r'\b([a-z]+)\s+(?:showed|demonstrated|exhibited)\b'
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, data_point.lower())
            entities.extend(matches)
        
        # Clean and deduplicate
        entities = list(set([e.strip() for e in entities if len(e.strip()) > 2]))
        
        return entities[:10]  # Limit to prevent noise
    
    async def _extract_attributes(self, data_point: str) -> Dict[str, Any]:
        """Extract attributes and properties"""
        
        attributes = {}
        
        # Extract qualitative attributes
        quality_patterns = [
            r'(?:is|was|became|seems?)\s+(very\s+)?(\w+)',
            r'(\w+)\s+(?:quality|property|characteristic)',
            r'(?:appears?|looks?)\s+(very\s+)?(\w+)'
        ]
        
        for pattern in quality_patterns:
            matches = re.findall(pattern, data_point.lower())
            for match in matches:
                if isinstance(match, tuple):
                    intensity, quality = match
                    attributes[f"quality_{quality}"] = "high" if intensity else "normal"
                else:
                    attributes[f"quality_{match}"] = "present"
        
        # Extract quantitative attributes
        number_patterns = [
            r'(\d+\.?\d*)\s*(percent|%|degrees?|units?|mg|kg|cm|mm|m|km)',
            r'(\d+\.?\d*)\s*(seconds?|minutes?|hours?|days?|years?)',
            r'(\d+\.?\d*)\s*(times?|cases?|instances?|occurrences?)'
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, data_point.lower())
            for value, unit in matches:
                attributes[f"measurement_{unit}"] = float(value)
        
        return attributes
    
    async def _extract_relationships(self, data_point: str) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities"""
        
        relationships = []
        
        # Relationship patterns
        rel_patterns = [
            r'(\w+)\s+(causes?|leads?\s+to|results?\s+in)\s+(\w+)',
            r'(\w+)\s+(increases?|decreases?|affects?|influences?)\s+(\w+)',
            r'(\w+)\s+(correlates?\s+with|is\s+related\s+to)\s+(\w+)',
            r'(\w+)\s+(follows?|precedes?|comes?\s+after)\s+(\w+)',
            r'(\w+)\s+(depends?\s+on|requires?)\s+(\w+)'
        ]
        
        for pattern in rel_patterns:
            matches = re.findall(pattern, data_point.lower())
            for entity1, relation, entity2 in matches:
                relationships.append((entity1.strip(), relation.strip(), entity2.strip()))
        
        return relationships
    
    async def _extract_measurements(self, data_point: str) -> Dict[str, float]:
        """Extract numerical measurements"""
        
        measurements = {}
        
        # Extract numerical values with units
        measurement_patterns = [
            r'(\w+):\s*(\d+\.?\d*)\s*([a-z%]+)',
            r'(\w+)\s+of\s+(\d+\.?\d*)\s*([a-z%]+)',
            r'(\w+)\s+was\s+(\d+\.?\d*)\s*([a-z%]+)'
        ]
        
        for pattern in measurement_patterns:
            matches = re.findall(pattern, data_point.lower())
            for measure, value, unit in matches:
                measurements[f"{measure}_{unit}"] = float(value)
        
        return measurements
    
    async def _validate_observation(self, observation: Observation, context: Dict[str, Any]) -> Observation:
        """Validate observation for quality and reliability"""
        
        # Reliability assessment
        observation.reliability = await self._assess_reliability(observation, context)
        
        # Precision assessment
        observation.precision = await self._assess_precision(observation)
        
        # Accuracy assessment
        observation.accuracy = await self._assess_accuracy(observation, context)
        
        # Completeness assessment
        observation.completeness = await self._assess_completeness(observation)
        
        # Representativeness assessment
        observation.representativeness = await self._assess_representativeness(observation, context)
        
        # Bias assessment
        observation.sample_bias = await self._assess_bias(observation, context)
        
        return observation
    
    async def _assess_reliability(self, observation: Observation, context: Dict[str, Any]) -> float:
        """Assess reliability of observation"""
        
        base_reliability = 0.8
        
        # Adjust based on source
        if context and "source_reliability" in context:
            base_reliability *= context["source_reliability"]
        
        # Adjust based on observation type
        type_reliability = {
            ObservationType.EXPERIMENTAL: 0.9,
            ObservationType.MEASUREMENT: 0.85,
            ObservationType.EMPIRICAL: 0.8,
            ObservationType.STATISTICAL: 0.85,
            ObservationType.ANECDOTAL: 0.4,
            ObservationType.TESTIMONIAL: 0.6
        }
        
        base_reliability *= type_reliability.get(observation.observation_type, 0.7)
        
        # Adjust based on content quality
        if observation.content:
            content_lower = observation.content.lower()
            
            # Certainty indicators increase reliability
            certainty_words = ["definitely", "clearly", "obviously", "certainly", "measured"]
            certainty_score = sum(1 for word in certainty_words if word in content_lower)
            base_reliability += certainty_score * 0.05
            
            # Uncertainty indicators decrease reliability
            uncertainty_words = ["maybe", "possibly", "perhaps", "might", "could"]
            uncertainty_score = sum(1 for word in uncertainty_words if word in content_lower)
            base_reliability -= uncertainty_score * 0.1
        
        return max(0.1, min(1.0, base_reliability))
    
    async def _assess_precision(self, observation: Observation) -> float:
        """Assess precision of observation"""
        
        # Base precision
        precision = 0.7
        
        # Increase precision for quantitative observations
        if observation.measurements:
            precision = 0.9
        
        # Increase precision for specific observations
        if observation.attributes:
            precision += 0.1
        
        # Increase precision for structured observations
        if observation.relationships:
            precision += 0.1
        
        return min(1.0, precision)
    
    async def _assess_accuracy(self, observation: Observation, context: Dict[str, Any]) -> float:
        """Assess accuracy of observation"""
        
        # Base accuracy
        accuracy = 0.8
        
        # Adjust based on verification
        if observation.verified:
            accuracy = 0.95
        
        # Adjust based on cross-validation
        if observation.cross_validated:
            accuracy = min(1.0, accuracy + 0.1)
        
        return accuracy
    
    async def _assess_completeness(self, observation: Observation) -> float:
        """Assess completeness of observation"""
        
        completeness_factors = []
        
        # Content completeness
        if observation.content and len(observation.content) > 10:
            completeness_factors.append(0.8)
        else:
            completeness_factors.append(0.3)
        
        # Entity completeness
        if observation.entities:
            completeness_factors.append(0.9)
        else:
            completeness_factors.append(0.5)
        
        # Attribute completeness
        if observation.attributes:
            completeness_factors.append(0.9)
        else:
            completeness_factors.append(0.6)
        
        # Relationship completeness
        if observation.relationships:
            completeness_factors.append(0.9)
        else:
            completeness_factors.append(0.7)
        
        return sum(completeness_factors) / len(completeness_factors)
    
    async def _assess_representativeness(self, observation: Observation, context: Dict[str, Any]) -> float:
        """Assess representativeness of observation"""
        
        # Base representativeness
        representativeness = 0.7
        
        # Adjust based on sample size
        if observation.sample_size:
            if observation.sample_size >= 100:
                representativeness = 0.9
            elif observation.sample_size >= 30:
                representativeness = 0.8
            elif observation.sample_size >= 10:
                representativeness = 0.7
            else:
                representativeness = 0.5
        
        # Adjust based on context
        if context and "population_coverage" in context:
            representativeness *= context["population_coverage"]
        
        return representativeness
    
    async def _assess_bias(self, observation: Observation, context: Dict[str, Any]) -> float:
        """Assess bias in observation"""
        
        bias_score = 0.0
        
        # Selection bias
        if context and "selection_method" in context:
            if context["selection_method"] == "random":
                bias_score += 0.0
            elif context["selection_method"] == "convenience":
                bias_score += 0.3
            elif context["selection_method"] == "purposive":
                bias_score += 0.2
        
        # Confirmation bias indicators
        if observation.content:
            bias_words = ["obviously", "clearly", "as expected", "predictably"]
            bias_count = sum(1 for word in bias_words if word in observation.content.lower())
            bias_score += bias_count * 0.1
        
        return min(1.0, bias_score)
    
    async def _assess_observation_quality(self, observation: Observation):
        """Assess overall observation quality"""
        
        # Quality is already calculated in __post_init__
        pass
    
    async def _assess_collection_quality(self, observations: List[Observation]):
        """Assess quality of the observation collection"""
        
        if not observations:
            return
        
        # Log collection quality metrics
        avg_quality = sum(obs.get_quality_score() for obs in observations) / len(observations)
        avg_reliability = sum(obs.reliability for obs in observations) / len(observations)
        avg_completeness = sum(obs.completeness for obs in observations) / len(observations)
        
        logger.info(f"Collection quality assessment: avg_quality={avg_quality:.2f}, avg_reliability={avg_reliability:.2f}, avg_completeness={avg_completeness:.2f}")
    
    def _initialize_collection_methods(self) -> Dict[str, Any]:
        """Initialize collection methods"""
        
        return {
            "direct": {"reliability": 0.8, "speed": 0.9},
            "survey": {"reliability": 0.7, "speed": 0.8},
            "experiment": {"reliability": 0.9, "speed": 0.5},
            "historical": {"reliability": 0.6, "speed": 0.9},
            "measurement": {"reliability": 0.9, "speed": 0.7}
        }
    
    def _initialize_quality_standards(self) -> Dict[str, float]:
        """Initialize quality standards"""
        
        return {
            "min_reliability": 0.5,
            "min_precision": 0.6,
            "min_accuracy": 0.7,
            "min_completeness": 0.5,
            "max_bias": 0.3
        }
    
    def _initialize_bias_detection(self) -> Dict[str, List[str]]:
        """Initialize bias detection patterns"""
        
        return {
            "confirmation_bias": ["obviously", "clearly", "as expected", "predictably"],
            "selection_bias": ["cherry-picked", "selected", "chosen"],
            "reporting_bias": ["significant", "important", "notable"]
        }


class PatternRecognitionEngine:
    """Engine for advanced pattern recognition with statistical validation"""
    
    def __init__(self):
        self.pattern_detectors = self._initialize_pattern_detectors()
        self.statistical_tests = self._initialize_statistical_tests()
        self.validation_methods = self._initialize_validation_methods()
    
    async def recognize_patterns(self, observations: List[Observation]) -> List[Pattern]:
        """Recognize patterns in observations with statistical validation"""
        
        logger.info(f"Recognizing patterns in {len(observations)} observations")
        
        patterns = []
        
        # Sequential patterns
        sequential_patterns = await self._detect_sequential_patterns(observations)
        patterns.extend(sequential_patterns)
        
        # Frequency patterns
        frequency_patterns = await self._detect_frequency_patterns(observations)
        patterns.extend(frequency_patterns)
        
        # Correlation patterns
        correlation_patterns = await self._detect_correlation_patterns(observations)
        patterns.extend(correlation_patterns)
        
        # Trend patterns
        trend_patterns = await self._detect_trend_patterns(observations)
        patterns.extend(trend_patterns)
        
        # Clustering patterns
        clustering_patterns = await self._detect_clustering_patterns(observations)
        patterns.extend(clustering_patterns)
        
        # Cyclical patterns
        cyclical_patterns = await self._detect_cyclical_patterns(observations)
        patterns.extend(cyclical_patterns)
        
        # Validate patterns
        validated_patterns = await self._validate_patterns(patterns, observations)
        
        logger.info(f"Recognized {len(validated_patterns)} validated patterns")
        return validated_patterns
    
    async def _detect_sequential_patterns(self, observations: List[Observation]) -> List[Pattern]:
        """Detect sequential patterns in observations"""
        
        patterns = []
        
        # Sort observations by sequence position
        sorted_obs = sorted([obs for obs in observations if obs.sequence_position is not None], 
                          key=lambda x: x.sequence_position)
        
        if len(sorted_obs) < 3:
            return patterns
        
        # Extract entity sequences
        entity_sequences = []
        for obs in sorted_obs:
            entity_sequences.extend(obs.entities)
        
        # Find repeating subsequences
        for length in range(2, min(6, len(entity_sequences))):
            subsequence_counts = Counter()
            
            for i in range(len(entity_sequences) - length + 1):
                subsequence = tuple(entity_sequences[i:i+length])
                subsequence_counts[subsequence] += 1
            
            # Create patterns for frequent subsequences
            for subsequence, count in subsequence_counts.items():
                if count >= 2:
                    pattern = Pattern(
                        id=f"seq_pattern_{len(patterns)+1}",
                        pattern_type=PatternType.SEQUENTIAL,
                        description=f"Sequential pattern: {' → '.join(subsequence)}",
                        pattern_elements=list(subsequence),
                        pattern_expression=f"seq({', '.join(subsequence)})",
                        frequency=count,
                        support=count / len(entity_sequences),
                        supporting_observations=[obs for obs in sorted_obs 
                                              if any(entity in obs.entities for entity in subsequence)]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_frequency_patterns(self, observations: List[Observation]) -> List[Pattern]:
        """Detect frequency patterns in observations"""
        
        patterns = []
        
        # Entity frequency patterns
        entity_counts = Counter()
        for obs in observations:
            entity_counts.update(obs.entities)
        
        total_entities = sum(entity_counts.values())
        
        # Create patterns for frequent entities
        for entity, count in entity_counts.items():
            if count >= 2:
                support = count / total_entities
                pattern = Pattern(
                    id=f"freq_pattern_{len(patterns)+1}",
                    pattern_type=PatternType.FREQUENCY,
                    description=f"Frequent entity: {entity} appears {count} times",
                    pattern_elements=[entity],
                    pattern_expression=f"freq({entity}) = {count}",
                    frequency=count,
                    support=support,
                    supporting_observations=[obs for obs in observations if entity in obs.entities]
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_correlation_patterns(self, observations: List[Observation]) -> List[Pattern]:
        """Detect correlation patterns between entities and attributes"""
        
        patterns = []
        
        # Entity-attribute correlations
        entity_attribute_pairs = []
        for obs in observations:
            for entity in obs.entities:
                for attr_name, attr_value in obs.attributes.items():
                    entity_attribute_pairs.append((entity, attr_name, attr_value))
        
        # Group by entity-attribute pairs
        pair_groups = defaultdict(list)
        for entity, attr_name, attr_value in entity_attribute_pairs:
            pair_groups[(entity, attr_name)].append(attr_value)
        
        # Find correlations
        for (entity, attr_name), values in pair_groups.items():
            if len(values) >= 2:
                # Check for consistent values
                if len(set(str(v) for v in values)) == 1:
                    # Perfect correlation
                    pattern = Pattern(
                        id=f"corr_pattern_{len(patterns)+1}",
                        pattern_type=PatternType.CORRELATION,
                        description=f"Correlation: {entity} consistently has {attr_name} = {values[0]}",
                        pattern_elements=[entity, attr_name],
                        pattern_expression=f"corr({entity}, {attr_name}) = {values[0]}",
                        frequency=len(values),
                        support=len(values) / len(observations),
                        confidence=1.0,
                        supporting_observations=[obs for obs in observations 
                                              if entity in obs.entities and attr_name in obs.attributes]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_trend_patterns(self, observations: List[Observation]) -> List[Pattern]:
        """Detect trend patterns in numerical data"""
        
        patterns = []
        
        # Extract numerical measurements over time
        numerical_series = defaultdict(list)
        for obs in observations:
            for measure_name, measure_value in obs.measurements.items():
                numerical_series[measure_name].append((obs.sequence_position or 0, measure_value))
        
        # Analyze trends
        for measure_name, series in numerical_series.items():
            if len(series) >= 3:
                # Sort by position
                series.sort(key=lambda x: x[0])
                values = [v for _, v in series]
                
                # Simple trend detection
                if len(values) >= 3:
                    increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
                    decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
                    
                    if increasing or decreasing:
                        trend_direction = "increasing" if increasing else "decreasing"
                        
                        pattern = Pattern(
                            id=f"trend_pattern_{len(patterns)+1}",
                            pattern_type=PatternType.TREND,
                            description=f"Trend pattern: {measure_name} is {trend_direction}",
                            pattern_elements=[measure_name],
                            pattern_expression=f"trend({measure_name}) = {trend_direction}",
                            frequency=len(values),
                            support=len(values) / len(observations),
                            confidence=1.0,
                            trend_direction=trend_direction,
                            supporting_observations=[obs for obs in observations 
                                                  if measure_name in obs.measurements]
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_clustering_patterns(self, observations: List[Observation]) -> List[Pattern]:
        """Detect clustering patterns in data"""
        
        patterns = []
        
        # Analyze numerical clustering
        numerical_data = defaultdict(list)
        for obs in observations:
            for measure_name, measure_value in obs.measurements.items():
                numerical_data[measure_name].append(measure_value)
        
        # Find clusters
        for measure_name, values in numerical_data.items():
            if len(values) >= 3:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                if std_val > 0:
                    # Values within one standard deviation
                    within_std = [v for v in values if abs(v - mean_val) <= std_val]
                    
                    if len(within_std) / len(values) >= 0.8:
                        pattern = Pattern(
                            id=f"cluster_pattern_{len(patterns)+1}",
                            pattern_type=PatternType.CLUSTERING,
                            description=f"Clustering pattern: {measure_name} clusters around {mean_val:.2f}",
                            pattern_elements=[measure_name],
                            pattern_expression=f"cluster({measure_name}) ~ {mean_val:.2f} ± {std_val:.2f}",
                            frequency=len(within_std),
                            support=len(within_std) / len(values),
                            confidence=len(within_std) / len(values),
                            supporting_observations=[obs for obs in observations 
                                                  if measure_name in obs.measurements]
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_cyclical_patterns(self, observations: List[Observation]) -> List[Pattern]:
        """Detect cyclical patterns in temporal data"""
        
        patterns = []
        
        # Simple cyclical detection (would be more sophisticated in practice)
        # Look for repeating patterns in entity sequences
        
        if len(observations) < 4:
            return patterns
        
        # Extract entity sequences
        entity_sequence = []
        for obs in sorted(observations, key=lambda x: x.sequence_position or 0):
            entity_sequence.extend(obs.entities)
        
        # Look for repeating cycles
        for cycle_length in range(2, min(6, len(entity_sequence) // 2)):
            cycles_found = 0
            
            for start in range(len(entity_sequence) - cycle_length * 2):
                cycle1 = entity_sequence[start:start + cycle_length]
                cycle2 = entity_sequence[start + cycle_length:start + cycle_length * 2]
                
                if cycle1 == cycle2:
                    cycles_found += 1
            
            if cycles_found >= 2:
                pattern = Pattern(
                    id=f"cycle_pattern_{len(patterns)+1}",
                    pattern_type=PatternType.CYCLICAL,
                    description=f"Cyclical pattern: cycle of length {cycle_length} repeats",
                    pattern_elements=[f"cycle_{cycle_length}"],
                    pattern_expression=f"cycle(length={cycle_length})",
                    frequency=cycles_found,
                    support=cycles_found / len(observations),
                    supporting_observations=observations
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _validate_patterns(self, patterns: List[Pattern], observations: List[Observation]) -> List[Pattern]:
        """Validate patterns using statistical tests"""
        
        validated_patterns = []
        
        for pattern in patterns:
            # Calculate confidence
            pattern.confidence = await self._calculate_confidence(pattern, observations)
            
            # Calculate statistical significance
            pattern.statistical_significance = await self._calculate_statistical_significance(pattern, observations)
            
            # Calculate p-value
            pattern.p_value = await self._calculate_p_value(pattern, observations)
            
            # Calculate confidence interval
            pattern.confidence_interval = await self._calculate_confidence_interval(pattern, observations)
            
            # Calculate effect size
            pattern.effect_size = await self._calculate_effect_size(pattern, observations)
            
            # Perform cross-validation
            pattern.cross_validation_score = await self._perform_cross_validation(pattern, observations)
            
            # Check if pattern meets validation criteria
            if (pattern.support >= 0.1 and 
                pattern.confidence >= 0.5 and 
                pattern.statistical_significance >= 0.05):
                
                validated_patterns.append(pattern)
        
        return validated_patterns
    
    async def _calculate_confidence(self, pattern: Pattern, observations: List[Observation]) -> float:
        """Calculate confidence in pattern"""
        
        # Base confidence on support
        base_confidence = pattern.support
        
        # Adjust for observation quality
        if pattern.supporting_observations:
            avg_quality = sum(obs.get_quality_score() for obs in pattern.supporting_observations) / len(pattern.supporting_observations)
            base_confidence *= avg_quality
        
        # Adjust for contradicting evidence
        if pattern.contradicting_observations:
            contradiction_penalty = len(pattern.contradicting_observations) / len(observations)
            base_confidence *= (1 - contradiction_penalty)
        
        return max(0.0, min(1.0, base_confidence))
    
    async def _calculate_statistical_significance(self, pattern: Pattern, observations: List[Observation]) -> float:
        """Calculate statistical significance of pattern"""
        
        # Simple significance based on frequency vs expected
        expected_frequency = 1.0 / len(observations)
        observed_frequency = pattern.frequency / len(observations)
        
        if expected_frequency > 0:
            significance = min(1.0, observed_frequency / expected_frequency)
        else:
            significance = 1.0
        
        return significance
    
    async def _calculate_p_value(self, pattern: Pattern, observations: List[Observation]) -> float:
        """Calculate p-value for pattern"""
        
        # Simplified p-value calculation
        # In practice, would use appropriate statistical tests
        
        # Based on frequency and sample size
        n = len(observations)
        observed = pattern.frequency
        expected = n * 0.1  # Assume 10% expected frequency
        
        if expected > 0:
            # Simple chi-square like calculation
            chi_square = ((observed - expected) ** 2) / expected
            p_value = max(0.001, 1.0 / (1.0 + chi_square))
        else:
            p_value = 1.0
        
        return p_value
    
    async def _calculate_confidence_interval(self, pattern: Pattern, observations: List[Observation]) -> Tuple[float, float]:
        """Calculate confidence interval for pattern"""
        
        # Simple confidence interval calculation
        p = pattern.support
        n = len(observations)
        
        if n > 0:
            # Wilson score interval (simplified)
            z = 1.96  # 95% confidence
            center = p + z**2 / (2*n)
            spread = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))
            denominator = 1 + z**2/n
            
            lower = max(0.0, (center - spread) / denominator)
            upper = min(1.0, (center + spread) / denominator)
        else:
            lower, upper = 0.0, 1.0
        
        return (lower, upper)
    
    async def _calculate_effect_size(self, pattern: Pattern, observations: List[Observation]) -> float:
        """Calculate effect size for pattern"""
        
        # Simple effect size based on frequency
        effect_size = pattern.frequency / len(observations)
        
        return effect_size
    
    async def _perform_cross_validation(self, pattern: Pattern, observations: List[Observation]) -> float:
        """Perform cross-validation on pattern"""
        
        # Simple cross-validation
        if len(observations) < 5:
            return 0.5
        
        # Split observations
        split_point = len(observations) // 2
        train_obs = observations[:split_point]
        test_obs = observations[split_point:]
        
        # Count pattern occurrences in test set
        test_support = 0
        for obs in test_obs:
            if any(element in obs.entities for element in pattern.pattern_elements):
                test_support += 1
        
        # Calculate validation score
        validation_score = test_support / len(test_obs) if test_obs else 0.0
        
        return validation_score
    
    def _initialize_pattern_detectors(self) -> Dict[str, Any]:
        """Initialize pattern detection methods"""
        
        return {
            PatternType.SEQUENTIAL: {"min_length": 2, "min_frequency": 2},
            PatternType.FREQUENCY: {"min_frequency": 2, "min_support": 0.1},
            PatternType.CORRELATION: {"min_observations": 3, "min_correlation": 0.5},
            PatternType.TREND: {"min_points": 3, "min_consistency": 0.8},
            PatternType.CLUSTERING: {"min_cluster_size": 0.8, "max_std_dev": 1.0},
            PatternType.CYCLICAL: {"min_cycle_length": 2, "min_repetitions": 2}
        }
    
    def _initialize_statistical_tests(self) -> Dict[str, Any]:
        """Initialize statistical tests"""
        
        return {
            "chi_square": {"min_expected": 5},
            "t_test": {"min_sample_size": 10},
            "correlation": {"min_sample_size": 10},
            "regression": {"min_sample_size": 20}
        }
    
    def _initialize_validation_methods(self) -> Dict[str, Any]:
        """Initialize validation methods"""
        
        return {
            "cross_validation": {"folds": 5, "min_sample_size": 10},
            "bootstrap": {"iterations": 1000, "confidence_level": 0.95},
            "permutation": {"iterations": 10000, "alpha": 0.05}
        }


class GeneralizationEngine:
    """Engine for creating probabilistic generalizations from patterns"""
    
    def __init__(self):
        self.generalization_strategies = self._initialize_generalization_strategies()
        self.uncertainty_quantification = self._initialize_uncertainty_quantification()
    
    async def create_generalization(self, patterns: List[Pattern], observations: List[Observation]) -> Generalization:
        """Create generalization from identified patterns"""
        
        logger.info(f"Creating generalization from {len(patterns)} patterns")
        
        if not patterns:
            return await self._create_empty_generalization(observations)
        
        # Select primary pattern
        primary_pattern = max(patterns, key=lambda p: p.get_strength_score())
        
        # Determine generalization type
        gen_type = await self._determine_generalization_type(primary_pattern, observations)
        
        # Generate generalization statement
        statement = await self._generate_generalization_statement(primary_pattern, patterns, gen_type)
        
        # Calculate probability and uncertainty
        probability = await self._calculate_probability(primary_pattern, patterns, observations)
        uncertainty = await self._calculate_uncertainty(primary_pattern, patterns, observations)
        
        # Calculate confidence interval
        confidence_interval = await self._calculate_confidence_interval(primary_pattern, observations)
        
        # Determine scope
        scope_conditions = await self._determine_scope_conditions(primary_pattern, patterns, observations)
        applicable_domains = await self._determine_applicable_domains(primary_pattern, observations)
        
        # Calculate evidence counts
        total_obs = len(observations)
        supporting_obs = len(primary_pattern.supporting_observations)
        contradicting_obs = len(primary_pattern.contradicting_observations)
        
        # Create generalization
        generalization = Generalization(
            id=str(uuid4()),
            generalization_type=gen_type,
            statement=statement,
            primary_pattern=primary_pattern,
            supporting_patterns=[p for p in patterns if p != primary_pattern],
            probability=probability,
            uncertainty=uncertainty,
            confidence_interval=confidence_interval,
            scope_conditions=scope_conditions,
            applicable_domains=applicable_domains,
            total_observations=total_obs,
            supporting_observations=supporting_obs,
            contradicting_observations=contradicting_obs
        )
        
        # Validate generalization
        validated_generalization = await self._validate_generalization(generalization, observations)
        
        logger.info(f"Created generalization with probability {probability:.2f}")
        return validated_generalization
    
    async def _create_empty_generalization(self, observations: List[Observation]) -> Generalization:
        """Create empty generalization when no patterns found"""
        
        empty_pattern = Pattern(
            id="empty_pattern",
            pattern_type=PatternType.FREQUENCY,
            description="No significant pattern found",
            pattern_elements=[],
            pattern_expression="none"
        )
        
        return Generalization(
            id=str(uuid4()),
            generalization_type=GeneralizationType.PROBABILISTIC,
            statement="No significant patterns found in the observations",
            primary_pattern=empty_pattern,
            probability=0.1,
            uncertainty=0.9,
            confidence_interval=(0.0, 0.2),
            total_observations=len(observations),
            supporting_observations=0,
            contradicting_observations=0
        )
    
    async def _determine_generalization_type(self, pattern: Pattern, observations: List[Observation]) -> GeneralizationType:
        """Determine appropriate generalization type"""
        
        # Based on pattern type and characteristics
        if pattern.pattern_type == PatternType.FREQUENCY:
            if pattern.confidence >= 0.95:
                return GeneralizationType.UNIVERSAL
            else:
                return GeneralizationType.STATISTICAL
        
        elif pattern.pattern_type == PatternType.CORRELATION:
            return GeneralizationType.CONDITIONAL
        
        elif pattern.pattern_type == PatternType.CAUSAL:
            return GeneralizationType.CAUSAL
        
        elif pattern.pattern_type == PatternType.TREND:
            return GeneralizationType.PREDICTIVE
        
        elif pattern.pattern_type == PatternType.SEQUENTIAL:
            return GeneralizationType.PREDICTIVE
        
        else:
            return GeneralizationType.PROBABILISTIC
    
    async def _generate_generalization_statement(self, primary_pattern: Pattern, patterns: List[Pattern], gen_type: GeneralizationType) -> str:
        """Generate generalization statement"""
        
        base_statement = primary_pattern.description
        
        # Modify based on generalization type
        if gen_type == GeneralizationType.UNIVERSAL:
            statement = f"All instances show that {base_statement}"
        
        elif gen_type == GeneralizationType.STATISTICAL:
            percentage = int(primary_pattern.confidence * 100)
            statement = f"Approximately {percentage}% of instances show that {base_statement}"
        
        elif gen_type == GeneralizationType.PROBABILISTIC:
            if primary_pattern.confidence >= 0.8:
                qualifier = "very likely"
            elif primary_pattern.confidence >= 0.6:
                qualifier = "likely"
            else:
                qualifier = "possibly"
            statement = f"It is {qualifier} that {base_statement}"
        
        elif gen_type == GeneralizationType.CONDITIONAL:
            statement = f"When certain conditions are met, {base_statement}"
        
        elif gen_type == GeneralizationType.CAUSAL:
            statement = f"There is evidence that {base_statement} represents a causal relationship"
        
        elif gen_type == GeneralizationType.PREDICTIVE:
            statement = f"Based on observed patterns, we can predict that {base_statement}"
        
        elif gen_type == GeneralizationType.EXPLANATORY:
            statement = f"The pattern suggests that {base_statement} explains the observed phenomena"
        
        else:
            statement = f"The evidence suggests that {base_statement}"
        
        # Add supporting information
        if len(patterns) > 1:
            statement += f" This conclusion is supported by {len(patterns)} consistent patterns."
        
        return statement
    
    async def _calculate_probability(self, primary_pattern: Pattern, patterns: List[Pattern], observations: List[Observation]) -> float:
        """Calculate probability of generalization"""
        
        # Base probability on primary pattern
        base_probability = primary_pattern.confidence
        
        # Adjust for supporting patterns
        if len(patterns) > 1:
            supporting_boost = min(0.2, (len(patterns) - 1) * 0.05)
            base_probability += supporting_boost
        
        # Adjust for observation quality
        if primary_pattern.supporting_observations:
            avg_quality = sum(obs.get_quality_score() for obs in primary_pattern.supporting_observations) / len(primary_pattern.supporting_observations)
            base_probability *= avg_quality
        
        # Adjust for sample size
        sample_size_factor = min(1.0, len(observations) / 20)  # Diminishing returns
        base_probability *= (0.5 + 0.5 * sample_size_factor)
        
        return max(0.0, min(1.0, base_probability))
    
    async def _calculate_uncertainty(self, primary_pattern: Pattern, patterns: List[Pattern], observations: List[Observation]) -> float:
        """Calculate uncertainty in generalization"""
        
        # Base uncertainty on pattern confidence
        base_uncertainty = 1.0 - primary_pattern.confidence
        
        # Adjust for contradicting evidence
        if primary_pattern.contradicting_observations:
            contradiction_factor = len(primary_pattern.contradicting_observations) / len(observations)
            base_uncertainty += contradiction_factor * 0.3
        
        # Adjust for sample size
        if len(observations) < 10:
            base_uncertainty += 0.2
        elif len(observations) < 5:
            base_uncertainty += 0.4
        
        # Adjust for observation quality variance
        if primary_pattern.supporting_observations:
            qualities = [obs.get_quality_score() for obs in primary_pattern.supporting_observations]
            if len(qualities) > 1:
                quality_variance = statistics.variance(qualities)
                base_uncertainty += quality_variance * 0.1
        
        return max(0.0, min(1.0, base_uncertainty))
    
    async def _calculate_confidence_interval(self, pattern: Pattern, observations: List[Observation]) -> Tuple[float, float]:
        """Calculate confidence interval for generalization"""
        
        # Use pattern's confidence interval
        return pattern.confidence_interval
    
    async def _determine_scope_conditions(self, primary_pattern: Pattern, patterns: List[Pattern], observations: List[Observation]) -> List[str]:
        """Determine conditions that define the scope of generalization"""
        
        conditions = []
        
        # Context conditions
        contexts = set()
        for obs in primary_pattern.supporting_observations:
            if obs.context:
                for key, value in obs.context.items():
                    contexts.add(f"{key}={value}")
        
        if contexts:
            conditions.append(f"Context conditions: {', '.join(list(contexts)[:3])}")
        
        # Domain conditions
        domains = set(obs.attributes.get("domain", "general") for obs in primary_pattern.supporting_observations)
        if len(domains) > 1:
            conditions.append(f"Applicable across domains: {', '.join(domains)}")
        elif len(domains) == 1:
            conditions.append(f"Limited to domain: {list(domains)[0]}")
        
        # Sample size conditions
        if len(observations) < 10:
            conditions.append("Limited sample size - generalization may not hold for larger populations")
        
        # Temporal conditions
        if primary_pattern.temporal_stability and primary_pattern.temporal_stability < 0.7:
            conditions.append("Time-dependent pattern - may not hold across all time periods")
        
        return conditions
    
    async def _determine_applicable_domains(self, pattern: Pattern, observations: List[Observation]) -> List[str]:
        """Determine domains where generalization applies"""
        
        domains = set()
        
        # Extract domains from supporting observations
        for obs in pattern.supporting_observations:
            if obs.context and "domain" in obs.context:
                domains.add(obs.context["domain"])
            elif hasattr(obs, "domain"):
                domains.add(obs.domain)
        
        if not domains:
            domains.add("general")
        
        return list(domains)
    
    async def _validate_generalization(self, generalization: Generalization, observations: List[Observation]) -> Generalization:
        """Validate generalization"""
        
        # Cross-validation
        generalization.cross_validation_accuracy = await self._perform_cross_validation(generalization, observations)
        
        # External validation check
        generalization.external_validation = await self._check_external_validation(generalization)
        
        # Temporal validity
        generalization.temporal_validity = await self._assess_temporal_validity(generalization, observations)
        
        return generalization
    
    async def _perform_cross_validation(self, generalization: Generalization, observations: List[Observation]) -> float:
        """Perform cross-validation on generalization"""
        
        # Simple cross-validation
        if len(observations) < 5:
            return 0.5
        
        # Split observations
        split_point = len(observations) // 2
        train_obs = observations[:split_point]
        test_obs = observations[split_point:]
        
        # Test generalization on test set
        pattern = generalization.primary_pattern
        test_support = 0
        
        for obs in test_obs:
            if any(element in obs.entities for element in pattern.pattern_elements):
                test_support += 1
        
        accuracy = test_support / len(test_obs) if test_obs else 0.0
        
        return accuracy
    
    async def _check_external_validation(self, generalization: Generalization) -> bool:
        """Check if generalization can be externally validated"""
        
        # Heuristic based on strength
        return (generalization.probability >= 0.7 and 
                generalization.uncertainty <= 0.3 and
                generalization.supporting_observations >= 5)
    
    async def _assess_temporal_validity(self, generalization: Generalization, observations: List[Observation]) -> float:
        """Assess temporal validity of generalization"""
        
        # Check if pattern holds across time
        if generalization.primary_pattern.temporal_stability:
            return generalization.primary_pattern.temporal_stability
        
        # Simple assessment based on observation spread
        return 0.8  # Default temporal validity
    
    def _initialize_generalization_strategies(self) -> Dict[str, Any]:
        """Initialize generalization strategies"""
        
        return {
            GeneralizationType.UNIVERSAL: {"min_confidence": 0.95, "min_support": 0.9},
            GeneralizationType.STATISTICAL: {"min_confidence": 0.7, "min_support": 0.5},
            GeneralizationType.PROBABILISTIC: {"min_confidence": 0.5, "min_support": 0.3},
            GeneralizationType.CONDITIONAL: {"min_confidence": 0.6, "min_support": 0.4},
            GeneralizationType.CAUSAL: {"min_confidence": 0.8, "min_support": 0.6},
            GeneralizationType.PREDICTIVE: {"min_confidence": 0.7, "min_support": 0.5}
        }
    
    def _initialize_uncertainty_quantification(self) -> Dict[str, Any]:
        """Initialize uncertainty quantification methods"""
        
        return {
            "epistemic": {"model_uncertainty": 0.1, "parameter_uncertainty": 0.05},
            "aleatoric": {"data_noise": 0.1, "measurement_error": 0.05},
            "confidence_intervals": {"method": "bootstrap", "alpha": 0.05}
        }


class ScopeExceptionEvaluator:
    """Engine for evaluating scope and exceptions in inductive reasoning"""
    
    def __init__(self):
        self.evaluation_criteria = self._initialize_evaluation_criteria()
        self.exception_detectors = self._initialize_exception_detectors()
    
    async def evaluate_scope_exceptions(self, generalization: Generalization, observations: List[Observation]) -> ScopeEvaluation:
        """Evaluate scope and exceptions for generalization"""
        
        logger.info("Evaluating scope and exceptions for generalization")
        
        # Assess scope
        scope_breadth = await self._assess_scope_breadth(generalization, observations)
        scope_depth = await self._assess_scope_depth(generalization, observations)
        scope_precision = await self._assess_scope_precision(generalization, observations)
        
        # Find exceptions
        exceptions = await self._find_exceptions(generalization, observations)
        exception_rate = await self._calculate_exception_rate(exceptions, observations)
        exception_patterns = await self._identify_exception_patterns(exceptions)
        
        # Identify limitations
        sample_limitations = await self._identify_sample_limitations(generalization, observations)
        methodological_limitations = await self._identify_methodological_limitations(generalization, observations)
        domain_limitations = await self._identify_domain_limitations(generalization, observations)
        temporal_limitations = await self._identify_temporal_limitations(generalization, observations)
        
        # Find boundary conditions
        boundary_conditions = await self._find_boundary_conditions(generalization, observations)
        edge_cases = await self._find_edge_cases(generalization, observations)
        
        # Assess robustness
        robustness_score = await self._assess_robustness(generalization, observations)
        sensitivity_analysis = await self._perform_sensitivity_analysis(generalization, observations)
        
        # Generate recommendations
        scope_recommendations = await self._generate_scope_recommendations(generalization, observations)
        improvement_suggestions = await self._generate_improvement_suggestions(generalization, observations)
        
        evaluation = ScopeEvaluation(
            scope_breadth=scope_breadth,
            scope_depth=scope_depth,
            scope_precision=scope_precision,
            exceptions_found=exceptions,
            exception_rate=exception_rate,
            exception_patterns=exception_patterns,
            sample_limitations=sample_limitations,
            methodological_limitations=methodological_limitations,
            domain_limitations=domain_limitations,
            temporal_limitations=temporal_limitations,
            boundary_conditions=boundary_conditions,
            edge_cases=edge_cases,
            robustness_score=robustness_score,
            sensitivity_analysis=sensitivity_analysis,
            scope_recommendations=scope_recommendations,
            improvement_suggestions=improvement_suggestions
        )
        
        logger.info(f"Scope evaluation complete: breadth={scope_breadth:.2f}, robustness={robustness_score:.2f}")
        return evaluation
    
    async def _assess_scope_breadth(self, generalization: Generalization, observations: List[Observation]) -> float:
        """Assess breadth of generalization scope"""
        
        # Count different contexts and domains
        contexts = set()
        domains = set()
        
        for obs in generalization.primary_pattern.supporting_observations:
            if obs.context:
                for key, value in obs.context.items():
                    contexts.add(f"{key}={value}")
            
            if hasattr(obs, "domain"):
                domains.add(obs.domain)
        
        # Normalize breadth score
        context_breadth = min(1.0, len(contexts) / 10)  # Up to 10 contexts
        domain_breadth = min(1.0, len(domains) / 5)     # Up to 5 domains
        
        return (context_breadth + domain_breadth) / 2
    
    async def _assess_scope_depth(self, generalization: Generalization, observations: List[Observation]) -> float:
        """Assess depth of generalization scope"""
        
        # Based on detail level and specificity
        pattern = generalization.primary_pattern
        
        # Detail level
        detail_score = len(pattern.pattern_elements) / 10  # Up to 10 elements
        
        # Specificity
        specificity_score = len(pattern.pattern_conditions) / 5  # Up to 5 conditions
        
        # Evidence depth
        evidence_depth = len(pattern.supporting_observations) / 20  # Up to 20 observations
        
        return min(1.0, (detail_score + specificity_score + evidence_depth) / 3)
    
    async def _assess_scope_precision(self, generalization: Generalization, observations: List[Observation]) -> float:
        """Assess precision of generalization scope"""
        
        # Based on confidence interval width and uncertainty
        ci_width = generalization.confidence_interval[1] - generalization.confidence_interval[0]
        precision_from_ci = 1.0 - ci_width
        
        # Uncertainty factor
        precision_from_uncertainty = 1.0 - generalization.uncertainty
        
        return (precision_from_ci + precision_from_uncertainty) / 2
    
    async def _find_exceptions(self, generalization: Generalization, observations: List[Observation]) -> List[str]:
        """Find exceptions to the generalization"""
        
        exceptions = []
        pattern = generalization.primary_pattern
        
        # Find observations that contradict the pattern
        for obs in observations:
            is_exception = False
            
            # Check if observation should support pattern but doesn't
            if any(element in obs.entities for element in pattern.pattern_elements):
                # This observation is relevant to the pattern
                if obs not in pattern.supporting_observations:
                    is_exception = True
                    exceptions.append(f"Observation {obs.id}: {obs.content[:100]}...")
        
        return exceptions
    
    async def _calculate_exception_rate(self, exceptions: List[str], observations: List[Observation]) -> float:
        """Calculate rate of exceptions"""
        
        if not observations:
            return 0.0
        
        return len(exceptions) / len(observations)
    
    async def _identify_exception_patterns(self, exceptions: List[str]) -> List[Pattern]:
        """Identify patterns in exceptions"""
        
        # Simplified - in practice would analyze exception characteristics
        exception_patterns = []
        
        if len(exceptions) >= 3:
            # Create a pattern for frequent exceptions
            exception_pattern = Pattern(
                id="exception_pattern",
                pattern_type=PatternType.FREQUENCY,
                description=f"Exception pattern: {len(exceptions)} exceptions found",
                pattern_elements=["exception"],
                pattern_expression=f"exceptions = {len(exceptions)}",
                frequency=len(exceptions)
            )
            exception_patterns.append(exception_pattern)
        
        return exception_patterns
    
    async def _identify_sample_limitations(self, generalization: Generalization, observations: List[Observation]) -> List[str]:
        """Identify sample-related limitations"""
        
        limitations = []
        
        # Sample size
        if len(observations) < 30:
            limitations.append(f"Small sample size ({len(observations)} observations)")
        
        # Sample quality
        if observations:
            avg_quality = sum(obs.get_quality_score() for obs in observations) / len(observations)
            if avg_quality < 0.7:
                limitations.append(f"Low average observation quality ({avg_quality:.2f})")
        
        # Sample bias
        if observations:
            avg_bias = sum(obs.sample_bias for obs in observations) / len(observations)
            if avg_bias > 0.3:
                limitations.append(f"High sample bias detected ({avg_bias:.2f})")
        
        # Representativeness
        if observations:
            avg_representativeness = sum(obs.representativeness for obs in observations) / len(observations)
            if avg_representativeness < 0.7:
                limitations.append(f"Low representativeness ({avg_representativeness:.2f})")
        
        return limitations
    
    async def _identify_methodological_limitations(self, generalization: Generalization, observations: List[Observation]) -> List[str]:
        """Identify methodological limitations"""
        
        limitations = []
        
        # Collection method diversity
        collection_methods = set(obs.collection_method for obs in observations)
        if len(collection_methods) == 1:
            limitations.append(f"Single collection method: {list(collection_methods)[0]}")
        
        # Source diversity
        sources = set(obs.source for obs in observations)
        if len(sources) == 1:
            limitations.append(f"Single source: {list(sources)[0]}")
        
        # Observation type diversity
        obs_types = set(obs.observation_type for obs in observations)
        if len(obs_types) == 1:
            limitations.append(f"Single observation type: {list(obs_types)[0].value}")
        
        return limitations
    
    async def _identify_domain_limitations(self, generalization: Generalization, observations: List[Observation]) -> List[str]:
        """Identify domain-related limitations"""
        
        limitations = []
        
        # Domain coverage
        domains = set(obs.context.get("domain", "general") for obs in observations if obs.context)
        if len(domains) == 1:
            limitations.append(f"Limited to single domain: {list(domains)[0]}")
        
        # Context coverage
        contexts = set()
        for obs in observations:
            if obs.context:
                for key, value in obs.context.items():
                    contexts.add(f"{key}={value}")
        
        if len(contexts) < 3:
            limitations.append("Limited context diversity")
        
        return limitations
    
    async def _identify_temporal_limitations(self, generalization: Generalization, observations: List[Observation]) -> List[str]:
        """Identify temporal limitations"""
        
        limitations = []
        
        # Time span coverage
        timestamps = [obs.collection_timestamp for obs in observations if obs.collection_timestamp]
        if len(timestamps) >= 2:
            time_span = max(timestamps) - min(timestamps)
            if time_span.days < 7:
                limitations.append(f"Short time span: {time_span.days} days")
        
        # Temporal stability
        if generalization.primary_pattern.temporal_stability and generalization.primary_pattern.temporal_stability < 0.7:
            limitations.append("Pattern shows temporal instability")
        
        return limitations
    
    async def _find_boundary_conditions(self, generalization: Generalization, observations: List[Observation]) -> List[str]:
        """Find boundary conditions for generalization"""
        
        boundary_conditions = []
        
        # Numerical boundaries
        pattern = generalization.primary_pattern
        for obs in pattern.supporting_observations:
            for measure_name, measure_value in obs.measurements.items():
                if isinstance(measure_value, (int, float)):
                    boundary_conditions.append(f"{measure_name} must be around {measure_value}")
        
        # Contextual boundaries
        contexts = set()
        for obs in pattern.supporting_observations:
            if obs.context:
                for key, value in obs.context.items():
                    contexts.add(f"{key} = {value}")
        
        if contexts:
            boundary_conditions.append(f"Context constraints: {', '.join(list(contexts)[:3])}")
        
        return boundary_conditions
    
    async def _find_edge_cases(self, generalization: Generalization, observations: List[Observation]) -> List[str]:
        """Find edge cases for generalization"""
        
        edge_cases = []
        
        # Extreme values
        pattern = generalization.primary_pattern
        for obs in pattern.supporting_observations:
            for measure_name, measure_value in obs.measurements.items():
                if isinstance(measure_value, (int, float)):
                    # Find extreme values
                    all_values = [o.measurements.get(measure_name, 0) for o in observations 
                                if measure_name in o.measurements]
                    if all_values:
                        min_val, max_val = min(all_values), max(all_values)
                        if measure_value <= min_val or measure_value >= max_val:
                            edge_cases.append(f"Extreme {measure_name}: {measure_value}")
        
        # Unusual combinations
        if len(pattern.pattern_elements) > 1:
            edge_cases.append("Unusual combination of pattern elements")
        
        return edge_cases
    
    async def _assess_robustness(self, generalization: Generalization, observations: List[Observation]) -> float:
        """Assess robustness of generalization"""
        
        robustness_factors = []
        
        # Stability across different conditions
        pattern = generalization.primary_pattern
        if pattern.stability_score:
            robustness_factors.append(pattern.stability_score)
        else:
            robustness_factors.append(0.5)  # Default
        
        # Cross-validation performance
        if generalization.cross_validation_accuracy:
            robustness_factors.append(generalization.cross_validation_accuracy)
        else:
            robustness_factors.append(0.5)  # Default
        
        # Exception rate (inverse)
        exception_rate = await self._calculate_exception_rate([], observations)
        robustness_factors.append(1.0 - exception_rate)
        
        # Uncertainty (inverse)
        robustness_factors.append(1.0 - generalization.uncertainty)
        
        return sum(robustness_factors) / len(robustness_factors)
    
    async def _perform_sensitivity_analysis(self, generalization: Generalization, observations: List[Observation]) -> Dict[str, float]:
        """Perform sensitivity analysis"""
        
        sensitivity = {}
        
        # Sensitivity to sample size
        if len(observations) > 5:
            # Test with smaller sample
            small_sample = observations[:len(observations)//2]
            small_support = len([obs for obs in small_sample 
                               if any(element in obs.entities 
                                     for element in generalization.primary_pattern.pattern_elements)])
            original_support = generalization.supporting_observations
            
            if original_support > 0:
                sensitivity["sample_size"] = abs(small_support - original_support) / original_support
            else:
                sensitivity["sample_size"] = 0.0
        
        # Sensitivity to observation quality
        high_quality_obs = [obs for obs in observations if obs.get_quality_score() > 0.8]
        if high_quality_obs:
            hq_support = len([obs for obs in high_quality_obs 
                             if any(element in obs.entities 
                                   for element in generalization.primary_pattern.pattern_elements)])
            if generalization.supporting_observations > 0:
                sensitivity["observation_quality"] = abs(hq_support - generalization.supporting_observations) / generalization.supporting_observations
            else:
                sensitivity["observation_quality"] = 0.0
        
        return sensitivity
    
    async def _generate_scope_recommendations(self, generalization: Generalization, observations: List[Observation]) -> List[str]:
        """Generate recommendations for scope improvement"""
        
        recommendations = []
        
        # Sample size recommendations
        if len(observations) < 30:
            recommendations.append("Increase sample size for better generalizability")
        
        # Domain diversity recommendations
        domains = set(obs.context.get("domain", "general") for obs in observations if obs.context)
        if len(domains) == 1:
            recommendations.append("Test generalization across multiple domains")
        
        # Temporal recommendations
        if generalization.primary_pattern.temporal_stability and generalization.primary_pattern.temporal_stability < 0.7:
            recommendations.append("Conduct longitudinal study to assess temporal stability")
        
        # Quality recommendations
        if observations:
            avg_quality = sum(obs.get_quality_score() for obs in observations) / len(observations)
            if avg_quality < 0.7:
                recommendations.append("Improve observation quality through better collection methods")
        
        return recommendations
    
    async def _generate_improvement_suggestions(self, generalization: Generalization, observations: List[Observation]) -> List[str]:
        """Generate suggestions for improvement"""
        
        suggestions = []
        
        # Statistical improvements
        if generalization.probability < 0.7:
            suggestions.append("Gather more supporting evidence to increase probability")
        
        # Uncertainty reduction
        if generalization.uncertainty > 0.5:
            suggestions.append("Reduce uncertainty through controlled experiments")
        
        # Validation improvements
        if not generalization.external_validation:
            suggestions.append("Seek external validation through independent studies")
        
        # Scope refinement
        if len(generalization.scope_conditions) < 2:
            suggestions.append("Better define scope conditions and boundary cases")
        
        return suggestions
    
    def _initialize_evaluation_criteria(self) -> Dict[str, Any]:
        """Initialize evaluation criteria"""
        
        return {
            "scope_breadth": {"min_contexts": 3, "min_domains": 2},
            "scope_depth": {"min_elements": 2, "min_conditions": 1},
            "scope_precision": {"max_ci_width": 0.4, "max_uncertainty": 0.5},
            "exception_rate": {"max_acceptable": 0.2},
            "robustness": {"min_stability": 0.6, "min_cross_validation": 0.6}
        }
    
    def _initialize_exception_detectors(self) -> Dict[str, Any]:
        """Initialize exception detection methods"""
        
        return {
            "outlier_detection": {"method": "iqr", "threshold": 1.5},
            "pattern_violation": {"method": "rule_based", "strictness": 0.8},
            "contextual_exception": {"method": "clustering", "min_cluster_size": 3}
        }


class PredictiveInferenceEngine:
    """Engine for predictive and explanatory inference application"""
    
    def __init__(self):
        self.inference_strategies = self._initialize_inference_strategies()
        self.validation_methods = self._initialize_validation_methods()
        self.uncertainty_models = self._initialize_uncertainty_models()
    
    async def generate_inference(self, generalization: Generalization, context: Dict[str, Any]) -> InductiveInference:
        """Generate predictive or explanatory inference"""
        
        logger.info("Generating predictive/explanatory inference")
        
        # Determine inference type
        inference_type = await self._determine_inference_type(generalization, context)
        
        # Generate inference statement
        statement = await self._generate_inference_statement(generalization, context, inference_type)
        
        # Calculate inference properties
        if inference_type == "predictive":
            prediction_confidence = await self._calculate_prediction_confidence(generalization, context)
            prediction_horizon = await self._determine_prediction_horizon(generalization, context)
            prediction_accuracy = await self._estimate_prediction_accuracy(generalization, context)
            
            explanatory_power = 0.0
            causal_strength = 0.0
            mechanism_clarity = 0.0
        else:  # explanatory
            explanatory_power = await self._calculate_explanatory_power(generalization, context)
            causal_strength = await self._assess_causal_strength(generalization, context)
            mechanism_clarity = await self._assess_mechanism_clarity(generalization, context)
            
            prediction_confidence = 0.0
            prediction_horizon = None
            prediction_accuracy = 0.0
        
        # Quantify uncertainty
        epistemic_uncertainty = await self._calculate_epistemic_uncertainty(generalization, context)
        aleatoric_uncertainty = await self._calculate_aleatoric_uncertainty(generalization, context)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Validation
        validation_tests = await self._design_validation_tests(generalization, context, inference_type)
        validation_results = await self._perform_validation_tests(generalization, context, validation_tests)
        
        inference = InductiveInference(
            id=str(uuid4()),
            inference_type=inference_type,
            statement=statement,
            generalization=generalization,
            application_context=context,
            prediction_confidence=prediction_confidence,
            prediction_horizon=prediction_horizon,
            prediction_accuracy=prediction_accuracy,
            explanatory_power=explanatory_power,
            causal_strength=causal_strength,
            mechanism_clarity=mechanism_clarity,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            validation_tests=validation_tests,
            validation_results=validation_results
        )
        
        logger.info(f"Generated {inference_type} inference with confidence {inference.prediction_confidence:.2f}")
        return inference
    
    async def _determine_inference_type(self, generalization: Generalization, context: Dict[str, Any]) -> str:
        """Determine whether to make predictive or explanatory inference"""
        
        # Check context hints
        if context.get("inference_type"):
            return context["inference_type"]
        
        # Check generalization type
        if generalization.generalization_type in [GeneralizationType.PREDICTIVE, GeneralizationType.STATISTICAL]:
            return "predictive"
        elif generalization.generalization_type in [GeneralizationType.EXPLANATORY, GeneralizationType.CAUSAL]:
            return "explanatory"
        
        # Default based on pattern type
        pattern_type = generalization.primary_pattern.pattern_type
        if pattern_type in [PatternType.TREND, PatternType.SEQUENTIAL, PatternType.CYCLICAL]:
            return "predictive"
        else:
            return "explanatory"
    
    async def _generate_inference_statement(self, generalization: Generalization, context: Dict[str, Any], inference_type: str) -> str:
        """Generate inference statement"""
        
        base_statement = generalization.statement
        
        if inference_type == "predictive":
            # Future-oriented prediction
            statement = f"Based on the observed pattern, we predict that {base_statement} will continue to hold in future cases"
            
            # Add context-specific prediction
            if context.get("target_domain"):
                statement += f" within the {context['target_domain']} domain"
            
            # Add time horizon
            if context.get("time_horizon"):
                statement += f" over the next {context['time_horizon']}"
        
        else:  # explanatory
            # Explanation-oriented inference
            statement = f"The observed pattern suggests that {base_statement} can be explained by underlying mechanisms"
            
            # Add causal explanation
            if generalization.generalization_type == GeneralizationType.CAUSAL:
                statement += " involving causal relationships"
            
            # Add context-specific explanation
            if context.get("phenomenon"):
                statement += f" related to {context['phenomenon']}"
        
        return statement
    
    async def _calculate_prediction_confidence(self, generalization: Generalization, context: Dict[str, Any]) -> float:
        """Calculate confidence in prediction"""
        
        # Base confidence on generalization probability
        base_confidence = generalization.probability
        
        # Adjust for context similarity
        if context.get("similarity_to_training"):
            base_confidence *= context["similarity_to_training"]
        
        # Adjust for temporal stability
        if generalization.temporal_validity:
            base_confidence *= generalization.temporal_validity
        
        # Adjust for uncertainty
        base_confidence *= (1.0 - generalization.uncertainty)
        
        return max(0.0, min(1.0, base_confidence))
    
    async def _determine_prediction_horizon(self, generalization: Generalization, context: Dict[str, Any]) -> Optional[str]:
        """Determine prediction horizon"""
        
        # Check context
        if context.get("time_horizon"):
            return context["time_horizon"]
        
        # Based on pattern type
        pattern_type = generalization.primary_pattern.pattern_type
        if pattern_type == PatternType.TREND:
            return "short_term"
        elif pattern_type == PatternType.CYCLICAL:
            return "medium_term"
        elif pattern_type == PatternType.SEQUENTIAL:
            return "immediate"
        else:
            return "short_term"
    
    async def _estimate_prediction_accuracy(self, generalization: Generalization, context: Dict[str, Any]) -> float:
        """Estimate prediction accuracy"""
        
        # Base accuracy on cross-validation
        if generalization.cross_validation_accuracy:
            base_accuracy = generalization.cross_validation_accuracy
        else:
            base_accuracy = generalization.probability
        
        # Adjust for context factors
        if context.get("complexity_factor"):
            base_accuracy *= (1.0 - context["complexity_factor"])
        
        # Adjust for temporal distance
        if context.get("temporal_distance"):
            decay_factor = 1.0 / (1.0 + context["temporal_distance"])
            base_accuracy *= decay_factor
        
        return max(0.0, min(1.0, base_accuracy))
    
    async def _calculate_explanatory_power(self, generalization: Generalization, context: Dict[str, Any]) -> float:
        """Calculate explanatory power"""
        
        # Base power on generalization strength
        base_power = generalization.get_reliability_score()
        
        # Adjust for mechanism clarity
        if generalization.primary_pattern.pattern_type == PatternType.CAUSAL:
            base_power *= 1.2
        
        # Adjust for coverage
        if generalization.population_coverage:
            base_power *= generalization.population_coverage
        
        return max(0.0, min(1.0, base_power))
    
    async def _assess_causal_strength(self, generalization: Generalization, context: Dict[str, Any]) -> float:
        """Assess causal strength"""
        
        # Base strength on pattern type and evidence
        if generalization.primary_pattern.pattern_type == PatternType.CAUSAL:
            base_strength = 0.8
        elif generalization.primary_pattern.pattern_type == PatternType.CORRELATION:
            base_strength = 0.4
        else:
            base_strength = 0.2
        
        # Adjust for evidence quality
        if generalization.primary_pattern.supporting_observations:
            avg_quality = sum(obs.get_quality_score() for obs in generalization.primary_pattern.supporting_observations) / len(generalization.primary_pattern.supporting_observations)
            base_strength *= avg_quality
        
        # Adjust for statistical significance
        if generalization.primary_pattern.statistical_significance:
            base_strength *= generalization.primary_pattern.statistical_significance
        
        return max(0.0, min(1.0, base_strength))
    
    async def _assess_mechanism_clarity(self, generalization: Generalization, context: Dict[str, Any]) -> float:
        """Assess mechanism clarity"""
        
        # Base clarity on pattern detail
        pattern = generalization.primary_pattern
        base_clarity = len(pattern.pattern_elements) / 10  # Up to 10 elements
        
        # Adjust for relationship clarity
        if pattern.pattern_type in [PatternType.CAUSAL, PatternType.CORRELATION]:
            base_clarity += 0.3
        
        # Adjust for supporting patterns
        if generalization.supporting_patterns:
            base_clarity += len(generalization.supporting_patterns) * 0.1
        
        return max(0.0, min(1.0, base_clarity))
    
    async def _calculate_epistemic_uncertainty(self, generalization: Generalization, context: Dict[str, Any]) -> float:
        """Calculate epistemic (model) uncertainty"""
        
        # Model uncertainty
        model_uncertainty = 0.1  # Base model uncertainty
        
        # Adjust for generalization type
        if generalization.generalization_type == GeneralizationType.PROBABILISTIC:
            model_uncertainty += 0.1
        elif generalization.generalization_type == GeneralizationType.UNIVERSAL:
            model_uncertainty -= 0.05
        
        # Adjust for pattern complexity
        pattern_complexity = len(generalization.primary_pattern.pattern_elements) / 10
        model_uncertainty += pattern_complexity * 0.1
        
        return max(0.0, min(0.5, model_uncertainty))
    
    async def _calculate_aleatoric_uncertainty(self, generalization: Generalization, context: Dict[str, Any]) -> float:
        """Calculate aleatoric (data) uncertainty"""
        
        # Data uncertainty
        data_uncertainty = generalization.uncertainty
        
        # Adjust for observation quality
        if generalization.primary_pattern.supporting_observations:
            avg_quality = sum(obs.get_quality_score() for obs in generalization.primary_pattern.supporting_observations) / len(generalization.primary_pattern.supporting_observations)
            data_uncertainty *= (1.0 - avg_quality)
        
        # Adjust for sample size
        sample_size = generalization.total_observations
        if sample_size < 10:
            data_uncertainty += 0.2
        elif sample_size < 5:
            data_uncertainty += 0.4
        
        return max(0.0, min(0.5, data_uncertainty))
    
    async def _design_validation_tests(self, generalization: Generalization, context: Dict[str, Any], inference_type: str) -> List[str]:
        """Design validation tests for inference"""
        
        tests = []
        
        if inference_type == "predictive":
            tests.append("Out-of-sample prediction test")
            tests.append("Temporal validation test")
            tests.append("Cross-domain validation test")
        else:  # explanatory
            tests.append("Mechanism verification test")
            tests.append("Alternative explanation test")
            tests.append("Causal intervention test")
        
        # General tests
        tests.append("Robustness test")
        tests.append("Sensitivity analysis")
        
        return tests
    
    async def _perform_validation_tests(self, generalization: Generalization, context: Dict[str, Any], tests: List[str]) -> Dict[str, float]:
        """Perform validation tests"""
        
        results = {}
        
        for test in tests:
            if test == "Out-of-sample prediction test":
                results[test] = generalization.cross_validation_accuracy or 0.5
            elif test == "Temporal validation test":
                results[test] = generalization.temporal_validity or 0.7
            elif test == "Cross-domain validation test":
                results[test] = 0.6  # Placeholder
            elif test == "Mechanism verification test":
                results[test] = 0.7  # Placeholder
            elif test == "Alternative explanation test":
                results[test] = 0.8  # Placeholder
            elif test == "Causal intervention test":
                results[test] = 0.5  # Placeholder
            elif test == "Robustness test":
                results[test] = generalization.get_reliability_score()
            elif test == "Sensitivity analysis":
                results[test] = 0.7  # Placeholder
            else:
                results[test] = 0.5  # Default
        
        return results
    
    def _initialize_inference_strategies(self) -> Dict[str, Any]:
        """Initialize inference strategies"""
        
        return {
            "predictive": {
                "methods": ["extrapolation", "interpolation", "pattern_matching"],
                "confidence_factors": ["temporal_stability", "pattern_strength", "context_similarity"]
            },
            "explanatory": {
                "methods": ["causal_analysis", "mechanism_identification", "pattern_explanation"],
                "confidence_factors": ["causal_strength", "mechanism_clarity", "evidence_quality"]
            }
        }
    
    def _initialize_validation_methods(self) -> Dict[str, Any]:
        """Initialize validation methods"""
        
        return {
            "predictive": ["out_of_sample", "temporal_validation", "cross_domain"],
            "explanatory": ["mechanism_verification", "alternative_explanation", "causal_intervention"],
            "general": ["robustness", "sensitivity_analysis"]
        }
    
    def _initialize_uncertainty_models(self) -> Dict[str, Any]:
        """Initialize uncertainty models"""
        
        return {
            "epistemic": {
                "model_uncertainty": 0.1,
                "parameter_uncertainty": 0.05,
                "structural_uncertainty": 0.08
            },
            "aleatoric": {
                "data_noise": 0.1,
                "measurement_error": 0.05,
                "natural_variability": 0.07
            }
        }


class EnhancedInductiveReasoningEngine:
    """
    Enhanced Inductive Reasoning Engine implementing all elemental components
    
    This engine provides comprehensive inductive reasoning capabilities with:
    1. Observation Collection and Validation
    2. Pattern Recognition and Statistical Validation
    3. Probabilistic Generalization
    4. Scope and Exception Evaluation
    5. Predictive and Explanatory Inference
    """
    
    def __init__(self):
        # Initialize elemental components
        self.observation_engine = ObservationCollectionEngine()
        self.pattern_engine = PatternRecognitionEngine()
        self.generalization_engine = GeneralizationEngine()
        self.scope_evaluator = ScopeExceptionEvaluator()
        self.inference_engine = PredictiveInferenceEngine()
        
        # Reasoning history
        self.reasoning_history: List[InductiveReasoning] = []
        
        # Configuration
        self.confidence_threshold = 0.5
        self.max_reasoning_depth = 5
        
        logger.info("Enhanced Inductive Reasoning Engine initialized")
    
    async def perform_inductive_reasoning(
        self,
        raw_observations: List[str],
        query: str,
        context: Dict[str, Any] = None
    ) -> InductiveReasoning:
        """
        Perform comprehensive inductive reasoning using all elemental components
        
        Args:
            raw_observations: List of raw observation data
            query: The question or hypothesis to investigate
            context: Additional context for reasoning
            
        Returns:
            InductiveReasoning: Complete reasoning with all elemental components
        """
        
        logger.info(f"Performing inductive reasoning: {query}")
        
        context = context or {}
        start_time = time.time()
        
        # Component 1: Observation Collection
        logger.info("Step 1: Collecting observations")
        observations = await self.observation_engine.collect_observations(raw_observations, context)
        
        # Component 2: Pattern Recognition
        logger.info("Step 2: Recognizing patterns")
        patterns = await self.pattern_engine.recognize_patterns(observations)
        
        # Component 3: Generalization
        logger.info("Step 3: Creating generalization")
        generalization = await self.generalization_engine.create_generalization(patterns, observations)
        
        # Component 4: Scope and Exception Evaluation
        logger.info("Step 4: Evaluating scope and exceptions")
        scope_evaluation = await self.scope_evaluator.evaluate_scope_exceptions(generalization, observations)
        
        # Component 5: Predictive/Explanatory Inference
        logger.info("Step 5: Generating inference")
        inference = await self.inference_engine.generate_inference(generalization, context)
        
        # Determine method used
        method_used = await self._determine_method_used(patterns, generalization, context)
        
        # Calculate overall reasoning quality
        reasoning_quality = await self._calculate_reasoning_quality(
            observations, patterns, generalization, scope_evaluation, inference
        )
        
        # Determine confidence level
        confidence_level = await self._determine_confidence_level(generalization.probability)
        
        # Create complete reasoning
        reasoning = InductiveReasoning(
            id=str(uuid4()),
            query=query,
            observation_collection=observations,
            pattern_recognition=patterns,
            generalization=generalization,
            scope_evaluation=scope_evaluation,
            inference=inference,
            method_used=method_used,
            reasoning_quality=reasoning_quality,
            processing_time=time.time() - start_time,
            conclusion=inference.statement,
            confidence_level=confidence_level,
            probability=generalization.probability
        )
        
        # Add to history
        self.reasoning_history.append(reasoning)
        
        logger.info(f"Inductive reasoning complete: probability={generalization.probability:.2f}, quality={reasoning_quality:.2f}")
        
        return reasoning
    
    async def _determine_method_used(self, patterns: List[Pattern], generalization: Generalization, context: Dict[str, Any]) -> InductiveMethodType:
        """Determine which inductive method was primarily used"""
        
        # Check context hints
        if context.get("method"):
            try:
                return InductiveMethodType(context["method"])
            except ValueError:
                pass
        
        # Determine based on patterns and generalization
        if generalization.generalization_type == GeneralizationType.STATISTICAL:
            return InductiveMethodType.STATISTICAL_INFERENCE
        elif generalization.generalization_type == GeneralizationType.CAUSAL:
            return InductiveMethodType.MILL_METHODS
        elif generalization.generalization_type == GeneralizationType.PROBABILISTIC:
            return InductiveMethodType.BAYESIAN_INFERENCE
        elif generalization.generalization_type == GeneralizationType.PREDICTIVE:
            return InductiveMethodType.TREND_EXTRAPOLATION
        else:
            return InductiveMethodType.ENUMERATION
    
    async def _calculate_reasoning_quality(
        self,
        observations: List[Observation],
        patterns: List[Pattern],
        generalization: Generalization,
        scope_evaluation: ScopeEvaluation,
        inference: InductiveInference
    ) -> float:
        """Calculate overall reasoning quality"""
        
        quality_factors = []
        
        # Observation quality
        if observations:
            avg_obs_quality = sum(obs.get_quality_score() for obs in observations) / len(observations)
            quality_factors.append(avg_obs_quality)
        
        # Pattern quality
        if patterns:
            avg_pattern_strength = sum(p.get_strength_score() for p in patterns) / len(patterns)
            quality_factors.append(avg_pattern_strength)
        
        # Generalization quality
        generalization_quality = generalization.get_reliability_score()
        quality_factors.append(generalization_quality)
        
        # Scope evaluation quality
        scope_quality = scope_evaluation.robustness_score
        quality_factors.append(scope_quality)
        
        # Inference quality
        inference_quality = inference.get_utility_score()
        quality_factors.append(inference_quality)
        
        return sum(quality_factors) / len(quality_factors)
    
    async def _determine_confidence_level(self, probability: float) -> ConfidenceLevel:
        """Determine confidence level from probability"""
        
        if probability >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif probability >= 0.7:
            return ConfidenceLevel.HIGH
        elif probability >= 0.5:
            return ConfidenceLevel.MODERATE
        elif probability >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    async def refine_reasoning(self, reasoning: InductiveReasoning, new_observations: List[str]) -> InductiveReasoning:
        """Refine existing reasoning with new observations"""
        
        logger.info(f"Refining reasoning with {len(new_observations)} new observations")
        
        # Collect new observations
        new_obs = await self.observation_engine.collect_observations(new_observations, reasoning.inference.application_context)
        
        # Combine with existing observations
        all_observations = reasoning.observation_collection + new_obs
        
        # Re-perform reasoning
        refined_reasoning = await self.perform_inductive_reasoning(
            [obs.content for obs in all_observations],
            reasoning.query,
            reasoning.inference.application_context
        )
        
        # Update revision tracking
        refined_reasoning.generalization.revision_count = reasoning.generalization.revision_count + 1
        refined_reasoning.generalization.last_revision = datetime.now(timezone.utc)
        
        return refined_reasoning
    
    async def validate_reasoning(self, reasoning: InductiveReasoning, validation_data: List[str]) -> Dict[str, float]:
        """Validate reasoning with independent data"""
        
        logger.info("Validating reasoning with independent data")
        
        # Collect validation observations
        validation_obs = await self.observation_engine.collect_observations(validation_data, reasoning.inference.application_context)
        
        # Test generalization against validation data
        pattern = reasoning.generalization.primary_pattern
        validation_support = 0
        
        for obs in validation_obs:
            if any(element in obs.entities for element in pattern.pattern_elements):
                validation_support += 1
        
        # Calculate validation metrics
        validation_accuracy = validation_support / len(validation_obs) if validation_obs else 0.0
        
        # Compare with expected probability
        expected_support = reasoning.generalization.probability * len(validation_obs)
        prediction_error = abs(validation_support - expected_support) / len(validation_obs) if validation_obs else 1.0
        
        # Overall validation score
        validation_score = validation_accuracy * (1.0 - prediction_error)
        
        return {
            "validation_accuracy": validation_accuracy,
            "prediction_error": prediction_error,
            "validation_score": validation_score,
            "expected_support": expected_support,
            "actual_support": validation_support
        }
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about inductive reasoning performance"""
        
        if not self.reasoning_history:
            return {"total_reasoning_sessions": 0}
        
        return {
            "total_reasoning_sessions": len(self.reasoning_history),
            "average_probability": sum(r.probability for r in self.reasoning_history) / len(self.reasoning_history),
            "average_quality": sum(r.reasoning_quality for r in self.reasoning_history) / len(self.reasoning_history),
            "average_processing_time": sum(r.processing_time for r in self.reasoning_history) / len(self.reasoning_history),
            "confidence_level_distribution": {
                cl.value: sum(1 for r in self.reasoning_history if r.confidence_level == cl) 
                for cl in ConfidenceLevel
            },
            "method_usage": {
                method.value: sum(1 for r in self.reasoning_history if r.method_used == method) 
                for method in InductiveMethodType
            },
            "generalization_types": {
                gt.value: sum(1 for r in self.reasoning_history if r.generalization.generalization_type == gt) 
                for gt in GeneralizationType
            },
            "average_observations_per_session": sum(len(r.observation_collection) for r in self.reasoning_history) / len(self.reasoning_history),
            "average_patterns_per_session": sum(len(r.pattern_recognition) for r in self.reasoning_history) / len(self.reasoning_history),
            "external_validation_rate": sum(1 for r in self.reasoning_history if r.generalization.external_validation) / len(self.reasoning_history)
        }


# Example usage and demonstration
async def demonstrate_enhanced_inductive_reasoning():
    """Demonstrate the enhanced inductive reasoning system"""
    
    print("🔍 ENHANCED INDUCTIVE REASONING SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Initialize engine
    engine = EnhancedInductiveReasoningEngine()
    
    # Example 1: Pattern recognition in observations
    print("\n1. 📊 PATTERN RECOGNITION DEMONSTRATION")
    print("-" * 40)
    
    observations = [
        "The sun rose at 6:30 AM on Monday",
        "The sun rose at 6:32 AM on Tuesday", 
        "The sun rose at 6:34 AM on Wednesday",
        "The sun rose at 6:36 AM on Thursday",
        "The sun rose at 6:38 AM on Friday"
    ]
    
    reasoning = await engine.perform_inductive_reasoning(
        observations, 
        "What pattern exists in sunrise times?",
        {"domain": "astronomy", "source": "observation"}
    )
    
    print(f"Query: What pattern exists in sunrise times?")
    print(f"Observations: {len(observations)}")
    print(f"Patterns found: {len(reasoning.pattern_recognition)}")
    print(f"Generalization: {reasoning.generalization.statement}")
    print(f"Probability: {reasoning.probability:.2f}")
    print(f"Confidence: {reasoning.confidence_level.value}")
    
    # Example 2: Statistical inference
    print("\n2. 📈 STATISTICAL INFERENCE DEMONSTRATION")
    print("-" * 40)
    
    observations = [
        "Patient A took medicine and recovered in 3 days",
        "Patient B took medicine and recovered in 4 days",
        "Patient C took medicine and recovered in 3 days",
        "Patient D took medicine and recovered in 5 days",
        "Patient E took medicine and recovered in 4 days"
    ]
    
    reasoning = await engine.perform_inductive_reasoning(
        observations,
        "Is the medicine effective?",
        {"domain": "medicine", "source": "clinical_trial"}
    )
    
    print(f"Query: Is the medicine effective?")
    print(f"Observations: {len(observations)}")
    print(f"Generalization: {reasoning.generalization.statement}")
    print(f"Probability: {reasoning.probability:.2f}")
    print(f"Scope limitations: {len(reasoning.scope_evaluation.sample_limitations)}")
    
    # Example 3: Predictive inference
    print("\n3. 🔮 PREDICTIVE INFERENCE DEMONSTRATION")
    print("-" * 40)
    
    observations = [
        "Stock price was $100 on day 1",
        "Stock price was $102 on day 2",
        "Stock price was $104 on day 3",
        "Stock price was $106 on day 4",
        "Stock price was $108 on day 5"
    ]
    
    reasoning = await engine.perform_inductive_reasoning(
        observations,
        "What will happen to the stock price?",
        {"domain": "finance", "inference_type": "predictive"}
    )
    
    print(f"Query: What will happen to the stock price?")
    print(f"Inference type: {reasoning.inference.inference_type}")
    print(f"Prediction: {reasoning.inference.statement}")
    print(f"Prediction confidence: {reasoning.inference.prediction_confidence:.2f}")
    print(f"Uncertainty: {reasoning.inference.total_uncertainty:.2f}")
    
    # Show statistics
    print("\n📊 REASONING STATISTICS")
    print("-" * 40)
    
    stats = engine.get_reasoning_statistics()
    print(f"Total reasoning sessions: {stats['total_reasoning_sessions']}")
    print(f"Average probability: {stats['average_probability']:.2f}")
    print(f"Average quality: {stats['average_quality']:.2f}")
    print(f"Average processing time: {stats['average_processing_time']:.2f}s")
    
    print("\n" + "=" * 70)
    print("✅ Enhanced inductive reasoning demonstration complete!")


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_inductive_reasoning())