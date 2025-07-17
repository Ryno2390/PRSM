#!/usr/bin/env python3
"""
Enhanced Causal Reasoning Engine for NWTN
=========================================

This module implements a comprehensive causal reasoning system based on
elemental components derived from causal inference research and philosophy of science.

The system follows the five elemental components of causal reasoning:
1. Observation of Events
2. Identification of Potential Causes
3. Causal Linking
4. Evaluation of Confounding Factors
5. Predictive or Explanatory Inference

Key Features:
- Comprehensive event observation and effect identification
- Systematic potential cause identification with multiple strategies
- Rigorous causal linking with mechanism identification
- Thorough confounding factor evaluation and control
- Predictive inference application with intervention analysis
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


class EventType(Enum):
    """Types of events in causal reasoning"""
    EFFECT = "effect"                    # The outcome to be explained
    CAUSE = "cause"                     # The causal factor
    INTERMEDIATE = "intermediate"        # Intermediate event in causal chain
    CONFOUNDING = "confounding"         # Confounding event
    MEDIATING = "mediating"             # Mediating event
    MODERATING = "moderating"           # Moderating event
    SPURIOUS = "spurious"               # Spurious correlation event
    BACKGROUND = "background"           # Background condition
    TEMPORAL = "temporal"               # Temporal marker event
    CONTEXTUAL = "contextual"           # Contextual event


class CausalRelationType(Enum):
    """Types of causal relationships"""
    DIRECT_CAUSE = "direct_cause"           # A directly causes B
    INDIRECT_CAUSE = "indirect_cause"       # A causes B through intermediates
    COMMON_CAUSE = "common_cause"           # A and B share a common cause
    BIDIRECTIONAL = "bidirectional"         # A and B cause each other
    NECESSARY_CAUSE = "necessary_cause"     # A is necessary for B
    SUFFICIENT_CAUSE = "sufficient_cause"   # A is sufficient for B
    PARTIAL_CAUSE = "partial_cause"         # A contributes to B with other factors
    PROBABILISTIC = "probabilistic"         # A increases probability of B
    ENABLING = "enabling"                   # A enables B to occur
    TRIGGERING = "triggering"               # A triggers B under conditions


class CausalStrength(Enum):
    """Strength of causal relationships"""
    VERY_STRONG = "very_strong"     # >0.8
    STRONG = "strong"               # 0.6-0.8
    MODERATE = "moderate"           # 0.4-0.6
    WEAK = "weak"                   # 0.2-0.4
    VERY_WEAK = "very_weak"         # <0.2


class CausalMechanism(Enum):
    """Types of causal mechanisms"""
    PHYSICAL = "physical"               # Physical processes
    BIOLOGICAL = "biological"           # Biological processes
    PSYCHOLOGICAL = "psychological"     # Mental/cognitive processes
    SOCIAL = "social"                  # Social processes
    ECONOMIC = "economic"              # Economic processes
    INFORMATIONAL = "informational"     # Information transfer
    CHEMICAL = "chemical"              # Chemical processes
    MECHANICAL = "mechanical"          # Mechanical processes
    ELECTRICAL = "electrical"          # Electrical processes
    STATISTICAL = "statistical"        # Statistical relationships
    UNKNOWN = "unknown"                # Mechanism unknown


class TemporalRelation(Enum):
    """Temporal relationships between cause and effect"""
    IMMEDIATE = "immediate"         # Effect occurs immediately
    SHORT_TERM = "short_term"      # Effect occurs within minutes/hours
    MEDIUM_TERM = "medium_term"    # Effect occurs within days/weeks
    LONG_TERM = "long_term"        # Effect occurs over months/years
    DELAYED = "delayed"            # Effect has significant delay
    CONTINUOUS = "continuous"       # Ongoing causal relationship
    CYCLICAL = "cyclical"          # Cyclical causal pattern
    CUMULATIVE = "cumulative"      # Cumulative effect over time


class CauseGenerationStrategy(Enum):
    """Strategies for generating potential causes"""
    DOMAIN_KNOWLEDGE = "domain_knowledge"    # Based on domain expertise
    TEMPORAL_PRECEDENCE = "temporal_precedence"  # Based on temporal order
    CORRELATION_ANALYSIS = "correlation_analysis"  # Based on correlations
    MECHANISM_THEORY = "mechanism_theory"    # Based on theoretical mechanisms
    ANALOGICAL = "analogical"               # Based on analogies
    ELIMINATIVE = "eliminative"             # Process of elimination
    EMPIRICAL = "empirical"                 # Based on empirical patterns
    EXPERT_KNOWLEDGE = "expert_knowledge"   # Based on expert input


class ConfoundingType(Enum):
    """Types of confounding factors"""
    COMMON_CAUSE = "common_cause"           # Common cause of both variables
    COLLIDER = "collider"                  # Common effect of both variables
    MEDIATOR = "mediator"                  # Intermediate variable
    MODERATOR = "moderator"                # Interaction variable
    SELECTION_BIAS = "selection_bias"       # Selection bias
    MEASUREMENT_ERROR = "measurement_error"  # Measurement error
    UNMEASURED_CONFOUNDER = "unmeasured_confounder"  # Unknown confounder
    REVERSE_CAUSATION = "reverse_causation"  # Reverse causal direction


class ConfidenceLevel(Enum):
    """Confidence levels for causal conclusions"""
    VERY_HIGH = "very_high"    # >90%
    HIGH = "high"              # 70-90%
    MODERATE = "moderate"      # 50-70%
    LOW = "low"                # 30-50%
    VERY_LOW = "very_low"      # <30%


@dataclass
class CausalEvent:
    """An event or phenomenon in causal reasoning"""
    
    # Core identification
    id: str
    description: str
    event_type: EventType
    
    # Event characteristics
    variables: List[str] = field(default_factory=list)
    measurements: Dict[str, float] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal information
    temporal_position: Optional[str] = None
    duration: Optional[str] = None
    frequency: Optional[str] = None
    
    # Context and domain
    domain: str = "general"
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Relevance and importance
    relevance_score: float = 0.0
    importance_score: float = 0.0
    observability: float = 0.0
    
    # Relationships
    related_events: List[str] = field(default_factory=list)
    
    # Quality assessment
    evidence_quality: float = 0.0
    reliability: float = 0.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_overall_score(self) -> float:
        """Calculate overall event score"""
        factors = [
            self.relevance_score,
            self.importance_score,
            self.evidence_quality,
            self.reliability,
            self.observability
        ]
        return sum(factors) / len(factors)


@dataclass
class PotentialCause:
    """A potential cause identified for an effect"""
    
    # Core identification
    id: str
    description: str
    generation_strategy: CauseGenerationStrategy
    cause_variables: List[str] = field(default_factory=list)
    
    # Generation information
    source: str = "unknown"
    
    # Causal properties
    plausibility: float = 0.0
    theoretical_support: float = 0.0
    empirical_support: float = 0.0
    
    # Evidence and validation
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    
    # Mechanism information
    proposed_mechanism: str = ""
    mechanism_type: CausalMechanism = CausalMechanism.UNKNOWN
    
    # Temporal properties
    temporal_precedence: float = 0.0
    temporal_relationship: TemporalRelation = TemporalRelation.MEDIUM_TERM
    
    # Domain and context
    domain: str = "general"
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    validation_tests: List[str] = field(default_factory=list)
    validation_results: Dict[str, float] = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_overall_score(self) -> float:
        """Calculate overall potential cause score"""
        factors = [
            self.plausibility,
            self.theoretical_support,
            self.empirical_support,
            self.temporal_precedence
        ]
        return sum(factors) / len(factors)


@dataclass
class CausalLink:
    """A causal link between cause and effect"""
    
    # Core identification
    id: str
    cause_id: str
    effect_id: str
    
    # Relationship properties
    relationship_type: CausalRelationType
    causal_strength: float = 0.0
    strength_category: CausalStrength = CausalStrength.MODERATE
    
    # Evidence and support
    evidence_strength: float = 0.0
    statistical_support: float = 0.0
    experimental_support: float = 0.0
    theoretical_support: float = 0.0
    
    # Mechanism information
    mechanism: str = ""
    mechanism_type: CausalMechanism = CausalMechanism.UNKNOWN
    mechanism_confidence: float = 0.0
    
    # Temporal properties
    temporal_relationship: TemporalRelation = TemporalRelation.MEDIUM_TERM
    temporal_consistency: float = 0.0
    
    # Dose-response information
    dose_response: bool = False
    dose_response_strength: float = 0.0
    
    # Validation and testing
    validation_tests: List[str] = field(default_factory=list)
    validation_results: Dict[str, float] = field(default_factory=dict)
    
    # Conditions and context
    necessary_conditions: List[str] = field(default_factory=list)
    sufficient_conditions: List[str] = field(default_factory=list)
    moderating_factors: List[str] = field(default_factory=list)
    
    # Confidence and uncertainty
    confidence: float = 0.0
    uncertainty_sources: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update_strength_category(self):
        """Update strength category based on numerical strength"""
        if self.causal_strength >= 0.8:
            self.strength_category = CausalStrength.VERY_STRONG
        elif self.causal_strength >= 0.6:
            self.strength_category = CausalStrength.STRONG
        elif self.causal_strength >= 0.4:
            self.strength_category = CausalStrength.MODERATE
        elif self.causal_strength >= 0.2:
            self.strength_category = CausalStrength.WEAK
        else:
            self.strength_category = CausalStrength.VERY_WEAK
    
    def get_overall_score(self) -> float:
        """Calculate overall causal link score"""
        factors = [
            self.causal_strength,
            self.evidence_strength,
            self.mechanism_confidence,
            self.temporal_consistency,
            self.confidence
        ]
        return sum(factors) / len(factors)


@dataclass
class ConfoundingFactor:
    """A confounding factor in causal analysis"""
    
    # Core identification
    id: str
    variable_name: str
    description: str
    
    # Confounding properties
    confounding_type: ConfoundingType
    confounding_strength: float = 0.0
    
    # Relationship to main variables
    relationship_to_cause: str = ""
    relationship_to_effect: str = ""
    
    # Control methods
    control_methods: List[str] = field(default_factory=list)
    control_feasibility: float = 0.0
    
    # Impact assessment
    bias_magnitude: float = 0.0
    bias_direction: str = ""  # "positive", "negative", "unknown"
    
    # Evidence and detection
    detection_method: str = ""
    detection_confidence: float = 0.0
    
    # Validation
    validation_tests: List[str] = field(default_factory=list)
    validation_results: Dict[str, float] = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_impact_score(self) -> float:
        """Calculate impact score of confounding"""
        return self.confounding_strength * self.bias_magnitude


@dataclass
class CausalInference:
    """Causal inference and prediction"""
    
    # Core identification
    id: str
    causal_link_id: str
    inference_type: str  # "predictive", "explanatory", "interventional"
    
    # Inference content
    inference_statement: str
    predictions: List[str] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    
    # Intervention analysis
    intervention_recommendations: List[str] = field(default_factory=list)
    intervention_effects: Dict[str, float] = field(default_factory=dict)
    
    # Confidence and uncertainty
    confidence: float = 0.0
    uncertainty_sources: List[str] = field(default_factory=list)
    
    # Validation and testing
    testable_predictions: List[str] = field(default_factory=list)
    validation_methods: List[str] = field(default_factory=list)
    
    # Practical implications
    practical_implications: List[str] = field(default_factory=list)
    decision_guidance: List[str] = field(default_factory=list)
    
    # Limitations
    limitations: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    
    # Additional fields for constructor compatibility
    interventions: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    confidence_level: Optional[Any] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    future_research: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CausalReasoning:
    """Complete causal reasoning process with all elemental components"""
    
    id: str
    query: str
    
    # Elemental components
    observed_events: List[CausalEvent]
    potential_causes: List[PotentialCause]
    causal_links: List[CausalLink]
    confounding_factors: List[ConfoundingFactor]
    causal_inferences: List[CausalInference]
    
    # Primary results
    primary_causal_link: CausalLink
    conclusion: str
    confidence_level: ConfidenceLevel
    
    # Process metadata
    reasoning_quality: float = 0.0
    processing_time: float = 0.0
    causal_certainty: float = 0.0
    
    # Alternative explanations
    alternative_explanations: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventObservationEngine:
    """Engine for comprehensive event observation and identification"""
    
    def __init__(self):
        self.event_types = self._initialize_event_types()
        self.observation_strategies = self._initialize_observation_strategies()
        self.domain_analyzers = self._initialize_domain_analyzers()
    
    async def observe_events(self, raw_observations: List[str], context: Dict[str, Any] = None) -> List[CausalEvent]:
        """Observe and identify events suggesting causal relationships"""
        
        logger.info(f"Observing events from {len(raw_observations)} observations")
        
        events = []
        
        for i, observation in enumerate(raw_observations):
            # Extract events from observation
            extracted_events = await self._extract_events_from_observation(observation, i, context)
            events.extend(extracted_events)
        
        # Classify and analyze events
        classified_events = await self._classify_events(events)
        
        # Assess event quality and relevance
        assessed_events = await self._assess_event_quality(classified_events)
        
        # Identify relationships between events
        related_events = await self._identify_event_relationships(assessed_events)
        
        # Filter and rank events
        filtered_events = await self._filter_and_rank_events(related_events)
        
        logger.info(f"Identified {len(filtered_events)} significant events")
        
        return filtered_events
    
    async def _extract_events_from_observation(self, observation: str, obs_id: int, context: Dict[str, Any] = None) -> List[CausalEvent]:
        """Extract events from a single observation"""
        
        events = []
        
        # Extract potential effects (outcomes)
        effects = await self._extract_effects(observation)
        
        # Extract potential causes
        causes = await self._extract_causes(observation)
        
        # Extract other event types
        other_events = await self._extract_other_events(observation)
        
        # Create event objects
        for i, effect in enumerate(effects):
            event = CausalEvent(
                id=f"event_{obs_id}_{i}_effect",
                description=effect,
                event_type=EventType.EFFECT,
                domain=await self._determine_domain(observation, context),
                context=context or {}
            )
            events.append(event)
        
        for i, cause in enumerate(causes):
            event = CausalEvent(
                id=f"event_{obs_id}_{i}_cause",
                description=cause,
                event_type=EventType.CAUSE,
                domain=await self._determine_domain(observation, context),
                context=context or {}
            )
            events.append(event)
        
        for i, other in enumerate(other_events):
            event = CausalEvent(
                id=f"event_{obs_id}_{i}_other",
                description=other,
                event_type=EventType.BACKGROUND,
                domain=await self._determine_domain(observation, context),
                context=context or {}
            )
            events.append(event)
        
        return events
    
    async def _extract_effects(self, observation: str) -> List[str]:
        """Extract potential effects from observation"""
        
        effects = []
        
        # Effect indicator patterns
        effect_patterns = [
            r'(.+?)\s+occurred',
            r'(.+?)\s+happened',
            r'(.+?)\s+resulted',
            r'(.+?)\s+was\s+observed',
            r'(.+?)\s+was\s+found',
            r'(.+?)\s+developed',
            r'(.+?)\s+appeared',
            r'(.+?)\s+increased',
            r'(.+?)\s+decreased',
            r'(.+?)\s+changed',
            r'outcome\s+was\s+(.+)',
            r'effect\s+was\s+(.+)',
            r'result\s+was\s+(.+)',
            r'consequence\s+was\s+(.+)'
        ]
        
        for pattern in effect_patterns:
            matches = re.findall(pattern, observation, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                effects.append(match.strip())
        
        return effects[:5]  # Limit to top 5
    
    async def _extract_causes(self, observation: str) -> List[str]:
        """Extract potential causes from observation"""
        
        causes = []
        
        # Cause indicator patterns
        cause_patterns = [
            r'(.+?)\s+causes?',
            r'(.+?)\s+leads?\s+to',
            r'(.+?)\s+results?\s+in',
            r'(.+?)\s+produces?',
            r'(.+?)\s+triggers?',
            r'(.+?)\s+influences?',
            r'(.+?)\s+affects?',
            r'because\s+of\s+(.+)',
            r'due\s+to\s+(.+)',
            r'caused\s+by\s+(.+)',
            r'triggered\s+by\s+(.+)',
            r'as\s+a\s+result\s+of\s+(.+)',
            r'following\s+(.+)',
            r'after\s+(.+)',
        ]
        
        for pattern in cause_patterns:
            matches = re.findall(pattern, observation, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                causes.append(match.strip())
        
        return causes[:5]  # Limit to top 5
    
    async def _extract_other_events(self, observation: str) -> List[str]:
        """Extract other types of events"""
        
        events = []
        
        # Extract temporal markers
        temporal_patterns = [
            r'(before\s+.+)',
            r'(after\s+.+)',
            r'(during\s+.+)',
            r'(while\s+.+)',
            r'(when\s+.+)',
            r'(simultaneously\s+.+)',
            r'(previously\s+.+)',
            r'(subsequently\s+.+)'
        ]
        
        for pattern in temporal_patterns:
            matches = re.findall(pattern, observation, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                events.append(match.strip())
        
        # Extract contextual information
        context_patterns = [
            r'(in\s+the\s+context\s+of\s+.+)',
            r'(under\s+conditions\s+of\s+.+)',
            r'(given\s+that\s+.+)',
            r'(assuming\s+.+)',
            r'(provided\s+that\s+.+)'
        ]
        
        for pattern in context_patterns:
            matches = re.findall(pattern, observation, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                events.append(match.strip())
        
        return events[:3]  # Limit to top 3
    
    async def _determine_domain(self, observation: str, context: Dict[str, Any] = None) -> str:
        """Determine the domain of the observation"""
        
        # Check context first
        if context and "domain" in context:
            return context["domain"]
        
        # Handle case where observation is a list
        if isinstance(observation, list):
            observation = ' '.join(str(item) for item in observation)
        elif not isinstance(observation, str):
            observation = str(observation)
        
        # Domain classification based on keywords
        domain_keywords = {
            "medical": ["patient", "symptom", "diagnosis", "treatment", "disease", "health", "medicine", "clinical", "therapy", "drug"],
            "scientific": ["experiment", "hypothesis", "data", "research", "study", "analysis", "theory", "scientific", "laboratory", "test"],
            "technical": ["system", "component", "failure", "error", "malfunction", "technical", "engineering", "software", "hardware", "computer"],
            "social": ["behavior", "society", "culture", "group", "social", "interaction", "community", "people", "population", "human"],
            "economic": ["market", "financial", "economic", "business", "trade", "money", "cost", "profit", "investment", "economy"],
            "environmental": ["climate", "weather", "environmental", "ecology", "nature", "pollution", "ecosystem", "atmosphere", "temperature"],
            "psychological": ["behavior", "mental", "cognitive", "emotional", "psychological", "mind", "consciousness", "perception", "memory"],
            "biological": ["organism", "biology", "life", "cell", "genetic", "evolution", "species", "biological", "protein", "gene"],
            "physical": ["physics", "force", "energy", "matter", "motion", "physical", "mechanical", "electrical", "chemical", "material"],
            "educational": ["student", "learning", "education", "academic", "school", "teaching", "knowledge", "instruction", "curriculum"]
        }
        
        obs_lower = str(observation).lower()
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
    
    async def _classify_events(self, events: List[CausalEvent]) -> List[CausalEvent]:
        """Classify events into appropriate types"""
        
        classified_events = []
        
        for event in events:
            # Refine event type based on description
            refined_type = await self._refine_event_type(event)
            event.event_type = refined_type
            
            # Extract variables
            event.variables = await self._extract_variables(event.description)
            
            # Extract measurements
            event.measurements = await self._extract_measurements(event.description)
            
            # Extract properties
            event.properties = await self._extract_properties(event.description)
            
            # Extract temporal information
            event.temporal_position = await self._extract_temporal_position(event.description)
            
            classified_events.append(event)
        
        return classified_events
    
    async def _refine_event_type(self, event: CausalEvent) -> EventType:
        """Refine event type based on description"""
        
        # Handle case where description is a list
        description = event.description
        if isinstance(description, list):
            description = ' '.join(str(item) for item in description)
        elif not isinstance(description, str):
            description = str(description)
        
        desc_lower = str(description).lower()
        
        # Check for specific event type indicators
        if any(indicator in desc_lower for indicator in ["outcome", "result", "consequence", "effect"]):
            return EventType.EFFECT
        
        if any(indicator in desc_lower for indicator in ["cause", "reason", "factor", "trigger"]):
            return EventType.CAUSE
        
        if any(indicator in desc_lower for indicator in ["confound", "bias", "artifact"]):
            return EventType.CONFOUNDING
        
        if any(indicator in desc_lower for indicator in ["mediat", "through", "via", "pathway"]):
            return EventType.MEDIATING
        
        if any(indicator in desc_lower for indicator in ["moderat", "interact", "depend"]):
            return EventType.MODERATING
        
        if any(indicator in desc_lower for indicator in ["spurious", "coincidence", "random"]):
            return EventType.SPURIOUS
        
        if any(indicator in desc_lower for indicator in ["context", "background", "condition"]):
            return EventType.CONTEXTUAL
        
        if any(indicator in desc_lower for indicator in ["time", "temporal", "sequence", "order"]):
            return EventType.TEMPORAL
        
        if any(indicator in desc_lower for indicator in ["intermediate", "step", "stage"]):
            return EventType.INTERMEDIATE
        
        # Default to existing type
        return event.event_type
    
    async def _extract_variables(self, description: str) -> List[str]:
        """Extract variables from event description"""
        
        variables = []
        
        # Extract nouns and noun phrases
        variable_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Proper nouns
            r'\b([a-z]+(?:\s+[a-z]+)*)\s+(?:level|rate|count|amount|degree|intensity)\b',  # Quantitative variables
            r'\b(?:the|a|an)\s+([a-z]+(?:\s+[a-z]+)*)\b',  # Articles + nouns
        ]
        
        for pattern in variable_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if str(match).lower() not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                    variables.append(match.strip())
        
        return variables[:5]  # Limit to top 5
    
    async def _extract_measurements(self, description: str) -> Dict[str, float]:
        """Extract measurements from event description"""
        
        measurements = {}
        
        # Extract numerical measurements
        measurement_patterns = [
            r'(\d+\.?\d*)\s*(percent|%|degrees?|units?|mg|kg|g|lb|oz|cm|mm|m|km|seconds?|minutes?|hours?|days?|years?)',
            r'(\d+\.?\d*)\s*(increased?|decreased?|changed?|rose|fell|dropped)',
            r'(\d+\.?\d*)\s*(times?|fold|factor)',
            r'(\d+\.?\d*)\s*(higher|lower|greater|less|more|fewer)'
        ]
        
        for pattern in measurement_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for value, unit in matches:
                try:
                    measurements[f"measurement_{unit}"] = float(value)
                except ValueError:
                    continue
        
        return measurements
    
    async def _extract_properties(self, description: str) -> Dict[str, Any]:
        """Extract properties from event description"""
        
        properties = {}
        
        # Extract qualitative properties
        property_patterns = [
            r'is\s+(very\s+)?(\w+)',
            r'was\s+(very\s+)?(\w+)',
            r'became\s+(very\s+)?(\w+)',
            r'appears?\s+(very\s+)?(\w+)',
            r'seems?\s+(very\s+)?(\w+)'
        ]
        
        for pattern in property_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for modifier, property_name in matches:
                intensity = "high" if modifier else "normal"
                properties[f"property_{property_name}"] = intensity
        
        # Extract categorical properties
        # Handle case where description is a list
        if isinstance(description, list):
            description = ' '.join(str(item) for item in description)
        elif not isinstance(description, str):
            description = str(description)
        
        if "type" in str(description).lower():
            type_match = re.search(r'type\s+of\s+(\w+)', description, re.IGNORECASE)
            if type_match:
                properties["type"] = type_match.group(1)
        
        return properties
    
    async def _extract_temporal_position(self, description: str) -> Optional[str]:
        """Extract temporal position from description"""
        
        temporal_indicators = [
            "before", "after", "during", "while", "when", "then", "next", "subsequently", 
            "previously", "initially", "finally", "first", "last", "earlier", "later",
            "immediately", "soon", "eventually", "gradually", "suddenly", "instantly"
        ]
        
        # Handle case where description is a list
        if isinstance(description, list):
            description = ' '.join(str(item) for item in description)
        elif not isinstance(description, str):
            description = str(description)
        
        desc_lower = str(description).lower()
        
        for indicator in temporal_indicators:
            if indicator in desc_lower:
                return indicator
        
        return None
    
    async def _assess_event_quality(self, events: List[CausalEvent]) -> List[CausalEvent]:
        """Assess quality and relevance of events"""
        
        assessed_events = []
        
        for event in events:
            # Assess relevance
            event.relevance_score = await self._assess_relevance(event)
            
            # Assess importance
            event.importance_score = await self._assess_importance(event)
            
            # Assess observability
            event.observability = await self._assess_observability(event)
            
            # Assess evidence quality
            event.evidence_quality = await self._assess_evidence_quality(event)
            
            # Assess reliability
            event.reliability = await self._assess_reliability(event)
            
            assessed_events.append(event)
        
        return assessed_events
    
    async def _assess_relevance(self, event: CausalEvent) -> float:
        """Assess relevance of event to causal analysis"""
        
        relevance = 0.6  # Base relevance
        
        # Event type relevance
        type_weights = {
            EventType.EFFECT: 0.9,
            EventType.CAUSE: 0.9,
            EventType.CONFOUNDING: 0.8,
            EventType.MEDIATING: 0.7,
            EventType.MODERATING: 0.6,
            EventType.INTERMEDIATE: 0.7,
            EventType.TEMPORAL: 0.5,
            EventType.CONTEXTUAL: 0.4,
            EventType.BACKGROUND: 0.3,
            EventType.SPURIOUS: 0.2
        }
        
        relevance = type_weights.get(event.event_type, 0.5)
        
        # Adjust for variables
        if event.variables:
            relevance += min(0.2, len(event.variables) * 0.05)
        
        # Adjust for measurements
        if event.measurements:
            relevance += min(0.2, len(event.measurements) * 0.05)
        
        # Adjust for temporal information
        if event.temporal_position:
            relevance += 0.1
        
        return max(0.0, min(1.0, relevance))
    
    async def _assess_importance(self, event: CausalEvent) -> float:
        """Assess importance of event"""
        
        importance = 0.5  # Base importance
        
        # Domain-specific importance
        domain_weights = {
            "medical": 0.9,
            "scientific": 0.8,
            "technical": 0.7,
            "social": 0.6,
            "economic": 0.7,
            "environmental": 0.8,
            "psychological": 0.6,
            "biological": 0.8,
            "physical": 0.7,
            "educational": 0.6
        }
        
        importance = domain_weights.get(event.domain, 0.5)
        
        # Adjust for event type
        if event.event_type in [EventType.EFFECT, EventType.CAUSE]:
            importance += 0.2
        elif event.event_type == EventType.CONFOUNDING:
            importance += 0.1
        
        # Adjust for complexity
        if len(event.variables) > 3:
            importance += 0.1
        
        return max(0.0, min(1.0, importance))
    
    async def _assess_observability(self, event: CausalEvent) -> float:
        """Assess observability of event"""
        
        observability = 0.7  # Base observability
        
        # Observability indicators
        observable_indicators = ["observed", "measured", "recorded", "documented", "reported", "detected"]
        unobservable_indicators = ["inferred", "assumed", "hypothetical", "theoretical", "supposed"]
        
        # Handle case where description is a list
        description = event.description
        if isinstance(description, list):
            description = ' '.join(str(item) for item in description)
        elif not isinstance(description, str):
            description = str(description)
        
        desc_lower = str(description).lower()
        
        for indicator in observable_indicators:
            if indicator in desc_lower:
                observability += 0.2
                break
        
        for indicator in unobservable_indicators:
            if indicator in desc_lower:
                observability -= 0.2
                break
        
        # Adjust for measurements
        if event.measurements:
            observability += 0.1
        
        return max(0.0, min(1.0, observability))
    
    async def _assess_evidence_quality(self, event: CausalEvent) -> float:
        """Assess quality of evidence for event"""
        
        quality = 0.6  # Base quality
        
        # Quality indicators
        high_quality_indicators = ["experimental", "controlled", "randomized", "peer-reviewed", "validated"]
        low_quality_indicators = ["anecdotal", "preliminary", "unverified", "suspected", "alleged"]
        
        # Handle case where description is a list
        description = event.description
        if isinstance(description, list):
            description = ' '.join(str(item) for item in description)
        elif not isinstance(description, str):
            description = str(description)
        
        desc_lower = str(description).lower()
        
        for indicator in high_quality_indicators:
            if indicator in desc_lower:
                quality += 0.2
                break
        
        for indicator in low_quality_indicators:
            if indicator in desc_lower:
                quality -= 0.2
                break
        
        # Adjust for specificity
        if event.measurements:
            quality += 0.1
        
        if event.variables:
            quality += 0.1
        
        return max(0.0, min(1.0, quality))
    
    async def _assess_reliability(self, event: CausalEvent) -> float:
        """Assess reliability of event"""
        
        reliability = 0.7  # Base reliability
        
        # Reliability indicators
        reliable_indicators = ["consistent", "repeatable", "reproducible", "verified", "confirmed"]
        unreliable_indicators = ["inconsistent", "variable", "unreliable", "questionable", "disputed"]
        
        # Handle case where description is a list
        description = event.description
        if isinstance(description, list):
            description = ' '.join(str(item) for item in description)
        elif not isinstance(description, str):
            description = str(description)
        
        desc_lower = str(description).lower()
        
        for indicator in reliable_indicators:
            if indicator in desc_lower:
                reliability += 0.2
                break
        
        for indicator in unreliable_indicators:
            if indicator in desc_lower:
                reliability -= 0.2
                break
        
        # Adjust for evidence quality
        reliability = (reliability + event.evidence_quality) / 2
        
        return max(0.0, min(1.0, reliability))
    
    async def _identify_event_relationships(self, events: List[CausalEvent]) -> List[CausalEvent]:
        """Identify relationships between events"""
        
        for i, event in enumerate(events):
            related_events = []
            
            for j, other_event in enumerate(events):
                if i != j:
                    # Check for relationships
                    relationship_score = await self._calculate_event_similarity(event, other_event)
                    
                    if relationship_score > 0.5:
                        related_events.append(other_event.id)
            
            event.related_events = related_events
        
        return events
    
    async def _calculate_event_similarity(self, event1: CausalEvent, event2: CausalEvent) -> float:
        """Calculate similarity between events"""
        
        # Domain similarity
        domain_similarity = 1.0 if event1.domain == event2.domain else 0.0
        
        # Variable overlap
        vars1 = set(event1.variables)
        vars2 = set(event2.variables)
        
        if vars1 and vars2:
            variable_similarity = len(vars1 & vars2) / len(vars1 | vars2)
        else:
            variable_similarity = 0.0
        
        # Description similarity (simple word overlap)
        # Handle case where descriptions are lists
        desc1 = event1.description
        if isinstance(desc1, list):
            desc1 = ' '.join(str(item) for item in desc1)
        elif not isinstance(desc1, str):
            desc1 = str(desc1)
        
        desc2 = event2.description
        if isinstance(desc2, list):
            desc2 = ' '.join(str(item) for item in desc2)
        elif not isinstance(desc2, str):
            desc2 = str(desc2)
        
        words1 = set(str(desc1).lower().split())
        words2 = set(str(desc2).lower().split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        words1 -= stop_words
        words2 -= stop_words
        
        if words1 and words2:
            description_similarity = len(words1 & words2) / len(words1 | words2)
        else:
            description_similarity = 0.0
        
        # Weighted combination
        similarity = (
            0.3 * domain_similarity +
            0.4 * variable_similarity +
            0.3 * description_similarity
        )
        
        return similarity
    
    async def _filter_and_rank_events(self, events: List[CausalEvent]) -> List[CausalEvent]:
        """Filter and rank events by quality"""
        
        # Filter out low-quality events
        filtered_events = [
            event for event in events
            if event.get_overall_score() > 0.3
        ]
        
        # Sort by overall score
        filtered_events.sort(key=lambda e: e.get_overall_score(), reverse=True)
        
        return filtered_events
    
    def _initialize_event_types(self) -> Dict[str, Any]:
        """Initialize event type configurations"""
        return {
            "effect": {"weight": 0.9, "indicators": ["outcome", "result", "consequence"]},
            "cause": {"weight": 0.9, "indicators": ["cause", "reason", "factor"]},
            "confounding": {"weight": 0.8, "indicators": ["confound", "bias", "artifact"]},
            "mediating": {"weight": 0.7, "indicators": ["mediat", "through", "via"]},
            "moderating": {"weight": 0.6, "indicators": ["moderat", "interact", "depend"]},
            "intermediate": {"weight": 0.7, "indicators": ["intermediate", "step", "stage"]},
            "temporal": {"weight": 0.5, "indicators": ["time", "temporal", "sequence"]},
            "contextual": {"weight": 0.4, "indicators": ["context", "background", "condition"]},
            "background": {"weight": 0.3, "indicators": ["background", "setting", "environment"]},
            "spurious": {"weight": 0.2, "indicators": ["spurious", "coincidence", "random"]}
        }
    
    def _initialize_observation_strategies(self) -> Dict[str, Any]:
        """Initialize observation strategies"""
        return {
            "effect_identification": {"focus": "identifying outcomes and effects"},
            "cause_extraction": {"focus": "extracting potential causes"},
            "temporal_analysis": {"focus": "temporal relationships"},
            "context_analysis": {"focus": "contextual factors"},
            "variable_extraction": {"focus": "identifying variables"}
        }
    
    def _initialize_domain_analyzers(self) -> Dict[str, Any]:
        """Initialize domain-specific analyzers"""
        return {
            "medical": {"keywords": ["patient", "symptom", "diagnosis", "treatment"], "weight": 0.9},
            "scientific": {"keywords": ["experiment", "hypothesis", "data", "research"], "weight": 0.8},
            "technical": {"keywords": ["system", "component", "failure", "error"], "weight": 0.7},
            "social": {"keywords": ["behavior", "society", "culture", "group"], "weight": 0.6},
            "economic": {"keywords": ["market", "financial", "business", "trade"], "weight": 0.7}
        }


class PotentialCauseIdentificationEngine:
    """Engine for identifying potential causes of observed effects"""
    
    def __init__(self):
        self.generation_strategies = self._initialize_generation_strategies()
        self.domain_knowledge = self._initialize_domain_knowledge()
        self.causal_theories = self._initialize_causal_theories()
    
    async def identify_potential_causes(self, events: List[CausalEvent], context: Dict[str, Any] = None) -> List[PotentialCause]:
        """Identify potential causes for observed effects"""
        
        logger.info(f"Identifying potential causes for {len(events)} events")
        
        # Find effect events
        effect_events = [event for event in events if event.event_type == EventType.EFFECT]
        
        if not effect_events:
            logger.warning("No effect events found for cause identification")
            return []
        
        all_potential_causes = []
        
        # Generate potential causes for each effect
        for effect_event in effect_events:
            # Multiple generation strategies
            for strategy in self.generation_strategies:
                strategy_causes = await self._apply_generation_strategy(strategy, effect_event, events, context)
                all_potential_causes.extend(strategy_causes)
        
        # Remove duplicates and enhance
        unique_causes = await self._remove_duplicate_causes(all_potential_causes)
        enhanced_causes = await self._enhance_potential_causes(unique_causes, events)
        
        # Validate and rank
        validated_causes = await self._validate_potential_causes(enhanced_causes, events)
        ranked_causes = await self._rank_potential_causes(validated_causes)
        
        logger.info(f"Identified {len(ranked_causes)} potential causes")
        
        return ranked_causes
    
    async def _apply_generation_strategy(self, strategy: str, effect_event: CausalEvent, events: List[CausalEvent], context: Dict[str, Any] = None) -> List[PotentialCause]:
        """Apply a specific generation strategy"""
        
        if strategy == "domain_knowledge":
            return await self._generate_domain_knowledge_causes(effect_event, context)
        elif strategy == "temporal_precedence":
            return await self._generate_temporal_precedence_causes(effect_event, events)
        elif strategy == "correlation_analysis":
            return await self._generate_correlation_causes(effect_event, events)
        elif strategy == "mechanism_theory":
            return await self._generate_mechanism_theory_causes(effect_event, context)
        elif strategy == "analogical":
            return await self._generate_analogical_causes(effect_event, context)
        elif strategy == "eliminative":
            return await self._generate_eliminative_causes(effect_event, events)
        elif strategy == "empirical":
            return await self._generate_empirical_causes(effect_event, events)
        else:
            return []
    
    async def _generate_domain_knowledge_causes(self, effect_event: CausalEvent, context: Dict[str, Any] = None) -> List[PotentialCause]:
        """Generate causes based on domain knowledge"""
        
        potential_causes = []
        
        # Domain-specific common causes
        domain_causes = {
            "medical": [
                "infection", "inflammation", "genetic factor", "environmental factor", 
                "medication", "lifestyle factor", "age", "gender", "stress", "diet"
            ],
            "technical": [
                "hardware failure", "software bug", "configuration error", "resource exhaustion",
                "network issue", "power failure", "overheating", "wear and tear", "human error"
            ],
            "scientific": [
                "experimental error", "measurement noise", "environmental factor", "instrument drift",
                "systematic bias", "random variation", "confounding variable", "sample contamination"
            ],
            "social": [
                "cultural factor", "economic factor", "social pressure", "group dynamics",
                "individual differences", "environmental influence", "historical context", "policy change"
            ],
            "economic": [
                "market forces", "policy change", "technological change", "demographic shift",
                "global events", "competition", "regulation", "consumer behavior"
            ]
        }
        
        causes = domain_causes.get(effect_event.domain, ["unknown factor"])
        
        for i, cause in enumerate(causes):
            potential_cause = PotentialCause(
                id=f"domain_{effect_event.id}_{i}",
                description=cause,
                generation_strategy=CauseGenerationStrategy.DOMAIN_KNOWLEDGE,
                source=f"domain knowledge ({effect_event.domain})",
                plausibility=0.7,
                theoretical_support=0.8,
                domain=effect_event.domain,
                context=context or {}
            )
            potential_causes.append(potential_cause)
        
        return potential_causes
    
    async def _generate_temporal_precedence_causes(self, effect_event: CausalEvent, events: List[CausalEvent]) -> List[PotentialCause]:
        """Generate causes based on temporal precedence"""
        
        potential_causes = []
        
        # Look for events that occurred before the effect
        temporal_indicators = {
            "before": 0.9,
            "previously": 0.8,
            "earlier": 0.7,
            "first": 0.9,
            "initially": 0.8,
            "prior": 0.8
        }
        
        for event in events:
            if event != effect_event and event.temporal_position:
                # Check if this event preceded the effect
                precedence_score = temporal_indicators.get(event.temporal_position, 0.5)
                
                if precedence_score > 0.6:
                    potential_cause = PotentialCause(
                        id=f"temporal_{effect_event.id}_{event.id}",
                        description=f"Temporal precedence: {event.description}",
                        generation_strategy=CauseGenerationStrategy.TEMPORAL_PRECEDENCE,
                        source=f"temporal analysis of {event.id}",
                        plausibility=0.6,
                        temporal_precedence=precedence_score,
                        temporal_relationship=TemporalRelation.MEDIUM_TERM,
                        domain=effect_event.domain
                    )
                    potential_causes.append(potential_cause)
        
        return potential_causes
    
    async def _generate_correlation_causes(self, effect_event: CausalEvent, events: List[CausalEvent]) -> List[PotentialCause]:
        """Generate causes based on correlation analysis"""
        
        potential_causes = []
        
        # Look for correlated events
        for event in events:
            if event != effect_event and event.event_type != EventType.EFFECT:
                # Simple correlation based on variable overlap
                correlation_score = await self._calculate_event_correlation(effect_event, event)
                
                if correlation_score > 0.5:
                    potential_cause = PotentialCause(
                        id=f"correlation_{effect_event.id}_{event.id}",
                        description=f"Correlation-based: {event.description}",
                        generation_strategy=CauseGenerationStrategy.CORRELATION_ANALYSIS,
                        source=f"correlation analysis with {event.id}",
                        plausibility=0.5,
                        empirical_support=correlation_score,
                        domain=effect_event.domain
                    )
                    potential_causes.append(potential_cause)
        
        return potential_causes
    
    async def _calculate_event_correlation(self, event1: CausalEvent, event2: CausalEvent) -> float:
        """Calculate correlation between events"""
        
        # Variable overlap
        vars1 = set(event1.variables)
        vars2 = set(event2.variables)
        
        if vars1 and vars2:
            variable_overlap = len(vars1 & vars2) / len(vars1 | vars2)
        else:
            variable_overlap = 0.0
        
        # Domain consistency
        domain_consistency = 1.0 if event1.domain == event2.domain else 0.0
        
        # Measurement correlation (simple)
        measurement_correlation = 0.0
        if event1.measurements and event2.measurements:
            common_measures = set(event1.measurements.keys()) & set(event2.measurements.keys())
            if common_measures:
                measurement_correlation = 0.5  # Simplified
        
        # Weighted combination
        correlation = (
            0.5 * variable_overlap +
            0.3 * domain_consistency +
            0.2 * measurement_correlation
        )
        
        return correlation
    
    async def _generate_mechanism_theory_causes(self, effect_event: CausalEvent, context: Dict[str, Any] = None) -> List[PotentialCause]:
        """Generate causes based on theoretical mechanisms"""
        
        potential_causes = []
        
        # Mechanism-based causes
        mechanism_causes = {
            "medical": [
                {"cause": "pathophysiological process", "mechanism": "disease mechanism"},
                {"cause": "pharmacological effect", "mechanism": "drug action"},
                {"cause": "immunological response", "mechanism": "immune system"},
                {"cause": "genetic expression", "mechanism": "gene regulation"}
            ],
            "technical": [
                {"cause": "system overload", "mechanism": "resource exhaustion"},
                {"cause": "component failure", "mechanism": "mechanical breakdown"},
                {"cause": "software error", "mechanism": "code execution"},
                {"cause": "network congestion", "mechanism": "data transmission"}
            ],
            "scientific": [
                {"cause": "physical process", "mechanism": "natural law"},
                {"cause": "chemical reaction", "mechanism": "molecular interaction"},
                {"cause": "environmental factor", "mechanism": "ecological process"},
                {"cause": "measurement artifact", "mechanism": "instrument response"}
            ]
        }
        
        causes = mechanism_causes.get(effect_event.domain, [])
        
        for i, cause_info in enumerate(causes):
            potential_cause = PotentialCause(
                id=f"mechanism_{effect_event.id}_{i}",
                description=cause_info["cause"],
                generation_strategy=CauseGenerationStrategy.MECHANISM_THEORY,
                source="theoretical mechanism",
                plausibility=0.8,
                theoretical_support=0.9,
                proposed_mechanism=cause_info["mechanism"],
                domain=effect_event.domain
            )
            potential_causes.append(potential_cause)
        
        return potential_causes
    
    async def _generate_analogical_causes(self, effect_event: CausalEvent, context: Dict[str, Any] = None) -> List[PotentialCause]:
        """Generate causes based on analogies"""
        
        potential_causes = []
        
        # Analogical causes based on similar effects
        analogical_causes = {
            "medical": [
                "similar condition in related population",
                "analogous symptom in different context",
                "comparable biological process",
                "related pathological mechanism"
            ],
            "technical": [
                "similar failure in comparable system",
                "analogous error in related software",
                "comparable component degradation",
                "related performance issue"
            ],
            "scientific": [
                "similar phenomenon in related field",
                "analogous process in different domain",
                "comparable experimental result",
                "related theoretical prediction"
            ]
        }
        
        causes = analogical_causes.get(effect_event.domain, ["analogous process"])
        
        for i, cause in enumerate(causes):
            potential_cause = PotentialCause(
                id=f"analogical_{effect_event.id}_{i}",
                description=cause,
                generation_strategy=CauseGenerationStrategy.ANALOGICAL,
                source="analogical reasoning",
                plausibility=0.6,
                theoretical_support=0.7,
                domain=effect_event.domain
            )
            potential_causes.append(potential_cause)
        
        return potential_causes
    
    async def _generate_eliminative_causes(self, effect_event: CausalEvent, events: List[CausalEvent]) -> List[PotentialCause]:
        """Generate causes through eliminative reasoning"""
        
        potential_causes = []
        
        # Common alternative explanations to eliminate
        common_alternatives = [
            "measurement error",
            "sampling bias",
            "confounding variable",
            "chance occurrence",
            "systematic error",
            "observer bias",
            "instrument malfunction",
            "data processing error"
        ]
        
        for i, alternative in enumerate(common_alternatives):
            potential_cause = PotentialCause(
                id=f"eliminative_{effect_event.id}_{i}",
                description=f"After eliminating {alternative}: remaining explanation",
                generation_strategy=CauseGenerationStrategy.ELIMINATIVE,
                source="eliminative reasoning",
                plausibility=0.5,
                empirical_support=0.6,
                domain=effect_event.domain
            )
            potential_causes.append(potential_cause)
        
        return potential_causes
    
    async def _generate_empirical_causes(self, effect_event: CausalEvent, events: List[CausalEvent]) -> List[PotentialCause]:
        """Generate causes based on empirical patterns"""
        
        potential_causes = []
        
        # Look for empirical patterns in the data
        for event in events:
            if event != effect_event and event.measurements:
                # Check for dose-response patterns
                dose_response_score = await self._assess_dose_response(event, effect_event)
                
                if dose_response_score > 0.6:
                    potential_cause = PotentialCause(
                        id=f"empirical_{effect_event.id}_{event.id}",
                        description=f"Dose-response pattern: {event.description}",
                        generation_strategy=CauseGenerationStrategy.EMPIRICAL,
                        source="empirical pattern analysis",
                        plausibility=0.7,
                        empirical_support=dose_response_score,
                        domain=effect_event.domain
                    )
                    potential_causes.append(potential_cause)
        
        return potential_causes
    
    async def _assess_dose_response(self, potential_cause_event: CausalEvent, effect_event: CausalEvent) -> float:
        """Assess dose-response relationship"""
        
        # Simplified dose-response assessment
        if potential_cause_event.measurements and effect_event.measurements:
            # Look for quantitative relationships
            return 0.7  # Simplified
        
        # Look for qualitative dose-response indicators
        # Handle case where descriptions are lists
        cause_desc = potential_cause_event.description
        if isinstance(cause_desc, list):
            cause_desc = ' '.join(str(item) for item in cause_desc)
        elif not isinstance(cause_desc, str):
            cause_desc = str(cause_desc)
        
        effect_desc = effect_event.description
        if isinstance(effect_desc, list):
            effect_desc = ' '.join(str(item) for item in effect_desc)
        elif not isinstance(effect_desc, str):
            effect_desc = str(effect_desc)
        
        desc_lower = f"{str(cause_desc)} {str(effect_desc)}".lower()
        
        dose_response_indicators = [
            "more", "less", "increase", "decrease", "higher", "lower",
            "dose", "amount", "level", "intensity", "degree"
        ]
        
        if any(indicator in desc_lower for indicator in dose_response_indicators):
            return 0.6
        
        return 0.3
    
    async def _remove_duplicate_causes(self, potential_causes: List[PotentialCause]) -> List[PotentialCause]:
        """Remove duplicate potential causes"""
        
        unique_causes = []
        seen_descriptions = set()
        
        for cause in potential_causes:
            # Normalize description for comparison
            # Handle case where description is a list
            description = cause.description
            if isinstance(description, list):
                description = ' '.join(str(item) for item in description)
            elif not isinstance(description, str):
                description = str(description)
            
            normalized_desc = str(description).lower().strip()
            
            if normalized_desc not in seen_descriptions:
                unique_causes.append(cause)
                seen_descriptions.add(normalized_desc)
        
        return unique_causes
    
    async def _enhance_potential_causes(self, potential_causes: List[PotentialCause], events: List[CausalEvent]) -> List[PotentialCause]:
        """Enhance potential causes with additional information"""
        
        enhanced_causes = []
        
        for cause in potential_causes:
            # Extract variables
            cause.cause_variables = await self._extract_cause_variables(cause.description)
            
            # Determine mechanism type
            cause.mechanism_type = await self._determine_mechanism_type(cause)
            
            # Generate validation tests
            cause.validation_tests = await self._generate_validation_tests(cause)
            
            # Assess supporting evidence
            cause.supporting_evidence = await self._assess_supporting_evidence(cause, events)
            
            enhanced_causes.append(cause)
        
        return enhanced_causes
    
    async def _extract_cause_variables(self, description: str) -> List[str]:
        """Extract variables from cause description"""
        
        variables = []
        
        # Extract nouns and key terms
        variable_patterns = [
            r'\b([a-z]+(?:\s+[a-z]+)*)\s+(?:level|rate|amount|factor|process|mechanism)\b',
            r'\b(?:the|a|an)\s+([a-z]+(?:\s+[a-z]+)*)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
        
        for pattern in variable_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if str(match).lower() not in ['the', 'a', 'an', 'and', 'or', 'but']:
                    variables.append(match.strip())
        
        return variables[:3]  # Limit to top 3
    
    async def _determine_mechanism_type(self, cause: PotentialCause) -> CausalMechanism:
        """Determine mechanism type for cause"""
        
        # Handle case where description is a list
        description = cause.description
        if isinstance(description, list):
            description = ' '.join(str(item) for item in description)
        elif not isinstance(description, str):
            description = str(description)
        
        desc_lower = str(description).lower()
        
        # Mechanism type indicators
        mechanism_keywords = {
            CausalMechanism.PHYSICAL: ["physical", "mechanical", "force", "energy", "motion"],
            CausalMechanism.BIOLOGICAL: ["biological", "genetic", "cellular", "metabolic", "physiological"],
            CausalMechanism.CHEMICAL: ["chemical", "reaction", "molecular", "compound", "catalyst"],
            CausalMechanism.PSYCHOLOGICAL: ["psychological", "mental", "cognitive", "behavioral", "emotional"],
            CausalMechanism.SOCIAL: ["social", "cultural", "group", "community", "interaction"],
            CausalMechanism.ECONOMIC: ["economic", "financial", "market", "monetary", "cost"],
            CausalMechanism.INFORMATIONAL: ["information", "data", "signal", "communication", "message"],
            CausalMechanism.ELECTRICAL: ["electrical", "electronic", "current", "voltage", "circuit"],
            CausalMechanism.MECHANICAL: ["mechanical", "machine", "component", "mechanism", "device"]
        }
        
        for mechanism_type, keywords in mechanism_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                return mechanism_type
        
        # Default based on domain
        domain_mechanisms = {
            "medical": CausalMechanism.BIOLOGICAL,
            "technical": CausalMechanism.MECHANICAL,
            "scientific": CausalMechanism.PHYSICAL,
            "social": CausalMechanism.SOCIAL,
            "economic": CausalMechanism.ECONOMIC
        }
        
        return domain_mechanisms.get(cause.domain, CausalMechanism.UNKNOWN)
    
    async def _generate_validation_tests(self, cause: PotentialCause) -> List[str]:
        """Generate validation tests for potential cause"""
        
        tests = []
        
        # Strategy-based tests
        if cause.generation_strategy == CauseGenerationStrategy.TEMPORAL_PRECEDENCE:
            tests.append("Verify temporal sequence through longitudinal study")
        elif cause.generation_strategy == CauseGenerationStrategy.CORRELATION_ANALYSIS:
            tests.append("Test correlation significance and control for confounders")
        elif cause.generation_strategy == CauseGenerationStrategy.MECHANISM_THEORY:
            tests.append("Experimental validation of proposed mechanism")
        
        # Domain-specific tests
        if cause.domain == "medical":
            tests.extend([
                "Clinical trial to test causal relationship",
                "Dose-response study",
                "Mechanism validation through biological assays"
            ])
        elif cause.domain == "technical":
            tests.extend([
                "Controlled system test",
                "Component isolation test",
                "Failure mode analysis"
            ])
        elif cause.domain == "scientific":
            tests.extend([
                "Controlled experiment",
                "Replication study",
                "Mechanism investigation"
            ])
        
        # General tests
        tests.extend([
            "Experimental manipulation of proposed cause",
            "Control for confounding variables",
            "Test temporal precedence"
        ])
        
        return tests[:5]  # Limit to top 5
    
    async def _assess_supporting_evidence(self, cause: PotentialCause, events: List[CausalEvent]) -> List[str]:
        """Assess supporting evidence for potential cause"""
        
        supporting_evidence = []
        
        # Look for supporting evidence in events
        for event in events:
            # Handle case where description is a list
            description = event.description
            if isinstance(description, list):
                description = ' '.join(str(item) for item in description)
            elif not isinstance(description, str):
                description = str(description)
            
            # Check for mentions of the cause
            if any(str(var).lower() in str(description).lower() for var in cause.cause_variables):
                supporting_evidence.append(f"Mentioned in event: {description}")
            
            # Check for mechanism support
            if cause.proposed_mechanism and str(cause.proposed_mechanism).lower() in str(description).lower():
                supporting_evidence.append(f"Mechanism support: {description}")
        
        return supporting_evidence[:3]  # Limit to top 3
    
    async def _validate_potential_causes(self, potential_causes: List[PotentialCause], events: List[CausalEvent]) -> List[PotentialCause]:
        """Validate potential causes"""
        
        validated_causes = []
        
        for cause in potential_causes:
            # Calculate validation scores
            validation_score = await self._calculate_validation_score(cause, events)
            
            if validation_score > 0.3:  # Threshold for inclusion
                # Update cause with validation results
                cause.validation_results = {"overall": validation_score}
                
                validated_causes.append(cause)
        
        return validated_causes
    
    async def _calculate_validation_score(self, cause: PotentialCause, events: List[CausalEvent]) -> float:
        """Calculate validation score for potential cause"""
        
        score = 0.0
        
        # Base score from generation strategy
        strategy_scores = {
            CauseGenerationStrategy.DOMAIN_KNOWLEDGE: 0.7,
            CauseGenerationStrategy.TEMPORAL_PRECEDENCE: 0.8,
            CauseGenerationStrategy.CORRELATION_ANALYSIS: 0.6,
            CauseGenerationStrategy.MECHANISM_THEORY: 0.8,
            CauseGenerationStrategy.ANALOGICAL: 0.5,
            CauseGenerationStrategy.ELIMINATIVE: 0.4,
            CauseGenerationStrategy.EMPIRICAL: 0.7
        }
        
        score += strategy_scores.get(cause.generation_strategy, 0.5)
        
        # Adjust for plausibility
        score = (score + cause.plausibility) / 2
        
        # Adjust for evidence
        if cause.supporting_evidence:
            score += min(0.2, len(cause.supporting_evidence) * 0.05)
        
        # Adjust for mechanism clarity
        if cause.proposed_mechanism:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _rank_potential_causes(self, potential_causes: List[PotentialCause]) -> List[PotentialCause]:
        """Rank potential causes by overall score"""
        
        # Calculate overall scores
        for cause in potential_causes:
            cause.plausibility = cause.get_overall_score()
        
        # Sort by overall score
        ranked_causes = sorted(potential_causes, key=lambda c: c.get_overall_score(), reverse=True)
        
        return ranked_causes
    
    def _initialize_generation_strategies(self) -> List[str]:
        """Initialize generation strategies"""
        return [
            "domain_knowledge",
            "temporal_precedence",
            "correlation_analysis",
            "mechanism_theory",
            "analogical",
            "eliminative",
            "empirical"
        ]
    
    def _initialize_domain_knowledge(self) -> Dict[str, Any]:
        """Initialize domain knowledge base"""
        return {
            "medical": {
                "common_causes": ["infection", "inflammation", "genetic", "environmental"],
                "mechanisms": ["pathophysiological", "pharmacological", "immunological"]
            },
            "technical": {
                "common_causes": ["hardware failure", "software bug", "configuration error"],
                "mechanisms": ["mechanical", "electrical", "software"]
            },
            "scientific": {
                "common_causes": ["experimental error", "environmental factor", "measurement noise"],
                "mechanisms": ["physical", "chemical", "biological"]
            }
        }
    
    def _initialize_causal_theories(self) -> Dict[str, Any]:
        """Initialize causal theories"""
        return {
            "mill_methods": ["method_of_agreement", "method_of_difference", "joint_method"],
            "counterfactual": ["counterfactual_dependence", "possible_worlds"],
            "probabilistic": ["probabilistic_causation", "causal_bayes_nets"],
            "mechanistic": ["mechanism_based", "process_tracing"]
        }


class CausalLinkingEngine:
    """Engine for linking causes and effects (Component 3: Causal Linking)"""
    
    def __init__(self):
        self.linking_methods = self._initialize_linking_methods()
        self.mechanism_database = self._initialize_mechanism_database()
        self.evidence_criteria = self._initialize_evidence_criteria()
        self.link_validation = self._initialize_link_validation()
        
    async def establish_causal_links(self, events: List[CausalEvent], potential_causes: List[PotentialCause]) -> List[CausalLink]:
        """Establish causal links between potential causes and effects"""
        
        logger.info(f"Establishing causal links between {len(potential_causes)} causes and {len(events)} events")
        
        # Identify candidate links
        candidate_links = await self._identify_candidate_links(events, potential_causes)
        
        # Assess mechanism plausibility
        assessed_links = await self._assess_mechanism_plausibility(candidate_links)
        
        # Evaluate evidence strength
        evidence_evaluated = await self._evaluate_evidence_strength(assessed_links)
        
        # Validate temporal relationships
        temporal_validated = await self._validate_temporal_relationships(evidence_evaluated)
        
        # Assess causal strength
        strength_assessed = await self._assess_causal_strength(temporal_validated)
        
        # Rank and filter links
        final_links = await self._rank_and_filter_links(strength_assessed)
        
        return final_links
    
    async def _identify_candidate_links(self, events: List[CausalEvent], potential_causes: List[PotentialCause]) -> List[CausalLink]:
        """Identify candidate causal links"""
        
        candidate_links = []
        
        # Find effect events
        effect_events = [e for e in events if e.event_type == EventType.EFFECT]
        
        for effect_event in effect_events:
            for potential_cause in potential_causes:
                # Create candidate link
                link = CausalLink(
                    id=str(uuid4()),
                    cause_id=potential_cause.id,
                    effect_id=effect_event.id,
                    relationship_type=await self._determine_relationship_type(potential_cause, effect_event),
                    temporal_relation=await self._determine_temporal_relation(potential_cause, effect_event),
                    mechanism_description=await self._describe_mechanism(potential_cause, effect_event),
                    evidence_support=await self._assess_evidence_support(potential_cause, effect_event),
                    strength=CausalStrength.MODERATE,  # Initial assessment
                    confidence=0.5,  # Initial confidence
                    confounding_factors=[],
                    alternative_explanations=[],
                    validation_tests=[],
                    context_factors={}
                )
                
                candidate_links.append(link)
        
        return candidate_links
    
    async def _determine_relationship_type(self, potential_cause: PotentialCause, effect_event: CausalEvent) -> CausalRelationType:
        """Determine type of causal relationship"""
        
        # Analyze temporal relationship
        if potential_cause.temporal_precedence == TemporalRelation.IMMEDIATE:
            return CausalRelationType.DIRECT_CAUSE
        elif potential_cause.temporal_precedence in [TemporalRelation.SHORT_TERM, TemporalRelation.MEDIUM_TERM]:
            return CausalRelationType.INDIRECT_CAUSE
        
        # Check for necessity/sufficiency
        if potential_cause.causal_necessity > 0.8:
            return CausalRelationType.NECESSARY_CAUSE
        elif potential_cause.causal_sufficiency > 0.8:
            return CausalRelationType.SUFFICIENT_CAUSE
        
        # Check for bidirectional relationships
        if "feedback" in str(potential_cause.description).lower() or "mutual" in str(potential_cause.description).lower():
            return CausalRelationType.BIDIRECTIONAL
        
        # Check for common causes
        if potential_cause.generation_strategy == CauseGenerationStrategy.DOMAIN_KNOWLEDGE:
            return CausalRelationType.COMMON_CAUSE
        
        # Default to partial cause
        return CausalRelationType.PARTIAL_CAUSE
    
    async def _determine_temporal_relation(self, potential_cause: PotentialCause, effect_event: CausalEvent) -> TemporalRelation:
        """Determine temporal relationship"""
        
        # Use cause temporal precedence if available
        if potential_cause.temporal_precedence:
            return potential_cause.temporal_precedence
        
        # Analyze descriptions for temporal clues
        # Handle case where descriptions are lists
        cause_desc = potential_cause.description
        if isinstance(cause_desc, list):
            cause_desc = ' '.join(str(item) for item in cause_desc)
        elif not isinstance(cause_desc, str):
            cause_desc = str(cause_desc)
        
        effect_desc = effect_event.description
        if isinstance(effect_desc, list):
            effect_desc = ' '.join(str(item) for item in effect_desc)
        elif not isinstance(effect_desc, str):
            effect_desc = str(effect_desc)
        
        cause_desc = str(cause_desc).lower()
        effect_desc = str(effect_desc).lower()
        
        if "immediately" in cause_desc or "instant" in cause_desc:
            return TemporalRelation.IMMEDIATE
        elif "shortly" in cause_desc or "soon" in cause_desc:
            return TemporalRelation.SHORT_TERM
        elif "eventually" in cause_desc or "later" in cause_desc:
            return TemporalRelation.MEDIUM_TERM
        elif "long" in cause_desc or "delayed" in cause_desc:
            return TemporalRelation.LONG_TERM
        
        # Default to medium term
        return TemporalRelation.MEDIUM_TERM
    
    async def _describe_mechanism(self, potential_cause: PotentialCause, effect_event: CausalEvent) -> str:
        """Describe causal mechanism"""
        
        # Map mechanism type to description
        mechanism_descriptions = {
            CausalMechanism.PHYSICAL: f"Physical process whereby {potential_cause.description} leads to {effect_event.description}",
            CausalMechanism.BIOLOGICAL: f"Biological mechanism connecting {potential_cause.description} to {effect_event.description}",
            CausalMechanism.PSYCHOLOGICAL: f"Psychological process through which {potential_cause.description} causes {effect_event.description}",
            CausalMechanism.SOCIAL: f"Social mechanism linking {potential_cause.description} to {effect_event.description}",
            CausalMechanism.ECONOMIC: f"Economic process connecting {potential_cause.description} to {effect_event.description}",
            CausalMechanism.INFORMATIONAL: f"Information transfer mechanism from {potential_cause.description} to {effect_event.description}",
            CausalMechanism.CHEMICAL: f"Chemical reaction or process linking {potential_cause.description} to {effect_event.description}",
            CausalMechanism.MECHANICAL: f"Mechanical process whereby {potential_cause.description} produces {effect_event.description}",
            CausalMechanism.ELECTRICAL: f"Electrical mechanism connecting {potential_cause.description} to {effect_event.description}",
            CausalMechanism.STATISTICAL: f"Statistical relationship between {potential_cause.description} and {effect_event.description}",
            CausalMechanism.UNKNOWN: f"Unknown mechanism linking {potential_cause.description} to {effect_event.description}"
        }
        
        return mechanism_descriptions.get(potential_cause.mechanism_type, mechanism_descriptions[CausalMechanism.UNKNOWN])
    
    async def _assess_evidence_support(self, potential_cause: PotentialCause, effect_event: CausalEvent) -> float:
        """Assess evidence support for causal link"""
        
        support = 0.5  # Base support
        
        # Adjust based on cause confidence
        support = (support + potential_cause.confidence) / 2
        
        # Adjust based on effect event reliability
        support = (support + effect_event.reliability) / 2
        
        # Adjust based on evidence quality
        support = (support + effect_event.evidence_quality) / 2
        
        return max(0.0, min(1.0, support))
    
    async def _assess_mechanism_plausibility(self, links: List[CausalLink]) -> List[CausalLink]:
        """Assess plausibility of causal mechanisms"""
        
        for link in links:
            # Get mechanism knowledge
            mechanism_knowledge = self.mechanism_database.get(link.mechanism_type, {})
            
            # Assess plausibility based on domain knowledge
            domain_plausibility = mechanism_knowledge.get("plausibility", 0.5)
            
            # Assess complexity (simpler mechanisms more plausible)
            complexity_penalty = min(0.3, len(link.mechanism_description.split()) * 0.01)
            
            # Calculate overall plausibility
            plausibility = domain_plausibility - complexity_penalty
            
            link.mechanism_plausibility = max(0.0, min(1.0, plausibility))
        
        return links
    
    async def _evaluate_evidence_strength(self, links: List[CausalLink]) -> List[CausalLink]:
        """Evaluate strength of evidence for causal links"""
        
        for link in links:
            evidence_strength = 0.5  # Base strength
            
            # Adjust based on evidence support
            evidence_strength = (evidence_strength + link.evidence_support) / 2
            
            # Adjust based on mechanism plausibility
            evidence_strength = (evidence_strength + link.mechanism_plausibility) / 2
            
            # Evidence quality indicators
            if "experimental" in str(link.mechanism_description).lower():
                evidence_strength += 0.2
            elif "observational" in str(link.mechanism_description).lower():
                evidence_strength += 0.1
            elif "theoretical" in str(link.mechanism_description).lower():
                evidence_strength -= 0.1
            
            link.evidence_strength = max(0.0, min(1.0, evidence_strength))
        
        return links
    
    async def _validate_temporal_relationships(self, links: List[CausalLink]) -> List[CausalLink]:
        """Validate temporal relationships in causal links"""
        
        for link in links:
            temporal_validity = 0.8  # Base validity
            
            # Check for temporal consistency
            if link.temporal_relation == TemporalRelation.IMMEDIATE:
                temporal_validity = 0.9
            elif link.temporal_relation in [TemporalRelation.SHORT_TERM, TemporalRelation.MEDIUM_TERM]:
                temporal_validity = 0.8
            elif link.temporal_relation == TemporalRelation.LONG_TERM:
                temporal_validity = 0.7
            elif link.temporal_relation == TemporalRelation.DELAYED:
                temporal_validity = 0.6
            
            # Check for temporal violations
            if "after" in str(link.mechanism_description).lower() and link.temporal_relation == TemporalRelation.IMMEDIATE:
                temporal_validity -= 0.3
            
            link.temporal_validity = max(0.0, min(1.0, temporal_validity))
        
        return links
    
    async def _assess_causal_strength(self, links: List[CausalLink]) -> List[CausalLink]:
        """Assess overall causal strength of links"""
        
        for link in links:
            # Combine multiple factors
            strength_score = (
                link.evidence_strength * 0.3 +
                link.mechanism_plausibility * 0.3 +
                link.temporal_validity * 0.2 +
                link.evidence_support * 0.2
            )
            
            # Map to strength categories
            if strength_score > 0.8:
                link.strength = CausalStrength.VERY_STRONG
            elif strength_score > 0.6:
                link.strength = CausalStrength.STRONG
            elif strength_score > 0.4:
                link.strength = CausalStrength.MODERATE
            elif strength_score > 0.2:
                link.strength = CausalStrength.WEAK
            else:
                link.strength = CausalStrength.VERY_WEAK
            
            # Update confidence
            link.confidence = strength_score
        
        return links
    
    async def _rank_and_filter_links(self, links: List[CausalLink]) -> List[CausalLink]:
        """Rank and filter causal links"""
        
        # Filter out very weak links
        filtered_links = [link for link in links if link.strength != CausalStrength.VERY_WEAK]
        
        # Sort by confidence
        ranked_links = sorted(filtered_links, key=lambda l: l.confidence, reverse=True)
        
        return ranked_links
    
    def _initialize_linking_methods(self) -> List[str]:
        """Initialize causal linking methods"""
        return [
            "correlation_analysis",
            "temporal_precedence",
            "mechanism_identification",
            "intervention_analysis",
            "counterfactual_reasoning",
            "mill_methods",
            "granger_causality",
            "process_tracing"
        ]
    
    def _initialize_mechanism_database(self) -> Dict[str, Any]:
        """Initialize mechanism database"""
        return {
            CausalMechanism.PHYSICAL: {"plausibility": 0.8, "complexity": 0.6},
            CausalMechanism.BIOLOGICAL: {"plausibility": 0.7, "complexity": 0.7},
            CausalMechanism.PSYCHOLOGICAL: {"plausibility": 0.6, "complexity": 0.8},
            CausalMechanism.SOCIAL: {"plausibility": 0.5, "complexity": 0.9},
            CausalMechanism.ECONOMIC: {"plausibility": 0.6, "complexity": 0.8},
            CausalMechanism.INFORMATIONAL: {"plausibility": 0.7, "complexity": 0.7},
            CausalMechanism.CHEMICAL: {"plausibility": 0.8, "complexity": 0.6},
            CausalMechanism.MECHANICAL: {"plausibility": 0.9, "complexity": 0.5},
            CausalMechanism.ELECTRICAL: {"plausibility": 0.8, "complexity": 0.6},
            CausalMechanism.STATISTICAL: {"plausibility": 0.4, "complexity": 0.9},
            CausalMechanism.UNKNOWN: {"plausibility": 0.3, "complexity": 1.0}
        }
    
    def _initialize_evidence_criteria(self) -> Dict[str, float]:
        """Initialize evidence criteria"""
        return {
            "experimental": 0.9,
            "observational": 0.7,
            "correlational": 0.5,
            "theoretical": 0.4,
            "anecdotal": 0.2
        }
    
    def _initialize_link_validation(self) -> Dict[str, Any]:
        """Initialize link validation methods"""
        return {
            "temporal_consistency": True,
            "mechanism_plausibility": True,
            "evidence_strength": True,
            "alternative_explanations": True,
            "confounding_factors": True
        }


class ConfoundingFactorEvaluationEngine:
    """Engine for evaluating confounding factors (Component 4: Evaluation of Confounding Factors)"""
    
    def __init__(self):
        self.confounding_types = self._initialize_confounding_types()
        self.detection_methods = self._initialize_detection_methods()
        self.control_strategies = self._initialize_control_strategies()
        self.assessment_criteria = self._initialize_assessment_criteria()
        
    async def evaluate_confounding_factors(self, causal_links: List[CausalLink], events: List[CausalEvent]) -> List[ConfoundingFactor]:
        """Evaluate confounding factors for causal links"""
        
        logger.info(f"Evaluating confounding factors for {len(causal_links)} causal links")
        
        # Detect potential confounders
        potential_confounders = await self._detect_potential_confounders(causal_links, events)
        
        # Assess confounding impact
        assessed_confounders = await self._assess_confounding_impact(potential_confounders, causal_links)
        
        # Evaluate control strategies
        controlled_confounders = await self._evaluate_control_strategies(assessed_confounders)
        
        # Validate confounding assessment
        validated_confounders = await self._validate_confounding_assessment(controlled_confounders)
        
        # Rank confounders by impact
        ranked_confounders = await self._rank_confounders_by_impact(validated_confounders)
        
        return ranked_confounders
    
    async def _detect_potential_confounders(self, causal_links: List[CausalLink], events: List[CausalEvent]) -> List[ConfoundingFactor]:
        """Detect potential confounding factors"""
        
        potential_confounders = []
        
        for link in causal_links:
            # Find events that might confound this link
            confounding_events = [e for e in events if e.event_type == EventType.CONFOUNDING]
            
            for event in confounding_events:
                confounder = ConfoundingFactor(
                    id=str(uuid4()),
                    causal_link_id=link.id,
                    confounder_variable=event.description,
                    confounder_type=await self._determine_confounder_type(event, link),
                    impact_on_cause=await self._assess_impact_on_cause(event, link),
                    impact_on_effect=await self._assess_impact_on_effect(event, link),
                    correlation_with_cause=await self._assess_correlation_with_cause(event, link),
                    correlation_with_effect=await self._assess_correlation_with_effect(event, link),
                    confounding_strength=0.5,  # Initial assessment
                    control_strategy=None,
                    residual_confounding=0.0,
                    bias_direction="unknown",
                    bias_magnitude=0.0,
                    detection_method="observational",
                    control_feasibility=0.5,
                    evidence_quality=event.evidence_quality
                )
                
                potential_confounders.append(confounder)
        
        return potential_confounders
    
    async def _determine_confounder_type(self, event: CausalEvent, link: CausalLink) -> str:
        """Determine type of confounding factor"""
        
        # Check for common confounding patterns
        # Handle case where description is a list
        description = event.description
        if isinstance(description, list):
            description = ' '.join(str(item) for item in description)
        elif not isinstance(description, str):
            description = str(description)
        
        desc_lower = str(description).lower()
        
        if "selection" in desc_lower:
            return "selection_bias"
        elif "measurement" in desc_lower:
            return "measurement_bias"
        elif "information" in desc_lower:
            return "information_bias"
        elif "time" in desc_lower or "temporal" in desc_lower:
            return "temporal_confounding"
        elif "common" in desc_lower:
            return "common_cause"
        elif "mediating" in desc_lower:
            return "mediator"
        elif "moderating" in desc_lower:
            return "moderator"
        else:
            return "unknown_confounder"
    
    async def _assess_impact_on_cause(self, event: CausalEvent, link: CausalLink) -> float:
        """Assess impact of confounder on cause"""
        
        impact = 0.5  # Base impact
        
        # Assess based on variable overlap
        if event.variables and link.mechanism_description:
            overlap_score = len(set(event.variables) & set(link.mechanism_description.split())) / max(len(event.variables), 1)
            impact += overlap_score * 0.3
        
        # Assess based on domain similarity
        if event.domain and link.mechanism_description:
            if str(event.domain).lower() in str(link.mechanism_description).lower():
                impact += 0.2
        
        return max(0.0, min(1.0, impact))
    
    async def _assess_impact_on_effect(self, event: CausalEvent, link: CausalLink) -> float:
        """Assess impact of confounder on effect"""
        
        impact = 0.5  # Base impact
        
        # Assess based on event type
        if event.event_type == EventType.CONFOUNDING:
            impact += 0.3
        elif event.event_type == EventType.MEDIATING:
            impact += 0.2
        elif event.event_type == EventType.MODERATING:
            impact += 0.1
        
        return max(0.0, min(1.0, impact))
    
    async def _assess_correlation_with_cause(self, event: CausalEvent, link: CausalLink) -> float:
        """Assess correlation between confounder and cause"""
        
        correlation = 0.3  # Base correlation
        
        # Assess based on temporal relationship
        if event.temporal_position and link.temporal_relation:
            if event.temporal_position == "before":
                correlation += 0.2
            elif event.temporal_position == "during":
                correlation += 0.1
        
        # Assess based on description similarity
        if event.description and link.mechanism_description:
            # Handle case where description is a list
            description = event.description
            if isinstance(description, list):
                description = ' '.join(str(item) for item in description)
            elif not isinstance(description, str):
                description = str(description)
            
            desc_words = set(str(description).lower().split())
            mech_words = set(str(link.mechanism_description).lower().split())
            overlap = len(desc_words & mech_words) / max(len(desc_words), 1)
            correlation += overlap * 0.3
        
        return max(0.0, min(1.0, correlation))
    
    async def _assess_correlation_with_effect(self, event: CausalEvent, link: CausalLink) -> float:
        """Assess correlation between confounder and effect"""
        
        correlation = 0.3  # Base correlation
        
        # Assess based on event type
        if event.event_type in [EventType.CONFOUNDING, EventType.MEDIATING]:
            correlation += 0.2
        
        # Assess based on reliability
        correlation = (correlation + event.reliability) / 2
        
        return max(0.0, min(1.0, correlation))
    
    async def _assess_confounding_impact(self, confounders: List[ConfoundingFactor], causal_links: List[CausalLink]) -> List[ConfoundingFactor]:
        """Assess impact of confounding factors"""
        
        for confounder in confounders:
            # Calculate confounding strength
            strength = (
                confounder.impact_on_cause * 0.3 +
                confounder.impact_on_effect * 0.3 +
                confounder.correlation_with_cause * 0.2 +
                confounder.correlation_with_effect * 0.2
            )
            
            confounder.confounding_strength = strength
            
            # Assess bias direction
            if confounder.impact_on_cause > 0.6 and confounder.impact_on_effect > 0.6:
                confounder.bias_direction = "positive"
            elif confounder.impact_on_cause < 0.4 and confounder.impact_on_effect < 0.4:
                confounder.bias_direction = "negative"
            else:
                confounder.bias_direction = "mixed"
            
            # Assess bias magnitude
            confounder.bias_magnitude = abs(confounder.impact_on_cause - confounder.impact_on_effect)
        
        return confounders
    
    async def _evaluate_control_strategies(self, confounders: List[ConfoundingFactor]) -> List[ConfoundingFactor]:
        """Evaluate control strategies for confounding factors"""
        
        for confounder in confounders:
            # Determine best control strategy
            if confounder.confounding_strength > 0.7:
                confounder.control_strategy = "stratification"
                confounder.control_feasibility = 0.8
            elif confounder.confounding_strength > 0.5:
                confounder.control_strategy = "matching"
                confounder.control_feasibility = 0.7
            elif confounder.confounding_strength > 0.3:
                confounder.control_strategy = "adjustment"
                confounder.control_feasibility = 0.6
            else:
                confounder.control_strategy = "none"
                confounder.control_feasibility = 1.0
            
            # Assess residual confounding
            if confounder.control_strategy != "none":
                confounder.residual_confounding = confounder.confounding_strength * (1 - confounder.control_feasibility)
            else:
                confounder.residual_confounding = confounder.confounding_strength
        
        return confounders
    
    async def _validate_confounding_assessment(self, confounders: List[ConfoundingFactor]) -> List[ConfoundingFactor]:
        """Validate confounding assessment"""
        
        for confounder in confounders:
            # Validate based on evidence quality
            validation_score = confounder.evidence_quality
            
            # Adjust based on detection method
            if confounder.detection_method == "experimental":
                validation_score += 0.2
            elif confounder.detection_method == "observational":
                validation_score += 0.1
            
            # Adjust based on control feasibility
            validation_score = (validation_score + confounder.control_feasibility) / 2
            
            confounder.validation_score = max(0.0, min(1.0, validation_score))
        
        return confounders
    
    async def _rank_confounders_by_impact(self, confounders: List[ConfoundingFactor]) -> List[ConfoundingFactor]:
        """Rank confounders by impact"""
        
        # Sort by confounding strength (descending)
        ranked_confounders = sorted(confounders, key=lambda c: c.confounding_strength, reverse=True)
        
        return ranked_confounders
    
    def _initialize_confounding_types(self) -> List[str]:
        """Initialize confounding types"""
        return [
            "selection_bias",
            "measurement_bias",
            "information_bias",
            "temporal_confounding",
            "common_cause",
            "mediator",
            "moderator",
            "collider",
            "unknown_confounder"
        ]
    
    def _initialize_detection_methods(self) -> List[str]:
        """Initialize detection methods"""
        return [
            "observational",
            "experimental",
            "statistical",
            "graphical",
            "sensitivity_analysis",
            "instrumental_variables",
            "randomization",
            "matching"
        ]
    
    def _initialize_control_strategies(self) -> List[str]:
        """Initialize control strategies"""
        return [
            "stratification",
            "matching",
            "adjustment",
            "randomization",
            "instrumental_variables",
            "sensitivity_analysis",
            "propensity_scores",
            "none"
        ]
    
    def _initialize_assessment_criteria(self) -> Dict[str, float]:
        """Initialize assessment criteria"""
        return {
            "strength_threshold": 0.3,
            "bias_threshold": 0.2,
            "control_threshold": 0.5,
            "validation_threshold": 0.6
        }


class PredictiveInferenceEngine:
    """Engine for predictive/explanatory inference (Component 5: Predictive or Explanatory Inference)"""
    
    def __init__(self):
        self.inference_types = self._initialize_inference_types()
        self.prediction_methods = self._initialize_prediction_methods()
        self.intervention_strategies = self._initialize_intervention_strategies()
        self.validation_metrics = self._initialize_validation_metrics()
        
    async def generate_predictive_inference(self, causal_links: List[CausalLink], confounding_factors: List[ConfoundingFactor]) -> CausalInference:
        """Generate predictive or explanatory inference"""
        
        logger.info(f"Generating predictive inference from {len(causal_links)} causal links")
        
        # Determine inference type
        inference_type = await self._determine_inference_type(causal_links)
        
        # Generate predictions
        predictions = await self._generate_predictions(causal_links, confounding_factors)
        
        # Analyze interventions
        interventions = await self._analyze_interventions(causal_links, confounding_factors)
        
        # Assess uncertainties
        uncertainties = await self._assess_uncertainties(causal_links, confounding_factors)
        
        # Provide explanations
        explanations = await self._provide_explanations(causal_links, confounding_factors)
        
        # Validate inference
        validation_results = await self._validate_inference(predictions, interventions, explanations)
        
        # Create inference object
        inference = CausalInference(
            id=str(uuid4()),
            causal_link_id=str(uuid4()),  # Add missing required field
            inference_type=inference_type,
            inference_statement=f"Causal inference for {inference_type} reasoning",  # Add missing required field
            predictions=predictions,
            interventions=interventions,
            explanations=explanations,
            uncertainties=uncertainties,
            confidence_level=await self._calculate_confidence_level(validation_results),
            validation_results=validation_results,
            practical_implications=await self._derive_practical_implications(predictions, interventions),
            recommendations=await self._generate_recommendations(predictions, interventions, explanations),
            limitations=await self._identify_limitations(confounding_factors, uncertainties),
            future_research=await self._suggest_future_research(causal_links, uncertainties)
        )
        
        return inference
    
    async def _determine_inference_type(self, causal_links: List[CausalLink]) -> str:
        """Determine type of inference"""
        
        # Count different types of causal relationships
        relationship_counts = Counter(link.relationship_type for link in causal_links)
        
        # Determine dominant inference type
        if relationship_counts.get(CausalRelationType.DIRECT_CAUSE, 0) > len(causal_links) * 0.5:
            return "explanatory"
        elif relationship_counts.get(CausalRelationType.PROBABILISTIC, 0) > len(causal_links) * 0.3:
            return "predictive"
        elif relationship_counts.get(CausalRelationType.BIDIRECTIONAL, 0) > 0:
            return "interactive"
        else:
            return "mixed"
    
    async def _generate_predictions(self, causal_links: List[CausalLink], confounding_factors: List[ConfoundingFactor]) -> List[Dict[str, Any]]:
        """Generate predictions based on causal links"""
        
        predictions = []
        
        for link in causal_links:
            # Generate prediction based on link strength
            if link.strength in [CausalStrength.STRONG, CausalStrength.VERY_STRONG]:
                prediction = {
                    "prediction": f"If {link.cause_id} occurs, {link.effect_id} will likely occur",
                    "confidence": link.confidence,
                    "strength": link.strength.value,
                    "mechanism": link.mechanism_description,
                    "temporal_frame": link.temporal_relation.value,
                    "conditions": [],
                    "probability": link.confidence * 0.9,
                    "evidence_support": link.evidence_support
                }
                
                # Add confounding considerations
                relevant_confounders = [cf for cf in confounding_factors if cf.causal_link_id == link.id]
                if relevant_confounders:
                    prediction["conditions"].append(f"Controlling for {len(relevant_confounders)} confounding factors")
                    prediction["probability"] *= (1 - max(cf.residual_confounding for cf in relevant_confounders))
                
                predictions.append(prediction)
        
        return predictions
    
    async def _analyze_interventions(self, causal_links: List[CausalLink], confounding_factors: List[ConfoundingFactor]) -> List[Dict[str, Any]]:
        """Analyze potential interventions"""
        
        interventions = []
        
        for link in causal_links:
            if link.strength in [CausalStrength.STRONG, CausalStrength.VERY_STRONG]:
                intervention = {
                    "intervention_type": "cause_manipulation",
                    "target": link.cause_id,
                    "expected_effect": link.effect_id,
                    "mechanism": link.mechanism_description,
                    "feasibility": await self._assess_intervention_feasibility(link),
                    "effectiveness": link.confidence,
                    "side_effects": await self._assess_side_effects(link),
                    "ethical_considerations": await self._assess_ethical_considerations(link),
                    "cost_benefit": await self._assess_cost_benefit(link),
                    "implementation_strategy": await self._suggest_implementation_strategy(link),
                    "monitoring_requirements": await self._suggest_monitoring_requirements(link),
                    "success_indicators": await self._identify_success_indicators(link)
                }
                
                interventions.append(intervention)
        
        return interventions
    
    async def _assess_intervention_feasibility(self, link: CausalLink) -> float:
        """Assess feasibility of intervention"""
        
        feasibility = 0.5  # Base feasibility
        
        # Adjust based on mechanism type
        mechanism_feasibility = {
            CausalMechanism.PHYSICAL: 0.8,
            CausalMechanism.BIOLOGICAL: 0.6,
            CausalMechanism.PSYCHOLOGICAL: 0.4,
            CausalMechanism.SOCIAL: 0.3,
            CausalMechanism.ECONOMIC: 0.5,
            CausalMechanism.INFORMATIONAL: 0.7,
            CausalMechanism.CHEMICAL: 0.7,
            CausalMechanism.MECHANICAL: 0.9,
            CausalMechanism.ELECTRICAL: 0.8,
            CausalMechanism.STATISTICAL: 0.2,
            CausalMechanism.UNKNOWN: 0.1
        }
        
        feasibility = mechanism_feasibility.get(link.mechanism_type, 0.5)
        
        # Adjust based on temporal relation
        if link.temporal_relation == TemporalRelation.IMMEDIATE:
            feasibility += 0.1
        elif link.temporal_relation == TemporalRelation.DELAYED:
            feasibility -= 0.1
        
        return max(0.0, min(1.0, feasibility))
    
    async def _assess_side_effects(self, link: CausalLink) -> List[str]:
        """Assess potential side effects of intervention"""
        
        side_effects = []
        
        # Common side effects based on mechanism
        if link.mechanism_type == CausalMechanism.BIOLOGICAL:
            side_effects.extend(["biological_disruption", "systemic_effects"])
        elif link.mechanism_type == CausalMechanism.PSYCHOLOGICAL:
            side_effects.extend(["behavioral_changes", "emotional_effects"])
        elif link.mechanism_type == CausalMechanism.SOCIAL:
            side_effects.extend(["social_disruption", "unintended_consequences"])
        elif link.mechanism_type == CausalMechanism.ECONOMIC:
            side_effects.extend(["economic_impacts", "resource_allocation"])
        
        # General side effects
        if link.strength == CausalStrength.VERY_STRONG:
            side_effects.append("potential_overcorrection")
        
        return side_effects
    
    async def _assess_ethical_considerations(self, link: CausalLink) -> List[str]:
        """Assess ethical considerations for intervention"""
        
        ethical_considerations = []
        
        # Based on mechanism type
        if link.mechanism_type == CausalMechanism.BIOLOGICAL:
            ethical_considerations.extend(["informed_consent", "safety_protocols"])
        elif link.mechanism_type == CausalMechanism.PSYCHOLOGICAL:
            ethical_considerations.extend(["psychological_wellbeing", "autonomy"])
        elif link.mechanism_type == CausalMechanism.SOCIAL:
            ethical_considerations.extend(["social_justice", "equity"])
        
        # General ethical considerations
        ethical_considerations.extend(["beneficence", "non_maleficence", "justice"])
        
        return ethical_considerations
    
    async def _assess_cost_benefit(self, link: CausalLink) -> Dict[str, float]:
        """Assess cost-benefit of intervention"""
        
        # Simple cost-benefit assessment
        benefit = link.confidence * 0.8
        cost = 1.0 - (link.confidence * 0.6)
        
        return {
            "benefit": benefit,
            "cost": cost,
            "ratio": benefit / cost if cost > 0 else float('inf')
        }
    
    async def _suggest_implementation_strategy(self, link: CausalLink) -> str:
        """Suggest implementation strategy"""
        
        if link.temporal_relation == TemporalRelation.IMMEDIATE:
            return "immediate_implementation"
        elif link.temporal_relation == TemporalRelation.SHORT_TERM:
            return "phased_implementation"
        elif link.temporal_relation == TemporalRelation.MEDIUM_TERM:
            return "gradual_implementation"
        else:
            return "long_term_implementation"
    
    async def _suggest_monitoring_requirements(self, link: CausalLink) -> List[str]:
        """Suggest monitoring requirements"""
        
        monitoring = ["baseline_measurement", "regular_assessment", "outcome_tracking"]
        
        if link.strength == CausalStrength.VERY_STRONG:
            monitoring.append("intensive_monitoring")
        
        return monitoring
    
    async def _identify_success_indicators(self, link: CausalLink) -> List[str]:
        """Identify success indicators"""
        
        indicators = [
            f"reduction_in_{link.effect_id}",
            f"improvement_in_{link.mechanism_type.value}_function",
            "sustained_effect_over_time"
        ]
        
        return indicators
    
    async def _assess_uncertainties(self, causal_links: List[CausalLink], confounding_factors: List[ConfoundingFactor]) -> Dict[str, float]:
        """Assess uncertainties in inference"""
        
        uncertainties = {}
        
        # Causal uncertainty
        link_confidences = [link.confidence for link in causal_links]
        uncertainties["causal_uncertainty"] = 1.0 - (sum(link_confidences) / len(link_confidences)) if link_confidences else 1.0
        
        # Confounding uncertainty
        confounding_strengths = [cf.confounding_strength for cf in confounding_factors]
        uncertainties["confounding_uncertainty"] = sum(confounding_strengths) / len(confounding_strengths) if confounding_strengths else 0.0
        
        # Measurement uncertainty
        uncertainties["measurement_uncertainty"] = 0.3  # Default assumption
        
        # Model uncertainty
        uncertainties["model_uncertainty"] = 0.2  # Default assumption
        
        return uncertainties
    
    async def _provide_explanations(self, causal_links: List[CausalLink], confounding_factors: List[ConfoundingFactor]) -> List[str]:
        """Provide explanations for causal relationships"""
        
        explanations = []
        
        for link in causal_links:
            explanation = f"The causal relationship between {link.cause_id} and {link.effect_id} operates through {link.mechanism_description}. This relationship has {link.strength.value} strength with {link.confidence:.2f} confidence."
            
            # Add confounding considerations
            relevant_confounders = [cf for cf in confounding_factors if cf.causal_link_id == link.id]
            if relevant_confounders:
                explanation += f" However, {len(relevant_confounders)} confounding factors may influence this relationship."
            
            explanations.append(explanation)
        
        return explanations
    
    async def _validate_inference(self, predictions: List[Dict], interventions: List[Dict], explanations: List[str]) -> Dict[str, Any]:
        """Validate inference results"""
        
        validation = {
            "prediction_quality": len(predictions) / 10.0,  # Simple quality metric
            "intervention_feasibility": sum(i.get("feasibility", 0.5) for i in interventions) / len(interventions) if interventions else 0.0,
            "explanation_completeness": len(explanations) / 10.0,  # Simple completeness metric
            "overall_validity": 0.0
        }
        
        validation["overall_validity"] = (
            validation["prediction_quality"] * 0.4 +
            validation["intervention_feasibility"] * 0.3 +
            validation["explanation_completeness"] * 0.3
        )
        
        return validation
    
    async def _calculate_confidence_level(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall confidence level"""
        
        return validation_results.get("overall_validity", 0.5)
    
    async def _derive_practical_implications(self, predictions: List[Dict], interventions: List[Dict]) -> List[str]:
        """Derive practical implications"""
        
        implications = []
        
        # From predictions
        for prediction in predictions:
            if prediction["confidence"] > 0.7:
                implications.append(f"High confidence prediction: {prediction['prediction']}")
        
        # From interventions
        for intervention in interventions:
            if intervention["feasibility"] > 0.7:
                implications.append(f"Feasible intervention: {intervention['intervention_type']} targeting {intervention['target']}")
        
        return implications
    
    async def _generate_recommendations(self, predictions: List[Dict], interventions: List[Dict], explanations: List[str]) -> List[str]:
        """Generate recommendations based on inference"""
        
        recommendations = []
        
        # Based on predictions
        high_confidence_predictions = [p for p in predictions if p["confidence"] > 0.7]
        if high_confidence_predictions:
            recommendations.append(f"Focus on {len(high_confidence_predictions)} high-confidence predictions")
        
        # Based on interventions
        feasible_interventions = [i for i in interventions if i["feasibility"] > 0.7]
        if feasible_interventions:
            recommendations.append(f"Implement {len(feasible_interventions)} feasible interventions")
        
        # General recommendations
        recommendations.extend([
            "Continue monitoring causal relationships",
            "Validate findings with additional data",
            "Consider ethical implications of interventions"
        ])
        
        return recommendations
    
    async def _identify_limitations(self, confounding_factors: List[ConfoundingFactor], uncertainties: Dict[str, float]) -> List[str]:
        """Identify limitations of inference"""
        
        limitations = []
        
        # Based on confounding factors
        high_confounding = [cf for cf in confounding_factors if cf.confounding_strength > 0.7]
        if high_confounding:
            limitations.append(f"High confounding from {len(high_confounding)} factors")
        
        # Based on uncertainties
        for uncertainty_type, value in uncertainties.items():
            if value > 0.5:
                limitations.append(f"High {uncertainty_type}: {value:.2f}")
        
        return limitations
    
    async def _suggest_future_research(self, causal_links: List[CausalLink], uncertainties: Dict[str, float]) -> List[str]:
        """Suggest future research directions"""
        
        suggestions = []
        
        # Based on causal links
        weak_links = [link for link in causal_links if link.strength in [CausalStrength.WEAK, CausalStrength.VERY_WEAK]]
        if weak_links:
            suggestions.append(f"Strengthen evidence for {len(weak_links)} weak causal links")
        
        # Based on uncertainties
        for uncertainty_type, value in uncertainties.items():
            if value > 0.5:
                suggestions.append(f"Reduce {uncertainty_type} through targeted research")
        
        return suggestions
    
    def _initialize_inference_types(self) -> List[str]:
        """Initialize inference types"""
        return [
            "explanatory",
            "predictive",
            "interactive",
            "mixed"
        ]
    
    def _initialize_prediction_methods(self) -> List[str]:
        """Initialize prediction methods"""
        return [
            "deterministic",
            "probabilistic",
            "bayesian",
            "machine_learning",
            "statistical",
            "simulation"
        ]
    
    def _initialize_intervention_strategies(self) -> List[str]:
        """Initialize intervention strategies"""
        return [
            "cause_manipulation",
            "effect_modification",
            "mechanism_disruption",
            "environmental_change",
            "behavioral_intervention",
            "policy_intervention"
        ]
    
    def _initialize_validation_metrics(self) -> Dict[str, float]:
        """Initialize validation metrics"""
        return {
            "prediction_accuracy": 0.8,
            "intervention_success": 0.7,
            "explanation_quality": 0.6,
            "overall_validity": 0.7
        }


class EnhancedCausalReasoningEngine:
    """Enhanced causal reasoning engine with all five elemental components"""
    
    def __init__(self):
        self.event_observation_engine = EventObservationEngine()
        self.cause_identification_engine = PotentialCauseIdentificationEngine()
        self.causal_linking_engine = CausalLinkingEngine()
        self.confounding_evaluation_engine = ConfoundingFactorEvaluationEngine()
        self.predictive_inference_engine = PredictiveInferenceEngine()
        
    async def perform_causal_reasoning(self, observations: List[str], query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform complete causal reasoning with all five elemental components"""
        
        start_time = time.time()
        
        logger.info(f"Starting enhanced causal reasoning with {len(observations)} observations")
        
        if context is None:
            context = {}
        
        # Component 1: Observation of Events
        events = await self.event_observation_engine.observe_events(observations, context)
        
        # Component 2: Identification of Potential Causes
        potential_causes = await self.cause_identification_engine.identify_potential_causes(events, context)
        
        # Component 3: Causal Linking
        causal_links = await self.causal_linking_engine.establish_causal_links(events, potential_causes)
        
        # Component 4: Evaluation of Confounding Factors
        confounding_factors = await self.confounding_evaluation_engine.evaluate_confounding_factors(causal_links, events)
        
        # Component 5: Predictive or Explanatory Inference
        causal_inference = await self.predictive_inference_engine.generate_predictive_inference(causal_links, confounding_factors)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create comprehensive result
        result = {
            "query": query,
            "observations": observations,
            "context": context,
            "events": events,
            "potential_causes": potential_causes,
            "causal_links": causal_links,
            "confounding_factors": confounding_factors,
            "causal_inference": causal_inference,
            "reasoning_quality": await self._assess_reasoning_quality(events, potential_causes, causal_links, confounding_factors, causal_inference),
            "processing_time": processing_time
        }
        
        logger.info(f"Enhanced causal reasoning completed in {processing_time:.2f} seconds")
        
        return result
    
    async def _assess_reasoning_quality(self, events: List[CausalEvent], potential_causes: List[PotentialCause], 
                                      causal_links: List[CausalLink], confounding_factors: List[ConfoundingFactor], 
                                      causal_inference: CausalInference) -> float:
        """Assess overall quality of causal reasoning"""
        
        quality_components = []
        
        # Event observation quality
        if events:
            event_quality = sum(e.get_overall_score() for e in events) / len(events)
            quality_components.append(event_quality)
        
        # Cause identification quality
        if potential_causes:
            cause_quality = sum(c.get_overall_score() for c in potential_causes) / len(potential_causes)
            quality_components.append(cause_quality)
        
        # Causal linking quality
        if causal_links:
            link_quality = sum(l.confidence for l in causal_links) / len(causal_links)
            quality_components.append(link_quality)
        
        # Confounding evaluation quality
        if confounding_factors:
            confounding_quality = 1.0 - (sum(cf.confounding_strength for cf in confounding_factors) / len(confounding_factors))
            quality_components.append(confounding_quality)
        
        # Inference quality
        inference_quality = causal_inference.confidence_level
        quality_components.append(inference_quality)
        
        # Overall quality
        overall_quality = sum(quality_components) / len(quality_components) if quality_components else 0.5
        
        return overall_quality