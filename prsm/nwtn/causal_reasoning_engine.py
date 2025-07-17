#!/usr/bin/env python3
"""
NWTN Causal Reasoning Engine
Understanding cause-and-effect relationships and building causal models

This module implements NWTN's causal reasoning capabilities, which allow the system to:
1. Identify potential causal relationships from observations
2. Build causal models and networks
3. Distinguish causation from correlation
4. Perform causal inference and intervention analysis
5. Handle confounding variables and causal complexity

Causal reasoning is fundamental to understanding how the world works and
making predictions about the effects of interventions.

Key Concepts:
- Causal discovery from observational data
- Causal model construction and validation
- Intervention analysis and counterfactual reasoning
- Confounding variable identification
- Causal strength assessment
- Temporal causality analysis

Usage:
    from prsm.nwtn.causal_reasoning_engine import CausalReasoningEngine
    
    engine = CausalReasoningEngine()
    result = await engine.analyze_causal_relationships(observations, context)
"""

import asyncio
import json
import math
import re
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime, timezone
from collections import defaultdict, Counter

import structlog
from pydantic import BaseModel, Field

from prsm.nwtn.hybrid_architecture import SOC, SOCType, ConfidenceLevel
from prsm.nwtn.world_model_engine import WorldModelEngine
from prsm.agents.executors.model_executor import ModelExecutor

logger = structlog.get_logger(__name__)


class CausalRelationType(str, Enum):
    """Types of causal relationships"""
    DIRECT_CAUSE = "direct_cause"           # A directly causes B
    INDIRECT_CAUSE = "indirect_cause"       # A causes B through intermediates
    COMMON_CAUSE = "common_cause"           # A and B share a common cause
    BIDIRECTIONAL = "bidirectional"         # A and B cause each other
    SPURIOUS = "spurious"                   # Apparent causation due to confounding
    NECESSARY_CAUSE = "necessary_cause"     # A is necessary for B
    SUFFICIENT_CAUSE = "sufficient_cause"   # A is sufficient for B
    PARTIAL_CAUSE = "partial_cause"         # A contributes to B with other factors


class CausalStrength(str, Enum):
    """Strength of causal relationships"""
    VERY_STRONG = "very_strong"     # >0.8
    STRONG = "strong"               # 0.6-0.8
    MODERATE = "moderate"           # 0.4-0.6
    WEAK = "weak"                   # 0.2-0.4
    VERY_WEAK = "very_weak"         # <0.2


class CausalMechanism(str, Enum):
    """Types of causal mechanisms"""
    PHYSICAL = "physical"           # Physical processes
    BIOLOGICAL = "biological"       # Biological processes
    PSYCHOLOGICAL = "psychological" # Mental/cognitive processes
    SOCIAL = "social"              # Social processes
    ECONOMIC = "economic"          # Economic processes
    INFORMATIONAL = "informational" # Information transfer
    STATISTICAL = "statistical"    # Statistical relationships
    UNKNOWN = "unknown"            # Mechanism unknown


class TemporalRelation(str, Enum):
    """Temporal relationships between cause and effect"""
    IMMEDIATE = "immediate"         # Effect occurs immediately
    SHORT_TERM = "short_term"      # Effect occurs within minutes/hours
    MEDIUM_TERM = "medium_term"    # Effect occurs within days/weeks
    LONG_TERM = "long_term"        # Effect occurs over months/years
    DELAYED = "delayed"            # Effect has significant delay
    CONTINUOUS = "continuous"       # Ongoing causal relationship


@dataclass
class CausalVariable:
    """A variable in a causal model"""
    
    id: str
    name: str
    description: str
    
    # Variable properties
    variable_type: str = "continuous"  # "continuous", "categorical", "binary"
    possible_values: List[Any] = field(default_factory=list)
    
    # Causal properties
    is_observed: bool = True
    is_confounding: bool = False
    is_instrumental: bool = False
    
    # Temporal properties
    temporal_stability: float = 0.5  # How stable over time
    measurement_error: float = 0.1   # Estimated measurement error
    
    # Statistical properties
    mean_value: Optional[float] = None
    variance: Optional[float] = None
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class CausalRelationship:
    """A causal relationship between variables"""
    
    id: str
    cause: CausalVariable
    effect: CausalVariable
    
    # Relationship properties
    relationship_type: CausalRelationType
    causal_strength: float = 0.5
    strength_category: CausalStrength = CausalStrength.MODERATE
    
    # Mechanism and timing
    mechanism: CausalMechanism = CausalMechanism.UNKNOWN
    temporal_relation: TemporalRelation = TemporalRelation.MEDIUM_TERM
    
    # Evidence and confidence
    evidence_strength: float = 0.5
    confidence: float = 0.5
    
    # Confounding and mediation
    confounding_variables: List[CausalVariable] = field(default_factory=list)
    mediating_variables: List[CausalVariable] = field(default_factory=list)
    
    # Conditions and context
    necessary_conditions: List[str] = field(default_factory=list)
    moderating_factors: List[str] = field(default_factory=list)
    
    # Validation
    experimental_support: bool = False
    observational_support: bool = False
    
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


@dataclass
class CausalModel:
    """A causal model representing relationships between variables"""
    
    id: str
    name: str
    description: str
    
    # Model components
    variables: List[CausalVariable] = field(default_factory=list)
    relationships: List[CausalRelationship] = field(default_factory=list)
    
    # Model properties
    model_type: str = "directed_acyclic_graph"  # "dag", "bidirectional", "temporal"
    complexity: float = 0.5
    
    # Validation metrics
    goodness_of_fit: float = 0.5
    predictive_accuracy: float = 0.5
    causal_validity: float = 0.5
    
    # Interventions and predictions
    testable_interventions: List[str] = field(default_factory=list)
    causal_predictions: List[str] = field(default_factory=list)
    
    # Limitations and assumptions
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_variable(self, variable: CausalVariable):
        """Add a variable to the model"""
        if variable not in self.variables:
            self.variables.append(variable)
    
    def add_relationship(self, relationship: CausalRelationship):
        """Add a causal relationship to the model"""
        # Ensure variables are in the model
        self.add_variable(relationship.cause)
        self.add_variable(relationship.effect)
        
        # Add relationship
        self.relationships.append(relationship)
    
    def get_causes(self, variable: CausalVariable) -> List[CausalVariable]:
        """Get all variables that cause the given variable"""
        return [rel.cause for rel in self.relationships if rel.effect == variable]
    
    def get_effects(self, variable: CausalVariable) -> List[CausalVariable]:
        """Get all variables that are affected by the given variable"""
        return [rel.effect for rel in self.relationships if rel.cause == variable]


@dataclass
class CausalAnalysis:
    """Result of causal reasoning analysis"""
    
    id: str
    query: str
    observations: List[str]
    
    # Discovered causal model
    causal_model: CausalModel
    
    # Analysis results
    primary_causal_relationships: List[CausalRelationship] = field(default_factory=list)
    confounding_factors: List[CausalVariable] = field(default_factory=list)
    
    # Causal conclusions
    causal_conclusions: List[str] = field(default_factory=list)
    intervention_recommendations: List[str] = field(default_factory=list)
    
    # Confidence and limitations
    overall_confidence: float = 0.5
    causal_certainty: float = 0.5
    
    # Validation and testing
    required_tests: List[str] = field(default_factory=list)
    alternative_explanations: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CausalReasoningEngine:
    """
    Engine for causal reasoning and building causal models
    
    This system enables NWTN to understand cause-and-effect relationships,
    build causal models, and make predictions about interventions.
    """
    
    def __init__(self):
        self.model_executor = ModelExecutor(agent_id="causal_reasoning_engine")
        self.world_model = WorldModelEngine()
        
        # Storage for causal models and analyses
        self.causal_models: List[CausalModel] = []
        self.causal_analyses: List[CausalAnalysis] = []
        
        # Configuration
        self.min_causal_strength = 0.3
        self.min_evidence_threshold = 0.5
        self.max_variables_per_model = 20
        
        # Causal discovery methods
        self.causal_discovery_methods = [
            "correlation_analysis",
            "temporal_precedence",
            "mechanism_analysis",
            "intervention_analysis",
            "confounding_control"
        ]
        
        # Common causal patterns
        self.causal_patterns = self._initialize_causal_patterns()
        
        logger.info("Initialized Causal Reasoning Engine")
    
    async def analyze_causal_relationships(
        self, 
        observations: List[str], 
        context: Dict[str, Any] = None
    ) -> CausalAnalysis:
        """
        Analyze causal relationships from observations
        
        Args:
            observations: List of observation statements
            context: Additional context for causal analysis
            
        Returns:
            CausalAnalysis: Causal model and analysis results
        """
        
        logger.info(
            "Starting causal analysis",
            observation_count=len(observations)
        )
        
        # Step 1: Extract variables and potential relationships
        variables, potential_relationships = await self._extract_causal_structure(observations, context)
        
        # Step 2: Analyze causal relationships
        validated_relationships = await self._validate_causal_relationships(
            potential_relationships, variables, observations
        )
        
        # Step 3: Build causal model
        causal_model = await self._build_causal_model(variables, validated_relationships)
        
        # Step 4: Identify confounding factors
        confounding_factors = await self._identify_confounding_factors(causal_model, observations)
        
        # Step 5: Generate causal conclusions
        analysis = await self._generate_causal_analysis(
            causal_model, confounding_factors, observations, context
        )
        
        # Step 6: Validate and enhance analysis
        enhanced_analysis = await self._enhance_causal_analysis(analysis)
        
        # Step 7: Store results
        self.causal_models.append(causal_model)
        self.causal_analyses.append(enhanced_analysis)
        
        logger.info(
            "Causal analysis complete",
            variables_found=len(variables),
            relationships_found=len(validated_relationships),
            overall_confidence=enhanced_analysis.overall_confidence
        )
        
        return enhanced_analysis
    
    async def _extract_causal_structure(
        self, 
        observations: List[str], 
        context: Dict[str, Any] = None
    ) -> Tuple[List[CausalVariable], List[CausalRelationship]]:
        """Extract variables and potential causal relationships from observations"""
        
        variables = []
        potential_relationships = []
        
        # Extract variables
        for i, obs in enumerate(observations):
            extracted_vars = await self._extract_variables_from_observation(obs, i)
            variables.extend(extracted_vars)
        
        # Remove duplicates
        variables = self._deduplicate_variables(variables)
        
        # Extract potential relationships
        for obs in observations:
            relationships = await self._extract_relationships_from_observation(obs, variables)
            potential_relationships.extend(relationships)
        
        # Pattern-based relationship extraction
        pattern_relationships = await self._extract_pattern_relationships(observations, variables)
        potential_relationships.extend(pattern_relationships)
        
        return variables, potential_relationships
    
    async def _extract_variables_from_observation(self, observation: str, obs_id: int) -> List[CausalVariable]:
        """Extract causal variables from a single observation"""
        
        variables = []
        
        # Extract nouns and noun phrases as potential variables
        variable_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Proper nouns
            r'\bthe\s+([a-z]+(?:\s+[a-z]+)*)\b',       # "the X"
            r'\ba\s+([a-z]+(?:\s+[a-z]+)*)\b',         # "a X"
            r'\ban\s+([a-z]+(?:\s+[a-z]+)*)\b',        # "an X"
        ]
        
        for pattern in variable_patterns:
            matches = re.findall(pattern, observation)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                
                # Filter out common words
                if str(match).lower() not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                    var_id = f"var_{obs_id}_{len(variables)}"
                    
                    # Determine variable type
                    var_type = await self._determine_variable_type(match, observation)
                    
                    variable = CausalVariable(
                        id=var_id,
                        name=str(match).lower(),
                        description=f"Variable '{match}' extracted from observation",
                        variable_type=var_type
                    )
                    variables.append(variable)
        
        return variables
    
    async def _determine_variable_type(self, variable_name: str, observation: str) -> str:
        """Determine the type of a variable"""
        
        var_lower = str(variable_name).lower()
        obs_lower = str(observation).lower()
        
        # Numeric indicators
        if any(indicator in obs_lower for indicator in ["amount", "number", "level", "rate", "count", "measure"]):
            return "continuous"
        
        # Binary indicators
        if any(indicator in obs_lower for indicator in ["yes", "no", "true", "false", "presence", "absence"]):
            return "binary"
        
        # Categorical indicators
        if any(indicator in obs_lower for indicator in ["type", "category", "kind", "class", "group"]):
            return "categorical"
        
        # Default to continuous
        return "continuous"
    
    def _deduplicate_variables(self, variables: List[CausalVariable]) -> List[CausalVariable]:
        """Remove duplicate variables based on name similarity"""
        
        unique_variables = []
        seen_names = set()
        
        for var in variables:
            # Normalize name for comparison
            normalized_name = str(var.name).lower().strip()
            
            if normalized_name not in seen_names:
                unique_variables.append(var)
                seen_names.add(normalized_name)
        
        return unique_variables
    
    async def _extract_relationships_from_observation(
        self, 
        observation: str, 
        variables: List[CausalVariable]
    ) -> List[CausalRelationship]:
        """Extract causal relationships from a single observation"""
        
        relationships = []
        
        # Causal indicator patterns
        causal_patterns = [
            (r'(.+?)\s+causes?\s+(.+)', CausalRelationType.DIRECT_CAUSE),
            (r'(.+?)\s+leads?\s+to\s+(.+)', CausalRelationType.DIRECT_CAUSE),
            (r'(.+?)\s+results?\s+in\s+(.+)', CausalRelationType.DIRECT_CAUSE),
            (r'(.+?)\s+produces?\s+(.+)', CausalRelationType.DIRECT_CAUSE),
            (r'(.+?)\s+triggers?\s+(.+)', CausalRelationType.DIRECT_CAUSE),
            (r'(.+?)\s+influences?\s+(.+)', CausalRelationType.PARTIAL_CAUSE),
            (r'(.+?)\s+affects?\s+(.+)', CausalRelationType.PARTIAL_CAUSE),
            (r'because\s+of\s+(.+?),\s+(.+)', CausalRelationType.DIRECT_CAUSE),
            (r'due\s+to\s+(.+?),\s+(.+)', CausalRelationType.DIRECT_CAUSE),
            (r'as\s+a\s+result\s+of\s+(.+?),\s+(.+)', CausalRelationType.DIRECT_CAUSE),
        ]
        
        for pattern, rel_type in causal_patterns:
            matches = re.findall(pattern, observation, re.IGNORECASE)
            
            for match in matches:
                if len(match) == 2:
                    cause_text, effect_text = match
                    
                    # Find corresponding variables
                    cause_var = await self._find_matching_variable(cause_text, variables)
                    effect_var = await self._find_matching_variable(effect_text, variables)
                    
                    if cause_var and effect_var and cause_var != effect_var:
                        relationship = CausalRelationship(
                            id=str(uuid4()),
                            cause=cause_var,
                            effect=effect_var,
                            relationship_type=rel_type,
                            causal_strength=0.6,  # Initial estimate
                            evidence_strength=0.5,
                            confidence=0.5,
                            observational_support=True
                        )
                        relationship.update_strength_category()
                        relationships.append(relationship)
        
        return relationships
    
    async def _find_matching_variable(self, text: str, variables: List[CausalVariable]) -> Optional[CausalVariable]:
        """Find variable that matches the given text"""
        
        text_lower = str(text).lower().strip()
        
        # Direct name match
        for var in variables:
            if str(var.name).lower() in text_lower or text_lower in str(var.name).lower():
                return var
        
        # Partial match
        for var in variables:
            var_words = set(str(var.name).lower().split())
            text_words = set(text_lower.split())
            
            if var_words & text_words:  # If there's any overlap
                return var
        
        return None
    
    async def _extract_pattern_relationships(
        self, 
        observations: List[str], 
        variables: List[CausalVariable]
    ) -> List[CausalRelationship]:
        """Extract relationships based on known causal patterns"""
        
        relationships = []
        
        for pattern in self.causal_patterns:
            pattern_relationships = await self._apply_causal_pattern(pattern, observations, variables)
            relationships.extend(pattern_relationships)
        
        return relationships
    
    async def _apply_causal_pattern(
        self, 
        pattern: Dict[str, Any], 
        observations: List[str], 
        variables: List[CausalVariable]
    ) -> List[CausalRelationship]:
        """Apply a causal pattern to identify relationships"""
        
        relationships = []
        
        # Check if pattern keywords are present
        pattern_keywords = pattern.get("keywords", [])
        
        for obs in observations:
            obs_lower = str(obs).lower()
            
            # Check if observation matches pattern
            if any(keyword in obs_lower for keyword in pattern_keywords):
                # Extract variables that might be involved
                involved_vars = []
                for var in variables:
                    if str(var.name).lower() in obs_lower:
                        involved_vars.append(var)
                
                # Create relationships based on pattern
                if len(involved_vars) >= 2:
                    cause_var = involved_vars[0]
                    effect_var = involved_vars[1]
                    
                    relationship = CausalRelationship(
                        id=str(uuid4()),
                        cause=cause_var,
                        effect=effect_var,
                        relationship_type=CausalRelationType(pattern["type"]),
                        causal_strength=pattern.get("strength", 0.5),
                        mechanism=CausalMechanism(pattern.get("mechanism", "unknown")),
                        evidence_strength=0.4,  # Lower for pattern-based
                        confidence=0.4,
                        observational_support=True
                    )
                    relationship.update_strength_category()
                    relationships.append(relationship)
        
        return relationships
    
    async def _validate_causal_relationships(
        self, 
        potential_relationships: List[CausalRelationship], 
        variables: List[CausalVariable], 
        observations: List[str]
    ) -> List[CausalRelationship]:
        """Validate potential causal relationships"""
        
        validated_relationships = []
        
        for relationship in potential_relationships:
            # Validate relationship
            validation_score = await self._validate_single_relationship(relationship, observations)
            
            if validation_score >= self.min_evidence_threshold:
                # Update relationship with validation results
                relationship.evidence_strength = validation_score
                relationship.confidence = validation_score
                
                # Check for confounding
                confounding_vars = await self._check_confounding(relationship, variables, observations)
                relationship.confounding_variables = confounding_vars
                
                # Assess temporal precedence
                temporal_score = await self._assess_temporal_precedence(relationship, observations)
                relationship.causal_strength *= temporal_score
                
                # Update strength category
                relationship.update_strength_category()
                
                validated_relationships.append(relationship)
        
        return validated_relationships
    
    async def _validate_single_relationship(
        self, 
        relationship: CausalRelationship, 
        observations: List[str]
    ) -> float:
        """Validate a single causal relationship"""
        
        validation_score = 0.0
        
        # Check for consistent mentions
        cause_mentions = sum(1 for obs in observations if str(relationship.cause.name).lower() in str(obs).lower())
        effect_mentions = sum(1 for obs in observations if str(relationship.effect.name).lower() in str(obs).lower())
        
        if cause_mentions > 0 and effect_mentions > 0:
            validation_score += 0.3
        
        # Check for explicit causal language
        causal_indicators = ["causes", "leads to", "results in", "produces", "triggers", "because of", "due to"]
        
        for obs in observations:
            if (str(relationship.cause.name).lower() in str(obs).lower() and 
                str(relationship.effect.name).lower() in str(obs).lower()):
                
                if any(indicator in str(obs).lower() for indicator in causal_indicators):
                    validation_score += 0.4
                    break
        
        # Check for mechanism description
        mechanism_indicators = ["mechanism", "process", "pathway", "method", "how", "through"]
        
        for obs in observations:
            if any(indicator in str(obs).lower() for indicator in mechanism_indicators):
                validation_score += 0.2
                break
        
        # Check for dose-response relationship
        dose_indicators = ["more", "less", "increase", "decrease", "higher", "lower"]
        
        for obs in observations:
            if (str(relationship.cause.name).lower() in str(obs).lower() and 
                str(relationship.effect.name).lower() in str(obs).lower()):
                
                if any(indicator in str(obs).lower() for indicator in dose_indicators):
                    validation_score += 0.1
                    break
        
        return min(1.0, validation_score)
    
    async def _check_confounding(
        self, 
        relationship: CausalRelationship, 
        variables: List[CausalVariable], 
        observations: List[str]
    ) -> List[CausalVariable]:
        """Check for potential confounding variables"""
        
        confounding_vars = []
        
        for var in variables:
            if var != relationship.cause and var != relationship.effect:
                # Check if this variable appears with both cause and effect
                var_with_cause = 0
                var_with_effect = 0
                
                for obs in observations:
                    obs_lower = str(obs).lower()
                    if str(var.name).lower() in obs_lower:
                        if str(relationship.cause.name).lower() in obs_lower:
                            var_with_cause += 1
                        if str(relationship.effect.name).lower() in obs_lower:
                            var_with_effect += 1
                
                # If variable appears with both cause and effect, it might be confounding
                if var_with_cause > 0 and var_with_effect > 0:
                    var.is_confounding = True
                    confounding_vars.append(var)
        
        return confounding_vars
    
    async def _assess_temporal_precedence(
        self, 
        relationship: CausalRelationship, 
        observations: List[str]
    ) -> float:
        """Assess temporal precedence of cause before effect"""
        
        # Look for temporal indicators
        temporal_indicators = {
            "before": 0.8,
            "after": 0.2,
            "then": 0.7,
            "next": 0.6,
            "subsequently": 0.7,
            "followed by": 0.8,
            "preceding": 0.8,
            "first": 0.9,
            "initially": 0.8,
            "later": 0.3,
            "finally": 0.1
        }
        
        temporal_score = 0.5  # Default neutral score
        
        for obs in observations:
            obs_lower = str(obs).lower()
            
            # Check if both cause and effect are mentioned
            if (str(relationship.cause.name).lower() in obs_lower and 
                str(relationship.effect.name).lower() in obs_lower):
                
                # Look for temporal indicators
                for indicator, score in temporal_indicators.items():
                    if indicator in obs_lower:
                        # Check relative position
                        cause_pos = obs_lower.find(str(relationship.cause.name).lower())
                        effect_pos = obs_lower.find(str(relationship.effect.name).lower())
                        indicator_pos = obs_lower.find(indicator)
                        
                        # Adjust score based on temporal logic
                        if cause_pos < indicator_pos < effect_pos:
                            temporal_score = max(temporal_score, score)
                        elif effect_pos < indicator_pos < cause_pos:
                            temporal_score = max(temporal_score, 1.0 - score)
        
        return temporal_score
    
    async def _build_causal_model(
        self, 
        variables: List[CausalVariable], 
        relationships: List[CausalRelationship]
    ) -> CausalModel:
        """Build a causal model from variables and relationships"""
        
        model = CausalModel(
            id=str(uuid4()),
            name="Discovered Causal Model",
            description="Causal model built from observational data",
            variables=variables,
            relationships=relationships
        )
        
        # Calculate model complexity
        model.complexity = len(relationships) / max(len(variables), 1)
        
        # Generate testable interventions
        model.testable_interventions = await self._generate_testable_interventions(model)
        
        # Generate causal predictions
        model.causal_predictions = await self._generate_causal_predictions(model)
        
        # Add assumptions
        model.assumptions = [
            "Observed variables capture the relevant causal structure",
            "No unmeasured confounders significantly affect the relationships",
            "Causal relationships are stable over the observation period",
            "Temporal precedence indicates causal direction where observed"
        ]
        
        # Add limitations
        model.limitations = [
            "Based on observational data - experimental validation needed",
            "May miss important confounding variables",
            "Temporal relationships may not reflect true causal timing",
            "Causal strength estimates are approximate"
        ]
        
        return model
    
    async def _generate_testable_interventions(self, model: CausalModel) -> List[str]:
        """Generate testable interventions from causal model"""
        
        interventions = []
        
        for relationship in model.relationships:
            if relationship.causal_strength >= 0.5:
                interventions.append(
                    f"Intervene on {relationship.cause.name} to test effect on {relationship.effect.name}"
                )
                
                interventions.append(
                    f"Control for {relationship.cause.name} to see if {relationship.effect.name} changes"
                )
        
        # Add confounding control interventions
        confounding_vars = set()
        for relationship in model.relationships:
            confounding_vars.update(relationship.confounding_variables)
        
        for confounder in confounding_vars:
            interventions.append(f"Control for {confounder.name} to test for confounding")
        
        return interventions
    
    async def _generate_causal_predictions(self, model: CausalModel) -> List[str]:
        """Generate causal predictions from model"""
        
        predictions = []
        
        for relationship in model.relationships:
            predictions.append(
                f"If {relationship.cause.name} changes, {relationship.effect.name} should change in the same direction"
            )
            
            if relationship.causal_strength >= 0.7:
                predictions.append(
                    f"Strong changes in {relationship.cause.name} should produce proportional changes in {relationship.effect.name}"
                )
        
        # Add interaction predictions
        cause_counts = Counter(rel.cause for rel in model.relationships)
        
        for cause, count in cause_counts.items():
            if count > 1:
                effects = [rel.effect.name for rel in model.relationships if rel.cause == cause]
                predictions.append(
                    f"Changes in {cause.name} should simultaneously affect {', '.join(effects)}"
                )
        
        return predictions
    
    async def _identify_confounding_factors(
        self, 
        model: CausalModel, 
        observations: List[str]
    ) -> List[CausalVariable]:
        """Identify confounding factors in the causal model"""
        
        confounding_factors = []
        
        # Collect all confounding variables from relationships
        for relationship in model.relationships:
            confounding_factors.extend(relationship.confounding_variables)
        
        # Remove duplicates
        confounding_factors = list(set(confounding_factors))
        
        # Look for additional potential confounders
        for var in model.variables:
            if var not in confounding_factors:
                # Check if this variable might be a confounder
                confounder_score = await self._assess_confounding_potential(var, model, observations)
                
                if confounder_score >= 0.6:
                    var.is_confounding = True
                    confounding_factors.append(var)
        
        return confounding_factors
    
    async def _assess_confounding_potential(
        self, 
        variable: CausalVariable, 
        model: CausalModel, 
        observations: List[str]
    ) -> float:
        """Assess the potential for a variable to be a confounder"""
        
        confounder_score = 0.0
        
        # Check if variable appears frequently with multiple other variables
        co_occurrence_count = 0
        
        for obs in observations:
            obs_lower = str(obs).lower()
            if str(variable.name).lower() in obs_lower:
                # Count co-occurrences with other variables
                for other_var in model.variables:
                    if other_var != variable and str(other_var.name).lower() in obs_lower:
                        co_occurrence_count += 1
        
        # Higher co-occurrence suggests potential confounding
        if co_occurrence_count >= 3:
            confounder_score += 0.4
        
        # Check for common cause indicators
        common_cause_indicators = ["common", "shared", "underlying", "both", "all"]
        
        for obs in observations:
            obs_lower = str(obs).lower()
            if str(variable.name).lower() in obs_lower:
                if any(indicator in obs_lower for indicator in common_cause_indicators):
                    confounder_score += 0.3
                    break
        
        # Check for third variable indicators
        third_var_indicators = ["third", "additional", "another", "also", "factor", "influence"]
        
        for obs in observations:
            obs_lower = str(obs).lower()
            if str(variable.name).lower() in obs_lower:
                if any(indicator in obs_lower for indicator in third_var_indicators):
                    confounder_score += 0.2
                    break
        
        return min(1.0, confounder_score)
    
    async def _generate_causal_analysis(
        self, 
        model: CausalModel, 
        confounding_factors: List[CausalVariable], 
        observations: List[str], 
        context: Dict[str, Any] = None
    ) -> CausalAnalysis:
        """Generate comprehensive causal analysis"""
        
        # Identify primary causal relationships
        primary_relationships = [
            rel for rel in model.relationships 
            if rel.causal_strength >= 0.5 and rel.confidence >= 0.5
        ]
        
        # Generate causal conclusions
        causal_conclusions = await self._generate_causal_conclusions(model, primary_relationships)
        
        # Generate intervention recommendations
        intervention_recommendations = await self._generate_intervention_recommendations(model)
        
        # Calculate overall confidence
        overall_confidence = await self._calculate_overall_confidence(model, primary_relationships)
        
        # Calculate causal certainty
        causal_certainty = await self._calculate_causal_certainty(model, observations)
        
        # Generate required tests
        required_tests = await self._generate_required_tests(model)
        
        # Generate alternative explanations
        alternative_explanations = await self._generate_alternative_explanations(model, observations)
        
        analysis = CausalAnalysis(
            id=str(uuid4()),
            query=context.get("query", "Causal analysis") if context else "Causal analysis",
            observations=observations,
            causal_model=model,
            primary_causal_relationships=primary_relationships,
            confounding_factors=confounding_factors,
            causal_conclusions=causal_conclusions,
            intervention_recommendations=intervention_recommendations,
            overall_confidence=overall_confidence,
            causal_certainty=causal_certainty,
            required_tests=required_tests,
            alternative_explanations=alternative_explanations
        )
        
        return analysis
    
    async def _generate_causal_conclusions(
        self, 
        model: CausalModel, 
        primary_relationships: List[CausalRelationship]
    ) -> List[str]:
        """Generate causal conclusions from the model"""
        
        conclusions = []
        
        if not primary_relationships:
            conclusions.append("No strong causal relationships identified in the data")
            return conclusions
        
        # Main causal relationships
        for relationship in primary_relationships:
            strength_desc = relationship.strength_category.value.replace("_", " ")
            conclusions.append(
                f"{relationship.cause.name} has a {strength_desc} causal effect on {relationship.effect.name}"
            )
        
        # Confounding analysis
        confounded_relationships = [rel for rel in primary_relationships if rel.confounding_variables]
        
        if confounded_relationships:
            conclusions.append(
                f"{len(confounded_relationships)} causal relationships may be confounded by other variables"
            )
        
        # Temporal analysis
        temporal_relationships = [rel for rel in primary_relationships if rel.temporal_relation != TemporalRelation.MEDIUM_TERM]
        
        if temporal_relationships:
            conclusions.append(
                f"Causal effects show varied temporal patterns: {len(temporal_relationships)} relationships have specific timing"
            )
        
        # Mechanism analysis
        known_mechanisms = [rel for rel in primary_relationships if rel.mechanism != CausalMechanism.UNKNOWN]
        
        if known_mechanisms:
            conclusions.append(
                f"Causal mechanisms are identified for {len(known_mechanisms)} relationships"
            )
        
        return conclusions
    
    async def _generate_intervention_recommendations(self, model: CausalModel) -> List[str]:
        """Generate intervention recommendations"""
        
        recommendations = []
        
        # Strongest causal relationships
        strong_relationships = [
            rel for rel in model.relationships 
            if rel.causal_strength >= 0.7 and rel.confidence >= 0.7
        ]
        
        for relationship in strong_relationships:
            recommendations.append(
                f"To affect {relationship.effect.name}, consider intervening on {relationship.cause.name}"
            )
        
        # Confounding control
        confounding_vars = set()
        for relationship in model.relationships:
            confounding_vars.update(relationship.confounding_variables)
        
        if confounding_vars:
            for confounder in confounding_vars:
                recommendations.append(
                    f"Control for {confounder.name} to isolate causal effects"
                )
        
        # Multiple cause variables
        effect_counts = Counter(rel.effect for rel in model.relationships)
        
        for effect, count in effect_counts.items():
            if count > 1:
                causes = [rel.cause.name for rel in model.relationships if rel.effect == effect]
                recommendations.append(
                    f"To maximize effect on {effect.name}, consider coordinated intervention on {', '.join(causes)}"
                )
        
        return recommendations
    
    async def _calculate_overall_confidence(
        self, 
        model: CausalModel, 
        primary_relationships: List[CausalRelationship]
    ) -> float:
        """Calculate overall confidence in causal analysis"""
        
        if not primary_relationships:
            return 0.1
        
        # Average relationship confidence
        avg_confidence = sum(rel.confidence for rel in primary_relationships) / len(primary_relationships)
        
        # Adjust for model complexity
        complexity_penalty = min(0.2, model.complexity * 0.1)
        
        # Adjust for confounding
        confounded_count = sum(1 for rel in primary_relationships if rel.confounding_variables)
        confounding_penalty = min(0.3, confounded_count * 0.1)
        
        overall_confidence = avg_confidence - complexity_penalty - confounding_penalty
        
        return max(0.1, min(1.0, overall_confidence))
    
    async def _calculate_causal_certainty(self, model: CausalModel, observations: List[str]) -> float:
        """Calculate certainty in causal claims"""
        
        certainty = 0.5  # Base certainty
        
        # Increase certainty for experimental evidence
        experimental_count = sum(1 for rel in model.relationships if rel.experimental_support)
        if experimental_count > 0:
            certainty += 0.3
        
        # Increase certainty for temporal evidence
        temporal_count = sum(1 for rel in model.relationships if rel.temporal_relation != TemporalRelation.MEDIUM_TERM)
        if temporal_count > 0:
            certainty += 0.2
        
        # Decrease certainty for confounding
        confounding_count = sum(len(rel.confounding_variables) for rel in model.relationships)
        if confounding_count > 0:
            certainty -= min(0.3, confounding_count * 0.05)
        
        # Decrease certainty for weak evidence
        weak_count = sum(1 for rel in model.relationships if rel.evidence_strength < 0.5)
        if weak_count > 0:
            certainty -= min(0.2, weak_count * 0.05)
        
        return max(0.1, min(1.0, certainty))
    
    async def _generate_required_tests(self, model: CausalModel) -> List[str]:
        """Generate required tests to validate causal model"""
        
        tests = []
        
        # Experimental tests for strong relationships
        for relationship in model.relationships:
            if relationship.causal_strength >= 0.6:
                tests.append(
                    f"Experimental test: Manipulate {relationship.cause.name} and measure {relationship.effect.name}"
                )
        
        # Confounding tests
        confounding_vars = set()
        for relationship in model.relationships:
            confounding_vars.update(relationship.confounding_variables)
        
        for confounder in confounding_vars:
            tests.append(f"Control for {confounder.name} to test for confounding")
        
        # Temporal tests
        tests.append("Longitudinal study to confirm temporal precedence")
        
        # Mechanism tests
        tests.append("Investigate causal mechanisms through process analysis")
        
        return tests
    
    async def _generate_alternative_explanations(
        self, 
        model: CausalModel, 
        observations: List[str]
    ) -> List[str]:
        """Generate alternative explanations for the observations"""
        
        alternatives = []
        
        # Reverse causation
        for relationship in model.relationships:
            alternatives.append(
                f"Reverse causation: {relationship.effect.name} might cause {relationship.cause.name}"
            )
        
        # Common cause
        alternatives.append("Common cause: An unmeasured variable causes both apparent cause and effect")
        
        # Spurious correlation
        alternatives.append("Spurious correlation: Relationship is coincidental, not causal")
        
        # Mediating variables
        alternatives.append("Mediation: Relationship is indirect through unmeasured mediating variables")
        
        # Selection bias
        alternatives.append("Selection bias: Observed relationships due to biased sampling")
        
        # Measurement error
        alternatives.append("Measurement error: Apparent relationships due to correlated measurement errors")
        
        return alternatives
    
    async def _enhance_causal_analysis(self, analysis: CausalAnalysis) -> CausalAnalysis:
        """Enhance causal analysis with additional insights"""
        
        # Add model validation metrics
        analysis.causal_model.goodness_of_fit = await self._calculate_goodness_of_fit(analysis.causal_model)
        analysis.causal_model.causal_validity = await self._calculate_causal_validity(analysis.causal_model)
        
        return analysis
    
    async def _calculate_goodness_of_fit(self, model: CausalModel) -> float:
        """Calculate goodness of fit for causal model"""
        
        # Simple metric based on relationship strength
        if not model.relationships:
            return 0.0
        
        avg_strength = sum(rel.causal_strength for rel in model.relationships) / len(model.relationships)
        avg_confidence = sum(rel.confidence for rel in model.relationships) / len(model.relationships)
        
        return (avg_strength + avg_confidence) / 2
    
    async def _calculate_causal_validity(self, model: CausalModel) -> float:
        """Calculate causal validity of the model"""
        
        if not model.relationships:
            return 0.0
        
        # Factors that increase validity
        experimental_support = sum(1 for rel in model.relationships if rel.experimental_support)
        temporal_support = sum(1 for rel in model.relationships if rel.temporal_relation != TemporalRelation.MEDIUM_TERM)
        
        # Factors that decrease validity
        confounding_issues = sum(len(rel.confounding_variables) for rel in model.relationships)
        weak_evidence = sum(1 for rel in model.relationships if rel.evidence_strength < 0.5)
        
        validity = 0.5  # Base validity
        
        # Adjust for supporting factors
        validity += (experimental_support / len(model.relationships)) * 0.3
        validity += (temporal_support / len(model.relationships)) * 0.2
        
        # Adjust for undermining factors
        validity -= min(0.3, confounding_issues * 0.05)
        validity -= min(0.2, weak_evidence * 0.05)
        
        return max(0.1, min(1.0, validity))
    
    def _initialize_causal_patterns(self) -> List[Dict[str, Any]]:
        """Initialize common causal patterns"""
        
        patterns = [
            {
                "name": "Dose-Response",
                "type": "direct_cause",
                "keywords": ["dose", "amount", "level", "more", "less", "increase", "decrease"],
                "strength": 0.7,
                "mechanism": "physical"
            },
            {
                "name": "Temporal Sequence",
                "type": "direct_cause",
                "keywords": ["before", "after", "then", "next", "following", "subsequent"],
                "strength": 0.6,
                "mechanism": "unknown"
            },
            {
                "name": "Necessary Condition",
                "type": "necessary_cause",
                "keywords": ["necessary", "required", "essential", "must", "need"],
                "strength": 0.8,
                "mechanism": "unknown"
            },
            {
                "name": "Sufficient Condition",
                "type": "sufficient_cause",
                "keywords": ["sufficient", "enough", "guarantee", "ensure", "always"],
                "strength": 0.8,
                "mechanism": "unknown"
            },
            {
                "name": "Mechanism Description",
                "type": "direct_cause",
                "keywords": ["mechanism", "process", "pathway", "through", "via", "by"],
                "strength": 0.7,
                "mechanism": "unknown"
            }
        ]
        
        return patterns
    
    def get_causal_stats(self) -> Dict[str, Any]:
        """Get statistics about causal reasoning usage"""
        
        total_relationships = sum(len(model.relationships) for model in self.causal_models)
        
        return {
            "total_models": len(self.causal_models),
            "total_analyses": len(self.causal_analyses),
            "total_relationships": total_relationships,
            "relationship_types": {
                rt.value: sum(
                    sum(1 for rel in model.relationships if rel.relationship_type == rt)
                    for model in self.causal_models
                ) for rt in CausalRelationType
            },
            "causal_strengths": {
                cs.value: sum(
                    sum(1 for rel in model.relationships if rel.strength_category == cs)
                    for model in self.causal_models
                ) for cs in CausalStrength
            },
            "mechanisms": {
                cm.value: sum(
                    sum(1 for rel in model.relationships if rel.mechanism == cm)
                    for model in self.causal_models
                ) for cm in CausalMechanism
            },
            "average_model_confidence": sum(
                analysis.overall_confidence for analysis in self.causal_analyses
            ) / max(len(self.causal_analyses), 1),
            "average_causal_certainty": sum(
                analysis.causal_certainty for analysis in self.causal_analyses
            ) / max(len(self.causal_analyses), 1)
        }