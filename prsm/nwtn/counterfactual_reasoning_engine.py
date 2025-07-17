#!/usr/bin/env python3
"""
NWTN Counterfactual Reasoning Engine
Hypothetical scenario evaluation and "what if" reasoning

This module implements NWTN's counterfactual reasoning capabilities, which allow the system to:
1. Generate hypothetical scenarios by altering key variables
2. Evaluate consequences of alternative actions or conditions
3. Assess causal relationships through counterfactual analysis
4. Support decision-making through scenario comparison
5. Handle temporal counterfactuals and alternative histories

Counterfactual reasoning is essential for learning from experience, planning
future actions, and understanding causal relationships by considering
alternative possibilities.

Key Concepts:
- Counterfactual scenario generation
- Alternative world modeling
- Causal intervention analysis
- Temporal reasoning and alternative histories
- Consequence evaluation and comparison
- Decision support through scenario analysis

Usage:
    from prsm.nwtn.counterfactual_reasoning_engine import CounterfactualReasoningEngine
    
    engine = CounterfactualReasoningEngine()
    result = await engine.evaluate_counterfactual(scenario, intervention, context)
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


class CounterfactualType(str, Enum):
    """Types of counterfactual scenarios"""
    CAUSAL = "causal"                   # "What if X had caused Y?"
    TEMPORAL = "temporal"               # "What if X happened at time T?"
    CONDITIONAL = "conditional"         # "What if condition X were true?"
    BEHAVIORAL = "behavioral"           # "What if agent X acted differently?"
    STRUCTURAL = "structural"           # "What if structure X were different?"
    PROBABILISTIC = "probabilistic"     # "What if probability of X were different?"
    COUNTERFACTUAL_HISTORY = "counterfactual_history"  # Alternative historical scenarios


class InterventionType(str, Enum):
    """Types of interventions in counterfactual scenarios"""
    ADDITION = "addition"               # Adding something that wasn't there
    REMOVAL = "removal"                 # Removing something that was there
    MODIFICATION = "modification"       # Changing a property or value
    SUBSTITUTION = "substitution"       # Replacing one thing with another
    TEMPORAL_SHIFT = "temporal_shift"   # Changing when something happened
    ORDERING_CHANGE = "ordering_change" # Changing the order of events
    MAGNITUDE_CHANGE = "magnitude_change" # Changing the size/intensity


class ModalityType(str, Enum):
    """Modalities for counterfactual reasoning"""
    NECESSITY = "necessity"             # Must have happened
    POSSIBILITY = "possibility"         # Could have happened
    IMPOSSIBILITY = "impossibility"     # Could not have happened
    PROBABILITY = "probability"         # Likelihood of happening
    INEVITABILITY = "inevitability"     # Would certainly happen


class TemporalRelation(str, Enum):
    """Temporal relationships in counterfactual scenarios"""
    BEFORE = "before"                   # Intervention before outcome
    AFTER = "after"                     # Intervention after outcome
    SIMULTANEOUS = "simultaneous"       # Intervention at same time
    SPANNING = "spanning"               # Intervention spans time period
    RECURRING = "recurring"             # Repeated intervention


class ConfidenceLevel(str, Enum):
    """Confidence levels for counterfactual conclusions"""
    VERY_HIGH = "very_high"      # >90% confidence
    HIGH = "high"                # 70-90% confidence
    MODERATE = "moderate"        # 50-70% confidence
    LOW = "low"                  # 30-50% confidence
    VERY_LOW = "very_low"        # <30% confidence


@dataclass
class CounterfactualScenario:
    """A hypothetical scenario for counterfactual reasoning"""
    
    id: str
    description: str
    scenario_type: CounterfactualType
    
    # Original context
    actual_world: Dict[str, Any]
    actual_outcome: str
    
    # Counterfactual world
    counterfactual_world: Dict[str, Any]
    hypothetical_intervention: str
    
    # Intervention details
    intervention_type: InterventionType
    intervention_target: str
    intervention_value: Any
    
    # Temporal information
    temporal_relation: TemporalRelation
    intervention_time: Optional[str] = None
    
    # Context and constraints
    domain: str = "general"
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    
    # Validation
    plausibility: float = 0.5
    consistency: float = 0.5
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CounterfactualConsequence:
    """A consequence in a counterfactual scenario"""
    
    id: str
    description: str
    consequence_type: str  # "direct", "indirect", "side_effect", "chain_reaction"
    
    # Causal information
    caused_by: str
    affects: List[str]
    
    # Probability and confidence
    probability: float = 0.5
    confidence: float = 0.5
    
    # Temporal information
    timing: str = "unknown"  # "immediate", "short_term", "long_term"
    duration: str = "unknown"  # "brief", "extended", "permanent"
    
    # Impact assessment
    magnitude: str = "moderate"  # "minimal", "moderate", "significant", "major"
    valence: str = "neutral"  # "positive", "negative", "neutral", "mixed"
    
    # Evidence
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CounterfactualComparison:
    """Comparison between actual and counterfactual scenarios"""
    
    id: str
    actual_scenario: str
    counterfactual_scenario: str
    
    # Outcome differences
    outcome_differences: List[str]
    key_changes: List[str]
    
    # Preference assessment
    preferable_scenario: str  # "actual", "counterfactual", "unclear"
    preference_reasons: List[str]
    
    # Metrics
    similarity_score: float = 0.5
    impact_score: float = 0.5
    desirability_score: float = 0.5
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CounterfactualAnalysis:
    """Complete analysis of a counterfactual reasoning task"""
    
    id: str
    query: str
    counterfactual_type: CounterfactualType
    
    # Scenario
    scenario: CounterfactualScenario
    
    # Consequences
    direct_consequences: List[CounterfactualConsequence]
    indirect_consequences: List[CounterfactualConsequence]
    
    # Analysis
    causal_chain: List[str]
    key_factors: List[str]
    
    # Comparison
    comparison: CounterfactualComparison
    
    # Assessment
    modality: ModalityType
    confidence_level: ConfidenceLevel
    overall_probability: float
    
    # Insights
    insights: List[str]
    implications: List[str]
    limitations: List[str]
    
    # Validation
    consistency_check: bool = True
    plausibility_check: bool = True
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CounterfactualReasoningEngine:
    """
    Engine for counterfactual reasoning and hypothetical scenario evaluation
    
    This system enables NWTN to explore alternative possibilities and
    understand causal relationships through "what if" analysis.
    """
    
    def __init__(self):
        self.model_executor = ModelExecutor(agent_id="counterfactual_reasoning_engine")
        self.world_model = WorldModelEngine()
        
        # Analysis storage
        self.counterfactual_analyses: List[CounterfactualAnalysis] = []
        self.scenario_library: List[CounterfactualScenario] = []
        
        # Configuration
        self.min_plausibility_threshold = 0.3
        self.min_consistency_threshold = 0.5
        self.max_causal_chain_length = 10
        self.temporal_reasoning_enabled = True
        
        logger.info("Initialized Counterfactual Reasoning Engine")
    
    async def evaluate_counterfactual(
        self, 
        query: str, 
        context: Dict[str, Any] = None
    ) -> CounterfactualAnalysis:
        """
        Evaluate a counterfactual reasoning query
        
        Args:
            query: Counterfactual query (e.g., "What if X had happened?")
            context: Additional context for reasoning
            
        Returns:
            CounterfactualAnalysis: Complete analysis of the counterfactual scenario
        """
        
        logger.info(
            "Starting counterfactual reasoning",
            query=query[:100] + "..." if len(query) > 100 else query
        )
        
        # Step 1: Parse counterfactual query
        counterfactual_type = await self._classify_counterfactual_type(query)
        
        # Step 2: Generate counterfactual scenario
        scenario = await self._generate_counterfactual_scenario(query, counterfactual_type, context)
        
        # Step 3: Evaluate consequences
        direct_consequences = await self._evaluate_direct_consequences(scenario)
        indirect_consequences = await self._evaluate_indirect_consequences(scenario, direct_consequences)
        
        # Step 4: Build causal chain
        causal_chain = await self._build_causal_chain(scenario, direct_consequences, indirect_consequences)
        
        # Step 5: Identify key factors
        key_factors = await self._identify_key_factors(scenario, causal_chain)
        
        # Step 6: Compare scenarios
        comparison = await self._compare_scenarios(scenario, direct_consequences, indirect_consequences)
        
        # Step 7: Assess modality and confidence
        modality = await self._assess_modality(scenario, direct_consequences)
        confidence_level = await self._assess_confidence_level(scenario, direct_consequences, indirect_consequences)
        overall_probability = await self._calculate_overall_probability(scenario, direct_consequences)
        
        # Step 8: Generate insights
        insights = await self._generate_insights(scenario, direct_consequences, indirect_consequences, causal_chain)
        implications = await self._identify_implications(scenario, comparison, insights)
        limitations = await self._identify_limitations(scenario, context)
        
        # Step 9: Validate analysis
        consistency_check = await self._check_consistency(scenario, direct_consequences, indirect_consequences)
        plausibility_check = await self._check_plausibility(scenario, direct_consequences)
        
        # Create analysis
        analysis = CounterfactualAnalysis(
            id=str(uuid4()),
            query=query,
            counterfactual_type=counterfactual_type,
            scenario=scenario,
            direct_consequences=direct_consequences,
            indirect_consequences=indirect_consequences,
            causal_chain=causal_chain,
            key_factors=key_factors,
            comparison=comparison,
            modality=modality,
            confidence_level=confidence_level,
            overall_probability=overall_probability,
            insights=insights,
            implications=implications,
            limitations=limitations,
            consistency_check=consistency_check,
            plausibility_check=plausibility_check
        )
        
        # Store results
        self.counterfactual_analyses.append(analysis)
        self.scenario_library.append(scenario)
        
        logger.info(
            "Counterfactual reasoning complete",
            scenario_type=counterfactual_type,
            direct_consequences=len(direct_consequences),
            indirect_consequences=len(indirect_consequences),
            confidence_level=confidence_level,
            overall_probability=overall_probability
        )
        
        return analysis
    
    async def _classify_counterfactual_type(self, query: str) -> CounterfactualType:
        """Classify the type of counterfactual query"""
        
        query_lower = str(query).lower()
        
        # Pattern matching for different types
        if any(pattern in query_lower for pattern in ["what if", "if only", "suppose", "imagine"]):
            if any(pattern in query_lower for pattern in ["caused", "because", "due to", "resulted in"]):
                return CounterfactualType.CAUSAL
            elif any(pattern in query_lower for pattern in ["earlier", "later", "before", "after", "when"]):
                return CounterfactualType.TEMPORAL
            elif any(pattern in query_lower for pattern in ["acted", "decided", "chose", "did"]):
                return CounterfactualType.BEHAVIORAL
            elif any(pattern in query_lower for pattern in ["structured", "organized", "designed"]):
                return CounterfactualType.STRUCTURAL
            elif any(pattern in query_lower for pattern in ["probability", "chance", "likely", "unlikely"]):
                return CounterfactualType.PROBABILISTIC
            elif any(pattern in query_lower for pattern in ["history", "past", "happened", "occurred"]):
                return CounterfactualType.COUNTERFACTUAL_HISTORY
            else:
                return CounterfactualType.CONDITIONAL
        
        # Default to conditional
        return CounterfactualType.CONDITIONAL
    
    async def _generate_counterfactual_scenario(
        self, 
        query: str, 
        counterfactual_type: CounterfactualType, 
        context: Dict[str, Any] = None
    ) -> CounterfactualScenario:
        """Generate a counterfactual scenario from the query"""
        
        # Parse intervention from query
        intervention = await self._parse_intervention(query)
        
        # Extract actual world state
        actual_world = await self._extract_actual_world(query, context)
        
        # Generate counterfactual world
        counterfactual_world = await self._generate_counterfactual_world(actual_world, intervention)
        
        # Determine intervention details
        intervention_type = await self._classify_intervention_type(intervention)
        intervention_target = await self._identify_intervention_target(intervention)
        intervention_value = await self._extract_intervention_value(intervention)
        
        # Determine temporal relation
        temporal_relation = await self._determine_temporal_relation(query, intervention)
        
        # Assess plausibility and consistency
        plausibility = await self._assess_scenario_plausibility(counterfactual_world, actual_world)
        consistency = await self._assess_scenario_consistency(counterfactual_world, intervention)
        
        # Create scenario
        scenario = CounterfactualScenario(
            id=str(uuid4()),
            description=await self._generate_scenario_description(query, intervention),
            scenario_type=counterfactual_type,
            actual_world=actual_world,
            actual_outcome=await self._extract_actual_outcome(query, actual_world),
            counterfactual_world=counterfactual_world,
            hypothetical_intervention=intervention,
            intervention_type=intervention_type,
            intervention_target=intervention_target,
            intervention_value=intervention_value,
            temporal_relation=temporal_relation,
            domain=await self._determine_domain(query, context),
            context=context or {},
            constraints=await self._identify_constraints(query, context),
            plausibility=plausibility,
            consistency=consistency
        )
        
        return scenario
    
    async def _parse_intervention(self, query: str) -> str:
        """Parse the intervention from the counterfactual query"""
        
        # Extract intervention patterns
        intervention_patterns = [
            r'what if (.+?) had',
            r'what if (.+?) was',
            r'what if (.+?) were',
            r'suppose (.+?) had',
            r'imagine (.+?) was',
            r'if only (.+?) had',
            r'if (.+?) then'
        ]
        
        query_lower = str(query).lower()
        
        for pattern in intervention_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                return matches[0].strip()
        
        # Fallback: extract everything after "what if"
        if "what if" in query_lower:
            return query_lower.split("what if")[1].strip()
        
        return "unknown intervention"
    
    async def _extract_actual_world(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract the actual world state from query and context"""
        
        actual_world = {}
        
        # Extract from context
        if context:
            actual_world.update(context)
        
        # Extract entities and properties from query
        entities = await self._extract_entities(query)
        properties = await self._extract_properties(query)
        
        actual_world["entities"] = entities
        actual_world["properties"] = properties
        
        # Extract relationships
        relationships = await self._extract_relationships(query)
        actual_world["relationships"] = relationships
        
        return actual_world
    
    async def _generate_counterfactual_world(
        self, 
        actual_world: Dict[str, Any], 
        intervention: str
    ) -> Dict[str, Any]:
        """Generate the counterfactual world state"""
        
        # Start with actual world
        counterfactual_world = actual_world.copy()
        
        # Apply intervention
        intervention_changes = await self._apply_intervention(intervention, actual_world)
        
        # Update world state
        for key, value in intervention_changes.items():
            counterfactual_world[key] = value
        
        # Propagate changes
        propagated_changes = await self._propagate_changes(counterfactual_world, intervention_changes)
        counterfactual_world.update(propagated_changes)
        
        return counterfactual_world
    
    async def _apply_intervention(self, intervention: str, actual_world: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the intervention to generate changes"""
        
        changes = {}
        
        # Parse intervention components
        intervention_lower = str(intervention).lower()
        
        # Handle different intervention types
        if any(word in intervention_lower for word in ["not", "didn't", "hadn't", "without"]):
            # Negation intervention
            changes["negation_applied"] = True
            changes["negated_element"] = intervention
        
        elif any(word in intervention_lower for word in ["more", "less", "increased", "decreased"]):
            # Magnitude intervention
            changes["magnitude_changed"] = True
            changes["magnitude_direction"] = "increased" if any(word in intervention_lower for word in ["more", "increased"]) else "decreased"
        
        elif any(word in intervention_lower for word in ["instead", "rather", "alternatively"]):
            # Substitution intervention
            changes["substitution_applied"] = True
            changes["substituted_element"] = intervention
        
        else:
            # General change
            changes["general_change"] = intervention
        
        return changes
    
    async def _propagate_changes(
        self, 
        counterfactual_world: Dict[str, Any], 
        intervention_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Propagate intervention changes through the world model"""
        
        propagated = {}
        
        # Simple propagation based on change type
        if "negation_applied" in intervention_changes:
            propagated["cascading_effects"] = ["negation_cascades"]
        
        if "magnitude_changed" in intervention_changes:
            propagated["cascading_effects"] = ["magnitude_cascades"]
        
        if "substitution_applied" in intervention_changes:
            propagated["cascading_effects"] = ["substitution_cascades"]
        
        return propagated
    
    async def _classify_intervention_type(self, intervention: str) -> InterventionType:
        """Classify the type of intervention"""
        
        intervention_lower = str(intervention).lower()
        
        if any(word in intervention_lower for word in ["not", "without", "didn't", "hadn't"]):
            return InterventionType.REMOVAL
        elif any(word in intervention_lower for word in ["more", "additional", "extra"]):
            return InterventionType.ADDITION
        elif any(word in intervention_lower for word in ["instead", "rather", "alternatively"]):
            return InterventionType.SUBSTITUTION
        elif any(word in intervention_lower for word in ["earlier", "later", "before", "after"]):
            return InterventionType.TEMPORAL_SHIFT
        elif any(word in intervention_lower for word in ["increased", "decreased", "bigger", "smaller"]):
            return InterventionType.MAGNITUDE_CHANGE
        elif any(word in intervention_lower for word in ["different", "changed", "modified"]):
            return InterventionType.MODIFICATION
        else:
            return InterventionType.MODIFICATION
    
    async def _identify_intervention_target(self, intervention: str) -> str:
        """Identify the target of the intervention"""
        
        # Extract nouns as potential targets
        entities = await self._extract_entities(intervention)
        
        if entities:
            return entities[0]
        
        return "unknown_target"
    
    async def _extract_intervention_value(self, intervention: str) -> Any:
        """Extract the value associated with the intervention"""
        
        # Extract numerical values
        numbers = re.findall(r'\d+\.?\d*', intervention)
        if numbers:
            return float(numbers[0])
        
        # Extract qualitative values
        if any(word in str(intervention).lower() for word in ["true", "false", "yes", "no"]):
            return any(word in str(intervention).lower() for word in ["true", "yes"])
        
        return intervention
    
    async def _determine_temporal_relation(self, query: str, intervention: str) -> TemporalRelation:
        """Determine the temporal relation of the intervention"""
        
        combined_text = str(query + " " + intervention).lower()
        
        if any(word in combined_text for word in ["before", "earlier", "prior"]):
            return TemporalRelation.BEFORE
        elif any(word in combined_text for word in ["after", "later", "following"]):
            return TemporalRelation.AFTER
        elif any(word in combined_text for word in ["during", "while", "simultaneously"]):
            return TemporalRelation.SIMULTANEOUS
        elif any(word in combined_text for word in ["throughout", "across", "spanning"]):
            return TemporalRelation.SPANNING
        elif any(word in combined_text for word in ["repeatedly", "regularly", "recurring"]):
            return TemporalRelation.RECURRING
        else:
            return TemporalRelation.SIMULTANEOUS
    
    async def _assess_scenario_plausibility(
        self, 
        counterfactual_world: Dict[str, Any], 
        actual_world: Dict[str, Any]
    ) -> float:
        """Assess the plausibility of the counterfactual scenario"""
        
        # Simple plausibility assessment
        base_plausibility = 0.7
        
        # Check for consistency with known facts
        if "negation_applied" in counterfactual_world:
            base_plausibility *= 0.8
        
        if "magnitude_changed" in counterfactual_world:
            base_plausibility *= 0.9
        
        if "substitution_applied" in counterfactual_world:
            base_plausibility *= 0.7
        
        return max(0.1, min(1.0, base_plausibility))
    
    async def _assess_scenario_consistency(
        self, 
        counterfactual_world: Dict[str, Any], 
        intervention: str
    ) -> float:
        """Assess the internal consistency of the counterfactual scenario"""
        
        # Simple consistency assessment
        base_consistency = 0.8
        
        # Check for logical contradictions
        if "negation_applied" in counterfactual_world and "general_change" in counterfactual_world:
            base_consistency *= 0.9
        
        return max(0.1, min(1.0, base_consistency))
    
    async def _generate_scenario_description(self, query: str, intervention: str) -> str:
        """Generate a description of the counterfactual scenario"""
        
        return f"Counterfactual scenario: {intervention} (derived from: {query})"
    
    async def _extract_actual_outcome(self, query: str, actual_world: Dict[str, Any]) -> str:
        """Extract the actual outcome from the query"""
        
        # Look for outcome indicators
        outcome_patterns = [
            r'then (.+?)$',
            r'resulted in (.+?)$',
            r'led to (.+?)$',
            r'caused (.+?)$'
        ]
        
        for pattern in outcome_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return "unknown outcome"
    
    async def _determine_domain(self, query: str, context: Dict[str, Any] = None) -> str:
        """Determine the domain of the counterfactual scenario"""
        
        if context and "domain" in context:
            return context["domain"]
        
        # Domain classification based on keywords
        domain_keywords = {
            "physics": ["force", "energy", "motion", "gravity", "quantum"],
            "history": ["war", "battle", "empire", "revolution", "historical"],
            "economics": ["market", "price", "trade", "economy", "financial"],
            "psychology": ["behavior", "decision", "emotion", "mental", "cognitive"],
            "medicine": ["disease", "treatment", "patient", "diagnosis", "health"],
            "technology": ["computer", "software", "system", "algorithm", "data"],
            "social": ["society", "community", "culture", "social", "group"]
        }
        
        query_lower = str(query).lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return "general"
    
    async def _identify_constraints(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """Identify constraints on the counterfactual scenario"""
        
        constraints = []
        
        # Extract constraint indicators
        if any(word in str(query).lower() for word in ["must", "should", "cannot", "impossible"]):
            constraints.append("logical_constraints")
        
        if any(word in str(query).lower() for word in ["physically", "naturally", "biologically"]):
            constraints.append("physical_constraints")
        
        if any(word in str(query).lower() for word in ["realistically", "practically", "feasibly"]):
            constraints.append("practical_constraints")
        
        if context and "constraints" in context:
            constraints.extend(context["constraints"])
        
        return constraints
    
    async def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text"""
        
        # Simple entity extraction
        entities = []
        
        # Extract proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(proper_nouns)
        
        # Extract common nouns
        common_patterns = [
            r'\bthe\s+([a-z]+)\b',
            r'\ba\s+([a-z]+)\b',
            r'\ban\s+([a-z]+)\b'
        ]
        
        for pattern in common_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        # Remove duplicates and stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        entities = list(set([str(entity).lower() for entity in entities if str(entity).lower() not in stop_words]))
        
        return entities[:5]  # Limit to top 5 entities
    
    async def _extract_properties(self, text: str) -> Dict[str, Any]:
        """Extract properties from text"""
        
        properties = {}
        
        # Extract numerical properties
        numbers = re.findall(r'(\d+\.?\d*)', text)
        if numbers:
            properties["numerical_values"] = [float(num) for num in numbers]
        
        # Extract qualitative properties
        qualities = re.findall(r'is\s+(very\s+)?(\w+)', text, re.IGNORECASE)
        if qualities:
            properties["qualities"] = [quality[1] for quality in qualities]
        
        return properties
    
    async def _extract_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relationships from text"""
        
        relationships = []
        
        # Extract relationship patterns
        relationship_patterns = [
            r'(\w+)\s+(causes?|leads?|results?)\s+(\w+)',
            r'(\w+)\s+(affects?|influences?)\s+(\w+)',
            r'(\w+)\s+(depends?\s+on|relies?\s+on)\s+(\w+)'
        ]
        
        for pattern in relationship_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for entity1, relation, entity2 in matches:
                relationships.append((str(entity1).lower(), str(relation).lower(), str(entity2).lower()))
        
        return relationships
    
    async def _evaluate_direct_consequences(self, scenario: CounterfactualScenario) -> List[CounterfactualConsequence]:
        """Evaluate direct consequences of the counterfactual scenario"""
        
        consequences = []
        
        # Analyze intervention effects
        intervention = scenario.hypothetical_intervention
        
        # Generate direct consequences based on intervention type
        if scenario.intervention_type == InterventionType.REMOVAL:
            consequence = CounterfactualConsequence(
                id=str(uuid4()),
                description=f"Direct consequence of removing {scenario.intervention_target}",
                consequence_type="direct",
                caused_by=intervention,
                affects=[scenario.intervention_target],
                probability=0.8,
                confidence=0.7,
                timing="immediate",
                magnitude="significant"
            )
            consequences.append(consequence)
        
        elif scenario.intervention_type == InterventionType.ADDITION:
            consequence = CounterfactualConsequence(
                id=str(uuid4()),
                description=f"Direct consequence of adding {scenario.intervention_target}",
                consequence_type="direct",
                caused_by=intervention,
                affects=[scenario.intervention_target],
                probability=0.8,
                confidence=0.7,
                timing="immediate",
                magnitude="moderate"
            )
            consequences.append(consequence)
        
        elif scenario.intervention_type == InterventionType.MODIFICATION:
            consequence = CounterfactualConsequence(
                id=str(uuid4()),
                description=f"Direct consequence of modifying {scenario.intervention_target}",
                consequence_type="direct",
                caused_by=intervention,
                affects=[scenario.intervention_target],
                probability=0.7,
                confidence=0.6,
                timing="immediate",
                magnitude="moderate"
            )
            consequences.append(consequence)
        
        # Add more specific consequences based on domain
        domain_consequences = await self._generate_domain_specific_consequences(scenario)
        consequences.extend(domain_consequences)
        
        return consequences
    
    async def _generate_domain_specific_consequences(self, scenario: CounterfactualScenario) -> List[CounterfactualConsequence]:
        """Generate domain-specific consequences"""
        
        consequences = []
        
        if scenario.domain == "physics":
            consequence = CounterfactualConsequence(
                id=str(uuid4()),
                description="Physical system behavior would change",
                consequence_type="direct",
                caused_by=scenario.hypothetical_intervention,
                affects=["physical_system"],
                probability=0.9,
                confidence=0.8,
                timing="immediate",
                magnitude="significant"
            )
            consequences.append(consequence)
        
        elif scenario.domain == "history":
            consequence = CounterfactualConsequence(
                id=str(uuid4()),
                description="Historical timeline would be altered",
                consequence_type="direct",
                caused_by=scenario.hypothetical_intervention,
                affects=["historical_timeline"],
                probability=0.8,
                confidence=0.6,
                timing="immediate",
                magnitude="major"
            )
            consequences.append(consequence)
        
        elif scenario.domain == "economics":
            consequence = CounterfactualConsequence(
                id=str(uuid4()),
                description="Economic conditions would change",
                consequence_type="direct",
                caused_by=scenario.hypothetical_intervention,
                affects=["economic_system"],
                probability=0.7,
                confidence=0.7,
                timing="short_term",
                magnitude="moderate"
            )
            consequences.append(consequence)
        
        return consequences
    
    async def _evaluate_indirect_consequences(
        self, 
        scenario: CounterfactualScenario, 
        direct_consequences: List[CounterfactualConsequence]
    ) -> List[CounterfactualConsequence]:
        """Evaluate indirect consequences of the counterfactual scenario"""
        
        indirect_consequences = []
        
        # Generate indirect consequences from direct ones
        for direct_consequence in direct_consequences:
            # Chain reactions
            for affected_entity in direct_consequence.affects:
                chain_consequence = CounterfactualConsequence(
                    id=str(uuid4()),
                    description=f"Chain reaction affecting {affected_entity}",
                    consequence_type="indirect",
                    caused_by=direct_consequence.description,
                    affects=[f"connected_to_{affected_entity}"],
                    probability=direct_consequence.probability * 0.7,
                    confidence=direct_consequence.confidence * 0.8,
                    timing="short_term",
                    magnitude="moderate"
                )
                indirect_consequences.append(chain_consequence)
            
            # Side effects
            side_effect = CounterfactualConsequence(
                id=str(uuid4()),
                description=f"Side effect of {direct_consequence.description}",
                consequence_type="side_effect",
                caused_by=direct_consequence.description,
                affects=["unintended_targets"],
                probability=direct_consequence.probability * 0.5,
                confidence=direct_consequence.confidence * 0.6,
                timing="medium_term",
                magnitude="minimal"
            )
            indirect_consequences.append(side_effect)
        
        return indirect_consequences
    
    async def _build_causal_chain(
        self, 
        scenario: CounterfactualScenario, 
        direct_consequences: List[CounterfactualConsequence], 
        indirect_consequences: List[CounterfactualConsequence]
    ) -> List[str]:
        """Build the causal chain for the counterfactual scenario"""
        
        chain = []
        
        # Start with intervention
        chain.append(f"Intervention: {scenario.hypothetical_intervention}")
        
        # Add direct consequences
        for consequence in direct_consequences:
            chain.append(f"Direct: {consequence.description}")
        
        # Add indirect consequences
        for consequence in indirect_consequences:
            chain.append(f"Indirect: {consequence.description}")
        
        # Limit chain length
        return chain[:self.max_causal_chain_length]
    
    async def _identify_key_factors(
        self, 
        scenario: CounterfactualScenario, 
        causal_chain: List[str]
    ) -> List[str]:
        """Identify key factors in the counterfactual scenario"""
        
        key_factors = []
        
        # Intervention target is always a key factor
        key_factors.append(scenario.intervention_target)
        
        # Extract key factors from causal chain
        for step in causal_chain:
            if "significant" in step or "major" in step:
                key_factors.append(step)
        
        # Domain-specific factors
        if scenario.domain == "physics":
            key_factors.append("physical_laws")
        elif scenario.domain == "history":
            key_factors.append("historical_context")
        elif scenario.domain == "economics":
            key_factors.append("market_conditions")
        
        return list(set(key_factors))
    
    async def _compare_scenarios(
        self, 
        scenario: CounterfactualScenario, 
        direct_consequences: List[CounterfactualConsequence], 
        indirect_consequences: List[CounterfactualConsequence]
    ) -> CounterfactualComparison:
        """Compare actual and counterfactual scenarios"""
        
        # Identify outcome differences
        outcome_differences = []
        for consequence in direct_consequences + indirect_consequences:
            outcome_differences.append(consequence.description)
        
        # Identify key changes
        key_changes = [
            f"Intervention: {scenario.hypothetical_intervention}",
            f"Target changed: {scenario.intervention_target}"
        ]
        
        # Assess preference (simplified)
        positive_consequences = sum(1 for c in direct_consequences + indirect_consequences if c.valence == "positive")
        negative_consequences = sum(1 for c in direct_consequences + indirect_consequences if c.valence == "negative")
        
        if positive_consequences > negative_consequences:
            preferable_scenario = "counterfactual"
            preference_reasons = ["More positive consequences"]
        elif negative_consequences > positive_consequences:
            preferable_scenario = "actual"
            preference_reasons = ["Fewer negative consequences"]
        else:
            preferable_scenario = "unclear"
            preference_reasons = ["Balanced consequences"]
        
        # Calculate metrics
        similarity_score = 1.0 - (len(outcome_differences) / 10)  # Simple similarity
        impact_score = len(direct_consequences) / 5  # Impact based on direct consequences
        desirability_score = 0.5  # Neutral default
        
        return CounterfactualComparison(
            id=str(uuid4()),
            actual_scenario=scenario.actual_outcome,
            counterfactual_scenario=scenario.description,
            outcome_differences=outcome_differences,
            key_changes=key_changes,
            preferable_scenario=preferable_scenario,
            preference_reasons=preference_reasons,
            similarity_score=max(0.0, min(1.0, similarity_score)),
            impact_score=max(0.0, min(1.0, impact_score)),
            desirability_score=desirability_score
        )
    
    async def _assess_modality(
        self, 
        scenario: CounterfactualScenario, 
        direct_consequences: List[CounterfactualConsequence]
    ) -> ModalityType:
        """Assess the modality of the counterfactual scenario"""
        
        # Calculate average probability of consequences
        if direct_consequences:
            avg_probability = sum(c.probability for c in direct_consequences) / len(direct_consequences)
        else:
            avg_probability = 0.5
        
        # Determine modality based on probability and plausibility
        if avg_probability >= 0.9 and scenario.plausibility >= 0.8:
            return ModalityType.INEVITABILITY
        elif avg_probability >= 0.7 and scenario.plausibility >= 0.6:
            return ModalityType.NECESSITY
        elif avg_probability >= 0.3 and scenario.plausibility >= 0.3:
            return ModalityType.POSSIBILITY
        elif avg_probability >= 0.1:
            return ModalityType.PROBABILITY
        else:
            return ModalityType.IMPOSSIBILITY
    
    async def _assess_confidence_level(
        self, 
        scenario: CounterfactualScenario, 
        direct_consequences: List[CounterfactualConsequence], 
        indirect_consequences: List[CounterfactualConsequence]
    ) -> ConfidenceLevel:
        """Assess confidence level in the counterfactual analysis"""
        
        # Calculate average confidence
        all_consequences = direct_consequences + indirect_consequences
        if all_consequences:
            avg_confidence = sum(c.confidence for c in all_consequences) / len(all_consequences)
        else:
            avg_confidence = 0.5
        
        # Adjust for scenario quality
        adjusted_confidence = avg_confidence * scenario.plausibility * scenario.consistency
        
        # Determine confidence level
        if adjusted_confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif adjusted_confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif adjusted_confidence >= 0.5:
            return ConfidenceLevel.MODERATE
        elif adjusted_confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    async def _calculate_overall_probability(
        self, 
        scenario: CounterfactualScenario, 
        direct_consequences: List[CounterfactualConsequence]
    ) -> float:
        """Calculate overall probability of the counterfactual scenario"""
        
        # Base probability on scenario plausibility
        base_probability = scenario.plausibility
        
        # Adjust for consequence probabilities
        if direct_consequences:
            avg_consequence_prob = sum(c.probability for c in direct_consequences) / len(direct_consequences)
            base_probability *= avg_consequence_prob
        
        # Adjust for consistency
        base_probability *= scenario.consistency
        
        return max(0.0, min(1.0, base_probability))
    
    async def _generate_insights(
        self, 
        scenario: CounterfactualScenario, 
        direct_consequences: List[CounterfactualConsequence], 
        indirect_consequences: List[CounterfactualConsequence], 
        causal_chain: List[str]
    ) -> List[str]:
        """Generate insights from the counterfactual analysis"""
        
        insights = []
        
        # Key insight about intervention
        insights.append(f"The intervention '{scenario.hypothetical_intervention}' would primarily affect {scenario.intervention_target}")
        
        # Insight about consequences
        if direct_consequences:
            insights.append(f"The analysis reveals {len(direct_consequences)} direct consequences")
        
        if indirect_consequences:
            insights.append(f"There would be {len(indirect_consequences)} indirect consequences")
        
        # Insight about causal chain
        if len(causal_chain) > 3:
            insights.append("The counterfactual would trigger a complex causal chain")
        
        # Domain-specific insights
        if scenario.domain == "physics":
            insights.append("Physical laws would constrain the possible outcomes")
        elif scenario.domain == "history":
            insights.append("Historical context would influence the alternative timeline")
        elif scenario.domain == "economics":
            insights.append("Economic forces would shape the counterfactual outcomes")
        
        return insights
    
    async def _identify_implications(
        self, 
        scenario: CounterfactualScenario, 
        comparison: CounterfactualComparison, 
        insights: List[str]
    ) -> List[str]:
        """Identify implications of the counterfactual analysis"""
        
        implications = []
        
        # Preference implications
        if comparison.preferable_scenario == "counterfactual":
            implications.append("The alternative scenario appears more desirable")
        elif comparison.preferable_scenario == "actual":
            implications.append("The actual scenario seems preferable")
        
        # Causal implications
        implications.append(f"The intervention would have {comparison.impact_score:.1f} impact level")
        
        # Learning implications
        implications.append("This analysis reveals important causal relationships")
        
        # Decision implications
        if scenario.plausibility >= 0.7:
            implications.append("The counterfactual scenario is plausible for decision-making")
        
        return implications
    
    async def _identify_limitations(
        self, 
        scenario: CounterfactualScenario, 
        context: Dict[str, Any] = None
    ) -> List[str]:
        """Identify limitations of the counterfactual analysis"""
        
        limitations = []
        
        # Plausibility limitations
        if scenario.plausibility < 0.5:
            limitations.append("Low plausibility reduces confidence in the analysis")
        
        # Consistency limitations
        if scenario.consistency < 0.5:
            limitations.append("Internal consistency issues limit reliability")
        
        # Data limitations
        if not context or len(context) < 3:
            limitations.append("Limited context affects analysis quality")
        
        # Domain limitations
        if scenario.domain == "general":
            limitations.append("General domain limits specific insights")
        
        # Temporal limitations
        if scenario.temporal_relation == TemporalRelation.SIMULTANEOUS:
            limitations.append("Temporal relationships are unclear")
        
        return limitations
    
    async def _check_consistency(
        self, 
        scenario: CounterfactualScenario, 
        direct_consequences: List[CounterfactualConsequence], 
        indirect_consequences: List[CounterfactualConsequence]
    ) -> bool:
        """Check consistency of the counterfactual analysis"""
        
        # Check scenario consistency
        if scenario.consistency < self.min_consistency_threshold:
            return False
        
        # Check consequence consistency
        all_consequences = direct_consequences + indirect_consequences
        if all_consequences:
            avg_confidence = sum(c.confidence for c in all_consequences) / len(all_consequences)
            if avg_confidence < 0.4:
                return False
        
        return True
    
    async def _check_plausibility(
        self, 
        scenario: CounterfactualScenario, 
        direct_consequences: List[CounterfactualConsequence]
    ) -> bool:
        """Check plausibility of the counterfactual analysis"""
        
        # Check scenario plausibility
        if scenario.plausibility < self.min_plausibility_threshold:
            return False
        
        # Check consequence plausibility
        if direct_consequences:
            avg_probability = sum(c.probability for c in direct_consequences) / len(direct_consequences)
            if avg_probability < 0.2:
                return False
        
        return True
    
    def get_counterfactual_stats(self) -> Dict[str, Any]:
        """Get statistics about counterfactual reasoning usage"""
        
        return {
            "total_analyses": len(self.counterfactual_analyses),
            "total_scenarios": len(self.scenario_library),
            "counterfactual_types": {ct.value: sum(1 for a in self.counterfactual_analyses if a.counterfactual_type == ct) for ct in CounterfactualType},
            "intervention_types": {it.value: sum(1 for s in self.scenario_library if s.intervention_type == it) for it in InterventionType},
            "confidence_levels": {cl.value: sum(1 for a in self.counterfactual_analyses if a.confidence_level == cl) for cl in ConfidenceLevel},
            "modality_types": {mt.value: sum(1 for a in self.counterfactual_analyses if a.modality == mt) for mt in ModalityType},
            "average_probability": sum(a.overall_probability for a in self.counterfactual_analyses) / max(len(self.counterfactual_analyses), 1),
            "average_plausibility": sum(s.plausibility for s in self.scenario_library) / max(len(self.scenario_library), 1),
            "consistency_rate": sum(1 for a in self.counterfactual_analyses if a.consistency_check) / max(len(self.counterfactual_analyses), 1),
            "plausibility_rate": sum(1 for a in self.counterfactual_analyses if a.plausibility_check) / max(len(self.counterfactual_analyses), 1)
        }