#!/usr/bin/env python3
"""
NWTN Analogical Breakthrough Engine
Systematic discovery of breakthrough insights through cross-domain analogical reasoning

This module implements NWTN's ability to find "eureka moments" by systematically
mapping successful patterns, principles, and solutions from one domain to another.
This is how NWTN can generate genuinely novel insights rather than just
retrieving existing knowledge.

Key Concepts:
1. Pattern Mining: Extract successful patterns from well-understood domains
2. Analogical Mapping: Systematically map patterns to target domains
3. Constraint Validation: Ensure analogies are physically/logically valid
4. Breakthrough Detection: Identify when analogies lead to novel insights
5. Knowledge Propagation: Share breakthrough discoveries across PRSM network

Historical Breakthrough Examples:
- Kekulé's benzene ring (snake → circular structure)
- Darwin's evolution (population theory → natural selection)
- Bohr's atomic model (planetary orbits → electron shells)
- Wave-particle duality (waves + particles → quantum mechanics)

Usage:
    from prsm.nwtn.analogical_breakthrough_engine import AnalogicalBreakthroughEngine
    
    engine = AnalogicalBreakthroughEngine()
    breakthroughs = await engine.discover_cross_domain_insights(source_domain, target_domain)
"""

import asyncio
import json
import math
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime, timezone
from collections import defaultdict

import structlog
from pydantic import BaseModel, Field

from prsm.nwtn.hybrid_architecture import SOC, SOCType, ConfidenceLevel, HybridNWTNEngine
from prsm.nwtn.world_model_engine import WorldModelEngine
from prsm.agents.executors.model_executor import ModelExecutor

logger = structlog.get_logger(__name__)


class AnalogicalRelationType(str, Enum):
    """Types of analogical relationships"""
    STRUCTURAL = "structural"           # Similar structure/organization
    FUNCTIONAL = "functional"           # Similar function/behavior
    CAUSAL = "causal"                  # Similar cause-effect patterns
    MATHEMATICAL = "mathematical"       # Similar mathematical relationships
    PROCESS = "process"                # Similar processes/mechanisms
    CONSTRAINT = "constraint"          # Similar constraints/limitations


class BreakthroughType(str, Enum):
    """Types of breakthrough insights"""
    MECHANISM_TRANSFER = "mechanism_transfer"        # Transfer of how something works
    PRINCIPLE_APPLICATION = "principle_application"  # Apply principle to new domain
    STRUCTURAL_MAPPING = "structural_mapping"        # Map structure to new context
    CONSTRAINT_SOLVING = "constraint_solving"        # Solve constraints using analogy
    PATTERN_COMPLETION = "pattern_completion"        # Complete patterns in new domain
    SYNTHESIS = "synthesis"                          # Combine multiple domain insights


@dataclass
class AnalogicalPattern:
    """A pattern that can be mapped across domains"""
    
    id: str
    name: str
    source_domain: str
    
    # Pattern structure
    structural_components: List[str]
    functional_relationships: Dict[str, str]
    causal_chains: List[Tuple[str, str]]
    mathematical_relationships: List[str]
    
    # Pattern properties
    success_rate: float  # How often this pattern works in source domain
    generalization_level: str  # "specific", "intermediate", "general"
    abstraction_level: str  # "concrete", "abstract", "mathematical"
    
    # Constraints and limitations
    domain_constraints: List[str]
    validity_conditions: List[str]
    known_failures: List[str]
    
    # Metadata
    confidence: float = 0.8
    usage_count: int = 0
    last_validated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AnalogicalMapping:
    """Mapping of a pattern from source to target domain"""
    
    id: str
    source_pattern: AnalogicalPattern
    target_domain: str
    
    # Mapping details
    component_mappings: Dict[str, str]  # source_component -> target_component
    relationship_mappings: Dict[str, str]  # source_relation -> target_relation
    constraint_mappings: Dict[str, str]  # source_constraint -> target_constraint
    
    # Validation results
    structural_validity: float  # 0-1, how well structure maps
    functional_validity: float  # 0-1, how well function maps
    causal_validity: float     # 0-1, how well causation maps
    overall_validity: float    # 0-1, overall mapping quality
    
    # Predictions and insights
    predicted_behaviors: List[str]
    novel_insights: List[str]
    testable_hypotheses: List[str]
    
    # Breakthrough potential
    breakthrough_type: BreakthroughType
    novelty_score: float  # 0-1, how novel is this insight
    impact_potential: float  # 0-1, potential impact if correct
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BreakthroughInsight(BaseModel):
    """A potential breakthrough insight discovered through analogical reasoning"""
    
    id: UUID = Field(default_factory=uuid4)
    title: str
    description: str
    
    # Source information
    source_domain: str
    source_pattern: str
    target_domain: str
    
    # Insight details
    breakthrough_type: BreakthroughType
    analogical_mapping: Dict[str, Any]
    
    # Validation and testing
    confidence_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    testable_predictions: List[str] = Field(default_factory=list)
    validation_experiments: List[str] = Field(default_factory=list)
    
    # Impact assessment
    potential_applications: List[str] = Field(default_factory=list)
    related_unsolved_problems: List[str] = Field(default_factory=list)
    
    # Metadata
    discovered_by: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnalogicalBreakthroughEngine:
    """
    Engine for discovering breakthrough insights through systematic analogical reasoning
    
    This system enables NWTN to have genuine "eureka moments" by systematically
    mapping successful patterns from well-understood domains to less-understood ones.
    """
    
    def __init__(self):
        self.model_executor = ModelExecutor(agent_id="analogical_breakthrough_engine")
        self.world_model = WorldModelEngine()
        
        # Pattern databases
        self.domain_patterns: Dict[str, List[AnalogicalPattern]] = defaultdict(list)
        self.successful_mappings: List[AnalogicalMapping] = []
        self.breakthrough_insights: List[BreakthroughInsight] = []
        
        # Domain knowledge bases
        self.domain_knowledge = {
            "physics": self._load_physics_patterns(),
            "chemistry": self._load_chemistry_patterns(),
            "biology": self._load_biology_patterns(),
            "mathematics": self._load_mathematics_patterns(),
            "engineering": self._load_engineering_patterns(),
            "computer_science": self._load_cs_patterns()
        }
        
        # Breakthrough thresholds
        self.novelty_threshold = 0.7
        self.confidence_threshold = 0.6
        self.impact_threshold = 0.5
        
        logger.info("Initialized Analogical Breakthrough Engine")
    
    async def discover_cross_domain_insights(
        self, 
        source_domain: str, 
        target_domain: str,
        focus_area: str = None
    ) -> List[BreakthroughInsight]:
        """
        Systematically discover breakthrough insights by mapping patterns 
        from source domain to target domain
        """
        
        logger.info(
            "Starting cross-domain insight discovery",
            source_domain=source_domain,
            target_domain=target_domain,
            focus_area=focus_area
        )
        
        # Step 1: Extract successful patterns from source domain
        source_patterns = await self._extract_domain_patterns(source_domain)
        
        # Step 2: Analyze target domain for analogical opportunities
        target_gaps = await self._identify_target_domain_gaps(target_domain, focus_area)
        
        # Step 3: Generate analogical mappings
        potential_mappings = await self._generate_analogical_mappings(
            source_patterns, target_domain, target_gaps
        )
        
        # Step 4: Validate mappings against world model
        validated_mappings = await self._validate_analogical_mappings(potential_mappings)
        
        # Step 5: Identify breakthrough insights
        breakthrough_insights = await self._identify_breakthrough_insights(validated_mappings)
        
        # Step 6: Rank and prioritize insights
        ranked_insights = await self._rank_insights_by_potential(breakthrough_insights)
        
        # Step 7: Generate testable hypotheses
        testable_insights = await self._generate_testable_hypotheses(ranked_insights)
        
        logger.info(
            "Completed cross-domain insight discovery",
            total_patterns=len(source_patterns),
            potential_mappings=len(potential_mappings),
            validated_mappings=len(validated_mappings),
            breakthrough_insights=len(breakthrough_insights)
        )
        
        return testable_insights
    
    async def _extract_domain_patterns(self, domain: str) -> List[AnalogicalPattern]:
        """Extract successful patterns from a source domain"""
        
        # Start with pre-loaded patterns
        patterns = self.domain_knowledge.get(domain, [])
        
        # Dynamically discover additional patterns
        pattern_discovery_prompt = f"""
        Analyze the {domain} domain and identify the most successful, transferable patterns:
        
        Focus on:
        1. Structural patterns (how things are organized)
        2. Functional patterns (how things work)
        3. Causal patterns (what causes what)
        4. Mathematical patterns (quantitative relationships)
        5. Process patterns (how things change over time)
        
        For each pattern, identify:
        - Core components and their relationships
        - Success conditions and constraints
        - Known applications and limitations
        - Abstraction level (concrete to mathematical)
        
        Prioritize patterns that have been successful across multiple contexts within {domain}.
        """
        
        try:
            analysis = await self.model_executor.execute_request(
                prompt=pattern_discovery_prompt,
                model_name="gpt-4",
                temperature=0.3
            )
            
            # Parse discovered patterns (simplified for demonstration)
            discovered_patterns = await self._parse_discovered_patterns(analysis, domain)
            patterns.extend(discovered_patterns)
            
        except Exception as e:
            logger.error("Error extracting domain patterns", error=str(e))
        
        return patterns
    
    async def _identify_target_domain_gaps(self, target_domain: str, focus_area: str = None) -> List[str]:
        """Identify gaps, unsolved problems, or areas needing innovation in target domain"""
        
        gap_analysis_prompt = f"""
        Analyze the {target_domain} domain to identify:
        
        1. Unsolved problems or mysteries
        2. Areas lacking good explanatory mechanisms
        3. Phenomena that need better models
        4. Inefficient processes that could be improved
        5. Missing connections between known facts
        
        {f"Focus specifically on: {focus_area}" if focus_area else ""}
        
        Prioritize gaps where analogical reasoning from other domains might help.
        """
        
        try:
            analysis = await self.model_executor.execute_request(
                prompt=gap_analysis_prompt,
                model_name="gpt-4",
                temperature=0.4
            )
            
            # Extract gaps (simplified parsing)
            gaps = await self._parse_domain_gaps(analysis, target_domain)
            return gaps
            
        except Exception as e:
            logger.error("Error identifying target domain gaps", error=str(e))
            return []
    
    async def _generate_analogical_mappings(
        self, 
        source_patterns: List[AnalogicalPattern], 
        target_domain: str, 
        target_gaps: List[str]
    ) -> List[AnalogicalMapping]:
        """Generate potential analogical mappings from source patterns to target domain"""
        
        mappings = []
        
        for pattern in source_patterns:
            for gap in target_gaps:
                mapping_prompt = f"""
                Consider mapping this pattern from {pattern.source_domain} to {target_domain}:
                
                Source Pattern: {pattern.name}
                - Structure: {pattern.structural_components}
                - Function: {pattern.functional_relationships}
                - Causation: {pattern.causal_chains}
                - Math: {pattern.mathematical_relationships}
                
                Target Gap: {gap}
                
                Create an analogical mapping:
                1. Map each source component to target domain equivalent
                2. Map relationships and causation patterns
                3. Identify what this mapping would predict
                4. Assess mapping validity and novelty
                
                Be specific about component mappings and predicted behaviors.
                """
                
                try:
                    mapping_analysis = await self.model_executor.execute_request(
                        prompt=mapping_prompt,
                        model_name="gpt-4",
                        temperature=0.5
                    )
                    
                    # Parse mapping (simplified)
                    mapping = await self._parse_analogical_mapping(
                        mapping_analysis, pattern, target_domain, gap
                    )
                    
                    if mapping:
                        mappings.append(mapping)
                        
                except Exception as e:
                    logger.error("Error generating analogical mapping", error=str(e))
        
        return mappings
    
    async def _validate_analogical_mappings(self, mappings: List[AnalogicalMapping]) -> List[AnalogicalMapping]:
        """Validate analogical mappings against world model and physical constraints"""
        
        validated_mappings = []
        
        for mapping in mappings:
            # Validate structural coherence
            structural_validity = await self._validate_structural_mapping(mapping)
            
            # Validate functional coherence
            functional_validity = await self._validate_functional_mapping(mapping)
            
            # Validate causal consistency
            causal_validity = await self._validate_causal_mapping(mapping)
            
            # Update mapping with validation results
            mapping.structural_validity = structural_validity
            mapping.functional_validity = functional_validity
            mapping.causal_validity = causal_validity
            mapping.overall_validity = (structural_validity + functional_validity + causal_validity) / 3
            
            # Only keep mappings that pass minimum validity threshold
            if mapping.overall_validity >= self.confidence_threshold:
                validated_mappings.append(mapping)
                
                logger.debug(
                    "Validated analogical mapping",
                    mapping_id=mapping.id,
                    validity=mapping.overall_validity
                )
            
        return validated_mappings
    
    async def _identify_breakthrough_insights(self, mappings: List[AnalogicalMapping]) -> List[BreakthroughInsight]:
        """Identify which mappings represent genuine breakthrough insights"""
        
        breakthrough_insights = []
        
        for mapping in mappings:
            # Assess novelty - how new is this insight?
            novelty_score = await self._assess_novelty(mapping)
            
            # Assess impact - how significant could this be?
            impact_score = await self._assess_impact_potential(mapping)
            
            # Check if this meets breakthrough thresholds
            if (novelty_score >= self.novelty_threshold and 
                impact_score >= self.impact_threshold and
                mapping.overall_validity >= self.confidence_threshold):
                
                insight = BreakthroughInsight(
                    title=f"Analogical Breakthrough: {mapping.source_pattern.name} → {mapping.target_domain}",
                    description=f"Applying {mapping.source_pattern.name} pattern to {mapping.target_domain}",
                    source_domain=mapping.source_pattern.source_domain,
                    source_pattern=mapping.source_pattern.name,
                    target_domain=mapping.target_domain,
                    breakthrough_type=mapping.breakthrough_type,
                    analogical_mapping=mapping.component_mappings,
                    confidence_score=mapping.overall_validity,
                    novelty_score=novelty_score,
                    testable_predictions=mapping.predicted_behaviors,
                    validation_experiments=mapping.testable_hypotheses,
                    discovered_by=f"analogical_breakthrough_engine"
                )
                
                breakthrough_insights.append(insight)
                
                logger.info(
                    "Breakthrough insight identified",
                    title=insight.title,
                    novelty=novelty_score,
                    impact=impact_score,
                    confidence=mapping.overall_validity
                )
        
        return breakthrough_insights
    
    async def _rank_insights_by_potential(self, insights: List[BreakthroughInsight]) -> List[BreakthroughInsight]:
        """Rank insights by their breakthrough potential"""
        
        def breakthrough_score(insight: BreakthroughInsight) -> float:
            return (
                insight.novelty_score * 0.4 +
                insight.confidence_score * 0.3 +
                len(insight.testable_predictions) * 0.1 +
                len(insight.potential_applications) * 0.2
            )
        
        return sorted(insights, key=breakthrough_score, reverse=True)
    
    async def _generate_testable_hypotheses(self, insights: List[BreakthroughInsight]) -> List[BreakthroughInsight]:
        """Generate concrete testable hypotheses for breakthrough insights"""
        
        for insight in insights:
            hypothesis_prompt = f"""
            Generate specific, testable hypotheses for this breakthrough insight:
            
            Insight: {insight.title}
            Description: {insight.description}
            Source Pattern: {insight.source_pattern}
            Target Domain: {insight.target_domain}
            
            Create hypotheses that:
            1. Make specific, measurable predictions
            2. Can be tested with available methods
            3. Would clearly validate or refute the analogical mapping
            4. Have clear success/failure criteria
            
            Focus on experiments that could be conducted to test this analogy.
            """
            
            try:
                hypothesis_analysis = await self.model_executor.execute_request(
                    prompt=hypothesis_prompt,
                    model_name="gpt-4",
                    temperature=0.4
                )
                
                # Parse and add testable hypotheses
                hypotheses = await self._parse_testable_hypotheses(hypothesis_analysis)
                insight.testable_predictions.extend(hypotheses)
                
            except Exception as e:
                logger.error("Error generating testable hypotheses", error=str(e))
        
        return insights
    
    # Pre-loaded pattern databases (simplified examples)
    def _load_physics_patterns(self) -> List[AnalogicalPattern]:
        """Load successful patterns from physics"""
        return [
            AnalogicalPattern(
                id="wave_pattern",
                name="Wave Interference",
                source_domain="physics",
                structural_components=["wave_source", "medium", "interference_pattern"],
                functional_relationships={"constructive": "amplification", "destructive": "cancellation"},
                causal_chains=[("wave_overlap", "interference"), ("phase_difference", "amplitude_change")],
                mathematical_relationships=["amplitude = A₁ + A₂", "phase_shift = 2π/λ"],
                success_rate=0.95,
                generalization_level="general",
                abstraction_level="mathematical"
            ),
            AnalogicalPattern(
                id="resonance_pattern",
                name="Resonance Amplification",
                source_domain="physics",
                structural_components=["driver", "resonator", "coupling", "amplification"],
                functional_relationships={"frequency_match": "energy_transfer", "coupling": "amplification"},
                causal_chains=[("frequency_match", "resonance"), ("resonance", "amplification")],
                mathematical_relationships=["Q = f₀/Δf", "amplitude ∝ 1/damping"],
                success_rate=0.9,
                generalization_level="general",
                abstraction_level="mathematical"
            )
        ]
    
    def _load_chemistry_patterns(self) -> List[AnalogicalPattern]:
        """Load successful patterns from chemistry"""
        return [
            AnalogicalPattern(
                id="catalyst_pattern",
                name="Catalytic Mechanism",
                source_domain="chemistry",
                structural_components=["catalyst", "reactants", "transition_state", "products"],
                functional_relationships={"catalyst": "lower_activation_energy", "pathway": "alternative_route"},
                causal_chains=[("catalyst_binding", "transition_state_stabilization"), ("lower_barrier", "faster_reaction")],
                mathematical_relationships=["rate = k[A][B]", "k = Ae^(-Ea/RT)"],
                success_rate=0.85,
                generalization_level="general",
                abstraction_level="concrete"
            )
        ]
    
    def _load_biology_patterns(self) -> List[AnalogicalPattern]:
        """Load successful patterns from biology"""
        return [
            AnalogicalPattern(
                id="feedback_regulation",
                name="Negative Feedback Control",
                source_domain="biology",
                structural_components=["sensor", "controller", "effector", "feedback_loop"],
                functional_relationships={"deviation": "correction", "stability": "homeostasis"},
                causal_chains=[("deviation_detection", "corrective_action"), ("corrective_action", "stability")],
                mathematical_relationships=["output = setpoint - error", "gain = output/input"],
                success_rate=0.9,
                generalization_level="general",
                abstraction_level="abstract"
            )
        ]
    
    def _load_mathematics_patterns(self) -> List[AnalogicalPattern]:
        """Load successful patterns from mathematics"""
        return [
            AnalogicalPattern(
                id="symmetry_pattern",
                name="Symmetry Conservation",
                source_domain="mathematics",
                structural_components=["symmetry_operation", "invariant", "transformation", "conservation"],
                functional_relationships={"symmetry": "conservation", "invariance": "constancy"},
                causal_chains=[("symmetry_operation", "invariance"), ("invariance", "conservation")],
                mathematical_relationships=["Noether's theorem", "∂L/∂q = 0 → conserved quantity"],
                success_rate=0.95,
                generalization_level="general",
                abstraction_level="mathematical"
            )
        ]
    
    def _load_engineering_patterns(self) -> List[AnalogicalPattern]:
        """Load successful patterns from engineering"""
        return [
            AnalogicalPattern(
                id="optimization_pattern",
                name="Constrained Optimization",
                source_domain="engineering",
                structural_components=["objective_function", "constraints", "variables", "optimum"],
                functional_relationships={"minimize": "cost", "maximize": "performance"},
                causal_chains=[("constraint_satisfaction", "feasible_solution"), ("optimization", "best_solution")],
                mathematical_relationships=["∇f = λ∇g", "KKT conditions"],
                success_rate=0.8,
                generalization_level="general",
                abstraction_level="mathematical"
            )
        ]
    
    def _load_cs_patterns(self) -> List[AnalogicalPattern]:
        """Load successful patterns from computer science"""
        return [
            AnalogicalPattern(
                id="recursive_pattern",
                name="Recursive Decomposition",
                source_domain="computer_science",
                structural_components=["base_case", "recursive_case", "decomposition", "combination"],
                functional_relationships={"decompose": "smaller_problems", "combine": "solution"},
                causal_chains=[("problem_decomposition", "subproblems"), ("subproblem_solution", "overall_solution")],
                mathematical_relationships=["T(n) = T(n/2) + O(n)", "divide and conquer"],
                success_rate=0.85,
                generalization_level="general",
                abstraction_level="abstract"
            )
        ]
    
    # Simplified helper methods (full implementation would be more sophisticated)
    async def _parse_discovered_patterns(self, analysis: str, domain: str) -> List[AnalogicalPattern]:
        """Parse discovered patterns from analysis text"""
        # Simplified implementation
        return []
    
    async def _parse_domain_gaps(self, analysis: str, domain: str) -> List[str]:
        """Parse domain gaps from analysis text"""
        # Simplified implementation
        return ["gap1", "gap2", "gap3"]
    
    async def _parse_analogical_mapping(self, analysis: str, pattern: AnalogicalPattern, target_domain: str, gap: str) -> Optional[AnalogicalMapping]:
        """Parse analogical mapping from analysis text"""
        # Simplified implementation
        return AnalogicalMapping(
            id=str(uuid4()),
            source_pattern=pattern,
            target_domain=target_domain,
            component_mappings={"source_comp": "target_comp"},
            relationship_mappings={"source_rel": "target_rel"},
            constraint_mappings={"source_constraint": "target_constraint"},
            structural_validity=0.8,
            functional_validity=0.7,
            causal_validity=0.75,
            overall_validity=0.75,
            predicted_behaviors=["prediction1", "prediction2"],
            novel_insights=["insight1", "insight2"],
            testable_hypotheses=["hypothesis1", "hypothesis2"],
            breakthrough_type=BreakthroughType.MECHANISM_TRANSFER,
            novelty_score=0.8,
            impact_potential=0.7
        )
    
    async def _validate_structural_mapping(self, mapping: AnalogicalMapping) -> float:
        """Validate structural coherence of mapping"""
        return 0.8  # Simplified
    
    async def _validate_functional_mapping(self, mapping: AnalogicalMapping) -> float:
        """Validate functional coherence of mapping"""
        return 0.7  # Simplified
    
    async def _validate_causal_mapping(self, mapping: AnalogicalMapping) -> float:
        """Validate causal consistency of mapping"""
        return 0.75  # Simplified
    
    async def _assess_novelty(self, mapping: AnalogicalMapping) -> float:
        """Assess how novel this mapping is"""
        return 0.8  # Simplified
    
    async def _assess_impact_potential(self, mapping: AnalogicalMapping) -> float:
        """Assess potential impact of this mapping"""
        return 0.7  # Simplified
    
    async def _parse_testable_hypotheses(self, analysis: str) -> List[str]:
        """Parse testable hypotheses from analysis text"""
        return ["hypothesis1", "hypothesis2"]  # Simplified
    
    async def systematic_breakthrough_search(
        self, 
        target_domain: str, 
        max_source_domains: int = 5
    ) -> List[BreakthroughInsight]:
        """
        Systematically search for breakthrough insights by testing analogies
        from multiple source domains to a target domain
        """
        
        all_insights = []
        
        # Get all available source domains
        source_domains = list(self.domain_knowledge.keys())
        
        # Remove target domain from sources
        if target_domain in source_domains:
            source_domains.remove(target_domain)
        
        # Limit to most promising source domains
        source_domains = source_domains[:max_source_domains]
        
        logger.info(
            "Starting systematic breakthrough search",
            target_domain=target_domain,
            source_domains=source_domains
        )
        
        # Test analogies from each source domain
        for source_domain in source_domains:
            insights = await self.discover_cross_domain_insights(source_domain, target_domain)
            all_insights.extend(insights)
        
        # Rank all insights together
        ranked_insights = await self._rank_insights_by_potential(all_insights)
        
        logger.info(
            "Completed systematic breakthrough search",
            total_insights=len(all_insights),
            top_insights=len(ranked_insights[:10])
        )
        
        return ranked_insights
    
    def get_breakthrough_stats(self) -> Dict[str, Any]:
        """Get statistics about breakthrough discoveries"""
        
        return {
            "total_insights": len(self.breakthrough_insights),
            "successful_mappings": len(self.successful_mappings),
            "domain_patterns": {domain: len(patterns) for domain, patterns in self.domain_patterns.items()},
            "breakthrough_types": {bt.value: sum(1 for insight in self.breakthrough_insights if insight.breakthrough_type == bt) for bt in BreakthroughType},
            "average_novelty": sum(insight.novelty_score for insight in self.breakthrough_insights) / max(1, len(self.breakthrough_insights)),
            "average_confidence": sum(insight.confidence_score for insight in self.breakthrough_insights) / max(1, len(self.breakthrough_insights))
        }