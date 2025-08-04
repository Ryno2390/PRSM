"""
Core data structures for rich reasoning context preservation in NWTN.

This module defines comprehensive data structures that capture and preserve
the sophisticated reasoning insights currently being lost between meta-reasoning
and final synthesis in NWTN's pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums and Constants
# ============================================================================

class EngineType(Enum):
    """Types of reasoning engines in NWTN"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    PROBABILISTIC = "probabilistic"


class PatternType(Enum):
    """Types of synthesis patterns between reasoning engines"""
    CONVERGENT = "convergent"          # Engines reach similar conclusions
    DIVERGENT = "divergent"            # Engines reach different conclusions
    COMPLEMENTARY = "complementary"    # Engines provide different aspects
    CONFLICTING = "conflicting"        # Engines contradict each other
    HIERARCHICAL = "hierarchical"      # One engine builds on another
    SYNERGISTIC = "synergistic"        # Combined result > sum of parts


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning insights"""
    LOW = "low"                    # 0.0 - 0.3
    MODERATE = "moderate"          # 0.3 - 0.6
    HIGH = "high"                  # 0.6 - 0.8
    VERY_HIGH = "very_high"        # 0.8 - 1.0


class BreakthroughPotential(Enum):
    """Potential for breakthrough insights"""
    INCREMENTAL = "incremental"
    SIGNIFICANT = "significant"
    MAJOR = "major"
    REVOLUTIONARY = "revolutionary"


# ============================================================================
# Supporting Data Structures
# ============================================================================

@dataclass
class Evidence:
    """Represents a piece of evidence supporting reasoning"""
    source: str                           # Source identifier (paper, principle, etc.)
    content: str                          # The actual evidence content
    strength: float                       # Evidence strength (0.0 - 1.0)
    relevance: float                      # Relevance to conclusion (0.0 - 1.0)
    evidence_type: str                    # empirical, theoretical, analogical, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """Represents a single step in a reasoning trace"""
    step_number: int
    operation: str                        # The reasoning operation performed
    input_state: str                      # State before this step
    output_state: str                     # State after this step
    justification: str                    # Why this step was taken
    confidence: float                     # Confidence in this step
    alternatives_considered: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EngineInsight:
    """Comprehensive insight from a single reasoning engine"""
    engine_type: EngineType
    primary_findings: List[str]           # Main insights from this engine
    supporting_evidence: List[Evidence]   # Evidence supporting the findings
    confidence_level: float               # Overall confidence (0.0 - 1.0)
    confidence_category: ConfidenceLevel
    reasoning_trace: List[ReasoningStep]  # Step-by-step reasoning process
    breakthrough_indicators: List[str]    # Indicators of potential breakthroughs
    processing_time: float                # Time spent on this reasoning
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisPattern:
    """Pattern of interaction between multiple reasoning engines"""
    pattern_type: PatternType
    participating_engines: List[EngineType]
    synthesis_description: str            # Description of the pattern
    strength: float                       # Strength of the pattern (0.0 - 1.0)
    evidence_support: List[Evidence]      # Evidence supporting this pattern
    emergent_properties: List[str]        # Properties that emerge from combination
    implications: List[str]               # What this pattern implies
    confidence: float                     # Confidence in pattern detection


@dataclass
class EngineInteraction:
    """Specific interaction between two reasoning engines"""
    engine_a: EngineType
    engine_b: EngineType
    interaction_type: str                 # support, conflict, complement, etc.
    interaction_strength: float           # Strength of interaction (0.0 - 1.0)
    description: str                      # Description of the interaction
    resolution: Optional[str] = None      # How conflicts were resolved
    synthesis_outcome: Optional[str] = None # Result of the interaction


@dataclass
class ReasoningConflict:
    """Conflict between reasoning engines"""
    conflicting_engines: List[EngineType]
    conflict_description: str
    conflict_severity: float              # Severity of conflict (0.0 - 1.0)
    potential_resolutions: List[str]
    evidence_for_each_side: Dict[str, List[Evidence]]
    resolution_status: str                # unresolved, partially_resolved, resolved
    resolution_method: Optional[str] = None


@dataclass
class ConvergencePoint:
    """Point where multiple reasoning engines converge"""
    converging_engines: List[EngineType]
    convergence_topic: str
    convergence_strength: float           # How strongly they converge (0.0 - 1.0)
    shared_conclusion: str
    supporting_evidence: List[Evidence]
    implications: List[str]


@dataclass
class EmergentInsight:
    """Insight that emerges from combination of reasoning engines"""
    insight_description: str
    contributing_engines: List[EngineType]
    emergence_mechanism: str              # How the insight emerged
    novelty_score: float                  # How novel is this insight (0.0 - 1.0)
    potential_impact: str                 # Potential impact of this insight
    confidence: float                     # Confidence in the insight
    supporting_evidence: List[Evidence]
    follow_up_questions: List[str] = field(default_factory=list)


@dataclass
class AnalogicalConnection:
    """Analogical connection discovered by reasoning"""
    source_domain: str
    target_domain: str
    connection_strength: float            # Strength of analogy (0.0 - 1.0)
    analogical_mapping: Dict[str, str]    # Element-to-element mappings
    insights_generated: List[str]         # Insights from this analogy
    confidence: float                     # Confidence in the analogy
    reasoning_trace: List[ReasoningStep]
    breakthrough_potential: float         # Potential for breakthrough (0.0 - 1.0)
    limitations: List[str] = field(default_factory=list)


@dataclass
class CrossDomainBridge:
    """Bridge concept that connects different domains"""
    bridge_concept: str
    connected_domains: List[str]
    bridge_strength: float                # Strength of the bridge (0.0 - 1.0)
    bridging_mechanism: str               # How the bridge works
    insights_enabled: List[str]           # Insights enabled by this bridge
    evidence: List[Evidence]


@dataclass
class MetaphoricalPattern:
    """Metaphorical pattern identified in reasoning"""
    metaphor_description: str
    literal_domain: str
    metaphorical_domain: str
    pattern_elements: Dict[str, str]      # Mapping between literal and metaphorical
    insights_generated: List[str]
    explanatory_power: float              # How well it explains (0.0 - 1.0)


@dataclass
class ConfidenceAnalysis:
    """Comprehensive analysis of confidence across reasoning"""
    overall_confidence: float             # Overall confidence (0.0 - 1.0)
    confidence_distribution: Dict[EngineType, float]  # Per-engine confidence
    confidence_factors: List[str]         # Factors affecting confidence
    uncertainty_sources: List[str]        # Sources of uncertainty
    confidence_boosters: List[str]        # What increases confidence
    confidence_detractors: List[str]      # What decreases confidence
    reliability_assessment: str           # Overall reliability assessment


@dataclass
class UncertaintyMapping:
    """Mapping of uncertainties in the reasoning process"""
    uncertainty_types: Dict[str, float]   # Types and levels of uncertainty
    epistemic_uncertainty: float          # Uncertainty due to lack of knowledge
    aleatoric_uncertainty: float          # Uncertainty due to inherent randomness
    model_uncertainty: float              # Uncertainty in reasoning models
    uncertainty_propagation: Dict[str, float]  # How uncertainty propagates
    mitigation_strategies: List[str]      # Ways to reduce uncertainty


@dataclass
class KnowledgeGap:
    """Identified gap in knowledge"""
    gap_description: str
    affected_reasoning: List[EngineType]
    impact_severity: float                # Impact on conclusions (0.0 - 1.0)
    potential_sources: List[str]          # Where to find missing knowledge
    research_questions: List[str]         # Questions to address the gap
    priority: str                         # low, medium, high, critical


@dataclass
class BreakthroughAnalysis:
    """Analysis of breakthrough potential"""
    overall_breakthrough_score: float     # Overall breakthrough potential (0.0 - 1.0)
    breakthrough_category: BreakthroughPotential
    breakthrough_areas: List[str]         # Areas where breakthroughs are possible
    paradigm_shift_indicators: List[str]  # Signs of potential paradigm shifts
    innovation_opportunities: List[str]   # Specific innovation opportunities
    risk_assessment: str                  # Risk of pursuing breakthrough directions
    evidence_strength: float             # Strength of evidence for breakthroughs


@dataclass
class NoveltyAssessment:
    """Assessment of novelty in reasoning results"""
    novelty_score: float                  # Overall novelty (0.0 - 1.0)
    novel_connections: List[str]          # Novel connections discovered
    novel_applications: List[str]         # Novel applications identified
    paradigm_challenges: List[str]        # Challenges to existing paradigms
    originality_indicators: List[str]     # Indicators of original thinking


@dataclass
class WorldModelValidation:
    """Validation against world model knowledge"""
    validation_score: float               # Overall validation score (0.0 - 1.0)
    validated_principles: List[str]       # Principles that validate reasoning
    conflicting_principles: List[str]     # Principles that conflict
    novel_principle_candidates: List[str] # Potential new principles
    consistency_check: str                # Overall consistency assessment
    domain_coverage: Dict[str, float]     # Coverage across knowledge domains


@dataclass
class PrincipleConsistency:
    """Consistency with established principles"""
    consistent_principles: List[str]
    inconsistent_principles: List[str]
    partially_consistent: Dict[str, float]  # Principle -> consistency score
    principle_conflicts: List[str]
    resolution_suggestions: List[str]


@dataclass
class DomainExpertise:
    """Domain-specific expertise integration"""
    relevant_domains: List[str]
    domain_confidence: Dict[str, float]   # Confidence per domain
    cross_domain_insights: List[str]
    domain_limitations: List[str]
    expertise_gaps: List[str]


@dataclass
class ReasoningQuality:
    """Quality metrics for reasoning process"""
    logical_consistency: float            # Logical consistency score
    evidence_integration: float           # How well evidence is integrated
    coherence_score: float               # Internal coherence
    completeness_score: float            # How complete the reasoning is
    depth_score: float                    # Depth of analysis
    breadth_score: float                  # Breadth of consideration
    originality_score: float             # Originality of insights


@dataclass
class CoherenceMetrics:
    """Metrics for reasoning coherence"""
    internal_coherence: float             # Internal logical coherence
    external_coherence: float             # Coherence with existing knowledge
    narrative_coherence: float            # Coherence of the reasoning story
    temporal_coherence: float             # Consistency across time
    cross_engine_coherence: float         # Coherence between engines


@dataclass
class CompletenessAssessment:
    """Assessment of reasoning completeness"""
    coverage_score: float                 # How comprehensive the analysis is
    missing_perspectives: List[str]       # Important perspectives not considered
    unexplored_angles: List[str]          # Angles that could be explored
    depth_gaps: List[str]                 # Areas needing deeper analysis
    breadth_gaps: List[str]               # Areas needing broader analysis


# ============================================================================
# Main Rich Reasoning Context Class
# ============================================================================

@dataclass
class RichReasoningContext:
    """
    Comprehensive reasoning context that captures all the rich insights
    from NWTN's multi-engine reasoning process.
    
    This class serves as the central data structure for preserving the
    sophisticated reasoning insights that are currently being lost in
    the transition from meta-reasoning to final synthesis.
    """
    
    # Core reasoning results
    original_query: str
    processing_timestamp: datetime = field(default_factory=datetime.now)
    
    # Individual Engine Insights
    engine_insights: Dict[EngineType, EngineInsight] = field(default_factory=dict)
    engine_confidence_levels: Dict[EngineType, float] = field(default_factory=dict)
    engine_processing_time: Dict[EngineType, float] = field(default_factory=dict)
    
    # Cross-Engine Analysis
    synthesis_patterns: List[SynthesisPattern] = field(default_factory=list)
    cross_engine_interactions: List[EngineInteraction] = field(default_factory=list)
    reasoning_conflicts: List[ReasoningConflict] = field(default_factory=list)
    convergence_points: List[ConvergencePoint] = field(default_factory=list)
    
    # Breakthrough Analysis
    emergent_insights: List[EmergentInsight] = field(default_factory=list)
    breakthrough_analysis: Optional[BreakthroughAnalysis] = None
    novelty_assessment: Optional[NoveltyAssessment] = None
    
    # Analogical Connections
    analogical_connections: List[AnalogicalConnection] = field(default_factory=list)
    cross_domain_bridges: List[CrossDomainBridge] = field(default_factory=list)
    metaphorical_patterns: List[MetaphoricalPattern] = field(default_factory=list)
    
    # Uncertainty & Confidence
    confidence_analysis: Optional[ConfidenceAnalysis] = None
    uncertainty_mapping: Optional[UncertaintyMapping] = None
    knowledge_gaps: List[KnowledgeGap] = field(default_factory=list)
    
    # World Model Integration
    world_model_validation: Optional[WorldModelValidation] = None
    principle_consistency: Optional[PrincipleConsistency] = None
    domain_expertise: Optional[DomainExpertise] = None
    
    # Quality Metrics
    reasoning_quality: Optional[ReasoningQuality] = None
    coherence_metrics: Optional[CoherenceMetrics] = None
    completeness_assessment: Optional[CompletenessAssessment] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    def get_top_insights(self, limit: int = 5) -> List[str]:
        """Get the top insights from all reasoning engines"""
        all_insights = []
        for insight in self.engine_insights.values():
            all_insights.extend(insight.primary_findings)
        
        # Sort by some relevance/importance metric
        # For now, just return first N
        return all_insights[:limit]
    
    def get_strongest_analogies(self, limit: int = 3) -> List[AnalogicalConnection]:
        """Get the strongest analogical connections"""
        sorted_analogies = sorted(
            self.analogical_connections,
            key=lambda x: x.connection_strength,
            reverse=True
        )
        return sorted_analogies[:limit]
    
    def get_highest_confidence_engines(self) -> List[EngineType]:
        """Get engines ordered by confidence level"""
        return sorted(
            self.engine_confidence_levels.keys(),
            key=lambda x: self.engine_confidence_levels[x],
            reverse=True
        )
    
    def has_breakthrough_potential(self) -> bool:
        """Check if the reasoning has breakthrough potential"""
        if self.breakthrough_analysis:
            return self.breakthrough_analysis.overall_breakthrough_score > 0.6
        return False
    
    def has_major_conflicts(self) -> bool:
        """Check if there are major conflicts between reasoning engines"""
        return any(conflict.conflict_severity > 0.7 for conflict in self.reasoning_conflicts)
    
    def get_confidence_summary(self) -> str:
        """Get a summary of confidence levels"""
        if self.confidence_analysis:
            return f"Overall confidence: {self.confidence_analysis.overall_confidence:.2f}"
        return "Confidence analysis not available"
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of processing metrics"""
        total_time = sum(self.engine_processing_time.values())
        return {
            'total_processing_time': total_time,
            'engines_used': len(self.engine_insights),
            'insights_generated': len(self.emergent_insights),
            'analogies_found': len(self.analogical_connections),
            'conflicts_detected': len(self.reasoning_conflicts),
            'convergence_points': len(self.convergence_points)
        }


# ============================================================================
# Related Data Structures for Integration
# ============================================================================

@dataclass
class ContextValidationResult:
    """Result of validating rich reasoning context"""
    overall_score: float                  # Overall validation score (0.0 - 1.0)
    validation_checks: Dict[str, float]   # Individual validation scores
    recommendations: List[str]            # Recommendations for improvement
    missing_components: List[str]         # Components that are missing
    quality_issues: List[str]             # Quality issues identified
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EnhancedReasoningResult:
    """Enhanced reasoning result with rich context"""
    standard_result: Any                  # Original meta-reasoning result
    rich_context: RichReasoningContext    # Rich reasoning context
    context_validation: ContextValidationResult  # Validation results
    processing_metadata: Dict[str, Any] = field(default_factory=dict)