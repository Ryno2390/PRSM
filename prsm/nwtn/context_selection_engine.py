"""
Context Selection Engine for NWTN - Phase 3 Implementation.

This module implements intelligent context selection that adapts the amount
and type of reasoning context provided to synthesis based on user parameters,
ensuring optimal information density for different user preferences.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .reasoning_context_types import (
    RichReasoningContext, EngineType, EngineInsight, SynthesisPattern,
    AnalogicalConnection, ConfidenceAnalysis, BreakthroughAnalysis,
    EmergentInsight, ReasoningConflict, ConvergencePoint
)
from .contextual_synthesizer import UserParameters, ResponseTone


logger = logging.getLogger(__name__)


# ============================================================================
# Context Selection Data Structures
# ============================================================================

@dataclass
class ContextSubset:
    """Subset of reasoning context selected for synthesis"""
    
    # Core engine insights (always included)
    selected_engine_insights: Dict[EngineType, EngineInsight] = field(default_factory=dict)
    
    # Synthesis elements (filtered based on parameters)
    synthesis_patterns: List[SynthesisPattern] = field(default_factory=list)
    reasoning_conflicts: List[ReasoningConflict] = field(default_factory=list)
    convergence_points: List[ConvergencePoint] = field(default_factory=list)
    
    # Breakthrough and novelty (parameter-dependent)
    emergent_insights: List[EmergentInsight] = field(default_factory=list)
    breakthrough_highlights: List[str] = field(default_factory=list)
    novelty_indicators: List[str] = field(default_factory=list)
    
    # Analogical connections (creativity-dependent)
    analogical_connections: List[AnalogicalConnection] = field(default_factory=list)
    cross_domain_bridges: List[Dict[str, Any]] = field(default_factory=list)
    
    # Confidence and uncertainty (transparency-dependent)
    confidence_summary: Optional[str] = None
    detailed_confidence: Optional[ConfidenceAnalysis] = None
    uncertainty_exploration: Optional[Dict[str, Any]] = None
    knowledge_gaps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Evidence and research (focus-dependent)
    evidence_summary: List[str] = field(default_factory=list)
    research_citations: List[Dict[str, Any]] = field(default_factory=list)
    methodological_details: List[str] = field(default_factory=list)
    
    # Quality and completeness metrics
    context_completeness_score: float = 0.0
    information_density: float = 0.0
    selection_rationale: List[str] = field(default_factory=list)


@dataclass
class SelectionCriteria:
    """Criteria for context selection based on user parameters"""
    
    # Verbosity-based criteria
    max_engine_insights: int = 7
    max_synthesis_patterns: int = 3
    max_analogical_connections: int = 5
    max_emergent_insights: int = 3
    
    # Depth-based criteria
    include_reasoning_traces: bool = False
    include_quality_metrics: bool = False
    include_methodology_details: bool = False
    
    # Focus-based criteria
    emphasize_breakthrough: bool = False
    emphasize_evidence: bool = False
    emphasize_methodology: bool = False
    emphasize_implications: bool = False
    
    # Mode-based criteria
    creativity_threshold: float = 0.5
    confidence_threshold: float = 0.6
    novelty_threshold: float = 0.5
    
    # Uncertainty handling criteria
    include_uncertainty_details: bool = True
    include_knowledge_gaps: bool = True
    uncertainty_emphasis_level: float = 0.5


class ContextSelectionEngine:
    """
    Intelligently selects appropriate context subsets based on user parameters
    to optimize information density and relevance for synthesis.
    """
    
    def __init__(self):
        """Initialize the context selection engine"""
        
        # Selection strategies for different parameter combinations
        self.selection_strategies = {
            'verbosity_based': self._select_by_verbosity,
            'depth_based': self._select_by_depth,
            'focus_based': self._select_by_focus,
            'mode_based': self._select_by_reasoning_mode,
            'integrated': self._integrated_selection
        }
        
        # Parameter-to-criteria mappings
        self.verbosity_criteria = {
            'BRIEF': SelectionCriteria(
                max_engine_insights=3,
                max_synthesis_patterns=1,
                max_analogical_connections=2,
                max_emergent_insights=1,
                include_reasoning_traces=False,
                include_quality_metrics=False
            ),
            'MODERATE': SelectionCriteria(
                max_engine_insights=5,
                max_synthesis_patterns=2,
                max_analogical_connections=3,
                max_emergent_insights=2,
                include_reasoning_traces=False,
                include_quality_metrics=True
            ),
            'COMPREHENSIVE': SelectionCriteria(
                max_engine_insights=7,
                max_synthesis_patterns=3,
                max_analogical_connections=5,
                max_emergent_insights=3,
                include_reasoning_traces=True,
                include_quality_metrics=True,
                include_methodology_details=True
            ),
            'EXHAUSTIVE': SelectionCriteria(
                max_engine_insights=7,
                max_synthesis_patterns=5,
                max_analogical_connections=8,
                max_emergent_insights=5,
                include_reasoning_traces=True,
                include_quality_metrics=True,
                include_methodology_details=True
            )
        }
        
        logger.info("ContextSelectionEngine initialized with selection strategies")
    
    async def select_context_for_parameters(self,
                                           rich_context: RichReasoningContext,
                                           user_parameters: UserParameters) -> ContextSubset:
        """
        Select appropriate context subset based on user parameters.
        
        Args:
            rich_context: Complete rich reasoning context
            user_parameters: User customization parameters
            
        Returns:
            ContextSubset optimized for user parameters
        """
        
        logger.info(f"Selecting context for parameters: verbosity={user_parameters.verbosity}, "
                   f"mode={user_parameters.reasoning_mode}, depth={user_parameters.depth}")
        
        try:
            # Generate selection criteria from user parameters
            selection_criteria = await self._generate_selection_criteria(user_parameters)
            
            # Apply integrated selection strategy
            context_subset = await self._integrated_selection(
                rich_context, selection_criteria, user_parameters
            )
            
            # Calculate completeness and density metrics
            context_subset.context_completeness_score = await self._calculate_completeness_score(
                context_subset, rich_context
            )
            context_subset.information_density = await self._calculate_information_density(
                context_subset
            )
            
            # Generate selection rationale
            context_subset.selection_rationale = await self._generate_selection_rationale(
                selection_criteria, user_parameters
            )
            
            logger.info(f"Context selection completed: completeness={context_subset.context_completeness_score:.2f}, "
                       f"density={context_subset.information_density:.2f}")
            
            return context_subset
            
        except Exception as e:
            logger.error(f"Error in context selection: {e}")
            # Return minimal context subset as fallback
            return await self._create_minimal_context_subset(rich_context)
    
    # ========================================================================
    # Selection Criteria Generation
    # ========================================================================
    
    async def _generate_selection_criteria(self, user_parameters: UserParameters) -> SelectionCriteria:
        """Generate selection criteria from user parameters"""
        
        # Start with verbosity-based criteria
        base_criteria = self.verbosity_criteria.get(
            user_parameters.verbosity, 
            self.verbosity_criteria['MODERATE']
        )
        
        # Create copy for modification
        criteria = SelectionCriteria(
            max_engine_insights=base_criteria.max_engine_insights,
            max_synthesis_patterns=base_criteria.max_synthesis_patterns,
            max_analogical_connections=base_criteria.max_analogical_connections,
            max_emergent_insights=base_criteria.max_emergent_insights,
            include_reasoning_traces=base_criteria.include_reasoning_traces,
            include_quality_metrics=base_criteria.include_quality_metrics,
            include_methodology_details=base_criteria.include_methodology_details
        )
        
        # Modify based on reasoning mode
        if user_parameters.reasoning_mode == 'REVOLUTIONARY':
            criteria.emphasize_breakthrough = True
            criteria.creativity_threshold = 0.3
            criteria.novelty_threshold = 0.3
            criteria.max_emergent_insights = min(criteria.max_emergent_insights + 2, 5)
            criteria.max_analogical_connections = min(criteria.max_analogical_connections + 3, 8)
            
        elif user_parameters.reasoning_mode == 'CREATIVE':
            criteria.creativity_threshold = 0.4
            criteria.novelty_threshold = 0.4
            criteria.max_analogical_connections = min(criteria.max_analogical_connections + 2, 6)
            
        elif user_parameters.reasoning_mode == 'CONSERVATIVE':
            criteria.emphasize_evidence = True
            criteria.confidence_threshold = 0.7
            criteria.creativity_threshold = 0.7
            criteria.novelty_threshold = 0.7
            criteria.max_analogical_connections = max(criteria.max_analogical_connections - 1, 1)
        
        # Modify based on depth
        if user_parameters.depth == 'EXHAUSTIVE':
            criteria.include_reasoning_traces = True
            criteria.include_quality_metrics = True
            criteria.include_methodology_details = True
            
        elif user_parameters.depth == 'DEEP':
            criteria.include_reasoning_traces = True
            criteria.include_quality_metrics = True
            
        elif user_parameters.depth == 'SURFACE':
            criteria.include_reasoning_traces = False
            criteria.include_quality_metrics = False
            criteria.include_methodology_details = False
        
        # Modify based on synthesis focus
        if user_parameters.synthesis_focus == 'EVIDENCE':
            criteria.emphasize_evidence = True
            criteria.confidence_threshold = 0.7
            
        elif user_parameters.synthesis_focus == 'METHODOLOGY':
            criteria.emphasize_methodology = True
            criteria.include_methodology_details = True
            
        elif user_parameters.synthesis_focus == 'IMPLICATIONS':
            criteria.emphasize_implications = True
            criteria.max_emergent_insights = min(criteria.max_emergent_insights + 1, 5)
        
        # Modify based on uncertainty handling
        if user_parameters.uncertainty_handling == 'HIDE':
            criteria.include_uncertainty_details = False
            criteria.include_knowledge_gaps = False
            criteria.uncertainty_emphasis_level = 0.0
            
        elif user_parameters.uncertainty_handling == 'EMPHASIZE':
            criteria.include_uncertainty_details = True
            criteria.include_knowledge_gaps = True
            criteria.uncertainty_emphasis_level = 1.0
            
        elif user_parameters.uncertainty_handling == 'EXPLORE':
            criteria.include_uncertainty_details = True
            criteria.include_knowledge_gaps = True
            criteria.uncertainty_emphasis_level = 0.8
        
        return criteria
    
    # ========================================================================
    # Context Selection Strategies
    # ========================================================================
    
    async def _integrated_selection(self,
                                  rich_context: RichReasoningContext,
                                  criteria: SelectionCriteria,
                                  user_parameters: UserParameters) -> ContextSubset:
        """Integrated context selection using all criteria"""
        
        context_subset = ContextSubset()
        
        # 1. Select engine insights
        context_subset.selected_engine_insights = await self._select_engine_insights(
            rich_context, criteria
        )
        
        # 2. Select synthesis patterns
        context_subset.synthesis_patterns = await self._select_synthesis_patterns(
            rich_context, criteria
        )
        
        # 3. Select analogical connections
        context_subset.analogical_connections = await self._select_analogical_connections(
            rich_context, criteria
        )
        
        # 4. Select breakthrough elements
        if criteria.emphasize_breakthrough or user_parameters.reasoning_mode in ['CREATIVE', 'REVOLUTIONARY']:
            context_subset.emergent_insights = await self._select_emergent_insights(
                rich_context, criteria
            )
            context_subset.breakthrough_highlights = await self._select_breakthrough_highlights(
                rich_context, criteria
            )
        
        # 5. Select confidence and uncertainty elements
        if criteria.include_uncertainty_details:
            context_subset.detailed_confidence = rich_context.confidence_analysis
            context_subset.uncertainty_exploration = await self._select_uncertainty_exploration(
                rich_context, criteria
            )
        else:
            context_subset.confidence_summary = await self._generate_confidence_summary(
                rich_context
            )
        
        # 6. Select knowledge gaps if requested
        if criteria.include_knowledge_gaps:
            context_subset.knowledge_gaps = await self._select_knowledge_gaps(
                rich_context, criteria
            )
        
        # 7. Select evidence and research elements
        if criteria.emphasize_evidence:
            context_subset.evidence_summary = await self._select_evidence_summary(
                rich_context, criteria
            )
            context_subset.research_citations = await self._select_research_citations(
                rich_context, criteria
            )
        
        # 8. Select methodological details if requested
        if criteria.include_methodology_details:
            context_subset.methodological_details = await self._select_methodological_details(
                rich_context, criteria
            )
        
        # 9. Select reasoning conflicts and convergence
        context_subset.reasoning_conflicts = await self._select_reasoning_conflicts(
            rich_context, criteria
        )
        context_subset.convergence_points = await self._select_convergence_points(
            rich_context, criteria
        )
        
        return context_subset
    
    async def _select_engine_insights(self,
                                    rich_context: RichReasoningContext,
                                    criteria: SelectionCriteria) -> Dict[EngineType, EngineInsight]:
        """Select engine insights based on criteria"""
        
        # Rank engines by relevance and quality
        engine_scores = {}
        
        for engine_type, insight in rich_context.engine_insights.items():
            score = 0.0
            
            # Base score from confidence
            score += insight.confidence_level * 0.4
            
            # Score from number of findings
            score += min(len(insight.primary_findings) / 5.0, 1.0) * 0.3
            
            # Score from breakthrough potential
            if criteria.emphasize_breakthrough and insight.breakthrough_indicators:
                score += len(insight.breakthrough_indicators) / 5.0 * 0.3
            
            # Score from evidence strength
            if criteria.emphasize_evidence and insight.supporting_evidence:
                avg_evidence_strength = sum(e.strength for e in insight.supporting_evidence) / len(insight.supporting_evidence)
                score += avg_evidence_strength * 0.2
            
            engine_scores[engine_type] = score
        
        # Select top engines up to max limit
        sorted_engines = sorted(engine_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_insights = {}
        for engine_type, score in sorted_engines[:criteria.max_engine_insights]:
            insight = rich_context.engine_insights[engine_type]
            
            # Filter insight content based on criteria
            filtered_insight = await self._filter_engine_insight(insight, criteria)
            selected_insights[engine_type] = filtered_insight
        
        return selected_insights
    
    async def _filter_engine_insight(self,
                                   insight: EngineInsight,
                                   criteria: SelectionCriteria) -> EngineInsight:
        """Filter engine insight content based on criteria"""
        
        filtered_insight = EngineInsight(
            engine_type=insight.engine_type,
            primary_findings=insight.primary_findings[:3],  # Limit findings
            supporting_evidence=insight.supporting_evidence[:2] if criteria.emphasize_evidence else [],
            confidence_level=insight.confidence_level,
            confidence_category=insight.confidence_category,
            reasoning_trace=insight.reasoning_trace if criteria.include_reasoning_traces else [],
            breakthrough_indicators=insight.breakthrough_indicators if criteria.emphasize_breakthrough else [],
            processing_time=insight.processing_time,
            quality_metrics=insight.quality_metrics if criteria.include_quality_metrics else {}
        )
        
        return filtered_insight
    
    async def _select_synthesis_patterns(self,
                                       rich_context: RichReasoningContext,
                                       criteria: SelectionCriteria) -> List[SynthesisPattern]:
        """Select synthesis patterns based on criteria"""
        
        if not rich_context.synthesis_patterns:
            return []
        
        # Score patterns by relevance
        pattern_scores = []
        
        for pattern in rich_context.synthesis_patterns:
            score = pattern.strength * 0.5
            
            # Bonus for multiple engines involved
            score += len(pattern.participating_engines) / 7.0 * 0.3
            
            # Bonus for complexity if depth is high
            if criteria.include_methodology_details:
                score += pattern.confidence * 0.2
            
            pattern_scores.append((pattern, score))
        
        # Select top patterns
        sorted_patterns = sorted(pattern_scores, key=lambda x: x[1], reverse=True)
        
        return [pattern for pattern, score in sorted_patterns[:criteria.max_synthesis_patterns]]
    
    async def _select_analogical_connections(self,
                                           rich_context: RichReasoningContext,
                                           criteria: SelectionCriteria) -> List[AnalogicalConnection]:
        """Select analogical connections based on criteria"""
        
        if not rich_context.analogical_connections:
            return []
        
        # Filter by creativity threshold
        qualified_connections = [
            conn for conn in rich_context.analogical_connections
            if conn.confidence >= criteria.creativity_threshold
        ]
        
        # Score remaining connections
        connection_scores = []
        
        for connection in qualified_connections:
            score = connection.connection_strength * connection.confidence * 0.6
            
            # Bonus for breakthrough potential
            if criteria.emphasize_breakthrough:
                score += connection.breakthrough_potential * 0.4
            
            # Bonus for novel connections
            if len(connection.insights_generated) > 2:
                score += 0.2
            
            connection_scores.append((connection, score))
        
        # Select top connections
        sorted_connections = sorted(connection_scores, key=lambda x: x[1], reverse=True)
        
        return [conn for conn, score in sorted_connections[:criteria.max_analogical_connections]]
    
    async def _select_emergent_insights(self,
                                      rich_context: RichReasoningContext,
                                      criteria: SelectionCriteria) -> List[EmergentInsight]:
        """Select emergent insights based on criteria"""
        
        if not rich_context.emergent_insights:
            return []
        
        # Filter by novelty threshold
        qualified_insights = [
            insight for insight in rich_context.emergent_insights
            if insight.novelty_score >= criteria.novelty_threshold
        ]
        
        # Score insights
        insight_scores = []
        
        for insight in qualified_insights:
            score = insight.novelty_score * insight.confidence * 0.7
            
            # Bonus for multiple contributing engines
            score += len(insight.contributing_engines) / 7.0 * 0.3
            
            insight_scores.append((insight, score))
        
        # Select top insights
        sorted_insights = sorted(insight_scores, key=lambda x: x[1], reverse=True)
        
        return [insight for insight, score in sorted_insights[:criteria.max_emergent_insights]]
    
    async def _select_breakthrough_highlights(self,
                                            rich_context: RichReasoningContext,
                                            criteria: SelectionCriteria) -> List[str]:
        """Select breakthrough highlights"""
        
        highlights = []
        
        if rich_context.breakthrough_analysis:
            highlights.extend(rich_context.breakthrough_analysis.breakthrough_areas[:3])
            
            if rich_context.breakthrough_analysis.overall_breakthrough_score > 0.7:
                highlights.append(f"Revolutionary potential: {rich_context.breakthrough_analysis.overall_breakthrough_score:.2f}")
        
        return highlights
    
    async def _select_uncertainty_exploration(self,
                                            rich_context: RichReasoningContext,
                                            criteria: SelectionCriteria) -> Optional[Dict[str, Any]]:
        """Select uncertainty exploration details"""
        
        if not rich_context.uncertainty_mapping:
            return None
        
        exploration = {
            'uncertainty_types': rich_context.uncertainty_mapping.uncertainty_types,
            'epistemic_uncertainty': rich_context.uncertainty_mapping.epistemic_uncertainty,
            'aleatoric_uncertainty': rich_context.uncertainty_mapping.aleatoric_uncertainty,
            'mitigation_strategies': rich_context.uncertainty_mapping.mitigation_strategies[:3]
        }
        
        return exploration
    
    async def _select_knowledge_gaps(self,
                                   rich_context: RichReasoningContext,
                                   criteria: SelectionCriteria) -> List[Dict[str, Any]]:
        """Select knowledge gaps for inclusion"""
        
        if not rich_context.knowledge_gaps:
            return []
        
        # Select high-impact gaps
        high_impact_gaps = [
            gap for gap in rich_context.knowledge_gaps
            if gap.impact_severity > 0.5
        ]
        
        # Format for inclusion
        formatted_gaps = []
        for gap in high_impact_gaps[:3]:
            formatted_gaps.append({
                'description': gap.gap_description,
                'impact': gap.impact_severity,
                'priority': gap.priority
            })
        
        return formatted_gaps
    
    async def _select_evidence_summary(self,
                                     rich_context: RichReasoningContext,
                                     criteria: SelectionCriteria) -> List[str]:
        """Select evidence summary"""
        
        evidence_summary = []
        
        # Collect evidence from engine insights
        for insight in rich_context.engine_insights.values():
            for evidence in insight.supporting_evidence[:2]:
                if evidence.strength > 0.6:
                    evidence_summary.append(f"{evidence.evidence_type}: {evidence.content[:100]}...")
        
        return evidence_summary[:5]
    
    async def _select_research_citations(self,
                                       rich_context: RichReasoningContext,
                                       criteria: SelectionCriteria) -> List[Dict[str, Any]]:
        """Select research citations"""
        
        # This would integrate with the actual research corpus
        # For now, return placeholder
        return [{
            'title': 'Supporting Research',
            'relevance': 0.8,
            'supporting_insights': ['Evidence-based analysis']
        }]
    
    async def _select_methodological_details(self,
                                           rich_context: RichReasoningContext,
                                           criteria: SelectionCriteria) -> List[str]:
        """Select methodological details"""
        
        details = []
        
        # Add processing information
        processing_summary = rich_context.get_processing_summary()
        details.append(f"Engines utilized: {processing_summary['engines_used']}")
        details.append(f"Processing time: {processing_summary['total_processing_time']:.2f}s")
        
        # Add quality metrics if available
        if rich_context.reasoning_quality:
            details.append(f"Logical consistency: {rich_context.reasoning_quality.logical_consistency:.2f}")
            details.append(f"Coherence score: {rich_context.reasoning_quality.coherence_score:.2f}")
        
        return details
    
    async def _select_reasoning_conflicts(self,
                                        rich_context: RichReasoningContext,
                                        criteria: SelectionCriteria) -> List[ReasoningConflict]:
        """Select reasoning conflicts"""
        
        if not rich_context.reasoning_conflicts:
            return []
        
        # Select significant conflicts
        significant_conflicts = [
            conflict for conflict in rich_context.reasoning_conflicts
            if conflict.conflict_severity > 0.5
        ]
        
        return significant_conflicts[:2]
    
    async def _select_convergence_points(self,
                                       rich_context: RichReasoningContext,
                                       criteria: SelectionCriteria) -> List[ConvergencePoint]:
        """Select convergence points"""
        
        if not rich_context.convergence_points:
            return []
        
        # Select strong convergence points
        strong_convergence = [
            point for point in rich_context.convergence_points
            if point.convergence_strength > 0.6
        ]
        
        return strong_convergence[:2]
    
    async def _generate_confidence_summary(self, rich_context: RichReasoningContext) -> str:
        """Generate confidence summary"""
        
        if rich_context.confidence_analysis:
            return f"Overall confidence: {rich_context.confidence_analysis.overall_confidence:.2f}"
        
        # Calculate from engine confidences
        if rich_context.engine_confidence_levels:
            avg_confidence = sum(rich_context.engine_confidence_levels.values()) / len(rich_context.engine_confidence_levels)
            return f"Average confidence: {avg_confidence:.2f}"
        
        return "Confidence analysis unavailable"
    
    # ========================================================================
    # Metrics and Utilities
    # ========================================================================
    
    async def _calculate_completeness_score(self,
                                          context_subset: ContextSubset,
                                          original_context: RichReasoningContext) -> float:
        """Calculate completeness score of context subset"""
        
        score = 0.0
        
        # Engine insights completeness
        if original_context.engine_insights:
            engine_ratio = len(context_subset.selected_engine_insights) / len(original_context.engine_insights)
            score += engine_ratio * 0.3
        
        # Analogical connections completeness
        if original_context.analogical_connections:
            analogy_ratio = len(context_subset.analogical_connections) / len(original_context.analogical_connections)
            score += min(analogy_ratio, 1.0) * 0.2
        
        # Synthesis patterns completeness
        if original_context.synthesis_patterns:
            pattern_ratio = len(context_subset.synthesis_patterns) / len(original_context.synthesis_patterns)
            score += min(pattern_ratio, 1.0) * 0.2
        
        # Confidence information completeness
        if context_subset.detailed_confidence or context_subset.confidence_summary:
            score += 0.15
        
        # Breakthrough information completeness
        if context_subset.emergent_insights or context_subset.breakthrough_highlights:
            score += 0.15
        
        return min(score, 1.0)
    
    async def _calculate_information_density(self, context_subset: ContextSubset) -> float:
        """Calculate information density of context subset"""
        
        # Count information elements
        element_count = (
            len(context_subset.selected_engine_insights) +
            len(context_subset.synthesis_patterns) +
            len(context_subset.analogical_connections) +
            len(context_subset.emergent_insights) +
            len(context_subset.breakthrough_highlights) +
            len(context_subset.evidence_summary) +
            len(context_subset.methodological_details)
        )
        
        # Normalize by expected maximum
        max_expected_elements = 30  # Approximate maximum for comprehensive context
        
        return min(element_count / max_expected_elements, 1.0)
    
    async def _generate_selection_rationale(self,
                                          criteria: SelectionCriteria,
                                          user_parameters: UserParameters) -> List[str]:
        """Generate rationale for context selection decisions"""
        
        rationale = []
        
        rationale.append(f"Selected {criteria.max_engine_insights} engine insights for {user_parameters.verbosity} verbosity")
        
        if criteria.emphasize_breakthrough:
            rationale.append("Emphasized breakthrough elements for creative/revolutionary mode")
        
        if criteria.emphasize_evidence:
            rationale.append("Emphasized evidence elements for evidence-focused synthesis")
        
        if criteria.include_reasoning_traces:
            rationale.append("Included reasoning traces for deep analysis depth")
        
        if not criteria.include_uncertainty_details:
            rationale.append("Minimized uncertainty details per user preference")
        
        return rationale
    
    async def _create_minimal_context_subset(self, rich_context: RichReasoningContext) -> ContextSubset:
        """Create minimal context subset as fallback"""
        
        minimal_subset = ContextSubset()
        
        # Include top 3 engine insights
        if rich_context.engine_insights:
            sorted_insights = sorted(
                rich_context.engine_insights.items(),
                key=lambda x: x[1].confidence_level,
                reverse=True
            )
            
            for engine_type, insight in sorted_insights[:3]:
                minimal_subset.selected_engine_insights[engine_type] = insight
        
        # Include basic confidence summary
        minimal_subset.confidence_summary = await self._generate_confidence_summary(rich_context)
        
        minimal_subset.context_completeness_score = 0.3
        minimal_subset.information_density = 0.2
        minimal_subset.selection_rationale = ["Minimal context due to selection error"]
        
        return minimal_subset