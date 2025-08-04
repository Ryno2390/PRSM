"""
Response Enhancement Engine for NWTN Contextual Synthesizer.

This module enhances raw Claude API responses with contextual elements including
citations, confidence indicators, reasoning transparency, and follow-up suggestions
based on the rich reasoning context.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from .reasoning_context_types import (
    RichReasoningContext, EngineType, EngineInsight, AnalogicalConnection,
    ConfidenceAnalysis, BreakthroughAnalysis
)
from .contextual_synthesizer import (
    EnhancedResponse, SynthesisConfiguration, UserParameters
)


logger = logging.getLogger(__name__)


# ============================================================================
# Enhancement Data Structures
# ============================================================================

@dataclass
class CitationContext:
    """Context for a citation"""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    relevance_score: float
    supporting_insights: List[str]
    reasoning_engines: List[EngineType]
    citation_type: str  # 'supporting', 'contrasting', 'analogical', 'methodological'


@dataclass
class ConfidenceIndicator:
    """Confidence indicator for response sections"""
    section: str
    confidence_level: float
    confidence_source: str  # 'engine_consensus', 'evidence_strength', 'analogical_support'
    uncertainty_factors: List[str]
    boosting_factors: List[str]


@dataclass
class ReasoningTransparency:
    """Transparency information about reasoning process"""
    engines_utilized: List[str]
    reasoning_path: List[str]
    key_insights: List[str]
    analogical_connections: List[str]
    synthesis_patterns: List[str]
    confidence_distribution: Dict[str, float]
    breakthrough_indicators: List[str]


class ResponseEnhancementEngine:
    """
    Enhances responses with citations, provenance, confidence indicators,
    and contextual information based on rich reasoning context.
    """
    
    def __init__(self):
        """Initialize the response enhancement engine"""
        self.citation_manager = ProvenanceCitationManager()
        self.confidence_presenter = ConfidencePresenter()
        self.transparency_generator = ReasoningTransparencyGenerator()
        self.follow_up_generator = FollowUpSuggestionGenerator()
        
        # Enhancement configuration
        self.enhancement_config = {
            'max_citations_per_section': 3,
            'confidence_threshold_for_indication': 0.3,
            'transparency_detail_level': 'moderate',
            'follow_up_suggestion_count': 3
        }
        
        logger.info("ResponseEnhancementEngine initialized with all components")
    
    async def enhance_response(self,
                             raw_response: str,
                             rich_context: RichReasoningContext,
                             search_corpus: List[Any],
                             synthesis_config: SynthesisConfiguration) -> EnhancedResponse:
        """
        Enhance raw Claude response with contextual elements.
        
        Args:
            raw_response: Raw response from Claude API
            rich_context: Rich reasoning context from Phase 1
            search_corpus: Search corpus used in reasoning
            synthesis_config: Configuration for synthesis strategy
            
        Returns:
            EnhancedResponse with all contextual enhancements
        """
        
        logger.info("Starting response enhancement with contextual elements")
        enhancement_start = datetime.now()
        
        try:
            # Step 1: Generate contextual citations
            logger.debug("Generating contextual citations")
            citations = await self.citation_manager.generate_contextual_citations(
                raw_response, rich_context, search_corpus, synthesis_config
            )
            
            # Step 2: Add confidence indicators
            logger.debug("Adding confidence indicators")
            confidence_indicators = await self.confidence_presenter.generate_confidence_indicators(
                raw_response, rich_context, synthesis_config
            )
            
            # Step 3: Generate reasoning transparency
            logger.debug("Generating reasoning transparency")
            reasoning_transparency = await self.transparency_generator.generate_transparency_info(
                rich_context, synthesis_config
            )
            
            # Step 4: Generate follow-up suggestions
            logger.debug("Generating follow-up suggestions")
            follow_up_suggestions = await self.follow_up_generator.generate_suggestions(
                raw_response, rich_context, synthesis_config
            )
            
            # Step 5: Extract breakthrough highlights
            breakthrough_highlights = self._extract_breakthrough_highlights(rich_context)
            
            # Step 6: Extract uncertainty acknowledgments
            uncertainty_acknowledgments = self._extract_uncertainty_acknowledgments(rich_context)
            
            # Step 7: Extract analogical insights
            analogical_insights = self._extract_analogical_insights(rich_context)
            
            # Step 8: Enhance the main response text with inline elements
            enhanced_text = await self._enhance_response_text(
                raw_response, citations, confidence_indicators, synthesis_config
            )
            
            # Create enhanced response
            processing_time = (datetime.now() - enhancement_start).total_seconds()
            
            enhanced_response = EnhancedResponse(
                enhanced_text=enhanced_text,
                citations=citations,
                confidence_indicators={ci.section: ci.confidence_level for ci in confidence_indicators},
                follow_up_suggestions=follow_up_suggestions,
                reasoning_transparency=reasoning_transparency.__dict__,
                breakthrough_highlights=breakthrough_highlights,
                uncertainty_acknowledgments=uncertainty_acknowledgments,
                analogical_insights=analogical_insights,
                synthesis_metadata={
                    'enhancement_time': processing_time,
                    'citations_added': len(citations),
                    'confidence_indicators_added': len(confidence_indicators),
                    'transparency_level': self.enhancement_config['transparency_detail_level']
                }
            )
            
            logger.info(f"Response enhancement completed in {processing_time:.2f}s with "
                       f"{len(citations)} citations and {len(confidence_indicators)} confidence indicators")
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error in response enhancement: {e}")
            # Return minimally enhanced response
            return await self._create_minimal_enhanced_response(raw_response, str(e))
    
    # ========================================================================
    # Response Text Enhancement
    # ========================================================================
    
    async def _enhance_response_text(self,
                                   raw_response: str,
                                   citations: List[Dict[str, Any]],
                                   confidence_indicators: List[ConfidenceIndicator],
                                   synthesis_config: SynthesisConfiguration) -> str:
        """Enhance the main response text with inline elements"""
        
        enhanced_text = raw_response
        
        # Add inline citations based on user preferences
        if synthesis_config.citation_integration in ['inline_attribution', 'contextual_citations']:
            enhanced_text = await self._add_inline_citations(enhanced_text, citations)
        
        # Add confidence indicators if requested
        if synthesis_config.confidence_presentation in ['explicit_confidence_levels', 'natural_acknowledgment']:
            enhanced_text = await self._add_confidence_indicators(enhanced_text, confidence_indicators)
        
        # Add reasoning transparency elements if appropriate
        if synthesis_config.user_parameters.depth in ['DEEP', 'EXHAUSTIVE']:
            enhanced_text = await self._add_reasoning_transparency(enhanced_text, synthesis_config)
        
        return enhanced_text
    
    async def _add_inline_citations(self, text: str, citations: List[Dict[str, Any]]) -> str:
        """Add inline citations to the response text"""
        
        # Simple implementation - in production this would be more sophisticated
        enhanced_text = text
        
        # Add citations at the end of major claims
        sentences = text.split('. ')
        
        for i, sentence in enumerate(sentences):
            # Look for relevant citations for this sentence
            relevant_citations = [
                cite for cite in citations
                if any(insight.lower() in sentence.lower() for insight in cite.get('supporting_insights', []))
            ]
            
            if relevant_citations and i < len(sentences) - 1:  # Don't add to last sentence
                citation_refs = ', '.join([f"({cite['title'][:30]}...)" for cite in relevant_citations[:2]])
                sentences[i] = f"{sentence} [{citation_refs}]"
        
        return '. '.join(sentences)
    
    async def _add_confidence_indicators(self, 
                                       text: str, 
                                       confidence_indicators: List[ConfidenceIndicator]) -> str:
        """Add confidence indicators to the response text"""
        
        # Add confidence phrases based on confidence levels
        enhanced_text = text
        
        # Simple confidence phrase insertion
        confidence_phrases = {
            0.9: "with high confidence",
            0.7: "based on strong evidence",
            0.5: "with moderate confidence",
            0.3: "tentatively",
            0.1: "speculatively"
        }
        
        # This is a simplified implementation
        return enhanced_text
    
    async def _add_reasoning_transparency(self, 
                                        text: str, 
                                        synthesis_config: SynthesisConfiguration) -> str:
        """Add reasoning transparency elements to the text"""
        
        # Add reasoning methodology explanation for deep/exhaustive responses
        if synthesis_config.user_parameters.depth == 'EXHAUSTIVE':
            transparency_note = "\n\n*This analysis integrates insights from multiple reasoning approaches including deductive, inductive, abductive, analogical, causal, counterfactual, and probabilistic reasoning engines.*"
            return text + transparency_note
        
        return text
    
    # ========================================================================
    # Content Extraction Methods
    # ========================================================================
    
    def _extract_breakthrough_highlights(self, rich_context: RichReasoningContext) -> List[str]:
        """Extract breakthrough highlights from rich context"""
        
        highlights = []
        
        # From breakthrough analysis
        if rich_context.breakthrough_analysis:
            highlights.extend(rich_context.breakthrough_analysis.breakthrough_areas)
            if rich_context.breakthrough_analysis.overall_breakthrough_score > 0.7:
                highlights.append(f"Revolutionary potential score: {rich_context.breakthrough_analysis.overall_breakthrough_score:.2f}")
        
        # From emergent insights
        for insight in rich_context.emergent_insights[:3]:
            if insight.novelty_score > 0.6:
                highlights.append(insight.insight_description)
        
        # From high-impact analogical connections
        for connection in rich_context.analogical_connections:
            if connection.breakthrough_potential > 0.7:
                highlights.append(f"Breakthrough analogy: {connection.source_domain} ↔ {connection.target_domain}")
        
        return highlights[:5]  # Limit to top 5 highlights
    
    def _extract_uncertainty_acknowledgments(self, rich_context: RichReasoningContext) -> List[str]:
        """Extract uncertainty acknowledgments from rich context"""
        
        acknowledgments = []
        
        # From uncertainty mapping
        if rich_context.uncertainty_mapping:
            for uncertainty_type, level in rich_context.uncertainty_mapping.uncertainty_types.items():
                if level > 0.5:
                    acknowledgments.append(f"Uncertainty in {uncertainty_type}: {level:.2f}")
        
        # From knowledge gaps
        for gap in rich_context.knowledge_gaps[:3]:
            if gap.impact_severity > 0.5:
                acknowledgments.append(f"Knowledge gap: {gap.gap_description}")
        
        # From confidence analysis
        if rich_context.confidence_analysis:
            for detractor in rich_context.confidence_analysis.confidence_detractors[:2]:
                acknowledgments.append(f"Confidence limitation: {detractor}")
        
        return acknowledgments
    
    def _extract_analogical_insights(self, rich_context: RichReasoningContext) -> List[str]:
        """Extract key analogical insights from rich context"""
        
        insights = []
        
        # From analogical connections
        for connection in rich_context.analogical_connections[:3]:
            if connection.confidence > 0.6:
                insight = f"{connection.source_domain} → {connection.target_domain}: {', '.join(connection.insights_generated[:2])}"
                insights.append(insight)
        
        # From cross-domain bridges
        for bridge in rich_context.cross_domain_bridges[:2]:
            insight = f"Cross-domain bridge '{bridge.bridge_concept}' connects {', '.join(bridge.connected_domains)}"
            insights.append(insight)
        
        return insights
    
    async def _create_minimal_enhanced_response(self, raw_response: str, error_msg: str) -> EnhancedResponse:
        """Create minimal enhanced response in case of enhancement errors"""
        
        return EnhancedResponse(
            enhanced_text=raw_response,
            citations=[],
            confidence_indicators={"overall": "Enhancement error occurred"},
            follow_up_suggestions=["Please try rephrasing your question"],
            reasoning_transparency={"error": error_msg},
            breakthrough_highlights=[],
            uncertainty_acknowledgments=["Enhancement process encountered errors"],
            analogical_insights=[],
            synthesis_metadata={"enhancement_error": True, "error": error_msg}
        )


# ============================================================================
# Specialized Enhancement Components
# ============================================================================

class ProvenanceCitationManager:
    """Manages generation of contextual citations and provenance information"""
    
    def __init__(self):
        """Initialize the citation manager"""
        self.citation_styles = {
            'formal_citations': self._generate_formal_citations,
            'contextual_citations': self._generate_contextual_citations,
            'inline_attribution': self._generate_inline_attribution,
            'minimal_attribution': self._generate_minimal_attribution
        }
        
        logger.info("ProvenanceCitationManager initialized")
    
    async def generate_contextual_citations(self,
                                          raw_response: str,
                                          rich_context: RichReasoningContext,
                                          search_corpus: List[Any],
                                          synthesis_config: SynthesisConfiguration) -> List[Dict[str, Any]]:
        """Generate contextual citations based on synthesis configuration"""
        
        citation_style = synthesis_config.citation_integration
        generator = self.citation_styles.get(citation_style, self._generate_contextual_citations)
        
        return await generator(raw_response, rich_context, search_corpus)
    
    async def _generate_formal_citations(self,
                                       raw_response: str,
                                       rich_context: RichReasoningContext,
                                       search_corpus: List[Any]) -> List[Dict[str, Any]]:
        """Generate formal academic citations"""
        
        citations = []
        
        # Map insights to papers from search corpus
        for i, paper in enumerate(search_corpus[:10]):  # Limit to top 10 papers
            citation = {
                'id': f"ref_{i+1}",
                'title': getattr(paper, 'title', f'Paper {i+1}'),
                'authors': getattr(paper, 'authors', ['Unknown']),
                'year': getattr(paper, 'year', 2024),
                'relevance_score': 0.8,  # Would be calculated based on actual relevance
                'supporting_insights': ['General research support'],
                'reasoning_engines': [EngineType.DEDUCTIVE],
                'citation_type': 'supporting',
                'formal_citation': f"Author et al. ({getattr(paper, 'year', 2024)}). {getattr(paper, 'title', 'Unknown Title')}."
            }
            citations.append(citation)
        
        return citations
    
    async def _generate_contextual_citations(self,
                                           raw_response: str,
                                           rich_context: RichReasoningContext,
                                           search_corpus: List[Any]) -> List[Dict[str, Any]]:
        """Generate contextual citations that map to specific insights"""
        
        citations = []
        
        # Map engine insights to supporting papers
        for engine_type, insight in rich_context.engine_insights.items():
            for finding in insight.primary_findings[:2]:  # Top 2 findings per engine
                # Find relevant papers (simplified matching)
                relevant_papers = [paper for paper in search_corpus[:5]]  # Simplified
                
                for paper in relevant_papers:
                    citation = {
                        'id': f"{engine_type.value}_{len(citations)+1}",
                        'title': getattr(paper, 'title', f'Supporting Research {len(citations)+1}'),
                        'authors': getattr(paper, 'authors', ['Research Team']),
                        'year': getattr(paper, 'year', 2024),
                        'relevance_score': insight.confidence_level,
                        'supporting_insights': [finding],
                        'reasoning_engines': [engine_type],
                        'citation_type': 'supporting',
                        'context': f"Supports {engine_type.value} reasoning finding: {finding}"
                    }
                    citations.append(citation)
                    break  # One paper per finding
        
        return citations
    
    async def _generate_inline_attribution(self,
                                         raw_response: str,
                                         rich_context: RichReasoningContext,
                                         search_corpus: List[Any]) -> List[Dict[str, Any]]:
        """Generate inline attribution for response text"""
        
        # Similar to contextual but designed for inline integration
        return await self._generate_contextual_citations(raw_response, rich_context, search_corpus)
    
    async def _generate_minimal_attribution(self,
                                          raw_response: str,
                                          rich_context: RichReasoningContext,
                                          search_corpus: List[Any]) -> List[Dict[str, Any]]:
        """Generate minimal attribution information"""
        
        # Just provide corpus size and general attribution
        return [{
            'id': 'corpus_attribution',
            'title': 'Research Corpus',
            'authors': ['NWTN Analysis'],
            'year': 2024,
            'relevance_score': 1.0,
            'supporting_insights': ['General research support'],
            'reasoning_engines': list(rich_context.engine_insights.keys()),
            'citation_type': 'corpus',
            'context': f"Analysis based on {len(search_corpus)} research papers"
        }]


class ConfidencePresenter:
    """Presents confidence information in natural language"""
    
    def __init__(self):
        """Initialize the confidence presenter"""
        self.confidence_phrases = {
            0.9: ["with high confidence", "strongly supported", "well-established"],
            0.7: ["with good confidence", "supported by evidence", "likely"],
            0.5: ["with moderate confidence", "tentatively", "possibly"],
            0.3: ["with low confidence", "speculatively", "uncertainly"],
            0.1: ["very tentatively", "with significant uncertainty", "speculatively"]
        }
        
        logger.info("ConfidencePresenter initialized")
    
    async def generate_confidence_indicators(self,
                                           raw_response: str,
                                           rich_context: RichReasoningContext,
                                           synthesis_config: SynthesisConfiguration) -> List[ConfidenceIndicator]:
        """Generate confidence indicators for response sections"""
        
        indicators = []
        
        # Overall confidence from confidence analysis
        if rich_context.confidence_analysis:
            overall_indicator = ConfidenceIndicator(
                section="overall",
                confidence_level=rich_context.confidence_analysis.overall_confidence,
                confidence_source="engine_consensus",
                uncertainty_factors=rich_context.confidence_analysis.uncertainty_sources[:3],
                boosting_factors=rich_context.confidence_analysis.confidence_boosters[:3]
            )
            indicators.append(overall_indicator)
        
        # Per-engine confidence indicators
        for engine_type, confidence in rich_context.engine_confidence_levels.items():
            if confidence != 0.5:  # Only include non-default confidence levels
                indicator = ConfidenceIndicator(
                    section=f"{engine_type.value}_analysis",
                    confidence_level=confidence,
                    confidence_source=f"{engine_type.value}_reasoning",
                    uncertainty_factors=[],
                    boosting_factors=[]
                )
                indicators.append(indicator)
        
        return indicators


class ReasoningTransparencyGenerator:
    """Generates reasoning transparency information"""
    
    def __init__(self):
        """Initialize the transparency generator"""
        logger.info("ReasoningTransparencyGenerator initialized")
    
    async def generate_transparency_info(self,
                                       rich_context: RichReasoningContext,
                                       synthesis_config: SynthesisConfiguration) -> ReasoningTransparency:
        """Generate reasoning transparency information"""
        
        # Extract transparency information from rich context
        engines_utilized = [engine.value for engine in rich_context.engine_insights.keys()]
        
        reasoning_path = []
        for pattern in rich_context.synthesis_patterns:
            reasoning_path.append(f"{pattern.pattern_type.value}: {pattern.synthesis_description}")
        
        key_insights = []
        for insight in rich_context.emergent_insights[:3]:
            key_insights.append(insight.insight_description)
        
        analogical_connections = []
        for connection in rich_context.analogical_connections[:3]:
            analogical_connections.append(f"{connection.source_domain} ↔ {connection.target_domain}")
        
        synthesis_patterns = [pattern.synthesis_description for pattern in rich_context.synthesis_patterns]
        
        confidence_distribution = {
            engine.value: confidence for engine, confidence in rich_context.engine_confidence_levels.items()
        }
        
        breakthrough_indicators = []
        if rich_context.breakthrough_analysis:
            breakthrough_indicators = rich_context.breakthrough_analysis.paradigm_shift_indicators
        
        return ReasoningTransparency(
            engines_utilized=engines_utilized,
            reasoning_path=reasoning_path,
            key_insights=key_insights,
            analogical_connections=analogical_connections,
            synthesis_patterns=synthesis_patterns,
            confidence_distribution=confidence_distribution,
            breakthrough_indicators=breakthrough_indicators
        )


class FollowUpSuggestionGenerator:
    """Generates intelligent follow-up suggestions"""
    
    def __init__(self):
        """Initialize the follow-up generator"""
        logger.info("FollowUpSuggestionGenerator initialized")
    
    async def generate_suggestions(self,
                                 raw_response: str,
                                 rich_context: RichReasoningContext,
                                 synthesis_config: SynthesisConfiguration) -> List[str]:
        """Generate follow-up suggestions based on reasoning context"""
        
        suggestions = []
        
        # Suggestions based on breakthrough potential
        if rich_context.breakthrough_analysis and rich_context.breakthrough_analysis.overall_breakthrough_score > 0.6:
            suggestions.append("Would you like me to explore the revolutionary implications of these findings in more detail?")
            suggestions.append("What specific applications of these breakthrough insights interest you most?")
        
        # Suggestions based on analogical connections
        if rich_context.analogical_connections:
            strong_analogies = [conn for conn in rich_context.analogical_connections if conn.connection_strength > 0.7]
            if strong_analogies:
                conn = strong_analogies[0]
                suggestions.append(f"Would you like to explore the analogy between {conn.source_domain} and {conn.target_domain} further?")
        
        # Suggestions based on knowledge gaps
        if rich_context.knowledge_gaps:
            high_priority_gaps = [gap for gap in rich_context.knowledge_gaps if gap.priority in ['high', 'critical']]
            if high_priority_gaps:
                gap = high_priority_gaps[0]
                suggestions.append(f"I notice a knowledge gap in {gap.gap_description}. Would you like me to suggest research directions?")
        
        # Suggestions based on conflicts
        if rich_context.reasoning_conflicts:
            unresolved_conflicts = [conflict for conflict in rich_context.reasoning_conflicts if conflict.resolution_status == 'unresolved']
            if unresolved_conflicts:
                suggestions.append("There are some unresolved tensions between different reasoning approaches. Would you like me to explore these further?")
        
        # Suggestions based on synthesis strategy
        if synthesis_config.strategy_type == SynthesisStrategy.UNCERTAINTY_NAVIGATION:
            suggestions.append("What aspects of the uncertainty would you like me to investigate further?")
            suggestions.append("Would you like suggestions for reducing the key uncertainties identified?")
        
        # Default suggestions if none generated
        if not suggestions:
            suggestions = [
                "What aspects of this analysis would you like me to explore in more depth?",
                "Are there related questions or applications you'd like me to investigate?",
                "Would you like me to focus on any particular reasoning approach or perspective?"
            ]
        
        # Limit to configured number of suggestions
        max_suggestions = 3
        return suggestions[:max_suggestions]