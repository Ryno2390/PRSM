"""
Contextual Synthesizer for NWTN - Phase 2 Implementation.

This module implements the sophisticated contextual synthesizer that leverages
rich reasoning context to generate nuanced, engaging natural language responses,
replacing the generic Voicebox that produces stilted outputs.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field

from .reasoning_context_types import (
    RichReasoningContext, EngineType, EngineInsight, SynthesisPattern,
    AnalogicalConnection, ConfidenceAnalysis, BreakthroughAnalysis
)


logger = logging.getLogger(__name__)


# ============================================================================
# Synthesis Configuration and Data Structures
# ============================================================================

class SynthesisStrategy(Enum):
    """Different synthesis strategies based on reasoning characteristics"""
    BREAKTHROUGH_NARRATIVE = "breakthrough_narrative"
    ANALOGICAL_EXPLORATION = "analogical_exploration"
    NUANCED_ANALYSIS = "nuanced_analysis"
    EVIDENCE_SYNTHESIS = "evidence_synthesis"
    UNCERTAINTY_NAVIGATION = "uncertainty_navigation"
    CONVERGENCE_SYNTHESIS = "convergence_synthesis"
    CONFLICT_RESOLUTION = "conflict_resolution"


class ResponseTone(Enum):
    """Response tone options"""
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"
    INNOVATIVE = "innovative"
    ANALYTICAL = "analytical"
    EXPLORATORY = "exploratory"


@dataclass
class UserParameters:
    """User parameters for synthesis customization"""
    verbosity: str = "MODERATE"              # BRIEF, MODERATE, COMPREHENSIVE, EXHAUSTIVE
    reasoning_mode: str = "BALANCED"         # CONSERVATIVE, BALANCED, CREATIVE, REVOLUTIONARY
    depth: str = "INTERMEDIATE"              # SURFACE, INTERMEDIATE, DEEP, EXHAUSTIVE
    synthesis_focus: str = "INSIGHTS"        # INSIGHTS, EVIDENCE, METHODOLOGY, IMPLICATIONS
    citation_style: str = "CONTEXTUAL"       # MINIMAL, CONTEXTUAL, COMPREHENSIVE, ACADEMIC
    uncertainty_handling: str = "ACKNOWLEDGE" # HIDE, ACKNOWLEDGE, EXPLORE, EMPHASIZE
    tone: ResponseTone = ResponseTone.CONVERSATIONAL


@dataclass
class SynthesisConfiguration:
    """Configuration for synthesis strategy"""
    strategy_type: SynthesisStrategy
    user_parameters: UserParameters
    reasoning_characteristics: Dict[str, Any]
    context_emphasis: Dict[str, float]       # What aspects of context to emphasize
    response_structure: List[str]            # Ordered list of response sections
    citation_integration: str               # How to integrate citations
    confidence_presentation: str            # How to present confidence levels


@dataclass
class EnhancedResponse:
    """Enhanced response with rich contextual information"""
    enhanced_text: str                       # Main response text
    citations: List[Dict[str, Any]]          # Contextual citations
    confidence_indicators: Dict[str, str]    # Confidence indicators
    follow_up_suggestions: List[str]         # Suggested follow-up questions
    reasoning_transparency: Dict[str, Any]   # Reasoning process transparency
    breakthrough_highlights: List[str]       # Key breakthrough insights
    uncertainty_acknowledgments: List[str]   # Acknowledged uncertainties
    analogical_insights: List[str]          # Key analogical connections
    synthesis_metadata: Dict[str, Any]      # Metadata about synthesis process


# ============================================================================
# Main Contextual Synthesizer
# ============================================================================

class ContextualSynthesizer:
    """
    Advanced synthesis engine that leverages rich reasoning context
    to generate nuanced, engaging natural language responses.
    
    This replaces the generic Voicebox and addresses the core problem
    of context starvation that leads to stilted, anodyne outputs.
    """
    
    def __init__(self):
        """Initialize the contextual synthesizer"""
        self.prompt_engineer = ContextRichPromptEngineer()
        self.synthesis_strategist = AdaptiveSynthesisStrategist()
        self.response_enhancer = ResponseEnhancementEngine()
        self.citation_manager = ProvenanceCitationManager()
        self.confidence_presenter = ConfidencePresenter()
        
        # Claude API configuration (would be configured externally)
        self.claude_api_config = {
            'model': 'claude-3-sonnet-20240229',
            'max_tokens': 4000,
            'temperature': 0.7
        }
        
        logger.info("ContextualSynthesizer initialized with all components")
    
    async def synthesize_response(self,
                                original_query: str,
                                rich_context: RichReasoningContext,
                                search_corpus: List[Any],
                                user_parameters: UserParameters) -> EnhancedResponse:
        """
        Main synthesis method that generates sophisticated natural language responses.
        
        Args:
            original_query: The user's original query
            rich_context: Rich reasoning context from Phase 1
            search_corpus: Search corpus used in reasoning
            user_parameters: User customization parameters
            
        Returns:
            EnhancedResponse with sophisticated natural language synthesis
        """
        
        logger.info(f"Starting contextual synthesis for query: {original_query[:100]}...")
        synthesis_start = datetime.now()
        
        try:
            # Step 1: Analyze reasoning characteristics and select synthesis strategy
            logger.debug("Analyzing reasoning characteristics and selecting synthesis strategy")
            synthesis_config = await self.synthesis_strategist.configure_synthesis(
                rich_context, user_parameters
            )
            
            # Step 2: Build context-rich prompt
            logger.debug(f"Building context-rich prompt with strategy: {synthesis_config.strategy_type.value}")
            synthesis_prompt = await self.prompt_engineer.build_contextual_prompt(
                original_query, rich_context, search_corpus, synthesis_config
            )
            
            # Step 3: Generate raw response via Claude API
            logger.debug("Generating response via Claude API with rich context")
            raw_response = await self._generate_claude_response(synthesis_prompt, synthesis_config)
            
            # Step 4: Enhance response with contextual elements
            logger.debug("Enhancing response with citations, confidence, and contextual elements")
            enhanced_response = await self.response_enhancer.enhance_response(
                raw_response, rich_context, search_corpus, synthesis_config
            )
            
            # Step 5: Add synthesis transparency and metadata
            enhanced_response = await self._add_synthesis_transparency(
                enhanced_response, rich_context, synthesis_config
            )
            
            processing_time = (datetime.now() - synthesis_start).total_seconds()
            enhanced_response.synthesis_metadata.update({
                'synthesis_time': processing_time,
                'strategy_used': synthesis_config.strategy_type.value,
                'user_parameters': user_parameters.__dict__,
                'context_utilization': self._calculate_context_utilization(rich_context)
            })
            
            logger.info(f"Contextual synthesis completed in {processing_time:.2f}s "
                       f"using {synthesis_config.strategy_type.value} strategy")
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error in contextual synthesis: {e}")
            # Fallback to basic synthesis
            return await self._fallback_synthesis(original_query, rich_context, str(e))
    
    # ========================================================================
    # Core Synthesis Methods
    # ========================================================================
    
    async def _generate_claude_response(self, 
                                      synthesis_prompt: str,
                                      synthesis_config: SynthesisConfiguration) -> str:
        """Generate response using Claude API with contextual prompt"""
        
        try:
            # Adjust Claude parameters based on synthesis configuration
            api_params = self._adapt_claude_parameters(synthesis_config)
            
            # In a real implementation, this would call the Claude API
            # For now, we'll simulate the API call
            simulated_response = await self._simulate_claude_api_call(
                synthesis_prompt, api_params
            )
            
            logger.debug(f"Generated response of {len(simulated_response)} characters")
            return simulated_response
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return f"I apologize, but I encountered an error while generating the response: {str(e)}"
    
    async def _simulate_claude_api_call(self, prompt: str, params: Dict[str, Any]) -> str:
        """Simulate Claude API call for testing purposes"""
        
        # This is a placeholder - in production this would be replaced with actual Claude API calls
        return f"""Based on the comprehensive reasoning analysis, I can provide you with a nuanced response that integrates insights from multiple reasoning approaches.

The analysis reveals fascinating convergences and tensions between different reasoning modes. The analogical engine identified striking parallels between the concepts in your query, while the causal analysis uncovered specific mechanisms that drive the phenomena you're asking about.

What's particularly intriguing is how the deductive reasoning validates these findings through first-principles analysis, while the probabilistic assessment suggests varying degrees of confidence across different aspects of the response.

The breakthrough potential here lies in the cross-domain connections that emerged during the reasoning process, suggesting novel applications and research directions that weren't immediately obvious from the surface-level analysis.

Let me walk you through the key insights and their implications for your specific question..."""
    
    def _adapt_claude_parameters(self, synthesis_config: SynthesisConfiguration) -> Dict[str, Any]:
        """Adapt Claude API parameters based on synthesis configuration"""
        
        params = self.claude_api_config.copy()
        
        # Adjust temperature based on reasoning mode
        if synthesis_config.user_parameters.reasoning_mode == "REVOLUTIONARY":
            params['temperature'] = 0.9  # Higher creativity
        elif synthesis_config.user_parameters.reasoning_mode == "CONSERVATIVE":
            params['temperature'] = 0.3  # Lower creativity
        
        # Adjust max_tokens based on verbosity
        verbosity_multipliers = {
            "BRIEF": 0.5,
            "MODERATE": 1.0,
            "COMPREHENSIVE": 1.5,
            "EXHAUSTIVE": 2.0
        }
        
        multiplier = verbosity_multipliers.get(synthesis_config.user_parameters.verbosity, 1.0)
        params['max_tokens'] = int(params['max_tokens'] * multiplier)
        
        return params
    
    async def _add_synthesis_transparency(self,
                                        enhanced_response: EnhancedResponse,
                                        rich_context: RichReasoningContext,
                                        synthesis_config: SynthesisConfiguration) -> EnhancedResponse:
        """Add transparency about the synthesis process"""
        
        # Add reasoning transparency
        enhanced_response.reasoning_transparency = {
            'engines_utilized': list(rich_context.engine_insights.keys()),
            'synthesis_patterns_found': len(rich_context.synthesis_patterns),
            'analogical_connections': len(rich_context.analogical_connections),
            'confidence_distribution': rich_context.engine_confidence_levels,
            'strategy_rationale': synthesis_config.strategy_type.value,
            'breakthrough_indicators': [
                insight.insight_description for insight in rich_context.emergent_insights
            ]
        }
        
        # Add breakthrough highlights
        if rich_context.breakthrough_analysis and rich_context.breakthrough_analysis.overall_breakthrough_score > 0.6:
            enhanced_response.breakthrough_highlights = rich_context.breakthrough_analysis.breakthrough_areas
        
        # Add uncertainty acknowledgments
        if rich_context.uncertainty_mapping:
            enhanced_response.uncertainty_acknowledgments = [
                f"Uncertainty in {domain}: {level:.2f}" 
                for domain, level in rich_context.uncertainty_mapping.uncertainty_types.items()
            ]
        
        return enhanced_response
    
    def _calculate_context_utilization(self, rich_context: RichReasoningContext) -> Dict[str, float]:
        """Calculate how well different aspects of context were utilized"""
        
        return {
            'engine_insights_utilized': len(rich_context.engine_insights) / 7.0,  # 7 total engines
            'analogical_richness': min(1.0, len(rich_context.analogical_connections) / 5.0),
            'synthesis_patterns': min(1.0, len(rich_context.synthesis_patterns) / 3.0),
            'breakthrough_potential': rich_context.breakthrough_analysis.overall_breakthrough_score if rich_context.breakthrough_analysis else 0.0
        }
    
    async def _fallback_synthesis(self, 
                                query: str, 
                                rich_context: RichReasoningContext,
                                error_msg: str) -> EnhancedResponse:
        """Fallback synthesis in case of errors"""
        
        fallback_text = f"""I apologize, but I encountered an issue during the advanced synthesis process. However, I can still provide you with insights based on the reasoning analysis.

Based on the available reasoning context, here are the key insights I can share:

{self._extract_basic_insights(rich_context)}

Error details: {error_msg}"""
        
        return EnhancedResponse(
            enhanced_text=fallback_text,
            citations=[],
            confidence_indicators={"overall": "Limited due to synthesis error"},
            follow_up_suggestions=["Please try rephrasing your question"],
            reasoning_transparency={"error": error_msg},
            breakthrough_highlights=[],
            uncertainty_acknowledgments=["Synthesis process encountered errors"],
            analogical_insights=[],
            synthesis_metadata={"fallback_mode": True, "error": error_msg}
        )
    
    def _extract_basic_insights(self, rich_context: RichReasoningContext) -> str:
        """Extract basic insights for fallback synthesis"""
        
        insights = []
        
        # Extract from engine insights
        for engine_type, insight in rich_context.engine_insights.items():
            if insight.primary_findings:
                insights.append(f"• {engine_type.value.title()}: {insight.primary_findings[0]}")
        
        # Add analogical connections if available
        if rich_context.analogical_connections:
            connection = rich_context.analogical_connections[0]
            insights.append(f"• Analogical insight: Connection between {connection.source_domain} and {connection.target_domain}")
        
        return "\n".join(insights) if insights else "Limited insights available due to processing error."


# ============================================================================
# Context-Rich Prompt Engineering
# ============================================================================

class ContextRichPromptEngineer:
    """Builds sophisticated prompts that leverage rich reasoning context"""
    
    def __init__(self):
        """Initialize the prompt engineer"""
        self.prompt_templates = {
            SynthesisStrategy.BREAKTHROUGH_NARRATIVE: self._build_breakthrough_prompt,
            SynthesisStrategy.ANALOGICAL_EXPLORATION: self._build_analogical_prompt,
            SynthesisStrategy.NUANCED_ANALYSIS: self._build_nuanced_prompt,
            SynthesisStrategy.EVIDENCE_SYNTHESIS: self._build_evidence_prompt,
            SynthesisStrategy.UNCERTAINTY_NAVIGATION: self._build_uncertainty_prompt,
            SynthesisStrategy.CONVERGENCE_SYNTHESIS: self._build_convergence_prompt,
            SynthesisStrategy.CONFLICT_RESOLUTION: self._build_conflict_prompt
        }
        
        logger.info("ContextRichPromptEngineer initialized with strategy templates")
    
    async def build_contextual_prompt(self,
                                    original_query: str,
                                    rich_context: RichReasoningContext,
                                    search_corpus: List[Any],
                                    synthesis_config: SynthesisConfiguration) -> str:
        """Build comprehensive contextual prompt"""
        
        # Get strategy-specific prompt builder
        prompt_builder = self.prompt_templates.get(
            synthesis_config.strategy_type,
            self._build_default_prompt
        )
        
        # Build the contextual prompt
        contextual_prompt = await prompt_builder(
            original_query, rich_context, search_corpus, synthesis_config
        )
        
        logger.debug(f"Built contextual prompt of {len(contextual_prompt)} characters "
                    f"using {synthesis_config.strategy_type.value} strategy")
        
        return contextual_prompt
    
    async def _build_breakthrough_prompt(self,
                                       query: str,
                                       rich_context: RichReasoningContext,
                                       search_corpus: List[Any],
                                       synthesis_config: SynthesisConfiguration) -> str:
        """Build prompt for breakthrough narrative synthesis"""
        
        prompt_sections = []
        
        # Context overview
        prompt_sections.append(f"""You are synthesizing breakthrough insights from sophisticated multi-modal reasoning analysis.

ORIGINAL QUERY: {query}

BREAKTHROUGH CONTEXT:""")
        
        # Breakthrough analysis
        if rich_context.breakthrough_analysis:
            breakthrough_score = rich_context.breakthrough_analysis.overall_breakthrough_score
            breakthrough_areas = rich_context.breakthrough_analysis.breakthrough_areas
            
            prompt_sections.append(f"""
BREAKTHROUGH POTENTIAL: {breakthrough_score:.2f}/1.0 ({rich_context.breakthrough_analysis.breakthrough_category.value})
BREAKTHROUGH AREAS: {', '.join(breakthrough_areas)}
PARADIGM SHIFT INDICATORS: {', '.join(rich_context.breakthrough_analysis.paradigm_shift_indicators)}""")
        
        # Engine insights focused on breakthrough potential
        prompt_sections.append("\nREASONING ENGINE BREAKTHROUGH INSIGHTS:")
        for engine_type, insight in rich_context.engine_insights.items():
            if insight.breakthrough_indicators:
                prompt_sections.append(f"""
{engine_type.value.upper()} ENGINE:
- Breakthrough Indicators: {', '.join(insight.breakthrough_indicators)}
- Key Findings: {', '.join(insight.primary_findings[:2])}
- Confidence: {insight.confidence_level:.2f}""")
        
        # Emergent insights
        if rich_context.emergent_insights:
            prompt_sections.append("\nEMERGENT INSIGHTS:")
            for insight in rich_context.emergent_insights[:3]:
                prompt_sections.append(f"• {insight.insight_description}")
        
        # Revolutionary analogical connections
        revolutionary_analogies = [
            conn for conn in rich_context.analogical_connections
            if conn.breakthrough_potential > 0.7
        ]
        
        if revolutionary_analogies:
            prompt_sections.append("\nREVOLUTIONARY ANALOGICAL CONNECTIONS:")
            for conn in revolutionary_analogies[:2]:
                prompt_sections.append(f"""
• {conn.source_domain} ↔ {conn.target_domain} (strength: {conn.connection_strength:.2f})
  Breakthrough potential: {conn.breakthrough_potential:.2f}
  Insights: {', '.join(conn.insights_generated[:2])}""")
        
        # Supporting research context
        prompt_sections.append(f"\nSUPPORTING RESEARCH: {len(search_corpus)} papers analyzed")
        
        # Synthesis instructions
        prompt_sections.append(f"""
SYNTHESIS INSTRUCTIONS:
Generate a compelling breakthrough narrative that:
1. Highlights the revolutionary potential identified by the analysis
2. Explains HOW different reasoning engines contributed to breakthrough insights
3. Presents analogical connections as paradigm-shifting opportunities
4. Acknowledges breakthrough risks while emphasizing transformative potential
5. Uses research papers to support specific breakthrough insights
6. Maintains scientific rigor while conveying excitement about possibilities

RESPONSE STYLE: {synthesis_config.user_parameters.tone.value}
VERBOSITY: {synthesis_config.user_parameters.verbosity}
FOCUS: Revolutionary insights and paradigm-shifting potential""")
        
        return "\n".join(prompt_sections)
    
    async def _build_analogical_prompt(self,
                                     query: str,
                                     rich_context: RichReasoningContext,
                                     search_corpus: List[Any],
                                     synthesis_config: SynthesisConfiguration) -> str:
        """Build prompt for analogical exploration synthesis"""
        
        prompt_sections = []
        
        prompt_sections.append(f"""You are exploring analogical connections discovered through sophisticated reasoning analysis.

ORIGINAL QUERY: {query}

ANALOGICAL EXPLORATION CONTEXT:""")
        
        # Primary analogical connections
        if rich_context.analogical_connections:
            prompt_sections.append("\nDISCOVERED ANALOGICAL CONNECTIONS:")
            for i, conn in enumerate(rich_context.analogical_connections[:3], 1):
                prompt_sections.append(f"""
{i}. {conn.source_domain} ↔ {conn.target_domain}
   Connection Strength: {conn.connection_strength:.2f}
   Insights Generated: {', '.join(conn.insights_generated)}
   Confidence: {conn.confidence:.2f}""")
        
        # Cross-domain bridges
        if rich_context.cross_domain_bridges:
            prompt_sections.append("\nCROSS-DOMAIN BRIDGES:")
            for bridge in rich_context.cross_domain_bridges[:2]:
                prompt_sections.append(f"""
• Bridge Concept: {bridge.bridge_concept}
  Connected Domains: {', '.join(bridge.connected_domains)}
  Insights Enabled: {', '.join(bridge.insights_enabled)}""")
        
        # Analogical reasoning engine specifics
        analogical_insight = rich_context.engine_insights.get(EngineType.ANALOGICAL)
        if analogical_insight:
            prompt_sections.append(f"""
ANALOGICAL REASONING ENGINE ANALYSIS:
- Primary Findings: {', '.join(analogical_insight.primary_findings)}
- Confidence Level: {analogical_insight.confidence_level:.2f}
- Processing Quality: {analogical_insight.quality_metrics}""")
        
        # Synthesis instructions
        prompt_sections.append(f"""
SYNTHESIS INSTRUCTIONS:
Generate an analogical exploration that:
1. Uses analogical connections as the primary narrative structure
2. Explains how cross-domain insights illuminate the original question
3. Builds bridges between seemingly unrelated concepts
4. Shows how analogical thinking reveals hidden patterns
5. Maintains rigor while embracing creative connections
6. Guides readers through the analogical reasoning process

RESPONSE STYLE: {synthesis_config.user_parameters.tone.value}
FOCUS: Analogical connections and cross-domain insights""")
        
        return "\n".join(prompt_sections)
    
    async def _build_nuanced_prompt(self,
                                  query: str,
                                  rich_context: RichReasoningContext,
                                  search_corpus: List[Any],
                                  synthesis_config: SynthesisConfiguration) -> str:
        """Build prompt for nuanced analysis synthesis"""
        
        prompt_sections = []
        
        prompt_sections.append(f"""You are providing nuanced analysis based on comprehensive multi-engine reasoning.

ORIGINAL QUERY: {query}

NUANCED ANALYSIS CONTEXT:""")
        
        # Multi-engine insights with nuances
        prompt_sections.append("\nMULTI-ENGINE REASONING INSIGHTS:")
        for engine_type, insight in rich_context.engine_insights.items():
            prompt_sections.append(f"""
{engine_type.value.upper()} REASONING:
- Primary Insights: {', '.join(insight.primary_findings[:2])}
- Confidence: {insight.confidence_level:.2f} ({insight.confidence_category.value})
- Nuances: {len(insight.reasoning_trace)} reasoning steps analyzed""")
        
        # Synthesis patterns showing complexity
        if rich_context.synthesis_patterns:
            prompt_sections.append("\nCROSS-ENGINE SYNTHESIS PATTERNS:")
            for pattern in rich_context.synthesis_patterns[:2]:
                prompt_sections.append(f"""
• {pattern.pattern_type.value.title()} Pattern (strength: {pattern.strength:.2f})
  Engines: {', '.join([e.value for e in pattern.participating_engines])}
  Description: {pattern.synthesis_description}""")
        
        # Confidence and uncertainty analysis
        if rich_context.confidence_analysis:
            prompt_sections.append(f"""
CONFIDENCE ANALYSIS:
- Overall Confidence: {rich_context.confidence_analysis.overall_confidence:.2f}
- Confidence Factors: {', '.join(rich_context.confidence_analysis.confidence_factors[:3])}
- Uncertainty Sources: {', '.join(rich_context.confidence_analysis.uncertainty_sources[:3])}""")
        
        # Reasoning conflicts and tensions
        if rich_context.reasoning_conflicts:
            prompt_sections.append("\nREASONING TENSIONS:")
            for conflict in rich_context.reasoning_conflicts[:2]:
                prompt_sections.append(f"""
• Conflict: {conflict.conflict_description}
  Engines: {', '.join([e.value for e in conflict.conflicting_engines])}
  Severity: {conflict.conflict_severity:.2f}""")
        
        # Synthesis instructions
        prompt_sections.append(f"""
SYNTHESIS INSTRUCTIONS:
Generate a nuanced analysis that:
1. Acknowledges complexity and multiple perspectives
2. Explores tensions between different reasoning approaches
3. Presents confidence levels and uncertainties naturally
4. Shows how different engines contribute different insights
5. Avoids oversimplification while remaining accessible
6. Demonstrates intellectual honesty about limitations

RESPONSE STYLE: {synthesis_config.user_parameters.tone.value}
FOCUS: Nuanced understanding with acknowledged complexity""")
        
        return "\n".join(prompt_sections)
    
    async def _build_evidence_prompt(self,
                                   query: str,
                                   rich_context: RichReasoningContext,
                                   search_corpus: List[Any],
                                   synthesis_config: SynthesisConfiguration) -> str:
        """Build prompt for evidence synthesis"""
        
        prompt_sections = []
        
        prompt_sections.append(f"""You are synthesizing evidence from comprehensive research and reasoning analysis.

ORIGINAL QUERY: {query}

EVIDENCE SYNTHESIS CONTEXT:""")
        
        # Evidence from reasoning engines
        prompt_sections.append("\nREASONING-BASED EVIDENCE:")
        for engine_type, insight in rich_context.engine_insights.items():
            if insight.supporting_evidence:
                prompt_sections.append(f"""
{engine_type.value.upper()} EVIDENCE:
- Number of evidence pieces: {len(insight.supporting_evidence)}
- Evidence types: {', '.join(set(e.evidence_type for e in insight.supporting_evidence))}
- Average strength: {sum(e.strength for e in insight.supporting_evidence) / len(insight.supporting_evidence):.2f}""")
        
        # Research corpus evidence
        prompt_sections.append(f"\nRESEARCH CORPUS: {len(search_corpus)} papers analyzed for supporting evidence")
        
        # Quality assessment
        if rich_context.reasoning_quality:
            prompt_sections.append(f"""
EVIDENCE QUALITY ASSESSMENT:
- Evidence Integration Score: {rich_context.reasoning_quality.evidence_integration:.2f}
- Logical Consistency: {rich_context.reasoning_quality.logical_consistency:.2f}
- Completeness: {rich_context.reasoning_quality.completeness_score:.2f}""")
        
        # Synthesis instructions
        prompt_sections.append(f"""
SYNTHESIS INSTRUCTIONS:
Generate an evidence-based synthesis that:
1. Prioritizes empirical evidence and research findings
2. Shows how different types of evidence support conclusions
3. Acknowledges evidence limitations and gaps
4. Uses proper academic citations and attribution
5. Distinguishes between strong and weak evidence
6. Builds conclusions systematically from evidence

RESPONSE STYLE: {synthesis_config.user_parameters.tone.value}
CITATION STYLE: {synthesis_config.user_parameters.citation_style}
FOCUS: Evidence-based conclusions with proper attribution""")
        
        return "\n".join(prompt_sections)
    
    async def _build_uncertainty_prompt(self,
                                      query: str,
                                      rich_context: RichReasoningContext,
                                      search_corpus: List[Any],
                                      synthesis_config: SynthesisConfiguration) -> str:
        """Build prompt for uncertainty navigation synthesis"""
        
        prompt_sections = []
        
        prompt_sections.append(f"""You are navigating uncertainties revealed through comprehensive reasoning analysis.

ORIGINAL QUERY: {query}

UNCERTAINTY NAVIGATION CONTEXT:""")
        
        # Uncertainty mapping
        if rich_context.uncertainty_mapping:
            prompt_sections.append(f"""
UNCERTAINTY ANALYSIS:
- Epistemic Uncertainty: {rich_context.uncertainty_mapping.epistemic_uncertainty:.2f}
- Aleatoric Uncertainty: {rich_context.uncertainty_mapping.aleatoric_uncertainty:.2f}
- Model Uncertainty: {rich_context.uncertainty_mapping.model_uncertainty:.2f}
- Uncertainty Types: {', '.join(rich_context.uncertainty_mapping.uncertainty_types.keys())}""")
        
        # Knowledge gaps
        if rich_context.knowledge_gaps:
            prompt_sections.append("\nIDENTIFIED KNOWLEDGE GAPS:")
            for gap in rich_context.knowledge_gaps[:3]:
                prompt_sections.append(f"""
• {gap.gap_description}
  Impact: {gap.impact_severity:.2f}
  Priority: {gap.priority}""")
        
        # Confidence distribution across engines
        prompt_sections.append("\nCONFIDENCE DISTRIBUTION:")
        for engine_type, confidence in rich_context.engine_confidence_levels.items():
            prompt_sections.append(f"• {engine_type.value}: {confidence:.2f}")
        
        # Synthesis instructions
        prompt_sections.append(f"""
SYNTHESIS INSTRUCTIONS:
Generate an uncertainty-aware synthesis that:
1. Explicitly acknowledges what we don't know
2. Distinguishes between different types of uncertainty
3. Shows how uncertainty affects conclusions
4. Suggests ways to reduce uncertainty
5. Presents uncertainty as opportunity for future research
6. Maintains intellectual honesty about limitations

RESPONSE STYLE: {synthesis_config.user_parameters.tone.value}
UNCERTAINTY HANDLING: {synthesis_config.user_parameters.uncertainty_handling}
FOCUS: Transparent uncertainty navigation""")
        
        return "\n".join(prompt_sections)
    
    async def _build_convergence_prompt(self,
                                      query: str,
                                      rich_context: RichReasoningContext,
                                      search_corpus: List[Any],
                                      synthesis_config: SynthesisConfiguration) -> str:
        """Build prompt for convergence synthesis"""
        
        prompt_sections = []
        
        prompt_sections.append(f"""You are synthesizing convergent insights from multiple reasoning approaches.

ORIGINAL QUERY: {query}

CONVERGENCE SYNTHESIS CONTEXT:""")
        
        # Convergence points
        if rich_context.convergence_points:
            prompt_sections.append("\nCONVERGENCE POINTS:")
            for point in rich_context.convergence_points:
                prompt_sections.append(f"""
• Topic: {point.convergence_topic}
  Converging Engines: {', '.join([e.value for e in point.converging_engines])}
  Strength: {point.convergence_strength:.2f}
  Shared Conclusion: {point.shared_conclusion}""")
        
        # High-confidence findings
        high_confidence_insights = [
            (engine_type, insight) for engine_type, insight in rich_context.engine_insights.items()
            if insight.confidence_level > 0.7
        ]
        
        if high_confidence_insights:
            prompt_sections.append("\nHIGH-CONFIDENCE CONVERGENT INSIGHTS:")
            for engine_type, insight in high_confidence_insights:
                prompt_sections.append(f"""
{engine_type.value.upper()}: {', '.join(insight.primary_findings[:2])}
Confidence: {insight.confidence_level:.2f}""")
        
        # Synthesis instructions
        prompt_sections.append(f"""
SYNTHESIS INSTRUCTIONS:
Generate a convergence-focused synthesis that:
1. Emphasizes areas where multiple reasoning approaches agree
2. Shows the strength that comes from convergent validation
3. Builds confidence through cross-engine confirmation
4. Presents unified conclusions where appropriate
5. Acknowledges remaining areas of non-convergence
6. Uses convergence as a basis for strong recommendations

RESPONSE STYLE: {synthesis_config.user_parameters.tone.value}
FOCUS: Convergent insights and unified conclusions""")
        
        return "\n".join(prompt_sections)
    
    async def _build_conflict_prompt(self,
                                   query: str,
                                   rich_context: RichReasoningContext,
                                   search_corpus: List[Any],
                                   synthesis_config: SynthesisConfiguration) -> str:
        """Build prompt for conflict resolution synthesis"""
        
        prompt_sections = []
        
        prompt_sections.append(f"""You are resolving conflicts between different reasoning approaches.

ORIGINAL QUERY: {query}

CONFLICT RESOLUTION CONTEXT:""")
        
        # Reasoning conflicts
        if rich_context.reasoning_conflicts:
            prompt_sections.append("\nIDENTIFIED REASONING CONFLICTS:")
            for conflict in rich_context.reasoning_conflicts:
                prompt_sections.append(f"""
• Conflict: {conflict.conflict_description}
  Conflicting Engines: {', '.join([e.value for e in conflict.conflicting_engines])}
  Severity: {conflict.conflict_severity:.2f}
  Resolution Status: {conflict.resolution_status}""")
        
        # Engine disagreements
        low_agreement_patterns = [
            pattern for pattern in rich_context.synthesis_patterns
            if pattern.pattern_type.value in ['conflicting', 'divergent']
        ]
        
        if low_agreement_patterns:
            prompt_sections.append("\nDIVERGENT PATTERNS:")
            for pattern in low_agreement_patterns:
                prompt_sections.append(f"""
• {pattern.synthesis_description}
  Participating Engines: {', '.join([e.value for e in pattern.participating_engines])}
  Pattern Strength: {pattern.strength:.2f}""")
        
        # Synthesis instructions
        prompt_sections.append(f"""
SYNTHESIS INSTRUCTIONS:
Generate a conflict-resolving synthesis that:
1. Acknowledges disagreements between reasoning approaches
2. Explores why different engines reach different conclusions
3. Seeks higher-level synthesis that reconciles conflicts
4. Shows how apparent conflicts can reveal deeper insights
5. Provides balanced perspective on unresolved tensions
6. Uses conflicts as opportunities for nuanced understanding

RESPONSE STYLE: {synthesis_config.user_parameters.tone.value}
FOCUS: Constructive conflict resolution and synthesis""")
        
        return "\n".join(prompt_sections)
    
    async def _build_default_prompt(self,
                                  query: str,
                                  rich_context: RichReasoningContext,
                                  search_corpus: List[Any],
                                  synthesis_config: SynthesisConfiguration) -> str:
        """Build default prompt when no specific strategy matches"""
        
        return f"""You are synthesizing insights from comprehensive multi-modal reasoning analysis.

ORIGINAL QUERY: {query}

REASONING CONTEXT:
{self._format_basic_context(rich_context)}

SYNTHESIS INSTRUCTIONS:
Generate a comprehensive response that integrates the reasoning insights above.
Use a {synthesis_config.user_parameters.tone.value} tone with {synthesis_config.user_parameters.verbosity} verbosity.
"""
    
    def _format_basic_context(self, rich_context: RichReasoningContext) -> str:
        """Format basic context for default prompt"""
        
        context_parts = []
        
        # Engine insights
        if rich_context.engine_insights:
            context_parts.append("REASONING ENGINE INSIGHTS:")
            for engine_type, insight in rich_context.engine_insights.items():
                context_parts.append(f"• {engine_type.value}: {', '.join(insight.primary_findings[:2])}")
        
        # Analogical connections
        if rich_context.analogical_connections:
            context_parts.append("\nANALOGICAL CONNECTIONS:")
            for conn in rich_context.analogical_connections[:2]:
                context_parts.append(f"• {conn.source_domain} ↔ {conn.target_domain}")
        
        return "\n".join(context_parts)