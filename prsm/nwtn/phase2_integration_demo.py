"""
Phase 2 Integration Demo for NWTN Contextual Synthesizer.

This module demonstrates the complete Phase 2 integration, showing how
the ContextualSynthesizer, AdaptiveSynthesisStrategist, and ResponseEnhancementEngine
work together to transform NWTN's outputs from stilted to sophisticated.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Phase 1 imports
from .reasoning_context_types import RichReasoningContext, EngineType, EngineInsight
from .enhanced_meta_reasoning_engine import EnhancedMetaReasoningEngine

# Phase 2 imports
from .contextual_synthesizer import ContextualSynthesizer, UserParameters, ResponseTone
from .adaptive_synthesis_strategist import AdaptiveSynthesisStrategist
from .response_enhancement_engine import ResponseEnhancementEngine


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase2IntegrationDemo:
    """
    Demonstrates complete Phase 2 integration and shows the transformation
    from stilted generic outputs to sophisticated contextual synthesis.
    """
    
    def __init__(self):
        """Initialize the Phase 2 integration demo"""
        
        # Initialize Phase 1 components
        self.enhanced_meta_engine = EnhancedMetaReasoningEngine()
        
        # Initialize Phase 2 components
        self.contextual_synthesizer = ContextualSynthesizer()
        self.synthesis_strategist = AdaptiveSynthesisStrategist()
        self.response_enhancer = ResponseEnhancementEngine()
        
        logger.info("Phase2IntegrationDemo initialized with all components")
    
    async def demonstrate_transformation(self, 
                                       demo_query: str,
                                       user_parameters: Optional[UserParameters] = None) -> Dict[str, Any]:
        """
        Demonstrate the complete transformation from reasoning to sophisticated synthesis.
        
        Args:
            demo_query: Query to demonstrate with
            user_parameters: Optional user parameters for customization
            
        Returns:
            Dictionary containing both old-style and new-style outputs for comparison
        """
        
        logger.info(f"Starting Phase 2 transformation demo for query: {demo_query[:100]}...")
        
        # Use default user parameters if none provided
        if user_parameters is None:
            user_parameters = UserParameters(
                verbosity="COMPREHENSIVE",
                reasoning_mode="CREATIVE",
                depth="DEEP",
                tone=ResponseTone.CONVERSATIONAL
            )
        
        try:
            # Step 1: Generate enhanced reasoning with rich context (Phase 1)
            logger.info("Step 1: Generating enhanced reasoning with rich context")
            enhanced_reasoning = await self.enhanced_meta_engine.process_with_rich_context(
                demo_query,
                search_corpus=self._create_mock_search_corpus(),
                user_parameters=user_parameters.__dict__
            )
            
            # Step 2: Generate old-style generic output for comparison
            logger.info("Step 2: Generating old-style generic output")
            old_style_output = await self._generate_old_style_output(enhanced_reasoning.rich_context)
            
            # Step 3: Generate new sophisticated synthesis (Phase 2)
            logger.info("Step 3: Generating sophisticated contextual synthesis")
            sophisticated_response = await self.contextual_synthesizer.synthesize_response(
                demo_query,
                enhanced_reasoning.rich_context,
                self._create_mock_search_corpus(),
                user_parameters
            )
            
            # Step 4: Analyze the transformation
            transformation_analysis = await self._analyze_transformation(
                old_style_output, sophisticated_response, enhanced_reasoning.rich_context
            )
            
            demo_results = {
                'query': demo_query,
                'user_parameters': user_parameters.__dict__,
                'rich_context_summary': self._summarize_rich_context(enhanced_reasoning.rich_context),
                'old_style_output': old_style_output,
                'sophisticated_response': sophisticated_response.__dict__,
                'transformation_analysis': transformation_analysis,
                'context_validation': enhanced_reasoning.context_validation.__dict__,
                'processing_metadata': enhanced_reasoning.processing_metadata
            }
            
            logger.info("Phase 2 transformation demo completed successfully")
            return demo_results
            
        except Exception as e:
            logger.error(f"Error in Phase 2 demo: {e}")
            return {
                'error': str(e),
                'query': demo_query,
                'status': 'failed'
            }
    
    async def demonstrate_synthesis_strategies(self, demo_query: str) -> Dict[str, Any]:
        """Demonstrate different synthesis strategies for the same query"""
        
        logger.info("Demonstrating multiple synthesis strategies")
        
        # Create rich context
        enhanced_reasoning = await self.enhanced_meta_engine.process_with_rich_context(
            demo_query,
            search_corpus=self._create_mock_search_corpus()
        )
        
        strategy_demos = {}
        
        # Test different reasoning modes
        modes = ["CONSERVATIVE", "BALANCED", "CREATIVE", "REVOLUTIONARY"]
        
        for mode in modes:
            logger.info(f"Testing synthesis with {mode} mode")
            
            user_params = UserParameters(
                reasoning_mode=mode,
                verbosity="MODERATE",
                depth="INTERMEDIATE"
            )
            
            try:
                response = await self.contextual_synthesizer.synthesize_response(
                    demo_query,
                    enhanced_reasoning.rich_context,
                    self._create_mock_search_corpus(),
                    user_params
                )
                
                strategy_demos[mode] = {
                    'response_text': response.enhanced_text,
                    'strategy_used': response.synthesis_metadata.get('strategy_used'),
                    'citations_count': len(response.citations),
                    'breakthrough_highlights': response.breakthrough_highlights,
                    'analogical_insights': response.analogical_insights
                }
                
            except Exception as e:
                strategy_demos[mode] = {'error': str(e)}
        
        return {
            'query': demo_query,
            'rich_context_summary': self._summarize_rich_context(enhanced_reasoning.rich_context),
            'strategy_demonstrations': strategy_demos
        }
    
    # ========================================================================
    # Helper Methods for Demo
    # ========================================================================
    
    def _create_mock_search_corpus(self) -> List[Any]:
        """Create mock search corpus for demonstration"""
        
        class MockPaper:
            def __init__(self, title: str, authors: List[str], year: int):
                self.title = title
                self.authors = authors
                self.year = year
        
        return [
            MockPaper("Advanced Reasoning in AI Systems", ["Smith, J.", "Jones, A."], 2023),
            MockPaper("Cross-Domain Knowledge Transfer", ["Brown, L."], 2024),
            MockPaper("Analogical Reasoning and Creativity", ["Davis, M.", "Wilson, K."], 2023),
            MockPaper("Uncertainty Quantification in ML", ["Taylor, R."], 2024),
            MockPaper("Breakthrough Detection in Research", ["Garcia, S."], 2023)
        ]
    
    async def _generate_old_style_output(self, rich_context: RichReasoningContext) -> str:
        """Generate old-style generic output for comparison"""
        
        # Simulate the current stilted, anodyne output style
        old_style = f"""Based on the research analysis, here is a comprehensive response to your query about {rich_context.original_query}.

The analysis has been completed using multiple reasoning approaches. The system has processed the available information and generated insights across different analytical frameworks.

Key findings include:
- Analysis has been conducted across multiple domains
- Various reasoning methods have been applied
- Research literature has been consulted
- Conclusions have been drawn based on available evidence

The results suggest that further investigation may be warranted. Additional research could provide more comprehensive insights into the topic of interest.

References: Multiple research papers were consulted during this analysis."""
        
        return old_style
    
    async def _analyze_transformation(self,
                                    old_output: str,
                                    new_response: Any,
                                    rich_context: RichReasoningContext) -> Dict[str, Any]:
        """Analyze the transformation from old to new output"""
        
        # Calculate transformation metrics
        analysis = {
            'length_comparison': {
                'old_length': len(old_output),
                'new_length': len(new_response.enhanced_text),
                'length_ratio': len(new_response.enhanced_text) / len(old_output) if len(old_output) > 0 else 0
            },
            'content_richness': {
                'old_specific_terms': self._count_specific_terms(old_output),
                'new_specific_terms': self._count_specific_terms(new_response.enhanced_text),
                'improvement_factor': 0  # Would be calculated
            },
            'contextual_elements': {
                'citations_added': len(new_response.citations),
                'confidence_indicators': len(new_response.confidence_indicators),
                'breakthrough_highlights': len(new_response.breakthrough_highlights),
                'analogical_insights': len(new_response.analogical_insights),
                'uncertainty_acknowledgments': len(new_response.uncertainty_acknowledgments)
            },
            'engagement_factors': {
                'reasoning_transparency': 'reasoning_transparency' in new_response.__dict__,
                'follow_up_suggestions': len(new_response.follow_up_suggestions),
                'strategy_used': new_response.synthesis_metadata.get('strategy_used', 'unknown')
            },
            'context_utilization': {
                'engines_utilized': len(rich_context.engine_insights),
                'analogical_connections': len(rich_context.analogical_connections),
                'synthesis_patterns': len(rich_context.synthesis_patterns),
                'emergent_insights': len(rich_context.emergent_insights)
            }
        }
        
        # Calculate improvement score
        improvement_score = self._calculate_improvement_score(analysis)
        analysis['overall_improvement_score'] = improvement_score
        
        return analysis
    
    def _count_specific_terms(self, text: str) -> int:
        """Count specific domain terms vs generic phrases"""
        
        generic_phrases = [
            "based on the analysis", "comprehensive response", "various methods",
            "research suggests", "findings include", "further investigation"
        ]
        
        specific_indicators = [
            "deductive reasoning", "analogical connection", "breakthrough potential",
            "confidence level", "cross-domain", "synthesis pattern", "emergent insight"
        ]
        
        text_lower = text.lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in text_lower)
        specific_count = sum(1 for indicator in specific_indicators if indicator in text_lower)
        
        return specific_count - generic_count  # Positive = more specific, negative = more generic
    
    def _calculate_improvement_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall improvement score"""
        
        score = 0.0
        
        # Length improvement (diminishing returns)
        length_ratio = analysis['length_comparison']['length_ratio']
        if length_ratio > 1.5:
            score += 0.2
        
        # Content richness
        content_improvement = analysis['content_richness']['new_specific_terms']
        score += min(0.3, content_improvement * 0.05)
        
        # Contextual elements
        contextual_score = (
            analysis['contextual_elements']['citations_added'] * 0.05 +
            analysis['contextual_elements']['breakthrough_highlights'] * 0.1 +
            analysis['contextual_elements']['analogical_insights'] * 0.1
        )
        score += min(0.3, contextual_score)
        
        # Context utilization
        utilization_score = (
            analysis['context_utilization']['engines_utilized'] * 0.02 +
            analysis['context_utilization']['analogical_connections'] * 0.05 +
            analysis['context_utilization']['emergent_insights'] * 0.1
        )
        score += min(0.2, utilization_score)
        
        return min(1.0, score)
    
    def _summarize_rich_context(self, rich_context: RichReasoningContext) -> Dict[str, Any]:
        """Summarize rich context for demo output"""
        
        return {
            'engines_utilized': [engine.value for engine in rich_context.engine_insights.keys()],
            'analogical_connections': len(rich_context.analogical_connections),
            'synthesis_patterns': len(rich_context.synthesis_patterns),
            'emergent_insights': len(rich_context.emergent_insights),
            'breakthrough_score': rich_context.breakthrough_analysis.overall_breakthrough_score if rich_context.breakthrough_analysis else 0.0,
            'confidence_distribution': {k.value: v for k, v in rich_context.engine_confidence_levels.items()},
            'processing_summary': rich_context.get_processing_summary()
        }


# ============================================================================
# Demo Execution Functions
# ============================================================================

async def run_phase2_demo():
    """Run the complete Phase 2 demonstration"""
    
    print("🚀 Starting NWTN Phase 2 Integration Demo")
    print("=" * 60)
    
    demo = Phase2IntegrationDemo()
    
    # Demo query
    demo_query = "What are the implications of analogical reasoning for breakthrough innovations in artificial intelligence?"
    
    print(f"Demo Query: {demo_query}")
    print("\n" + "=" * 60)
    
    # Run transformation demo
    print("Running transformation demonstration...")
    transformation_results = await demo.demonstrate_transformation(demo_query)
    
    if 'error' not in transformation_results:
        print("\n📊 TRANSFORMATION ANALYSIS:")
        print("-" * 40)
        
        analysis = transformation_results['transformation_analysis']
        print(f"Overall Improvement Score: {analysis['overall_improvement_score']:.2f}")
        print(f"Length Ratio (New/Old): {analysis['length_comparison']['length_ratio']:.2f}")
        print(f"Citations Added: {analysis['contextual_elements']['citations_added']}")
        print(f"Breakthrough Highlights: {analysis['contextual_elements']['breakthrough_highlights']}")
        print(f"Analogical Insights: {analysis['contextual_elements']['analogical_insights']}")
        
        print("\n📝 OLD STYLE OUTPUT:")
        print("-" * 40)
        print(transformation_results['old_style_output'][:300] + "...")
        
        print("\n✨ NEW SOPHISTICATED OUTPUT:")
        print("-" * 40)
        print(transformation_results['sophisticated_response']['enhanced_text'][:500] + "...")
        
        print("\n🔍 FOLLOW-UP SUGGESTIONS:")
        print("-" * 40)
        for suggestion in transformation_results['sophisticated_response']['follow_up_suggestions']:
            print(f"• {suggestion}")
    
    print("\n" + "=" * 60)
    
    # Run strategy demonstration
    print("Running synthesis strategy demonstration...")
    strategy_results = await demo.demonstrate_synthesis_strategies(demo_query)
    
    print("\n🎯 SYNTHESIS STRATEGY COMPARISON:")
    print("-" * 40)
    
    for mode, result in strategy_results['strategy_demonstrations'].items():
        if 'error' not in result:
            print(f"\n{mode} MODE:")
            print(f"Strategy Used: {result.get('strategy_used', 'Unknown')}")
            print(f"Response Preview: {result['response_text'][:150]}...")
        else:
            print(f"\n{mode} MODE: Error - {result['error']}")
    
    print("\n🎉 Phase 2 Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    """Run the demo when executed directly"""
    asyncio.run(run_phase2_demo())