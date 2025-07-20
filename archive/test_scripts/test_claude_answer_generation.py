#!/usr/bin/env python3
"""
Test Claude API Integration for Answer Generation
==============================================

Demonstrates the new Claude API-powered answer generation with user-configurable
verbosity levels from brief summaries to academic paper length responses.
"""

import sys
import asyncio
sys.path.insert(0, '.')

# Import all necessary classes
from prsm.nwtn.data_models import PaperEmbedding, PaperData, SemanticSearchResult
from prsm.nwtn.candidate_answer_generator import CandidateAnswerGenerator, ConceptSynthesizer, CandidateType, AnswerVerbosity
from prsm.nwtn.content_analyzer import ContentAnalyzer, ContentSummary, ExtractedConcept, ContentQuality

async def test_claude_integration():
    """Test Claude API integration with different verbosity levels"""
    print("🚀 Testing Claude API Integration for Answer Generation")
    print("=" * 60)
    
    query = "What are the latest advances in machine learning algorithms for natural language processing?"
    print(f"📝 Query: {query}")
    print()
    
    # Create mock content analysis with realistic concepts (not just "finding")
    mock_concepts = [
        ExtractedConcept(
            concept="transformer architectures", 
            category="methodology", 
            confidence=0.9, 
            context="using transformer architectures for NLP",
            paper_id="paper_1"
        ),
        ExtractedConcept(
            concept="attention mechanisms", 
            category="methodology", 
            confidence=0.85, 
            context="attention mechanisms improve performance",
            paper_id="paper_2" 
        ),
        ExtractedConcept(
            concept="significant performance improvements in language understanding", 
            category="finding", 
            confidence=0.8, 
            context="results show significant performance improvements",
            paper_id="paper_3"
        ),
        ExtractedConcept(
            concept="breakthrough results in text generation", 
            category="finding", 
            confidence=0.75, 
            context="demonstrated breakthrough results",
            paper_id="paper_4"
        ),
        ExtractedConcept(
            concept="conversational AI systems", 
            category="application", 
            confidence=0.7, 
            context="applied to conversational AI systems",
            paper_id="paper_5"
        )
    ]
    
    # Create mock paper summaries with realistic content
    mock_papers = [
        ContentSummary(
            paper_id="paper_1",
            title="Advanced Transformer Networks for Natural Language Processing",
            quality_score=0.9,
            quality_level=ContentQuality.EXCELLENT,
            key_concepts=mock_concepts[:2],
            main_contributions=["novel transformer architecture design", "improved training efficiency"],
            methodologies=["transformer networks", "attention mechanisms"],
            findings=["significant performance improvements in language understanding"],
            applications=["conversational AI systems", "text generation"],
            limitations=["computational complexity", "data requirements"]
        ),
        ContentSummary(
            paper_id="paper_2", 
            title="Breakthrough Attention Mechanisms in Deep Learning",
            quality_score=0.85,
            quality_level=ContentQuality.EXCELLENT, 
            key_concepts=mock_concepts[2:4],
            main_contributions=["novel attention mechanism", "state-of-the-art performance"],
            methodologies=["deep learning", "attention mechanisms"],
            findings=["breakthrough results in text generation"],
            applications=["language modeling", "machine translation"],
            limitations=["training complexity"]
        ),
        ContentSummary(
            paper_id="paper_3",
            title="Modern Applications of AI in Conversational Systems", 
            quality_score=0.8,
            quality_level=ContentQuality.GOOD,
            key_concepts=[mock_concepts[4]],
            main_contributions=["practical AI implementation", "system optimization"],
            methodologies=["neural networks", "optimization techniques"],
            findings=["improved user experience", "enhanced system performance"],
            applications=["conversational AI systems", "chatbots"],
            limitations=["deployment challenges"]
        )
    ]
    
    # Initialize the generator
    generator = CandidateAnswerGenerator()
    await generator.initialize()
    
    # Check Claude API availability
    if generator.claude_api_key:
        print(f"✅ Claude API Key found: {generator.claude_api_key[:10]}...")
        print(f"🤖 Using Claude Model: {generator.claude_model}")
    else:
        print("⚠️  No Claude API key found - will use template fallback")
        print("💡 Set ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable for Claude integration")
    print()
    
    # Test different verbosity levels
    verbosity_levels = [
        (AnswerVerbosity.BRIEF, "Brief Summary"),
        (AnswerVerbosity.STANDARD, "Standard Response"), 
        (AnswerVerbosity.DETAILED, "Detailed Analysis"),
        (AnswerVerbosity.COMPREHENSIVE, "Comprehensive Report"),
        (AnswerVerbosity.ACADEMIC, "Academic Paper Style")
    ]
    
    for verbosity, description in verbosity_levels:
        print(f"🎯 Testing {description} ({verbosity.value.upper()})")
        print("-" * 50)
        
        try:
            # Set verbosity for this test
            await generator.set_verbosity(verbosity)
            
            # Generate a candidate answer using Claude API with specific verbosity
            candidate = await generator._generate_single_candidate(
                query=query,
                papers=mock_papers,
                answer_type=CandidateType.SYNTHESIS,
                candidate_index=0
            )
            
            if candidate:
                print(f"✅ Generated {description}:")
                print(f"   Type: {candidate.answer_type.value}")
                print(f"   Confidence: {candidate.confidence_score:.2f}")
                print(f"   Sources Used: {len(candidate.source_contributions)}")
                print()
                print(f"📝 Answer Text ({len(candidate.answer_text)} characters):")
                print(f"   {candidate.answer_text}")
                print()
                print(f"🔗 Source Citations:")
                for i, source in enumerate(candidate.source_contributions, 1):
                    print(f"   {i}. {source.title} (weight: {source.contribution_weight:.2f})")
                print()
                
                # Estimate word count and reading time
                word_count = len(candidate.answer_text.split())
                reading_time = max(1, word_count // 200)  # ~200 words per minute
                print(f"📊 Stats: {word_count} words, ~{reading_time} min reading time")
                
            else:
                print(f"❌ Failed to generate {description}")
                
        except Exception as e:
            print(f"❌ Error generating {description}: {e}")
        
        print()
        print("=" * 60)
        print()

if __name__ == "__main__":
    success = asyncio.run(test_claude_integration())
    print("🎉 Claude API Integration Test Complete!")