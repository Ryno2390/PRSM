#!/usr/bin/env python3
"""
Test New Answer Generation with All Fixes Applied
=================================================

Generate a sample answer with our embedding and quality fixes to show the improvement.
"""

import sys
import asyncio
sys.path.insert(0, '.')

# Import all necessary classes
from prsm.nwtn.data_models import PaperEmbedding, PaperData, SemanticSearchResult
from prsm.nwtn.candidate_answer_generator import CandidateAnswerGenerator, ConceptSynthesizer, CandidateType
from prsm.nwtn.content_analyzer import ContentAnalyzer, ContentSummary, ExtractedConcept, ContentQuality

async def generate_improved_answer():
    """Generate a sample answer with all fixes applied"""
    print("🚀 Generating New Answer with All Fixes Applied")
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
    
    print("🔍 Generating candidate answer with SYNTHESIS type (multiple sources)...")
    
    # Generate a candidate answer using our improved synthesis
    candidate = await generator._generate_single_candidate(
        query=query,
        papers=mock_papers,
        answer_type=CandidateType.SYNTHESIS,
        candidate_index=0
    )
    
    if candidate:
        print(f"✅ Generated Answer:")
        print(f"   Type: {candidate.answer_type.value}")
        print(f"   Confidence: {candidate.confidence_score:.2f}")
        print(f"   Sources Used: {len(candidate.source_contributions)}")
        print()
        print(f"📝 Final Answer Text:")
        print(f"   {candidate.answer_text}")
        print()
        print(f"🔗 Source Citations:")
        for i, source in enumerate(candidate.source_contributions, 1):
            print(f"   {i}. {source.title} (weight: {source.contribution_weight:.2f})")
        print()
        print(f"🧠 Reasoning Chain:")
        for i, reason in enumerate(candidate.reasoning_chain, 1):
            print(f"   {i}. {reason}")
        print()
        print(f"💡 Key Concepts Used:")
        print(f"   {', '.join(candidate.key_concepts_used[:5])}")
        
        # Compare with old problematic patterns
        print(f"\n🔍 Quality Check:")
        if "finding and finding" in candidate.answer_text.lower():
            print(f"   ❌ Still contains 'finding and finding'")
        else:
            print(f"   ✅ No 'finding and finding' placeholders")
            
        if candidate.answer_text.count("finding") > 2:
            print(f"   ⚠️  Multiple generic 'finding' references")
        else:
            print(f"   ✅ Proper concept usage")
            
        # Check source diversity
        unique_titles = set(source.title for source in candidate.source_contributions)
        if len(unique_titles) > 1:
            print(f"   ✅ Diverse sources: {len(unique_titles)} unique papers")
        else:
            print(f"   ⚠️  Limited source diversity")
            
        return True
    else:
        print(f"❌ Failed to generate candidate")
        return False

if __name__ == "__main__":
    success = asyncio.run(generate_improved_answer())
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILED'}: Answer generation with all fixes applied!")