#!/usr/bin/env python3
"""
Core Pipeline Test
==================

Tests just the semantic retrieval and content analysis stages.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prsm.nwtn.system_integrator import SystemIntegrator

async def test_core_pipeline():
    """Test the core pipeline stages"""
    print("🚀 Testing Core Pipeline Stages")
    print("=" * 40)
    
    # Create integrator with forced mock retriever
    integrator = SystemIntegrator(force_mock_retriever=True)
    
    # Initialize
    print("\n📋 Initializing...")
    await integrator.initialize()
    
    if not integrator.initialized:
        print("❌ Initialization failed")
        return False
    
    print("✅ Initialization successful")
    
    # Test Stage 1: Semantic Retrieval
    print("\n📋 Testing Semantic Retrieval...")
    query = "How can quantum error correction improve qubit stability?"
    
    try:
        retrieval_result = await integrator.semantic_retriever.semantic_search(query)
        print(f"✅ Semantic retrieval successful: {len(retrieval_result.retrieved_papers)} papers found")
        
        for i, paper in enumerate(retrieval_result.retrieved_papers[:2]):
            print(f"   Paper {i+1}: {paper.title[:50]}... (score: {paper.relevance_score:.2f})")
        
        # Test Stage 2: Content Analysis
        print("\n📋 Testing Content Analysis...")
        analysis_result = await integrator.content_analyzer.analyze_retrieved_papers(retrieval_result)
        print(f"✅ Content analysis successful: {len(analysis_result.analyzed_papers)} papers analyzed")
        
        for i, paper in enumerate(analysis_result.analyzed_papers[:2]):
            print(f"   Paper {i+1}: {paper.quality_level.value} quality, {len(paper.key_concepts)} concepts")
        
        # Test Stage 3: Candidate Generation
        print("\n📋 Testing Candidate Generation...")
        candidate_result = await integrator.candidate_generator.generate_candidates(analysis_result)
        print(f"✅ Candidate generation: {len(candidate_result.candidate_answers)} candidates generated")
        
        for i, candidate in enumerate(candidate_result.candidate_answers[:2]):
            print(f"   Candidate {i+1}: {candidate.answer_type.value} (confidence: {candidate.confidence_score:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        return False

async def main():
    """Run the core pipeline test"""
    success = await test_core_pipeline()
    
    if success:
        print("\n🎉 Core Pipeline Test PASSED!")
        print("   System 1 components are working correctly")
    else:
        print("\n❌ Core Pipeline Test FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())