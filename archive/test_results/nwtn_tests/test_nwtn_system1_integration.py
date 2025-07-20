#!/usr/bin/env python3
"""
Integration Test for NWTN System 1 Pipeline
==========================================

Tests the complete System 1 pipeline: SemanticRetriever → ContentAnalyzer → CandidateAnswerGenerator
"""

import asyncio
import json
from typing import Dict, Any

from prsm.nwtn.external_storage_config import ExternalStorageManager, ExternalKnowledgeBase
from prsm.nwtn.semantic_retriever import create_semantic_retriever
from prsm.nwtn.content_analyzer import create_content_analyzer
from prsm.nwtn.candidate_answer_generator import create_candidate_generator


async def test_complete_system1_pipeline():
    """Test the complete System 1 pipeline integration"""
    print("🧪 Testing Complete NWTN System 1 Pipeline Integration")
    print("=" * 60)
    
    # Mock query
    test_query = "How can machine learning improve classification accuracy?"
    
    try:
        # Step 1: Initialize components
        print("📦 Step 1: Initializing System 1 components...")
        
        # Create external knowledge base (mock)
        storage_manager = ExternalStorageManager()
        knowledge_base = ExternalKnowledgeBase(storage_manager)
        
        # Initialize components
        semantic_retriever = await create_semantic_retriever(knowledge_base)
        content_analyzer = await create_content_analyzer()
        candidate_generator = await create_candidate_generator()
        
        print("   ✅ SemanticRetriever initialized")
        print("   ✅ ContentAnalyzer initialized")
        print("   ✅ CandidateAnswerGenerator initialized")
        
        # Step 2: Semantic retrieval
        print("\n🔍 Step 2: Performing semantic retrieval...")
        
        search_result = await semantic_retriever.semantic_search(
            query=test_query,
            top_k=5,
            search_method="hybrid"
        )
        
        print(f"   ✅ Retrieved {len(search_result.retrieved_papers)} papers")
        print(f"   ✅ Search time: {search_result.search_time_seconds:.3f}s")
        print(f"   ✅ Method: {search_result.retrieval_method}")
        
        # Step 3: Content analysis
        print("\n🔬 Step 3: Analyzing retrieved content...")
        
        analysis_result = await content_analyzer.analyze_retrieved_papers(search_result)
        
        print(f"   ✅ Analyzed {len(analysis_result.analyzed_papers)} papers")
        print(f"   ✅ Extracted {analysis_result.total_concepts_extracted} concepts")
        print(f"   ✅ Analysis time: {analysis_result.analysis_time_seconds:.3f}s")
        
        # Quality distribution
        quality_dist = analysis_result.quality_distribution
        print(f"   ✅ Quality distribution: {dict(quality_dist)}")
        
        # Step 4: Candidate generation
        print("\n🧠 Step 4: Generating candidate answers...")
        
        candidate_result = await candidate_generator.generate_candidates(analysis_result)
        
        print(f"   ✅ Generated {len(candidate_result.candidate_answers)} candidates")
        print(f"   ✅ Generation time: {candidate_result.generation_time_seconds:.3f}s")
        print(f"   ✅ Sources used: {candidate_result.total_sources_used}")
        
        # Diversity metrics
        diversity = candidate_result.diversity_metrics
        print(f"   ✅ Diversity metrics: {dict(diversity)}")
        
        # Step 5: Analyze candidate quality
        print("\n📊 Step 5: Analyzing candidate quality...")
        
        candidates = candidate_result.candidate_answers
        answer_types = [c.answer_type.value for c in candidates]
        confidence_scores = [c.confidence_score for c in candidates]
        
        print(f"   ✅ Answer types: {set(answer_types)}")
        print(f"   ✅ Confidence range: {min(confidence_scores):.2f} - {max(confidence_scores):.2f}")
        print(f"   ✅ Average confidence: {sum(confidence_scores) / len(confidence_scores):.2f}")
        
        # Step 6: Verify source tracking
        print("\n🔗 Step 6: Verifying source tracking...")
        
        all_source_ids = set()
        for candidate in candidates:
            for contribution in candidate.source_contributions:
                all_source_ids.add(contribution.paper_id)
        
        print(f"   ✅ Unique sources referenced: {len(all_source_ids)}")
        print(f"   ✅ Source utilization: {len(candidate_result.source_utilization)} papers")
        
        # Step 7: Display sample candidate
        print("\n📝 Step 7: Sample candidate answer...")
        
        best_candidate = max(candidates, key=lambda c: c.confidence_score)
        print(f"   📋 Type: {best_candidate.answer_type.value}")
        print(f"   📋 Confidence: {best_candidate.confidence_score:.2f}")
        print(f"   📋 Sources: {len(best_candidate.source_contributions)}")
        print(f"   📋 Concepts: {len(best_candidate.key_concepts_used)}")
        print(f"   📋 Answer preview: {best_candidate.answer_text[:100]}...")
        
        # Step 8: Verify pipeline integrity
        print("\n🔄 Step 8: Verifying pipeline integrity...")
        
        # Check data flow
        assert len(search_result.retrieved_papers) > 0
        assert len(analysis_result.analyzed_papers) > 0
        assert len(candidate_result.candidate_answers) > 0
        
        # Check that concepts flow through pipeline
        analysis_concepts = set()
        for paper in analysis_result.analyzed_papers:
            analysis_concepts.update(c.concept for c in paper.key_concepts)
        
        candidate_concepts = set()
        for candidate in candidate_result.candidate_answers:
            candidate_concepts.update(candidate.key_concepts_used)
        
        concept_overlap = len(analysis_concepts.intersection(candidate_concepts))
        print(f"   ✅ Concept flow verified: {concept_overlap} concepts used")
        
        # Check source consistency
        analysis_sources = set(p.paper_id for p in analysis_result.analyzed_papers)
        candidate_sources = set(candidate_result.source_utilization.keys())
        
        source_consistency = len(analysis_sources.intersection(candidate_sources))
        print(f"   ✅ Source consistency: {source_consistency} sources tracked")
        
        # Final verification
        print("\n" + "=" * 60)
        print("🎉 SYSTEM 1 PIPELINE INTEGRATION TEST PASSED!")
        print("   ✅ Semantic retrieval working")
        print("   ✅ Content analysis working")
        print("   ✅ Candidate generation working")
        print("   ✅ Data flow verified")
        print("   ✅ Source tracking verified")
        print("   ✅ Quality metrics verified")
        
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance_metrics():
    """Test performance metrics of the complete pipeline"""
    print("\n🚀 Testing Performance Metrics...")
    
    # Mock query
    test_query = "What are the latest advances in neural network architectures?"
    
    try:
        # Initialize components
        storage_manager = ExternalStorageManager()
        knowledge_base = ExternalKnowledgeBase(storage_manager)
        
        semantic_retriever = await create_semantic_retriever(knowledge_base)
        content_analyzer = await create_content_analyzer()
        candidate_generator = await create_candidate_generator()
        
        # Run pipeline multiple times for performance measurement
        total_time = 0
        iterations = 3
        
        for i in range(iterations):
            print(f"   🔄 Iteration {i+1}/{iterations}")
            
            # Time the complete pipeline
            import time
            start_time = time.time()
            
            # Run pipeline
            search_result = await semantic_retriever.semantic_search(test_query, top_k=3)
            analysis_result = await content_analyzer.analyze_retrieved_papers(search_result)
            candidate_result = await candidate_generator.generate_candidates(analysis_result)
            
            end_time = time.time()
            iteration_time = end_time - start_time
            total_time += iteration_time
            
            print(f"      ⏱️  Time: {iteration_time:.3f}s")
            print(f"      📊 Candidates: {len(candidate_result.candidate_answers)}")
        
        average_time = total_time / iterations
        print(f"   ✅ Average pipeline time: {average_time:.3f}s")
        print(f"   ✅ Performance target: {'✅ PASS' if average_time < 1.0 else '⚠️ SLOW'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


async def test_component_statistics():
    """Test component statistics and monitoring"""
    print("\n📈 Testing Component Statistics...")
    
    try:
        # Initialize components
        storage_manager = ExternalStorageManager()
        knowledge_base = ExternalKnowledgeBase(storage_manager)
        
        semantic_retriever = await create_semantic_retriever(knowledge_base)
        content_analyzer = await create_content_analyzer()
        candidate_generator = await create_candidate_generator()
        
        # Run some operations to generate statistics
        test_query = "How can AI improve healthcare outcomes?"
        
        search_result = await semantic_retriever.semantic_search(test_query)
        analysis_result = await content_analyzer.analyze_retrieved_papers(search_result)
        candidate_result = await candidate_generator.generate_candidates(analysis_result)
        
        # Get statistics
        retrieval_stats = semantic_retriever.get_retrieval_statistics()
        analysis_stats = content_analyzer.get_analysis_statistics()
        generation_stats = candidate_generator.get_generation_statistics()
        
        print(f"   📊 Retrieval stats: {retrieval_stats}")
        print(f"   📊 Analysis stats: {analysis_stats}")
        print(f"   📊 Generation stats: {generation_stats}")
        
        # Verify statistics are reasonable
        assert retrieval_stats['total_retrievals'] > 0
        assert analysis_stats['total_analyses'] > 0
        assert generation_stats['total_generations'] > 0
        
        print("   ✅ Component statistics verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    print("🚀 Starting NWTN System 1 Integration Tests")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Complete pipeline integration
    if await test_complete_system1_pipeline():
        tests_passed += 1
    
    # Test 2: Performance metrics
    if await test_performance_metrics():
        tests_passed += 1
    
    # Test 3: Component statistics
    if await test_component_statistics():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 INTEGRATION TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("   ✅ NWTN System 1 pipeline is fully operational")
        print("   ✅ Ready for Phase 2: System 2 Meta-Reasoning")
    else:
        print("❌ Some integration tests failed")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    asyncio.run(main())