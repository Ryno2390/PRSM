#!/usr/bin/env python3
"""
System 1 → System 2 Integration Test
===================================

Tests the complete System 1 → System 2 pipeline:
SemanticRetriever → ContentAnalyzer → CandidateAnswerGenerator → CandidateEvaluator
"""

import asyncio
import json
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

from prsm.nwtn.external_storage_config import ExternalStorageManager, ExternalKnowledgeBase
from prsm.nwtn.semantic_retriever import create_semantic_retriever
from prsm.nwtn.content_analyzer import create_content_analyzer
from prsm.nwtn.candidate_answer_generator import create_candidate_generator
from prsm.nwtn.candidate_evaluator import create_candidate_evaluator
from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode


async def test_complete_system1_system2_pipeline():
    """Test the complete System 1 → System 2 pipeline"""
    print("🧪 Testing Complete System 1 → System 2 Pipeline Integration")
    print("=" * 70)
    
    # Test query
    test_query = "How can machine learning improve classification accuracy?"
    
    try:
        # Step 1: Initialize all components
        print("📦 Step 1: Initializing System 1 and System 2 components...")
        
        # System 1 components
        storage_manager = ExternalStorageManager()
        knowledge_base = ExternalKnowledgeBase(storage_manager)
        
        semantic_retriever = await create_semantic_retriever(knowledge_base)
        content_analyzer = await create_content_analyzer()
        candidate_generator = await create_candidate_generator()
        
        # System 2 components
        meta_reasoning_engine = MetaReasoningEngine()
        candidate_evaluator = await create_candidate_evaluator(meta_reasoning_engine)
        
        print("   ✅ System 1 components initialized")
        print("      - SemanticRetriever: Ready")
        print("      - ContentAnalyzer: Ready")
        print("      - CandidateAnswerGenerator: Ready")
        print("   ✅ System 2 components initialized")
        print("      - MetaReasoningEngine: Ready")
        print("      - CandidateEvaluator: Ready")
        
        # Step 2: System 1 - Semantic Retrieval
        print("\n🔍 Step 2: System 1 - Semantic Retrieval...")
        
        search_result = await semantic_retriever.semantic_search(
            query=test_query,
            top_k=5,
            search_method="hybrid"
        )
        
        print(f"   ✅ Retrieved {len(search_result.retrieved_papers)} papers")
        print(f"   ✅ Search time: {search_result.search_time_seconds:.3f}s")
        print(f"   ✅ Method: {search_result.retrieval_method}")
        
        # Step 3: System 1 - Content Analysis
        print("\n🔬 Step 3: System 1 - Content Analysis...")
        
        analysis_result = await content_analyzer.analyze_retrieved_papers(search_result)
        
        print(f"   ✅ Analyzed {len(analysis_result.analyzed_papers)} papers")
        print(f"   ✅ Extracted {analysis_result.total_concepts_extracted} concepts")
        print(f"   ✅ Analysis time: {analysis_result.analysis_time_seconds:.3f}s")
        
        # Step 4: System 1 - Candidate Generation (Brainstorming)
        print("\n🧠 Step 4: System 1 - Candidate Generation (Brainstorming)...")
        
        candidate_result = await candidate_generator.generate_candidates(analysis_result)
        
        print(f"   ✅ Generated {len(candidate_result.candidate_answers)} candidates")
        print(f"   ✅ Generation time: {candidate_result.generation_time_seconds:.3f}s")
        print(f"   ✅ Sources used: {candidate_result.total_sources_used}")
        print(f"   ✅ Diversity metrics: {candidate_result.diversity_metrics}")
        
        # Display candidates
        for i, candidate in enumerate(candidate_result.candidate_answers[:3]):  # Show first 3
            print(f"      Candidate {i+1}: {candidate.answer_type.value} (conf: {candidate.confidence_score:.2f})")
        
        # Step 5: System 2 - Methodical Evaluation
        print("\n🤔 Step 5: System 2 - Methodical Evaluation...")
        
        # Use different thinking modes for demonstration
        evaluation_result = await candidate_evaluator.evaluate_candidates(
            candidate_result,
            thinking_mode=ThinkingMode.INTERMEDIATE
        )
        
        print(f"   ✅ Evaluated {len(evaluation_result.candidate_evaluations)} candidates")
        print(f"   ✅ Evaluation time: {evaluation_result.evaluation_time_seconds:.3f}s")
        print(f"   ✅ Thinking mode: {evaluation_result.thinking_mode_used.value}")
        print(f"   ✅ Overall confidence: {evaluation_result.overall_confidence:.2f}")
        
        # Step 6: System 2 - Best Candidate Selection
        print("\n🏆 Step 6: System 2 - Best Candidate Selection...")
        
        if evaluation_result.best_candidate:
            best = evaluation_result.best_candidate
            print(f"   ✅ Best candidate: {best.candidate_answer.answer_type.value}")
            print(f"   ✅ Overall score: {best.overall_score:.2f}")
            print(f"   ✅ Ranking position: {best.ranking_position}")
            print(f"   ✅ Sources used: {len(best.candidate_answer.source_contributions)}")
            print(f"   ✅ Answer preview: {best.candidate_answer.answer_text[:100]}...")
        
        # Step 7: Source Lineage Verification
        print("\n🔗 Step 7: Source Lineage Verification...")
        
        # Verify source tracking through the pipeline
        retrieval_sources = set(p.paper_id for p in search_result.retrieved_papers)
        analysis_sources = set(p.paper_id for p in analysis_result.analyzed_papers)
        generation_sources = set(candidate_result.source_utilization.keys())
        evaluation_sources = set(evaluation_result.source_lineage.keys())
        
        print(f"   ✅ Retrieval sources: {len(retrieval_sources)}")
        print(f"   ✅ Analysis sources: {len(analysis_sources)}")
        print(f"   ✅ Generation sources: {len(generation_sources)}")
        print(f"   ✅ Evaluation sources: {len(evaluation_sources)}")
        
        # Verify source flow consistency
        analysis_consistent = len(retrieval_sources.intersection(analysis_sources)) > 0
        generation_consistent = len(analysis_sources.intersection(generation_sources)) > 0
        evaluation_consistent = len(generation_sources.intersection(evaluation_sources)) > 0
        
        print(f"   ✅ Source flow consistency:")
        print(f"      - Retrieval → Analysis: {'✅' if analysis_consistent else '❌'}")
        print(f"      - Analysis → Generation: {'✅' if generation_consistent else '❌'}")
        print(f"      - Generation → Evaluation: {'✅' if evaluation_consistent else '❌'}")
        
        # Step 8: Performance Metrics
        print("\n📊 Step 8: Performance Metrics...")
        
        total_pipeline_time = (
            search_result.search_time_seconds +
            analysis_result.analysis_time_seconds +
            candidate_result.generation_time_seconds +
            evaluation_result.evaluation_time_seconds
        )
        
        print(f"   ✅ Total pipeline time: {total_pipeline_time:.3f}s")
        print(f"      - System 1 time: {search_result.search_time_seconds + analysis_result.analysis_time_seconds + candidate_result.generation_time_seconds:.3f}s")
        print(f"      - System 2 time: {evaluation_result.evaluation_time_seconds:.3f}s")
        
        # Step 9: Quality Assessment
        print("\n🎯 Step 9: Quality Assessment...")
        
        quality_metrics = {
            "candidates_generated": len(candidate_result.candidate_answers),
            "candidates_evaluated": len(evaluation_result.candidate_evaluations),
            "best_candidate_score": evaluation_result.best_candidate.overall_score if evaluation_result.best_candidate else 0,
            "diversity_score": candidate_result.diversity_metrics.get("overall_diversity", 0),
            "source_coverage": len(evaluation_result.source_lineage) / max(1, len(search_result.retrieved_papers)),
            "evaluation_confidence": evaluation_result.overall_confidence
        }
        
        print(f"   ✅ Quality metrics:")
        for metric, value in quality_metrics.items():
            print(f"      - {metric}: {value:.2f}")
        
        # Step 10: Pipeline Validation
        print("\n✅ Step 10: Pipeline Validation...")
        
        # Validation criteria
        validations = {
            "Papers retrieved": len(search_result.retrieved_papers) > 0,
            "Content analyzed": len(analysis_result.analyzed_papers) > 0,
            "Candidates generated": len(candidate_result.candidate_answers) > 0,
            "Candidates evaluated": len(evaluation_result.candidate_evaluations) > 0,
            "Best candidate selected": evaluation_result.best_candidate is not None,
            "Source lineage tracked": len(evaluation_result.source_lineage) > 0,
            "Performance acceptable": total_pipeline_time < 10.0,  # Less than 10 seconds
            "Quality threshold met": quality_metrics["best_candidate_score"] > 0.3
        }
        
        all_passed = all(validations.values())
        
        print(f"   🔍 Validation results:")
        for criterion, passed in validations.items():
            print(f"      - {criterion}: {'✅' if passed else '❌'}")
        
        print("\n" + "=" * 70)
        if all_passed:
            print("🎉 SYSTEM 1 → SYSTEM 2 PIPELINE INTEGRATION TEST PASSED!")
            print("   ✅ System 1 (Fast brainstorming) working correctly")
            print("   ✅ System 2 (Methodical evaluation) working correctly")
            print("   ✅ Source lineage tracking working correctly")
            print("   ✅ Quality and performance metrics acceptable")
            print("   ✅ Ready for Phase 3: Attribution and Natural Language Generation")
        else:
            print("❌ Some pipeline validations failed")
            print("   Please review the failing criteria above")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_system2_thinking_modes():
    """Test System 2 evaluation with different thinking modes"""
    print("\n🧪 Testing System 2 Thinking Modes...")
    
    try:
        # Initialize components
        storage_manager = ExternalStorageManager()
        knowledge_base = ExternalKnowledgeBase(storage_manager)
        
        semantic_retriever = await create_semantic_retriever(knowledge_base)
        content_analyzer = await create_content_analyzer()
        candidate_generator = await create_candidate_generator()
        
        meta_reasoning_engine = MetaReasoningEngine()
        candidate_evaluator = await create_candidate_evaluator(meta_reasoning_engine)
        
        # Generate candidates
        test_query = "What are the latest advances in neural network architectures?"
        
        search_result = await semantic_retriever.semantic_search(test_query, top_k=3)
        analysis_result = await content_analyzer.analyze_retrieved_papers(search_result)
        candidate_result = await candidate_generator.generate_candidates(analysis_result)
        
        # Test different thinking modes
        thinking_modes = [ThinkingMode.QUICK, ThinkingMode.INTERMEDIATE, ThinkingMode.DEEP]
        
        for mode in thinking_modes:
            print(f"\n   🤔 Testing {mode.value} thinking mode...")
            
            evaluation_result = await candidate_evaluator.evaluate_candidates(
                candidate_result,
                thinking_mode=mode
            )
            
            print(f"      ✅ Candidates evaluated: {len(evaluation_result.candidate_evaluations)}")
            print(f"      ✅ Evaluation time: {evaluation_result.evaluation_time_seconds:.3f}s")
            print(f"      ✅ Overall confidence: {evaluation_result.overall_confidence:.2f}")
            
            if evaluation_result.best_candidate:
                print(f"      ✅ Best score: {evaluation_result.best_candidate.overall_score:.2f}")
        
        print("   ✅ All thinking modes working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Thinking mode test failed: {e}")
        return False


async def test_pipeline_scalability():
    """Test pipeline scalability with multiple queries"""
    print("\n🧪 Testing Pipeline Scalability...")
    
    try:
        # Initialize components
        storage_manager = ExternalStorageManager()
        knowledge_base = ExternalKnowledgeBase(storage_manager)
        
        semantic_retriever = await create_semantic_retriever(knowledge_base)
        content_analyzer = await create_content_analyzer()
        candidate_generator = await create_candidate_generator()
        
        meta_reasoning_engine = MetaReasoningEngine()
        candidate_evaluator = await create_candidate_evaluator(meta_reasoning_engine)
        
        # Test queries
        test_queries = [
            "How does machine learning improve healthcare outcomes?",
            "What are the environmental impacts of renewable energy?",
            "How can AI enhance cybersecurity measures?"
        ]
        
        total_time = 0
        successful_runs = 0
        
        for i, query in enumerate(test_queries):
            print(f"\n   🔄 Processing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            try:
                import time
                start_time = time.time()
                
                # Run pipeline
                search_result = await semantic_retriever.semantic_search(query, top_k=3)
                analysis_result = await content_analyzer.analyze_retrieved_papers(search_result)
                candidate_result = await candidate_generator.generate_candidates(analysis_result)
                evaluation_result = await candidate_evaluator.evaluate_candidates(candidate_result)
                
                end_time = time.time()
                query_time = end_time - start_time
                total_time += query_time
                successful_runs += 1
                
                print(f"      ✅ Completed in {query_time:.3f}s")
                print(f"      ✅ Candidates: {len(evaluation_result.candidate_evaluations)}")
                if evaluation_result.best_candidate:
                    print(f"      ✅ Best score: {evaluation_result.best_candidate.overall_score:.2f}")
                
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
                continue
        
        if successful_runs > 0:
            avg_time = total_time / successful_runs
            print(f"\n   ✅ Scalability test results:")
            print(f"      - Successful runs: {successful_runs}/{len(test_queries)}")
            print(f"      - Average time per query: {avg_time:.3f}s")
            print(f"      - Total time: {total_time:.3f}s")
            
            return successful_runs == len(test_queries)
        else:
            print("   ❌ No successful runs")
            return False
            
    except Exception as e:
        print(f"   ❌ Scalability test failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    print("🚀 Starting System 1 → System 2 Integration Tests")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Complete pipeline integration
    if await test_complete_system1_system2_pipeline():
        tests_passed += 1
    
    # Test 2: System 2 thinking modes
    if await test_system2_thinking_modes():
        tests_passed += 1
    
    # Test 3: Pipeline scalability
    if await test_pipeline_scalability():
        tests_passed += 1
    
    print("\n" + "=" * 70)
    print(f"🎯 INTEGRATION TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("   ✅ System 1 → System 2 pipeline is fully operational")
        print("   ✅ Ready for Phase 3: Attribution and Natural Language Generation")
        print("   ✅ Academic methodology: System 1 brainstorming + System 2 evaluation")
        print("   ✅ Source tracking: Complete lineage through pipeline")
        print("   ✅ Quality assurance: Methodical evaluation working")
    else:
        print("❌ Some integration tests failed")
        print("   Please review the failing tests above")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    asyncio.run(main())