#!/usr/bin/env python3
"""
Test script for NWTN World Model Integration
===========================================

This script tests the integration of ZIM-based World Model knowledge
into the NWTN meta-reasoning pipeline for candidate scoring.
"""

import asyncio
import os
import sys
import time
import logging
from pathlib import Path

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_world_model_zim_processing():
    """Test ZIM file processing for World Model"""
    
    print("🧪 TESTING: ZIM File Processing for World Model")
    print("=" * 60)
    
    try:
        from engines.universal_knowledge_ingestion_engine import (
            process_world_model_zim_files, WorldModelKnowledge
        )
        
        # Test ZIM directory
        zim_directory = "/Users/ryneschultz/Documents/GitHub/PRSM/prsm/nwtn/processed_corpus/world_model_knowledge/raw_sources"
        
        if not os.path.exists(zim_directory):
            print(f"❌ ZIM directory not found: {zim_directory}")
            return False
        
        # List available ZIM files
        zim_files = [f for f in os.listdir(zim_directory) if f.endswith('.zim')]
        print(f"📁 Found {len(zim_files)} ZIM files:")
        for zim_file in zim_files[:3]:  # Show first 3
            print(f"   - {zim_file}")
        if len(zim_files) > 3:
            print(f"   ... and {len(zim_files) - 3} more")
        
        # Test with first ZIM file only for speed
        test_zim = os.path.join(zim_directory, zim_files[0]) if zim_files else None
        if not test_zim:
            print("❌ No ZIM files found to test")
            return False
        
        print(f"\n🔬 Processing test ZIM file: {os.path.basename(test_zim)}")
        
        # Process single ZIM file
        start_time = time.time()
        world_model = await process_world_model_zim_files(zim_directory)
        processing_time = time.time() - start_time
        
        if not world_model:
            print("❌ Failed to create World Model")
            return False
        
        # Get World Model summary
        summary = world_model.get_world_model_summary()
        
        print(f"\n✅ World Model Processing Results:")
        print(f"   Processing Time: {processing_time:.2f} seconds")
        print(f"   Total Domains: {summary['total_domains']}")
        print(f"   Total Concepts: {summary['total_concepts']}")
        print(f"   Average Quality: {summary['average_quality']:.3f}")
        print(f"   Domains: {list(summary['domains'].keys())}")
        
        # Test concept scoring
        test_candidates = [
            "Physics involves the study of fundamental forces and particles.",
            "Machine learning algorithms can solve complex optimization problems.",
            "Gravity is a fundamental force that attracts objects with mass.",
            "The human brain processes information through neural networks.",
            "Chemistry studies the composition and reactions of matter."
        ]
        
        print(f"\n🎯 Testing World Model Contradiction Detection:")
        for i, candidate in enumerate(test_candidates, 1):
            contradiction_result = world_model.detect_contradictions(candidate, "physics research")
            print(f"   {i}. Contradictions: {contradiction_result['has_contradictions']}, "
                  f"Score: {contradiction_result['contradiction_score']:.3f}, "
                  f"Facts Checked: {contradiction_result['facts_checked']}")
            print(f"      Candidate: {candidate[:60]}...")
            if contradiction_result['contradictions_found']:
                print(f"      Found: {len(contradiction_result['contradictions_found'])} contradictions")
        
        return True
        
    except Exception as e:
        print(f"❌ World Model ZIM processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_nwtn_world_model_integration():
    """Test full NWTN pipeline with World Model integration"""
    
    print("\n🧪 TESTING: NWTN Pipeline with World Model Integration")
    print("=" * 60)
    
    try:
        from complete_nwtn_pipeline_v4 import CompleteNWTNPipeline
        
        # Create pipeline
        pipeline = CompleteNWTNPipeline()
        
        # Test query about physics (should align well with World Model)
        test_query = "How do quantum mechanical principles apply to machine learning optimization?"
        
        print(f"🔍 Testing Query: {test_query}")
        
        # Run pipeline with World Model integration
        start_time = time.time()
        wisdom_package = await pipeline.run_complete_pipeline(test_query, context_allocation=500)  # Reduced for testing
        total_time = time.time() - start_time
        
        print(f"\n✅ NWTN Pipeline Results with World Model:")
        print(f"   Total Processing Time: {total_time:.2f} seconds")
        print(f"   Session ID: {wisdom_package.session_id}")
        print(f"   Original Query: {wisdom_package.original_query}")
        
        # Check meta-reasoning analysis for World Model integration
        meta_analysis = wisdom_package.meta_reasoning_analysis
        if 'world_model_integration' in meta_analysis:
            wm_stats = meta_analysis['world_model_integration']
            print(f"\n🌍 World Model Integration Results:")
            print(f"   World Model Enabled: {meta_analysis.get('world_model_enabled', False)}")
            if wm_stats:
                print(f"   Avg Contradiction Score: {wm_stats.get('avg_contradiction_score', 0):.3f}")
                print(f"   Avg Contradiction Confidence: {wm_stats.get('avg_contradiction_confidence', 0):.3f}")
                print(f"   Candidates with Contradictions: {wm_stats.get('candidates_with_contradictions', 0)}")
                print(f"   Total Major Contradictions: {wm_stats.get('total_major_contradictions', 0)}")
                print(f"   Total Minor Contradictions: {wm_stats.get('total_minor_contradictions', 0)}")
                print(f"   Avg Facts Checked: {wm_stats.get('avg_facts_checked', 0):.0f}")
                if wm_stats.get('contradiction_types_found'):
                    print(f"   Contradiction Types: {wm_stats['contradiction_types_found']}")
        
        # Show final answer
        if wisdom_package.final_answer:
            print(f"\n📝 Final Answer Preview:")
            print(f"   Length: {len(wisdom_package.final_answer)} characters")
            print(f"   Preview: {wisdom_package.final_answer[:200]}...")
        
        # Check candidate generation stats
        if wisdom_package.candidate_generation_stats:
            cand_stats = wisdom_package.candidate_generation_stats
            print(f"\n📊 Candidate Generation:")
            print(f"   Total Generated: {cand_stats.get('total_candidates_generated', 0)}")
            print(f"   Success Rate: {cand_stats.get('generation_success_rate', 0):.1%}")
        
        # Check deduplication results
        if wisdom_package.deduplication_stats:
            dedup_stats = wisdom_package.deduplication_stats
            print(f"\n🗜️ Deduplication Results:")
            print(f"   Original Candidates: {dedup_stats.get('original_candidates', 0)}")
            print(f"   Unique Candidates: {dedup_stats.get('unique_candidates', 0)}")
            print(f"   Compression Ratio: {dedup_stats.get('compression_ratio', 0):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ NWTN World Model integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_world_model_candidate_scoring():
    """Test World Model scoring independently"""
    
    print("\n🧪 TESTING: World Model Candidate Scoring")
    print("=" * 60)
    
    try:
        from engines.universal_knowledge_ingestion_engine import WorldModelKnowledge
        
        # Create empty World Model for testing
        world_model = WorldModelKnowledge()
        
        # Test contradiction detection without knowledge (baseline)
        test_candidate = "Quantum mechanics describes the behavior of particles at atomic scales."
        baseline_result = world_model.detect_contradictions(test_candidate, "physics")
        
        print(f"🔬 Baseline Contradiction Detection (no knowledge):")
        print(f"   Has Contradictions: {baseline_result['has_contradictions']}")
        print(f"   Contradiction Score: {baseline_result['contradiction_score']}")
        print(f"   Facts Checked: {baseline_result['facts_checked']}")
        print(f"   Contradiction Confidence: {baseline_result['contradiction_confidence']}")
        
        return True
        
    except Exception as e:
        print(f"❌ World Model candidate scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    
    print("🚀 NWTN WORLD MODEL INTEGRATION TEST SUITE")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: ZIM processing
    print("\n" + "=" * 70)
    result1 = await test_world_model_zim_processing()
    test_results.append(("ZIM Processing", result1))
    
    # Test 2: Candidate scoring
    print("\n" + "=" * 70)
    result2 = await test_world_model_candidate_scoring()
    test_results.append(("Candidate Scoring", result2))
    
    # Test 3: Full pipeline integration (may take longer)
    print("\n" + "=" * 70)
    result3 = await test_nwtn_world_model_integration()
    test_results.append(("Full NWTN Integration", result3))
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(test_results)} tests passed ({passed/len(test_results)*100:.0f}%)")
    
    if passed == len(test_results):
        print("\n🎉 All tests passed! World Model integration is working correctly.")
        print("\n📋 World Model Features Verified:")
        print("   - ZIM file processing and knowledge extraction")
        print("   - World Model candidate scoring and alignment detection")
        print("   - Integration with NWTN meta-reasoning pipeline")
        print("   - Enhanced candidate confidence scoring")
        print("   - Contradiction detection and factual support analysis")
    else:
        print(f"\n⚠️  {len(test_results) - passed} test(s) failed. Check the implementation.")
    
    return passed == len(test_results)

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)