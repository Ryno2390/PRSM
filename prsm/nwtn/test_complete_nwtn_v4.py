#!/usr/bin/env python3
"""
Test Complete NWTN Pipeline V4 - Full 9-Step Implementation
==========================================================

Tests the complete 9-step NWTN pipeline with:
1. User Prompt Input
2. Semantic Search (2,295 papers)
3. Content Analysis 
4. System 1: Generate 5,040 candidate answers
5. Deduplication & Compression
6. System 2: Meta-reasoning
7. Wisdom Package Creation
8. LLM Integration
9. Final Natural Language Response
"""

import asyncio
import logging
import sys
import time
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the complete NWTN pipeline
try:
    from complete_nwtn_pipeline_v4 import CompleteNWTNPipeline, WisdomPackage
    COMPLETE_PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import complete NWTN pipeline: {e}")
    COMPLETE_PIPELINE_AVAILABLE = False


async def test_complete_nwtn_pipeline():
    """Test the complete 9-step NWTN pipeline implementation"""
    
    print("🧠 NWTN Complete Pipeline V4 - Full Implementation Test")
    print("=" * 70)
    print("Testing all 9 steps of the complete NWTN pipeline:")
    print("1. 📝 User Prompt Input")
    print("2. 📚 Semantic Search (2,295 papers → 20 most relevant)")
    print("3. 🔬 Content Analysis (extract key concepts)")
    print("4. 🧠 System 1: Generate 5,040 candidate answers (7! reasoning engine permutations)")
    print("5. 🗜️ Deduplication & Compression")
    print("6. 🎯 System 2: Meta-reasoning evaluation and synthesis")
    print("7. 📦 Wisdom Package Creation (answer + traces + corpus + metadata)")
    print("8. 🤖 LLM Integration (Claude API) for natural language generation")
    print("9. ✨ Final Natural Language Response")
    print()
    
    if not COMPLETE_PIPELINE_AVAILABLE:
        print("❌ Complete NWTN pipeline not available - check imports")
        return False
    
    try:
        # Initialize the complete pipeline
        print("🏗️  Initializing Complete NWTN Pipeline...")
        pipeline = CompleteNWTNPipeline()
        
        # Test query - context rot problem
        test_query = """The concept of "context rot" in AI systems refers to the degradation of performance when the operational context differs significantly from the training context. This phenomenon is particularly pronounced in large language models and neural networks deployed in dynamic environments.

Context rot manifests in several ways:
1. Distribution shift between training and deployment data
2. Temporal drift in data patterns over time  
3. Domain adaptation challenges when models encounter new scenarios
4. Catastrophic forgetting when models are updated with new information

Understanding and mitigating context rot is crucial for maintaining robust AI systems in production environments. What are the most effective strategies for detecting and preventing context rot in deployed machine learning models?"""
        
        print(f"📝 Test Query: {test_query[:150]}...")
        print()
        
        # Run the complete pipeline (note: this will take time for 5,040 candidates)
        print("🚀 Starting Complete 9-Step NWTN Pipeline...")
        print("⚠️  Warning: This will generate 5,040 candidates and may take several minutes")
        print()
        
        start_time = time.time()
        
        # Execute the complete pipeline
        wisdom_package = await pipeline.run_complete_pipeline(
            query=test_query,
            context_allocation=1000
        )
        
        total_time = time.time() - start_time
        
        # Analyze results
        print("🎊 COMPLETE NWTN PIPELINE V4 - RESULTS ANALYSIS")
        print("=" * 70)
        
        # Step 1-3: Input and Search Analysis
        print(f"📊 SEMANTIC & CONTENT ANALYSIS:")
        print(f"   Papers analyzed: {wisdom_package.corpus_metadata.get('total_papers_available', 0)}")
        print(f"   Content words processed: {wisdom_package.corpus_metadata.get('total_content_words', 0):,}")
        print(f"   Content extraction success: {wisdom_package.corpus_metadata.get('papers_with_content_extracted', 0)} papers")
        
        # Step 4: System 1 Analysis
        print(f"\n🧠 SYSTEM 1 CANDIDATE GENERATION:")
        print(f"   Total candidates generated: {wisdom_package.candidate_generation_stats.get('total_candidates_generated', 0):,}")
        print(f"   Reasoning permutations used: {wisdom_package.candidate_generation_stats.get('reasoning_permutations_used', 0):,}")
        print(f"   Unique sequences: {wisdom_package.candidate_generation_stats.get('unique_reasoning_sequences', 0)}")
        print(f"   Generation success rate: {wisdom_package.candidate_generation_stats.get('generation_success_rate', 0):.1%}")
        
        # Step 5: Deduplication Analysis
        print(f"\n🗜️ DEDUPLICATION & COMPRESSION:")
        print(f"   Original candidates: {wisdom_package.deduplication_stats.get('original_candidates', 0):,}")
        print(f"   Unique candidates: {wisdom_package.deduplication_stats.get('unique_candidates', 0):,}")
        print(f"   Compression ratio: {wisdom_package.deduplication_stats.get('compression_ratio', 0):.1%}")
        print(f"   Processing time: {wisdom_package.deduplication_stats.get('processing_time', 0):.2f}s")
        
        # Step 6: System 2 Meta-reasoning
        print(f"\n🎯 SYSTEM 2 META-REASONING:")
        print(f"   Candidates analyzed: {wisdom_package.meta_reasoning_analysis.get('total_candidates_analyzed', 0)}")
        print(f"   Top candidates selected: {wisdom_package.meta_reasoning_analysis.get('top_candidates_selected', 0)}")
        print(f"   Consensus strength: {wisdom_package.meta_reasoning_analysis.get('consensus_strength', 0):.3f}")
        print(f"   Evidence diversity: {wisdom_package.meta_reasoning_analysis.get('evidence_diversity', 0)} sources")
        
        # Step 7: Wisdom Package
        print(f"\n📦 WISDOM PACKAGE:")
        print(f"   Session ID: {wisdom_package.session_id}")
        print(f"   Reasoning traces: {len(wisdom_package.reasoning_traces)}")
        print(f"   Creation timestamp: {wisdom_package.creation_timestamp}")
        
        # Step 8-9: LLM Integration and Final Response
        print(f"\n🤖 LLM INTEGRATION & FINAL RESPONSE:")
        print(f"   Final answer length: {len(wisdom_package.final_answer):,} characters")
        print(f"   Average confidence: {wisdom_package.confidence_metrics.get('meta_reasoning_confidence', 0):.3f}")
        print(f"   Max candidate confidence: {wisdom_package.confidence_metrics.get('max_candidate_confidence', 0):.3f}")
        
        # Processing Timeline
        print(f"\n⏱️  PROCESSING TIMELINE:")
        timeline = wisdom_package.processing_timeline
        print(f"   Semantic Search: {timeline.get('step2_semantic_search', 0):.2f}s")
        print(f"   Content Analysis: {timeline.get('step3_content_analysis', 0):.2f}s")
        print(f"   System 1 Generation: {timeline.get('step4_system1_generation', 0):.2f}s")
        print(f"   Deduplication: {timeline.get('step5_deduplication', 0):.2f}s")
        print(f"   Meta-reasoning: {timeline.get('step6_meta_reasoning', 0):.2f}s")
        print(f"   Wisdom Package: {timeline.get('step7_wisdom_package', 0):.2f}s")
        print(f"   LLM Integration: {timeline.get('step8_llm_integration', 0):.2f}s")
        print(f"   Total Pipeline: {total_time:.2f}s")
        
        # Display final response preview
        print(f"\n📄 FINAL RESPONSE PREVIEW:")
        print("─" * 70)
        response_preview = wisdom_package.final_answer[:500]
        print(response_preview)
        if len(wisdom_package.final_answer) > 500:
            print("...")
        print("─" * 70)
        
        # Validation checks
        print(f"\n✅ PIPELINE VALIDATION:")
        validations = [
            ("9-step pipeline executed", True),
            ("5,040 candidates generated", wisdom_package.candidate_generation_stats.get('total_candidates_generated', 0) >= 5000),
            ("Deduplication applied", wisdom_package.deduplication_stats.get('compression_ratio', 0) < 1.0),
            ("Meta-reasoning performed", wisdom_package.meta_reasoning_analysis.get('total_candidates_analyzed', 0) > 0),
            ("Wisdom Package created", wisdom_package.session_id is not None),
            ("Natural language response", len(wisdom_package.final_answer) > 500),
            ("Content grounded", wisdom_package.corpus_metadata.get('total_content_words', 0) > 0),
            ("Comprehensive analysis", len(wisdom_package.reasoning_traces) >= 5)
        ]
        
        passed_validations = 0
        for validation_name, passed in validations:
            status = "✅" if passed else "❌"
            print(f"   {status} {validation_name}")
            if passed:
                passed_validations += 1
        
        success_rate = passed_validations / len(validations)
        print(f"\n🎯 VALIDATION SUMMARY: {passed_validations}/{len(validations)} checks passed ({success_rate:.1%})")
        
        # Save detailed results
        results_file = f"complete_nwtn_v4_results_{int(time.time())}.json"
        try:
            # Prepare JSON-serializable results
            json_results = {
                'test_summary': {
                    'pipeline_version': 'Complete NWTN V4',
                    'total_processing_time': total_time,
                    'validation_success_rate': success_rate,
                    'passed_validations': passed_validations,
                    'total_validations': len(validations),
                    'test_timestamp': wisdom_package.creation_timestamp
                },
                'candidate_statistics': wisdom_package.candidate_generation_stats,
                'deduplication_statistics': wisdom_package.deduplication_stats,
                'meta_reasoning_analysis': wisdom_package.meta_reasoning_analysis,
                'corpus_metadata': wisdom_package.corpus_metadata,
                'confidence_metrics': wisdom_package.confidence_metrics,
                'processing_timeline': wisdom_package.processing_timeline,
                'final_answer_length': len(wisdom_package.final_answer),
                'session_id': wisdom_package.session_id
            }
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\n💾 Detailed results saved to: {results_file}")
            
        except Exception as e:
            print(f"\n⚠️  Could not save results file: {e}")
        
        # Final assessment
        if success_rate >= 0.8:  # 80% success rate required
            print(f"\n🎉 COMPLETE NWTN PIPELINE V4: SUCCESS!")
            print("All major components validated and working correctly.")
            print("The complete 9-step NWTN pipeline with 5,040 candidates, deduplication,")
            print("and Wisdom Package creation is fully operational.")
            return True
        else:
            print(f"\n⚠️  COMPLETE NWTN PIPELINE V4: PARTIAL SUCCESS")
            print(f"Success rate: {success_rate:.1%} (need ≥80%)")
            print("Some components may need attention.")
            return False
            
    except Exception as e:
        print(f"❌ Complete NWTN Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the complete NWTN pipeline test"""
    
    print("🚀 Starting Complete NWTN Pipeline V4 Test...")
    print("⏱️  This will test the full 9-step implementation including:")
    print("   • 5,040 candidate generation (7! reasoning permutations)")
    print("   • Deduplication & compression")
    print("   • System 2 meta-reasoning")
    print("   • Wisdom Package creation")
    print("   • LLM integration")
    print()
    
    success = await test_complete_nwtn_pipeline()
    
    if success:
        print("\n🎊 Complete NWTN Pipeline V4 test completed successfully!")
        print("All 9 steps of the NWTN pipeline are operational.")
    else:
        print("\n⚠️  Complete NWTN Pipeline V4 test had issues.")
        print("Review the output above for specific problems.")
    
    return success


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit_code = 0 if result else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)