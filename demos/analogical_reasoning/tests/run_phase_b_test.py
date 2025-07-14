#!/usr/bin/env python3
"""
Run Phase B: Multi-Tier Relevance Analysis Test
Execute the hybrid approach with domain relevance filtering + calibrated scoring
"""

from enhanced_batch_processor import EnhancedBatchProcessor

def main():
    """Execute Phase B multi-tier relevance analysis"""
    
    print("🎯 PHASE B: MULTI-TIER RELEVANCE ANALYSIS")
    print("=" * 80)
    print("🔬 Purpose: Test hybrid approach with domain relevance tiers")
    print("📊 Method: Categorize 100 unique papers by relevance distance")
    print("🎯 Goal: Compare discovery rates across relevance tiers")
    print("💡 Insight: Does domain distance affect breakthrough quality?")
    
    # Create enhanced batch processor with Phase B configuration
    processor = EnhancedBatchProcessor(test_mode="phase_b", use_calibrated_scoring=True)
    
    # Execute unified Phase B test
    results = processor.run_unified_test(
        test_mode="phase_b",
        paper_count=100,
        paper_source="unique"
    )
    
    if results:
        print(f"\n🎉 PHASE B TEST COMPLETE!")
        print(f"Results saved in enhanced_phase_b_processing/results/")
        print(f"🎯 Phase B tier analysis demonstrates hybrid approach effectiveness")
    else:
        print(f"\n❌ PHASE B TEST FAILED")
        print(f"Check logs for error details")
    
    return results

if __name__ == "__main__":
    main()