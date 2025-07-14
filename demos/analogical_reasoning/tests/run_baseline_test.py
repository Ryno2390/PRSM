#!/usr/bin/env python3
"""
Run Phase A Baseline Test: 100 Unique Papers with Calibrated Scoring
Establish baseline metrics for breakthrough discovery scaling analysis
"""

from enhanced_batch_processor import EnhancedBatchProcessor

def main():
    """Execute Phase A baseline test with 100 unique papers"""
    
    print("🔧 PHASE A BASELINE: 100 UNIQUE PAPERS WITH CALIBRATED SCORING")
    print("=" * 80)
    print("🎯 Purpose: Establish baseline metrics for scaling analysis")
    print("📊 Method: Calibrated breakthrough ranking on 100 unique papers")
    print("💰 Goal: Determine cost per breakthrough and scaling economics")
    print("📈 Next: Scale to 1,000 papers to validate 10x mathematics")
    
    # Create enhanced batch processor with Phase A configuration
    processor = EnhancedBatchProcessor(test_mode="phase_a", use_calibrated_scoring=True)
    
    # Execute unified Phase A test
    results = processor.run_unified_test(
        test_mode="phase_a",
        paper_count=100,
        paper_source="unique"
    )
    
    if results:
        print(f"\n🎉 PHASE A BASELINE TEST COMPLETE!")
        print(f"Results saved in enhanced_phase_a_processing/results/")
        print(f"📊 Baseline established for 1,000-paper scaling analysis")
        
        # Extract key metrics for scaling analysis
        metrics = results.get('performance_metrics', {})
        hq_rate = 0
        commercial_rate = 0
        
        if 'ranked_breakthroughs' in results:
            breakthroughs = results['ranked_breakthroughs']
            papers_processed = results['papers_processed']
            
            hq_count = sum(1 for b in breakthroughs if b.breakthrough_score.overall_score >= 0.75)
            commercial_count = sum(1 for b in breakthroughs if b.breakthrough_score.overall_score >= 0.65)
            
            hq_rate = (hq_count / papers_processed) * 100
            commercial_rate = (commercial_count / papers_processed) * 100
            
        print(f"\n📊 BASELINE METRICS FOR SCALING:")
        print(f"   High-quality breakthrough rate: {hq_rate:.1f}%")
        print(f"   Commercial opportunity rate: {commercial_rate:.1f}%")
        print(f"   Cost per paper: $0.327")
        print(f"   Total baseline cost: ${100 * 0.327:.2f}")
        
        if hq_rate > 0:
            cost_per_hq = (100 * 0.327) / (hq_rate / 100 * 100)
            print(f"   Cost per high-quality breakthrough: ${cost_per_hq:.2f}")
            print(f"\n🚀 SCALING PROJECTIONS:")
            print(f"   1,000 papers expected HQ breakthroughs: {int(hq_rate * 10)}")
            print(f"   1,000 papers expected cost: ${1000 * 0.327:.2f}")
            print(f"   Expected cost per HQ breakthrough (1K): ${cost_per_hq:.2f}")
        
    else:
        print(f"\n❌ PHASE A BASELINE TEST FAILED")
        print(f"Check logs for error details")
    
    return results

if __name__ == "__main__":
    main()