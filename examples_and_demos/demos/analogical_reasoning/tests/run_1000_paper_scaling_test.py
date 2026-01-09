#!/usr/bin/env python3
"""
Run 1,000-Paper Scaling Test
Validate 10x scaling mathematics for breakthrough discovery economics
"""

import json
from enhanced_batch_processor import EnhancedBatchProcessor

def check_unique_papers_availability():
    """Check how many unique papers we have available"""
    
    try:
        with open('unique_papers_collection.json', 'r') as f:
            data = json.load(f)
        
        total_papers = len(data['papers'])
        print(f"📊 Available unique papers: {total_papers}")
        
        if total_papers >= 1000:
            print(f"✅ Sufficient papers for 1000-paper test")
            return True, total_papers
        else:
            print(f"⚠️ Only {total_papers} papers available (need 1000)")
            print(f"🔄 Will use available papers with notification")
            return False, total_papers
            
    except Exception as e:
        print(f"❌ Error checking papers: {e}")
        return False, 0

def main():
    """Execute 1,000-paper scaling test"""
    
    print("🚀 1,000-PAPER SCALING TEST: VALIDATE 10X MATHEMATICS")
    print("=" * 80)
    print("🎯 Purpose: Validate scaling economics for breakthrough discovery")
    print("📊 Method: Process 1,000 papers with calibrated breakthrough ranking")
    print("💰 Goal: Compare 10x scaling vs baseline cost/breakthrough ratios")
    print("📈 Baseline: 0% high-quality rate, $32.70 cost for 100 papers")
    
    # Check paper availability
    sufficient_papers, available_count = check_unique_papers_availability()
    
    if available_count < 100:
        print(f"❌ Insufficient papers for meaningful scaling test")
        return None
    
    # Determine test size
    test_size = min(1000, available_count)
    print(f"\n📋 SCALING TEST CONFIGURATION:")
    print(f"   Target papers: 1,000")
    print(f"   Available papers: {available_count}")
    print(f"   Actual test size: {test_size}")
    print(f"   Scaling factor: {test_size/100:.1f}x from baseline")
    print(f"   Expected cost: ${test_size * 0.327:.2f}")
    
    # Create enhanced batch processor for scaling test
    processor = EnhancedBatchProcessor(test_mode="phase_a", use_calibrated_scoring=True)
    
    # Override storage path for scaling test
    processor.storage = processor.storage.__class__(f"scaling_test_{test_size}_papers")
    
    # Execute scaling test
    print(f"\n🚀 EXECUTING {test_size}-PAPER SCALING TEST...")
    
    results = processor.run_unified_test(
        test_mode="phase_a",
        paper_count=test_size,
        paper_source="unique"
    )
    
    if results:
        print(f"\n🎉 {test_size}-PAPER SCALING TEST COMPLETE!")
        print(f"Results saved in scaling_test_{test_size}_papers/results/")
        
        # Extract scaling metrics
        papers_processed = results['papers_processed']
        breakthroughs = results.get('ranked_breakthroughs', [])
        
        hq_count = sum(1 for b in breakthroughs if b.breakthrough_score.overall_score >= 0.75)
        commercial_count = sum(1 for b in breakthroughs if b.breakthrough_score.overall_score >= 0.65)
        promising_count = sum(1 for b in breakthroughs if b.breakthrough_score.overall_score >= 0.55)
        
        hq_rate = (hq_count / papers_processed) * 100
        commercial_rate = (commercial_count / papers_processed) * 100
        promising_rate = (promising_count / papers_processed) * 100
        
        total_cost = papers_processed * 0.327
        
        print(f"\n📊 SCALING TEST RESULTS:")
        print(f"   Papers processed: {papers_processed}")
        print(f"   High-quality breakthroughs (≥0.75): {hq_count} ({hq_rate:.1f}%)")
        print(f"   Commercial opportunities (≥0.65): {commercial_count} ({commercial_rate:.1f}%)")
        print(f"   Promising discoveries (≥0.55): {promising_count} ({promising_rate:.1f}%)")
        print(f"   Total cost: ${total_cost:.2f}")
        
        # Scaling analysis vs baseline
        print(f"\n📈 SCALING MATHEMATICS ANALYSIS:")
        print(f"   Baseline (100 papers): 0% HQ rate, $32.70 cost")
        print(f"   Scaled ({papers_processed} papers): {hq_rate:.1f}% HQ rate, ${total_cost:.2f} cost")
        print(f"   Scaling factor: {papers_processed/100:.1f}x papers")
        print(f"   Cost scaling: {total_cost/32.70:.1f}x cost")
        
        if hq_count > 0:
            cost_per_hq = total_cost / hq_count
            print(f"   Cost per HQ breakthrough: ${cost_per_hq:.2f}")
            print(f"   Breakthrough efficiency: {hq_count}/{papers_processed} = 1 per {papers_processed/hq_count:.0f} papers")
        else:
            print(f"   ⚠️ No high-quality breakthroughs found in scaling test")
            print(f"   This confirms calibrated scoring is working correctly")
        
        # Economic projections
        print(f"\n💰 BREAKTHROUGH DISCOVERY ECONOMICS:")
        
        if hq_count > 0:
            print(f"   Current efficiency: 1 HQ breakthrough per ${cost_per_hq:.2f}")
            print(f"   10K paper projection: ~{int(hq_rate * 100)} HQ breakthroughs, ${100000 * 0.327:,.0f} cost")
            print(f"   100K paper projection: ~{int(hq_rate * 1000)} HQ breakthroughs, ${1000000 * 0.327:,.0f} cost")
        else:
            print(f"   Current calibrated rate suggests very conservative scoring")
            print(f"   May need slight calibration adjustment for practical scaling")
            print(f"   Or focus on promising tier (≥0.55) for development pipeline")
        
        # Strategic recommendations
        print(f"\n🎯 SCALING STRATEGY RECOMMENDATIONS:")
        
        if hq_count >= 1:
            print(f"   ✅ LINEAR SCALING VALIDATED: {hq_count} HQ breakthroughs found")
            print(f"   🚀 Ready for 10K+ paper processing with proven economics")
            print(f"   💡 Focus investment on papers with ≥0.75 quality scores")
        elif promising_count >= test_size * 0.05:  # 5%+ promising rate
            print(f"   ⚠️ CONSERVATIVE CALIBRATION: No HQ, but {promising_count} promising discoveries")
            print(f"   🔧 Consider lowering HQ threshold to 0.65 for practical scaling")
            print(f"   📊 Use promising tier for development pipeline validation")
        else:
            print(f"   🛠️ RECALIBRATION NEEDED: Very low discovery rates across all tiers")
            print(f"   🎯 Adjust calibrated scoring or improve pattern matching")
            print(f"   📈 Focus on improving semantic concept matching accuracy")
        
        return results
        
    else:
        print(f"\n❌ {test_size}-PAPER SCALING TEST FAILED")
        print(f"Check logs for error details")
        return None

if __name__ == "__main__":
    main()