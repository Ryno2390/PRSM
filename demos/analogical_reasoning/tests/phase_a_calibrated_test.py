#!/usr/bin/env python3
"""
Phase A: Calibrated Quality Scoring Test
Tests 100 unique papers with calibrated (realistic) quality scoring
"""

import json
from enhanced_batch_processor import EnhancedBatchProcessor
from calibrated_breakthrough_ranker import CalibratedBreakthroughRanker

class PhaseACalibratedTest(EnhancedBatchProcessor):
    """Phase A test with calibrated breakthrough ranking"""
    
    def __init__(self):
        super().__init__()
        self.paper_count = 100
        self.storage = self.storage.__class__("phase_a_calibrated_100_papers")
        
        # Replace with calibrated ranker
        self.breakthrough_ranker = CalibratedBreakthroughRanker()
        
        # Phase A specific metrics
        self.phase_a_metrics = {
            'calibration_type': 'Conservative Quality Scoring',
            'expected_breakthrough_rate': '0.5-2%',
            'expected_commercial_rate': '5-15%',
            'comparison_baseline': 'Original inflated scoring'
        }
    
    def load_unique_papers(self, count: int = 100) -> list:
        """Load unique papers for Phase A test"""
        
        try:
            with open('unique_papers_collection.json', 'r') as f:
                data = json.load(f)
            
            papers = data['papers']
            selected_papers = papers[:count]
            
            print(f"✅ Phase A: Loaded {len(selected_papers)} unique papers")
            print(f"🔧 Using calibrated breakthrough ranking system")
            
            return selected_papers
            
        except Exception as e:
            print(f"❌ Error loading unique papers: {e}")
            return []
    
    def run_phase_a_test(self):
        """Run Phase A calibrated quality test"""
        
        print(f"\n🔧 PHASE A: CALIBRATED QUALITY SCORING TEST")
        print(f"=" * 70)
        print(f"🎯 Purpose: Establish realistic quality scoring baseline")
        print(f"📊 Data: 100 unique scientific papers")
        print(f"🔧 Method: Calibrated breakthrough ranking (conservative scoring)")
        print(f"📈 Expected: 0.5-2% breakthrough rate (vs previous 21%)")
        print(f"💰 Expected: 5-15% commercial rate (vs previous 98%)")
        
        # Show calibration details
        calibration_summary = self.breakthrough_ranker.get_calibration_summary()
        print(f"\n🎯 CALIBRATION CHANGES:")
        for change in calibration_summary['key_changes'][:3]:
            print(f"   • {change}")
        print(f"   • And {len(calibration_summary['key_changes'])-3} more...")
        
        # Load unique papers
        self.papers = self.load_unique_papers(100)
        
        if len(self.papers) < 100:
            print(f"⚠️ Only {len(self.papers)} unique papers available")
        
        # Update paper count
        self.paper_count = len(self.papers)
        
        # Initialize breakthrough metrics for Phase A
        self.breakthrough_metrics = {
            'high_quality_breakthroughs': [],      # Score ≥0.75 (calibrated)
            'commercial_opportunities': [],        # Score ≥0.65 (calibrated)
            'promising_discoveries': [],           # Score ≥0.55 (calibrated)
            'baseline_discoveries': [],            # Score ≥0.45 (calibrated)
            
            'high_quality_count': 0,
            'commercial_count': 0,
            'promising_count': 0,
            'baseline_count': 0,
            
            'cost_per_high_quality_breakthrough': 0.0,
            'cost_per_commercial_opportunity': 0.0,
            'total_processing_cost': self.paper_count * 0.327,
            
            'calibrated_breakthrough_discovery_rate': 0.0,
            'calibrated_commercial_discovery_rate': 0.0
        }
        
        # Execute enhanced batch processing with calibrated ranking
        results = self.run_enhanced_batch_processing()
        
        # Phase A specific analysis
        self._analyze_phase_a_results()
        
        return results
    
    def _categorize_breakthrough(self, breakthrough: object, quality_score: float):
        """Categorize breakthrough using CALIBRATED thresholds"""
        
        # Use calibrated thresholds
        if quality_score >= 0.75:  # HIGH_POTENTIAL threshold
            self.breakthrough_metrics['high_quality_breakthroughs'].append(breakthrough)
            self.breakthrough_metrics['high_quality_count'] += 1
        
        if quality_score >= 0.65:  # PROMISING threshold
            self.breakthrough_metrics['commercial_opportunities'].append(breakthrough)
            self.breakthrough_metrics['commercial_count'] += 1
        
        if quality_score >= 0.55:  # MODERATE threshold  
            self.breakthrough_metrics['promising_discoveries'].append(breakthrough)
            self.breakthrough_metrics['promising_count'] += 1
        
        if quality_score >= 0.45:  # LOW_PRIORITY threshold
            self.breakthrough_metrics['baseline_discoveries'].append(breakthrough)
            self.breakthrough_metrics['baseline_count'] += 1
    
    def _analyze_phase_a_results(self):
        """Phase A specific analysis with calibrated scoring"""
        
        print(f"\n🔧 PHASE A CALIBRATED ANALYSIS")
        print(f"=" * 70)
        
        hq_count = self.breakthrough_metrics['high_quality_count']
        commercial_count = self.breakthrough_metrics['commercial_count']
        promising_count = self.breakthrough_metrics['promising_count']
        papers_processed = self.batch_results['papers_processed']
        
        print(f"📊 CALIBRATED BREAKTHROUGH METRICS:")
        print(f"   Papers processed: {papers_processed} unique papers")
        print(f"   High-quality breakthroughs (≥0.75): {hq_count}")
        print(f"   Commercial opportunities (≥0.65): {commercial_count}")
        print(f"   Promising discoveries (≥0.55): {promising_count}")
        
        # Calculate realistic rates
        hq_rate = (hq_count / papers_processed) * 100
        commercial_rate = (commercial_count / papers_processed) * 100
        promising_rate = (promising_count / papers_processed) * 100
        
        print(f"\n📈 CALIBRATED DISCOVERY RATES:")
        print(f"   High-quality rate: {hq_rate:.1f}% (target: 0.5-2%)")
        print(f"   Commercial rate: {commercial_rate:.1f}% (target: 5-15%)")
        print(f"   Promising rate: {promising_rate:.1f}% (target: 10-25%)")
        
        # Validation against targets
        print(f"\n✅ CALIBRATION VALIDATION:")
        
        if 0.5 <= hq_rate <= 2.0:
            print(f"   🎯 High-quality rate WITHIN TARGET: {hq_rate:.1f}%")
        elif hq_rate < 0.5:
            print(f"   ⚠️ High-quality rate BELOW TARGET: {hq_rate:.1f}% (too conservative)")
        else:
            print(f"   ❌ High-quality rate ABOVE TARGET: {hq_rate:.1f}% (still too generous)")
        
        if 5.0 <= commercial_rate <= 15.0:
            print(f"   🎯 Commercial rate WITHIN TARGET: {commercial_rate:.1f}%")
        elif commercial_rate < 5.0:
            print(f"   ⚠️ Commercial rate BELOW TARGET: {commercial_rate:.1f}% (too conservative)")
        else:
            print(f"   ❌ Commercial rate ABOVE TARGET: {commercial_rate:.1f}% (still too generous)")
        
        # Cost efficiency with calibrated metrics
        if hq_count > 0:
            cost_per_hq = (papers_processed * 0.327) / hq_count
            print(f"\n💰 CALIBRATED COST EFFICIENCY:")
            print(f"   Cost per high-quality breakthrough: ${cost_per_hq:.2f}")
            print(f"   Expected value per breakthrough: $100K-1M+")
            print(f"   ROI ratio: {(500000/cost_per_hq):.0f}x (assuming $500K avg value)")
        
        # Scaling projections with calibrated rates
        print(f"\n📈 PHASE A SCALING PROJECTIONS:")
        if hq_count > 0:
            print(f"   1K papers: ~{int(hq_rate * 10)} high-quality breakthroughs")
            print(f"   10K papers: ~{int(hq_rate * 100)} high-quality breakthroughs")
            print(f"   Cost for 1K: ${1000 * 0.327:.2f}")
            print(f"   Expected ROI (1K): ${int(hq_rate * 10) * 500000:,.0f}+ value")
        
        # Phase B preview
        print(f"\n🎯 PHASE B PREPARATION:")
        print(f"   Phase A baseline established: {hq_rate:.1f}% high-quality rate")
        print(f"   Phase B target: 10-20% high-quality rate from relevant papers only")
        print(f"   Phase B method: Domain relevance filtering + calibrated scoring")
        print(f"   Expected improvement: 3-10x better discovery rate")
        
        # Success assessment for Phase A
        if 0.5 <= hq_rate <= 3.0 and 3.0 <= commercial_rate <= 20.0:
            print(f"\n✅ PHASE A SUCCESS: Calibrated scoring working correctly")
            print(f"   🚀 Ready for Phase B domain relevance filtering")
        elif hq_rate == 0 and commercial_rate < 3:
            print(f"\n⚠️ PHASE A: Over-calibrated (too conservative)")
            print(f"   🔧 Slightly relax scoring thresholds for Phase B")
        else:
            print(f"\n❌ PHASE A: Under-calibrated (still too generous)")
            print(f"   🔧 Further tighten scoring before Phase B")

def main():
    """Run Phase A calibrated quality test"""
    
    tester = PhaseACalibratedTest()
    results = tester.run_phase_a_test()
    
    if results:
        print(f"\n🎉 PHASE A CALIBRATED TEST COMPLETE!")
        print(f"Results saved in phase_a_calibrated_100_papers/results/")
        print(f"🎯 Phase B (domain filtering) ready for implementation")
    
    return results

if __name__ == "__main__":
    main()