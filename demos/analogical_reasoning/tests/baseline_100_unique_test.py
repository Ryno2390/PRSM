#!/usr/bin/env python3
"""
Baseline 100 Unique Papers Test
Establishes true baseline for breakthrough discovery with 100 unique scientific papers
"""

import json
from breakthrough_focused_processor import BreakthroughFocusedProcessor

class Baseline100UniqueTest(BreakthroughFocusedProcessor):
    """Test with 100 unique papers to establish true baseline"""
    
    def __init__(self):
        super().__init__(paper_count=100)
        self.storage = self.storage.__class__("baseline_100_unique_papers")
    
    def load_unique_papers(self, count: int = 100) -> list:
        """Load unique papers from collection"""
        
        try:
            with open('unique_papers_collection.json', 'r') as f:
                data = json.load(f)
            
            papers = data['papers']
            print(f"✅ Loaded {len(papers)} unique papers from collection")
            
            # Select first 100 unique papers
            selected_papers = papers[:count]
            print(f"📊 Selected {len(selected_papers)} papers for baseline test")
            
            # Ensure no duplicates
            unique_ids = set(p['arxiv_id'] for p in selected_papers)
            print(f"🔍 Verified {len(unique_ids)} unique arXiv IDs")
            
            if len(unique_ids) != len(selected_papers):
                print("⚠️ Duplicate detection - filtering...")
                seen_ids = set()
                filtered_papers = []
                for paper in selected_papers:
                    if paper['arxiv_id'] not in seen_ids:
                        seen_ids.add(paper['arxiv_id'])
                        filtered_papers.append(paper)
                selected_papers = filtered_papers
            
            return selected_papers
            
        except Exception as e:
            print(f"❌ Error loading unique papers: {e}")
            return []
    
    def run_baseline_test(self):
        """Run baseline test with 100 unique papers"""
        
        print(f"\n🎯 BASELINE TEST: 100 UNIQUE PAPERS")
        print(f"=" * 60)
        print(f"🔬 Purpose: Establish true baseline for breakthrough discovery")
        print(f"📊 Data: 100 unique scientific papers across diverse domains")
        print(f"🏆 Primary Metric: High-quality breakthrough count (≥0.7)")
        print(f"💰 Secondary Metric: Commercial opportunities (≥0.6)")
        
        # Load unique papers
        self.papers = self.load_unique_papers(100)
        
        if len(self.papers) < 100:
            print(f"⚠️ Only {len(self.papers)} unique papers available")
            proceed = input("Proceed with available papers? (y/n): ")
            if proceed.lower() != 'y':
                return None
        
        # Update paper count for processing
        self.paper_count = len(self.papers)
        
        # Execute breakthrough-focused test
        results = self.run_breakthrough_focused_test()
        
        # Additional baseline analysis
        self._analyze_baseline_results()
        
        return results
    
    def _analyze_baseline_results(self):
        """Additional analysis for baseline establishment"""
        
        print(f"\n📊 BASELINE ANALYSIS")
        print(f"=" * 60)
        
        hq_count = self.breakthrough_metrics['high_quality_count']
        commercial_count = self.breakthrough_metrics['commercial_count']
        papers_processed = self.batch_results['papers_processed']
        
        # Realistic expectations for 100 unique papers
        print(f"🎯 BASELINE METRICS:")
        print(f"   Papers processed: {papers_processed} unique papers")
        print(f"   High-quality breakthroughs: {hq_count}")
        print(f"   Commercial opportunities: {commercial_count}")
        print(f"   Discovery rate: {(hq_count/papers_processed)*100:.1f}% high-quality")
        print(f"   Commercial rate: {(commercial_count/papers_processed)*100:.1f}% commercial")
        
        # Set realistic scaling expectations
        print(f"\n📈 SCALING PROJECTIONS (based on unique paper baseline):")
        
        if hq_count > 0:
            hq_rate = hq_count / papers_processed
            print(f"   1K unique papers: ~{int(1000 * hq_rate)} high-quality breakthroughs")
            print(f"   10K unique papers: ~{int(10000 * hq_rate)} high-quality breakthroughs")
        else:
            print(f"   ⚠️ No high-quality breakthroughs found in baseline")
            print(f"   Quality scoring calibration needed before scaling")
        
        if commercial_count > 0:
            comm_rate = commercial_count / papers_processed
            print(f"   1K unique papers: ~{int(1000 * comm_rate)} commercial opportunities")
        
        # Quality assessment
        print(f"\n🔍 BASELINE QUALITY ASSESSMENT:")
        
        if hq_count >= 2:
            print(f"   ✅ Strong baseline: {hq_count} high-quality breakthroughs")
            print(f"   🚀 Ready for 1K unique paper scaling")
        elif hq_count >= 1:
            print(f"   ⚠️ Moderate baseline: {hq_count} high-quality breakthrough")
            print(f"   🔧 Consider quality improvements before scaling")
        else:
            print(f"   ❌ Weak baseline: No high-quality breakthroughs")
            print(f"   🛠️ Quality scoring calibration required")
        
        # Cost efficiency baseline
        if hq_count > 0:
            cost_per_hq = (papers_processed * 0.327) / hq_count
            print(f"\n💰 COST EFFICIENCY BASELINE:")
            print(f"   Cost per high-quality breakthrough: ${cost_per_hq:.2f}")
            print(f"   Cost for 1K papers: ${1000 * 0.327:.2f}")
            print(f"   Expected ROI: {int(hq_count * 10)} breakthroughs × $100K+ value each")

def main():
    """Run baseline test with 100 unique papers"""
    
    tester = Baseline100UniqueTest()
    results = tester.run_baseline_test()
    
    if results:
        print(f"\n🎉 BASELINE TEST COMPLETE!")
        print(f"Results saved in baseline_100_unique_papers/results/")
    
    return results

if __name__ == "__main__":
    main()