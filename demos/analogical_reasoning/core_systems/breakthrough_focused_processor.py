#!/usr/bin/env python3
"""
Breakthrough-Focused Batch Processor
Optimized for tracking high-quality breakthrough discoveries as primary success metric
"""

import json
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import traceback

# Import our validated pipeline components
from enhanced_batch_processor import EnhancedBatchProcessor

class BreakthroughFocusedProcessor(EnhancedBatchProcessor):
    """
    Batch processor focused on high-quality breakthrough discovery count
    Primary success metric: Number of breakthroughs with quality score ≥0.7
    """
    
    def __init__(self, paper_count: int = 1000):
        super().__init__()
        self.paper_count = paper_count
        self.storage = self.storage.__class__(f"breakthrough_focused_{paper_count}_papers")
        
        # Breakthrough-focused metrics
        self.breakthrough_metrics = {
            'high_quality_breakthroughs': [],      # Score ≥0.7
            'commercial_opportunities': [],        # Score ≥0.6  
            'promising_discoveries': [],           # Score ≥0.5
            'baseline_discoveries': [],            # Score ≥0.4
            
            'high_quality_count': 0,
            'commercial_count': 0,
            'promising_count': 0,
            'baseline_count': 0,
            
            'cost_per_high_quality_breakthrough': 0.0,
            'cost_per_commercial_opportunity': 0.0,
            'total_processing_cost': 0.0,
            
            'quality_improvement_rate': 0.0,
            'breakthrough_discovery_rate': 0.0,     # High-quality per 100 papers
            'commercial_discovery_rate': 0.0        # Commercial per 100 papers
        }
    
    def load_papers_for_scale_test(self, count: int) -> List[Dict]:
        """Load papers for large-scale breakthrough testing"""
        
        try:
            with open('selected_papers.json', 'r') as f:
                papers = json.load(f)
            
            if len(papers) < count:
                print(f"⚠️ Only {len(papers)} papers available, need {count}")
                print("🔄 Cycling through available papers to reach target count...")
                
                # Cycle through papers to reach target count
                extended_papers = []
                while len(extended_papers) < count:
                    remaining_needed = count - len(extended_papers)
                    papers_to_add = min(remaining_needed, len(papers))
                    extended_papers.extend(papers[:papers_to_add])
                
                papers = extended_papers
            
            selected_papers = papers[:count]
            print(f"✅ Loaded {len(selected_papers)} papers for {count}-paper breakthrough test")
            return selected_papers
            
        except Exception as e:
            print(f"❌ Error loading papers: {e}")
            return []
    
    def run_breakthrough_focused_test(self, paper_count: int = None) -> Dict:
        """Execute breakthrough-focused batch processing"""
        
        if paper_count:
            self.paper_count = paper_count
            self.papers = self.load_papers_for_scale_test(paper_count)
        
        print(f"\n🏆 STARTING BREAKTHROUGH-FOCUSED {self.paper_count}-PAPER TEST")
        print(f"=" * 80)
        print(f"🎯 PRIMARY METRIC: High-Quality Breakthrough Count (score ≥0.7)")
        print(f"💰 SECONDARY METRIC: Commercial Opportunities (score ≥0.6)")  
        print(f"📊 TERTIARY METRIC: Cost per High-Quality Breakthrough")
        print(f"📈 Target Papers: {self.paper_count}")
        print(f"💵 Expected Cost: ${self.paper_count * 0.327:.2f}")
        print(f"🎯 Success Threshold: ≥{max(2, self.paper_count // 100)} high-quality breakthroughs")
        
        self.breakthrough_metrics['total_processing_cost'] = self.paper_count * 0.327
        
        # Execute enhanced batch processing
        self.batch_results['start_time'] = time.time()
        
        # Process in optimized batches
        batch_size = min(50, max(20, self.paper_count // 20))  # Adaptive batch sizing
        total_batches = (len(self.papers) + batch_size - 1) // batch_size
        
        print(f"\n📦 PROCESSING STRATEGY:")
        print(f"   Batch size: {batch_size} papers")
        print(f"   Total batches: {total_batches}")
        print(f"   Progress updates every: {max(10, self.paper_count // 20)} papers")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(self.papers))
            batch_papers = self.papers[start_idx:end_idx]
            
            print(f"\n📦 PROCESSING BATCH {batch_num + 1}/{total_batches}")
            print(f"Papers {start_idx + 1}-{end_idx} ({len(batch_papers)} papers)")
            
            self._process_breakthrough_focused_batch(batch_papers, batch_num + 1, batch_size)
            
            # Breakthrough progress update every batch
            self._print_breakthrough_progress(end_idx)
            
            # Storage monitoring
            if (batch_num + 1) % max(1, total_batches // 10) == 0:
                print(f"\n📊 Storage status after batch {batch_num + 1}:")
                storage_ok = self.storage.monitor_storage()
                if not storage_ok:
                    print("⚠️ Storage cleanup...")
                    self.storage.cleanup_storage(target_free_gb=2.0)
        
        # Final breakthrough analysis
        self._finalize_breakthrough_analysis()
        
        return self.batch_results
    
    def _process_breakthrough_focused_batch(self, papers: List[Dict], batch_num: int, batch_size: int):
        """Process batch with breakthrough focus"""
        
        batch_start_time = time.time()
        batch_breakthroughs = []
        
        for i, paper in enumerate(papers):
            paper_num = (batch_num - 1) * batch_size + i + 1
            
            try:
                # Process through enhanced pipeline
                success, quality_score, ranked_breakthrough = self._process_enhanced_single_paper(paper, paper_num)
                
                if success and ranked_breakthrough:
                    self.batch_results['successful_papers'] += 1
                    
                    # Categorize breakthrough by quality
                    self._categorize_breakthrough(ranked_breakthrough, quality_score)
                    batch_breakthroughs.append(ranked_breakthrough)
                    
                self.batch_results['papers_processed'] += 1
                
            except Exception as e:
                print(f"   💥 Error processing paper {paper_num}: {e}")
                self.batch_results['processing_errors'].append({
                    'paper_id': paper.get('arxiv_id', 'unknown'),
                    'paper_num': paper_num,
                    'error': str(e)
                })
        
        batch_time = time.time() - batch_start_time
        
        # Batch breakthrough summary
        batch_high_quality = len([b for b in batch_breakthroughs if b.breakthrough_score.overall_score >= 0.7])
        batch_commercial = len([b for b in batch_breakthroughs if b.breakthrough_score.overall_score >= 0.6])
        
        print(f"📊 Batch {batch_num} Results ({batch_time:.1f}s):")
        print(f"   🏆 High-quality breakthroughs: {batch_high_quality}")
        print(f"   💰 Commercial opportunities: {batch_commercial}")
        print(f"   📈 Success rate: {len(batch_breakthroughs)/len(papers)*100:.1f}%")
    
    def _categorize_breakthrough(self, breakthrough: object, quality_score: float):
        """Categorize breakthrough by quality tier"""
        
        if quality_score >= 0.7:
            self.breakthrough_metrics['high_quality_breakthroughs'].append(breakthrough)
            self.breakthrough_metrics['high_quality_count'] += 1
        
        if quality_score >= 0.6:
            self.breakthrough_metrics['commercial_opportunities'].append(breakthrough)
            self.breakthrough_metrics['commercial_count'] += 1
        
        if quality_score >= 0.5:
            self.breakthrough_metrics['promising_discoveries'].append(breakthrough)
            self.breakthrough_metrics['promising_count'] += 1
        
        if quality_score >= 0.4:
            self.breakthrough_metrics['baseline_discoveries'].append(breakthrough)
            self.breakthrough_metrics['baseline_count'] += 1
    
    def _print_breakthrough_progress(self, papers_processed: int):
        """Print breakthrough-focused progress update"""
        
        print(f"\n🏆 BREAKTHROUGH PROGRESS - {papers_processed}/{self.paper_count} papers")
        print(f"=" * 60)
        
        # Core breakthrough metrics
        hq_count = self.breakthrough_metrics['high_quality_count']
        commercial_count = self.breakthrough_metrics['commercial_count']
        promising_count = self.breakthrough_metrics['promising_count']
        
        print(f"🎯 HIGH-QUALITY BREAKTHROUGHS (≥0.7): {hq_count}")
        print(f"💰 COMMERCIAL OPPORTUNITIES (≥0.6): {commercial_count}")
        print(f"📊 PROMISING DISCOVERIES (≥0.5): {promising_count}")
        
        # Discovery rates per 100 papers
        hq_rate = (hq_count / papers_processed) * 100
        commercial_rate = (commercial_count / papers_processed) * 100
        
        print(f"\n📈 DISCOVERY RATES:")
        print(f"   High-quality per 100 papers: {hq_rate:.1f}")
        print(f"   Commercial per 100 papers: {commercial_rate:.1f}")
        
        # Cost efficiency
        if hq_count > 0:
            cost_per_hq = (papers_processed * 0.327) / hq_count
            print(f"   Cost per high-quality breakthrough: ${cost_per_hq:.2f}")
        
        if commercial_count > 0:
            cost_per_commercial = (papers_processed * 0.327) / commercial_count
            print(f"   Cost per commercial opportunity: ${cost_per_commercial:.2f}")
        
        # Success assessment
        expected_hq = max(1, papers_processed // 200)  # Expect 1 per 200 papers minimum
        if hq_count >= expected_hq:
            print(f"✅ ON TRACK: {hq_count} ≥ {expected_hq} expected high-quality breakthroughs")
        else:
            print(f"⚠️ BELOW TARGET: {hq_count} < {expected_hq} expected high-quality breakthroughs")
    
    def _finalize_breakthrough_analysis(self):
        """Finalize with breakthrough-focused analysis"""
        
        self.batch_results['end_time'] = time.time()
        processing_time = self.batch_results['end_time'] - self.batch_results['start_time']
        
        print(f"\n🏆 BREAKTHROUGH-FOCUSED ANALYSIS COMPLETE")
        print(f"=" * 80)
        
        # Calculate final breakthrough metrics
        self._calculate_breakthrough_metrics()
        
        # Print comprehensive breakthrough results
        self._print_breakthrough_results()
        
        # Generate scaling recommendation
        self._generate_scaling_recommendation()
        
        # Save breakthrough-focused results
        self._save_breakthrough_results()
    
    def _calculate_breakthrough_metrics(self):
        """Calculate comprehensive breakthrough metrics"""
        
        total_cost = self.paper_count * 0.327
        
        # Cost efficiency metrics
        if self.breakthrough_metrics['high_quality_count'] > 0:
            self.breakthrough_metrics['cost_per_high_quality_breakthrough'] = (
                total_cost / self.breakthrough_metrics['high_quality_count']
            )
        
        if self.breakthrough_metrics['commercial_count'] > 0:
            self.breakthrough_metrics['cost_per_commercial_opportunity'] = (
                total_cost / self.breakthrough_metrics['commercial_count']
            )
        
        # Discovery rates per 100 papers
        papers_processed = self.batch_results['papers_processed']
        self.breakthrough_metrics['breakthrough_discovery_rate'] = (
            self.breakthrough_metrics['high_quality_count'] / papers_processed
        ) * 100
        
        self.breakthrough_metrics['commercial_discovery_rate'] = (
            self.breakthrough_metrics['commercial_count'] / papers_processed
        ) * 100
    
    def _print_breakthrough_results(self):
        """Print comprehensive breakthrough results"""
        
        print(f"🎯 PRIMARY SUCCESS METRIC:")
        print(f"   HIGH-QUALITY BREAKTHROUGHS (≥0.7): {self.breakthrough_metrics['high_quality_count']}")
        
        if self.breakthrough_metrics['high_quality_count'] > 0:
            print(f"   💵 Cost per breakthrough: ${self.breakthrough_metrics['cost_per_high_quality_breakthrough']:.2f}")
            print(f"   📈 Discovery rate: {self.breakthrough_metrics['breakthrough_discovery_rate']:.2f} per 100 papers")
        
        print(f"\n💰 SECONDARY SUCCESS METRICS:")
        print(f"   COMMERCIAL OPPORTUNITIES (≥0.6): {self.breakthrough_metrics['commercial_count']}")
        print(f"   PROMISING DISCOVERIES (≥0.5): {self.breakthrough_metrics['promising_count']}")
        
        if self.breakthrough_metrics['commercial_count'] > 0:
            print(f"   💵 Cost per commercial opportunity: ${self.breakthrough_metrics['cost_per_commercial_opportunity']:.2f}")
            print(f"   📈 Commercial discovery rate: {self.breakthrough_metrics['commercial_discovery_rate']:.2f} per 100 papers")
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   Papers processed: {self.batch_results['papers_processed']}")
        print(f"   Total cost: ${self.breakthrough_metrics['total_processing_cost']:.2f}")
        print(f"   Processing time: {(self.batch_results['end_time'] - self.batch_results['start_time'])/3600:.2f} hours")
        
        # Show top breakthroughs
        if self.breakthrough_metrics['high_quality_breakthroughs']:
            print(f"\n🏆 TOP HIGH-QUALITY BREAKTHROUGHS:")
            for i, breakthrough in enumerate(self.breakthrough_metrics['high_quality_breakthroughs'][:3], 1):
                print(f"   #{i}: Score {breakthrough.breakthrough_score.overall_score:.3f}")
                print(f"       {breakthrough.discovery_description[:70]}...")
    
    def _generate_scaling_recommendation(self):
        """Generate scaling recommendation based on breakthrough results"""
        
        hq_count = self.breakthrough_metrics['high_quality_count']
        commercial_count = self.breakthrough_metrics['commercial_count']
        cost = self.breakthrough_metrics['total_processing_cost']
        
        # Define scaling thresholds
        if self.paper_count == 100:
            hq_threshold = 1
            commercial_threshold = 3
        elif self.paper_count == 1000:
            hq_threshold = 5
            commercial_threshold = 15
        elif self.paper_count == 10000:
            hq_threshold = 50
            commercial_threshold = 150
        else:
            hq_threshold = max(1, self.paper_count // 200)
            commercial_threshold = max(3, self.paper_count // 67)
        
        print(f"\n🎯 SCALING RECOMMENDATION:")
        print(f"=" * 60)
        
        if hq_count >= hq_threshold and commercial_count >= commercial_threshold:
            next_scale = self.paper_count * 10
            next_cost = next_scale * 0.327
            expected_hq = hq_count * 10
            expected_commercial = commercial_count * 10
            
            print(f"✅ FULL GO: Proceed to {next_scale:,} papers")
            print(f"   💰 Investment: ${next_cost:,.2f}")
            print(f"   🎯 Expected high-quality breakthroughs: {expected_hq}")
            print(f"   💰 Expected commercial opportunities: {expected_commercial}")
            print(f"   📈 ROI potential: ${expected_commercial * 100000:,.0f}+ value")
            
        elif hq_count >= max(1, hq_threshold // 2):
            print(f"⚠️ CONDITIONAL GO: Improve quality then scale")
            print(f"   Current: {hq_count} high-quality (target: {hq_threshold})")
            print(f"   Recommendation: Enhance quality filtering first")
            print(f"   Then proceed to {self.paper_count * 5:,} papers (conservative scaling)")
            
        else:
            print(f"❌ QUALITY ENHANCEMENT NEEDED")
            print(f"   Current: {hq_count} high-quality (target: {hq_threshold})")
            print(f"   Recommendation: Implement quality improvements before scaling")
            print(f"   Focus on relevance filtering and confidence calibration")
    
    def _save_breakthrough_results(self):
        """Save breakthrough-focused results"""
        
        # Combine all results
        comprehensive_results = {
            **self.batch_results,
            'breakthrough_metrics': self.breakthrough_metrics,
            'test_parameters': {
                'paper_count': self.paper_count,
                'focus_metric': 'high_quality_breakthrough_count',
                'success_threshold': 0.7,
                'commercial_threshold': 0.6
            }
        }
        
        # Save comprehensive results
        results_file = self.storage.save_json_compressed(
            comprehensive_results,
            f"breakthrough_focused_results_{self.paper_count}_papers",
            "results"
        )
        
        # Save breakthrough summary
        summary = {
            'test_name': f'{self.paper_count}-Paper Breakthrough-Focused Test',
            'completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'high_quality_breakthroughs': self.breakthrough_metrics['high_quality_count'],
            'commercial_opportunities': self.breakthrough_metrics['commercial_count'],
            'total_cost': self.breakthrough_metrics['total_processing_cost'],
            'cost_per_breakthrough': self.breakthrough_metrics.get('cost_per_high_quality_breakthrough', 0),
            'breakthrough_discovery_rate': self.breakthrough_metrics['breakthrough_discovery_rate'],
            'scaling_recommendation': self._get_scaling_verdict()
        }
        
        summary_file = self.storage.save_json_compressed(
            summary,
            f"{self.paper_count}_paper_breakthrough_summary",
            "results"
        )
        
        print(f"\n💾 BREAKTHROUGH RESULTS SAVED:")
        print(f"   Comprehensive results: {results_file}")
        print(f"   Breakthrough summary: {summary_file}")
    
    def _get_scaling_verdict(self) -> str:
        """Get scaling verdict for summary"""
        
        hq_count = self.breakthrough_metrics['high_quality_count']
        
        if self.paper_count <= 100:
            threshold = 1
        elif self.paper_count <= 1000:
            threshold = 5
        else:
            threshold = self.paper_count // 200
        
        if hq_count >= threshold:
            return f"PROCEED: {hq_count} high-quality breakthroughs exceed {threshold} threshold"
        else:
            return f"ENHANCE: {hq_count} high-quality breakthroughs below {threshold} threshold"

def main():
    """Execute breakthrough-focused test"""
    
    print("🏆 BREAKTHROUGH-FOCUSED BATCH PROCESSOR")
    print("=" * 60)
    print("Primary metric: High-quality breakthrough count (score ≥0.7)")
    
    # Default to 1000 papers for scaling test
    paper_count = 1000
    print(f"Processing {paper_count} papers for breakthrough discovery scaling test...")
    
    processor = BreakthroughFocusedProcessor(paper_count)
    
    # Load papers
    processor.papers = processor.load_papers_for_scale_test(paper_count)
    
    if not processor.papers:
        print("❌ No papers loaded. Cannot proceed.")
        return
    
    # Execute breakthrough-focused test
    results = processor.run_breakthrough_focused_test()
    
    print(f"\n🎉 {paper_count}-PAPER BREAKTHROUGH TEST COMPLETE!")
    print(f"Check results in breakthrough_focused_{paper_count}_papers/results/")
    
    return results

if __name__ == "__main__":
    main()