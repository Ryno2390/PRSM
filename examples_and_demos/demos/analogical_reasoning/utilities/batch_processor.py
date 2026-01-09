#!/usr/bin/env python3
"""
100-Paper Batch Processor
Processes 100 scientific papers through the complete NWTN analogical reasoning pipeline

This demonstrates genuine breakthrough discovery capability by:
1. Processing 100 real scientific papers from arXiv
2. Extracting ~260 patterns from authentic research literature
3. Generating cross-domain analogical mappings for innovation
4. Validating NWTN's scalability and discovery potential
"""

import json
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import traceback

# Import our validated pipeline components
from pipeline_tester import PipelineTester
from storage_manager import StorageManager

class BatchProcessor:
    """Processes 100 papers through the complete NWTN pipeline"""
    
    def __init__(self):
        self.storage = StorageManager("batch_processing_storage")
        self.pipeline_tester = PipelineTester()
        
        # Batch processing metrics
        self.batch_results = {
            'start_time': None,
            'end_time': None,
            'papers_processed': 0,
            'successful_papers': 0,
            'total_socs': 0,
            'total_patterns': 0,
            'total_mappings': 0,
            'total_discoveries': 0,
            'processing_errors': [],
            'performance_metrics': {},
            'breakthrough_discoveries': [],
            'pattern_catalog': {},
            'discovery_confidence_scores': []
        }
        
        # Load the selected papers
        self.papers = self._load_selected_papers()
        
    def _load_selected_papers(self) -> List[Dict]:
        """Load the 100 selected papers"""
        
        try:
            with open('selected_papers.json', 'r') as f:
                papers = json.load(f)
            
            print(f"✅ Loaded {len(papers)} papers for batch processing")
            return papers[:100]  # Ensure exactly 100 papers
            
        except Exception as e:
            print(f"❌ Error loading papers: {e}")
            return []
    
    def run_batch_processing(self) -> Dict:
        """Execute the complete 100-paper batch processing test"""
        
        print(f"\n🚀 STARTING 100-PAPER BATCH PROCESSING")
        print(f"=" * 70)
        print(f"Processing {len(self.papers)} papers through complete NWTN pipeline")
        print(f"Expected patterns: ~{len(self.papers) * 2.6:.0f}")
        print(f"Expected cost: ${len(self.papers) * 0.327:.2f}")
        
        self.batch_results['start_time'] = time.time()
        
        # Initial storage check
        print(f"\n📊 Initial storage status:")
        self.storage.monitor_storage()
        
        # Process papers in batches for memory management
        batch_size = 20  # Process 20 papers at a time
        total_batches = (len(self.papers) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(self.papers))
            batch_papers = self.papers[start_idx:end_idx]
            
            print(f"\n📦 PROCESSING BATCH {batch_num + 1}/{total_batches}")
            print(f"Papers {start_idx + 1}-{end_idx} ({len(batch_papers)} papers)")
            
            self._process_paper_batch(batch_papers, batch_num + 1)
            
            # Storage monitoring every 2 batches
            if (batch_num + 1) % 2 == 0:
                print(f"\n📊 Storage status after batch {batch_num + 1}:")
                storage_ok = self.storage.monitor_storage()
                
                if not storage_ok:
                    print("⚠️ Storage getting low, running cleanup...")
                    self.storage.cleanup_storage(target_free_gb=1.0)
        
        # Final processing and analysis
        self._finalize_batch_processing()
        
        return self.batch_results
    
    def _process_paper_batch(self, papers: List[Dict], batch_num: int):
        """Process a batch of papers"""
        
        batch_start_time = time.time()
        
        for i, paper in enumerate(papers):
            paper_num = (batch_num - 1) * 20 + i + 1
            
            print(f"\n📄 PAPER {paper_num}/100")
            print(f"Title: {paper['title'][:50]}...")
            print(f"arXiv ID: {paper['arxiv_id']}")
            
            try:
                # Process through complete pipeline
                success = self._process_single_paper(paper, paper_num)
                
                if success:
                    self.batch_results['successful_papers'] += 1
                    print(f"   ✅ Paper {paper_num} completed successfully")
                else:
                    print(f"   ❌ Paper {paper_num} failed processing")
                
                self.batch_results['papers_processed'] += 1
                
                # Progress update every 10 papers
                if paper_num % 10 == 0:
                    self._print_progress_update(paper_num)
                
            except Exception as e:
                print(f"   💥 Unexpected error processing paper {paper_num}: {e}")
                self.batch_results['processing_errors'].append({
                    'paper_id': paper['arxiv_id'],
                    'paper_num': paper_num,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
        
        batch_time = time.time() - batch_start_time
        print(f"\n📊 Batch {batch_num} completed in {batch_time:.1f}s")
    
    def _process_single_paper(self, paper: Dict, paper_num: int) -> bool:
        """Process a single paper through the complete pipeline"""
        
        try:
            # Step 1: Content ingestion
            success, content_result, content = self.pipeline_tester.test_content_ingestion(paper)
            if not success:
                return False
            
            # Step 2: SOC extraction
            success, soc_result, socs = self.pipeline_tester.test_soc_extraction(paper, content)
            if not success:
                return False
            
            self.batch_results['total_socs'] += len(socs)
            
            # Step 3: Pattern extraction
            success, pattern_result, patterns = self.pipeline_tester.test_pattern_extraction(paper, socs)
            if not success:
                return False
            
            # Count patterns
            if isinstance(patterns, dict):
                pattern_count = sum(len(pattern_list) for pattern_list in patterns.values())
                self.batch_results['total_patterns'] += pattern_count
                
                # Store patterns in catalog
                self.batch_results['pattern_catalog'][paper['arxiv_id']] = {
                    'paper_title': paper['title'],
                    'pattern_count': pattern_count,
                    'pattern_types': {k: len(v) for k, v in patterns.items()}
                }
            
            # Step 4: Cross-domain mapping
            success, mapping_count = self.pipeline_tester.test_cross_domain_mapping(patterns)
            if success:
                self.batch_results['total_mappings'] += mapping_count
                
                # Count as discovery if mappings found
                if mapping_count > 0:
                    self.batch_results['total_discoveries'] += 1
            
            return True
            
        except Exception as e:
            print(f"      ❌ Pipeline error: {e}")
            return False
    
    def _print_progress_update(self, paper_num: int):
        """Print progress update"""
        
        success_rate = (self.batch_results['successful_papers'] / paper_num) * 100
        avg_patterns = self.batch_results['total_patterns'] / max(1, self.batch_results['successful_papers'])
        avg_mappings = self.batch_results['total_mappings'] / max(1, self.batch_results['successful_papers'])
        
        print(f"\n📈 PROGRESS UPDATE - {paper_num}/100 papers")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Total SOCs: {self.batch_results['total_socs']}")
        print(f"   Total patterns: {self.batch_results['total_patterns']}")
        print(f"   Total mappings: {self.batch_results['total_mappings']}")
        print(f"   Papers with discoveries: {self.batch_results['total_discoveries']}")
        print(f"   Avg patterns/paper: {avg_patterns:.1f}")
        print(f"   Avg mappings/paper: {avg_mappings:.1f}")
    
    def _finalize_batch_processing(self):
        """Finalize batch processing and generate comprehensive results"""
        
        self.batch_results['end_time'] = time.time()
        processing_time = self.batch_results['end_time'] - self.batch_results['start_time']
        
        print(f"\n🎯 BATCH PROCESSING COMPLETE")
        print(f"=" * 70)
        
        # Calculate final metrics
        success_rate = (self.batch_results['successful_papers'] / 
                       self.batch_results['papers_processed']) * 100
        
        discovery_rate = (self.batch_results['total_discoveries'] / 
                         max(1, self.batch_results['successful_papers'])) * 100
        
        avg_patterns_per_paper = (self.batch_results['total_patterns'] / 
                                 max(1, self.batch_results['successful_papers']))
        
        avg_mappings_per_paper = (self.batch_results['total_mappings'] / 
                                 max(1, self.batch_results['successful_papers']))
        
        # Store performance metrics
        self.batch_results['performance_metrics'] = {
            'total_processing_time_seconds': processing_time,
            'processing_time_hours': processing_time / 3600,
            'papers_per_hour': self.batch_results['papers_processed'] / (processing_time / 3600),
            'success_rate_percent': success_rate,
            'discovery_rate_percent': discovery_rate,
            'avg_patterns_per_paper': avg_patterns_per_paper,
            'avg_mappings_per_paper': avg_mappings_per_paper,
            'patterns_per_hour': self.batch_results['total_patterns'] / (processing_time / 3600),
            'mappings_per_hour': self.batch_results['total_mappings'] / (processing_time / 3600)
        }
        
        # Print comprehensive results
        self._print_final_results()
        
        # Save results
        self._save_batch_results()
    
    def _print_final_results(self):
        """Print comprehensive final results"""
        
        metrics = self.batch_results['performance_metrics']
        
        print(f"📊 PROCESSING STATISTICS:")
        print(f"   Papers processed: {self.batch_results['papers_processed']}")
        print(f"   Successful papers: {self.batch_results['successful_papers']}")
        print(f"   Success rate: {metrics['success_rate_percent']:.1f}%")
        print(f"   Processing time: {metrics['processing_time_hours']:.2f} hours")
        print(f"   Papers per hour: {metrics['papers_per_hour']:.1f}")
        
        print(f"\n🔬 DISCOVERY STATISTICS:")
        print(f"   Total SOCs extracted: {self.batch_results['total_socs']}")
        print(f"   Total patterns generated: {self.batch_results['total_patterns']}")
        print(f"   Total cross-domain mappings: {self.batch_results['total_mappings']}")
        print(f"   Papers with discoveries: {self.batch_results['total_discoveries']}")
        print(f"   Discovery rate: {metrics['discovery_rate_percent']:.1f}%")
        
        print(f"\n📈 EFFICIENCY METRICS:")
        print(f"   Average patterns per paper: {metrics['avg_patterns_per_paper']:.1f}")
        print(f"   Average mappings per paper: {metrics['avg_mappings_per_paper']:.1f}")
        print(f"   Patterns generated per hour: {metrics['patterns_per_hour']:.0f}")
        print(f"   Mappings generated per hour: {metrics['mappings_per_hour']:.0f}")
        
        if self.batch_results['processing_errors']:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            print(f"   Total errors: {len(self.batch_results['processing_errors'])}")
            print(f"   Error rate: {(len(self.batch_results['processing_errors']) / self.batch_results['papers_processed']) * 100:.1f}%")
        
        # Final storage status
        print(f"\n📊 FINAL STORAGE STATUS:")
        self.storage.monitor_storage()
        
        # Success assessment
        if metrics['success_rate_percent'] >= 80:
            print(f"\n✅ BATCH PROCESSING: HIGHLY SUCCESSFUL")
            print(f"   Ready for 1K paper scaling!")
        elif metrics['success_rate_percent'] >= 60:
            print(f"\n⚠️ BATCH PROCESSING: SUCCESSFUL WITH IMPROVEMENTS NEEDED")
            print(f"   Consider optimizations before scaling")
        else:
            print(f"\n❌ BATCH PROCESSING: NEEDS SIGNIFICANT IMPROVEMENT")
            print(f"   Address critical issues before scaling")
    
    def _save_batch_results(self):
        """Save comprehensive batch results"""
        
        # Save detailed results
        results_file = self.storage.save_json_compressed(
            self.batch_results,
            "batch_processing_results_100_papers",
            "results"
        )
        
        # Save pattern catalog separately
        catalog_file = self.storage.save_json_compressed(
            self.batch_results['pattern_catalog'],
            "pattern_catalog_100_papers", 
            "results"
        )
        
        print(f"\n💾 RESULTS SAVED:")
        print(f"   Batch results: {results_file}")
        print(f"   Pattern catalog: {catalog_file}")
        
        # Create summary report
        summary = {
            'test_name': '100-Paper NWTN Batch Processing Test',
            'completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'papers_processed': self.batch_results['papers_processed'],
            'success_rate': self.batch_results['performance_metrics']['success_rate_percent'],
            'total_patterns': self.batch_results['total_patterns'],
            'total_mappings': self.batch_results['total_mappings'],
            'discovery_rate': self.batch_results['performance_metrics']['discovery_rate_percent'],
            'processing_time_hours': self.batch_results['performance_metrics']['processing_time_hours'],
            'recommendation': self._generate_recommendation()
        }
        
        summary_file = self.storage.save_json_compressed(
            summary,
            "100_paper_test_summary",
            "results"
        )
        print(f"   Test summary: {summary_file}")
    
    def _generate_recommendation(self) -> str:
        """Generate recommendation for next steps"""
        
        success_rate = self.batch_results['performance_metrics']['success_rate_percent']
        discovery_rate = self.batch_results['performance_metrics']['discovery_rate_percent']
        
        if success_rate >= 80 and discovery_rate >= 70:
            return "FULL GO: Proceed immediately to 1K paper processing"
        elif success_rate >= 70 and discovery_rate >= 50:
            return "CONDITIONAL GO: Minor optimizations then proceed to 1K papers"
        elif success_rate >= 60:
            return "ITERATE: Improve pipeline efficiency before scaling"
        else:
            return "REASSESS: Address fundamental issues before proceeding"

def main():
    """Execute 100-paper batch processing test"""
    
    processor = BatchProcessor()
    
    if not processor.papers:
        print("❌ No papers loaded. Cannot proceed with batch processing.")
        return
    
    if len(processor.papers) < 100:
        print(f"⚠️ Only {len(processor.papers)} papers available (need 100)")
        response = input("Proceed anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Execute batch processing
    results = processor.run_batch_processing()
    
    print(f"\n🎉 100-PAPER BATCH PROCESSING COMPLETE!")
    print(f"Check results in batch_processing_storage/results/")
    
    return results

if __name__ == "__main__":
    main()