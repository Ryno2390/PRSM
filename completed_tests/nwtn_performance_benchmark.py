#!/usr/bin/env python3
"""
NWTN Performance Benchmark
==========================

Comprehensive performance benchmark of NWTN with full corpus access,
measuring speed, accuracy, and scalability metrics.
"""

import asyncio
import json
import sys
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
from enhanced_semantic_search import EnhancedSemanticSearchEngine, SearchQuery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NWTNPerformanceBenchmark:
    """Comprehensive performance benchmark for NWTN with full corpus"""
    
    def __init__(self):
        self.meta_engine = None
        self.semantic_engine = None
        self.benchmark_results = []
        
        # Performance test scenarios
        self.benchmark_scenarios = [
            {
                "name": "Simple Query",
                "query": "machine learning algorithms",
                "papers_requested": 5,
                "expected_time": 2.0,
                "complexity": "low"
            },
            {
                "name": "Complex Cross-Domain Query",
                "query": "quantum computing applications in machine learning optimization",
                "papers_requested": 10,
                "expected_time": 5.0,
                "complexity": "medium"
            },
            {
                "name": "Multi-Disciplinary Analysis",
                "query": "interdisciplinary approaches combining biology, physics, and computer science for protein structure prediction",
                "papers_requested": 15,
                "expected_time": 8.0,
                "complexity": "high"
            },
            {
                "name": "High-Volume Retrieval",
                "query": "neural networks deep learning",
                "papers_requested": 25,
                "expected_time": 10.0,
                "complexity": "high"
            },
            {
                "name": "Specialized Domain Query",
                "query": "topological quantum error correction codes",
                "papers_requested": 8,
                "expected_time": 4.0,
                "complexity": "medium"
            }
        ]
    
    async def initialize_systems(self):
        """Initialize NWTN and semantic search systems"""
        logger.info("⚡ NWTN PERFORMANCE BENCHMARK")
        logger.info("=" * 60)
        logger.info("🎯 Testing speed, accuracy, and scalability")
        logger.info("📊 Full corpus: 151,120 papers")
        logger.info("🔍 HNSW semantic search index")
        logger.info("=" * 60)
        
        # Initialize NWTN
        self.meta_engine = MetaReasoningEngine()
        await self.meta_engine.initialize()
        logger.info("✅ NWTN Meta-Reasoning Engine initialized")
        
        # Initialize semantic search
        self.semantic_engine = EnhancedSemanticSearchEngine(
            index_dir="/Volumes/My Passport/PRSM_Storage/PRSM_Indices",
            index_type="HNSW"
        )
        
        if not self.semantic_engine.initialize():
            logger.error("❌ Failed to initialize semantic search")
            return False
        
        logger.info("✅ Semantic search engine initialized")
        return True
    
    async def benchmark_scenario(self, scenario: Dict, run_number: int = 1) -> Dict:
        """Benchmark a specific scenario"""
        logger.info(f"🏃 Run {run_number}: {scenario['name']}")
        
        overall_start = time.time()
        
        # Measure semantic search performance
        search_start = time.time()
        search_query = SearchQuery(
            query_text=scenario['query'],
            max_results=scenario['papers_requested'],
            similarity_threshold=0.1
        )
        
        papers = await self.semantic_engine.search(search_query)
        search_time = time.time() - search_start
        
        # Measure reasoning performance
        reasoning_start = time.time()
        
        reasoning_context = {
            'query': scenario['query'],
            'papers': [
                {
                    'id': p.paper_id,
                    'title': p.title,
                    'abstract': p.abstract,
                    'domain': p.domain,
                    'relevance_score': p.similarity_score
                } for p in papers
            ],
            'corpus_size': 151120
        }
        
        try:
            meta_result = await self.meta_engine.reason(
                query=scenario['query'],
                context=reasoning_context,
                thinking_mode=ThinkingMode.INTERMEDIATE
            )
            
            reasoning_time = time.time() - reasoning_start
            total_time = time.time() - overall_start
            
            # Calculate performance metrics
            papers_per_second = len(papers) / search_time if search_time > 0 else 0
            avg_relevance = sum(p.similarity_score for p in papers) / len(papers) if papers else 0
            
            result = {
                'scenario': scenario['name'],
                'run_number': run_number,
                'success': True,
                'total_time': total_time,
                'search_time': search_time,
                'reasoning_time': reasoning_time,
                'papers_retrieved': len(papers),
                'papers_requested': scenario['papers_requested'],
                'papers_per_second': papers_per_second,
                'avg_relevance': avg_relevance,
                'meta_confidence': meta_result.meta_confidence,
                'reasoning_engines_used': len(meta_result.reasoning_engines_used.keys()) if hasattr(meta_result, 'reasoning_engines_used') else 0,
                'complexity': scenario['complexity'],
                'expected_time': scenario['expected_time'],
                'performance_ratio': scenario['expected_time'] / total_time if total_time > 0 else 0,
                'query_length': len(scenario['query'])
            }
            
            logger.info(f"⏱️  {total_time:.2f}s (search: {search_time:.2f}s, reasoning: {reasoning_time:.2f}s)")
            logger.info(f"📄 {len(papers)} papers, avg relevance: {avg_relevance:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            return {
                'scenario': scenario['name'],
                'run_number': run_number,
                'success': False,
                'error': str(e),
                'total_time': time.time() - overall_start
            }
    
    async def run_comprehensive_benchmark(self, runs_per_scenario: int = 3) -> Dict:
        """Run comprehensive performance benchmark"""
        logger.info(f"🚀 Starting comprehensive benchmark ({runs_per_scenario} runs per scenario)")
        
        benchmark_start = time.time()
        
        # Run each scenario multiple times
        for scenario in self.benchmark_scenarios:
            logger.info(f"\n📊 Benchmarking: {scenario['name']}")
            
            scenario_results = []
            for run in range(runs_per_scenario):
                result = await self.benchmark_scenario(scenario, run + 1)
                scenario_results.append(result)
                self.benchmark_results.append(result)
                
                # Brief pause between runs
                await asyncio.sleep(0.1)
            
            # Calculate scenario statistics
            successful_runs = [r for r in scenario_results if r['success']]
            if successful_runs:
                avg_time = statistics.mean(r['total_time'] for r in successful_runs)
                std_time = statistics.stdev(r['total_time'] for r in successful_runs) if len(successful_runs) > 1 else 0
                avg_confidence = statistics.mean(r['meta_confidence'] for r in successful_runs)
                
                logger.info(f"📈 Scenario average: {avg_time:.2f}s ± {std_time:.2f}s")
                logger.info(f"🎯 Average confidence: {avg_confidence:.3f}")
        
        total_benchmark_time = time.time() - benchmark_start
        
        # Generate comprehensive report
        report = self.generate_benchmark_report(total_benchmark_time, runs_per_scenario)
        return report
    
    def generate_benchmark_report(self, total_time: float, runs_per_scenario: int) -> Dict:
        """Generate comprehensive benchmark report"""
        successful_results = [r for r in self.benchmark_results if r['success']]
        
        if not successful_results:
            return {'error': 'No successful benchmark runs'}
        
        # Overall performance metrics
        total_runs = len(successful_results)
        avg_total_time = statistics.mean(r['total_time'] for r in successful_results)
        avg_search_time = statistics.mean(r['search_time'] for r in successful_results)
        avg_reasoning_time = statistics.mean(r['reasoning_time'] for r in successful_results)
        avg_papers_retrieved = statistics.mean(r['papers_retrieved'] for r in successful_results)
        avg_relevance = statistics.mean(r['avg_relevance'] for r in successful_results)
        avg_confidence = statistics.mean(r['meta_confidence'] for r in successful_results)
        
        # Performance by complexity
        complexity_stats = {}
        for complexity in ['low', 'medium', 'high']:
            complexity_results = [r for r in successful_results if r['complexity'] == complexity]
            if complexity_results:
                complexity_stats[complexity] = {
                    'count': len(complexity_results),
                    'avg_time': statistics.mean(r['total_time'] for r in complexity_results),
                    'avg_confidence': statistics.mean(r['meta_confidence'] for r in complexity_results),
                    'avg_papers': statistics.mean(r['papers_retrieved'] for r in complexity_results)
                }
        
        # Speed metrics
        total_papers_processed = sum(r['papers_retrieved'] for r in successful_results)
        papers_per_second = total_papers_processed / sum(r['total_time'] for r in successful_results)
        
        # Throughput calculation
        queries_per_minute = (total_runs / total_time) * 60
        
        report = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'total_scenarios': len(self.benchmark_scenarios),
            'runs_per_scenario': runs_per_scenario,
            'total_runs': total_runs,
            'successful_runs': len(successful_results),
            'success_rate': len(successful_results) / len(self.benchmark_results) * 100,
            'total_benchmark_time': total_time,
            'performance_metrics': {
                'avg_total_time': avg_total_time,
                'avg_search_time': avg_search_time,
                'avg_reasoning_time': avg_reasoning_time,
                'avg_papers_retrieved': avg_papers_retrieved,
                'avg_relevance': avg_relevance,
                'avg_confidence': avg_confidence,
                'papers_per_second': papers_per_second,
                'queries_per_minute': queries_per_minute
            },
            'complexity_analysis': complexity_stats,
            'detailed_results': self.benchmark_results,
            'system_performance': {
                'corpus_size': 151120,
                'index_type': 'HNSW',
                'reasoning_engines': 7,
                'scalability_rating': self.calculate_scalability_rating(avg_total_time, papers_per_second)
            }
        }
        
        return report
    
    def calculate_scalability_rating(self, avg_time: float, papers_per_second: float) -> str:
        """Calculate scalability rating"""
        if avg_time < 3.0 and papers_per_second > 5:
            return "EXCELLENT"
        elif avg_time < 5.0 and papers_per_second > 3:
            return "GOOD"
        elif avg_time < 10.0 and papers_per_second > 1:
            return "FAIR"
        else:
            return "POOR"
    
    def print_benchmark_summary(self, report: Dict):
        """Print comprehensive benchmark summary"""
        print("\n⚡ NWTN PERFORMANCE BENCHMARK RESULTS")
        print("=" * 80)
        print(f"📅 Benchmark Date: {report['benchmark_timestamp']}")
        print(f"🧪 Total Scenarios: {report['total_scenarios']}")
        print(f"🔄 Runs per Scenario: {report['runs_per_scenario']}")
        print(f"✅ Success Rate: {report['success_rate']:.1f}%")
        print(f"⏱️  Total Time: {report['total_benchmark_time']:.1f}s")
        print()
        
        print("🎯 PERFORMANCE METRICS:")
        print("-" * 40)
        metrics = report['performance_metrics']
        print(f"⏱️  Average Total Time: {metrics['avg_total_time']:.2f}s")
        print(f"🔍 Average Search Time: {metrics['avg_search_time']:.2f}s")
        print(f"🧠 Average Reasoning Time: {metrics['avg_reasoning_time']:.2f}s")
        print(f"📄 Average Papers Retrieved: {metrics['avg_papers_retrieved']:.1f}")
        print(f"🎯 Average Relevance: {metrics['avg_relevance']:.3f}")
        print(f"💪 Average Confidence: {metrics['avg_confidence']:.3f}")
        print(f"🚀 Papers per Second: {metrics['papers_per_second']:.1f}")
        print(f"📊 Queries per Minute: {metrics['queries_per_minute']:.1f}")
        print()
        
        print("📊 COMPLEXITY ANALYSIS:")
        print("-" * 40)
        for complexity, stats in report['complexity_analysis'].items():
            print(f"{complexity.upper()}: {stats['avg_time']:.2f}s avg, {stats['avg_confidence']:.3f} confidence")
        print()
        
        print("🎖️  SYSTEM PERFORMANCE:")
        print("-" * 40)
        sys_perf = report['system_performance']
        print(f"📚 Corpus Size: {sys_perf['corpus_size']:,} papers")
        print(f"🔍 Index Type: {sys_perf['index_type']}")
        print(f"⚙️  Reasoning Engines: {sys_perf['reasoning_engines']}")
        print(f"📈 Scalability Rating: {sys_perf['scalability_rating']}")
        print("=" * 80)

async def main():
    """Main benchmark function"""
    benchmark = NWTNPerformanceBenchmark()
    
    # Initialize systems
    if not await benchmark.initialize_systems():
        logger.error("❌ Failed to initialize systems")
        return
    
    # Run comprehensive benchmark
    report = await benchmark.run_comprehensive_benchmark(runs_per_scenario=3)
    
    # Print summary
    benchmark.print_benchmark_summary(report)
    
    # Save report
    report_file = f"nwtn_performance_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📄 Benchmark report saved to: {report_file}")
    logger.info("🎉 NWTN Performance Benchmark Complete!")

if __name__ == "__main__":
    asyncio.run(main())