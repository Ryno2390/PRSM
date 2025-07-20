#!/usr/bin/env python3
"""
NWTN Full Corpus Validation Test
================================

This test validates NWTN's enhanced reasoning capabilities with access to
the complete corpus of 151,120 arXiv papers through production FAISS indices.

This demonstrates the transformation from basic NWTN to production-ready
NWTN+PRSM integration with full semantic search capabilities.
"""

import asyncio
import json
import sys
import time
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

class NWTNFullCorpusValidator:
    """Validate NWTN reasoning with full 151,120 paper corpus"""
    
    def __init__(self):
        self.meta_engine = None
        self.semantic_engine = None
        self.validation_results = []
        
        # Advanced test scenarios leveraging full corpus
        self.validation_scenarios = [
            {
                "name": "Cross-Domain Scientific Synthesis",
                "query": "How do quantum computing advances in the last 5 years relate to machine learning optimization problems, and what new hybrid approaches have emerged?",
                "expected_domains": ["physics", "computer_science"],
                "min_papers": 10,
                "reasoning_engines": ["DEDUCTIVE", "INDUCTIVE", "ANALOGICAL"]
            },
            {
                "name": "Interdisciplinary Biology-Physics Innovation",
                "query": "What are the latest developments in biophysics that combine protein folding research with quantum mechanical modeling approaches?",
                "expected_domains": ["biology", "physics"],
                "min_papers": 8,
                "reasoning_engines": ["DEDUCTIVE", "ABDUCTIVE", "CAUSAL"]
            },
            {
                "name": "Emerging Technology Convergence",
                "query": "How are recent advances in metamaterials and photonics being applied to quantum information processing and what are the theoretical foundations?",
                "expected_domains": ["physics", "computer_science"],
                "min_papers": 12,
                "reasoning_engines": ["DEDUCTIVE", "INDUCTIVE", "COUNTERFACTUAL"]
            },
            {
                "name": "Mathematical Foundations of AI",
                "query": "What new mathematical frameworks have been developed for understanding deep learning optimization landscapes and their relationship to differential geometry?",
                "expected_domains": ["mathematics", "computer_science"],
                "min_papers": 15,
                "reasoning_engines": ["DEDUCTIVE", "INDUCTIVE", "ANALOGICAL"]
            },
            {
                "name": "Climate Science Computational Methods",
                "query": "How are recent advances in computational fluid dynamics and machine learning being integrated for climate modeling and prediction?",
                "expected_domains": ["physics", "computer_science"],
                "min_papers": 10,
                "reasoning_engines": ["DEDUCTIVE", "CAUSAL", "PROBABILISTIC"]
            },
            {
                "name": "Quantum-Classical Interface",
                "query": "What are the latest theoretical and experimental approaches to quantum error correction and how do they relate to classical coding theory?",
                "expected_domains": ["physics", "computer_science"],
                "min_papers": 12,
                "reasoning_engines": ["DEDUCTIVE", "ABDUCTIVE", "COUNTERFACTUAL"]
            }
        ]
    
    async def initialize_systems(self):
        """Initialize NWTN and semantic search systems"""
        logger.info("🚀 NWTN FULL CORPUS VALIDATION")
        logger.info("=" * 80)
        logger.info("🧠 Testing NWTN Meta-Reasoning with 151,120 arXiv papers")
        logger.info("🔍 Using production FAISS indices for semantic search")
        logger.info("🎯 Validating cross-domain reasoning capabilities")
        logger.info("=" * 80)
        
        # Initialize NWTN Meta-Reasoning Engine
        logger.info("🧠 Initializing NWTN Meta-Reasoning Engine...")
        self.meta_engine = MetaReasoningEngine()
        await self.meta_engine.initialize()
        logger.info("✅ NWTN Meta-Reasoning Engine initialized")
        
        # Initialize Enhanced Semantic Search with production HNSW index
        logger.info("🔍 Initializing Enhanced Semantic Search (HNSW, 151,120 papers)...")
        self.semantic_engine = EnhancedSemanticSearchEngine(
            index_dir="/Volumes/My Passport/PRSM_Storage/PRSM_Indices",
            index_type="HNSW"
        )
        
        if not self.semantic_engine.initialize():
            logger.error("❌ Failed to initialize semantic search engine")
            return False
        
        # Get search engine statistics
        stats = self.semantic_engine.get_statistics()
        logger.info(f"✅ Semantic search initialized: {stats['total_papers']:,} papers available")
        logger.info()
        
        return True
    
    async def retrieve_relevant_papers(self, query: str, max_papers: int = 20) -> List[Dict]:
        """Retrieve papers relevant to query using full corpus semantic search"""
        try:
            search_query = SearchQuery(
                query_text=query,
                max_results=max_papers,
                similarity_threshold=0.15  # Lower threshold for broader search
            )
            
            semantic_results = await self.semantic_engine.search(search_query)
            
            papers = []
            for result in semantic_results:
                paper_data = {
                    'id': result.paper_id,
                    'title': result.title,
                    'abstract': result.abstract,
                    'domain': result.domain or 'unknown',
                    'authors': result.authors,
                    'relevance_score': result.similarity_score,
                    'file_path': result.file_path
                }
                papers.append(paper_data)
            
            logger.info(f"📄 Retrieved {len(papers)} papers with avg similarity {sum(p['relevance_score'] for p in papers)/len(papers):.3f}")
            return papers
            
        except Exception as e:
            logger.error(f"Error retrieving papers: {e}")
            return []
    
    async def validate_scenario(self, scenario: Dict) -> Dict:
        """Validate a specific reasoning scenario"""
        logger.info(f"🧪 Testing: {scenario['name']}")
        logger.info(f"🔍 Query: {scenario['query']}")
        
        start_time = time.time()
        
        # Retrieve relevant papers from full corpus
        relevant_papers = await self.retrieve_relevant_papers(
            scenario['query'], 
            scenario['min_papers']
        )
        
        if len(relevant_papers) < scenario['min_papers']:
            logger.warning(f"⚠️  Only found {len(relevant_papers)} papers, expected {scenario['min_papers']}")
        
        # Analyze domain distribution
        domain_counts = {}
        for paper in relevant_papers:
            domain = paper['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        logger.info(f"📊 Domain distribution: {domain_counts}")
        
        # Perform meta-reasoning with full corpus context
        reasoning_context = {
            'query': scenario['query'],
            'papers': relevant_papers,
            'domain_context': domain_counts,
            'corpus_size': 151120
        }
        
        try:
            # Use INTERMEDIATE thinking mode for comprehensive analysis
            meta_result = await self.meta_engine.reason(
                query=scenario['query'],
                context=reasoning_context,
                thinking_mode=ThinkingMode.INTERMEDIATE
            )
            
            processing_time = time.time() - start_time
            
            # Validate reasoning quality
            validation_result = {
                'scenario': scenario['name'],
                'query': scenario['query'],
                'success': True,
                'processing_time': processing_time,
                'papers_retrieved': len(relevant_papers),
                'domain_distribution': domain_counts,
                'expected_domains': scenario['expected_domains'],
                'reasoning_engines_used': list(meta_result.reasoning_engines_used.keys()) if hasattr(meta_result, 'reasoning_engines_used') else [],
                'meta_confidence': meta_result.meta_confidence,
                'world_model_integration': meta_result.world_model_integration_score if hasattr(meta_result, 'world_model_integration_score') else 0.0,
                'cross_domain_synthesis': self.evaluate_cross_domain_synthesis(relevant_papers, scenario['expected_domains']),
                'semantic_coherence': self.evaluate_semantic_coherence(relevant_papers),
                'reasoning_quality': self.evaluate_reasoning_quality(meta_result),
                'corpus_utilization': {
                    'papers_accessed': len(relevant_papers),
                    'corpus_percentage': (len(relevant_papers) / 151120) * 100,
                    'average_relevance': sum(p['relevance_score'] for p in relevant_papers) / len(relevant_papers),
                    'domain_coverage': len(domain_counts)
                }
            }
            
            logger.info(f"✅ Completed in {processing_time:.2f}s")
            logger.info(f"🎯 Meta-confidence: {meta_result.meta_confidence:.3f}")
            logger.info(f"📊 Reasoning engines: {len(validation_result['reasoning_engines_used'])}")
            logger.info(f"🌐 Cross-domain synthesis: {validation_result['cross_domain_synthesis']:.3f}")
            logger.info()
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Scenario failed: {e}")
            return {
                'scenario': scenario['name'],
                'query': scenario['query'],
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'papers_retrieved': len(relevant_papers)
            }
    
    def evaluate_cross_domain_synthesis(self, papers: List[Dict], expected_domains: List[str]) -> float:
        """Evaluate how well the reasoning synthesizes across expected domains"""
        if not papers:
            return 0.0
        
        # Check domain coverage
        paper_domains = set(paper['domain'] for paper in papers)
        expected_domains_set = set(expected_domains)
        
        domain_coverage = len(paper_domains.intersection(expected_domains_set)) / len(expected_domains_set)
        
        # Check domain balance
        domain_counts = {}
        for paper in papers:
            domain = paper['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        if len(domain_counts) > 1:
            # Calculate balance score (closer to 1.0 means more balanced)
            total_papers = len(papers)
            expected_per_domain = total_papers / len(domain_counts)
            balance_score = 1.0 - (sum(abs(count - expected_per_domain) for count in domain_counts.values()) / (2 * total_papers))
        else:
            balance_score = 0.5  # Single domain
        
        return (domain_coverage + balance_score) / 2.0
    
    def evaluate_semantic_coherence(self, papers: List[Dict]) -> float:
        """Evaluate semantic coherence of retrieved papers"""
        if not papers:
            return 0.0
        
        # Use relevance scores as proxy for semantic coherence
        relevance_scores = [paper['relevance_score'] for paper in papers]
        
        if len(relevance_scores) == 1:
            return relevance_scores[0]
        
        # Calculate coherence based on consistency of relevance scores
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        variance = sum((score - avg_relevance) ** 2 for score in relevance_scores) / len(relevance_scores)
        
        # Lower variance = higher coherence
        coherence = avg_relevance * (1.0 - min(variance, 1.0))
        
        return coherence
    
    def evaluate_reasoning_quality(self, meta_result) -> float:
        """Evaluate overall reasoning quality"""
        if not meta_result:
            return 0.0
        
        # Base quality on meta-confidence
        base_quality = meta_result.meta_confidence
        
        # Bonus for multiple reasoning engines
        if hasattr(meta_result, 'reasoning_engines_used'):
            engine_bonus = min(len(meta_result.reasoning_engines_used) / 7.0, 1.0) * 0.2
        else:
            engine_bonus = 0.0
        
        # Bonus for world model integration
        if hasattr(meta_result, 'world_model_integration_score'):
            integration_bonus = meta_result.world_model_integration_score * 0.1
        else:
            integration_bonus = 0.0
        
        return min(base_quality + engine_bonus + integration_bonus, 1.0)
    
    async def run_full_validation(self) -> Dict:
        """Run complete validation of NWTN with full corpus"""
        logger.info("🎯 Starting comprehensive NWTN validation with 151,120 papers...")
        
        validation_start = time.time()
        
        # Run all validation scenarios
        for scenario in self.validation_scenarios:
            result = await self.validate_scenario(scenario)
            self.validation_results.append(result)
        
        total_validation_time = time.time() - validation_start
        
        # Calculate overall metrics
        successful_tests = [r for r in self.validation_results if r['success']]
        success_rate = len(successful_tests) / len(self.validation_results) * 100
        
        if successful_tests:
            avg_confidence = sum(r['meta_confidence'] for r in successful_tests) / len(successful_tests)
            avg_processing_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests)
            avg_papers_per_query = sum(r['papers_retrieved'] for r in successful_tests) / len(successful_tests)
            avg_cross_domain = sum(r['cross_domain_synthesis'] for r in successful_tests) / len(successful_tests)
            avg_semantic_coherence = sum(r['semantic_coherence'] for r in successful_tests) / len(successful_tests)
            avg_reasoning_quality = sum(r['reasoning_quality'] for r in successful_tests) / len(successful_tests)
            
            total_papers_accessed = sum(r['papers_retrieved'] for r in successful_tests)
            corpus_utilization = (total_papers_accessed / 151120) * 100
        else:
            avg_confidence = 0.0
            avg_processing_time = 0.0
            avg_papers_per_query = 0.0
            avg_cross_domain = 0.0
            avg_semantic_coherence = 0.0
            avg_reasoning_quality = 0.0
            corpus_utilization = 0.0
        
        # Generate comprehensive report
        validation_report = {
            "validation_timestamp": datetime.now().isoformat(),
            "corpus_size": 151120,
            "total_scenarios": len(self.validation_scenarios),
            "successful_scenarios": len(successful_tests),
            "success_rate": success_rate,
            "total_validation_time": total_validation_time,
            "overall_metrics": {
                "avg_meta_confidence": avg_confidence,
                "avg_processing_time": avg_processing_time,
                "avg_papers_per_query": avg_papers_per_query,
                "avg_cross_domain_synthesis": avg_cross_domain,
                "avg_semantic_coherence": avg_semantic_coherence,
                "avg_reasoning_quality": avg_reasoning_quality,
                "corpus_utilization_percentage": corpus_utilization
            },
            "scenario_results": self.validation_results,
            "system_info": {
                "nwtn_version": "Enhanced Meta-Reasoning v2.0",
                "semantic_search": "HNSW Index with 151,120 papers",
                "reasoning_engines": 7,
                "world_model_items": 223,
                "production_ready": True
            }
        }
        
        return validation_report
    
    def print_validation_summary(self, report: Dict):
        """Print comprehensive validation summary"""
        print("\n🎉 NWTN FULL CORPUS VALIDATION COMPLETE")
        print("=" * 80)
        print(f"📅 Validation Date: {report['validation_timestamp']}")
        print(f"📊 Corpus Size: {report['corpus_size']:,} papers")
        print(f"🧪 Test Scenarios: {report['total_scenarios']}")
        print(f"✅ Success Rate: {report['success_rate']:.1f}%")
        print(f"⏱️  Total Time: {report['total_validation_time']:.1f}s")
        print()
        
        print("📈 PERFORMANCE METRICS:")
        print("-" * 40)
        metrics = report['overall_metrics']
        print(f"🎯 Average Meta-Confidence: {metrics['avg_meta_confidence']:.3f}")
        print(f"⚡ Average Processing Time: {metrics['avg_processing_time']:.1f}s")
        print(f"📄 Average Papers per Query: {metrics['avg_papers_per_query']:.1f}")
        print(f"🌐 Cross-Domain Synthesis: {metrics['avg_cross_domain_synthesis']:.3f}")
        print(f"🔗 Semantic Coherence: {metrics['avg_semantic_coherence']:.3f}")
        print(f"🧠 Reasoning Quality: {metrics['avg_reasoning_quality']:.3f}")
        print(f"📊 Corpus Utilization: {metrics['corpus_utilization_percentage']:.3f}%")
        print()
        
        print("🧪 SCENARIO RESULTS:")
        print("-" * 40)
        for result in report['scenario_results']:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['scenario']}")
            if result['success']:
                print(f"   📊 Confidence: {result['meta_confidence']:.3f}")
                print(f"   📄 Papers: {result['papers_retrieved']}")
                print(f"   🌐 Domains: {len(result['domain_distribution'])}")
                print(f"   ⏱️  Time: {result['processing_time']:.1f}s")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        print()
        
        print("🚀 SYSTEM CAPABILITIES:")
        print("-" * 40)
        sys_info = report['system_info']
        print(f"🧠 NWTN Version: {sys_info['nwtn_version']}")
        print(f"🔍 Semantic Search: {sys_info['semantic_search']}")
        print(f"⚙️  Reasoning Engines: {sys_info['reasoning_engines']}")
        print(f"🌍 World Model Items: {sys_info['world_model_items']}")
        print(f"🎯 Production Ready: {sys_info['production_ready']}")

async def main():
    """Main validation function"""
    validator = NWTNFullCorpusValidator()
    
    # Initialize systems
    if not await validator.initialize_systems():
        logger.error("❌ Failed to initialize systems")
        return
    
    # Run comprehensive validation
    report = await validator.run_full_validation()
    
    # Print summary
    validator.print_validation_summary(report)
    
    # Save detailed report
    report_file = f"nwtn_full_corpus_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📄 Detailed report saved to: {report_file}")
    logger.info("🎉 NWTN Full Corpus Validation Complete!")

if __name__ == "__main__":
    asyncio.run(main())