#!/usr/bin/env python3
"""
Test Enhanced Analogical Reasoning System on Yesterday's Papers
============================================================

This script tests the enhanced analogical reasoning system on the 194 ArXiv papers
downloaded yesterday (July 14, 2025) and compares results with baseline performance.

Key comparisons:
- Breakthrough detection accuracy and potential scores
- Cross-domain analogical connections
- Processing efficiency and learning velocity
- Quality of analogical mappings and inferences
"""

import asyncio
import json
import gzip
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import numpy as np

import sys
sys.path.append('/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.analogical_integration import NWTNAnalogicalIntegration
from prsm.nwtn.enhanced_analogical_reasoning import (
    AnalogicalReasoningEngine, AnalogicalReasoningType, 
    ConceptualDomain, ConceptualObject, StructuralRelation, RelationType
)

import structlog

logger = structlog.get_logger(__name__)


class EnhancedAnalogicalTester:
    """Enhanced analogical reasoning system tester"""
    
    def __init__(self):
        self.storage_path = Path("/Volumes/My Passport/PRSM_Storage")
        self.content_path = self.storage_path / "PRSM_Content" / "hot"
        self.results_path = Path("/Users/ryneschultz/Documents/GitHub/PRSM")
        
        # Initialize systems
        self.integration = NWTNAnalogicalIntegration(self.storage_path)
        self.engine = AnalogicalReasoningEngine()
        
        # Test results
        self.test_results = {
            "test_timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_info": {},
            "performance_metrics": {},
            "breakthrough_results": {},
            "analogical_mappings": {},
            "comparison_with_baseline": {},
            "processing_stats": {}
        }
        
        # Load baseline results for comparison
        self.baseline_results = self._load_baseline_results()
        
        logger.info("Enhanced Analogical Tester initialized")
    
    def _load_baseline_results(self) -> Dict[str, Any]:
        """Load baseline results from yesterday's tests"""
        baseline_files = [
            self.results_path / "test_results_20250714_164809.json",
            self.results_path / "test_results_20250714_165021.json"
        ]
        
        baseline_results = {}
        for result_file in baseline_files:
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        timestamp = result_file.stem.split('_')[-1]
                        baseline_results[timestamp] = data
                        logger.info(f"Loaded baseline results from {result_file}")
                except Exception as e:
                    logger.warning(f"Could not load baseline from {result_file}: {e}")
        
        return baseline_results
    
    async def find_yesterday_papers(self) -> List[Dict[str, Any]]:
        """Find papers downloaded yesterday (July 14, 2025)"""
        yesterday_papers = []
        
        if not self.content_path.exists():
            logger.error(f"Content path does not exist: {self.content_path}")
            return yesterday_papers
        
        # Find all .dat files modified yesterday
        for dat_file in self.content_path.rglob("*.dat"):
            try:
                # Check modification time
                mod_time = datetime.fromtimestamp(dat_file.stat().st_mtime, tz=timezone.utc)
                if mod_time.date() == datetime(2025, 7, 14).date():
                    
                    # Load and parse the content
                    with gzip.open(dat_file, 'rb') as f:
                        content = pickle.load(f)
                    
                    # Add file metadata
                    content['file_path'] = str(dat_file)
                    content['file_mod_time'] = mod_time.isoformat()
                    
                    yesterday_papers.append(content)
                    
            except Exception as e:
                logger.warning(f"Could not process {dat_file}: {e}")
        
        logger.info(f"Found {len(yesterday_papers)} papers from yesterday")
        return yesterday_papers
    
    async def process_papers_for_analogical_reasoning(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process papers through enhanced analogical reasoning system"""
        
        logger.info(f"Processing {len(papers)} papers for analogical reasoning")
        
        processing_stats = {
            "total_papers": len(papers),
            "processed_papers": 0,
            "failed_papers": 0,
            "processing_time": 0.0,
            "domain_distribution": {},
            "topographical_complexity": []
        }
        
        start_time = time.time()
        
        # Process each paper
        for i, paper in enumerate(papers):
            try:
                # Extract paper info
                paper_id = paper.get('id', f'paper_{i}')
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                domain = paper.get('domain', 'unknown')
                
                # Create content structure for integration
                content_data = {
                    'id': paper_id,
                    'title': title,
                    'abstract': abstract,
                    'domain': domain,
                    'authors': paper.get('authors', []),
                    'categories': paper.get('categories', []),
                    'keywords': paper.get('keywords', [])
                }
                
                # Process through analogical integration
                await self.integration.process_content_for_analogical_reasoning(content_data)
                
                # Update stats
                processing_stats["processed_papers"] += 1
                processing_stats["domain_distribution"][domain] = processing_stats["domain_distribution"].get(domain, 0) + 1
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(papers)} papers")
                
            except Exception as e:
                logger.error(f"Failed to process paper {paper.get('id', i)}: {e}")
                processing_stats["failed_papers"] += 1
        
        processing_stats["processing_time"] = time.time() - start_time
        
        # Calculate topographical complexity statistics
        for content_id, topo in self.integration.content_topographies.items():
            processing_stats["topographical_complexity"].append({
                "content_id": content_id,
                "complexity": topo.complexity_score,
                "maturity": topo.maturity_level,
                "breakthrough_potential": topo.breakthrough_potential
            })
        
        logger.info(f"Processing complete: {processing_stats['processed_papers']} papers processed in {processing_stats['processing_time']:.2f} seconds")
        
        return processing_stats
    
    async def test_breakthrough_detection(self) -> Dict[str, Any]:
        """Test breakthrough detection capabilities"""
        
        logger.info("Testing breakthrough detection capabilities")
        
        breakthrough_results = {
            "total_content_analyzed": len(self.integration.content_topographies),
            "breakthrough_candidates": [],
            "high_potential_papers": [],
            "cross_domain_breakthroughs": [],
            "average_breakthrough_potential": 0.0,
            "domain_breakthrough_analysis": {}
        }
        
        # Analyze each domain for breakthrough potential
        domains = set(topo.domain for topo in self.integration.content_topographies.values())
        
        for domain in domains:
            domain_inferences = await self.integration.find_analogical_breakthroughs(domain)
            
            breakthrough_results["domain_breakthrough_analysis"][domain] = {
                "inference_count": len(domain_inferences),
                "average_confidence": np.mean([inf.confidence for inf in domain_inferences]) if domain_inferences else 0.0,
                "high_confidence_count": len([inf for inf in domain_inferences if inf.confidence > 0.8])
            }
        
        # Find high-potential papers
        high_potential_threshold = 0.7
        for content_id, topo in self.integration.content_topographies.items():
            if topo.breakthrough_potential > high_potential_threshold:
                breakthrough_results["high_potential_papers"].append({
                    "content_id": content_id,
                    "domain": topo.domain,
                    "breakthrough_potential": topo.breakthrough_potential,
                    "complexity": topo.complexity_score,
                    "maturity": topo.maturity_level
                })
        
        # Calculate average breakthrough potential
        all_potentials = [topo.breakthrough_potential for topo in self.integration.content_topographies.values()]
        breakthrough_results["average_breakthrough_potential"] = np.mean(all_potentials) if all_potentials else 0.0
        
        # Find cross-domain breakthrough opportunities
        domain_pairs = [(d1, d2) for d1 in domains for d2 in domains if d1 != d2]
        
        for source_domain, target_domain in domain_pairs[:5]:  # Test top 5 pairs
            cross_inferences = await self.integration.find_cross_domain_analogies(source_domain, target_domain)
            
            if cross_inferences:
                breakthrough_results["cross_domain_breakthroughs"].append({
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "inference_count": len(cross_inferences),
                    "top_confidence": max(inf.confidence for inf in cross_inferences),
                    "top_inference": cross_inferences[0].content if cross_inferences else None
                })
        
        logger.info(f"Breakthrough detection complete: {len(breakthrough_results['high_potential_papers'])} high-potential papers found")
        
        return breakthrough_results
    
    async def test_analogical_mappings(self) -> Dict[str, Any]:
        """Test analogical mapping capabilities"""
        
        logger.info("Testing analogical mapping capabilities")
        
        mapping_results = {
            "total_mappings_tested": 0,
            "successful_mappings": 0,
            "mapping_quality_distribution": [],
            "reasoning_type_results": {},
            "domain_pair_analysis": {}
        }
        
        # Test different reasoning types
        reasoning_types = [
            AnalogicalReasoningType.DEVELOPMENTAL,
            AnalogicalReasoningType.EXPLANATORY,
            AnalogicalReasoningType.PROBLEM_SOLVING,
            AnalogicalReasoningType.CREATIVE
        ]
        
        domains = list(set(topo.domain for topo in self.integration.content_topographies.values()))
        
        for reasoning_type in reasoning_types:
            type_results = {
                "mappings_found": 0,
                "average_quality": 0.0,
                "inferences_generated": 0,
                "average_confidence": 0.0
            }
            
            # Test domain pairs
            for i, source_domain in enumerate(domains[:3]):  # Test first 3 domains
                for target_domain in domains[i+1:4]:  # Against next 3 domains
                    
                    # Find mappings
                    mappings = self.engine.find_analogical_mappings(
                        f"{source_domain}_sample", 
                        f"{target_domain}_sample", 
                        reasoning_type
                    )
                    
                    mapping_results["total_mappings_tested"] += 1
                    
                    if mappings:
                        mapping_results["successful_mappings"] += 1
                        type_results["mappings_found"] += len(mappings)
                        
                        # Analyze mapping quality
                        qualities = [m.mapping_quality for m in mappings]
                        mapping_results["mapping_quality_distribution"].extend(qualities)
                        
                        # Generate inferences
                        for mapping in mappings:
                            inferences = self.engine.generate_inferences(mapping)
                            type_results["inferences_generated"] += len(inferences)
                            
                            confidences = [inf.confidence for inf in inferences]
                            if confidences:
                                type_results["average_confidence"] = np.mean(confidences)
            
            # Calculate type averages
            if mapping_results["mapping_quality_distribution"]:
                type_results["average_quality"] = np.mean(mapping_results["mapping_quality_distribution"])
            
            mapping_results["reasoning_type_results"][reasoning_type.value] = type_results
        
        logger.info(f"Analogical mapping test complete: {mapping_results['successful_mappings']}/{mapping_results['total_mappings_tested']} successful mappings")
        
        return mapping_results
    
    async def compare_with_baseline(self) -> Dict[str, Any]:
        """Compare enhanced results with baseline performance"""
        
        logger.info("Comparing enhanced results with baseline performance")
        
        comparison = {
            "baseline_available": len(self.baseline_results) > 0,
            "performance_improvements": {},
            "breakthrough_improvements": {},
            "processing_improvements": {},
            "overall_enhancement_score": 0.0
        }
        
        if not self.baseline_results:
            logger.warning("No baseline results available for comparison")
            return comparison
        
        # Get latest baseline results
        latest_baseline = max(self.baseline_results.values(), key=lambda x: x.get('timestamp', ''))
        
        # Compare breakthrough detection
        baseline_breakthrough = latest_baseline.get('breakthrough_detection', {})
        enhanced_breakthrough = self.test_results.get('breakthrough_results', {})
        
        if baseline_breakthrough and enhanced_breakthrough:
            baseline_avg = baseline_breakthrough.get('average_potential', 0.0)
            enhanced_avg = enhanced_breakthrough.get('average_breakthrough_potential', 0.0)
            
            comparison["breakthrough_improvements"] = {
                "baseline_average_potential": baseline_avg,
                "enhanced_average_potential": enhanced_avg,
                "improvement_factor": enhanced_avg / baseline_avg if baseline_avg > 0 else 0.0,
                "improvement_percentage": ((enhanced_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0.0
            }
        
        # Compare processing performance
        baseline_processing = latest_baseline.get('processing_stats', {})
        enhanced_processing = self.test_results.get('processing_stats', {})
        
        if baseline_processing and enhanced_processing:
            baseline_time = baseline_processing.get('total_time', 0.0)
            enhanced_time = enhanced_processing.get('processing_time', 0.0)
            
            comparison["processing_improvements"] = {
                "baseline_processing_time": baseline_time,
                "enhanced_processing_time": enhanced_time,
                "speed_improvement": baseline_time / enhanced_time if enhanced_time > 0 else 0.0,
                "efficiency_gain": ((baseline_time - enhanced_time) / baseline_time * 100) if baseline_time > 0 else 0.0
            }
        
        # Calculate overall enhancement score
        breakthrough_score = comparison["breakthrough_improvements"].get("improvement_factor", 1.0)
        processing_score = comparison["processing_improvements"].get("speed_improvement", 1.0)
        
        comparison["overall_enhancement_score"] = (breakthrough_score + processing_score) / 2
        
        logger.info(f"Comparison complete: Overall enhancement score = {comparison['overall_enhancement_score']:.2f}")
        
        return comparison
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of enhanced analogical reasoning system"""
        
        logger.info("🚀 Starting comprehensive test of enhanced analogical reasoning system")
        
        # Find yesterday's papers
        yesterday_papers = await self.find_yesterday_papers()
        
        if not yesterday_papers:
            logger.error("No papers found from yesterday - cannot run test")
            return {"error": "No papers found from yesterday"}
        
        # Update test results with dataset info
        self.test_results["dataset_info"] = {
            "total_papers": len(yesterday_papers),
            "date_range": "2025-07-14",
            "domains": list(set(paper.get('domain', 'unknown') for paper in yesterday_papers)),
            "source": "ArXiv papers downloaded yesterday"
        }
        
        # Process papers through enhanced system
        processing_stats = await self.process_papers_for_analogical_reasoning(yesterday_papers)
        self.test_results["processing_stats"] = processing_stats
        
        # Test breakthrough detection
        breakthrough_results = await self.test_breakthrough_detection()
        self.test_results["breakthrough_results"] = breakthrough_results
        
        # Test analogical mappings
        mapping_results = await self.test_analogical_mappings()
        self.test_results["analogical_mappings"] = mapping_results
        
        # Compare with baseline
        comparison = await self.compare_with_baseline()
        self.test_results["comparison_with_baseline"] = comparison
        
        # Calculate overall performance metrics
        self.test_results["performance_metrics"] = {
            "processing_efficiency": processing_stats["processed_papers"] / processing_stats["processing_time"] if processing_stats["processing_time"] > 0 else 0,
            "breakthrough_detection_rate": len(breakthrough_results["high_potential_papers"]) / len(yesterday_papers) if yesterday_papers else 0,
            "analogical_mapping_success_rate": mapping_results["successful_mappings"] / mapping_results["total_mappings_tested"] if mapping_results["total_mappings_tested"] > 0 else 0,
            "overall_system_score": comparison.get("overall_enhancement_score", 0.0)
        }
        
        # Save results
        await self.save_test_results()
        
        logger.info("✅ Comprehensive test completed successfully")
        
        return self.test_results
    
    async def save_test_results(self):
        """Save test results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_path / f"enhanced_analogical_test_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            logger.info(f"Test results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
    
    def print_test_summary(self):
        """Print a summary of test results"""
        
        print("\n" + "="*80)
        print("🧠 ENHANCED ANALOGICAL REASONING SYSTEM TEST RESULTS")
        print("="*80)
        
        # Dataset info
        dataset_info = self.test_results.get("dataset_info", {})
        print(f"\n📊 DATASET INFORMATION:")
        print(f"   Total Papers: {dataset_info.get('total_papers', 0)}")
        print(f"   Date Range: {dataset_info.get('date_range', 'Unknown')}")
        print(f"   Domains: {len(dataset_info.get('domains', []))}")
        print(f"   Source: {dataset_info.get('source', 'Unknown')}")
        
        # Processing stats
        processing_stats = self.test_results.get("processing_stats", {})
        print(f"\n⚡ PROCESSING PERFORMANCE:")
        print(f"   Papers Processed: {processing_stats.get('processed_papers', 0)}/{processing_stats.get('total_papers', 0)}")
        print(f"   Processing Time: {processing_stats.get('processing_time', 0):.2f} seconds")
        print(f"   Processing Rate: {processing_stats.get('processed_papers', 0) / processing_stats.get('processing_time', 1):.2f} papers/second")
        
        # Breakthrough detection
        breakthrough_results = self.test_results.get("breakthrough_results", {})
        print(f"\n💡 BREAKTHROUGH DETECTION:")
        print(f"   Average Breakthrough Potential: {breakthrough_results.get('average_breakthrough_potential', 0):.3f}")
        print(f"   High-Potential Papers: {len(breakthrough_results.get('high_potential_papers', []))}")
        print(f"   Cross-Domain Breakthroughs: {len(breakthrough_results.get('cross_domain_breakthroughs', []))}")
        
        # Analogical mappings
        mapping_results = self.test_results.get("analogical_mappings", {})
        print(f"\n🔗 ANALOGICAL MAPPING PERFORMANCE:")
        print(f"   Successful Mappings: {mapping_results.get('successful_mappings', 0)}/{mapping_results.get('total_mappings_tested', 0)}")
        print(f"   Success Rate: {mapping_results.get('successful_mappings', 0) / mapping_results.get('total_mappings_tested', 1) * 100:.1f}%")
        
        # Comparison with baseline
        comparison = self.test_results.get("comparison_with_baseline", {})
        if comparison.get("baseline_available"):
            print(f"\n📈 COMPARISON WITH BASELINE:")
            
            breakthrough_improvements = comparison.get("breakthrough_improvements", {})
            if breakthrough_improvements:
                print(f"   Breakthrough Potential Improvement: {breakthrough_improvements.get('improvement_percentage', 0):.1f}%")
            
            processing_improvements = comparison.get("processing_improvements", {})
            if processing_improvements:
                print(f"   Processing Speed Improvement: {processing_improvements.get('efficiency_gain', 0):.1f}%")
            
            print(f"   Overall Enhancement Score: {comparison.get('overall_enhancement_score', 0):.2f}")
        else:
            print(f"\n📈 COMPARISON WITH BASELINE: No baseline data available")
        
        # Overall performance
        performance_metrics = self.test_results.get("performance_metrics", {})
        print(f"\n🎯 OVERALL PERFORMANCE METRICS:")
        print(f"   Processing Efficiency: {performance_metrics.get('processing_efficiency', 0):.2f} papers/second")
        print(f"   Breakthrough Detection Rate: {performance_metrics.get('breakthrough_detection_rate', 0) * 100:.1f}%")
        print(f"   Analogical Mapping Success Rate: {performance_metrics.get('analogical_mapping_success_rate', 0) * 100:.1f}%")
        print(f"   Overall System Score: {performance_metrics.get('overall_system_score', 0):.2f}")
        
        print("\n" + "="*80)


async def main():
    """Main test execution"""
    
    print("🧠 ENHANCED ANALOGICAL REASONING SYSTEM TEST")
    print("Testing on papers downloaded yesterday (July 14, 2025)")
    print("="*80)
    
    # Create tester
    tester = EnhancedAnalogicalTester()
    
    # Run comprehensive test
    results = await tester.run_comprehensive_test()
    
    # Print summary
    tester.print_test_summary()
    
    return results


if __name__ == "__main__":
    asyncio.run(main())