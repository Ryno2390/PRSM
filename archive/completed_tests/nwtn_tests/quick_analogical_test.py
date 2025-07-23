#!/usr/bin/env python3
"""
Quick Enhanced Analogical Reasoning Test
=======================================

A focused test to quickly compare enhanced analogical reasoning performance 
with yesterday's baseline results.
"""

import asyncio
import json
import gzip
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone
import numpy as np

import sys
sys.path.append('/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.analogical_integration import NWTNAnalogicalIntegration
from prsm.nwtn.enhanced_analogical_reasoning import AnalogicalReasoningEngine, AnalogicalReasoningType

import structlog
logger = structlog.get_logger(__name__)


async def load_sample_papers(limit: int = 20) -> List[Dict[str, Any]]:
    """Load a sample of yesterday's papers for testing"""
    
    storage_path = Path("/Volumes/My Passport/PRSM_Storage")
    content_path = storage_path / "PRSM_Content" / "hot"
    
    papers = []
    
    if not content_path.exists():
        logger.error(f"Content path not found: {content_path}")
        return papers
    
    # Find papers from yesterday
    for dat_file in content_path.rglob("*.dat"):
        if len(papers) >= limit:
            break
            
        try:
            mod_time = datetime.fromtimestamp(dat_file.stat().st_mtime, tz=timezone.utc)
            if mod_time.date() == datetime(2025, 7, 14).date():
                
                with gzip.open(dat_file, 'rb') as f:
                    content = pickle.load(f)
                
                papers.append({
                    'id': content.get('id', str(dat_file.name)),
                    'title': content.get('title', ''),
                    'abstract': content.get('abstract', '')[:500],  # Truncate for speed
                    'domain': content.get('domain', 'unknown'),
                    'authors': content.get('authors', [])[:3],  # Limit authors
                    'categories': content.get('categories', [])[:2]  # Limit categories
                })
                
        except Exception as e:
            logger.warning(f"Could not load {dat_file}: {e}")
    
    logger.info(f"Loaded {len(papers)} sample papers from yesterday")
    return papers


async def test_enhanced_analogical_reasoning():
    """Test enhanced analogical reasoning on sample papers"""
    
    print("🧠 ENHANCED ANALOGICAL REASONING - QUICK TEST")
    print("=" * 60)
    
    # Load sample papers
    papers = await load_sample_papers(limit=20)
    
    if not papers:
        print("❌ No papers found to test")
        return
    
    print(f"📄 Testing with {len(papers)} papers from yesterday")
    
    # Initialize systems
    storage_path = Path("/tmp/test_storage")
    storage_path.mkdir(exist_ok=True)
    
    integration = NWTNAnalogicalIntegration(storage_path)
    engine = AnalogicalReasoningEngine()
    
    # Test results
    results = {
        "papers_processed": 0,
        "domains_found": set(),
        "topographical_analysis": [],
        "analogical_mappings": 0,
        "breakthrough_candidates": 0,
        "processing_time": 0.0
    }
    
    start_time = time.time()
    
    # Process papers
    print("\n🔄 Processing papers...")
    for paper in papers:
        try:
            await integration.process_content_for_analogical_reasoning(paper)
            results["papers_processed"] += 1
            results["domains_found"].add(paper['domain'])
            
            # Analyze topography
            if paper['id'] in integration.content_topographies:
                topo = integration.content_topographies[paper['id']]
                results["topographical_analysis"].append({
                    "id": paper['id'],
                    "domain": topo.domain,
                    "concepts": len(topo.concepts),
                    "complexity": topo.complexity_score,
                    "maturity": topo.maturity_level,
                    "breakthrough_potential": topo.breakthrough_potential
                })
                
                # Check for breakthrough candidates
                if topo.breakthrough_potential > 0.7:
                    results["breakthrough_candidates"] += 1
            
        except Exception as e:
            logger.error(f"Failed to process paper {paper['id']}: {e}")
    
    # Test analogical mappings
    print("🔗 Testing analogical mappings...")
    domains = list(results["domains_found"])
    
    if len(domains) >= 2:
        # Test cross-domain analogies
        for i, source_domain in enumerate(domains[:3]):
            for target_domain in domains[i+1:4]:
                try:
                    cross_inferences = await integration.find_cross_domain_analogies(
                        source_domain, target_domain
                    )
                    results["analogical_mappings"] += len(cross_inferences)
                except Exception as e:
                    logger.warning(f"Could not find analogies between {source_domain} and {target_domain}: {e}")
    
    results["processing_time"] = time.time() - start_time
    
    # Load baseline results for comparison
    baseline_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/test_results_20250714_165021.json")
    baseline_data = {}
    
    if baseline_file.exists():
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load baseline data: {e}")
    
    # Print results
    print("\n📊 ENHANCED SYSTEM RESULTS:")
    print(f"   Papers Processed: {results['papers_processed']}")
    print(f"   Domains Found: {len(results['domains_found'])}")
    print(f"   Analogical Mappings: {results['analogical_mappings']}")
    print(f"   Breakthrough Candidates: {results['breakthrough_candidates']}")
    print(f"   Processing Time: {results['processing_time']:.2f}s")
    
    # Topographical analysis
    if results["topographical_analysis"]:
        avg_complexity = np.mean([t["complexity"] for t in results["topographical_analysis"]])
        avg_maturity = np.mean([t["maturity"] for t in results["topographical_analysis"]])
        avg_breakthrough = np.mean([t["breakthrough_potential"] for t in results["topographical_analysis"]])
        
        print(f"\n🗺️ TOPOGRAPHICAL ANALYSIS:")
        print(f"   Average Complexity: {avg_complexity:.3f}")
        print(f"   Average Maturity: {avg_maturity:.3f}")
        print(f"   Average Breakthrough Potential: {avg_breakthrough:.3f}")
    
    # Comparison with baseline
    if baseline_data:
        print(f"\n📈 COMPARISON WITH BASELINE:")
        
        # Compare breakthrough detection
        baseline_breakthrough = baseline_data.get('breakthrough_detection', {})
        baseline_avg_potential = baseline_breakthrough.get('average_potential', 0)
        
        if results["topographical_analysis"]:
            enhanced_avg_potential = np.mean([t["breakthrough_potential"] for t in results["topographical_analysis"]])
            improvement = ((enhanced_avg_potential - baseline_avg_potential) / baseline_avg_potential * 100) if baseline_avg_potential > 0 else 0
            print(f"   Breakthrough Potential: {baseline_avg_potential:.3f} → {enhanced_avg_potential:.3f} ({improvement:+.1f}%)")
        
        # Compare processing efficiency
        baseline_processing = baseline_data.get('processing_stats', {})
        baseline_time = baseline_processing.get('total_time', 0)
        
        if baseline_time > 0:
            efficiency_improvement = ((baseline_time - results["processing_time"]) / baseline_time * 100)
            print(f"   Processing Time: {baseline_time:.2f}s → {results['processing_time']:.2f}s ({efficiency_improvement:+.1f}%)")
        
        # Overall assessment
        print(f"\n🎯 ENHANCED SYSTEM ASSESSMENT:")
        
        if results["breakthrough_candidates"] > 0:
            print(f"   ✅ Breakthrough Detection: {results['breakthrough_candidates']} candidates found")
        else:
            print(f"   ⚠️  Breakthrough Detection: No high-potential candidates in sample")
        
        if results["analogical_mappings"] > 0:
            print(f"   ✅ Analogical Reasoning: {results['analogical_mappings']} cross-domain mappings")
        else:
            print(f"   ⚠️  Analogical Reasoning: Limited cross-domain mappings")
        
        if results["processing_time"] < 10:
            print(f"   ✅ Processing Speed: Efficient ({results['processing_time']:.2f}s for {results['papers_processed']} papers)")
        else:
            print(f"   ⚠️  Processing Speed: Could be improved ({results['processing_time']:.2f}s)")
    
    # Sample breakthrough candidates
    if results["breakthrough_candidates"] > 0:
        print(f"\n💡 BREAKTHROUGH CANDIDATES:")
        for analysis in results["topographical_analysis"]:
            if analysis["breakthrough_potential"] > 0.7:
                print(f"   • {analysis['id']} ({analysis['domain']}): {analysis['breakthrough_potential']:.3f}")
    
    print(f"\n{'='*60}")
    print(f"✅ Enhanced analogical reasoning test completed!")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_enhanced_analogical_reasoning())