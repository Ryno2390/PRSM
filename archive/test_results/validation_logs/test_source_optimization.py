#!/usr/bin/env python3
"""
Test Source Retrieval Optimization
==================================

This test verifies that our parameter optimizations successfully increase
the number of sources found per query from 1 to 3-15 sources.

Key optimizations applied:
1. Meta Reasoning: max_results 5 → 20
2. Embedding similarity: >0.5 → >0.25  
3. Citation Filter: relevance 0.3→0.15, contribution 0.2→0.1
4. Semantic Retriever: top_k 10→25, similarity 0.3→0.2
5. External Storage: max_results 10→25
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.external_storage_config import ExternalStorageConfig

async def test_source_optimization():
    """Test that source optimization increases source count per query"""
    print("🔧 Testing Source Retrieval Optimization...")
    print("=" * 50)
    
    # Initialize system integrator
    external_storage = ExternalStorageConfig()
    integrator = SystemIntegrator(external_storage_config=external_storage)
    await integrator.initialize()
    print("✅ NWTN System initialized")
    
    # Test with a query that should find multiple sources
    test_query = 'machine learning neural networks deep learning'
    
    print(f"\n🧠 Testing optimized retrieval:")
    print(f"   Query: {test_query}")
    print(f"   Expected: 3-15 sources (was 1 before optimization)")
    
    start_time = time.time()
    result = await integrator.process_complete_query(
        query=test_query,
        user_id='test_source_optimization',
        query_cost=3.0
    )
    end_time = time.time()
    
    print(f"\n📊 Source Optimization Test Results:")
    print(f"   Success: {result.success}")
    print(f"   Processing time: {end_time - start_time:.2f}s")
    
    if result.success:
        sources_found = len(result.citations)
        print(f"   Sources found: {sources_found}")
        print(f"   Quality score: {result.quality_score:.3f}")
        print(f"   Total cost: {result.total_cost:.2f} FTNS")
        
        # Check if optimization worked
        optimization_successful = sources_found >= 3
        
        print(f"\n🎯 Optimization Assessment:")
        if optimization_successful:
            print(f"   ✅ SUCCESS: Found {sources_found} sources (target: ≥3)")
            print(f"   📈 Significant improvement from previous 1-source limitation")
            if sources_found >= 5:
                print(f"   🌟 EXCELLENT: {sources_found} sources provides great diversity")
        else:
            print(f"   ⚠️  NEEDS MORE WORK: Only {sources_found} sources found")
            print(f"   🔧 Additional parameter tuning may be needed")
        
        print(f"\n📚 Retrieved Sources:")
        for i, citation in enumerate(result.citations, 1):
            # Handle both string and dict citation formats
            if isinstance(citation, dict):
                title = citation.get('title', 'Unknown Title')
                relevance = citation.get('relevance_score', 0)
            else:
                # If citation is a string, use it as title
                title = str(citation)
                relevance = 0.0
            
            # Truncate long titles
            display_title = title[:60] + '...' if len(title) > 60 else title
            print(f"   {i}. {display_title} (Score: {relevance:.3f})")
        
        return optimization_successful
    else:
        print(f"   ❌ FAILED: {result.error_message}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_source_optimization())
    
    if success:
        print(f"\n🎉 Source retrieval optimization SUCCESSFUL!")
        print(f"   Ready for optimized 5-prompt production test")
    else:
        print(f"\n🔧 Source retrieval needs additional tuning")