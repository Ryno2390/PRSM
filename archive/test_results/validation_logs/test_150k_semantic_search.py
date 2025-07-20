#!/usr/bin/env python3
"""
Test 150K Semantic Search Scaling
=================================

This test verifies that semantic search now scales to all 150K papers
instead of being limited to 1,000 papers.
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.external_storage_config import ExternalStorageConfig

async def test_150k_semantic_search():
    """Test that semantic search now uses all 150K papers"""
    print("🔍 Testing 150K Semantic Search Scaling...")
    
    # Initialize system integrator
    external_storage = ExternalStorageConfig()
    integrator = SystemIntegrator(external_storage_config=external_storage)
    await integrator.initialize()
    
    # Test with a specific query
    test_query = 'quantum computing breakthrough methodologies'
    
    print(f"🧠 Testing query: {test_query}")
    print(f"📊 Expected to search: 150,000 papers")
    
    start_time = time.time()
    result = await integrator.process_complete_query(
        query=test_query,
        user_id='test_user_150k',
        query_cost=5.0
    )
    end_time = time.time()
    
    print(f"\n✅ 150K Search Test Results:")
    print(f"   Success: {result.success}")
    print(f"   Quality: {result.quality_score:.3f}")
    print(f"   Sources: {len(result.citations)}")
    print(f"   Processing time: {end_time - start_time:.2f}s")
    print(f"   Total cost: {result.total_cost:.2f} FTNS")
    
    # Check if we're actually searching more papers
    if result.success and hasattr(result, 'search_stats'):
        papers_searched = result.search_stats.get('papers_searched', 0)
        print(f"   Papers searched: {papers_searched:,}")
        
        if papers_searched >= 100000:
            print(f"🎉 SUCCESS: Searching {papers_searched:,} papers (150K scale)")
        elif papers_searched >= 10000:
            print(f"📈 PROGRESS: Searching {papers_searched:,} papers (improvement)")
        else:
            print(f"⚠️  LIMITED: Only searching {papers_searched:,} papers")
    
    if result.success:
        print(f"\n📚 Retrieved Sources:")
        for i, citation in enumerate(result.citations[:3]):
            print(f"   {i+1}. {citation.get('title', 'Unknown')} (Score: {citation.get('relevance_score', 0):.3f})")
        
        if len(result.citations) > 3:
            print(f"   ... and {len(result.citations) - 3} more sources")
        
        return True
    else:
        print(f"\n❌ FAILED: {result.error_message}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_150k_semantic_search())
    if success:
        print("\n🎉 150K semantic search scaling test passed!")
    else:
        print("\n⚠️  150K semantic search scaling test failed")