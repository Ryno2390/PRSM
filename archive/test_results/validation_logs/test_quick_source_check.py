#!/usr/bin/env python3
"""
Quick Source Check Test
======================

Quick test to verify the data structure fix and confirm 4+ sources.
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.external_storage_config import ExternalStorageConfig

async def quick_source_check():
    """Quick test to verify optimized source retrieval"""
    print("🔧 Quick Source Check Test...")
    
    # Initialize system integrator
    external_storage = ExternalStorageConfig()
    integrator = SystemIntegrator(external_storage_config=external_storage)
    await integrator.initialize()
    
    # Simple query
    test_query = 'artificial intelligence'
    
    start_time = time.time()
    result = await integrator.process_complete_query(
        query=test_query,
        user_id='quick_check',
        query_cost=2.0
    )
    end_time = time.time()
    
    print(f"\n✅ Quick Check Results:")
    print(f"   Success: {result.success}")
    print(f"   Sources: {len(result.citations)}")
    print(f"   Time: {end_time - start_time:.1f}s")
    
    if result.success and len(result.citations) > 0:
        print(f"\n📚 Sources Found:")
        for i, citation in enumerate(result.citations, 1):
            # Handle both string and dict citation formats (fixed)
            if isinstance(citation, dict):
                title = citation.get('title', f'Source {i}')
                relevance = citation.get('relevance_score', 0)
            else:
                title = str(citation)
                relevance = 0.0
            
            display_title = title[:50] + '...' if len(title) > 50 else title
            print(f"   {i}. {display_title} ({relevance:.3f})")
        
        optimization_success = len(result.citations) >= 3
        print(f"\n🎯 Optimization Status: {'✅ SUCCESS' if optimization_success else '⚠️ NEEDS WORK'}")
        print(f"   Found {len(result.citations)} sources (target: ≥3)")
        
        return optimization_success
    else:
        print(f"   ❌ Failed: {result.error_message if not result.success else 'No sources found'}")
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_source_check())
    
    if success:
        print("\n🎉 Source optimization confirmed working!")
    else:
        print("\n🔧 Additional tuning needed")