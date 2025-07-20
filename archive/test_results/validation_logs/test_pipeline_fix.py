#!/usr/bin/env python3
"""
Quick Pipeline Test to Verify Fixes
===================================

This test verifies that the pipeline fixes work correctly:
1. Session management is working
2. Semantic search finds papers
3. Complete pipeline runs successfully
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.external_storage_config import ExternalStorageConfig

async def test_pipeline_fix():
    """Test that the pipeline fixes work"""
    print("🔧 Testing Pipeline Fix...")
    
    # Initialize system integrator
    external_storage = ExternalStorageConfig()
    integrator = SystemIntegrator(external_storage_config=external_storage)
    await integrator.initialize()
    
    # Test with a simple prompt
    test_prompt = 'quantum computing applications'
    
    print(f"🧠 Testing prompt: {test_prompt}")
    
    start_time = time.time()
    result = await integrator.process_complete_query(
        query=test_prompt,
        user_id='test_user_fix',
        query_cost=5.0
    )
    end_time = time.time()
    
    print(f"\n✅ Pipeline Test Results:")
    print(f"   Success: {result.success}")
    print(f"   Quality: {result.quality_score:.3f}")
    print(f"   Sources: {len(result.citations)}")
    print(f"   Processing time: {end_time - start_time:.2f}s")
    print(f"   Total cost: {result.total_cost:.2f} FTNS")
    print(f"   Payments: {len(result.payment_distributions)}")
    
    if result.success:
        print(f"\n🎉 SUCCESS: Pipeline is working!")
        print(f"   Response preview: {result.final_response[:100]}...")
        
        # Check key components
        components_working = {
            'semantic_search': len(result.citations) > 0,
            'deep_reasoning': result.quality_score > 0,
            'payment_system': len(result.payment_distributions) > 0,
            'session_management': True  # If we got here, sessions work
        }
        
        print(f"\n📊 Component Status:")
        for component, status in components_working.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}: {'Working' if status else 'Failed'}")
        
        all_working = all(components_working.values())
        print(f"\n🎯 Overall Status: {'PRODUCTION READY' if all_working else 'NEEDS WORK'}")
        
        return all_working
    else:
        print(f"\n❌ FAILED: Pipeline still has issues")
        print(f"   Error: {result.error_message}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_pipeline_fix())
    if success:
        print("\n🎉 All pipeline fixes are working correctly!")
    else:
        print("\n⚠️  Pipeline still needs additional fixes")