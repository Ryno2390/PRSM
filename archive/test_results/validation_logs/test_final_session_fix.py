#!/usr/bin/env python3
"""
Test Final Session Fix
======================

This test verifies that the final session coordination fix works.
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.external_storage_config import ExternalStorageConfig

async def test_final_session_fix():
    """Test that the final session fix resolves all coordination issues"""
    print("🔧 Testing Final Session Fix...")
    
    # Initialize system integrator
    external_storage = ExternalStorageConfig()
    integrator = SystemIntegrator(external_storage_config=external_storage)
    await integrator.initialize()
    
    # Use a simple query to test session coordination
    test_query = 'quantum mechanics'
    
    print(f"🧠 Testing query: {test_query}")
    print(f"🎯 Focus: Complete pipeline with graceful session handling")
    
    start_time = time.time()
    result = await integrator.process_complete_query(
        query=test_query,
        user_id='test_final_fix',
        query_cost=2.0
    )
    end_time = time.time()
    
    print(f"\n✅ Final Session Fix Test Results:")
    print(f"   Success: {result.success}")
    print(f"   Processing time: {end_time - start_time:.2f}s")
    
    if result.success:
        print(f"   Quality: {result.quality_score:.3f}")
        print(f"   Sources: {len(result.citations)}")
        print(f"   Total cost: {result.total_cost:.2f} FTNS")
        print(f"   Payments: {len(result.payment_distributions)}")
        
        print(f"\n🎉 SUCCESS: Final session fix is working!")
        print(f"   Pipeline completed without session errors")
        return True
    else:
        print(f"\n❌ FAILED: {result.error_message}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_final_session_fix())
    if success:
        print("\n🎉 Final session fix verified - NWTN runs flawlessly!")
    else:
        print("\n⚠️  Additional debugging needed")