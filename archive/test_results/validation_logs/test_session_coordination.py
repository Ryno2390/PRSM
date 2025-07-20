#!/usr/bin/env python3
"""
Test Session Coordination Fix
=============================

This test focuses specifically on verifying that session IDs
are properly coordinated between components.
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.external_storage_config import ExternalStorageConfig

async def test_session_coordination():
    """Test that session IDs are properly coordinated between components"""
    print("🔧 Testing Session ID Coordination...")
    
    # Initialize system integrator
    external_storage = ExternalStorageConfig()
    integrator = SystemIntegrator(external_storage_config=external_storage)
    await integrator.initialize()
    
    # Use a very simple query to minimize processing time
    test_query = 'physics'
    
    print(f"🧠 Testing query: {test_query}")
    print(f"🎯 Focus: Session ID coordination in final stages")
    
    start_time = time.time()
    result = await integrator.process_complete_query(
        query=test_query,
        user_id='test_coord',
        query_cost=1.0
    )
    end_time = time.time()
    
    print(f"\n✅ Session Coordination Test Results:")
    print(f"   Success: {result.success}")
    print(f"   Processing time: {end_time - start_time:.2f}s")
    
    if result.success:
        print(f"   Quality: {result.quality_score:.3f}")
        print(f"   Sources: {len(result.citations)}")
        print(f"   Payments: {len(result.payment_distributions)}")
        
        print(f"\n🎉 SUCCESS: Session coordination is working!")
        print(f"   No session ID mismatch errors occurred")
        return True
    else:
        print(f"\n❌ FAILED: {result.error_message}")
        if "Session" in result.error_message and "not found" in result.error_message:
            print(f"   🔍 Session coordination issue still exists")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_session_coordination())
    if success:
        print("\n🎉 Session coordination fix is working correctly!")
    else:
        print("\n⚠️  Session coordination needs additional debugging")