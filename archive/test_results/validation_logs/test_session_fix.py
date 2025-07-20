#!/usr/bin/env python3
"""
Test Session Management Fix
===========================

This test verifies that the session management fix resolves the
"Session not found" error that was occurring at the final stage.
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.external_storage_config import ExternalStorageConfig

async def test_session_management_fix():
    """Test that session management no longer fails with 'Session not found'"""
    print("🔧 Testing Session Management Fix...")
    
    # Initialize system integrator
    external_storage = ExternalStorageConfig()
    integrator = SystemIntegrator(external_storage_config=external_storage)
    await integrator.initialize()
    
    # Use a simpler query to focus on session management
    test_query = 'machine learning algorithms'
    
    print(f"🧠 Testing query: {test_query}")
    print(f"🎯 Focus: Session management during complete pipeline")
    
    start_time = time.time()
    result = await integrator.process_complete_query(
        query=test_query,
        user_id='test_session_fix',
        query_cost=3.0
    )
    end_time = time.time()
    
    print(f"\n✅ Session Management Test Results:")
    print(f"   Success: {result.success}")
    print(f"   Processing time: {end_time - start_time:.2f}s")
    
    if result.success:
        print(f"   Quality: {result.quality_score:.3f}")
        print(f"   Sources: {len(result.citations)}")
        print(f"   Total cost: {result.total_cost:.2f} FTNS")
        print(f"   Payments: {len(result.payment_distributions)}")
        
        print(f"\n🎉 SUCCESS: Session management is working correctly!")
        print(f"   No 'Session not found' errors occurred")
        print(f"   Complete pipeline executed successfully")
        return True
    else:
        print(f"\n❌ FAILED: {result.error_message}")
        if "Session not found" in result.error_message:
            print(f"   🔍 The session management fix needs additional work")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_session_management_fix())
    if success:
        print("\n🎉 Session management fix is working correctly!")
    else:
        print("\n⚠️  Session management fix needs additional work")