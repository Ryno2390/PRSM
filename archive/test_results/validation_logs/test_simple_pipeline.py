#!/usr/bin/env python3
"""
Simple Pipeline Test
===================

Tests the core components of the NWTN pipeline without the heavy reasoning engines.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prsm.nwtn.system_integrator import SystemIntegrator

async def test_simple_pipeline():
    """Test the simple pipeline with mock data"""
    print("🚀 Starting Simple Pipeline Test")
    print("=" * 50)
    
    # Create integrator with forced mock retriever
    integrator = SystemIntegrator(force_mock_retriever=True)
    
    # Initialize
    print("\n📋 Initializing System Integrator...")
    await integrator.initialize()
    
    if integrator.initialized:
        print("✅ System Integrator initialized successfully")
    else:
        print("❌ System Integrator initialization failed")
        return False
    
    # Process a simple query
    print("\n📋 Processing Query...")
    query = "How can quantum error correction improve qubit stability?"
    user_id = "test_user"
    
    try:
        result = await integrator.process_complete_query(query, user_id)
        
        if result.success:
            print("✅ Query processed successfully!")
            print(f"   Session ID: {result.session_id}")
            print(f"   Final Response: {result.final_response[:100]}...")
            print(f"   Citations: {len(result.citations)}")
            print(f"   Payment Distributions: {len(result.payment_distributions)}")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            print(f"   Quality Score: {result.quality_score:.2f}")
            
            # Show payment distributions
            if result.payment_distributions:
                print("\n💰 Payment Distributions:")
                for dist in result.payment_distributions:
                    print(f"   - {dist['creator_id']}: {dist['payment_amount']} FTNS")
            
            return True
        else:
            print("❌ Query processing failed")
            print(f"   Error: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during query processing: {e}")
        return False

async def main():
    """Run the simple pipeline test"""
    success = await test_simple_pipeline()
    
    if success:
        print("\n🎉 Simple Pipeline Test PASSED!")
        print("   The core System 1 → System 2 → Attribution → Payment pipeline is working")
    else:
        print("\n❌ Simple Pipeline Test FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())