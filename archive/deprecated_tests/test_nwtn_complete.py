#!/usr/bin/env python3
"""
Test Script for NWTN Complete System
====================================

This script validates the complete NWTN system including:
1. NWTN Voicebox natural language interface
2. Multi-modal reasoning engine integration
3. API key configuration and management
4. Query processing pipeline
5. Cost estimation and FTNS integration

Usage:
    python test_nwtn_complete.py
"""

import asyncio
import sys
import os

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.complete_system import NWTNCompleteSystem


async def test_system_initialization():
    """Test system initialization"""
    print("🧪 Testing NWTN Complete System Initialization...")
    
    try:
        system = NWTNCompleteSystem()
        await system.initialize()
        
        status = await system.get_system_status()
        
        print(f"✅ System initialized successfully")
        print(f"   Voicebox: {'✅' if status.voicebox_initialized else '❌'}")
        print(f"   Multi-Modal Engine: {'✅' if status.multi_modal_engine_initialized else '❌'}")
        print(f"   FTNS Service: {'✅' if status.ftns_service_initialized else '❌'}")
        print(f"   Supported Providers: {', '.join(status.supported_providers)}")
        
        return system, True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return None, False


async def test_api_configuration(system):
    """Test API key configuration"""
    print("\n🧪 Testing API Key Configuration...")
    
    try:
        user_id = "test_user_123"
        
        # Test valid configuration
        success = await system.configure_user_api(
            user_id=user_id,
            provider="claude",
            api_key="sk-test-key-12345678901234567890"
        )
        
        if success:
            print(f"✅ API key configured for user {user_id}")
            return user_id, True
        else:
            print(f"❌ API key configuration failed")
            return None, False
            
    except Exception as e:
        print(f"❌ API configuration test failed: {e}")
        return None, False


async def test_query_cost_estimation(system, user_id):
    """Test query cost estimation"""
    print("\n🧪 Testing Query Cost Estimation...")
    
    try:
        test_queries = [
            "What is water?",
            "Compare different approaches to renewable energy",
            "What are the most promising breakthrough technologies for commercial atomically precise manufacturing?",
            "If we could manipulate matter at the atomic level, what would be the implications for manufacturing, medicine, and environmental sustainability?"
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\n📊 Query {i+1}: {query[:50]}...")
            
            estimate = await system.estimate_query_cost(user_id, query)
            
            print(f"   Complexity: {estimate.get('complexity', 'unknown')}")
            print(f"   Estimated Cost: {estimate.get('estimated_cost_ftns', 0):.2f} FTNS")
            print(f"   Reasoning Modes: {', '.join(estimate.get('estimated_reasoning_modes', []))}")
            print(f"   Needs Clarification: {estimate.get('requires_clarification', False)}")
            
        print(f"✅ Cost estimation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Cost estimation test failed: {e}")
        return False


async def test_query_processing(system, user_id):
    """Test query processing pipeline"""
    print("\n🧪 Testing Query Processing Pipeline...")
    
    try:
        # Test simple query
        simple_query = "What is the chemical formula for water?"
        print(f"📝 Processing simple query: {simple_query}")
        
        response = await system.process_query(
            user_id=user_id,
            query=simple_query,
            show_reasoning_trace=True
        )
        
        print(f"✅ Query processed successfully")
        print(f"   Response Length: {len(response.natural_language_response)} characters")
        print(f"   Confidence Score: {response.confidence_score:.2f}")
        print(f"   Reasoning Modes: {', '.join(response.used_reasoning_modes)}")
        print(f"   Processing Time: {response.processing_time_seconds:.2f}s")
        print(f"   Cost: {response.total_cost_ftns:.2f} FTNS")
        
        # Show truncated response
        print(f"   Response Preview: {response.natural_language_response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Query processing test failed: {e}")
        print(f"   Error details: {str(e)}")
        return False


async def test_batch_processing(system, user_id):
    """Test batch query processing"""
    print("\n🧪 Testing Batch Query Processing...")
    
    try:
        batch_queries = [
            "What is photosynthesis?",
            "How does gravity work?",
            "What are the benefits of renewable energy?"
        ]
        
        print(f"📝 Processing batch of {len(batch_queries)} queries...")
        
        responses = await system.batch_process_queries(
            user_id=user_id,
            queries=batch_queries
        )
        
        print(f"✅ Batch processing completed")
        print(f"   Queries processed: {len(responses)}")
        
        total_cost = sum(r.total_cost_ftns for r in responses)
        avg_confidence = sum(r.confidence_score for r in responses) / len(responses)
        
        print(f"   Total cost: {total_cost:.2f} FTNS")
        print(f"   Average confidence: {avg_confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        return False


async def test_system_status(system):
    """Test system status reporting"""
    print("\n🧪 Testing System Status...")
    
    try:
        status = await system.get_system_status()
        
        print(f"✅ System status retrieved")
        print(f"   Users Configured: {status.total_users_configured}")
        print(f"   Queries Processed: {status.total_queries_processed}")
        print(f"   Average Confidence: {status.average_confidence_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ System status test failed: {e}")
        return False


async def run_all_tests():
    """Run comprehensive test suite"""
    print("🚀 NWTN Complete System Test Suite")
    print("=" * 50)
    
    # Track test results
    test_results = []
    
    # Test 1: System initialization
    system, init_success = await test_system_initialization()
    test_results.append(("System Initialization", init_success))
    
    if not init_success:
        print("❌ Cannot proceed with tests - system initialization failed")
        return
    
    # Test 2: API configuration
    user_id, config_success = await test_api_configuration(system)
    test_results.append(("API Configuration", config_success))
    
    if not config_success:
        print("❌ Cannot proceed with tests - API configuration failed")
        return
    
    # Test 3: Cost estimation
    cost_success = await test_query_cost_estimation(system, user_id)
    test_results.append(("Query Cost Estimation", cost_success))
    
    # Test 4: Query processing
    query_success = await test_query_processing(system, user_id)
    test_results.append(("Query Processing", query_success))
    
    # Test 5: Batch processing
    batch_success = await test_batch_processing(system, user_id)
    test_results.append(("Batch Processing", batch_success))
    
    # Test 6: System status
    status_success = await test_system_status(system)
    test_results.append(("System Status", status_success))
    
    # Test results summary
    print("\n" + "=" * 50)
    print("🧪 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Total Tests: {len(test_results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! NWTN Complete System is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. See details above.")
    
    # Shutdown system
    await system.shutdown()


if __name__ == "__main__":
    print("Starting NWTN Complete System tests...")
    asyncio.run(run_all_tests())