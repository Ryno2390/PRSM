#!/usr/bin/env python3
"""
Simple Performance Instrumentation Test
Tests the core performance measurement functionality directly.
"""

import asyncio
import sys
from pathlib import Path

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent
sys.path.insert(0, str(PRSM_ROOT))

from prsm.performance.benchmark_collector import (
    get_global_collector, reset_global_collector, 
    time_async_operation, performance_timer
)


async def test_basic_performance_measurement():
    """Test basic performance measurement functionality"""
    
    print("🧪 Testing Basic Performance Measurement")
    print("=" * 50)
    
    # Reset collector for clean test
    reset_global_collector()
    collector = get_global_collector()
    
    # Test 1: Async context manager
    print("Test 1: Async context manager timing...")
    async with time_async_operation("test_operation", {"test": "context_manager"}):
        await asyncio.sleep(0.1)  # Simulate 100ms operation
    
    # Test 2: Decorator timing
    print("Test 2: Decorator timing...")
    
    @performance_timer("decorated_operation", {"test": "decorator"})
    async def test_decorated_function():
        await asyncio.sleep(0.05)  # Simulate 50ms operation
        return "decorated_result"
    
    result = await test_decorated_function()
    
    # Test 3: Multiple operations
    print("Test 3: Multiple operations...")
    for i in range(3):
        async with time_async_operation("batch_operation", {"batch": i}):
            await asyncio.sleep(0.02)  # Simulate 20ms operations
    
    # Check collected metrics
    print("\n📊 Collected Performance Metrics:")
    print("-" * 40)
    
    all_metrics = collector.get_all_metrics()
    
    if not all_metrics:
        print("❌ No metrics collected!")
        return False
    
    success_count = 0
    for operation, metrics in all_metrics.items():
        print(f"Operation: {operation}")
        print(f"  Sample Count: {metrics.sample_count}")
        print(f"  Mean Time: {metrics.mean_ms:.2f}ms")
        print(f"  Min/Max: {metrics.min_ms:.2f}ms / {metrics.max_ms:.2f}ms")
        print(f"  Standard Deviation: {metrics.std_dev_ms:.2f}ms")
        print()
        success_count += 1
    
    # Check summary
    summary = collector.get_summary_stats()
    print("📈 Summary Statistics:")
    print(f"  Total Operations: {summary['total_operations']}")
    print(f"  Total Measurements: {summary['total_measurements']}")
    print(f"  Operations: {summary['operations']}")
    
    if success_count >= 3:  # test_operation, decorated_operation, batch_operation
        print(f"\n✅ Performance measurement working correctly!")
        print(f"   - Captured {success_count} operation types")
        print(f"   - Total measurements: {summary['total_measurements']}")
        return True
    else:
        print(f"\n❌ Performance measurement failed")
        print(f"   - Expected 3+ operations, got {success_count}")
        return False


async def test_p2p_network_integration():
    """Test performance instrumentation with actual P2P code"""
    
    print("\n🌐 Testing P2P Network Integration")
    print("=" * 50)
    
    try:
        from prsm.compute.federation.p2p_network import P2PModelNetwork
        from prsm.core.models import PeerNode, ArchitectTask
        from uuid import uuid4
        
        # Create P2P network
        p2p_network = P2PModelNetwork()
        
        # Add multiple test peers (need at least 2 for execution)
        test_peers = []
        for i in range(3):
            test_peer = PeerNode(
                node_id=f"test_node_{i}",
                peer_id=f"test_peer_{i}",
                multiaddr=f"/ip4/127.0.0.1/tcp/400{i}",
                capabilities=["model_execution"],
                reputation_score=0.8,
                active=True
            )
            test_peers.append(test_peer)
            await p2p_network.add_peer(test_peer)
        
        print(f"✅ Added {len(test_peers)} test peers to P2P network")
        
        # Create and execute a test task (this should trigger performance measurement)
        test_task = ArchitectTask(
            task_id=uuid4(),
            session_id=uuid4(),
            instruction="Test performance measurement",
            complexity_score=0.3
        )
        
        print("🚀 Executing task to test performance instrumentation...")
        
        # This should trigger our performance instrumentation in _execute_on_peer
        execution_futures = await p2p_network.coordinate_distributed_execution(test_task)
        results = await asyncio.gather(*execution_futures, return_exceptions=True)
        
        print(f"✅ Task executed with {len(results)} results")
        
        # Check if peer execution metrics were collected
        collector = get_global_collector()
        all_metrics = collector.get_all_metrics()
        
        print(f"\n📊 All Performance Metrics After P2P Execution:")
        if all_metrics:
            for operation, metrics in all_metrics.items():
                print(f"  {operation}: {metrics.mean_ms:.2f}ms avg ({metrics.sample_count} samples)")
        else:
            print("  No metrics collected")
        
        # Check for peer execution metrics specifically
        peer_metrics = {k: v for k, v in all_metrics.items() 
                       if "peer_execution" in k}
        
        if peer_metrics:
            print("\n✅ P2P Performance Metrics Found:")
            for operation, metrics in peer_metrics.items():
                print(f"  {operation}: {metrics.mean_ms:.2f}ms avg ({metrics.sample_count} samples)")
            return True
        else:
            print("\n❌ No P2P performance metrics collected")
            print("   This suggests the performance instrumentation in _execute_on_peer may not be working")
            return False
            
    except Exception as e:
        print(f"❌ P2P integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    async def run_tests():
        """Run all performance tests"""
        
        print("🔬 Performance Instrumentation Validation")
        print("=" * 60)
        print("Testing performance measurement and collection framework.")
        print()
        
        # Test basic functionality
        test1_success = await test_basic_performance_measurement()
        
        # Test P2P integration
        test2_success = await test_p2p_network_integration()
        
        print("\n" + "=" * 60)
        print("🏁 PERFORMANCE INSTRUMENTATION TEST RESULTS:")
        print(f"   Basic Measurement: {'✅ PASS' if test1_success else '❌ FAIL'}")
        print(f"   P2P Integration: {'✅ PASS' if test2_success else '❌ FAIL'}")
        
        if test1_success and test2_success:
            print("\n🎉 Performance instrumentation is working correctly!")
            print("✅ Real timing measurements successfully implemented")
            print("📊 Performance data collection operational")
            return 0
        else:
            print("\n❌ Performance instrumentation tests failed")
            return 1
    
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)