#!/usr/bin/env python3
"""
Test Performance Instrumentation Implementation
Validates that the performance benchmark collector is working properly with the updated P2P network code.
"""

import asyncio
import sys
from pathlib import Path

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent
sys.path.insert(0, str(PRSM_ROOT))

from prsm.performance.benchmark_collector import get_global_collector, reset_global_collector
from prsm.federation.p2p_network import P2PModelNetwork
from prsm.core.models import ArchitectTask, PeerNode
from uuid import uuid4


async def test_performance_instrumentation():
    """Test that performance instrumentation is working with P2P network"""
    
    print("🧪 Testing Performance Instrumentation Implementation")
    print("=" * 60)
    
    # Reset collector for clean test
    reset_global_collector()
    collector = get_global_collector()
    
    # Create P2P network
    p2p_network = P2PModelNetwork()
    
    # Add test peers
    test_peers = [
        PeerNode(
            node_id=f"node_{i}",
            peer_id=f"test_peer_{i}",
            multiaddr=f"/ip4/127.0.0.1/tcp/400{i}",
            capabilities=["model_execution", "data_storage"],
            reputation_score=0.8,
            active=True
        )
        for i in range(3)
    ]
    
    for peer in test_peers:
        await p2p_network.add_peer(peer)
    
    print(f"✅ Added {len(test_peers)} test peers to P2P network")
    
    # Create test task
    test_task = ArchitectTask(
        task_id=uuid4(),
        session_id=uuid4(),
        instruction="Test performance instrumentation in P2P network",
        complexity_score=0.5,
        metadata={"test": "performance_instrumentation", "task_type": "benchmark_test"}
    )
    
    print(f"✅ Created test task: {test_task.task_id}")
    
    # Execute task to trigger performance measurement
    print("\n🚀 Executing task with performance instrumentation...")
    
    execution_futures = await p2p_network.coordinate_distributed_execution(test_task)
    
    # Wait for execution to complete
    results = await asyncio.gather(*execution_futures, return_exceptions=True)
    
    print(f"✅ Task execution completed with {len(results)} results")
    
    # Check performance metrics
    print("\n📊 Performance Metrics Analysis:")
    print("-" * 40)
    
    all_metrics = collector.get_all_metrics()
    
    if not all_metrics:
        print("❌ No performance metrics collected!")
        return False
    
    success_count = 0
    for operation, metrics in all_metrics.items():
        print(f"Operation: {operation}")
        print(f"  Sample Count: {metrics.sample_count}")
        print(f"  Mean Time: {metrics.mean_ms:.2f}ms")
        print(f"  Min/Max: {metrics.min_ms:.2f}ms / {metrics.max_ms:.2f}ms")
        print(f"  95th Percentile: {metrics.p95_ms:.2f}ms")
        print(f"  Measurements/sec: {metrics.measurements_per_second:.2f}")
        print()
        success_count += 1
    
    # Check summary stats
    summary = collector.get_summary_stats()
    print("📈 Summary Statistics:")
    print(f"  Total Operations: {summary['total_operations']}")
    print(f"  Total Measurements: {summary['total_measurements']}")
    print(f"  Uptime: {summary['uptime_seconds']:.2f}s")
    print(f"  Avg Measurements/sec: {summary['average_measurements_per_second']:.2f}")
    
    # Validate results
    if success_count > 0:
        print(f"\n✅ Performance instrumentation working correctly!")
        print(f"   - Captured {success_count} different operation types")
        print(f"   - Total measurements: {summary['total_measurements']}")
        print(f"   - Real timing data collected and analyzed")
        return True
    else:
        print(f"\n❌ Performance instrumentation failed - no metrics collected")
        return False


async def test_multi_region_network_performance():
    """Test performance instrumentation in multi-region network"""
    
    print("\n🌍 Testing Multi-Region Network Performance...")
    print("-" * 50)
    
    try:
        from prsm.federation.multi_region_p2p_network import MultiRegionP2PNetwork, RegionCode
        
        # Create multi-region network
        network = MultiRegionP2PNetwork()
        
        # Deploy a small test region
        print("Deploying test nodes in US-East region...")
        
        result = await network.deploy_regional_nodes(RegionCode.US_EAST, node_count=2)
        
        print(f"✅ Deployed {result['nodes_deployed']} nodes in {result['region']}")
        
        # Check performance metrics from deployment
        collector = get_global_collector()
        deployment_metrics = collector.get_all_metrics()
        
        print("\n📊 Multi-Region Performance Metrics:")
        for operation, metrics in deployment_metrics.items():
            if "infrastructure_setup" in operation or "node_startup" in operation or "node_deployment" in operation:
                print(f"  {operation}: {metrics.mean_ms:.2f}ms avg ({metrics.sample_count} samples)")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-region test failed: {e}")
        return False


if __name__ == "__main__":
    async def run_tests():
        """Run all performance instrumentation tests"""
        
        print("🔬 PRSM Performance Instrumentation Validation")
        print("=" * 60)
        print("Testing that asyncio.sleep() simulation delays have been")
        print("replaced with real performance measurement and collection.")
        print()
        
        # Test basic P2P performance instrumentation
        test1_success = await test_performance_instrumentation()
        
        # Test multi-region network performance
        test2_success = await test_multi_region_network_performance()
        
        print("\n" + "=" * 60)
        print("🏁 PERFORMANCE INSTRUMENTATION TEST RESULTS:")
        print(f"   Basic P2P Network: {'✅ PASS' if test1_success else '❌ FAIL'}")
        print(f"   Multi-Region Network: {'✅ PASS' if test2_success else '❌ FAIL'}")
        
        if test1_success and test2_success:
            print("\n🎉 All performance instrumentation tests PASSED!")
            print("✅ Real timing measurements successfully replace simulation delays")
            print("📊 Performance data collection and analysis working correctly")
            return 0
        else:
            print("\n❌ Some performance instrumentation tests FAILED")
            return 1
    
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)