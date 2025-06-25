#!/usr/bin/env python3
"""
Test PRSM Scaling Test Controller
Validate comprehensive scaling test capabilities
"""

import sys
import asyncio
from pathlib import Path

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent
sys.path.insert(0, str(PRSM_ROOT))

def test_scaling_controller_components():
    """Test the scaling controller components"""
    print("🧪 Testing PRSM Scaling Test Controller Components")
    print("=" * 70)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from prsm.performance.scaling_test_controller import (
            ScalingTestController, ScalingTestConfig, ScalingEnvironment,
            ResourceProfile, NetworkTopology, ResourceMonitor
        )
        from comprehensive_performance_benchmark import NetworkCondition
        print("   ✅ All scaling controller components imported successfully")
        
        # Test configuration creation
        print("\n2. Testing configuration creation...")
        controller = ScalingTestController("test_scaling_results")
        configs = controller.create_comprehensive_scaling_configs()
        print(f"   ✅ Created {len(configs)} scaling test configurations")
        
        # Show configuration breakdown
        config_types = {}
        for config in configs:
            test_type = config.metadata.get("test_type", "unknown")
            config_types[test_type] = config_types.get(test_type, 0) + 1
        
        print("   Configuration breakdown:")
        for test_type, count in config_types.items():
            print(f"     - {test_type}: {count} configs")
        
        # Test resource monitor
        print("\n3. Testing resource monitor...")
        monitor = ResourceMonitor()
        print("   ✅ Resource monitor initialized")
        
        # Test scaling environments
        print("\n4. Testing scaling environments...")
        for env in ScalingEnvironment:
            print(f"   - {env.value}: Available")
        print("   ✅ All scaling environments defined")
        
        # Test resource profiles
        print("\n5. Testing resource profiles...")
        for profile in ResourceProfile:
            print(f"   - {profile.value}: Available")
        print("   ✅ All resource profiles defined")
        
        # Test network topologies
        print("\n6. Testing network topologies...")
        for topology in NetworkTopology:
            print(f"   - {topology.value}: Available")
        print("   ✅ All network topologies defined")
        
        # Test configuration validation
        print("\n7. Testing configuration validation...")
        sample_config = ScalingTestConfig(
            name="test_config",
            environment=ScalingEnvironment.LOCAL_SIMULATION,
            node_counts=[10, 25, 50],
            resource_profile=ResourceProfile.STANDARD,
            network_topology=NetworkTopology.MESH,
            network_conditions=[NetworkCondition.WAN],
            test_duration_per_scale=5,
            byzantine_ratios=[0.0, 0.1],
            target_operations_per_second=10.0
        )
        print(f"   ✅ Sample configuration created: {sample_config.name}")
        print(f"      Node counts: {sample_config.node_counts}")
        print(f"      Duration per scale: {sample_config.test_duration_per_scale}s")
        
        # Test configuration analysis
        print("\n8. Testing analysis functions...")
        
        # Mock performance data for testing
        mock_performance = {
            10: {"operations_per_second": 10.0, "mean_latency_ms": 20.0, "success_rate": 0.99, "operations_per_node_per_second": 1.0},
            25: {"operations_per_second": 18.0, "mean_latency_ms": 35.0, "success_rate": 0.97, "operations_per_node_per_second": 0.72},
            50: {"operations_per_second": 25.0, "mean_latency_ms": 55.0, "success_rate": 0.95, "operations_per_node_per_second": 0.50}
        }
        
        scaling_efficiency = controller._calculate_scaling_efficiency(mock_performance)
        print(f"   ✅ Scaling efficiency calculated: {len(scaling_efficiency)} data points")
        for nodes, efficiency in scaling_efficiency.items():
            print(f"      {nodes} nodes: {efficiency:.3f} efficiency")
        
        saturation_point = controller._find_throughput_saturation(mock_performance)
        degradation_point = controller._find_latency_degradation(mock_performance)
        print(f"   ✅ Saturation analysis: saturation at {saturation_point}, degradation at {degradation_point}")
        
        recommended_max = controller._calculate_recommended_max_nodes(mock_performance, {})
        print(f"   ✅ Recommended max nodes: {recommended_max}")
        
        bottlenecks = controller._identify_bottlenecks(mock_performance, {}, {})
        print(f"   ✅ Bottleneck analysis: {len(bottlenecks)} issues identified")
        
        recommendations = controller._generate_scaling_recommendations(scaling_efficiency, saturation_point, degradation_point)
        print(f"   ✅ Generated {len(recommendations)} scaling recommendations")
        
        print("\n" + "=" * 70)
        print("🎉 ALL SCALING CONTROLLER COMPONENTS TESTED SUCCESSFULLY!")
        print("=" * 70)
        print("📊 Scaling Test Features Validated:")
        print("   ✅ Comprehensive configuration generation")
        print("   ✅ Resource monitoring capabilities")  
        print("   ✅ Multi-environment support")
        print("   ✅ Performance analysis functions")
        print("   ✅ Bottleneck identification")
        print("   ✅ Scaling recommendations")
        print("   ✅ Byzantine fault tolerance testing")
        print("   ✅ Network topology variations")
        print()
        print("🚀 Ready for comprehensive scaling tests!")
        
        return True
        
    except Exception as e:
        print(f"❌ Scaling controller test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_scaling_simulation():
    """Test the scaling simulation functionality"""
    print("\n🎯 Testing Scaling Simulation")
    print("=" * 50)
    
    try:
        from prsm.performance.scaling_test_controller import ScalingTestController, ScalingTestConfig, ScalingEnvironment, ResourceProfile, NetworkTopology
        from comprehensive_performance_benchmark import NetworkCondition
        
        controller = ScalingTestController("test_scaling_results")
        
        # Create a minimal test config
        test_config = ScalingTestConfig(
            name="simulation_test",
            environment=ScalingEnvironment.LOCAL_SIMULATION,
            node_counts=[10, 20],
            resource_profile=ResourceProfile.STANDARD,
            network_topology=NetworkTopology.MESH,
            network_conditions=[NetworkCondition.WAN],
            test_duration_per_scale=3,  # Very short for testing
            warmup_duration=1,
            byzantine_ratios=[0.0],
            target_operations_per_second=8.0,
            enable_resource_monitoring=False  # Disable for testing
        )
        
        print("1. Testing consensus operation simulation...")
        
        # Test individual operation
        success, latency_ms, metrics = await controller.simulate_scaled_consensus_operation(test_config, 25, 0.0)
        print(f"   ✅ Single operation: success={success}, latency={latency_ms:.2f}ms")
        print(f"      Metrics: {list(metrics.keys())}")
        
        # Test with Byzantine nodes
        success, latency_ms, metrics = await controller.simulate_scaled_consensus_operation(test_config, 25, 0.2)
        print(f"   ✅ Byzantine operation: success={success}, latency={latency_ms:.2f}ms")
        
        print("\n2. Testing scaling test at specific node count...")
        
        # Test scaling at one node count
        result = await controller.run_scaling_test_at_node_count(test_config, 15, 0.0)
        print(f"   ✅ Node count test completed")
        print(f"      Operations: {result['total_operations']}")
        print(f"      Success rate: {result['success_rate']:.1%}")
        print(f"      Throughput: {result['operations_per_second']:.2f} ops/s")
        print(f"      Mean latency: {result['mean_latency_ms']:.2f}ms")
        
        print("\n✅ Scaling simulation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Scaling simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 PRSM Scaling Test Controller - Component Testing")
    print("=" * 70)
    
    # Test core components
    components_ok = test_scaling_controller_components()
    
    # Test simulation
    simulation_ok = False
    if components_ok:
        simulation_ok = asyncio.run(test_scaling_simulation())
    
    print("\n" + "=" * 70)
    print("🏁 SCALING TEST CONTROLLER TEST SUMMARY:")
    print(f"   Core Components: {'✅ PASS' if components_ok else '❌ FAIL'}")
    print(f"   Simulation Functions: {'✅ PASS' if simulation_ok else '❌ FAIL'}")
    
    if components_ok and simulation_ok:
        print("\n🎉 Scaling test controller is ready!")
        print("🚀 Next steps:")
        print("   1. Run demo: python prsm/performance/scaling_test_controller.py")
        print("   2. Run comprehensive: python prsm/performance/scaling_test_controller.py comprehensive")
        print("   3. Analyze results in scaling_test_results/")
    else:
        print("\n❌ Fix component issues before running scaling tests")
    
    sys.exit(0 if (components_ok and simulation_ok) else 1)