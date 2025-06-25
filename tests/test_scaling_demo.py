#!/usr/bin/env python3
"""
Quick Scaling Test Demo
Fast demonstration of PRSM scaling test capabilities
"""

import sys
import asyncio
from pathlib import Path

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent
sys.path.insert(0, str(PRSM_ROOT))

async def run_quick_scaling_demo():
    """Run a very quick scaling demo"""
    print("🎯 PRSM Scaling Test Controller - Quick Demo")
    print("=" * 60)
    
    from prsm.performance.scaling_test_controller import (
        ScalingTestController, ScalingTestConfig, ScalingEnvironment,
        ResourceProfile, NetworkTopology
    )
    from comprehensive_performance_benchmark import NetworkCondition
    
    controller = ScalingTestController("demo_scaling_results")
    
    # Create a very lightweight demo config
    demo_config = ScalingTestConfig(
        name="quick_scaling_demo",
        environment=ScalingEnvironment.LOCAL_SIMULATION,
        node_counts=[10, 20, 40],  # Small node counts
        resource_profile=ResourceProfile.STANDARD,
        network_topology=NetworkTopology.MESH,
        network_conditions=[NetworkCondition.WAN],
        test_duration_per_scale=2,  # Very short
        warmup_duration=0,  # No warmup
        cooldown_duration=0,  # No cooldown
        byzantine_ratios=[0.0],  # No Byzantine nodes for speed
        target_operations_per_second=15.0,
        enable_resource_monitoring=False,  # Disable for speed
        metadata={"demo": True, "quick_test": True}
    )
    
    print(f"🚀 Running quick scaling test: {demo_config.name}")
    print(f"   Node counts: {demo_config.node_counts}")
    print(f"   Duration per scale: {demo_config.test_duration_per_scale}s")
    
    try:
        # Run the test
        result = await controller.run_comprehensive_scaling_test(demo_config)
        
        # Save results
        controller.save_scaling_results(result)
        
        print(f"\n✅ Quick demo completed successfully!")
        
        # Show key results
        print(f"\n📊 QUICK RESULTS SUMMARY:")
        for node_count in sorted(result.node_performance.keys()):
            perf = result.node_performance[node_count]
            efficiency = result.scaling_efficiency.get(node_count, 0)
            print(f"   {node_count:2d} nodes: {perf['operations_per_second']:6.2f} ops/s, "
                  f"{perf['mean_latency_ms']:5.1f}ms, {efficiency:.3f} efficiency")
        
        print(f"\n🎯 Key Findings:")
        print(f"   Recommended max nodes: {result.recommended_max_nodes}")
        if result.throughput_saturation_point:
            print(f"   Throughput saturation: {result.throughput_saturation_point} nodes")
        if result.latency_degradation_point:
            print(f"   Latency degradation: {result.latency_degradation_point} nodes")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("⚡ PRSM Quick Scaling Demo")
    print("=" * 40)
    
    success = asyncio.run(run_quick_scaling_demo())
    
    if success:
        print("\n🎉 Quick scaling demo completed successfully!")
        print("📈 Scaling test controller is working correctly")
        print("🚀 Ready for comprehensive scaling analysis")
    else:
        print("\n❌ Quick scaling demo failed")
    
    sys.exit(0 if success else 1)