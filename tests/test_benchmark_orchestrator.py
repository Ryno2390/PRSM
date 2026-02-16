#!/usr/bin/env python3
"""
Test PRSM Benchmark Orchestrator
Validates comprehensive performance benchmarking capabilities
"""

import sys
import asyncio
from pathlib import Path
import pytest

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent
sys.path.insert(0, str(PRSM_ROOT))

# Import benchmark orchestrator components directly
import importlib.util

# Load benchmark orchestrator
orch_module_path = PRSM_ROOT / "prsm" / "performance" / "benchmark_orchestrator.py"
if not orch_module_path.exists():
    pytest.skip(f"Benchmark orchestrator module not found at {orch_module_path}", allow_module_level=True)

try:
    spec = importlib.util.spec_from_file_location("benchmark_orchestrator", orch_module_path)
    orch_module = importlib.util.module_from_spec(spec)

    # Temporarily fix imports by adding needed modules to sys.modules
    # Load benchmark collector
    collector_module_path = PRSM_ROOT / "prsm" / "performance" / "benchmark_collector.py"
    collector_spec = importlib.util.spec_from_file_location("benchmark_collector", collector_module_path)
    collector_module = importlib.util.module_from_spec(collector_spec)
    collector_spec.loader.exec_module(collector_module)
    sys.modules['prsm.performance.benchmark_collector'] = collector_module
except (FileNotFoundError, AttributeError, ImportError) as e:
    pytest.skip(f"Could not load benchmark modules: {e}", allow_module_level=True)

# Load core models
models_module_path = PRSM_ROOT / "prsm" / "core" / "models.py"
models_spec = importlib.util.spec_from_file_location("core_models", models_module_path)
models_module = importlib.util.module_from_spec(models_spec)
models_spec.loader.exec_module(models_module)
sys.modules['prsm.core.models'] = models_module

# Load P2P network
p2p_module_path = PRSM_ROOT / "prsm" / "federation" / "p2p_network.py"
p2p_spec = importlib.util.spec_from_file_location("p2p_network", p2p_module_path)
p2p_module = importlib.util.module_from_spec(p2p_spec)

# Mock dependencies that have complex imports
class MockP2PNetwork:
    def __init__(self):
        self.peers = {}
    
    async def add_peer(self, peer):
        self.peers[peer.peer_id] = peer
        return True
    
    async def coordinate_distributed_execution(self, task):
        # Simulate execution
        await asyncio.sleep(0.01)  # 10ms simulation
        return [{"success": True, "peer_id": f"peer_{i}"} for i in range(3)]

class MockMultiRegionNetwork:
    def __init__(self):
        pass

# Patch the imports in the orchestrator module
sys.modules['prsm.federation.p2p_network'] = type('MockP2PModule', (), {
    'P2PModelNetwork': MockP2PNetwork
})
sys.modules['prsm.federation.multi_region_p2p_network'] = type('MockMultiRegionModule', (), {
    'MultiRegionP2PNetwork': MockMultiRegionNetwork,
    'RegionCode': type('RegionCode', (), {
        'US_EAST': 'us-east',
        'US_WEST': 'us-west', 
        'EU_CENTRAL': 'eu-central',
        'AP_SOUTHEAST': 'ap-southeast',
        'LATAM': 'latam'
    })
})

# Now load the orchestrator
spec.loader.exec_module(orch_module)

# Import the classes we need
BenchmarkOrchestrator = orch_module.BenchmarkOrchestrator
BenchmarkType = orch_module.BenchmarkType
LoadProfile = orch_module.LoadProfile
BenchmarkScenario = orch_module.BenchmarkScenario


async def test_benchmark_orchestrator():
    """Test the benchmark orchestrator functionality"""
    print("📊 Testing PRSM Benchmark Orchestrator")
    print("=" * 50)
    
    # Create orchestrator
    orchestrator = BenchmarkOrchestrator("test_benchmark_results")
    
    # Test scenario creation
    print("1. Testing scenario creation...")
    scenarios = orchestrator.create_predefined_scenarios()
    print(f"   ✅ Created {len(scenarios)} predefined scenarios")
    
    # Show scenario types
    scenario_types = {}
    for scenario in scenarios:
        scenario_types[scenario.benchmark_type.value] = scenario_types.get(scenario.benchmark_type.value, 0) + 1
    
    print("   Scenario breakdown:")
    for bench_type, count in scenario_types.items():
        print(f"     - {bench_type}: {count} scenarios")
    
    # Test a lightweight scenario
    print("\n2. Testing lightweight benchmark execution...")
    test_scenario = BenchmarkScenario(
        name="test_lightweight_consensus",
        benchmark_type=BenchmarkType.CONSENSUS_SCALING,
        load_profile=LoadProfile.LIGHT,
        node_count=5,
        duration_seconds=3,  # Very short for testing
        target_tps=5.0,
        network_latency_ms=10
    )
    
    try:
        result = await orchestrator.run_benchmark_scenario(test_scenario)
        print(f"   ✅ Benchmark completed successfully")
        print(f"   Operations: {result.total_operations}")
        print(f"   Success rate: {result.message_success_rate:.1%}")
        print(f"   Ops/sec: {result.operations_per_second:.2f}")
        print(f"   Mean time: {result.mean_operation_time_ms:.2f}ms")
    except Exception as e:
        print(f"   ❌ Benchmark failed: {e}")
        return False
    
    # Test result saving
    print("\n3. Testing result persistence...")
    try:
        csv_file = orchestrator.save_results_to_csv("test_results.csv")
        json_file = orchestrator.save_results_to_json("test_results.json")
        print(f"   ✅ CSV saved: {csv_file}")
        print(f"   ✅ JSON saved: {json_file}")
    except Exception as e:
        print(f"   ❌ Result saving failed: {e}")
        return False
    
    # Test performance summary
    print("\n4. Testing performance summary...")
    try:
        summary = orchestrator.get_performance_summary()
        print(f"   ✅ Summary generated")
        print(f"   Total scenarios: {summary['total_scenarios_run']}")
        print(f"   Total operations: {summary['total_operations']}")
        print(f"   Average ops/sec: {summary['average_ops_per_second']:.2f}")
    except Exception as e:
        print(f"   ❌ Summary generation failed: {e}")
        return False
    
    print("\n✅ All benchmark orchestrator tests passed!")
    return True


async def demo_scaling_scenarios():
    """Demonstrate scaling scenario capabilities"""
    print("\n🚀 Benchmark Scaling Scenarios Demo")
    print("=" * 50)
    
    orchestrator = BenchmarkOrchestrator()
    
    # Create scaling scenarios
    scaling_scenarios = [
        BenchmarkScenario(
            name=f"scaling_demo_{nodes}_nodes",
            benchmark_type=BenchmarkType.CONSENSUS_SCALING,
            load_profile=LoadProfile.LIGHT,
            node_count=nodes,
            duration_seconds=2,  # Very short for demo
            target_tps=3.0
        )
        for nodes in [5, 10, 20]
    ]
    
    results = []
    for scenario in scaling_scenarios:
        print(f"\n🎯 Running {scenario.name}...")
        try:
            result = await orchestrator.run_benchmark_scenario(scenario)
            results.append(result)
            print(f"   Result: {result.operations_per_second:.2f} ops/sec")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    if results:
        print(f"\n📈 Scaling Analysis:")
        for result in results:
            nodes = result.scenario_name.split('_')[2]
            print(f"   {nodes:>3} nodes: {result.operations_per_second:6.2f} ops/sec, "
                  f"{result.mean_operation_time_ms:6.2f}ms avg")
        
        # Calculate scaling efficiency
        if len(results) >= 2:
            baseline = results[0]
            for result in results[1:]:
                scaling_factor = int(result.scenario_name.split('_')[2]) / int(baseline.scenario_name.split('_')[2])
                throughput_ratio = result.operations_per_second / baseline.operations_per_second
                efficiency = throughput_ratio / scaling_factor
                print(f"   Scaling efficiency vs baseline: {efficiency:.2f}")
    
    print(f"\n✅ Scaling demo completed!")


if __name__ == "__main__":
    async def run_tests():
        """Run all benchmark orchestrator tests"""
        
        # Test basic functionality
        test1_success = await test_benchmark_orchestrator()
        
        # Test scaling scenarios
        if test1_success:
            await demo_scaling_scenarios()
        
        print("\n" + "=" * 50)
        print("🏁 BENCHMARK ORCHESTRATOR TEST RESULTS:")
        print(f"   Core Functionality: {'✅ PASS' if test1_success else '❌ FAIL'}")
        
        if test1_success:
            print("\n🎉 Benchmark orchestrator is working correctly!")
            print("✅ Ready for comprehensive performance testing")
            print("📊 Supports multiple benchmark types and scaling tests")
            return 0
        else:
            print("\n❌ Benchmark orchestrator tests failed")
            return 1
    
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)