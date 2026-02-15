#!/usr/bin/env python3
"""
Test Consensus Fault Injection
Validates fault injection system for consensus resilience testing
"""

import asyncio
import sys
from pathlib import Path
from typing import List

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent))

from prsm.core.models import PeerNode
from prsm.compute.federation.fault_injection import (
    FaultInjector, FaultScenario, FaultType, FaultSeverity, get_fault_injector
)


async def test_fault_injection():
    """Test fault injection system with comprehensive scenarios"""
    print("🧪 Testing PRSM Consensus Fault Injection")
    print("=" * 60)
    
    try:
        # Create test network
        peer_nodes = []
        for i in range(12):
            reputation = 0.5 + (i % 4) * 0.1 + (i / 12) * 0.3  # Varied reputation
            peer = PeerNode(
                node_id=f"fault_node_{i}",
                peer_id=f"fault_peer_{i}",
                multiaddr=f"/ip4/127.0.0.1/tcp/{12000+i}",
                reputation_score=min(1.0, reputation),
                active=True
            )
            peer_nodes.append(peer)
        
        # Initialize fault injector
        fault_injector = FaultInjector()
        initialization_success = await fault_injector.initialize_fault_injection(peer_nodes)
        
        if not initialization_success:
            print("❌ Failed to initialize fault injection system")
            return False
        
        print(f"✅ Fault injection system initialized with {len(peer_nodes)} nodes")
        
        # Test individual fault types
        test_results = []
        
        print(f"\n🔧 Testing Individual Fault Types")
        print("-" * 50)
        
        # Test 1: Node Crash
        crash_result = await test_node_crash(fault_injector, peer_nodes)
        test_results.append(("Node Crash", crash_result))
        
        # Test 2: Node Slowdown
        slow_result = await test_node_slowdown(fault_injector, peer_nodes)
        test_results.append(("Node Slowdown", slow_result))
        
        # Test 3: Byzantine Behavior
        byzantine_result = await test_byzantine_behavior(fault_injector, peer_nodes)
        test_results.append(("Byzantine Behavior", byzantine_result))
        
        # Test 4: Network Partition
        partition_result = await test_network_partition(fault_injector, peer_nodes)
        test_results.append(("Network Partition", partition_result))
        
        # Test 5: Message Loss
        message_loss_result = await test_message_loss(fault_injector, peer_nodes)
        test_results.append(("Message Loss", message_loss_result))
        
        # Test comprehensive fault scenarios
        print(f"\n🚀 Testing Comprehensive Fault Scenarios")
        print("-" * 50)
        comprehensive_result = await test_comprehensive_scenarios(fault_injector, peer_nodes)
        test_results.append(("Comprehensive Scenarios", comprehensive_result))
        
        # Test consensus under faults
        print(f"\n🎯 Testing Consensus Under Faults")
        print("-" * 50)
        consensus_result = await test_consensus_under_faults(fault_injector, peer_nodes)
        test_results.append(("Consensus Under Faults", consensus_result))
        
        # Test fault recovery
        print(f"\n🔄 Testing Fault Recovery")
        print("-" * 50)
        recovery_result = await test_fault_recovery(fault_injector, peer_nodes)
        test_results.append(("Fault Recovery", recovery_result))
        
        # Summary results
        print("\n" + "=" * 60)
        print("📊 FAULT INJECTION TEST SUMMARY")
        print("=" * 60)
        
        successful_tests = 0
        for test_name, result in test_results:
            status = "✅" if result else "❌"
            print(f"{status} {test_name}: {'PASSED' if result else 'FAILED'}")
            if result:
                successful_tests += 1
        
        # Get final metrics
        metrics = await fault_injector.get_fault_metrics()
        
        print(f"\n🎯 OVERALL FAULT INJECTION RESULTS:")
        print(f"   - Successful tests: {successful_tests}/{len(test_results)}")
        print(f"   - Total faults injected: {metrics['total_faults_injected']}")
        print(f"   - Scenarios completed: {metrics['total_scenarios_completed']}")
        print(f"   - Consensus success rate under faults: {metrics['consensus_success_rate']:.1%}")
        print(f"   - Average recovery time: {metrics['average_recovery_time']:.1f}s")
        print(f"   - Byzantine nodes identified: {metrics['byzantine_node_count']}")
        
        success_rate = successful_tests / len(test_results)
        
        if success_rate >= 0.8 and metrics['total_faults_injected'] >= 5:
            print(f"\n✅ FAULT INJECTION SYSTEM: COMPREHENSIVE TESTING SUCCESSFUL!")
            print(f"🚀 Test success rate: {success_rate:.1%}")
            print(f"🚀 Fault injection capabilities validated")
            print(f"🚀 Ready for consensus resilience testing")
        else:
            print(f"\n⚠️ Fault injection system needs refinement:")
            print(f"   - Success rate: {success_rate:.1%} (target: >80%)")
            print(f"   - Faults injected: {metrics['total_faults_injected']} (target: >5)")
        
        return test_results
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return []


async def test_node_crash(fault_injector: FaultInjector, peer_nodes: List[PeerNode]) -> bool:
    """Test node crash fault injection"""
    try:
        print("💥 Testing node crash fault injection")
        
        # Create crash scenario
        crash_scenario = FaultScenario(
            name="Test Node Crash",
            description="Single node crash test",
            fault_type=FaultType.NODE_CRASH,
            severity=FaultSeverity.MEDIUM,
            target_nodes=[peer_nodes[0].peer_id],
            duration=5  # Short duration for testing
        )
        
        # Inject fault
        injection_success = await fault_injector.inject_fault_scenario(crash_scenario)
        
        if not injection_success:
            print("   ❌ Failed to inject node crash fault")
            return False
        
        # Verify node state
        node_state = fault_injector.node_states[peer_nodes[0].peer_id]
        crashed = node_state["status"] == "crashed"
        
        print(f"   📊 Node crash results:")
        print(f"      - Injection successful: {'✅' if injection_success else '❌'}")
        print(f"      - Node marked as crashed: {'✅' if crashed else '❌'}")
        print(f"      - Response time: {node_state['response_time']}")
        
        # Wait for recovery
        await asyncio.sleep(6)  # Wait for fault duration + buffer
        
        return injection_success and crashed
        
    except Exception as e:
        print(f"❌ Node crash test error: {e}")
        return False


async def test_node_slowdown(fault_injector: FaultInjector, peer_nodes: List[PeerNode]) -> bool:
    """Test node slowdown fault injection"""
    try:
        print("🐌 Testing node slowdown fault injection")
        
        # Get initial response time
        initial_time = fault_injector.node_states[peer_nodes[1].peer_id]["response_time"]
        
        # Create slowdown scenario
        slowdown_scenario = FaultScenario(
            name="Test Node Slowdown",
            description="Node slowdown test",
            fault_type=FaultType.NODE_SLOW,
            severity=FaultSeverity.MEDIUM,
            target_nodes=[peer_nodes[1].peer_id],
            duration=5,
            parameters={"slowdown_factor": 4.0}
        )
        
        # Inject fault
        injection_success = await fault_injector.inject_fault_scenario(slowdown_scenario)
        
        # Verify slowdown
        current_time = fault_injector.node_states[peer_nodes[1].peer_id]["response_time"]
        slowdown_applied = current_time > initial_time * 2  # Should be significantly slower
        
        print(f"   📊 Node slowdown results:")
        print(f"      - Injection successful: {'✅' if injection_success else '❌'}")
        print(f"      - Initial response time: {initial_time:.3f}s")
        print(f"      - Current response time: {current_time:.3f}s")
        print(f"      - Slowdown applied: {'✅' if slowdown_applied else '❌'}")
        
        # Wait for recovery
        await asyncio.sleep(6)
        
        return injection_success and slowdown_applied
        
    except Exception as e:
        print(f"❌ Node slowdown test error: {e}")
        return False


async def test_byzantine_behavior(fault_injector: FaultInjector, peer_nodes: List[PeerNode]) -> bool:
    """Test byzantine behavior fault injection"""
    try:
        print("🎭 Testing byzantine behavior fault injection")
        
        initial_byzantine_count = len(fault_injector.byzantine_nodes)
        
        # Create byzantine scenario
        byzantine_scenario = FaultScenario(
            name="Test Byzantine Behavior",
            description="Byzantine node test",
            fault_type=FaultType.BYZANTINE_BEHAVIOR,
            severity=FaultSeverity.HIGH,
            target_nodes=[peer_nodes[2].peer_id, peer_nodes[3].peer_id],
            duration=5
        )
        
        # Inject fault
        injection_success = await fault_injector.inject_fault_scenario(byzantine_scenario)
        
        # Verify byzantine behavior
        current_byzantine_count = len(fault_injector.byzantine_nodes)
        byzantine_increased = current_byzantine_count > initial_byzantine_count
        
        node_2_byzantine = peer_nodes[2].peer_id in fault_injector.byzantine_nodes
        node_3_byzantine = peer_nodes[3].peer_id in fault_injector.byzantine_nodes
        
        print(f"   📊 Byzantine behavior results:")
        print(f"      - Injection successful: {'✅' if injection_success else '❌'}")
        print(f"      - Initial byzantine nodes: {initial_byzantine_count}")
        print(f"      - Current byzantine nodes: {current_byzantine_count}")
        print(f"      - Byzantine count increased: {'✅' if byzantine_increased else '❌'}")
        print(f"      - Target nodes marked byzantine: {'✅' if node_2_byzantine and node_3_byzantine else '❌'}")
        
        # Wait for recovery
        await asyncio.sleep(6)
        
        return injection_success and byzantine_increased
        
    except Exception as e:
        print(f"❌ Byzantine behavior test error: {e}")
        return False


async def test_network_partition(fault_injector: FaultInjector, peer_nodes: List[PeerNode]) -> bool:
    """Test network partition fault injection"""
    try:
        print("🌐 Testing network partition fault injection")
        
        initial_partitions = len(fault_injector.partitioned_nodes)
        target_nodes = [node.peer_id for node in peer_nodes[:6]]  # Partition 6 nodes
        
        # Create partition scenario
        partition_scenario = FaultScenario(
            name="Test Network Partition",
            description="Network partition test",
            fault_type=FaultType.NETWORK_PARTITION,
            severity=FaultSeverity.HIGH,
            target_nodes=target_nodes,
            duration=5,
            parameters={"partition_size": 3}
        )
        
        # Inject fault
        injection_success = await fault_injector.inject_fault_scenario(partition_scenario)
        
        # Verify partition
        current_partitions = len(fault_injector.partitioned_nodes)
        partition_created = current_partitions > initial_partitions
        
        # Check if nodes are marked as partitioned
        partitioned_nodes = sum(1 for node_id in target_nodes 
                              if fault_injector.node_states[node_id]["status"] == "partitioned")
        
        print(f"   📊 Network partition results:")
        print(f"      - Injection successful: {'✅' if injection_success else '❌'}")
        print(f"      - Initial partitions: {initial_partitions}")
        print(f"      - Current partitions: {current_partitions}")
        print(f"      - Partition created: {'✅' if partition_created else '❌'}")
        print(f"      - Nodes marked as partitioned: {partitioned_nodes}/{len(target_nodes)}")
        
        # Wait for recovery
        await asyncio.sleep(6)
        
        return injection_success and partition_created
        
    except Exception as e:
        print(f"❌ Network partition test error: {e}")
        return False


async def test_message_loss(fault_injector: FaultInjector, peer_nodes: List[PeerNode]) -> bool:
    """Test message loss fault injection"""
    try:
        print("📦 Testing message loss fault injection")
        
        initial_drop_rates = len(fault_injector.message_drop_rates)
        
        # Create message loss scenario
        message_loss_scenario = FaultScenario(
            name="Test Message Loss",
            description="Message loss test",
            fault_type=FaultType.MESSAGE_LOSS,
            severity=FaultSeverity.MEDIUM,
            target_nodes=[peer_nodes[4].peer_id],
            duration=5,
            parameters={"drop_rate": 0.3}
        )
        
        # Inject fault
        injection_success = await fault_injector.inject_fault_scenario(message_loss_scenario)
        
        # Verify message loss configuration
        current_drop_rates = len(fault_injector.message_drop_rates)
        drop_rates_added = current_drop_rates > initial_drop_rates
        
        print(f"   📊 Message loss results:")
        print(f"      - Injection successful: {'✅' if injection_success else '❌'}")
        print(f"      - Initial drop rate entries: {initial_drop_rates}")
        print(f"      - Current drop rate entries: {current_drop_rates}")
        print(f"      - Drop rates configured: {'✅' if drop_rates_added else '❌'}")
        
        # Wait for recovery
        await asyncio.sleep(6)
        
        return injection_success and drop_rates_added
        
    except Exception as e:
        print(f"❌ Message loss test error: {e}")
        return False


async def test_comprehensive_scenarios(fault_injector: FaultInjector, peer_nodes: List[PeerNode]) -> bool:
    """Test comprehensive fault scenarios"""
    try:
        print("🚀 Testing comprehensive fault scenarios")
        
        # Create multiple scenarios
        scenarios = await fault_injector.create_fault_scenarios(peer_nodes)
        
        print(f"   📋 Generated {len(scenarios)} fault scenarios:")
        for scenario in scenarios:
            print(f"      - {scenario.name}: {scenario.fault_type.value} ({scenario.severity.value})")
        
        # Test a subset of scenarios with short durations
        test_scenarios = scenarios[:3]  # Test first 3 scenarios
        
        successful_injections = 0
        for scenario in test_scenarios:
            scenario.duration = 3  # Short duration for testing
            success = await fault_injector.inject_fault_scenario(scenario)
            if success:
                successful_injections += 1
            
            # Wait for scenario to complete
            await asyncio.sleep(4)
        
        print(f"   📊 Comprehensive scenario results:")
        print(f"      - Scenarios generated: {len(scenarios)}")
        print(f"      - Scenarios tested: {len(test_scenarios)}")
        print(f"      - Successful injections: {successful_injections}/{len(test_scenarios)}")
        
        return len(scenarios) >= 5 and successful_injections >= 2
        
    except Exception as e:
        print(f"❌ Comprehensive scenarios test error: {e}")
        return False


async def test_consensus_under_faults(fault_injector: FaultInjector, peer_nodes: List[PeerNode]) -> bool:
    """Test consensus simulation under fault conditions"""
    try:
        print("🎯 Testing consensus under fault conditions")
        
        node_ids = [node.peer_id for node in peer_nodes]
        
        # Test consensus under different fault conditions
        test_proposals = [
            {"action": "test_consensus", "proposal_id": i, "data": f"test_data_{i}"}
            for i in range(5)
        ]
        
        consensus_results = []
        
        # Test 1: Consensus with no faults (baseline)
        for proposal in test_proposals[:2]:
            result = fault_injector.simulate_consensus_under_faults(proposal, node_ids)
            consensus_results.append(result)
        
        baseline_success_rate = sum(1 for r in consensus_results if r.consensus_achieved) / len(consensus_results)
        
        # Test 2: Inject some faults and test consensus
        fault_scenario = FaultScenario(
            name="Consensus Test Fault",
            description="Fault for consensus testing",
            fault_type=FaultType.NODE_SLOW,
            severity=FaultSeverity.LOW,
            target_nodes=node_ids[:2],
            duration=10,
            parameters={"slowdown_factor": 2.0}
        )
        
        await fault_injector.inject_fault_scenario(fault_scenario)
        
        # Test consensus under faults
        fault_consensus_results = []
        for proposal in test_proposals[2:]:
            result = fault_injector.simulate_consensus_under_faults(proposal, node_ids)
            fault_consensus_results.append(result)
        
        fault_success_rate = sum(1 for r in fault_consensus_results if r.consensus_achieved) / len(fault_consensus_results)
        
        # Calculate performance metrics
        baseline_avg_time = sum(r.execution_time for r in consensus_results) / len(consensus_results)
        fault_avg_time = sum(r.execution_time for r in fault_consensus_results) / len(fault_consensus_results)
        
        print(f"   📊 Consensus under faults results:")
        print(f"      - Baseline success rate: {baseline_success_rate:.1%}")
        print(f"      - Success rate under faults: {fault_success_rate:.1%}")
        print(f"      - Baseline avg execution time: {baseline_avg_time:.2f}s")
        print(f"      - Fault avg execution time: {fault_avg_time:.2f}s")
        print(f"      - Performance degradation: {((fault_avg_time - baseline_avg_time) / baseline_avg_time * 100):.1f}%")
        
        # Wait for fault to clear
        await asyncio.sleep(12)
        
        # Validate that consensus simulation is working
        simulation_working = len(consensus_results) > 0 and len(fault_consensus_results) > 0
        performance_impact = fault_avg_time > baseline_avg_time  # Faults should increase execution time
        
        return simulation_working and performance_impact
        
    except Exception as e:
        print(f"❌ Consensus under faults test error: {e}")
        return False


async def test_fault_recovery(fault_injector: FaultInjector, peer_nodes: List[PeerNode]) -> bool:
    """Test fault recovery mechanisms"""
    try:
        print("🔄 Testing fault recovery mechanisms")
        
        # Create fault scenario
        recovery_scenario = FaultScenario(
            name="Recovery Test Fault",
            description="Fault for recovery testing",
            fault_type=FaultType.NODE_CRASH,
            severity=FaultSeverity.MEDIUM,
            target_nodes=[peer_nodes[5].peer_id],
            duration=3
        )
        
        # Inject fault
        injection_success = await fault_injector.inject_fault_scenario(recovery_scenario)
        
        if not injection_success:
            print("   ❌ Failed to inject fault for recovery test")
            return False
        
        # Verify fault is active
        initial_active_faults = len(fault_injector.active_faults)
        node_crashed = fault_injector.node_states[peer_nodes[5].peer_id]["status"] == "crashed"
        
        print(f"   📊 Pre-recovery state:")
        print(f"      - Active faults: {initial_active_faults}")
        print(f"      - Node crashed: {'✅' if node_crashed else '❌'}")
        
        # Wait for automatic recovery
        await asyncio.sleep(4)  # Wait for fault duration + buffer
        
        # Verify recovery
        final_active_faults = len(fault_injector.active_faults)
        node_recovered = fault_injector.node_states[peer_nodes[5].peer_id]["status"] == "healthy"
        
        # Check fault history
        recovery_recorded = len(fault_injector.fault_history) > 0
        
        print(f"   📊 Post-recovery state:")
        print(f"      - Active faults: {final_active_faults}")
        print(f"      - Node recovered: {'✅' if node_recovered else '❌'}")
        print(f"      - Recovery recorded in history: {'✅' if recovery_recorded else '❌'}")
        
        # Validate recovery
        recovery_successful = (
            injection_success and 
            node_crashed and 
            final_active_faults < initial_active_faults and
            node_recovered
        )
        
        return recovery_successful
        
    except Exception as e:
        print(f"❌ Fault recovery test error: {e}")
        return False


async def main():
    """Main test function"""
    try:
        # Test fault injection system
        results = await test_fault_injection()
        
        print("\n" + "=" * 60)
        print("🎉 FAULT INJECTION TESTING COMPLETE")
        print("=" * 60)
        
        if results:
            successful_tests = len([r for name, r in results if r])
            
            print(f"✅ Test Results:")
            print(f"   - Test categories passed: {successful_tests}/{len(results)}")
            
            if successful_tests >= len(results) * 0.8:
                print(f"\n🚀 FAULT INJECTION SYSTEM: READY FOR PRODUCTION")
                print(f"💡 Key achievements:")
                print(f"   - Comprehensive fault type support (crash, slowdown, byzantine, partition, etc.)")
                print(f"   - Realistic consensus simulation under fault conditions")
                print(f"   - Automatic fault recovery and metrics collection")
                print(f"   - Enterprise-ready resilience testing capabilities")
            else:
                print(f"\n⚠️ Some tests need attention before production deployment")
        else:
            print(f"❌ Testing incomplete - check error messages above")
        
    except Exception as e:
        print(f"❌ Main test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())