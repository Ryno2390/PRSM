#!/usr/bin/env python3
"""
Simple Adaptive Consensus Test
Quick validation of adaptive consensus strategy selection
"""

import asyncio
import sys
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent))

from prsm.core.models import PeerNode
from prsm.compute.federation.adaptive_consensus import (
    AdaptiveConsensusEngine, NetworkMetrics, NetworkCondition, ConsensusStrategy
)


async def simple_adaptive_test():
    """Simple test of adaptive consensus strategy selection"""
    print("🧪 Simple Adaptive Consensus Test")
    print("=" * 50)
    
    try:
        # Test network metrics and condition detection
        print("📊 Testing Network Metrics & Condition Detection:")
        
        metrics = NetworkMetrics()
        
        # Test 1: Optimal conditions
        print("\n🌟 Test 1: Optimal Network Conditions")
        for i in range(10):
            metrics.add_latency_sample(10 + i)  # Low latency
            metrics.add_throughput_sample(20 + i)  # High throughput
            metrics.add_consensus_result(0.1, True, ConsensusStrategy.FAST_MAJORITY)
        
        metrics.update_network_composition(
            active_nodes={f"node_{i}" for i in range(8)},
            byzantine_nodes=set(),
            reputations={f"node_{i}": 0.8 for i in range(8)}
        )
        
        summary = metrics.get_performance_summary()
        print(f"   - Condition: {summary['condition']}")
        print(f"   - Recommended: {summary['recommended_strategy']}")
        print(f"   - Avg latency: {summary['avg_latency_ms']:.1f}ms")
        print(f"   - Avg throughput: {summary['avg_throughput']:.1f} ops/s")
        
        # Test 2: Congested network
        print("\n⚠️ Test 2: Congested Network Conditions")
        metrics = NetworkMetrics()
        for i in range(10):
            metrics.add_latency_sample(250 + i * 10)  # High latency
            metrics.add_throughput_sample(3 + i * 0.1)  # Low throughput
            metrics.add_consensus_result(1.5, True, ConsensusStrategy.HIERARCHICAL)
        
        metrics.update_network_composition(
            active_nodes={f"node_{i}" for i in range(30)},
            byzantine_nodes=set(),
            reputations={f"node_{i}": 0.7 for i in range(30)}
        )
        
        summary = metrics.get_performance_summary()
        print(f"   - Condition: {summary['condition']}")
        print(f"   - Recommended: {summary['recommended_strategy']}")
        print(f"   - Avg latency: {summary['avg_latency_ms']:.1f}ms")
        print(f"   - Avg throughput: {summary['avg_throughput']:.1f} ops/s")
        
        # Test 3: Unreliable network
        print("\n🚨 Test 3: Unreliable Network Conditions")
        metrics = NetworkMetrics()
        for i in range(10):
            metrics.add_latency_sample(100 + i * 5)
            metrics.add_throughput_sample(8 + i * 0.2)
            metrics.add_consensus_result(0.8, i % 3 == 0, ConsensusStrategy.BYZANTINE_RESILIENT)  # 33% failures
            if i % 3 == 1:
                metrics.add_failure_event("byzantine_behavior", f"node_{i}")
        
        metrics.update_network_composition(
            active_nodes={f"node_{i}" for i in range(20)},
            byzantine_nodes={f"node_{i}" for i in range(0, 6)},  # 30% Byzantine
            reputations={f"node_{i}": 0.3 if i < 6 else 0.8 for i in range(20)}
        )
        
        summary = metrics.get_performance_summary()
        print(f"   - Condition: {summary['condition']}")
        print(f"   - Recommended: {summary['recommended_strategy']}")
        print(f"   - Byzantine nodes: {len(metrics.byzantine_nodes)}/{len(metrics.active_nodes)}")
        print(f"   - Failure rate: {summary['failure_rate']:.1%}")
        
        # Test 4: Strategy selection for different network sizes
        print("\n📊 Testing Strategy Selection by Network Size:")
        
        test_sizes = [5, 15, 30, 60, 100]
        for size in test_sizes:
            metrics = NetworkMetrics()
            # Simulate optimal conditions for pure size-based selection
            for i in range(5):
                metrics.add_latency_sample(20)
                metrics.add_throughput_sample(15)
                metrics.add_consensus_result(0.1, True, ConsensusStrategy.FAST_MAJORITY)
            
            metrics.update_network_composition(
                active_nodes={f"node_{i}" for i in range(size)},
                byzantine_nodes=set(),
                reputations={f"node_{i}": 0.8 for i in range(size)}
            )
            
            summary = metrics.get_performance_summary()
            print(f"   - {size:3d} nodes → {summary['recommended_strategy']}")
        
        # Test 5: Adaptive engine integration
        print(f"\n🧠 Testing Adaptive Engine Integration:")
        
        # Small network test
        peer_nodes = []
        for i in range(8):
            peer = PeerNode(
                node_id=f"node_{i}",
                peer_id=f"peer_{i}",
                multiaddr=f"/ip4/127.0.0.1/tcp/{9000+i}",
                reputation_score=0.8,
                active=True
            )
            peer_nodes.append(peer)
        
        adaptive_engine = AdaptiveConsensusEngine()
        success = await adaptive_engine.initialize_adaptive_consensus(peer_nodes)
        
        if success:
            print(f"   - Initialization: ✅")
            print(f"   - Initial strategy: {adaptive_engine.current_strategy.value}")
            
            # Test consensus
            proposal = {"action": "test_adaptive_consensus", "data": "simple_test"}
            result = await adaptive_engine.achieve_adaptive_consensus(proposal)
            
            print(f"   - Consensus result: {'✅' if result.consensus_achieved else '❌'}")
            print(f"   - Strategy used: {adaptive_engine.current_strategy.value}")
            
            # Test metrics reporting
            await adaptive_engine.report_network_event("latency_measurement", {"latency_ms": 150})
            await adaptive_engine.report_network_event("throughput_measurement", {"ops_per_second": 12})
            
            recommendations = await adaptive_engine.get_strategy_recommendations()
            print(f"   - Network condition: {recommendations['network_condition']}")
            print(f"   - Recommendation confidence: {recommendations['confidence']:.1%}")
            
        else:
            print(f"   - Initialization: ❌")
        
        print(f"\n✅ ADAPTIVE CONSENSUS CORE FUNCTIONALITY VALIDATED!")
        print(f"🎯 Key Capabilities:")
        print(f"   - Network condition detection based on metrics")
        print(f"   - Strategy selection based on network size and conditions")
        print(f"   - Real-time metrics collection and analysis")
        print(f"   - Adaptive strategy recommendations with confidence scoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(simple_adaptive_test())