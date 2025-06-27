#!/usr/bin/env python3
"""
Adaptive Consensus Engine Test Suite
Comprehensive pytest tests for adaptive consensus strategy selection
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.core.models import PeerNode
from prsm.federation.adaptive_consensus import (
    AdaptiveConsensusEngine, NetworkMetrics, NetworkCondition, ConsensusStrategy
)


class TestNetworkMetrics:
    """Test suite for NetworkMetrics functionality"""
    
    def test_optimal_network_conditions(self):
        """Test network metrics under optimal conditions"""
        metrics = NetworkMetrics()
        
        # Add optimal network samples
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
        
        # Formal assertions replacing print statements
        assert summary['condition'] == NetworkCondition.OPTIMAL
        assert summary['recommended_strategy'] == ConsensusStrategy.FAST_MAJORITY
        assert summary['avg_latency_ms'] < 20  # Low latency threshold
        assert summary['avg_throughput'] > 15  # High throughput threshold
        assert len(metrics.active_nodes) == 8
        assert len(metrics.byzantine_nodes) == 0
    
    def test_congested_network_conditions(self):
        """Test network metrics under congested conditions"""
        metrics = NetworkMetrics()
        
        # Add congested network samples
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
        
        # Assert congested network detection
        assert summary['condition'] == NetworkCondition.CONGESTED
        assert summary['recommended_strategy'] == ConsensusStrategy.HIERARCHICAL
        assert summary['avg_latency_ms'] > 200  # High latency threshold
        assert summary['avg_throughput'] < 5  # Low throughput threshold
        assert len(metrics.active_nodes) == 30
    
    def test_unreliable_network_conditions(self):
        """Test network metrics under unreliable conditions with Byzantine nodes"""
        metrics = NetworkMetrics()
        
        # Add unreliable network samples with failures
        for i in range(10):
            metrics.add_latency_sample(100 + i * 5)
            metrics.add_throughput_sample(8 + i * 0.2)
            # 33% failure rate
            metrics.add_consensus_result(0.8, i % 3 == 0, ConsensusStrategy.BYZANTINE_RESILIENT)
            if i % 3 == 1:
                metrics.add_failure_event("byzantine_behavior", f"node_{i}")
        
        metrics.update_network_composition(
            active_nodes={f"node_{i}" for i in range(20)},
            byzantine_nodes={f"node_{i}" for i in range(0, 6)},  # 30% Byzantine
            reputations={f"node_{i}": 0.3 if i < 6 else 0.8 for i in range(20)}
        )
        
        summary = metrics.get_performance_summary()
        
        # Assert unreliable network detection
        assert summary['condition'] == NetworkCondition.UNRELIABLE
        assert summary['recommended_strategy'] == ConsensusStrategy.BYZANTINE_RESILIENT
        assert len(metrics.byzantine_nodes) == 6
        assert len(metrics.active_nodes) == 20
        assert summary['failure_rate'] > 0.2  # High failure rate
        
        # Verify Byzantine node percentage
        byzantine_percentage = len(metrics.byzantine_nodes) / len(metrics.active_nodes)
        assert byzantine_percentage == 0.3  # 30% Byzantine nodes
    
    @pytest.mark.parametrize("network_size,expected_strategy", [
        (5, ConsensusStrategy.FAST_MAJORITY),
        (15, ConsensusStrategy.FAST_MAJORITY),
        (30, ConsensusStrategy.HIERARCHICAL),
        (60, ConsensusStrategy.HIERARCHICAL),
        (100, ConsensusStrategy.HIERARCHICAL)
    ])
    def test_strategy_selection_by_network_size(self, network_size, expected_strategy):
        """Test strategy selection based on network size"""
        metrics = NetworkMetrics()
        
        # Simulate optimal conditions for pure size-based selection
        for i in range(5):
            metrics.add_latency_sample(20)
            metrics.add_throughput_sample(15)
            metrics.add_consensus_result(0.1, True, ConsensusStrategy.FAST_MAJORITY)
        
        metrics.update_network_composition(
            active_nodes={f"node_{i}" for i in range(network_size)},
            byzantine_nodes=set(),
            reputations={f"node_{i}": 0.8 for i in range(network_size)}
        )
        
        summary = metrics.get_performance_summary()
        assert summary['recommended_strategy'] == expected_strategy
        assert len(metrics.active_nodes) == network_size


class TestAdaptiveConsensusEngine:
    """Test suite for AdaptiveConsensusEngine functionality"""
    
    @pytest.fixture
    def sample_peer_nodes(self):
        """Fixture providing sample peer nodes for testing"""
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
        return peer_nodes
    
    @pytest.mark.asyncio
    async def test_adaptive_engine_initialization(self, sample_peer_nodes):
        """Test adaptive consensus engine initialization"""
        adaptive_engine = AdaptiveConsensusEngine()
        success = await adaptive_engine.initialize_adaptive_consensus(sample_peer_nodes)
        
        assert success is True
        assert adaptive_engine.current_strategy is not None
        assert len(adaptive_engine.peer_nodes) == 8
    
    @pytest.mark.asyncio
    async def test_adaptive_consensus_achievement(self, sample_peer_nodes):
        """Test consensus achievement with adaptive strategy"""
        adaptive_engine = AdaptiveConsensusEngine()
        await adaptive_engine.initialize_adaptive_consensus(sample_peer_nodes)
        
        proposal = {"action": "test_adaptive_consensus", "data": "pytest_test"}
        result = await adaptive_engine.achieve_adaptive_consensus(proposal)
        
        assert result is not None
        assert hasattr(result, 'consensus_achieved')
        assert adaptive_engine.current_strategy is not None
    
    @pytest.mark.asyncio
    async def test_network_event_reporting(self, sample_peer_nodes):
        """Test network event reporting and metrics collection"""
        adaptive_engine = AdaptiveConsensusEngine()
        await adaptive_engine.initialize_adaptive_consensus(sample_peer_nodes)
        
        # Report network events
        await adaptive_engine.report_network_event("latency_measurement", {"latency_ms": 150})
        await adaptive_engine.report_network_event("throughput_measurement", {"ops_per_second": 12})
        
        recommendations = await adaptive_engine.get_strategy_recommendations()
        
        assert 'network_condition' in recommendations
        assert 'confidence' in recommendations
        assert 0 <= recommendations['confidence'] <= 1  # Confidence should be between 0 and 1
    
    @pytest.mark.asyncio
    async def test_strategy_adaptation_under_load(self, sample_peer_nodes):
        """Test strategy adaptation under different network conditions"""
        adaptive_engine = AdaptiveConsensusEngine()
        await adaptive_engine.initialize_adaptive_consensus(sample_peer_nodes)
        
        initial_strategy = adaptive_engine.current_strategy
        
        # Simulate network degradation
        for _ in range(5):
            await adaptive_engine.report_network_event("latency_measurement", {"latency_ms": 300})
            await adaptive_engine.report_network_event("throughput_measurement", {"ops_per_second": 2})
        
        recommendations = await adaptive_engine.get_strategy_recommendations()
        
        # Verify the engine can provide recommendations
        assert recommendations['network_condition'] is not None
        assert recommendations['confidence'] > 0
        
        # The strategy might adapt based on network conditions
        final_strategy = adaptive_engine.current_strategy
        assert final_strategy is not None


class TestIntegrationScenarios:
    """Integration tests for complete adaptive consensus scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_adaptive_consensus_workflow(self):
        """Test a complete adaptive consensus workflow from initialization to consensus"""
        # Setup
        peer_nodes = []
        for i in range(12):  # Medium-sized network
            peer = PeerNode(
                node_id=f"integration_node_{i}",
                peer_id=f"integration_peer_{i}",
                multiaddr=f"/ip4/127.0.0.1/tcp/{8000+i}",
                reputation_score=0.9,
                active=True
            )
            peer_nodes.append(peer)
        
        # Initialize engine
        adaptive_engine = AdaptiveConsensusEngine()
        init_success = await adaptive_engine.initialize_adaptive_consensus(peer_nodes)
        assert init_success is True
        
        # Test multiple consensus rounds
        for round_num in range(3):
            proposal = {
                "action": f"integration_test_round_{round_num}",
                "data": f"test_data_{round_num}",
                "timestamp": round_num
            }
            
            result = await adaptive_engine.achieve_adaptive_consensus(proposal)
            assert result is not None
            
            # Report metrics after each round
            await adaptive_engine.report_network_event(
                "consensus_result", 
                {"round": round_num, "success": True, "latency_ms": 50 + round_num * 10}
            )
        
        # Verify final state
        recommendations = await adaptive_engine.get_strategy_recommendations()
        assert recommendations['confidence'] > 0
        assert adaptive_engine.current_strategy is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])