
import pytest
import asyncio
from decimal import Decimal
from prsm.compute.network.distributed_rlt_network import DistributedRLTNetwork, NetworkMessage, MessageType
from prsm.economy.governance.resilience import ResilienceManager

@pytest.mark.asyncio
async def test_dynamic_pricing_fairness():
    """Verify that niche tasks (few nodes) get higher rewards"""
    net = DistributedRLTNetwork(node_id="master")
    
    # Simulate few nodes
    reward_niche = net.resilience_manager.calculate_fairness_reward(task_complexity=100, node_count=2)
    
    # Simulate many nodes
    reward_popular = net.resilience_manager.calculate_fairness_reward(task_complexity=100, node_count=20)
    
    assert reward_niche > reward_popular
    assert reward_niche == Decimal("20.0") # 100 * 0.1 * 2.0

@pytest.mark.asyncio
async def test_vrf_validator_rotation():
    """Verify that assigned validators are deterministic but 'shuffled' via VRF"""
    res = ResilienceManager()
    
    nodes = ["node1", "node2", "node3", "node4", "node5"]
    for n in nodes:
        res.register_node(n, Decimal("500.0"))
        
    task_a = "task_alpha"
    task_b = "task_beta"
    
    validators_a1 = res.get_shuffled_validators(task_a, nodes, count=2)
    validators_a2 = res.get_shuffled_validators(task_a, nodes, count=2)
    validators_b = res.get_shuffled_validators(task_b, nodes, count=2)
    
    # Same task -> Same validators (deterministic)
    assert validators_a1 == validators_a2
    
    # Different task -> Different validators (rotated)
    assert validators_a1 != validators_b

@pytest.mark.asyncio
async def test_slashing_mechanics():
    """Verify that slashing correctly nullifies a node's stake and reputation"""
    res = ResilienceManager()
    node_id = "malicious_node"
    
    res.register_node(node_id, Decimal("1000.0"))
    assert res.stakes[node_id].staked_amount == Decimal("1000.0")
    
    res.slash_node(node_id, "Fraud detected")
    
    assert res.stakes[node_id].staked_amount == Decimal("0.0")
    assert res.stakes[node_id].is_slashed is True
    assert res.stakes[node_id].reputation_score == 0.0
