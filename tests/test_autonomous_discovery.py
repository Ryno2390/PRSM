
import pytest
import asyncio
from decimal import Decimal
from prsm.compute.nwtn.reasoning.autonomous_discovery import DiscoveryPipeline, MoonshotFund, BreakthroughLevel
from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator
from prsm.compute.nwtn.knowledge_graph import RecursiveKnowledgeGraph, DiscoveryUpdate, GlobalBrainSync
from prsm.knowledge_system import UnifiedKnowledgeSystem

@pytest.mark.asyncio
async def test_discovery_pipeline_impact():
    """Verify that the pipeline correctly identifies and assesses breakthroughs"""
    orchestrator = NeuroSymbolicOrchestrator(node_id="discovery_node")
    ks = UnifiedKnowledgeSystem() # Uninitialized but for mock it's okay
    pipeline = DiscoveryPipeline(orchestrator, ks)
    
    # Simulate a high-reward result (Level 5)
    high_reward_result = {"reward": 0.98}
    level = pipeline.assess_breakthrough_impact(high_reward_result)
    assert level == BreakthroughLevel.LEVEL_5
    
    # Simulate a low-reward result (Level 1)
    low_reward_result = {"reward": 0.2}
    level = pipeline.assess_breakthrough_impact(low_reward_result)
    assert level == BreakthroughLevel.LEVEL_1

@pytest.mark.asyncio
async def test_moonshot_payout():
    """Verify that Level 5 breakthroughs trigger massive treasury payouts"""
    fund = MoonshotFund(treasury_balance=Decimal("1000000.0"))
    
    payout_l5 = fund.calculate_impact_payout(BreakthroughLevel.LEVEL_5)
    assert payout_l5 == Decimal("100000.0") # 10%
    
    # Distribute among 5 contributors
    per_node = fund.distribute_payout(payout_l5, ["node1", "node2", "node3", "node4", "node5"])
    assert per_node == Decimal("20000.0")
    assert fund.treasury == Decimal("900000.0")

@pytest.mark.asyncio
async def test_recursive_knowledge_graph_sync():
    """Verify that agents learn from each other in real-time via the graph"""
    kg = RecursiveKnowledgeGraph()
    
    # Two nodes subscribing to climate data
    node_a_sync = GlobalBrainSync("node_a", kg)
    node_b_sync = GlobalBrainSync("node_b", kg)
    
    node_a_sync.start_sync(["climate"])
    node_b_sync.start_sync(["climate"])
    
    # Node C publishes a discovery
    discovery = DiscoveryUpdate(
        cid="cid_new_battery_chemistry",
        domain="climate",
        content="Graphene-enhanced zinc-air battery exceeds lithium-ion energy density.",
        impact_level=BreakthroughLevel.LEVEL_4
    )
    
    await kg.publish_discovery(discovery)
    
    # Both node_a and node_b should have received it
    assert len(node_a_sync.local_context_buffer) == 1
    assert len(node_b_sync.local_context_buffer) == 1
    assert node_a_sync.local_context_buffer[0].cid == "cid_new_battery_chemistry"
