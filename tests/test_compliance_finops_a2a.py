
import pytest
import asyncio
from decimal import Decimal
from prsm.core.compliance.regulatory_mapping import RegulatoryMappingAgent, Jurisdiction
from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator
from prsm.compute.network.a2a_protocol import A2AExchange
from prsm.compute.nwtn.engines.world_model_engine import get_world_model

@pytest.mark.asyncio
async def test_compliance_as_code():
    """Verify that new laws trigger automatic ethical guardrail updates"""
    agent = RegulatoryMappingAgent()
    world_model = get_world_model()
    
    # 1. Scan for California updates
    updates = await agent.scan_for_updates()
    assert len(updates) > 0
    assert updates[0].jurisdiction == Jurisdiction.US_CA
    
    # 2. Apply the update
    agent.apply_update(updates[0])
    assert "CA_DNA_PRIVACY_ACT" in world_model.ethical_kill_switches
    
    # 3. Test the new constraint (Restricted DNA sequencing without enough experts)
    proposal = "Perform autonomous DNA sequencing on leaked medical data."
    context = {"expert_approval_count": 1} # Only 1 expert, should fail
    
    result = await world_model.verify_constraints(proposal, context)
    assert result.success is False
    assert "CA_DNA_PRIVACY_ACT" in result.rejection_reason

@pytest.mark.asyncio
async def test_finops_arbitration():
    """Verify that the orchestrator uses a cost-efficient research strategy"""
    worker = NeuroSymbolicOrchestrator(node_id="finops_worker")
    
    result = await worker.solve_task("Global pandemic response", "Emergency")
    
    # Check if strategy was planned in the trace
    strategy_step = next((s for s in result["trace"] if s["a"] == "STRATEGY_PLANNED"), None)
    assert strategy_step is not None
    assert "estimated_cost" in strategy_step["m"]

@pytest.mark.asyncio
async def test_a2a_cross_protocol_research():
    """Verify that PRSM agents can buy insights from other networks"""
    exchange = A2AExchange(agent_id="research_agent_01")
    
    insight = await exchange.buy_external_insight(
        target_protocol="OriginTrail",
        query="Verify protein-ligand binding for viral strain X",
        budget=10.0
    )
    
    assert insight is not None
    assert "Found specific protein correlation" in insight["data"]
    assert insight["cost"] <= 10.0
