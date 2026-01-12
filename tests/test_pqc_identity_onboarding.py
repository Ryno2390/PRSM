
import pytest
import asyncio
import base64
from decimal import Decimal
from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator
from prsm.core.identity.nhi import NHIManager
from prsm.core.institutional.gateway import InstitutionalGateway, InstitutionalCapacity, ParticipationMode

@pytest.mark.asyncio
async def test_quantum_resilient_provenance():
    """Verify that task results are signed with PQC and verified by validators"""
    worker = NeuroSymbolicOrchestrator(node_id="quantum_worker", seed=42)
    validator = NeuroSymbolicOrchestrator(node_id="quantum_validator")
    
    # Solve task - should include PQC signature
    result = await worker.solve_task("Room temperature superconductor", "PQC-Test")
    assert result["pq_signature"] is not None
    
    # Verify via validator
    is_valid = await validator.verify_remote_node(result, seed=42)
    assert is_valid is True

@pytest.mark.asyncio
async def test_nhi_meritocracy():
    """Verify that agent reputation reduces staking requirements"""
    nhi_manager = NHIManager()
    
    # Register an agent
    agent, keypair = nhi_manager.register_agent("research_lab_01", "orchestrator")
    
    base_stake = 1000.0
    initial_req = agent.calculate_staking_requirement(base_stake)
    assert initial_req == 1000.0
    
    # Agent makes a Level 5 breakthrough
    nhi_manager.update_reputation(agent.agent_id, 5)
    
    # Reputation should be 1.5 (1.0 + 5 * 0.1)
    # Multiplier: 1 / sqrt(1.5) approx 0.816
    reduced_req = agent.calculate_staking_requirement(base_stake)
    assert reduced_req < initial_req
    assert 810 < reduced_req < 820

@pytest.mark.asyncio
async def test_institutional_private_shard():
    """Verify that institutions can create private shards via the bridge"""
    gateway = InstitutionalGateway()
    
    # Onboard MIT
    capacity = InstitutionalCapacity(
        compute_tflops=500000, storage_petabytes=100, 
        bandwidth_gbps=1000, model_parameters=10**12,
        research_personnel=500, annual_ai_budget_usd=10**9
    )
    
    mit = await gateway.onboard_institution(
        "MIT AI Lab", capacity, [ParticipationMode.MODEL_CONTRIBUTOR]
    )
    
    # Create private shard
    shard = await gateway.create_private_shard(mit.participant_id, ["agent_01", "agent_02"])
    
    assert shard.institution_name == "MIT AI Lab"
    assert "agent_01" in shard.members
    assert gateway.private_shards[shard.shard_id] == shard
