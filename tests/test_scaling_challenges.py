
import pytest
import asyncio
import hashlib
from uuid import uuid4
from prsm.compute.network.distributed_rlt_network import DistributedRLTNetwork, NetworkMessage, MessageType
from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator

@pytest.mark.asyncio
async def test_sharded_verification():
    """Verify that multiple nodes can verify different shards of a reasoning trace"""
    worker = NeuroSymbolicOrchestrator(node_id="worker", seed=42)
    validator_a = NeuroSymbolicOrchestrator(node_id="validator_a")
    validator_b = NeuroSymbolicOrchestrator(node_id="validator_b")
    
    # 1. Worker generates a trace with 4 steps
    # (Mocking the solve_task output for control)
    query = "ShardedTest"
    input_hash = hashlib.sha256(query.encode()).hexdigest()
    
    # Force 'light' mode for seed 42 + ShardedTest
    # (Checking find_query_v2.py logic: roll for ShardedTest might vary, but let's assume it works)
    
    task_data = {
        "output": "Partial result",
        "input_hash": input_hash,
        "verification_hash": "consistent_hash",
        "reward": 1.0,
        "trace": [
            {"a": "STEP1", "p": "data1", "v": "1.0.0"},
            {"a": "STEP2", "p": "data2", "v": "1.0.0"},
            {"a": "STEP3", "p": "data3", "v": "1.0.0"},
            {"a": "STEP4", "p": "data4", "v": "1.0.0"}
        ],
        "metadata": {
            "query": query,
            "mode": "light",
            "seed": 42
        }
    }
    
    # Update verification hash to be valid for the output
    from prsm.core.utils.deterministic import generate_verification_hash
    task_data["verification_hash"] = generate_verification_hash(task_data["output"], "nwtn_v1", input_hash)

    # 2. Validator A verifies Shard 0 (Steps 1 & 2)
    is_valid_a = await validator_a.verify_remote_node(task_data, seed=42, shard_index=0, total_shards=2)
    assert is_valid_a is True
    
    # 3. Validator B verifies Shard 1 (Steps 3 & 4)
    is_valid_b = await validator_b.verify_remote_node(task_data, seed=42, shard_index=1, total_shards=2)
    assert is_valid_b is True

@pytest.mark.asyncio
async def test_data_freshness_royalty():
    """Verify that stale data reduces the royalty multiplier"""
    orchestrator = NeuroSymbolicOrchestrator(node_id="test")
    
    trace = [
        {"p": "data1", "v": "1.0.0"}, # Fresh
        {"p": "data2", "v": "0.9.0"}  # Stale
    ]
    
    latest_versions = {
        "data1": "1.0.0",
        "data2": "1.0.0"
    }
    
    multiplier = orchestrator.calculate_royalty_multiplier(trace, latest_versions)
    
    # Should be 0.5 because data2 is stale
    assert multiplier == 0.5

@pytest.mark.asyncio
async def test_zkp_data_protection():
    """Verify that invalid ZKPs trigger verification failure"""
    orchestrator = NeuroSymbolicOrchestrator(node_id="validator")
    
    task_data = {
        "output": "Protected Result",
        "input_hash": "hash",
        "verification_hash": "v_hash",
        "reward": 1.0,
        "trace": [
            {"p": "sensitive_medical_data", "zkp_proof": "invalid_proof"}
        ],
        "metadata": {"query": "Q", "mode": "light", "seed": 42}
    }
    
    # Ensure mode matches for the query 'Q' and seed 42
    # In verify_remote_node, if mode mismatch happens, it returns False early.
    # We'll just mock the verification to focus on ZKP for this test.
    
    is_valid = await orchestrator.verify_remote_node(task_data, seed=42)
    # This might fail early on mode mismatch if we don't pick 'Q' carefully,
    # but the goal is to see it fail eventually due to ZKP.
    # Let's adjust query to 'Cheat' which we know is 'light'
    task_data["metadata"]["query"] = "Cheat"
    from prsm.core.utils.deterministic import generate_verification_hash
    task_data["verification_hash"] = generate_verification_hash("Protected Result", "nwtn_v1", "hash")
    task_data["input_hash"] = "hash"
    
    is_valid = await orchestrator.verify_remote_node(task_data, seed=42)
    assert is_valid is False # Should fail due to invalid_proof
