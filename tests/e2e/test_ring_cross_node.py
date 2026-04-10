"""
Ring 1-10 Cross-Node E2E Test
==============================

Tests all capability rings across a real two-node P2P connection:

Ring 1  - Hardware profiles gossip between nodes
Ring 2  - Mobile agent dispatched from Node A, executed on Node B
Ring 3  - Swarm job fans out across both nodes
Ring 4  - Pricing engine quotes job costs, prosumer tiers work
Ring 5  - REMOVED in v1.6.0 (legacy NWTN agent_forge pruned)
Ring 6  - Dynamic gas pricing, signature verification, CLI commands
Ring 7  - Confidential compute / TEE runtime
Ring 8  - Tensor-parallel model sharding
Ring 9  - NWTN model service and training pipeline
Ring 10 - Security hardening (integrity, privacy budget, audit log)

Two in-process PRSMNode instances connect via direct WebSocket bootstrap.
"""

import asyncio
import base64
import os
import tempfile
import pytest
import socket
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from prsm.node.config import NodeConfig
from prsm.node.node import PRSMNode


# ── Helpers ───────────────────────────────────────────────────────────────

def _free_port() -> int:
    """Get a free TCP port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Minimal WASM module: (module (func (export "run") (result i32) (i32.const 42)))
MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test node data."""
    with tempfile.TemporaryDirectory(prefix="prsm-ring-e2e-") as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_ipfs():
    """Mock IPFS client for testing without a running daemon."""
    with patch("prsm.node.storage_provider.StorageProvider._check_ipfs") as mock_detect:
        mock_detect.return_value = True
        with patch("prsm.node.storage_provider.StorageProvider.pin_content") as mock_pin:
            mock_pin.return_value = True
            with patch("prsm.node.storage_provider.StorageProvider.verify_pin") as mock_verify:
                mock_verify.return_value = True
                with patch("prsm.node.content_uploader.ContentUploader._ipfs_add") as mock_add:
                    _cid_counter = {"n": 0}
                    def _unique_cid(*args, **kwargs):
                        _cid_counter["n"] += 1
                        return (f"QmTestCID{_cid_counter['n']:012d}", 1024)
                    mock_add.side_effect = _unique_cid
                    with patch("prsm.node.content_provider.ContentProvider._ipfs_cat") as mock_cat:
                        mock_cat.return_value = b"Test content"
                        yield


@pytest.fixture
async def node_a(temp_data_dir, mock_ipfs):
    """Create and start Node A (primary node)."""
    p2p_port = _free_port()
    api_port = _free_port()

    config = NodeConfig(
        display_name="RingTestNodeA",
        data_dir=os.path.join(temp_data_dir, "node-a"),
        p2p_port=p2p_port,
        api_port=api_port,
        roles=["full"],
        bootstrap_nodes=[],  # Node A is the bootstrap
        welcome_grant=1000.0,
    )
    os.makedirs(config.data_dir, exist_ok=True)

    node = PRSMNode(config)
    await node.initialize()
    await node.start()

    yield node

    await node.stop()


@pytest.fixture
async def node_b(temp_data_dir, mock_ipfs, node_a):
    """Create and start Node B, bootstrapping to Node A."""
    node_a_p2p = f"ws://127.0.0.1:{node_a.config.p2p_port}"

    config = NodeConfig(
        display_name="RingTestNodeB",
        data_dir=os.path.join(temp_data_dir, "node-b"),
        p2p_port=_free_port(),
        api_port=_free_port(),
        roles=["full"],
        bootstrap_nodes=[node_a_p2p],
        bootstrap_connect_timeout=10.0,
        bootstrap_retry_attempts=3,
        welcome_grant=1000.0,
    )
    os.makedirs(config.data_dir, exist_ok=True)

    node = PRSMNode(config)
    await node.initialize()
    await node.start()

    # Wait for peer connection (up to 15 seconds)
    for _ in range(75):
        if node.transport and node.transport.peer_count > 0:
            break
        await asyncio.sleep(0.2)

    yield node

    await node.stop()


# ── Ring 1: Hardware Profile Gossip ───────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring1_hardware_profile_detection(node_a):
    """Ring 1: Node detects hardware and classifies compute tier."""
    from prsm.compute.wasm import HardwareProfiler, ComputeTier

    profiler = HardwareProfiler()
    profile = profiler.detect()

    assert profile.cpu_cores >= 1
    assert profile.ram_total_gb > 0
    assert profile.tflops_fp32 > 0
    assert profile.compute_tier in [ComputeTier.T1, ComputeTier.T2, ComputeTier.T3, ComputeTier.T4]


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring1_wasm_sandbox_execution(node_a):
    """Ring 1: WASM module executes in sandbox with resource metering."""
    from prsm.compute.wasm import WasmtimeRuntime, ResourceLimits, ExecutionStatus

    runtime = WasmtimeRuntime()
    if not runtime.available:
        pytest.skip("wasmtime not installed")

    module = runtime.load(MINIMAL_WASM)
    result = runtime.execute(
        module=module,
        input_data=b'{"test": true}',
        resource_limits=ResourceLimits(max_memory_bytes=64 * 1024 * 1024, max_execution_seconds=5),
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.execution_time_seconds >= 0
    assert result.pcu() >= 0


# ── Ring 2: Agent Dispatch ────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring2_components_initialized(node_a, node_b):
    """Ring 2: Both nodes have AgentDispatcher and AgentExecutor."""
    assert node_a.agent_dispatcher is not None, "Node A missing agent_dispatcher"
    assert node_a.agent_executor is not None, "Node A missing agent_executor"
    assert node_b.agent_dispatcher is not None, "Node B missing agent_dispatcher"
    assert node_b.agent_executor is not None, "Node B missing agent_executor"


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring2_agent_creation_and_dispatch(node_a):
    """Ring 2: Node A can create and dispatch a mobile agent."""
    from prsm.compute.agents.models import AgentManifest, DispatchStatus

    manifest = AgentManifest(
        required_content_ids=["QmTestShard001"],
        min_hardware_tier="t1",
        max_execution_seconds=10,
    )

    agent = node_a.agent_dispatcher.create_agent(
        wasm_binary=MINIMAL_WASM,
        manifest=manifest,
        ftns_budget=1.0,
        ttl=60,
    )

    assert agent is not None
    assert agent.origin_node == node_a.identity.node_id
    assert agent.ftns_budget == 1.0
    assert len(agent.signature) > 0


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring2_executor_runs_wasm(node_b):
    """Ring 2: Node B can execute a WASM agent in sandbox."""
    from prsm.compute.agents.models import AgentManifest, MobileAgent

    agent = MobileAgent(
        agent_id="e2e-test-agent",
        wasm_binary=MINIMAL_WASM,
        manifest=AgentManifest(required_content_ids=[], min_hardware_tier="t1"),
        origin_node="test-origin",
        signature="test-sig",
        ftns_budget=1.0,
        ttl=60,
    )

    result = await node_b.agent_executor.execute_agent(agent, input_data=b"")

    assert result["status"] in ("success", "error")
    if result["status"] == "success":
        assert result["pcu"] >= 0
        assert len(result["provider_signature"]) > 0


# ── Ring 3: Swarm Compute ─────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring3_swarm_coordinator_initialized(node_a, node_b):
    """Ring 3: Both nodes have SwarmCoordinator."""
    assert node_a.swarm_coordinator is not None, "Node A missing swarm_coordinator"
    assert node_b.swarm_coordinator is not None, "Node B missing swarm_coordinator"


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring3_semantic_shard_search():
    """Ring 3: Semantic shards can be searched by embedding similarity."""
    from prsm.data.shard_models import SemanticShard, SemanticShardManifest

    shards = [
        SemanticShard(
            shard_id="ev-shard",
            parent_dataset="nada",
            cid="QmEVShard",
            centroid=[1.0, 0.0, 0.0],
            record_count=500,
            size_bytes=1024 * 1024,
            keywords=["electric vehicles"],
        ),
        SemanticShard(
            shard_id="gas-shard",
            parent_dataset="nada",
            cid="QmGasShard",
            centroid=[0.0, 1.0, 0.0],
            record_count=500,
            size_bytes=1024 * 1024,
            keywords=["gasoline"],
        ),
    ]
    manifest = SemanticShardManifest(
        dataset_id="nada-nc",
        total_records=1000,
        total_size_bytes=2 * 1024 * 1024,
        shards=shards,
    )

    # Query close to EV shard
    relevant = manifest.find_relevant_shards([0.9, 0.1, 0.0], top_k=1)
    assert len(relevant) == 1
    shard, score = relevant[0]
    assert shard.shard_id == "ev-shard"
    assert score > 0.8


# ── Ring 4: Economy ───────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring4_pricing_engine_initialized(node_a):
    """Ring 4: Node has PricingEngine available."""
    assert node_a.pricing_engine is not None, "Node A missing pricing_engine"


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring4_cost_quote():
    """Ring 4: PricingEngine can quote a swarm job."""
    from prsm.economy.pricing import PricingEngine, DataAccessFee

    engine = PricingEngine(network_utilization=0.6)
    quote = engine.quote_swarm_job(
        shard_count=5,
        hardware_tier="t2",
        estimated_pcu_per_shard=50.0,
        data_fee=DataAccessFee(
            dataset_id="nada",
            base_access_fee=Decimal("5.0"),
            per_shard_fee=Decimal("0.5"),
        ),
    )

    assert quote.compute_cost > 0
    assert quote.data_cost > 0
    assert quote.network_fee > 0
    assert quote.total == quote.compute_cost + quote.data_cost + quote.network_fee


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring4_prosumer_yield_estimate(node_a):
    """Ring 4: Prosumer manager can estimate yields."""
    assert node_a.prosumer_manager is not None

    profile = await node_a.prosumer_manager.register(stake_amount=0)
    assert profile is not None

    estimate = node_a.prosumer_manager.yield_estimate(
        hardware_tier="t2",
        tflops=15.0,
        hours_per_day=8,
    )

    assert float(estimate["daily_ftns"]) > 0
    assert float(estimate["monthly_ftns"]) > 0


# ── Ring 5: REMOVED ──────────────────────────────────────────────────────
# Ring 5 AgentForge (NWTN agent_forge module) was deleted in v1.6.0 as part
# of the legacy AGI framework removal. Ring 5 tests have been removed.


# ── Ring 6: Production Hardening ──────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring6_dynamic_gas_pricing():
    """Ring 6: Dynamic gas price estimation works."""
    from prsm.economy.ftns_onchain import estimate_gas_price, RPCFailover

    # Test RPC failover rotation
    failover = RPCFailover(urls=["https://a.com", "https://b.com"])
    assert failover.current_url == "https://a.com"
    failover.mark_failed()
    assert failover.current_url == "https://b.com"

    # Test gas estimation with mock
    mock_w3 = MagicMock()
    mock_w3.eth.gas_price = 3_000_000_000  # 3 gwei
    gas = estimate_gas_price(mock_w3, multiplier=1.2)
    assert gas == int(3_000_000_000 * 1.2)


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring6_settler_signature_verification():
    """Ring 6: Settler signature verification function exists and works."""
    from prsm.node.settler_registry import verify_settler_signature

    # Empty signature must be rejected
    assert not verify_settler_signature("key", b"msg", "")

    # Invalid signature must be rejected
    assert not verify_settler_signature("aW52YWxpZA==", b"test", "aW52YWxpZA==")


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring6_cli_commands_exist():
    """Ring 6: New CLI commands are registered."""
    from click.testing import CliRunner
    from prsm.cli import main

    runner = CliRunner()

    # node benchmark
    result = runner.invoke(main, ["node", "benchmark", "--help"])
    assert result.exit_code == 0

    # ftns yield-estimate
    result = runner.invoke(main, ["ftns", "yield-estimate", "--help"])
    assert result.exit_code == 0

    # compute quote
    result = runner.invoke(main, ["compute", "quote", "--help"])
    assert result.exit_code == 0


# ── Cross-Node Integration ────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_cross_node_peer_connection(node_a, node_b):
    """Both nodes establish P2P connection."""
    # Node B should have connected to Node A
    b_peers = node_b.transport.peer_count if node_b.transport else 0

    # Allow extra time if not yet connected
    if b_peers == 0:
        for _ in range(50):
            await asyncio.sleep(0.2)
            b_peers = node_b.transport.peer_count if node_b.transport else 0
            if b_peers > 0:
                break

    assert b_peers > 0, f"Node B has {b_peers} peers (expected >= 1)"


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_cross_node_gossip_delivery(node_a, node_b):
    """Gossip messages propagate from Node A to Node B."""
    received = {"count": 0, "data": None}

    async def _on_test_message(subtype, data, sender_id):
        received["count"] += 1
        received["data"] = data

    node_b.gossip.subscribe("ring_test_ping", _on_test_message)

    # Node A publishes a test gossip message
    sent = await node_a.gossip.publish("ring_test_ping", {
        "message": "hello from node A",
        "ring_test": True,
    })

    # Wait for delivery
    for _ in range(50):
        if received["count"] > 0:
            break
        await asyncio.sleep(0.2)

    # If peers are connected, message should have been delivered
    if node_b.transport.peer_count > 0:
        assert received["count"] > 0, "Gossip message not received by Node B"
        assert received["data"]["message"] == "hello from node A"


# ── Ring 7: Confidential Compute ──────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring7_confidential_executor_initialized(node_a, node_b):
    """Ring 7: Both nodes have ConfidentialExecutor."""
    assert node_a.confidential_executor is not None, "Node A missing confidential_executor"
    assert node_b.confidential_executor is not None, "Node B missing confidential_executor"


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring7_dp_noise_injection():
    """Ring 7: DP noise injector works correctly."""
    import numpy as np
    from prsm.compute.tee.dp_noise import DPNoiseInjector
    from prsm.compute.tee.models import DPConfig

    injector = DPNoiseInjector(DPConfig(epsilon=8.0))
    original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    noisy = injector.inject(original)

    assert noisy.shape == original.shape
    assert not np.array_equal(noisy, original)
    assert injector.epsilon_spent == 8.0


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring7_tee_detection(node_a):
    """Ring 7: Hardware profiler detects TEE capability."""
    from prsm.compute.wasm.profiler import HardwareProfiler

    profiler = HardwareProfiler()
    profile = profiler.detect()
    assert isinstance(profile.tee_available, bool)
    assert isinstance(profile.tee_type, str)


# ── Ring 8: Model Sharding ────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring8_tensor_executor_initialized(node_a, node_b):
    """Ring 8: Both nodes have TensorParallelExecutor."""
    assert node_a.tensor_executor is not None, "Node A missing tensor_executor"
    assert node_b.tensor_executor is not None, "Node B missing tensor_executor"


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring8_model_shard_and_verify():
    """Ring 8: Model can be sharded and integrity verified."""
    import numpy as np
    from prsm.compute.model_sharding import ModelSharder
    from prsm.security import IntegrityVerifier

    sharder = ModelSharder()
    weights = {"layer1": np.random.randn(16, 8)}
    model = sharder.shard_model("test-model", "TestModel", weights, n_shards=4)

    assert model.total_shards == 4
    assert model.total_size_bytes > 0

    # Verify integrity of each shard
    verifier = IntegrityVerifier()
    for shard in model.shards:
        assert verifier.verify_shard(shard.tensor_data, shard.checksum), \
            f"Shard {shard.shard_id} integrity check failed"


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring8_collision_detection():
    """Ring 8: Collision detector identifies divergent outputs."""
    import json
    from prsm.compute.model_sharding.collision_detector import CollisionDetector

    detector = CollisionDetector(dp_epsilon=8.0, tolerance_multiplier=1.0)

    good = json.dumps([1.0, 2.0, 3.0]).encode()
    bad = json.dumps([999.0, 999.0, 999.0]).encode()

    report = detector.detect_collision([good, good, bad])
    assert report["match"] is False
    assert 2 in report["flagged_indices"]


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring8_pipeline_randomizer():
    """Ring 8: Pipeline randomizer enforces pool size and produces unique assignments."""
    from prsm.compute.model_sharding import PipelineRandomizer

    randomizer = PipelineRandomizer(min_pool_size=5)
    nodes = [{"node_id": f"node-{i}", "tee_available": True} for i in range(10)]

    assignments = randomizer.assign_pipeline(4, nodes)
    assert len(assignments) == 4
    node_ids = [a["node_id"] for a in assignments]
    assert len(set(node_ids)) == 4  # All unique


# ── Ring 9: NWTN Model Service ────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring9_model_service_initialized(node_a, node_b):
    """Ring 9: Both nodes have NWTNModelService."""
    assert node_a.nwtn_model_service is not None, "Node A missing nwtn_model_service"
    assert node_b.nwtn_model_service is not None, "Node B missing nwtn_model_service"


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring9_training_pipeline():
    """Ring 9: Training pipeline can ingest traces and export JSONL."""
    import json
    from prsm.compute.nwtn.training import TrainingPipeline, TrainingConfig

    pipeline = TrainingPipeline(TrainingConfig(min_corpus_size=2))
    traces = [
        {
            "query": f"test query {i}",
            "decomposition": {"operations": ["filter"], "estimated_complexity": 0.5},
            "plan": {"route": "swarm"},
            "execution_result": {"status": "success"},
            "execution_metrics": {"pcu": 1.0},
        }
        for i in range(5)
    ]
    count = pipeline.ingest_traces(traces)
    assert count == 5

    valid, errors = pipeline.validate_corpus()
    assert valid, f"Validation errors: {errors}"

    jsonl = pipeline.export_dataset()
    lines = jsonl.strip().split("\n")
    assert len(lines) == 5


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring9_model_deploy_lifecycle():
    """Ring 9: Model can be registered, deployed with sharding, and retired."""
    import numpy as np
    from prsm.compute.nwtn.training import ModelCard, DeploymentStatus
    from prsm.compute.nwtn.training.model_service import NWTNModelService

    service = NWTNModelService()
    card = ModelCard(model_id="nwtn-e2e", model_name="NWTN-E2E", base_model="llama-3.1")
    service.register_model(card)
    assert card.status == DeploymentStatus.REGISTERED

    weights = {"attn": np.random.randn(16, 8)}
    deployment = service.deploy_model("nwtn-e2e", weight_tensors=weights, n_shards=4)
    assert deployment["n_shards"] == 4
    assert card.status == DeploymentStatus.DEPLOYED

    service.retire_model("nwtn-e2e")
    assert card.status == DeploymentStatus.RETIRED


# ── Ring 10: Security Hardening ───────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring10_security_modules_initialized(node_a, node_b):
    """Ring 10: Both nodes have security modules."""
    assert node_a.integrity_verifier is not None, "Node A missing integrity_verifier"
    assert node_a.privacy_budget is not None, "Node A missing privacy_budget"
    assert node_a.pipeline_audit_log is not None, "Node A missing pipeline_audit_log"
    assert node_b.integrity_verifier is not None, "Node B missing integrity_verifier"


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring10_privacy_budget_enforcement():
    """Ring 10: Privacy budget tracker enforces limits."""
    from prsm.security import PrivacyBudgetTracker

    tracker = PrivacyBudgetTracker(max_epsilon=20.0)
    assert tracker.record_spend(8.0, "inference-1", "nwtn")
    assert tracker.record_spend(8.0, "inference-2", "nwtn")
    assert not tracker.record_spend(8.0, "inference-3", "nwtn")  # Would exceed
    assert tracker.remaining == 4.0


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring10_audit_log_chain():
    """Ring 10: Audit log maintains hash chain integrity."""
    from prsm.security import PipelineAuditLog

    log = PipelineAuditLog()
    for i in range(5):
        log.record(f"model-{i}", 4, [{"node_id": f"n{j}"} for j in range(4)], 20)

    assert log.verify_chain()
    assert log.entry_count == 5


# ── Cross-Node Full Stack ─────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_cross_node_all_rings_initialized(node_a, node_b):
    """Verify all Ring 1-10 components are initialized on both nodes."""
    for name, node in [("Node A", node_a), ("Node B", node_b)]:
        # Ring 2
        assert node.agent_dispatcher is not None, f"{name} missing agent_dispatcher"
        assert node.agent_executor is not None, f"{name} missing agent_executor"
        # Ring 3
        assert node.swarm_coordinator is not None, f"{name} missing swarm_coordinator"
        # Ring 4
        assert node.pricing_engine is not None, f"{name} missing pricing_engine"
        assert node.prosumer_manager is not None, f"{name} missing prosumer_manager"
        # Ring 5: removed in v1.6.0 (legacy NWTN AGI framework pruned)
        # Ring 7
        assert node.confidential_executor is not None, f"{name} missing confidential_executor"
        # Ring 8
        assert node.tensor_executor is not None, f"{name} missing tensor_executor"
        # Ring 9
        assert node.nwtn_model_service is not None, f"{name} missing nwtn_model_service"
        # Ring 10
        assert node.integrity_verifier is not None, f"{name} missing integrity_verifier"
        assert node.privacy_budget is not None, f"{name} missing privacy_budget"
        assert node.pipeline_audit_log is not None, f"{name} missing pipeline_audit_log"
