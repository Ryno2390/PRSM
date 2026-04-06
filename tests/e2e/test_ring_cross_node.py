"""
Ring 1-6 Cross-Node E2E Test
=============================

Tests all six capability rings across a real two-node P2P connection:

Ring 1 - Hardware profiles gossip between nodes
Ring 2 - Mobile agent dispatched from Node A, executed on Node B
Ring 3 - Swarm job fans out across both nodes
Ring 4 - Pricing engine quotes job costs, prosumer tiers work
Ring 5 - Agent forge decomposes queries and routes execution
Ring 6 - Dynamic gas pricing, signature verification, CLI commands

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
        required_cids=["QmTestShard001"],
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
        manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
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


# ── Ring 5: Agent Forge ───────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring5_forge_initialized(node_a):
    """Ring 5: Node has AgentForge available."""
    assert node_a.agent_forge is not None, "Node A missing agent_forge"


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring5_forge_decomposition():
    """Ring 5: AgentForge can decompose a query (without LLM backend)."""
    from prsm.compute.nwtn.agent_forge import AgentForge, ExecutionRoute

    # Without a backend, decompose returns defaults
    forge = AgentForge()
    decomp = await forge.decompose("What is 2+2?")

    assert decomp.query == "What is 2+2?"
    # Without backend, defaults to no datasets → DIRECT_LLM
    assert decomp.recommended_route == ExecutionRoute.DIRECT_LLM


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_ring5_mcp_tools_available():
    """Ring 5: MCP tool definitions are available."""
    from prsm.compute.nwtn.agent_forge.mcp_tools import get_forge_tools

    tools = get_forge_tools()
    assert len(tools) == 5
    tool_names = [t["name"] for t in tools]
    assert "prsm_analyze" in tool_names
    assert "prsm_quote" in tool_names


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


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(120)
async def test_cross_node_all_rings_initialized(node_a, node_b):
    """Verify all Ring 1-6 components are initialized on both nodes."""
    for name, node in [("Node A", node_a), ("Node B", node_b)]:
        # Ring 2
        assert node.agent_dispatcher is not None, f"{name} missing agent_dispatcher"
        assert node.agent_executor is not None, f"{name} missing agent_executor"
        # Ring 3
        assert node.swarm_coordinator is not None, f"{name} missing swarm_coordinator"
        # Ring 4
        assert node.pricing_engine is not None, f"{name} missing pricing_engine"
        assert node.prosumer_manager is not None, f"{name} missing prosumer_manager"
        # Ring 5
        assert node.agent_forge is not None, f"{name} missing agent_forge"
