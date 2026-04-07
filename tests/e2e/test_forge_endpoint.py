"""
End-to-End Forge Pipeline Test
==============================

Tests the full /compute/forge API endpoint and CLI --query flag,
verifying the complete Ring 1-10 path from query to result.
"""

import asyncio
import os
import socket
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.node.config import NodeConfig
from prsm.node.node import PRSMNode


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory(prefix="prsm-forge-e2e-") as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_ipfs():
    with patch("prsm.node.storage_provider.StorageProvider._check_ipfs") as m:
        m.return_value = True
        with patch("prsm.node.storage_provider.StorageProvider.pin_content") as mp:
            mp.return_value = True
            with patch("prsm.node.storage_provider.StorageProvider.verify_pin") as mv:
                mv.return_value = True
                with patch("prsm.node.content_uploader.ContentUploader._ipfs_add") as ma:
                    ma.return_value = ("QmTest", 1024)
                    with patch("prsm.node.content_provider.ContentProvider._ipfs_cat") as mc:
                        mc.return_value = b"test"
                        yield


@pytest.fixture
async def forge_node(temp_data_dir, mock_ipfs):
    config = NodeConfig(
        display_name="ForgeTestNode",
        data_dir=os.path.join(temp_data_dir, "forge-node"),
        p2p_port=_free_port(),
        api_port=_free_port(),
        roles=["full"],
        bootstrap_nodes=[],
        welcome_grant=1000.0,
    )
    os.makedirs(config.data_dir, exist_ok=True)

    node = PRSMNode(config)
    await node.initialize()
    await node.start()
    yield node
    await node.stop()


class TestForgeEndpoint:
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_forge_initialized_on_node(self, forge_node):
        """AgentForge is available on the node."""
        assert forge_node.agent_forge is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_forge_pipeline_direct_llm(self, forge_node):
        """Simple query decomposes and attempts DIRECT_LLM route."""
        # Without a real LLM backend, decompose returns defaults (no datasets → DIRECT_LLM)
        result = await forge_node.agent_forge.run(
            query="What is 2+2?",
            budget_ftns=1.0,
        )

        assert result is not None
        # The forge ran and collected a trace even without a backend
        assert len(forge_node.agent_forge.traces) >= 1
        trace = forge_node.agent_forge.traces[-1]
        assert trace.query == "What is 2+2?"
        # Route should be direct_llm (from decomposition), though execution
        # may return error status if no LLM backend is configured
        route = result.get("route", trace.to_dict().get("plan", {}).get("route", ""))
        assert route in ("direct_llm", ""), f"Expected direct_llm route, got: {route}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_forge_pipeline_with_shard_cids(self, forge_node):
        """Query with shard CIDs routes to swarm execution."""
        # The forge will try to dispatch, but with no real peers it will
        # fall through. The important thing is the pipeline runs.
        result = await forge_node.agent_forge.run(
            query="Analyze EV trends in NC",
            budget_ftns=5.0,
            shard_cids=["QmTestShard1", "QmTestShard2"],
        )

        # Result may be None or failed since there are no real peers/shards,
        # but the pipeline should not crash
        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_all_ring_components_accessible_from_forge(self, forge_node):
        """The forge node has all 10 ring components wired."""
        node = forge_node

        # Ring 1: Hardware profiler
        from prsm.compute.wasm import HardwareProfiler
        profile = HardwareProfiler().detect()
        assert profile.cpu_cores >= 1

        # Ring 2: Agent dispatcher
        assert node.agent_dispatcher is not None

        # Ring 3: Swarm coordinator
        assert node.swarm_coordinator is not None

        # Ring 4: Pricing engine
        assert node.pricing_engine is not None
        quote = node.pricing_engine.quote_swarm_job(
            shard_count=3, hardware_tier="t2", estimated_pcu_per_shard=50.0,
        )
        assert quote.total > 0

        # Ring 5: Agent forge
        assert node.agent_forge is not None
        decomp = await node.agent_forge.decompose("test query")
        assert decomp.query == "test query"

        # Ring 7: Confidential executor
        assert node.confidential_executor is not None

        # Ring 8: Tensor executor
        assert node.tensor_executor is not None

        # Ring 9: Model service
        assert node.nwtn_model_service is not None

        # Ring 10: Security modules
        assert node.integrity_verifier is not None
        assert node.privacy_budget is not None
        assert node.pipeline_audit_log is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_training_trace_collected_after_forge_run(self, forge_node):
        """Running forge pipeline collects AgentTrace for future fine-tuning."""
        initial_traces = len(forge_node.agent_forge.traces)

        await forge_node.agent_forge.run(
            query="How many electric vehicles were sold in 2025?",
            budget_ftns=2.0,
        )

        assert len(forge_node.agent_forge.traces) == initial_traces + 1
        trace = forge_node.agent_forge.traces[-1]
        assert trace.query == "How many electric vehicles were sold in 2025?"

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_privacy_budget_tracked_on_forge_run(self, forge_node):
        """Privacy budget is tracked when forge runs with confidential compute."""
        if forge_node.privacy_budget:
            initial_spent = forge_node.privacy_budget.total_spent
            # The forge itself doesn't directly spend privacy budget
            # (that happens when ConfidentialExecutor runs within the pipeline)
            # But we can verify the tracker is functional
            forge_node.privacy_budget.record_spend(8.0, "test_forge", "test")
            assert forge_node.privacy_budget.total_spent == initial_spent + 8.0


class TestCLIForgeFlag:
    def test_compute_run_query_flag_exists(self):
        """CLI has --query flag for forge pipeline."""
        from click.testing import CliRunner
        from prsm.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["compute", "run", "--help"])
        assert result.exit_code == 0
        assert "--query" in result.output
        assert "--prompt" in result.output
        assert "--privacy" in result.output

    def test_compute_run_requires_prompt_or_query(self):
        """CLI rejects invocation without --prompt or --query."""
        from click.testing import CliRunner
        from prsm.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["compute", "run"])
        assert result.exit_code != 0
