"""Ring 5 Smoke Test — full forge pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from decimal import Decimal

from prsm.compute.nwtn.agent_forge import AgentForge, ExecutionRoute
from prsm.economy.pricing.models import CostQuote
from prsm.compute.nwtn.agent_forge.mcp_tools import get_forge_tools


MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


class TestRing5Smoke:
    @pytest.mark.asyncio
    async def test_full_forge_pipeline(self):
        backend = AsyncMock()
        backend.execute_with_fallback = AsyncMock(return_value=MagicMock(
            content='{"required_datasets": ["nada-nc"], "operations": ["filter", "aggregate"], "parallelizable": true, "min_hardware_tier": "t2", "estimated_complexity": 0.6}',
        ))

        pricing = MagicMock()
        pricing.estimate_pcu = MagicMock(return_value=50.0)
        pricing.quote_swarm_job = MagicMock(return_value=CostQuote(
            compute_cost=Decimal("2.50"),
            data_cost=Decimal("5.00"),
            network_fee=Decimal("0.375"),
        ))

        swarm = AsyncMock()
        swarm.create_swarm_job = MagicMock(return_value=MagicMock(job_id="swarm-1"))
        swarm.execute = AsyncMock(return_value=MagicMock(
            job_id="swarm-1",
            shards_completed=3,
            total_pcu=1.5,
            aggregated_output={"shard_count": 3, "shard_outputs": [{"data": {"ev_count": 142}}]},
        ))

        forge = AgentForge(
            backend_registry=backend,
            pricing_engine=pricing,
            swarm_coordinator=swarm,
            template_wasm=MINIMAL_WASM,
        )

        result = await forge.run("EV adoption trends in NC 2025", budget_ftns=10.0, shard_cids=["QmA", "QmB", "QmC"])

        assert result is not None
        # run() returns {"query", "decomposition", "plan", "result", ...}
        # result["result"] contains the execution output with "route"
        assert result["result"]["route"] == "swarm"
        assert result["result"]["shards_completed"] == 3
        assert len(forge.traces) == 1

    @pytest.mark.asyncio
    async def test_simple_query_routes_to_llm(self):
        backend = AsyncMock()
        backend.execute_with_fallback = AsyncMock(side_effect=[
            MagicMock(content='{"required_datasets": [], "operations": [], "parallelizable": false, "min_hardware_tier": "t1", "estimated_complexity": 0.1}'),
            MagicMock(content="Paris is the capital of France."),
        ])

        forge = AgentForge(backend_registry=backend, template_wasm=MINIMAL_WASM)
        result = await forge.run("What is the capital of France?")

        assert result is not None
        assert result["result"]["route"] == "direct_llm"
        assert "Paris" in result["result"].get("response", "")

    def test_mcp_tools_defined(self):
        tools = get_forge_tools()
        assert len(tools) == 5
        names = [t["name"] for t in tools]
        assert "prsm_analyze" in names
        assert "prsm_quote" in names
        assert "prsm_list_datasets" in names
        assert "prsm_dispatch_agent" in names
        assert "prsm_swarm_status" in names

    def test_mcp_tools_have_required_fields(self):
        for tool in get_forge_tools():
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert len(tool["description"]) > 20
