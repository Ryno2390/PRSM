"""Sprint 236 — prsm_forge_quote MCP tool.

POST /compute/forge/quote returns a network-aware CostQuote that
the JS + Go SDKs call. The existing prsm_quote MCP tool uses a
LOCAL PricingEngine — it doesn't reflect real network state
(queue depth, peer availability) or run through server-side
validation. Need a separate MCP wrapper that hits the endpoint.

Pairs with prsm_forge_submit (which hits /compute/forge): use
prsm_forge_quote first for an accurate pre-flight cost estimate,
then prsm_forge_submit to execute.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_forge_quote


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_forge_quote" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_query_rejected(self):
        result = await handle_prsm_forge_quote({})
        assert "query" in result.lower()

    @pytest.mark.asyncio
    async def test_excessive_shard_count_rejected_locally(self):
        result = await handle_prsm_forge_quote({
            "query": "x", "shard_count": 1000,
        })
        assert "shard_count" in result.lower()

    @pytest.mark.asyncio
    async def test_bad_hardware_tier_rejected_locally(self):
        result = await handle_prsm_forge_quote({
            "query": "x", "hardware_tier": "t99",
        })
        assert "tier" in result.lower()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_routes_correctly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "compute_cost": 1.5,
                "data_cost": 0.1,
                "network_fee": 0.05,
                "total": 1.65,
                "hardware_tier": "t2",
                "shard_count": 3,
            }),
        ) as mock_call:
            result = await handle_prsm_forge_quote({
                "query": "summarize this dataset",
                "shard_count": 3,
                "hardware_tier": "t2",
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1] == "/compute/forge/quote"
        body = args[2] if len(args) > 2 else {}
        assert body.get("query") == "summarize this dataset"
        assert body.get("shard_count") == 3
        assert "1.65" in result
        assert "t2" in result.lower()

    @pytest.mark.asyncio
    async def test_shard_cids_count_overrides(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "compute_cost": 5.0, "data_cost": 0.3,
                "network_fee": 0.15, "total": 5.45,
                "hardware_tier": "t2", "shard_count": 5,
            }),
        ) as mock_call:
            await handle_prsm_forge_quote({
                "query": "x",
                "shard_cids": ["a", "b", "c", "d", "e"],
            })
        args, _ = mock_call.await_args
        body = args[2] if len(args) > 2 else {}
        # Pass shard_cids through so the server count derivation
        # is authoritative.
        assert body.get("shard_cids") == ["a", "b", "c", "d", "e"]


class TestServer422:
    @pytest.mark.asyncio
    async def test_server_validation_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"detail": "shard_count must be in [1, 100]"}),
        ):
            result = await handle_prsm_forge_quote({"query": "x"})
        # Server-side error gets surfaced.
        assert "shard_count" in result.lower() or "refused" in result.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_forge_quote({"query": "x"})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
