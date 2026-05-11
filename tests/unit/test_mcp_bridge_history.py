"""Sprint 230 — prsm_bridge_history MCP tool.

Three bridge read endpoints had no MCP coverage:
  - GET /bridge/status — current bridge state
  - GET /bridge/transactions — list bridge transactions
  - GET /bridge/transactions/{tx_id} — single bridge tx lookup

Consolidates into single tool with `view` selector
(status|list|lookup).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_bridge_history


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_bridge_history" in TOOL_HANDLERS


class TestRouting:
    @pytest.mark.asyncio
    async def test_status_routes(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"available": True, "operational": True}),
        ) as mock_call:
            await handle_prsm_bridge_history({"view": "status"})
        args, _ = mock_call.await_args
        assert args[1] == "/bridge/status"

    @pytest.mark.asyncio
    async def test_list_routes(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"transactions": [], "count": 0}),
        ) as mock_call:
            await handle_prsm_bridge_history({"view": "list"})
        args, _ = mock_call.await_args
        assert args[1].startswith("/bridge/transactions")
        assert "/{tx_id}" not in args[1]

    @pytest.mark.asyncio
    async def test_lookup_routes(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"transaction_id": "tx-1"}),
        ) as mock_call:
            await handle_prsm_bridge_history({
                "view": "lookup", "tx_id": "tx-1",
            })
        args, _ = mock_call.await_args
        assert args[1] == "/bridge/transactions/tx-1"


class TestValidation:
    @pytest.mark.asyncio
    async def test_unknown_view_rejected(self):
        result = await handle_prsm_bridge_history({"view": "bogus"})
        assert "must be" in result.lower()

    @pytest.mark.asyncio
    async def test_lookup_requires_tx_id(self):
        result = await handle_prsm_bridge_history({"view": "lookup"})
        assert "tx_id" in result.lower()


class TestRenderStatus:
    @pytest.mark.asyncio
    async def test_renders_status(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "available": True,
                "operational": True,
                "supported_chains": [1, 137, 8453],
            }),
        ):
            result = await handle_prsm_bridge_history({"view": "status"})
        assert "available" in result.lower()
        assert "137" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_bridge_history({"view": "status"})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
