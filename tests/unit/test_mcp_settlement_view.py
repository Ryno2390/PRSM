"""Sprint 231 — prsm_settlement_view MCP tool.

Three settlement endpoints had no MCP coverage:
  - GET  /settlement/pending  — un-settled transfers
  - GET  /settlement/history  — recent settlement results
  - POST /settlement/flush    — manually trigger batch settlement

Consolidates into single tool with `action` selector
(pending|history|flush). prsm_settlement_stats already exists.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_settlement_view


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_settlement_view" in TOOL_HANDLERS


class TestRouting:
    @pytest.mark.asyncio
    async def test_pending_routes(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"pending": [], "count": 0}),
        ) as mock_call:
            await handle_prsm_settlement_view({"action": "pending"})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/settlement/pending"

    @pytest.mark.asyncio
    async def test_history_routes(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"history": [], "count": 0}),
        ) as mock_call:
            await handle_prsm_settlement_view({"action": "history"})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1].startswith("/settlement/history")

    @pytest.mark.asyncio
    async def test_flush_routes(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "settled_count": 3,
                "total_amount": 100.0,
                "net_transfers": 2,
                "tx_hashes": ["0xabc"],
                "errors": [],
                "duration_seconds": 1.5,
            }),
        ) as mock_call:
            result = await handle_prsm_settlement_view({"action": "flush"})
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1] == "/settlement/flush"
        assert "3" in result and "1.5" in result


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        result = await handle_prsm_settlement_view({})
        assert "action" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        result = await handle_prsm_settlement_view({"action": "bogus"})
        assert "must be" in result.lower()


class TestRender:
    @pytest.mark.asyncio
    async def test_pending_empty_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"pending": [], "count": 0}),
        ):
            result = await handle_prsm_settlement_view({"action": "pending"})
        assert "no pending" in result.lower() or "0" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_settlement_view({"action": "pending"})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
