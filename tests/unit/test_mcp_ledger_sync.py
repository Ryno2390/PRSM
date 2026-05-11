"""Sprint 233 — prsm_ledger_sync MCP tool.

GET /ledger/sync/stats returns ledger gossip-sync statistics
(messages broadcast, peers in sync, last sync timestamp). No
MCP wrapper. Useful for operators verifying their node is
participating in ledger gossip.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_ledger_sync


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_ledger_sync" in TOOL_HANDLERS


class TestRender:
    @pytest.mark.asyncio
    async def test_renders_stats(self):
        mock_resp = {
            "messages_broadcast": 1234,
            "messages_received": 5678,
            "peers_in_sync": 7,
            "last_sync_ts": 1715000000.0,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_ledger_sync({})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/ledger/sync/stats"
        assert "1234" in result
        assert "5678" in result
        assert "7" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_ledger_sync({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
