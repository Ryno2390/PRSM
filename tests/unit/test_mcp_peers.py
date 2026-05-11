"""Sprint 213 — prsm_peers MCP tool.

GET /peers returns the node's connected peer list. No MCP wrapper
existed, so operators triaging cross-node connectivity had to
curl manually. Useful for verifying bootstrap connectivity per
the DO bootstrap server entry in MEMORY.md.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_peers


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_peers" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_peers"] is handle_prsm_peers


class TestRender:
    @pytest.mark.asyncio
    async def test_renders_peer_list(self):
        mock_resp = {
            "connected": [
                {
                    "peer_id": "peer-1",
                    "address": "wss://bootstrap1.prsm-network.com:8765",
                    "display_name": "bootstrap-1",
                    "outbound": True,
                },
                {
                    "peer_id": None,
                    "address": "/ip4/10.0.0.5/tcp/4001",
                    "display_name": "",
                    "outbound": False,
                },
            ],
            "connected_count": 2,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_peers({})
        mock_call.assert_awaited_once()
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/peers"
        assert "peer-1" in result
        assert "bootstrap" in result.lower()
        assert "10.0.0.5" in result
        assert "outbound" in result.lower() or "inbound" in result.lower()
        assert "2" in result  # count

    @pytest.mark.asyncio
    async def test_no_peers_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"connected": [], "connected_count": 0}),
        ):
            result = await handle_prsm_peers({})
        assert isinstance(result, str)
        assert "no peers" in result.lower() or "0" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_node_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_peers({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
