"""Sprint 211 — prsm_info MCP tool.

GET /info shipped 2026-05-09 (`info-endpoint-merge-ready-20260509`)
surfaces static node metadata: node_id, api_version, network,
chain_id, rpc_host, operator_address, agent_forge_wired,
canonical_addresses. No MCP wrapper existed, so end-users
verifying which network/contracts a node is pinned to had to
curl manually.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_info


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_info" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_info"] is handle_prsm_info


class TestRender:
    @pytest.mark.asyncio
    async def test_renders_full_info(self):
        mock_resp = {
            "node_id": "node-abc",
            "api_version": "1.6.0",
            "network": "base-mainnet",
            "chain_id": 8453,
            "rpc_host": "mainnet.base.org",
            "operator_address": "0xdEADbEef0000000000000000000000000000B005",
            "agent_forge_wired": True,
            "canonical_addresses": {
                "ftns_token": "0x5276a3756C85f2E9e46f6D34386167a209aa16e5",
                "royalty_distributor": "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e",
            },
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_info({})
        mock_call.assert_awaited_once()
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/info"
        assert "node-abc" in result
        assert "1.6.0" in result
        assert "base-mainnet" in result
        assert "8453" in result
        assert "mainnet.base.org" in result
        assert "0xdEADbEef" in result.replace("...", "0xdEADbEef")
        assert "ftns_token" in result
        assert "agent_forge_wired" in result.lower() or "wired" in result.lower()

    @pytest.mark.asyncio
    async def test_minimal_response_renders(self):
        """/info always returns at least node_id + api_version."""
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "node_id": "n1", "api_version": "1.6.0",
            }),
        ):
            result = await handle_prsm_info({})
        assert "n1" in result
        assert "1.6.0" in result

    @pytest.mark.asyncio
    async def test_qo_error_surfaced(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "node_id": "n1",
                "api_version": "1.6.0",
                "agent_forge_wired": False,
                "query_orchestrator_state": "disabled",
                "query_orchestrator_error": "PRSM_QUERY_ORCHESTRATOR_ENABLED not set",
            }),
        ):
            result = await handle_prsm_info({})
        assert "disabled" in result.lower()
        assert "PRSM_QUERY_ORCHESTRATOR_ENABLED" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_node_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_info({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
