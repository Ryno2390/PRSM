"""Sprint 214 — prsm_agents + prsm_agent_spending MCP tools.

GET /agents (with local_only filter) + GET /agents/search (by
capability) + GET /agents/spending had no MCP coverage. Operators
managing agents had to curl by hand.

prsm_agents: list or search agents. If `capability` provided,
            routes to /agents/search; else /agents with optional
            `local_only` flag.
prsm_agent_spending: aggregate spending dashboard.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS,
    handle_prsm_agents,
    handle_prsm_agent_spending,
)


class TestRegistration:
    def test_agents_tool_in_handlers(self):
        assert "prsm_agents" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_agents"] is handle_prsm_agents

    def test_agent_spending_tool_in_handlers(self):
        assert "prsm_agent_spending" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_agent_spending"] is handle_prsm_agent_spending


class TestAgentsList:
    @pytest.mark.asyncio
    async def test_list_renders(self):
        mock_resp = {
            "agents": [
                {"agent_id": "a-1", "display_name": "Agent One", "status": "online"},
                {"agent_id": "a-2", "display_name": "Agent Two", "status": "paused"},
            ],
            "count": 2,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_agents({})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1].startswith("/agents")
        assert "a-1" in result
        assert "a-2" in result

    @pytest.mark.asyncio
    async def test_local_only_passes_query(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"agents": [], "count": 0}),
        ) as mock_call:
            await handle_prsm_agents({"local_only": True})
        args, _ = mock_call.await_args
        assert "local_only=true" in args[1].lower()

    @pytest.mark.asyncio
    async def test_capability_routes_to_search(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "capability": "image-gen",
                "agents": [{"agent_id": "a-3"}],
                "count": 1,
            }),
        ) as mock_call:
            result = await handle_prsm_agents({"capability": "image-gen"})
        args, _ = mock_call.await_args
        assert "/agents/search" in args[1]
        assert "image-gen" in args[1]
        assert "a-3" in result


class TestAgentSpending:
    @pytest.mark.asyncio
    async def test_renders(self):
        mock_resp = {
            "agents": [
                {"agent_id": "a-1", "spent": 5.5, "allowance": 100.0},
            ],
            "total_spent": 5.5,
            "total_allowance": 100.0,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_agent_spending({})
        args, _ = mock_call.await_args
        assert args[1] == "/agents/spending"
        assert "5.5" in result
        assert "100" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_agents_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_agents({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
