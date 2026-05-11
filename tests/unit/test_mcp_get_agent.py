"""Sprint 228 — prsm_get_agent MCP tool.

GET /agents/{agent_id} returns full agent details + current
allowance. Distinct from prsm_agents (list/search) which returns
short-form per agent. No MCP wrapper for the deep single-agent
view existed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_get_agent


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_get_agent" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_agent_id_rejected(self):
        result = await handle_prsm_get_agent({})
        assert "agent_id" in result.lower()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_renders_full_record(self):
        mock_resp = {
            "agent_id": "a-1",
            "display_name": "Agent One",
            "status": "online",
            "capabilities": ["image-gen", "summary"],
            "spent": 12.5,
            "allowance": 100.0,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_get_agent({"agent_id": "a-1"})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/agents/a-1"
        assert "a-1" in result
        assert "online" in result
        assert "image-gen" in result
        assert "12.5" in result
        assert "100" in result


class TestNotFound:
    @pytest.mark.asyncio
    async def test_404_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"detail": "Agent not found"}),
        ):
            result = await handle_prsm_get_agent({
                "agent_id": "missing",
            })
        assert "not found" in result.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_get_agent({"agent_id": "a-1"})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
