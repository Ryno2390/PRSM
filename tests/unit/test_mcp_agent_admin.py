"""Sprint 221 — prsm_agent_admin MCP tool.

Four agent admin endpoints had no MCP coverage:
  - POST   /agents/{id}/allowance    (set allowance)
  - DELETE /agents/{id}/allowance    (revoke allowance)
  - POST   /agents/{id}/pause        (pause agent)
  - POST   /agents/{id}/resume       (resume agent)

Consolidates into single MCP tool with `action` selector. All
four operate on a single agent_id and have nearly-identical
response shapes — separate tools would sprawl the catalog.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_agent_admin


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_agent_admin" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_agent_id_rejected(self):
        result = await handle_prsm_agent_admin({"action": "pause"})
        assert "agent_id" in result.lower()

    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        result = await handle_prsm_agent_admin({"agent_id": "a-1"})
        assert "action" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        result = await handle_prsm_agent_admin({
            "agent_id": "a-1", "action": "bogus",
        })
        assert "must be" in result.lower()

    @pytest.mark.asyncio
    async def test_set_allowance_requires_amount(self):
        result = await handle_prsm_agent_admin({
            "agent_id": "a-1", "action": "set_allowance",
        })
        assert "amount" in result.lower()

    @pytest.mark.asyncio
    async def test_set_allowance_inf_amount_rejected(self):
        result = await handle_prsm_agent_admin({
            "agent_id": "a-1", "action": "set_allowance",
            "amount": float("inf"),
        })
        assert "amount" in result.lower() or "finite" in result.lower()


class TestRouting:
    @pytest.mark.asyncio
    async def test_set_allowance_routes_correctly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"agent_id": "a-1", "amount": 50.0}),
        ) as mock_call:
            await handle_prsm_agent_admin({
                "agent_id": "a-1", "action": "set_allowance",
                "amount": 50.0,
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert "/agents/a-1/allowance" in args[1]
        assert "amount=50" in args[1]

    @pytest.mark.asyncio
    async def test_revoke_uses_delete(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"agent_id": "a-1", "revoked": True}),
        ) as mock_call:
            await handle_prsm_agent_admin({
                "agent_id": "a-1", "action": "revoke",
            })
        args, _ = mock_call.await_args
        assert args[0] == "DELETE"
        assert args[1] == "/agents/a-1/allowance"

    @pytest.mark.asyncio
    async def test_pause_routes_correctly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"agent_id": "a-1", "status": "paused"}),
        ) as mock_call:
            await handle_prsm_agent_admin({
                "agent_id": "a-1", "action": "pause",
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1] == "/agents/a-1/pause"

    @pytest.mark.asyncio
    async def test_resume_routes_correctly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"agent_id": "a-1", "status": "online"}),
        ) as mock_call:
            await handle_prsm_agent_admin({
                "agent_id": "a-1", "action": "resume",
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1] == "/agents/a-1/resume"


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_agent_admin({
                "agent_id": "a-1", "action": "pause",
            })
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
