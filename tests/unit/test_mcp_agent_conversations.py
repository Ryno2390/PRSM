"""Sprint 227 — prsm_agent_conversations MCP tool.

GET /agents/{agent_id}/conversations returns recent conversation
threads involving an agent (with last-5-messages preview). No
MCP wrapper existed. Operators monitoring agent activity had to
curl.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS,
    handle_prsm_agent_conversations,
)


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_agent_conversations" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_agent_id_rejected(self):
        result = await handle_prsm_agent_conversations({})
        assert "agent_id" in result.lower()

    @pytest.mark.asyncio
    async def test_excessive_limit_rejected_locally(self):
        result = await handle_prsm_agent_conversations({
            "agent_id": "a-1", "limit": 1000,
        })
        assert "limit" in result.lower()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_renders(self):
        mock_resp = {
            "conversations": [
                {
                    "conversation_id": "c-1",
                    "message_count": 10,
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "agent", "content": "hi back"},
                    ],
                },
            ],
            "count": 1,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_agent_conversations({
                "agent_id": "a-1",
            })
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert "/agents/a-1/conversations" in args[1]
        assert "c-1" in result
        assert "10" in result  # message_count

    @pytest.mark.asyncio
    async def test_no_conversations_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"conversations": [], "count": 0}),
        ):
            result = await handle_prsm_agent_conversations({
                "agent_id": "a-1",
            })
        assert "no conversations" in result.lower() or "0" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_agent_conversations({
                "agent_id": "a-1",
            })
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
