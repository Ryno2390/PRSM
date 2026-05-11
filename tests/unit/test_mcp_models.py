"""Sprint 235 — prsm_models MCP tool.

Pairs with the new GET /compute/models endpoint. End-users
running prsm_inference can now discover available model_ids
via MCP instead of guessing from the tool description.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_models


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_models" in TOOL_HANDLERS


class TestRender:
    @pytest.mark.asyncio
    async def test_renders_model_list(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "models": [
                    "mock-llama-3-8b",
                    "mock-mistral-7b",
                    "mock-phi-3",
                ],
                "count": 3,
            }),
        ) as mock_call:
            result = await handle_prsm_models({})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/compute/models"
        assert "mock-llama-3-8b" in result
        assert "mock-mistral-7b" in result
        assert "3" in result

    @pytest.mark.asyncio
    async def test_empty_list_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"models": [], "count": 0}),
        ):
            result = await handle_prsm_models({})
        assert "no models" in result.lower() or "0" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_models({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
