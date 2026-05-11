"""Sprint 237 — prsm_inference_quote MCP tool.

Pairs with the new POST /compute/inference/quote endpoint.
End-users can now budget for inference jobs without locking
escrow.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_inference_quote


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_inference_quote" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_prompt_rejected(self):
        result = await handle_prsm_inference_quote({
            "model_id": "mock-llama-3-8b",
        })
        assert "prompt" in result.lower()

    @pytest.mark.asyncio
    async def test_missing_model_id_rejected(self):
        result = await handle_prsm_inference_quote({"prompt": "hi"})
        assert "model_id" in result.lower()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_routes_correctly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "model_id": "mock-llama-3-8b",
                "cost_ftns": "0.25",
                "privacy_tier": "standard",
                "content_tier": "A",
            }),
        ) as mock_call:
            result = await handle_prsm_inference_quote({
                "prompt": "hi",
                "model_id": "mock-llama-3-8b",
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1] == "/compute/inference/quote"
        body = args[2] if len(args) > 2 else {}
        assert body.get("prompt") == "hi"
        assert body.get("model_id") == "mock-llama-3-8b"
        assert "0.25" in result
        assert "mock-llama-3-8b" in result


class TestErrors:
    @pytest.mark.asyncio
    async def test_unsupported_model_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"detail": "Unknown model_id: bogus"}),
        ):
            result = await handle_prsm_inference_quote({
                "prompt": "hi", "model_id": "bogus",
            })
        assert "unknown" in result.lower() or "refused" in result.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_inference_quote({
                "prompt": "hi", "model_id": "mock-llama-3-8b",
            })
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
