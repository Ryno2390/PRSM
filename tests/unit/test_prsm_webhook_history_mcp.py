"""prsm_webhook_history MCP tool handler.

Wraps GET /admin/webhook-history for AI-side-panel inspection
of recent webhook dispatch attempts.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_webhook_history,
)


class TestRegistration:
    def test_handler_registered(self):
        assert "prsm_webhook_history" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_webhook_history" in names


class TestHandler:
    @pytest.mark.asyncio
    async def test_renders_recent_dispatches(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "GET"
            assert "/admin/webhook-history" in path
            return {
                "entries": [
                    {
                        "timestamp": 1700000000.0,
                        "event": "daemon.crashed",
                        "url": "https://hook.example.com",
                        "success": True,
                        "attempts": 1,
                        "status_code": 200,
                        "error": None,
                    },
                    {
                        "timestamp": 1700000010.0,
                        "event": "escrow.leaked",
                        "url": "https://hook.example.com",
                        "success": False,
                        "attempts": 3,
                        "status_code": 503,
                        "error": "service unavailable",
                    },
                ],
                "total": 2,
                "offset": 0,
                "limit": 20,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_webhook_history({})
        assert "daemon.crashed" in result
        assert "escrow.leaked" in result
        assert "[ok]" in result
        assert "[!]" in result
        assert "service unavailable" in result

    @pytest.mark.asyncio
    async def test_empty_buffer_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "entries": [], "total": 0,
                "offset": 0, "limit": 20,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_webhook_history({})
        assert "No webhook dispatches" in result

    @pytest.mark.asyncio
    async def test_pagination_propagates(self):
        captured = {}

        async def fake_call_node_api(method, path, data=None):
            captured["path"] = path
            return {
                "entries": [], "total": 0,
                "offset": 5, "limit": 3,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            await handle_prsm_webhook_history({"limit": 3, "offset": 5})
        assert "limit=3" in captured["path"]
        assert "offset=5" in captured["path"]


class TestErrors:
    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_webhook_history({})
        assert "cannot reach" in result.lower()

    @pytest.mark.asyncio
    async def test_503_not_configured_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "detail": "Webhook log not initialized.",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_webhook_history({})
        assert "not configured" in result.lower()
        assert "PRSM_WEBHOOK_URL" in result
