"""prsm_webhook_test MCP tool handler.

Wraps POST /admin/webhook-test for AI-side-panel webhook config
verification.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_webhook_test,
)


class TestRegistration:
    def test_handler_registered(self):
        assert "prsm_webhook_test" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_webhook_test" in names


class TestHandler:
    @pytest.mark.asyncio
    async def test_renders_pass_on_success(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "POST"
            assert "/admin/webhook-test" in path
            return {
                "success": True,
                "status_code": 200,
                "attempts": 1,
                "error": None,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_webhook_test({})
        assert "PASS" in result
        assert "200" in result
        assert "FAIL" not in result

    @pytest.mark.asyncio
    async def test_renders_fail_with_error_detail(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "success": False,
                "status_code": 503,
                "attempts": 3,
                "error": "service unavailable",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_webhook_test({})
        assert "FAIL" in result
        assert "503" in result
        assert "service unavailable" in result
        assert "operator action" in result.lower()

    @pytest.mark.asyncio
    async def test_503_not_configured_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "detail": "Webhook not configured. Set PRSM_WEBHOOK_URL.",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_webhook_test({})
        assert "not configured" in result.lower()
        assert "PRSM_WEBHOOK_URL" in result


class TestErrors:
    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_webhook_test({})
        assert "cannot reach" in result.lower()
