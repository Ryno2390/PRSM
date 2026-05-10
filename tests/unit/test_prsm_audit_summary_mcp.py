"""prsm_audit_summary MCP tool handler.

Wraps GET /audit/summary for AI-side-panel dashboard view.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_audit_summary,
)


class TestRegistration:
    def test_handler_registered(self):
        assert "prsm_audit_summary" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_audit_summary" in names


class TestHandler:
    @pytest.mark.asyncio
    async def test_renders_full_summary(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "GET"
            return {
                "total": 47,
                "status_buckets": {"2xx": 38, "4xx": 6, "5xx": 3},
                "method_buckets": {"POST": 42, "DELETE": 5},
                "top_paths": [
                    {"path": "/compute/forge", "count": 18},
                    {"path": "/wallet/royalty/claim", "count": 12},
                ],
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_audit_summary({})
        # Status bucket counts surface
        assert "38" in result
        assert "6" in result
        assert "3" in result
        # Method counts
        assert "POST" in result and "42" in result
        # Top paths sorted
        assert "/compute/forge" in result
        assert "18" in result
        assert "/wallet/royalty/claim" in result

    @pytest.mark.asyncio
    async def test_top_paths_arg_propagates(self):
        captured = {}

        async def fake_call_node_api(method, path, data=None):
            captured["path"] = path
            return {
                "total": 0,
                "status_buckets": {},
                "method_buckets": {},
                "top_paths": [],
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            await handle_prsm_audit_summary({"top_paths": 25})
        assert "top_paths=25" in captured["path"]

    @pytest.mark.asyncio
    async def test_empty_buffer_renders_empty_indicators(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "total": 0,
                "status_buckets": {},
                "method_buckets": {},
                "top_paths": [],
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_audit_summary({})
        assert "(empty)" in result


class TestErrors:
    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_audit_summary({})
        assert "cannot reach" in result.lower()
