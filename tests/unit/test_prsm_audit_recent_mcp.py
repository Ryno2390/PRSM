"""prsm_audit_recent MCP tool handler."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_audit_recent,
)


class TestRegistration:
    def test_handler_registered(self):
        assert "prsm_audit_recent" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_audit_recent" in names


class TestHandler:
    @pytest.mark.asyncio
    async def test_renders_entries(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "GET"
            assert "/audit/recent" in path
            return {
                "entries": [
                    {
                        "timestamp": 1700000000.0,
                        "method": "POST",
                        "path": "/compute/forge",
                        "requester": "node-1",
                        "status_code": 200,
                        "request_id": "r1",
                    },
                    {
                        "timestamp": 1700000010.0,
                        "method": "POST",
                        "path": "/compute/cancel/job-x",
                        "requester": "node-1",
                        "status_code": 200,
                        "request_id": "r2",
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
            result = await handle_prsm_audit_recent({})
        assert "/compute/forge" in result
        assert "/compute/cancel/job-x" in result
        assert "POST" in result
        assert "200" in result

    @pytest.mark.asyncio
    async def test_empty_buffer_renders_friendly_message(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "entries": [],
                "total": 0,
                "offset": 0,
                "limit": 20,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_audit_recent({})
        assert "No state-changing" in result

    @pytest.mark.asyncio
    async def test_pagination_propagates(self):
        captured = {}

        async def fake_call_node_api(method, path, data=None):
            captured["path"] = path
            return {"entries": [], "total": 0, "offset": 5, "limit": 3}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            await handle_prsm_audit_recent({"limit": 3, "offset": 5})
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
            result = await handle_prsm_audit_recent({})
        assert "cannot reach" in result.lower()
