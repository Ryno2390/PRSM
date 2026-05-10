"""prsm_content_info MCP wrapper for /content/{cid}."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_content_info,
)


class TestRegistration:
    def test_handler_registered(self):
        assert "prsm_content_info" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_content_info" in names


class TestHandler:
    @pytest.mark.asyncio
    async def test_renders_content_record(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "GET"
            assert "/content/cid123" in path
            return {
                "cid": "cid123",
                "filename": "paper.pdf",
                "size_bytes": 12345,
                "content_hash": "0xCAFE",
                "creator_id": "creator1",
                "providers": ["nodeA", "nodeB"],
                "royalty_rate": 0.05,
                "parent_cids": ["citedA"],
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_content_info({"cid": "cid123"})
        assert "paper.pdf" in result
        assert "0.0500" in result
        assert "creator1" in result
        assert "nodeA" in result

    @pytest.mark.asyncio
    async def test_missing_cid(self):
        result = await handle_prsm_content_info({})
        assert "Missing required" in result

    @pytest.mark.asyncio
    async def test_404_friendly(self):
        async def fake(method, path, data=None):
            return {"detail": "Content not found in index"}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake,
        ):
            result = await handle_prsm_content_info({"cid": "missing"})
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_content_info({"cid": "x"})
        assert "Cannot reach PRSM node" in result
