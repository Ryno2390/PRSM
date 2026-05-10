"""prsm_heartbeat_trigger MCP wrapper."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_heartbeat_trigger,
)


class TestRegistration:
    def test_handler_registered(self):
        assert "prsm_heartbeat_trigger" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_heartbeat_trigger" in names


class TestHandler:
    @pytest.mark.asyncio
    async def test_renders_tx_hash_on_success(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "POST"
            assert "/admin/heartbeat/trigger" in path
            return {"tx_hash": "0xABC123", "status": "CONFIRMED"}

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_heartbeat_trigger({})
        assert "0xABC123" in result
        assert "CONFIRMED" in result

    @pytest.mark.asyncio
    async def test_503_not_wired_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {"detail": "StorageSlashingClient not wired."}

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_heartbeat_trigger({})
        assert "not wired" in result.lower()

    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_heartbeat_trigger({})
        assert "Cannot reach PRSM node" in result
