"""prsm_slash_history MCP wrapper."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_slash_history,
)


class TestRegistration:
    def test_handler_registered(self):
        assert "prsm_slash_history" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_slash_history" in names


class TestHandler:
    @pytest.mark.asyncio
    async def test_renders_recent_slash_events(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "GET"
            assert "/admin/slash-history" in path
            return {
                "entries": [
                    {
                        "timestamp": 1700000000.0,
                        "kind": "proof_failure_slashed",
                        "provider": "0xPROVIDER",
                        "challenger": "0xCHAL",
                        "slash_id": "0x" + "ab" * 32,
                        "extras": {"shard_id": "0xfffe"},
                    },
                    {
                        "timestamp": 1700000010.0,
                        "kind": "heartbeat_missing_slashed",
                        "provider": "0xPROV2",
                        "challenger": "0xCHAL2",
                        "slash_id": "0x" + "cd" * 32,
                        "extras": {"last_heartbeat_at": 1699996400},
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
            result = await handle_prsm_slash_history({})
        assert "proof_failure_slashed" in result
        assert "heartbeat_missing_slashed" in result
        assert "PRSM Slash Events" in result

    @pytest.mark.asyncio
    async def test_empty_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "entries": [], "total": 0,
                "offset": 0, "limit": 20,
            }

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_slash_history({})
        assert "No slash events" in result

    @pytest.mark.asyncio
    async def test_provider_filter_passthrough(self):
        captured = {}

        async def fake_call_node_api(method, path, data=None):
            captured["path"] = path
            return {"entries": [], "total": 0, "offset": 0, "limit": 20}

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            await handle_prsm_slash_history({"provider": "0xMINE"})
        assert "provider=0xMINE" in captured["path"]

    @pytest.mark.asyncio
    async def test_503_not_configured_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "detail": "Slash event log not initialized.",
            }

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_slash_history({})
        assert "not configured" in result.lower()
        assert "PRSM_STORAGE_SLASHING_WATCHER_ENABLED" in result

    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_slash_history({})
        assert "Cannot reach PRSM node" in result
