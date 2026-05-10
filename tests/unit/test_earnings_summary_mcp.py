"""prsm_earnings_summary MCP wrapper."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_earnings_summary,
)


class TestRegistration:
    def test_handler_registered(self):
        assert "prsm_earnings_summary" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_earnings_summary" in names


class TestHandler:
    @pytest.mark.asyncio
    async def test_renders_all_streams_wired(self):
        now = int(time.time())

        async def fake_call_node_api(method, path, data=None):
            assert method == "GET"
            assert "/admin/earnings-summary" in path
            return {
                "operator_address": "0xABC",
                "royalty": {
                    "available": True,
                    "claimable_wei": 1_500_000_000_000_000_000,  # 1.5 FTNS
                    "address": "0xROY",
                },
                "heartbeat": {
                    "available": True,
                    "last_heartbeat": now - 100,
                    "grace_seconds": 3600,
                    "grace_remaining": 3500,
                    "expired": False,
                    "at_risk": False,
                },
                "distribution": {
                    "available": True,
                    "last_distribution": now - 7200,
                    "seconds_since": 7200,
                },
            }

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_earnings_summary({})
        assert "0xABC" in result
        assert "1.500000 FTNS" in result
        assert "ok" in result
        assert "2h ago" in result

    @pytest.mark.asyncio
    async def test_renders_at_risk_heartbeat(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "operator_address": "0xABC",
                "royalty": {"available": False},
                "heartbeat": {
                    "available": True,
                    "last_heartbeat": int(time.time()) - 3500,
                    "grace_seconds": 3600,
                    "grace_remaining": 100,
                    "expired": False,
                    "at_risk": True,
                },
                "distribution": {"available": False},
            }

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_earnings_summary({})
        assert "at-risk" in result
        assert "100s grace remaining" in result

    @pytest.mark.asyncio
    async def test_renders_expired_heartbeat(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "operator_address": "0xABC",
                "royalty": {"available": False},
                "heartbeat": {
                    "available": True,
                    "last_heartbeat": 1700000000,
                    "grace_seconds": 3600,
                    "grace_remaining": 0,
                    "expired": True,
                    "at_risk": True,
                },
                "distribution": {"available": False},
            }

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_earnings_summary({})
        assert "EXPIRED" in result
        assert "slashing window open" in result

    @pytest.mark.asyncio
    async def test_renders_unwired_streams_clearly(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "operator_address": None,
                "royalty": {"available": False},
                "heartbeat": {"available": False},
                "distribution": {"available": False},
            }

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_earnings_summary({})
        assert "PRSM_OPERATOR_ADDRESS unset" in result
        assert "Royalty:" in result
        assert "not wired" in result

    @pytest.mark.asyncio
    async def test_renders_per_stream_error(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "operator_address": "0xABC",
                "royalty": {
                    "available": False,
                    "error": "rpc connection refused",
                },
                "heartbeat": {"available": False},
                "distribution": {"available": False},
            }

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_earnings_summary({})
        assert "rpc connection refused" in result
        assert "[!]" in result

    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_earnings_summary({})
        assert "Cannot reach PRSM node" in result
