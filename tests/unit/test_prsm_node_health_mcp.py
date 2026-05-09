"""prsm_node_health MCP tool handler.

Wraps GET /health/detailed for one-shot operator diagnostics
from the AI side panel. Distinct from prsm_node_status which
focuses on Ring activation; this surfaces subsystem readiness
(ftns_ledger, payment_escrow, job_history, royalty_distributor)
for ops/troubleshooting.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_node_health,
)


class TestToolRegistration:
    def test_handler_registered(self):
        assert "prsm_node_health" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_node_health" in names


class TestNodeHealthHandler:
    @pytest.mark.asyncio
    async def test_renders_healthy_status(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "GET"
            assert path == "/health/detailed"
            return {
                "status": "healthy",
                "node_id": "test-node",
                "subsystems": {
                    "ftns_ledger": {
                        "available": True, "status": "ok",
                        "connected_address": "0x" + "11" * 20,
                    },
                    "payment_escrow": {
                        "available": True, "status": "ok",
                        "pending_count": 0,
                        "default_timeout_sec": 3600.0,
                    },
                    "job_history": {
                        "available": True, "status": "ok",
                        "count": 5, "max_entries": 1024,
                        "persisted": False,
                    },
                    "royalty_distributor": {
                        "available": True, "status": "ok",
                        "claimable_wei": 0,
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_node_health({})
        assert "healthy" in result.lower()
        assert "ftns_ledger" in result
        assert "payment_escrow" in result
        assert "job_history" in result
        assert "royalty_distributor" in result

    @pytest.mark.asyncio
    async def test_renders_degraded_with_unavailable_subsystems(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "degraded",
                "node_id": "test-node",
                "subsystems": {
                    "ftns_ledger": {
                        "available": True, "status": "ok",
                        "connected_address": "0x" + "11" * 20,
                    },
                    "payment_escrow": {
                        "available": True, "status": "ok",
                        "pending_count": 2,
                        "default_timeout_sec": 3600.0,
                    },
                    "job_history": {
                        "available": False, "status": "not_wired",
                    },
                    "royalty_distributor": {
                        "available": False, "status": "not_wired",
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_node_health({})
        assert "degraded" in result.lower()
        # Missing subsystems flagged.
        # Output should include some marker for unavailable ones.
        assert "not_wired" in result or "unavailable" in result.lower()

    @pytest.mark.asyncio
    async def test_renders_unhealthy_when_core_missing(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "unhealthy",
                "node_id": "test-node",
                "subsystems": {
                    "ftns_ledger": {
                        "available": False, "status": "not_wired",
                    },
                    "payment_escrow": {
                        "available": False, "status": "not_wired",
                    },
                    "job_history": {
                        "available": False, "status": "not_wired",
                    },
                    "royalty_distributor": {
                        "available": False, "status": "not_wired",
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_node_health({})
        assert "unhealthy" in result.lower()

    @pytest.mark.asyncio
    async def test_subsystem_error_surfaces_in_output(self):
        """If a subsystem reports an error, show the error message
        so the operator can debug from the side panel."""
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "degraded",
                "node_id": "test-node",
                "subsystems": {
                    "ftns_ledger": {"available": True, "status": "ok"},
                    "payment_escrow": {"available": True, "status": "ok"},
                    "job_history": {"available": True, "status": "ok"},
                    "royalty_distributor": {
                        "available": False, "status": "error",
                        "error": "RPC unreachable: connection timeout",
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_node_health({})
        assert "RPC unreachable" in result or \
            "connection timeout" in result


class TestNodeHealthErrors:
    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_node_health({})
        assert "cannot reach" in result.lower()
