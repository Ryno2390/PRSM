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


class TestNodeHealthCanonicalMatch:
    """Renders canonical-match status for on-chain subsystems
    (post-A-08 ceremony 2026-05-09)."""

    @pytest.mark.asyncio
    async def test_match_renders_concise_confirmation(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "healthy",
                "node_id": "test-node",
                "subsystems": {
                    "ftns_ledger": {
                        "available": True, "status": "ok",
                        "wired_address": "0x5276a3756C85f2E9e46f6D34386167a209aa16e5",
                        "canonical_address": "0x5276a3756C85f2E9e46f6D34386167a209aa16e5",
                        "canonical_match": True,
                    },
                    "payment_escrow": {"available": True, "status": "ok"},
                    "job_history": {"available": True, "status": "ok"},
                    "royalty_distributor": {
                        "available": True, "status": "ok",
                        "wired_address": "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e",
                        "canonical_address": "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e",
                        "canonical_match": True,
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_node_health({})
        assert "canonical pin matches" in result
        # Both subsystems should show match indicator.
        assert result.count("canonical pin matches") >= 2

    @pytest.mark.asyncio
    async def test_mismatch_renders_loudly_with_action_hint(self):
        """canonical_match: False operator-side surfaces loudly with
        action hint. Specifically validates the post-A-08 v1→v2
        scenario."""
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "healthy",
                "node_id": "test-node",
                "subsystems": {
                    "ftns_ledger": {"available": True, "status": "ok"},
                    "payment_escrow": {"available": True, "status": "ok"},
                    "job_history": {"available": True, "status": "ok"},
                    "royalty_distributor": {
                        "available": True, "status": "ok",
                        "wired_address": "0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2",  # v1
                        "canonical_address": "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e",  # v2
                        "canonical_match": False,
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_node_health({})
        assert "MISMATCH" in result
        # Both wired and canonical addresses surfaced.
        assert "0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2" in result  # wired (v1)
        assert "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e" in result  # canonical (v2)
        # Action hint present.
        assert "operator action" in result.lower()
        assert "PRSM_" in result  # env override hint

    @pytest.mark.asyncio
    async def test_no_canonical_field_omitted_silently(self):
        """When canonical_match is absent (e.g., local network),
        the indicator line is simply not rendered."""
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "healthy",
                "node_id": "test-node",
                "subsystems": {
                    "ftns_ledger": {"available": True, "status": "ok"},
                    "payment_escrow": {"available": True, "status": "ok"},
                    "job_history": {"available": True, "status": "ok"},
                    "royalty_distributor": {"available": True, "status": "ok"},
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_node_health({})
        assert "canonical pin matches" not in result
        assert "MISMATCH" not in result


class TestCleanupTaskRendering:
    """payment_escrow cleanup_task_running surfaces explicitly in
    the rendered output so operators see silent-crash signal in
    the AI side panel."""

    @pytest.mark.asyncio
    async def test_cleanup_task_crashed_renders_loud_marker(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "healthy",
                "node_id": "test-node",
                "subsystems": {
                    "ftns_ledger": {"available": True, "status": "ok"},
                    "payment_escrow": {
                        "available": True, "status": "ok",
                        "pending_count": 2,
                        "cleanup_task_running": False,  # CRASHED
                    },
                    "job_history": {"available": True, "status": "ok"},
                    "royalty_distributor": {
                        "available": True, "status": "ok",
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_node_health({})
        assert "CRASHED" in result
        assert "[!]" in result

    @pytest.mark.asyncio
    async def test_cleanup_task_ok_renders_subtle_indicator(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "healthy",
                "node_id": "test-node",
                "subsystems": {
                    "ftns_ledger": {"available": True, "status": "ok"},
                    "payment_escrow": {
                        "available": True, "status": "ok",
                        "pending_count": 2,
                        "cleanup_task_running": True,
                    },
                    "job_history": {"available": True, "status": "ok"},
                    "royalty_distributor": {
                        "available": True, "status": "ok",
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_node_health({})
        assert "cleanup_task: ok" in result
        assert "CRASHED" not in result


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


class TestTickAgeRendering:
    """Sprint 404 — surface the sprint 399-401 tick_status +
    last_tick_age_seconds fields in the prsm_node_health
    MCP renderer. Operators triaging via Claude Code see
    silent-economic-failure modes (task running but every
    tick failing) directly in the side panel."""

    def _fleet(self, *, heartbeat_tick_status, age_seconds):
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "healthy",
                "node_id": "test-node",
                "subsystems": {
                    "ftns_ledger": {"available": True, "status": "ok"},
                    "payment_escrow": {
                        "available": True, "status": "ok",
                        "pending_count": 0,
                    },
                    "job_history": {"available": True, "status": "ok"},
                    "royalty_distributor": {
                        "available": True, "status": "ok",
                    },
                    "heartbeat_scheduler": {
                        "available": True, "status": "ok",
                        "interval_seconds": 900,
                        "task_running": True,
                        "last_tick_age_seconds": age_seconds,
                        "tick_status": heartbeat_tick_status,
                    },
                },
            }
        return fake_call_node_api

    @pytest.mark.asyncio
    async def test_stale_tick_renders_loud_marker(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=self._fleet(
                heartbeat_tick_status="stale",
                age_seconds=5000,
            ),
        ):
            result = await handle_prsm_node_health({})
        # Stale = silent-economic-failure. Loud marker like
        # the cleanup_task CRASHED treatment.
        assert "stale" in result.lower()
        assert "heartbeat_scheduler" in result
        # Age surfaces so operators see how bad it is
        assert "5000" in result or "5,000" in result

    @pytest.mark.asyncio
    async def test_degraded_tick_renders_warning_marker(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=self._fleet(
                heartbeat_tick_status="degraded",
                age_seconds=2200,
            ),
        ):
            result = await handle_prsm_node_health({})
        assert "degraded" in result.lower()
        assert "2200" in result or "2,200" in result

    @pytest.mark.asyncio
    async def test_healthy_tick_subtle_or_omitted(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=self._fleet(
                heartbeat_tick_status="healthy",
                age_seconds=12,
            ),
        ):
            result = await handle_prsm_node_health({})
        # Healthy = don't shout. Output should NOT contain
        # the loud "stale" / "degraded" markers.
        assert "stale" not in result.lower()
        assert "[!]" not in result or "cleanup_task" in result
        # heartbeat_scheduler still appears in the subsystem
        # list at minimum
        assert "heartbeat_scheduler" in result

    @pytest.mark.asyncio
    async def test_subsystem_without_tick_status_unchanged(self):
        """Backwards-compat: subsystems that don't have
        tick_status at all (ftns_ledger, payment_escrow, etc.)
        render the same as pre-sprint-404."""
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "healthy",
                "node_id": "test-node",
                "subsystems": {
                    "ftns_ledger": {"available": True, "status": "ok"},
                    "payment_escrow": {
                        "available": True, "status": "ok",
                        "pending_count": 5,
                    },
                    "job_history": {"available": True, "status": "ok"},
                    "royalty_distributor": {
                        "available": True, "status": "ok",
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_node_health({})
        # No tick_status / stale / degraded markers anywhere
        assert "stale" not in result.lower()
        assert "tick" not in result.lower()
