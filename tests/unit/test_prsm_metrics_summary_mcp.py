"""prsm_metrics_summary MCP tool handler.

Wraps GET /metrics and renders the Prometheus exposition into
a human-readable side-panel summary. Distinct from
prsm_node_health (subsystem readiness) — this surfaces actual
operational metric values for triage.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_metrics_summary,
)


class TestToolRegistration:
    def test_handler_registered(self):
        assert "prsm_metrics_summary" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_metrics_summary" in names


class TestMetricsSummaryHandler:
    @pytest.mark.asyncio
    async def test_renders_parsed_gauges(self):
        async def fake_call_node_api(method, path, data=None,
                                      raw_text=False):
            assert method == "GET"
            assert path == "/metrics"
            return (
                "# HELP prsm_pending_escrow_count Pending escrow count\n"
                "# TYPE prsm_pending_escrow_count gauge\n"
                "prsm_pending_escrow_count 3\n"
                "# HELP prsm_total_locked_ftns ...\n"
                "# TYPE prsm_total_locked_ftns gauge\n"
                "prsm_total_locked_ftns 12.5\n"
                "# HELP prsm_job_history_size ...\n"
                "# TYPE prsm_job_history_size gauge\n"
                "prsm_job_history_size 42\n"
                "# HELP prsm_node_up ...\n"
                "# TYPE prsm_node_up gauge\n"
                "prsm_node_up 1\n"
            )
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_metrics_summary({})
        # Each gauge surfaced.
        assert "pending_escrow_count" in result
        assert "3" in result
        assert "total_locked_ftns" in result
        assert "12.5" in result
        assert "job_history_size" in result
        assert "42" in result
        assert "node_up" in result

    @pytest.mark.asyncio
    async def test_skips_help_and_type_lines(self):
        """Output should not include the Prometheus # HELP /
        # TYPE lines — those are scraper bookkeeping, not
        operator-relevant info."""
        async def fake_call_node_api(method, path, data=None,
                                      raw_text=False):
            return (
                "# HELP prsm_node_up ...\n"
                "# TYPE prsm_node_up gauge\n"
                "prsm_node_up 1\n"
            )
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_metrics_summary({})
        assert "# HELP" not in result
        assert "# TYPE" not in result


class TestMetricsSummaryErrors:
    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None, raw_text=False):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_metrics_summary({})
        assert "cannot reach" in result.lower()
