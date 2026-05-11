"""Sprint 209 — prsm_status_stream MCP tool.

Pre-fix: /compute/status/{job_id}/stream endpoint shipped 2026-05-09
but no MCP wrapper exists. End-users submitting jobs via
prsm_forge_submit had to poll prsm_agent_status by hand to see
progress. A stale reference at mcp_server.py:2900 ("Use
prsm_jobs_list / prsm_status_stream to observe progress") advertised
a tool that didn't exist.

prsm_status_stream:
  - Consumes the SSE stream and returns when terminal event arrives
    OR max_wait_sec elapses (default 60, capped at 600).
  - Renders unique status-snapshot transitions as a trajectory log.
  - Validates job_id (required, non-empty).
  - Optionally emits progress for each transition when emit_progress
    is provided.
  - Registered in TOOL_HANDLERS dispatch table.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_status_stream


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_status_stream" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_status_stream"] is handle_prsm_status_stream


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_job_id_rejected(self):
        result = await handle_prsm_status_stream({})
        assert "job_id" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_job_id_rejected(self):
        result = await handle_prsm_status_stream({"job_id": "  "})
        assert "job_id" in result.lower()

    @pytest.mark.asyncio
    async def test_max_wait_capped_at_600(self):
        """max_wait_sec >600 clamps to 600 (no error, just clamp)."""
        with patch(
            "prsm.mcp_server._consume_status_stream",
            new=AsyncMock(return_value=(
                [{"status": "completed"}], "terminal", "completed",
            )),
        ) as mock_consume:
            await handle_prsm_status_stream({
                "job_id": "j1", "max_wait_sec": 10_000,
            })
        # Called with capped value, not 10_000.
        _, kwargs = mock_consume.await_args
        assert kwargs["max_wait_sec"] <= 600


class TestTrajectoryRender:
    @pytest.mark.asyncio
    async def test_renders_status_transitions(self):
        snapshots = [
            {"status": "PENDING"},
            {"status": "IN_PROGRESS"},
            {"status": "COMPLETED"},
        ]
        with patch(
            "prsm.mcp_server._consume_status_stream",
            new=AsyncMock(return_value=(
                snapshots, "terminal", "completed",
            )),
        ):
            result = await handle_prsm_status_stream({"job_id": "j1"})
        assert "PENDING" in result
        assert "IN_PROGRESS" in result
        assert "COMPLETED" in result
        assert "j1" in result

    @pytest.mark.asyncio
    async def test_timeout_terminal_rendered(self):
        with patch(
            "prsm.mcp_server._consume_status_stream",
            new=AsyncMock(return_value=(
                [{"status": "IN_PROGRESS"}], "terminal", "timeout",
            )),
        ):
            result = await handle_prsm_status_stream({
                "job_id": "j1", "max_wait_sec": 5,
            })
        assert "timeout" in result.lower()


class TestEmitProgress:
    @pytest.mark.asyncio
    async def test_emits_progress_when_emitter_provided(self):
        emitter = AsyncMock()
        snapshots = [
            {"status": "PENDING"},
            {"status": "COMPLETED"},
        ]
        with patch(
            "prsm.mcp_server._consume_status_stream",
            new=AsyncMock(return_value=(
                snapshots, "terminal", "completed",
            )),
        ):
            await handle_prsm_status_stream(
                {"job_id": "j1"}, emit_progress=emitter,
            )
        # At least one progress emit per transition.
        assert emitter.await_count >= 1


class TestNetworkFailureHandling:
    @pytest.mark.asyncio
    async def test_network_error_returns_friendly_string(self):
        with patch(
            "prsm.mcp_server._consume_status_stream",
            new=AsyncMock(side_effect=RuntimeError("connection refused")),
        ):
            result = await handle_prsm_status_stream({"job_id": "j1"})
        # Returns a friendly message, doesn't raise.
        assert isinstance(result, str)
        assert "connection refused" in result.lower() or "failed" in result.lower()
