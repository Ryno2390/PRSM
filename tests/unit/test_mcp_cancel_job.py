"""Sprint 210 — prsm_cancel_job MCP tool.

The POST /compute/cancel/{job_id} endpoint shipped 2026-05-09
(`compute-cancel-endpoint-merge-ready-20260509`) but no MCP tool
wraps it. End-users running prsm_forge_submit get a job_id back
and can stream status via prsm_status_stream (sprint 209) but
can't cancel via MCP — they'd have to curl manually.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_cancel_job


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_cancel_job" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_cancel_job"] is handle_prsm_cancel_job


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_job_id_rejected(self):
        result = await handle_prsm_cancel_job({})
        assert "job_id" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_job_id_rejected(self):
        result = await handle_prsm_cancel_job({"job_id": "  "})
        assert "job_id" in result.lower()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_cancellation_success(self):
        mock_resp = {
            "job_id": "j1",
            "status": "cancelled",
            "history_marked": True,
            "escrow_refunded": True,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_cancel_job({"job_id": "j1"})
        mock_call.assert_awaited_once()
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert "/compute/cancel/j1" in args[1]
        assert "j1" in result
        assert "cancelled" in result.lower()


class TestNotFoundPath:
    @pytest.mark.asyncio
    async def test_404_rendered_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"detail": "job not found"}),
        ):
            result = await handle_prsm_cancel_job({"job_id": "missing"})
        assert "not found" in result.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_node_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_cancel_job({"job_id": "j1"})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
