"""Sprint 265 — prsm_royalty_dispatch_summary MCP wrapper."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_royalty_dispatch_summary,
)


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_royalty_dispatch_summary" in TOOL_HANDLERS


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_renders_summary(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "total": 4,
                "status_counts": {
                    "sent": 2, "failed": 1,
                    "skipped_zero_amount": 1,
                },
                "total_sent_wei": 400,
                "by_allocation_mode": {
                    "uniform": 2, "rate_weighted": 2,
                },
                "earliest_ts": 100.0,
                "latest_ts": 400.0,
            }),
        ) as mock_call:
            result = await handle_prsm_royalty_dispatch_summary({})
        args, _ = mock_call.await_args
        assert args[1] == "/admin/royalty-dispatch-summary"
        assert "total=4" in result
        assert "total_sent_wei" in result
        assert "400" in result
        assert "sent" in result
        assert "uniform" in result


class TestEmpty:
    @pytest.mark.asyncio
    async def test_zero_total_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "total": 0,
                "status_counts": {},
                "total_sent_wei": 0,
                "by_allocation_mode": {},
                "earliest_ts": None,
                "latest_ts": None,
            }),
        ):
            result = await handle_prsm_royalty_dispatch_summary({})
        assert "no" in result.lower()
        assert "PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED" in result


class TestUnwired:
    @pytest.mark.asyncio
    async def test_503_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "Royalty dispatch ring not initialized.",
            }),
        ):
            result = await handle_prsm_royalty_dispatch_summary({})
        assert "not wired" in result.lower()
        assert "PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_royalty_dispatch_summary({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
