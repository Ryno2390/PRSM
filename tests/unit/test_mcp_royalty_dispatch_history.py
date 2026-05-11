"""Sprint 249 — prsm_royalty_dispatch_history MCP tool."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS,
    handle_prsm_royalty_dispatch_history,
)


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_royalty_dispatch_history" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_bad_limit_rejected_locally(self):
        result = await handle_prsm_royalty_dispatch_history({
            "limit": 5000,
        })
        assert "limit" in result.lower()

    @pytest.mark.asyncio
    async def test_negative_offset_rejected_locally(self):
        result = await handle_prsm_royalty_dispatch_history({
            "offset": -1,
        })
        assert "offset" in result.lower()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_renders_entries(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "entries": [
                    {
                        "timestamp": 1715000000.0,
                        "job_id": "job-1",
                        "cid": "cid-a",
                        "status": "sent",
                        "tx_hash": "0xdeadbeef",
                        "gross_wei": 10**15,
                    },
                    {
                        "timestamp": 1715000010.0,
                        "job_id": "job-2",
                        "cid": "cid-b",
                        "status": "failed",
                        "tx_hash": None,
                        "gross_wei": 10**15,
                        "error": "rpc down",
                    },
                ],
                "total": 2,
                "offset": 0,
                "limit": 20,
            }),
        ) as mock_call:
            result = await handle_prsm_royalty_dispatch_history({})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert "/admin/royalty-dispatch-history" in args[1]
        assert "job-1" in result
        assert "job-2" in result
        assert "sent" in result
        assert "failed" in result
        assert "rpc down" in result


class TestStatusFilter:
    @pytest.mark.asyncio
    async def test_filter_passed_to_query_string(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "entries": [], "total": 0, "offset": 0, "limit": 20,
            }),
        ) as mock_call:
            await handle_prsm_royalty_dispatch_history({
                "status": "failed",
            })
        args, _ = mock_call.await_args
        assert "status=failed" in args[1]


class TestEmptyState:
    @pytest.mark.asyncio
    async def test_no_entries_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "entries": [], "total": 0,
                "offset": 0, "limit": 20,
            }),
        ):
            result = await handle_prsm_royalty_dispatch_history({})
        assert (
            "no" in result.lower()
            and "dispatch" in result.lower()
        )
        assert "PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED" in result


class TestRingUnwired:
    @pytest.mark.asyncio
    async def test_friendly_503(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "Royalty dispatch ring not initialized.",
            }),
        ):
            result = await handle_prsm_royalty_dispatch_history({})
        assert "not wired" in result.lower()
        assert "PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_royalty_dispatch_history({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
