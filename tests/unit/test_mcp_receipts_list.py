"""Sprint 250 — prsm_receipts_list MCP wrapper."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_receipts_list,
)


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_receipts_list" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_bad_limit_rejected(self):
        result = await handle_prsm_receipts_list({"limit": 5000})
        assert "limit" in result.lower()

    @pytest.mark.asyncio
    async def test_negative_offset_rejected(self):
        result = await handle_prsm_receipts_list({"offset": -1})
        assert "offset" in result.lower()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_renders_listing(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "receipts": [
                    {
                        "job_id": "j1",
                        "model_id": "m1",
                        "cost_ftns": "0.10",
                        "settler_node_id": "settler-7",
                    },
                ],
                "total": 1, "offset": 0, "limit": 20,
            }),
        ) as mock_call:
            result = await handle_prsm_receipts_list({})
        args, _ = mock_call.await_args
        assert "/compute/receipts" in args[1]
        assert "j1" in result
        assert "m1" in result


class TestModelFilter:
    @pytest.mark.asyncio
    async def test_filter_passed_through(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "receipts": [], "total": 0,
                "offset": 0, "limit": 20,
            }),
        ) as mock_call:
            await handle_prsm_receipts_list({"model_id": "m1"})
        args, _ = mock_call.await_args
        assert "model_id=m1" in args[1]


class TestEmptyState:
    @pytest.mark.asyncio
    async def test_empty_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "receipts": [], "total": 0,
                "offset": 0, "limit": 20,
            }),
        ):
            result = await handle_prsm_receipts_list({})
        assert "no stored receipts" in result.lower() or "0" in result


class TestUnwired:
    @pytest.mark.asyncio
    async def test_503_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "Receipt store not initialized.",
            }),
        ):
            result = await handle_prsm_receipts_list({})
        assert "not wired" in result.lower()
        assert "PRSM_RECEIPT_STORE_DIR" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_receipts_list({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
