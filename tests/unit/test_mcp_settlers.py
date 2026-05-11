"""Sprint 220 — prsm_settlers + prsm_settler_batches MCP tools.

GET /settler/list/active + GET /settler/{id} + GET
/settler/batch/pending had no MCP coverage.

prsm_settlers: list active settlers OR lookup specific by id.
prsm_settler_batches: list pending multi-sig batches.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS,
    handle_prsm_settlers,
    handle_prsm_settler_batches,
)


class TestRegistration:
    def test_settlers_tool_in_handlers(self):
        assert "prsm_settlers" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_settlers"] is handle_prsm_settlers

    def test_settler_batches_tool_in_handlers(self):
        assert "prsm_settler_batches" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_settler_batches"] is handle_prsm_settler_batches


class TestSettlersList:
    @pytest.mark.asyncio
    async def test_list_active(self):
        mock_resp = [
            {
                "settler_id": "s-1",
                "address": "0xabc",
                "bond_amount": 10000.0,
                "total_settled": 42,
            },
            {
                "settler_id": "s-2",
                "address": "0xdef",
                "bond_amount": 15000.0,
                "total_settled": 17,
            },
        ]
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_settlers({})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/settler/list/active"
        assert "s-1" in result
        assert "s-2" in result
        assert "10000" in result

    @pytest.mark.asyncio
    async def test_empty_list_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=[]),
        ):
            result = await handle_prsm_settlers({})
        assert "no active settlers" in result.lower() or "0" in result


class TestSettlerLookup:
    @pytest.mark.asyncio
    async def test_lookup_by_id(self):
        mock_resp = {
            "settler_id": "s-1",
            "address": "0xabc",
            "bond_amount": 10000.0,
            "status": "active",
            "can_settle": True,
            "total_settled": 42,
            "slashed_amount": 0.0,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_settlers({"settler_id": "s-1"})
        args, _ = mock_call.await_args
        assert args[1] == "/settler/s-1"
        assert "s-1" in result
        assert "active" in result.lower()
        assert "can_settle" in result.lower()

    @pytest.mark.asyncio
    async def test_lookup_not_found(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"detail": "Settler x not found"}),
        ):
            result = await handle_prsm_settlers({"settler_id": "x"})
        assert "not found" in result.lower()


class TestSettlerBatches:
    @pytest.mark.asyncio
    async def test_list_pending(self):
        mock_resp = [
            {
                "batch_id": "b-1",
                "batch_hash": "0xhash",
                "transfer_count": 5,
                "total_amount": 250.0,
                "signature_count": 2,
                "threshold": 3,
                "approved": False,
            },
        ]
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_settler_batches({})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/settler/batch/pending"
        assert "b-1" in result
        assert "2" in result and "3" in result

    @pytest.mark.asyncio
    async def test_no_pending_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=[]),
        ):
            result = await handle_prsm_settler_batches({})
        assert "no pending" in result.lower() or "0" in result
