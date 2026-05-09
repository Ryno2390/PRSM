"""prsm_escrow_lookup MCP tool handler.

Wraps GET /wallet/escrows/{escrow_id} for AI-side-panel detail
lookup of a specific escrow.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_escrow_lookup,
)


class TestRegistration:
    def test_handler_registered(self):
        assert "prsm_escrow_lookup" in TOOL_HANDLERS

    def test_required_arg(self):
        tool = next(
            t for t in TOOLS if t.name == "prsm_escrow_lookup"
        )
        assert "escrow_id" in tool.inputSchema["required"]


class TestHandler:
    @pytest.mark.asyncio
    async def test_renders_full_detail(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "GET"
            assert "/wallet/escrows/esc-abc" in path
            return {
                "escrow_id": "esc-abc",
                "job_id": "forge-aaa",
                "requester_id": "0x1111",
                "amount_ftns": 12.5,
                "status": "pending",
                "provider_winner": None,
                "tx_lock": "0xtx-lock",
                "tx_release": None,
                "created_at": 1700000000,
                "completed_at": None,
                "metadata": {},
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_escrow_lookup({
                "escrow_id": "esc-abc",
            })
        assert "esc-abc" in result
        assert "forge-aaa" in result
        assert "12.5" in result or "12.500000" in result
        assert "PENDING" in result.upper()
        assert "0xtx-lock" in result

    @pytest.mark.asyncio
    async def test_renders_released_lifecycle(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "escrow_id": "esc-rel",
                "job_id": "forge-x",
                "requester_id": "0x1111",
                "amount_ftns": 5.0,
                "status": "released",
                "provider_winner": "0xprovider",
                "tx_lock": "0xtx-lock",
                "tx_release": "0xtx-release",
                "created_at": 1700000000,
                "completed_at": 1700000100,
                "metadata": {},
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_escrow_lookup({
                "escrow_id": "esc-rel",
            })
        assert "RELEASED" in result.upper()
        assert "0xprovider" in result
        assert "0xtx-release" in result


class TestErrors:
    @pytest.mark.asyncio
    async def test_missing_escrow_id(self):
        result = await handle_prsm_escrow_lookup({})
        assert "missing" in result.lower()

    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_escrow_lookup({
                "escrow_id": "x",
            })
        assert "cannot reach" in result.lower()

    @pytest.mark.asyncio
    async def test_404_renders_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {"detail": "No escrow record for escrow_id='missing'"}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_escrow_lookup({
                "escrow_id": "missing",
            })
        assert "not found" in result.lower()
