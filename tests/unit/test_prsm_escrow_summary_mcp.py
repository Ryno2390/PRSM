"""prsm_escrow_summary MCP tool handler.

Wraps GET /wallet/escrows to surface operator's outstanding
compute-budget commitments via the AI side-panel.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_escrow_summary,
)


class TestToolRegistration:
    def test_handler_registered(self):
        assert "prsm_escrow_summary" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_escrow_summary" in names


class TestEscrowSummaryHandler:
    @pytest.mark.asyncio
    async def test_renders_escrow_table(self):
        async def fake_call_node_api(method, path, data=None):
            assert "/wallet/escrows" in path
            return {
                "address": "0x" + "11" * 20,
                "escrows": [
                    {"escrow_id": "esc-1", "job_id": "forge-aaa",
                     "amount_ftns": 5.0, "status": "pending",
                     "provider_winner": None, "tx_lock": None,
                     "tx_release": None, "created_at": 0,
                     "completed_at": None},
                    {"escrow_id": "esc-2", "job_id": "forge-bbb",
                     "amount_ftns": 3.5, "status": "pending",
                     "provider_winner": None, "tx_lock": None,
                     "tx_release": None, "created_at": 0,
                     "completed_at": None},
                ],
                "total": 2,
                "total_locked_ftns": 8.5,
                "include_terminal": False,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_escrow_summary({})
        assert "forge-aaa" in result
        assert "forge-bbb" in result
        assert "8.5" in result or "8.500000" in result  # total locked

    @pytest.mark.asyncio
    async def test_empty_renders_friendly_message(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "address": "0x" + "11" * 20,
                "escrows": [],
                "total": 0,
                "total_locked_ftns": 0.0,
                "include_terminal": False,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_escrow_summary({})
        assert "No active escrows" in result

    @pytest.mark.asyncio
    async def test_passes_include_terminal_flag(self):
        captured = {}

        async def fake_call_node_api(method, path, data=None):
            captured["path"] = path
            return {"address": "0x", "escrows": [], "total": 0,
                    "total_locked_ftns": 0.0, "include_terminal": True}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            await handle_prsm_escrow_summary({"include_terminal": True})
        assert "include_terminal=true" in captured["path"]


class TestEscrowSummaryErrors:
    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_escrow_summary({})
        assert "cannot reach" in result.lower()

    @pytest.mark.asyncio
    async def test_503_escrow_not_configured(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "detail": "PaymentEscrow not initialized on this node.",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_escrow_summary({})
        assert "not configured" in result.lower() or \
            "not initialized" in result.lower()
