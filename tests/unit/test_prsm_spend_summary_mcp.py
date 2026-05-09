"""prsm_spend_summary MCP tool handler.

Wraps GET /wallet/spend.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_spend_summary,
)


class TestToolRegistration:
    def test_handler_registered(self):
        assert "prsm_spend_summary" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_spend_summary" in names

    def test_days_in_schema(self):
        tool = next(t for t in TOOLS if t.name == "prsm_spend_summary")
        assert "days" in tool.inputSchema["properties"]


class TestSpendSummaryHandler:
    @pytest.mark.asyncio
    async def test_renders_spend_summary(self):
        async def fake_call_node_api(method, path, data=None):
            assert "/wallet/spend" in path
            return {
                "address": "0x" + "11" * 20,
                "days": 30,
                "total_spent_ftns": 12.5,
                "escrows_count": 4,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_spend_summary({})
        assert "12.5" in result or "12.500000" in result
        assert "30" in result  # days
        assert "4" in result  # count

    @pytest.mark.asyncio
    async def test_passes_days_param(self):
        captured = {}

        async def fake_call_node_api(method, path, data=None):
            captured["path"] = path
            return {
                "address": "0x", "days": 7,
                "total_spent_ftns": 5.0, "escrows_count": 2,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            await handle_prsm_spend_summary({"days": 7})
        assert "days=7" in captured["path"]

    @pytest.mark.asyncio
    async def test_zero_count_avg_is_zero(self):
        """No spend → avg should not divide by zero."""
        async def fake_call_node_api(method, path, data=None):
            return {
                "address": "0x", "days": 30,
                "total_spent_ftns": 0.0, "escrows_count": 0,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_spend_summary({})
        # No exception, sensible output.
        assert "0.000000 FTNS" in result


class TestSpendSummaryErrors:
    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_spend_summary({})
        assert "cannot reach" in result.lower()

    @pytest.mark.asyncio
    async def test_503_returns_failed_message(self):
        async def fake_call_node_api(method, path, data=None):
            return {"detail": "PaymentEscrow not initialized."}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_spend_summary({})
        assert "failed" in result.lower() or "not initialized" in result.lower()
