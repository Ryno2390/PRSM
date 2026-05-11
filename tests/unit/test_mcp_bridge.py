"""Sprint 222 — prsm_bridge MCP tool.

POST /bridge/deposit + POST /bridge/withdraw had no MCP coverage.
Consolidated into a single tool with `direction` selector
(deposit|withdraw). Bridges FTNS between local balance and
external chain (Polygon default).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_bridge


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_bridge" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_direction_rejected(self):
        result = await handle_prsm_bridge({
            "amount": 1.0, "chain_address": "0xabc",
        })
        assert "direction" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_direction_rejected(self):
        result = await handle_prsm_bridge({
            "direction": "bogus", "amount": 1.0,
            "chain_address": "0xabc",
        })
        assert "must be" in result.lower()

    @pytest.mark.asyncio
    async def test_missing_amount_rejected(self):
        result = await handle_prsm_bridge({
            "direction": "deposit", "chain_address": "0xabc",
        })
        assert "amount" in result.lower()

    @pytest.mark.asyncio
    async def test_missing_chain_address_rejected(self):
        result = await handle_prsm_bridge({
            "direction": "deposit", "amount": 1.0,
        })
        assert "chain_address" in result.lower()

    @pytest.mark.asyncio
    async def test_inf_amount_rejected_locally(self):
        result = await handle_prsm_bridge({
            "direction": "deposit",
            "amount": float("inf"),
            "chain_address": "0xabc",
        })
        assert "finite" in result.lower() or "amount" in result.lower()


class TestRouting:
    @pytest.mark.asyncio
    async def test_deposit_routes_correctly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "success": True,
                "transaction": {
                    "transaction_id": "tx-1",
                    "amount": "1.0",
                    "status": "pending",
                },
            }),
        ) as mock_call:
            await handle_prsm_bridge({
                "direction": "deposit",
                "amount": 1.0,
                "chain_address": "0xabc",
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1] == "/bridge/deposit"
        body = args[2] if len(args) > 2 else {}
        assert body.get("amount") == 1.0
        assert body.get("chain_address") == "0xabc"
        assert body.get("destination_chain") == 137  # default polygon

    @pytest.mark.asyncio
    async def test_withdraw_routes_correctly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "success": True,
                "transaction": {
                    "transaction_id": "tx-2",
                    "amount": "5.0",
                    "status": "pending",
                },
            }),
        ) as mock_call:
            await handle_prsm_bridge({
                "direction": "withdraw",
                "amount": 5.0,
                "chain_address": "0xdef",
                "source_chain": 8453,
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1] == "/bridge/withdraw"
        body = args[2] if len(args) > 2 else {}
        assert body.get("source_chain") == 8453


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_bridge({
                "direction": "deposit",
                "amount": 1.0,
                "chain_address": "0xabc",
            })
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
