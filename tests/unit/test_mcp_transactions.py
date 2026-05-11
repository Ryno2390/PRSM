"""Sprint 212 — prsm_transactions MCP tool.

GET /transactions returns a node's FTNS transaction history
(credit/debit + tx_type + from/to wallet + amount + description
+ timestamp), bounded by limit param [1, 200]. No MCP wrapper
existed, so end-users tracking their FTNS flows via the MCP
side-panel had to curl manually.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_transactions


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_transactions" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_transactions"] is handle_prsm_transactions


class TestRender:
    @pytest.mark.asyncio
    async def test_renders_transactions(self):
        mock_resp = {
            "transactions": [
                {
                    "tx_id": "tx-1",
                    "type": "WELCOME_GRANT",
                    "from": "faucet",
                    "to": "wallet-a",
                    "amount": 100.0,
                    "description": "Faucet grant: 100 FTNS",
                    "timestamp": 1715000000.0,
                },
                {
                    "tx_id": "tx-2",
                    "type": "TRANSFER",
                    "from": "wallet-a",
                    "to": "wallet-b",
                    "amount": 25.5,
                    "description": "Payment",
                    "timestamp": 1715000100.0,
                },
            ],
            "count": 2,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_transactions({})
        mock_call.assert_awaited_once()
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert "/transactions" in args[1]
        assert "tx-1" in result
        assert "tx-2" in result
        assert "WELCOME_GRANT" in result
        assert "100" in result
        assert "25.5" in result

    @pytest.mark.asyncio
    async def test_empty_history_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"transactions": [], "count": 0}),
        ):
            result = await handle_prsm_transactions({})
        assert "no transactions" in result.lower() or "0" in result

    @pytest.mark.asyncio
    async def test_limit_passed_to_query_string(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"transactions": [], "count": 0}),
        ) as mock_call:
            await handle_prsm_transactions({"limit": 10})
        args, _ = mock_call.await_args
        assert "limit=10" in args[1]


class TestValidation:
    @pytest.mark.asyncio
    async def test_limit_out_of_range_rejected_locally(self):
        """Client-side guard so we don't waste a round-trip on a
        provably-invalid limit. Server caps at [1, 200]."""
        result = await handle_prsm_transactions({"limit": 1000})
        assert "limit" in result.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_node_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_transactions({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
