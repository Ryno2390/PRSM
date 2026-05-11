"""Sprint 225 — prsm_local_balance MCP tool.

GET /balance returns the node's local-ledger FTNS balance plus
the 20 most-recent transactions. Distinct from prsm_balance_check
(which hits /balance/onchain — aggregates on-chain + claimable
royalties + escrowed). Useful when you just want the local-ledger
view without the aggregate round-trip.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_local_balance


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_local_balance" in TOOL_HANDLERS


class TestRender:
    @pytest.mark.asyncio
    async def test_renders_balance_and_history(self):
        mock_resp = {
            "wallet_id": "wallet-a",
            "balance": 123.45,
            "recent_transactions": [
                {
                    "tx_id": "tx-1",
                    "type": "WELCOME_GRANT",
                    "from": "faucet",
                    "to": "wallet-a",
                    "amount": 100.0,
                    "description": "Faucet grant",
                    "timestamp": 1715000000.0,
                },
            ],
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_local_balance({})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/balance"
        assert "123.45" in result
        assert "wallet-a" in result
        assert "tx-1" in result

    @pytest.mark.asyncio
    async def test_empty_history_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "wallet_id": "wallet-a",
                "balance": 0,
                "recent_transactions": [],
            }),
        ):
            result = await handle_prsm_local_balance({})
        assert "0" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_local_balance({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
