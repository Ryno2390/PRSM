"""Sprint 224 — prsm_transfer MCP tool.

POST /ledger/transfer had no MCP wrapper. Anyone wanting to send
FTNS to another wallet via MCP had to curl. End-user-impactful.

Endpoint validates positive amount + finite (sprint 199). MCP
tool adds local-side validation for friendlier UX.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_transfer


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_transfer" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_to_wallet_rejected(self):
        result = await handle_prsm_transfer({"amount": 1.0})
        assert "to_wallet" in result.lower()

    @pytest.mark.asyncio
    async def test_missing_amount_rejected(self):
        result = await handle_prsm_transfer({"to_wallet": "wallet-b"})
        assert "amount" in result.lower()

    @pytest.mark.asyncio
    async def test_negative_amount_rejected(self):
        result = await handle_prsm_transfer({
            "to_wallet": "wallet-b", "amount": -1.0,
        })
        assert "amount" in result.lower()

    @pytest.mark.asyncio
    async def test_zero_amount_rejected(self):
        result = await handle_prsm_transfer({
            "to_wallet": "wallet-b", "amount": 0,
        })
        assert "amount" in result.lower()

    @pytest.mark.asyncio
    async def test_inf_amount_rejected_locally(self):
        result = await handle_prsm_transfer({
            "to_wallet": "wallet-b", "amount": float("inf"),
        })
        assert "amount" in result.lower() or "finite" in result.lower()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_transfer_succeeds(self):
        mock_resp = {
            "tx_id": "tx-1",
            "from": "wallet-a",
            "to": "wallet-b",
            "amount": 5.5,
            "timestamp": 1715000000.0,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_transfer({
                "to_wallet": "wallet-b", "amount": 5.5,
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert "to_wallet=wallet-b" in args[1]
        assert "amount=5.5" in args[1]
        assert "tx-1" in result
        assert "5.5" in result


class TestInsufficientBalance:
    @pytest.mark.asyncio
    async def test_400_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"detail": "Insufficient balance"}),
        ):
            result = await handle_prsm_transfer({
                "to_wallet": "wallet-b", "amount": 1e9,
            })
        assert "insufficient" in result.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_transfer({
                "to_wallet": "wallet-b", "amount": 1.0,
            })
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
