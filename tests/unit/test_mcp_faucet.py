"""Sprint 223 — prsm_faucet MCP tool.

POST /ftns/faucet had no MCP wrapper. Testnet operators trying to
grant their wallets initial FTNS had to curl. Endpoint is 100
FTNS max per request, 1000 FTNS max per wallet, disabled in
production via PRSM_FAUCET_ENABLED=0.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_faucet


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_faucet" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_negative_amount_rejected_locally(self):
        result = await handle_prsm_faucet({"amount": -10})
        assert "amount" in result.lower()

    @pytest.mark.asyncio
    async def test_inf_amount_rejected_locally(self):
        result = await handle_prsm_faucet({"amount": float("inf")})
        assert "amount" in result.lower() or "finite" in result.lower()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_default_amount(self):
        mock_resp = {
            "granted": 100.0,
            "new_balance": 100.0,
            "wallet_id": "wallet-a",
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_faucet({})
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1] == "/ftns/faucet"
        assert "100" in result

    @pytest.mark.asyncio
    async def test_explicit_amount_and_wallet(self):
        mock_resp = {
            "granted": 50.0,
            "new_balance": 150.0,
            "wallet_id": "wallet-b",
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            await handle_prsm_faucet({
                "amount": 50.0, "wallet_id": "wallet-b",
            })
        args, _ = mock_call.await_args
        body = args[2] if len(args) > 2 else {}
        assert body.get("amount") == 50.0
        assert body.get("wallet_id") == "wallet-b"


class TestDisabled:
    @pytest.mark.asyncio
    async def test_403_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "Faucet disabled in production",
            }),
        ):
            result = await handle_prsm_faucet({})
        assert "disabled" in result.lower()


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_429_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "Wallet already has 1000 FTNS (max 1000)",
            }),
        ):
            result = await handle_prsm_faucet({})
        assert "1000" in result or "max" in result.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_faucet({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
