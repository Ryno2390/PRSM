"""Sprint 279 — prsm_pool_quote MCP tool.

LLM-facing surface for Aerodrome pool inspection. action
selector: state | quote. Read-only — no commission gate; works
end-to-end the moment BASE_RPC_URL + AERODROME_USDC_FTNS_POOL_
ADDRESS are configured (post Vision-gantt-2026-06-15 seeding
ceremony).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_pool_quote,
)


def test_tool_registered():
    assert "prsm_pool_quote" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        r = await handle_prsm_pool_quote({})
        assert "action" in r.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        r = await handle_prsm_pool_quote({"action": "explode"})
        assert "must be" in r.lower()


class TestState:
    @pytest.mark.asyncio
    async def test_state_not_configured(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "status": "NOT_CONFIGURED",
                "note": "seeding ceremony pending",
            }),
        ) as mock_call:
            r = await handle_prsm_pool_quote({"action": "state"})
        args = mock_call.await_args[0]
        assert args[0] == "GET"
        assert args[1] == "/wallet/pool/state"
        assert "NOT_CONFIGURED" in r
        assert "seeding" in r.lower()

    @pytest.mark.asyncio
    async def test_state_ok(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "status": "OK",
                "pool_address": "0xpool",
                "token0": "0xusdc",
                "token1": "0xftns",
                "reserve0": 1_000_000,
                "reserve1": 2_000_000,
                "stable": False,
                "fee_bps": 30,
                "total_supply": 1_414_213,
                "block_number": 42,
            }),
        ):
            r = await handle_prsm_pool_quote({"action": "state"})
        assert "0xpool" in r
        assert "1000000" in r or "1,000,000" in r
        assert "2000000" in r or "2,000,000" in r
        assert "30" in r  # fee_bps
        assert "42" in r  # block_number

    @pytest.mark.asyncio
    async def test_state_pool_unavailable(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "status": "POOL_UNAVAILABLE",
                "pool_address": "0xpool",
                "note": "RPC down",
            }),
        ):
            r = await handle_prsm_pool_quote({"action": "state"})
        assert "POOL_UNAVAILABLE" in r
        assert "RPC down" in r


class TestQuote:
    @pytest.mark.asyncio
    async def test_quote_requires_amount_in(self):
        r = await handle_prsm_pool_quote({
            "action": "quote",
            "token_in": "0xusdc",
        })
        assert "amount_in" in r

    @pytest.mark.asyncio
    async def test_quote_requires_token_in(self):
        r = await handle_prsm_pool_quote({
            "action": "quote",
            "amount_in": 1000,
        })
        assert "token_in" in r

    @pytest.mark.asyncio
    async def test_quote_happy_path(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "status": "OK",
                "amount_in": 1000,
                "token_in": "0xusdc",
                "token_out": "0xftns",
                "amount_out": 996,
                "price_impact_bps": 100,
                "route": "aerodrome",
                "fee_bps": 30,
            }),
        ) as mock_call:
            r = await handle_prsm_pool_quote({
                "action": "quote",
                "amount_in": 1000,
                "token_in": "0xusdc",
            })
        path = mock_call.await_args[0][1]
        assert "/wallet/pool/quote" in path
        assert "amount_in=1000" in path
        assert "token_in=0xusdc" in path
        assert "1000" in r
        assert "996" in r
        assert "aerodrome" in r

    @pytest.mark.asyncio
    async def test_quote_not_configured(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "status": "NOT_CONFIGURED",
                "amount_in": 1000,
                "token_in": "0xusdc",
            }),
        ):
            r = await handle_prsm_pool_quote({
                "action": "quote",
                "amount_in": 1000,
                "token_in": "0xusdc",
            })
        assert "NOT_CONFIGURED" in r

    @pytest.mark.asyncio
    async def test_quote_unknown_token_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": (
                    "token_in='0xeth' not in pool "
                    "(token0='0xusdc', token1='0xftns')"
                ),
            }),
        ):
            r = await handle_prsm_pool_quote({
                "action": "quote",
                "amount_in": 1000,
                "token_in": "0xeth",
            })
        assert "not in pool" in r.lower()

    @pytest.mark.asyncio
    async def test_quote_stable_unsupported(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": (
                    "Stable-pool swap math not implemented in v1"
                ),
            }),
        ):
            r = await handle_prsm_pool_quote({
                "action": "quote",
                "amount_in": 1000,
                "token_in": "0xusdc",
            })
        assert "stable" in r.lower()


class TestServiceUnavailable:
    @pytest.mark.asyncio
    async def test_503_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "Aerodrome client not initialized.",
            }),
        ):
            r = await handle_prsm_pool_quote({"action": "state"})
        assert "not wired" in r.lower() or "not initialized" in r.lower()
