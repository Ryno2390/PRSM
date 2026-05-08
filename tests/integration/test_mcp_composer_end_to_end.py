"""End-to-end integration test for the MCP composer surface
(prsm_balance_check + coinbase_offramp_initiate).

Where the unit tests at tests/unit/test_*_mcp.py mock
``_call_node_api`` and exercise the handler in isolation, this
suite exercises the full path:

  Handler → real httpx call → live FastAPI TestClient →
  endpoint → stub OnChainFTNSLedger → response → handler
  formats output

Catches serialization-boundary bugs that mocked handlers can
miss (e.g., Pydantic body shape mismatch, response field type
divergence between unit-stub and real endpoint).
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.mcp_server import (
    handle_coinbase_offramp_initiate,
    handle_prsm_balance_check,
)
from prsm.node.api import create_api_app


# ──────────────────────────────────────────────────────────────────────
# Stub Node + harness bridging MCP handler → live FastAPI TestClient
# ──────────────────────────────────────────────────────────────────────


def _node_with_ftns(*, balance_ftns: float = 100.0,
                    address: str = "0x" + "11" * 20):
    """Build a stub Node whose ftns_ledger.get_balance returns the
    configured FTNS amount synchronously via AsyncMock."""
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = "test-node"
    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = address
    ftns_ledger._decimals = 18
    ftns_ledger.get_balance = AsyncMock(return_value=balance_ftns)
    node.ftns_ledger = ftns_ledger
    return node


def _bridge_call_node_api(test_client: TestClient):
    """Build an async function with `_call_node_api`'s shape that
    routes through the in-process FastAPI TestClient instead of
    aiohttp + a real network call.

    This is what makes "end-to-end" possible without spinning up an
    actual node process — the MCP handler invokes _call_node_api as
    if it were the real one, but the request flows through the
    TestClient ASGI dispatcher and hits the real endpoint code.
    """
    async def fake_call_node_api(method, path, data=None):
        if method == "GET":
            response = test_client.get(path)
        else:
            response = test_client.post(path, json=data or {})
        return response.json()
    return fake_call_node_api


# ──────────────────────────────────────────────────────────────────────
# prsm_balance_check end-to-end
# ──────────────────────────────────────────────────────────────────────


class TestBalanceCheckEndToEnd:
    @pytest.mark.asyncio
    async def test_full_path_happy(self):
        node = _node_with_ftns(balance_ftns=42.5)
        client = TestClient(create_api_app(node, enable_security=False))

        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.5"}), patch(
            "prsm.mcp_server._call_node_api",
            side_effect=_bridge_call_node_api(client),
        ):
            output = await handle_prsm_balance_check({})

        # Output reflects the live endpoint computation, not a unit
        # stub's pre-canned shape.
        assert "42.500000 FTNS" in output
        # USD: 42.5 * 1.5 = 63.75
        assert "$63.75" in output
        assert "1.5 USD/FTNS" in output
        assert "onchain" in output.lower()

    @pytest.mark.asyncio
    async def test_full_path_503_when_ftns_ledger_missing(self):
        node = MagicMock()
        node.identity = MagicMock()
        node.identity.node_id = "test-node"
        node.ftns_ledger = None  # explicit; not auto-magic'd
        client = TestClient(create_api_app(node, enable_security=False))

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=_bridge_call_node_api(client),
        ):
            output = await handle_prsm_balance_check({})

        # Endpoint returns 503 + detail envelope; handler formats it.
        assert "not configured" in output.lower() or "not initialized" in output.lower()
        assert "PRSM_ONCHAIN_FTNS" in output

    @pytest.mark.asyncio
    async def test_full_path_with_custom_address(self):
        node = _node_with_ftns(balance_ftns=10.0)
        client = TestClient(create_api_app(node, enable_security=False))
        target = "0x" + "ab" * 20

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=_bridge_call_node_api(client),
        ):
            output = await handle_prsm_balance_check({"address": target})

        # Live endpoint calls ftns_ledger.get_balance with the override.
        node.ftns_ledger.get_balance.assert_awaited_with(target)
        # Address rendered (truncated form: first 10 chars).
        assert target[:10] in output


# ──────────────────────────────────────────────────────────────────────
# coinbase_offramp_initiate end-to-end
# ──────────────────────────────────────────────────────────────────────


class TestOfframpInitiateEndToEnd:
    @pytest.mark.asyncio
    async def test_full_path_pending_commission(self):
        node = _node_with_ftns(balance_ftns=4200.0)
        client = TestClient(create_api_app(node, enable_security=False))

        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}), patch(
            "prsm.mcp_server._call_node_api",
            side_effect=_bridge_call_node_api(client),
        ):
            output = await handle_coinbase_offramp_initiate({
                "usd_amount": 500.0,
            })

        # Critical: status field present + composer-only framing.
        assert "PENDING_COMMISSION" in output
        # Quote details present from the live endpoint.
        assert "500.00 USD" in output or "$500.00" in output
        assert "aerodrome" in output.lower()
        assert "coinbase-cdp" in output.lower()
        # Note: the gate explanation flows from endpoint to handler
        # to user.
        assert "gantt" in output.lower() or "2026-06-15" in output

    @pytest.mark.asyncio
    async def test_full_path_insufficient_balance_422(self):
        node = _node_with_ftns(balance_ftns=10.0)
        client = TestClient(create_api_app(node, enable_security=False))

        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}), patch(
            "prsm.mcp_server._call_node_api",
            side_effect=_bridge_call_node_api(client),
        ):
            output = await handle_coinbase_offramp_initiate({
                "usd_amount": 500.0,
            })

        # 422 detail flows through; handler formats with cross-ref to
        # prsm_balance_check.
        assert "insufficient" in output.lower() or "balance" in output.lower()
        assert "prsm_balance_check" in output.lower()

    @pytest.mark.asyncio
    async def test_full_path_400_on_negative_amount(self):
        node = _node_with_ftns(balance_ftns=100.0)
        client = TestClient(create_api_app(node, enable_security=False))

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=_bridge_call_node_api(client),
        ):
            output = await handle_coinbase_offramp_initiate({
                "usd_amount": -10.0,
            })

        # 400 detail envelope flows through; "must be positive" message.
        assert "positive" in output.lower() or "must be > 0" in output.lower()

    @pytest.mark.asyncio
    async def test_full_path_503_when_ftns_ledger_missing(self):
        node = MagicMock()
        node.identity = MagicMock()
        node.identity.node_id = "test-node"
        node.ftns_ledger = None
        client = TestClient(create_api_app(node, enable_security=False))

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=_bridge_call_node_api(client),
        ):
            output = await handle_coinbase_offramp_initiate({
                "usd_amount": 100.0,
            })

        assert "not configured" in output.lower() or "not initialized" in output.lower()


# ──────────────────────────────────────────────────────────────────────
# Composability check: read → quote (the documented flow)
# ──────────────────────────────────────────────────────────────────────


class TestComposability:
    @pytest.mark.asyncio
    async def test_balance_check_then_offramp_initiate(self):
        """The Vision §13 Phase 5 documented flow: read first, quote
        second. Both should produce coherent outputs against the same
        underlying balance state."""
        node = _node_with_ftns(balance_ftns=1000.0)
        client = TestClient(create_api_app(node, enable_security=False))

        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}), patch(
            "prsm.mcp_server._call_node_api",
            side_effect=_bridge_call_node_api(client),
        ):
            balance_output = await handle_prsm_balance_check({})
            quote_output = await handle_coinbase_offramp_initiate({
                "usd_amount": 200.0,
            })

        # Balance reads 1000 FTNS = $1000 USD at rate 1.0.
        assert "1000.000000 FTNS" in balance_output
        assert "$1,000.00" in balance_output

        # Quote reads same source, swaps 200 FTNS for $200 USD.
        assert "200.000000 FTNS" in quote_output
        assert "PENDING_COMMISSION" in quote_output

        # Both reference the same node-connected address.
        # Default _node_with_ftns address is 0x111...111; handler
        # truncates to first 10 chars (0x + 8 hex) + "…" + last 4.
        assert "0x" + "11" * 4 in balance_output  # 0x + 8 hex chars
        assert "0x" + "11" * 4 in quote_output
