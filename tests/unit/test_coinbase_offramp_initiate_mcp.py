"""coinbase_offramp_initiate MCP tool handler.

V1 scope: pre-flight transaction-summary composer matching Vision
§13 Phase 5 step 2 ('Gemini presents an Artifact in your side
panel'). Companion to prsm_balance_check (the read side); this is
the write side of the cash-out flow, but actual execution gates
on CDP commission per Vision gantt 2026-06-15. Today's tool
returns a PENDING_COMMISSION envelope with the full transaction
summary the AI displays.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    handle_coinbase_offramp_initiate,
    TOOL_HANDLERS,
    TOOLS,
)


# ──────────────────────────────────────────────────────────────────────
# Tool registration
# ──────────────────────────────────────────────────────────────────────


class TestToolRegistration:
    def test_handler_registered(self):
        assert "coinbase_offramp_initiate" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "coinbase_offramp_initiate" in names

    def test_schema_requires_usd_amount(self):
        tool = next(t for t in TOOLS if t.name == "coinbase_offramp_initiate")
        assert "usd_amount" in tool.inputSchema["properties"]
        assert "usd_amount" in tool.inputSchema.get("required", [])

    def test_schema_optional_bank_account_alias(self):
        tool = next(t for t in TOOLS if t.name == "coinbase_offramp_initiate")
        assert "bank_account_alias" in tool.inputSchema["properties"]
        assert "bank_account_alias" not in tool.inputSchema.get("required", [])


# ──────────────────────────────────────────────────────────────────────
# Handler — happy path
# ──────────────────────────────────────────────────────────────────────


class TestOfframpInitiateHandler:
    @pytest.mark.asyncio
    async def test_returns_pre_flight_summary(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "POST"
            assert path == "/wallet/offramp/quote"
            return {
                "requested_usd": 500.0,
                "source_address": "0x" + "11" * 20,
                "source_balance_ftns": 4200.0,
                "source_balance_usd": 4200.0,
                "quote": {
                    "ftns_to_swap": 500.0,
                    "usdc_received": 500.0,
                    "usd_settled": 500.0,
                    "swap_route": "aerodrome",
                    "offramp_route": "coinbase-cdp",
                    "bank_account_alias": "primary",
                },
                "usd_rate": 1.0,
                "status": "PENDING_COMMISSION",
                "commission_gate_note": (
                    "Coinbase CDP commission gates on Aerodrome pool seeding."
                ),
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_coinbase_offramp_initiate(
                {"usd_amount": 500.0},
            )
        # Output text must contain the load-bearing summary fields:
        assert "$500" in result or "500.00" in result
        assert "FTNS" in result
        assert "PENDING_COMMISSION" in result or "pending" in result.lower()

    @pytest.mark.asyncio
    async def test_passes_usd_amount_and_bank_alias(self):
        captured = {}

        async def fake_call_node_api(method, path, data=None):
            captured["method"] = method
            captured["path"] = path
            captured["data"] = data
            return {
                "requested_usd": 100.0,
                "source_address": "0x" + "22" * 20,
                "source_balance_ftns": 1000.0,
                "source_balance_usd": 1000.0,
                "quote": {
                    "ftns_to_swap": 100.0,
                    "usdc_received": 100.0,
                    "usd_settled": 100.0,
                    "swap_route": "aerodrome",
                    "offramp_route": "coinbase-cdp",
                    "bank_account_alias": "savings",
                },
                "usd_rate": 1.0,
                "status": "PENDING_COMMISSION",
                "commission_gate_note": "...",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            await handle_coinbase_offramp_initiate({
                "usd_amount": 100.0,
                "bank_account_alias": "savings",
            })
        assert captured["method"] == "POST"
        assert captured["data"]["usd_amount"] == 100.0
        assert captured["data"]["bank_account_alias"] == "savings"

    @pytest.mark.asyncio
    async def test_renders_swap_and_offramp_routes(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "requested_usd": 50.0,
                "source_address": "0x" + "11" * 20,
                "source_balance_ftns": 100.0,
                "source_balance_usd": 100.0,
                "quote": {
                    "ftns_to_swap": 50.0,
                    "usdc_received": 50.0,
                    "usd_settled": 50.0,
                    "swap_route": "aerodrome",
                    "offramp_route": "coinbase-cdp",
                    "bank_account_alias": "primary",
                },
                "usd_rate": 1.0,
                "status": "PENDING_COMMISSION",
                "commission_gate_note": "...",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_coinbase_offramp_initiate(
                {"usd_amount": 50.0},
            )
        # Both routes must appear so users know what infrastructure
        # the transaction will route through.
        assert "aerodrome" in result.lower()
        assert "coinbase" in result.lower() or "cdp" in result.lower()


# ──────────────────────────────────────────────────────────────────────
# Handler — error paths
# ──────────────────────────────────────────────────────────────────────


class TestOfframpInitiateHandlerErrors:
    @pytest.mark.asyncio
    async def test_missing_usd_amount_returns_user_error(self):
        result = await handle_coinbase_offramp_initiate({})
        assert "usd_amount" in result.lower() or "required" in result.lower()

    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_coinbase_offramp_initiate(
                {"usd_amount": 100.0},
            )
        assert "cannot reach" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_insufficient_balance_surfaces_422(self):
        # Endpoint returns 422 detail envelope for insufficient balance.
        async def fake_call_node_api(method, path, data=None):
            return {"detail": "Insufficient balance: requested $500.00, available $10.00"}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_coinbase_offramp_initiate(
                {"usd_amount": 500.0},
            )
        assert "insufficient" in result.lower() or "balance" in result.lower()

    @pytest.mark.asyncio
    async def test_503_ftns_ledger_missing(self):
        async def fake_call_node_api(method, path, data=None):
            return {"detail": "On-chain ftns_ledger not initialized"}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_coinbase_offramp_initiate(
                {"usd_amount": 100.0},
            )
        assert (
            "not initialized" in result.lower()
            or "not configured" in result.lower()
        )
