"""Sprint 278 — coinbase_onramp_initiate MCP tool.

Composer-only PENDING_COMMISSION tool that mirrors the
sprint-2026-05-08 coinbase_offramp_initiate, in the opposite
direction (USD → FTNS). No execute path. Per the WaaS
seamlessness principle, the user just supplies a USD amount
and an MPC-managed user_id; the LLM-facing artifact summarizes
what the transaction WILL look like once Coinbase CDP
commissions.

This tool intentionally does NOT have a sibling of the
R-2026-05-08-1 CI invariant test until that rule is extended
via a superseding council ratification. The IMPLEMENTATION
follows the composer-only contract so the future ratification
slots in without code change.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOLS, TOOL_HANDLERS, handle_coinbase_onramp_initiate,
)


def test_tool_registered():
    assert "coinbase_onramp_initiate" in TOOL_HANDLERS


def test_tool_in_catalog():
    matches = [t for t in TOOLS if t.name == "coinbase_onramp_initiate"]
    assert len(matches) == 1


def test_tool_description_signals_pending_commission():
    t = next(t for t in TOOLS if t.name == "coinbase_onramp_initiate")
    assert "PENDING_COMMISSION" in t.description


def test_tool_schema_no_execute_class_tokens():
    """Composer-only contract: no schema property suggesting an
    execute-class capability. Mirrors the R-2026-05-08-1
    schema-tier protection on the offramp side, defensively
    applied here so the future R-2026-05-11-* ratification
    finds no surprises."""
    t = next(t for t in TOOLS if t.name == "coinbase_onramp_initiate")
    forbidden_tokens = (
        "submit", "execute", "broadcast", "send_tx",
        "sign_and_send", "dry_run",
    )
    for prop_name in t.inputSchema.get("properties", {}):
        for token in forbidden_tokens:
            assert token not in prop_name.lower(), (
                f"coinbase_onramp_initiate schema property "
                f"{prop_name!r} contains execute-class token "
                f"{token!r}; composer-only contract requires "
                f"removal."
            )


# ── Validation ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_usd_amount_rejected():
    r = await handle_coinbase_onramp_initiate({
        "destination_user_id": "alice",
    })
    assert "usd_amount" in r.lower()


@pytest.mark.asyncio
async def test_missing_destination_rejected_client_side():
    r = await handle_coinbase_onramp_initiate({
        "usd_amount": 100.0,
    })
    assert (
        "destination_user_id" in r.lower()
        or "destination_address" in r.lower()
    )


# ── Composer rendering ───────────────────────────────────


@pytest.mark.asyncio
async def test_renders_pending_commission_artifact_user_id():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "status": "PENDING_COMMISSION",
            "requested_usd": 100.0,
            "destination_user_id": "alice",
            "destination_address": "0xabc",
            "ftns_to_receive": 100.0,
            "usd_rate": 1.0,
            "quote": {
                "usd_in": 100.0,
                "usdc_acquired": 100.0,
                "ftns_received": 100.0,
                "onramp_route": "coinbase-cdp",
                "swap_route": "aerodrome",
                "payment_method_alias": "primary",
            },
            "note": "preview only",
        }),
    ) as mock_call:
        r = await handle_coinbase_onramp_initiate({
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        })
    call_args = mock_call.await_args[0]
    assert call_args[0] == "POST"
    assert call_args[1] == "/wallet/onramp/quote"
    body = call_args[2]
    assert body["usd_amount"] == 100.0
    assert body["destination_user_id"] == "alice"
    # Output mentions PENDING_COMMISSION + key fields
    assert "PENDING_COMMISSION" in r
    assert "100" in r
    assert "alice" in r or "0xabc" in r
    # Composer-only language present
    assert (
        "preview" in r.lower()
        or "pending" in r.lower()
        or "artifact" in r.lower()
    )


@pytest.mark.asyncio
async def test_renders_explicit_address_path():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "status": "PENDING_COMMISSION",
            "requested_usd": 50.0,
            "destination_user_id": None,
            "destination_address": "0xrecipient",
            "ftns_to_receive": 50.0,
            "usd_rate": 1.0,
            "quote": {
                "usd_in": 50.0,
                "usdc_acquired": 50.0,
                "ftns_received": 50.0,
                "onramp_route": "coinbase-cdp",
                "swap_route": "aerodrome",
                "payment_method_alias": "primary",
            },
            "note": "preview only",
        }),
    ) as mock_call:
        r = await handle_coinbase_onramp_initiate({
            "usd_amount": 50.0,
            "destination_address": "0xrecipient",
        })
    body = mock_call.await_args[0][2]
    assert body["destination_address"] == "0xrecipient"
    assert "0xrecipient" in r


@pytest.mark.asyncio
async def test_renders_pending_wallet_note():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "status": "PENDING_COMMISSION",
            "requested_usd": 100.0,
            "destination_user_id": "alice",
            "destination_address": None,
            "ftns_to_receive": 100.0,
            "usd_rate": 1.0,
            "quote": {
                "usd_in": 100.0, "usdc_acquired": 100.0,
                "ftns_received": 100.0,
                "onramp_route": "coinbase-cdp",
                "swap_route": "aerodrome",
                "payment_method_alias": "primary",
            },
            "note": (
                "Destination WaaS wallet is "
                "status=PENDING_COMMISSION (no address yet)."
            ),
        }),
    ):
        r = await handle_coinbase_onramp_initiate({
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        })
    assert "PENDING_COMMISSION" in r
    assert "wallet" in r.lower()


# ── Error paths ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_renders_404_user_id_missing():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "detail": "no WaaS wallet for destination_user_id='ghost'",
        }),
    ):
        r = await handle_coinbase_onramp_initiate({
            "usd_amount": 100.0,
            "destination_user_id": "ghost",
        })
    assert "no waas wallet" in r.lower()
    assert "provision" in r.lower()


@pytest.mark.asyncio
async def test_renders_400_negative_amount():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "detail": "usd_amount must be positive (> 0)",
        }),
    ):
        r = await handle_coinbase_onramp_initiate({
            "usd_amount": -10.0,
            "destination_user_id": "alice",
        })
    assert "positive" in r.lower() or "> 0" in r


@pytest.mark.asyncio
async def test_payment_method_alias_passed_through():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "status": "PENDING_COMMISSION",
            "requested_usd": 100.0,
            "destination_user_id": "alice",
            "destination_address": "0xabc",
            "ftns_to_receive": 100.0,
            "usd_rate": 1.0,
            "quote": {
                "usd_in": 100.0, "usdc_acquired": 100.0,
                "ftns_received": 100.0,
                "onramp_route": "coinbase-cdp",
                "swap_route": "aerodrome",
                "payment_method_alias": "savings",
            },
            "note": "x",
        }),
    ) as mock_call:
        await handle_coinbase_onramp_initiate({
            "usd_amount": 100.0,
            "destination_user_id": "alice",
            "payment_method_alias": "savings",
        })
    body = mock_call.await_args[0][2]
    assert body["payment_method_alias"] == "savings"
