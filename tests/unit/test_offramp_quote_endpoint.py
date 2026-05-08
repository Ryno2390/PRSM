"""POST /wallet/offramp/quote — backend endpoint feeding the
coinbase_offramp_initiate MCP tool.

V1 scope: pre-flight transaction-summary composer. Returns the
quote shape that Vision §13 Phase 5 step 2 ("Gemini presents an
Artifact in your side panel") describes; does NOT initiate any
on-chain swap or Coinbase off-ramp. Actual execution gates on
CDP commission per Vision gantt 2026-06-15. Until then the
endpoint returns a `status: "PENDING_COMMISSION"` envelope.

Response shape:
    {
        "requested_usd": 500.0,
        "source_address": "0x...",
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
        "commission_gate_note": "...",
    }
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node_with_ftns_ledger(*, balance_ftns: float = 100.0,
                           connected_address: str | None = "0x" + "11" * 20):
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = "test-node"
    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = connected_address
    ftns_ledger._decimals = 18
    ftns_ledger.get_balance = AsyncMock(return_value=balance_ftns)
    node.ftns_ledger = ftns_ledger
    return node


def _client(node):
    app = create_api_app(node, enable_security=False)
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


class TestOfframpQuoteHappyPath:
    def test_quote_for_default_bank_alias(self):
        node = _node_with_ftns_ledger(balance_ftns=4200.0)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}):
            response = _client(node).post(
                "/wallet/offramp/quote",
                json={"usd_amount": 500.0},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["requested_usd"] == 500.0
        assert body["source_balance_ftns"] == 4200.0
        assert body["status"] == "PENDING_COMMISSION"
        assert body["quote"]["bank_account_alias"] == "primary"
        assert body["quote"]["swap_route"] == "aerodrome"
        assert body["quote"]["offramp_route"] == "coinbase-cdp"

    def test_ftns_to_swap_computed_via_usd_rate(self):
        # At rate=2.0, $500 should require 250 FTNS.
        node = _node_with_ftns_ledger(balance_ftns=4200.0)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "2.0"}):
            response = _client(node).post(
                "/wallet/offramp/quote",
                json={"usd_amount": 500.0},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["quote"]["ftns_to_swap"] == 250.0
        assert body["quote"]["usdc_received"] == 500.0
        assert body["usd_rate"] == 2.0

    def test_custom_bank_alias_propagates(self):
        node = _node_with_ftns_ledger(balance_ftns=4200.0)
        response = _client(node).post(
            "/wallet/offramp/quote",
            json={"usd_amount": 100.0, "bank_account_alias": "savings"},
        )
        assert response.status_code == 200
        assert response.json()["quote"]["bank_account_alias"] == "savings"

    def test_commission_gate_note_present(self):
        node = _node_with_ftns_ledger(balance_ftns=4200.0)
        response = _client(node).post(
            "/wallet/offramp/quote",
            json={"usd_amount": 100.0},
        )
        body = response.json()
        # Note must explicitly reference the gate so consumers see why
        # the status is PENDING.
        assert "commission" in body["commission_gate_note"].lower() or \
               "aerodrome" in body["commission_gate_note"].lower()

    def test_address_query_param_overrides_default(self):
        node = _node_with_ftns_ledger(balance_ftns=10.0)
        target = "0x" + "ab" * 20
        response = _client(node).post(
            f"/wallet/offramp/quote?address={target}",
            json={"usd_amount": 1.0},
        )
        assert response.status_code == 200
        assert response.json()["source_address"] == target


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


class TestOfframpQuoteValidation:
    def test_negative_usd_amount_rejected(self):
        node = _node_with_ftns_ledger(balance_ftns=100.0)
        response = _client(node).post(
            "/wallet/offramp/quote",
            json={"usd_amount": -10.0},
        )
        assert response.status_code == 400
        assert "positive" in response.json()["detail"].lower() or \
               "must be > 0" in response.json()["detail"].lower()

    def test_zero_usd_amount_rejected(self):
        node = _node_with_ftns_ledger(balance_ftns=100.0)
        response = _client(node).post(
            "/wallet/offramp/quote",
            json={"usd_amount": 0.0},
        )
        assert response.status_code == 400

    def test_insufficient_balance_returns_422(self):
        # Balance is 10 FTNS at rate 1.0 USD/FTNS = $10 total
        # available; $500 requested = insufficient.
        node = _node_with_ftns_ledger(balance_ftns=10.0)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}):
            response = _client(node).post(
                "/wallet/offramp/quote",
                json={"usd_amount": 500.0},
            )
        assert response.status_code == 422
        body = response.json()
        assert "insufficient" in body["detail"].lower() or \
               "balance" in body["detail"].lower()


# ──────────────────────────────────────────────────────────────────────
# Error paths
# ──────────────────────────────────────────────────────────────────────


class TestOfframpQuoteErrors:
    def test_503_when_ftns_ledger_missing(self):
        node = MagicMock()
        node.identity = MagicMock()
        node.identity.node_id = "test-node"
        node.ftns_ledger = None
        response = _client(node).post(
            "/wallet/offramp/quote",
            json={"usd_amount": 100.0},
        )
        assert response.status_code == 503

    def test_missing_usd_amount_rejected(self):
        node = _node_with_ftns_ledger(balance_ftns=100.0)
        response = _client(node).post(
            "/wallet/offramp/quote",
            json={},
        )
        # FastAPI/pydantic validation returns 422 for missing required.
        assert response.status_code in (400, 422)
