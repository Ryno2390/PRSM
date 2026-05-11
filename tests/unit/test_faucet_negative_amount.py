"""Sprint 181 — /ftns/faucet rejects non-positive amounts (was DoS / debit bug).

Pre-fix the handler did:
    amount = min(float(body.get("amount", 100)), 100)

Capped at 100 max but had NO lower bound. `amount=-1` returned
200 with `"granted": -1.0` and DEBITED the wallet via the
canonical `ledger.credit(amount=-1.0, ...)` call. Effectively
converted the testnet faucet into an arbitrary-debit endpoint —
operator could zero out their own balance via repeated calls,
and (if combined with the existing wallet_id passthrough) drain
any wallet they could target.

Live dogfood reproduced 2026-05-11:
    POST /ftns/faucet {"amount":-1}
    → {"granted":-1.0,"new_balance":197.55,...}[200]

Post-fix:
    → 422 "amount must be > 0; got -1.0."
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    ledger = MagicMock()
    ledger.get_balance = AsyncMock(return_value=50.0)
    ledger.credit = AsyncMock()
    node.ledger = ledger
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestFaucetAmountValidation:
    def test_negative_amount_returns_422(self):
        resp = _client(_node()).post(
            "/ftns/faucet", json={"amount": -1},
        )
        assert resp.status_code == 422
        assert "must be > 0" in resp.json()["detail"]

    def test_zero_amount_returns_422(self):
        resp = _client(_node()).post(
            "/ftns/faucet", json={"amount": 0},
        )
        assert resp.status_code == 422

    def test_non_numeric_amount_returns_422(self):
        resp = _client(_node()).post(
            "/ftns/faucet", json={"amount": "not_a_number"},
        )
        assert resp.status_code == 422
        assert "must be a positive number" in resp.json()["detail"]

    def test_valid_amount_returns_200(self):
        resp = _client(_node()).post(
            "/ftns/faucet", json={"amount": 10},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["granted"] == 10.0

    def test_default_amount_returns_200(self):
        """Empty body defaults to 100 — preserved behavior."""
        resp = _client(_node()).post("/ftns/faucet", json={})
        assert resp.status_code == 200
        assert resp.json()["granted"] == 100.0

    def test_amount_above_cap_clamped_to_100(self):
        """Sprint 181 invariant — upper-bound cap (100) preserved
        from pre-fix behavior."""
        resp = _client(_node()).post(
            "/ftns/faucet", json={"amount": 500},
        )
        assert resp.status_code == 200
        assert resp.json()["granted"] == 100.0

    def test_negative_amount_does_not_credit_ledger(self):
        """Sprint 181 invariant — bad input MUST NOT reach the
        ledger.credit() call. Regression-pin against future
        validation-order drift."""
        node = _node()
        _client(node).post("/ftns/faucet", json={"amount": -50})
        node.ledger.credit.assert_not_called()
