"""Sprint 890 — swap slippage ceiling (anti-sandwich).

sp887 finding #7. /wallet/swap/quote + /wallet/swap/execute
validated slippage_bps in [0, 10000], but slippage_bps=10000 makes
amountOutMin = amount_out * (10000-10000)//10000 = 0 — the swap
would accept ANY output, i.e. unbounded sandwich/MEV loss. Even
values like 5000 (50%) are economically reckless.

Sp890 enforces a sane max-slippage ceiling
(PRSM_SWAP_MAX_SLIPPAGE_BPS, default 1000 = 10%, raisable by
operators for thin-liquidity launch conditions) on both swap
endpoints, plus an amountOutMin>0 invariant in the orchestrator's
envelope builder (never accept zero output).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


class _UnconfiguredAerodrome:
    """is_configured()=False → endpoints return POOL_NOT_CONFIGURED
    (200), so a slippage-gate PASS surfaces as 200 (not the 503 a
    None client would give). Lets us prove the slippage guard fires
    BEFORE the pool-config check."""
    def is_configured(self): return False
    def get_pool_state(self, *a, **k): return None


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._aerodrome_client = _UnconfiguredAerodrome()
    node._coinbase_waas_client = None
    node._kyc_client = None
    node._fiat_compliance_ring = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── Quote endpoint slippage ceiling ──────────────────────────

def test_quote_rejects_slippage_10000():
    """100% slippage (amountOutMin → 0, accept-anything sandwich)
    is rejected with 422 — before any pool work."""
    resp = _client().post(
        "/wallet/swap/quote",
        json={
            "amount_in": 100.0, "token_in": "USDC",
            "slippage_bps": 10000,
        },
    )
    assert resp.status_code == 422


def test_quote_rejects_slippage_above_default_max():
    """5000 bps (50%) exceeds the 1000-bps (10%) default ceiling."""
    resp = _client().post(
        "/wallet/swap/quote",
        json={
            "amount_in": 100.0, "token_in": "USDC",
            "slippage_bps": 5000,
        },
    )
    assert resp.status_code == 422
    assert "slippage" in resp.text.lower()


def test_quote_accepts_slippage_at_default_max():
    """Exactly 1000 bps (10%) is allowed; pool unwired →
    POOL_NOT_CONFIGURED (200), proving the slippage gate passed."""
    resp = _client().post(
        "/wallet/swap/quote",
        json={
            "amount_in": 100.0, "token_in": "USDC",
            "slippage_bps": 1000,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "POOL_NOT_CONFIGURED"


def test_quote_accepts_normal_slippage():
    resp = _client().post(
        "/wallet/swap/quote",
        json={
            "amount_in": 100.0, "token_in": "USDC",
            "slippage_bps": 100,
        },
    )
    assert resp.status_code == 200


def test_quote_negative_slippage_still_rejected():
    resp = _client().post(
        "/wallet/swap/quote",
        json={
            "amount_in": 100.0, "token_in": "USDC",
            "slippage_bps": -1,
        },
    )
    assert resp.status_code == 422


# ── Env-tunable ceiling ──────────────────────────────────────

def test_quote_env_raises_ceiling(monkeypatch):
    """Operators can raise the ceiling for thin-liquidity launch
    conditions via PRSM_SWAP_MAX_SLIPPAGE_BPS."""
    monkeypatch.setenv("PRSM_SWAP_MAX_SLIPPAGE_BPS", "3000")
    resp = _client().post(
        "/wallet/swap/quote",
        json={
            "amount_in": 100.0, "token_in": "USDC",
            "slippage_bps": 2500,  # now under the raised 3000 cap
        },
    )
    assert resp.status_code == 200


def test_quote_env_ceiling_still_caps_at_10000(monkeypatch):
    """Even a misconfigured huge env ceiling can't allow 10000
    (amountOutMin=0) — the absolute zero-output guard holds."""
    monkeypatch.setenv("PRSM_SWAP_MAX_SLIPPAGE_BPS", "99999")
    resp = _client().post(
        "/wallet/swap/quote",
        json={
            "amount_in": 100.0, "token_in": "USDC",
            "slippage_bps": 10000,
        },
    )
    assert resp.status_code == 422


# ── Execute endpoint slippage ceiling ────────────────────────

def test_execute_rejects_high_slippage():
    resp = _client().post(
        "/wallet/swap/execute",
        json={
            "amount_in": 100.0, "token_in": "USDC",
            "slippage_bps": 9000,
            "from_address": "0x" + "11" * 20,
        },
    )
    assert resp.status_code == 422


def test_execute_accepts_normal_slippage():
    """Normal slippage passes the gate (then POOL_NOT_CONFIGURED)."""
    resp = _client().post(
        "/wallet/swap/execute",
        json={
            "amount_in": 100.0, "token_in": "USDC",
            "slippage_bps": 100,
            "from_address": "0x" + "11" * 20,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "POOL_NOT_CONFIGURED"


# ── Envelope builder amountOutMin>0 invariant ────────────────

class _FakeQuote:
    def __init__(self, amount_out):
        self.amount_out = amount_out
        self.price_impact_bps = 10
        self.fee_bps = 30


class _FakeAerodrome:
    def is_configured(self): return True
    def quote_swap(self, *, amount_in, token_in, **k):
        return _FakeQuote(amount_out=1000)  # small output


class _Intent:
    intent_id = "ix"
    user_id = "alice"
    destination_address = "0x" + "11" * 20
    usdc_received = 5.0


def test_envelope_never_zero_amount_out_min():
    """build_envelope_for_intent must never emit amountOutMin=0
    when amount_out>0 — even with an extreme slippage, a swap that
    accepts zero output is invalid."""
    from prsm.economy.web3.onramp_to_swap_orchestrator import (
        build_envelope_for_intent,
    )
    env = build_envelope_for_intent(
        _Intent(),
        aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ab" * 20,
        slippage_bps=10000,  # would zero out the min
    )
    assert env is not None
    assert env["args"]["amountOutMin"] >= 1
