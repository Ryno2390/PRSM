"""Sprint 279 — Aerodrome pool quoter HTTP endpoints.

Read-only operator surface for the AerodromeClient. Real
production code (no commission gate). Before the seeding
ceremony lands, endpoints surface NOT_CONFIGURED so operators
can verify plumbing without waiting on liquidity.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.economy.web3.aerodrome_client import AerodromeClient
from prsm.node.api import create_api_app


class FakeBackend:
    def get_pool_state(self, pool_address):
        return {
            "pool_address": pool_address,
            "token0": "0xusdc",
            "token1": "0xftns",
            "reserve0": 1_000_000,
            "reserve1": 1_000_000,
            "stable": False,
            "fee_bps": 30,
            "total_supply": 1_000_000,
            "block_number": 42,
        }


class StableBackend:
    def get_pool_state(self, pool_address):
        return {
            "pool_address": pool_address,
            "token0": "0xusdc", "token1": "0xftns",
            "reserve0": 1_000_000, "reserve1": 1_000_000,
            "stable": True, "fee_bps": 5,
            "total_supply": 1_000_000, "block_number": 42,
        }


def _client(pool=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._aerodrome_client = pool
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _configured_pool():
    return AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=FakeBackend(),
    )


# ── GET /wallet/pool/state ───────────────────────────────


def test_pool_state_503_when_unwired():
    resp = _client(None).get("/wallet/pool/state")
    assert resp.status_code == 503


def test_pool_state_unconfigured_returns_200():
    """When client exists but pool/RPC not yet configured,
    return a NOT_CONFIGURED record so operators can see the
    plumbing is wired and just waiting on the seeding ceremony."""
    pool = AerodromeClient()  # nothing configured
    resp = _client(pool).get("/wallet/pool/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "NOT_CONFIGURED"


def test_pool_state_happy_path():
    resp = _client(_configured_pool()).get("/wallet/pool/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "OK"
    assert body["pool_address"] == "0xpool"
    assert body["reserve0"] == 1_000_000
    assert body["reserve1"] == 1_000_000
    assert body["fee_bps"] == 30


def test_pool_state_rpc_error_returns_unavailable():
    class BoomBackend:
        def get_pool_state(self, addr):
            raise RuntimeError("rpc down")
    pool = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=BoomBackend(),
    )
    resp = _client(pool).get("/wallet/pool/state")
    # Backend exceptions surface as POOL_UNAVAILABLE so operators
    # see the difference between "not configured" and "tried and
    # failed."
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "POOL_UNAVAILABLE"


# ── GET /wallet/pool/quote ───────────────────────────────


def test_pool_quote_503_when_unwired():
    resp = _client(None).get(
        "/wallet/pool/quote?amount_in=1000&token_in=0xusdc",
    )
    assert resp.status_code == 503


def test_pool_quote_unconfigured_returns_200():
    pool = AerodromeClient()
    resp = _client(pool).get(
        "/wallet/pool/quote?amount_in=1000&token_in=0xusdc",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "NOT_CONFIGURED"


def test_pool_quote_happy_path():
    resp = _client(_configured_pool()).get(
        "/wallet/pool/quote?amount_in=1000&token_in=0xusdc",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "OK"
    assert body["amount_in"] == 1000
    assert body["token_in"] == "0xusdc"
    assert body["token_out"] == "0xftns"
    assert 995 <= body["amount_out"] <= 997
    assert body["route"] == "aerodrome"


def test_pool_quote_422_amount_zero():
    resp = _client(_configured_pool()).get(
        "/wallet/pool/quote?amount_in=0&token_in=0xusdc",
    )
    assert resp.status_code == 422


def test_pool_quote_422_amount_negative():
    resp = _client(_configured_pool()).get(
        "/wallet/pool/quote?amount_in=-10&token_in=0xusdc",
    )
    assert resp.status_code == 422


def test_pool_quote_422_missing_token_in():
    resp = _client(_configured_pool()).get(
        "/wallet/pool/quote?amount_in=1000",
    )
    # FastAPI returns 422 for missing required query param
    assert resp.status_code == 422


def test_pool_quote_422_unknown_token():
    resp = _client(_configured_pool()).get(
        "/wallet/pool/quote?amount_in=1000&token_in=0xeth",
    )
    assert resp.status_code == 422
    assert "not in pool" in resp.json()["detail"].lower()


def test_pool_quote_422_stable_pool_unsupported():
    pool = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=StableBackend(),
    )
    resp = _client(pool).get(
        "/wallet/pool/quote?amount_in=1000&token_in=0xusdc",
    )
    assert resp.status_code == 422
    assert "stable" in resp.json()["detail"].lower()
