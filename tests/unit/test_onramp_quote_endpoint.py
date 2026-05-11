"""Sprint 278 — Coinbase onramp quote endpoint (USD → FTNS).

Mirrors the sprint-2026-05-08 offramp quote: composer-only
PENDING_COMMISSION artifact, no execute path. The user has USD
(via Coinbase-managed bank/card), wants FTNS deposited to a
WaaS-managed wallet or explicit address.

V1 surface intentionally narrow:
  POST body: {usd_amount, destination_user_id?, destination_address?,
              payment_method_alias?}
  Either destination_user_id (WaaS-resolved) or destination_address
  required; XOR enforced.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient
from prsm.node.api import create_api_app


class FakeWaasBackend:
    def create_wallet(self, user_id, email):
        return {
            "wallet_id": f"w-{user_id}",
            "address": f"0x{user_id:0>40}",
            "network": "base-mainnet",
        }


def _client(waas=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._coinbase_waas_client = waas
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _commissioned_waas():
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        backend=FakeWaasBackend(),
    )
    c.provision_wallet(user_id="alice", email="a@x.io")
    return c


# ── POST /wallet/onramp/quote ────────────────────────────


def test_onramp_quote_happy_path_with_user_id():
    resp = _client(_commissioned_waas()).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PENDING_COMMISSION"
    assert body["requested_usd"] == 100.0
    assert body["destination_address"].startswith("0x")
    assert body["destination_user_id"] == "alice"
    assert "ftns_to_receive" in body
    assert body["ftns_to_receive"] > 0


def test_onramp_quote_happy_path_with_explicit_address():
    resp = _client(_commissioned_waas()).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 50.0,
            "destination_address": "0xabc",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["destination_address"] == "0xabc"
    assert body["destination_user_id"] is None


def test_onramp_quote_400_negative_amount():
    resp = _client(_commissioned_waas()).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": -10.0,
            "destination_user_id": "alice",
        },
    )
    assert resp.status_code == 400


def test_onramp_quote_400_zero_amount():
    resp = _client(_commissioned_waas()).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 0,
            "destination_user_id": "alice",
        },
    )
    assert resp.status_code == 400


def test_onramp_quote_422_missing_destination():
    resp = _client(_commissioned_waas()).post(
        "/wallet/onramp/quote",
        json={"usd_amount": 100.0},
    )
    assert resp.status_code == 422
    assert (
        "destination_user_id" in resp.json()["detail"].lower()
        or "destination_address" in resp.json()["detail"].lower()
    )


def test_onramp_quote_422_both_destinations():
    resp = _client(_commissioned_waas()).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
            "destination_address": "0xabc",
        },
    )
    assert resp.status_code == 422


def test_onramp_quote_404_unknown_user_id():
    waas = CoinbaseWaaSClient()  # no users
    resp = _client(waas).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "ghost",
        },
    )
    assert resp.status_code == 404


def test_onramp_quote_pending_when_sender_wallet_pending():
    """If user_id resolves to a PENDING_COMMISSION WaaS wallet
    (no address yet), the quote should surface that explicitly
    and not silently substitute."""
    waas = CoinbaseWaaSClient()  # uncommissioned client
    waas.provision_wallet(user_id="alice", email="a@x.io")
    resp = _client(waas).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PENDING_COMMISSION"
    # Address null because the WaaS wallet itself is pending
    assert body["destination_address"] is None
    assert "wallet" in body["note"].lower()


def test_onramp_quote_503_when_waas_unwired_and_user_id_used():
    """User-id resolution requires WaaS; absent it should 503."""
    resp = _client(None).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    assert resp.status_code == 503


def test_onramp_quote_explicit_address_works_without_waas():
    """Explicit address bypasses WaaS lookup entirely."""
    resp = _client(None).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_address": "0xabc",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["destination_address"] == "0xabc"


def test_onramp_quote_uses_env_rate(monkeypatch):
    """PRSM_FTNS_USD_RATE drives the rate; default is 1.0."""
    monkeypatch.setenv("PRSM_FTNS_USD_RATE", "0.5")
    resp = _client(_commissioned_waas()).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    # At 0.5 USD/FTNS, $100 buys 200 FTNS
    assert abs(body["ftns_to_receive"] - 200.0) < 1e-9


def test_onramp_quote_includes_swap_route_and_onramp_route():
    resp = _client(_commissioned_waas()).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["quote"]["onramp_route"] == "coinbase-cdp"
    assert body["quote"]["swap_route"] == "aerodrome"
    assert body["quote"]["payment_method_alias"] == "primary"


def test_onramp_quote_payment_method_alias_passthrough():
    resp = _client(_commissioned_waas()).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
            "payment_method_alias": "savings",
        },
    )
    body = resp.json()
    assert body["quote"]["payment_method_alias"] == "savings"
