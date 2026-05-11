"""Sprint 276 — WaaS HTTP endpoints.

Operator-facing surface for the CoinbaseWaaSClient. The endpoints
honor the PENDING_COMMISSION pattern: when CDP keys are absent the
endpoint still returns 200 with a PENDING_COMMISSION record (it's
a useful preview, not an error condition).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient
from prsm.node.api import create_api_app


class FakeBackend:
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


# ── POST /wallet/waas/provision ──────────────────────────


def test_provision_503_when_unwired():
    resp = _client(None).post(
        "/wallet/waas/provision",
        json={"user_id": "alice", "email": "a@x.io"},
    )
    assert resp.status_code == 503


def test_provision_uncommissioned_returns_pending():
    c = CoinbaseWaaSClient()  # no keys
    resp = _client(c).post(
        "/wallet/waas/provision",
        json={"user_id": "alice", "email": "a@x.io"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "alice"
    assert body["status"] == "PENDING_COMMISSION"
    assert body["wallet_id"] is None


def test_provision_commissioned_returns_provisioned():
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        backend=FakeBackend(),
    )
    resp = _client(c).post(
        "/wallet/waas/provision",
        json={"user_id": "alice", "email": "a@x.io"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PROVISIONED"
    assert body["wallet_id"] == "w-alice"
    assert body["address"].startswith("0x")


def test_provision_422_missing_user_id():
    c = CoinbaseWaaSClient()
    resp = _client(c).post(
        "/wallet/waas/provision",
        json={"email": "a@x.io"},
    )
    assert resp.status_code == 422


def test_provision_422_missing_email():
    c = CoinbaseWaaSClient()
    resp = _client(c).post(
        "/wallet/waas/provision",
        json={"user_id": "alice"},
    )
    assert resp.status_code == 422


def test_provision_idempotent():
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        backend=FakeBackend(),
    )
    cli = _client(c)
    r1 = cli.post(
        "/wallet/waas/provision",
        json={"user_id": "alice", "email": "a@x.io"},
    )
    r2 = cli.post(
        "/wallet/waas/provision",
        json={"user_id": "alice", "email": "a@x.io"},
    )
    assert r1.json()["wallet_id"] == r2.json()["wallet_id"]


# ── GET /wallet/waas/{user_id} ───────────────────────────


def test_get_one_503_when_unwired():
    resp = _client(None).get("/wallet/waas/alice")
    assert resp.status_code == 503


def test_get_one_404_when_missing():
    c = CoinbaseWaaSClient()
    resp = _client(c).get("/wallet/waas/nobody")
    assert resp.status_code == 404


def test_get_one_happy_path():
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        backend=FakeBackend(),
    )
    c.provision_wallet(user_id="alice", email="a@x.io")
    resp = _client(c).get("/wallet/waas/alice")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "alice"
    assert body["status"] == "PROVISIONED"


# ── GET /wallet/waas ─────────────────────────────────────


def test_list_503_when_unwired():
    resp = _client(None).get("/wallet/waas")
    assert resp.status_code == 503


def test_list_empty():
    c = CoinbaseWaaSClient()
    resp = _client(c).get("/wallet/waas")
    assert resp.status_code == 200
    body = resp.json()
    assert body["wallets"] == []
    assert body["count"] == 0


def test_list_returns_provisioned_wallets():
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        backend=FakeBackend(),
    )
    c.provision_wallet(user_id="alice", email="a@x.io")
    c.provision_wallet(user_id="bob", email="b@x.io")
    resp = _client(c).get("/wallet/waas")
    body = resp.json()
    assert body["count"] == 2
    ids = sorted(w["user_id"] for w in body["wallets"])
    assert ids == ["alice", "bob"]


def test_list_invalid_limit_422():
    c = CoinbaseWaaSClient()
    resp = _client(c).get("/wallet/waas?limit=0")
    assert resp.status_code == 422
    resp = _client(c).get("/wallet/waas?limit=10001")
    assert resp.status_code == 422


# ── GET /wallet/waas/status ──────────────────────────────


def test_status_503_when_unwired():
    resp = _client(None).get("/wallet/waas/status")
    assert resp.status_code == 503


def test_status_uncommissioned():
    c = CoinbaseWaaSClient()
    resp = _client(c).get("/wallet/waas/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["commissioned"] is False
    assert body["network"] == "base-mainnet"


def test_status_commissioned():
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        backend=FakeBackend(),
    )
    resp = _client(c).get("/wallet/waas/status")
    body = resp.json()
    assert body["commissioned"] is True
