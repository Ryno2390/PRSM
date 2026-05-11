"""Sprint 277 — Gasless FTNS transfer endpoints.

Composes a UserOperation from a WaaS-managed sender wallet and
routes it through the paymaster for sponsored submission. Per
Vision §14 the user never sees gas; the operator's paymaster
reserve covers it.

dry_run=True (default) → estimate-only artifact.
dry_run=False           → real sponsored submission.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient
from prsm.economy.web3.paymaster_client import PaymasterClient
from prsm.node.api import create_api_app


class FakeWaasBackend:
    def create_wallet(self, user_id, email):
        return {
            "wallet_id": f"w-{user_id}",
            "address": f"0x{user_id:0>40}",
            "network": "base-mainnet",
        }


class FakePaymasterBackend:
    def estimate_gas(self, op):
        return {
            "gas_estimate_wei": 100_000,
            "max_fee_per_gas_wei": 50_000_000,
        }
    def submit_sponsored(self, op):
        return {
            "user_op_hash": "0xdeadbeef",
            "tx_hash": "0xsuccess",
            "sponsor_amount_wei": 5_000_000_000_000_000,
        }


def _client(waas=None, paymaster=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._coinbase_waas_client = waas
    node._paymaster_client = paymaster
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


# ── POST /wallet/transfer/gasless ────────────────────────


def test_gasless_503_when_paymaster_unwired():
    resp = _client(_commissioned_waas(), None).post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "to_address": "0xrecipient",
            "ftns_amount": "10.5",
        },
    )
    assert resp.status_code == 503


def test_gasless_503_when_waas_unwired():
    pm = PaymasterClient(
        endpoint="x", api_key="y",
        backend=FakePaymasterBackend(),
    )
    resp = _client(None, pm).post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "to_address": "0xrecipient",
            "ftns_amount": "10.5",
        },
    )
    assert resp.status_code == 503


def test_gasless_404_unknown_sender():
    waas = CoinbaseWaaSClient()  # no users
    pm = PaymasterClient()
    resp = _client(waas, pm).post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "ghost",
            "to_address": "0xrecipient",
            "ftns_amount": "10.5",
        },
    )
    assert resp.status_code == 404


def test_gasless_422_missing_recipient():
    resp = _client(_commissioned_waas(), PaymasterClient()).post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "ftns_amount": "10.5",
        },
    )
    assert resp.status_code == 422


def test_gasless_422_zero_amount():
    resp = _client(_commissioned_waas(), PaymasterClient()).post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "to_address": "0xrecipient",
            "ftns_amount": "0",
        },
    )
    assert resp.status_code == 422


def test_gasless_422_negative_amount():
    resp = _client(_commissioned_waas(), PaymasterClient()).post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "to_address": "0xrecipient",
            "ftns_amount": "-5",
        },
    )
    assert resp.status_code == 422


def test_gasless_pending_commission_when_paymaster_uncommissioned():
    """Both WaaS and PaymasterClient wired, but paymaster has no
    keys yet — endpoint should return 200 with PENDING_COMMISSION
    record (preview artifact)."""
    waas = _commissioned_waas()
    pm = PaymasterClient()  # uncommissioned
    resp = _client(waas, pm).post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "to_address": "0xrecipient",
            "ftns_amount": "10.5",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PENDING_COMMISSION"
    assert body["tx_hash"] is None


def test_gasless_dry_run_returns_estimate():
    waas = _commissioned_waas()
    pm = PaymasterClient(
        endpoint="x", api_key="y",
        backend=FakePaymasterBackend(),
    )
    resp = _client(waas, pm).post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "to_address": "0xrecipient",
            "ftns_amount": "10.5",
            "dry_run": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ESTIMATED"
    assert body["gas_estimate_wei"] == 100_000
    assert body["tx_hash"] is None


def test_gasless_dry_run_is_default():
    """Omitting dry_run defaults to True (composer-first UX)."""
    waas = _commissioned_waas()
    pm = PaymasterClient(
        endpoint="x", api_key="y",
        backend=FakePaymasterBackend(),
    )
    resp = _client(waas, pm).post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "to_address": "0xrecipient",
            "ftns_amount": "10.5",
        },
    )
    body = resp.json()
    assert body["status"] == "ESTIMATED"


def test_gasless_execute_returns_tx_hash():
    waas = _commissioned_waas()
    pm = PaymasterClient(
        endpoint="x", api_key="y",
        backend=FakePaymasterBackend(),
    )
    resp = _client(waas, pm).post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "to_address": "0xrecipient",
            "ftns_amount": "10.5",
            "dry_run": False,
        },
    )
    body = resp.json()
    assert body["status"] == "SUBMITTED"
    assert body["tx_hash"] == "0xsuccess"
    assert body["sponsor_amount_wei"] == 5_000_000_000_000_000


# ── GET /wallet/paymaster/status ─────────────────────────


def test_paymaster_status_503_when_unwired():
    resp = _client(_commissioned_waas(), None).get(
        "/wallet/paymaster/status",
    )
    assert resp.status_code == 503


def test_paymaster_status_uncommissioned():
    resp = _client(_commissioned_waas(), PaymasterClient()).get(
        "/wallet/paymaster/status",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["commissioned"] is False
    assert body["sponsorships"] == 0


def test_paymaster_status_commissioned_with_spend():
    waas = _commissioned_waas()
    pm = PaymasterClient(
        endpoint="x", api_key="y",
        backend=FakePaymasterBackend(),
    )
    cli = _client(waas, pm)
    cli.post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "to_address": "0xrecipient",
            "ftns_amount": "10",
            "dry_run": False,
        },
    )
    resp = cli.get("/wallet/paymaster/status")
    body = resp.json()
    assert body["commissioned"] is True
    assert body["sponsorships"] == 1
    assert body["total_sponsored_wei"] == 5_000_000_000_000_000
