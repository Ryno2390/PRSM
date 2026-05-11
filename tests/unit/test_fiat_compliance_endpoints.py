"""Sprint 282 — Fiat compliance admin endpoints + quote-side
recording.

GET  /admin/fiat-compliance             — paginated list
GET  /admin/fiat-compliance/summary     — by-kind aggregate
GET  /admin/fiat-compliance/{entry_id}  — single entry
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient
from prsm.economy.web3.fiat_compliance_ring import FiatComplianceRing
from prsm.economy.web3.kyc_client import (
    KYCClient, KYC_STATUS_VERIFIED,
)
from prsm.economy.web3.paymaster_client import PaymasterClient
from prsm.node.api import create_api_app


class FakeWaas:
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
            "user_op_hash": "0xdead",
            "tx_hash": "0xtxhash",
            "sponsor_amount_wei": 1_000_000_000_000_000,
        }


def _client(ring=None, waas=None, kyc=None, paymaster=None,
            ftns_ledger=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = ftns_ledger
    node._fiat_compliance_ring = ring
    node._coinbase_waas_client = waas
    node._kyc_client = kyc
    node._paymaster_client = paymaster
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _commissioned_waas():
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        backend=FakeWaas(),
    )
    c.provision_wallet(user_id="alice", email="a@x.io")
    return c


# ── GET /admin/fiat-compliance ───────────────────────────


def test_list_503_when_unwired():
    resp = _client(None).get("/admin/fiat-compliance")
    assert resp.status_code == 503


def test_list_empty():
    resp = _client(FiatComplianceRing()).get(
        "/admin/fiat-compliance",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["entries"] == []
    assert body["count"] == 0


def test_list_populated_newest_first():
    r = FiatComplianceRing()
    e1 = r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="OK",
        timestamp=100.0,
    )
    e2 = r.record(
        kind="offramp_quote", user_id="bob",
        usd_amount=50.0, ftns_amount=50.0, status="OK",
        timestamp=200.0,
    )
    resp = _client(r).get("/admin/fiat-compliance")
    body = resp.json()
    ids = [e["entry_id"] for e in body["entries"]]
    assert ids == [e2.entry_id, e1.entry_id]


def test_list_kind_filter():
    r = FiatComplianceRing()
    r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="OK",
    )
    r.record(
        kind="offramp_quote", user_id="alice",
        usd_amount=50.0, ftns_amount=50.0, status="OK",
    )
    resp = _client(r).get(
        "/admin/fiat-compliance?kind=onramp_quote",
    )
    body = resp.json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["kind"] == "onramp_quote"


def test_list_user_id_filter():
    r = FiatComplianceRing()
    r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="OK",
    )
    r.record(
        kind="onramp_quote", user_id="bob",
        usd_amount=50.0, ftns_amount=50.0, status="OK",
    )
    resp = _client(r).get(
        "/admin/fiat-compliance?user_id=alice",
    )
    body = resp.json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["user_id"] == "alice"


def test_list_invalid_limit():
    resp = _client(FiatComplianceRing()).get(
        "/admin/fiat-compliance?limit=0",
    )
    assert resp.status_code == 422


def test_list_invalid_kind():
    resp = _client(FiatComplianceRing()).get(
        "/admin/fiat-compliance?kind=bogus",
    )
    assert resp.status_code == 422


# ── GET /admin/fiat-compliance/summary ───────────────────


def test_summary_503_when_unwired():
    resp = _client(None).get("/admin/fiat-compliance/summary")
    assert resp.status_code == 503


def test_summary_empty():
    resp = _client(FiatComplianceRing()).get(
        "/admin/fiat-compliance/summary",
    )
    body = resp.json()
    assert body["by_kind"] == {}
    assert body["total_entries"] == 0


def test_summary_populated():
    r = FiatComplianceRing()
    r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="OK",
    )
    r.record(
        kind="onramp_quote", user_id="bob",
        usd_amount=200.0, ftns_amount=200.0, status="OK",
    )
    r.record(
        kind="offramp_quote", user_id="alice",
        usd_amount=50.0, ftns_amount=50.0, status="OK",
    )
    resp = _client(r).get("/admin/fiat-compliance/summary")
    body = resp.json()
    assert body["total_entries"] == 3
    assert body["by_kind"]["onramp_quote"]["count"] == 2
    assert body["by_kind"]["onramp_quote"]["total_usd"] == 300.0


# ── GET /admin/fiat-compliance/{entry_id} ────────────────


def test_get_one_503_when_unwired():
    resp = _client(None).get("/admin/fiat-compliance/abc")
    assert resp.status_code == 503


def test_get_one_404_when_missing():
    resp = _client(FiatComplianceRing()).get(
        "/admin/fiat-compliance/nonexistent",
    )
    assert resp.status_code == 404


def test_get_one_happy_path():
    r = FiatComplianceRing()
    e = r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="OK",
    )
    resp = _client(r).get(
        f"/admin/fiat-compliance/{e.entry_id}",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["entry_id"] == e.entry_id
    assert body["user_id"] == "alice"


# ── Onramp quote records to ring ─────────────────────────


def test_onramp_quote_records_to_ring():
    ring = FiatComplianceRing()
    waas = _commissioned_waas()
    cli = _client(ring=ring, waas=waas)
    resp = cli.post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    assert resp.status_code == 200
    assert ring.count() == 1
    entry = ring.recent(limit=1)[0]
    assert entry.kind == "onramp_quote"
    assert entry.user_id == "alice"
    assert entry.usd_amount == 100.0
    assert entry.ftns_amount == 100.0


def test_onramp_explicit_address_records_with_empty_user():
    ring = FiatComplianceRing()
    cli = _client(ring=ring)
    cli.post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 50.0,
            "destination_address": "0xrecipient",
        },
    )
    entry = ring.recent(limit=1)[0]
    assert entry.user_id == ""
    assert entry.address == "0xrecipient"


def test_onramp_records_kyc_status():
    ring = FiatComplianceRing()
    waas = _commissioned_waas()
    kyc = KYCClient(
        vendor="persona", api_key="k",
        backend=type(
            "FakeKYC", (),
            {"initiate_session": lambda self, u, e, l: {
                "vendor_ref": "ref",
                "session_url": "url",
                "status": "INITIATED",
            }},
        )(),
    )
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    kyc.update_status("alice", KYC_STATUS_VERIFIED)
    cli = _client(ring=ring, waas=waas, kyc=kyc)
    cli.post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    entry = ring.recent(limit=1)[0]
    assert entry.kyc_status == "VERIFIED"


# ── Offramp quote records to ring ────────────────────────


class _FakeLedger:
    _connected_address = "0xdefault"
    _decimals = 18

    async def get_balance(self, addr):
        return 1000.0


def test_offramp_quote_records_to_ring():
    ring = FiatComplianceRing()
    cli = _client(ring=ring, ftns_ledger=_FakeLedger())
    cli.post(
        "/wallet/offramp/quote",
        json={"usd_amount": 50.0},
    )
    assert ring.count() == 1
    entry = ring.recent(limit=1)[0]
    assert entry.kind == "offramp_quote"
    assert entry.usd_amount == 50.0


# ── Gasless transfer records to ring ─────────────────────


def test_gasless_quote_records_to_ring():
    ring = FiatComplianceRing()
    waas = _commissioned_waas()
    pm = PaymasterClient(
        endpoint="x", api_key="y",
        backend=FakePaymasterBackend(),
    )
    cli = _client(ring=ring, waas=waas, paymaster=pm)
    cli.post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "to_address": "0xrecipient",
            "ftns_amount": "10.5",
            "dry_run": True,
        },
    )
    assert ring.count() == 1
    entry = ring.recent(limit=1)[0]
    assert entry.kind == "gasless_transfer_quote"
    assert entry.user_id == "alice"
    assert entry.ftns_amount == 10.5


def test_gasless_execute_records_to_ring():
    ring = FiatComplianceRing()
    waas = _commissioned_waas()
    pm = PaymasterClient(
        endpoint="x", api_key="y",
        backend=FakePaymasterBackend(),
    )
    cli = _client(ring=ring, waas=waas, paymaster=pm)
    cli.post(
        "/wallet/transfer/gasless",
        json={
            "from_user_id": "alice",
            "to_address": "0xrecipient",
            "ftns_amount": "10.5",
            "dry_run": False,
        },
    )
    assert ring.count() == 1
    entry = ring.recent(limit=1)[0]
    assert entry.kind == "gasless_transfer_execute"
    assert entry.tx_hash == "0xtxhash"


# ── Ring failure does not break primary surface ──────────


def test_ring_disabled_does_not_break_onramp_quote():
    """If the compliance ring isn't wired, quote endpoints
    must still function — never deny service over telemetry."""
    waas = _commissioned_waas()
    cli = _client(ring=None, waas=waas)
    resp = cli.post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    assert resp.status_code == 200
