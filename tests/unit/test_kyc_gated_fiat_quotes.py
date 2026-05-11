"""Sprint 281 — KYC gating on onramp + offramp quotes.

Wires the sprint-280 KYCClient.is_verified() predicate into
the fiat quote endpoints. Mirrors the proven claim_required
UX pattern from offramp/quote: surface the prerequisite in
the artifact (kyc_required + kyc_status + kyc_session_url)
without failing — the preview is still useful, just flagged.

Onramp: gates on destination_user_id (WaaS-resolved). Explicit
destination_address bypasses (operator-side reasoning: if the
caller is paying directly to an address, KYC is the caller's
responsibility, not PRSM's).

Offramp: gates on optional source_user_id. When absent
(legacy callers using `address` query param), no gating —
preserves backwards compat.

Field contract:
  kyc_required:    bool
  kyc_status:      str  ("VERIFIED" | "PENDING_COMMISSION" |
                         "NOT_STARTED" | "INITIATED" | "PENDING" |
                         "REJECTED" | "EXPIRED")
  kyc_session_url: Optional[str] (None when no INITIATED session)
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient
from prsm.economy.web3.kyc_client import (
    KYCClient, KYC_STATUS_VERIFIED, KYC_STATUS_REJECTED,
)
from prsm.node.api import create_api_app


class FakeWaasBackend:
    def create_wallet(self, user_id, email):
        return {
            "wallet_id": f"w-{user_id}",
            "address": f"0x{user_id:0>40}",
            "network": "base-mainnet",
        }


class FakeKYCBackend:
    def initiate_session(self, user_id, email, level):
        return {
            "vendor_ref": f"persona-{user_id}",
            "session_url": f"https://persona.example/v/{user_id}",
            "status": "INITIATED",
        }


def _client(waas=None, kyc=None, ftns_ledger=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = ftns_ledger
    node._coinbase_waas_client = waas
    node._kyc_client = kyc
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


def _commissioned_kyc():
    return KYCClient(
        vendor="persona", api_key="k",
        backend=FakeKYCBackend(),
    )


# ── Onramp quote — KYC gating ────────────────────────────


def test_onramp_no_kyc_client_no_gating():
    """When _kyc_client is unwired, kyc_required is False
    regardless of destination_user_id. Preserves pre-sprint-281
    behavior."""
    resp = _client(_commissioned_waas(), None).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["kyc_required"] is False
    assert body["kyc_status"] is None


def test_onramp_unverified_user_flags_kyc_required():
    """destination_user_id resolves, but the user has no KYC
    record — kyc_required=true + status=NOT_STARTED."""
    waas = _commissioned_waas()
    kyc = _commissioned_kyc()  # no KYC initiated for alice yet
    resp = _client(waas, kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kyc_required"] is True
    assert body["kyc_status"] == "NOT_STARTED"
    assert body["kyc_session_url"] is None


def test_onramp_in_progress_user_flags_kyc_required_with_url():
    """User has INITIATED but not VERIFIED session — surface
    session_url so caller can resume the vendor flow."""
    waas = _commissioned_waas()
    kyc = _commissioned_kyc()
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    resp = _client(waas, kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["kyc_required"] is True
    assert body["kyc_status"] == "INITIATED"
    assert "persona.example/v/alice" in body["kyc_session_url"]


def test_onramp_verified_user_kyc_not_required():
    waas = _commissioned_waas()
    kyc = _commissioned_kyc()
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    kyc.update_status("alice", KYC_STATUS_VERIFIED)
    resp = _client(waas, kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["kyc_required"] is False
    assert body["kyc_status"] == "VERIFIED"


def test_onramp_rejected_user_flags_kyc_required():
    waas = _commissioned_waas()
    kyc = _commissioned_kyc()
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    kyc.update_status("alice", KYC_STATUS_REJECTED)
    resp = _client(waas, kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["kyc_required"] is True
    assert body["kyc_status"] == "REJECTED"


def test_onramp_explicit_address_bypasses_kyc():
    """Explicit destination_address — KYC is caller's
    responsibility, not PRSM's. No gating."""
    kyc = _commissioned_kyc()  # no records
    resp = _client(None, kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_address": "0xabc",
        },
    )
    body = resp.json()
    assert body["kyc_required"] is False
    assert body["kyc_status"] is None


# ── Offramp quote — KYC gating via source_user_id ────────


class _FakeLedger:
    """Minimal ftns_ledger stub for offramp quote tests."""
    _connected_address = "0xdefault"
    _decimals = 18

    async def get_balance(self, addr):
        return 1000.0


def test_offramp_legacy_no_source_user_id_no_gating():
    """Existing callers using `address` query param — no KYC
    field touched. Backwards-compat preserved."""
    ledger = _FakeLedger()
    kyc = _commissioned_kyc()
    resp = _client(
        waas=None, kyc=kyc, ftns_ledger=ledger,
    ).post(
        "/wallet/offramp/quote",
        json={"usd_amount": 50.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("kyc_required") is False
    assert body.get("kyc_status") is None


def test_offramp_source_user_id_unverified_flags_kyc():
    waas = _commissioned_waas()
    kyc = _commissioned_kyc()
    ledger = _FakeLedger()
    resp = _client(waas, kyc, ledger).post(
        "/wallet/offramp/quote",
        json={
            "usd_amount": 50.0,
            "source_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["kyc_required"] is True
    assert body["kyc_status"] == "NOT_STARTED"


def test_offramp_source_user_id_verified_no_gating():
    waas = _commissioned_waas()
    kyc = _commissioned_kyc()
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    kyc.update_status("alice", KYC_STATUS_VERIFIED)
    ledger = _FakeLedger()
    resp = _client(waas, kyc, ledger).post(
        "/wallet/offramp/quote",
        json={
            "usd_amount": 50.0,
            "source_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["kyc_required"] is False
    assert body["kyc_status"] == "VERIFIED"


def test_offramp_source_user_id_resolves_address_via_waas():
    """source_user_id should resolve to the user's WaaS
    address (consistent with onramp behavior)."""
    waas = _commissioned_waas()
    kyc = _commissioned_kyc()
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")
    kyc.update_status("alice", KYC_STATUS_VERIFIED)
    ledger = _FakeLedger()
    resp = _client(waas, kyc, ledger).post(
        "/wallet/offramp/quote",
        json={
            "usd_amount": 50.0,
            "source_user_id": "alice",
        },
    )
    body = resp.json()
    # source_address should reflect the WaaS-resolved address
    assert body["source_address"].startswith("0x")
    assert body["source_address"] != "0xdefault"


def test_offramp_source_user_id_404_when_no_waas_wallet():
    """source_user_id but no WaaS record — same 404 path as
    onramp."""
    kyc = _commissioned_kyc()
    ledger = _FakeLedger()
    # No WaaS client at all
    resp = _client(None, kyc, ledger).post(
        "/wallet/offramp/quote",
        json={
            "usd_amount": 50.0,
            "source_user_id": "alice",
        },
    )
    assert resp.status_code == 503  # WaaS required to resolve


def test_offramp_source_user_id_404_with_waas_but_no_record():
    waas = CoinbaseWaaSClient()  # nothing provisioned
    kyc = _commissioned_kyc()
    ledger = _FakeLedger()
    resp = _client(waas, kyc, ledger).post(
        "/wallet/offramp/quote",
        json={
            "usd_amount": 50.0,
            "source_user_id": "ghost",
        },
    )
    assert resp.status_code == 404


# ── MCP renderer — surfaces kyc_required prereq block ────


import pytest
from unittest.mock import AsyncMock, patch
from prsm.mcp_server import (
    handle_coinbase_onramp_initiate,
    handle_coinbase_offramp_initiate,
)


@pytest.mark.asyncio
async def test_onramp_mcp_renders_kyc_required_block():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "status": "PENDING_COMMISSION",
            "requested_usd": 100.0,
            "destination_user_id": "alice",
            "destination_address": "0xabc",
            "ftns_to_receive": 100.0,
            "usd_rate": 1.0,
            "kyc_required": True,
            "kyc_status": "INITIATED",
            "kyc_session_url": "https://persona.example/v/alice",
            "quote": {
                "usd_in": 100.0, "usdc_acquired": 100.0,
                "ftns_received": 100.0,
                "onramp_route": "coinbase-cdp",
                "swap_route": "aerodrome",
                "payment_method_alias": "primary",
            },
            "note": "preview",
        }),
    ):
        r = await handle_coinbase_onramp_initiate({
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        })
    # KYC prerequisite block should be present
    assert "KYC" in r or "kyc" in r.lower()
    assert "persona.example/v/alice" in r
    assert "INITIATED" in r or "complete" in r.lower()


@pytest.mark.asyncio
async def test_onramp_mcp_no_kyc_block_when_verified():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "status": "PENDING_COMMISSION",
            "requested_usd": 100.0,
            "destination_user_id": "alice",
            "destination_address": "0xabc",
            "ftns_to_receive": 100.0,
            "usd_rate": 1.0,
            "kyc_required": False,
            "kyc_status": "VERIFIED",
            "kyc_session_url": None,
            "quote": {
                "usd_in": 100.0, "usdc_acquired": 100.0,
                "ftns_received": 100.0,
                "onramp_route": "coinbase-cdp",
                "swap_route": "aerodrome",
                "payment_method_alias": "primary",
            },
            "note": "preview",
        }),
    ):
        r = await handle_coinbase_onramp_initiate({
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        })
    # No "kyc required" prerequisite line when not gated
    assert "kyc required" not in r.lower()


@pytest.mark.asyncio
async def test_offramp_mcp_renders_kyc_required_block():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "requested_usd": 50.0,
            "source_address": "0xabc",
            "source_balance_ftns": 1000.0,
            "source_balance_usd": 1000.0,
            "available_ftns": 1000.0,
            "available_usd": 1000.0,
            "claimable_royalties_ftns": 0.0,
            "claim_required": False,
            "claim_amount_ftns": 0.0,
            "kyc_required": True,
            "kyc_status": "NOT_STARTED",
            "kyc_session_url": None,
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
            "commission_gate_note": "x",
        }),
    ):
        r = await handle_coinbase_offramp_initiate({
            "usd_amount": 50.0,
        })
    assert "KYC" in r or "kyc" in r.lower()
    assert "NOT_STARTED" in r or "kyc required" in r.lower()
