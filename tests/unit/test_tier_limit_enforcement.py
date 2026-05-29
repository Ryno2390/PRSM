"""Sprint 285 — KYC-tier rolling-total enforcement.

Vendors (Persona / Onfido / regulators) enforce per-tier USD/day
limits. Without quote-time enforcement, our system would let a
basic-tier user submit unlimited fiat quotes — the audit log
catches it after the fact, but doesn't prevent it.

Tier limits (defaults, tunable via env):
  basic    — $1,000 / 24h  (FinCEN MSB low-threshold; selfie+ID
                              KYC is sufficient)
  enhanced — $10,000 / 24h (proof of address + source of funds
                              required to unlock)

Computation: per-user rolling sum of usd_amount across SETTLED
fiat-execute kinds (onramp_execute + offramp_execute) within the
window. Sp885 correction: quotes are NON-binding price checks and
do NOT count toward the limit (counting them let a user exhaust
their cap by price-shopping; and executes — the real settled
volume — recorded nothing). Gasless transfers excluded (FTNS-
denominated). KYC events excluded (zero USD).

New response fields on /wallet/onramp/quote +
/wallet/offramp/quote:
  tier_limit_usd:           per-day cap for the user's tier
  tier_limit_remaining_usd: cap minus rolling 24h total
  tier_limit_exceeded:      bool — True if requested > remaining
  tier_level:               "basic" | "enhanced" | None
                             (None when KYC not VERIFIED)
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient
from prsm.economy.web3.fiat_compliance_ring import FiatComplianceRing
from prsm.economy.web3.kyc_client import (
    KYCClient, KYC_STATUS_VERIFIED,
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


def _client(ring=None, waas=None, kyc=None, ftns_ledger=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = ftns_ledger
    node._fiat_compliance_ring = ring
    node._coinbase_waas_client = waas
    node._kyc_client = kyc
    node._kyc_webhook_replay_ring = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _verified_user(level="basic"):
    waas = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        backend=FakeWaasBackend(),
    )
    waas.provision_wallet(user_id="alice", email="a@x.io")
    kyc = KYCClient(
        vendor="persona", api_key="k", backend=FakeKYCBackend(),
    )
    kyc.initiate(user_id="alice", email="a@x.io", level=level)
    kyc.update_status("alice", KYC_STATUS_VERIFIED)
    return waas, kyc


# ── FiatComplianceRing.total_usd_for_user ────────────────


def test_total_usd_zero_for_unknown_user():
    r = FiatComplianceRing()
    assert r.total_usd_for_user("ghost") == 0.0


def test_total_usd_sums_recent_events():
    # Sp885: counts EXECUTES (settled volume), not quotes.
    r = FiatComplianceRing()
    now = time.time()
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="CONFIRMED",
        timestamp=now - 100,
    )
    r.record(
        kind="offramp_execute", user_id="alice",
        usd_amount=50.0, ftns_amount=50.0, status="CONFIRMED",
        timestamp=now - 50,
    )
    assert r.total_usd_for_user("alice") == 150.0


def test_total_usd_window_excludes_old_events():
    r = FiatComplianceRing()
    now = time.time()
    # 25h ago — outside default 24h window
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=1000.0, ftns_amount=1000.0, status="CONFIRMED",
        timestamp=now - 25 * 3600,
    )
    # 1h ago — inside window
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="CONFIRMED",
        timestamp=now - 3600,
    )
    assert r.total_usd_for_user("alice") == 100.0


def test_total_usd_filters_by_user():
    r = FiatComplianceRing()
    now = time.time()
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="CONFIRMED",
        timestamp=now,
    )
    r.record(
        kind="onramp_execute", user_id="bob",
        usd_amount=500.0, ftns_amount=500.0, status="CONFIRMED",
        timestamp=now,
    )
    assert r.total_usd_for_user("alice") == 100.0
    assert r.total_usd_for_user("bob") == 500.0


def test_total_usd_quotes_do_not_count_sp885():
    """Sp885 regression: non-binding quotes must NOT burn the limit
    (the bug this sprint fixed — they previously did)."""
    r = FiatComplianceRing()
    now = time.time()
    r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=900.0, ftns_amount=900.0, status="OK",
        timestamp=now,
    )
    r.record(
        kind="offramp_quote", user_id="alice",
        usd_amount=900.0, ftns_amount=900.0, status="OK",
        timestamp=now,
    )
    assert r.total_usd_for_user("alice") == 0.0


def test_total_usd_excludes_gasless_kinds():
    """Gasless transfers are FTNS-denominated; their
    usd_amount field is 0 anyway, but the kind-exclusion
    list makes the semantics explicit."""
    r = FiatComplianceRing()
    now = time.time()
    r.record(
        kind="gasless_transfer_execute", user_id="alice",
        usd_amount=999.0, ftns_amount=10.0, status="OK",
        timestamp=now,
    )
    # Even though usd_amount=999, gasless is excluded
    assert r.total_usd_for_user("alice") == 0.0


def test_total_usd_excludes_kyc_events():
    r = FiatComplianceRing()
    now = time.time()
    r.record(
        kind="kyc_initiate", user_id="alice",
        usd_amount=0.0, ftns_amount=0.0, status="OK",
        timestamp=now,
    )
    r.record(
        kind="kyc_status_change", user_id="alice",
        usd_amount=0.0, ftns_amount=0.0, status="OK",
        timestamp=now,
    )
    assert r.total_usd_for_user("alice") == 0.0


def test_total_usd_custom_window():
    r = FiatComplianceRing()
    now = time.time()
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="CONFIRMED",
        timestamp=now - 2 * 3600,  # 2h ago
    )
    # 24h window: included
    assert r.total_usd_for_user("alice", window_sec=86400) == 100.0
    # 1h window: excluded
    assert r.total_usd_for_user("alice", window_sec=3600) == 0.0


def test_total_usd_empty_user_id_returns_zero():
    """Explicit-address flows have empty user_id; querying
    by empty must NOT aggregate across all of them."""
    r = FiatComplianceRing()
    r.record(
        kind="onramp_quote", user_id="",
        address="0xabc", usd_amount=100.0,
        ftns_amount=100.0, status="OK",
    )
    assert r.total_usd_for_user("") == 0.0


# ── Onramp quote tier-limit fields ───────────────────────


def test_onramp_no_kyc_no_tier_fields_set():
    """Unverified user: existing kyc_required gate handles
    this. Tier fields should be neutral (None / 0)."""
    r = FiatComplianceRing()
    waas = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        backend=FakeWaasBackend(),
    )
    waas.provision_wallet(user_id="alice", email="a@x.io")
    resp = _client(ring=r, waas=waas).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["tier_level"] is None
    assert body["tier_limit_exceeded"] is False


def test_onramp_verified_basic_under_limit():
    r = FiatComplianceRing()
    waas, kyc = _verified_user(level="basic")
    resp = _client(ring=r, waas=waas, kyc=kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["tier_level"] == "basic"
    assert body["tier_limit_usd"] == 1000.0
    # remaining is PRE-quote (current daily budget), so 1000
    # for a fresh user. Consistent across under/over cases.
    assert body["tier_limit_remaining_usd"] == 1000.0
    assert body["tier_limit_exceeded"] is False


def test_onramp_verified_basic_exactly_at_limit():
    """$1000 requested with $0 spent — exactly at limit, NOT
    exceeded (inclusive)."""
    r = FiatComplianceRing()
    waas, kyc = _verified_user(level="basic")
    resp = _client(ring=r, waas=waas, kyc=kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 1000.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["tier_limit_exceeded"] is False


def test_onramp_verified_basic_over_limit():
    r = FiatComplianceRing()
    waas, kyc = _verified_user(level="basic")
    resp = _client(ring=r, waas=waas, kyc=kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 1500.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["tier_limit_exceeded"] is True
    assert body["tier_limit_usd"] == 1000.0
    assert body["tier_limit_remaining_usd"] == 1000.0


def test_onramp_verified_basic_pushed_over_by_rolling():
    """User already has $900 in 24h spend; requesting $200
    pushes over the $1000 cap."""
    r = FiatComplianceRing()
    waas, kyc = _verified_user(level="basic")
    now = time.time()
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=900.0, ftns_amount=900.0,
        status="CONFIRMED", timestamp=now - 100,
    )
    resp = _client(ring=r, waas=waas, kyc=kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 200.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["tier_limit_exceeded"] is True
    assert body["tier_limit_remaining_usd"] == 100.0


def test_onramp_verified_enhanced_higher_limit():
    r = FiatComplianceRing()
    waas, kyc = _verified_user(level="enhanced")
    resp = _client(ring=r, waas=waas, kyc=kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 5000.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["tier_level"] == "enhanced"
    assert body["tier_limit_usd"] == 10000.0
    assert body["tier_limit_exceeded"] is False


def test_onramp_env_var_overrides_basic_limit(monkeypatch):
    monkeypatch.setenv(
        "PRSM_KYC_TIER_LIMIT_BASIC_USD", "500",
    )
    r = FiatComplianceRing()
    waas, kyc = _verified_user(level="basic")
    resp = _client(ring=r, waas=waas, kyc=kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 750.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["tier_limit_usd"] == 500.0
    assert body["tier_limit_exceeded"] is True


def test_onramp_explicit_address_no_tier_check():
    r = FiatComplianceRing()
    resp = _client(ring=r).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100000.0,  # huge
            "destination_address": "0xabc",
        },
    )
    body = resp.json()
    assert body["tier_level"] is None
    assert body["tier_limit_exceeded"] is False


def test_onramp_no_compliance_ring_neutral_tier_fields():
    """If the compliance ring isn't wired, tier check can't
    compute rolling total — surface neutral fields rather
    than fail."""
    waas, kyc = _verified_user(level="basic")
    resp = _client(ring=None, waas=waas, kyc=kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_user_id": "alice",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # When ring is absent, tier_level still surfaces (we know
    # the KYC record's level) but rolling total is unknowable
    # → exceeded=False, remaining=full_limit
    assert body["tier_level"] == "basic"
    assert body["tier_limit_exceeded"] is False


# ── Offramp tier-limit symmetric ─────────────────────────


class _FakeLedger:
    _connected_address = "0xdefault"
    _decimals = 18
    async def get_balance(self, addr):
        # Plenty of balance so tier-limit check is the
        # binding constraint, not balance.
        return 50_000.0


def test_offramp_verified_basic_under_limit():
    r = FiatComplianceRing()
    waas, kyc = _verified_user(level="basic")
    resp = _client(
        ring=r, waas=waas, kyc=kyc,
        ftns_ledger=_FakeLedger(),
    ).post(
        "/wallet/offramp/quote",
        json={
            "usd_amount": 100.0,
            "source_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["tier_level"] == "basic"
    assert body["tier_limit_exceeded"] is False


def test_offramp_verified_basic_over_limit():
    r = FiatComplianceRing()
    waas, kyc = _verified_user(level="basic")
    resp = _client(
        ring=r, waas=waas, kyc=kyc,
        ftns_ledger=_FakeLedger(),
    ).post(
        "/wallet/offramp/quote",
        json={
            "usd_amount": 1500.0,
            "source_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["tier_limit_exceeded"] is True


def test_offramp_no_source_user_id_no_tier_check():
    """Legacy callers using `address` query param — no tier
    check applies."""
    r = FiatComplianceRing()
    resp = _client(
        ring=r, ftns_ledger=_FakeLedger(),
    ).post(
        "/wallet/offramp/quote",
        json={"usd_amount": 50.0},
    )
    body = resp.json()
    assert body.get("tier_level") is None
    assert body.get("tier_limit_exceeded") is False


# ── Rolling total spans BOTH directions ──────────────────


def test_rolling_total_combines_onramp_and_offramp():
    """A user's daily limit is across BOTH directions — they
    can't onramp $900 then offramp $900 on basic tier."""
    r = FiatComplianceRing()
    waas, kyc = _verified_user(level="basic")
    now = time.time()
    # User already onramped $700 + offramped $200 today (settled)
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=700.0, ftns_amount=700.0, status="CONFIRMED",
        timestamp=now - 1000,
    )
    r.record(
        kind="offramp_execute", user_id="alice",
        usd_amount=200.0, ftns_amount=200.0, status="CONFIRMED",
        timestamp=now - 500,
    )
    # Requesting $200 more → 700+200+200 = $1100 > $1000 limit
    resp = _client(ring=r, waas=waas, kyc=kyc).post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 200.0,
            "destination_user_id": "alice",
        },
    )
    body = resp.json()
    assert body["tier_limit_exceeded"] is True
    assert body["tier_limit_remaining_usd"] == 100.0


# ── MCP renderer surfaces tier-limit prereq block ────────


import pytest
from unittest.mock import AsyncMock, patch
from prsm.mcp_server import (
    handle_coinbase_onramp_initiate,
    handle_coinbase_offramp_initiate,
)


@pytest.mark.asyncio
async def test_onramp_mcp_renders_tier_limit_exceeded_block():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "status": "PENDING_COMMISSION",
            "requested_usd": 1500.0,
            "destination_user_id": "alice",
            "destination_address": "0xabc",
            "ftns_to_receive": 1500.0,
            "usd_rate": 1.0,
            "kyc_required": False,
            "kyc_status": "VERIFIED",
            "kyc_session_url": None,
            "tier_level": "basic",
            "tier_limit_usd": 1000.0,
            "tier_limit_remaining_usd": 1000.0,
            "tier_limit_exceeded": True,
            "quote": {
                "usd_in": 1500.0, "usdc_acquired": 1500.0,
                "ftns_received": 1500.0,
                "onramp_route": "coinbase-cdp",
                "swap_route": "aerodrome",
                "payment_method_alias": "primary",
            },
            "note": "preview",
        }),
    ):
        r = await handle_coinbase_onramp_initiate({
            "usd_amount": 1500.0,
            "destination_user_id": "alice",
        })
    # The tier-exceeded prereq block is present
    assert "tier" in r.lower()
    assert "1,000" in r or "1000" in r  # the limit
    assert "exceeded" in r.lower()


@pytest.mark.asyncio
async def test_offramp_mcp_renders_tier_limit_block():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "requested_usd": 1500.0,
            "source_address": "0xabc",
            "source_user_id": "alice",
            "source_balance_ftns": 5000.0,
            "source_balance_usd": 5000.0,
            "available_ftns": 5000.0,
            "available_usd": 5000.0,
            "claimable_royalties_ftns": 0.0,
            "claim_required": False,
            "claim_amount_ftns": 0.0,
            "kyc_required": False,
            "kyc_status": "VERIFIED",
            "kyc_session_url": None,
            "tier_level": "basic",
            "tier_limit_usd": 1000.0,
            "tier_limit_remaining_usd": 1000.0,
            "tier_limit_exceeded": True,
            "quote": {
                "ftns_to_swap": 1500.0,
                "usdc_received": 1500.0,
                "usd_settled": 1500.0,
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
            "usd_amount": 1500.0,
        })
    assert "tier" in r.lower()
    assert "exceeded" in r.lower()


@pytest.mark.asyncio
async def test_onramp_mcp_no_tier_block_when_under_limit():
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
            "tier_level": "basic",
            "tier_limit_usd": 1000.0,
            "tier_limit_remaining_usd": 900.0,
            "tier_limit_exceeded": False,
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
    # No exceeded prereq line
    assert "tier limit exceeded" not in r.lower()
