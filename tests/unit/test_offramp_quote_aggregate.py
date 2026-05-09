"""POST /wallet/offramp/quote — aggregate-source balance check.

Closes the false-422 regression that aggregate-source quoting on
/balance/onchain exposed: users with claimable royalties bridging
the gap between on-chain balance and requested USD were getting
"insufficient balance" rejections even though their total
spendable funds were sufficient.

Contract:
  - "Available" for offramp = on-chain FTNS + claimable royalties.
  - Escrowed FTNS does NOT count (locked in pending compute jobs).
  - When on-chain alone covers the request → 200, no claim needed.
  - When on-chain insufficient but on-chain + claimable covers →
    200 with `claim_required: True` + `claim_amount_ftns` set.
  - When even aggregate is insufficient → 422.
  - Royalty client unwired or RPC-flaked → fail-soft: treated as
    0 claimable, falls back to on-chain-only check.

Forward-compat: the response always includes `available_ftns` /
`available_usd` so consumers can reason about the aggregate;
legacy `source_balance_ftns` / `source_balance_usd` preserved
bit-identically as on-chain-only fields.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _node(
    *,
    balance_ftns: float = 100.0,
    claimable_wei: int | None = None,
    address: str = "0x" + "11" * 20,
):
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = "test-node"
    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = address
    ftns_ledger._decimals = 18
    ftns_ledger.get_balance = AsyncMock(return_value=balance_ftns)
    node.ftns_ledger = ftns_ledger
    # Suppress MagicMock auto-attr — same pattern as
    # test_balance_check_aggregate.
    node._payment_escrow = None
    if claimable_wei is None:
        node._royalty_distributor_client = None
    else:
        royalty_client = MagicMock()
        royalty_client.claimable = MagicMock(return_value=claimable_wei)
        node._royalty_distributor_client = royalty_client
    return node


def _client(node):
    app = create_api_app(node, enable_security=False)
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────
# Backwards-compat: legacy v1 fields preserved
# ──────────────────────────────────────────────────────────────────────


class TestBackwardsCompat:
    def test_legacy_response_shape_when_royalty_client_unwired(self):
        """No royalty client → response uses on-chain-only as both
        legacy fields AND aggregate fields. Behavior bit-identical
        to v1 except for the additive `available_*` mirror."""
        node = _node(balance_ftns=4200.0)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}):
            response = _client(node).post(
                "/wallet/offramp/quote",
                json={"usd_amount": 500.0},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["source_balance_ftns"] == 4200.0
        assert body["source_balance_usd"] == 4200.0
        # Aggregate fields == on-chain when no claimable wired.
        assert body["available_ftns"] == 4200.0
        assert body["available_usd"] == 4200.0
        assert body["claim_required"] is False

    def test_v1_422_unchanged_when_aggregate_also_insufficient(self):
        """Legacy 422 message format preserved."""
        node = _node(balance_ftns=10.0)
        response = _client(node).post(
            "/wallet/offramp/quote",
            json={"usd_amount": 500.0},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        # Legacy phrasing: "Insufficient balance"
        assert "insufficient" in detail.lower()


# ──────────────────────────────────────────────────────────────────────
# Aggregate-source behavior
# ──────────────────────────────────────────────────────────────────────


class TestAggregateAvailable:
    def test_onchain_alone_sufficient_no_claim_required(self):
        """When on-chain balance covers the request, claim_required
        is False even if claimable royalties exist."""
        node = _node(balance_ftns=600.0, claimable_wei=100 * 10**18)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}):
            response = _client(node).post(
                "/wallet/offramp/quote",
                json={"usd_amount": 500.0},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["claim_required"] is False
        assert body["claim_amount_ftns"] == 0.0
        assert body["available_ftns"] == 700.0  # 600 + 100
        assert body["source_balance_ftns"] == 600.0  # legacy field

    def test_onchain_insufficient_but_claimable_bridges_gap(self):
        """on-chain (10) + claimable (50) = 60 USD covers a 50 USD
        request. Returns 200 + claim_required: True + claim_amount =
        the FTNS shortfall on on-chain that needs claiming."""
        node = _node(balance_ftns=10.0, claimable_wei=50 * 10**18)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}):
            response = _client(node).post(
                "/wallet/offramp/quote",
                json={"usd_amount": 50.0},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["available_ftns"] == 60.0
        assert body["available_usd"] == 60.0
        assert body["claim_required"] is True
        # Need 50 FTNS for swap; have 10 on-chain → must claim 40.
        assert body["claim_amount_ftns"] == 40.0

    def test_aggregate_insufficient_returns_422_with_breakdown(self):
        """Even with claimable, aggregate insufficient → 422 with
        breakdown showing both on-chain AND claimable in error msg
        so user knows a claim won't help."""
        node = _node(balance_ftns=10.0, claimable_wei=20 * 10**18)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}):
            response = _client(node).post(
                "/wallet/offramp/quote",
                json={"usd_amount": 100.0},
            )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "insufficient" in detail.lower()
        # Both numbers visible in error so user can decide next step.
        assert "30" in detail  # 10 + 20 = 30 available
        assert "100" in detail  # requested

    def test_escrowed_ftns_not_counted_in_available(self):
        """FTNS held in PENDING escrows for in-flight compute jobs
        is locked — must NOT be counted as available for offramp.
        Even if escrow manager wired, available stays = on-chain +
        claimable only."""
        from prsm.node.payment_escrow import EscrowEntry, EscrowStatus
        addr = "0x" + "11" * 20
        node = _node(balance_ftns=10.0, claimable_wei=5 * 10**18,
                     address=addr)
        # Wire escrow with 100 FTNS locked — should NOT contribute.
        payment_escrow = MagicMock()
        payment_escrow.list_escrows_by_requester = MagicMock(
            return_value=[
                EscrowEntry(
                    escrow_id="e1", job_id="j1",
                    requester_id=addr, amount=100.0,
                    status=EscrowStatus.PENDING,
                ),
            ],
        )
        node._payment_escrow = payment_escrow
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}):
            response = _client(node).post(
                "/wallet/offramp/quote",
                json={"usd_amount": 50.0},
            )
        # Available = 10 on-chain + 5 claimable = 15. NOT 115.
        # Request for $50 → 422.
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# Failure modes — fail-soft
# ──────────────────────────────────────────────────────────────────────


class TestRoyaltyClientFailSoft:
    def test_royalty_rpc_error_falls_back_to_onchain_only(self):
        """If royalty_client.claimable() raises, treat claimable as
        0 (fail-soft) and continue with on-chain-only validation.
        Same fail-soft contract as /balance/onchain."""
        node = _node(balance_ftns=600.0, claimable_wei=100 * 10**18)
        node._royalty_distributor_client.claimable = MagicMock(
            side_effect=RuntimeError("rpc unreachable"),
        )
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}):
            response = _client(node).post(
                "/wallet/offramp/quote",
                json={"usd_amount": 500.0},
            )
        # On-chain alone covers (600 >= 500).
        assert response.status_code == 200
        body = response.json()
        assert body["available_ftns"] == 600.0
        assert body["claim_required"] is False

    def test_royalty_rpc_error_does_not_inflate_422(self):
        """When RPC errors AND on-chain is insufficient, 422 must
        NOT include phantom claimable in the error msg."""
        node = _node(balance_ftns=10.0, claimable_wei=200 * 10**18)
        node._royalty_distributor_client.claimable = MagicMock(
            side_effect=RuntimeError("rpc"),
        )
        response = _client(node).post(
            "/wallet/offramp/quote",
            json={"usd_amount": 100.0},
        )
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# Quote field semantics
# ──────────────────────────────────────────────────────────────────────


class TestQuoteFieldSemantics:
    def test_ftns_to_swap_uses_full_request_not_just_onchain(self):
        """ftns_to_swap reflects the FULL swap amount the user is
        quoting for, regardless of where the FTNS came from
        (existing on-chain vs. about-to-be-claimed). The CDP
        commission flow ultimately swaps the full ftns_to_swap."""
        node = _node(balance_ftns=10.0, claimable_wei=50 * 10**18)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}):
            response = _client(node).post(
                "/wallet/offramp/quote",
                json={"usd_amount": 50.0},
            )
        body = response.json()
        assert body["quote"]["ftns_to_swap"] == 50.0

    def test_status_remains_pending_commission_either_path(self):
        """No matter which path (no-claim or claim-required), v1
        status stays PENDING_COMMISSION until CDP ships."""
        node1 = _node(balance_ftns=600.0)
        r1 = _client(node1).post(
            "/wallet/offramp/quote",
            json={"usd_amount": 500.0},
        )
        assert r1.json()["status"] == "PENDING_COMMISSION"

        node2 = _node(balance_ftns=10.0, claimable_wei=100 * 10**18)
        r2 = _client(node2).post(
            "/wallet/offramp/quote",
            json={"usd_amount": 50.0},
        )
        assert r2.json()["status"] == "PENDING_COMMISSION"
