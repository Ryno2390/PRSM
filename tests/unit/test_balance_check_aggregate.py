"""GET /balance/onchain — aggregate-source extension.

Closes the audit-prep §7.23 honest-scope deferred item:
v1 read only on-chain FTNS; v2 aggregates across multiple sources
(on-chain + claimable royalties + escrowed in pending jobs).

Forward-compat invariants:
  - Existing v1 fields preserved bit-identically (`balance_wei`,
    `balance_ftns`, `usd_rate`, `usd_equivalent`, `source`,
    `address`).
  - New fields are additive: `claimable_royalties_ftns`,
    `escrowed_ftns`, `total_ftns`, `total_usd_equivalent`,
    `sources` (per-source breakdown with `available` flag).
  - Sources unavailable (client not wired, RPC error, no escrow
    manager) report `available: false` and contribute 0 to
    aggregates — don't crash the endpoint.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.payment_escrow import EscrowEntry, EscrowStatus


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _node_with_ftns_only(*, balance_ftns: float = 100.0,
                          address: str = "0x" + "11" * 20):
    """Legacy v1 node — only ftns_ledger wired. Aggregate-source
    sources unavailable; backwards-compat path."""
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = "test-node"
    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = address
    ftns_ledger._decimals = 18
    ftns_ledger.get_balance = AsyncMock(return_value=balance_ftns)
    node.ftns_ledger = ftns_ledger
    # Critical: explicit None to prevent MagicMock auto-attribute spawn.
    node._royalty_distributor_client = None
    node._payment_escrow = None
    return node


def _node_with_all_sources(
    *,
    balance_ftns: float = 100.0,
    claimable_wei: int = 0,
    escrows: list = None,
    address: str = "0x" + "11" * 20,
):
    """Fully-wired node — all 3 aggregate sources available."""
    node = _node_with_ftns_only(balance_ftns=balance_ftns, address=address)

    royalty_client = MagicMock()
    royalty_client.claimable = MagicMock(return_value=claimable_wei)
    node._royalty_distributor_client = royalty_client

    payment_escrow = MagicMock()
    payment_escrow.list_escrows_by_requester = MagicMock(
        return_value=escrows or [],
    )
    node._payment_escrow = payment_escrow
    return node


def _client(node):
    app = create_api_app(node, enable_security=False)
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────
# Backwards-compat: legacy v1 fields preserved
# ──────────────────────────────────────────────────────────────────────


class TestBackwardsCompat:
    def test_legacy_v1_fields_present_when_only_ftns_wired(self):
        node = _node_with_ftns_only(balance_ftns=42.5)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}):
            response = _client(node).get("/balance/onchain")
        assert response.status_code == 200
        body = response.json()
        # All v1 fields still present + identical semantics.
        assert body["balance_ftns"] == 42.5
        assert body["balance_wei"] == int(42.5 * 10**18)
        assert body["usd_rate"] == 1.0
        assert body["usd_equivalent"] == 42.5
        assert body["source"] == "onchain"

    def test_v1_legacy_does_not_break_on_unwired_aggregate_sources(self):
        """When royalty client + escrow manager aren't wired, the
        endpoint must NOT crash — fall back to ftns-only and report
        sources as unavailable."""
        node = _node_with_ftns_only(balance_ftns=10.0)
        response = _client(node).get("/balance/onchain")
        assert response.status_code == 200
        body = response.json()
        # Aggregate fields present but reflect "no claimable, no escrow".
        assert body["claimable_royalties_ftns"] == 0.0
        assert body["escrowed_ftns"] == 0.0
        # Total = balance only.
        assert body["total_ftns"] == 10.0


# ──────────────────────────────────────────────────────────────────────
# New aggregate fields when sources are wired
# ──────────────────────────────────────────────────────────────────────


class TestAggregateSources:
    def test_claimable_royalties_aggregated(self):
        # 5 FTNS in claimable royalties (5 * 10^18 wei).
        node = _node_with_all_sources(
            balance_ftns=10.0,
            claimable_wei=5 * 10**18,
        )
        response = _client(node).get("/balance/onchain")
        assert response.status_code == 200
        body = response.json()
        assert body["claimable_royalties_ftns"] == 5.0
        # Total = 10 (on-chain) + 5 (claimable) + 0 (no escrows) = 15.
        assert body["total_ftns"] == 15.0

    def test_escrowed_ftns_aggregated(self):
        addr = "0x" + "11" * 20
        escrows = [
            EscrowEntry(
                escrow_id="e1", job_id="j1",
                requester_id=addr, amount=2.5,
                status=EscrowStatus.PENDING,
            ),
            EscrowEntry(
                escrow_id="e2", job_id="j2",
                requester_id=addr, amount=1.5,
                status=EscrowStatus.PENDING,
            ),
        ]
        node = _node_with_all_sources(
            balance_ftns=10.0,
            escrows=escrows,
        )
        response = _client(node).get("/balance/onchain")
        body = response.json()
        assert body["escrowed_ftns"] == 4.0
        # Total = 10 + 0 + 4 = 14.
        assert body["total_ftns"] == 14.0

    def test_all_three_sources_aggregated(self):
        addr = "0x" + "11" * 20
        escrows = [
            EscrowEntry(
                escrow_id="e1", job_id="j1",
                requester_id=addr, amount=3.0,
                status=EscrowStatus.PENDING,
            ),
        ]
        node = _node_with_all_sources(
            balance_ftns=10.0,
            claimable_wei=2 * 10**18,
            escrows=escrows,
        )
        response = _client(node).get("/balance/onchain")
        body = response.json()
        assert body["balance_ftns"] == 10.0
        assert body["claimable_royalties_ftns"] == 2.0
        assert body["escrowed_ftns"] == 3.0
        assert body["total_ftns"] == 15.0

    def test_total_usd_equivalent_uses_aggregate_total(self):
        node = _node_with_all_sources(
            balance_ftns=10.0,
            claimable_wei=2 * 10**18,
        )
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "2.5"}):
            response = _client(node).get("/balance/onchain")
        body = response.json()
        # Total = 10 + 2 = 12; total_usd = 12 * 2.5 = 30.
        assert body["total_ftns"] == 12.0
        assert body["total_usd_equivalent"] == 30.0
        # Legacy usd_equivalent still on-chain-only = 25.
        assert body["usd_equivalent"] == 25.0


# ──────────────────────────────────────────────────────────────────────
# Sources breakdown
# ──────────────────────────────────────────────────────────────────────


class TestSourcesBreakdown:
    def test_sources_object_present_when_any_source_wired(self):
        node = _node_with_all_sources(balance_ftns=10.0)
        response = _client(node).get("/balance/onchain")
        body = response.json()
        assert "sources" in body
        sources = body["sources"]
        assert "onchain" in sources
        assert "claimable_royalties" in sources
        assert "escrowed" in sources

    def test_sources_report_availability_correctly(self):
        node = _node_with_all_sources(
            balance_ftns=10.0,
            claimable_wei=5 * 10**18,
        )
        response = _client(node).get("/balance/onchain")
        body = response.json()
        sources = body["sources"]
        assert sources["onchain"]["available"] is True
        assert sources["onchain"]["ftns"] == 10.0
        assert sources["claimable_royalties"]["available"] is True
        assert sources["claimable_royalties"]["ftns"] == 5.0
        assert sources["escrowed"]["available"] is True
        assert sources["escrowed"]["ftns"] == 0.0  # no escrows

    def test_sources_report_unavailable_when_clients_missing(self):
        node = _node_with_ftns_only(balance_ftns=10.0)
        response = _client(node).get("/balance/onchain")
        body = response.json()
        sources = body["sources"]
        assert sources["onchain"]["available"] is True
        # Royalty client + escrow manager weren't wired.
        assert sources["claimable_royalties"]["available"] is False
        assert sources["claimable_royalties"]["ftns"] == 0.0
        assert sources["escrowed"]["available"] is False
        assert sources["escrowed"]["ftns"] == 0.0


# ──────────────────────────────────────────────────────────────────────
# Failure modes
# ──────────────────────────────────────────────────────────────────────


class TestAggregateFailureModes:
    def test_royalty_client_raises_falls_back_gracefully(self):
        """If royalty_client.claimable() raises (e.g., RPC error),
        endpoint must NOT crash — report claimable as unavailable
        + log + continue serving the rest of the response."""
        node = _node_with_all_sources(balance_ftns=10.0)
        node._royalty_distributor_client.claimable = MagicMock(
            side_effect=RuntimeError("rpc unreachable"),
        )
        response = _client(node).get("/balance/onchain")
        assert response.status_code == 200
        body = response.json()
        assert body["claimable_royalties_ftns"] == 0.0
        assert body["sources"]["claimable_royalties"]["available"] is False
        # Other sources unaffected.
        assert body["balance_ftns"] == 10.0
        assert body["sources"]["onchain"]["available"] is True

    def test_escrow_listing_raises_falls_back_gracefully(self):
        node = _node_with_all_sources(balance_ftns=10.0)
        node._payment_escrow.list_escrows_by_requester = MagicMock(
            side_effect=RuntimeError("escrow store down"),
        )
        response = _client(node).get("/balance/onchain")
        assert response.status_code == 200
        body = response.json()
        assert body["escrowed_ftns"] == 0.0
        assert body["sources"]["escrowed"]["available"] is False
        # Other sources unaffected.
        assert body["balance_ftns"] == 10.0
