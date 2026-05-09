"""POST /wallet/royalty/claim — backend endpoint for royalty
withdrawal.

Closes the loop on the offramp-quote claim_required path
(shipped same session): when /wallet/offramp/quote returns
`claim_required: True`, the operator now has a backend endpoint
(and MCP tool composing on top) to execute the claim.

Behavior:
  - 503 if RoyaltyDistributorClient not wired
  - dry_run=True (default): read claimable + return artifact
    without on-chain action; status="DRY_RUN"
  - dry_run=False: call client.claim(); status="EXECUTED" with
    tx_hash + amount_claimed
  - claimable=0 + dry_run=False: skip the claim() call (avoids
    on-chain ZeroClaim revert + gas burn); status="SKIPPED_ZERO"
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node(*, claimable_wei=None, claim_returns=None, claim_raises=None,
          decimals=18, address="0x" + "11" * 20):
    node = MagicMock()
    node.identity.node_id = "test-node"
    if claimable_wei is None:
        node._royalty_distributor_client = None
    else:
        client = MagicMock()
        client.claimable = MagicMock(return_value=claimable_wei)
        if claim_raises is not None:
            client.claim = MagicMock(side_effect=claim_raises)
        else:
            client.claim = MagicMock(
                return_value=claim_returns or ("0xabcd" + "00" * 30, "OK"),
            )
        # The client carries decimals indirectly via ftns_ledger; expose
        # it where the endpoint reads it.
        ftns_ledger = MagicMock()
        ftns_ledger._is_initialized = True
        ftns_ledger._connected_address = address
        ftns_ledger._decimals = decimals
        node.ftns_ledger = ftns_ledger
        node._royalty_distributor_client = client
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Service availability
# ──────────────────────────────────────────────────────────────────────


class TestRoyaltyClaimAvailability:
    def test_503_when_client_not_wired(self):
        node = _node()
        resp = _client(node).post("/wallet/royalty/claim", json={})
        assert resp.status_code == 503
        assert "royalty" in resp.json()["detail"].lower() or \
            "distributor" in resp.json()["detail"].lower()


# ──────────────────────────────────────────────────────────────────────
# Dry-run path (default)
# ──────────────────────────────────────────────────────────────────────


class TestRoyaltyClaimDryRun:
    def test_default_is_dry_run(self):
        """Without dry_run param, endpoint defaults to dry_run=True
        — no on-chain action."""
        node = _node(claimable_wei=5 * 10**18)
        resp = _client(node).post("/wallet/royalty/claim", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "DRY_RUN"
        assert body["claimable_ftns"] == 5.0
        assert "tx_hash" not in body or body["tx_hash"] is None
        # claim() was NOT called.
        node._royalty_distributor_client.claim.assert_not_called()

    def test_explicit_dry_run_true(self):
        node = _node(claimable_wei=2 * 10**18)
        resp = _client(node).post(
            "/wallet/royalty/claim", json={"dry_run": True},
        )
        body = resp.json()
        assert body["status"] == "DRY_RUN"
        assert body["claimable_ftns"] == 2.0
        node._royalty_distributor_client.claim.assert_not_called()

    def test_dry_run_with_zero_claimable(self):
        node = _node(claimable_wei=0)
        resp = _client(node).post("/wallet/royalty/claim", json={})
        body = resp.json()
        assert body["status"] == "DRY_RUN"
        assert body["claimable_ftns"] == 0.0


# ──────────────────────────────────────────────────────────────────────
# Execute path
# ──────────────────────────────────────────────────────────────────────


class TestRoyaltyClaimExecute:
    def test_execute_with_positive_claimable_calls_claim(self):
        node = _node(
            claimable_wei=3 * 10**18,
            claim_returns=("0x" + "ab" * 32, "OK"),
        )
        resp = _client(node).post(
            "/wallet/royalty/claim", json={"dry_run": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "EXECUTED"
        assert body["claimable_ftns"] == 3.0
        assert body["amount_claimed_ftns"] == 3.0
        assert body["tx_hash"] == "0x" + "ab" * 32
        # claim() was called.
        node._royalty_distributor_client.claim.assert_called_once()

    def test_execute_with_zero_claimable_skips_call(self):
        """0 claimable → ZeroClaim revert on-chain. Skip the call
        entirely + return a clear status so the operator knows
        no gas was burned."""
        node = _node(claimable_wei=0)
        resp = _client(node).post(
            "/wallet/royalty/claim", json={"dry_run": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "SKIPPED_ZERO"
        assert body["claimable_ftns"] == 0.0
        assert body["amount_claimed_ftns"] == 0.0
        # claim() NOT called.
        node._royalty_distributor_client.claim.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# Failure modes
# ──────────────────────────────────────────────────────────────────────


class TestRoyaltyClaimFailures:
    def test_claim_raising_returns_502(self):
        """If claim() raises (e.g., RPC error, on-chain revert),
        surface as 502 Bad Gateway with the error message."""
        node = _node(
            claimable_wei=5 * 10**18,
            claim_raises=RuntimeError("rpc unreachable"),
        )
        resp = _client(node).post(
            "/wallet/royalty/claim", json={"dry_run": False},
        )
        assert resp.status_code == 502
        assert "rpc unreachable" in resp.json()["detail"]

    def test_claimable_raising_returns_502(self):
        """If claimable() raises, treat as RPC failure."""
        node = _node(claimable_wei=5 * 10**18)
        node._royalty_distributor_client.claimable = MagicMock(
            side_effect=RuntimeError("read failed"),
        )
        resp = _client(node).post("/wallet/royalty/claim", json={})
        assert resp.status_code == 502
