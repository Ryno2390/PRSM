"""Sprint 163 — 503 detail messages on admin trigger endpoints
should mention the sprint-144 canonical-fallback path, not just
the explicit PRSM_*_ADDRESS env-var route.

Pre-fix the messages told operators to "Set PRSM_*_ADDRESS +
FTNS_WALLET_PRIVATE_KEY" — but sprint 144 made that a fallback
path: a node with PRSM_NETWORK declared automatically wires the
canonical address from networks.py. So the only env var still
strictly required is FTNS_WALLET_PRIVATE_KEY (for the write
clients).

Updated detail messages give operators both options.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node_no_clients():
    """Node with no on-chain clients wired so 503 paths fire."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._royalty_distributor_client = None
    node._storage_slashing_client = None
    node._compensation_distributor_client = None
    node._provenance_client = None
    node._webhook_log = None
    node._operator_address = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def test_distribution_trigger_503_mentions_canonical_fallback():
    """Sprint 163 — operator-facing detail must surface both
    config paths (explicit env var OR PRSM_NETWORK)."""
    resp = _client(_node_no_clients()).post(
        "/admin/distribution/trigger",
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"].lower()
    assert "prsm_network" in detail


def test_heartbeat_trigger_503_mentions_canonical_fallback():
    resp = _client(_node_no_clients()).post(
        "/admin/heartbeat/trigger",
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"].lower()
    assert "prsm_network" in detail


def test_royalty_claim_503_mentions_canonical_fallback():
    """The /wallet/royalty/claim 503 detail also pre-dated sprint
    144 canonical fallback — should be updated."""
    resp = _client(_node_no_clients()).post(
        "/wallet/royalty/claim", json={},
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"].lower()
    assert "prsm_network" in detail
