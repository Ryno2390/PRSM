"""Phase 7-storage + Phase 8 canonical-match coverage in
/health/detailed.

Operators wiring StorageSlashing / CompensationDistributor /
KeyDistribution clients via env get a canonical-match signal
analogous to the existing ftns_ledger / royalty_distributor /
provenance_registry coverage. Closes the gap where watcher
subsystems showed `available=True, status=ok` without
verifying the wired contract address matches the canonical
networks.py pin.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._provenance_client = None
    node._royalty_distributor_client = None
    node._webhook_log = None
    node._slash_event_log = None
    node._heartbeat_log = None
    # Default no Phase 7/8 clients wired
    node._storage_slashing_client = None
    node._compensation_distributor_client = None
    node._key_distribution_client = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _mock_client(addr):
    c = MagicMock()
    c.address = addr
    return c


def test_storage_slashing_canonical_match_when_wired():
    node = _node()
    canonical_addr = "0x0e9cAfadCCCe0987C773B5FdFF295c2Aa6F03337"
    node._storage_slashing_client = _mock_client(canonical_addr)
    with patch.dict("os.environ", {"PRSM_NETWORK": "mainnet"}):
        resp = _client(node).get("/health/detailed")
    body = resp.json()
    sub = body["subsystems"].get("storage_slashing")
    assert sub is not None
    assert sub["available"] is True
    assert sub["wired_address"] == canonical_addr
    assert sub["canonical_match"] is True


def test_storage_slashing_canonical_mismatch_flagged():
    node = _node()
    node._storage_slashing_client = _mock_client("0xWRONG")
    with patch.dict("os.environ", {"PRSM_NETWORK": "mainnet"}):
        resp = _client(node).get("/health/detailed")
    sub = resp.json()["subsystems"]["storage_slashing"]
    assert sub["canonical_match"] is False


def test_storage_slashing_not_wired():
    node = _node()
    resp = _client(node).get("/health/detailed")
    sub = resp.json()["subsystems"]["storage_slashing"]
    assert sub["available"] is False
    assert sub["status"] == "not_wired"


def test_compensation_distributor_canonical_match():
    node = _node()
    canonical_addr = "0xa9551F5a3AeAB39cc8315AcD8caC2886Bd04f244"
    node._compensation_distributor_client = _mock_client(canonical_addr)
    with patch.dict("os.environ", {"PRSM_NETWORK": "mainnet"}):
        resp = _client(node).get("/health/detailed")
    sub = resp.json()["subsystems"]["compensation_distributor"]
    assert sub["wired_address"] == canonical_addr
    assert sub["canonical_match"] is True


def test_key_distribution_canonical_match():
    node = _node()
    canonical_addr = "0x51AF73Aa098E3b12Da78167c25c3d1D98059c8Ff"
    node._key_distribution_client = _mock_client(canonical_addr)
    with patch.dict("os.environ", {"PRSM_NETWORK": "mainnet"}):
        resp = _client(node).get("/health/detailed")
    sub = resp.json()["subsystems"]["key_distribution"]
    assert sub["wired_address"] == canonical_addr
    assert sub["canonical_match"] is True


def test_canonical_match_robust_to_missing_address_attr():
    node = _node()
    bad_client = MagicMock(spec=[])  # no `.address` attr
    node._storage_slashing_client = bad_client
    resp = _client(node).get("/health/detailed")
    sub = resp.json()["subsystems"]["storage_slashing"]
    # Should still be available but no wired_address surfaced
    assert sub["available"] is True
    assert "wired_address" not in sub
