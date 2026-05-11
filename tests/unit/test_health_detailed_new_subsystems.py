"""Sprint 255 — /health/detailed surfaces receipt_store +
royalty_dispatch_ring subsystem state.

Pre-fix the new state added in sprints 242 + 249 was invisible
to /health/detailed. Operators monitoring per-subsystem
readiness saw no row for either.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.receipt_store import ReceiptStore
from prsm.node.royalty_dispatch_log import RoyaltyDispatchRing


def _client(receipt_store=None, royalty_ring=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._provenance_client = None
    node._royalty_distributor_client = None
    node._receipt_store = receipt_store
    node._royalty_dispatch_ring = royalty_ring
    for attr in (
        "_escrow_cleanup_task", "_heartbeat_scheduler_task",
        "_compensation_scheduler_task",
        "_key_distribution_watcher_task",
        "_storage_slashing_watcher_task",
        "_compensation_distributor_watcher_task",
    ):
        setattr(node, attr, None)
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_receipt_store_subsystem_present_when_wired():
    s = ReceiptStore()
    s.put("j1", {"job_id": "j1"})
    resp = _client(receipt_store=s).get("/health/detailed")
    assert resp.status_code in (200, 503)
    body = resp.json()
    sub = body["subsystems"]["receipt_store"]
    assert sub["available"] is True
    assert sub["status"] == "ok"
    assert sub["count"] == 1
    assert sub["persisted"] is False


def test_receipt_store_subsystem_not_wired():
    resp = _client(receipt_store=None).get("/health/detailed")
    body = resp.json()
    sub = body["subsystems"]["receipt_store"]
    assert sub["available"] is False
    assert sub["status"] == "not_wired"


def test_royalty_ring_subsystem_present_when_wired():
    r = RoyaltyDispatchRing()
    r.append(
        job_id="j1", cid="c1", status="sent",
        tx_hash="0xtx", gross_wei=1,
    )
    resp = _client(royalty_ring=r).get("/health/detailed")
    body = resp.json()
    sub = body["subsystems"]["royalty_dispatch_ring"]
    assert sub["available"] is True
    assert sub["status"] == "ok"
    assert sub["count"] == 1
    assert sub["persisted"] is False


def test_royalty_ring_subsystem_not_wired():
    resp = _client(royalty_ring=None).get("/health/detailed")
    body = resp.json()
    sub = body["subsystems"]["royalty_dispatch_ring"]
    assert sub["available"] is False
    assert sub["status"] == "not_wired"


def test_persisted_true_when_dir_set(tmp_path):
    s = ReceiptStore(persist_dir=tmp_path)
    resp = _client(receipt_store=s).get("/health/detailed")
    sub = resp.json()["subsystems"]["receipt_store"]
    assert sub["persisted"] is True
