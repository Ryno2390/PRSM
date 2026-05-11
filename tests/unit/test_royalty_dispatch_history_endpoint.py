"""Sprint 249 — GET /admin/royalty-dispatch-history endpoint."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.royalty_dispatch_log import RoyaltyDispatchRing


def _client(ring=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._royalty_dispatch_ring = ring
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_503_when_ring_unwired():
    resp = _client(None).get("/admin/royalty-dispatch-history")
    assert resp.status_code == 503


def test_returns_entries():
    ring = RoyaltyDispatchRing()
    ring.append(
        job_id="job-1", cid="cid-a", status="sent",
        tx_hash="0xtx", gross_wei=10**15,
    )
    resp = _client(ring).get("/admin/royalty-dispatch-history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["entries"][0]["job_id"] == "job-1"
    assert body["entries"][0]["status"] == "sent"


def test_status_filter_passes_through():
    ring = RoyaltyDispatchRing()
    ring.append(
        job_id="j1", cid="c1", status="sent",
        tx_hash="0x1", gross_wei=1,
    )
    ring.append(
        job_id="j2", cid="c2", status="failed",
        tx_hash=None, gross_wei=1, error="rpc",
    )
    resp = _client(ring).get(
        "/admin/royalty-dispatch-history?status=failed",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["status"] == "failed"


def test_job_id_filter_passes_through():
    ring = RoyaltyDispatchRing()
    ring.append(
        job_id="j1", cid="c1", status="sent",
        tx_hash="0x1", gross_wei=1,
    )
    ring.append(
        job_id="j2", cid="c2", status="sent",
        tx_hash="0x2", gross_wei=1,
    )
    resp = _client(ring).get(
        "/admin/royalty-dispatch-history?job_id=j2",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["job_id"] == "j2"


def test_422_on_bad_limit():
    resp = _client(RoyaltyDispatchRing()).get(
        "/admin/royalty-dispatch-history?limit=0",
    )
    assert resp.status_code == 422


def test_422_on_negative_offset():
    resp = _client(RoyaltyDispatchRing()).get(
        "/admin/royalty-dispatch-history?offset=-1",
    )
    assert resp.status_code == 422
