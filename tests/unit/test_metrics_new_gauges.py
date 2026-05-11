"""Sprint 254 — /metrics exposes prsm_receipt_store_size +
prsm_royalty_dispatch_ring_size gauges.

Pre-fix the new state added in sprints 242 (ReceiptStore) + 249
(RoyaltyDispatchRing) was invisible to Prometheus scrapers.
Operators tracking long-term audit-trail growth or on-chain
royalty dispatch volume had no time-series data.
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
    node._royalty_distributor_client = None
    node._receipt_store = receipt_store
    node._royalty_dispatch_ring = royalty_ring
    # Avoid task probes blowing up
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


def test_receipt_store_gauge_present():
    s = ReceiptStore()
    s.put("j1", {"job_id": "j1", "model_id": "m1"})
    s.put("j2", {"job_id": "j2", "model_id": "m1"})
    resp = _client(receipt_store=s).get("/metrics")
    assert resp.status_code == 200
    assert "prsm_receipt_store_size 2" in resp.text


def test_receipt_store_gauge_zero_when_empty():
    resp = _client(receipt_store=ReceiptStore()).get("/metrics")
    assert resp.status_code == 200
    assert "prsm_receipt_store_size 0" in resp.text


def test_receipt_store_gauge_absent_when_unwired():
    """Missing store = gauge omitted (not zero) — same pattern
    as the existing prsm_job_history_size probe."""
    resp = _client(receipt_store=None).get("/metrics")
    assert resp.status_code == 200
    assert "prsm_receipt_store_size" not in resp.text


def test_royalty_ring_gauge_present():
    r = RoyaltyDispatchRing()
    r.append(
        job_id="j1", cid="c1", status="sent",
        tx_hash="0xtx", gross_wei=10**15,
    )
    r.append(
        job_id="j2", cid="c2", status="failed",
        tx_hash=None, gross_wei=10**15, error="rpc",
    )
    resp = _client(royalty_ring=r).get("/metrics")
    assert resp.status_code == 200
    assert "prsm_royalty_dispatch_ring_size 2" in resp.text


def test_royalty_ring_gauge_absent_when_unwired():
    resp = _client(royalty_ring=None).get("/metrics")
    assert resp.status_code == 200
    assert "prsm_royalty_dispatch_ring_size" not in resp.text


def test_metrics_endpoint_text_plain():
    """Existing /metrics contract: text/plain content-type for
    Prometheus scraping."""
    resp = _client(receipt_store=ReceiptStore()).get("/metrics")
    assert resp.headers["content-type"].startswith("text/plain")
