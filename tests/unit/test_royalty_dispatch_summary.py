"""Sprint 265 — /admin/royalty-dispatch-summary aggregate view.

The sprint-249 audit ring + endpoint surface per-row history.
Operators tracking paid-out volume had to page through entries
to compute counts. This sprint adds an aggregate summary
endpoint + matching MCP wrapper.

Returned fields:
  - total: ring entry count
  - status_counts: {sent, failed, skipped_no_record,
    skipped_bad_hash, skipped_zero_amount}
  - total_sent_wei: sum of gross_wei across status=sent entries
  - by_allocation_mode: {uniform: N, rate_weighted: N}
  - earliest_ts, latest_ts: timestamp bookends
"""
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
    resp = _client(None).get("/admin/royalty-dispatch-summary")
    assert resp.status_code == 503


def test_empty_ring_returns_zeroes():
    resp = _client(RoyaltyDispatchRing()).get(
        "/admin/royalty-dispatch-summary",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["total_sent_wei"] == 0
    assert body["status_counts"] == {}
    assert body["by_allocation_mode"] == {}
    assert body["earliest_ts"] is None
    assert body["latest_ts"] is None


def test_aggregates_counts_and_volume():
    r = RoyaltyDispatchRing()
    r.append(
        job_id="j1", cid="c1", status="sent",
        tx_hash="0x1", gross_wei=100, timestamp=100.0,
        allocation_mode="uniform",
    )
    r.append(
        job_id="j2", cid="c2", status="sent",
        tx_hash="0x2", gross_wei=300, timestamp=200.0,
        allocation_mode="uniform",
    )
    r.append(
        job_id="j3", cid="c3", status="failed",
        tx_hash=None, gross_wei=500, error="rpc",
        timestamp=300.0, allocation_mode="rate_weighted",
    )
    r.append(
        job_id="j4", cid="c4", status="skipped_zero_amount",
        tx_hash=None, gross_wei=0, timestamp=400.0,
        allocation_mode="rate_weighted",
    )
    resp = _client(r).get("/admin/royalty-dispatch-summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 4
    # total_sent_wei = sum of gross_wei across status="sent" only
    assert body["total_sent_wei"] == 400
    assert body["status_counts"] == {
        "sent": 2, "failed": 1, "skipped_zero_amount": 1,
    }
    assert body["by_allocation_mode"] == {
        "uniform": 2, "rate_weighted": 2,
    }
    assert body["earliest_ts"] == 100.0
    assert body["latest_ts"] == 400.0


def test_handles_legacy_entries_without_allocation_mode():
    """Pre-258 entries lack allocation_mode (None). Group under
    a special 'unknown' bucket so the count survives."""
    r = RoyaltyDispatchRing()
    r.append(
        job_id="j1", cid="c1", status="sent",
        tx_hash="0x1", gross_wei=100, timestamp=100.0,
        # allocation_mode omitted → None
    )
    resp = _client(r).get("/admin/royalty-dispatch-summary")
    body = resp.json()
    assert body["by_allocation_mode"] == {"unknown": 1}
