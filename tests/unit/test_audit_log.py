"""Audit log ring buffer + GET /audit/recent endpoint."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.audit_log import AuditEntry, AuditLogRing


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._royalty_distributor_client = None
    node._audit_log = AuditLogRing()
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# AuditLogRing
# ──────────────────────────────────────────────────────────────────────


class TestAuditLogRing:
    def test_append_and_read(self):
        ring = AuditLogRing()
        ring.append(
            method="POST", path="/x", requester="n1",
            status_code=200, request_id="r1",
        )
        results = ring.recent()
        assert len(results) == 1
        assert results[0].method == "POST"
        assert results[0].path == "/x"

    def test_most_recent_first(self):
        ring = AuditLogRing()
        for i in range(5):
            ring.append(
                method="POST", path=f"/x{i}", requester="n1",
                status_code=200, request_id=f"r{i}",
            )
        results = ring.recent()
        # Most recent first: x4, x3, x2, x1, x0
        assert [e.path for e in results] == [
            "/x4", "/x3", "/x2", "/x1", "/x0",
        ]

    def test_bounded_by_max_entries(self):
        ring = AuditLogRing(max_entries=3)
        for i in range(10):
            ring.append(
                method="POST", path=f"/x{i}", requester="n1",
                status_code=200, request_id=f"r{i}",
            )
        # Only the last 3 retained.
        results = ring.recent()
        assert len(results) == 3
        assert [e.path for e in results] == ["/x9", "/x8", "/x7"]

    def test_pagination(self):
        ring = AuditLogRing()
        for i in range(20):
            ring.append(
                method="POST", path=f"/x{i}", requester="n1",
                status_code=200, request_id=f"r{i}",
            )
        page1 = ring.recent(limit=5, offset=0)
        page2 = ring.recent(limit=5, offset=5)
        assert [e.path for e in page1] == [
            "/x19", "/x18", "/x17", "/x16", "/x15",
        ]
        assert [e.path for e in page2] == [
            "/x14", "/x13", "/x12", "/x11", "/x10",
        ]

    def test_validation(self):
        ring = AuditLogRing()
        with pytest.raises(ValueError):
            ring.recent(limit=0)
        with pytest.raises(ValueError):
            ring.recent(limit=10000)
        with pytest.raises(ValueError):
            ring.recent(offset=-1)
        with pytest.raises(ValueError):
            AuditLogRing(max_entries=0)

    def test_to_dict_round_trip(self):
        entry = AuditEntry(
            timestamp=1700000000.0, method="POST", path="/x",
            requester="n1", status_code=201, request_id="r1",
        )
        d = entry.to_dict()
        assert d["timestamp"] == 1700000000.0
        assert d["method"] == "POST"
        assert d["status_code"] == 201


# ──────────────────────────────────────────────────────────────────────
# GET /audit/recent endpoint
# ──────────────────────────────────────────────────────────────────────


class TestAuditEndpoint:
    def test_503_when_ring_not_wired(self):
        node = _node()
        node._audit_log = None
        resp = _client(node).get("/audit/recent")
        assert resp.status_code == 503

    def test_returns_empty_when_ring_fresh(self):
        node = _node()
        resp = _client(node).get("/audit/recent")
        assert resp.status_code == 200
        body = resp.json()
        assert body["entries"] == []
        assert body["total"] == 0

    def test_returns_seeded_entries(self):
        node = _node()
        node._audit_log.append(
            method="POST", path="/compute/forge",
            requester="n1", status_code=200, request_id="r1",
        )
        resp = _client(node).get("/audit/recent")
        body = resp.json()
        assert body["total"] == 1
        assert body["entries"][0]["path"] == "/compute/forge"

    def test_pagination_propagates(self):
        node = _node()
        for i in range(10):
            node._audit_log.append(
                method="POST", path=f"/x{i}",
                requester="n1", status_code=200, request_id=f"r{i}",
            )
        resp = _client(node).get("/audit/recent?limit=3&offset=2")
        body = resp.json()
        assert len(body["entries"]) == 3
        # offset=2 from most-recent-first: skip x9, x8 → start at x7
        assert [e["path"] for e in body["entries"]] == [
            "/x7", "/x6", "/x5",
        ]

    def test_invalid_limit_returns_422(self):
        node = _node()
        resp = _client(node).get("/audit/recent?limit=0")
        assert resp.status_code == 422
