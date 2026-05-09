"""Audit log ring buffer + GET /audit/recent endpoint + filesystem persistence."""
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


# ──────────────────────────────────────────────────────────────────────
# Middleware auto-population
# ──────────────────────────────────────────────────────────────────────


class TestMiddlewareAutoPopulate:
    """Non-GET requests auto-append to the audit log via middleware.
    GET requests are NOT recorded to keep the buffer focused on
    writes."""

    def test_post_request_recorded(self):
        node = _node()
        # POST to a route that returns something deterministic.
        # /info doesn't exist as POST so we'll get 405; that's
        # still a state-change attempt worth logging.
        client = _client(node)
        client.post("/some/nonexistent/path", json={})
        entries = node._audit_log.recent()
        # At least one entry recorded — the POST attempt.
        assert any(e.method == "POST" for e in entries)

    def test_get_request_not_recorded(self):
        """GET requests are read-only; don't pollute the audit
        buffer with them."""
        node = _node()
        client = _client(node)
        client.get("/health")
        client.get("/info")
        entries = node._audit_log.recent()
        # No GET entries (the audit log is for state changes only).
        assert all(e.method != "GET" for e in entries)

    def test_self_referential_audit_get_not_recorded(self):
        """GET /audit/recent itself shouldn't appear in the
        results (it's a GET so already excluded; this confirms
        the GET filter works on this self-call)."""
        node = _node()
        client = _client(node)
        client.get("/audit/recent")
        client.get("/audit/recent")
        entries = node._audit_log.recent()
        # No /audit/recent entries since they were GET.
        assert all(e.path != "/audit/recent" for e in entries)

    def test_status_code_recorded(self):
        node = _node()
        client = _client(node)
        # POST to a path that returns 405 (no POST handler).
        resp = client.post("/health", json={})
        entries = node._audit_log.recent()
        # Entry recorded with the actual response status code.
        assert any(e.status_code == resp.status_code for e in entries)


# ──────────────────────────────────────────────────────────────────────
# Filesystem persistence (PRSM_AUDIT_LOG_DIR opt-in)
# ──────────────────────────────────────────────────────────────────────


class TestPersistence:
    """Operator audit log survives restart when persist_dir set —
    forensic continuity. Default unset preserves v1 in-memory-only
    behavior bit-identically."""

    def test_v1_behavior_when_persist_dir_none(self, tmp_path):
        ring = AuditLogRing()
        ring.append(
            method="POST", path="/x", requester="n1",
            status_code=200, request_id="r1",
        )
        # No files written.
        assert list(tmp_path.iterdir()) == []
        assert ring.count() == 1

    def test_append_writes_to_disk_when_persist_dir_set(self, tmp_path):
        ring = AuditLogRing(persist_dir=tmp_path)
        ring.append(
            method="POST", path="/x", requester="n1",
            status_code=200, request_id="r1",
        )
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        import json
        data = json.loads(files[0].read_text())
        assert data["method"] == "POST"
        assert data["path"] == "/x"

    def test_startup_scan_loads_existing_entries(self, tmp_path):
        ring_a = AuditLogRing(persist_dir=tmp_path)
        ring_a.append(
            method="POST", path="/x1", requester="n1",
            status_code=200, request_id="r1",
            timestamp=1000.0,
        )
        ring_a.append(
            method="POST", path="/x2", requester="n1",
            status_code=200, request_id="r2",
            timestamp=2000.0,
        )
        # Fresh ring on same dir.
        ring_b = AuditLogRing(persist_dir=tmp_path)
        results = ring_b.recent()
        assert len(results) == 2
        # Most-recent-first by timestamp.
        assert results[0].path == "/x2"
        assert results[1].path == "/x1"

    def test_corrupt_disk_file_skipped_fail_soft(self, tmp_path):
        # Write a junk file directly.
        (tmp_path / "corrupt.json").write_text("{not valid json")
        # Write a valid one.
        ring_a = AuditLogRing(persist_dir=tmp_path)
        ring_a.append(
            method="POST", path="/good", requester="n1",
            status_code=200, request_id="r1",
        )
        # Reload — must not raise.
        ring_b = AuditLogRing(persist_dir=tmp_path)
        results = ring_b.recent()
        # Valid entry loaded; corrupt one skipped.
        assert any(e.path == "/good" for e in results)

    def test_persist_dir_created_if_missing(self, tmp_path):
        target = tmp_path / "subdir" / "audit"
        assert not target.exists()
        ring = AuditLogRing(persist_dir=target)
        assert target.exists() and target.is_dir()
        ring.append(
            method="POST", path="/x", requester="n1",
            status_code=200, request_id="r1",
        )
        assert list(target.glob("*.json"))

    def test_retention_prunes_old_disk_entries(self, tmp_path):
        """retention_days arg deletes disk files older than the
        retention window on startup. Closes the unbounded-disk-
        growth concern for long-running operator nodes."""
        import time as _time
        # Pre-seed disk with one OLD and one RECENT entry.
        ring_a = AuditLogRing(persist_dir=tmp_path)
        old_ts = _time.time() - 10 * 86400  # 10 days ago
        recent_ts = _time.time() - 1 * 3600  # 1 hour ago
        ring_a.append(
            method="POST", path="/old", requester="n1",
            status_code=200, request_id="r-old",
            timestamp=old_ts,
        )
        ring_a.append(
            method="POST", path="/recent", requester="n1",
            status_code=200, request_id="r-recent",
            timestamp=recent_ts,
        )
        # Both files on disk.
        assert len(list(tmp_path.glob("*.json"))) == 2
        # Fresh ring with 7-day retention → prune the 10-day-old one.
        ring_b = AuditLogRing(persist_dir=tmp_path, retention_days=7.0)
        # Disk now has only the recent file.
        on_disk = list(tmp_path.glob("*.json"))
        assert len(on_disk) == 1
        # In-memory ring also reflects pruning.
        results = ring_b.recent()
        assert len(results) == 1
        assert results[0].path == "/recent"

    def test_no_retention_keeps_all_disk_entries(self, tmp_path):
        """Without retention_days set, all disk entries are
        preserved (v1 behavior preserved bit-identically)."""
        import time as _time
        ring_a = AuditLogRing(persist_dir=tmp_path)
        old_ts = _time.time() - 100 * 86400  # 100 days ago
        ring_a.append(
            method="POST", path="/old", requester="n1",
            status_code=200, request_id="r-old",
            timestamp=old_ts,
        )
        # Reload without retention.
        ring_b = AuditLogRing(persist_dir=tmp_path)
        results = ring_b.recent()
        assert len(results) == 1
        assert results[0].path == "/old"

    def test_lru_evicts_oldest_disk_on_startup_when_over_cap(self, tmp_path):
        """If disk has more entries than max_entries, oldest are
        dropped on startup so the in-memory ring respects its bound."""
        ring_a = AuditLogRing(persist_dir=tmp_path, max_entries=3)
        for i in range(5):
            ring_a.append(
                method="POST", path=f"/x{i}", requester="n1",
                status_code=200, request_id=f"r{i}",
                timestamp=float(i),
            )
        # All 5 on disk.
        assert len(list(tmp_path.glob("*.json"))) == 5
        # Fresh ring with max_entries=3 → only 3 most-recent loaded.
        ring_b = AuditLogRing(persist_dir=tmp_path, max_entries=3)
        results = ring_b.recent()
        assert len(results) == 3
        # Oldest dropped.
        paths = {e.path for e in results}
        assert "/x4" in paths and "/x3" in paths and "/x2" in paths
        assert "/x0" not in paths and "/x1" not in paths
