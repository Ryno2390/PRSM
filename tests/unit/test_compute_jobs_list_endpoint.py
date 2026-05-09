"""GET /compute/jobs — paginated operator-side job list.

Closes the operator UX gap that JobHistoryStore filesystem
persistence alone left open: now that the store survives
restarts, operators need a way to enumerate active + recent
jobs without knowing each job_id ahead of time.

Backs the ``prsm_jobs_list`` MCP tool.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.job_history import (
    JobHistoryRecord, JobHistoryStore, JobStatus,
)


def _record(job_id, *, status=JobStatus.IN_PROGRESS, started_at=None):
    return JobHistoryRecord(
        job_id=job_id, query=f"q-{job_id}", status=status,
        started_at=time.time() if started_at is None else started_at,
    )


def _node(history=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._job_history = history
    node._payment_escrow = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestJobsListAvailability:
    def test_503_when_history_not_wired(self):
        node = _node()
        resp = _client(node).get("/compute/jobs")
        assert resp.status_code == 503


class TestJobsListHappyPath:
    def test_empty_history_returns_empty_jobs(self):
        node = _node(JobHistoryStore())
        resp = _client(node).get("/compute/jobs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["jobs"] == []
        assert body["total"] == 0
        assert body["offset"] == 0

    def test_returns_jobs_most_recent_first(self):
        history = JobHistoryStore()
        history.put(_record("forge-old", started_at=100.0))
        history.put(_record("forge-mid", started_at=200.0))
        history.put(_record("forge-new", started_at=300.0))
        node = _node(history)
        resp = _client(node).get("/compute/jobs")
        body = resp.json()
        ids = [j["job_id"] for j in body["jobs"]]
        assert ids == ["forge-new", "forge-mid", "forge-old"]
        assert body["total"] == 3

    def test_status_filter_narrows_result(self):
        history = JobHistoryStore()
        history.put(_record(
            "forge-1", status=JobStatus.IN_PROGRESS, started_at=100.0,
        ))
        history.put(_record(
            "forge-2", status=JobStatus.COMPLETED, started_at=200.0,
        ))
        history.put(_record(
            "forge-3", status=JobStatus.FAILED, started_at=300.0,
        ))
        node = _node(history)
        resp = _client(node).get("/compute/jobs?status=completed")
        body = resp.json()
        assert [j["job_id"] for j in body["jobs"]] == ["forge-2"]
        assert body["total"] == 1

    def test_limit_caps_output(self):
        history = JobHistoryStore()
        for i in range(5):
            history.put(_record(f"forge-{i}", started_at=float(i)))
        node = _node(history)
        resp = _client(node).get("/compute/jobs?limit=2")
        body = resp.json()
        assert len(body["jobs"]) == 2
        assert body["total"] == 5  # total reflects all matching, not page size

    def test_offset_paginates(self):
        history = JobHistoryStore()
        for i in range(5):
            history.put(_record(f"forge-{i}", started_at=float(i)))
        node = _node(history)
        resp = _client(node).get("/compute/jobs?limit=2&offset=2")
        body = resp.json()
        # most-recent-first: forge-4, 3, [2, 1], 0 → offset 2 + limit 2
        # = forge-2, forge-1
        assert [j["job_id"] for j in body["jobs"]] == ["forge-2", "forge-1"]
        assert body["offset"] == 2
        assert body["limit"] == 2
        assert body["total"] == 5


class TestJobsListValidation:
    def test_invalid_status_returns_422(self):
        node = _node(JobHistoryStore())
        resp = _client(node).get("/compute/jobs?status=invalid")
        assert resp.status_code == 422

    def test_negative_offset_returns_422(self):
        node = _node(JobHistoryStore())
        resp = _client(node).get("/compute/jobs?offset=-1")
        assert resp.status_code == 422

    def test_zero_limit_returns_422(self):
        node = _node(JobHistoryStore())
        resp = _client(node).get("/compute/jobs?limit=0")
        assert resp.status_code == 422

    def test_excessive_limit_clamped_to_max(self):
        """Limit must not allow a single request to enumerate the
        entire LRU cache (DoS shape). Default max 100; oversized
        requests get 422 with a clear message."""
        node = _node(JobHistoryStore())
        resp = _client(node).get("/compute/jobs?limit=10000")
        assert resp.status_code == 422
