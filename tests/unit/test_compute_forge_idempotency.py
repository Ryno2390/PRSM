"""POST /compute/forge — Idempotency-Key header handling.

Operators retrying a failed POST due to network blip must not
double-charge their escrow. The forge endpoint reads the
optional ``Idempotency-Key`` header and, when seen previously,
returns the cached job's status without locking a new escrow
or running compute.

Tests focus on the entry-level idempotency check + the
register-on-write path. The full forge pipeline is mocked so
these tests don't require a real Ring 1-10 stack.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.job_history import (
    JobHistoryRecord, JobHistoryStore, JobStatus,
)


def _node(*, with_history=True, with_agent_forge=False):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._job_history = JobHistoryStore() if with_history else None
    if not with_agent_forge:
        node.agent_forge = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Idempotent replay — header present + key seen before
# ──────────────────────────────────────────────────────────────────────


class TestIdempotentReplay:
    def test_returns_cached_job_when_key_seen_before(self):
        """Pre-seed JobHistoryStore + idempotency index. Forge POST
        with the same Idempotency-Key returns the cached job
        without invoking agent_forge (which is None — so any
        execution attempt would 503)."""
        node = _node()
        # Pre-seed a completed job with idempotency mapping.
        completed = JobHistoryRecord(
            job_id="forge-cached-abc",
            query="prior query",
            status=JobStatus.COMPLETED,
            started_at=time.time() - 10,
            completed_at=time.time(),
            response="prior response",
        )
        node._job_history.put_with_idempotency(
            completed, idempotency_key="abc-123",
        )

        # Forge with same key → cache hit, no execute.
        # Even though agent_forge is None (would 503 normally),
        # the idempotency check returns BEFORE that 503.
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "anything"},
            headers={"Idempotency-Key": "abc-123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "idempotent_replay"
        assert body["job_id"] == "forge-cached-abc"
        assert body["history"]["status"] == "completed"
        assert body["history"]["response"] == "prior response"

    def test_no_header_falls_through_to_normal_forge(self):
        """Without Idempotency-Key header, the endpoint proceeds
        as v1 — agent_forge being None hits the normal 503."""
        node = _node()
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "test"},
            # No Idempotency-Key header.
        )
        assert resp.status_code == 503
        assert "Agent forge not initialized" in resp.json()["detail"]

    def test_unknown_key_falls_through_to_normal_forge(self):
        """Idempotency-Key present but key never seen → fall
        through to normal forge logic."""
        node = _node()
        # No pre-seeded mapping for this key.
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "test"},
            headers={"Idempotency-Key": "never-seen-key"},
        )
        # Hits the 503 since agent_forge is None — proves we
        # didn't short-circuit.
        assert resp.status_code == 503

    def test_no_history_store_disables_idempotency(self):
        """Without JobHistoryStore wired, idempotency check is
        skipped (no cache to consult). Falls through to normal
        forge logic."""
        node = _node(with_history=False)
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "test"},
            headers={"Idempotency-Key": "abc-123"},
        )
        # 503 from agent_forge None.
        assert resp.status_code == 503


# ──────────────────────────────────────────────────────────────────────
# Robustness
# ──────────────────────────────────────────────────────────────────────


class TestIdempotencyRobustness:
    def test_lookup_raising_falls_through(self):
        """If lookup_by_idempotency_key raises (corrupted index,
        disk error), the endpoint must NOT 500 — log + continue
        with normal forge logic."""
        node = _node()
        node._job_history.lookup_by_idempotency_key = MagicMock(
            side_effect=RuntimeError("index corrupt"),
        )
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "test"},
            headers={"Idempotency-Key": "abc-123"},
        )
        # Falls through to normal 503 (agent_forge None).
        assert resp.status_code == 503

    def test_cached_record_pointer_dangling_returns_normal_forge(self):
        """Index points to a job_id but the record is LRU-evicted +
        not on disk — treat as cache miss + proceed normally."""
        node = _node()
        # Manually insert a dangling index entry.
        node._job_history._idempotency_index["abc-123"] = "forge-evicted"
        # No record actually present in store.
        resp = _client(node).post(
            "/compute/forge",
            json={"query": "test"},
            headers={"Idempotency-Key": "abc-123"},
        )
        # Falls through; not a cache hit since record is missing.
        assert resp.status_code == 503
