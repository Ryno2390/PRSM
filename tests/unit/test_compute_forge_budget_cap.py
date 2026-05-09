"""POST /compute/forge — PRSM_MAX_FTNS_PER_JOB cap enforcement.

Cost-control feature: operators worried about misbehaving AI
agents draining their FTNS balance with a single oversized
request can set a per-job cap via env var. Requests exceeding
the cap return 422 with a clear remaining-budget message
before any escrow is locked.

Default: unlimited (preserves v1 behavior bit-identically).
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.job_history import JobHistoryStore


def _node_minimal():
    """Bare node — no agent_forge, no escrow. The budget-cap
    check fires BEFORE the agent_forge availability check, so
    422 with cap exceeded returns even when nothing else is
    wired."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._job_history = JobHistoryStore()
    node.agent_forge = None  # would normally 503
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Cap enforcement
# ──────────────────────────────────────────────────────────────────────


class TestBudgetCapEnforcement:
    def test_request_exceeding_cap_returns_422(self):
        node = _node_minimal()
        with patch.dict(os.environ, {"PRSM_MAX_FTNS_PER_JOB": "5.0"}):
            resp = _client(node).post(
                "/compute/forge",
                json={"query": "test", "budget_ftns": 10.0},
            )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "5.0" in detail or "5" in detail
        assert "10" in detail
        assert "PRSM_MAX_FTNS_PER_JOB" in detail or \
            "cap" in detail.lower()

    def test_request_within_cap_proceeds(self):
        """Request within cap falls through to normal forge logic
        (which 503s here because agent_forge is None)."""
        node = _node_minimal()
        with patch.dict(os.environ, {"PRSM_MAX_FTNS_PER_JOB": "100.0"}):
            resp = _client(node).post(
                "/compute/forge",
                json={"query": "test", "budget_ftns": 50.0},
            )
        # 503 from agent_forge None — proves we didn't 422.
        assert resp.status_code == 503

    def test_request_at_exact_cap_proceeds(self):
        """Exact-equal is allowed; only strictly-greater rejects."""
        node = _node_minimal()
        with patch.dict(os.environ, {"PRSM_MAX_FTNS_PER_JOB": "10.0"}):
            resp = _client(node).post(
                "/compute/forge",
                json={"query": "test", "budget_ftns": 10.0},
            )
        # Through to 503.
        assert resp.status_code == 503


# ──────────────────────────────────────────────────────────────────────
# Default + env-var resolution
# ──────────────────────────────────────────────────────────────────────


class TestBudgetCapDefault:
    def test_no_env_means_unlimited(self):
        """Without PRSM_MAX_FTNS_PER_JOB set, default is unlimited
        — v1 behavior preserved."""
        node = _node_minimal()
        # Make sure env is unset.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_FTNS_PER_JOB", None)
            resp = _client(node).post(
                "/compute/forge",
                json={"query": "test", "budget_ftns": 1_000_000.0},
            )
        # Through to 503 (cap not enforced).
        assert resp.status_code == 503

    def test_invalid_env_falls_back_to_unlimited(self):
        node = _node_minimal()
        with patch.dict(
            os.environ, {"PRSM_MAX_FTNS_PER_JOB": "not_a_number"},
        ):
            resp = _client(node).post(
                "/compute/forge",
                json={"query": "test", "budget_ftns": 1_000_000.0},
            )
        # Through to 503 (cap silently disabled, log WARN).
        assert resp.status_code == 503

    def test_zero_or_negative_env_disables_cap(self):
        """Zero or negative cap is non-sensical (every request would
        reject) — fall back to unlimited."""
        node = _node_minimal()
        with patch.dict(os.environ, {"PRSM_MAX_FTNS_PER_JOB": "0"}):
            resp = _client(node).post(
                "/compute/forge",
                json={"query": "test", "budget_ftns": 100.0},
            )
        # Through to 503 — zero cap disabled.
        assert resp.status_code == 503

        with patch.dict(os.environ, {"PRSM_MAX_FTNS_PER_JOB": "-1"}):
            resp = _client(node).post(
                "/compute/forge",
                json={"query": "test", "budget_ftns": 100.0},
            )
        assert resp.status_code == 503


# ──────────────────────────────────────────────────────────────────────
# Order: cap fires AFTER idempotency cache hit (cache is free)
# ──────────────────────────────────────────────────────────────────────


class TestBudgetCapOrdering:
    def test_idempotent_replay_bypasses_cap(self):
        """A cache-hit by Idempotency-Key does NOT re-validate
        against the cap — the cached job already settled. This
        prevents a cap-tightened-since-original-request from
        breaking idempotent retries."""
        from prsm.node.job_history import (
            JobHistoryRecord, JobStatus,
        )
        import time
        node = _node_minimal()
        rec = JobHistoryRecord(
            job_id="forge-cached", query="prior",
            status=JobStatus.COMPLETED,
            started_at=time.time() - 10,
            completed_at=time.time(),
            response="cached",
        )
        node._job_history.put_with_idempotency(
            rec, idempotency_key="key-1",
        )

        with patch.dict(os.environ, {"PRSM_MAX_FTNS_PER_JOB": "1.0"}):
            resp = _client(node).post(
                "/compute/forge",
                json={"query": "anything", "budget_ftns": 1_000_000.0},
                headers={"Idempotency-Key": "key-1"},
            )
        # Cache hit returns 200 even though budget exceeds cap.
        assert resp.status_code == 200
        assert resp.json()["status"] == "idempotent_replay"
