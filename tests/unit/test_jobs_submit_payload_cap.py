"""Sprint 197 — /api/jobs/submit payload size cap.

Pre-fix a JSON payload of arbitrary size was accepted with 200
and gossip-propagated to the swarm. Network-DoS vector: a single
1MB payload submitted at high frequency could saturate cross-
operator bandwidth.

Cap: 100KB default (matches PRSM_MAX_QUERY_BYTES). Operator
override via PRSM_MAX_JOB_PAYLOAD_BYTES.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


def _client():
    from prsm.dashboard.app import DashboardServer
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.compute_requester = MagicMock()
    server = DashboardServer(node=node)
    return TestClient(server.app, raise_server_exceptions=False)


def test_oversized_payload_returns_413():
    """Sprint 197 — 1MB payload now 413 (was 200)."""
    big_payload = {"data": "x" * 1_000_000}
    resp = _client().post("/api/jobs/submit", json={
        "job_type": "inference",
        "payload": big_payload,
    })
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"].lower()


def test_payload_at_cap_passes_validation():
    """Sprint 197 — payload just under cap reaches downstream
    (validation gate passes)."""
    # ~50KB
    payload = {"data": "x" * 50_000}
    resp = _client().post("/api/jobs/submit", json={
        "job_type": "inference",
        "payload": payload,
    })
    # NOT 413 — validation passed. May be other code (200 or 5xx
    # depending on downstream mock fidelity).
    assert resp.status_code != 413


def test_typical_payload_works(monkeypatch):
    """Sprint 197 invariant — small typical payload accepted."""
    resp = _client().post("/api/jobs/submit", json={
        "job_type": "inference",
        "payload": {"prompt": "hello world"},
    })
    assert resp.status_code != 413


def test_env_override(monkeypatch):
    """Operator can raise the cap via env var."""
    monkeypatch.setenv("PRSM_MAX_JOB_PAYLOAD_BYTES", "10")
    resp = _client().post("/api/jobs/submit", json={
        "job_type": "inference",
        "payload": {"prompt": "hello world"},
    })
    # 21 bytes after JSON serialization > 10-byte cap → 413
    assert resp.status_code == 413


def test_env_invalid_falls_back_to_default(monkeypatch):
    """Garbage env value silently falls back to 100KB default."""
    monkeypatch.setenv("PRSM_MAX_JOB_PAYLOAD_BYTES", "not_an_int")
    resp = _client().post("/api/jobs/submit", json={
        "job_type": "inference",
        "payload": {"data": "x" * 100},  # 100 bytes — under 100KB
    })
    assert resp.status_code != 413
