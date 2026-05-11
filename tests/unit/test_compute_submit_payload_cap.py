"""Sprint 197b — /compute/submit (main API sibling of sprint 197a)
payload-size cap.

Same gossip-DoS surface as /api/jobs/submit (sprint 197a) — both
endpoints accept arbitrary-size payloads that propagate via
swarm gossip. Cap applies to both.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.compute_requester = MagicMock()
    node.compute_requester.submit_job = AsyncMock()
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_oversized_payload_returns_413():
    big = {"data": "x" * 1_000_000}
    resp = _client().post("/compute/submit", json={
        "job_type": "inference",
        "payload": big,
    })
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"].lower()


def test_typical_payload_passes_validation():
    resp = _client().post("/compute/submit", json={
        "job_type": "inference",
        "payload": {"prompt": "hi"},
    })
    assert resp.status_code != 413


def test_env_override(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_JOB_PAYLOAD_BYTES", "10")
    resp = _client().post("/compute/submit", json={
        "job_type": "inference",
        "payload": {"prompt": "long enough to exceed 10 bytes"},
    })
    assert resp.status_code == 413
