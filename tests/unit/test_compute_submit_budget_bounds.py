"""Sprint 204 — /compute/submit ftns_budget bounded.

JobSubmission.ftns_budget at api.py:41 is a bare `float = 1.0`
with no Pydantic constraints. Pre-fix the following slipped
through:
  - ftns_budget=-100 → submitted to compute_requester (negative-
                       budget job, downstream undefined behavior)
  - ftns_budget=0    → silent zero-budget job

The middleware (sprint 201) catches Infinity/NaN, but signed
finite values bypassed validation entirely.

Post-fix: Pydantic `Field(..., gt=0)` rejects negative/zero at
422; `allow_inf_nan=False` is redundant after the middleware but
included as defense-in-depth.
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


def test_negative_budget_rejected():
    resp = _client().post("/compute/submit", json={
        "job_type": "inference", "payload": {},
        "ftns_budget": -100,
    })
    assert resp.status_code == 422


def test_zero_budget_rejected():
    resp = _client().post("/compute/submit", json={
        "job_type": "inference", "payload": {},
        "ftns_budget": 0,
    })
    assert resp.status_code == 422


def test_positive_budget_passes_validation():
    resp = _client().post("/compute/submit", json={
        "job_type": "inference", "payload": {},
        "ftns_budget": 1.0,
    })
    assert resp.status_code != 422


def test_excessive_budget_rejected():
    """Cap upper bound — PRSM_MAX_FTNS_PER_JOB applies on /forge,
    but /compute/submit had no per-job cap. Add a sane absolute
    upper bound to reject obvious garbage like 1e308."""
    resp = _client().post("/compute/submit", json={
        "job_type": "inference", "payload": {},
        "ftns_budget": 1e15,
    })
    assert resp.status_code == 422
