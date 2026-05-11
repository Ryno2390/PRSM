"""Sprint 186 — /health supports HEAD method (was 404).

Sprint 186 dogfood probe revealed `HEAD /health` returned 404
instead of 200. Infrastructure-side health probes
(Kubernetes liveness, AWS ELB, generic monitoring) default to
HEAD per RFC 7231 — those probes were seeing PRSM nodes as
unhealthy.

Root cause: `app.mount("", _dash_app)` at api.py:6753 mounts a
catch-all dashboard sub-app. FastAPI auto-supports HEAD for GET
routes, but the mount intercepts HEAD requests before they
reach FastAPI's auto-HEAD path, treating them as "unknown
endpoint" → 404 from the dashboard sub-app.

Fix: explicit `@app.api_route("/health", methods=["GET", "HEAD"])`
registers both methods up-front so the parent app handles them
before the mount can intercept.

Other HEAD routes can be fixed similarly when needed. /health is
the load-bearing one because it's the canonical health-probe
endpoint.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    return TestClient(create_api_app(node, enable_security=False))


def test_head_health_returns_200():
    """Sprint 186 — HEAD /health is now 200 (was 404)."""
    resp = _client().head("/health")
    assert resp.status_code == 200


def test_get_health_still_200():
    """Regression-pin — adding HEAD must not break GET."""
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_head_health_returns_empty_body():
    """RFC 7231 — HEAD response MUST NOT include a body."""
    resp = _client().head("/health")
    assert resp.content == b""


def test_head_health_carries_correct_headers():
    """HEAD response should include the same headers GET would."""
    head_resp = _client().head("/health")
    get_resp = _client().get("/health")
    # Both should set application/json content-type (HEAD's content-
    # length should match GET's response body size).
    assert head_resp.headers.get("content-type") == get_resp.headers.get("content-type")
