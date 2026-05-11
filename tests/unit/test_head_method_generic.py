"""Sprint 187 — HEAD method support generalized via middleware.

Sprint 186 registered HEAD explicitly on /health to support
infra health probes. Sprint 187 generalizes: HEAD on ANY GET
route now returns the same status + headers as GET, with body
stripped per RFC 7231 §4.3.2.

Pre-fix (sprint 186 narrow): only /health supported HEAD; every
other GET route returned 404 because the catch-all dashboard
mount swallowed HEAD before FastAPI's auto-HEAD path fired.

Post-fix (sprint 187): head_as_get_middleware rewrites HEAD
requests to GET, then strips body from the GET response. Works
for every defined GET route uniformly — no per-route changes
needed.
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
    node._operator_address = None
    return TestClient(create_api_app(node, enable_security=False))


class TestHeadGeneric:
    def test_head_health(self):
        resp = _client().head("/health")
        assert resp.status_code == 200
        assert resp.content == b""

    def test_head_info(self):
        """Sprint 187 — HEAD /info now works (was 404 in 186)."""
        resp = _client().head("/info")
        assert resp.status_code == 200
        assert resp.content == b""

    def test_head_api_info(self):
        """HEAD /api-info — another canonical metadata probe."""
        resp = _client().head("/api-info")
        assert resp.status_code == 200
        assert resp.content == b""

    def test_head_metrics(self):
        """HEAD /metrics — Prometheus probe pattern."""
        resp = _client().head("/metrics")
        assert resp.status_code == 200
        assert resp.content == b""

    def test_get_still_works_unchanged(self):
        """Regression-pin — middleware doesn't break GET."""
        resp = _client().get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_head_strips_body(self):
        """RFC 7231 §4.3.2 — HEAD MUST NOT include body."""
        resp = _client().head("/info")
        assert resp.content == b""

    def test_head_preserves_content_type(self):
        """HEAD response carries the GET response's Content-Type so
        clients can sniff response shape without fetching body."""
        head_resp = _client().head("/info")
        get_resp = _client().get("/info")
        assert (
            head_resp.headers.get("content-type")
            == get_resp.headers.get("content-type")
        )

    def test_post_endpoint_head_does_not_match(self):
        """HEAD on a POST-only endpoint should NOT find a GET to
        rewrite to. The middleware tries internal rewrite, but the
        GET route doesn't exist, so the response is whatever the
        mount catch-all returns (404 for unknown method)."""
        resp = _client().head("/compute/forge")
        # The dashboard catch-all mount returns 404 here. The
        # middleware still strips body, so 404 with empty content.
        # This pins that the middleware doesn't accidentally turn
        # POST endpoints into reachable HEAD endpoints.
        assert resp.status_code in (404, 405)
