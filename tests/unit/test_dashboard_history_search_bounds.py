"""Sprint 193 — /api/ftns/history + /api/content/search limit bounds.

Same DoS-via-unbounded-limit pattern that sprint 172 fixed on the
main /transactions endpoint. The dashboard sub-app duplicated the
identical `min(limit, 200)` and `min(limit, 100)` patterns —
caps upper, accepts negative. `limit=-1` passed through to the
underlying lookup returning the FULL history / search index.

DoS vector + metadata exfil: an unauthenticated attacker (in
dev-mode) or any caller with the right scope could enumerate all
transactions / all indexed content with a single negative-limit
request.

Plus content search adds a query-size cap (1024 chars) — operators
shouldn't be able to overload the index with pathological-pattern
queries.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


def _dash_client():
    """Build a dashboard-only TestClient with a stub node."""
    from prsm.dashboard.app import DashboardServer
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ledger = MagicMock()
    node.ledger.get_transaction_history = AsyncMock(return_value=[])
    node.content_index = MagicMock()
    node.content_index.search = MagicMock(return_value=[])
    server = DashboardServer(node=node)
    return TestClient(server.app)


class TestHistoryLimitBounds:
    def test_negative_limit_returns_422(self):
        resp = _dash_client().get("/api/ftns/history?limit=-1")
        assert resp.status_code == 422
        assert "limit" in resp.json()["detail"].lower()

    def test_zero_limit_returns_422(self):
        resp = _dash_client().get("/api/ftns/history?limit=0")
        assert resp.status_code == 422

    def test_excessive_limit_returns_422(self):
        resp = _dash_client().get("/api/ftns/history?limit=99999")
        assert resp.status_code == 422

    def test_valid_limit_passes(self):
        resp = _dash_client().get("/api/ftns/history?limit=50")
        assert resp.status_code == 200

    def test_boundary_limits_pass(self):
        for lim in (1, 200):
            resp = _dash_client().get(f"/api/ftns/history?limit={lim}")
            assert resp.status_code == 200


class TestContentSearchBounds:
    def test_negative_limit_returns_422(self):
        resp = _dash_client().get("/api/content/search?q=test&limit=-1")
        assert resp.status_code == 422

    def test_zero_limit_returns_422(self):
        resp = _dash_client().get("/api/content/search?q=test&limit=0")
        assert resp.status_code == 422

    def test_excessive_limit_returns_422(self):
        resp = _dash_client().get("/api/content/search?q=test&limit=9999")
        assert resp.status_code == 422

    def test_long_query_returns_413(self):
        big = "x" * 2000
        resp = _dash_client().get(f"/api/content/search?q={big}")
        assert resp.status_code == 413
        assert "exceeds" in resp.json()["detail"].lower()

    def test_boundary_query_passes(self):
        """Sprint 193 — query at exactly 1024 still accepted."""
        big = "x" * 1024
        resp = _dash_client().get(f"/api/content/search?q={big}")
        assert resp.status_code == 200

    def test_valid_search_passes(self):
        resp = _dash_client().get("/api/content/search?q=test&limit=10")
        assert resp.status_code == 200
