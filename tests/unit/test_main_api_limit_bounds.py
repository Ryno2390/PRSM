"""Sprint 194 — main API endpoints share the same negative-limit
bypass that sprints 172 + 193 fixed on /transactions + dashboard
duplicates. Three more sites swept this sprint:

  /content/search       limit ∈ [1, 100]   + q max 1024 chars
  /agents/search        limit ∈ [1, 100]   + capability max 256
  /bridge/transactions  limit ∈ [1, 200]

Pre-fix all three used `min(limit, X)` — capped upper, accepted
negative → underlying lookup returned full data set.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.content_index = MagicMock()
    node.content_index.search = MagicMock(return_value=[])
    node.agent_registry = MagicMock()
    node.agent_registry.search = MagicMock(return_value=[])
    node.ftns_bridge = MagicMock()
    node.ftns_bridge.get_user_transactions = AsyncMock(return_value=[])
    return node


def _client():
    return TestClient(create_api_app(_node(), enable_security=False))


class TestContentSearchBounds:
    def test_negative_limit_returns_422(self):
        resp = _client().get("/content/search?q=test&limit=-1")
        assert resp.status_code == 422

    def test_excessive_limit_returns_422(self):
        resp = _client().get("/content/search?q=test&limit=9999")
        assert resp.status_code == 422

    def test_long_query_returns_413(self):
        big = "x" * 2000
        resp = _client().get(f"/content/search?q={big}")
        assert resp.status_code == 413

    def test_boundary_passes(self):
        resp = _client().get(f"/content/search?q={'x'*1024}&limit=100")
        assert resp.status_code == 200


class TestAgentsSearchBounds:
    def test_negative_limit_returns_422(self):
        resp = _client().get("/agents/search?capability=compute&limit=-1")
        assert resp.status_code == 422

    def test_excessive_limit_returns_422(self):
        resp = _client().get("/agents/search?capability=compute&limit=9999")
        assert resp.status_code == 422

    def test_long_capability_returns_413(self):
        big = "x" * 500
        resp = _client().get(f"/agents/search?capability={big}")
        assert resp.status_code == 413

    def test_valid_passes(self):
        resp = _client().get("/agents/search?capability=compute&limit=20")
        assert resp.status_code == 200


class TestBridgeTransactionsBounds:
    def test_negative_limit_returns_422(self):
        resp = _client().get("/bridge/transactions?limit=-1")
        assert resp.status_code == 422

    def test_excessive_limit_returns_422(self):
        resp = _client().get("/bridge/transactions?limit=99999")
        assert resp.status_code == 422

    def test_valid_passes(self):
        resp = _client().get("/bridge/transactions?limit=50")
        assert resp.status_code == 200

    def test_boundary_passes(self):
        for lim in (1, 200):
            resp = _client().get(f"/bridge/transactions?limit={lim}")
            assert resp.status_code == 200
