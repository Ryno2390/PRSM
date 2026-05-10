"""Sprint 172 — /transactions `limit` query param bounds validation.

Pre-fix the handler took `limit: int = 50` and passed
`min(limit, 200)` to the ledger. A negative limit (limit=-1)
became -1, which downstream interpreted as "unlimited" and
returned every transaction in history. Real DoS vector — an
operator with thousands of transactions hits /transactions?limit=-1
and the server serializes them all.

Sprint 172 adds upfront 422 validation matching the pattern
used on /compute/jobs and /admin/*-history endpoints:
  limit must be in [1, 200].
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ledger = MagicMock()
    node.ledger.get_transaction_history = AsyncMock(return_value=[])
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestTransactionsLimitBounds:
    def test_negative_limit_returns_422(self):
        resp = _client(_node()).get("/transactions?limit=-1")
        assert resp.status_code == 422
        assert "limit" in resp.json()["detail"].lower()

    def test_zero_limit_returns_422(self):
        resp = _client(_node()).get("/transactions?limit=0")
        assert resp.status_code == 422

    def test_excessive_limit_returns_422(self):
        resp = _client(_node()).get("/transactions?limit=99999")
        assert resp.status_code == 422

    def test_valid_limit_passes(self):
        resp = _client(_node()).get("/transactions?limit=50")
        assert resp.status_code == 200

    def test_boundary_limit_1_passes(self):
        resp = _client(_node()).get("/transactions?limit=1")
        assert resp.status_code == 200

    def test_boundary_limit_200_passes(self):
        resp = _client(_node()).get("/transactions?limit=200")
        assert resp.status_code == 200

    def test_default_limit_passes(self):
        resp = _client(_node()).get("/transactions")
        assert resp.status_code == 200
