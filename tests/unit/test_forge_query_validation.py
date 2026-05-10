"""Query validation on /compute/forge (sprint 108).

Hardening: whitespace-only queries rejected; query size cap
prevents prompt-injection DoS via multi-MB amplification
through LLM tokens.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    # agent_forge intentionally None so we focus on validation
    # without triggering downstream paths
    node.agent_forge = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _post(node, body):
    return _client(node).post("/compute/forge", json=body)


class TestQueryValidation:
    def test_missing_query_400(self):
        resp = _post(_node(), {})
        assert resp.status_code == 400
        assert "query" in resp.json()["detail"].lower()

    def test_whitespace_only_query_400(self):
        resp = _post(_node(), {"query": "   \t\n  "})
        assert resp.status_code == 400
        assert "whitespace" in resp.json()["detail"].lower()

    def test_empty_string_query_400(self):
        resp = _post(_node(), {"query": ""})
        assert resp.status_code == 400

    def test_oversized_query_413(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_QUERY_BYTES", None)
            big = "x" * (101 * 1024)  # 101KB > 100KB default
            resp = _post(_node(), {"query": big, "budget_ftns": 1.0})
        assert resp.status_code == 413
        assert "PRSM_MAX_QUERY_BYTES" in resp.json()["detail"]

    def test_env_override_cap(self):
        with patch.dict(os.environ, {"PRSM_MAX_QUERY_BYTES": "100"}):
            resp = _post(
                _node(),
                {"query": "x" * 200, "budget_ftns": 1.0},
            )
        assert resp.status_code == 413

    def test_invalid_env_falls_back(self):
        with patch.dict(os.environ, {"PRSM_MAX_QUERY_BYTES": "boom"}):
            resp = _post(
                _node(),
                {"query": "small query", "budget_ftns": 1.0},
            )
        # Default 100KB; small passes the cap (downstream may
        # 503 / fail since agent_forge=None — that's OK,
        # validation order means cap runs before forge check)
        assert resp.status_code != 413

    def test_normal_query_passes_validation(self):
        resp = _post(
            _node(),
            {"query": "What is the capital of France?", "budget_ftns": 1.0},
        )
        # Should NOT 400/413 — validation passed (downstream may
        # 503 since agent_forge=None on test node)
        assert resp.status_code not in (400, 413)
