"""Sprint 196 — /compute/query + /content/upload/shard body field validation.

Dogfood found 6 bad-input bugs across two endpoints:

  /compute/query:
    timeout="x"          → 500 (uncaught ValueError)
    budget="x"           → 500
    timeout=-1           → 200 (silently accepted, 0-effective)
    timeout=99999999     → 200 (no upper cap, hours-long wait)
    budget=-100          → 200 (silently accepted as 0)

  /content/upload/shard:
    shard_count="x"      → 500 (uncaught ValueError)
    royalty_rate="x"     → 500
    royalty_rate=100     → 502 (downstream; should be 422)

Post-fix all surface as 422 with structured detail.
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
    node.compute_requester = MagicMock()
    node.compute_provider = MagicMock()
    return node


def _client():
    # raise_server_exceptions=False lets downstream mock-driven
    # 500s return as response objects so we can assert on
    # status_code instead of catching the propagated exception.
    return TestClient(
        create_api_app(_node(), enable_security=False),
        raise_server_exceptions=False,
    )


class TestComputeQueryValidation:
    def test_non_numeric_timeout(self):
        resp = _client().post("/compute/query", json={
            "prompt": "hi", "timeout": "x",
        })
        assert resp.status_code == 422
        assert "timeout" in resp.json()["detail"].lower()

    def test_non_numeric_budget(self):
        resp = _client().post("/compute/query", json={
            "prompt": "hi", "budget": "x",
        })
        assert resp.status_code == 422
        assert "budget" in resp.json()["detail"].lower()

    def test_negative_timeout_rejected(self):
        resp = _client().post("/compute/query", json={
            "prompt": "hi", "timeout": -1,
        })
        assert resp.status_code == 422

    def test_excessive_timeout_rejected(self):
        resp = _client().post("/compute/query", json={
            "prompt": "hi", "timeout": 99999999,
        })
        assert resp.status_code == 422

    def test_negative_budget_rejected(self):
        resp = _client().post("/compute/query", json={
            "prompt": "hi", "budget": -100,
        })
        assert resp.status_code == 422

    def test_zero_budget_accepted_at_validation_layer(self):
        """Sprint 196 invariant — budget=0 (free-tier) passes the
        validation layer. Downstream may fail with mocked node,
        but the validation step doesn't return 422."""
        resp = _client().post("/compute/query", json={
            "prompt": "hi", "budget": 0,
        })
        # 422 means the validation gate triggered — must NOT happen
        # for the canonical free-tier case.
        assert resp.status_code != 422


class TestUploadShardValidation:
    def test_non_numeric_shard_count(self):
        resp = _client().post("/content/upload/shard", json={
            "dataset_id": "x", "content_b64": "aGk=",
            "shard_count": "x",
        })
        assert resp.status_code == 422
        assert "shard_count" in resp.json()["detail"].lower()

    def test_non_numeric_royalty(self):
        resp = _client().post("/content/upload/shard", json={
            "dataset_id": "x", "content_b64": "aGk=",
            "royalty_rate": "x",
        })
        assert resp.status_code == 422
        assert "royalty_rate" in resp.json()["detail"].lower()

    def test_royalty_above_max(self):
        resp = _client().post("/content/upload/shard", json={
            "dataset_id": "x", "content_b64": "aGk=",
            "royalty_rate": 100,
        })
        assert resp.status_code == 422

    def test_royalty_below_min(self):
        resp = _client().post("/content/upload/shard", json={
            "dataset_id": "x", "content_b64": "aGk=",
            "royalty_rate": 0.0001,
        })
        assert resp.status_code == 422

    def test_royalty_boundary_passes_validation(self):
        """Sprint 196 — values at exactly the docs range pass
        the validation layer. Downstream may 500 with mocked
        node, but validation step must NOT return 422."""
        for r in (0.001, 0.05, 0.1):
            resp = _client().post("/content/upload/shard", json={
                "dataset_id": "x", "content_b64": "aGk=",
                "royalty_rate": r,
            })
            assert resp.status_code != 422, (
                f"royalty_rate={r} should pass validation; got 422"
            )
