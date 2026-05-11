"""Sprint 195 — /compute/forge/quote upfront field validation.

Dogfood probe found multiple 500s + hangs from bad body inputs:
  shard_count="not_an_int"          → 500 (ValueError uncaught)
  estimated_pcu_per_shard="x"       → 500 (ValueError uncaught)
  shard_count=99999999              → hang (no upper cap)
  estimated_pcu_per_shard=-1        → hang (no lower cap)
  hardware_tier="<script>"          → hang (no enum check)

Post-fix all four conditions return 422 with structured detail
before reaching PricingEngine.
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


class TestForgeQuoteValidation:
    def test_non_numeric_shard_count(self):
        resp = _client().post("/compute/forge/quote", json={
            "query": "hi", "shard_count": "not_an_int",
        })
        assert resp.status_code == 422
        assert "shard_count" in resp.json()["detail"].lower()

    def test_excessive_shard_count(self):
        resp = _client().post("/compute/forge/quote", json={
            "query": "hi", "shard_count": 99999,
        })
        assert resp.status_code == 422

    def test_negative_shard_count(self):
        resp = _client().post("/compute/forge/quote", json={
            "query": "hi", "shard_count": -5,
        })
        assert resp.status_code == 422

    def test_zero_shard_count(self):
        resp = _client().post("/compute/forge/quote", json={
            "query": "hi", "shard_count": 0,
        })
        assert resp.status_code == 422

    def test_non_numeric_pcu(self):
        resp = _client().post("/compute/forge/quote", json={
            "query": "hi", "estimated_pcu_per_shard": "x",
        })
        assert resp.status_code == 422
        assert "estimated_pcu_per_shard" in resp.json()["detail"].lower()

    def test_negative_pcu(self):
        resp = _client().post("/compute/forge/quote", json={
            "query": "hi", "estimated_pcu_per_shard": -1.0,
        })
        assert resp.status_code == 422

    def test_zero_pcu(self):
        resp = _client().post("/compute/forge/quote", json={
            "query": "hi", "estimated_pcu_per_shard": 0,
        })
        assert resp.status_code == 422

    def test_invalid_hardware_tier(self):
        resp = _client().post("/compute/forge/quote", json={
            "query": "hi", "hardware_tier": "<script>",
        })
        assert resp.status_code == 422
        assert "hardware_tier" in resp.json()["detail"].lower()
        assert "t1" in resp.json()["detail"]

    def test_valid_request_passes(self):
        resp = _client().post("/compute/forge/quote", json={
            "query": "hi", "shard_count": 3, "hardware_tier": "t2",
        })
        assert resp.status_code == 200

    def test_boundary_shard_count_100(self):
        resp = _client().post("/compute/forge/quote", json={
            "query": "hi", "shard_count": 100,
        })
        assert resp.status_code == 200
