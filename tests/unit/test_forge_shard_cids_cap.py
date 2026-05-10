"""Cap shard_cids on /compute/forge (sprint 109)."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

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
    node.agent_forge = None  # Forces 503 IF cap doesn't fire first
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _post(node, shard_cids):
    return _client(node).post(
        "/compute/forge",
        json={
            "query": "test query",
            "budget_ftns": 1.0,
            "shard_cids": shard_cids,
        },
    )


class TestShardCidsCap:
    def test_default_100_rejects_101(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_FORGE_SHARDS", None)
            cids = [f"cid{i}" for i in range(101)]
            resp = _post(_node(), cids)
        assert resp.status_code == 422
        assert "100" in resp.json()["detail"]

    def test_env_override_smaller(self):
        with patch.dict(os.environ, {"PRSM_MAX_FORGE_SHARDS": "5"}):
            resp = _post(_node(), [f"cid{i}" for i in range(10)])
        assert resp.status_code == 422

    def test_below_cap_passes(self):
        # Below cap → goes through to agent_forge=None 503
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_FORGE_SHARDS", None)
            resp = _post(_node(), ["cid1", "cid2"])
        assert resp.status_code != 422

    def test_none_passes(self):
        # No shard_cids = no cap check
        resp = _client(_node()).post(
            "/compute/forge",
            json={"query": "test", "budget_ftns": 1.0},
        )
        assert resp.status_code != 422

    def test_invalid_env_falls_back(self):
        with patch.dict(os.environ, {"PRSM_MAX_FORGE_SHARDS": "boom"}):
            resp = _post(_node(), [f"cid{i}" for i in range(50)])
        assert resp.status_code != 422
