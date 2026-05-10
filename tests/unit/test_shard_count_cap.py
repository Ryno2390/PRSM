"""Cap shard_count on /content/upload/shard (sprint 104)."""
from __future__ import annotations

import base64
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
    cu = AsyncMock()
    cu.upload = AsyncMock(
        return_value=MagicMock(
            content_id="cid", filename="x", size_bytes=1,
            content_hash="h",
        ),
    )
    node.content_uploader = cu
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _post(node, shard_count):
    return _client(node).post(
        "/content/upload/shard",
        json={
            "dataset_id": "test",
            "content_b64": base64.b64encode(b"hello").decode("ascii"),
            "shard_count": shard_count,
        },
    )


class TestShardCountCap:
    def test_default_1000_rejects_1001(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_SHARD_COUNT", None)
            resp = _post(_node(), 1001)
        assert resp.status_code == 422
        assert "1000" in resp.json()["detail"]

    def test_env_override_smaller(self):
        with patch.dict(os.environ, {"PRSM_MAX_SHARD_COUNT": "10"}):
            resp = _post(_node(), 50)
        assert resp.status_code == 422
        assert "10" in resp.json()["detail"]

    def test_invalid_env_falls_back(self):
        with patch.dict(os.environ, {"PRSM_MAX_SHARD_COUNT": "boom"}):
            resp = _post(_node(), 5000)
        # Default is 1000; 5000 over → 422
        assert resp.status_code == 422

    def test_below_cap_passes_validation(self):
        # We don't care about downstream success — only that the
        # cap doesn't fire for in-bound requests.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_SHARD_COUNT", None)
            resp = _post(_node(), 100)
        assert resp.status_code != 422
