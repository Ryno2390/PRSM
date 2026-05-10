"""Cap parent_cids on /content/upload (sprint 105)."""
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
    cu = AsyncMock()
    cu.upload_text = AsyncMock(
        return_value=MagicMock(
            content_id="cid", filename="x", size_bytes=1,
            content_hash="h", creator_id="me", royalty_rate=0.01,
            parent_cids=[],
        ),
    )
    node.content_uploader = cu
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _post(node, parent_cids):
    return _client(node).post(
        "/content/upload",
        json={
            "text": "hi",
            "filename": "x.txt",
            "replicas": 1,
            "parent_cids": parent_cids,
        },
    )


class TestParentCidsCap:
    def test_default_100_rejects_101(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_PARENT_CIDS", None)
            cids = [f"cid{i}" for i in range(101)]
            resp = _post(_node(), cids)
        assert resp.status_code == 422
        assert "100" in resp.json()["detail"]

    def test_env_override_smaller(self):
        with patch.dict(os.environ, {"PRSM_MAX_PARENT_CIDS": "5"}):
            resp = _post(_node(), [f"cid{i}" for i in range(10)])
        assert resp.status_code == 422

    def test_below_cap_passes(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_PARENT_CIDS", None)
            resp = _post(_node(), ["cid1", "cid2", "cid3"])
        assert resp.status_code == 200

    def test_empty_list_passes(self):
        resp = _post(_node(), [])
        assert resp.status_code == 200

    def test_invalid_env_falls_back(self):
        with patch.dict(os.environ, {"PRSM_MAX_PARENT_CIDS": "boom"}):
            resp = _post(_node(), [f"cid{i}" for i in range(50)])
        # Default 100; 50 should pass
        assert resp.status_code == 200
