"""Cap replicas on /content/upload (sprint 101).

Production hardening: a malicious or buggy uploader could
request millions of replicas, DoS-ing the storage layer.
PRSM_MAX_REPLICAS (default 100) caps the request at the API
boundary; over-cap requests get 422 with clear remediation.
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
    node.content_uploader = AsyncMock()
    node.content_uploader.upload_text = AsyncMock(
        return_value=MagicMock(
            content_id="cid1",
            filename="x.txt",
            size_bytes=100,
            content_hash="0xCAFE",
        ),
    )
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _post(node, replicas: int):
    return _client(node).post(
        "/content/upload",
        json={
            "text": "hello world",
            "filename": "test.txt",
            "replicas": replicas,
        },
    )


class TestReplicasCap:
    def test_default_100_accepts_50(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_REPLICAS", None)
            resp = _post(_node(), 50)
        assert resp.status_code == 200

    def test_default_100_rejects_101(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_REPLICAS", None)
            resp = _post(_node(), 101)
        assert resp.status_code == 422
        assert "100" in resp.json()["detail"]

    def test_env_override_higher(self):
        with patch.dict(os.environ, {"PRSM_MAX_REPLICAS": "500"}):
            resp = _post(_node(), 200)
        assert resp.status_code == 200

    def test_env_override_lower_rejects_default(self):
        with patch.dict(os.environ, {"PRSM_MAX_REPLICAS": "5"}):
            resp = _post(_node(), 10)
        assert resp.status_code == 422
        assert "5" in resp.json()["detail"]

    def test_invalid_env_falls_back_to_default(self):
        # Non-numeric env should silently fall back to 100
        with patch.dict(os.environ, {"PRSM_MAX_REPLICAS": "not-a-num"}):
            resp = _post(_node(), 50)
        assert resp.status_code == 200
        # And rejects above default
        with patch.dict(os.environ, {"PRSM_MAX_REPLICAS": "not-a-num"}):
            resp = _post(_node(), 200)
        assert resp.status_code == 422

    def test_zero_replicas_accepted(self):
        # 0 is a valid choice — content stored locally only
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_REPLICAS", None)
            resp = _post(_node(), 0)
        assert resp.status_code == 200
