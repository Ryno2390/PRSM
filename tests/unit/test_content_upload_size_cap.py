"""Cap upload payload size on /content/upload (sprint 102).

Production hardening: no upper bound on text size could DoS
the storage layer with multi-GB uploads. PRSM_MAX_UPLOAD_BYTES
(default 10MB) caps at API boundary.
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
            content_id="cid1", filename="x.txt",
            size_bytes=100, content_hash="0xCAFE",
        ),
    )
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _post(node, text):
    return _client(node).post(
        "/content/upload",
        json={"text": text, "filename": "test.txt", "replicas": 1},
    )


class TestSizeCap:
    def test_default_10mb_accepts_1kb(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_UPLOAD_BYTES", None)
            resp = _post(_node(), "x" * 1024)
        assert resp.status_code == 200

    def test_default_10mb_rejects_11mb(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_MAX_UPLOAD_BYTES", None)
            big = "x" * (11 * 1024 * 1024)
            resp = _post(_node(), big)
        assert resp.status_code == 413
        detail = resp.json()["detail"]
        assert "PRSM_MAX_UPLOAD_BYTES" in detail

    def test_env_override_smaller(self):
        with patch.dict(os.environ, {"PRSM_MAX_UPLOAD_BYTES": "1024"}):
            resp = _post(_node(), "x" * 2048)
        assert resp.status_code == 413

    def test_env_override_larger(self):
        with patch.dict(
            os.environ,
            {"PRSM_MAX_UPLOAD_BYTES": str(50 * 1024 * 1024)},
        ):
            resp = _post(_node(), "x" * (20 * 1024 * 1024))
        assert resp.status_code == 200

    def test_invalid_env_falls_back_to_default(self):
        with patch.dict(os.environ, {"PRSM_MAX_UPLOAD_BYTES": "not-a-num"}):
            # Default 10MB; 1KB should pass
            resp = _post(_node(), "x" * 1024)
        assert resp.status_code == 200
