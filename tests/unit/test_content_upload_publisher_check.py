"""/content/upload pre-flight ContentPublisher check (sprint 124).

Pre-fix: when content_publisher was None (e.g., libtorrent not
installed), upload_text returned None → API returned generic
502 'content store unavailable?' which hid the real cause.

Post-fix: API checks content_publisher upfront and returns
503 with actionable libtorrent install hint.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node(*, publisher_wired: bool = True):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    cu = AsyncMock()
    cu.content_publisher = MagicMock() if publisher_wired else None
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


def _post(node, text="hi"):
    return _client(node).post(
        "/content/upload",
        json={"text": text, "filename": "t.txt", "replicas": 1},
    )


class TestPublisherCheck:
    def test_503_when_publisher_unwired(self):
        resp = _post(_node(publisher_wired=False))
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert "libtorrent" in detail.lower()
        assert "BitTorrent" in detail

    def test_proceeds_when_publisher_wired(self):
        resp = _post(_node(publisher_wired=True))
        # Should NOT be 503-because-of-publisher (might still 200
        # OR another status from downstream — just not THIS 503)
        if resp.status_code == 503:
            assert "libtorrent" not in resp.json()["detail"].lower()


class TestPreservesOtherChecks:
    def test_503_when_uploader_missing(self):
        node = _node()
        node.content_uploader = None
        resp = _post(node)
        assert resp.status_code == 503
        # Original "Content uploader not initialized" message
        # (NOT the new libtorrent message)
        assert "uploader" in resp.json()["detail"].lower()
