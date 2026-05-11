"""Sprint 208 — ContentUploadRequest field-level bounds.

Pre-fix gaps:
  - filename: unbounded — could be 100MB filename
  - text: unbounded at Pydantic layer (PRSM_MAX_UPLOAD_BYTES caps
          but only AFTER Pydantic materializes the full str in
          memory). Add Pydantic-side max_length for memory DoS
          defense-in-depth.
  - parent_cids items: each unbounded — 100 entries x 1MB each
          would crash on JSON parse

Add:
  - filename max_length=512
  - text max_length=100MB (matches PRSM_MAX_UPLOAD_BYTES default;
    operator can raise the env var but Pydantic ceiling is fixed)
  - parent_cids item max_length=256 (CIDs are <100 chars typical)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.content_uploader = MagicMock()
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_excessive_filename_rejected():
    resp = _client().post("/content/upload", json={
        "text": "hi",
        "filename": "x" * 100_000,
    })
    assert resp.status_code == 422


def test_excessive_parent_cid_item_rejected():
    resp = _client().post("/content/upload", json={
        "text": "hi",
        "parent_cids": ["x" * 10_000],
    })
    assert resp.status_code == 422


def test_typical_passes():
    resp = _client().post("/content/upload", json={
        "text": "hi",
        "filename": "doc.txt",
        "parent_cids": ["bafy123"],
    })
    assert resp.status_code != 422
