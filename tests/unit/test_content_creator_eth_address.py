"""Sprint 243 — foundation for on-chain content-access royalty leg.

Captures `creator_eth_address` at upload time so the eventual
RoyaltyDistributor.distribute_royalty() call has a destination
address. v1 contract:
  - Optional field on ContentUploadRequest + ContentUploadShard
    body shape (default None, preserves backwards-compat).
  - Validated as 0x-prefixed 40-hex-char Ethereum address when
    set; rejected upfront at 422 otherwise.
  - Persisted into UploadedContent record.
  - Surfaced via /content/{cid} when present.

This sprint is data-layer prep ONLY. The actual on-chain dispatch
remains gated on additional infrastructure (per-shard content
hash, gas budget management, settlement-layer integration).
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
    node.content_uploader.content_publisher = MagicMock()
    node.content_uploader.upload_text = AsyncMock(return_value={
        "content_id": "cid-1",
        "filename": "doc.txt",
        "content_hash": "00" * 32,
        "creator_id": "creator-a",
        "size_bytes": 10,
    })
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── Validation ───────────────────────────────────────────


def test_invalid_eth_address_rejected():
    """Missing 0x prefix → 422."""
    resp = _client().post("/content/upload", json={
        "text": "hi",
        "creator_eth_address": "dEADbEef" * 5,  # 40 hex, no 0x
    })
    assert resp.status_code == 422


def test_wrong_length_rejected():
    resp = _client().post("/content/upload", json={
        "text": "hi",
        "creator_eth_address": "0xabc",  # too short
    })
    assert resp.status_code == 422


def test_non_hex_rejected():
    resp = _client().post("/content/upload", json={
        "text": "hi",
        "creator_eth_address": "0x" + "z" * 40,  # not hex
    })
    assert resp.status_code == 422


# ── Backwards-compat ────────────────────────────────────


def test_field_omitted_is_fine():
    """Existing uploads work — field is optional."""
    resp = _client().post("/content/upload", json={"text": "hi"})
    assert resp.status_code != 422


def test_field_null_is_fine():
    resp = _client().post("/content/upload", json={
        "text": "hi", "creator_eth_address": None,
    })
    assert resp.status_code != 422


# ── Happy path ──────────────────────────────────────────


def test_valid_address_accepted():
    resp = _client().post("/content/upload", json={
        "text": "hi",
        "creator_eth_address": "0xdEADbEef" + "0" * 32,
    })
    assert resp.status_code != 422
