"""Sprint 267 — /storage/pinned-stats + /storage/provider-reputations.

Pre-fix StorageProvider exposed get_pinned_content_stats() and
get_provider_stats_summary() but no HTTP endpoint surfaced
either. Storage operators triaging "is my data being
challenged?" and "which providers are reliable?" had no view.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client(storage_provider=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.storage_provider = storage_provider
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── /storage/pinned-stats ────────────────────────────────


def test_pinned_stats_503_when_storage_unwired():
    resp = _client(None).get("/storage/pinned-stats")
    assert resp.status_code == 503


def test_pinned_stats_returns_list():
    sp = MagicMock()
    sp.get_pinned_content_stats = MagicMock(return_value=[
        {
            "cid": "cid-a",
            "size_bytes": 1024,
            "pinned_at": 100.0,
            "requester_id": "user-1",
            "last_verified": 200.0,
            "successful_challenges": 5,
            "failed_challenges": 0,
        },
        {
            "cid": "cid-b",
            "size_bytes": 4096,
            "pinned_at": 150.0,
            "requester_id": "user-2",
            "last_verified": None,
            "successful_challenges": 0,
            "failed_challenges": 1,
        },
    ])
    resp = _client(sp).get("/storage/pinned-stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["pinned"][0]["cid"] == "cid-a"
    assert body["pinned"][1]["failed_challenges"] == 1


def test_pinned_stats_empty():
    sp = MagicMock()
    sp.get_pinned_content_stats = MagicMock(return_value=[])
    resp = _client(sp).get("/storage/pinned-stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["pinned"] == []


# ── /storage/provider-reputations ────────────────────────


def test_provider_reps_503_when_storage_unwired():
    resp = _client(None).get("/storage/provider-reputations")
    assert resp.status_code == 503


def test_provider_reps_returns_dict():
    sp = MagicMock()
    sp.get_provider_stats_summary = MagicMock(return_value={
        "provider-a": {
            "reputation": 0.95,
            "total_challenges": 100,
            "successful_proofs": 95,
            "failed_proofs": 3,
            "expired_challenges": 2,
        },
        "provider-b": {
            "reputation": 0.50,
            "total_challenges": 10,
            "successful_proofs": 5,
            "failed_proofs": 5,
            "expired_challenges": 0,
        },
    })
    resp = _client(sp).get("/storage/provider-reputations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert "provider-a" in body["providers"]
    assert body["providers"]["provider-a"]["reputation"] == 0.95


def test_provider_reps_empty_dict():
    sp = MagicMock()
    sp.get_provider_stats_summary = MagicMock(return_value={})
    resp = _client(sp).get("/storage/provider-reputations")
    body = resp.json()
    assert body["count"] == 0
    assert body["providers"] == {}
