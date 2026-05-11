"""Sprint 268 — GET /content/provider-stats.

Pre-fix ContentProvider.get_stats() was only called internally
by /content/retrieve to compute providers_tried delta. The
underlying stats (local_content_count, pending_requests,
discovery sub-stats, telemetry) had no external surface.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client(content_provider=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.content_provider = content_provider
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_503_when_provider_unwired():
    resp = _client(None).get("/content/provider-stats")
    assert resp.status_code == 503


def test_returns_stats_dict():
    cp = MagicMock()
    cp.get_stats = MagicMock(return_value={
        "local_content_count": 42,
        "pending_requests": 3,
        "discovery": {
            "queries_sent": 100,
            "responses_received": 75,
        },
        "total_fetches": 200,
        "successful_fetches": 180,
        "failed_fetches": 20,
    })
    resp = _client(cp).get("/content/provider-stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["local_content_count"] == 42
    assert body["pending_requests"] == 3
    assert body["discovery"]["queries_sent"] == 100
    assert body["total_fetches"] == 200


def test_provider_get_stats_raise_returns_500():
    cp = MagicMock()
    cp.get_stats = MagicMock(side_effect=RuntimeError("boom"))
    resp = _client(cp).get("/content/provider-stats")
    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]
