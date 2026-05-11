"""Sprint 203 — /content/retrieve/{cid}?timeout bounds.

Pre-fix: `timeout: float = 30.0` query param is passed unmodified
to `content_provider.request_content(timeout=...)`. Sending
?timeout=Infinity or ?timeout=999999 ties up a worker thread
indefinitely — slow-loris-style DoS.

Body-guard middleware (sprint 201) only catches JSON body; query
params bypass it. Add explicit bounds: [0.1, PRSM_MAX_RETRIEVE_
TIMEOUT_SEC] (default 300 = 5min).
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
    node.content_provider = MagicMock()
    node.content_provider.get_stats = MagicMock(return_value={
        "providers_tried": 0,
    })
    node.content_provider.request_content = AsyncMock(return_value=None)
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_inf_timeout_rejected():
    resp = _client().get(
        "/content/retrieve/somecid", params={"timeout": "Infinity"},
    )
    assert resp.status_code == 422


def test_nan_timeout_rejected():
    resp = _client().get(
        "/content/retrieve/somecid", params={"timeout": "NaN"},
    )
    assert resp.status_code == 422


def test_negative_timeout_rejected():
    resp = _client().get(
        "/content/retrieve/somecid", params={"timeout": "-1"},
    )
    assert resp.status_code == 422


def test_excessive_timeout_rejected():
    resp = _client().get(
        "/content/retrieve/somecid", params={"timeout": "999999"},
    )
    assert resp.status_code == 422


def test_typical_timeout_passes():
    resp = _client().get(
        "/content/retrieve/somecid", params={"timeout": "30"},
    )
    assert resp.status_code != 422


def test_default_timeout_passes():
    resp = _client().get("/content/retrieve/somecid")
    assert resp.status_code != 422


def test_env_override_cap(monkeypatch):
    monkeypatch.setenv("PRSM_MAX_RETRIEVE_TIMEOUT_SEC", "5")
    resp = _client().get(
        "/content/retrieve/somecid", params={"timeout": "10"},
    )
    assert resp.status_code == 422
