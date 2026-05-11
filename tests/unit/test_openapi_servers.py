"""Sprint 189 — OpenAPI spec declares a `servers` field.

Pre-fix /openapi.json had `servers: null` — openapi-generator and
similar code-generation tools couldn't prefill an API base URL,
leaving operators to manually configure each generated client.

Post-fix /openapi.json carries:
  servers: [{"url": "http://127.0.0.1:8000",
              "description": "PRSM node"}]

Operators on non-default deploys override via `PRSM_API_BASE_URL`
env var (e.g. when the node sits behind a reverse proxy at
`https://prsm.example.com`).
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    return TestClient(create_api_app(node, enable_security=False))


def test_openapi_servers_present():
    """Sprint 189 — servers field is present + non-empty."""
    spec = _client().get("/openapi.json").json()
    servers = spec.get("servers")
    assert servers is not None
    assert isinstance(servers, list)
    assert len(servers) >= 1


def test_openapi_servers_default_url():
    """Default url is the local-dev endpoint."""
    spec = _client().get("/openapi.json").json()
    servers = spec["servers"]
    assert servers[0]["url"] == "http://127.0.0.1:8000"


def test_openapi_servers_override_via_env(monkeypatch):
    """Sprint 189 — PRSM_API_BASE_URL env overrides."""
    monkeypatch.setenv(
        "PRSM_API_BASE_URL", "https://prsm.example.com",
    )
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    client = TestClient(create_api_app(node, enable_security=False))
    spec = client.get("/openapi.json").json()
    assert spec["servers"][0]["url"] == "https://prsm.example.com"


def test_openapi_servers_empty_env_falls_back_to_default(monkeypatch):
    """Whitespace-only env value falls back to default — not an
    empty URL that would break clients."""
    monkeypatch.setenv("PRSM_API_BASE_URL", "   ")
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    client = TestClient(create_api_app(node, enable_security=False))
    spec = client.get("/openapi.json").json()
    assert spec["servers"][0]["url"] == "http://127.0.0.1:8000"
