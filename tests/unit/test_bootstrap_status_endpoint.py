"""Sprint 266 — GET /bootstrap/status exposes PeerDiscovery
bootstrap state for operator triage.

Pre-fix the rich state from PeerDiscovery.get_bootstrap_status()
(configured nodes, attempted/failed sets, connected count,
degraded mode, retry attempts, fallback state, etc.) was
unsurfaced — operators couldn't tell from outside whether the
node was actually connected to bootstrap or just to random peers.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client(discovery=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.discovery = discovery
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_503_when_discovery_unwired():
    resp = _client(None).get("/bootstrap/status")
    assert resp.status_code == 503


def test_returns_status_dict():
    disco = MagicMock()
    disco.get_bootstrap_status = MagicMock(return_value={
        "configured_nodes": [
            "wss://bootstrap1.prsm-network.com:8765",
        ],
        "attempted_nodes": [
            "wss://bootstrap1.prsm-network.com:8765",
        ],
        "failed_nodes": [],
        "success_node": "wss://bootstrap1.prsm-network.com:8765",
        "connected_count": 1,
        "degraded_mode": False,
        "retry_attempts": 0,
        "connect_timeout_seconds": 5.0,
        "fallback_enabled": True,
        "fallback_activated": False,
        "fallback_succeeded": False,
        "addresses_rejected": 0,
        "source_policy": "primary_only",
        "bootstrap_client_active": True,
    })
    resp = _client(disco).get("/bootstrap/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected_count"] == 1
    assert body["degraded_mode"] is False
    assert (
        body["success_node"]
        == "wss://bootstrap1.prsm-network.com:8765"
    )


def test_surfaces_degraded_mode():
    disco = MagicMock()
    disco.get_bootstrap_status = MagicMock(return_value={
        "configured_nodes": [
            "wss://bootstrap1.prsm-network.com:8765",
        ],
        "attempted_nodes": [
            "wss://bootstrap1.prsm-network.com:8765",
        ],
        "failed_nodes": [
            "wss://bootstrap1.prsm-network.com:8765",
        ],
        "success_node": None,
        "connected_count": 0,
        "degraded_mode": True,
        "retry_attempts": 3,
        "connect_timeout_seconds": 5.0,
        "fallback_enabled": True,
        "fallback_activated": True,
        "fallback_succeeded": False,
        "addresses_rejected": 2,
        "source_policy": "primary_then_fallback",
        "bootstrap_client_active": False,
    })
    resp = _client(disco).get("/bootstrap/status")
    body = resp.json()
    assert body["degraded_mode"] is True
    assert body["fallback_activated"] is True
    assert body["fallback_succeeded"] is False
    assert body["addresses_rejected"] == 2
