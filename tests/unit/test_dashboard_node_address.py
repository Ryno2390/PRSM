"""Sprint 159 — DashboardServer /api/node `address` field reflects
the local node's LISTENING address, not a remote inbound peer.

Pre-fix the dashboard read `transport.peer_addresses[0]` which is
the list of REMOTE inbound peer addresses, NOT the local node's
listen address. So /api/node either:
  - Returned a peer's address as our own (wrong)
  - Returned "unknown" when no inbound peers existed (the common
    dogfood state — also wrong since we DO have listen addresses)

Live dogfood reproduced:
  /status   → p2p_address: "ws://0.0.0.0:9001"  (correct)
  /api/node → address:     "unknown"            (wrong)

Fix: derive from `node.config.listen_host:p2p_port` matching how
the main /status endpoint computes it.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")


def _build_node(*, listen_host="0.0.0.0", p2p_port=9001,
                inbound_peer_addrs=()):
    node = MagicMock()
    node.identity.node_id = "test-node-id"
    node.transport = MagicMock()
    node.transport.peer_addresses = list(inbound_peer_addrs)
    node.transport.peer_count = len(inbound_peer_addrs)
    node.transport.peers = {}
    node.config = MagicMock()
    node.config.listen_host = listen_host
    node.config.p2p_port = p2p_port
    node.compute_provider = None
    node.storage_provider = None
    node.agent_registry = None
    return node


def _client(node):
    from prsm.dashboard.app import DashboardServer
    server = DashboardServer(node=node)
    return TestClient(server.app)


def test_address_derived_from_node_config():
    """Sprint 159 — /api/node returns ws://listen_host:p2p_port."""
    node = _build_node(listen_host="0.0.0.0", p2p_port=9001)
    resp = _client(node).get("/api/node")
    body = resp.json()
    assert body["address"] == "ws://0.0.0.0:9001"


def test_address_does_not_leak_inbound_peer():
    """Sprint 159 invariant — even with inbound peers connected,
    the node's `address` is its OWN listen URL, not a peer's."""
    node = _build_node(
        listen_host="192.168.1.10", p2p_port=9100,
        inbound_peer_addrs=[
            "ws://203.0.113.5:9001",  # remote peer
            "ws://198.51.100.7:9001",
        ],
    )
    resp = _client(node).get("/api/node")
    body = resp.json()
    assert body["address"] == "ws://192.168.1.10:9100"
    assert "203.0.113.5" not in body["address"]


def test_address_unknown_when_no_config():
    """Defensive — if config is missing, surface "unknown" rather
    than crashing or returning a wrong value."""
    node = _build_node()
    node.config = None
    resp = _client(node).get("/api/node")
    body = resp.json()
    assert body["address"] == "unknown"
