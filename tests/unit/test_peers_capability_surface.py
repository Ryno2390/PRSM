"""Sprint 326 — Surface peer capabilities on /peers + prsm_peers.

Sprint 322 threaded bootstrap-reported capabilities into
PeerInfo so `find_peers_with_capability` finally returns the
right candidate set. But the operator-facing surfaces lagged:

  - `GET /peers` `known[]` items rendered only `node_id`,
    `address`, `display_name`, `last_seen` — no capabilities
    field, so operators couldn't see which peers are GPU
    nodes etc.
  - `prsm_peers` MCP handler doesn't render the `known` list
    at all; operators relying on it for triage only see
    transport-level connected peers (typically empty since
    libp2p direct-dialing isn't wired to the wss:// bootstrap
    server).

Sprint 326 threads capabilities through both surfaces so the
sprint-322 work pays off in operator UX.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.mcp_server import handle_prsm_peers
from prsm.node.discovery import PeerInfo


# ── /peers endpoint surface ───────────────────────────────


def _make_test_client(known_peers: List[PeerInfo]):
    """Build a minimal TestClient with the /peers endpoint
    wired against a stub discovery."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    node.identity.node_id = "self"
    node.transport.peers = {}
    node.transport.peer_addresses = []
    node.transport.peer_count = 0
    discovery = MagicMock()
    discovery.get_known_peers = MagicMock(return_value=known_peers)
    node.discovery = discovery
    node.ftns_ledger = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_peers_endpoint_renders_capabilities_in_known():
    """Known peers must surface their capabilities list so
    operators can find compute/gpu/storage nodes via /peers."""
    client = _make_test_client([
        PeerInfo(
            node_id="gpu-node",
            address="10.0.0.1:9001",
            capabilities=["compute", "gpu"],
            last_seen=time.time(),
        ),
    ])
    resp = client.get("/peers")
    assert resp.status_code == 200
    body = resp.json()
    known = body.get("known") or []
    assert len(known) == 1
    entry = known[0]
    assert entry["node_id"] == "gpu-node"
    # Sprint 326 — capabilities must surface
    assert "capabilities" in entry
    assert set(entry["capabilities"]) == {"compute", "gpu"}


def test_peers_endpoint_renders_empty_capabilities_for_legacy_peers():
    """Peers with no capabilities should still render — caps
    field present as empty list (not missing)."""
    client = _make_test_client([
        PeerInfo(
            node_id="bare-peer",
            address="10.0.0.2:9001",
            last_seen=time.time(),
        ),
    ])
    resp = client.get("/peers")
    body = resp.json()
    known = body[0] if isinstance(body, list) else (body.get("known") or [])
    assert known
    assert "capabilities" in known[0]
    assert known[0]["capabilities"] == []


# ── prsm_peers MCP renders the known list with capabilities


def test_prsm_peers_renders_known_list_with_capabilities():
    """The MCP handler must surface known peers (not just
    transport-connected) including their capabilities — that's
    where bootstrap-discovered peers live."""
    payload = {
        "connected": [],
        "connected_count": 0,
        "known": [
            {
                "node_id": "gpu-node",
                "address": "10.0.0.1:9001",
                "display_name": "",
                "last_seen": time.time(),
                "capabilities": ["compute", "gpu"],
            },
            {
                "node_id": "storage-node",
                "address": "10.0.0.2:9001",
                "display_name": "",
                "last_seen": time.time(),
                "capabilities": ["storage"],
            },
        ],
        "known_count": 2,
    }
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_peers({}))
    # Both peers appear in the output
    assert "gpu-node" in out
    assert "storage-node" in out
    # Their capabilities are rendered
    assert "compute" in out
    assert "gpu" in out
    assert "storage" in out


def test_prsm_peers_shows_known_count_when_no_connected():
    """If transport has 0 connected but discovery has known
    peers, the handler must NOT say 'No peers connected' as
    the only message — it must surface the known list."""
    payload = {
        "connected": [],
        "connected_count": 0,
        "known": [
            {"node_id": "k1", "address": "x:1",
             "display_name": "", "last_seen": time.time(),
             "capabilities": []},
        ],
        "known_count": 1,
    }
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_peers({}))
    assert "k1" in out, (
        "Known peers should be surfaced even when "
        "transport-connected count is zero"
    )


def test_prsm_peers_handles_legacy_payload_without_known_field():
    """Backward-compat: older nodes don't include the `known`
    field. Handler must still render connected peers cleanly."""
    payload = {
        "connected": [
            {"peer_id": "p1", "address": "x:1",
             "outbound": True, "display_name": ""},
        ],
        "connected_count": 1,
    }
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_peers({}))
    assert "p1" in out
