"""Sprint 569 — POST /peers/connect endpoint (auto-dial gap fix).

Sprint 567 identified the auto-dial gap: `_hydrate_peers_from_bootstrap`
populates `known[]` (capability index) but never invokes
`transport.connect_to_peer()` on discovered peers, so `connected[]`
stays empty. Operators have no way to TURN a known peer into a
connected one.

Sprint 569 ships the simplest operator-facing trigger:

    POST /peers/connect {address: "host:port"}
      200 → {connected: true, peer_id, address}
      400 → invalid address
      502 → transport.connect_to_peer returned None
      503 → transport not initialized
      500 → transport raised

This is the foundation for any future "auto-dial on discovery"
sprint — the manual trigger first, then a background sweep that
calls the same endpoint per new known entry. It also enables
operator-side debugging of why a known peer can't be connected
(403 from a remote firewall, 404 from a stale registration, etc.).

Live-verify path: with F20 cleared (sprint 568), Mac can POST to
its own /peers/connect with the droplet's external IP. If the WS
handshake completes both sides exchange public_key + capabilities,
Mac's connected[] populates with the droplet, and direct cross-host
P2P works end-to-end for the first time.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient


def _stub_node(transport=None):
    n = MagicMock()
    n.identity = MagicMock(node_id="stub-node")
    n.transport = transport
    return n


def _make_app(transport=None):
    from prsm.node.api import create_api_app
    return create_api_app(
        _stub_node(transport=transport), enable_security=False,
    )


def test_endpoint_503_when_transport_not_initialized():
    """Pre-init / no transport → 503 with actionable detail."""
    app = _make_app(transport=None)
    client = TestClient(app)
    response = client.post(
        "/peers/connect", json={"address": "1.2.3.4:9001"},
    )
    assert response.status_code == 503
    assert "transport" in response.json()["detail"].lower()


def test_endpoint_400_on_empty_address():
    """Empty address → 400 with hint."""
    transport = MagicMock()
    app = _make_app(transport=transport)
    client = TestClient(app)
    response = client.post(
        "/peers/connect", json={"address": ""},
    )
    assert response.status_code == 400


def test_endpoint_400_on_missing_address_field():
    """No `address` in body → 422 from Pydantic (FastAPI default)."""
    transport = MagicMock()
    app = _make_app(transport=transport)
    client = TestClient(app)
    response = client.post("/peers/connect", json={})
    assert response.status_code == 422


def test_endpoint_502_when_connect_returns_none():
    """transport.connect_to_peer returned None (remote unreachable,
    handshake failed, etc.) → 502 so operators distinguish remote
    failure from a local transport problem."""
    transport = MagicMock()
    transport.connect_to_peer = AsyncMock(return_value=None)
    app = _make_app(transport=transport)
    client = TestClient(app)
    response = client.post(
        "/peers/connect", json={"address": "1.2.3.4:9001"},
    )
    assert response.status_code == 502
    transport.connect_to_peer.assert_awaited_once_with("1.2.3.4:9001")


def test_endpoint_500_when_connect_raises():
    """transport.connect_to_peer raised an unexpected exception →
    500 with the exception class + message (operator triage)."""
    transport = MagicMock()
    transport.connect_to_peer = AsyncMock(
        side_effect=RuntimeError("OS error 113 no route"),
    )
    app = _make_app(transport=transport)
    client = TestClient(app)
    response = client.post(
        "/peers/connect", json={"address": "1.2.3.4:9001"},
    )
    assert response.status_code == 500
    body = response.json()["detail"]
    assert "no route" in body or "OS error" in body


def test_endpoint_200_on_successful_connect():
    """transport.connect_to_peer returned a PeerConnection-shaped
    object → 200 with peer_id + address."""
    transport = MagicMock()
    peer = MagicMock()
    peer.peer_id = "remote-peer-123"
    peer.address = "1.2.3.4:9001"
    transport.connect_to_peer = AsyncMock(return_value=peer)

    app = _make_app(transport=transport)
    client = TestClient(app)
    response = client.post(
        "/peers/connect", json={"address": "1.2.3.4:9001"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["connected"] is True
    assert body["peer_id"] == "remote-peer-123"
    assert body["address"] == "1.2.3.4:9001"


def test_endpoint_accepts_ws_url_form():
    """Operator can pass `ws://host:port` or `wss://host:port` —
    transport's connect_to_peer handles all three forms (host:port,
    ws://, wss://)."""
    transport = MagicMock()
    peer = MagicMock()
    peer.peer_id = "remote-peer"
    peer.address = "ws://1.2.3.4:9001"
    transport.connect_to_peer = AsyncMock(return_value=peer)

    app = _make_app(transport=transport)
    client = TestClient(app)
    response = client.post(
        "/peers/connect", json={"address": "ws://1.2.3.4:9001"},
    )
    assert response.status_code == 200
    transport.connect_to_peer.assert_awaited_once_with(
        "ws://1.2.3.4:9001",
    )
