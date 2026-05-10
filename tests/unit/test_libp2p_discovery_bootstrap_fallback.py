"""Sprint 164 — Libp2pDiscovery bootstrap fallback to BootstrapClient.

Pre-fix Libp2pDiscovery.bootstrap() called only
``transport.connect_to_peer(addr)`` for each bootstrap address. The
PRSM bootstrap server speaks a simpler register/heartbeat
WebSocket protocol on ``wss://bootstrap1.prsm-network.com:8765``,
NOT the libp2p P2P handshake. So when the configured bootstrap
addresses are wss:// URLs (the canonical sprint 148 default), the
libp2p connect fails silently and discovery sits in degraded
mode forever — `attempted=1, connected=0, degraded=true`.

Live dogfood reproduced repeatedly throughout sprints 148-150.

Legacy PeerDiscovery (prsm/node/discovery.py) had a
``_try_bootstrap_client`` fallback that switched to BootstrapClient
when the libp2p path failed. Libp2pDiscovery never got that
fallback ported over.

Sprint 164 ports the fallback so a node configured with a wss://
bootstrap address actually connects through the right protocol.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.node.libp2p_discovery import Libp2pDiscovery


def _make_discovery(*, bootstrap_nodes=None, transport=None):
    if transport is None:
        transport = MagicMock()
        transport.identity.node_id = "test-node"
        transport.port = 9001
        transport.connect_to_peer = AsyncMock(return_value=None)
    return Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=bootstrap_nodes or [],
    )


def test_libp2p_only_path_when_succeeds():
    """When libp2p connect_to_peer returns a peer, no fallback fires."""
    transport = MagicMock()
    transport.identity.node_id = "n"
    transport.port = 9001
    transport.connect_to_peer = AsyncMock(return_value=MagicMock())
    d = _make_discovery(
        bootstrap_nodes=["wss://bootstrap.example.com:8765"],
        transport=transport,
    )
    connected = asyncio.run(d.bootstrap())
    assert connected == 1
    assert d._bootstrap_status["connected"] == 1
    assert d._bootstrap_status["degraded"] is False


def test_fallback_fires_when_libp2p_returns_none_and_addr_is_wss():
    """Sprint 164 — when libp2p path returns None for a wss:// addr,
    Libp2pDiscovery falls back to BootstrapClient just like the
    legacy PeerDiscovery does."""
    transport = MagicMock()
    transport.identity.node_id = "n"
    transport.port = 9001
    transport.connect_to_peer = AsyncMock(return_value=None)

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def connect(self):
            return []  # no peers discovered, but successful registration

        async def start_heartbeat(self):
            pass

    with patch(
        "prsm.bootstrap.client.BootstrapClient", _FakeClient,
    ):
        d = _make_discovery(
            bootstrap_nodes=["wss://bootstrap.example.com:8765"],
            transport=transport,
        )
        connected = asyncio.run(d.bootstrap())

    assert connected == 1, (
        "fallback path should mark connected=1 on successful "
        "BootstrapClient registration"
    )
    assert d._bootstrap_status["degraded"] is False
    assert captured.get("bootstrap_url") == "wss://bootstrap.example.com:8765"


def test_no_fallback_when_addr_is_not_websocket():
    """Multiaddr-style addresses (/dns4/.../tcp/.../p2p/...) don't
    use the BootstrapClient protocol — fallback must SKIP them."""
    transport = MagicMock()
    transport.identity.node_id = "n"
    transport.port = 9001
    transport.connect_to_peer = AsyncMock(return_value=None)

    with patch(
        "prsm.bootstrap.client.BootstrapClient",
    ) as MockClient:
        d = _make_discovery(
            bootstrap_nodes=["/dns4/peer.example.com/tcp/9001/p2p/QmFoo"],
            transport=transport,
        )
        connected = asyncio.run(d.bootstrap())
    assert connected == 0
    assert d._bootstrap_status["degraded"] is True
    MockClient.assert_not_called()


def test_fallback_failure_keeps_degraded_state():
    """Fallback BootstrapClient.connect() raises → keeps degraded
    mode, doesn't crash."""
    transport = MagicMock()
    transport.identity.node_id = "n"
    transport.port = 9001
    transport.connect_to_peer = AsyncMock(return_value=None)

    class _BrokenClient:
        def __init__(self, **kwargs):
            pass

        async def connect(self):
            raise RuntimeError("simulated network failure")

    with patch(
        "prsm.bootstrap.client.BootstrapClient", _BrokenClient,
    ):
        d = _make_discovery(
            bootstrap_nodes=["wss://bootstrap.example.com:8765"],
            transport=transport,
        )
        connected = asyncio.run(d.bootstrap())

    assert connected == 0
    assert d._bootstrap_status["degraded"] is True
