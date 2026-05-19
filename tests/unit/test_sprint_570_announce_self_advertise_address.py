"""Sprint 570 — F28 fix: announce_self must not gossip 0.0.0.0:port.

Sprint 569 surfaced F28 during multi-host live verification: Mac's
``known_peers[droplet_id].address`` showed ``"0.0.0.0:9001"`` even
though the bootstrap-server's peers.db correctly stored the droplet
as ``"159.203.129.218:9001"`` (sprint-566 wiring works end-to-end at
the bootstrap layer).

Root cause: ``PeerDiscovery.announce_self`` (and the mirror in
libp2p_discovery) builds a DISCOVERY_ANNOUNCE gossip with
``"address": f"{self.transport.host}:{self.transport.port}"``. But
``transport.host`` is the LISTEN host (typically "0.0.0.0" — bind
to all interfaces). It is NOT a routable advertise address.

When that gossip arrives on a remote peer, ``_handle_announce``
unconditionally OVERWRITES ``known_peers[sender].address`` with the
``0.0.0.0:port`` value — wiping out the correct address that
bootstrap-server's peer_list response had just installed via
``_try_bootstrap_client``.

Operator-facing impact: ``maintain_connections`` then tries to
dial ``0.0.0.0:port``, which fails. /peers UI shows wrong address.
Sprint-569's ``/peers/connect`` workaround works only because the
operator manually supplies the correct address — but any gossip-
mediated reconnect goes through ``known_peers[id].address``, which
is poisoned.

Sprint 570 fix:

1. ``announce_self``: include ``"address"`` field ONLY when
   ``PRSM_ADVERTISE_ADDRESS`` is set (sprint-566's existing env
   var). If unset, omit the field entirely — recipients fall back
   to ``peer.address`` (the WS source-connection IP), which is the
   real reachable address for any inbound connection.

2. ``_handle_announce``: defensively ignore ``0.0.0.0:*``,
   ``0.0.0.0``, or empty payload addresses from peers still on
   pre-sprint-570 code. Falls back to ``peer.address`` (same as
   the missing-key path).

These are mutually-reinforcing: fix #1 prevents future poisoning;
fix #2 disinfects legacy peers we're talking to.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── announce_self payload shape ──────────────────────────────────


@pytest.mark.asyncio
async def test_announce_self_omits_address_when_advertise_unset():
    """Without PRSM_ADVERTISE_ADDRESS, announce_self payload must
    NOT contain a 'address' key (so 0.0.0.0:port can't leak).
    """
    from prsm.node.discovery import PeerDiscovery

    transport = MagicMock()
    transport.identity = MagicMock(node_id="self-node", display_name="x")
    transport.host = "0.0.0.0"
    transport.port = 9001
    transport.peer_count = 0
    transport.gossip = AsyncMock(return_value=0)

    d = PeerDiscovery(transport=transport, bootstrap_nodes=[])
    d._local_capabilities = []
    d._local_backends = []
    d._local_gpu_available = False

    with patch.dict(os.environ, {"PRSM_ADVERTISE_ADDRESS": ""}, clear=False):
        os.environ.pop("PRSM_ADVERTISE_ADDRESS", None)
        await d.announce_self()

    assert transport.gossip.await_count == 1
    sent_msg = transport.gossip.await_args.args[0]
    assert "address" not in sent_msg.payload, (
        "announce_self must not include 'address' when "
        "PRSM_ADVERTISE_ADDRESS is unset — listen_host (0.0.0.0) is "
        "not a routable advertise value"
    )


@pytest.mark.asyncio
async def test_announce_self_uses_advertise_address_when_set():
    """With PRSM_ADVERTISE_ADDRESS=ip, announce_self payload
    contains 'address': 'ip:port'.
    """
    from prsm.node.discovery import PeerDiscovery

    transport = MagicMock()
    transport.identity = MagicMock(node_id="self-node", display_name="x")
    transport.host = "0.0.0.0"
    transport.port = 9001
    transport.peer_count = 0
    transport.gossip = AsyncMock(return_value=0)

    d = PeerDiscovery(transport=transport, bootstrap_nodes=[])
    d._local_capabilities = []
    d._local_backends = []
    d._local_gpu_available = False

    with patch.dict(
        os.environ,
        {"PRSM_ADVERTISE_ADDRESS": "159.203.129.218"},
        clear=False,
    ):
        await d.announce_self()

    sent_msg = transport.gossip.await_args.args[0]
    assert sent_msg.payload.get("address") == "159.203.129.218:9001"


@pytest.mark.asyncio
async def test_announce_self_does_not_send_0_0_0_0_address():
    """No matter how transport.host is configured, announce_self
    must never emit a 0.0.0.0:port address (the F28 invariant).
    """
    from prsm.node.discovery import PeerDiscovery

    transport = MagicMock()
    transport.identity = MagicMock(node_id="self-node", display_name="x")
    transport.host = "0.0.0.0"
    transport.port = 9001
    transport.peer_count = 0
    transport.gossip = AsyncMock(return_value=0)

    d = PeerDiscovery(transport=transport, bootstrap_nodes=[])
    d._local_capabilities = []
    d._local_backends = []
    d._local_gpu_available = False

    os.environ.pop("PRSM_ADVERTISE_ADDRESS", None)
    await d.announce_self()

    sent_msg = transport.gossip.await_args.args[0]
    addr = sent_msg.payload.get("address", "")
    assert "0.0.0.0" not in addr, (
        "announce_self must never gossip 0.0.0.0 — it poisons "
        "remote known_peers registries"
    )


# ── _handle_announce defensive guard ─────────────────────────────


@pytest.mark.asyncio
async def test_handle_announce_rejects_0_0_0_0_address():
    """If a legacy pre-sprint-570 peer sends 'address': '0.0.0.0:port',
    _handle_announce must fall back to peer.address (the WS source
    connection IP), NOT store the bogus value in known_peers.
    """
    from prsm.node.discovery import PeerDiscovery
    from prsm.node.transport import P2PMessage, PeerConnection

    transport = MagicMock()
    transport.identity = MagicMock(node_id="self-node")
    transport.host = "0.0.0.0"
    transport.port = 9001

    d = PeerDiscovery(transport=transport, bootstrap_nodes=[])
    d._local_capabilities = []
    d._local_backends = []
    d._local_gpu_available = False

    peer = MagicMock(spec=PeerConnection)
    peer.address = "159.203.129.218:9001"
    peer.peer_id = "remote-node"

    msg = P2PMessage(
        msg_type=1,  # MSG_GOSSIP — int value, real constants vary
        sender_id="remote-node",
        payload={
            "subtype": "discovery_announce",
            "address": "0.0.0.0:9001",  # bogus from legacy peer
            "display_name": "remote",
            "roles": [],
            "capabilities": [],
            "supported_backends": [],
            "gpu_available": False,
        },
        ttl=1,
    )

    await d._handle_announce(msg, peer)

    stored = d.known_peers.get("remote-node")
    assert stored is not None
    assert stored.address != "0.0.0.0:9001", (
        "_handle_announce must defensively reject 0.0.0.0:port "
        "addresses from legacy peers — falling back to peer.address"
    )
    assert stored.address == "159.203.129.218:9001"


@pytest.mark.asyncio
async def test_handle_announce_accepts_real_address():
    """Valid (non-0.0.0.0) addresses in announce payload are stored
    as-is — no regression on the happy path.
    """
    from prsm.node.discovery import PeerDiscovery
    from prsm.node.transport import P2PMessage, PeerConnection

    transport = MagicMock()
    transport.identity = MagicMock(node_id="self-node")
    transport.host = "0.0.0.0"
    transport.port = 9001

    d = PeerDiscovery(transport=transport, bootstrap_nodes=[])
    d._local_capabilities = []
    d._local_backends = []
    d._local_gpu_available = False

    peer = MagicMock(spec=PeerConnection)
    peer.address = "8.8.8.8:9001"
    peer.peer_id = "remote-node"

    msg = P2PMessage(
        msg_type=1,
        sender_id="remote-node",
        payload={
            "subtype": "discovery_announce",
            "address": "159.203.129.218:9001",  # real
            "display_name": "remote",
            "roles": [],
            "capabilities": [],
            "supported_backends": [],
            "gpu_available": False,
        },
        ttl=1,
    )

    await d._handle_announce(msg, peer)

    stored = d.known_peers.get("remote-node")
    assert stored is not None
    assert stored.address == "159.203.129.218:9001", (
        "Real addresses must be stored as-is — no regression"
    )
