"""
Tests for prsm.node.discovery — peer discovery with bootstrap and gossip.
"""

import asyncio
from unittest.mock import patch

import pytest

_REAL_SLEEP = asyncio.sleep

from prsm.node.discovery import PeerDiscovery, PeerInfo


@pytest.fixture(autouse=True)
def real_asyncio_sleep():
    """Restore real asyncio.sleep for network tests."""
    with patch("asyncio.sleep", _REAL_SLEEP):
        yield
from prsm.node.identity import generate_node_identity
from prsm.node.transport import WebSocketTransport


class TestPeerDiscovery:
    @pytest.mark.asyncio
    async def test_bootstrap_connects(self):
        """Bootstrap connects to a known node."""
        id1 = generate_node_identity("bootstrap-node")
        id2 = generate_node_identity("joining-node")
        t1 = WebSocketTransport(id1, host="127.0.0.1", port=19100)
        t2 = WebSocketTransport(id2, host="127.0.0.1", port=19101)

        d1 = PeerDiscovery(t1, bootstrap_nodes=[])
        d2 = PeerDiscovery(t2, bootstrap_nodes=["127.0.0.1:19100"])

        try:
            await t1.start()
            await t2.start()

            connected = await d2.bootstrap()
            assert connected == 1
            await asyncio.sleep(0.2)
            assert t2.peer_count == 1
            assert t1.peer_count == 1

        finally:
            await d1.stop()
            await d2.stop()
            await t2.stop()
            await t1.stop()

    @pytest.mark.asyncio
    async def test_bootstrap_no_nodes(self):
        """Bootstrap with empty list succeeds (first node on network)."""
        identity = generate_node_identity()
        transport = WebSocketTransport(identity, host="127.0.0.1", port=19102)
        discovery = PeerDiscovery(transport, bootstrap_nodes=[])

        try:
            await transport.start()
            connected = await discovery.bootstrap()
            assert connected == 0
        finally:
            await discovery.stop()
            await transport.stop()

    @pytest.mark.asyncio
    async def test_announce_self(self):
        """Announcement reaches connected peers."""
        id1 = generate_node_identity()
        id2 = generate_node_identity()
        t1 = WebSocketTransport(id1, host="127.0.0.1", port=19103)
        t2 = WebSocketTransport(id2, host="127.0.0.1", port=19104)

        d1 = PeerDiscovery(t1)
        d2 = PeerDiscovery(t2, bootstrap_nodes=["127.0.0.1:19103"])

        try:
            await t1.start()
            await t2.start()
            await d2.bootstrap()
            await asyncio.sleep(0.2)

            sent = await d2.announce_self()
            assert sent >= 1  # sent to at least 1 peer

            await asyncio.sleep(0.3)
            # d1 should now know about d2
            assert id2.node_id in d1.known_peers

        finally:
            await d2.stop()
            await d1.stop()
            await t2.stop()
            await t1.stop()

    @pytest.mark.asyncio
    async def test_peer_list_exchange(self):
        """Nodes exchange peer lists during discovery."""
        ids = [generate_node_identity(f"node-{i}") for i in range(3)]
        transports = [WebSocketTransport(ids[i], host="127.0.0.1", port=19110 + i) for i in range(3)]
        discoveries = [
            PeerDiscovery(transports[0]),
            PeerDiscovery(transports[1], bootstrap_nodes=["127.0.0.1:19110"]),
            PeerDiscovery(transports[2], bootstrap_nodes=["127.0.0.1:19110"]),
        ]

        try:
            for t in transports:
                await t.start()

            # Node 1 and 2 bootstrap to node 0
            await discoveries[1].bootstrap()
            await discoveries[2].bootstrap()
            await asyncio.sleep(0.3)

            # Node 0 should see both peers
            assert transports[0].peer_count == 2

        finally:
            for d in discoveries:
                await d.stop()
            for t in transports:
                await t.stop()

    @pytest.mark.asyncio
    async def test_known_peers_tracking(self):
        """Known peers are tracked from announcements."""
        discovery = PeerDiscovery.__new__(PeerDiscovery)
        discovery.known_peers = {}
        discovery.bootstrap_nodes = []
        discovery._running = False
        discovery._tasks = []

        # Manually add a known peer
        discovery.known_peers["node-abc"] = PeerInfo(
            node_id="node-abc",
            address="10.0.0.1:9001",
            display_name="Test",
        )

        peers = discovery.get_known_peers()
        assert len(peers) == 1
        assert peers[0].node_id == "node-abc"
