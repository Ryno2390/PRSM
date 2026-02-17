"""
Tests for prsm.node.transport — WebSocket P2P transport.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

# Capture REAL asyncio.sleep before conftest autouse fixtures patch it
_REAL_SLEEP = asyncio.sleep

from prsm.node.identity import generate_node_identity


@pytest.fixture(autouse=True)
def real_asyncio_sleep():
    """Restore real asyncio.sleep for transport tests (needs real I/O timing)."""
    with patch("asyncio.sleep", _REAL_SLEEP):
        yield
from prsm.node.transport import (
    MSG_DIRECT,
    MSG_GOSSIP,
    MSG_HANDSHAKE,
    MSG_PING,
    P2PMessage,
    PeerConnection,
    WebSocketTransport,
)


class TestP2PMessage:
    def test_roundtrip_serialization(self):
        msg = P2PMessage(
            msg_type="test",
            sender_id="abc123",
            payload={"key": "value"},
            ttl=3,
        )
        raw = msg.to_json()
        restored = P2PMessage.from_json(raw)
        assert restored.msg_type == "test"
        assert restored.sender_id == "abc123"
        assert restored.payload == {"key": "value"}
        assert restored.ttl == 3
        assert restored.nonce == msg.nonce

    def test_signing(self):
        identity = generate_node_identity()
        msg = P2PMessage(
            msg_type="test",
            sender_id=identity.node_id,
            payload={"data": 42},
        )
        msg.sign(identity)
        assert msg.signature != ""

        # Verify
        data_bytes = msg.to_bytes()
        assert identity.verify(data_bytes, msg.signature)

    def test_unique_nonces(self):
        msgs = [P2PMessage(msg_type="test", sender_id="x", payload={}) for _ in range(100)]
        nonces = {m.nonce for m in msgs}
        assert len(nonces) == 100


class TestWebSocketTransport:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        identity = generate_node_identity()
        transport = WebSocketTransport(identity, host="127.0.0.1", port=19001)
        await transport.start()
        assert transport._server is not None
        await transport.stop()
        assert transport._server is None

    @pytest.mark.asyncio
    async def test_two_nodes_connect(self):
        """Two transport instances can connect and exchange messages."""
        id1 = generate_node_identity("node-1")
        id2 = generate_node_identity("node-2")

        t1 = WebSocketTransport(id1, host="127.0.0.1", port=19002)
        t2 = WebSocketTransport(id2, host="127.0.0.1", port=19003)

        received = []

        async def on_msg(msg, peer):
            received.append(msg)

        t1.on_message(MSG_DIRECT, on_msg)

        try:
            await t1.start()
            await t2.start()

            # t2 connects to t1
            peer = await t2.connect_to_peer("127.0.0.1:19002")
            assert peer is not None
            assert peer.peer_id == id1.node_id

            # Wait for handshake to complete on both sides
            await asyncio.sleep(0.2)
            assert t1.peer_count == 1
            assert t2.peer_count == 1

            # t2 sends a message to t1
            msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id=id2.node_id,
                payload={"greeting": "hello from node-2"},
            )
            success = await t2.send_to_peer(id1.node_id, msg)
            assert success

            await asyncio.sleep(0.2)
            assert len(received) == 1
            assert received[0].payload["greeting"] == "hello from node-2"

        finally:
            await t2.stop()
            await t1.stop()

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Broadcast sends to all connected peers."""
        ids = [generate_node_identity(f"node-{i}") for i in range(3)]
        transports = [WebSocketTransport(ids[i], host="127.0.0.1", port=19010 + i) for i in range(3)]

        received_counts = [0, 0, 0]

        async def counter_factory(idx):
            async def handler(msg, peer):
                nonlocal received_counts
                received_counts[idx] += 1
            return handler

        try:
            for i, t in enumerate(transports):
                handler = await counter_factory(i)
                t.on_message(MSG_DIRECT, handler)
                await t.start()

            # Node 1 and 2 connect to node 0
            await transports[1].connect_to_peer("127.0.0.1:19010")
            await transports[2].connect_to_peer("127.0.0.1:19010")
            await asyncio.sleep(0.2)

            # Node 0 broadcasts
            msg = P2PMessage(msg_type=MSG_DIRECT, sender_id=ids[0].node_id, payload={"broadcast": True})
            sent = await transports[0].broadcast(msg)
            assert sent == 2

            await asyncio.sleep(0.2)
            # Nodes 1 and 2 should receive
            assert received_counts[1] == 1
            assert received_counts[2] == 1
            # Node 0 should not receive its own broadcast
            assert received_counts[0] == 0

        finally:
            for t in transports:
                await t.stop()

    @pytest.mark.asyncio
    async def test_self_connection_rejected(self):
        """A node should not be able to connect to itself."""
        identity = generate_node_identity()
        t = WebSocketTransport(identity, host="127.0.0.1", port=19020)
        try:
            await t.start()
            peer = await t.connect_to_peer("127.0.0.1:19020")
            # Should be rejected (self-connection)
            assert peer is None or peer.peer_id != identity.node_id
        finally:
            await t.stop()

    @pytest.mark.asyncio
    async def test_nonce_dedup(self):
        """Messages with the same nonce should be deduplicated."""
        id1 = generate_node_identity()
        id2 = generate_node_identity()
        t1 = WebSocketTransport(id1, host="127.0.0.1", port=19030)
        t2 = WebSocketTransport(id2, host="127.0.0.1", port=19031)

        received = []

        async def on_msg(msg, peer):
            received.append(msg)

        t1.on_message(MSG_DIRECT, on_msg)

        try:
            await t1.start()
            await t2.start()
            await t2.connect_to_peer("127.0.0.1:19030")
            await asyncio.sleep(0.2)

            # Send same message twice (same nonce)
            msg = P2PMessage(msg_type=MSG_DIRECT, sender_id=id2.node_id, payload={"dup": True})
            msg.sign(id2)
            raw = msg.to_json()

            peer = list(t2.peers.values())[0]
            await peer.websocket.send(raw)
            await peer.websocket.send(raw)  # duplicate

            await asyncio.sleep(0.3)
            assert len(received) == 1  # dedup should filter the second

        finally:
            await t2.stop()
            await t1.stop()
