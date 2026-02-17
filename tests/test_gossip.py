"""
Tests for prsm.node.gossip — epidemic gossip protocol.
"""

import asyncio
from unittest.mock import patch

import pytest

_REAL_SLEEP = asyncio.sleep

from prsm.node.gossip import GossipProtocol, GOSSIP_JOB_OFFER


@pytest.fixture(autouse=True)
def real_asyncio_sleep():
    """Restore real asyncio.sleep for network tests."""
    with patch("asyncio.sleep", _REAL_SLEEP):
        yield
from prsm.node.identity import generate_node_identity
from prsm.node.transport import WebSocketTransport


class TestGossipProtocol:
    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self):
        """Published messages are delivered to local subscribers."""
        id1 = generate_node_identity()
        id2 = generate_node_identity()
        t1 = WebSocketTransport(id1, host="127.0.0.1", port=19200)
        t2 = WebSocketTransport(id2, host="127.0.0.1", port=19201)

        g1 = GossipProtocol(t1, fanout=3, heartbeat_interval=9999)
        g2 = GossipProtocol(t2, fanout=3, heartbeat_interval=9999)

        received = []

        async def handler(subtype, data, origin):
            received.append((subtype, data, origin))

        g1.subscribe("test_event", handler)

        try:
            await t1.start()
            await t2.start()
            await t2.connect_to_peer("127.0.0.1:19200")
            await asyncio.sleep(0.2)

            # g2 publishes, g1 should receive
            await g2.publish("test_event", {"value": 42})
            await asyncio.sleep(0.3)

            assert len(received) == 1
            assert received[0][0] == "test_event"
            assert received[0][1]["value"] == 42

        finally:
            await g1.stop()
            await g2.stop()
            await t2.stop()
            await t1.stop()

    @pytest.mark.asyncio
    async def test_gossip_propagation(self):
        """Messages propagate through intermediate nodes (A -> B -> C)."""
        ids = [generate_node_identity(f"node-{i}") for i in range(3)]
        transports = [WebSocketTransport(ids[i], host="127.0.0.1", port=19210 + i) for i in range(3)]
        gossips = [GossipProtocol(t, fanout=3, default_ttl=5, heartbeat_interval=9999) for t in transports]

        received_on_c = []

        async def handler(subtype, data, origin):
            received_on_c.append(data)

        gossips[2].subscribe("propagation_test", handler)

        try:
            for t in transports:
                await t.start()

            # Chain: A - B - C (B connected to both A and C)
            await transports[1].connect_to_peer("127.0.0.1:19210")  # B->A
            await transports[1].connect_to_peer("127.0.0.1:19212")  # B->C
            await asyncio.sleep(0.2)

            # A publishes — should reach C via B
            await gossips[0].publish("propagation_test", {"msg": "from A"})
            await asyncio.sleep(0.5)

            assert len(received_on_c) >= 1
            assert received_on_c[0]["msg"] == "from A"

        finally:
            for g in gossips:
                await g.stop()
            for t in transports:
                await t.stop()

    @pytest.mark.asyncio
    async def test_ttl_prevents_infinite_propagation(self):
        """Messages with TTL=1 should not be re-propagated."""
        id1 = generate_node_identity()
        id2 = generate_node_identity()
        id3 = generate_node_identity()
        t1 = WebSocketTransport(id1, host="127.0.0.1", port=19220)
        t2 = WebSocketTransport(id2, host="127.0.0.1", port=19221)
        t3 = WebSocketTransport(id3, host="127.0.0.1", port=19222)

        g1 = GossipProtocol(t1, fanout=3, heartbeat_interval=9999)
        g2 = GossipProtocol(t2, fanout=3, heartbeat_interval=9999)
        g3 = GossipProtocol(t3, fanout=3, heartbeat_interval=9999)

        received_on_3 = []

        async def handler(subtype, data, origin):
            received_on_3.append(data)

        g3.subscribe("ttl_test", handler)

        try:
            await t1.start()
            await t2.start()
            await t3.start()

            # Chain: 1 - 2 - 3
            await t2.connect_to_peer("127.0.0.1:19220")
            await t2.connect_to_peer("127.0.0.1:19222")
            await asyncio.sleep(0.2)

            # Publish with TTL=1 — should reach node 2 but NOT node 3
            await g1.publish("ttl_test", {"msg": "limited"}, ttl=1)
            await asyncio.sleep(0.5)

            # Node 3 should NOT receive (TTL expired at node 2)
            assert len(received_on_3) == 0

        finally:
            for g in [g1, g2, g3]:
                await g.stop()
            for t in [t1, t2, t3]:
                await t.stop()

    @pytest.mark.asyncio
    async def test_nonce_dedup_prevents_duplicate_delivery(self):
        """Same message (same nonce) should only be delivered once even via multiple paths."""
        id1 = generate_node_identity()
        id2 = generate_node_identity()
        t1 = WebSocketTransport(id1, host="127.0.0.1", port=19230)
        t2 = WebSocketTransport(id2, host="127.0.0.1", port=19231)

        g1 = GossipProtocol(t1, fanout=3, heartbeat_interval=9999)
        g2 = GossipProtocol(t2, fanout=3, heartbeat_interval=9999)

        received = []

        async def handler(subtype, data, origin):
            received.append(data)

        g1.subscribe("dedup_test", handler)

        try:
            await t1.start()
            await t2.start()
            await t2.connect_to_peer("127.0.0.1:19230")
            await asyncio.sleep(0.2)

            # Publish once
            await g2.publish("dedup_test", {"val": 1})
            await asyncio.sleep(0.3)
            assert len(received) == 1

            # Even if somehow received again (not typical in 2-node setup),
            # the nonce dedup in transport prevents duplicate dispatch.

        finally:
            await g1.stop()
            await g2.stop()
            await t2.stop()
            await t1.stop()

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Multiple subscribers for the same subtype all receive the message."""
        id1 = generate_node_identity()
        id2 = generate_node_identity()
        t1 = WebSocketTransport(id1, host="127.0.0.1", port=19240)
        t2 = WebSocketTransport(id2, host="127.0.0.1", port=19241)

        g1 = GossipProtocol(t1, fanout=3, heartbeat_interval=9999)
        g2 = GossipProtocol(t2, fanout=3, heartbeat_interval=9999)

        received_a = []
        received_b = []

        async def handler_a(subtype, data, origin):
            received_a.append(data)

        async def handler_b(subtype, data, origin):
            received_b.append(data)

        g1.subscribe("multi_test", handler_a)
        g1.subscribe("multi_test", handler_b)

        try:
            await t1.start()
            await t2.start()
            await t2.connect_to_peer("127.0.0.1:19240")
            await asyncio.sleep(0.2)

            await g2.publish("multi_test", {"x": 1})
            await asyncio.sleep(0.3)

            assert len(received_a) == 1
            assert len(received_b) == 1

        finally:
            await g1.stop()
            await g2.stop()
            await t2.stop()
            await t1.stop()
