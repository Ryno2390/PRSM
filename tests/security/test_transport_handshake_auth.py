"""Focused security tests for transport handshake authentication hardening."""

import asyncio

import pytest
import websockets

from prsm.node.identity import generate_node_identity
from prsm.node.transport import MSG_HANDSHAKE, P2PMessage, WebSocketTransport


@pytest.mark.asyncio
async def test_valid_signed_handshake_still_connects() -> None:
    """Legitimate peers with valid signed handshakes should still connect."""
    id1 = generate_node_identity("secure-node-1")
    id2 = generate_node_identity("secure-node-2")
    t1 = WebSocketTransport(id1, host="127.0.0.1", port=19910)
    t2 = WebSocketTransport(id2, host="127.0.0.1", port=19911)

    try:
        await t1.start()
        await t2.start()

        peer = await t2.connect_to_peer("127.0.0.1:19910")
        assert peer is not None
        await asyncio.sleep(0.2)

        assert t1.peer_count == 1
        assert t2.peer_count == 1
    finally:
        await t2.stop()
        await t1.stop()


@pytest.mark.asyncio
async def test_unsigned_handshake_is_rejected_before_peer_promotion() -> None:
    """Unauthenticated handshakes fail closed and never become active peers."""
    server_id = generate_node_identity("server")
    attacker_id = generate_node_identity("attacker")
    transport = WebSocketTransport(server_id, host="127.0.0.1", port=19912)

    try:
        await transport.start()

        uri = "ws://127.0.0.1:19912"
        async with websockets.client.connect(uri) as ws:
            hs = P2PMessage(
                msg_type=MSG_HANDSHAKE,
                sender_id=attacker_id.node_id,
                payload={
                    "public_key": attacker_id.public_key_b64,
                    "display_name": "attacker",
                },
                signature="",  # weak-auth path must be rejected deterministically
            )
            await ws.send(hs.to_json())

            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await ws.recv()

        await asyncio.sleep(0.1)
        assert transport.peer_count == 0
    finally:
        await transport.stop()


@pytest.mark.asyncio
async def test_sender_id_must_match_public_key_identity() -> None:
    """Session identity must be bound to the validated public key-derived node identity."""
    server_id = generate_node_identity("server")
    attacker_id = generate_node_identity("attacker")
    transport = WebSocketTransport(server_id, host="127.0.0.1", port=19913)

    try:
        await transport.start()

        uri = "ws://127.0.0.1:19913"
        async with websockets.client.connect(uri) as ws:
            hs = P2PMessage(
                msg_type=MSG_HANDSHAKE,
                sender_id="0" * 32,  # does not match attacker_id.public_key_b64
                payload={
                    "public_key": attacker_id.public_key_b64,
                    "display_name": "attacker",
                },
            )
            hs.sign(attacker_id)
            await ws.send(hs.to_json())

            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await ws.recv()

        await asyncio.sleep(0.1)
        assert transport.peer_count == 0
    finally:
        await transport.stop()


@pytest.mark.asyncio
async def test_missing_public_key_handshake_is_rejected_as_weak_auth() -> None:
    """Downgrade-like weak-auth handshakes without identity material are rejected."""
    server_id = generate_node_identity("server")
    peer_id = generate_node_identity("peer")
    transport = WebSocketTransport(server_id, host="127.0.0.1", port=19914)

    try:
        await transport.start()

        uri = "ws://127.0.0.1:19914"
        async with websockets.client.connect(uri) as ws:
            hs = P2PMessage(
                msg_type=MSG_HANDSHAKE,
                sender_id=peer_id.node_id,
                payload={"display_name": "peer"},  # missing public_key must fail closed
            )
            hs.sign(peer_id)
            await ws.send(hs.to_json())

            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await ws.recv()

        await asyncio.sleep(0.1)
        assert transport.peer_count == 0
    finally:
        await transport.stop()

