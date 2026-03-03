"""Replay/downgrade-focused handshake tests for WebSocketTransport."""

import pytest

from prsm.node.identity import generate_node_identity
from prsm.node.transport import MSG_HANDSHAKE, MSG_HANDSHAKE_ACK, P2PMessage, WebSocketTransport


@pytest.mark.asyncio
async def test_handshake_nonce_replay_rejected_deterministically() -> None:
    """The same handshake nonce cannot be accepted twice during authentication."""
    server_id = generate_node_identity("server")
    peer_id = generate_node_identity("peer")
    transport = WebSocketTransport(server_id, host="127.0.0.1", port=19920)

    msg = P2PMessage(
        msg_type=MSG_HANDSHAKE,
        sender_id=peer_id.node_id,
        payload={
            "public_key": peer_id.public_key_b64,
            "display_name": "peer",
        },
    )
    msg.sign(peer_id)

    ok1, reason1 = await transport._validate_handshake_message(msg, require_ack_for=False)
    ok2, reason2 = await transport._validate_handshake_message(msg, require_ack_for=False)

    assert ok1 is True
    assert reason1 == ""
    assert ok2 is False
    assert reason2 == "Replay nonce"


@pytest.mark.asyncio
async def test_ack_without_binding_rejected_as_downgrade_like() -> None:
    """Handshake ACK must bind to the original handshake nonce (ack_for required)."""
    local_id = generate_node_identity("local")
    remote_id = generate_node_identity("remote")
    transport = WebSocketTransport(local_id, host="127.0.0.1", port=19921)

    ack = P2PMessage(
        msg_type=MSG_HANDSHAKE_ACK,
        sender_id=remote_id.node_id,
        payload={
            "public_key": remote_id.public_key_b64,
            "display_name": "remote",
            # missing ack_for: downgrade-like weak-auth attempt
        },
    )
    ack.sign(remote_id)

    ok, reason = await transport._validate_handshake_message(
        ack,
        require_ack_for=True,
        expected_ack_for="expected-client-handshake-nonce",
    )

    assert ok is False
    assert reason == "Missing ack binding"


@pytest.mark.asyncio
async def test_ack_with_wrong_binding_rejected_deterministically() -> None:
    """Handshake ACK with mismatched ack_for is rejected deterministically."""
    local_id = generate_node_identity("local")
    remote_id = generate_node_identity("remote")
    transport = WebSocketTransport(local_id, host="127.0.0.1", port=19922)

    ack = P2PMessage(
        msg_type=MSG_HANDSHAKE_ACK,
        sender_id=remote_id.node_id,
        payload={
            "public_key": remote_id.public_key_b64,
            "display_name": "remote",
            "ack_for": "different-nonce",
        },
    )
    ack.sign(remote_id)

    ok, reason = await transport._validate_handshake_message(
        ack,
        require_ack_for=True,
        expected_ack_for="expected-client-handshake-nonce",
    )

    assert ok is False
    assert reason == "Ack nonce mismatch"

