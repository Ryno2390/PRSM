"""Sprint 601 (Phase 2E-1) — server-side request handler scaffolding.

Phase 2 client-side wire (sprints 592-600) ships chain-executor
REQUEST messages from a node + processes RESPONSE messages. Phase
2 server-side ships the inverse: receive REQUESTs + send RESPONSEs.

Sprint 601 is Phase 2E-1: the request handler scaffolding. Decodes
the wire-protocol envelope, constructs a structured "not yet
implemented" response, and dispatches it back to the sender. Real
stage execution (Phase 2E-2+) plugs into the same wire later.

Tests:
  - handle_chain_executor_request reads CHAIN_REQ_KEY + decodes
    base64 payload
  - It constructs a response P2PMessage with CHAIN_RESP_KEY set
  - The response payload includes an error code indicating the
    stage handler hasn't landed yet (Phase 2E-2+)
  - Ignores messages with wrong subtype
  - Ignores messages already carrying CHAIN_RESP_KEY (those are
    responses to OUR outbound requests — sprint 597 handles those)
  - Tolerates malformed base64 (sends error response)
"""
from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock


def _make_msg(payload, sender_id="remote-peer"):
    m = MagicMock()
    m.payload = payload
    m.sender_id = sender_id
    return m


async def _no_peer():
    return None


def test_handler_function_exists():
    from prsm.node import chain_executor_adapters as m
    assert hasattr(m, "handle_chain_executor_request")


def test_handler_ignores_wrong_subtype():
    """Returns False for non-chain_executor_rpc messages."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_request, CHAIN_REQ_KEY,
    )

    node = MagicMock()
    node.identity.node_id = "self"
    node.transport.send_to_peer = AsyncMock(return_value=True)

    msg = _make_msg({
        "subtype": "something_else",
        CHAIN_REQ_KEY: "req-1",
    })

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            handle_chain_executor_request(node, msg),
        )
    finally:
        loop.close()
    assert result is False
    node.transport.send_to_peer.assert_not_called()


def test_handler_ignores_response_messages():
    """A msg with CHAIN_RESP_KEY is a RESPONSE to our outbound
    request (handled by sprint 597), not a REQUEST. The
    server-side handler must NOT process it.
    """
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_request,
        CHAIN_MSG_TYPE, CHAIN_REQ_KEY, CHAIN_RESP_KEY,
    )

    node = MagicMock()
    node.identity.node_id = "self"
    node.transport.send_to_peer = AsyncMock(return_value=True)

    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_REQ_KEY: "req-1",
        CHAIN_RESP_KEY: "req-1",  # presence of resp key → ignore
        "chain_payload_b64": base64.b64encode(b"x").decode(),
    })

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            handle_chain_executor_request(node, msg),
        )
    finally:
        loop.close()
    assert result is False
    node.transport.send_to_peer.assert_not_called()


def test_handler_sends_response_for_valid_request():
    """A valid REQUEST → handler sends back a P2PMessage with
    CHAIN_RESP_KEY set + the same request_id.
    """
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_request,
        CHAIN_MSG_TYPE, CHAIN_REQ_KEY, CHAIN_RESP_KEY,
        CHAIN_PAYLOAD_KEY,
    )

    node = MagicMock()
    node.identity.node_id = "self"
    sent_messages = []
    async def _capture_send(peer_id, msg):
        sent_messages.append((peer_id, msg))
        return True
    node.transport.send_to_peer = _capture_send

    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_REQ_KEY: "req-abc123",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"request payload").decode(),
    }, sender_id="remote-peer-id")

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            handle_chain_executor_request(node, msg),
        )
    finally:
        loop.close()

    assert result is True
    assert len(sent_messages) == 1
    peer_id, resp = sent_messages[0]
    assert peer_id == "remote-peer-id"
    assert resp.payload["subtype"] == CHAIN_MSG_TYPE
    assert resp.payload[CHAIN_RESP_KEY] == "req-abc123"


def test_handler_response_includes_phase_2e_pending_indicator():
    """Phase 2E-1: the response payload signals that the stage
    handler hasn't landed yet — so the requester can distinguish
    "stage handler returned error" from "no stage handler exists".
    """
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_request,
        CHAIN_MSG_TYPE, CHAIN_REQ_KEY, CHAIN_PAYLOAD_KEY,
        CHAIN_ERROR_KEY,
    )

    node = MagicMock()
    node.identity.node_id = "self"
    sent_messages = []
    async def _capture_send(peer_id, msg):
        sent_messages.append((peer_id, msg))
        return True
    node.transport.send_to_peer = _capture_send

    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_REQ_KEY: "req-1",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"x").decode(),
    })

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            handle_chain_executor_request(node, msg),
        )
    finally:
        loop.close()

    payload = sent_messages[0][1].payload
    err = payload.get(CHAIN_ERROR_KEY) or payload.get("chain_error")
    assert err is not None, (
        "Phase 2E-1 response must carry an error indicator key"
    )
    # Error string mentions phase 2E or not yet implemented
    assert (
        "phase 2e" in err.lower()
        or "not yet implemented" in err.lower()
        or "stage handler" in err.lower()
    )


def test_handler_tolerates_malformed_base64():
    """Malformed payload still produces a response (error indicator)
    rather than silently dropping the request.
    """
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_request,
        CHAIN_MSG_TYPE, CHAIN_REQ_KEY, CHAIN_PAYLOAD_KEY,
    )

    node = MagicMock()
    node.identity.node_id = "self"
    sent_messages = []
    async def _capture_send(peer_id, msg):
        sent_messages.append((peer_id, msg))
        return True
    node.transport.send_to_peer = _capture_send

    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_REQ_KEY: "req-1",
        CHAIN_PAYLOAD_KEY: "!!! bad-base64 !!!",
    })

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            handle_chain_executor_request(node, msg),
        )
    finally:
        loop.close()
    # Returns True (we DID handle it, with an error response)
    assert result is True
    assert len(sent_messages) == 1
