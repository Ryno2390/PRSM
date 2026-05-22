"""Sprint 711 F40 — remote token-stream wire protocol.

Closes audit-doc §7.3 (cross-host streaming). The wire protocol
extends the existing chain-executor RPC (sprints 596-606 unary) with
a streaming variant:

  Requester → Server: ONE CHAIN_STREAM_REQ + payload
  Server → Requester: MULTIPLE CHAIN_STREAM_FRAME (one per chunk)
  Server → Requester: ONE terminal CHAIN_STREAM_END (optional error)

Sprint 711 ships:
  - Wire constants (CHAIN_STREAM_MSG_TYPE, CHAIN_STREAM_REQ_KEY, etc.)
  - Server-side `handle_chain_stream_request` (decodes req, iterates
    `LayerStageServer.handle_token_stream`, ships frames + end)
  - Client-side `handle_chain_stream_response` (routes frames to a
    per-stream asyncio.Queue created by the dispatcher)
  - `_remote_token_stream_dispatch` (issues req + collects frames)
  - Wired into `build_token_stream_send_message_adapter` remote path

Sprint 712 will live-attest end-to-end across NYC+SFO. Sprint 713
will add back-pressure + integration with the sprint-704 OOM gate
for streaming.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def test_stream_wire_protocol_constants_distinct_from_unary():
    """Wire constants must NOT collide with the existing unary
    CHAIN_MSG_TYPE — otherwise unary + streaming responses would
    cross-route into wrong handlers."""
    from prsm.node.chain_executor_adapters import (
        CHAIN_MSG_TYPE,
        CHAIN_STREAM_MSG_TYPE,
        CHAIN_REQ_KEY,
        CHAIN_STREAM_REQ_KEY,
        CHAIN_RESP_KEY,
        CHAIN_STREAM_FRAME_KEY,
        CHAIN_STREAM_END_KEY,
    )
    # Subtypes are distinct strings — server-side router uses
    # subtype to dispatch.
    assert CHAIN_MSG_TYPE != CHAIN_STREAM_MSG_TYPE
    # Stream-side request/frame/end keys distinct from unary keys
    assert CHAIN_STREAM_REQ_KEY != CHAIN_REQ_KEY
    assert CHAIN_STREAM_REQ_KEY != CHAIN_RESP_KEY
    assert CHAIN_STREAM_FRAME_KEY != CHAIN_STREAM_REQ_KEY
    assert CHAIN_STREAM_END_KEY != CHAIN_STREAM_REQ_KEY
    assert CHAIN_STREAM_END_KEY != CHAIN_STREAM_FRAME_KEY


@pytest.mark.asyncio
async def test_handle_chain_stream_request_ignores_non_stream_msg():
    """Server handler returns False for non-stream messages so
    other dispatchers can claim them."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_request, CHAIN_MSG_TYPE,
    )
    node = MagicMock()
    msg = MagicMock()
    msg.payload = {"subtype": CHAIN_MSG_TYPE}  # unary, not stream
    msg.sender_id = "peerA"
    handled = await handle_chain_stream_request(node, msg)
    assert handled is False


@pytest.mark.asyncio
async def test_handle_chain_stream_request_ignores_frame_msg():
    """Server handler returns False for FRAME / END messages
    (those are client-side; routed by handle_chain_stream_response)."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_request,
        CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY,
    )
    node = MagicMock()
    msg = MagicMock()
    msg.payload = {
        "subtype": CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY: "stream-x",
    }
    msg.sender_id = "peerA"
    handled = await handle_chain_stream_request(node, msg)
    assert handled is False


def test_handle_chain_stream_response_ignores_non_stream_msg():
    """Response handler returns False for non-stream messages."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_response, CHAIN_MSG_TYPE,
    )
    node = MagicMock()
    msg = MagicMock()
    msg.payload = {"subtype": CHAIN_MSG_TYPE}
    assert handle_chain_stream_response(node, msg) is False


def test_handle_chain_stream_response_ignores_request_msg():
    """Response handler returns False for REQUEST messages (those
    are server-side; routed by handle_chain_stream_request)."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_response,
        CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_REQ_KEY,
    )
    node = MagicMock()
    msg = MagicMock()
    msg.payload = {
        "subtype": CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_REQ_KEY: "stream-x",
    }
    assert handle_chain_stream_response(node, msg) is False


def test_handle_chain_stream_response_ignores_unknown_stream():
    """Frames for streams we never originated (e.g., late arrivals
    after the requester gave up) are dropped silently → False."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_response,
        CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY,
        CHAIN_PAYLOAD_KEY,
    )
    import base64
    node = MagicMock()
    node._chain_executor_pending_streams = {}
    msg = MagicMock()
    msg.payload = {
        "subtype": CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY: "unknown-stream",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"frame1").decode(),
    }
    assert handle_chain_stream_response(node, msg) is False


def test_send_message_adapter_remote_path_uses_dispatch():
    """Pin: build_token_stream_send_message_adapter's remote branch
    must delegate to _remote_token_stream_dispatch (sprint 711)
    rather than raise the sprint-691 "not yet wired" RuntimeError."""
    import inspect
    from prsm.node.chain_executor_adapters import (
        build_token_stream_send_message_adapter,
    )
    src = inspect.getsource(build_token_stream_send_message_adapter)
    assert "_remote_token_stream_dispatch" in src, (
        "remote-streaming wiring must delegate to "
        "_remote_token_stream_dispatch (sprint 711 F40)"
    )
    # Ensure the sprint-691 "not yet wired" placeholder is gone
    assert "not yet wired" not in src, (
        "sprint-691 placeholder still present — sprint 711 should "
        "have replaced it with the real dispatch"
    )


def test_node_py_registers_stream_handlers():
    """Pin: PRSMNode.start must register both
    handle_chain_stream_request + handle_chain_stream_response on
    MSG_DIRECT so incoming stream messages are routed."""
    import inspect
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "handle_chain_stream_request" in src
    assert "handle_chain_stream_response" in src
