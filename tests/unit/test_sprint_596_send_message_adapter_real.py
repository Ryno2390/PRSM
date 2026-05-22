"""Sprint 596 (Phase 2D step 2) — real SendMessage adapter behavior.

Replaces Phase 2A scaffolding (sprint 592) with the real impl that
uses node._loop (sprint 595) + run_async_on_loop (sprint 594) +
pending dict (sprint 595) to bridge sync SendMessage over async
transport.

Tests:
  - Adapter raises _Phase2AdapterNotReady when node._loop is None.
  - Adapter constructs proper P2PMessage shape (subtype + chain_req_id
    + base64 payload).
  - Adapter installs Future in pending dict keyed by sha256(request_bytes).
  - Adapter resolves to the response bytes when the future is set.
  - Adapter cleans up the pending dict entry on both success + error.
  - Adapter raises RuntimeError when transport.send_to_peer returns False.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import threading
from unittest.mock import AsyncMock, MagicMock


def test_adapter_raises_phase2_not_ready_when_loop_is_none():
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter, _Phase2AdapterNotReady,
    )
    import pytest
    node = MagicMock()
    node._loop = None
    adapter = build_send_message_adapter(node)
    with pytest.raises(_Phase2AdapterNotReady):
        adapter("peer-1", b"req")


def test_adapter_resolves_response_when_future_set():
    """End-to-end happy path: send request → manually resolve the
    future from another thread → adapter returns response bytes.
    """
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
    )

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run_loop():
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    t = threading.Thread(target=_run_loop, daemon=True)
    t.start()
    started.wait(timeout=2)

    node = MagicMock()
    node._loop = loop
    node._chain_executor_pending = {}
    node.identity.node_id = "self-id"
    node.transport.send_to_peer = AsyncMock(return_value=True)

    adapter = build_send_message_adapter(node, timeout=3.0)
    request_bytes = b"sample-request-payload"

    # Resolve the pending Future from a watchdog thread once it
    # appears in the dict. Sprint 724 made request_id non-
    # derivable from request_bytes (8 bytes os.urandom mixed in)
    # + sprint 727 changed pending[] to store (future, stage_addr)
    # tuples — so we iterate keys + unpack tuple shape rather than
    # looking up a pre-computed id.
    def _watch_and_resolve():
        import time
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            for _key, entry in list(
                node._chain_executor_pending.items()
            ):
                future = (
                    entry[0]
                    if isinstance(entry, tuple) else entry
                )
                if future is not None and not future.done():
                    loop.call_soon_threadsafe(
                        future.set_result, b"response-bytes",
                    )
                    return
            time.sleep(0.01)

    threading.Thread(target=_watch_and_resolve, daemon=True).start()

    try:
        result = adapter("remote-peer", request_bytes)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)

    assert result == b"response-bytes"
    # send_to_peer was called with the remote peer + a P2PMessage
    args, kwargs = node.transport.send_to_peer.await_args
    assert args[0] == "remote-peer"
    msg = args[1]
    assert msg.payload["subtype"] == "chain_executor_rpc"
    # Sprint 724 made request_id non-derivable from request_bytes
    # alone (mixed in os.urandom). Pin the SHAPE rather than the
    # exact bytes: must be a 64-char hex string.
    chain_req_id = msg.payload["chain_req_id"]
    assert isinstance(chain_req_id, str) and len(chain_req_id) == 64
    assert all(c in "0123456789abcdef" for c in chain_req_id)
    assert msg.payload["chain_payload_b64"] == base64.b64encode(
        request_bytes,
    ).decode("ascii")


def test_adapter_raises_when_send_to_peer_returns_false():
    """transport.send_to_peer returning False → RuntimeError surfaces."""
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
    )

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run_loop():
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    threading.Thread(target=_run_loop, daemon=True).start()
    started.wait(timeout=2)

    node = MagicMock()
    node._loop = loop
    node._chain_executor_pending = {}
    node.identity.node_id = "self-id"
    node.transport.send_to_peer = AsyncMock(return_value=False)

    adapter = build_send_message_adapter(node, timeout=1.0)
    raised = None
    try:
        try:
            adapter("unreachable-peer", b"req")
        except Exception as exc:  # noqa: BLE001
            raised = exc
    finally:
        loop.call_soon_threadsafe(loop.stop)

    assert raised is not None
    assert isinstance(raised, RuntimeError)
    assert "unreachable-peer" in str(raised) or "send_to_peer" in str(raised)


def test_adapter_cleans_pending_dict_after_completion():
    """After successful return, the request_id must NOT linger in
    node._chain_executor_pending (memory leak protection).
    """
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
    )

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run_loop():
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    threading.Thread(target=_run_loop, daemon=True).start()
    started.wait(timeout=2)

    node = MagicMock()
    node._loop = loop
    node._chain_executor_pending = {}
    node.identity.node_id = "self-id"
    node.transport.send_to_peer = AsyncMock(return_value=True)

    adapter = build_send_message_adapter(node, timeout=3.0)
    request_bytes = b"clean-req"

    def _watch_and_resolve():
        # Sprint 724 + 727: iterate keys + unpack tuple shape (see
        # the other watcher above for context).
        import time
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            for _key, entry in list(
                node._chain_executor_pending.items()
            ):
                future = (
                    entry[0]
                    if isinstance(entry, tuple) else entry
                )
                if future is not None and not future.done():
                    loop.call_soon_threadsafe(
                        future.set_result, b"x",
                    )
                    return
            time.sleep(0.01)

    threading.Thread(target=_watch_and_resolve, daemon=True).start()
    try:
        adapter("peer", request_bytes)
    finally:
        loop.call_soon_threadsafe(loop.stop)

    # Sprint 724: request_id is non-derivable from request_bytes
    # alone; the cleanup invariant is "dict is empty post-completion"
    # not "specific key is gone".
    assert not node._chain_executor_pending


def test_wire_protocol_constants_exposed():
    """Sprint 597's response handler needs these constants."""
    from prsm.node import chain_executor_adapters as m
    assert m.CHAIN_MSG_TYPE == "chain_executor_rpc"
    assert m.CHAIN_REQ_KEY == "chain_req_id"
    assert m.CHAIN_RESP_KEY == "chain_resp_id"
    assert m.CHAIN_PAYLOAD_KEY == "chain_payload_b64"
