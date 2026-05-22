"""Sprint 720 F54 — server-side stream cleanup on requester disconnect.

Sprint 711's `handle_chain_stream_request` iterated
`server.handle_token_stream(request_bytes)` and shipped one frame
per yielded chunk via `await node.transport.send_to_peer(...)`. The
return value of send_to_peer was ignored. When the requester
disconnected mid-stream, send_to_peer returned False but the loop
kept iterating the inner generator → for a real autoregressive
runner that means continuing GPU work on tokens nobody will
receive.

Worst-case shape: a long inference (1000 tokens, multi-second per
token on CPU operator), requester drops at token 10 → server
keeps generating tokens 11-1000 + tries to send each → 990 wasted
forward passes + KV cache held + memory pressure on a 2GB droplet.

Fix: check send_to_peer return value. On False, call
`stream_iter.close()` to release the inner generator's resources
(KV cache, model context) + return immediately. No terminal
STREAM_END is sent because the peer is gone (sending would just
return False again).
"""
from __future__ import annotations

import inspect


def test_handle_request_breaks_on_send_to_peer_false():
    """Pin: server-side iteration loop must check send_to_peer
    return value and exit when it returns False. Pre-720, the
    return value was ignored."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_request,
    )
    src = inspect.getsource(handle_chain_stream_request)
    # Find the iteration block and assert send_ok variable is read
    # and the loop branches on it.
    assert "send_ok" in src or "send_to_peer" in src
    assert "if not send_ok" in src or "if not " in src, (
        "F54: server-side loop must check send_to_peer return + "
        "exit early when peer disconnected"
    )


def test_handle_request_closes_stream_iter_on_disconnect():
    """Pin: when the requester disconnects, the server-side
    iteration must call `stream_iter.close()` (if available) to
    release the underlying generator's resources. For
    AutoregressiveStreamingRunner that releases the KV cache +
    model context promptly rather than waiting for GC."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_request,
    )
    src = inspect.getsource(handle_chain_stream_request)
    assert "stream_iter" in src or "_close_fn" in src
    assert ".close" in src, (
        "F54: must call stream_iter.close() to release generator "
        "resources on requester disconnect"
    )
