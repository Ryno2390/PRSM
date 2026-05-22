"""Sprint 719 F53 — stream hijack protection (sender binding).

Sprint 711's wire protocol authenticated messages (msg.sign in
WebSocketTransport) but did NOT bind sender identity to stream_id
on the receiver side. A peer that learned the stream_id (by
network observation, sprint-718-pre entropy guess, or protocol
quirk) could send forged CHAIN_STREAM_FRAME / CHAIN_STREAM_END
messages and the victim's `handle_chain_stream_response` would
route them into the per-stream queue alongside legitimate frames.

P2P networks are open by default: any peer can connect to any
other peer, so the attack surface is real. A malicious aggregator
or a compromised peer could inject zero-frames, terminate streams
early, or inject false output bytes.

Fix: `pending[stream_id]` now stores `(queue, expected_sender)`.
The response handler verifies `msg.sender_id == expected_sender`
BEFORE routing. Frames from any other sender are silently dropped
(return False so other dispatchers in the same daemon can claim
them if needed; never poison the victim's queue).
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock


def test_dispatcher_registers_pending_as_queue_and_sender_tuple():
    """Pin: pre-719 `pending[stream_id] = queue` (bare queue);
    post-719 `pending[stream_id] = (queue, stage_address)`. Source-
    level invariant."""
    from prsm.node.chain_executor_adapters import (
        _remote_token_stream_dispatch,
    )
    src = inspect.getsource(_remote_token_stream_dispatch)
    assert "pending[stream_id] = (queue, stage_address)" in src, (
        "F53 fix requires pending[] to store (queue, expected_sender) "
        "tuple so handle_chain_stream_response can verify sender"
    )


def test_response_handler_drops_frame_with_wrong_sender():
    """Behavioral check: response handler returns False (drops the
    frame) when msg.sender_id does NOT match the dispatcher's
    expected_sender for that stream_id."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_response,
        CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY,
        CHAIN_PAYLOAD_KEY,
    )
    import base64

    node = MagicMock()
    # Sprint 719 tuple shape: (queue, expected_sender)
    fake_queue = MagicMock()
    expected_peer = "legit-peer-id-aaaaaaaaaaaaaaaaa"
    node._chain_executor_pending_streams = {
        "stream-xyz": (fake_queue, expected_peer),
    }
    node._loop = MagicMock()

    msg = MagicMock()
    msg.payload = {
        "subtype": CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY: "stream-xyz",
        CHAIN_PAYLOAD_KEY: base64.b64encode(
            b"forged-frame-from-attacker",
        ).decode(),
    }
    msg.sender_id = "attacker-peer-id-zzzzzzzzzzzzzzzz"
    handled = handle_chain_stream_response(node, msg)
    # Frame must NOT have been routed to the queue. Handler returns
    # False so other dispatch paths could try to claim — but the
    # frame must NEVER reach `queue.put`.
    assert handled is False, (
        "F53: frame from wrong sender must be dropped, not routed"
    )
    # Verify no scheduler was called on the loop (no run_coroutine_*,
    # no call_soon).
    node._loop.call_soon_threadsafe.assert_not_called()
    # Also verify queue itself was never put_nowait'd directly
    fake_queue.put_nowait.assert_not_called()


def test_response_handler_routes_frame_with_matching_sender():
    """Legitimate path: msg.sender_id matches expected_sender →
    frame is routed (handler returns True + scheduler invoked)."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_response,
        CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY,
        CHAIN_PAYLOAD_KEY,
    )
    import base64

    node = MagicMock()
    legit_peer = "legit-peer-id-aaaaaaaaaaaaaaaaa"
    fake_queue = MagicMock()
    node._chain_executor_pending_streams = {
        "stream-xyz": (fake_queue, legit_peer),
    }
    node._loop = MagicMock()

    msg = MagicMock()
    msg.payload = {
        "subtype": CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY: "stream-xyz",
        CHAIN_PAYLOAD_KEY: base64.b64encode(
            b"legitimate-frame",
        ).decode(),
    }
    msg.sender_id = legit_peer  # matches expected
    handled = handle_chain_stream_response(node, msg)
    assert handled is True, (
        "matching sender path must return True (routed)"
    )


def test_legacy_bare_queue_shape_still_works_defensively():
    """Defensive: if pending[] still has the legacy bare-queue
    shape (no tuple) — e.g., from a test fixture or pre-719
    in-flight stream that survived an upgrade — the response
    handler should NOT crash. It falls back to non-binding mode
    (legacy behavior). Should never happen in production but the
    fallback is what prevents an upgrade-time outage."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_response,
        CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY,
        CHAIN_PAYLOAD_KEY,
    )
    import base64

    node = MagicMock()
    fake_queue = MagicMock()
    # Legacy: bare queue, no expected_sender
    node._chain_executor_pending_streams = {
        "stream-xyz": fake_queue,
    }
    node._loop = MagicMock()

    msg = MagicMock()
    msg.payload = {
        "subtype": CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY: "stream-xyz",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"frame").decode(),
    }
    msg.sender_id = "any-peer"
    # Should NOT crash + should still return True (legacy routes).
    handled = handle_chain_stream_response(node, msg)
    assert handled is True


def test_empty_expected_sender_skips_binding_check():
    """If expected_sender is empty/None (defensive), the binding
    check is skipped — frame routes. Avoids breaking edge cases
    where dispatch didn't have a peer id at registration time
    (shouldn't happen in production but pin the fallback)."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_response,
        CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY,
        CHAIN_PAYLOAD_KEY,
    )
    import base64

    node = MagicMock()
    fake_queue = MagicMock()
    node._chain_executor_pending_streams = {
        "stream-xyz": (fake_queue, ""),  # empty expected_sender
    }
    node._loop = MagicMock()
    msg = MagicMock()
    msg.payload = {
        "subtype": CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_FRAME_KEY: "stream-xyz",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"f").decode(),
    }
    msg.sender_id = "any-peer"
    handled = handle_chain_stream_response(node, msg)
    assert handled is True
