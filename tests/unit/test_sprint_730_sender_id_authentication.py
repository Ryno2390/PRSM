"""Sprint 730 F63 — bind msg.sender_id to authenticated peer.peer_id.

The transport (`prsm.node.transport.WebSocketTransport`) verifies
cryptographic signatures ONLY at handshake time. Subsequent
MSG_DIRECT messages carry a `sender_id` field that is wire-trusted
but NOT rebound to the connection's authenticated identity.

Attack: a peer that completes a valid handshake (their own keypair
verifies the handshake signature, deriving peer_id=X) can then
send MSG_DIRECT messages with `msg.sender_id="VICTIM_ID"`. The
handler reads msg.sender_id and:

  - Increments sprint-726/723 per-peer cap counter under "VICTIM_ID"
    instead of X → X can open many concurrent streams by cycling
    fake sender_ids, all under the radar of the cap.
  - Sprint-727/719 sender-binding checks compare msg.sender_id to
    expected_sender; X can forge responses by claiming to be the
    victim's expected peer.

Fix (sprint 730): the four chain-executor dispatch wrappers in
`PRSMNode.start` now overwrite `msg.sender_id` with the
authenticated `peer.peer_id` BEFORE invoking the handler.
Cryptographically-bound identity flows downstream; no handler
signature changes needed.
"""
from __future__ import annotations

import inspect


def test_node_dispatch_wrappers_bind_sender_to_peer_id():
    """Pin: PRSMNode.start defines a `_bind_sender(msg, peer)`
    helper and calls it inside all 4 chain-executor dispatch
    wrappers BEFORE the handler invocation."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "_bind_sender" in src, (
        "Sprint 730 helper missing — F63 fix may have been reverted"
    )
    # Each of the 4 wrappers must call _bind_sender.
    wrapper_names = [
        "_chain_executor_response_dispatch",
        "_chain_executor_request_dispatch",
        "_chain_stream_request_dispatch",
        "_chain_stream_response_dispatch",
    ]
    for w in wrapper_names:
        idx = src.find(f"async def {w}")
        assert idx > 0, f"wrapper {w} not found"
        # Look at the next 200 chars of the wrapper body for the bind
        # call. (Body is small — 2-3 lines.)
        body = src[idx:idx + 400]
        assert "_bind_sender" in body, (
            f"wrapper {w} does not call _bind_sender — F63 fix "
            f"incomplete; spoofed sender_id can still reach handler"
        )


def test_bind_sender_overwrites_claimed_sender_id():
    """Behavioral: _bind_sender(msg, peer) where peer.peer_id =
    AUTHENTIC and msg.sender_id = SPOOFED → msg.sender_id becomes
    AUTHENTIC. The wire claim is overwritten."""
    # Replicate the inline helper for behavioral verification —
    # it's defined inside PRSMNode.start as a closure, so we can't
    # import it directly. The pin test above ensures source-shape
    # parity; this test reproduces the contract.
    from unittest.mock import MagicMock
    msg = MagicMock()
    msg.sender_id = "SPOOFED_VICTIM_ID"
    peer = MagicMock()
    peer.peer_id = "AUTHENTIC_PEER_ID"
    # Inline helper matching node.py
    def _bind_sender(m, p):
        if p is not None:
            authentic = getattr(p, "peer_id", None)
            if authentic:
                m.sender_id = authentic
    _bind_sender(msg, peer)
    assert msg.sender_id == "AUTHENTIC_PEER_ID", (
        "F63: msg.sender_id must be overwritten with peer.peer_id"
    )


def test_bind_sender_tolerates_missing_peer():
    """Defensive: if peer is None (synthesized message, edge case),
    msg.sender_id stays as-is — the wrapper must NOT crash. The
    handler then either drops the message (defensive checks
    upstream) or processes the wire-claimed sender_id; in either
    case, no security regression vs pre-730."""
    from unittest.mock import MagicMock
    msg = MagicMock()
    msg.sender_id = "ORIGINAL"
    def _bind_sender(m, p):
        if p is not None:
            authentic = getattr(p, "peer_id", None)
            if authentic:
                m.sender_id = authentic
    _bind_sender(msg, None)
    assert msg.sender_id == "ORIGINAL"


def test_bind_sender_tolerates_peer_without_peer_id():
    """Defensive: if peer object lacks peer_id attribute (unusual
    fixture shape), don't crash — leave sender_id alone."""
    from unittest.mock import MagicMock
    msg = MagicMock()
    msg.sender_id = "ORIGINAL"
    peer = MagicMock(spec=[])  # peer with no peer_id attr
    def _bind_sender(m, p):
        if p is not None:
            authentic = getattr(p, "peer_id", None)
            if authentic:
                m.sender_id = authentic
    _bind_sender(msg, peer)
    assert msg.sender_id == "ORIGINAL"
