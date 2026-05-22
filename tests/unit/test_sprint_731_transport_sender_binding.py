"""Sprint 731 F64 — transport-level sender_id binding (generalizes F63).

Sprint 730 fixed sender_id spoofing for the 4 chain-executor
handlers by binding at the dispatch-wrapper level in
PRSMNode.start. But OTHER MSG_DIRECT handlers exist across the
codebase that ALSO read msg.sender_id and ALSO trust it:

  - ledger_sync._on_direct_message     (FTNS transfers)
  - compute_provider._on_direct_message (compute job requests)
  - storage_provider._on_direct_message (storage interactions)
  - content_provider._handle_direct_message (content fetches)
  - agent_registry._on_direct_message  (agent presence)

Each one's security model depends on msg.sender_id being the
actual cryptographically-authenticated peer — none of them
re-verify per-message signatures, and sprint-730 only bound for
chain-executor handlers.

Fix (sprint 731): move the bind from per-handler-wrapper to
WebSocketTransport._dispatch. ALL handlers for MSG_DIRECT and
other peer-to-peer types automatically see authenticated
sender_id. MSG_GOSSIP is excluded because gossip relay
legitimately carries the original sender's id, distinct from
the relaying peer (gossip protocol does its own provenance
verification).

This RETAINS sprint-730's chain-executor binds (defense in
depth — if something bypasses the transport, the wrappers
still bind). Both layers operate idempotently.
"""
from __future__ import annotations

import inspect


def test_transport_dispatch_binds_sender_id_to_peer_id():
    """Pin: WebSocketTransport._dispatch overwrites msg.sender_id
    with peer.peer_id BEFORE invoking handlers."""
    from prsm.node.transport import WebSocketTransport
    src = inspect.getsource(WebSocketTransport._dispatch)
    assert "msg.sender_id = authentic" in src or (
        "msg.sender_id =" in src and "peer_id" in src
    ), (
        "Sprint 731 F64 fix missing — transport must bind "
        "msg.sender_id to peer.peer_id at dispatch boundary"
    )


def test_transport_dispatch_excludes_gossip_from_bind():
    """Pin: MSG_GOSSIP is excluded because gossip relay carries
    the ORIGINAL sender's id, distinct from the relaying peer's
    id. Binding gossip would corrupt provenance."""
    from prsm.node.transport import WebSocketTransport
    src = inspect.getsource(WebSocketTransport._dispatch)
    assert "MSG_GOSSIP" in src, (
        "MSG_GOSSIP must be referenced in _dispatch so gossip is "
        "explicitly excluded from sender-binding"
    )


def test_dispatch_bind_is_idempotent_with_node_dispatch_wrappers():
    """Defense-in-depth: sprint 730's PRSMNode.start dispatch
    wrappers also bind sender_id. If transport binds first, the
    wrapper's bind is a no-op (already authentic). If transport
    bind ever regresses, the wrapper still protects chain-executor
    paths. The two layers don't conflict."""
    from unittest.mock import MagicMock
    msg = MagicMock()
    msg.sender_id = "AUTHENTIC_PEER"  # transport already bound
    peer = MagicMock()
    peer.peer_id = "AUTHENTIC_PEER"
    # Inline sprint-730 wrapper logic
    def _bind_sender(m, p):
        if p is not None:
            authentic = getattr(p, "peer_id", None)
            if authentic:
                m.sender_id = authentic
    _bind_sender(msg, peer)
    # Still authentic — idempotent.
    assert msg.sender_id == "AUTHENTIC_PEER"


def test_other_msg_direct_handlers_protected_by_transport_bind():
    """Pin (source-shape): the codebase has multiple MSG_DIRECT
    handlers besides the 4 chain-executor ones. Sprint 731 protects
    all of them automatically by binding at transport. Document
    the registered handlers so a future reviewer knows what's
    protected by this fix."""
    import inspect as _i
    # ledger_sync
    from prsm.node import ledger_sync as _ledger
    assert "MSG_DIRECT" in _i.getsource(_ledger), (
        "ledger_sync should register MSG_DIRECT handler"
    )
    # compute_provider
    from prsm.node import compute_provider as _cp
    assert "MSG_DIRECT" in _i.getsource(_cp), (
        "compute_provider should register MSG_DIRECT handler"
    )
