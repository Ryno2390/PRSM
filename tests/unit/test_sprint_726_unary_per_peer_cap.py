"""Sprint 726 F59 — unary per-peer concurrent request cap (F56 sibling).

Sprint 723 closed F56 on the streaming server-side handler. Sprint
726 closes the identical fix-class on the UNARY handler
(`handle_chain_executor_request`).

Pre-726, one peer could open unlimited concurrent CHAIN_REQs
against this server. Each request held an in-flight
`executor.execute(payload_bytes)` coroutine + (for real
autoregressive runner) KV cache + intermediate tensors. One
malicious or buggy peer = trivial memory-exhaustion DoS,
especially on 2GB-RAM operator droplets.

Fix mirrors F56:
- `_resolve_per_peer_unary_concurrency()` reads
  `PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY` (default 8).
- `handle_chain_executor_request` increments
  `node._chain_executor_serving_unary_by_peer[sender_id]` on
  accepted REQ; rejects with CHAIN_ERROR_KEY response when
  count >= cap.
- Body extracted to `_handle_chain_executor_request_body` so
  wrapper places try/finally around every return path.
- Dict key popped when counter hits 0 (no unbounded growth).
"""
from __future__ import annotations

import inspect
import os

import pytest


def test_resolve_per_peer_unary_concurrency_default():
    """Unset env → 8."""
    from prsm.node.chain_executor_adapters import (
        _resolve_per_peer_unary_concurrency,
    )
    os.environ.pop("PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY", None)
    assert _resolve_per_peer_unary_concurrency() == 8


def test_resolve_per_peer_unary_concurrency_explicit():
    """Valid int env → that value."""
    from prsm.node.chain_executor_adapters import (
        _resolve_per_peer_unary_concurrency,
    )
    os.environ["PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY"] = "32"
    try:
        assert _resolve_per_peer_unary_concurrency() == 32
    finally:
        del os.environ["PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY"]


def test_resolve_per_peer_unary_concurrency_zero_unbounded():
    """0 → unbounded."""
    from prsm.node.chain_executor_adapters import (
        _resolve_per_peer_unary_concurrency,
    )
    os.environ["PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY"] = "0"
    try:
        assert _resolve_per_peer_unary_concurrency() == 0
    finally:
        del os.environ["PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY"]


def test_resolve_per_peer_unary_concurrency_typo_safe_default():
    """Non-int → safe-default 8."""
    from prsm.node.chain_executor_adapters import (
        _resolve_per_peer_unary_concurrency,
    )
    os.environ["PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY"] = "eight"
    try:
        assert _resolve_per_peer_unary_concurrency() == 8
    finally:
        del os.environ["PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY"]


def test_unary_helper_function_extracted_for_try_finally_wrap():
    """Pin: `_handle_chain_executor_request_body` exists so the
    wrapper's increment + try/finally + decrement structure can be
    applied around every return path."""
    from prsm.node import chain_executor_adapters as _mod
    assert hasattr(_mod, "_handle_chain_executor_request_body"), (
        "sprint 726 helper missing — refactor may have been reverted; "
        "counter leak risk reintroduced"
    )


def test_unary_wrapper_structure_increment_try_finally_decrement():
    """Pin via source: wrapper does increment → try-call-helper →
    finally-decrement. Order matters: increment before body so the
    cap is enforced; decrement in finally so error paths don't
    leak slots."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_request,
    )
    src = inspect.getsource(handle_chain_executor_request)
    inc_idx = src.find(
        "_serving_unary[sender_id] = _serving_unary.get(sender_id, 0) + 1"
    )
    try_idx = src.find("try:", inc_idx)
    finally_idx = src.find("finally:", try_idx)
    dec_idx = src.find("_serving_unary.pop", finally_idx)
    assert 0 < inc_idx < try_idx < finally_idx < dec_idx, (
        "wrapper structure broken — must be increment → try → "
        "finally → decrement"
    )


@pytest.mark.asyncio
async def test_unary_handler_rejects_over_cap():
    """Behavioral: when sender_id already has cap-many active
    requests, the next CHAIN_REQ is rejected with CHAIN_ERROR_KEY
    response naming the env var. Counter NOT incremented (rejection
    pre-increment)."""
    from unittest.mock import MagicMock
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_request, CHAIN_MSG_TYPE,
        CHAIN_REQ_KEY, CHAIN_PAYLOAD_KEY, CHAIN_RESP_KEY,
        CHAIN_ERROR_KEY,
    )
    import base64

    node = MagicMock()
    node.identity.node_id = "server" * 6
    sent = []
    async def _send(peer, msg):
        sent.append((peer, msg.payload))
        return True
    node.transport.send_to_peer = _send
    os.environ["PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY"] = "2"
    try:
        node._chain_executor_serving_unary_by_peer = {
            "attacker": 2,  # at cap
        }
        msg = MagicMock()
        msg.payload = {
            "subtype": CHAIN_MSG_TYPE,
            CHAIN_REQ_KEY: "req-726",
            CHAIN_PAYLOAD_KEY: base64.b64encode(b"req").decode(),
        }
        msg.sender_id = "attacker"
        result = await handle_chain_executor_request(node, msg)
        assert result is True
        # Counter not incremented (still 2)
        assert node._chain_executor_serving_unary_by_peer.get(
            "attacker"
        ) == 2
        # Error response sent
        assert len(sent) == 1
        _, payload = sent[0]
        assert payload.get(CHAIN_RESP_KEY) == "req-726"
        err = payload.get(CHAIN_ERROR_KEY, "")
        assert "concurrent unary-request limit exceeded" in err
        assert "PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY" in err
    finally:
        del os.environ["PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY"]
