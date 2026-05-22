"""Sprint 723 F56 — per-peer concurrent stream cap on server side.

Pre-723, one peer could open unlimited CHAIN_STREAM_REQs against
this server. Each request started a new generator + bound an
asyncio.Queue + (for real autoregressive runner) allocated KV
cache. Memory grows unbounded → trivial DoS, especially on
2GB-RAM operator droplets.

Fix:
- `_resolve_per_peer_stream_concurrency()` env-tunable (default 8).
- `handle_chain_stream_request` increments a counter at
  `node._chain_executor_serving_streams_by_peer[sender_id]` when
  accepting; rejects with terminal STREAM_END when count >= cap.
- Body extracted to `_handle_stream_request_body` so the wrapper
  can place the increment + try/finally + decrement around EVERY
  return-True path — counter never leaks.
"""
from __future__ import annotations

import os

import pytest


def test_resolve_per_peer_stream_concurrency_default():
    """Unset env → 8 (default)."""
    from prsm.node.chain_executor_adapters import (
        _resolve_per_peer_stream_concurrency,
    )
    os.environ.pop("PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY", None)
    assert _resolve_per_peer_stream_concurrency() == 8


def test_resolve_per_peer_stream_concurrency_explicit_override():
    """Valid int env → that value."""
    from prsm.node.chain_executor_adapters import (
        _resolve_per_peer_stream_concurrency,
    )
    os.environ["PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY"] = "16"
    try:
        assert _resolve_per_peer_stream_concurrency() == 16
    finally:
        del os.environ["PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY"]


def test_resolve_per_peer_stream_concurrency_zero_unbounded():
    """0 → unbounded (pre-723 behavior)."""
    from prsm.node.chain_executor_adapters import (
        _resolve_per_peer_stream_concurrency,
    )
    os.environ["PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY"] = "0"
    try:
        assert _resolve_per_peer_stream_concurrency() == 0
    finally:
        del os.environ["PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY"]


def test_resolve_per_peer_stream_concurrency_typo_safely_defaults():
    """Non-int → safe-default 8."""
    from prsm.node.chain_executor_adapters import (
        _resolve_per_peer_stream_concurrency,
    )
    os.environ["PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY"] = "eight"
    try:
        assert _resolve_per_peer_stream_concurrency() == 8
    finally:
        del os.environ["PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY"]


def test_helper_function_extracted_for_try_finally_wrap():
    """Pin: sprint 723 refactor extracted the body of
    `handle_chain_stream_request` into `_handle_stream_request_body`
    so the per-peer-cap counter wraps every return-True path. This
    is the structural invariant that prevents the counter from
    leaking; pin it via module-level lookup."""
    from prsm.node import chain_executor_adapters as _mod
    assert hasattr(_mod, "_handle_stream_request_body"), (
        "sprint 723 helper missing — refactor may have been reverted; "
        "counter leak risk reintroduced"
    )


def test_wrapper_increments_before_body_and_decrements_in_finally():
    """Pin via source: the wrapper does increment → try-call-helper
    → finally-decrement. Order matters because:
      - If increment happened AFTER body, the cap couldn't see the
        new stream and we'd over-accept.
      - If decrement weren't in finally, every error path would
        leak a slot.
    """
    import inspect
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_request,
    )
    src = inspect.getsource(handle_chain_stream_request)
    inc_idx = src.find(
        "_serving[sender_id] = _serving.get(sender_id, 0) + 1"
    )
    try_idx = src.find("try:", inc_idx)
    finally_idx = src.find("finally:", try_idx)
    dec_idx = src.find("_serving.pop", finally_idx)
    assert 0 < inc_idx < try_idx < finally_idx < dec_idx, (
        "wrapper structure broken — must be increment → try → "
        "finally → decrement"
    )


@pytest.mark.asyncio
async def test_handle_request_rejects_over_cap():
    """Behavioral: when sender_id already has cap-many active
    streams, the next REQ is rejected with terminal STREAM_END
    carrying the env var name in the error message. The counter
    is NOT incremented (the rejection happens before the
    increment line). Verified by inspecting send_to_peer calls."""
    from unittest.mock import AsyncMock, MagicMock
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_request, CHAIN_STREAM_MSG_TYPE,
        CHAIN_STREAM_REQ_KEY, CHAIN_PAYLOAD_KEY,
        CHAIN_STREAM_END_KEY, CHAIN_ERROR_KEY,
    )
    import base64

    node = MagicMock()
    node.identity.node_id = "server" * 6
    sent = []
    async def _send(peer, msg):
        sent.append((peer, msg.payload))
        return True
    node.transport.send_to_peer = _send
    # Pre-fill cap (default 8) so the new REQ is rejected.
    os.environ["PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY"] = "2"
    try:
        node._chain_executor_serving_streams_by_peer = {
            "attacker-peer": 2,  # already at cap
        }
        msg = MagicMock()
        msg.payload = {
            "subtype": CHAIN_STREAM_MSG_TYPE,
            CHAIN_STREAM_REQ_KEY: "stream-723",
            CHAIN_PAYLOAD_KEY: base64.b64encode(b"req").decode(),
        }
        msg.sender_id = "attacker-peer"
        result = await handle_chain_stream_request(node, msg)
        assert result is True  # handled (with rejection)
        # Cap counter should NOT have been incremented — rejection
        # happened before that.
        assert node._chain_executor_serving_streams_by_peer.get(
            "attacker-peer"
        ) == 2
        # Terminal STREAM_END with cap error must have been sent.
        assert len(sent) == 1
        _, payload = sent[0]
        assert payload.get(CHAIN_STREAM_END_KEY) == "stream-723"
        err = payload.get(CHAIN_ERROR_KEY, "")
        assert "concurrent stream limit exceeded" in err
        assert "PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY" in err
    finally:
        del os.environ["PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY"]
