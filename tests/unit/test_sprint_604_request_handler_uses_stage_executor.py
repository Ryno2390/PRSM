"""Sprint 604 (Phase 2E-4) — handle_chain_executor_request uses StageExecutor.

Replaces sprint 601's hardcoded "not yet implemented" error response
with delegation to a StageExecutor (sprint 602 Protocol). Selection:

  1. node._chain_stage_executor if set (test injection / future
     production wiring)
  2. else _build_stage_executor_from_env() reading
     PRSM_PARALLAX_STAGE_EXECUTOR_KIND:
       - stub (default)  → raises (sprint 602)
       - echo            → returns input unchanged (sprint 603)

Tests:
  - Env=echo → response payload echoes request bytes
  - Env unset (stub) → response has CHAIN_ERROR_KEY w/ phase-2e-3 hint
  - node._chain_stage_executor injection overrides env
  - StageExecutionError → response has CHAIN_ERROR_KEY
  - Generic exception in executor → response has CHAIN_ERROR_KEY
"""
from __future__ import annotations

import asyncio
import base64
import os
from unittest.mock import MagicMock, patch


def _make_msg(payload, sender_id="remote-peer"):
    m = MagicMock()
    m.payload = payload
    m.sender_id = sender_id
    return m


def _make_node():
    node = MagicMock()
    node.identity.node_id = "self"
    sent = []
    async def _capture(peer_id, msg):
        sent.append((peer_id, msg))
        return True
    node.transport.send_to_peer = _capture
    node._chain_stage_executor = None  # default: use env
    node._sent = sent
    return node


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_env_echo_returns_input_bytes_in_response():
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_request,
        CHAIN_MSG_TYPE, CHAIN_REQ_KEY, CHAIN_RESP_KEY,
        CHAIN_PAYLOAD_KEY, CHAIN_ERROR_KEY,
    )

    node = _make_node()
    req_bytes = b"forward-pass input"
    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_REQ_KEY: "req-1",
        CHAIN_PAYLOAD_KEY: base64.b64encode(req_bytes).decode(),
    })

    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_STAGE_EXECUTOR_KIND": "echo"},
        clear=False,
    ):
        result = _run(handle_chain_executor_request(node, msg))
    assert result is True

    sent = node._sent
    assert len(sent) == 1
    resp_payload = sent[0][1].payload
    assert resp_payload[CHAIN_RESP_KEY] == "req-1"
    assert CHAIN_ERROR_KEY not in resp_payload
    decoded = base64.b64decode(resp_payload[CHAIN_PAYLOAD_KEY])
    assert decoded == req_bytes


def test_env_default_stub_sends_error_response():
    """With env unset, _build_stage_executor_from_env returns the
    stub which raises — response has CHAIN_ERROR_KEY."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_request,
        CHAIN_MSG_TYPE, CHAIN_REQ_KEY,
        CHAIN_PAYLOAD_KEY, CHAIN_ERROR_KEY,
    )

    node = _make_node()
    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_REQ_KEY: "req-1",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"x").decode(),
    })

    os.environ.pop("PRSM_PARALLAX_STAGE_EXECUTOR_KIND", None)
    result = _run(handle_chain_executor_request(node, msg))
    assert result is True

    resp_payload = node._sent[0][1].payload
    err = resp_payload.get(CHAIN_ERROR_KEY) or ""
    assert "stage" in err.lower() or "phase 2e" in err.lower()


def test_node_attached_executor_overrides_env():
    """Test injection via node._chain_stage_executor takes precedence
    over env var — useful for live tests with custom executors.
    """
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_request,
        CHAIN_MSG_TYPE, CHAIN_REQ_KEY,
        CHAIN_PAYLOAD_KEY, CHAIN_ERROR_KEY,
    )

    class _ConstExecutor:
        async def execute(self, request_bytes: bytes) -> bytes:
            return b"CONSTANT-RESPONSE"

    node = _make_node()
    node._chain_stage_executor = _ConstExecutor()
    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_REQ_KEY: "req-1",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"ignored").decode(),
    })

    # Env=echo but injected executor should win
    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_STAGE_EXECUTOR_KIND": "echo"},
        clear=False,
    ):
        _run(handle_chain_executor_request(node, msg))

    resp_payload = node._sent[0][1].payload
    assert CHAIN_ERROR_KEY not in resp_payload
    assert base64.b64decode(
        resp_payload[CHAIN_PAYLOAD_KEY],
    ) == b"CONSTANT-RESPONSE"


def test_stage_executor_raising_yields_error_response():
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_request,
        CHAIN_MSG_TYPE, CHAIN_REQ_KEY, CHAIN_PAYLOAD_KEY,
        CHAIN_ERROR_KEY, StageExecutionError,
    )

    class _RaisingExecutor:
        async def execute(self, request_bytes: bytes) -> bytes:
            raise StageExecutionError("forward-pass failed: OOM")

    node = _make_node()
    node._chain_stage_executor = _RaisingExecutor()
    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_REQ_KEY: "req-1",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"x").decode(),
    })

    _run(handle_chain_executor_request(node, msg))
    err = node._sent[0][1].payload[CHAIN_ERROR_KEY]
    assert "forward-pass failed" in err or "OOM" in err
