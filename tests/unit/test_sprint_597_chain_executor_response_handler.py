"""Sprint 597 (Phase 2D step 3) — chain-executor response handler.

Tests:
  - Matching request_id → Future resolved with response bytes
  - Wrong subtype → returns False (ignored)
  - Missing request_id → returns False
  - No pending entry → returns False (silent drop, not error)
  - Already-resolved Future → returns False (duplicate response)
  - Malformed base64 → Future gets RuntimeError (not bogus bytes)
"""
from __future__ import annotations

import asyncio
import base64
from unittest.mock import MagicMock


def _make_msg(payload):
    m = MagicMock()
    m.payload = payload
    return m


def test_handler_resolves_matching_future():
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY, CHAIN_PAYLOAD_KEY,
    )

    loop = asyncio.new_event_loop()
    future = loop.create_future()
    node = MagicMock()
    node._chain_executor_pending = {"req-123": future}

    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "req-123",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"response").decode(),
    })

    result = handle_chain_executor_response(node, msg)
    assert result is True
    assert future.done()
    assert future.result() == b"response"


def test_handler_ignores_wrong_subtype():
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_RESP_KEY,
    )
    node = MagicMock()
    node._chain_executor_pending = {}
    msg = _make_msg({
        "subtype": "something_else",
        CHAIN_RESP_KEY: "req-123",
    })
    assert handle_chain_executor_response(node, msg) is False


def test_handler_silent_drop_when_no_pending_entry():
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY, CHAIN_PAYLOAD_KEY,
    )
    node = MagicMock()
    node._chain_executor_pending = {}
    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "unknown-req",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"x").decode(),
    })
    # Silent drop — not an error
    assert handle_chain_executor_response(node, msg) is False


def test_handler_silent_drop_on_duplicate_response():
    """If the Future is already done (duplicate response or post-timeout
    cleanup), drop silently."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY, CHAIN_PAYLOAD_KEY,
    )

    loop = asyncio.new_event_loop()
    future = loop.create_future()
    future.set_result(b"original")
    node = MagicMock()
    node._chain_executor_pending = {"req-1": future}
    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "req-1",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"duplicate").decode(),
    })
    assert handle_chain_executor_response(node, msg) is False
    # Original result preserved
    assert future.result() == b"original"


def test_handler_sets_exception_on_malformed_base64():
    """A bogus base64 payload must NOT set the Future to garbage
    bytes — it must surface a RuntimeError so the caller knows the
    response was malformed.
    """
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY, CHAIN_PAYLOAD_KEY,
    )

    loop = asyncio.new_event_loop()
    future = loop.create_future()
    node = MagicMock()
    node._chain_executor_pending = {"req-1": future}
    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "req-1",
        CHAIN_PAYLOAD_KEY: "!!! not-base64 !!!",
    })
    assert handle_chain_executor_response(node, msg) is True
    assert future.done()
    import pytest
    with pytest.raises(RuntimeError, match="base64"):
        future.result()


def test_handler_no_pending_attr_returns_false():
    """Node without _chain_executor_pending attr (sprint 595 NOT
    applied) → silent False rather than AttributeError."""
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY,
    )

    class _Bare:
        pass

    node = _Bare()
    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "x",
    })
    assert handle_chain_executor_response(node, msg) is False


# --------------------------------------------------------------------------
# Sprint 630 — CHAIN_ERROR_KEY propagation
# --------------------------------------------------------------------------


def test_handler_propagates_chain_error_key_as_exception():
    """Sprint 630 fix: CHAIN_ERROR_KEY in response → Future raises.

    Pre-630 the response-build path (server side) set
    CHAIN_ERROR_KEY="..." + CHAIN_PAYLOAD_KEY="" when a stage
    raised. The handler decoded the empty payload to b"" and
    called `future.set_result(b"")`. Caller's await returned
    empty bytes silently — sprint 624 hit this and saw
    "size_bytes=0" instead of a useful error.

    Post-630: when CHAIN_ERROR_KEY is set, future is rejected
    with StageExecutionError carrying the server-side message
    so callers see the actual failure mode.
    """
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY, CHAIN_PAYLOAD_KEY, CHAIN_ERROR_KEY,
        StageExecutionError,
    )

    loop = asyncio.new_event_loop()
    future = loop.create_future()
    node = MagicMock()
    node._chain_executor_pending = {"req-err": future}

    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "req-err",
        CHAIN_ERROR_KEY: "stage-executor raised RuntimeError: bad model",
        CHAIN_PAYLOAD_KEY: "",
    })

    result = handle_chain_executor_response(node, msg)
    assert result is True
    assert future.done()
    # Future must reject with StageExecutionError, not silently
    # resolve to b"".
    exc = future.exception()
    assert isinstance(exc, StageExecutionError), (
        f"expected StageExecutionError, got {type(exc).__name__}: {exc}"
    )
    assert "bad model" in str(exc)


def test_handler_chain_error_key_takes_precedence_over_payload():
    """Defense in depth: even if payload is also non-empty, the
    presence of CHAIN_ERROR_KEY signals failure — caller must
    not see a "success" result. This protects against any future
    server-side branch that fills both fields by mistake.
    """
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY, CHAIN_PAYLOAD_KEY, CHAIN_ERROR_KEY,
        StageExecutionError,
    )

    loop = asyncio.new_event_loop()
    future = loop.create_future()
    node = MagicMock()
    node._chain_executor_pending = {"req-mixed": future}

    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "req-mixed",
        CHAIN_ERROR_KEY: "diagnostic error message",
        # Defensive: even with non-empty bytes, error key wins.
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"junk").decode(),
    })

    assert handle_chain_executor_response(node, msg) is True
    assert isinstance(future.exception(), StageExecutionError)


def test_handler_empty_error_string_still_resolves_payload():
    """CHAIN_ERROR_KEY="" (empty) is NOT an error; treat as success
    branch. Distinguishes "no error field" / "empty error" / "real
    error" cleanly.
    """
    from prsm.node.chain_executor_adapters import (
        handle_chain_executor_response, CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY, CHAIN_PAYLOAD_KEY, CHAIN_ERROR_KEY,
    )

    loop = asyncio.new_event_loop()
    future = loop.create_future()
    node = MagicMock()
    node._chain_executor_pending = {"req-empty-err": future}

    msg = _make_msg({
        "subtype": CHAIN_MSG_TYPE,
        CHAIN_RESP_KEY: "req-empty-err",
        CHAIN_ERROR_KEY: "",
        CHAIN_PAYLOAD_KEY: base64.b64encode(b"ok").decode(),
    })

    assert handle_chain_executor_response(node, msg) is True
    assert future.result() == b"ok"
