"""Sprint 728 F61 — unary execution timeout.

Pre-728, `handle_chain_executor_request` called
`await executor.execute(payload_bytes)` with no timeout. If the
executor hangs (model load deadlock, infinite loop, blocked
network call to unreachable backend), the unary handler hangs
indefinitely. Consequences:

  - Sprint 726's per-peer cap slot stays held forever → peer's
    legitimate subsequent requests get rejected at the cap.
  - Requester's pending future never resolves (until the
    requester's own send_message_adapter timeout — 30s default —
    fires + cleans up).
  - Server-side resources (memory, possibly KV cache for real
    autoregressive backends) held indefinitely.

Fix:
- `_resolve_unary_execution_timeout()` reads
  `PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S` (default 60s; <=0 =
  disabled; non-float defaults to 60).
- Wrap `executor.execute(...)` in `asyncio.wait_for(..., timeout)`.
- Convert asyncio.TimeoutError (a TimeoutError subclass) into a
  CHAIN_ERROR_KEY response naming the env var so operators
  reading logs know what to tune.
"""
from __future__ import annotations

import inspect
import os


def test_resolve_unary_execution_timeout_default():
    """Unset env → 60.0 seconds."""
    from prsm.node.chain_executor_adapters import (
        _resolve_unary_execution_timeout,
    )
    os.environ.pop("PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S", None)
    assert _resolve_unary_execution_timeout() == 60.0


def test_resolve_unary_execution_timeout_explicit_override():
    """Valid float env → that value."""
    from prsm.node.chain_executor_adapters import (
        _resolve_unary_execution_timeout,
    )
    os.environ["PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S"] = "5.5"
    try:
        assert _resolve_unary_execution_timeout() == 5.5
    finally:
        del os.environ["PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S"]


def test_resolve_unary_execution_timeout_zero_disabled():
    """0 → disabled (pre-728 behavior)."""
    from prsm.node.chain_executor_adapters import (
        _resolve_unary_execution_timeout,
    )
    os.environ["PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S"] = "0"
    try:
        assert _resolve_unary_execution_timeout() == 0.0
    finally:
        del os.environ["PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S"]


def test_resolve_unary_execution_timeout_negative_disabled():
    """Negative → disabled."""
    from prsm.node.chain_executor_adapters import (
        _resolve_unary_execution_timeout,
    )
    os.environ["PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S"] = "-1"
    try:
        assert _resolve_unary_execution_timeout() == 0.0
    finally:
        del os.environ["PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S"]


def test_resolve_unary_execution_timeout_typo_safely_defaults():
    """Non-float → safe-default 60.0."""
    from prsm.node.chain_executor_adapters import (
        _resolve_unary_execution_timeout,
    )
    os.environ["PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S"] = "sixty"
    try:
        assert _resolve_unary_execution_timeout() == 60.0
    finally:
        del os.environ["PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S"]


def _unary_handler_source() -> str:
    """Sprint 723/726 refactor: handler body lives in
    `_handle_chain_executor_request_body`."""
    from prsm.node import chain_executor_adapters as _mod
    return inspect.getsource(_mod._handle_chain_executor_request_body)


def test_unary_handler_uses_wait_for_with_resolver():
    """Pin: handler uses `asyncio.wait_for(executor.execute(...),
    timeout=...)` so a hung executor doesn't block forever."""
    src = _unary_handler_source()
    assert "_resolve_unary_execution_timeout" in src, (
        "F61 requires the timeout to be wired into the unary "
        "handler"
    )
    assert "wait_for" in src, (
        "F61 requires asyncio.wait_for() around executor.execute() "
        "to enforce the timeout"
    )


def test_unary_handler_timeout_error_references_env_var():
    """Pin: timeout error message references the env var so
    operators reading logs know what to tune."""
    src = _unary_handler_source()
    assert "PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S" in src, (
        "timeout-exceeded error must reference the env var so "
        "operators know what to tune"
    )
    assert "execution timeout" in src
