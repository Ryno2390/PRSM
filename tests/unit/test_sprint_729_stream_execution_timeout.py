"""Sprint 729 F62 — streaming server total wall-time bound.

Sprint 728 closed F61 on the UNARY path (executor.execute()
timeout). Sprint 729 closes the analogous concern for STREAMING:
a slow or malicious StageExecutor that yields frames at a glacial
rate could hold sprint-723's per-peer cap slot far longer than
any honest stream needs.

This is NOT a sibling of F61 — streaming iteration is sync inside
the async handler, so `asyncio.wait_for` doesn't wrap a `for`-
loop. Instead we check wall-clock between yields. If a single
`__next__()` call hangs indefinitely, the event loop blocks too
(thread-offload refactor would be needed for that separate
attack surface).

Fix:
- `_resolve_stream_execution_timeout()` reads
  `PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S` (default 300s; <=0
  disables; non-float safely defaults to 300s).
- On each iteration, check `time.monotonic() - start > timeout`.
  Exceeded → close inner iterator + ship terminal STREAM_END
  with actionable error referencing the env var.
"""
from __future__ import annotations

import inspect
import os


def test_resolve_stream_execution_timeout_default():
    """Unset env → 300.0s."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_execution_timeout,
    )
    os.environ.pop("PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S", None)
    assert _resolve_stream_execution_timeout() == 300.0


def test_resolve_stream_execution_timeout_explicit_override():
    """Valid float → that value."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_execution_timeout,
    )
    os.environ["PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S"] = "120.5"
    try:
        assert _resolve_stream_execution_timeout() == 120.5
    finally:
        del os.environ["PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S"]


def test_resolve_stream_execution_timeout_zero_disabled():
    """0 → disabled."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_execution_timeout,
    )
    os.environ["PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S"] = "0"
    try:
        assert _resolve_stream_execution_timeout() == 0.0
    finally:
        del os.environ["PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S"]


def test_resolve_stream_execution_timeout_typo_safely_defaults():
    """Non-float → safe-default 300s."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_execution_timeout,
    )
    os.environ["PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S"] = "five-min"
    try:
        assert _resolve_stream_execution_timeout() == 300.0
    finally:
        del os.environ["PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S"]


def _stream_handler_body_source() -> str:
    """Sprint 723 refactor moved body to
    `_handle_stream_request_body`."""
    from prsm.node import chain_executor_adapters as _mod
    return inspect.getsource(_mod._handle_stream_request_body)


def test_stream_handler_iterates_with_wall_time_check():
    """Pin: streaming handler uses `time.monotonic()` to check
    cumulative wall-time against `_resolve_stream_execution_timeout()`
    on each iteration."""
    src = _stream_handler_body_source()
    assert "_resolve_stream_execution_timeout" in src, (
        "F62 fix requires the timeout to be wired into the "
        "streaming handler"
    )
    assert "time.monotonic" in src or "_time.monotonic" in src, (
        "F62 requires wall-clock check; asyncio.wait_for can't "
        "wrap a sync for-loop"
    )


def test_stream_handler_emits_terminal_end_with_env_var_on_timeout():
    """Pin: timeout exceedance ships terminal STREAM_END with
    CHAIN_ERROR_KEY referencing the env var so operators reading
    logs know what to tune."""
    src = _stream_handler_body_source()
    assert "PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S" in src, (
        "timeout-exceeded error must reference the env var so "
        "operators know what to tune"
    )
    assert "exceeded total" in src or "wall-time" in src
