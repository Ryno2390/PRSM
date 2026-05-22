"""Sprint 713 — bounded receive-queue back-pressure for remote streaming.

Sprint 711 shipped the F40 wire protocol with an UNBOUNDED per-stream
asyncio.Queue on the requester side. Under cold-load + many concurrent
streams that's an OOM vector the sprint-704 unary semaphore doesn't
cover. Sprint 713 closes it:

  - Default queue maxsize = 64 (2× typical inference chunk-count)
  - Operators override via PRSM_CHAIN_STREAM_QUEUE_MAXSIZE
  - Values <= 0 mean unbounded (disabled gate)
  - Non-int values default to 64 safely (no streaming-setup failure)

Frame-put path now uses `run_coroutine_threadsafe(queue.put(...), loop)`
so the producer (transport handler) AWAITS at the put rather than
dropping frames on QueueFull. Terminal STREAM_END bypasses
back-pressure via put_nowait so the requester always sees the close
signal even when the queue is saturated.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock

import pytest


def test_resolve_stream_queue_maxsize_default():
    """Unset env → 64 (sprint 713 chosen baseline)."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_queue_maxsize,
    )
    os.environ.pop("PRSM_CHAIN_STREAM_QUEUE_MAXSIZE", None)
    assert _resolve_stream_queue_maxsize() == 64


def test_resolve_stream_queue_maxsize_explicit_override():
    """Valid int env → that value."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_queue_maxsize,
    )
    os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"] = "16"
    try:
        assert _resolve_stream_queue_maxsize() == 16
    finally:
        del os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"]


def test_resolve_stream_queue_maxsize_zero_means_unbounded():
    """0 → unbounded (pre-713 behavior preserved for operators who
    explicitly opt out of the gate)."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_queue_maxsize,
    )
    os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"] = "0"
    try:
        assert _resolve_stream_queue_maxsize() == 0
    finally:
        del os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"]


def test_resolve_stream_queue_maxsize_negative_means_unbounded():
    """Negative values → unbounded (defensive; treats as opt-out)."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_queue_maxsize,
    )
    os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"] = "-1"
    try:
        assert _resolve_stream_queue_maxsize() == 0
    finally:
        del os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"]


def test_resolve_stream_queue_maxsize_typo_safely_defaults():
    """Non-int values must NOT raise — streaming setup must not
    fail because of a malformed env var. Falls back to 64 (the
    safe default) rather than 0 (unbounded) because typos likely
    came from operators who DID want a gate."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_queue_maxsize,
    )
    os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"] = "sixty-four"
    try:
        assert _resolve_stream_queue_maxsize() == 64
    finally:
        del os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"]


def test_resolve_stream_queue_maxsize_empty_string_defaults():
    """Empty string (e.g., `export PRSM_CHAIN_STREAM_QUEUE_MAXSIZE=`)
    must default to 64, not raise."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_queue_maxsize,
    )
    os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"] = ""
    try:
        assert _resolve_stream_queue_maxsize() == 64
    finally:
        del os.environ["PRSM_CHAIN_STREAM_QUEUE_MAXSIZE"]


@pytest.mark.asyncio
async def test_make_async_queue_respects_maxsize():
    """`_make_async_queue(loop, maxsize=N)` must produce an
    asyncio.Queue with that maxsize so put_nowait raises QueueFull
    at the bound."""
    from prsm.node.chain_executor_adapters import _make_async_queue
    loop = asyncio.get_event_loop()
    q = await _make_async_queue(loop, maxsize=2)
    q.put_nowait(b"f1")
    q.put_nowait(b"f2")
    with pytest.raises(asyncio.QueueFull):
        q.put_nowait(b"f3")


@pytest.mark.asyncio
async def test_make_async_queue_maxsize_zero_unbounded():
    """maxsize=0 → unbounded (pre-713 behavior preserved)."""
    from prsm.node.chain_executor_adapters import _make_async_queue
    loop = asyncio.get_event_loop()
    q = await _make_async_queue(loop, maxsize=0)
    # Drop 1000 frames without QueueFull
    for i in range(1000):
        q.put_nowait(i)
    assert q.qsize() == 1000


def test_response_handler_uses_run_coroutine_threadsafe_for_frames():
    """Pin: sprint 713 frame-put path must use
    `run_coroutine_threadsafe` (which awaits put → back-pressure)
    instead of `put_nowait` (which would raise QueueFull and lose
    frames on a saturated queue)."""
    import inspect
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_response,
    )
    src = inspect.getsource(handle_chain_stream_response)
    assert "run_coroutine_threadsafe" in src, (
        "frame-put path must await on a full queue (sprint 713 "
        "back-pressure), not drop frames"
    )
    assert "queue.put(" in src, (
        "frame-put must use queue.put (awaitable) not put_nowait"
    )


def test_response_handler_end_entry_still_bypasses_backpressure():
    """Pin: terminal STREAM_END must keep using put_nowait (NOT
    awaited) so the requester always sees the close signal — even
    when the queue is saturated. Otherwise a saturated queue would
    deadlock the requester waiting for end."""
    import inspect
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_response,
    )
    src = inspect.getsource(handle_chain_stream_response)
    # Find the END branch (between "STREAM_END_KEY" check and the
    # frame-decode branch) and assert put_nowait is the path.
    end_branch_marker = "CHAIN_STREAM_END_KEY"
    frame_marker = "Mid-stream frame"
    assert end_branch_marker in src and frame_marker in src
    end_idx = src.find(end_branch_marker)
    frame_idx = src.find(frame_marker)
    assert 0 < end_idx < frame_idx
    end_branch_src = src[end_idx:frame_idx]
    assert "put_nowait" in end_branch_src, (
        "terminal STREAM_END must use put_nowait so requester always"
        " sees close — never blocked by full queue"
    )
