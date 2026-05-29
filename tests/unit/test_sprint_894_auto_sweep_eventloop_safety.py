"""Sprint 894 — funnel auto-sweep must not block the event loop.

sp887 DoS finding. FunnelAutoSweepWorker._run_one_sweep called the
SYNCHRONOUS sweep_fn inline (`result = self._sweep_fn()`). The node
wires sweep_fn to a sync closure (node.py `_do_sweep`) that runs
OnrampFunnel.sweep → sp862 WalletBalanceReader.get_balances → blocking
Base RPC (httpx, 15s timeout) for EACH open intent (N intents × 3
calls). Running that inline blocks the daemon's asyncio event loop for
up to N×3×15s — stalling EVERY concurrent HTTP request the daemon
serves. A single slow/hanging RPC near its timeout becomes a
daemon-wide liveness DoS, not just a stalled sweep.

Sp894 offloads a synchronous sweep_fn to the default thread-pool
executor so a slow sweep can't stall the loop. An ASYNC sweep_fn is
still awaited cooperatively on the loop (it yields on its own I/O).

The proof is thread identity: a sync sweep must run on a DIFFERENT
thread than the event-loop thread (deterministic, not timing-based).
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from prsm.node.funnel_auto_sweep import (
    FunnelAutoSweepWorker,
    FunnelAutoSweepConfig,
)


def _cfg() -> FunnelAutoSweepConfig:
    return FunnelAutoSweepConfig(interval_seconds=300.0)


# ── The fix: sync sweep runs OFF the event-loop thread ───────

@pytest.mark.asyncio
async def test_sync_sweep_offloaded_to_worker_thread():
    """A synchronous sweep_fn must execute on a thread-pool thread,
    NOT the event-loop thread — otherwise a blocking RPC stalls the
    whole daemon. Pre-sp894 it ran inline (same thread) → this fails."""
    loop_thread = threading.get_ident()
    captured = {}

    def sync_sweep():
        captured["thread"] = threading.get_ident()
        return {"checked": 1, "confirmed_new": 0, "expired_new": 0}

    worker = FunnelAutoSweepWorker(sweep_fn=sync_sweep, config=_cfg())
    await worker._run_one_sweep()

    assert "thread" in captured
    assert captured["thread"] != loop_thread, (
        "sync sweep ran on the event-loop thread — a blocking RPC "
        "would stall the entire daemon"
    )


@pytest.mark.asyncio
async def test_blocking_sync_sweep_keeps_loop_responsive():
    """End-to-end behavioral proof: while a sync sweep BLOCKS (on a
    threading primitive), a concurrent coroutine still makes
    progress. If the sweep ran on the loop, the loop would be frozen
    and the concurrent task could never release the sweep → the test
    would deadlock/timeout."""
    release = threading.Event()
    started = threading.Event()

    def blocking_sweep():
        started.set()
        # Block in the worker thread until the loop signals release.
        # If this ran ON the loop, the loop could never set `release`.
        assert release.wait(timeout=5.0), "loop never released sweep"
        return {"checked": 1, "confirmed_new": 1, "expired_new": 0}

    worker = FunnelAutoSweepWorker(
        sweep_fn=blocking_sweep, config=_cfg(),
    )
    sweep_task = asyncio.create_task(worker._run_one_sweep())

    # The loop is free to run THIS code only if the sweep isn't
    # blocking it. Spin (yielding) until the sweep signals it started
    # in its thread, then release it.
    for _ in range(500):
        if started.is_set():
            break
        await asyncio.sleep(0)
    assert started.is_set(), "sweep never started — event loop blocked"
    release.set()

    result = await asyncio.wait_for(sweep_task, timeout=5.0)
    assert result["checked"] == 1
    assert worker.total_confirmed == 1


# ── Async sweep_fn still cooperatively awaited on the loop ───

@pytest.mark.asyncio
async def test_async_sweep_runs_on_loop_thread():
    """An async sweep_fn yields on its own I/O — it should be awaited
    on the loop thread, NOT shoved into a thread (which would defeat
    its cooperative design)."""
    loop_thread = threading.get_ident()
    captured = {}

    async def async_sweep():
        captured["thread"] = threading.get_ident()
        return {"checked": 2, "confirmed_new": 0, "expired_new": 0}

    worker = FunnelAutoSweepWorker(sweep_fn=async_sweep, config=_cfg())
    result = await worker._run_one_sweep()

    assert captured["thread"] == loop_thread
    assert result["checked"] == 2


# ── Regression: counters + error propagation preserved ───────

@pytest.mark.asyncio
async def test_offloaded_sync_sweep_still_aggregates_counters():
    def sync_sweep():
        return {"checked": 3, "confirmed_new": 2, "expired_new": 1}

    worker = FunnelAutoSweepWorker(sweep_fn=sync_sweep, config=_cfg())
    await worker._run_one_sweep()
    await worker._run_one_sweep()
    assert worker.sweeps_run == 2
    assert worker.total_confirmed == 4
    assert worker.total_expired == 2


@pytest.mark.asyncio
async def test_offloaded_sync_sweep_propagates_errors():
    """A raising sync sweep must still propagate out of
    _run_one_sweep (the run-loop's except catches it + increments
    sweep_failures) — offloading must not swallow the error."""
    def boom():
        raise RuntimeError("rpc exploded")

    worker = FunnelAutoSweepWorker(sweep_fn=boom, config=_cfg())
    with pytest.raises(RuntimeError, match="rpc exploded"):
        await worker._run_one_sweep()
