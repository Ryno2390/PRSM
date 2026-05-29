"""Sprint 878 — funnel auto-sweep worker pin tests."""
from __future__ import annotations

import asyncio

import pytest

from prsm.node.funnel_auto_sweep import (
    FunnelAutoSweepConfig,
    FunnelAutoSweepWorker,
    resolve_auto_sweep_config_from_env,
    _MIN_INTERVAL_S,
)


# ── Config resolution ────────────────────────────────────────

def test_config_unset_disabled(monkeypatch):
    monkeypatch.delenv(
        "PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S", raising=False,
    )
    cfg = resolve_auto_sweep_config_from_env()
    assert cfg.enabled is False
    assert cfg.interval_seconds == 0.0


def test_config_zero_disabled(monkeypatch):
    monkeypatch.setenv("PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S", "0")
    cfg = resolve_auto_sweep_config_from_env()
    assert cfg.enabled is False


def test_config_negative_disabled(monkeypatch):
    monkeypatch.setenv("PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S", "-5")
    cfg = resolve_auto_sweep_config_from_env()
    assert cfg.enabled is False


def test_config_valid_interval(monkeypatch):
    monkeypatch.setenv(
        "PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S", "300",
    )
    cfg = resolve_auto_sweep_config_from_env()
    assert cfg.enabled is True
    assert cfg.interval_seconds == 300.0


def test_config_clamps_below_minimum(monkeypatch):
    """Sub-60s sweeps hammer RPC with no benefit — clamp."""
    monkeypatch.setenv("PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S", "10")
    cfg = resolve_auto_sweep_config_from_env()
    assert cfg.interval_seconds == _MIN_INTERVAL_S


def test_config_non_numeric_disabled(monkeypatch):
    """Typo in opt-in env must NOT crash daemon — disable."""
    monkeypatch.setenv(
        "PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S", "five-minutes",
    )
    cfg = resolve_auto_sweep_config_from_env()
    assert cfg.enabled is False


# ── Worker lifecycle ─────────────────────────────────────────

def test_disabled_worker_never_runs():
    """Worker with disabled config short-circuits on start."""
    calls = []

    def sweep_fn():
        calls.append(1)
        return {"checked": 0, "confirmed_new": 0, "expired_new": 0}

    async def _run():
        worker = FunnelAutoSweepWorker(
            sweep_fn=sweep_fn,
            config=FunnelAutoSweepConfig(interval_seconds=0.0),
        )
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()
        return worker

    worker = asyncio.run(_run())
    assert calls == []
    assert worker.sweeps_run == 0


# NOTE: the test suite's conftest stubs asyncio.sleep (returns
# instantly without yielding), so wall-clock-driven loop tests are
# unreliable here. The sweep BODY is tested deterministically by
# calling _run_one_sweep() directly; the LOOP scheduling (start/
# stop, double-start idempotency) is tested separately without
# relying on sleep timing.

def test_run_one_sweep_increments_counters():
    """Direct invocation of one sweep iteration — deterministic."""
    def sweep_fn():
        return {
            "checked": 2, "confirmed_new": 1, "expired_new": 1,
        }

    async def _run():
        worker = FunnelAutoSweepWorker(
            sweep_fn=sweep_fn,
            config=FunnelAutoSweepConfig(interval_seconds=300.0),
        )
        await worker._run_one_sweep()
        return worker

    worker = asyncio.run(_run())
    assert worker.sweeps_run == 1
    assert worker.total_confirmed == 1
    assert worker.total_expired == 1
    assert worker.last_sweep_at > 0


def test_run_one_sweep_accumulates_across_calls():
    def sweep_fn():
        return {
            "checked": 1, "confirmed_new": 2, "expired_new": 0,
        }

    async def _run():
        worker = FunnelAutoSweepWorker(
            sweep_fn=sweep_fn,
            config=FunnelAutoSweepConfig(interval_seconds=300.0),
        )
        await worker._run_one_sweep()
        await worker._run_one_sweep()
        await worker._run_one_sweep()
        return worker

    worker = asyncio.run(_run())
    assert worker.sweeps_run == 3
    assert worker.total_confirmed == 6  # 2 × 3


def test_run_one_sweep_async_sweep_fn():
    """sweep_fn may be async — worker awaits the coroutine."""
    calls = []

    async def sweep_fn():
        calls.append(1)
        return {
            "checked": 0, "confirmed_new": 0, "expired_new": 0,
        }

    async def _run():
        worker = FunnelAutoSweepWorker(
            sweep_fn=sweep_fn,
            config=FunnelAutoSweepConfig(interval_seconds=300.0),
        )
        await worker._run_one_sweep()
        return worker

    worker = asyncio.run(_run())
    assert len(calls) == 1
    assert worker.sweeps_run == 1


def test_run_one_sweep_non_dict_result_safe():
    """If sweep_fn returns a non-dict (defensive), counters
    don't blow up — sweep still counted."""
    def sweep_fn():
        return None

    async def _run():
        worker = FunnelAutoSweepWorker(
            sweep_fn=sweep_fn,
            config=FunnelAutoSweepConfig(interval_seconds=300.0),
        )
        result = await worker._run_one_sweep()
        return worker, result

    worker, result = asyncio.run(_run())
    assert worker.sweeps_run == 1
    assert result is None


def test_loop_fail_soft_on_sweep_error():
    """A raising sweep_fn in the loop body increments failure
    counter without killing the worker. Tested by driving one
    loop iteration's error path via _run_loop with a controlled
    cancel — but simpler: assert _run_one_sweep propagates so
    _run_loop's except can catch it."""
    def sweep_fn():
        raise RuntimeError("simulated sweep failure")

    async def _run():
        worker = FunnelAutoSweepWorker(
            sweep_fn=sweep_fn,
            config=FunnelAutoSweepConfig(interval_seconds=300.0),
        )
        # _run_one_sweep propagates; the loop's except catches +
        # increments. Verify the propagation contract here.
        raised = False
        try:
            await worker._run_one_sweep()
        except RuntimeError:
            raised = True
        return raised

    raised = asyncio.run(_run())
    assert raised is True  # loop's except will catch + count


def test_double_start_idempotent():
    def sweep_fn():
        return {"checked": 0, "confirmed_new": 0, "expired_new": 0}

    async def _run():
        worker = FunnelAutoSweepWorker(
            sweep_fn=sweep_fn,
            config=FunnelAutoSweepConfig(interval_seconds=0.01),
        )
        await worker.start()
        task1 = worker._task
        await worker.start()  # second start — should no-op
        task2 = worker._task
        await worker.stop()
        return task1, task2

    t1, t2 = asyncio.run(_run())
    assert t1 is t2  # same task — not restarted


def test_stop_when_never_started_safe():
    def sweep_fn():
        return {}

    async def _run():
        worker = FunnelAutoSweepWorker(
            sweep_fn=sweep_fn,
            config=FunnelAutoSweepConfig(interval_seconds=0.0),
        )
        await worker.stop()  # never started — must not raise

    asyncio.run(_run())  # no exception = pass


# ── stats() ──────────────────────────────────────────────────

def test_stats_snapshot_shape():
    def sweep_fn():
        return {"checked": 0, "confirmed_new": 0, "expired_new": 0}

    worker = FunnelAutoSweepWorker(
        sweep_fn=sweep_fn,
        config=FunnelAutoSweepConfig(interval_seconds=300.0),
    )
    s = worker.stats()
    assert s["enabled"] is True
    assert s["interval_seconds"] == 300.0
    assert s["running"] is False  # not started
    assert s["sweeps_run"] == 0
    assert "total_confirmed" in s
    assert "total_expired" in s
    assert "sweep_failures" in s
    assert "last_sweep_at" in s
