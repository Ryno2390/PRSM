"""DaemonWatchdog stop() interrupts sleep promptly (sprint 116).

Pre-fix: stop() set a flag, but the watch loop's
asyncio.sleep(interval_seconds) wasn't interrupted. With
interval=30s, stop() didn't return until the next sleep
elapsed — past node.stop()'s 5s timeout, causing TimeoutError
during shutdown.

Post-fix: asyncio.Event-based interruptible sleep wakes the
loop immediately on stop().
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from prsm.node.daemon_watchdog import DaemonWatchdog


def _watchdog(interval=10.0):
    node = MagicMock()
    node.identity.node_id = "test"
    deliverer = MagicMock()
    return DaemonWatchdog(
        node=node,
        webhook_deliverer=deliverer,
        webhook_url="https://hook.example.com",
        interval_seconds=interval,
    )


@pytest.mark.asyncio
async def test_stop_interrupts_long_sleep():
    """stop() should return quickly even when interval=10s."""
    wd = _watchdog(interval=10.0)
    task = asyncio.create_task(wd.watch())
    # Let the loop enter its sleep
    await asyncio.sleep(0.05)
    # Stop should wake the sleep within the next event-loop tick
    t0 = time.monotonic()
    await wd.stop()
    await asyncio.wait_for(task, timeout=1.0)  # was 5s default
    elapsed = time.monotonic() - t0
    # Pre-fix this took ~10s (interval). Post-fix should be <100ms.
    assert elapsed < 1.0, (
        f"stop() took {elapsed:.2f}s — should be <1s. "
        f"Regression to bare-sleep pattern?"
    )


@pytest.mark.asyncio
async def test_stop_event_initialized_in_watch():
    """Watch must initialize _stop_event lazily so it binds to
    the correct event loop (not module load time)."""
    wd = _watchdog(interval=0.5)
    assert wd._stop_event is None
    task = asyncio.create_task(wd.watch())
    await asyncio.sleep(0.1)
    assert wd._stop_event is not None
    await wd.stop()
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_repeat_start_after_stop():
    """After stop(), watch() should be re-runnable (event resets)."""
    wd = _watchdog(interval=0.5)
    task = asyncio.create_task(wd.watch())
    await asyncio.sleep(0.05)
    await wd.stop()
    await asyncio.wait_for(task, timeout=1.0)
    # Restart
    task2 = asyncio.create_task(wd.watch())
    await asyncio.sleep(0.05)
    await wd.stop()
    await asyncio.wait_for(task2, timeout=1.0)
