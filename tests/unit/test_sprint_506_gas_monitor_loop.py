"""Sprint 506 — periodic background gas-status sampler.

Sprint 504 logs gas status once at daemon startup. Sprint 506
adds a periodic background task that re-samples every N seconds
and logs ONLY on status transitions (ok → low → critical and
back). This gives operators a real-time signal without spamming
logs.

Boundary: tests drive the sampler's tick logic directly with a
mocked w3.get_balance. The asyncio loop wrapper
(`run_forever`) is exercised end-to-end by a short-interval
integration test.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from prsm.economy.ftns_onchain import (
    OnChainFTNSLedger,
    GasStatusMonitor,
)


def _build_ledger(eth_wei=None, has_w3=True):
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
    )
    led._connected_address = "0xAAAA"
    if has_w3:
        led.w3 = MagicMock()
        led.w3.eth.get_balance.return_value = eth_wei
    else:
        led.w3 = None
    return led


def test_monitor_tick_transitions_ok_to_low_logs_warning(caplog):
    """First tick reports ok, second tick (after balance
    drops below threshold) must log a WARNING transition
    message."""
    led = _build_ledger(eth_wei=10**15)  # 0.001 → ok
    mon = GasStatusMonitor(led, interval_seconds=60)

    with caplog.at_level(logging.WARNING):
        # 1st tick: ok — no warning yet (initial baseline)
        mon._tick_sync()
        # Balance drops to 0.0003 ETH (low)
        led.w3.eth.get_balance.return_value = 3 * 10**14
        mon._tick_sync()

    transitions = [
        r for r in caplog.records
        if "transition" in r.getMessage().lower()
        or "ok → low" in r.getMessage()
        or "ok->low" in r.getMessage()
    ]
    assert transitions, (
        f"expected ok→low transition log, got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_monitor_tick_does_not_log_when_status_unchanged(
    caplog,
):
    """Repeated ok ticks must NOT spam logs."""
    led = _build_ledger(eth_wei=10**15)
    mon = GasStatusMonitor(led, interval_seconds=60)
    with caplog.at_level(logging.WARNING):
        mon._tick_sync()
        mon._tick_sync()
        mon._tick_sync()
        mon._tick_sync()
    transitions = [
        r for r in caplog.records
        if "transition" in r.getMessage().lower()
    ]
    assert not transitions, (
        "repeated same-status ticks must not log"
    )


def test_monitor_tick_low_to_critical_escalates_to_error(
    caplog,
):
    """Transition from low → critical must log at ERROR
    level (more urgent)."""
    led = _build_ledger(eth_wei=3 * 10**14)  # low
    mon = GasStatusMonitor(led, interval_seconds=60)
    with caplog.at_level(logging.WARNING):
        mon._tick_sync()  # baseline = low (no log)
        led.w3.eth.get_balance.return_value = 5 * 10**13  # critical
        mon._tick_sync()
    error_records = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR
    ]
    assert error_records


def test_monitor_recovery_logs_info(caplog):
    """Transition from low → ok (operator topped up) must
    log an INFO confirmation."""
    led = _build_ledger(eth_wei=3 * 10**14)  # low
    mon = GasStatusMonitor(led, interval_seconds=60)
    with caplog.at_level(logging.INFO):
        mon._tick_sync()  # baseline low
        led.w3.eth.get_balance.return_value = 10**15  # ok
        mon._tick_sync()
    recovery = [
        r for r in caplog.records
        if "recovered" in r.getMessage().lower()
        or "low → ok" in r.getMessage()
        or "low->ok" in r.getMessage()
    ]
    assert recovery


def test_monitor_handles_rpc_exception_gracefully(caplog):
    """RPC failure must not crash the monitor loop."""
    led = _build_ledger(eth_wei=0)
    led.w3.eth.get_balance.side_effect = RuntimeError("rpc down")
    mon = GasStatusMonitor(led, interval_seconds=60)
    with caplog.at_level(logging.DEBUG):
        # Should not raise
        mon._tick_sync()


@pytest.mark.asyncio
async def test_monitor_run_forever_can_be_cancelled():
    """The async run_forever must respond to cancellation
    cleanly (daemon shutdown path)."""
    led = _build_ledger(eth_wei=10**15)
    mon = GasStatusMonitor(led, interval_seconds=0.05)
    task = asyncio.create_task(mon.run_forever())
    await asyncio.sleep(0.15)  # let it tick a few times
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # No assertion needed — the test passes if no
    # exception leaks out.
