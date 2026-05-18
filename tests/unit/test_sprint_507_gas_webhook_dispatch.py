"""Sprint 507 — webhook firing on gas-status transitions.

Sprint 506's GasStatusMonitor logs transitions but only to
stdout/log. Operators running off-shift want a push to PagerDuty,
Slack, or their own alerting endpoint. Sprint 507 wires the
sprint-446 WebhookDeliverer into GasStatusMonitor so every
transition (ok↔low↔critical) fires a POST to PRSM_WEBHOOK_URL
with a structured payload.

Webhook payload:
  {
    "event": "gas.transition",
    "node_id": <node-id>,
    "address": <wallet address>,
    "previous_status": "ok|low|critical",
    "new_status": "ok|low|critical",
    "eth_balance": float,
    "timestamp": float
  }

Boundary: tests inject a fake AsyncMock deliverer + verify it
was called with the right payload shape. Real wire test is via
the daemon log + a mock HTTP receiver.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.economy.ftns_onchain import (
    GasStatusMonitor,
    OnChainFTNSLedger,
)


def _build_ledger(eth_wei=10**15, has_w3=True):
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


@pytest.mark.asyncio
async def test_monitor_fires_webhook_on_ok_to_low():
    deliverer = MagicMock()
    deliverer.deliver = AsyncMock()
    led = _build_ledger(eth_wei=10**15)
    mon = GasStatusMonitor(
        led, interval_seconds=60,
        webhook_deliverer=deliverer,
        webhook_url="http://example.test/hook",
        webhook_secret="s",
    )

    # baseline tick (ok) — no webhook fire
    await mon._tick_async()
    deliverer.deliver.assert_not_called()

    # drop to low
    led.w3.eth.get_balance.return_value = 3 * 10**14
    await mon._tick_async()
    deliverer.deliver.assert_called_once()
    kwargs = deliverer.deliver.call_args.kwargs
    assert kwargs["url"] == "http://example.test/hook"
    assert kwargs["event"] == "gas.transition"
    assert kwargs["secret"] == "s"
    payload = kwargs["payload"]
    assert payload["previous_status"] == "ok"
    assert payload["new_status"] == "low"
    assert payload["address"] == "0xAAAA"
    assert payload["eth_balance"] == 0.0003


@pytest.mark.asyncio
async def test_monitor_no_webhook_when_no_deliverer():
    """Sprint-506 behavior preserved: without deliverer,
    transitions still log but no webhook firing."""
    led = _build_ledger(eth_wei=10**15)
    mon = GasStatusMonitor(led, interval_seconds=60)
    await mon._tick_async()
    led.w3.eth.get_balance.return_value = 3 * 10**14
    await mon._tick_async()
    # No exception, no deliverer to assert against —
    # behavior preserved.
    assert mon._last_status == "low"


@pytest.mark.asyncio
async def test_monitor_no_webhook_on_unchanged_status():
    deliverer = MagicMock()
    deliverer.deliver = AsyncMock()
    led = _build_ledger(eth_wei=10**15)
    mon = GasStatusMonitor(
        led, interval_seconds=60,
        webhook_deliverer=deliverer,
        webhook_url="http://example.test/hook",
    )
    await mon._tick_async()
    await mon._tick_async()
    await mon._tick_async()
    deliverer.deliver.assert_not_called()


@pytest.mark.asyncio
async def test_monitor_fires_for_recovery_too():
    """Recovery (low → ok) must also fire — operators
    want to know the alert is cleared."""
    deliverer = MagicMock()
    deliverer.deliver = AsyncMock()
    led = _build_ledger(eth_wei=3 * 10**14)  # low
    mon = GasStatusMonitor(
        led, interval_seconds=60,
        webhook_deliverer=deliverer,
        webhook_url="http://example.test/hook",
    )
    await mon._tick_async()  # baseline low
    led.w3.eth.get_balance.return_value = 10**15  # ok
    await mon._tick_async()
    deliverer.deliver.assert_called_once()
    payload = deliverer.deliver.call_args.kwargs["payload"]
    assert payload["previous_status"] == "low"
    assert payload["new_status"] == "ok"


@pytest.mark.asyncio
async def test_monitor_webhook_failure_does_not_crash():
    """If webhook delivery raises, the monitor must NOT
    crash — the log still gets the transition, the tick
    loop continues."""
    deliverer = MagicMock()
    deliverer.deliver = AsyncMock(
        side_effect=RuntimeError("hook down"),
    )
    led = _build_ledger(eth_wei=10**15)
    mon = GasStatusMonitor(
        led, interval_seconds=60,
        webhook_deliverer=deliverer,
        webhook_url="http://example.test/hook",
    )
    await mon._tick_async()
    led.w3.eth.get_balance.return_value = 3 * 10**14
    # Must not raise
    await mon._tick_async()
    assert mon._last_status == "low"
