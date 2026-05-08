"""HeartbeatScheduler — async daemon that calls
StorageSlashingClient.record_heartbeat() on cadence.

Closes the deferred-follow-on item from EXPLOIT_RESPONSE_PLAYBOOK_
ANNEX_2026_05.md §6.2: providers running v1.7.0 had to invoke
record_heartbeat externally (cron, manual, custom service) until a
daemon shipped. Without the daemon, providers become vulnerable to
permissionless slash_for_missing_heartbeat() once their grace window
elapses.

Tests use a stub StorageSlashingClient to verify the daemon's
loop shape, error swallowing, and stop semantics without requiring
a live chain.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List
from unittest.mock import MagicMock

import pytest

from prsm.economy.web3.heartbeat_scheduler import HeartbeatScheduler
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
    TransferStatus,
)


# ──────────────────────────────────────────────────────────────────────
# Stub StorageSlashingClient
# ──────────────────────────────────────────────────────────────────────


class _FakeSlashingClient:
    """Records calls + lets tests inject per-call outcomes.

    `outcomes` is a list of values: each is either
      - a (tx_hash_hex, TransferStatus) tuple — happy path
      - an Exception class to instantiate and raise
      - an Exception instance to raise directly
    Consumed FIFO; once exhausted, subsequent calls return a default
    happy result.
    """

    def __init__(self, outcomes=None):
        self._outcomes = list(outcomes or [])
        self.calls: List[None] = []
        self.address = "0x" + "11" * 20

    def record_heartbeat(self):
        self.calls.append(None)
        if not self._outcomes:
            return ("0x" + "ab" * 32, TransferStatus.CONFIRMED)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, type) and issubclass(outcome, Exception):
            raise outcome("injected")
        return outcome


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_requires_client(self):
        with pytest.raises(TypeError):
            HeartbeatScheduler()  # type: ignore[call-arg]

    def test_default_interval_is_positive(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.interval_seconds > 0

    def test_custom_interval(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client, interval_seconds=10.0)
        assert scheduler.interval_seconds == 10.0

    def test_zero_interval_rejected(self):
        client = _FakeSlashingClient()
        with pytest.raises(ValueError, match="interval"):
            HeartbeatScheduler(client=client, interval_seconds=0)

    def test_negative_interval_rejected(self):
        client = _FakeSlashingClient()
        with pytest.raises(ValueError, match="interval"):
            HeartbeatScheduler(client=client, interval_seconds=-5)


# ──────────────────────────────────────────────────────────────────────
# Single tick
# ──────────────────────────────────────────────────────────────────────


class TestSingleTick:
    @pytest.mark.asyncio
    async def test_happy_path_calls_record_heartbeat(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_happy_path_increments_success_counter(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.success_count == 0
        await scheduler.tick()
        assert scheduler.success_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_failure_swallowed(self):
        client = _FakeSlashingClient(outcomes=[BroadcastFailedError])
        scheduler = HeartbeatScheduler(client=client)
        # MUST NOT raise — daemon stays alive.
        await scheduler.tick()
        assert scheduler.success_count == 0
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_pending_error_swallowed_but_logged_loud(self, caplog):
        # OnChainPendingError is the concerning case — receipt unknown.
        # The daemon should NOT crash but SHOULD log at WARNING+.
        client = _FakeSlashingClient(
            outcomes=[OnChainPendingError("pending", tx_hash="0xdead")],
        )
        scheduler = HeartbeatScheduler(client=client)
        with caplog.at_level(logging.WARNING):
            await scheduler.tick()
        assert any(
            "pending" in r.message.lower() or "unknown" in r.message.lower()
            for r in caplog.records
        )
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_reverted_error_swallowed(self):
        client = _FakeSlashingClient(outcomes=[OnChainRevertedError])
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_unexpected_exception_swallowed(self):
        # If client raises an unexpected type, daemon stays alive.
        client = _FakeSlashingClient(outcomes=[RuntimeError("weird")])
        scheduler = HeartbeatScheduler(client=client)
        await scheduler.tick()
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_callback_fires_on_success(self):
        client = _FakeSlashingClient()
        events = []

        async def cb(tx_hash):
            events.append(tx_hash)

        scheduler = HeartbeatScheduler(client=client, on_success=cb)
        await scheduler.tick()
        assert len(events) == 1
        assert events[0].startswith("0x")

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash_daemon(self):
        client = _FakeSlashingClient()

        def cb(tx_hash):
            raise RuntimeError("callback exploded")

        scheduler = HeartbeatScheduler(client=client, on_success=cb)
        # Must not propagate — the bug is in the operator's callback,
        # not the daemon, and the daemon must keep running.
        await scheduler.tick()
        assert scheduler.success_count == 1


# ──────────────────────────────────────────────────────────────────────
# Run loop
# ──────────────────────────────────────────────────────────────────────


class TestRunForever:
    @pytest.mark.asyncio
    async def test_run_forever_exits_when_stop_called(self):
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client, interval_seconds=0.05)

        task = asyncio.create_task(scheduler.run_forever())
        # Let it tick at least once.
        await asyncio.sleep(0.15)
        await scheduler.stop()
        # run_forever should now return promptly.
        await asyncio.wait_for(task, timeout=1.0)

        # At least one heartbeat occurred.
        assert len(client.calls) >= 1

    @pytest.mark.asyncio
    async def test_multiple_direct_ticks_increment_counters(self):
        # Avoids loop-timing flakiness under pytest-asyncio fixture
        # overhead. Direct `tick()` calls verify the cadence-relevant
        # property: successive ticks accumulate counter state.
        client = _FakeSlashingClient()
        scheduler = HeartbeatScheduler(client=client, interval_seconds=60.0)

        for _ in range(5):
            await scheduler.tick()

        assert scheduler.success_count == 5
        assert scheduler.failure_count == 0
        assert len(client.calls) == 5

    @pytest.mark.asyncio
    async def test_multiple_direct_ticks_keep_running_through_failures(self):
        # Mix successes + failures via outcome injection. The
        # cadence-irrelevant property under test: scheduler counters
        # correctly classify each call regardless of order, and no
        # exception escapes tick() to crash a hypothetical loop.
        client = _FakeSlashingClient(outcomes=[
            BroadcastFailedError,
            ("0x" + "01" * 32, TransferStatus.CONFIRMED),
            OnChainRevertedError,
            ("0x" + "02" * 32, TransferStatus.CONFIRMED),
            OnChainPendingError("pending", tx_hash="0xfe"),
            ("0x" + "03" * 32, TransferStatus.CONFIRMED),
        ])
        scheduler = HeartbeatScheduler(client=client, interval_seconds=60.0)

        for _ in range(6):
            await scheduler.tick()

        # 3 successes, 3 failures (broadcast + revert + pending).
        assert scheduler.success_count == 3
        assert scheduler.failure_count == 3
