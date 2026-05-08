"""PullAndDistributeScheduler — async daemon that calls
CompensationDistributorClient.pull_and_distribute() on cadence.

Closes the final deferred-follow-on item from
EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md §6.2: per
CompensationDistributor.sol §3.5 contract source comments,
monitoring should fire on call-gap > 7 days. Today's daemon ships
the periodic invocation half of that surface; alerting is
naturally handled by cadence < 7 days (default 86400s = 24h).

Structurally near-twin of HeartbeatScheduler, with longer cadence
and a different surface name. Tests use a stub
CompensationDistributorClient to verify loop shape, error
swallowing, and stop semantics.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

import pytest

from prsm.economy.web3.pull_and_distribute_scheduler import (
    PullAndDistributeScheduler,
)
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
    TransferStatus,
)


class _FakeDistributorClient:
    """Records calls + lets tests inject per-call outcomes.

    Same outcome-injection contract as the heartbeat-scheduler test
    fixture: each `outcomes` entry is either a (tx_hash, status)
    happy tuple, an Exception class to instantiate-and-raise, or an
    Exception instance to raise directly.
    """

    def __init__(self, outcomes=None):
        self._outcomes = list(outcomes or [])
        self.calls: List[None] = []
        self.address = "0x" + "11" * 20

    def pull_and_distribute(self):
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
            PullAndDistributeScheduler()  # type: ignore[call-arg]

    def test_default_interval_is_24_hours(self):
        # Contract source §3.5 says monitoring alerts on call-gap > 7 days.
        # Default cadence well below that, with operator headroom.
        client = _FakeDistributorClient()
        scheduler = PullAndDistributeScheduler(client=client)
        assert scheduler.interval_seconds == 86400.0

    def test_custom_interval(self):
        client = _FakeDistributorClient()
        scheduler = PullAndDistributeScheduler(
            client=client, interval_seconds=3600.0,
        )
        assert scheduler.interval_seconds == 3600.0

    def test_zero_interval_rejected(self):
        client = _FakeDistributorClient()
        with pytest.raises(ValueError, match="interval"):
            PullAndDistributeScheduler(client=client, interval_seconds=0)

    def test_interval_above_seven_days_rejected(self):
        # Hard-fail above the contract's monitoring threshold so an
        # operator misconfiguration cannot silently drift the daemon
        # into a state where it would itself trigger the call-gap > 7
        # days alert it is supposed to prevent.
        client = _FakeDistributorClient()
        with pytest.raises(ValueError, match="7 days"):
            PullAndDistributeScheduler(
                client=client, interval_seconds=86400 * 8,
            )


# ──────────────────────────────────────────────────────────────────────
# Single tick
# ──────────────────────────────────────────────────────────────────────


class TestSingleTick:
    @pytest.mark.asyncio
    async def test_happy_path_calls_pull_and_distribute(self):
        client = _FakeDistributorClient()
        scheduler = PullAndDistributeScheduler(client=client)
        await scheduler.tick()
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_happy_path_increments_success_counter(self):
        client = _FakeDistributorClient()
        scheduler = PullAndDistributeScheduler(client=client)
        assert scheduler.success_count == 0
        await scheduler.tick()
        assert scheduler.success_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_failure_swallowed(self):
        client = _FakeDistributorClient(outcomes=[BroadcastFailedError])
        scheduler = PullAndDistributeScheduler(client=client)
        await scheduler.tick()
        assert scheduler.success_count == 0
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_pending_error_swallowed_and_logged_loud(self, caplog):
        # OnChainPendingError is the most concerning case here:
        # pull_and_distribute is NOT idempotent at the FTNS-flow level
        # (a successful tx mints + distributes; a duplicate tx with
        # zero balance distributes nothing; either way state is
        # consistent, but operator wants to know).
        client = _FakeDistributorClient(
            outcomes=[OnChainPendingError("pending", tx_hash="0xdead")],
        )
        scheduler = PullAndDistributeScheduler(client=client)
        with caplog.at_level(logging.WARNING):
            await scheduler.tick()
        assert any(
            "pending" in r.message.lower() or "unknown" in r.message.lower()
            for r in caplog.records
        )
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_reverted_error_swallowed(self):
        # On-chain revert path: TransferFailed when the contract's FTNS
        # transfer to a pool fails. Daemon stays alive.
        client = _FakeDistributorClient(outcomes=[OnChainRevertedError])
        scheduler = PullAndDistributeScheduler(client=client)
        await scheduler.tick()
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_unexpected_exception_swallowed(self):
        client = _FakeDistributorClient(outcomes=[RuntimeError("weird")])
        scheduler = PullAndDistributeScheduler(client=client)
        await scheduler.tick()
        assert scheduler.failure_count == 1

    @pytest.mark.asyncio
    async def test_success_callback_fires(self):
        client = _FakeDistributorClient()
        events = []

        async def cb(tx_hash):
            events.append(tx_hash)

        scheduler = PullAndDistributeScheduler(
            client=client, on_success=cb,
        )
        await scheduler.tick()
        assert len(events) == 1
        assert events[0].startswith("0x")

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash_daemon(self):
        client = _FakeDistributorClient()

        def cb(tx_hash):
            raise RuntimeError("callback exploded")

        scheduler = PullAndDistributeScheduler(
            client=client, on_success=cb,
        )
        await scheduler.tick()
        assert scheduler.success_count == 1


# ──────────────────────────────────────────────────────────────────────
# Run loop (timing-based tests use direct multi-tick — same property
# pattern as HeartbeatScheduler, avoids pytest-asyncio fixture
# overhead flakiness)
# ──────────────────────────────────────────────────────────────────────


class TestRunForever:
    @pytest.mark.asyncio
    async def test_run_forever_exits_when_stop_called(self):
        client = _FakeDistributorClient()
        scheduler = PullAndDistributeScheduler(
            client=client, interval_seconds=0.05,
        )

        task = asyncio.create_task(scheduler.run_forever())
        await asyncio.sleep(0.15)
        await scheduler.stop()
        await asyncio.wait_for(task, timeout=1.0)

        assert len(client.calls) >= 1

    @pytest.mark.asyncio
    async def test_multiple_direct_ticks_increment_counters(self):
        client = _FakeDistributorClient()
        scheduler = PullAndDistributeScheduler(
            client=client, interval_seconds=86400.0,
        )
        for _ in range(5):
            await scheduler.tick()
        assert scheduler.success_count == 5
        assert scheduler.failure_count == 0
        assert len(client.calls) == 5

    @pytest.mark.asyncio
    async def test_multiple_direct_ticks_keep_running_through_failures(self):
        client = _FakeDistributorClient(outcomes=[
            BroadcastFailedError,
            ("0x" + "01" * 32, TransferStatus.CONFIRMED),
            OnChainRevertedError,
            ("0x" + "02" * 32, TransferStatus.CONFIRMED),
            OnChainPendingError("pending", tx_hash="0xfe"),
            ("0x" + "03" * 32, TransferStatus.CONFIRMED),
        ])
        scheduler = PullAndDistributeScheduler(
            client=client, interval_seconds=86400.0,
        )
        for _ in range(6):
            await scheduler.tick()
        assert scheduler.success_count == 3
        assert scheduler.failure_count == 3
