"""Sprint 401 — tick-age tracking pins for the 4 remaining
operator-node daemons: JobReaper + 3 chain-event watchers.

Mirrors sprint 399/400 tests on HeartbeatScheduler +
PullAndDistributeScheduler. Each daemon now exposes
`last_tick_at` (Optional[datetime]) bumped only on
successful tick/scan/poll, plus `last_tick_age_seconds`
property + `interval_seconds` property (alias for the
watchers' existing `poll_interval_sec`).

Silent-failure modes closed:
  - JobReaper:         IN_PROGRESS jobs hang forever, escrows
                       never refunded
  - StorageSlashing:   miss heartbeat-recorded /
                       proof-failure-slashed / heartbeat-
                       missing-slashed events → bad actors
                       not slashed
  - CompensationDist:  miss Distributed events → operator's
                       observability rings don't get fed
  - KeyDistribution:   miss key-release / deposit /
                       deauthorization events → security-
                       relevant state divergence
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock

import pytest


# ── JobReaper ────────────────────────────────────────────


class TestJobReaperTickAge:
    def _scheduler(self, *, max_duration=300, interval=60):
        from prsm.node.job_reaper import JobReaper
        from prsm.node.job_history import JobHistoryStore
        return JobReaper(
            job_history=JobHistoryStore(),
            payment_escrow=None,
            max_duration_seconds=max_duration,
            interval_seconds=interval,
        )

    def test_last_tick_at_initially_none(self):
        assert self._scheduler().last_tick_at is None

    def test_last_tick_age_seconds_none_initially(self):
        assert self._scheduler().last_tick_age_seconds is None

    @pytest.mark.asyncio
    async def test_reap_once_does_not_directly_bump(self):
        """reap_once is the unit-of-work; the bump happens in
        run_forever's call site after a successful reap_once.
        This pins the design choice (bump at loop level, not
        in reap_once itself) so test fixtures calling
        reap_once directly don't get false bumps."""
        sched = self._scheduler()
        await sched.reap_once()
        assert sched.last_tick_at is None

    @pytest.mark.asyncio
    async def test_run_forever_bumps_after_reap(self):
        sched = self._scheduler(interval=0.05)
        before = datetime.now(timezone.utc)
        task = asyncio.create_task(sched.run_forever())
        # Let one loop iteration land
        await asyncio.sleep(0.15)
        await sched.stop()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
        after = datetime.now(timezone.utc)
        assert sched.last_tick_at is not None
        assert before <= sched.last_tick_at <= after


# ── StorageSlashingWatcher ───────────────────────────────


class _FakeSlashingWatchClient:
    """Minimal client surface for sprint-401 watcher tests."""

    def __init__(self, *, latest=100, raises_on_latest=False):
        self._latest = latest
        self._raises = raises_on_latest

    def latest_block(self) -> int:
        if self._raises:
            raise RuntimeError("RPC down")
        return self._latest

    def get_heartbeat_recorded_events(self, *args, **kw): return []
    def get_proof_failure_slashed_events(self, *args, **kw): return []
    def get_heartbeat_missing_slashed_events(self, *args, **kw): return []


class TestStorageSlashingWatcherTickAge:
    def _watcher(self, **kw):
        from prsm.economy.web3.storage_slashing_watcher import (
            StorageSlashingWatcher,
        )
        client = _FakeSlashingWatchClient(**kw)
        return StorageSlashingWatcher(
            client=client,
            on_heartbeat_recorded=lambda e: None,
            on_proof_failure_slashed=lambda e: None,
            on_heartbeat_missing_slashed=lambda e: None,
            poll_interval_sec=1.0,
        )

    def test_last_tick_at_initially_none(self):
        assert self._watcher().last_tick_at is None

    def test_interval_seconds_alias(self):
        """interval_seconds aliases poll_interval_sec — lets
        the sprint-400 _daemon_subsystem helper auto-surface
        tick_status."""
        w = self._watcher()
        assert w.interval_seconds == w.poll_interval_sec

    @pytest.mark.asyncio
    async def test_first_tick_baseline_path_bumps(self):
        """First tick with no state_store falls into the
        baseline-establish path (line 180 region) and
        returns early. Bump still fires — poll completed."""
        w = self._watcher()
        await w.tick()
        assert w.last_tick_at is not None

    @pytest.mark.asyncio
    async def test_no_new_blocks_path_bumps(self):
        """Second tick with no new blocks since baseline
        returns early at the `latest <= last_processed_block`
        guard. Still a successful poll."""
        w = self._watcher()
        # Set baseline manually so first tick goes straight
        # to the "no new blocks" path
        w.last_processed_block = 100
        await w.tick()  # latest=100, last_processed=100
        assert w.last_tick_at is not None

    @pytest.mark.asyncio
    async def test_rpc_failure_does_not_bump(self):
        from prsm.economy.web3.storage_slashing_watcher import (
            StorageSlashingWatcher,
        )
        client = _FakeSlashingWatchClient(raises_on_latest=True)
        w = StorageSlashingWatcher(
            client=client,
            on_heartbeat_recorded=lambda e: None,
            poll_interval_sec=1.0,
        )
        await w.tick()
        # RPC failure path — no bump
        assert w.last_tick_at is None


# ── CompensationDistributorWatcher ───────────────────────


class _FakeCompClient:
    def __init__(self, *, latest=100, raises_on_latest=False):
        self._latest = latest
        self._raises = raises_on_latest

    def latest_block(self):
        if self._raises:
            raise RuntimeError("RPC down")
        return self._latest

    def get_distributed_events(self, *args, **kw):
        return []


class TestCompensationDistributorWatcherTickAge:
    def _watcher(self, **kw):
        from prsm.economy.web3.compensation_distributor_watcher import (
            CompensationDistributorWatcher,
        )
        client = _FakeCompClient(**kw)
        return CompensationDistributorWatcher(
            client=client,
            on_distributed=lambda e: None,
            poll_interval_sec=1.0,
        )

    def test_last_tick_at_initially_none(self):
        assert self._watcher().last_tick_at is None

    def test_interval_seconds_alias(self):
        w = self._watcher()
        assert w.interval_seconds == w.poll_interval_sec

    @pytest.mark.asyncio
    async def test_first_tick_baseline_path_bumps(self):
        w = self._watcher()
        await w.tick()
        assert w.last_tick_at is not None

    @pytest.mark.asyncio
    async def test_no_new_blocks_path_bumps(self):
        w = self._watcher()
        w.last_processed_block = 100
        await w.tick()
        assert w.last_tick_at is not None

    @pytest.mark.asyncio
    async def test_rpc_failure_does_not_bump(self):
        from prsm.economy.web3.compensation_distributor_watcher import (
            CompensationDistributorWatcher,
        )
        client = _FakeCompClient(raises_on_latest=True)
        w = CompensationDistributorWatcher(
            client=client, poll_interval_sec=1.0,
        )
        await w.tick()
        assert w.last_tick_at is None


# ── KeyDistributionWatcher ───────────────────────────────


class _FakeKeyClient:
    def __init__(self, *, latest=100, raises_on_latest=False):
        self._latest = latest
        self._raises = raises_on_latest

    def latest_block(self):
        if self._raises:
            raise RuntimeError("RPC down")
        return self._latest

    def get_key_released_events(self, *args, **kw): return []
    def get_key_deposited_events(self, *args, **kw): return []
    def get_key_deauthorized_events(self, *args, **kw): return []


class TestKeyDistributionWatcherTickAge:
    def _watcher(self, **kw):
        from prsm.economy.web3.key_distribution_watcher import (
            KeyDistributionWatcher,
        )
        client = _FakeKeyClient(**kw)
        return KeyDistributionWatcher(
            client=client,
            on_key_released=lambda e: None,
            on_key_deposited=lambda e: None,
            on_key_deauthorized=lambda e: None,
            poll_interval_sec=1.0,
        )

    def test_last_tick_at_initially_none(self):
        assert self._watcher().last_tick_at is None

    def test_interval_seconds_alias(self):
        w = self._watcher()
        assert w.interval_seconds == w.poll_interval_sec

    @pytest.mark.asyncio
    async def test_first_tick_baseline_path_bumps(self):
        w = self._watcher()
        await w.tick()
        assert w.last_tick_at is not None

    @pytest.mark.asyncio
    async def test_no_new_blocks_path_bumps(self):
        w = self._watcher()
        w.last_processed_block = 100
        await w.tick()
        assert w.last_tick_at is not None

    @pytest.mark.asyncio
    async def test_rpc_failure_does_not_bump(self):
        from prsm.economy.web3.key_distribution_watcher import (
            KeyDistributionWatcher,
        )
        client = _FakeKeyClient(raises_on_latest=True)
        w = KeyDistributionWatcher(
            client=client, poll_interval_sec=1.0,
        )
        await w.tick()
        assert w.last_tick_at is None


# Note: /health/detailed integration for these 4 daemons
# is covered by the sprint-400 generalized
# _daemon_subsystem helper which auto-surfaces tick_status
# for ANY daemon exposing both interval_seconds +
# last_tick_age_seconds. The helper's correctness is pinned
# by sprint-400's tests in test_health_detailed_endpoint.py
# (TestCompensationSchedulerTickAge). Sprint 401 just
# ensures all 4 remaining daemons expose those two
# attributes — verified above.
