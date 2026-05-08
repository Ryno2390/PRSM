"""StorageSlashingWatcher — async daemon polling StorageSlashing
contract for HeartbeatMissingSlashed / ProofFailureSlashed /
HeartbeatRecorded events.

Operational use cases:
  - Storage providers monitoring for slashing of their own address
    (real-time alert on permissionless or proof-failure slashes).
  - Foundation council monitoring fleet-wide slashing rate (anomaly
    detection).
  - Dashboards / forensics tooling.

Same loop shape as KeyDistributionWatcher; per-event-type polling
gated on callback subscription to save RPC bandwidth.
"""
from __future__ import annotations

import asyncio
from typing import List

import pytest

from prsm.economy.web3.storage_slashing import (
    HeartbeatMissingSlashedEvent,
    HeartbeatRecordedEvent,
    ProofFailureSlashedEvent,
)
from prsm.economy.web3.storage_slashing_watcher import StorageSlashingWatcher


class _FakeClient:
    def __init__(self, *, latest_block: int = 100):
        self._latest_block = latest_block
        self._heartbeat_recorded: List[HeartbeatRecordedEvent] = []
        self._proof_failure_slashed: List[ProofFailureSlashedEvent] = []
        self._heartbeat_missing_slashed: List[HeartbeatMissingSlashedEvent] = []
        self.recorded_calls = []
        self.proof_calls = []
        self.missing_calls = []

    def latest_block(self) -> int:
        return self._latest_block

    def advance_to(self, block):
        self._latest_block = block

    def queue_heartbeat_recorded(self, ev):
        self._heartbeat_recorded.append(ev)

    def queue_proof_failure(self, ev):
        self._proof_failure_slashed.append(ev)

    def queue_heartbeat_missing(self, ev):
        self._heartbeat_missing_slashed.append(ev)

    def get_heartbeat_recorded_events(self, from_block, to_block):
        self.recorded_calls.append((from_block, to_block))
        events = self._heartbeat_recorded
        self._heartbeat_recorded = []
        return events

    def get_proof_failure_slashed_events(self, from_block, to_block):
        self.proof_calls.append((from_block, to_block))
        events = self._proof_failure_slashed
        self._proof_failure_slashed = []
        return events

    def get_heartbeat_missing_slashed_events(self, from_block, to_block):
        self.missing_calls.append((from_block, to_block))
        events = self._heartbeat_missing_slashed
        self._heartbeat_missing_slashed = []
        return events


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_requires_client(self):
        with pytest.raises(TypeError):
            StorageSlashingWatcher()  # type: ignore[call-arg]

    def test_zero_poll_interval_rejected(self):
        with pytest.raises(ValueError, match="poll"):
            StorageSlashingWatcher(client=_FakeClient(), poll_interval_sec=0)


# ──────────────────────────────────────────────────────────────────────
# Single tick
# ──────────────────────────────────────────────────────────────────────


class TestSingleTick:
    @pytest.mark.asyncio
    async def test_first_tick_baseline_no_history_replay(self):
        client = _FakeClient(latest_block=500)
        client.queue_heartbeat_missing(HeartbeatMissingSlashedEvent(
            provider="0x" + "11" * 20, challenger="0x" + "22" * 20,
            last_heartbeat_at=1700000000, slash_id=b"\x55" * 32,
        ))
        events = []

        async def cb(ev):
            events.append(ev)

        watcher = StorageSlashingWatcher(
            client=client, on_heartbeat_missing_slashed=cb,
        )
        await watcher.tick()
        assert len(events) == 0
        assert watcher.last_processed_block == 500

    @pytest.mark.asyncio
    async def test_subsequent_tick_fires_callback_for_proof_failure(self):
        client = _FakeClient(latest_block=100)
        events = []

        async def cb(ev):
            events.append(ev)

        watcher = StorageSlashingWatcher(
            client=client, on_proof_failure_slashed=cb,
        )
        await watcher.tick()  # baseline
        client.advance_to(110)
        client.queue_proof_failure(ProofFailureSlashedEvent(
            provider="0x" + "11" * 20, challenger="0x" + "22" * 20,
            shard_id=b"\x33" * 32, evidence_hash=b"\x44" * 32,
            slash_id=b"\x55" * 32,
        ))
        await watcher.tick()
        assert len(events) == 1
        assert events[0].provider == "0x" + "11" * 20
        assert client.proof_calls[-1] == (101, 110)

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash_watcher(self):
        client = _FakeClient(latest_block=100)

        async def bad_cb(ev):
            raise RuntimeError("boom")

        watcher = StorageSlashingWatcher(
            client=client, on_heartbeat_missing_slashed=bad_cb,
        )
        await watcher.tick()
        client.advance_to(105)
        client.queue_heartbeat_missing(HeartbeatMissingSlashedEvent(
            provider="0x" + "11" * 20, challenger="0x" + "22" * 20,
            last_heartbeat_at=100, slash_id=b"\x55" * 32,
        ))
        await watcher.tick()
        assert watcher.last_processed_block == 105

    @pytest.mark.asyncio
    async def test_rpc_failure_no_progress(self):
        client = _FakeClient(latest_block=100)

        async def cb(ev):
            pass

        watcher = StorageSlashingWatcher(
            client=client, on_proof_failure_slashed=cb,
        )
        await watcher.tick()  # baseline 100
        client.advance_to(110)

        def boom(*a, **k):
            raise RuntimeError("rpc")

        client.get_proof_failure_slashed_events = boom
        await watcher.tick()
        assert watcher.last_processed_block == 100

    @pytest.mark.asyncio
    async def test_no_callback_means_no_polling(self):
        client = _FakeClient(latest_block=100)

        async def cb(ev):
            pass

        watcher = StorageSlashingWatcher(
            client=client, on_proof_failure_slashed=cb,
        )
        await watcher.tick()
        client.advance_to(105)
        await watcher.tick()
        assert len(client.proof_calls) >= 1
        assert len(client.recorded_calls) == 0
        assert len(client.missing_calls) == 0


# ──────────────────────────────────────────────────────────────────────
# Multi-event fanout
# ──────────────────────────────────────────────────────────────────────


class TestMultiEventFanout:
    @pytest.mark.asyncio
    async def test_three_event_types_independent(self):
        client = _FakeClient(latest_block=100)
        a, b, c = [], [], []

        async def on_recorded(ev):
            a.append(ev)

        async def on_proof(ev):
            b.append(ev)

        async def on_missing(ev):
            c.append(ev)

        watcher = StorageSlashingWatcher(
            client=client,
            on_heartbeat_recorded=on_recorded,
            on_proof_failure_slashed=on_proof,
            on_heartbeat_missing_slashed=on_missing,
        )
        await watcher.tick()
        client.advance_to(105)

        client.queue_heartbeat_recorded(HeartbeatRecordedEvent(
            provider="0x" + "11" * 20, timestamp=1700000000,
        ))
        client.queue_proof_failure(ProofFailureSlashedEvent(
            provider="0x" + "11" * 20, challenger="0x" + "22" * 20,
            shard_id=b"\x33" * 32, evidence_hash=b"\x44" * 32,
            slash_id=b"\x55" * 32,
        ))
        client.queue_heartbeat_missing(HeartbeatMissingSlashedEvent(
            provider="0x" + "11" * 20, challenger="0x" + "22" * 20,
            last_heartbeat_at=1700000000, slash_id=b"\x66" * 32,
        ))

        await watcher.tick()
        assert len(a) == 1
        assert len(b) == 1
        assert len(c) == 1


# ──────────────────────────────────────────────────────────────────────
# Run loop
# ──────────────────────────────────────────────────────────────────────


class TestRunForever:
    @pytest.mark.asyncio
    async def test_run_forever_exits_on_stop(self):
        client = _FakeClient(latest_block=100)
        watcher = StorageSlashingWatcher(
            client=client, poll_interval_sec=0.05,
        )
        task = asyncio.create_task(watcher.run_forever())
        await asyncio.sleep(0.15)
        await watcher.stop()
        await asyncio.wait_for(task, timeout=1.0)
