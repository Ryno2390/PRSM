"""CompensationDistributorWatcher — async daemon polling
CompensationDistributor contract for Distributed events.

Operationally smaller surface than the other two watchers — only
one event class is operationally meaningful (admin-triggered
WeightsScheduled / WeightsActivated / PoolAddressesUpdated events
are visible on Basescan and don't drive operator-side automation).
"""
from __future__ import annotations

import asyncio
from typing import List

import pytest

from prsm.economy.web3.compensation_distributor import DistributedEvent
from prsm.economy.web3.compensation_distributor_watcher import (
    CompensationDistributorWatcher,
)


class _FakeClient:
    def __init__(self, *, latest_block: int = 100):
        self._latest_block = latest_block
        self._distributed: List[DistributedEvent] = []
        self.distributed_calls = []

    def latest_block(self) -> int:
        return self._latest_block

    def advance_to(self, block):
        self._latest_block = block

    def queue_distributed(self, ev):
        self._distributed.append(ev)

    def get_distributed_events(self, from_block, to_block):
        self.distributed_calls.append((from_block, to_block))
        events = self._distributed
        self._distributed = []
        return events


class TestConstruction:
    def test_requires_client(self):
        with pytest.raises(TypeError):
            CompensationDistributorWatcher()  # type: ignore[call-arg]

    def test_zero_poll_interval_rejected(self):
        with pytest.raises(ValueError, match="poll"):
            CompensationDistributorWatcher(
                client=_FakeClient(), poll_interval_sec=0,
            )


class TestSingleTick:
    @pytest.mark.asyncio
    async def test_first_tick_baseline(self):
        client = _FakeClient(latest_block=200)
        client.queue_distributed(DistributedEvent(
            to_creator=10**17, to_operator=10**17, to_grant=10**17,
        ))
        events = []

        async def cb(ev):
            events.append(ev)

        watcher = CompensationDistributorWatcher(
            client=client, on_distributed=cb,
        )
        await watcher.tick()
        assert len(events) == 0
        assert watcher.last_processed_block == 200

    @pytest.mark.asyncio
    async def test_subsequent_tick_fires_callback(self):
        client = _FakeClient(latest_block=100)
        events = []

        async def cb(ev):
            events.append(ev)

        watcher = CompensationDistributorWatcher(
            client=client, on_distributed=cb,
        )
        await watcher.tick()
        client.advance_to(150)
        client.queue_distributed(DistributedEvent(
            to_creator=4500 * 10**14,
            to_operator=3500 * 10**14,
            to_grant=2000 * 10**14,
        ))
        await watcher.tick()
        assert len(events) == 1
        assert events[0].to_creator == 4500 * 10**14
        assert client.distributed_calls[-1] == (101, 150)

    @pytest.mark.asyncio
    async def test_no_callback_no_polling(self):
        client = _FakeClient(latest_block=100)
        watcher = CompensationDistributorWatcher(client=client)
        await watcher.tick()
        client.advance_to(150)
        await watcher.tick()
        assert len(client.distributed_calls) == 0

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash(self):
        client = _FakeClient(latest_block=100)

        async def bad_cb(ev):
            raise RuntimeError("boom")

        watcher = CompensationDistributorWatcher(
            client=client, on_distributed=bad_cb,
        )
        await watcher.tick()
        client.advance_to(105)
        client.queue_distributed(DistributedEvent(
            to_creator=10**17, to_operator=10**17, to_grant=10**17,
        ))
        await watcher.tick()
        assert watcher.last_processed_block == 105

    @pytest.mark.asyncio
    async def test_rpc_failure_no_progress(self):
        client = _FakeClient(latest_block=100)

        async def cb(ev):
            pass

        watcher = CompensationDistributorWatcher(
            client=client, on_distributed=cb,
        )
        await watcher.tick()
        client.advance_to(110)

        def boom(*a, **k):
            raise RuntimeError("rpc")

        client.get_distributed_events = boom
        await watcher.tick()
        assert watcher.last_processed_block == 100


class TestRunForever:
    @pytest.mark.asyncio
    async def test_run_forever_exits_on_stop(self):
        client = _FakeClient(latest_block=100)
        watcher = CompensationDistributorWatcher(
            client=client, poll_interval_sec=0.05,
        )
        task = asyncio.create_task(watcher.run_forever())
        await asyncio.sleep(0.15)
        await watcher.stop()
        await asyncio.wait_for(task, timeout=1.0)
