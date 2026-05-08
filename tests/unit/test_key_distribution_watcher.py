"""KeyDistributionWatcher — async daemon that polls KeyDistribution
contract for KeyReleased / KeyDeposited / KeyDeauthorized events
and fires user-supplied callbacks.

Closes the operationally-meaningful half of annex §5.4
(KeyDistribution release-without-payment P0 detection scenario):
without a watcher, the only way to detect KeyReleased events is
manual Basescan polling — too slow for a P0 surface where Tier C
trust depends on payment-verification correctness. This watcher
surfaces events to operator code in seconds.

Mirrors prsm/emission/watcher.py (EmissionWatcher) shape.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

import pytest

from prsm.economy.web3.key_distribution import (
    KeyReleasedEvent,
)
from prsm.economy.web3.key_distribution_watcher import (
    KeyDeauthorizedEvent,
    KeyDepositedEvent,
    KeyDistributionWatcher,
)


# ──────────────────────────────────────────────────────────────────────
# Stub KeyDistributionClient (with extended event-stream methods)
# ──────────────────────────────────────────────────────────────────────


class _FakeClient:
    """Stub mimicking KeyDistributionClient + the new event-stream
    surface. Tests inject events to be served on the next get_*_events
    call within the requested block range.
    """

    def __init__(self, *, latest_block: int = 100):
        self._latest_block = latest_block
        self._released_events: List[KeyReleasedEvent] = []
        self._deposited_events: List[KeyDepositedEvent] = []
        self._deauthorized_events: List[KeyDeauthorizedEvent] = []
        self.released_calls = []
        self.deposited_calls = []
        self.deauthorized_calls = []

    def latest_block(self) -> int:
        return self._latest_block

    def advance_to(self, block: int):
        self._latest_block = block

    def queue_released(self, event: KeyReleasedEvent):
        self._released_events.append(event)

    def queue_deposited(self, event: KeyDepositedEvent):
        self._deposited_events.append(event)

    def queue_deauthorized(self, event: KeyDeauthorizedEvent):
        self._deauthorized_events.append(event)

    def get_key_released_events(self, from_block, to_block):
        self.released_calls.append((from_block, to_block))
        events = self._released_events
        self._released_events = []
        return events

    def get_key_deposited_events(self, from_block, to_block):
        self.deposited_calls.append((from_block, to_block))
        events = self._deposited_events
        self._deposited_events = []
        return events

    def get_key_deauthorized_events(self, from_block, to_block):
        self.deauthorized_calls.append((from_block, to_block))
        events = self._deauthorized_events
        self._deauthorized_events = []
        return events


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_requires_client(self):
        with pytest.raises(TypeError):
            KeyDistributionWatcher()  # type: ignore[call-arg]

    def test_default_poll_interval_is_positive(self):
        watcher = KeyDistributionWatcher(client=_FakeClient())
        assert watcher.poll_interval_sec > 0

    def test_zero_poll_interval_rejected(self):
        with pytest.raises(ValueError, match="poll"):
            KeyDistributionWatcher(client=_FakeClient(), poll_interval_sec=0)


# ──────────────────────────────────────────────────────────────────────
# Single tick
# ──────────────────────────────────────────────────────────────────────


class TestSingleTick:
    @pytest.mark.asyncio
    async def test_first_tick_starts_at_tip_no_history_replay(self):
        # On first tick, should NOT poll [0, latest] — that would replay
        # months of history. Instead, mark current tip as the baseline
        # and start watching forward from there.
        client = _FakeClient(latest_block=1000)
        client.queue_released(KeyReleasedEvent(
            content_hash=b"\xaa" * 32,
            recipient="0x" + "11" * 20,
            encrypted_key=b"ct1",
        ))
        events_seen = []

        async def cb(ev):
            events_seen.append(ev)

        watcher = KeyDistributionWatcher(client=client, on_key_released=cb)
        await watcher.tick()

        # First tick was a no-op for events; next tick polls forward.
        assert len(events_seen) == 0
        assert watcher.last_processed_block == 1000

    @pytest.mark.asyncio
    async def test_subsequent_tick_fires_callback_for_new_events(self):
        client = _FakeClient(latest_block=1000)
        events_seen = []

        async def cb(ev):
            events_seen.append(ev)

        watcher = KeyDistributionWatcher(client=client, on_key_released=cb)
        await watcher.tick()  # baseline at 1000

        # Now advance and queue an event.
        client.advance_to(1010)
        new_event = KeyReleasedEvent(
            content_hash=b"\xbb" * 32,
            recipient="0x" + "22" * 20,
            encrypted_key=b"ct2",
        )
        client.queue_released(new_event)

        await watcher.tick()
        assert len(events_seen) == 1
        assert events_seen[0].content_hash == b"\xbb" * 32
        # Polling range was [1001, 1010] — strict from_block monotonicity.
        assert client.released_calls[-1] == (1001, 1010)
        assert watcher.last_processed_block == 1010

    @pytest.mark.asyncio
    async def test_empty_block_range_no_op(self):
        # If latest_block hasn't advanced, no get_logs call needed.
        client = _FakeClient(latest_block=500)
        watcher = KeyDistributionWatcher(client=client)
        await watcher.tick()
        assert watcher.last_processed_block == 500

        # Tick again with no advance.
        await watcher.tick()
        # Per-event get methods may or may not be called depending on
        # implementation; what matters is no extra forward progress
        # and no error.
        assert watcher.last_processed_block == 500

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash_watcher(self):
        client = _FakeClient(latest_block=100)

        async def bad_cb(ev):
            raise RuntimeError("callback exploded")

        watcher = KeyDistributionWatcher(
            client=client, on_key_released=bad_cb,
        )
        await watcher.tick()  # baseline
        client.advance_to(105)
        client.queue_released(KeyReleasedEvent(
            content_hash=b"\x11" * 32,
            recipient="0x" + "11" * 20,
            encrypted_key=b"ct",
        ))
        # Must NOT raise — daemon stays alive across user-callback bugs.
        await watcher.tick()
        assert watcher.last_processed_block == 105

    @pytest.mark.asyncio
    async def test_rpc_failure_swallowed_no_progress(self):
        # If get_*_events raises (for a SUBSCRIBED event type), watcher
        # should log + retry next tick WITHOUT advancing
        # last_processed_block (so no events are silently lost on a
        # transient RPC failure).
        client = _FakeClient(latest_block=100)

        async def cb(ev):
            pass

        watcher = KeyDistributionWatcher(
            client=client, on_key_released=cb,
        )
        await watcher.tick()  # baseline at 100

        client.advance_to(110)
        # Stub raise.
        def boom(*args, **kwargs):
            raise RuntimeError("rpc unreachable")
        client.get_key_released_events = boom

        # Must not raise; last_processed_block must NOT advance to 110.
        await watcher.tick()
        assert watcher.last_processed_block == 100


# ──────────────────────────────────────────────────────────────────────
# Multi-event-type fanout
# ──────────────────────────────────────────────────────────────────────


class TestMultiEventFanout:
    @pytest.mark.asyncio
    async def test_three_event_types_fire_separate_callbacks(self):
        client = _FakeClient(latest_block=100)
        released_seen = []
        deposited_seen = []
        deauthorized_seen = []

        async def on_released(ev):
            released_seen.append(ev)

        async def on_deposited(ev):
            deposited_seen.append(ev)

        async def on_deauthorized(ev):
            deauthorized_seen.append(ev)

        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=on_released,
            on_key_deposited=on_deposited,
            on_key_deauthorized=on_deauthorized,
        )
        await watcher.tick()  # baseline

        client.advance_to(105)
        client.queue_released(KeyReleasedEvent(
            content_hash=b"\x01" * 32, recipient="0x" + "01" * 20,
            encrypted_key=b"ct1",
        ))
        client.queue_deposited(KeyDepositedEvent(
            content_hash=b"\x02" * 32, publisher="0x" + "02" * 20,
            royalty="0x" + "03" * 20, release_fee_ftns_wei=10**18,
        ))
        client.queue_deauthorized(KeyDeauthorizedEvent(
            content_hash=b"\x03" * 32, publisher="0x" + "04" * 20,
        ))

        await watcher.tick()
        assert len(released_seen) == 1
        assert len(deposited_seen) == 1
        assert len(deauthorized_seen) == 1

    @pytest.mark.asyncio
    async def test_no_callback_means_no_polling_for_that_event(self):
        # If only on_key_released is set, watcher should NOT poll for
        # the other two event types — saves RPC bandwidth.
        client = _FakeClient(latest_block=100)

        async def cb(ev):
            pass

        watcher = KeyDistributionWatcher(
            client=client, on_key_released=cb,
        )
        await watcher.tick()  # baseline

        client.advance_to(110)
        await watcher.tick()
        # released was polled.
        assert len(client.released_calls) >= 1
        # deposited / deauthorized were NOT polled.
        assert len(client.deposited_calls) == 0
        assert len(client.deauthorized_calls) == 0


# ──────────────────────────────────────────────────────────────────────
# Run loop (direct multi-tick — avoids pytest-asyncio timing flakiness)
# ──────────────────────────────────────────────────────────────────────


class TestRunForever:
    @pytest.mark.asyncio
    async def test_run_forever_exits_on_stop(self):
        client = _FakeClient(latest_block=100)
        watcher = KeyDistributionWatcher(
            client=client, poll_interval_sec=0.05,
        )
        task = asyncio.create_task(watcher.run_forever())
        await asyncio.sleep(0.15)
        await watcher.stop()
        await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_multiple_direct_ticks_advance_baseline(self):
        client = _FakeClient(latest_block=100)
        watcher = KeyDistributionWatcher(
            client=client, poll_interval_sec=60.0,
        )
        await watcher.tick()
        assert watcher.last_processed_block == 100
        client.advance_to(200)
        await watcher.tick()
        assert watcher.last_processed_block == 200
        client.advance_to(300)
        await watcher.tick()
        assert watcher.last_processed_block == 300


# ──────────────────────────────────────────────────────────────────────
# Event dataclasses
# ──────────────────────────────────────────────────────────────────────


class TestKeyDepositedEvent:
    def test_from_decoded_args_happy_path(self):
        event = KeyDepositedEvent.from_decoded_args({
            "contentHash": b"\xaa" * 32,
            "publisher": "0x" + "11" * 20,
            "royalty": "0x" + "22" * 20,
            "releaseFeeFtnsWei": 10**18,
        })
        assert event.content_hash == b"\xaa" * 32
        assert event.publisher == "0x" + "11" * 20
        assert event.royalty == "0x" + "22" * 20
        assert event.release_fee_ftns_wei == 10**18

    def test_validates_content_hash_length(self):
        with pytest.raises(ValueError, match="32 bytes"):
            KeyDepositedEvent(
                content_hash=b"\x00" * 16,
                publisher="0x" + "11" * 20,
                royalty="0x" + "22" * 20,
                release_fee_ftns_wei=10**18,
            )


class TestKeyDeauthorizedEvent:
    def test_from_decoded_args_happy_path(self):
        event = KeyDeauthorizedEvent.from_decoded_args({
            "contentHash": b"\xaa" * 32,
            "publisher": "0x" + "11" * 20,
        })
        assert event.content_hash == b"\xaa" * 32
        assert event.publisher == "0x" + "11" * 20
