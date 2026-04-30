"""
Tests for the EventBus and WhiteboardPushHandler.

These tests verify the event-driven bridge between BSC pipeline and
LiveScribe (Sub-phase 10.2).
"""

import asyncio
import logging
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prsm.compute.nwtn.bsc.event_bus import BSCEvent, EventBus, EventType
from prsm.compute.nwtn.bsc.whiteboard_push import WhiteboardPushHandler, PushStats


# ======================================================================
# EventBus Tests
# ======================================================================

class TestEventBus:
    """Test EventBus publish/subscribe mechanics."""

    def test_init(self) -> None:
        """EventBus starts with zero subscribers."""
        bus = EventBus()
        for et in EventType:
            assert bus.subscriber_count(et) == 0

    async def test_subscribe_unsubscribe(self) -> None:
        """Subscribers can be added and removed."""
        bus = EventBus()
        callback1 = AsyncMock()
        callback2 = AsyncMock()

        # Subscribe
        await bus.subscribe(EventType.CHUNK_PROMOTED, callback1)
        assert bus.subscriber_count(EventType.CHUNK_PROMOTED) == 1

        await bus.subscribe(EventType.CHUNK_PROMOTED, callback2)
        assert bus.subscriber_count(EventType.CHUNK_PROMOTED) == 2

        # Unsubscribe
        await bus.unsubscribe(EventType.CHUNK_PROMOTED, callback1)
        assert bus.subscriber_count(EventType.CHUNK_PROMOTED) == 1

        # Unsubscribe non-existent is safe
        await bus.unsubscribe(EventType.CHUNK_REJECTED, callback1)  # never subscribed
        assert bus.subscriber_count(EventType.CHUNK_REJECTED) == 0

    async def test_publish_calls_all_subscribers(self) -> None:
        """Publishing calls all subscribers concurrently."""
        bus = EventBus()
        call_order = []
        async def cb1(event: BSCEvent) -> None:
            await asyncio.sleep(0.01)
            call_order.append(1)
        async def cb2(event: BSCEvent) -> None:
            call_order.append(2)

        await bus.subscribe(EventType.ROUND_ADVANCED, cb1)
        await bus.subscribe(EventType.ROUND_ADVANCED, cb2)

        event = BSCEvent(
            event_type=EventType.ROUND_ADVANCED,
            data={"round": 5},
            session_id="sess-1",
        )
        await bus.publish(event)

        # Both should have been called (order not guaranteed)
        assert set(call_order) == {1, 2}

    async def test_publish_fire_and_forget(self) -> None:
        """Subscriber exceptions are caught and counted."""
        bus = EventBus()
        good_cb = AsyncMock()
        bad_cb = AsyncMock(side_effect=RuntimeError("Boom"))

        await bus.subscribe(EventType.CHUNK_REJECTED, good_cb)
        await bus.subscribe(EventType.CHUNK_REJECTED, bad_cb)

        event = BSCEvent(
            event_type=EventType.CHUNK_REJECTED,
            data={"reason": "test"},
            session_id="sess-1",
        )
        await bus.publish(event)

        # Good callback was called
        good_cb.assert_called_once_with(event)
        # Bad callback raised but didn't stop the bus
        bad_cb.assert_called_once_with(event)

        stats = bus.get_stats()
        assert stats["published"] == 1
        assert stats["delivery_errors"] == 1

    async def test_event_isolation(self) -> None:
        """Subscribers only receive events of their subscribed type."""
        bus = EventBus()
        promoted_cb = AsyncMock()
        rejected_cb = AsyncMock()

        await bus.subscribe(EventType.CHUNK_PROMOTED, promoted_cb)
        await bus.subscribe(EventType.CHUNK_REJECTED, rejected_cb)

        promoted_event = BSCEvent(
            event_type=EventType.CHUNK_PROMOTED,
            data={"decision": {"promoted": True}},
            session_id="sess-1",
        )
        rejected_event = BSCEvent(
            event_type=EventType.CHUNK_REJECTED,
            data={"reason": "test"},
            session_id="sess-1",
        )

        await bus.publish(promoted_event)
        await bus.publish(rejected_event)

        promoted_cb.assert_called_once_with(promoted_event)
        rejected_cb.assert_called_once_with(rejected_event)

    async def test_clear_subscribers(self) -> None:
        """clear() removes all subscribers (optionally per event type)."""
        bus = EventBus()
        cb1 = AsyncMock()
        cb2 = AsyncMock()

        await bus.subscribe(EventType.CHUNK_PROMOTED, cb1)
        await bus.subscribe(EventType.CHUNK_REJECTED, cb2)

        assert bus.subscriber_count() == 2
        await bus.clear(EventType.CHUNK_PROMOTED)
        assert bus.subscriber_count(EventType.CHUNK_PROMOTED) == 0
        assert bus.subscriber_count(EventType.CHUNK_REJECTED) == 1

        await bus.clear()  # clear all
        assert bus.subscriber_count() == 0

    def test_event_data_immutable(self) -> None:
        """BSCEvent fields are frozen (frozen dataclass)."""
        event = BSCEvent(
            event_type=EventType.CHUNK_PROMOTED,
            data={"key": "value"},
            session_id="sess-1",
        )
        with pytest.raises(FrozenInstanceError):
            event.event_type = EventType.CHUNK_REJECTED  # type: ignore


# ======================================================================
# WhiteboardPushHandler Tests
# ======================================================================

class TestWhiteboardPushHandler:
    """Test WhiteboardPushHandler (BSC → LiveScribe bridge)."""

    async def test_start_stop_subscribes(self) -> None:
        """start() subscribes to CHUNK_PROMOTED, stop() unsubscribes."""
        mock_bus = AsyncMock(spec=EventBus)
        mock_scribe = MagicMock()

        handler = WhiteboardPushHandler(event_bus=mock_bus, live_scribe=mock_scribe)
        assert not handler.running

        await handler.start()
        assert handler.running
        mock_bus.subscribe.assert_called_once_with(
            EventType.CHUNK_PROMOTED,
            handler._on_promoted,
        )

        await handler.stop()
        assert not handler.running
        mock_bus.unsubscribe.assert_called_once_with(
            EventType.CHUNK_PROMOTED,
            handler._on_promoted,
        )

    async def test_on_promoted_calls_scribe(self) -> None:
        """CHUNK_PROMOTED event calls scribe.on_chunk_promoted()."""
        mock_bus = AsyncMock()
        mock_scribe = AsyncMock()
        handler = WhiteboardPushHandler(event_bus=mock_bus, live_scribe=mock_scribe)

        # Simulate a PromotionDecision (simplified mock)
        mock_decision = MagicMock()
        mock_decision.metadata.source_agent = "agent/coder"
        mock_decision.surprise_score = 0.75

        event = BSCEvent(
            event_type=EventType.CHUNK_PROMOTED,
            data={"decision": mock_decision},
            session_id="sess-1",
        )

        await handler._on_promoted(event)

        mock_scribe.on_chunk_promoted.assert_called_once_with(mock_decision)
        stats = handler.get_stats()
        assert stats["pushed"] == 1
        assert stats["failed"] == 0

    async def test_on_promoted_missing_decision_skips(self) -> None:
        """Event without decision data increments skipped counter."""
        mock_bus = AsyncMock()
        mock_scribe = AsyncMock()
        handler = WhiteboardPushHandler(event_bus=mock_bus, live_scribe=mock_scribe)

        event = BSCEvent(
            event_type=EventType.CHUNK_PROMOTED,
            data={},  # no 'decision' key
            session_id="sess-1",
        )

        await handler._on_promoted(event)

        mock_scribe.on_chunk_promoted.assert_not_called()
        stats = handler.get_stats()
        assert stats["skipped"] == 1
        assert stats["pushed"] == 0

    async def test_on_promoted_scribe_raises_counts_failed(self) -> None:
        """Scribe exception increments failed counter, doesn't propagate."""
        mock_bus = AsyncMock()
        mock_scribe = AsyncMock()
        mock_scribe.on_chunk_promoted.side_effect = RuntimeError("DB down")
        handler = WhiteboardPushHandler(event_bus=mock_bus, live_scribe=mock_scribe)

        mock_decision = MagicMock()
        mock_decision.metadata.source_agent = "agent/coder"

        event = BSCEvent(
            event_type=EventType.CHUNK_PROMOTED,
            data={"decision": mock_decision},
            session_id="sess-1",
        )

        # Should not raise
        await handler._on_promoted(event)

        mock_scribe.on_chunk_promoted.assert_called_once_with(mock_decision)
        stats = handler.get_stats()
        assert stats["failed"] == 1
        assert stats["pushed"] == 0

    async def test_stats_reset(self) -> None:
        """reset_stats() zeroes all counters."""
        mock_bus = AsyncMock()
        mock_scribe = AsyncMock()
        handler = WhiteboardPushHandler(event_bus=mock_bus, live_scribe=mock_scribe)

        handler._stats.pushed = 5
        handler._stats.failed = 2
        handler._stats.conflicts = 1
        handler._stats.skipped = 3

        handler.reset_stats()
        stats = handler.get_stats()
        assert stats == {"pushed": 0, "failed": 0, "conflicts": 0, "skipped": 0}

    def test_conflict_detection(self) -> None:
        """conflict_detected attribute increments conflict counter."""
        stats = PushStats()
        assert stats.conflicts == 0
        stats.conflicts += 1
        assert stats.to_dict()["conflicts"] == 1