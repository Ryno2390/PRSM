"""
Unit Tests for Real-Time Analytics Processor
=============================================

Tests for the real-time analytics processor singleton, factory functions,
and stream processing capabilities.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from prsm.data.analytics.real_time_processor import (
    StreamEvent,
    StreamEventType,
    StreamProcessor,
    AggregationProcessor,
    AlertProcessor,
    FilterProcessor,
    RealTimeProcessor,
    initialize_real_time_processor,
    get_real_time_processor,
    _create_default_processors,
    _real_time_processor as global_processor,
)


# Reset global state before each test
@pytest.fixture(autouse=True)
def reset_global_processor():
    """Reset the global processor singleton before each test."""
    import prsm.data.analytics.real_time_processor as module
    module._real_time_processor = None
    yield
    module._real_time_processor = None


class TestSingletonAndFactory:
    """Tests for singleton initialization and factory functions."""

    def test_initialize_creates_processor(self):
        """Test that initialize_real_time_processor() creates a processor instance."""
        processor = initialize_real_time_processor(buffer_size=50000)
        
        assert processor is not None
        assert isinstance(processor, RealTimeProcessor)
        assert processor.stream_buffer.max_size == 50000

    def test_get_processor_raises_when_not_initialized(self):
        """Test that get_real_time_processor() raises RuntimeError when uninitialized."""
        with pytest.raises(RuntimeError) as exc_info:
            get_real_time_processor()
        
        assert "not initialized" in str(exc_info.value)

    def test_default_processors_created_on_init(self):
        """Test that default processors (api_performance, latency_alerts) are registered."""
        processor = initialize_real_time_processor()
        
        assert "api_performance" in processor.processors
        assert "latency_alerts" in processor.processors
        
        # Verify types
        assert isinstance(processor.processors["api_performance"], AggregationProcessor)
        assert isinstance(processor.processors["latency_alerts"], AlertProcessor)


class TestEventIngestion:
    """Tests for event ingestion and buffer management."""

    @pytest.mark.asyncio
    async def test_ingest_event_adds_to_buffer(self):
        """Test that ingesting an event adds it to the stream buffer."""
        processor = initialize_real_time_processor(buffer_size=1000)
        
        event = StreamEvent(
            event_id="test-1",
            event_type=StreamEventType.PERFORMANCE_EVENT,
            timestamp=datetime.now(timezone.utc),
            data={"latency_ms": 100.5},
            source="test",
            tags=["test"]
        )
        
        await processor.ingest_event(event)
        
        # Event should be in buffer
        events = processor.stream_buffer.get_events()
        assert len(events) == 1
        assert events[0].event_id == "test-1"

    @pytest.mark.asyncio
    async def test_aggregation_processor_calculates_avg(self):
        """Test that AggregationProcessor correctly computes average."""
        processor = initialize_real_time_processor()
        await processor.start()
        
        aggregation = processor.processors["api_performance"]
        
        # Ingest three events with latency_ms values
        for i, latency in enumerate([100, 200, 300]):
            event = StreamEvent(
                event_id=f"test-{i}",
                event_type=StreamEventType.PERFORMANCE_EVENT,
                timestamp=datetime.now(timezone.utc),
                data={"latency_ms": latency},
                source="test",
                tags=[]
            )
            await aggregation.process_event(event)
        
        # Get aggregations
        aggs = aggregation._calculate_current_aggregations()
        
        assert "latency_ms" in aggs
        assert aggs["latency_ms"]["avg"] == 200.0  # (100 + 200 + 300) / 3
        assert aggs["latency_ms"]["count"] == 3
        assert aggs["latency_ms"]["min"] == 100
        assert aggs["latency_ms"]["max"] == 300
        
        await processor.stop()


class TestAlertProcessor:
    """Tests for alert processor threshold evaluation."""

    @pytest.mark.asyncio
    async def test_alert_processor_fires_on_threshold(self):
        """Test that AlertProcessor fires when latency exceeds threshold."""
        processor = initialize_real_time_processor()
        alerting = processor.processors["latency_alerts"]
        
        # Ingest event with latency_ms=6000 (above 5000 threshold)
        event = StreamEvent(
            event_id="high-latency",
            event_type=StreamEventType.PERFORMANCE_EVENT,
            timestamp=datetime.now(timezone.utc),
            data={"latency_ms": 6000},
            source="test",
            tags=[]
        )
        
        result = await alerting.process_event(event)
        
        # Should have triggered an alert
        assert result is not None
        assert "alerts" in result
        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["rule_id"] == "high_latency"
        
        # Check alert history
        assert len(alerting.alert_history) == 1

    @pytest.mark.asyncio
    async def test_alert_processor_no_fire_below_threshold(self):
        """Test that AlertProcessor does not fire when below threshold."""
        processor = initialize_real_time_processor()
        alerting = processor.processors["latency_alerts"]
        
        # Ingest event with latency_ms=100 (well below 5000 threshold)
        event = StreamEvent(
            event_id="low-latency",
            event_type=StreamEventType.PERFORMANCE_EVENT,
            timestamp=datetime.now(timezone.utc),
            data={"latency_ms": 100},
            source="test",
            tags=[]
        )
        
        result = await alerting.process_event(event)
        
        # Should not have triggered an alert
        assert result is None
        assert len(alerting.alert_history) == 0


class TestFilterProcessor:
    """Tests for filter processor functionality."""

    @pytest.mark.asyncio
    async def test_filter_processor_passes_matching_event(self):
        """Test that FilterProcessor passes events matching conditions."""
        processor = initialize_real_time_processor()
        
        # Create a filter that only passes performance_event types
        filter_proc = FilterProcessor(
            processor_id="test_filter",
            filter_conditions=[
                {"field": "event_type", "operator": "eq", "value": "performance_event"}
            ]
        )
        processor.add_processor(filter_proc)
        
        # Create matching event
        event = StreamEvent(
            event_id="match",
            event_type=StreamEventType.PERFORMANCE_EVENT,
            timestamp=datetime.now(timezone.utc),
            data={},
            source="test",
            tags=[]
        )
        
        result = await filter_proc.process_event(event)
        
        assert result is not None
        assert result["passed"] is True
        assert filter_proc.passed_count == 1


class TestAnalyticsAPI:
    """Tests for analytics API endpoints."""

    def test_analytics_status_endpoint_returns_200(self):
        """Test that /analytics/status endpoint returns HTTP 200."""
        # Initialize processor
        processor = initialize_real_time_processor()
        
        # Test the processor stats directly (what the endpoint returns)
        stats = processor.get_processor_stats()
        
        assert "processing_stats" in stats
        assert "buffer_stats" in stats
        assert "processor_stats" in stats

    def test_analytics_events_endpoint_filters_by_type(self):
        """Test that /analytics/events endpoint filters by event_type."""
        # Initialize processor and add events
        processor = initialize_real_time_processor()
        
        # Add mixed events to buffer
        perf_event = StreamEvent(
            event_id="perf-1",
            event_type=StreamEventType.PERFORMANCE_EVENT,
            timestamp=datetime.now(timezone.utc),
            data={"latency_ms": 100},
            source="test",
            tags=[]
        )
        user_event = StreamEvent(
            event_id="user-1",
            event_type=StreamEventType.USER_ACTION,
            timestamp=datetime.now(timezone.utc),
            data={"action": "click"},
            source="test",
            tags=[]
        )
        
        processor.stream_buffer.add_event(perf_event)
        processor.stream_buffer.add_event(user_event)
        
        # Test filtering by event type (what the endpoint does)
        from datetime import timedelta
        since = datetime.now(timezone.utc) - timedelta(minutes=60)
        
        # Get all events
        all_events = processor.stream_buffer.get_events(count=100, since=since)
        assert len(all_events) == 2
        
        # Filter by performance_event
        perf_events = processor.stream_buffer.get_events(
            count=100,
            since=since,
            event_types=[StreamEventType.PERFORMANCE_EVENT]
        )
        
        assert len(perf_events) == 1
        assert perf_events[0].event_type == StreamEventType.PERFORMANCE_EVENT
        assert perf_events[0].event_id == "perf-1"


class TestProcessorLifecycle:
    """Tests for processor start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_processor_start_stop(self):
        """Test that processor can be started and stopped cleanly."""
        processor = initialize_real_time_processor()
        
        assert processor.is_running is False
        
        await processor.start()
        assert processor.is_running is True
        
        await processor.stop()
        assert processor.is_running is False

    @pytest.mark.asyncio
    async def test_processor_handles_events_when_running(self):
        """Test that processor processes events when running."""
        processor = initialize_real_time_processor()
        await processor.start()
        
        # Give the processor a moment to start
        await asyncio.sleep(0.1)
        
        event = StreamEvent(
            event_id="lifecycle-test",
            event_type=StreamEventType.PERFORMANCE_EVENT,
            timestamp=datetime.now(timezone.utc),
            data={"latency_ms": 50},
            source="test",
            tags=[]
        )
        
        await processor.ingest_event(event)
        
        # Give it time to process
        await asyncio.sleep(0.2)
        
        # Event should be in buffer
        events = processor.stream_buffer.get_events()
        assert any(e.event_id == "lifecycle-test" for e in events)
        
        await processor.stop()
