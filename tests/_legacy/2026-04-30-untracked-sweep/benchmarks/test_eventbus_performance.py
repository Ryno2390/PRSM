"""
EventBus Performance Benchmarks

Tests throughput, latency, concurrency, memory, and error resilience
of the EventBus pub/sub system.
"""

import asyncio
import time
import tracemalloc
from dataclasses import dataclass
from typing import List

import pytest

from prsm.compute.nwtn.bsc.event_bus import EventBus, BSCEvent, EventType


class TestEventBusPerformance:
    """Performance benchmarks for EventBus."""

    @pytest.mark.asyncio
    async def test_single_subscriber_throughput(self):
        """Benchmark: 1 subscriber, 10_000 events, measure events/sec."""
        bus = EventBus()
        call_count = 0

        async def callback(event: BSCEvent):
            nonlocal call_count
            call_count += 1

        await bus.subscribe(EventType.CHUNK_PROMOTED, callback)

        # Warm up
        for _ in range(100):
            event = BSCEvent(
                event_type=EventType.CHUNK_PROMOTED,
                session_id="warmup",
                data={}
            )
            await bus.publish(event)

        # Reset counter after warmup
        call_count = 0

        # Measure
        N = 10_000
        start = time.perf_counter()
        for _ in range(N):
            event = BSCEvent(
                event_type=EventType.CHUNK_PROMOTED,
                session_id="bench",
                data={}
            )
            await bus.publish(event)
        elapsed = time.perf_counter() - start

        rate = N / elapsed
        print(f"[BENCH] single_subscriber_throughput: {rate:.0f} ops/sec (n={N}, total={elapsed:.3f}s)")

        assert rate >= 5_000, f"Expected >= 5_000 ops/sec, got {rate:.0f}"
        assert call_count == N, f"Expected {N} calls, got {call_count}"

    @pytest.mark.asyncio
    async def test_multi_subscriber_throughput(self):
        """Benchmark: 10 subscribers, 5_000 events, measure events/sec."""
        bus = EventBus()
        call_counts = [0] * 10

        callbacks = []
        for i in range(10):
            async def make_callback(idx):
                async def callback(event: BSCEvent):
                    call_counts[idx] += 1
                return callback
            callbacks.append(await make_callback(i))

        for callback in callbacks:
            await bus.subscribe(EventType.CHUNK_REJECTED, callback)

        # Warm up
        for _ in range(50):
            event = BSCEvent(
                event_type=EventType.CHUNK_REJECTED,
                session_id="warmup",
                data={}
            )
            await bus.publish(event)

        # Reset counters after warmup
        call_counts = [0] * 10

        # Measure
        N = 5_000
        start = time.perf_counter()
        for _ in range(N):
            event = BSCEvent(
                event_type=EventType.CHUNK_REJECTED,
                session_id="bench",
                data={}
            )
            await bus.publish(event)
        elapsed = time.perf_counter() - start

        rate = N / elapsed
        print(f"[BENCH] multi_subscriber_throughput: {rate:.0f} ops/sec (n={N}, total={elapsed:.3f}s)")

        assert rate >= 1_000, f"Expected >= 1_000 ops/sec, got {rate:.0f}"
        total_calls = sum(call_counts)
        assert total_calls == N * 10, f"Expected {N * 10} total calls, got {total_calls}"

    @pytest.mark.asyncio
    async def test_publish_latency_p99(self):
        """Benchmark: Measure p50, p95, p99 latency for single event publish."""
        bus = EventBus()

        async def callback(event: BSCEvent):
            pass

        await bus.subscribe(EventType.ROUND_ADVANCED, callback)

        # Warm up
        for _ in range(100):
            event = BSCEvent(
                event_type=EventType.ROUND_ADVANCED,
                session_id="warmup",
                data={}
            )
            await bus.publish(event)

        # Measure individual latencies
        N = 1_000
        latencies = []
        for _ in range(N):
            event = BSCEvent(
                event_type=EventType.ROUND_ADVANCED,
                session_id="bench",
                data={}
            )
            start = time.perf_counter()
            await bus.publish(event)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1_000_000)  # Convert to microseconds

        latencies.sort()
        p50 = latencies[int(N * 0.50)]
        p95 = latencies[int(N * 0.95)]
        p99 = latencies[int(N * 0.99)]

        print(f"[BENCH] publish_latency_p99: p50={p50:.0f}μs, p95={p95:.0f}μs, p99={p99:.0f}μs (n={N})")

        assert p99 < 5_000, f"Expected p99 < 5_000μs (5ms), got {p99:.0f}μs"

    @pytest.mark.asyncio
    async def test_concurrent_publishers(self):
        """Benchmark: 5 concurrent tasks each publishing 1_000 events."""
        bus = EventBus()
        received_count = 0

        async def callback(event: BSCEvent):
            nonlocal received_count
            received_count += 1

        await bus.subscribe(EventType.ROUND_ADVANCED, callback)

        # Warm up
        for _ in range(100):
            event = BSCEvent(
                event_type=EventType.ROUND_ADVANCED,
                session_id="warmup",
                data={}
            )
            await bus.publish(event)

        # Reset counter after warmup
        received_count = 0

        # Concurrent publishers
        async def publisher(publisher_id: int):
            for i in range(1_000):
                event = BSCEvent(
                    event_type=EventType.ROUND_ADVANCED,
                    session_id=f"publisher_{publisher_id}",
                    data={"index": i}
                )
                await bus.publish(event)

        N = 5_000  # Total events
        start = time.perf_counter()
        await asyncio.gather(
            publisher(0),
            publisher(1),
            publisher(2),
            publisher(3),
            publisher(4),
        )
        elapsed = time.perf_counter() - start

        rate = N / elapsed
        print(f"[BENCH] concurrent_publishers: {rate:.0f} ops/sec (n={N}, total={elapsed:.3f}s)")

        assert received_count == N, f"Expected {N} events received, got {received_count}"
        assert rate >= 2_000, f"Expected >= 2_000 ops/sec, got {rate:.0f}"

    @pytest.mark.asyncio
    async def test_memory_under_load(self):
        """Benchmark: Memory growth under 100_000 events."""
        bus = EventBus()

        async def callback(event: BSCEvent):
            pass  # No-op callback

        await bus.subscribe(EventType.CHUNK_PROMOTED, callback)

        # Warm up
        for _ in range(100):
            event = BSCEvent(
                event_type=EventType.CHUNK_PROMOTED,
                session_id="warmup",
                data={}
            )
            await bus.publish(event)

        # Track memory
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        # Publish many events
        N = 100_000
        for _ in range(N):
            event = BSCEvent(
                event_type=EventType.CHUNK_PROMOTED,
                session_id="bench",
                data={}
            )
            await bus.publish(event)

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate memory growth
        stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_growth_kb = sum(stat.size_diff for stat in stats) / 1024
        total_growth_mb = total_growth_kb / 1024

        elapsed = 0  # Not measuring time for this one
        print(f"[BENCH] memory_under_load: {total_growth_mb:.2f} MB growth (n={N})")

        assert total_growth_mb < 50, f"Expected < 50 MB growth, got {total_growth_mb:.2f} MB"

    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe_overhead(self):
        """Benchmark: Subscribe/publish/unsubscribe cycle overhead."""
        bus = EventBus()

        # Warm up
        async def warmup_callback(event: BSCEvent):
            pass
        await bus.subscribe(EventType.ROUND_ADVANCED, warmup_callback)
        for _ in range(10):
            event = BSCEvent(
                event_type=EventType.ROUND_ADVANCED,
                session_id="warmup",
                data={}
            )
            await bus.publish(event)
        await bus.unsubscribe(EventType.ROUND_ADVANCED, warmup_callback)

        # Measure
        N = 1_000
        start = time.perf_counter()
        for i in range(N):
            async def callback(event: BSCEvent):
                pass
            await bus.subscribe(EventType.ROUND_ADVANCED, callback)
            event = BSCEvent(
                event_type=EventType.ROUND_ADVANCED,
                session_id=f"bench_{i}",
                data={}
            )
            await bus.publish(event)
            await bus.unsubscribe(EventType.ROUND_ADVANCED, callback)
        elapsed = time.perf_counter() - start

        rate = N / elapsed
        print(f"[BENCH] subscribe_unsubscribe_overhead: {rate:.0f} ops/sec (n={N}, total={elapsed:.3f}s)")

        assert elapsed < 5, f"Expected < 5 seconds total, got {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_error_resilience_throughput(self):
        """Benchmark: Error in one subscriber doesn't affect others."""
        bus = EventBus()
        good_count = 0

        async def failing_callback(event: BSCEvent):
            raise Exception("Simulated error")

        async def good_callback(event: BSCEvent):
            nonlocal good_count
            good_count += 1

        await bus.subscribe(EventType.CHUNK_REJECTED, failing_callback)
        await bus.subscribe(EventType.CHUNK_REJECTED, good_callback)

        # Warm up
        for _ in range(50):
            event = BSCEvent(
                event_type=EventType.CHUNK_REJECTED,
                session_id="warmup",
                data={}
            )
            await bus.publish(event)

        # Reset counter after warmup
        good_count = 0

        # Measure
        N = 5_000
        start = time.perf_counter()
        for _ in range(N):
            event = BSCEvent(
                event_type=EventType.CHUNK_REJECTED,
                session_id="bench",
                data={}
            )
            await bus.publish(event)
        elapsed = time.perf_counter() - start

        rate = N / elapsed
        print(f"[BENCH] error_resilience_throughput: {rate:.0f} ops/sec (n={N}, total={elapsed:.3f}s)")

        assert good_count == N, f"Expected good subscriber to receive {N} events, got {good_count}"
        assert rate >= 1_000, f"Expected >= 1_000 ops/sec, got {rate:.0f}"
