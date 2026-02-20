"""
PRSM Performance Instrumentation Module

Provides timing decorators, memory profiling context managers, and performance
measurement utilities for instrumenting PRSM code paths.

Usage:
    from prsm.performance.instrumentation import timing_decorator, measure_performance
    
    @timing_decorator("process_query")
    async def process_query(query: str):
        ...
    
    async with measure_performance("database_query"):
        result = await db.execute(query)
"""

import asyncio
import functools
import time
import tracemalloc
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypeVar, ParamSpec
import structlog

logger = structlog.get_logger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


@dataclass
class TimingResult:
    """Result from a timing measurement"""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    current_mb: float
    peak_mb: float
    allocated_blocks: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "current_mb": self.current_mb,
            "peak_mb": self.peak_mb,
            "allocated_blocks": self.allocated_blocks,
        }


class PerformanceTimer:
    """
    High-resolution timer for performance measurements
    
    Thread-safe timer that can be used across async contexts.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._timings: List[TimingResult] = []
    
    def start(self) -> "PerformanceTimer":
        """Start the timer"""
        self._start_time = time.perf_counter()
        return self
    
    def stop(self) -> TimingResult:
        """Stop the timer and return result"""
        self._end_time = time.perf_counter()
        duration_ms = (self._end_time - self._start_time) * 1000 if self._start_time else 0
        result = TimingResult(
            operation=self.name,
            start_time=self._start_time or 0,
            end_time=self._end_time or 0,
            duration_ms=duration_ms,
        )
        self._timings.append(result)
        return result
    
    def record(self, operation: str, duration_ms: float, **metadata) -> TimingResult:
        """Record a timing result manually"""
        result = TimingResult(
            operation=operation,
            start_time=time.perf_counter() - duration_ms / 1000,
            end_time=time.perf_counter(),
            duration_ms=duration_ms,
            metadata=metadata,
        )
        self._timings.append(result)
        return result
    
    @property
    def total_time_ms(self) -> float:
        """Total time across all measurements"""
        return sum(t.duration_ms for t in self._timings)
    
    @property
    def average_time_ms(self) -> float:
        """Average time per measurement"""
        if not self._timings:
            return 0.0
        return self.total_time_ms / len(self._timings)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get timing summary"""
        return {
            "name": self.name,
            "total_measurements": len(self._timings),
            "total_time_ms": self.total_time_ms,
            "average_time_ms": self.average_time_ms,
            "min_time_ms": min(t.duration_ms for t in self._timings) if self._timings else 0,
            "max_time_ms": max(t.duration_ms for t in self._timings) if self._timings else 0,
        }


_timers: Dict[str, PerformanceTimer] = {}


def get_timer(name: str = "default") -> PerformanceTimer:
    """Get or create a named timer"""
    if name not in _timers:
        _timers[name] = PerformanceTimer(name)
    return _timers[name]


def timing_decorator(operation_name: Optional[str] = None, timer_name: str = "default"):
    """
    Decorator to measure function execution time
    
    Args:
        operation_name: Name for the operation (defaults to function name)
        timer_name: Name of the timer to use
    
    Usage:
        @timing_decorator("process_query")
        async def process_query(query: str):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            timer = get_timer(timer_name)
            timer.start()
            try:
                result = await func(*args, **kwargs)
                timing_result = timer.stop()
                timing_result.success = True
                logger.debug(
                    "Operation timed",
                    operation=op_name,
                    duration_ms=timing_result.duration_ms,
                )
                return result
            except Exception as e:
                timing_result = timer.stop()
                timing_result.success = False
                timing_result.error = str(e)
                logger.error(
                    "Operation failed",
                    operation=op_name,
                    duration_ms=timing_result.duration_ms,
                    error=str(e),
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            timer = get_timer(timer_name)
            timer.start()
            try:
                result = func(*args, **kwargs)
                timing_result = timer.stop()
                timing_result.success = True
                logger.debug(
                    "Operation timed",
                    operation=op_name,
                    duration_ms=timing_result.duration_ms,
                )
                return result
            except Exception as e:
                timing_result = timer.stop()
                timing_result.success = False
                timing_result.error = str(e)
                logger.error(
                    "Operation failed",
                    operation=op_name,
                    duration_ms=timing_result.duration_ms,
                    error=str(e),
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class TimingContext:
    """
    Context manager for timing code blocks
    
    Usage:
        with TimingContext("database_query") as tc:
            result = db.execute(query)
        print(f"Query took {tc.duration_ms:.2f}ms")
    """
    
    def __init__(self, operation: str, timer_name: str = "default", **metadata):
        self.operation = operation
        self.timer_name = timer_name
        self.metadata = metadata
        self.timer = get_timer(timer_name)
        self._start_time: Optional[float] = None
        self._result: Optional[TimingResult] = None
    
    def __enter__(self) -> "TimingContext":
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = time.perf_counter()
        duration_ms = (end_time - self._start_time) * 1000 if self._start_time else 0
        
        self._result = TimingResult(
            operation=self.operation,
            start_time=self._start_time or 0,
            end_time=end_time,
            duration_ms=duration_ms,
            success=exc_type is None,
            error=str(exc_val) if exc_val else None,
            metadata=self.metadata,
        )
        self.timer._timings.append(self._result)
        
        if exc_type:
            logger.error(
                "Timed operation failed",
                operation=self.operation,
                duration_ms=duration_ms,
                error=str(exc_val),
            )
        else:
            logger.debug(
                "Timed operation completed",
                operation=self.operation,
                duration_ms=duration_ms,
            )
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds"""
        return self._result.duration_ms if self._result else 0.0
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds"""
        return self.duration_ms / 1000.0
    
    @property
    def result(self) -> Optional[TimingResult]:
        """Get the timing result"""
        return self._result


class AsyncTimingContext:
    """
    Async context manager for timing code blocks
    
    Usage:
        async with AsyncTimingContext("api_call") as tc:
            result = await client.get(url)
        print(f"API call took {tc.duration_ms:.2f}ms")
    """
    
    def __init__(self, operation: str, timer_name: str = "default", **metadata):
        self.operation = operation
        self.timer_name = timer_name
        self.metadata = metadata
        self.timer = get_timer(timer_name)
        self._start_time: Optional[float] = None
        self._result: Optional[TimingResult] = None
    
    async def __aenter__(self) -> "AsyncTimingContext":
        self._start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = time.perf_counter()
        duration_ms = (end_time - self._start_time) * 1000 if self._start_time else 0
        
        self._result = TimingResult(
            operation=self.operation,
            start_time=self._start_time or 0,
            end_time=end_time,
            duration_ms=duration_ms,
            success=exc_type is None,
            error=str(exc_val) if exc_val else None,
            metadata=self.metadata,
        )
        self.timer._timings.append(self._result)
        
        if exc_type:
            logger.error(
                "Timed async operation failed",
                operation=self.operation,
                duration_ms=duration_ms,
                error=str(exc_val),
            )
        else:
            logger.debug(
                "Timed async operation completed",
                operation=self.operation,
                duration_ms=duration_ms,
            )
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds"""
        return self._result.duration_ms if self._result else 0.0


@asynccontextmanager
async def measure_performance(operation: str, timer_name: str = "default", **metadata):
    """
    Async context manager for measuring performance
    
    Usage:
        async with measure_performance("query_processing") as m:
            result = await process_query(query)
        print(f"Duration: {m.duration_ms:.2f}ms")
    """
    ctx = AsyncTimingContext(operation, timer_name, **metadata)
    await ctx.__aenter__()
    try:
        yield ctx
    finally:
        await ctx.__aexit__(None, None, None)


class MemoryProfiler:
    """
    Memory profiling utility using tracemalloc
    
    Usage:
        profiler = MemoryProfiler()
        profiler.start()
        
        # ... code to profile ...
        
        snapshot = profiler.stop()
        print(f"Peak memory: {snapshot.peak_mb:.2f} MB")
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._started = False
        self._snapshots: List[MemorySnapshot] = []
    
    def start(self) -> None:
        """Start memory profiling"""
        if not self.enabled:
            return
        
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._started = True
        logger.debug("Memory profiling started")
    
    def stop(self) -> MemorySnapshot:
        """Stop memory profiling and return snapshot"""
        if not self.enabled or not self._started:
            return MemorySnapshot(0, 0, 0, 0)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        snapshot = MemorySnapshot(
            timestamp=time.perf_counter(),
            current_mb=current / (1024 * 1024),
            peak_mb=peak / (1024 * 1024),
            allocated_blocks=len(tracemalloc.take_snapshot().statistics('lineno')) if tracemalloc.is_tracing() else 0,
        )
        
        self._snapshots.append(snapshot)
        logger.debug(
            "Memory profiling stopped",
            current_mb=snapshot.current_mb,
            peak_mb=snapshot.peak_mb,
        )
        
        return snapshot
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a snapshot without stopping profiling"""
        if not self.enabled or not tracemalloc.is_tracing():
            return MemorySnapshot(0, 0, 0, 0)
        
        current, peak = tracemalloc.get_traced_memory()
        
        snapshot = MemorySnapshot(
            timestamp=time.perf_counter(),
            current_mb=current / (1024 * 1024),
            peak_mb=peak / (1024 * 1024),
            allocated_blocks=0,
        )
        
        self._snapshots.append(snapshot)
        return snapshot
    
    @contextmanager
    def profile_context(self, operation: str = "memory_profile"):
        """
        Context manager for memory profiling a code block
        
        Usage:
            profiler = MemoryProfiler()
            with profiler.profile_context("data_processing"):
                process_large_dataset(data)
        """
        self.start()
        try:
            yield self
        finally:
            snapshot = self.stop()
            logger.info(
                "Memory profile complete",
                operation=operation,
                peak_mb=snapshot.peak_mb,
                current_mb=snapshot.current_mb,
            )
    
    @property
    def peak_memory_mb(self) -> float:
        """Get peak memory usage in MB"""
        if not self._snapshots:
            return 0.0
        return max(s.peak_mb for s in self._snapshots)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory profiling summary"""
        return {
            "enabled": self.enabled,
            "total_snapshots": len(self._snapshots),
            "peak_memory_mb": self.peak_memory_mb,
            "snapshots": [s.to_dict() for s in self._snapshots],
        }
