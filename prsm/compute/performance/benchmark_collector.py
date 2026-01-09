"""
Performance Benchmark Collection Framework
Replaces simulated metrics with real instrumentation and data collection.
"""

import asyncio
import time
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import json
import csv
from pathlib import Path
import threading

from prsm.core.models import BaseModel


@dataclass
class TimingMeasurement:
    """Individual timing measurement"""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics for an operation"""
    operation: str
    sample_count: int
    mean_ms: float
    median_ms: float
    std_dev_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    measurements_per_second: float
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, collector: 'BenchmarkCollector', operation: str, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.collector = collector
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            duration_ms = (self.end_time - self.start_time) * 1000
            measurement = TimingMeasurement(
                operation=self.operation,
                start_time=self.start_time,
                end_time=self.end_time,
                duration_ms=duration_ms,
                metadata=self.metadata
            )
            self.collector.record_measurement(measurement)


class AsyncPerformanceTimer:
    """Async context manager for timing operations"""
    
    def __init__(self, collector: 'BenchmarkCollector', operation: str,
                 metadata: Optional[Dict[str, Any]] = None):
        self.collector = collector
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            duration_ms = (self.end_time - self.start_time) * 1000
            measurement = TimingMeasurement(
                operation=self.operation,
                start_time=self.start_time,
                end_time=self.end_time,
                duration_ms=duration_ms,
                metadata=self.metadata
            )
            await self.collector.record_measurement_async(measurement)


def performance_timer(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator for timing function execution"""
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                collector = get_global_collector()
                async with AsyncPerformanceTimer(collector, operation, metadata):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                collector = get_global_collector()
                with PerformanceTimer(collector, operation, metadata):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


class BenchmarkCollector:
    """Main performance benchmark collection system"""
    
    def __init__(self, max_measurements_per_operation: int = 10000):
        self.max_measurements = max_measurements_per_operation
        self.measurements: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_measurements)
        )
        self.metrics_cache: Dict[str, PerformanceMetrics] = {}
        self.cache_valid_until: Dict[str, datetime] = {}
        self.cache_duration_seconds = 30  # Cache metrics for 30 seconds
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance counters
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()
    
    def record_measurement(self, measurement: TimingMeasurement):
        """Record a performance measurement (sync)"""
        with self._lock:
            self.measurements[measurement.operation].append(measurement)
            self.operation_counts[measurement.operation] += 1
            # Invalidate cache for this operation
            if measurement.operation in self.cache_valid_until:
                del self.cache_valid_until[measurement.operation]
    
    async def record_measurement_async(self, measurement: TimingMeasurement):
        """Record a performance measurement (async)"""
        # For now, just call sync version - could be enhanced for async storage
        self.record_measurement(measurement)
    
    def time_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> PerformanceTimer:
        """Create a performance timer for an operation"""
        return PerformanceTimer(self, operation, metadata)
    
    def time_async_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> AsyncPerformanceTimer:
        """Create an async performance timer for an operation"""
        return AsyncPerformanceTimer(self, operation, metadata)
    
    def get_metrics(self, operation: str) -> Optional[PerformanceMetrics]:
        """Get aggregated performance metrics for an operation"""
        with self._lock:
            # Check cache first
            now = datetime.now(timezone.utc)
            if (operation in self.cache_valid_until and 
                now < self.cache_valid_until[operation]):
                return self.metrics_cache.get(operation)
            
            # Calculate metrics if we have measurements
            if operation not in self.measurements or not self.measurements[operation]:
                return None
            
            measurements = list(self.measurements[operation])
            durations = [m.duration_ms for m in measurements]
            
            if not durations:
                return None
            
            # Calculate statistics
            durations_sorted = sorted(durations)
            sample_count = len(durations)
            
            # Time range for rate calculation
            time_range = measurements[-1].timestamp - measurements[0].timestamp
            time_range_seconds = max(time_range.total_seconds(), 1)  # Avoid division by zero
            
            metrics = PerformanceMetrics(
                operation=operation,
                sample_count=sample_count,
                mean_ms=statistics.mean(durations),
                median_ms=statistics.median(durations),
                std_dev_ms=statistics.stdev(durations) if len(durations) > 1 else 0.0,
                min_ms=min(durations),
                max_ms=max(durations),
                p95_ms=durations_sorted[int(0.95 * len(durations_sorted))] if durations_sorted else 0.0,
                p99_ms=durations_sorted[int(0.99 * len(durations_sorted))] if durations_sorted else 0.0,
                measurements_per_second=sample_count / time_range_seconds
            )
            
            # Cache the result
            self.metrics_cache[operation] = metrics
            self.cache_valid_until[operation] = now + timedelta(seconds=self.cache_duration_seconds)
            
            return metrics
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get metrics for all operations"""
        with self._lock:
            result = {}
            for operation in self.measurements.keys():
                metrics = self.get_metrics(operation)
                if metrics:
                    result[operation] = metrics
            return result
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all operations"""
        with self._lock:
            total_measurements = sum(len(measurements) for measurements in self.measurements.values())
            uptime_seconds = time.time() - self.start_time
            
            return {
                "total_operations": len(self.measurements),
                "total_measurements": total_measurements,
                "uptime_seconds": uptime_seconds,
                "average_measurements_per_second": total_measurements / max(uptime_seconds, 1),
                "operations": list(self.measurements.keys()),
                "operation_counts": dict(self.operation_counts)
            }
    
    def export_to_json(self, file_path: Union[str, Path]) -> None:
        """Export all metrics to JSON file"""
        with self._lock:
            data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": self.get_summary_stats(),
                "metrics": {}
            }
            
            # Add detailed metrics for each operation
            for operation in self.measurements.keys():
                metrics = self.get_metrics(operation)
                if metrics:
                    data["metrics"][operation] = {
                        "sample_count": metrics.sample_count,
                        "mean_ms": metrics.mean_ms,
                        "median_ms": metrics.median_ms,
                        "std_dev_ms": metrics.std_dev_ms,
                        "min_ms": metrics.min_ms,
                        "max_ms": metrics.max_ms,
                        "p95_ms": metrics.p95_ms,
                        "p99_ms": metrics.p99_ms,
                        "measurements_per_second": metrics.measurements_per_second
                    }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_raw_measurements_csv(self, operation: str, file_path: Union[str, Path]) -> None:
        """Export raw measurements for an operation to CSV"""
        with self._lock:
            if operation not in self.measurements:
                raise ValueError(f"No measurements found for operation: {operation}")
            
            measurements = list(self.measurements[operation])
            
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'duration_ms', 'metadata'])
                
                for measurement in measurements:
                    writer.writerow([
                        measurement.timestamp.isoformat(),
                        measurement.duration_ms,
                        json.dumps(measurement.metadata)
                    ])
    
    def clear_measurements(self, operation: Optional[str] = None):
        """Clear measurements for a specific operation or all operations"""
        with self._lock:
            if operation:
                if operation in self.measurements:
                    self.measurements[operation].clear()
                    self.operation_counts[operation] = 0
                if operation in self.cache_valid_until:
                    del self.cache_valid_until[operation]
                if operation in self.metrics_cache:
                    del self.metrics_cache[operation]
            else:
                self.measurements.clear()
                self.operation_counts.clear()
                self.cache_valid_until.clear()
                self.metrics_cache.clear()
                self.start_time = time.time()


# Global collector instance
_global_collector: Optional[BenchmarkCollector] = None


def get_global_collector() -> BenchmarkCollector:
    """Get or create the global benchmark collector"""
    global _global_collector
    if _global_collector is None:
        _global_collector = BenchmarkCollector()
    return _global_collector


def set_global_collector(collector: BenchmarkCollector):
    """Set a custom global collector"""
    global _global_collector
    _global_collector = collector


def reset_global_collector():
    """Reset the global collector"""
    global _global_collector
    _global_collector = BenchmarkCollector()


# Convenience functions
def time_operation(operation: str, metadata: Optional[Dict[str, Any]] = None) -> PerformanceTimer:
    """Time an operation using the global collector"""
    return get_global_collector().time_operation(operation, metadata)


def time_async_operation(operation: str, metadata: Optional[Dict[str, Any]] = None) -> AsyncPerformanceTimer:
    """Time an async operation using the global collector"""
    return get_global_collector().time_async_operation(operation, metadata)


def get_operation_metrics(operation: str) -> Optional[PerformanceMetrics]:
    """Get metrics for an operation from the global collector"""
    return get_global_collector().get_metrics(operation)


def get_all_performance_metrics() -> Dict[str, PerformanceMetrics]:
    """Get all performance metrics from the global collector"""
    return get_global_collector().get_all_metrics()


def export_performance_data(json_path: Union[str, Path]):
    """Export performance data to JSON file"""
    get_global_collector().export_to_json(json_path)


# Example usage and testing
async def example_usage():
    """Example of how to use the performance instrumentation"""
    collector = get_global_collector()
    
    # Time a sync operation
    with collector.time_operation("database_query", {"table": "users"}):
        time.sleep(0.05)  # Simulate 50ms database query
    
    # Time an async operation
    async with collector.time_async_operation("api_request", {"endpoint": "/users"}):
        await asyncio.sleep(0.1)  # Simulate 100ms API request
    
    # Use decorators
    @performance_timer("computation", {"algorithm": "fast_sort"})
    def expensive_computation():
        time.sleep(0.02)  # Simulate 20ms computation
        return "result"
    
    @performance_timer("async_computation")
    async def expensive_async_computation():
        await asyncio.sleep(0.03)  # Simulate 30ms async computation
        return "async_result"
    
    # Execute operations
    expensive_computation()
    await expensive_async_computation()
    
    # Get metrics
    db_metrics = collector.get_metrics("database_query")
    if db_metrics:
        print(f"Database queries: {db_metrics.mean_ms:.2f}ms average")
    
    # Export data
    collector.export_to_json("performance_results.json")


if __name__ == "__main__":
    asyncio.run(example_usage())