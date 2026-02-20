"""
PRSM Benchmark Collector Module

Global performance metric collection and aggregation for the PRSM system.
Provides real-time performance monitoring and historical benchmark tracking.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricStats:
    """Statistics for a single metric"""
    operation: str
    sample_count: int = 0
    total_time_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    sum_squared: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    samples: List[float] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    @property
    def mean_ms(self) -> float:
        return self.total_time_ms / self.sample_count if self.sample_count > 0 else 0.0
    
    @property
    def variance(self) -> float:
        if self.sample_count < 2:
            return 0.0
        return (self.sum_squared / self.sample_count) - (self.mean_ms ** 2)
    
    @property
    def std_dev(self) -> float:
        return self.variance ** 0.5
    
    @property
    def measurements_per_second(self) -> float:
        if not self.first_seen or not self.last_seen:
            return 0.0
        duration = (self.last_seen - self.first_seen).total_seconds()
        return self.sample_count / duration if duration > 0 else 0.0
    
    def add_sample(self, duration_ms: float, timestamp: Optional[datetime] = None, max_samples: int = 10000):
        self.sample_count += 1
        self.total_time_ms += duration_ms
        self.sum_squared += duration_ms ** 2
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        
        now = timestamp or datetime.now(timezone.utc)
        if not self.first_seen:
            self.first_seen = now
        self.last_seen = now
        
        self.samples.append(duration_ms)
        if len(self.samples) > max_samples:
            self.samples = self.samples[-max_samples:]
        
        self._calculate_percentiles()
    
    def _calculate_percentiles(self):
        if not self.samples:
            return
        
        sorted_samples = sorted(self.samples)
        n = len(sorted_samples)
        
        def percentile(p: float) -> float:
            if n == 1:
                return sorted_samples[0]
            k = (n - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < n else f
            return sorted_samples[f] + (k - f) * (sorted_samples[c] - sorted_samples[f])
        
        self.p50_ms = percentile(50)
        self.p95_ms = percentile(95)
        self.p99_ms = percentile(99)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "sample_count": self.sample_count,
            "mean_ms": round(self.mean_ms, 3),
            "min_ms": round(self.min_ms, 3) if self.min_ms != float('inf') else 0,
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "std_dev": round(self.std_dev, 3),
            "measurements_per_second": round(self.measurements_per_second, 3),
        }


class BenchmarkCollector:
    """
    Global benchmark collector for performance metrics
    
    Collects, aggregates, and reports performance metrics across the PRSM system.
    Thread-safe singleton pattern for global access.
    """
    
    _instance: Optional['BenchmarkCollector'] = None
    _lock = asyncio.Lock()
    
    def __init__(self, max_samples_per_metric: int = 10000):
        self._metrics: Dict[str, MetricStats] = defaultdict(lambda: MetricStats(operation=""))
        self._max_samples = max_samples_per_metric
        self._start_time = datetime.now(timezone.utc)
        self._enabled = True
        self._callbacks: List[Callable[[str, float], None]] = []
    
    @classmethod
    async def get_instance(cls) -> 'BenchmarkCollector':
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def get_instance_sync(cls) -> 'BenchmarkCollector':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False
    
    def record(self, operation: str, duration_ms: float, timestamp: Optional[datetime] = None) -> None:
        if not self._enabled:
            return
        
        stats = self._metrics[operation]
        stats.operation = operation
        stats.add_sample(duration_ms, timestamp, self._max_samples)
        
        for callback in self._callbacks:
            try:
                callback(operation, duration_ms)
            except Exception as e:
                logger.warning("Benchmark callback failed", error=str(e))
    
    async def record_async(self, operation: str, duration_ms: float, timestamp: Optional[datetime] = None) -> None:
        self.record(operation, duration_ms, timestamp)
    
    def add_callback(self, callback: Callable[[str, float], None]) -> None:
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[str, float], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_metric(self, operation: str) -> Optional[MetricStats]:
        return self._metrics.get(operation)
    
    def get_all_metrics(self) -> Dict[str, MetricStats]:
        return dict(self._metrics)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        total_operations = len(self._metrics)
        total_measurements = sum(m.sample_count for m in self._metrics.values())
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        
        return {
            "total_operations": total_operations,
            "total_measurements": total_measurements,
            "uptime_seconds": uptime,
            "average_measurements_per_second": total_measurements / uptime if uptime > 0 else 0,
            "operations": {op: stats.to_dict() for op, stats in self._metrics.items()},
        }
    
    def get_top_operations(self, by: str = "mean_ms", limit: int = 10) -> List[Dict[str, Any]]:
        sorted_ops = sorted(
            self._metrics.items(),
            key=lambda x: getattr(x[1], by, 0),
            reverse=True
        )
        return [{"operation": op, **stats.to_dict()} for op, stats in sorted_ops[:limit]]
    
    def reset(self) -> None:
        self._metrics.clear()
        self._start_time = datetime.now(timezone.utc)
    
    def export_metrics(self, format: str = "json") -> str:
        import json
        
        data = {
            "export_time": datetime.now(timezone.utc).isoformat(),
            "summary": self.get_summary_stats(),
            "metrics": {op: stats.to_dict() for op, stats in self._metrics.items()},
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        return str(data)


def get_global_collector() -> BenchmarkCollector:
    """Get the global benchmark collector instance"""
    return BenchmarkCollector.get_instance_sync()


def reset_global_collector() -> None:
    """Reset the global benchmark collector"""
    collector = get_global_collector()
    collector.reset()


class PerformanceRecorder:
    """
    Context manager for recording performance metrics
    
    Usage:
        async with PerformanceRecorder("database_query") as recorder:
            await db.execute(query)
        # Duration automatically recorded
    """
    
    def __init__(self, operation: str, collector: Optional[BenchmarkCollector] = None):
        self.operation = operation
        self.collector = collector or get_global_collector()
        self._start_time: Optional[float] = None
        self._duration_ms: float = 0.0
    
    def __enter__(self) -> 'PerformanceRecorder':
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = time.perf_counter()
        self._duration_ms = (end_time - self._start_time) * 1000 if self._start_time else 0
        self.collector.record(self.operation, self._duration_ms)
    
    async def __aenter__(self) -> 'PerformanceRecorder':
        self._start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = time.perf_counter()
        self._duration_ms = (end_time - self._start_time) * 1000 if self._start_time else 0
        self.collector.record(self.operation, self._duration_ms)
    
    @property
    def duration_ms(self) -> float:
        return self._duration_ms
