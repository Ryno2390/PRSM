#!/usr/bin/env python3
"""
Real-Time Metrics Collection System
===================================

High-performance metrics collection, aggregation, and storage system
for comprehensive analytics and monitoring.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Set
from collections import defaultdict, deque
import threading
import json
from pathlib import Path

from prsm.compute.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"           # Monotonically increasing values
    GAUGE = "gauge"              # Instantaneous values that can go up/down
    HISTOGRAM = "histogram"       # Distribution of values
    SUMMARY = "summary"          # Statistical summary of observations
    RATE = "rate"               # Rate of change over time
    PERCENTAGE = "percentage"    # Percentage values
    DURATION = "duration"        # Time duration measurements


class AggregationType(Enum):
    """Types of aggregation for metrics"""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    P95 = "p95"
    P99 = "p99"
    STDDEV = "stddev"
    RATE_PER_SECOND = "rate_per_second"


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected"""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    aggregations: List[AggregationType] = field(default_factory=list)
    retention_days: int = 30
    collection_interval: float = 60.0  # seconds
    enabled: bool = True
    
    def __post_init__(self):
        if not self.aggregations:
            # Default aggregations based on metric type
            if self.metric_type == MetricType.COUNTER:
                self.aggregations = [AggregationType.SUM, AggregationType.RATE_PER_SECOND]
            elif self.metric_type == MetricType.GAUGE:
                self.aggregations = [AggregationType.AVERAGE, AggregationType.MIN, AggregationType.MAX]
            elif self.metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                self.aggregations = [AggregationType.AVERAGE, AggregationType.P95, AggregationType.P99]
            else:
                self.aggregations = [AggregationType.AVERAGE]


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: Union[float, int]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricPoint':
        """Create from dictionary"""
        return cls(
            name=data["name"],
            value=data["value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            labels=data.get("labels", {}),
            metadata=data.get("metadata", {})
        )


class MetricsBuffer:
    """High-performance circular buffer for metrics"""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.total_points = 0
        self.dropped_points = 0
    
    def add_point(self, point: MetricPoint):
        """Add a metric point to the buffer"""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self.dropped_points += 1
            
            self.buffer.append(point)
            self.total_points += 1
    
    def get_points(self, count: Optional[int] = None, 
                   since: Optional[datetime] = None) -> List[MetricPoint]:
        """Get metric points from buffer"""
        with self.lock:
            points = list(self.buffer)
            
            # Filter by time if specified
            if since:
                points = [p for p in points if p.timestamp >= since]
            
            # Limit count if specified
            if count and len(points) > count:
                points = points[-count:]
            
            return points
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                "current_size": len(self.buffer),
                "max_size": self.max_size,
                "total_points": self.total_points,
                "dropped_points": self.dropped_points,
                "drop_rate": self.dropped_points / max(1, self.total_points)
            }


class MetricsAggregator:
    """Aggregates metrics over time windows"""
    
    def __init__(self, window_size: timedelta = timedelta(minutes=5)):
        self.window_size = window_size
        self.aggregation_cache = {}
        self.lock = threading.RLock()
    
    def aggregate_points(self, points: List[MetricPoint], 
                        aggregation: AggregationType,
                        group_by: List[str] = None) -> Dict[str, float]:
        """Aggregate metric points"""
        if not points:
            return {}
        
        # Group points by labels if specified
        if group_by:
            groups = defaultdict(list)
            for point in points:
                key = tuple(point.labels.get(label, "") for label in group_by)
                groups[key].append(point.value)
        else:
            groups = {"all": [p.value for p in points]}
        
        results = {}
        for group_key, values in groups.items():
            if not values:
                continue
                
            group_name = ":".join(group_key) if isinstance(group_key, tuple) else str(group_key)
            
            if aggregation == AggregationType.SUM:
                results[group_name] = sum(values)
            elif aggregation == AggregationType.AVERAGE:
                results[group_name] = sum(values) / len(values)
            elif aggregation == AggregationType.MIN:
                results[group_name] = min(values)
            elif aggregation == AggregationType.MAX:
                results[group_name] = max(values)
            elif aggregation == AggregationType.COUNT:
                results[group_name] = len(values)
            elif aggregation == AggregationType.MEDIAN:
                sorted_values = sorted(values)
                n = len(sorted_values)
                results[group_name] = sorted_values[n // 2] if n % 2 == 1 else \
                    (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
            elif aggregation in [AggregationType.P95, AggregationType.P99]:
                percentile = 0.95 if aggregation == AggregationType.P95 else 0.99
                sorted_values = sorted(values)
                index = int(len(sorted_values) * percentile)
                results[group_name] = sorted_values[min(index, len(sorted_values) - 1)]
            elif aggregation == AggregationType.STDDEV:
                if len(values) > 1:
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                    results[group_name] = variance ** 0.5
                else:
                    results[group_name] = 0.0
            elif aggregation == AggregationType.RATE_PER_SECOND:
                if len(points) > 1:
                    time_span = (points[-1].timestamp - points[0].timestamp).total_seconds()
                    if time_span > 0:
                        results[group_name] = (values[-1] - values[0]) / time_span
                    else:
                        results[group_name] = 0.0
                else:
                    results[group_name] = 0.0
        
        return results
    
    def get_windowed_aggregation(self, points: List[MetricPoint],
                                aggregation: AggregationType,
                                window_count: int = 12) -> List[Dict[str, Any]]:
        """Get aggregated data over multiple time windows"""
        if not points:
            return []
        
        # Sort points by timestamp
        points.sort(key=lambda p: p.timestamp)
        
        # Calculate window boundaries
        end_time = points[-1].timestamp
        start_time = end_time - (self.window_size * window_count)
        
        windows = []
        for i in range(window_count):
            window_start = start_time + (self.window_size * i)
            window_end = window_start + self.window_size
            
            # Get points in this window
            window_points = [
                p for p in points 
                if window_start <= p.timestamp < window_end
            ]
            
            # Aggregate points in window
            aggregated = self.aggregate_points(window_points, aggregation)
            
            windows.append({
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "point_count": len(window_points),
                "aggregated_values": aggregated
            })
        
        return windows


class MetricsCollector:
    """Main metrics collection and management system"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./metrics_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Core components
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.buffers: Dict[str, MetricsBuffer] = {}
        self.aggregator = MetricsAggregator()
        
        # Collection state
        self.collectors: Dict[str, Callable] = {}
        self.collection_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
        # Statistics
        self.collection_stats = {
            "total_metrics_collected": 0,
            "collection_errors": 0,
            "last_collection_time": None,
            "active_collectors": 0
        }
        
        # Optional dependencies
        self._setup_optional_features()
        
        # Initialize built-in metrics
        self._register_builtin_metrics()
    
    def _setup_optional_features(self):
        """Setup features based on available optional dependencies"""
        self.redis_client = require_optional("redis")
        self.numpy = require_optional("numpy")
        
        if self.redis_client:
            logger.info("Redis available - enabling distributed metrics collection")
        if self.numpy:
            logger.info("NumPy available - enabling advanced statistical aggregations")
    
    def _register_builtin_metrics(self):
        """Register built-in system metrics"""
        builtin_metrics = [
            MetricDefinition(
                name="system.cpu_usage",
                metric_type=MetricType.GAUGE,
                description="System CPU usage percentage",
                unit="percent",
                aggregations=[AggregationType.AVERAGE, AggregationType.MAX]
            ),
            MetricDefinition(
                name="system.memory_usage",
                metric_type=MetricType.GAUGE,
                description="System memory usage",
                unit="bytes",
                aggregations=[AggregationType.AVERAGE, AggregationType.MAX]
            ),
            MetricDefinition(
                name="reasoning.engine_executions",
                metric_type=MetricType.COUNTER,
                description="Number of reasoning engine executions",
                labels=["engine_type", "status"],
                aggregations=[AggregationType.SUM, AggregationType.RATE_PER_SECOND]
            ),
            MetricDefinition(
                name="reasoning.execution_time",
                metric_type=MetricType.HISTOGRAM,
                description="Reasoning engine execution time",
                unit="seconds",
                labels=["engine_type"],
                aggregations=[AggregationType.AVERAGE, AggregationType.P95, AggregationType.P99]
            ),
            MetricDefinition(
                name="reasoning.quality_score",
                metric_type=MetricType.GAUGE,
                description="Reasoning quality score",
                labels=["engine_type"],
                aggregations=[AggregationType.AVERAGE, AggregationType.MIN, AggregationType.MAX]
            ),
            MetricDefinition(
                name="api.request_count",
                metric_type=MetricType.COUNTER,
                description="API request count",
                labels=["endpoint", "method", "status"],
                aggregations=[AggregationType.SUM, AggregationType.RATE_PER_SECOND]
            ),
            MetricDefinition(
                name="api.response_time",
                metric_type=MetricType.HISTOGRAM,
                description="API response time",
                unit="seconds",
                labels=["endpoint", "method"],
                aggregations=[AggregationType.AVERAGE, AggregationType.P95, AggregationType.P99]
            )
        ]
        
        for metric_def in builtin_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric definition"""
        self.metric_definitions[metric_def.name] = metric_def
        self.buffers[metric_def.name] = MetricsBuffer()
        
        logger.info(f"Registered metric: {metric_def.name} ({metric_def.metric_type.value})")
    
    def record_metric(self, name: str, value: Union[float, int], 
                     labels: Dict[str, str] = None, 
                     timestamp: Optional[datetime] = None):
        """Record a single metric value"""
        if name not in self.metric_definitions:
            logger.warning(f"Unknown metric: {name}")
            return
        
        timestamp = timestamp or datetime.now(timezone.utc)
        labels = labels or {}
        
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp,
            labels=labels
        )
        
        self.buffers[name].add_point(point)
        self.collection_stats["total_metrics_collected"] += 1
    
    def record_multiple_metrics(self, metrics: List[Dict[str, Any]]):
        """Record multiple metrics efficiently"""
        timestamp = datetime.now(timezone.utc)
        
        for metric_data in metrics:
            name = metric_data.get("name")
            value = metric_data.get("value")
            labels = metric_data.get("labels", {})
            
            if name and value is not None:
                self.record_metric(name, value, labels, timestamp)
    
    def get_metric_data(self, name: str, 
                       since: Optional[datetime] = None,
                       limit: Optional[int] = None) -> List[MetricPoint]:
        """Get raw metric data points"""
        if name not in self.buffers:
            return []
        
        return self.buffers[name].get_points(limit, since)
    
    def get_aggregated_data(self, name: str,
                           aggregation: AggregationType,
                           window_size: Optional[timedelta] = None,
                           window_count: int = 12,
                           group_by: List[str] = None) -> List[Dict[str, Any]]:
        """Get aggregated metric data"""
        if name not in self.buffers:
            return []
        
        # Set window size from metric definition if not provided
        if window_size:
            self.aggregator.window_size = window_size
        
        points = self.get_metric_data(name, since=datetime.now(timezone.utc) - 
                                     (self.aggregator.window_size * window_count))
        
        return self.aggregator.get_windowed_aggregation(points, aggregation, window_count)
    
    def register_collector(self, name: str, collector_func: Callable):
        """Register a custom metric collector function"""
        self.collectors[name] = collector_func
        logger.info(f"Registered collector: {name}")
    
    async def start_collection(self):
        """Start automated metric collection"""
        if self.running:
            logger.warning("Metrics collection already running")
            return
        
        self.running = True
        
        # Start collection tasks for each registered collector
        for name, collector_func in self.collectors.items():
            task = asyncio.create_task(self._run_collector(name, collector_func))
            self.collection_tasks[name] = task
        
        # Start built-in system metrics collection
        if has_optional_dependency("psutil"):
            system_task = asyncio.create_task(self._collect_system_metrics())
            self.collection_tasks["system_metrics"] = system_task
        
        self.collection_stats["active_collectors"] = len(self.collection_tasks)
        logger.info(f"Started metrics collection with {len(self.collection_tasks)} collectors")
    
    async def stop_collection(self):
        """Stop automated metric collection"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all collection tasks
        for name, task in self.collection_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.collection_tasks.clear()
        self.collection_stats["active_collectors"] = 0
        logger.info("Stopped metrics collection")
    
    async def _run_collector(self, name: str, collector_func: Callable):
        """Run a single collector function"""
        while self.running:
            try:
                # Execute collector function
                if asyncio.iscoroutinefunction(collector_func):
                    metrics = await collector_func()
                else:
                    metrics = collector_func()
                
                # Record collected metrics
                if isinstance(metrics, list):
                    self.record_multiple_metrics(metrics)
                elif isinstance(metrics, dict):
                    self.record_multiple_metrics([metrics])
                
                self.collection_stats["last_collection_time"] = datetime.now(timezone.utc)
                
            except Exception as e:
                logger.error(f"Error in collector {name}: {e}")
                self.collection_stats["collection_errors"] += 1
            
            # Wait before next collection
            await asyncio.sleep(60)  # Default 1-minute interval
    
    async def _collect_system_metrics(self):
        """Collect built-in system metrics"""
        psutil = require_optional("psutil")
        if not psutil:
            return
        
        while self.running:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_metric("system.cpu_usage", cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_metric("system.memory_usage", memory.used)
                
                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                self.record_metric("system.process_memory", process_memory.rss)
                self.record_metric("system.process_cpu", process.cpu_percent())
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        summary = {
            "collection_stats": self.collection_stats.copy(),
            "registered_metrics": len(self.metric_definitions),
            "active_buffers": len(self.buffers),
            "buffer_stats": {},
            "metric_definitions": {}
        }
        
        # Add buffer statistics
        for name, buffer in self.buffers.items():
            summary["buffer_stats"][name] = buffer.get_stats()
        
        # Add metric definitions
        for name, metric_def in self.metric_definitions.items():
            summary["metric_definitions"][name] = {
                "type": metric_def.metric_type.value,
                "description": metric_def.description,
                "unit": metric_def.unit,
                "labels": metric_def.labels,
                "enabled": metric_def.enabled
            }
        
        return summary
    
    def export_metrics(self, format_type: str = "json", 
                      since: Optional[datetime] = None) -> str:
        """Export metrics in various formats"""
        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {}
        }
        
        for name in self.metric_definitions.keys():
            points = self.get_metric_data(name, since)
            export_data["metrics"][name] = [p.to_dict() for p in points]
        
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def clear_all_metrics(self):
        """Clear all collected metrics data"""
        for buffer in self.buffers.values():
            buffer.clear()
        
        self.collection_stats = {
            "total_metrics_collected": 0,
            "collection_errors": 0,
            "last_collection_time": None,
            "active_collectors": len(self.collection_tasks)
        }
        
        logger.info("Cleared all metrics data")


# Export main classes
__all__ = [
    'MetricType',
    'AggregationType', 
    'MetricDefinition',
    'MetricPoint',
    'MetricsBuffer',
    'MetricsAggregator',
    'MetricsCollector'
]