"""
PRSM Advanced Metrics Collection and Export System
Comprehensive metrics collection with Prometheus integration, custom metrics, and real-time monitoring
"""

from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import time
import statistics
import logging
from collections import defaultdict, deque, Counter
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

# Prometheus client imports
try:
    from prometheus_client import (
        Counter as PrometheusCounter,
        Gauge as PrometheusGauge,
        Histogram as PrometheusHistogram,
        Summary as PrometheusSummary,
        Info as PrometheusInfo,
        Enum as PrometheusEnum,
        generate_latest,
        CollectorRegistry,
        CONTENT_TYPE_LATEST,
        start_http_server,
        push_to_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback implementations
    class PrometheusCounter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, amount=1): pass
        def labels(self, **kwargs): return self
    
    class PrometheusGauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, value): pass
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
        def labels(self, **kwargs): return self
    
    class PrometheusHistogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, value): pass
        def labels(self, **kwargs): return self
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


class MetricScope(Enum):
    """Metric collection scopes"""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    CUSTOM = "custom"


@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: Optional[str] = None
    scope: MetricScope = MetricScope.APPLICATION


@dataclass
class MetricDataPoint:
    """Individual metric data point"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    """Metrics system configuration"""
    service_name: str = "prsm-api"
    service_version: str = "1.0.0"
    environment: str = "production"
    
    # Collection settings
    collection_interval: int = 30  # seconds
    retention_period_hours: int = 24
    max_metric_points: int = 10000
    
    # Export settings
    enable_prometheus: bool = True
    prometheus_port: int = 8000
    prometheus_pushgateway_url: Optional[str] = None
    
    # Storage settings
    store_in_redis: bool = True
    redis_key_prefix: str = "metrics"
    
    # System metrics
    collect_system_metrics: bool = True
    collect_process_metrics: bool = True
    collect_runtime_metrics: bool = True
    
    # Custom metrics
    enable_business_metrics: bool = True
    enable_performance_metrics: bool = True


class SystemMetricsCollector:
    """Collector for system-level metrics"""
    
    def __init__(self):
        self.last_cpu_times = None
        self.last_network_io = None
        self.last_disk_io = None
        
        try:
            import psutil
            self.psutil = psutil
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
            logger.warning("psutil not available, system metrics will be limited")
    
    async def collect_metrics(self) -> Dict[str, MetricDataPoint]:
        """Collect system metrics"""
        metrics = {}
        
        if not self.psutil_available:
            return metrics
        
        try:
            # CPU metrics
            cpu_percent = self.psutil.cpu_percent(interval=1)
            cpu_count = self.psutil.cpu_count()
            
            metrics["system_cpu_usage_percent"] = MetricDataPoint(
                name="system_cpu_usage_percent",
                value=cpu_percent
            )
            
            metrics["system_cpu_count"] = MetricDataPoint(
                name="system_cpu_count",
                value=cpu_count
            )
            
            # Memory metrics
            memory = self.psutil.virtual_memory()
            
            metrics["system_memory_total_bytes"] = MetricDataPoint(
                name="system_memory_total_bytes",
                value=memory.total
            )
            
            metrics["system_memory_available_bytes"] = MetricDataPoint(
                name="system_memory_available_bytes",
                value=memory.available
            )
            
            metrics["system_memory_usage_percent"] = MetricDataPoint(
                name="system_memory_usage_percent",
                value=memory.percent
            )
            
            # Disk metrics
            disk_usage = self.psutil.disk_usage('/')
            
            metrics["system_disk_total_bytes"] = MetricDataPoint(
                name="system_disk_total_bytes",
                value=disk_usage.total
            )
            
            metrics["system_disk_free_bytes"] = MetricDataPoint(
                name="system_disk_free_bytes",
                value=disk_usage.free
            )
            
            metrics["system_disk_usage_percent"] = MetricDataPoint(
                name="system_disk_usage_percent",
                value=(disk_usage.used / disk_usage.total) * 100
            )
            
            # Network metrics
            network_io = self.psutil.net_io_counters()
            
            metrics["system_network_bytes_sent"] = MetricDataPoint(
                name="system_network_bytes_sent",
                value=network_io.bytes_sent
            )
            
            metrics["system_network_bytes_recv"] = MetricDataPoint(
                name="system_network_bytes_recv",
                value=network_io.bytes_recv
            )
            
            # Load average (Unix only)
            try:
                load_avg = self.psutil.getloadavg()
                metrics["system_load_average_1m"] = MetricDataPoint(
                    name="system_load_average_1m",
                    value=load_avg[0]
                )
                metrics["system_load_average_5m"] = MetricDataPoint(
                    name="system_load_average_5m",
                    value=load_avg[1]
                )
                metrics["system_load_average_15m"] = MetricDataPoint(
                    name="system_load_average_15m",
                    value=load_avg[2]
                )
            except (AttributeError, OSError):
                # Not available on Windows
                pass
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics


class ProcessMetricsCollector:
    """Collector for process-level metrics"""
    
    def __init__(self):
        try:
            import psutil
            self.process = psutil.Process()
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
    
    async def collect_metrics(self) -> Dict[str, MetricDataPoint]:
        """Collect process metrics"""
        metrics = {}
        
        if not self.psutil_available:
            return metrics
        
        try:
            # Process CPU
            cpu_percent = self.process.cpu_percent()
            
            metrics["process_cpu_usage_percent"] = MetricDataPoint(
                name="process_cpu_usage_percent",
                value=cpu_percent
            )
            
            # Process memory
            memory_info = self.process.memory_info()
            
            metrics["process_memory_rss_bytes"] = MetricDataPoint(
                name="process_memory_rss_bytes",
                value=memory_info.rss
            )
            
            metrics["process_memory_vms_bytes"] = MetricDataPoint(
                name="process_memory_vms_bytes",
                value=memory_info.vms
            )
            
            # Process file descriptors
            try:
                num_fds = self.process.num_fds()
                metrics["process_open_fds"] = MetricDataPoint(
                    name="process_open_fds",
                    value=num_fds
                )
            except (AttributeError, OSError):
                # Not available on Windows
                pass
            
            # Process threads
            num_threads = self.process.num_threads()
            metrics["process_threads"] = MetricDataPoint(
                name="process_threads",
                value=num_threads
            )
            
            # Process uptime
            create_time = self.process.create_time()
            uptime_seconds = time.time() - create_time
            
            metrics["process_uptime_seconds"] = MetricDataPoint(
                name="process_uptime_seconds",
                value=uptime_seconds
            )
            
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
        
        return metrics


class RuntimeMetricsCollector:
    """Collector for Python runtime metrics"""
    
    def __init__(self):
        self.start_time = time.time()
    
    async def collect_metrics(self) -> Dict[str, MetricDataPoint]:
        """Collect runtime metrics"""
        metrics = {}
        
        try:
            import gc
            import sys
            
            # Garbage collection
            gc_stats = gc.get_stats()
            for i, stat in enumerate(gc_stats):
                metrics[f"python_gc_generation_{i}_collections"] = MetricDataPoint(
                    name=f"python_gc_generation_{i}_collections",
                    value=stat['collections']
                )
                
                metrics[f"python_gc_generation_{i}_collected"] = MetricDataPoint(
                    name=f"python_gc_generation_{i}_collected",
                    value=stat['collected']
                )
                
                metrics[f"python_gc_generation_{i}_uncollectable"] = MetricDataPoint(
                    name=f"python_gc_generation_{i}_uncollectable",
                    value=stat['uncollectable']
                )
            
            # Object counts by type
            object_counts = Counter(type(obj).__name__ for obj in gc.get_objects())
            for obj_type, count in object_counts.most_common(10):
                metrics[f"python_objects_{obj_type}"] = MetricDataPoint(
                    name=f"python_objects_{obj_type}",
                    value=count,
                    labels={"object_type": obj_type}
                )
            
            # Runtime information
            metrics["python_info"] = MetricDataPoint(
                name="python_info",
                value=1,
                labels={
                    "version": sys.version.split()[0],
                    "implementation": sys.implementation.name
                }
            )
            
        except Exception as e:
            logger.error(f"Error collecting runtime metrics: {e}")
        
        return metrics


class MetricsCollector:
    """Advanced metrics collection and management system"""
    
    def __init__(self, config: MetricsConfig, redis_client: Optional[aioredis.Redis] = None):
        self.config = config
        self.redis = redis_client
        
        # Metric storage
        self.custom_metrics: Dict[str, MetricDefinition] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_metric_points))
        
        # Prometheus integration
        self.prometheus_registry = None
        self.prometheus_metrics: Dict[str, Any] = {}
        
        # System collectors
        self.system_collector = SystemMetricsCollector() if config.collect_system_metrics else None
        self.process_collector = ProcessMetricsCollector() if config.collect_process_metrics else None
        self.runtime_collector = RuntimeMetricsCollector() if config.collect_runtime_metrics else None
        
        # Collection state
        self.collection_active = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "metrics_collected": 0,
            "collection_errors": 0,
            "export_attempts": 0,
            "export_failures": 0,
            "custom_metrics_registered": 0
        }
        
        # Initialize Prometheus
        if config.enable_prometheus:
            self._initialize_prometheus()
    
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available")
            return
        
        try:
            self.prometheus_registry = CollectorRegistry()
            
            # Create default metrics
            self._create_default_prometheus_metrics()
            
            # Start HTTP server if configured
            if self.config.prometheus_port:
                start_http_server(self.config.prometheus_port, registry=self.prometheus_registry)
                logger.info(f"✅ Prometheus metrics server started on port {self.config.prometheus_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus: {e}")
    
    def _create_default_prometheus_metrics(self):
        """Create default Prometheus metrics"""
        
        # Request metrics
        self.prometheus_metrics["http_requests_total"] = PrometheusCounter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics["http_request_duration_seconds"] = PrometheusHistogram(
            "http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0],
            registry=self.prometheus_registry
        )
        
        # Database metrics
        self.prometheus_metrics["database_connections_active"] = PrometheusGauge(
            "database_connections_active",
            "Active database connections",
            ["pool", "role"],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics["database_query_duration_seconds"] = PrometheusHistogram(
            "database_query_duration_seconds",
            "Database query duration",
            ["query_type", "table"],
            buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0, 2.5, 5.0],
            registry=self.prometheus_registry
        )
        
        # Cache metrics
        self.prometheus_metrics["cache_operations_total"] = PrometheusCounter(
            "cache_operations_total",
            "Total cache operations",
            ["operation", "cache_layer", "result"],
            registry=self.prometheus_registry
        )
        
        # Task queue metrics
        self.prometheus_metrics["task_queue_size"] = PrometheusGauge(
            "task_queue_size",
            "Current task queue size",
            ["queue_name", "status"],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics["task_processing_duration_seconds"] = PrometheusHistogram(
            "task_processing_duration_seconds",
            "Task processing duration",
            ["queue_name", "task_type"],
            buckets=[.1, .5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.prometheus_registry
        )
        
        # System metrics
        self.prometheus_metrics["system_cpu_usage_percent"] = PrometheusGauge(
            "system_cpu_usage_percent",
            "System CPU usage percentage",
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics["system_memory_usage_bytes"] = PrometheusGauge(
            "system_memory_usage_bytes",
            "System memory usage in bytes",
            ["type"],
            registry=self.prometheus_registry
        )
    
    def register_metric(self, definition: MetricDefinition) -> bool:
        """Register a custom metric"""
        try:
            self.custom_metrics[definition.name] = definition
            
            # Create Prometheus metric if enabled
            if self.config.enable_prometheus and PROMETHEUS_AVAILABLE:
                self._create_prometheus_metric(definition)
            
            self.stats["custom_metrics_registered"] += 1
            logger.debug(f"Registered metric: {definition.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering metric {definition.name}: {e}")
            return False
    
    def _create_prometheus_metric(self, definition: MetricDefinition):
        """Create Prometheus metric from definition"""
        
        if definition.name in self.prometheus_metrics:
            return
        
        try:
            if definition.metric_type == MetricType.COUNTER:
                self.prometheus_metrics[definition.name] = PrometheusCounter(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.prometheus_registry
                )
            
            elif definition.metric_type == MetricType.GAUGE:
                self.prometheus_metrics[definition.name] = PrometheusGauge(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.prometheus_registry
                )
            
            elif definition.metric_type == MetricType.HISTOGRAM:
                buckets = definition.buckets or [.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0]
                self.prometheus_metrics[definition.name] = PrometheusHistogram(
                    definition.name,
                    definition.description,
                    definition.labels,
                    buckets=buckets,
                    registry=self.prometheus_registry
                )
            
            elif definition.metric_type == MetricType.SUMMARY:
                self.prometheus_metrics[definition.name] = PrometheusSummary(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.prometheus_registry
                )
        
        except Exception as e:
            logger.error(f"Error creating Prometheus metric {definition.name}: {e}")
    
    async def start_collection(self):
        """Start metric collection"""
        if self.collection_active:
            logger.warning("Metric collection is already active")
            return
        
        self.collection_active = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        
        logger.info("✅ Metrics collection started")
    
    async def stop_collection(self):
        """Stop metric collection"""
        if not self.collection_active:
            return
        
        self.collection_active = False
        
        if self.collection_task and not self.collection_task.done():
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.collection_active:
            try:
                # Collect all metrics
                await self._collect_all_metrics()
                
                # Export metrics if configured
                await self._export_metrics()
                
                # Sleep until next collection
                await asyncio.sleep(self.config.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                self.stats["collection_errors"] += 1
                await asyncio.sleep(self.config.collection_interval)
    
    async def _collect_all_metrics(self):
        """Collect metrics from all sources"""
        
        all_metrics = {}
        
        # Collect system metrics
        if self.system_collector:
            try:
                system_metrics = await self.system_collector.collect_metrics()
                all_metrics.update(system_metrics)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
        
        # Collect process metrics
        if self.process_collector:
            try:
                process_metrics = await self.process_collector.collect_metrics()
                all_metrics.update(process_metrics)
            except Exception as e:
                logger.error(f"Error collecting process metrics: {e}")
        
        # Collect runtime metrics
        if self.runtime_collector:
            try:
                runtime_metrics = await self.runtime_collector.collect_metrics()
                all_metrics.update(runtime_metrics)
            except Exception as e:
                logger.error(f"Error collecting runtime metrics: {e}")
        
        # Store metrics
        for metric_name, metric_data in all_metrics.items():
            self.metric_history[metric_name].append(metric_data)
            
            # Update Prometheus metrics
            if self.config.enable_prometheus and metric_name in self.prometheus_metrics:
                try:
                    prometheus_metric = self.prometheus_metrics[metric_name]
                    if hasattr(prometheus_metric, 'set'):
                        prometheus_metric.set(metric_data.value)
                    elif hasattr(prometheus_metric, 'inc'):
                        prometheus_metric.inc(metric_data.value)
                except Exception as e:
                    logger.debug(f"Error updating Prometheus metric {metric_name}: {e}")
        
        # Store in Redis if configured
        if self.redis and self.config.store_in_redis:
            await self._store_metrics_in_redis(all_metrics)
        
        self.stats["metrics_collected"] += len(all_metrics)
    
    async def _store_metrics_in_redis(self, metrics: Dict[str, MetricDataPoint]):
        """Store metrics in Redis"""
        try:
            pipeline = self.redis.pipeline()
            
            for metric_name, metric_data in metrics.items():
                key = f"{self.config.redis_key_prefix}:{metric_name}"
                
                # Store current value
                value_data = {
                    "value": metric_data.value,
                    "timestamp": metric_data.timestamp.isoformat(),
                    "labels": metric_data.labels,
                    "metadata": metric_data.metadata
                }
                
                pipeline.setex(f"{key}:current", 300, json.dumps(value_data))  # 5 minute TTL
                
                # Store in time series (simplified)
                time_bucket = int(metric_data.timestamp.timestamp() // 60)  # 1-minute buckets
                pipeline.hset(f"{key}:timeseries", time_bucket, metric_data.value)
                pipeline.expire(f"{key}:timeseries", self.config.retention_period_hours * 3600)
            
            await pipeline.execute()
            
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
    
    async def _export_metrics(self):
        """Export metrics to configured targets"""
        
        self.stats["export_attempts"] += 1
        
        try:
            # Push to Prometheus gateway if configured
            if (self.config.prometheus_pushgateway_url and 
                PROMETHEUS_AVAILABLE and 
                self.prometheus_registry):
                
                try:
                    push_to_gateway(
                        self.config.prometheus_pushgateway_url,
                        job=self.config.service_name,
                        registry=self.prometheus_registry,
                        grouping_key={
                            "instance": f"{self.config.service_name}-{self.config.environment}"
                        }
                    )
                except Exception as e:
                    logger.error(f"Error pushing to Prometheus gateway: {e}")
                    self.stats["export_failures"] += 1
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            self.stats["export_failures"] += 1
    
    # Metric recording methods
    
    def increment_counter(self, name: str, amount: float = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        try:
            if self.config.enable_prometheus and name in self.prometheus_metrics:
                prometheus_metric = self.prometheus_metrics[name]
                if labels:
                    prometheus_metric.labels(**labels).inc(amount)
                else:
                    prometheus_metric.inc(amount)
        except Exception as e:
            logger.debug(f"Error incrementing counter {name}: {e}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        try:
            if self.config.enable_prometheus and name in self.prometheus_metrics:
                prometheus_metric = self.prometheus_metrics[name]
                if labels:
                    prometheus_metric.labels(**labels).set(value)
                else:
                    prometheus_metric.set(value)
        except Exception as e:
            logger.debug(f"Error setting gauge {name}: {e}")
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value in a histogram metric"""
        try:
            if self.config.enable_prometheus and name in self.prometheus_metrics:
                prometheus_metric = self.prometheus_metrics[name]
                if labels:
                    prometheus_metric.labels(**labels).observe(value)
                else:
                    prometheus_metric.observe(value)
        except Exception as e:
            logger.debug(f"Error observing histogram {name}: {e}")
    
    @asynccontextmanager
    async def time_histogram(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to time an operation and record in histogram"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe_histogram(name, duration, labels)
    
    def record_custom_metric(self, name: str, value: float, 
                           labels: Optional[Dict[str, str]] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Record a custom metric value"""
        
        metric_data = MetricDataPoint(
            name=name,
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        # Store in history
        self.metric_history[name].append(metric_data)
        
        # Update Prometheus if registered
        if self.config.enable_prometheus and name in self.prometheus_metrics:
            try:
                prometheus_metric = self.prometheus_metrics[name]
                
                if hasattr(prometheus_metric, 'set'):
                    if labels:
                        prometheus_metric.labels(**labels).set(value)
                    else:
                        prometheus_metric.set(value)
                elif hasattr(prometheus_metric, 'observe'):
                    if labels:
                        prometheus_metric.labels(**labels).observe(value)
                    else:
                        prometheus_metric.observe(value)
                elif hasattr(prometheus_metric, 'inc'):
                    if labels:
                        prometheus_metric.labels(**labels).inc(value)
                    else:
                        prometheus_metric.inc(value)
            
            except Exception as e:
                logger.debug(f"Error updating Prometheus metric {name}: {e}")
    
    async def get_metric_data(self, name: str, 
                            time_range_minutes: int = 60) -> List[MetricDataPoint]:
        """Get historical metric data"""
        
        if name not in self.metric_history:
            return []
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_range_minutes)
        recent_data = [
            point for point in self.metric_history[name]
            if point.timestamp >= cutoff_time
        ]
        
        return recent_data
    
    async def get_metric_summary(self, name: str, 
                               time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get metric summary statistics"""
        
        data_points = await self.get_metric_data(name, time_range_minutes)
        
        if not data_points:
            return {"error": f"No data available for metric '{name}'"}
        
        values = [point.value for point in data_points]
        
        return {
            "metric_name": name,
            "time_range_minutes": time_range_minutes,
            "sample_count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0,
            "latest": values[-1] if values else None,
            "first_timestamp": data_points[0].timestamp.isoformat() if data_points else None,
            "last_timestamp": data_points[-1].timestamp.isoformat() if data_points else None
        }
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system metrics overview"""
        
        overview = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service_info": {
                "name": self.config.service_name,
                "version": self.config.service_version,
                "environment": self.config.environment
            },
            "collection_stats": self.stats.copy(),
            "metrics_available": len(self.metric_history),
            "custom_metrics": len(self.custom_metrics),
            "system_metrics": {},
            "performance_metrics": {}
        }
        
        # Get latest system metrics
        system_metric_names = [
            "system_cpu_usage_percent",
            "system_memory_usage_percent",
            "system_disk_usage_percent",
            "process_cpu_usage_percent",
            "process_memory_rss_bytes"
        ]
        
        for metric_name in system_metric_names:
            if metric_name in self.metric_history and self.metric_history[metric_name]:
                latest_point = self.metric_history[metric_name][-1]
                overview["system_metrics"][metric_name] = {
                    "value": latest_point.value,
                    "timestamp": latest_point.timestamp.isoformat()
                }
        
        # Get performance metrics summaries
        performance_metrics = [
            "http_request_duration_seconds",
            "database_query_duration_seconds",
            "task_processing_duration_seconds"
        ]
        
        for metric_name in performance_metrics:
            if metric_name in self.metric_history:
                summary = await self.get_metric_summary(metric_name, 15)  # Last 15 minutes
                if "error" not in summary:
                    overview["performance_metrics"][metric_name] = {
                        "mean": summary["mean"],
                        "median": summary["median"],
                        "p95": summary.get("p95", summary["max"]),  # Simplified
                        "sample_count": summary["sample_count"]
                    }
        
        return overview
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        if not PROMETHEUS_AVAILABLE or not self.prometheus_registry:
            return "# Prometheus not available\n"
        
        return generate_latest(self.prometheus_registry).decode('utf-8')


# Global metrics collector instance
metrics_collector: Optional[MetricsCollector] = None


def initialize_metrics(config: MetricsConfig, redis_client: Optional[aioredis.Redis] = None):
    """Initialize the metrics collection system"""
    global metrics_collector
    
    metrics_collector = MetricsCollector(config, redis_client)
    logger.info("✅ Metrics collection system initialized")


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    if metrics_collector is None:
        raise RuntimeError("Metrics system not initialized")
    return metrics_collector


async def start_metrics_collection():
    """Start metrics collection"""
    if metrics_collector:
        await metrics_collector.start_collection()


async def stop_metrics_collection():
    """Stop metrics collection"""
    if metrics_collector:
        await metrics_collector.stop_collection()


# Convenience functions for common metrics operations

def increment_counter(name: str, amount: float = 1, **labels):
    """Increment a counter metric"""
    if metrics_collector:
        metrics_collector.increment_counter(name, amount, labels or None)


def set_gauge(name: str, value: float, **labels):
    """Set a gauge metric"""
    if metrics_collector:
        metrics_collector.set_gauge(name, value, labels or None)


def observe_histogram(name: str, value: float, **labels):
    """Observe a histogram metric"""
    if metrics_collector:
        metrics_collector.observe_histogram(name, value, labels or None)


def record_metric(name: str, value: float, **labels):
    """Record a custom metric"""
    if metrics_collector:
        metrics_collector.record_custom_metric(name, value, labels or None)


@asynccontextmanager
async def time_operation(metric_name: str, **labels):
    """Time an operation and record in histogram"""
    if metrics_collector:
        async with metrics_collector.time_histogram(metric_name, labels or None):
            yield
    else:
        yield


# Decorator for automatic function timing
def time_function(metric_name: Optional[str] = None, **labels):
    """Decorator to automatically time function execution"""
    
    def decorator(func: Callable) -> Callable:
        nonlocal metric_name
        if metric_name is None:
            metric_name = f"function_duration_seconds"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with time_operation(metric_name, function=func.__name__, **labels):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    observe_histogram(metric_name, duration, function=func.__name__, **labels)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    observe_histogram(metric_name, duration, function=func.__name__, status="error", **labels)
                    raise
            return sync_wrapper
    
    return decorator