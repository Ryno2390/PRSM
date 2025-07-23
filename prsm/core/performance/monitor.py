"""
Performance Monitor
===================

Central performance monitoring system with real-time metrics collection
and alerting capabilities.
"""

import asyncio
import logging
import psutil
import threading
import time
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
from enum import Enum
import weakref
import gc

logger = logging.getLogger(__name__)

T = TypeVar('T')

class MetricType(Enum):
    """Types of performance metrics"""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)
    component: Optional[str] = None

@dataclass
class SystemResources:
    """System resource utilization"""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    open_file_descriptors: int
    active_threads: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ComponentPerformance:
    """Performance metrics for a specific component"""
    component_name: str
    request_count: int = 0
    error_count: int = 0
    avg_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    throughput_per_second: float = 0.0
    active_operations: int = 0
    memory_usage_mb: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    severity: AlertSeverity
    component: str
    metric: str
    message: str
    threshold_value: float
    current_value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False

class PerformanceMonitor:
    """Central performance monitoring system"""
    
    _instance: Optional['PerformanceMonitor'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'PerformanceMonitor':
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if hasattr(self, '_initialized'):
            return
        
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.component_metrics: Dict[str, ComponentPerformance] = {}
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.subscribers: weakref.WeakSet = weakref.WeakSet()
        
        # System monitoring
        self.system_resources: deque = deque(maxlen=1000)
        self.monitoring_enabled = True
        self.monitoring_interval = 10.0  # seconds
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self._operation_timers: Dict[str, float] = {}
        self._operation_counts: Dict[str, int] = defaultdict(int)
        
        self._initialized = True
        logger.info("Performance monitor initialized")
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        component: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a performance metric"""
        if not self.monitoring_enabled:
            return
        
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                metric_type=metric_type,
                component=component,
                labels=labels or {}
            )
            
            self.metrics[name].append(metric)
            
            # Check for alerts
            self._check_alert_conditions(metric)
            
            # Update component metrics
            if component:
                self._update_component_metrics(component, metric)
    
    def record_timer(self, name: str, duration_ms: float, component: Optional[str] = None) -> None:
        """Record timing metric"""
        self.record_metric(
            name=f"{name}_duration",
            value=duration_ms,
            metric_type=MetricType.TIMER,
            component=component
        )
        
        # Update operation counts
        with self._lock:
            self._operation_counts[name] += 1
    
    def start_timer(self, operation_name: str) -> str:
        """Start timing an operation"""
        timer_key = f"{operation_name}_{time.time()}"
        self._operation_timers[timer_key] = time.perf_counter()
        return timer_key
    
    def end_timer(self, timer_key: str, component: Optional[str] = None) -> float:
        """End timing and record duration"""
        if timer_key not in self._operation_timers:
            logger.warning(f"Timer key not found: {timer_key}")
            return 0.0
        
        start_time = self._operation_timers.pop(timer_key)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract operation name from timer key
        operation_name = timer_key.rsplit('_', 1)[0]
        self.record_timer(operation_name, duration_ms, component)
        
        return duration_ms
    
    def record_error(self, component: str, error_type: str, message: str) -> None:
        """Record an error metric"""
        self.record_metric(
            name="error_count",
            value=1,
            metric_type=MetricType.COUNTER,
            component=component,
            labels={"error_type": error_type, "message": message[:100]}
        )
        
        # Update component error count
        if component in self.component_metrics:
            self.component_metrics[component].error_count += 1
    
    def record_throughput(self, component: str, operations_count: int) -> None:
        """Record throughput metric"""
        self.record_metric(
            name="throughput",
            value=operations_count,
            metric_type=MetricType.COUNTER,
            component=component
        )
    
    def set_alert_threshold(
        self,
        metric_name: str,
        threshold_value: float,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        component: Optional[str] = None
    ) -> None:
        """Set alert threshold for a metric"""
        with self._lock:
            key = f"{component}:{metric_name}" if component else metric_name
            self.alert_thresholds[key] = {
                'threshold': threshold_value,
                'severity': severity,
                'component': component or 'system'
            }
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        component: Optional[str] = None,
        time_window_minutes: int = 60
    ) -> List[PerformanceMetric]:
        """Get performance metrics"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            if metric_name:
                metrics = list(self.metrics.get(metric_name, []))
            else:
                metrics = []
                for metric_list in self.metrics.values():
                    metrics.extend(metric_list)
            
            # Filter by time window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            # Filter by component
            if component:
                metrics = [m for m in metrics if m.component == component]
            
            return sorted(metrics, key=lambda m: m.timestamp)
    
    def get_system_resources(self) -> Optional[SystemResources]:
        """Get current system resource utilization"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_io_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_sent_mb = network_io.bytes_sent / (1024 * 1024) if network_io else 0
            network_recv_mb = network_io.bytes_recv / (1024 * 1024) if network_io else 0
            
            # Process info
            process = psutil.Process()
            open_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
            active_threads = process.num_threads()
            
            return SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                open_file_descriptors=open_fds,
                active_threads=active_threads
            )
            
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            return None
    
    def get_component_performance(self, component: str) -> Optional[ComponentPerformance]:
        """Get performance metrics for a specific component"""
        with self._lock:
            return self.component_metrics.get(component)
    
    def get_all_component_performance(self) -> Dict[str, ComponentPerformance]:
        """Get performance metrics for all components"""
        with self._lock:
            return dict(self.component_metrics)
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        metrics = self.get_metrics(time_window_minutes=time_window_minutes)
        system_resources = self.get_system_resources()
        component_performance = self.get_all_component_performance()
        
        # Calculate aggregated statistics
        if metrics:
            avg_response_times = defaultdict(list)
            error_counts = defaultdict(int)
            throughput_counts = defaultdict(int)
            
            for metric in metrics:
                if metric.name.endswith('_duration'):
                    component = metric.component or 'unknown'
                    avg_response_times[component].append(metric.value)
                elif metric.name == 'error_count':
                    component = metric.component or 'unknown'
                    error_counts[component] += metric.value
                elif metric.name == 'throughput':
                    component = metric.component or 'unknown'
                    throughput_counts[component] += metric.value
            
            # Calculate averages
            avg_response_by_component = {
                comp: sum(times) / len(times) 
                for comp, times in avg_response_times.items()
            }
        else:
            avg_response_by_component = {}
            error_counts = {}
            throughput_counts = {}
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "time_window_minutes": time_window_minutes,
            "system_resources": system_resources.__dict__ if system_resources else None,
            "total_metrics": len(metrics),
            "active_alerts": len([a for a in self.active_alerts.values() if not a.resolved]),
            "component_performance": {
                name: perf.__dict__ for name, perf in component_performance.items()
            },
            "average_response_times": avg_response_by_component,
            "error_counts": dict(error_counts),
            "throughput_counts": dict(throughput_counts)
        }
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active performance alerts"""
        with self._lock:
            return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a performance alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                return True
            return False
    
    def subscribe(self, callback: Callable[[str, Any], None]) -> None:
        """Subscribe to performance events"""
        self.subscribers.add(callback)
    
    def enable_monitoring(self) -> None:
        """Enable performance monitoring"""
        self.monitoring_enabled = True
        logger.info("Performance monitoring enabled")
    
    def disable_monitoring(self) -> None:
        """Disable performance monitoring"""
        self.monitoring_enabled = False
        logger.info("Performance monitoring disabled")
    
    async def start_background_monitoring(self) -> None:
        """Start background monitoring tasks"""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_system_resources())
        
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        logger.info("Started background performance monitoring")
    
    async def stop_background_monitoring(self) -> None:
        """Stop background monitoring tasks"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Wait for tasks to complete
        tasks = [self._monitoring_task, self._cleanup_task]
        tasks = [t for t in tasks if t is not None]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Stopped background performance monitoring")
    
    async def _monitor_system_resources(self) -> None:
        """Monitor system resources in background"""
        while self.monitoring_enabled:
            try:
                resources = self.get_system_resources()
                if resources:
                    with self._lock:
                        self.system_resources.append(resources)
                    
                    # Record as metrics
                    self.record_metric("system_cpu_percent", resources.cpu_percent)
                    self.record_metric("system_memory_percent", resources.memory_percent)
                    self.record_metric("system_memory_available_mb", resources.memory_available_mb)
                    self.record_metric("system_open_fds", resources.open_file_descriptors)
                    self.record_metric("system_threads", resources.active_threads)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System resource monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old data"""
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                with self._lock:
                    # Clean up resolved alerts older than 1 hour
                    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
                    resolved_alerts = [
                        alert_id for alert_id, alert in self.active_alerts.items()
                        if alert.resolved and alert.timestamp < cutoff_time
                    ]
                    
                    for alert_id in resolved_alerts:
                        del self.active_alerts[alert_id]
                    
                    if resolved_alerts:
                        logger.debug(f"Cleaned up {len(resolved_alerts)} resolved alerts")
                    
                    # Force garbage collection
                    gc.collect()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic cleanup error: {e}")
    
    def _check_alert_conditions(self, metric: PerformanceMetric) -> None:
        """Check if metric triggers any alerts"""
        metric_key = f"{metric.component}:{metric.name}" if metric.component else metric.name
        global_key = metric.name
        
        # Check component-specific threshold
        threshold_config = self.alert_thresholds.get(metric_key)
        if not threshold_config:
            # Check global threshold
            threshold_config = self.alert_thresholds.get(global_key)
        
        if not threshold_config:
            return
        
        threshold_value = threshold_config['threshold']
        severity = threshold_config['severity']
        
        # Check if threshold is exceeded
        if metric.value > threshold_value:
            alert_id = f"{metric_key}_{metric.timestamp.timestamp()}"
            
            if alert_id not in self.active_alerts:
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    severity=severity,
                    component=metric.component or 'system',
                    metric=metric.name,
                    message=f"{metric.name} exceeded threshold: {metric.value} > {threshold_value}",
                    threshold_value=threshold_value,
                    current_value=metric.value
                )
                
                self.active_alerts[alert_id] = alert
                
                # Notify subscribers
                self._notify_subscribers('alert_created', alert)
                
                logger.warning(f"Performance alert: {alert.message}")
    
    def _update_component_metrics(self, component: str, metric: PerformanceMetric) -> None:
        """Update component-specific performance metrics"""
        if component not in self.component_metrics:
            self.component_metrics[component] = ComponentPerformance(component_name=component)
        
        comp_metrics = self.component_metrics[component]
        
        # Update based on metric type
        if metric.name.endswith('_duration'):
            # Response time metric
            comp_metrics.request_count += 1
            
            # Update response time statistics
            if comp_metrics.avg_response_time_ms == 0:
                comp_metrics.avg_response_time_ms = metric.value
            else:
                # Exponential moving average
                alpha = 0.1
                comp_metrics.avg_response_time_ms = (
                    alpha * metric.value + 
                    (1 - alpha) * comp_metrics.avg_response_time_ms
                )
            
            comp_metrics.max_response_time_ms = max(comp_metrics.max_response_time_ms, metric.value)
            comp_metrics.min_response_time_ms = min(comp_metrics.min_response_time_ms, metric.value)
        
        comp_metrics.last_updated = metric.timestamp
    
    def _notify_subscribers(self, event_type: str, data: Any) -> None:
        """Notify subscribers of performance events"""
        for callback in list(self.subscribers):
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error notifying performance subscriber: {e}")

# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor