"""
PRSM Task Queue Monitoring and Alerting System
Comprehensive monitoring, alerting, and analytics for distributed task processing
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import time
import statistics
import logging
from collections import defaultdict, deque
import redis.asyncio as aioredis

from .task_queue import TaskQueue, TaskStatus, TaskPriority
from .task_worker import WorkerManager, WorkerStatus

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Task monitoring alert types"""
    QUEUE_BACKLOG = "queue_backlog"
    HIGH_FAILURE_RATE = "high_failure_rate"
    WORKER_DOWN = "worker_down"
    SLOW_PROCESSING = "slow_processing"
    MEMORY_PRESSURE = "memory_pressure"
    TASK_TIMEOUT = "task_timeout"
    QUEUE_STARVATION = "queue_starvation"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class TaskAlert:
    """Task monitoring alert"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    queue_name: Optional[str]
    worker_id: Optional[str]
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueMetrics:
    """Queue performance metrics"""
    queue_name: str
    pending_tasks: int
    processing_tasks: int
    completed_tasks: int
    failed_tasks: int
    avg_processing_time_ms: float
    throughput_per_minute: float
    failure_rate_percent: float
    oldest_pending_age_seconds: Optional[float]
    backlog_size: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WorkerClusterMetrics:
    """Worker cluster performance metrics"""
    total_workers: int
    active_workers: int
    idle_workers: int
    busy_workers: int
    failed_workers: int
    total_active_tasks: int
    avg_cpu_usage_percent: float
    avg_memory_usage_mb: float
    cluster_throughput_per_minute: float
    avg_task_duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TaskMonitor:
    """Comprehensive task queue and worker monitoring system"""
    
    def __init__(self, redis_client: aioredis.Redis,
                 alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None):
        self.redis = redis_client
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 30  # seconds
        
        # Metrics storage
        self.queue_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.worker_metrics_history: deque = deque(maxlen=100)
        self.active_alerts: Dict[str, TaskAlert] = {}
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        self.anomaly_detection_enabled = True
        
        # Statistics
        self.stats = {
            "monitoring_cycles": 0,
            "alerts_generated": 0,
            "alerts_resolved": 0,
            "queues_monitored": 0,
            "workers_monitored": 0
        }
    
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default monitoring thresholds"""
        return {
            "queue_backlog": {
                "max_pending_tasks": 1000,
                "max_oldest_task_age_minutes": 30,
                "max_backlog_growth_rate": 10  # tasks per minute
            },
            "failure_rate": {
                "max_failure_rate_percent": 10.0,
                "min_sample_size": 10
            },
            "processing_time": {
                "max_avg_processing_time_ms": 5000,
                "max_processing_time_increase_percent": 200
            },
            "worker_health": {
                "max_cpu_usage_percent": 90,
                "max_memory_usage_mb": 2048,
                "min_active_workers": 1
            },
            "throughput": {
                "min_throughput_per_minute": 1,
                "max_throughput_decrease_percent": 50
            }
        }
    
    async def start_monitoring(self, task_queues: Dict[str, TaskQueue],
                             worker_manager: Optional[WorkerManager] = None):
        """Start comprehensive monitoring"""
        
        if self.monitoring_active:
            logger.warning("Task monitoring is already active")
            return
        
        self.task_queues = task_queues
        self.worker_manager = worker_manager
        self.monitoring_active = True
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("✅ Task monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Task monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect queue metrics
                await self._collect_queue_metrics()
                
                # Collect worker metrics
                if self.worker_manager:
                    await self._collect_worker_metrics()
                
                # Check alert conditions
                await self._check_alert_conditions()
                
                # Update performance baselines
                await self._update_performance_baselines()
                
                # Store metrics in Redis
                await self._store_monitoring_data()
                
                self.stats["monitoring_cycles"] += 1
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_queue_metrics(self):
        """Collect metrics for all queues"""
        
        for queue_name, task_queue in self.task_queues.items():
            try:
                queue_stats = await task_queue.get_queue_stats()
                
                # Calculate additional metrics
                oldest_pending_age = await self._get_oldest_pending_task_age(queue_name)
                backlog_size = queue_stats["pending_tasks"]
                
                # Calculate throughput (tasks completed in last minute)
                throughput = await self._calculate_queue_throughput(queue_name)
                
                metrics = QueueMetrics(
                    queue_name=queue_name,
                    pending_tasks=queue_stats["pending_tasks"],
                    processing_tasks=queue_stats["processing_tasks"],
                    completed_tasks=queue_stats["completed_tasks"],
                    failed_tasks=queue_stats["failed_tasks"],
                    avg_processing_time_ms=queue_stats["avg_processing_time_ms"],
                    throughput_per_minute=throughput,
                    failure_rate_percent=100 - queue_stats["success_rate"],
                    oldest_pending_age_seconds=oldest_pending_age,
                    backlog_size=backlog_size
                )
                
                self.queue_metrics_history[queue_name].append(metrics)
                
                # Store current metrics in Redis for real-time access
                await self.redis.setex(
                    f"queue_metrics:{queue_name}",
                    300,  # 5 minute TTL
                    json.dumps({
                        "queue_name": queue_name,
                        "pending_tasks": metrics.pending_tasks,
                        "processing_tasks": metrics.processing_tasks,
                        "completed_tasks": metrics.completed_tasks,
                        "failed_tasks": metrics.failed_tasks,
                        "avg_processing_time_ms": metrics.avg_processing_time_ms,
                        "throughput_per_minute": metrics.throughput_per_minute,
                        "failure_rate_percent": metrics.failure_rate_percent,
                        "oldest_pending_age_seconds": metrics.oldest_pending_age_seconds,
                        "backlog_size": metrics.backlog_size,
                        "timestamp": metrics.timestamp.isoformat()
                    })
                )
                
            except Exception as e:
                logger.error(f"Error collecting metrics for queue {queue_name}: {e}")
        
        self.stats["queues_monitored"] = len(self.task_queues)
    
    async def _collect_worker_metrics(self):
        """Collect worker cluster metrics"""
        
        try:
            cluster_status = await self.worker_manager.get_cluster_status()
            cluster_data = cluster_status.get("cluster_metrics", {})
            
            metrics = WorkerClusterMetrics(
                total_workers=cluster_data.get("total_workers", 0),
                active_workers=cluster_data.get("active_workers", 0),
                idle_workers=0,  # Calculate based on worker states
                busy_workers=0,  # Calculate based on worker states
                failed_workers=0,  # Calculate based on worker states
                total_active_tasks=cluster_data.get("total_active_tasks", 0),
                avg_cpu_usage_percent=cluster_data.get("avg_cpu_usage", 0.0),
                avg_memory_usage_mb=cluster_data.get("avg_memory_usage", 0.0),
                cluster_throughput_per_minute=0.0,  # Calculate from worker stats
                avg_task_duration_ms=0.0  # Calculate from worker stats
            )
            
            # Calculate derived metrics from individual workers
            workers_data = cluster_status.get("workers", {})
            if workers_data:
                idle_count = 0
                busy_count = 0
                failed_count = 0
                throughput_sum = 0.0
                duration_sum = 0.0
                
                for worker_stats in workers_data.values():
                    status = worker_stats.get("status", "")
                    if status == "idle":
                        idle_count += 1
                    elif status == "busy":
                        busy_count += 1
                    elif status == "failed":
                        failed_count += 1
                    
                    performance = worker_stats.get("performance", {})
                    throughput_sum += performance.get("tasks_per_minute", 0.0)
                    duration_sum += performance.get("avg_duration_ms", 0.0)
                
                metrics.idle_workers = idle_count
                metrics.busy_workers = busy_count
                metrics.failed_workers = failed_count
                metrics.cluster_throughput_per_minute = throughput_sum
                metrics.avg_task_duration_ms = duration_sum / len(workers_data) if workers_data else 0.0
            
            self.worker_metrics_history.append(metrics)
            self.stats["workers_monitored"] = metrics.total_workers
            
        except Exception as e:
            logger.error(f"Error collecting worker metrics: {e}")
    
    async def _get_oldest_pending_task_age(self, queue_name: str) -> Optional[float]:
        """Get age of oldest pending task in seconds"""
        
        try:
            # Get the first task from pending queue
            pending_key = f"queue:{queue_name}:pending"
            oldest_task_data = await self.redis.lindex(pending_key, -1)  # Last item (oldest)
            
            if oldest_task_data:
                task_info = json.loads(oldest_task_data)
                created_at = datetime.fromisoformat(task_info["created_at"])
                age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
                return age_seconds
            
        except Exception as e:
            logger.debug(f"Error getting oldest task age for {queue_name}: {e}")
        
        return None
    
    async def _calculate_queue_throughput(self, queue_name: str) -> float:
        """Calculate queue throughput (tasks per minute)"""
        
        try:
            # Get recent metrics
            if queue_name in self.queue_metrics_history:
                recent_metrics = list(self.queue_metrics_history[queue_name])[-2:]  # Last 2 measurements
                
                if len(recent_metrics) >= 2:
                    time_diff_minutes = (
                        recent_metrics[-1].timestamp - recent_metrics[-2].timestamp
                    ).total_seconds() / 60
                    
                    if time_diff_minutes > 0:
                        completed_diff = (
                            recent_metrics[-1].completed_tasks - recent_metrics[-2].completed_tasks
                        )
                        return completed_diff / time_diff_minutes
            
        except Exception as e:
            logger.debug(f"Error calculating throughput for {queue_name}: {e}")
        
        return 0.0
    
    async def _check_alert_conditions(self):
        """Check all metrics against alert thresholds"""
        
        # Check queue alerts
        for queue_name, metrics_history in self.queue_metrics_history.items():
            if not metrics_history:
                continue
            
            latest_metrics = metrics_history[-1]
            await self._check_queue_alerts(queue_name, latest_metrics)
        
        # Check worker alerts
        if self.worker_metrics_history:
            latest_worker_metrics = self.worker_metrics_history[-1]
            await self._check_worker_alerts(latest_worker_metrics)
    
    async def _check_queue_alerts(self, queue_name: str, metrics: QueueMetrics):
        """Check queue-specific alerts"""
        
        # Queue backlog alert
        max_pending = self.alert_thresholds["queue_backlog"]["max_pending_tasks"]
        if metrics.pending_tasks > max_pending:
            await self._create_alert(
                alert_type=AlertType.QUEUE_BACKLOG,
                severity=AlertSeverity.WARNING,
                message=f"Queue '{queue_name}' has high backlog: {metrics.pending_tasks} tasks",
                queue_name=queue_name,
                metric_value=metrics.pending_tasks,
                threshold=max_pending
            )
        
        # Old task alert
        max_age_minutes = self.alert_thresholds["queue_backlog"]["max_oldest_task_age_minutes"]
        if metrics.oldest_pending_age_seconds:
            age_minutes = metrics.oldest_pending_age_seconds / 60
            if age_minutes > max_age_minutes:
                await self._create_alert(
                    alert_type=AlertType.QUEUE_STARVATION,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Oldest task in '{queue_name}' is {age_minutes:.1f} minutes old",
                    queue_name=queue_name,
                    metric_value=age_minutes,
                    threshold=max_age_minutes
                )
        
        # High failure rate alert
        max_failure_rate = self.alert_thresholds["failure_rate"]["max_failure_rate_percent"]
        min_sample = self.alert_thresholds["failure_rate"]["min_sample_size"]
        
        if (metrics.completed_tasks + metrics.failed_tasks) >= min_sample:
            if metrics.failure_rate_percent > max_failure_rate:
                await self._create_alert(
                    alert_type=AlertType.HIGH_FAILURE_RATE,
                    severity=AlertSeverity.CRITICAL,
                    message=f"High failure rate in '{queue_name}': {metrics.failure_rate_percent:.1f}%",
                    queue_name=queue_name,
                    metric_value=metrics.failure_rate_percent,
                    threshold=max_failure_rate
                )
        
        # Slow processing alert
        max_processing_time = self.alert_thresholds["processing_time"]["max_avg_processing_time_ms"]
        if metrics.avg_processing_time_ms > max_processing_time:
            await self._create_alert(
                alert_type=AlertType.SLOW_PROCESSING,
                severity=AlertSeverity.WARNING,
                message=f"Slow processing in '{queue_name}': {metrics.avg_processing_time_ms:.1f}ms avg",
                queue_name=queue_name,
                metric_value=metrics.avg_processing_time_ms,
                threshold=max_processing_time
            )
        
        # Low throughput alert
        min_throughput = self.alert_thresholds["throughput"]["min_throughput_per_minute"]
        if metrics.throughput_per_minute < min_throughput and metrics.pending_tasks > 0:
            await self._create_alert(
                alert_type=AlertType.SLOW_PROCESSING,
                severity=AlertSeverity.WARNING,
                message=f"Low throughput in '{queue_name}': {metrics.throughput_per_minute:.1f} tasks/min",
                queue_name=queue_name,
                metric_value=metrics.throughput_per_minute,
                threshold=min_throughput
            )
    
    async def _check_worker_alerts(self, metrics: WorkerClusterMetrics):
        """Check worker cluster alerts"""
        
        # Worker availability alert
        min_workers = self.alert_thresholds["worker_health"]["min_active_workers"]
        if metrics.active_workers < min_workers:
            await self._create_alert(
                alert_type=AlertType.WORKER_DOWN,
                severity=AlertSeverity.CRITICAL,
                message=f"Low worker availability: {metrics.active_workers} active workers",
                queue_name=None,
                metric_value=metrics.active_workers,
                threshold=min_workers
            )
        
        # High CPU usage alert
        max_cpu = self.alert_thresholds["worker_health"]["max_cpu_usage_percent"]
        if metrics.avg_cpu_usage_percent > max_cpu:
            await self._create_alert(
                alert_type=AlertType.MEMORY_PRESSURE,
                severity=AlertSeverity.WARNING,
                message=f"High CPU usage: {metrics.avg_cpu_usage_percent:.1f}%",
                queue_name=None,
                metric_value=metrics.avg_cpu_usage_percent,
                threshold=max_cpu
            )
        
        # High memory usage alert
        max_memory = self.alert_thresholds["worker_health"]["max_memory_usage_mb"]
        if metrics.avg_memory_usage_mb > max_memory:
            await self._create_alert(
                alert_type=AlertType.MEMORY_PRESSURE,
                severity=AlertSeverity.WARNING,
                message=f"High memory usage: {metrics.avg_memory_usage_mb:.1f}MB",
                queue_name=None,
                metric_value=metrics.avg_memory_usage_mb,
                threshold=max_memory
            )
    
    async def _create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                          message: str, queue_name: Optional[str],
                          metric_value: float, threshold: float,
                          worker_id: Optional[str] = None):
        """Create and process alert"""
        
        # Create unique alert key to prevent duplicates
        alert_key = f"{alert_type.value}_{queue_name or 'cluster'}_{worker_id or 'all'}"
        
        # Check if similar alert already exists (avoid spam)
        if alert_key in self.active_alerts:
            existing_alert = self.active_alerts[alert_key]
            if not existing_alert.resolved:
                # Update existing alert with new metric value
                existing_alert.metric_value = metric_value
                existing_alert.timestamp = datetime.now(timezone.utc)
                return
        
        # Create new alert
        alert_id = f"{alert_type.value}_{int(time.time())}"
        
        alert = TaskAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            queue_name=queue_name,
            worker_id=worker_id,
            metric_value=metric_value,
            threshold=threshold
        )
        
        self.active_alerts[alert_key] = alert
        
        # Store alert in Redis
        alert_data = {
            "alert_id": alert_id,
            "alert_type": alert_type.value,
            "severity": severity.value,
            "message": message,
            "queue_name": queue_name,
            "worker_id": worker_id,
            "metric_value": metric_value,
            "threshold": threshold,
            "timestamp": alert.timestamp.isoformat(),
            "resolved": False
        }
        
        await self.redis.lpush("task_alerts", json.dumps(alert_data))
        await self.redis.ltrim("task_alerts", 0, 999)  # Keep last 1000 alerts
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        self.stats["alerts_generated"] += 1
        logger.warning(f"Task alert: {message}")
    
    async def resolve_alert(self, alert_key: str) -> bool:
        """Manually resolve an alert"""
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_at = datetime.now(timezone.utc)
            
            # Update in Redis
            await self.redis.hset(f"alert:{alert.alert_id}", mapping={
                "resolved": True,
                "resolved_at": alert.resolved_at.isoformat()
            })
            
            self.stats["alerts_resolved"] += 1
            logger.info(f"Resolved alert: {alert.message}")
            return True
        
        return False
    
    async def _update_performance_baselines(self):
        """Update performance baselines for anomaly detection"""
        
        try:
            # Update queue baselines
            for queue_name, metrics_history in self.queue_metrics_history.items():
                if len(metrics_history) >= 10:  # Need sufficient data
                    recent_throughput = [m.throughput_per_minute for m in list(metrics_history)[-10:]]
                    recent_processing_time = [m.avg_processing_time_ms for m in list(metrics_history)[-10:]]
                    
                    self.performance_baselines[f"{queue_name}_throughput"] = statistics.mean(recent_throughput)
                    self.performance_baselines[f"{queue_name}_processing_time"] = statistics.mean(recent_processing_time)
            
            # Update worker baselines
            if len(self.worker_metrics_history) >= 10:
                recent_cpu = [m.avg_cpu_usage_percent for m in list(self.worker_metrics_history)[-10:]]
                recent_memory = [m.avg_memory_usage_mb for m in list(self.worker_metrics_history)[-10:]]
                
                self.performance_baselines["cluster_cpu"] = statistics.mean(recent_cpu)
                self.performance_baselines["cluster_memory"] = statistics.mean(recent_memory)
        
        except Exception as e:
            logger.error(f"Error updating performance baselines: {e}")
    
    async def _store_monitoring_data(self):
        """Store monitoring data in Redis"""
        
        try:
            # Store monitoring statistics
            monitoring_stats = {
                "monitoring_cycles": self.stats["monitoring_cycles"],
                "alerts_generated": self.stats["alerts_generated"],
                "alerts_resolved": self.stats["alerts_resolved"],
                "queues_monitored": self.stats["queues_monitored"],
                "workers_monitored": self.stats["workers_monitored"],
                "active_alerts": len([a for a in self.active_alerts.values() if not a.resolved]),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            await self.redis.setex(
                "task_monitoring_stats",
                300,  # 5 minute TTL
                json.dumps(monitoring_stats)
            )
        
        except Exception as e:
            logger.error(f"Error storing monitoring data: {e}")
    
    def add_alert_handler(self, handler: Callable[[TaskAlert], Any]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        
        dashboard = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_health": {
                "monitoring_active": self.monitoring_active,
                "monitoring_cycles": self.stats["monitoring_cycles"],
                "queues_monitored": self.stats["queues_monitored"],
                "workers_monitored": self.stats["workers_monitored"]
            },
            "alerts": {
                "total_generated": self.stats["alerts_generated"],
                "total_resolved": self.stats["alerts_resolved"],
                "active_count": len([a for a in self.active_alerts.values() if not a.resolved]),
                "active_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "type": alert.alert_type.value,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "queue_name": alert.queue_name,
                        "metric_value": alert.metric_value,
                        "threshold": alert.threshold,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in self.active_alerts.values()
                    if not alert.resolved
                ]
            },
            "queue_metrics": {},
            "worker_metrics": None
        }
        
        # Add latest queue metrics
        for queue_name, metrics_history in self.queue_metrics_history.items():
            if metrics_history:
                latest = metrics_history[-1]
                dashboard["queue_metrics"][queue_name] = {
                    "pending_tasks": latest.pending_tasks,
                    "processing_tasks": latest.processing_tasks,
                    "completed_tasks": latest.completed_tasks,
                    "failed_tasks": latest.failed_tasks,
                    "throughput_per_minute": latest.throughput_per_minute,
                    "failure_rate_percent": latest.failure_rate_percent,
                    "avg_processing_time_ms": latest.avg_processing_time_ms,
                    "backlog_size": latest.backlog_size,
                    "oldest_pending_age_seconds": latest.oldest_pending_age_seconds
                }
        
        # Add latest worker metrics
        if self.worker_metrics_history:
            latest_worker = self.worker_metrics_history[-1]
            dashboard["worker_metrics"] = {
                "total_workers": latest_worker.total_workers,
                "active_workers": latest_worker.active_workers,
                "idle_workers": latest_worker.idle_workers,
                "busy_workers": latest_worker.busy_workers,
                "failed_workers": latest_worker.failed_workers,
                "total_active_tasks": latest_worker.total_active_tasks,
                "avg_cpu_usage_percent": latest_worker.avg_cpu_usage_percent,
                "avg_memory_usage_mb": latest_worker.avg_memory_usage_mb,
                "cluster_throughput_per_minute": latest_worker.cluster_throughput_per_minute,
                "avg_task_duration_ms": latest_worker.avg_task_duration_ms
            }
        
        return dashboard
    
    async def get_queue_analytics(self, queue_name: str, 
                                time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get detailed analytics for a specific queue"""
        
        if queue_name not in self.queue_metrics_history:
            return {"error": f"No metrics available for queue '{queue_name}'"}
        
        metrics_history = list(self.queue_metrics_history[queue_name])
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_range_minutes)
        
        # Filter metrics by time range
        recent_metrics = [m for m in metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": f"No recent metrics available for queue '{queue_name}'"}
        
        # Calculate analytics
        throughput_values = [m.throughput_per_minute for m in recent_metrics]
        processing_times = [m.avg_processing_time_ms for m in recent_metrics]
        failure_rates = [m.failure_rate_percent for m in recent_metrics]
        backlog_sizes = [m.backlog_size for m in recent_metrics]
        
        analytics = {
            "queue_name": queue_name,
            "time_range_minutes": time_range_minutes,
            "sample_count": len(recent_metrics),
            "throughput": {
                "min": min(throughput_values),
                "max": max(throughput_values),
                "avg": statistics.mean(throughput_values),
                "current": throughput_values[-1] if throughput_values else 0
            },
            "processing_time": {
                "min": min(processing_times),
                "max": max(processing_times),
                "avg": statistics.mean(processing_times),
                "current": processing_times[-1] if processing_times else 0
            },
            "failure_rate": {
                "min": min(failure_rates),
                "max": max(failure_rates),
                "avg": statistics.mean(failure_rates),
                "current": failure_rates[-1] if failure_rates else 0
            },
            "backlog": {
                "min": min(backlog_sizes),
                "max": max(backlog_sizes),
                "avg": statistics.mean(backlog_sizes),
                "current": backlog_sizes[-1] if backlog_sizes else 0
            },
            "trends": {
                "throughput_trend": "stable",  # Could calculate actual trends
                "processing_time_trend": "stable",
                "failure_rate_trend": "stable",
                "backlog_trend": "stable"
            }
        }
        
        return analytics


# Global task monitor instance
task_monitor: Optional[TaskMonitor] = None


async def initialize_task_monitoring(redis_client: aioredis.Redis,
                                   alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None):
    """Initialize task monitoring system"""
    global task_monitor
    
    task_monitor = TaskMonitor(redis_client, alert_thresholds)
    logger.info("✅ Task monitoring system initialized")


def get_task_monitor() -> TaskMonitor:
    """Get the global task monitor instance"""
    if task_monitor is None:
        raise RuntimeError("Task monitoring not initialized")
    return task_monitor


async def start_task_monitoring(task_queues: Dict[str, TaskQueue],
                              worker_manager: Optional[WorkerManager] = None):
    """Start task monitoring"""
    if task_monitor:
        await task_monitor.start_monitoring(task_queues, worker_manager)


async def stop_task_monitoring():
    """Stop task monitoring"""
    if task_monitor:
        await task_monitor.stop_monitoring()


# Alert handler examples
async def log_alert_handler(alert: TaskAlert):
    """Log task alerts"""
    logger.warning(f"TASK ALERT [{alert.severity.value.upper()}]: {alert.message}")


async def slack_alert_handler(alert: TaskAlert):
    """Send task alerts to Slack (placeholder)"""
    # This would integrate with Slack API
    pass


async def email_alert_handler(alert: TaskAlert):
    """Send task alerts via email (placeholder)"""
    # This would integrate with email service
    pass