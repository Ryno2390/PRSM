"""
PRSM Database Performance Monitoring
Real-time database performance monitoring, alerting, and analytics dashboard
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import time
import statistics
import json
import logging
from collections import defaultdict, deque
import redis.asyncio as aioredis
from .database_optimization import get_connection_pool, QueryMetrics, ConnectionPoolMetrics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Database metric types"""
    QUERY_PERFORMANCE = "query_performance"
    CONNECTION_POOL = "connection_pool"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    LATENCY = "latency"


@dataclass
class DatabaseAlert:
    """Database performance alert"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold: float
    database_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceMetric:
    """Database performance metric"""
    metric_id: str
    metric_type: MetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    database_name: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlowQueryAnalysis:
    """Slow query analysis result"""
    query_hash: str
    query_text: str
    avg_execution_time_ms: float
    max_execution_time_ms: float
    execution_count: int
    first_seen: datetime
    last_seen: datetime
    optimization_suggestions: List[str] = field(default_factory=list)
    impact_score: float = 0.0  # 0-100 score based on frequency and slowness


class DatabaseMonitor:
    """Comprehensive database performance monitor"""
    
    def __init__(self, redis_client: aioredis.Redis, 
                 alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None):
        self.redis = redis_client
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 30  # seconds
        
        # Metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, DatabaseAlert] = {}
        self.alert_handlers: List[Callable] = []
        
        # Query analysis
        self.slow_query_threshold_ms = 1000
        self.slow_queries: Dict[str, SlowQueryAnalysis] = {}
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        self.anomaly_detection_enabled = True
        
        # Statistics
        self.stats = {
            "metrics_collected": 0,
            "alerts_generated": 0,
            "slow_queries_detected": 0,
            "monitoring_uptime_seconds": 0
        }
    
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default alert thresholds"""
        return {
            "query_performance": {
                "avg_execution_time_ms": 2000,
                "max_execution_time_ms": 10000,
                "queries_per_second": 1000
            },
            "connection_pool": {
                "active_connections_ratio": 0.8,  # 80% of max connections
                "checkout_time_ms": 100,
                "pool_exhaustion_ratio": 0.95
            },
            "resource_usage": {
                "memory_usage_mb": 2048,
                "cpu_usage_percent": 80,
                "disk_io_mb_per_sec": 100
            },
            "error_rate": {
                "error_rate_percent": 5.0,
                "connection_failures_per_minute": 10
            }
        }
    
    async def start_monitoring(self):
        """Start database performance monitoring"""
        if self.monitoring_active:
            logger.warning("Database monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.stats["monitoring_start_time"] = time.time()
        
        logger.info("✅ Database performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop database performance monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Database performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        start_time = time.time()
        
        while self.monitoring_active:
            try:
                loop_start = time.time()
                
                # Collect metrics
                await self._collect_performance_metrics()
                await self._analyze_slow_queries()
                await self._check_alert_conditions()
                await self._update_performance_baselines()
                
                # Update uptime
                self.stats["monitoring_uptime_seconds"] = time.time() - start_time
                
                # Sleep until next interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.monitoring_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_performance_metrics(self):
        """Collect database performance metrics"""
        try:
            connection_pool = get_connection_pool()
            
            # Get pool statistics
            pool_stats = await connection_pool.get_pool_statistics()
            
            # Collect connection pool metrics
            await self._collect_pool_metrics(pool_stats)
            
            # Collect query performance metrics
            await self._collect_query_metrics()
            
            # Store metrics in Redis for historical analysis
            await self._store_metrics_in_redis()
            
            self.stats["metrics_collected"] += 1
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    async def _collect_pool_metrics(self, pool_stats: Dict[str, Any]):
        """Collect connection pool metrics"""
        timestamp = datetime.now(timezone.utc)
        
        for pool_name, stats in pool_stats["pools"].items():
            # Active connections ratio
            active_ratio = stats["active_connections"] / stats["max_connections"]
            
            metric = PerformanceMetric(
                metric_id=f"pool_active_ratio_{pool_name}",
                metric_type=MetricType.CONNECTION_POOL,
                name="active_connections_ratio",
                value=active_ratio,
                unit="ratio",
                timestamp=timestamp,
                database_name=pool_name,
                tags={"role": stats["role"]},
                metadata=stats
            )
            
            self.metrics_history[f"pool_active_ratio_{pool_name}"].append(metric)
            
            # Average checkout time
            checkout_metric = PerformanceMetric(
                metric_id=f"pool_checkout_time_{pool_name}",
                metric_type=MetricType.CONNECTION_POOL,
                name="avg_checkout_time_ms",
                value=stats["average_checkout_time_ms"],
                unit="milliseconds",
                timestamp=timestamp,
                database_name=pool_name,
                tags={"role": stats["role"]}
            )
            
            self.metrics_history[f"pool_checkout_time_{pool_name}"].append(checkout_metric)
    
    async def _collect_query_metrics(self):
        """Collect query performance metrics"""
        try:
            # Get slow queries from Redis
            slow_queries_data = await self.redis.lrange("slow_queries", 0, -1)
            
            if not slow_queries_data:
                return
            
            timestamp = datetime.now(timezone.utc)
            execution_times = []
            
            for query_data in slow_queries_data[-100:]:  # Last 100 slow queries
                try:
                    query_info = json.loads(query_data)
                    execution_times.append(query_info["execution_time_ms"])
                except json.JSONDecodeError:
                    continue
            
            if execution_times:
                # Average query time metric
                avg_time_metric = PerformanceMetric(
                    metric_id="avg_query_time",
                    metric_type=MetricType.QUERY_PERFORMANCE,
                    name="avg_execution_time_ms",
                    value=statistics.mean(execution_times),
                    unit="milliseconds",
                    timestamp=timestamp,
                    database_name="prsm"
                )
                
                self.metrics_history["avg_query_time"].append(avg_time_metric)
                
                # Max query time metric
                max_time_metric = PerformanceMetric(
                    metric_id="max_query_time",
                    metric_type=MetricType.QUERY_PERFORMANCE,
                    name="max_execution_time_ms",
                    value=max(execution_times),
                    unit="milliseconds",
                    timestamp=timestamp,
                    database_name="prsm"
                )
                
                self.metrics_history["max_query_time"].append(max_time_metric)
        
        except Exception as e:
            logger.error(f"Error collecting query metrics: {e}")
    
    async def _analyze_slow_queries(self):
        """Analyze slow queries for patterns and optimization opportunities"""
        try:
            # Get recent slow queries
            slow_queries_data = await self.redis.lrange("slow_queries", 0, 999)
            
            if not slow_queries_data:
                return
            
            # Group queries by hash
            query_groups = defaultdict(list)
            
            for query_data in slow_queries_data:
                try:
                    query_info = json.loads(query_data)
                    query_hash = query_info["query_hash"]
                    query_groups[query_hash].append(query_info)
                except json.JSONDecodeError:
                    continue
            
            # Analyze each query group
            for query_hash, executions in query_groups.items():
                if len(executions) < 2:  # Need at least 2 executions for analysis
                    continue
                
                execution_times = [e["execution_time_ms"] for e in executions]
                
                # Create or update slow query analysis
                if query_hash not in self.slow_queries:
                    first_execution = min(executions, key=lambda x: x["timestamp"])
                    
                    self.slow_queries[query_hash] = SlowQueryAnalysis(
                        query_hash=query_hash,
                        query_text="",  # Would need to store actual query text
                        avg_execution_time_ms=statistics.mean(execution_times),
                        max_execution_time_ms=max(execution_times),
                        execution_count=len(executions),
                        first_seen=datetime.fromisoformat(first_execution["timestamp"]),
                        last_seen=datetime.fromisoformat(executions[-1]["timestamp"])
                    )
                else:
                    analysis = self.slow_queries[query_hash]
                    analysis.avg_execution_time_ms = statistics.mean(execution_times)
                    analysis.max_execution_time_ms = max(execution_times)
                    analysis.execution_count = len(executions)
                    analysis.last_seen = datetime.fromisoformat(executions[-1]["timestamp"])
                
                # Calculate impact score (frequency * average time)
                analysis = self.slow_queries[query_hash]
                frequency_score = min(analysis.execution_count / 100, 1.0) * 50
                time_score = min(analysis.avg_execution_time_ms / 5000, 1.0) * 50
                analysis.impact_score = frequency_score + time_score
                
                # Generate optimization suggestions
                analysis.optimization_suggestions = self._generate_optimization_suggestions(analysis)
            
            self.stats["slow_queries_detected"] = len(self.slow_queries)
            
        except Exception as e:
            logger.error(f"Error analyzing slow queries: {e}")
    
    def _generate_optimization_suggestions(self, analysis: SlowQueryAnalysis) -> List[str]:
        """Generate optimization suggestions for slow query"""
        suggestions = []
        
        if analysis.avg_execution_time_ms > 5000:
            suggestions.append("Query is extremely slow - consider complete rewrite or caching")
        elif analysis.avg_execution_time_ms > 2000:
            suggestions.append("Query is slow - review indexes and query plan")
        
        if analysis.execution_count > 100:
            suggestions.append("Frequently executed query - consider caching results")
        
        if analysis.impact_score > 80:
            suggestions.append("High-impact query - prioritize optimization")
        
        return suggestions
    
    async def _check_alert_conditions(self):
        """Check metrics against alert thresholds"""
        current_time = datetime.now(timezone.utc)
        
        # Check query performance alerts
        await self._check_query_performance_alerts(current_time)
        
        # Check connection pool alerts
        await self._check_connection_pool_alerts(current_time)
        
        # Check for anomalies if enabled
        if self.anomaly_detection_enabled:
            await self._check_anomalies(current_time)
    
    async def _check_query_performance_alerts(self, timestamp: datetime):
        """Check query performance for alert conditions"""
        try:
            # Check average execution time
            if "avg_query_time" in self.metrics_history:
                recent_metrics = list(self.metrics_history["avg_query_time"])[-10:]  # Last 10 metrics
                if recent_metrics:
                    avg_time = statistics.mean([m.value for m in recent_metrics])
                    threshold = self.alert_thresholds["query_performance"]["avg_execution_time_ms"]
                    
                    if avg_time > threshold:
                        await self._create_alert(
                            alert_type="slow_queries",
                            severity=AlertSeverity.WARNING,
                            message=f"Average query execution time ({avg_time:.1f}ms) exceeds threshold ({threshold}ms)",
                            metric_value=avg_time,
                            threshold=threshold,
                            database_name="prsm",
                            timestamp=timestamp
                        )
            
            # Check for query execution spikes
            if "max_query_time" in self.metrics_history:
                recent_metrics = list(self.metrics_history["max_query_time"])[-5:]
                if recent_metrics:
                    max_time = max([m.value for m in recent_metrics])
                    threshold = self.alert_thresholds["query_performance"]["max_execution_time_ms"]
                    
                    if max_time > threshold:
                        await self._create_alert(
                            alert_type="query_spike",
                            severity=AlertSeverity.CRITICAL,
                            message=f"Query execution spike detected ({max_time:.1f}ms)",
                            metric_value=max_time,
                            threshold=threshold,
                            database_name="prsm",
                            timestamp=timestamp
                        )
        
        except Exception as e:
            logger.error(f"Error checking query performance alerts: {e}")
    
    async def _check_connection_pool_alerts(self, timestamp: datetime):
        """Check connection pool metrics for alert conditions"""
        try:
            connection_pool = get_connection_pool()
            pool_stats = await connection_pool.get_pool_statistics()
            
            for pool_name, stats in pool_stats["pools"].items():
                # Check active connections ratio
                active_ratio = stats["active_connections"] / stats["max_connections"]
                threshold = self.alert_thresholds["connection_pool"]["active_connections_ratio"]
                
                if active_ratio > threshold:
                    await self._create_alert(
                        alert_type="pool_exhaustion",
                        severity=AlertSeverity.WARNING,
                        message=f"Connection pool {pool_name} usage high ({active_ratio:.1%})",
                        metric_value=active_ratio,
                        threshold=threshold,
                        database_name=pool_name,
                        timestamp=timestamp
                    )
                
                # Check checkout time
                checkout_time = stats["average_checkout_time_ms"]
                checkout_threshold = self.alert_thresholds["connection_pool"]["checkout_time_ms"]
                
                if checkout_time > checkout_threshold:
                    await self._create_alert(
                        alert_type="slow_connection_checkout",
                        severity=AlertSeverity.WARNING,
                        message=f"Slow connection checkout in {pool_name} ({checkout_time:.1f}ms)",
                        metric_value=checkout_time,
                        threshold=checkout_threshold,
                        database_name=pool_name,
                        timestamp=timestamp
                    )
        
        except Exception as e:
            logger.error(f"Error checking connection pool alerts: {e}")
    
    async def _check_anomalies(self, timestamp: datetime):
        """Check for performance anomalies using baseline comparison"""
        try:
            for metric_name, metrics in self.metrics_history.items():
                if len(metrics) < 10:  # Need sufficient data for anomaly detection
                    continue
                
                recent_values = [m.value for m in list(metrics)[-10:]]
                historical_values = [m.value for m in list(metrics)[:-10]]
                
                if not historical_values:
                    continue
                
                # Calculate baseline statistics
                baseline_mean = statistics.mean(historical_values)
                baseline_stdev = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
                
                # Check for anomalies (values outside 2 standard deviations)
                current_mean = statistics.mean(recent_values)
                
                if baseline_stdev > 0:
                    z_score = abs(current_mean - baseline_mean) / baseline_stdev
                    
                    if z_score > 2:  # Anomaly detected
                        await self._create_alert(
                            alert_type="performance_anomaly",
                            severity=AlertSeverity.INFO,
                            message=f"Performance anomaly detected in {metric_name} (z-score: {z_score:.2f})",
                            metric_value=current_mean,
                            threshold=baseline_mean + 2 * baseline_stdev,
                            database_name="prsm",
                            timestamp=timestamp,
                            metadata={"z_score": z_score, "baseline_mean": baseline_mean}
                        )
        
        except Exception as e:
            logger.error(f"Error checking anomalies: {e}")
    
    async def _create_alert(self, alert_type: str, severity: AlertSeverity,
                          message: str, metric_value: float, threshold: float,
                          database_name: str, timestamp: datetime,
                          metadata: Optional[Dict[str, Any]] = None):
        """Create and process database alert"""
        
        alert_id = f"{alert_type}_{database_name}_{int(timestamp.timestamp())}"
        
        # Check if similar alert already exists (avoid spam)
        existing_alert_key = f"{alert_type}_{database_name}"
        if existing_alert_key in self.active_alerts:
            existing_alert = self.active_alerts[existing_alert_key]
            if not existing_alert.resolved and (timestamp - existing_alert.timestamp).seconds < 300:
                return  # Skip duplicate alert within 5 minutes
        
        alert = DatabaseAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_value=metric_value,
            threshold=threshold,
            database_name=database_name,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        self.active_alerts[existing_alert_key] = alert
        
        # Store alert in Redis
        await self.redis.lpush("database_alerts", json.dumps({
            "alert_id": alert_id,
            "alert_type": alert_type,
            "severity": severity.value,
            "message": message,
            "metric_value": metric_value,
            "threshold": threshold,
            "database_name": database_name,
            "timestamp": timestamp.isoformat(),
            "metadata": metadata or {}
        }))
        await self.redis.ltrim("database_alerts", 0, 999)  # Keep last 1000 alerts
        
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
        logger.warning(f"Database alert: {message}")
    
    async def _update_performance_baselines(self):
        """Update performance baselines for anomaly detection"""
        try:
            for metric_name, metrics in self.metrics_history.items():
                if len(metrics) >= 100:  # Need sufficient data
                    values = [m.value for m in metrics]
                    self.performance_baselines[metric_name] = statistics.mean(values)
        
        except Exception as e:
            logger.error(f"Error updating performance baselines: {e}")
    
    async def _store_metrics_in_redis(self):
        """Store current metrics in Redis for external monitoring"""
        try:
            metrics_data = {}
            
            for metric_name, metrics in self.metrics_history.items():
                if metrics:
                    latest_metric = metrics[-1]
                    metrics_data[metric_name] = {
                        "value": latest_metric.value,
                        "unit": latest_metric.unit,
                        "timestamp": latest_metric.timestamp.isoformat(),
                        "database_name": latest_metric.database_name
                    }
            
            if metrics_data:
                await self.redis.setex(
                    "database_metrics",
                    300,  # 5 minute TTL
                    json.dumps(metrics_data)
                )
        
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
    
    def add_alert_handler(self, handler: Callable[[DatabaseAlert], Any]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    async def get_performance_summary(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time range"""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_range_minutes)
        summary = {
            "time_range_minutes": time_range_minutes,
            "metrics": {},
            "slow_queries": [],
            "active_alerts": [],
            "statistics": self.stats.copy()
        }
        
        # Summarize metrics
        for metric_name, metrics in self.metrics_history.items():
            recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary["metrics"][metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": statistics.mean(values),
                    "latest": values[-1],
                    "unit": recent_metrics[-1].unit
                }
        
        # Top slow queries by impact score
        top_slow_queries = sorted(
            self.slow_queries.values(),
            key=lambda q: q.impact_score,
            reverse=True
        )[:10]
        
        summary["slow_queries"] = [
            {
                "query_hash": q.query_hash,
                "avg_execution_time_ms": q.avg_execution_time_ms,
                "execution_count": q.execution_count,
                "impact_score": q.impact_score,
                "optimization_suggestions": q.optimization_suggestions
            }
            for q in top_slow_queries
        ]
        
        # Active alerts
        summary["active_alerts"] = [
            {
                "alert_type": alert.alert_type,
                "severity": alert.severity.value,
                "message": alert.message,
                "database_name": alert.database_name,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in self.active_alerts.values()
            if not alert.resolved
        ]
        
        return summary
    
    async def get_query_analysis(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis for specific query"""
        
        if query_hash not in self.slow_queries:
            return None
        
        analysis = self.slow_queries[query_hash]
        
        return {
            "query_hash": analysis.query_hash,
            "query_text": analysis.query_text,
            "performance": {
                "avg_execution_time_ms": analysis.avg_execution_time_ms,
                "max_execution_time_ms": analysis.max_execution_time_ms,
                "execution_count": analysis.execution_count
            },
            "timeline": {
                "first_seen": analysis.first_seen.isoformat(),
                "last_seen": analysis.last_seen.isoformat()
            },
            "impact_score": analysis.impact_score,
            "optimization_suggestions": analysis.optimization_suggestions
        }


# Global database monitor instance
database_monitor: Optional[DatabaseMonitor] = None


async def initialize_database_monitoring(redis_client: aioredis.Redis,
                                       alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None):
    """Initialize database performance monitoring"""
    global database_monitor
    
    database_monitor = DatabaseMonitor(redis_client, alert_thresholds)
    await database_monitor.start_monitoring()
    
    logger.info("✅ Database performance monitoring initialized")


def get_database_monitor() -> DatabaseMonitor:
    """Get the global database monitor instance"""
    if database_monitor is None:
        raise RuntimeError("Database monitoring not initialized.")
    return database_monitor


async def shutdown_database_monitoring():
    """Shutdown database monitoring"""
    if database_monitor:
        await database_monitor.stop_monitoring()


# Alert handler examples
async def log_alert_handler(alert: DatabaseAlert):
    """Log database alerts"""
    logger.warning(f"DATABASE ALERT [{alert.severity.value.upper()}]: {alert.message}")


async def slack_alert_handler(alert: DatabaseAlert):
    """Send database alerts to Slack (placeholder)"""
    # This would integrate with Slack API
    pass


async def email_alert_handler(alert: DatabaseAlert):
    """Send database alerts via email (placeholder)"""
    # This would integrate with email service
    pass