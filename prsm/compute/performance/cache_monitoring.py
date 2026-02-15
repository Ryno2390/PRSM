"""
PRSM Cache Performance Monitoring
Real-time cache metrics, performance analytics, and optimization recommendations
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import time
import statistics
from collections import defaultdict, deque
import redis.asyncio as aioredis
import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    HAS_PLOTLY = True
except ImportError:
    go = px = PlotlyJSONEncoder = None
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of cache metrics"""
    HIT_RATIO = "hit_ratio"
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    KEY_COUNT = "key_count"
    EVICTION_RATE = "eviction_rate"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CACHE_SIZE = "cache_size"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CacheMetric:
    """Cache performance metric"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    layer: str  # Cache layer (L1, L2, etc.)
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheAlert:
    """Cache performance alert"""
    alert_id: str
    alert_type: str
    level: AlertLevel
    message: str
    timestamp: datetime
    metric_type: MetricType
    current_value: float
    threshold_value: float
    layer: str
    recommendations: List[str] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class CacheHealthScore:
    """Overall cache health assessment"""
    overall_score: float  # 0-100
    component_scores: Dict[str, float]
    issues: List[str]
    recommendations: List[str]
    last_updated: datetime


class CacheMetricsCollector:
    """Collects cache metrics from multiple sources"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.collection_interval = 30  # seconds
        self.collection_task: Optional[asyncio.Task] = None
        
        # Metric thresholds for alerts
        self.alert_thresholds = {
            MetricType.HIT_RATIO: {"warning": 0.7, "critical": 0.5},
            MetricType.RESPONSE_TIME: {"warning": 100, "critical": 500},  # ms
            MetricType.MEMORY_USAGE: {"warning": 0.8, "critical": 0.95},  # ratio
            MetricType.ERROR_RATE: {"warning": 0.01, "critical": 0.05},  # ratio
            MetricType.EVICTION_RATE: {"warning": 100, "critical": 1000},  # per minute
        }
        
        # Alert state tracking
        self.active_alerts: Dict[str, CacheAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
    
    async def start_collection(self):
        """Start metrics collection background task"""
        if self.collection_task is None or self.collection_task.done():
            self.collection_task = asyncio.create_task(self._collection_loop())
            logger.info("✅ Cache metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        if self.collection_task and not self.collection_task.done():
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Cache metrics collection stopped")
    
    async def collect_metrics(self, cache_manager) -> List[CacheMetric]:
        """Collect current cache metrics"""
        
        metrics = []
        timestamp = datetime.now(timezone.utc)
        
        try:
            # Get comprehensive cache stats
            cache_stats = await cache_manager.get_comprehensive_stats()
            
            # Memory cache metrics
            if "memory_cache" in cache_stats:
                memory_stats = cache_stats["memory_cache"]
                
                metrics.append(CacheMetric(
                    metric_type=MetricType.HIT_RATIO,
                    value=memory_stats.get("hit_ratio", 0.0),
                    timestamp=timestamp,
                    layer="L1_MEMORY"
                ))
                
                metrics.append(CacheMetric(
                    metric_type=MetricType.MEMORY_USAGE,
                    value=memory_stats.get("memory_usage_bytes", 0),
                    timestamp=timestamp,
                    layer="L1_MEMORY"
                ))
                
                metrics.append(CacheMetric(
                    metric_type=MetricType.KEY_COUNT,
                    value=memory_stats.get("key_count", 0),
                    timestamp=timestamp,
                    layer="L1_MEMORY"
                ))
                
                metrics.append(CacheMetric(
                    metric_type=MetricType.EVICTION_RATE,
                    value=memory_stats.get("evictions", 0),
                    timestamp=timestamp,
                    layer="L1_MEMORY"
                ))
            
            # Redis cache metrics
            if "redis_cache" in cache_stats and cache_stats["redis_cache"]:
                redis_stats = cache_stats["redis_cache"]
                
                metrics.append(CacheMetric(
                    metric_type=MetricType.HIT_RATIO,
                    value=redis_stats.get("hit_ratio", 0.0),
                    timestamp=timestamp,
                    layer="L2_REDIS"
                ))
                
                metrics.append(CacheMetric(
                    metric_type=MetricType.RESPONSE_TIME,
                    value=redis_stats.get("avg_response_time_ms", 0.0),
                    timestamp=timestamp,
                    layer="L2_REDIS"
                ))
                
                metrics.append(CacheMetric(
                    metric_type=MetricType.KEY_COUNT,
                    value=redis_stats.get("key_count", 0),
                    timestamp=timestamp,
                    layer="L2_REDIS"
                ))
            
            # Redis cluster metrics
            if "redis_cluster_info" in cache_stats and cache_stats["redis_cluster_info"]:
                cluster_info = cache_stats["redis_cluster_info"]
                
                metrics.append(CacheMetric(
                    metric_type=MetricType.MEMORY_USAGE,
                    value=cluster_info.get("total_memory_usage", 0),
                    timestamp=timestamp,
                    layer="L2_REDIS",
                    metadata={"cluster_state": cluster_info.get("cluster_state")}
                ))
            
            # Calculate derived metrics
            throughput_metric = await self._calculate_throughput()
            if throughput_metric:
                metrics.append(throughput_metric)
            
            error_rate_metric = await self._calculate_error_rate()
            if error_rate_metric:
                metrics.append(error_rate_metric)
            
            # Store metrics in buffer
            for metric in metrics:
                metric_key = f"{metric.layer}:{metric.metric_type.value}"
                self.metrics_buffer[metric_key].append({
                    "value": metric.value,
                    "timestamp": metric.timestamp.timestamp(),
                    "metadata": metric.metadata
                })
            
            # Check for alerts
            await self._check_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting cache metrics: {e}")
            return []
    
    async def _collection_loop(self):
        """Background metrics collection loop"""
        
        while True:
            try:
                # Import here to avoid circular imports
                from .caching_system import get_cache_manager
                
                cache_manager = get_cache_manager()
                await self.collect_metrics(cache_manager)
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _calculate_throughput(self) -> Optional[CacheMetric]:
        """Calculate cache throughput (requests per second)"""
        
        try:
            # Get request counts from recent metrics
            recent_metrics = []
            
            for layer in ["L1_MEMORY", "L2_REDIS"]:
                hit_key = f"{layer}:{MetricType.HIT_RATIO.value}"
                if hit_key in self.metrics_buffer and len(self.metrics_buffer[hit_key]) >= 2:
                    recent_metrics.extend(list(self.metrics_buffer[hit_key])[-2:])
            
            if len(recent_metrics) < 2:
                return None
            
            # Calculate throughput based on metric collection frequency
            time_diff = recent_metrics[-1]["timestamp"] - recent_metrics[-2]["timestamp"]
            if time_diff <= 0:
                return None
            
            # Approximate throughput (this would be more accurate with actual request counters)
            throughput = 1.0 / time_diff  # Basic approximation
            
            return CacheMetric(
                metric_type=MetricType.THROUGHPUT,
                value=throughput,
                timestamp=datetime.now(timezone.utc),
                layer="COMBINED"
            )
            
        except Exception as e:
            logger.error(f"Error calculating throughput: {e}")
            return None
    
    async def _calculate_error_rate(self) -> Optional[CacheMetric]:
        """Calculate cache error rate"""
        
        try:
            # Count Redis connection errors in recent period
            error_count = 0
            total_operations = 0
            
            # This would typically track actual errors from cache operations
            # For now, use a simple approximation
            
            return CacheMetric(
                metric_type=MetricType.ERROR_RATE,
                value=error_count / max(total_operations, 1),
                timestamp=datetime.now(timezone.utc),
                layer="COMBINED"
            )
            
        except Exception as e:
            logger.error(f"Error calculating error rate: {e}")
            return None
    
    async def _check_alerts(self, metrics: List[CacheMetric]):
        """Check metrics against alert thresholds"""
        
        for metric in metrics:
            if metric.metric_type not in self.alert_thresholds:
                continue
            
            thresholds = self.alert_thresholds[metric.metric_type]
            alert_key = f"{metric.layer}:{metric.metric_type.value}"
            
            # Determine alert level
            alert_level = None
            threshold_value = None
            
            if metric.value <= thresholds.get("critical", float('inf')):
                alert_level = AlertLevel.CRITICAL
                threshold_value = thresholds["critical"]
            elif metric.value <= thresholds.get("warning", float('inf')):
                alert_level = AlertLevel.WARNING
                threshold_value = thresholds["warning"]
            
            # Handle special cases (higher values are worse)
            if metric.metric_type in [MetricType.RESPONSE_TIME, MetricType.ERROR_RATE, MetricType.EVICTION_RATE]:
                if metric.value >= thresholds.get("critical", float('inf')):
                    alert_level = AlertLevel.CRITICAL
                    threshold_value = thresholds["critical"]
                elif metric.value >= thresholds.get("warning", float('inf')):
                    alert_level = AlertLevel.WARNING
                    threshold_value = thresholds["warning"]
            
            if alert_level:
                # Create or update alert
                if alert_key not in self.active_alerts:
                    alert = CacheAlert(
                        alert_id=f"alert_{int(time.time())}_{alert_key}",
                        alert_type=f"{metric.metric_type.value}_threshold",
                        level=alert_level,
                        message=self._generate_alert_message(metric, alert_level, threshold_value),
                        timestamp=datetime.now(timezone.utc),
                        metric_type=metric.metric_type,
                        current_value=metric.value,
                        threshold_value=threshold_value,
                        layer=metric.layer,
                        recommendations=self._generate_recommendations(metric, alert_level)
                    )
                    
                    self.active_alerts[alert_key] = alert
                    self.alert_history.append(alert)
                    
                    logger.warning(f"Cache alert: {alert.message}")
                else:
                    # Update existing alert
                    alert = self.active_alerts[alert_key]
                    alert.current_value = metric.value
                    alert.level = alert_level
                    alert.message = self._generate_alert_message(metric, alert_level, threshold_value)
            
            else:
                # Resolve alert if it exists
                if alert_key in self.active_alerts:
                    alert = self.active_alerts[alert_key]
                    alert.resolved = True
                    alert.resolved_at = datetime.now(timezone.utc)
                    del self.active_alerts[alert_key]
                    
                    logger.info(f"Cache alert resolved: {alert.message}")
    
    def _generate_alert_message(self, metric: CacheMetric, 
                              level: AlertLevel, threshold: float) -> str:
        """Generate alert message"""
        
        messages = {
            MetricType.HIT_RATIO: f"Cache hit ratio is {level.value}: {metric.value:.2%} (threshold: {threshold:.2%})",
            MetricType.RESPONSE_TIME: f"Cache response time is {level.value}: {metric.value:.1f}ms (threshold: {threshold}ms)",
            MetricType.MEMORY_USAGE: f"Cache memory usage is {level.value}: {metric.value / (1024*1024):.1f}MB",
            MetricType.ERROR_RATE: f"Cache error rate is {level.value}: {metric.value:.2%} (threshold: {threshold:.2%})",
            MetricType.EVICTION_RATE: f"Cache eviction rate is {level.value}: {metric.value:.0f}/min (threshold: {threshold}/min)"
        }
        
        return messages.get(metric.metric_type, f"Cache metric {metric.metric_type.value} is {level.value}")
    
    def _generate_recommendations(self, metric: CacheMetric, level: AlertLevel) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = {
            MetricType.HIT_RATIO: [
                "Increase cache TTL for frequently accessed data",
                "Implement cache warming for popular keys",
                "Review cache key strategies and patterns",
                "Consider increasing cache size limits"
            ],
            MetricType.RESPONSE_TIME: [
                "Check Redis cluster health and connectivity",
                "Optimize serialization/deserialization processes",
                "Consider adding more Redis nodes",
                "Review network latency between application and cache"
            ],
            MetricType.MEMORY_USAGE: [
                "Implement more aggressive eviction policies",
                "Reduce cache TTL for less critical data",
                "Consider compression for large cache values",
                "Add more memory or scale cache cluster"
            ],
            MetricType.ERROR_RATE: [
                "Check Redis cluster connectivity",
                "Review error logs for connection issues",
                "Implement circuit breaker patterns",
                "Add retry mechanisms for cache operations"
            ],
            MetricType.EVICTION_RATE: [
                "Increase cache memory limits",
                "Optimize cache key expiration strategies",
                "Review data access patterns",
                "Consider implementing cache partitioning"
            ]
        }
        
        return recommendations.get(metric.metric_type, ["Review cache configuration and usage patterns"])
    
    def get_metrics_history(self, metric_type: MetricType, 
                          layer: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics data"""
        
        metric_key = f"{layer}:{metric_type.value}"
        if metric_key not in self.metrics_buffer:
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            metric_data for metric_data in self.metrics_buffer[metric_key]
            if metric_data["timestamp"] >= cutoff_time
        ]
    
    def get_active_alerts(self) -> List[CacheAlert]:
        """Get current active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[CacheAlert]:
        """Get alert history"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]


class CacheHealthMonitor:
    """Monitors overall cache health and provides optimization recommendations"""
    
    def __init__(self, metrics_collector: CacheMetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_weights = {
            MetricType.HIT_RATIO: 0.3,
            MetricType.RESPONSE_TIME: 0.25,
            MetricType.MEMORY_USAGE: 0.2,
            MetricType.ERROR_RATE: 0.15,
            MetricType.EVICTION_RATE: 0.1
        }
    
    async def calculate_health_score(self) -> CacheHealthScore:
        """Calculate overall cache health score"""
        
        component_scores = {}
        issues = []
        recommendations = []
        
        try:
            # Calculate component scores
            for metric_type, weight in self.health_weights.items():
                score = await self._calculate_component_score(metric_type)
                component_scores[metric_type.value] = score
                
                if score < 60:  # Poor score
                    issues.append(f"Poor {metric_type.value} performance (score: {score:.1f})")
                    recommendations.extend(self._get_component_recommendations(metric_type, score))
            
            # Calculate weighted overall score
            overall_score = sum(
                score * self.health_weights[MetricType(metric_type)]
                for metric_type, score in component_scores.items()
                if MetricType(metric_type) in self.health_weights
            )
            
            # Add general recommendations based on overall score
            if overall_score < 70:
                recommendations.extend([
                    "Consider comprehensive cache architecture review",
                    "Implement automated cache optimization",
                    "Monitor cache patterns more closely"
                ])
            
            return CacheHealthScore(
                overall_score=overall_score,
                component_scores=component_scores,
                issues=list(set(issues)),  # Remove duplicates
                recommendations=list(set(recommendations)),  # Remove duplicates
                last_updated=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error calculating cache health score: {e}")
            return CacheHealthScore(
                overall_score=0.0,
                component_scores={},
                issues=[f"Health calculation error: {e}"],
                recommendations=["Check cache monitoring system"],
                last_updated=datetime.now(timezone.utc)
            )
    
    async def _calculate_component_score(self, metric_type: MetricType) -> float:
        """Calculate score for individual component"""
        
        # Get recent metrics for all layers
        scores = []
        
        for layer in ["L1_MEMORY", "L2_REDIS"]:
            recent_metrics = self.metrics_collector.get_metrics_history(
                metric_type, layer, hours=1
            )
            
            if not recent_metrics:
                continue
            
            # Calculate score based on metric type
            latest_value = recent_metrics[-1]["value"]
            
            if metric_type == MetricType.HIT_RATIO:
                # Higher is better (0-1 range)
                score = min(100, max(0, latest_value * 100))
            
            elif metric_type == MetricType.RESPONSE_TIME:
                # Lower is better (ms)
                if latest_value <= 10:
                    score = 100
                elif latest_value <= 50:
                    score = 80
                elif latest_value <= 100:
                    score = 60
                elif latest_value <= 500:
                    score = 40
                else:
                    score = 20
            
            elif metric_type == MetricType.MEMORY_USAGE:
                # Optimal around 70-80% usage
                usage_ratio = latest_value / (512 * 1024 * 1024)  # Assume 512MB max
                if usage_ratio <= 0.8:
                    score = 100 - (usage_ratio * 25)  # Linear decrease
                else:
                    score = max(0, 100 - ((usage_ratio - 0.8) * 250))  # Steep decrease
            
            elif metric_type == MetricType.ERROR_RATE:
                # Lower is better (ratio)
                if latest_value <= 0.001:
                    score = 100
                elif latest_value <= 0.01:
                    score = 80
                elif latest_value <= 0.05:
                    score = 50
                else:
                    score = 20
            
            elif metric_type == MetricType.EVICTION_RATE:
                # Lower is better (per minute)
                if latest_value <= 10:
                    score = 100
                elif latest_value <= 50:
                    score = 80
                elif latest_value <= 100:
                    score = 60
                else:
                    score = max(0, 60 - (latest_value - 100) / 10)
            
            else:
                score = 75  # Default neutral score
            
            scores.append(score)
        
        return statistics.mean(scores) if scores else 75.0
    
    def _get_component_recommendations(self, metric_type: MetricType, score: float) -> List[str]:
        """Get recommendations for component based on score"""
        
        if score >= 80:
            return []
        
        recommendations = {
            MetricType.HIT_RATIO: [
                "Implement intelligent cache warming",
                "Analyze cache access patterns",
                "Optimize cache key design"
            ],
            MetricType.RESPONSE_TIME: [
                "Check network connectivity to Redis",
                "Optimize serialization methods",
                "Consider read replicas"
            ],
            MetricType.MEMORY_USAGE: [
                "Implement better eviction policies",
                "Optimize data structures",
                "Consider memory expansion"
            ],
            MetricType.ERROR_RATE: [
                "Implement robust error handling",
                "Add connection pooling",
                "Monitor Redis cluster health"
            ],
            MetricType.EVICTION_RATE: [
                "Increase cache capacity",
                "Optimize TTL strategies",
                "Review access patterns"
            ]
        }
        
        return recommendations.get(metric_type, [])


class CacheMonitoringDashboard:
    """Web dashboard for cache monitoring"""
    
    def __init__(self, metrics_collector: CacheMetricsCollector,
                 health_monitor: CacheHealthMonitor):
        self.metrics_collector = metrics_collector
        self.health_monitor = health_monitor
    
    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for monitoring dashboard"""
        
        # Get health score
        health_score = await self.health_monitor.calculate_health_score()
        
        # Get active alerts
        active_alerts = self.metrics_collector.get_active_alerts()
        
        # Get recent metrics
        chart_data = await self._generate_chart_data()
        
        return {
            "health_score": {
                "overall_score": health_score.overall_score,
                "component_scores": health_score.component_scores,
                "issues": health_score.issues,
                "recommendations": health_score.recommendations,
                "last_updated": health_score.last_updated.isoformat()
            },
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "layer": alert.layer,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "recommendations": alert.recommendations
                }
                for alert in active_alerts
            ],
            "charts": chart_data,
            "summary_stats": await self._get_summary_stats()
        }
    
    async def _generate_chart_data(self) -> Dict[str, Any]:
        """Generate chart data for dashboard"""
        
        charts = {}
        
        # Hit ratio over time
        charts["hit_ratio_timeline"] = self._create_timeline_chart(
            MetricType.HIT_RATIO, "Cache Hit Ratio Over Time", "Hit Ratio (%)"
        )
        
        # Response time over time
        charts["response_time_timeline"] = self._create_timeline_chart(
            MetricType.RESPONSE_TIME, "Response Time Over Time", "Response Time (ms)"
        )
        
        # Memory usage over time
        charts["memory_usage_timeline"] = self._create_timeline_chart(
            MetricType.MEMORY_USAGE, "Memory Usage Over Time", "Memory Usage (MB)"
        )
        
        # Layer comparison
        charts["layer_comparison"] = await self._create_layer_comparison_chart()
        
        # Alert distribution
        charts["alert_distribution"] = self._create_alert_distribution_chart()
        
        return charts
    
    def _create_timeline_chart(self, metric_type: MetricType, 
                             title: str, y_axis: str) -> Dict[str, Any]:
        """Create timeline chart for metric"""
        
        chart_data = {"type": "line", "title": title, "y_axis": y_axis, "data": {}}
        
        for layer in ["L1_MEMORY", "L2_REDIS"]:
            metrics = self.metrics_collector.get_metrics_history(metric_type, layer, hours=24)
            
            if metrics:
                timestamps = [datetime.fromtimestamp(m["timestamp"]).isoformat() for m in metrics]
                values = [m["value"] for m in metrics]
                
                # Convert to appropriate units
                if metric_type == MetricType.HIT_RATIO:
                    values = [v * 100 for v in values]  # Convert to percentage
                elif metric_type == MetricType.MEMORY_USAGE:
                    values = [v / (1024 * 1024) for v in values]  # Convert to MB
                
                chart_data["data"][layer] = {
                    "timestamps": timestamps,
                    "values": values
                }
        
        return chart_data
    
    async def _create_layer_comparison_chart(self) -> Dict[str, Any]:
        """Create layer performance comparison chart"""
        
        layers = ["L1_MEMORY", "L2_REDIS"]
        metrics = [MetricType.HIT_RATIO, MetricType.RESPONSE_TIME]
        
        comparison_data = {}
        
        for layer in layers:
            layer_data = {}
            
            for metric_type in metrics:
                recent_metrics = self.metrics_collector.get_metrics_history(
                    metric_type, layer, hours=1
                )
                
                if recent_metrics:
                    latest_value = recent_metrics[-1]["value"]
                    
                    if metric_type == MetricType.HIT_RATIO:
                        layer_data["hit_ratio"] = latest_value * 100
                    elif metric_type == MetricType.RESPONSE_TIME:
                        layer_data["response_time"] = latest_value
            
            comparison_data[layer] = layer_data
        
        return {
            "type": "bar",
            "title": "Cache Layer Performance Comparison",
            "data": comparison_data
        }
    
    def _create_alert_distribution_chart(self) -> Dict[str, Any]:
        """Create alert distribution chart"""
        
        alert_history = self.metrics_collector.get_alert_history(hours=24)
        
        alert_counts = {"warning": 0, "critical": 0}
        for alert in alert_history:
            alert_counts[alert.level.value] += 1
        
        return {
            "type": "pie",
            "title": "Alert Distribution (24h)",
            "data": {
                "labels": list(alert_counts.keys()),
                "values": list(alert_counts.values())
            }
        }
    
    async def _get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        
        active_alerts = self.metrics_collector.get_active_alerts()
        
        # Get latest metrics for summary
        summary = {
            "total_active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
            "warning_alerts": len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
        }
        
        # Add latest metric values
        for layer in ["L1_MEMORY", "L2_REDIS"]:
            hit_ratio_metrics = self.metrics_collector.get_metrics_history(
                MetricType.HIT_RATIO, layer, hours=1
            )
            
            if hit_ratio_metrics:
                summary[f"{layer.lower()}_hit_ratio"] = hit_ratio_metrics[-1]["value"] * 100
        
        return summary
    
    async def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard"""
        
        dashboard_data = await self.generate_dashboard_data()
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PRSM Cache Performance Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8fafc; }}
                .dashboard {{ max-width: 1400px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; }}
                .health-score {{ text-align: center; margin: 20px 0; }}
                .score-circle {{ width: 120px; height: 120px; border-radius: 50%; margin: 0 auto 20px; display: flex; align-items: center; justify-content: center; font-size: 2em; font-weight: bold; color: white; }}
                .score-excellent {{ background: linear-gradient(135deg, #11998e, #38ef7d); }}
                .score-good {{ background: linear-gradient(135deg, #f093fb, #f5576c); }}
                .score-poor {{ background: linear-gradient(135deg, #ff6b6b, #ee5a24); }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .metric-card {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 4px solid #3b82f6; }}
                .metric-value {{ font-size: 2.5em; font-weight: bold; color: #1f2937; margin-bottom: 5px; }}
                .metric-label {{ color: #6b7280; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.5px; }}
                .alerts-section {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 30px; }}
                .alert {{ padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid; }}
                .alert.warning {{ background: #fef3c7; border-color: #f59e0b; }}
                .alert.critical {{ background: #fee2e2; border-color: #ef4444; }}
                .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }}
                .chart-container {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
                .recommendations {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
                .recommendation {{ background: #f0f9ff; padding: 15px; margin: 10px 0; border-left: 4px solid #0ea5e9; border-radius: 6px; }}
                .last-updated {{ text-align: center; color: #6b7280; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>🚀 PRSM Cache Performance Dashboard</h1>
                    <p>Real-time cache monitoring and performance optimization</p>
                </div>
                
                <div class="health-score">
                    <div class="score-circle {self._get_score_class(dashboard_data['health_score']['overall_score'])}">
                        {dashboard_data['health_score']['overall_score']:.0f}
                    </div>
                    <h3>Overall Cache Health Score</h3>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_data['summary_stats'].get('total_active_alerts', 0)}</div>
                        <div class="metric-label">Active Alerts</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_data['summary_stats'].get('l1_memory_hit_ratio', 0):.1f}%</div>
                        <div class="metric-label">L1 Hit Ratio</div>
                    </div>
        """
        
        if 'l2_redis_hit_ratio' in dashboard_data['summary_stats']:
            html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{dashboard_data['summary_stats']['l2_redis_hit_ratio']:.1f}%</div>
                        <div class="metric-label">L2 Hit Ratio</div>
                    </div>
            """
        
        html += """
                </div>
        """
        
        # Add alerts section
        if dashboard_data['active_alerts']:
            html += """
                <div class="alerts-section">
                    <h3>🚨 Active Alerts</h3>
            """
            
            for alert in dashboard_data['active_alerts']:
                html += f"""
                    <div class="alert {alert['level']}">
                        <strong>{alert['level'].upper()}:</strong> {alert['message']}
                        <br><small>Layer: {alert['layer']} | Current: {alert['current_value']:.2f} | Threshold: {alert['threshold_value']:.2f}</small>
                    </div>
                """
            
            html += "</div>"
        
        # Add recommendations
        if dashboard_data['health_score']['recommendations']:
            html += """
                <div class="recommendations">
                    <h3>💡 Optimization Recommendations</h3>
            """
            
            for recommendation in dashboard_data['health_score']['recommendations'][:5]:  # Show top 5
                html += f'<div class="recommendation">{recommendation}</div>'
            
            html += "</div>"
        
        html += f"""
                <div class="charts-grid">
                    <div class="chart-container">
                        <div id="hit-ratio-chart"></div>
                    </div>
                    <div class="chart-container">
                        <div id="response-time-chart"></div>
                    </div>
                    <div class="chart-container">
                        <div id="layer-comparison-chart"></div>
                    </div>
                    <div class="chart-container">
                        <div id="alert-distribution-chart"></div>
                    </div>
                </div>
                
                <div class="last-updated">
                    Last updated: {dashboard_data['health_score']['last_updated']}
                </div>
            </div>
            
            <script>
                // Chart data
                const chartData = {json.dumps(dashboard_data['charts'], cls=PlotlyJSONEncoder)};
                
                // Hit ratio chart
                if (chartData.hit_ratio_timeline && chartData.hit_ratio_timeline.data) {{
                    const traces = [];
                    for (const [layer, data] of Object.entries(chartData.hit_ratio_timeline.data)) {{
                        traces.push({{
                            x: data.timestamps,
                            y: data.values,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: layer,
                            line: {{ width: 3 }}
                        }});
                    }}
                    
                    Plotly.newPlot('hit-ratio-chart', traces, {{
                        title: chartData.hit_ratio_timeline.title,
                        xaxis: {{ title: 'Time' }},
                        yaxis: {{ title: chartData.hit_ratio_timeline.y_axis, range: [0, 100] }}
                    }});
                }}
                
                // Response time chart
                if (chartData.response_time_timeline && chartData.response_time_timeline.data) {{
                    const traces = [];
                    for (const [layer, data] of Object.entries(chartData.response_time_timeline.data)) {{
                        traces.push({{
                            x: data.timestamps,
                            y: data.values,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: layer,
                            line: {{ width: 3 }}
                        }});
                    }}
                    
                    Plotly.newPlot('response-time-chart', traces, {{
                        title: chartData.response_time_timeline.title,
                        xaxis: {{ title: 'Time' }},
                        yaxis: {{ title: chartData.response_time_timeline.y_axis }}
                    }});
                }}
                
                // Auto-refresh every 30 seconds
                setTimeout(() => {{
                    location.reload();
                }}, 30000);
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class for health score"""
        if score >= 80:
            return "score-excellent"
        elif score >= 60:
            return "score-good"
        else:
            return "score-poor"


# Global monitoring instances
metrics_collector: Optional[CacheMetricsCollector] = None
health_monitor: Optional[CacheHealthMonitor] = None
monitoring_dashboard: Optional[CacheMonitoringDashboard] = None


async def initialize_cache_monitoring(redis_client: aioredis.Redis):
    """Initialize cache monitoring system"""
    global metrics_collector, health_monitor, monitoring_dashboard
    
    metrics_collector = CacheMetricsCollector(redis_client)
    health_monitor = CacheHealthMonitor(metrics_collector)
    monitoring_dashboard = CacheMonitoringDashboard(metrics_collector, health_monitor)
    
    await metrics_collector.start_collection()
    logger.info("✅ Cache monitoring system initialized")


def get_cache_monitoring_dashboard() -> CacheMonitoringDashboard:
    """Get the global monitoring dashboard instance"""
    if monitoring_dashboard is None:
        raise RuntimeError("Cache monitoring not initialized.")
    return monitoring_dashboard


def get_metrics_collector() -> CacheMetricsCollector:
    """Get the global metrics collector instance"""
    if metrics_collector is None:
        raise RuntimeError("Cache monitoring not initialized.")
    return metrics_collector


async def shutdown_cache_monitoring():
    """Shutdown cache monitoring system"""
    if metrics_collector:
        await metrics_collector.stop_collection()


def create_cache_monitoring_endpoints(app: FastAPI):
    """Create cache monitoring API endpoints"""
    
    @app.get("/cache/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def get_cache_dashboard():
        """Get cache monitoring dashboard"""
        dashboard = get_cache_monitoring_dashboard()
        return await dashboard.generate_dashboard_html()
    
    @app.get("/api/cache/metrics", include_in_schema=False)
    async def get_cache_metrics():
        """Get current cache metrics"""
        dashboard = get_cache_monitoring_dashboard()
        return await dashboard.generate_dashboard_data()
    
    @app.get("/api/cache/health", include_in_schema=False)
    async def get_cache_health():
        """Get cache health score"""
        if not health_monitor:
            return {"error": "Health monitor not initialized"}
        
        health_score = await health_monitor.calculate_health_score()
        return {
            "overall_score": health_score.overall_score,
            "component_scores": health_score.component_scores,
            "issues": health_score.issues,
            "recommendations": health_score.recommendations,
            "last_updated": health_score.last_updated.isoformat()
        }
    
    @app.get("/api/cache/alerts", include_in_schema=False)
    async def get_cache_alerts():
        """Get current cache alerts"""
        if not metrics_collector:
            return {"active_alerts": [], "alert_history": []}
        
        active_alerts = metrics_collector.get_active_alerts()
        alert_history = metrics_collector.get_alert_history(hours=24)
        
        return {
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "layer": alert.layer,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "recommendations": alert.recommendations,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ],
            "alert_history": [
                {
                    "alert_id": alert.alert_id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "resolved": alert.resolved,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in alert_history
            ]
        }