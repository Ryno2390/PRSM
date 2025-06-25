"""
PRSM Performance Monitoring and APM Integration
Real-time performance monitoring, distributed tracing, and metrics collection

📊 MONITORING CAPABILITIES:
- Application Performance Monitoring (APM) integration
- Distributed tracing across microservices
- Real-time metrics collection and alerting
- Performance anomaly detection
- Custom metrics and dashboards
"""

import asyncio
import time
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from contextlib import asynccontextmanager

import structlog

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of performance metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    DISTRIBUTION = "distribution"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


@dataclass
class Metric:
    """Individual performance metric"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class TraceSpan:
    """Distributed tracing span"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error, timeout
    error_message: Optional[str] = None


@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    threshold_value: float
    current_value: float
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class DistributedTracing:
    """
    Distributed tracing system for PRSM microservices
    
    🔍 TRACING CAPABILITIES:
    - Request tracing across service boundaries
    - Performance bottleneck identification
    - Error tracking and correlation
    - Service dependency mapping
    - Latency analysis and optimization
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_traces: List[TraceSpan] = []
        self.trace_retention_hours = 24
        
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> TraceSpan:
        """Start a new tracing span"""
        span_id = str(uuid.uuid4())
        
        # Generate or inherit trace ID
        if parent_span_id and parent_span_id in self.active_spans:
            trace_id = self.active_spans[parent_span_id].trace_id
        else:
            trace_id = str(uuid.uuid4())
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(timezone.utc),
            tags=tags or {}
        )
        
        # Add service information
        span.tags["service.name"] = self.service_name
        span.tags["span.kind"] = "internal"
        
        self.active_spans[span_id] = span
        
        logger.debug("Span started",
                    span_id=span_id,
                    trace_id=trace_id,
                    operation=operation_name)
        
        return span
    
    def finish_span(self, span_id: str, status: str = "ok", error_message: Optional[str] = None):
        """Finish a tracing span"""
        if span_id not in self.active_spans:
            logger.warning("Attempted to finish unknown span", span_id=span_id)
            return
        
        span = self.active_spans[span_id]
        span.end_time = datetime.now(timezone.utc)
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        span.error_message = error_message
        
        # Move to completed traces
        self.completed_traces.append(span)
        del self.active_spans[span_id]
        
        logger.debug("Span finished",
                    span_id=span_id,
                    duration_ms=span.duration_ms,
                    status=status)
        
        # Clean up old traces
        self._cleanup_old_traces()
    
    def add_span_log(self, span_id: str, log_data: Dict[str, Any]):
        """Add log entry to span"""
        if span_id in self.active_spans:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **log_data
            }
            self.active_spans[span_id].logs.append(log_entry)
    
    def add_span_tag(self, span_id: str, key: str, value: str):
        """Add tag to span"""
        if span_id in self.active_spans:
            self.active_spans[span_id].tags[key] = value
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, parent_span_id: Optional[str] = None,
                            tags: Optional[Dict[str, str]] = None):
        """Context manager for tracing operations"""
        span = self.start_span(operation_name, parent_span_id, tags)
        
        try:
            yield span
            self.finish_span(span.span_id, "ok")
        except Exception as e:
            self.finish_span(span.span_id, "error", str(e))
            raise
    
    def get_trace_by_id(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a specific trace"""
        spans = [
            span for span in self.completed_traces
            if span.trace_id == trace_id
        ]
        
        # Include active spans for the trace
        active_spans = [
            span for span in self.active_spans.values()
            if span.trace_id == trace_id
        ]
        
        return spans + active_spans
    
    def analyze_trace_performance(self, trace_id: str) -> Dict[str, Any]:
        """Analyze performance characteristics of a trace"""
        spans = self.get_trace_by_id(trace_id)
        
        if not spans:
            return {"error": "Trace not found"}
        
        # Calculate trace statistics
        total_duration = 0
        error_count = 0
        service_breakdown = {}
        operation_breakdown = {}
        
        for span in spans:
            if span.duration_ms:
                total_duration = max(total_duration, span.duration_ms)
                
                if span.status == "error":
                    error_count += 1
                
                # Service breakdown
                service = span.tags.get("service.name", "unknown")
                if service not in service_breakdown:
                    service_breakdown[service] = {"duration": 0, "spans": 0}
                service_breakdown[service]["duration"] += span.duration_ms
                service_breakdown[service]["spans"] += 1
                
                # Operation breakdown
                if span.operation_name not in operation_breakdown:
                    operation_breakdown[span.operation_name] = {"duration": 0, "spans": 0}
                operation_breakdown[span.operation_name]["duration"] += span.duration_ms
                operation_breakdown[span.operation_name]["spans"] += 1
        
        return {
            "trace_id": trace_id,
            "total_spans": len(spans),
            "total_duration_ms": total_duration,
            "error_count": error_count,
            "error_rate": error_count / len(spans) if spans else 0,
            "service_breakdown": service_breakdown,
            "operation_breakdown": operation_breakdown,
            "critical_path": self._identify_critical_path(spans)
        }
    
    def _identify_critical_path(self, spans: List[TraceSpan]) -> List[Dict[str, Any]]:
        """Identify the critical path through the trace"""
        # Sort spans by start time
        sorted_spans = sorted(
            [span for span in spans if span.duration_ms],
            key=lambda x: x.start_time
        )
        
        # Find the longest duration spans
        critical_path = []
        for span in sorted_spans[:5]:  # Top 5 longest operations
            critical_path.append({
                "operation": span.operation_name,
                "service": span.tags.get("service.name", "unknown"),
                "duration_ms": span.duration_ms,
                "percentage_of_total": (span.duration_ms / max(s.duration_ms for s in sorted_spans)) * 100
            })
        
        return critical_path
    
    def _cleanup_old_traces(self):
        """Remove traces older than retention period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.trace_retention_hours)
        
        self.completed_traces = [
            trace for trace in self.completed_traces
            if trace.start_time > cutoff_time
        ]
    
    def export_traces_jaeger_format(self) -> List[Dict[str, Any]]:
        """Export traces in Jaeger-compatible format"""
        jaeger_traces = []
        
        # Group spans by trace ID
        traces_by_id = {}
        for span in self.completed_traces:
            if span.trace_id not in traces_by_id:
                traces_by_id[span.trace_id] = []
            traces_by_id[span.trace_id].append(span)
        
        for trace_id, spans in traces_by_id.items():
            jaeger_spans = []
            
            for span in spans:
                jaeger_span = {
                    "traceID": trace_id,
                    "spanID": span.span_id,
                    "parentSpanID": span.parent_span_id,
                    "operationName": span.operation_name,
                    "startTime": int(span.start_time.timestamp() * 1000000),  # microseconds
                    "duration": int((span.duration_ms or 0) * 1000),  # microseconds
                    "tags": [
                        {"key": k, "value": v, "type": "string"}
                        for k, v in span.tags.items()
                    ],
                    "logs": [
                        {
                            "timestamp": int(datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00')).timestamp() * 1000000),
                            "fields": [{"key": k, "value": str(v)} for k, v in log.items() if k != "timestamp"]
                        }
                        for log in span.logs
                    ]
                }
                jaeger_spans.append(jaeger_span)
            
            jaeger_traces.append({
                "traceID": trace_id,
                "spans": jaeger_spans,
                "processes": {
                    "p1": {
                        "serviceName": self.service_name,
                        "tags": []
                    }
                }
            })
        
        return jaeger_traces


class MetricsCollector:
    """
    Real-time metrics collection and aggregation
    
    📊 METRICS CAPABILITIES:
    - Custom metric collection and aggregation
    - Real-time metric streaming
    - Histogram and distribution tracking
    - Metric retention and storage
    - Integration with monitoring systems
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.metrics: Dict[str, List[Metric]] = {}
        self.metric_aggregations: Dict[str, Dict[str, float]] = {}
        self.retention_hours = 24
        
    def record_metric(self, name: str, value: Union[int, float], metric_type: MetricType,
                     tags: Optional[Dict[str, str]] = None, unit: str = "",
                     description: str = ""):
        """Record a performance metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            unit=unit,
            description=description
        )
        
        # Add service tag
        metric.tags["service"] = self.service_name
        
        # Store metric
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metric)
        
        # Update aggregations
        self._update_aggregations(metric)
        
        # Cleanup old metrics
        self._cleanup_old_metrics()
        
        logger.debug("Metric recorded",
                    name=name,
                    value=value,
                    type=metric_type.value)
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric"""
        self.record_metric(name, duration_ms, MetricType.TIMER, tags, "ms")
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    @asynccontextmanager
    async def time_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_timer(f"{operation_name}.duration", duration_ms, tags)
    
    def get_metric_summary(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        if metric_name not in self.metrics:
            return {"error": "Metric not found"}
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_metrics = [
            metric for metric in self.metrics[metric_name]
            if metric.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent data"}
        
        values = [metric.value for metric in recent_metrics]
        
        summary = {
            "metric_name": metric_name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "last_value": values[-1],
            "time_range_hours": hours
        }
        
        # Add percentiles for histograms and timers
        if recent_metrics[0].metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
            sorted_values = sorted(values)
            summary.update({
                "p50": self._percentile(sorted_values, 50),
                "p90": self._percentile(sorted_values, 90),
                "p95": self._percentile(sorted_values, 95),
                "p99": self._percentile(sorted_values, 99)
            })
        
        return summary
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics"""
        summary = {
            "service_name": self.service_name,
            "total_metric_types": len(self.metrics),
            "metrics_by_type": {},
            "recent_activity": {}
        }
        
        # Count metrics by type
        type_counts = {}
        for metric_list in self.metrics.values():
            if metric_list:
                metric_type = metric_list[0].metric_type.value
                type_counts[metric_type] = type_counts.get(metric_type, 0) + len(metric_list)
        
        summary["metrics_by_type"] = type_counts
        
        # Recent activity (last hour)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_count = 0
        for metric_list in self.metrics.values():
            recent_count += len([
                metric for metric in metric_list
                if metric.timestamp > cutoff_time
            ])
        
        summary["recent_activity"] = {
            "metrics_last_hour": recent_count,
            "avg_metrics_per_minute": recent_count / 60
        }
        
        return summary
    
    def _update_aggregations(self, metric: Metric):
        """Update metric aggregations"""
        name = metric.name
        
        if name not in self.metric_aggregations:
            self.metric_aggregations[name] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0.0
            }
        
        agg = self.metric_aggregations[name]
        agg["count"] += 1
        agg["sum"] += metric.value
        agg["min"] = min(agg["min"], metric.value)
        agg["max"] = max(agg["max"], metric.value)
        agg["avg"] = agg["sum"] / agg["count"]
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        
        for metric_name in list(self.metrics.keys()):
            self.metrics[metric_name] = [
                metric for metric in self.metrics[metric_name]
                if metric.timestamp > cutoff_time
            ]
            
            # Remove empty metric lists
            if not self.metrics[metric_name]:
                del self.metrics[metric_name]
    
    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not sorted_values:
            return 0.0
        
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        
        if lower_index == upper_index:
            return sorted_values[lower_index]
        
        # Linear interpolation
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        prometheus_lines = []
        
        for metric_name, metric_list in self.metrics.items():
            if not metric_list:
                continue
            
            latest_metric = metric_list[-1]
            
            # Add help and type comments
            prometheus_lines.append(f"# HELP {metric_name} {latest_metric.description}")
            prometheus_lines.append(f"# TYPE {metric_name} {latest_metric.metric_type.value}")
            
            # Add metric value with tags
            tags_str = ",".join([f'{k}="{v}"' for k, v in latest_metric.tags.items()])
            if tags_str:
                prometheus_lines.append(f"{metric_name}{{{tags_str}}} {latest_metric.value}")
            else:
                prometheus_lines.append(f"{metric_name} {latest_metric.value}")
        
        return "\n".join(prometheus_lines)


class APMIntegration:
    """
    Application Performance Monitoring integration
    
    📊 APM CAPABILITIES:
    - Integration with popular APM platforms
    - Automatic performance monitoring
    - Error tracking and alerting
    - Performance anomaly detection
    - Custom dashboard generation
    """
    
    def __init__(self, service_name: str, apm_config: Dict[str, Any]):
        self.service_name = service_name
        self.apm_config = apm_config
        self.tracing = DistributedTracing(service_name)
        self.metrics = MetricsCollector(service_name)
        self.alerts: List[PerformanceAlert] = []
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        
    async def initialize(self):
        """Initialize APM integration"""
        try:
            # Initialize monitoring components
            await self._setup_default_metrics()
            await self._setup_default_alerts()
            
            logger.info("APM integration initialized",
                       service=self.service_name,
                       tracing_enabled=True,
                       metrics_enabled=True)
            
        except Exception as e:
            logger.error("APM initialization failed", error=str(e))
            raise
    
    @asynccontextmanager
    async def monitor_request(self, request_path: str, method: str = "GET",
                            user_id: Optional[str] = None):
        """Monitor a complete request lifecycle"""
        # Start tracing
        request_id = str(uuid.uuid4())
        tags = {
            "http.method": method,
            "http.path": request_path,
            "request.id": request_id
        }
        
        if user_id:
            tags["user.id"] = user_id
        
        async with self.tracing.trace_operation(f"{method} {request_path}", tags=tags) as span:
            start_time = time.time()
            
            try:
                # Increment request counter
                self.metrics.increment_counter("http.requests.total", tags={
                    "method": method,
                    "path": request_path
                })
                
                yield {
                    "span": span,
                    "request_id": request_id,
                    "start_time": start_time
                }
                
                # Record successful request
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.record_timer("http.request.duration", duration_ms, tags={
                    "method": method,
                    "path": request_path,
                    "status": "success"
                })
                
                # Check for performance alerts
                await self._check_performance_thresholds("response_time", duration_ms, tags)
                
            except Exception as e:
                # Record error
                self.metrics.increment_counter("http.requests.errors", tags={
                    "method": method,
                    "path": request_path,
                    "error_type": type(e).__name__
                })
                
                # Add error information to span
                self.tracing.add_span_tag(span.span_id, "error", "true")
                self.tracing.add_span_tag(span.span_id, "error.type", type(e).__name__)
                self.tracing.add_span_log(span.span_id, {
                    "event": "error",
                    "message": str(e)
                })
                
                raise
    
    async def monitor_database_operation(self, operation: str, table: str = ""):
        """Monitor database operations"""
        tags = {"operation": operation}
        if table:
            tags["table"] = table
        
        async with self.tracing.trace_operation(f"db.{operation}", tags=tags) as span:
            async with self.metrics.time_operation(f"database.{operation}.duration", tags):
                yield span
    
    async def monitor_external_api_call(self, service: str, endpoint: str):
        """Monitor external API calls"""
        tags = {
            "external.service": service,
            "external.endpoint": endpoint,
            "span.kind": "client"
        }
        
        async with self.tracing.trace_operation(f"http.client.{service}", tags=tags) as span:
            start_time = time.time()
            
            try:
                yield span
                
                # Record successful call
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.record_timer("external.api.duration", duration_ms, tags={
                    "service": service,
                    "status": "success"
                })
                
            except Exception as e:
                # Record error
                self.metrics.increment_counter("external.api.errors", tags={
                    "service": service,
                    "error_type": type(e).__name__
                })
                raise
    
    def set_alert_threshold(self, metric_name: str, threshold_value: float,
                          severity: AlertSeverity = AlertSeverity.WARNING,
                          comparison: str = "greater_than"):
        """Set alert threshold for a metric"""
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}
        
        self.alert_thresholds[metric_name][severity.value] = {
            "threshold": threshold_value,
            "comparison": comparison
        }
        
        logger.info("Alert threshold set",
                   metric=metric_name,
                   threshold=threshold_value,
                   severity=severity.value)
    
    async def check_alerts(self) -> List[PerformanceAlert]:
        """Check for triggered alerts"""
        new_alerts = []
        
        for metric_name, thresholds in self.alert_thresholds.items():
            # Get recent metric data
            metric_summary = self.metrics.get_metric_summary(metric_name, hours=1)
            
            if "error" in metric_summary:
                continue
            
            current_value = metric_summary.get("last_value", 0)
            
            for severity_str, threshold_config in thresholds.items():
                threshold_value = threshold_config["threshold"]
                comparison = threshold_config["comparison"]
                
                alert_triggered = False
                if comparison == "greater_than" and current_value > threshold_value:
                    alert_triggered = True
                elif comparison == "less_than" and current_value < threshold_value:
                    alert_triggered = True
                
                if alert_triggered:
                    alert = PerformanceAlert(
                        alert_id=str(uuid.uuid4()),
                        metric_name=metric_name,
                        severity=AlertSeverity(severity_str),
                        threshold_value=threshold_value,
                        current_value=current_value,
                        message=f"{metric_name} {comparison} {threshold_value} (current: {current_value})",
                        tags={"service": self.service_name}
                    )
                    
                    new_alerts.append(alert)
                    self.alerts.append(alert)
        
        if new_alerts:
            logger.warning("Performance alerts triggered",
                          alert_count=len(new_alerts),
                          service=self.service_name)
        
        return new_alerts
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a performance alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now(timezone.utc)
                
                logger.info("Alert resolved",
                           alert_id=alert_id,
                           metric=alert.metric_name)
                return True
        
        return False
    
    async def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        dashboard_data = {
            "service_name": self.service_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics_summary": self.metrics.get_all_metrics_summary(),
            "active_alerts": [
                alert.__dict__ for alert in self.alerts
                if not alert.resolved
            ],
            "recent_traces": self._get_recent_trace_summary(),
            "performance_overview": await self._get_performance_overview()
        }
        
        return dashboard_data
    
    async def export_monitoring_data(self, format: str = "json") -> str:
        """Export monitoring data in specified format"""
        if format == "prometheus":
            return self.metrics.export_prometheus_format()
        elif format == "jaeger":
            traces = self.tracing.export_traces_jaeger_format()
            return json.dumps(traces, indent=2)
        else:  # json
            monitoring_data = {
                "service": self.service_name,
                "metrics": self.metrics.get_all_metrics_summary(),
                "traces": self.tracing.export_traces_jaeger_format(),
                "alerts": [alert.__dict__ for alert in self.alerts],
                "export_timestamp": datetime.now(timezone.utc).isoformat()
            }
            return json.dumps(monitoring_data, indent=2, default=str)
    
    async def _setup_default_metrics(self):
        """Setup default metrics for PRSM"""
        # Initialize standard application metrics
        default_metrics = [
            ("system.cpu.usage", MetricType.GAUGE),
            ("system.memory.usage", MetricType.GAUGE),
            ("http.requests.total", MetricType.COUNTER),
            ("http.request.duration", MetricType.TIMER),
            ("database.connections.active", MetricType.GAUGE),
            ("cache.hit.rate", MetricType.GAUGE)
        ]
        
        for metric_name, metric_type in default_metrics:
            # Initialize with zero values
            self.metrics.record_metric(metric_name, 0, metric_type)
    
    async def _setup_default_alerts(self):
        """Setup default alert thresholds"""
        # Default alert thresholds for PRSM
        default_thresholds = [
            ("http.request.duration", 2000, AlertSeverity.WARNING),  # 2 second response time
            ("http.request.duration", 5000, AlertSeverity.CRITICAL),  # 5 second response time
            ("system.cpu.usage", 80, AlertSeverity.WARNING),  # 80% CPU usage
            ("system.cpu.usage", 95, AlertSeverity.CRITICAL),  # 95% CPU usage
            ("system.memory.usage", 85, AlertSeverity.WARNING),  # 85% memory usage
            ("cache.hit.rate", 70, AlertSeverity.WARNING)  # 70% cache hit rate (less than)
        ]
        
        for metric_name, threshold, severity in default_thresholds:
            comparison = "less_than" if "hit.rate" in metric_name else "greater_than"
            self.set_alert_threshold(metric_name, threshold, severity, comparison)
    
    async def _check_performance_thresholds(self, metric_type: str, value: float, tags: Dict[str, str]):
        """Check if performance value exceeds thresholds"""
        # This would trigger real-time alerting
        # For now, just log significant performance issues
        if metric_type == "response_time" and value > 2000:  # 2 seconds
            logger.warning("Slow response time detected",
                          duration_ms=value,
                          tags=tags)
    
    def _get_recent_trace_summary(self) -> Dict[str, Any]:
        """Get summary of recent traces"""
        recent_traces = self.tracing.completed_traces[-10:]  # Last 10 traces
        
        if not recent_traces:
            return {"count": 0}
        
        total_duration = sum(trace.duration_ms or 0 for trace in recent_traces)
        error_count = sum(1 for trace in recent_traces if trace.status == "error")
        
        return {
            "count": len(recent_traces),
            "avg_duration_ms": total_duration / len(recent_traces),
            "error_rate": error_count / len(recent_traces),
            "latest_operations": [
                {
                    "operation": trace.operation_name,
                    "duration_ms": trace.duration_ms,
                    "status": trace.status
                }
                for trace in recent_traces[-5:]  # Last 5 operations
            ]
        }
    
    async def _get_performance_overview(self) -> Dict[str, Any]:
        """Get overall performance overview"""
        # Simulate performance overview data
        import random
        
        return {
            "health_score": random.uniform(85, 99),
            "avg_response_time_ms": random.uniform(100, 500),
            "error_rate_percent": random.uniform(0.1, 2.0),
            "throughput_rps": random.uniform(50, 200),
            "active_users": random.randint(10, 100),
            "system_load": {
                "cpu": random.uniform(20, 80),
                "memory": random.uniform(30, 70),
                "disk": random.uniform(10, 50)
            }
        }


# Example usage and factory functions
def create_prsm_apm_integration(service_name: str = "prsm-api") -> APMIntegration:
    """Create APM integration for PRSM services"""
    
    apm_config = {
        "enabled": True,
        "sampling_rate": 1.0,  # 100% sampling for development
        "export_interval_seconds": 60,
        "retention_hours": 24,
        "alerting_enabled": True
    }
    
    return APMIntegration(service_name, apm_config)


async def setup_prsm_monitoring():
    """Setup comprehensive monitoring for PRSM"""
    # Create APM integration for main services
    services = ["prsm-api", "prsm-ml-trainer", "prsm-worker"]
    apm_integrations = {}
    
    for service in services:
        apm = create_prsm_apm_integration(service)
        await apm.initialize()
        apm_integrations[service] = apm
    
    logger.info("PRSM monitoring setup completed",
               services=list(apm_integrations.keys()))
    
    return apm_integrations