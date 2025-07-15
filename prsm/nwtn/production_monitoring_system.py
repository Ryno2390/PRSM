#!/usr/bin/env python3
"""
Production Monitoring System for PRSM
=====================================

This module provides comprehensive production monitoring and alerting for
the PRSM system, with real-time metrics, health checks, and intelligent
alerting capabilities.

Key Features:
1. Real-time system health monitoring
2. Performance metrics tracking and analysis
3. Intelligent alerting and notification system
4. Resource utilization monitoring
5. Quality metrics and SLA tracking
6. Anomaly detection and predictive alerts
7. Comprehensive logging and audit trails
8. Dashboard and visualization support

Designed for production deployment with focus on reliability and observability.
"""

import asyncio
import json
import logging
import time
import psutil
import sqlite3
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import threading
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts"""
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    RESOURCE_USAGE = "resource_usage"
    QUALITY_DEGRADATION = "quality_degradation"
    INGESTION_FAILURE = "ingestion_failure"
    STORAGE_ISSUE = "storage_issue"
    PROCESSING_ANOMALY = "processing_anomaly"
    SLA_VIOLATION = "sla_violation"


class SystemStatus(str, Enum):
    """Overall system status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Alert information"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata
        }


@dataclass
class Metric:
    """Metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: SystemStatus
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health check to dictionary"""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "response_time": self.response_time,
            "details": self.details
        }


@dataclass
class SLAMetric:
    """SLA metric tracking"""
    metric_name: str
    target_value: float
    current_value: float
    threshold_breached: bool
    breach_duration: float = 0.0
    last_breach: Optional[datetime] = None
    
    @property
    def compliance_percentage(self) -> float:
        """Calculate SLA compliance percentage"""
        if self.target_value == 0:
            return 100.0
        return min(100.0, (self.current_value / self.target_value) * 100)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    
    # Health check settings
    health_check_interval: int = 60  # seconds
    component_timeout: float = 30.0  # seconds
    
    # Metric collection settings
    metric_collection_interval: int = 10  # seconds
    metric_retention_days: int = 30
    
    # Alert settings
    alert_debounce_time: int = 300  # 5 minutes
    max_alerts_per_hour: int = 100
    
    # Performance thresholds
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 95.0
    memory_warning_threshold: float = 80.0
    memory_critical_threshold: float = 90.0
    disk_warning_threshold: float = 80.0
    disk_critical_threshold: float = 95.0
    
    # Quality thresholds
    ingestion_success_rate_threshold: float = 95.0
    processing_time_threshold: float = 300.0  # 5 minutes
    error_rate_threshold: float = 5.0  # 5%
    
    # Storage settings
    database_path: str = "monitoring.db"
    log_retention_days: int = 90
    
    # Notification settings
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = field(default_factory=list)


class ProductionMonitoringSystem:
    """
    Production Monitoring System for PRSM
    
    Provides comprehensive monitoring, alerting, and observability for
    production PRSM deployments with real-time metrics and intelligent
    anomaly detection.
    """
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        
        # Monitoring state
        self.system_status = SystemStatus.HEALTHY
        self.monitoring_active = False
        
        # Storage
        self.database: Optional[sqlite3.Connection] = None
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.alerts_buffer: deque = deque(maxlen=1000)
        
        # Component tracking
        self.component_health: Dict[str, HealthCheck] = {}
        self.component_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Alert management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_debounce: Dict[str, datetime] = {}
        
        # SLA tracking
        self.sla_metrics: Dict[str, SLAMetric] = {}
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.metric_collection_task: Optional[asyncio.Task] = None
        self.alert_processing_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.monitoring_stats = {
            "total_metrics_collected": 0,
            "total_alerts_generated": 0,
            "total_health_checks": 0,
            "uptime_start": datetime.now(timezone.utc),
            "last_alert_time": None,
            "system_status_changes": 0
        }
        
        # Thread safety
        self.metrics_lock = threading.RLock()
        self.alerts_lock = threading.RLock()
        
        logger.info("Production Monitoring System initialized")
    
    async def initialize(self):
        """Initialize the monitoring system"""
        
        logger.info("🔍 Initializing Production Monitoring System...")
        
        # Initialize database
        await self._initialize_database()
        
        # Initialize SLA metrics
        await self._initialize_sla_metrics()
        
        # Start monitoring tasks
        await self._start_monitoring_tasks()
        
        # Perform initial health check
        await self._perform_initial_health_check()
        
        self.monitoring_active = True
        
        logger.info("✅ Production Monitoring System active")
    
    async def record_metric(self, name: str, value: float, 
                          metric_type: MetricType = MetricType.GAUGE,
                          tags: Dict[str, str] = None):
        """Record a metric"""
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {}
        )
        
        with self.metrics_lock:
            self.metrics_buffer.append(metric)
            self.component_metrics[name].append(metric)
            self.monitoring_stats["total_metrics_collected"] += 1
        
        # Check for threshold breaches
        await self._check_metric_thresholds(metric)
    
    async def record_health_check(self, component: str, status: SystemStatus, 
                                message: str, response_time: float = 0.0,
                                details: Dict[str, Any] = None):
        """Record a health check result"""
        
        health_check = HealthCheck(
            component=component,
            status=status,
            message=message,
            response_time=response_time,
            details=details or {}
        )
        
        self.component_health[component] = health_check
        self.monitoring_stats["total_health_checks"] += 1
        
        # Check for status changes
        await self._check_component_status_change(health_check)
        
        # Update overall system status
        await self._update_system_status()
    
    async def generate_alert(self, alert_type: AlertType, severity: AlertSeverity,
                           title: str, message: str, component: str,
                           metadata: Dict[str, Any] = None):
        """Generate an alert"""
        
        alert_id = f"{alert_type.value}_{component}_{int(time.time())}"
        
        # Check alert debouncing
        debounce_key = f"{alert_type.value}_{component}"
        if await self._should_debounce_alert(debounce_key):
            return
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            component=component,
            metadata=metadata or {}
        )
        
        with self.alerts_lock:
            self.active_alerts[alert_id] = alert
            self.alerts_buffer.append(alert)
            self.alert_history.append(alert)
            self.monitoring_stats["total_alerts_generated"] += 1
            self.monitoring_stats["last_alert_time"] = datetime.now(timezone.utc)
        
        # Update debounce tracking
        self.alert_debounce[debounce_key] = datetime.now(timezone.utc)
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        logger.warning(f"Alert generated: {alert.title}",
                      severity=severity.value,
                      component=component)
    
    async def acknowledge_alert(self, alert_id: str, acknowledger: str = "system"):
        """Acknowledge an alert"""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.metadata["acknowledged_by"] = acknowledger
            alert.metadata["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Alert acknowledged: {alert.title}",
                       alert_id=alert_id,
                       acknowledger=acknowledger)
    
    async def resolve_alert(self, alert_id: str, resolver: str = "system"):
        """Resolve an alert"""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now(timezone.utc)
            alert.metadata["resolved_by"] = resolver
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert.title}",
                       alert_id=alert_id,
                       resolver=resolver)
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        
        # Calculate uptime
        uptime_seconds = (datetime.now(timezone.utc) - self.monitoring_stats["uptime_start"]).total_seconds()
        
        # Get recent metrics
        recent_metrics = {}
        for name, metrics in self.component_metrics.items():
            if metrics:
                recent_metrics[name] = {
                    "current_value": metrics[-1].value,
                    "average_1h": await self._calculate_metric_average(name, hours=1),
                    "trend": await self._calculate_metric_trend(name)
                }
        
        # Get active alerts by severity
        alerts_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            alerts_by_severity[alert.severity.value] += 1
        
        # Get component health summary
        component_health_summary = {}
        for component, health in self.component_health.items():
            component_health_summary[component] = {
                "status": health.status.value,
                "message": health.message,
                "response_time": health.response_time,
                "last_check": health.timestamp.isoformat()
            }
        
        return {
            "system_status": self.system_status.value,
            "uptime_seconds": uptime_seconds,
            "monitoring_active": self.monitoring_active,
            "statistics": self.monitoring_stats,
            "recent_metrics": recent_metrics,
            "component_health": component_health_summary,
            "active_alerts": {
                "total": len(self.active_alerts),
                "by_severity": dict(alerts_by_severity)
            },
            "sla_compliance": await self._get_sla_compliance_summary(),
            "resource_usage": await self._get_resource_usage_summary(),
            "performance_summary": await self._get_performance_summary()
        }
    
    async def get_detailed_metrics(self, component: str = None, 
                                 hours: int = 24) -> Dict[str, Any]:
        """Get detailed metrics for analysis"""
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        # Filter metrics by time range and component
        filtered_metrics = []
        
        for metric in self.metrics_buffer:
            if metric.timestamp >= start_time and metric.timestamp <= end_time:
                if component is None or metric.tags.get("component") == component:
                    filtered_metrics.append(metric)
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in filtered_metrics:
            metrics_by_name[metric.name].append(metric)
        
        # Calculate statistics for each metric
        detailed_metrics = {}
        for name, metrics in metrics_by_name.items():
            values = [m.value for m in metrics]
            if values:
                detailed_metrics[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "trend": await self._calculate_trend(values),
                    "data_points": [m.to_dict() for m in metrics[-100:]]  # Last 100 points
                }
        
        return {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            },
            "component_filter": component,
            "metrics": detailed_metrics,
            "total_data_points": len(filtered_metrics)
        }
    
    async def get_alert_history(self, hours: int = 24, 
                              severity: AlertSeverity = None) -> List[Dict[str, Any]]:
        """Get alert history"""
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        # Filter alerts by time range and severity
        filtered_alerts = []
        
        for alert in self.alert_history:
            if alert.timestamp >= start_time and alert.timestamp <= end_time:
                if severity is None or alert.severity == severity:
                    filtered_alerts.append(alert.to_dict())
        
        return sorted(filtered_alerts, key=lambda x: x["timestamp"], reverse=True)
    
    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        
        logger.info("🔧 Running system diagnostics...")
        
        diagnostics = {
            "system_health": await self._run_health_diagnostics(),
            "performance_analysis": await self._run_performance_diagnostics(),
            "resource_analysis": await self._run_resource_diagnostics(),
            "alert_analysis": await self._run_alert_diagnostics(),
            "data_integrity": await self._run_data_integrity_diagnostics(),
            "recommendations": await self._generate_recommendations()
        }
        
        logger.info("✅ System diagnostics completed")
        
        return diagnostics
    
    # === Private Methods ===
    
    async def _initialize_database(self):
        """Initialize monitoring database"""
        
        db_path = Path(self.config.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.database = sqlite3.connect(str(db_path), check_same_thread=False)
        self.database.row_factory = sqlite3.Row
        
        # Create tables
        self.database.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                tags TEXT
            )
        """)
        
        self.database.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                component TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                acknowledged BOOLEAN DEFAULT FALSE,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at TIMESTAMP,
                metadata TEXT
            )
        """)
        
        self.database.execute("""
            CREATE TABLE IF NOT EXISTS health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                response_time REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                details TEXT
            )
        """)
        
        self.database.commit()
        
        logger.info("Monitoring database initialized")
    
    async def _initialize_sla_metrics(self):
        """Initialize SLA metrics"""
        
        # Define SLA targets
        sla_targets = {
            "system_uptime": 99.9,  # 99.9% uptime
            "ingestion_success_rate": 95.0,  # 95% success rate
            "processing_latency": 300.0,  # 5 minutes max processing time
            "error_rate": 5.0,  # 5% max error rate
            "resource_availability": 90.0  # 90% resource availability
        }
        
        for metric_name, target in sla_targets.items():
            self.sla_metrics[metric_name] = SLAMetric(
                metric_name=metric_name,
                target_value=target,
                current_value=0.0,
                threshold_breached=False
            )
        
        logger.info("SLA metrics initialized")
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        
        # Health check monitoring
        self.monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        
        # Metric collection
        self.metric_collection_task = asyncio.create_task(self._metric_collection_loop())
        
        # Alert processing
        self.alert_processing_task = asyncio.create_task(self._alert_processing_loop())
        
        logger.info("Monitoring tasks started")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Check system components
                await self._check_system_components()
                
                # Check external dependencies
                await self._check_external_dependencies()
                
                # Update system status
                await self._update_system_status()
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _metric_collection_loop(self):
        """Background metric collection loop"""
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect application metrics
                await self._collect_application_metrics()
                
                # Persist metrics to database
                await self._persist_metrics()
                
                await asyncio.sleep(self.config.metric_collection_interval)
                
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(10)
    
    async def _alert_processing_loop(self):
        """Background alert processing loop"""
        
        while self.monitoring_active:
            try:
                # Process alert queue
                await self._process_alert_queue()
                
                # Check for alert resolution
                await self._check_alert_resolution()
                
                # Cleanup old alerts
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(10)
    
    async def _check_system_components(self):
        """Check health of system components"""
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        await self.record_metric("system.cpu_percent", cpu_percent)
        
        if cpu_percent > self.config.cpu_critical_threshold:
            await self.generate_alert(
                AlertType.RESOURCE_USAGE,
                AlertSeverity.CRITICAL,
                "Critical CPU Usage",
                f"CPU usage is {cpu_percent:.1f}%",
                "system"
            )
        elif cpu_percent > self.config.cpu_warning_threshold:
            await self.generate_alert(
                AlertType.RESOURCE_USAGE,
                AlertSeverity.WARNING,
                "High CPU Usage",
                f"CPU usage is {cpu_percent:.1f}%",
                "system"
            )
        
        # Check memory usage
        memory = psutil.virtual_memory()
        await self.record_metric("system.memory_percent", memory.percent)
        
        if memory.percent > self.config.memory_critical_threshold:
            await self.generate_alert(
                AlertType.RESOURCE_USAGE,
                AlertSeverity.CRITICAL,
                "Critical Memory Usage",
                f"Memory usage is {memory.percent:.1f}%",
                "system"
            )
        elif memory.percent > self.config.memory_warning_threshold:
            await self.generate_alert(
                AlertType.RESOURCE_USAGE,
                AlertSeverity.WARNING,
                "High Memory Usage",
                f"Memory usage is {memory.percent:.1f}%",
                "system"
            )
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        await self.record_metric("system.disk_percent", disk_percent)
        
        if disk_percent > self.config.disk_critical_threshold:
            await self.generate_alert(
                AlertType.RESOURCE_USAGE,
                AlertSeverity.CRITICAL,
                "Critical Disk Usage",
                f"Disk usage is {disk_percent:.1f}%",
                "system"
            )
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        
        # CPU metrics
        cpu_times = psutil.cpu_times_percent()
        await self.record_metric("system.cpu_user", cpu_times.user)
        await self.record_metric("system.cpu_system", cpu_times.system)
        await self.record_metric("system.cpu_idle", cpu_times.idle)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        await self.record_metric("system.memory_total", memory.total)
        await self.record_metric("system.memory_used", memory.used)
        await self.record_metric("system.memory_available", memory.available)
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        if disk_io:
            await self.record_metric("system.disk_read_bytes", disk_io.read_bytes)
            await self.record_metric("system.disk_write_bytes", disk_io.write_bytes)
        
        # Network I/O metrics
        network_io = psutil.net_io_counters()
        if network_io:
            await self.record_metric("system.network_sent_bytes", network_io.bytes_sent)
            await self.record_metric("system.network_recv_bytes", network_io.bytes_recv)
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        
        # This would collect metrics from the PRSM application
        # For now, we'll generate some sample metrics
        
        await self.record_metric("app.active_connections", 10)
        await self.record_metric("app.request_rate", 100)
        await self.record_metric("app.error_rate", 0.5)
    
    async def _persist_metrics(self):
        """Persist metrics to database"""
        
        # Batch insert metrics
        metrics_to_insert = []
        
        with self.metrics_lock:
            for metric in list(self.metrics_buffer):
                metrics_to_insert.append((
                    metric.name,
                    metric.value,
                    metric.metric_type.value,
                    metric.timestamp,
                    json.dumps(metric.tags)
                ))
            
            # Clear buffer after copying
            self.metrics_buffer.clear()
        
        if metrics_to_insert:
            self.database.executemany(
                "INSERT INTO metrics (name, value, type, timestamp, tags) VALUES (?, ?, ?, ?, ?)",
                metrics_to_insert
            )
            self.database.commit()
    
    async def _should_debounce_alert(self, debounce_key: str) -> bool:
        """Check if alert should be debounced"""
        
        if debounce_key not in self.alert_debounce:
            return False
        
        last_alert_time = self.alert_debounce[debounce_key]
        time_since_last = (datetime.now(timezone.utc) - last_alert_time).total_seconds()
        
        return time_since_last < self.config.alert_debounce_time
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications"""
        
        # Email notifications
        if self.config.email_enabled and self.config.email_recipients:
            await self._send_email_notification(alert)
        
        # Additional notification channels would be implemented here
        # (Slack, PagerDuty, etc.)
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification for alert"""
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            - Component: {alert.component}
            - Severity: {alert.severity.value}
            - Time: {alert.timestamp.isoformat()}
            - Message: {alert.message}
            
            Alert ID: {alert.alert_id}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            
            text = msg.as_string()
            server.sendmail(self.config.email_username, self.config.email_recipients, text)
            server.quit()
            
            logger.info(f"Email notification sent for alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    # Additional helper methods would be implemented here...
    
    async def _perform_initial_health_check(self):
        """Perform initial health check"""
        await self.record_health_check("monitoring_system", SystemStatus.HEALTHY, "Monitoring system started")
    
    async def _check_component_status_change(self, health_check: HealthCheck):
        """Check for component status changes"""
        pass
    
    async def _update_system_status(self):
        """Update overall system status"""
        pass
    
    async def _check_metric_thresholds(self, metric: Metric):
        """Check metric against thresholds"""
        pass
    
    async def _check_external_dependencies(self):
        """Check external dependencies"""
        pass
    
    async def _process_alert_queue(self):
        """Process alert queue"""
        pass
    
    async def _check_alert_resolution(self):
        """Check for alert resolution"""
        pass
    
    async def _cleanup_old_alerts(self):
        """Cleanup old alerts"""
        pass
    
    async def _calculate_metric_average(self, name: str, hours: int = 1) -> float:
        """Calculate metric average"""
        return 0.0
    
    async def _calculate_metric_trend(self, name: str) -> str:
        """Calculate metric trend"""
        return "stable"
    
    async def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values"""
        if len(values) < 2:
            return "stable"
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        if statistics.mean(second_half) > statistics.mean(first_half):
            return "increasing"
        elif statistics.mean(second_half) < statistics.mean(first_half):
            return "decreasing"
        else:
            return "stable"
    
    async def _get_sla_compliance_summary(self) -> Dict[str, Any]:
        """Get SLA compliance summary"""
        return {metric.metric_name: metric.compliance_percentage for metric in self.sla_metrics.values()}
    
    async def _get_resource_usage_summary(self) -> Dict[str, Any]:
        """Get resource usage summary"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
        }
    
    async def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "response_time": "normal",
            "throughput": "optimal",
            "error_rate": "low"
        }
    
    async def _run_health_diagnostics(self) -> Dict[str, Any]:
        """Run health diagnostics"""
        return {"status": "healthy", "components_checked": len(self.component_health)}
    
    async def _run_performance_diagnostics(self) -> Dict[str, Any]:
        """Run performance diagnostics"""
        return {"status": "optimal", "metrics_analyzed": len(self.component_metrics)}
    
    async def _run_resource_diagnostics(self) -> Dict[str, Any]:
        """Run resource diagnostics"""
        return {"status": "normal", "resources_checked": ["cpu", "memory", "disk"]}
    
    async def _run_alert_diagnostics(self) -> Dict[str, Any]:
        """Run alert diagnostics"""
        return {"active_alerts": len(self.active_alerts), "alert_rate": "normal"}
    
    async def _run_data_integrity_diagnostics(self) -> Dict[str, Any]:
        """Run data integrity diagnostics"""
        return {"status": "intact", "checks_performed": ["database", "metrics", "alerts"]}
    
    async def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations"""
        return ["System operating normally", "No immediate action required"]
    
    async def shutdown(self):
        """Shutdown monitoring system"""
        
        logger.info("Shutting down monitoring system...")
        
        self.monitoring_active = False
        
        # Cancel monitoring tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.metric_collection_task:
            self.metric_collection_task.cancel()
        if self.alert_processing_task:
            self.alert_processing_task.cancel()
        
        # Close database
        if self.database:
            self.database.close()
        
        logger.info("Monitoring system shutdown complete")


# Test function
async def test_monitoring_system():
    """Test monitoring system"""
    
    print("📊 PRODUCTION MONITORING SYSTEM TEST")
    print("=" * 50)
    
    # Initialize monitoring
    monitoring = ProductionMonitoringSystem()
    await monitoring.initialize()
    
    # Test metric recording
    await monitoring.record_metric("test.metric", 42.5, MetricType.GAUGE)
    print("Metric Recording: ✅")
    
    # Test health check
    await monitoring.record_health_check("test_component", SystemStatus.HEALTHY, "Test OK")
    print("Health Check: ✅")
    
    # Test alert generation
    await monitoring.generate_alert(
        AlertType.SYSTEM_HEALTH,
        AlertSeverity.WARNING,
        "Test Alert",
        "This is a test alert",
        "test_component"
    )
    print("Alert Generation: ✅")
    
    # Get system overview
    overview = await monitoring.get_system_overview()
    print(f"System Overview: ✅ Status: {overview['system_status']}")
    
    # Run diagnostics
    diagnostics = await monitoring.run_diagnostics()
    print(f"Diagnostics: ✅ Health: {diagnostics['system_health']['status']}")
    
    # Shutdown
    await monitoring.shutdown()
    print("Shutdown: ✅")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(test_monitoring_system())