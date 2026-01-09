#!/usr/bin/env python3
"""
Real-time Pipeline Health Monitoring for NWTN
=============================================

Implements comprehensive monitoring and alerting system for NWTN pipeline
health, addressing the critical issue where pipeline components fail silently.

Features:
1. Real-time component health monitoring
2. Performance tracking and trend analysis
3. Alert system for critical failures
4. Health dashboard data generation
5. Predictive failure detection
6. Recovery recommendations

This ensures pipeline issues are detected immediately rather than
discovered after failed executions.
"""

import asyncio
import logging
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from collections import deque
import statistics
import threading

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Component health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: float
    timestamp: datetime
    status: HealthStatus
    threshold_warning: float = 0.7
    threshold_critical: float = 0.5
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate_status(self) -> HealthStatus:
        """Evaluate health status based on thresholds"""
        if self.value >= self.threshold_warning:
            return HealthStatus.HEALTHY
        elif self.value >= self.threshold_critical:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL


@dataclass
class ComponentHealth:
    """Health status for a pipeline component"""
    component_name: str
    status: HealthStatus
    last_updated: datetime
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    recent_errors: List[str] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    availability: float = 1.0
    response_time_avg: float = 0.0
    failure_count: int = 0
    success_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_metric(self, metric: HealthMetric):
        """Add or update health metric"""
        self.metrics[metric.name] = metric
        self.last_updated = datetime.now(timezone.utc)
        self._update_overall_status()
    
    def add_performance_data(self, response_time: float, success: bool):
        """Add performance data point"""
        self.performance_history.append(response_time)
        if len(self.performance_history) > 100:  # Keep last 100 data points
            self.performance_history.pop(0)
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # Update availability and response time
        total_attempts = self.success_count + self.failure_count
        self.availability = self.success_count / total_attempts if total_attempts > 0 else 1.0
        self.response_time_avg = statistics.mean(self.performance_history) if self.performance_history else 0.0
        
        self.last_updated = datetime.now(timezone.utc)
        self._update_overall_status()
    
    def add_error(self, error_message: str):
        """Add error message to recent errors"""
        self.recent_errors.append(f"{datetime.now(timezone.utc).isoformat()}: {error_message}")
        if len(self.recent_errors) > 10:  # Keep last 10 errors
            self.recent_errors.pop(0)
        
        self._update_overall_status()
    
    def _update_overall_status(self):
        """Update overall component health status"""
        if self.availability < 0.5:
            self.status = HealthStatus.FAILED
        elif self.availability < 0.8 or any(m.status == HealthStatus.CRITICAL for m in self.metrics.values()):
            self.status = HealthStatus.CRITICAL
        elif self.availability < 0.95 or any(m.status == HealthStatus.WARNING for m in self.metrics.values()):
            self.status = HealthStatus.WARNING
        else:
            self.status = HealthStatus.HEALTHY
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get component status summary"""
        return {
            'component_name': self.component_name,
            'status': self.status.value,
            'availability': self.availability,
            'response_time_avg': self.response_time_avg,
            'success_rate': self.success_count / max(self.success_count + self.failure_count, 1),
            'recent_errors_count': len(self.recent_errors),
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'metrics_count': len(self.metrics)
        }


@dataclass
class HealthAlert:
    """Health monitoring alert"""
    alert_id: str
    component_name: str
    alert_level: AlertLevel
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def resolve(self):
        """Mark alert as resolved"""
        self.resolved = True
        self.resolution_time = datetime.now(timezone.utc)


class AlertManager:
    """Manages health alerts and notifications"""
    
    def __init__(self, max_alerts: int = 1000):
        self.alerts: deque = deque(maxlen=max_alerts)
        self.alert_handlers: List[Callable] = []
        self.alert_count = 0
        
    def add_alert_handler(self, handler: Callable[[HealthAlert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    async def create_alert(self, 
                          component_name: str,
                          level: AlertLevel,
                          message: str,
                          metadata: Optional[Dict[str, Any]] = None) -> HealthAlert:
        """Create and process new alert"""
        alert = HealthAlert(
            alert_id=f"alert_{self.alert_count:06d}",
            component_name=component_name,
            alert_level=level,
            message=message,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        self.alert_count += 1
        
        logger.info(f"Created {level.value} alert for {component_name}: {message} (alert_id: {alert.alert_id})")
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                await self._safe_call_handler(handler, alert)
            except Exception as e:
                logger.error(f"Alert handler {handler.__name__} failed: {e}")
        
        return alert
    
    async def _safe_call_handler(self, handler: Callable, alert: HealthAlert):
        """Safely call alert handler"""
        if asyncio.iscoroutinefunction(handler):
            await handler(alert)
        else:
            handler(alert)
    
    def get_active_alerts(self, component_name: Optional[str] = None) -> List[HealthAlert]:
        """Get active (unresolved) alerts"""
        alerts = [alert for alert in self.alerts if not alert.resolved]
        
        if component_name:
            alerts = [alert for alert in alerts if alert.component_name == component_name]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_alerts = self.get_active_alerts()
        
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.value] = sum(1 for alert in active_alerts if alert.alert_level == level)
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'level_counts': level_counts,
            'most_recent': active_alerts[0].timestamp.isoformat() if active_alerts else None
        }


class PipelineHealthMonitor:
    """Main health monitoring system for NWTN pipeline"""
    
    def __init__(self, monitoring_interval: float = 30.0):
        self.components: Dict[str, ComponentHealth] = {}
        self.alert_manager = AlertManager()
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.session_history: List[Dict[str, Any]] = []
        self.system_metrics: Dict[str, Any] = {}
        
        # Component definitions
        self.expected_components = [
            'semantic_retrieval',
            'content_analysis', 
            'candidate_generation',
            'deduplication',
            'meta_reasoning',
            'wisdom_package',
            'synthesis'
        ]
        
        # Initialize component health tracking
        self._initialize_component_tracking()
        
        logger.info(f"Pipeline Health Monitor initialized with {len(self.expected_components)} expected components, monitoring interval: {monitoring_interval}s")
    
    def _initialize_component_tracking(self):
        """Initialize health tracking for expected components"""
        for component_name in self.expected_components:
            self.components[component_name] = ComponentHealth(
                component_name=component_name,
                status=HealthStatus.UNKNOWN,
                last_updated=datetime.now(timezone.utc)
            )
    
    async def start_monitoring(self):
        """Start real-time health monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Started real-time pipeline health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped pipeline health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._perform_health_checks()
                await self._analyze_trends()
                await self._check_alert_conditions()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all components"""
        current_time = datetime.now(timezone.utc)
        
        for component_name, health in self.components.items():
            # Check if component is stale (no updates in reasonable time)
            if health.last_updated:
                time_since_update = (current_time - health.last_updated).total_seconds()
                if time_since_update > 300:  # 5 minutes
                    health.status = HealthStatus.UNKNOWN
                    await self.alert_manager.create_alert(
                        component_name=component_name,
                        level=AlertLevel.WARNING,
                        message=f"No health updates for {time_since_update:.0f} seconds",
                        metadata={'time_since_update': time_since_update}
                    )
    
    async def _analyze_trends(self):
        """Analyze performance trends for predictive alerts"""
        for component_name, health in self.components.items():
            if len(health.performance_history) < 10:
                continue
            
            recent_performance = health.performance_history[-10:]
            older_performance = health.performance_history[-20:-10] if len(health.performance_history) >= 20 else []
            
            if older_performance:
                recent_avg = statistics.mean(recent_performance)
                older_avg = statistics.mean(older_performance)
                
                # Check for significant performance degradation
                if recent_avg > older_avg * 1.5:  # 50% slower
                    await self.alert_manager.create_alert(
                        component_name=component_name,
                        level=AlertLevel.WARNING,
                        message=f"Performance degradation detected: {recent_avg:.2f}s vs {older_avg:.2f}s average",
                        metadata={
                            'recent_avg': recent_avg,
                            'older_avg': older_avg,
                            'degradation_factor': recent_avg / older_avg
                        }
                    )
    
    async def _check_alert_conditions(self):
        """Check for alert conditions across all components"""
        critical_components = [
            name for name, health in self.components.items() 
            if health.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]
        ]
        
        if len(critical_components) > 2:
            await self.alert_manager.create_alert(
                component_name="pipeline_system",
                level=AlertLevel.CRITICAL,
                message=f"Multiple components in critical state: {', '.join(critical_components)}",
                metadata={'critical_components': critical_components}
            )
    
    async def record_component_execution(self,
                                       component_name: str,
                                       execution_time: float,
                                       success: bool,
                                       error_message: Optional[str] = None,
                                       metadata: Optional[Dict[str, Any]] = None):
        """Record component execution for health tracking"""
        if component_name not in self.components:
            self.components[component_name] = ComponentHealth(
                component_name=component_name,
                status=HealthStatus.UNKNOWN,
                last_updated=datetime.now(timezone.utc)
            )
        
        health = self.components[component_name]
        
        # Record performance data
        health.add_performance_data(execution_time, success)
        
        # Record error if provided
        if error_message:
            health.add_error(error_message)
        
        # Add execution metadata as metric
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    metric = HealthMetric(
                        name=f"execution_{key}",
                        value=value,
                        timestamp=datetime.now(timezone.utc),
                        status=HealthStatus.HEALTHY
                    )
                    health.add_metric(metric)
        
        # Create alerts for failures
        if not success:
            level = AlertLevel.ERROR if health.availability > 0.5 else AlertLevel.CRITICAL
            await self.alert_manager.create_alert(
                component_name=component_name,
                level=level,
                message=f"Component execution failed: {error_message or 'Unknown error'}",
                metadata={'execution_time': execution_time, 'availability': health.availability}
            )
        
        logger.debug(f"Recorded execution for {component_name} - component: {component_name}, success: {success}, execution_time: {execution_time:.3f}s, availability: {health.availability:.2f}")
    
    async def record_pipeline_session(self, session_data: Dict[str, Any]):
        """Record complete pipeline session for analysis"""
        self.session_history.append({
            **session_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Keep only recent sessions
        if len(self.session_history) > 100:
            self.session_history.pop(0)
        
        # Analyze session for health insights
        await self._analyze_session_health(session_data)
    
    async def _analyze_session_health(self, session_data: Dict[str, Any]):
        """Analyze session data for health insights"""
        pipeline_success = session_data.get('pipeline_success', False)
        health_score = session_data.get('pipeline_health_score', 0.0)
        components_executed = session_data.get('components_executed', 0)
        total_components = session_data.get('total_components', 7)
        
        # Create alerts based on session results
        if not pipeline_success:
            await self.alert_manager.create_alert(
                component_name="pipeline_system",
                level=AlertLevel.ERROR,
                message=f"Pipeline session failed: {components_executed}/{total_components} components executed",
                metadata=session_data
            )
        
        if health_score < 0.5:
            await self.alert_manager.create_alert(
                component_name="pipeline_system",
                level=AlertLevel.WARNING,
                message=f"Low pipeline health score: {health_score:.2f}",
                metadata={'health_score': health_score}
            )
    
    def get_system_health_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive health dashboard data"""
        current_time = datetime.now(timezone.utc)
        
        # Component health summary
        component_summary = {}
        for name, health in self.components.items():
            component_summary[name] = health.get_status_summary()
        
        # Overall system health
        healthy_count = sum(1 for h in self.components.values() if h.status == HealthStatus.HEALTHY)
        warning_count = sum(1 for h in self.components.values() if h.status == HealthStatus.WARNING)
        critical_count = sum(1 for h in self.components.values() if h.status in [HealthStatus.CRITICAL, HealthStatus.FAILED])
        
        overall_health = HealthStatus.HEALTHY
        if critical_count > 0:
            overall_health = HealthStatus.CRITICAL
        elif warning_count > 2:
            overall_health = HealthStatus.WARNING
        elif warning_count > 0:
            overall_health = HealthStatus.WARNING
        
        # Performance metrics
        recent_sessions = [s for s in self.session_history if 
                          (current_time - datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00'))).days < 1]
        
        avg_health_score = statistics.mean([s.get('pipeline_health_score', 0.0) for s in recent_sessions]) if recent_sessions else 0.0
        success_rate = sum(1 for s in recent_sessions if s.get('pipeline_success', False)) / max(len(recent_sessions), 1)
        
        return {
            'timestamp': current_time.isoformat(),
            'overall_health': overall_health.value,
            'component_health': component_summary,
            'health_counts': {
                'healthy': healthy_count,
                'warning': warning_count, 
                'critical': critical_count,
                'total': len(self.components)
            },
            'performance_metrics': {
                'avg_health_score_24h': avg_health_score,
                'success_rate_24h': success_rate,
                'total_sessions_24h': len(recent_sessions)
            },
            'alerts': self.alert_manager.get_alert_summary(),
            'monitoring_active': self.monitoring_active,
            'last_updated': current_time.isoformat()
        }
    
    def get_component_details(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for specific component"""
        if component_name not in self.components:
            return None
        
        health = self.components[component_name]
        
        return {
            'component_name': component_name,
            'status': health.status.value,
            'availability': health.availability,
            'response_time_avg': health.response_time_avg,
            'success_count': health.success_count,
            'failure_count': health.failure_count,
            'recent_errors': health.recent_errors,
            'performance_history': health.performance_history[-20:],  # Last 20 data points
            'metrics': {name: {
                'value': metric.value,
                'status': metric.status.value,
                'timestamp': metric.timestamp.isoformat(),
                'unit': metric.unit
            } for name, metric in health.metrics.items()},
            'last_updated': health.last_updated.isoformat() if health.last_updated else None,
            'active_alerts': len(self.alert_manager.get_active_alerts(component_name))
        }
    
    def get_health_recommendations(self) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        # Check for components with low availability
        low_availability_components = [
            name for name, health in self.components.items()
            if health.availability < 0.8
        ]
        
        if low_availability_components:
            recommendations.append(f"Investigate reliability issues in: {', '.join(low_availability_components)}")
        
        # Check for slow components
        slow_components = [
            name for name, health in self.components.items()
            if health.response_time_avg > 10.0  # 10 second threshold
        ]
        
        if slow_components:
            recommendations.append(f"Optimize performance of slow components: {', '.join(slow_components)}")
        
        # Check for components with many errors
        error_prone_components = [
            name for name, health in self.components.items()
            if len(health.recent_errors) > 5
        ]
        
        if error_prone_components:
            recommendations.append(f"Address recurring errors in: {', '.join(error_prone_components)}")
        
        # Check overall system health
        critical_count = sum(1 for h in self.components.values() if h.status in [HealthStatus.CRITICAL, HealthStatus.FAILED])
        if critical_count > 2:
            recommendations.append("CRITICAL: Multiple components failing - immediate attention required")
        
        # Recent session analysis
        if self.session_history:
            recent_success_rate = sum(1 for s in self.session_history[-10:] if s.get('pipeline_success', False)) / min(len(self.session_history), 10)
            if recent_success_rate < 0.7:
                recommendations.append(f"Pipeline success rate ({recent_success_rate:.1%}) below target - review component failures")
        
        return recommendations


# Export main classes
__all__ = [
    'PipelineHealthMonitor',
    'AlertManager',
    'ComponentHealth',
    'HealthAlert',
    'HealthStatus',
    'AlertLevel'
]