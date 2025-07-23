#!/usr/bin/env python3
"""
NWTN Meta-Reasoning Engine Monitoring Components
===============================================

Health monitoring, performance tracking, and system observability components
for the NWTN meta-reasoning system.
"""

from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from .types import (
    ReasoningEngine, 
    EngineHealthStatus,
    EnginePerformanceMetrics,
    PerformanceMetric,
    PerformanceCategory,
    PerformanceSnapshot,
    PerformanceProfile,
    FailureEvent,
    RecoveryStrategy
)


class EngineHealthMonitor:
    """Monitor and track health of reasoning engines"""
    
    def __init__(self) -> None:
        self.engine_metrics: Dict[ReasoningEngine, EnginePerformanceMetrics] = {}
        self.monitoring_enabled = True
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = None
        
        # Initialize metrics for all engines
        for engine_type in ReasoningEngine:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(
                engine_id=f"{engine_type.value}_engine",
                engine_name=engine_type.value
            )
    
    def record_execution(self, engine_type: ReasoningEngine, execution_time: float, 
                        quality_score: float, confidence_score: float, success: bool, 
                        error_type: Optional[str] = None) -> None:
        """Record an execution result"""
        if not self.monitoring_enabled:
            return
        
        if engine_type not in self.engine_metrics:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(
                engine_id=f"{engine_type.value}_engine",
                engine_name=engine_type.value
            )
        
        metrics = self.engine_metrics[engine_type]
        metrics.response_time_ms = execution_time * 1000  # Convert to ms
        metrics.reasoning_quality_score = quality_score
        metrics.accuracy_score = confidence_score
        
        if success:
            metrics.success_rate = min(100.0, metrics.success_rate + 1.0)
            metrics.error_rate = max(0.0, metrics.error_rate - 0.5)
        else:
            metrics.error_rate = min(100.0, metrics.error_rate + 2.0)
            metrics.success_rate = max(0.0, metrics.success_rate - 1.0)
            metrics.last_error = error_type
        
        # Update health status based on performance
        composite_score = metrics.calculate_composite_score()
        if composite_score >= 80:
            metrics.health_status = EngineHealthStatus.HEALTHY
        elif composite_score >= 60:
            metrics.health_status = EngineHealthStatus.DEGRADED
        elif composite_score >= 40:
            metrics.health_status = EngineHealthStatus.UNHEALTHY
        else:
            metrics.health_status = EngineHealthStatus.OFFLINE
    
    def record_timeout(self, engine_type: ReasoningEngine) -> None:
        """Record a timeout event"""
        if not self.monitoring_enabled:
            return
        
        if engine_type not in self.engine_metrics:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(
                engine_id=f"{engine_type.value}_engine",
                engine_name=engine_type.value
            )
        
        metrics = self.engine_metrics[engine_type]
        metrics.timeout_rate = min(100.0, metrics.timeout_rate + 5.0)
        metrics.error_rate = min(100.0, metrics.error_rate + 3.0)
        metrics.last_error = "timeout"
        metrics.health_status = EngineHealthStatus.DEGRADED
    
    def record_engine_execution(self, engine_type: ReasoningEngine, execution_time: float, 
                              success: bool, error: Optional[str] = None) -> None:
        """Record engine execution result with simplified signature"""
        if not self.monitoring_enabled:
            return
        
        # Use defaults for quality and confidence when not provided
        quality_score = 0.8 if success else 0.0
        confidence_score = 0.7 if success else 0.0
        
        self.record_execution(engine_type, execution_time, quality_score, 
                            confidence_score, success, error)
    
    def record_engine_timeout(self, engine_type: ReasoningEngine, timeout: float):
        """Record engine timeout event"""
        self.record_timeout(engine_type)
    
    def get_engine_health_status(self, engine_type: ReasoningEngine) -> EngineHealthStatus:
        """Get current health status for an engine"""
        if engine_type not in self.engine_metrics:
            return EngineHealthStatus.OFFLINE
        
        return self.engine_metrics[engine_type].health_status
    
    def get_engine_health_report(self, engine_type: ReasoningEngine) -> Dict[str, Any]:
        """Get comprehensive health report for an engine"""
        if engine_type not in self.engine_metrics:
            return {
                "engine_type": engine_type.value,
                "status": EngineHealthStatus.OFFLINE.value,
                "health_score": 0.0,
                "message": "No metrics available"
            }
        
        metrics = self.engine_metrics[engine_type]
        status = self.get_engine_health_status(engine_type)
        
        return {
            "engine_type": engine_type.value,
            "status": status.value,
            "health_score": metrics.calculate_composite_score(),
            "success_rate": metrics.success_rate,
            "error_rate": metrics.error_rate,
            "timeout_rate": metrics.timeout_rate,
            "average_execution_time": metrics.response_time_ms,
            "quality_score": metrics.reasoning_quality_score,
            "last_error": metrics.last_error,
            "uptime_seconds": metrics.uptime_seconds
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        healthy_engines = []
        degraded_engines = []
        unhealthy_engines = []
        offline_engines = []
        
        total_health_score = 0.0
        total_engines = 0
        
        for engine_type in ReasoningEngine:
            status = self.get_engine_health_status(engine_type)
            health_score = self.engine_metrics[engine_type].calculate_composite_score()
            
            if status == EngineHealthStatus.HEALTHY:
                healthy_engines.append(engine_type)
            elif status == EngineHealthStatus.DEGRADED:
                degraded_engines.append(engine_type)
            elif status == EngineHealthStatus.UNHEALTHY:
                unhealthy_engines.append(engine_type)
            else:
                offline_engines.append(engine_type)
            
            total_health_score += health_score
            total_engines += 1
        
        overall_health_score = total_health_score / total_engines if total_engines > 0 else 0.0
        
        return {
            "overall_health_score": overall_health_score,
            "total_engines": total_engines,
            "healthy_engines": [e.value for e in healthy_engines],
            "degraded_engines": [e.value for e in degraded_engines],
            "unhealthy_engines": [e.value for e in unhealthy_engines],
            "offline_engines": [e.value for e in offline_engines]
        }
    
    def get_overall_system_health(self) -> Dict[str, Any]:
        """Alias for get_system_health_summary for backward compatibility"""
        return self.get_system_health_summary()
    
    def get_healthy_engines(self) -> List[ReasoningEngine]:
        """Get list of healthy engines"""
        healthy_engines = []
        for engine_type in ReasoningEngine:
            status = self.get_engine_health_status(engine_type)
            if status in [EngineHealthStatus.HEALTHY, EngineHealthStatus.DEGRADED]:
                healthy_engines.append(engine_type)
        return healthy_engines
    
    def get_unhealthy_engines(self) -> List[ReasoningEngine]:
        """Get list of unhealthy engines"""
        unhealthy_engines = []
        for engine_type in ReasoningEngine:
            status = self.get_engine_health_status(engine_type)
            if status in [EngineHealthStatus.UNHEALTHY, EngineHealthStatus.OFFLINE]:
                unhealthy_engines.append(engine_type)
        return unhealthy_engines
    
    def enable_monitoring(self) -> None:
        """Enable health monitoring"""
        self.monitoring_enabled = True
    
    def disable_monitoring(self) -> None:
        """Disable health monitoring"""
        self.monitoring_enabled = False
    
    def reset_engine_metrics(self, engine_type: ReasoningEngine) -> None:
        """Reset metrics for a specific engine"""
        if engine_type in self.engine_metrics:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(
                engine_id=f"{engine_type.value}_engine",
                engine_name=engine_type.value
            )
    
    def reset_all_metrics(self) -> None:
        """Reset all engine metrics"""
        for engine_type in ReasoningEngine:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(
                engine_id=f"{engine_type.value}_engine",
                engine_name=engine_type.value
            )
    
    def should_use_engine(self, engine_type: ReasoningEngine) -> bool:
        """Check if an engine should be used based on health status"""
        if not self.monitoring_enabled:
            return True
        
        status = self.get_engine_health_status(engine_type)
        return status in [EngineHealthStatus.HEALTHY, EngineHealthStatus.DEGRADED]


class PerformanceTracker:
    """Advanced performance tracking system"""
    
    def __init__(self):
        self.profiles: Dict[ReasoningEngine, PerformanceProfile] = {}
        self.snapshots: List[PerformanceSnapshot] = []
        self.tracking_enabled = True
        self.max_snapshots = 10000  # Keep last 10k snapshots
        
        # Initialize profiles for all engines
        for engine_type in ReasoningEngine:
            self.profiles[engine_type] = PerformanceProfile(
                profile_id=f"{engine_type.value}_profile",
                name=f"{engine_type.value} Performance Profile",
                description=f"Performance tracking for {engine_type.value} reasoning engine"
            )
    
    def record_performance(self, engine_type: ReasoningEngine, metric_type: PerformanceMetric, 
                          value: float, context: Dict[str, Any] = None):
        """Record a performance measurement"""
        if not self.tracking_enabled:
            return
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_engines=len(ReasoningEngine),
            active_engines=len([e for e in ReasoningEngine if self.is_engine_active(e)]),
            healthy_engines=len([e for e in ReasoningEngine if self.is_engine_healthy(e)]),
            avg_response_time=value if metric_type == PerformanceMetric.RESPONSE_TIME else 0.0,
            total_requests=1,
            success_rate=100.0 if metric_type == PerformanceMetric.SUCCESS_RATE else 0.0,
            overall_quality_score=value if metric_type == PerformanceMetric.QUALITY_SCORE else 0.0,
            system_load=0.0,
            memory_usage_mb=value if metric_type == PerformanceMetric.RESOURCE_USAGE else 0.0
        )
        
        self.snapshots.append(snapshot)
        
        # Maintain snapshot limit
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        
        # Update profile
        if engine_type in self.profiles:
            self.profiles[engine_type].add_snapshot(snapshot)
    
    def is_engine_active(self, engine_type: ReasoningEngine) -> bool:
        """Check if engine is currently active"""
        # Simplified implementation - in real system would check actual engine status
        return True
    
    def is_engine_healthy(self, engine_type: ReasoningEngine) -> bool:
        """Check if engine is healthy"""
        # Simplified implementation - in real system would check health monitor
        return True
    
    def get_performance_profile(self, engine_type: ReasoningEngine) -> Optional[PerformanceProfile]:
        """Get performance profile for an engine"""
        return self.profiles.get(engine_type)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        if not self.snapshots:
            return {
                "total_measurements": 0,
                "avg_response_time": 0.0,
                "system_health": "unknown"
            }
        
        recent_snapshots = self.snapshots[-100:]  # Last 100 measurements
        
        avg_response_time = sum(s.avg_response_time for s in recent_snapshots) / len(recent_snapshots)
        avg_quality = sum(s.overall_quality_score for s in recent_snapshots) / len(recent_snapshots)
        avg_success_rate = sum(s.success_rate for s in recent_snapshots) / len(recent_snapshots)
        
        return {
            "total_measurements": len(self.snapshots),
            "recent_measurements": len(recent_snapshots),
            "avg_response_time": avg_response_time,
            "avg_quality_score": avg_quality,
            "avg_success_rate": avg_success_rate,
            "system_health": "healthy" if avg_success_rate > 80 else "degraded"
        }
    
    def get_engine_performance_stats(self, engine_type: ReasoningEngine) -> Dict[str, Any]:
        """Get detailed performance statistics for an engine"""
        profile = self.profiles.get(engine_type)
        if not profile:
            return {"error": "No profile found for engine"}
        
        return {
            "engine_type": engine_type.value,
            "avg_response_time": profile.avg_response_time,
            "avg_success_rate": profile.avg_success_rate,
            "avg_quality_score": profile.avg_quality_score,
            "min_response_time": profile.min_response_time,
            "max_response_time": profile.max_response_time,
            "reliability_score": profile.reliability_score,
            "total_snapshots": len(profile.snapshots)
        }
    
    def enable_tracking(self) -> None:
        """Enable performance tracking"""
        self.tracking_enabled = True
    
    def disable_tracking(self) -> None:
        """Disable performance tracking"""
        self.tracking_enabled = False
    
    def clear_snapshots(self) -> None:
        """Clear all performance snapshots"""
        self.snapshots.clear()
        for profile in self.profiles.values():
            profile.snapshots.clear()
    
    def export_performance_data(self, engine_type: Optional[ReasoningEngine] = None) -> Dict[str, Any]:
        """Export performance data for analysis"""
        if engine_type:
            profile = self.profiles.get(engine_type)
            if profile:
                return {
                    "engine_type": engine_type.value,
                    "profile": profile,
                    "snapshots": [s for s in self.snapshots if hasattr(s, 'engine_type') and s.engine_type == engine_type]
                }
            return {"error": "Engine not found"}
        
        return {
            "all_profiles": {e.value: profile for e, profile in self.profiles.items()},
            "all_snapshots": self.snapshots,
            "summary": self.get_performance_summary()
        }


class SystemHealthChecker:
    """Comprehensive system health checking"""
    
    def __init__(self, health_monitor: EngineHealthMonitor, performance_tracker: PerformanceTracker):
        self.health_monitor = health_monitor
        self.performance_tracker = performance_tracker
        self.check_interval = 60  # seconds
        self.last_check_time = None
        self.health_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        check_time = datetime.now(timezone.utc)
        
        # Get system health from monitor
        system_health = self.health_monitor.get_system_health_summary()
        
        # Get performance summary
        performance_summary = self.performance_tracker.get_performance_summary()
        
        # Calculate overall system status
        overall_status = self._calculate_overall_status(system_health, performance_summary)
        
        health_report = {
            "timestamp": check_time.isoformat(),
            "overall_status": overall_status,
            "system_health": system_health,
            "performance_summary": performance_summary,
            "engine_details": {
                engine.value: self.health_monitor.get_engine_health_report(engine)
                for engine in ReasoningEngine
            }
        }
        
        # Store in history
        self.health_history.append(health_report)
        if len(self.health_history) > self.max_history:
            self.health_history = self.health_history[-self.max_history:]
        
        self.last_check_time = check_time
        return health_report
    
    def _calculate_overall_status(self, system_health: Dict[str, Any], 
                                 performance_summary: Dict[str, Any]) -> str:
        """Calculate overall system status"""
        health_score = system_health.get("overall_health_score", 0)
        performance_health = performance_summary.get("system_health", "unknown")
        
        if health_score >= 80 and performance_health == "healthy":
            return "excellent"
        elif health_score >= 60 and performance_health in ["healthy", "degraded"]:
            return "good"
        elif health_score >= 40:
            return "fair"
        else:
            return "poor"
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over specified time period"""
        if not self.health_history:
            return {"error": "No health history available"}
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_checks = [
            check for check in self.health_history 
            if datetime.fromisoformat(check["timestamp"].replace('Z', '+00:00')) > cutoff_time
        ]
        
        if not recent_checks:
            return {"error": "No recent health data"}
        
        # Calculate trends
        health_scores = [check["system_health"]["overall_health_score"] for check in recent_checks]
        status_counts = {}
        for check in recent_checks:
            status = check["overall_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "time_period_hours": hours,
            "total_checks": len(recent_checks),
            "avg_health_score": sum(health_scores) / len(health_scores),
            "min_health_score": min(health_scores),
            "max_health_score": max(health_scores),
            "status_distribution": status_counts,
            "trend": self._calculate_trend(health_scores)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 3:
            return "stable"
        
        # Simple trend calculation
        first_half_avg = sum(values[:len(values)//2]) / (len(values)//2)
        second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        diff = second_half_avg - first_half_avg
        if diff > 5:
            return "improving"
        elif diff < -5:
            return "declining"
        else:
            return "stable"
    
    def get_alert_conditions(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        
        if not self.health_history:
            return alerts
        
        latest_check = self.health_history[-1]
        
        # Check overall health
        overall_health = latest_check["system_health"]["overall_health_score"]
        if overall_health < 50:
            alerts.append({
                "type": "critical",
                "message": f"System health critically low: {overall_health:.1f}%",
                "timestamp": latest_check["timestamp"]
            })
        elif overall_health < 70:
            alerts.append({
                "type": "warning", 
                "message": f"System health degraded: {overall_health:.1f}%",
                "timestamp": latest_check["timestamp"]
            })
        
        # Check individual engines
        for engine_name, engine_details in latest_check["engine_details"].items():
            if engine_details["status"] == "offline":
                alerts.append({
                    "type": "critical",
                    "message": f"Engine {engine_name} is offline",
                    "timestamp": latest_check["timestamp"]
                })
            elif engine_details["status"] == "unhealthy":
                alerts.append({
                    "type": "warning",
                    "message": f"Engine {engine_name} is unhealthy",
                    "timestamp": latest_check["timestamp"]
                })
        
        return alerts


# Export classes for use in other modules
__all__ = [
    'EngineHealthMonitor',
    'PerformanceTracker', 
    'SystemHealthChecker'
]