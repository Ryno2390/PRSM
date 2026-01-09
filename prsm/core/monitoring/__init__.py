"""
PRSM Monitoring and Observability
=================================

Comprehensive monitoring, metrics, and observability framework for PRSM.
Provides real-time insights into system performance, resource usage,
and operational health.

Key Components:
- MetricsCollector: Collects and aggregates system metrics
- DashboardManager: Manages monitoring dashboards
- AlertManager: Handles alerting and notifications
- PerformanceProfiler: Profiles system performance
- HealthChecker: Monitors system health
- LogAnalyzer: Analyzes logs for insights

Example Usage:
    from prsm.core.monitoring import MetricsCollector, DashboardManager
    
    # Initialize monitoring
    metrics = MetricsCollector()
    dashboard = DashboardManager(metrics_collector=metrics)
    
    # Start monitoring
    await metrics.start_collection()
    await dashboard.start_server(port=3000)
"""

from .metrics import MetricsCollector, MetricsRegistry, CustomMetric
from .dashboard import DashboardManager, DashboardConfig
from .alerts import AlertManager, AlertRule, AlertChannel
from .profiler import PerformanceProfiler, ProfileResult
from .health import HealthChecker, HealthCheck
from .logs import LogAnalyzer, LogMetrics
from .validators import ValidationSuite, ValidationResult

__all__ = [
    "MetricsCollector",
    "MetricsRegistry", 
    "CustomMetric",
    "DashboardManager",
    "DashboardConfig",
    "AlertManager",
    "AlertRule",
    "AlertChannel",
    "PerformanceProfiler",
    "ProfileResult",
    "HealthChecker",
    "HealthCheck",
    "LogAnalyzer",
    "LogMetrics",
    "ValidationSuite",
    "ValidationResult",
]

# Version info
__version__ = "0.1.0"
__author__ = "PRSM Team"
__description__ = "Monitoring and observability framework for PRSM"
