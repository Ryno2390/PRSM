#!/usr/bin/env python3
"""
PRSM Advanced Analytics & Business Intelligence System
=====================================================

Enterprise-grade analytics and business intelligence platform for PRSM,
providing real-time insights, interactive dashboards, and comprehensive
reporting capabilities.

Core Components:
- Real-time metrics collection and aggregation
- Interactive visualization and dashboard framework
- Business intelligence query engine
- Customizable reporting system
- Multi-dimensional analytics
"""

from .dashboard_manager import DashboardManager, Dashboard
from .metrics_collector import MetricsCollector, MetricDefinition
from .visualization_engine import VisualizationEngine, ChartType
from .bi_query_engine import BusinessIntelligenceEngine, QueryBuilder
from .real_time_processor import RealTimeProcessor, StreamProcessor

__version__ = "1.0.0"
__author__ = "PRSM Core Team"

__all__ = [
    'DashboardManager',
    'Dashboard', 
    'MetricsCollector',
    'MetricDefinition',
    'VisualizationEngine',
    'ChartType',
    'BusinessIntelligenceEngine',
    'QueryBuilder',
    'RealTimeProcessor',
    'StreamProcessor'
]