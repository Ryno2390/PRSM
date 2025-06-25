"""
PRSM Performance Optimization and Scaling Module

🚀 PERFORMANCE OPTIMIZATION SYSTEM:
This module provides comprehensive performance optimization, load testing, and scaling
infrastructure for PRSM's production deployment.

Key Components:
- Load Testing: Comprehensive load testing suite with realistic workloads
- Horizontal Scaling: Auto-scaling infrastructure with Kubernetes integration
- Caching Strategy: Multi-layer caching with Redis cluster and CDN integration
- Performance Monitoring: Real-time metrics, APM, and distributed tracing
- Database Optimization: Query optimization, read replicas, and connection pooling

Integration with PRSM:
- Works with existing performance monitoring in improvement/performance_monitor.py
- Extends agent performance tracking in agents/routers/performance_tracker.py
- Integrates with Redis caching in core/redis_client.py
- Optimizes database operations in core/database.py
- Enhances WebSocket scaling in api/websocket_auth.py
"""

from .load_testing import LoadTestSuite, LoadTestResult, LoadTestConfig
from .scaling import AutoScaler, ScalingPolicy, HorizontalScaler, ScalingMetric, ScalingTrigger
from .caching import CacheManager, CDNIntegration, DistributedCache, CacheConfig
from .optimization import PerformanceOptimizer, QueryOptimizer, APIOptimizer
from .monitoring import APMIntegration, DistributedTracing, MetricsCollector

__all__ = [
    # Load Testing
    "LoadTestSuite",
    "LoadTestResult", 
    "LoadTestConfig",
    
    # Scaling
    "AutoScaler",
    "ScalingPolicy",
    "HorizontalScaler",
    "ScalingMetric",
    "ScalingTrigger",
    
    # Caching
    "CacheManager",
    "CDNIntegration", 
    "DistributedCache",
    "CacheConfig",
    
    # Optimization
    "PerformanceOptimizer",
    "QueryOptimizer",
    "APIOptimizer",
    
    # Monitoring
    "APMIntegration",
    "DistributedTracing",
    "MetricsCollector"
]