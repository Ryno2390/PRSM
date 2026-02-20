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

# Legacy imports (maintained for backward compatibility)
from .load_testing import LoadTestSuite, LoadTestResult, LoadTestConfig
from .scaling import AutoScaler, ScalingPolicy, HorizontalScaler, ScalingMetric, ScalingTrigger
from .caching import CacheManager, CDNIntegration, DistributedCache, CacheConfig
from .optimization import PerformanceOptimizer, QueryOptimizer, APIOptimizer
from .monitoring import APMIntegration, DistributedTracing, MetricsCollector

# New advanced performance optimization components
from .caching_system import (
    ComprehensiveCacheManager,
    CacheLayer,
    CacheStrategy,
    CacheKey,
    initialize_cache_system,
    get_cache_manager,
    shutdown_cache_system,
    cache_result
)

from .cache_invalidation import (
    CacheInvalidationManager,
    InvalidationEvent,
    InvalidationScope,
    InvalidationRule,
    initialize_invalidation_manager,
    get_invalidation_manager,
    shutdown_invalidation_manager
)

from .cache_monitoring import (
    CacheHealthMonitor,
    initialize_cache_monitoring,
    get_cache_monitoring_dashboard,
    shutdown_cache_monitoring
)

def get_cache_monitor():
    """Alias for backward compatibility"""
    return get_cache_monitoring_dashboard()

from .database_optimization import (
    AdvancedConnectionPool,
    QueryOptimizer as AdvancedQueryOptimizer,
    DatabaseConfig,
    ConnectionRole,
    QueryType,
    initialize_database_optimization,
    get_connection_pool,
    shutdown_database_optimization,
    execute_read_query,
    execute_write_query,
    execute_analytical_query,
    get_database_session
)

from .db_monitoring import (
    DatabaseMonitor,
    AlertSeverity,
    MetricType,
    initialize_database_monitoring,
    get_database_monitor,
    shutdown_database_monitoring
)

from .indexing_strategies import (
    IndexAnalyzer,
    IndexType,
    IndexRecommendation,
    initialize_index_analyzer,
    get_index_analyzer,
    shutdown_index_analyzer
)

from .replica_manager import (
    ReplicaManager,
    LoadBalancingStrategy,
    ReplicaStatus,
    initialize_replica_manager,
    get_replica_manager,
    shutdown_replica_manager,
    get_optimal_read_replica
)

__all__ = [
    # Legacy components (backward compatibility)
    "LoadTestSuite",
    "LoadTestResult", 
    "LoadTestConfig",
    "AutoScaler",
    "ScalingPolicy",
    "HorizontalScaler",
    "ScalingMetric",
    "ScalingTrigger",
    "CacheManager",
    "CDNIntegration", 
    "DistributedCache",
    "CacheConfig",
    "PerformanceOptimizer",
    "QueryOptimizer",
    "APIOptimizer",
    "APMIntegration",
    "DistributedTracing",
    "MetricsCollector",
    
    # Advanced Caching System
    "ComprehensiveCacheManager",
    "CacheLayer",
    "CacheStrategy", 
    "CacheKey",
    "initialize_cache_system",
    "get_cache_manager",
    "shutdown_cache_system",
    "cache_result",
    
    # Cache Invalidation
    "CacheInvalidationManager",
    "InvalidationEvent",
    "InvalidationScope",
    "InvalidationRule",
    "initialize_invalidation_manager",
    "get_invalidation_manager",
    "shutdown_invalidation_manager",
    
    # Cache Monitoring
    "CacheHealthMonitor",
    "initialize_cache_monitoring",
    "get_cache_monitor",
    "shutdown_cache_monitoring",
    
    # Advanced Database Optimization
    "AdvancedConnectionPool",
    "AdvancedQueryOptimizer",
    "DatabaseConfig",
    "ConnectionRole",
    "QueryType",
    "initialize_database_optimization",
    "get_connection_pool",
    "shutdown_database_optimization",
    "execute_read_query",
    "execute_write_query",
    "execute_analytical_query",
    "get_database_session",
    
    # Database Monitoring
    "DatabaseMonitor",
    "AlertSeverity",
    "MetricType",
    "initialize_database_monitoring",
    "get_database_monitor",
    "shutdown_database_monitoring",
    
    # Index Analysis
    "IndexAnalyzer",
    "IndexType",
    "IndexRecommendation",
    "initialize_index_analyzer",
    "get_index_analyzer",
    "shutdown_index_analyzer",
    
    # Replica Management
    "ReplicaManager",
    "LoadBalancingStrategy",
    "ReplicaStatus",
    "initialize_replica_manager",
    "get_replica_manager",
    "shutdown_replica_manager",
    "get_optimal_read_replica"
]