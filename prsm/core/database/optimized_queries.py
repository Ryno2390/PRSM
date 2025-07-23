"""
Database Query Optimization
============================

Optimized database queries with caching, connection pooling, and performance monitoring.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import threading

from sqlalchemy import text, select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.sql import ClauseElement

from ..config.manager import get_config
from ..errors.exceptions import ProcessingError, ConfigurationError
from ..caching.decorators import db_cache, invalidate_cache

logger = logging.getLogger(__name__)

T = TypeVar('T')

class QueryType(Enum):
    """Types of database queries for optimization"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"
    COMPLEX_JOIN = "complex_join"

@dataclass
class QueryPerformanceMetrics:
    """Performance metrics for database queries"""
    query_type: QueryType
    execution_time_ms: float
    rows_affected: int
    cache_hit: bool = False
    connection_wait_time_ms: float = 0.0
    query_hash: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass 
class ConnectionPoolStats:
    """Connection pool statistics"""
    pool_size: int
    checked_out: int
    overflow: int
    checked_in: int
    total_connections: int
    wait_time_avg_ms: float = 0.0

class OptimizedQueryBuilder:
    """Builder for creating optimized database queries"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self._query = None
        self._eager_loading = []
        self._filters = []
        self._ordering = []
        self._limit_value = None
        self._offset_value = None
        
    def select(self, *entities):
        """Create SELECT query"""
        self._query = select(*entities)
        return self
    
    def filter(self, *filters):
        """Add WHERE filters"""
        self._filters.extend(filters)
        return self
    
    def filter_by(self, **kwargs):
        """Add WHERE filters by keyword arguments"""
        for key, value in kwargs.items():
            # This would need to be adapted based on your actual model structure
            self._filters.append(text(f"{key} = :value").bindparam(value=value))
        return self
    
    def eager_load(self, *relationships):
        """Add eager loading for relationships"""
        self._eager_loading.extend(relationships)
        return self
    
    def order_by(self, *order_clauses):
        """Add ORDER BY clauses"""
        self._ordering.extend(order_clauses)
        return self
    
    def limit(self, limit: int):
        """Add LIMIT clause"""
        self._limit_value = limit
        return self
    
    def offset(self, offset: int):
        """Add OFFSET clause"""
        self._offset_value = offset
        return self
    
    def build(self) -> ClauseElement:
        """Build the final optimized query"""
        if self._query is None:
            raise ValueError("No base query specified")
        
        query = self._query
        
        # Apply filters
        if self._filters:
            query = query.where(and_(*self._filters))
        
        # Apply eager loading
        for relationship in self._eager_loading:
            if hasattr(relationship, '_is_relationship'):
                query = query.options(selectinload(relationship))
            else:
                query = query.options(joinedload(relationship))
        
        # Apply ordering
        if self._ordering:
            query = query.order_by(*self._ordering)
        
        # Apply pagination
        if self._limit_value is not None:
            query = query.limit(self._limit_value)
        
        if self._offset_value is not None:
            query = query.offset(self._offset_value)
        
        return query

class QueryOptimizer:
    """Database query optimizer with performance monitoring"""
    
    def __init__(self, engine, performance_monitor=None):
        self.engine = engine
        self.performance_monitor = performance_monitor
        self._query_cache: Dict[str, Any] = {}
        self._prepared_statements: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
    async def execute_optimized(
        self,
        query: Union[str, ClauseElement],
        parameters: Optional[Dict[str, Any]] = None,
        query_type: QueryType = QueryType.SELECT,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None
    ) -> Any:
        """Execute query with optimization and caching"""
        
        start_time = time.perf_counter()
        
        # Check cache first for SELECT queries
        if query_type == QueryType.SELECT and cache_key:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result is not None:
                execution_time = (time.perf_counter() - start_time) * 1000
                self._record_metrics(query_type, execution_time, 0, cache_hit=True)
                return cached_result
        
        # Execute query
        async with self._get_session() as session:
            connection_start = time.perf_counter()
            
            try:
                if isinstance(query, str):
                    # Text query
                    result = await session.execute(text(query), parameters or {})
                else:
                    # SQLAlchemy query object
                    result = await session.execute(query, parameters or {})
                
                # Handle different result types
                if query_type == QueryType.SELECT:
                    data = result.fetchall()
                    rows_affected = len(data)
                elif query_type in [QueryType.INSERT, QueryType.UPDATE, QueryType.DELETE]:
                    await session.commit()
                    data = result.rowcount
                    rows_affected = result.rowcount
                else:
                    data = result.fetchall()
                    rows_affected = len(data)
                
                execution_time = (time.perf_counter() - start_time) * 1000
                connection_wait_time = (time.perf_counter() - connection_start) * 1000
                
                # Cache result for SELECT queries
                if query_type == QueryType.SELECT and cache_key and data:
                    await self._cache_result(cache_key, data, cache_ttl)
                
                # Record metrics
                self._record_metrics(
                    query_type, execution_time, rows_affected,
                    connection_wait_time=connection_wait_time
                )
                
                return data
                
            except Exception as e:
                await session.rollback()
                execution_time = (time.perf_counter() - start_time) * 1000
                self._record_metrics(query_type, execution_time, 0)
                
                logger.error(f"Query execution error: {e}")
                raise ProcessingError(
                    f"Database query failed: {str(e)}",
                    component="database",
                    operation="query_execution",
                    context={
                        "query_type": query_type.value,
                        "execution_time_ms": execution_time
                    }
                ) from e
    
    @asynccontextmanager
    async def _get_session(self):
        """Get database session from pool"""
        SessionLocal = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        session = SessionLocal()
        try:
            yield session
        finally:
            await session.close()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached query result"""
        # This would integrate with the caching system
        from ..caching.cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        return await cache_manager.get("database", cache_key)
    
    async def _cache_result(self, cache_key: str, result: Any, ttl: Optional[int] = None) -> bool:
        """Cache query result"""
        from ..caching.cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        return await cache_manager.set("database", cache_key, result, ttl)
    
    def _record_metrics(
        self,
        query_type: QueryType,
        execution_time: float,
        rows_affected: int,
        cache_hit: bool = False,
        connection_wait_time: float = 0.0
    ) -> None:
        """Record query performance metrics"""
        if self.performance_monitor:
            metrics = QueryPerformanceMetrics(
                query_type=query_type,
                execution_time_ms=execution_time,
                rows_affected=rows_affected,
                cache_hit=cache_hit,
                connection_wait_time_ms=connection_wait_time
            )
            self.performance_monitor.record_query_metrics(metrics)
    
    def get_pool_stats(self) -> Optional[ConnectionPoolStats]:
        """Get connection pool statistics"""
        if hasattr(self.engine.pool, 'size'):
            pool = self.engine.pool
            return ConnectionPoolStats(
                pool_size=pool.size(),
                checked_out=pool.checkedout(),
                overflow=getattr(pool, 'overflow', 0)(),
                checked_in=pool.checkedin(),
                total_connections=pool.size() + getattr(pool, 'overflow', 0)()
            )
        return None

class DatabasePerformanceMonitor:
    """Monitor database performance and connection health"""
    
    def __init__(self):
        self.query_metrics: List[QueryPerformanceMetrics] = []
        self.connection_stats: List[ConnectionPoolStats] = []
        self._lock = threading.RLock()
        self.max_metrics_history = 10000
        
    def record_query_metrics(self, metrics: QueryPerformanceMetrics) -> None:
        """Record query performance metrics"""
        with self._lock:
            self.query_metrics.append(metrics)
            
            # Limit history size
            if len(self.query_metrics) > self.max_metrics_history:
                self.query_metrics = self.query_metrics[-self.max_metrics_history:]
    
    def record_connection_stats(self, stats: ConnectionPoolStats) -> None:
        """Record connection pool statistics"""
        with self._lock:
            self.connection_stats.append(stats)
            
            # Limit history size
            if len(self.connection_stats) > 1000:
                self.connection_stats = self.connection_stats[-1000:]
    
    def get_performance_summary(
        self,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get performance summary for the specified time window"""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            # Filter metrics by time window
            recent_metrics = [
                m for m in self.query_metrics 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {"message": "No metrics available for the specified time window"}
            
            # Calculate statistics
            total_queries = len(recent_metrics)
            avg_execution_time = sum(m.execution_time_ms for m in recent_metrics) / total_queries
            cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
            cache_hit_rate = cache_hits / total_queries if total_queries > 0 else 0
            
            # Query type breakdown
            query_type_stats = {}
            for query_type in QueryType:
                type_metrics = [m for m in recent_metrics if m.query_type == query_type]
                if type_metrics:
                    query_type_stats[query_type.value] = {
                        "count": len(type_metrics),
                        "avg_time_ms": sum(m.execution_time_ms for m in type_metrics) / len(type_metrics),
                        "total_rows": sum(m.rows_affected for m in type_metrics)
                    }
            
            # Slow queries (top 10)
            slow_queries = sorted(
                recent_metrics,
                key=lambda m: m.execution_time_ms,
                reverse=True
            )[:10]
            
            return {
                "time_window_minutes": time_window_minutes,
                "total_queries": total_queries,
                "average_execution_time_ms": round(avg_execution_time, 2),
                "cache_hit_rate": round(cache_hit_rate, 3),
                "query_type_breakdown": query_type_stats,
                "slowest_queries": [
                    {
                        "type": q.query_type.value,
                        "execution_time_ms": round(q.execution_time_ms, 2),
                        "rows_affected": q.rows_affected,
                        "timestamp": q.timestamp.isoformat()
                    }
                    for q in slow_queries
                ],
                "connection_pool_status": self._get_latest_connection_stats()
            }
    
    def _get_latest_connection_stats(self) -> Optional[Dict[str, Any]]:
        """Get the most recent connection pool statistics"""
        if not self.connection_stats:
            return None
        
        latest = self.connection_stats[-1]
        return {
            "pool_size": latest.pool_size,
            "checked_out": latest.checked_out,
            "checked_in": latest.checked_in,
            "overflow": latest.overflow,
            "utilization_rate": latest.checked_out / max(latest.total_connections, 1)
        }
    
    def identify_performance_issues(self) -> List[Dict[str, Any]]:
        """Identify potential performance issues"""
        issues = []
        
        if not self.query_metrics:
            return issues
        
        # Recent metrics (last hour)
        recent_metrics = [
            m for m in self.query_metrics
            if m.timestamp >= datetime.now(timezone.utc) - timedelta(hours=1)
        ]
        
        if not recent_metrics:
            return issues
        
        # Check for slow queries
        slow_threshold = 1000  # 1 second
        slow_queries = [m for m in recent_metrics if m.execution_time_ms > slow_threshold]
        if slow_queries:
            issues.append({
                "type": "slow_queries",
                "severity": "high" if len(slow_queries) > 10 else "medium",
                "description": f"Found {len(slow_queries)} slow queries (>{slow_threshold}ms)",
                "recommendation": "Review query optimization and indexing"
            })
        
        # Check cache hit rate
        cache_hit_rate = sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics)
        if cache_hit_rate < 0.3:  # Less than 30% cache hit rate
            issues.append({
                "type": "low_cache_hit_rate",
                "severity": "medium",
                "description": f"Low cache hit rate: {cache_hit_rate:.1%}",
                "recommendation": "Review caching strategy and TTL settings"
            })
        
        # Check connection pool utilization
        if self.connection_stats:
            latest_stats = self.connection_stats[-1]
            utilization = latest_stats.checked_out / max(latest_stats.total_connections, 1)
            if utilization > 0.8:  # More than 80% utilization
                issues.append({
                    "type": "high_connection_utilization",
                    "severity": "high",
                    "description": f"High connection pool utilization: {utilization:.1%}",
                    "recommendation": "Consider increasing pool size or optimizing query patterns"
                })
        
        return issues

# Optimized query helper functions

@db_cache(ttl_seconds=1800)  # 30 minutes cache
async def get_user_sessions_optimized(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    include_reasoning: bool = False
) -> List[Dict[str, Any]]:
    """Get user sessions with optimized query and caching"""
    # This would be implemented with your actual database models
    # Example implementation:
    
    # query = (
    #     select(Session)
    #     .where(Session.user_id == user_id)
    #     .order_by(Session.created_at.desc())
    #     .limit(limit)
    #     .offset(offset)
    # )
    
    # if include_reasoning:
    #     query = query.options(selectinload(Session.reasoning_steps))
    
    # This is a placeholder - implement with actual models
    return []

@db_cache(ttl_seconds=300)  # 5 minutes cache for stats
async def get_session_statistics(
    user_id: Optional[str] = None,
    time_window_hours: int = 24
) -> Dict[str, Any]:
    """Get session statistics with caching"""
    # Placeholder for actual implementation
    return {
        "total_sessions": 0,
        "average_duration": 0.0,
        "success_rate": 0.0
    }

async def create_optimized_engine(database_url: str, **kwargs) -> Any:
    """Create optimized database engine with connection pooling"""
    
    # Default optimized settings
    engine_kwargs = {
        "poolclass": QueuePool,
        "pool_size": 20,
        "max_overflow": 30,
        "pool_timeout": 30,
        "pool_recycle": 3600,  # Recycle connections every hour
        "pool_pre_ping": True,  # Validate connections
        "echo": False,  # Set to True for SQL debugging
        **kwargs
    }
    
    try:
        engine = create_async_engine(database_url, **engine_kwargs)
        
        logger.info("Created optimized database engine",
                   pool_size=engine_kwargs["pool_size"],
                   max_overflow=engine_kwargs["max_overflow"])
        
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise ConfigurationError(
            "Database engine creation failed",
            config_key="database_url",
            config_value=database_url
        ) from e

# Query analysis and optimization suggestions
class QueryAnalyzer:
    """Analyze queries and provide optimization suggestions"""
    
    def __init__(self, performance_monitor: DatabasePerformanceMonitor):
        self.performance_monitor = performance_monitor
    
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze query patterns and suggest optimizations"""
        metrics = self.performance_monitor.query_metrics
        
        if not metrics:
            return {"message": "No query data available for analysis"}
        
        # Analyze query frequency
        query_frequency = {}
        for metric in metrics:
            query_type = metric.query_type.value
            query_frequency[query_type] = query_frequency.get(query_type, 0) + 1
        
        # Find performance bottlenecks
        slow_query_threshold = 500  # 500ms
        slow_queries_by_type = {}
        
        for metric in metrics:
            if metric.execution_time_ms > slow_query_threshold:
                query_type = metric.query_type.value
                if query_type not in slow_queries_by_type:
                    slow_queries_by_type[query_type] = []
                slow_queries_by_type[query_type].append(metric)
        
        # Generate optimization suggestions
        suggestions = []
        
        # Frequent slow SELECT queries
        if QueryType.SELECT.value in slow_queries_by_type:
            select_slow = slow_queries_by_type[QueryType.SELECT.value]
            if len(select_slow) > 5:
                suggestions.append({
                    "type": "indexing",
                    "priority": "high",
                    "description": f"Consider adding indexes for {len(select_slow)} slow SELECT queries",
                    "query_count": len(select_slow)
                })
        
        # High cache miss rate
        cached_queries = [m for m in metrics if m.query_type == QueryType.SELECT]
        if cached_queries:
            cache_hits = sum(1 for m in cached_queries if m.cache_hit)
            cache_hit_rate = cache_hits / len(cached_queries)
            
            if cache_hit_rate < 0.4:
                suggestions.append({
                    "type": "caching",
                    "priority": "medium", 
                    "description": f"Low cache hit rate ({cache_hit_rate:.1%}). Consider longer TTL or better cache keys",
                    "current_hit_rate": cache_hit_rate
                })
        
        return {
            "query_frequency": query_frequency,
            "slow_queries_by_type": {
                k: len(v) for k, v in slow_queries_by_type.items()
            },
            "optimization_suggestions": suggestions,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }