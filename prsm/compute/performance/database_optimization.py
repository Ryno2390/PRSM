"""
PRSM Database Performance Optimization
Advanced connection pooling, query optimization, read replicas, and performance monitoring
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import time
import statistics
import hashlib
import json
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool, Connection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy import text, event, Engine
from sqlalchemy.engine.events import PoolEvents
import redis.asyncio as aioredis
from .cache_invalidation import get_invalidation_manager, InvalidationEvent

logger = logging.getLogger(__name__)


class ConnectionRole(Enum):
    """Database connection roles"""
    PRIMARY = "primary"      # Read/write operations
    REPLICA = "replica"      # Read-only operations
    ANALYTICS = "analytics"  # Long-running analytical queries


class QueryType(Enum):
    """Query classification types"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    BULK_INSERT = "bulk_insert"
    ANALYTICAL = "analytical"
    MIGRATION = "migration"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str
    role: ConnectionRole = ConnectionRole.PRIMARY
    min_connections: int = 5
    max_connections: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    enable_ssl: bool = True
    ssl_ca_file: Optional[str] = None
    application_name: str = "prsm_api"


@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_hash: str
    query_type: QueryType
    execution_time_ms: float
    rows_affected: int
    connection_role: ConnectionRole
    timestamp: datetime
    database_name: str
    table_names: List[str] = field(default_factory=list)
    index_usage: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    error: Optional[str] = None


@dataclass
class ConnectionPoolMetrics:
    """Connection pool metrics"""
    pool_name: str
    role: ConnectionRole
    active_connections: int
    idle_connections: int
    total_connections: int
    max_connections: int
    checked_out_connections: int
    overflow_connections: int
    invalid_connections: int
    pool_hits: int
    pool_misses: int
    connection_creation_time_ms: float
    average_checkout_time_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class QueryOptimizer:
    """Intelligent query optimization and analysis"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.slow_query_threshold_ms = 1000
        self.query_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Query optimization rules
        self.optimization_rules = {
            "missing_index": self._suggest_missing_indexes,
            "inefficient_joins": self._suggest_join_optimizations,
            "full_table_scan": self._suggest_index_usage,
            "n_plus_one": self._detect_n_plus_one_queries,
            "unused_columns": self._suggest_column_pruning
        }
    
    def analyze_query(self, query: str, execution_plan: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze query for optimization opportunities"""
        
        query_hash = self._hash_query(query)
        
        analysis = {
            "query_hash": query_hash,
            "query_type": self._classify_query(query),
            "tables_involved": self._extract_table_names(query),
            "estimated_complexity": self._estimate_query_complexity(query),
            "optimization_suggestions": [],
            "index_recommendations": [],
            "rewrite_suggestions": []
        }
        
        # Apply optimization rules
        for rule_name, rule_func in self.optimization_rules.items():
            try:
                suggestions = rule_func(query, execution_plan)
                if suggestions:
                    analysis["optimization_suggestions"].extend(suggestions)
            except Exception as e:
                logger.error(f"Error applying optimization rule {rule_name}: {e}")
        
        return analysis
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query normalization"""
        # Normalize query by removing parameters and whitespace
        normalized = query.strip().lower()
        # Replace parameter placeholders with generic markers
        import re
        normalized = re.sub(r'\$\d+|\?|:[a-zA-Z_]\w*', '?', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify query type"""
        query_lower = query.strip().lower()
        
        if query_lower.startswith('select'):
            if 'join' in query_lower and 'group by' in query_lower:
                return QueryType.ANALYTICAL
            return QueryType.SELECT
        elif query_lower.startswith('insert'):
            if 'values' in query_lower and query_lower.count('values') > 1:
                return QueryType.BULK_INSERT
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        elif any(keyword in query_lower for keyword in ['create table', 'alter table', 'drop table']):
            return QueryType.MIGRATION
        else:
            return QueryType.SELECT
    
    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from query"""
        import re
        
        # Simple regex to extract table names (can be improved)
        table_pattern = r'\b(?:from|join|update|into)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(table_pattern, query.lower())
        
        return list(set(matches))
    
    def _estimate_query_complexity(self, query: str) -> str:
        """Estimate query complexity"""
        query_lower = query.lower()
        
        complexity_score = 0
        
        # Count complexity indicators
        complexity_score += query_lower.count('join') * 2
        complexity_score += query_lower.count('subquery') * 3
        complexity_score += query_lower.count('group by') * 2
        complexity_score += query_lower.count('order by') * 1
        complexity_score += query_lower.count('having') * 2
        complexity_score += query_lower.count('union') * 2
        
        if complexity_score >= 10:
            return "high"
        elif complexity_score >= 5:
            return "medium"
        else:
            return "low"
    
    def _suggest_missing_indexes(self, query: str, execution_plan: Optional[Dict]) -> List[str]:
        """Suggest missing indexes based on query patterns"""
        suggestions = []
        
        # Look for WHERE clauses without indexes
        import re
        where_conditions = re.findall(r'where\s+([a-zA-Z_]\w*)\s*[=<>]', query.lower())
        
        for column in where_conditions:
            suggestions.append(
                f"Consider adding index on column '{column}' for better WHERE clause performance"
            )
        
        # Look for JOIN conditions
        join_conditions = re.findall(r'on\s+([a-zA-Z_]\w*\.[a-zA-Z_]\w*)', query.lower())
        
        for join_col in join_conditions:
            suggestions.append(
                f"Consider adding index on join column '{join_col}'"
            )
        
        return suggestions
    
    def _suggest_join_optimizations(self, query: str, execution_plan: Optional[Dict]) -> List[str]:
        """Suggest JOIN optimizations"""
        suggestions = []
        
        join_count = query.lower().count('join')
        if join_count > 3:
            suggestions.append(
                f"Query has {join_count} JOINs - consider denormalization or materialized views"
            )
        
        if 'left join' in query.lower():
            suggestions.append(
                "Consider if LEFT JOINs can be replaced with INNER JOINs for better performance"
            )
        
        return suggestions
    
    def _suggest_index_usage(self, query: str, execution_plan: Optional[Dict]) -> List[str]:
        """Suggest better index usage"""
        suggestions = []
        
        if execution_plan and 'seq_scan' in str(execution_plan).lower():
            suggestions.append(
                "Query is performing sequential scans - consider adding appropriate indexes"
            )
        
        return suggestions
    
    def _detect_n_plus_one_queries(self, query: str, execution_plan: Optional[Dict]) -> List[str]:
        """Detect N+1 query patterns"""
        suggestions = []
        
        query_hash = self._hash_query(query)
        
        # Track query frequency
        if query_hash in self.query_stats:
            recent_executions = len([
                q for q in self.query_stats[query_hash]
                if time.time() - q['timestamp'] < 60  # Last minute
            ])
            
            if recent_executions > 50:  # High frequency execution
                suggestions.append(
                    "Potential N+1 query detected - consider using JOINs or batch loading"
                )
        
        return suggestions
    
    def _suggest_column_pruning(self, query: str, execution_plan: Optional[Dict]) -> List[str]:
        """Suggest removing unused columns"""
        suggestions = []
        
        if 'select *' in query.lower():
            suggestions.append(
                "Avoid SELECT * - specify only needed columns for better performance"
            )
        
        return suggestions
    
    async def record_query_execution(self, metrics: QueryMetrics):
        """Record query execution metrics"""
        
        query_data = {
            "execution_time_ms": metrics.execution_time_ms,
            "rows_affected": metrics.rows_affected,
            "timestamp": time.time(),
            "connection_role": metrics.connection_role.value,
            "cache_hit": metrics.cache_hit,
            "error": metrics.error
        }
        
        self.query_stats[metrics.query_hash].append(query_data)
        
        # Store slow queries for analysis
        if metrics.execution_time_ms > self.slow_query_threshold_ms:
            await self._store_slow_query(metrics)
    
    async def _store_slow_query(self, metrics: QueryMetrics):
        """Store slow query for analysis"""
        slow_query_data = {
            "query_hash": metrics.query_hash,
            "execution_time_ms": metrics.execution_time_ms,
            "timestamp": metrics.timestamp.isoformat(),
            "database": metrics.database_name,
            "tables": metrics.table_names,
            "connection_role": metrics.connection_role.value
        }
        
        await self.redis.lpush("slow_queries", json.dumps(slow_query_data))
        await self.redis.ltrim("slow_queries", 0, 999)  # Keep last 1000
        await self.redis.expire("slow_queries", 86400)  # 24 hour TTL
    
    def get_query_recommendations(self, query_hash: str) -> Dict[str, Any]:
        """Get optimization recommendations for a query"""
        
        if query_hash not in self.query_stats:
            return {"recommendations": [], "statistics": {}}
        
        stats = self.query_stats[query_hash]
        
        # Calculate statistics
        execution_times = [s["execution_time_ms"] for s in stats]
        
        statistics_data = {
            "total_executions": len(stats),
            "avg_execution_time_ms": statistics.mean(execution_times),
            "median_execution_time_ms": statistics.median(execution_times),
            "max_execution_time_ms": max(execution_times),
            "min_execution_time_ms": min(execution_times),
            "cache_hit_ratio": sum(1 for s in stats if s["cache_hit"]) / len(stats),
            "error_rate": sum(1 for s in stats if s["error"]) / len(stats)
        }
        
        # Generate recommendations
        recommendations = []
        
        if statistics_data["avg_execution_time_ms"] > self.slow_query_threshold_ms:
            recommendations.append("Query is consistently slow - consider optimization")
        
        if statistics_data["cache_hit_ratio"] < 0.3:
            recommendations.append("Low cache hit ratio - consider caching strategy")
        
        if statistics_data["error_rate"] > 0.01:
            recommendations.append("High error rate detected - investigate query issues")
        
        return {
            "recommendations": recommendations,
            "statistics": statistics_data
        }


class AdvancedConnectionPool:
    """Advanced database connection pool with load balancing and monitoring"""
    
    def __init__(self, configs: List[DatabaseConfig], redis_client: aioredis.Redis):
        self.configs = configs
        self.redis = redis_client
        self.pools: Dict[str, Pool] = {}
        self.engines: Dict[str, AsyncEngine] = {}
        self.session_makers: Dict[str, sessionmaker] = {}
        self.pool_metrics: Dict[str, ConnectionPoolMetrics] = {}
        
        # Load balancing
        self.replica_pools: List[str] = []
        self.primary_pool: Optional[str] = None
        self.analytics_pools: List[str] = []
        
        # Connection routing
        self.connection_weights: Dict[str, float] = {}
        self.health_check_interval = 30
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Query optimizer
        self.query_optimizer = QueryOptimizer(redis_client)
    
    async def initialize(self):
        """Initialize all connection pools"""
        
        for config in self.configs:
            try:
                # Create asyncpg pool
                pool_name = f"{config.role.value}_{config.host}_{config.port}"
                
                pool = await asyncpg.create_pool(
                    host=config.host,
                    port=config.port,
                    database=config.database,
                    user=config.username,
                    password=config.password,
                    min_size=config.min_connections,
                    max_size=config.max_connections,
                    command_timeout=config.pool_timeout,
                    server_settings={
                        'application_name': config.application_name,
                        'jit': 'off'  # Disable JIT for consistent performance
                    }
                )
                
                self.pools[pool_name] = pool
                
                # Create SQLAlchemy engine
                connection_string = self._build_connection_string(config)
                engine = create_async_engine(
                    connection_string,
                    poolclass=QueuePool if config.role == ConnectionRole.PRIMARY else NullPool,
                    pool_size=config.max_connections,
                    max_overflow=config.max_overflow,
                    pool_timeout=config.pool_timeout,
                    pool_recycle=config.pool_recycle,
                    pool_pre_ping=config.pool_pre_ping,
                    echo=False  # Set to True for SQL debugging
                )
                
                self.engines[pool_name] = engine
                
                # Create session maker
                session_maker = sessionmaker(
                    engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
                self.session_makers[pool_name] = session_maker
                
                # Set up connection routing
                if config.role == ConnectionRole.PRIMARY:
                    self.primary_pool = pool_name
                    self.connection_weights[pool_name] = 1.0
                elif config.role == ConnectionRole.REPLICA:
                    self.replica_pools.append(pool_name)
                    self.connection_weights[pool_name] = 1.0
                elif config.role == ConnectionRole.ANALYTICS:
                    self.analytics_pools.append(pool_name)
                    self.connection_weights[pool_name] = 1.0
                
                # Initialize metrics
                self.pool_metrics[pool_name] = ConnectionPoolMetrics(
                    pool_name=pool_name,
                    role=config.role,
                    active_connections=0,
                    idle_connections=config.min_connections,
                    total_connections=config.min_connections,
                    max_connections=config.max_connections,
                    checked_out_connections=0,
                    overflow_connections=0,
                    invalid_connections=0,
                    pool_hits=0,
                    pool_misses=0,
                    connection_creation_time_ms=0.0,
                    average_checkout_time_ms=0.0
                )
                
                logger.info(f"✅ Initialized {config.role.value} pool: {pool_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize pool for {config.host}:{config.port}: {e}")
                raise
        
        # Start health checking
        await self._start_health_checking()
        
        logger.info(f"✅ Database connection pools initialized ({len(self.pools)} pools)")
    
    def _build_connection_string(self, config: DatabaseConfig) -> str:
        """Build SQLAlchemy connection string"""
        
        connection_parts = [
            f"postgresql+asyncpg://{config.username}:{config.password}",
            f"@{config.host}:{config.port}/{config.database}"
        ]
        
        params = []
        if config.enable_ssl:
            params.append("sslmode=require")
            if config.ssl_ca_file:
                params.append(f"sslrootcert={config.ssl_ca_file}")
        
        if params:
            connection_parts.append("?" + "&".join(params))
        
        return "".join(connection_parts)
    
    async def _start_health_checking(self):
        """Start background health checking"""
        if self.health_check_task is None or self.health_check_task.done():
            self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self):
        """Background health checking loop"""
        
        while True:
            try:
                for pool_name, pool in self.pools.items():
                    try:
                        # Test connection
                        start_time = time.time()
                        async with pool.acquire() as conn:
                            await conn.fetchval("SELECT 1")
                        
                        response_time = (time.time() - start_time) * 1000
                        
                        # Update health weight based on response time
                        if response_time < 10:
                            self.connection_weights[pool_name] = 1.0
                        elif response_time < 50:
                            self.connection_weights[pool_name] = 0.8
                        elif response_time < 100:
                            self.connection_weights[pool_name] = 0.5
                        else:
                            self.connection_weights[pool_name] = 0.2
                        
                        # Update pool metrics
                        await self._update_pool_metrics(pool_name, pool)
                        
                    except Exception as e:
                        logger.warning(f"Health check failed for {pool_name}: {e}")
                        self.connection_weights[pool_name] = 0.1  # Severely downweight
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _update_pool_metrics(self, pool_name: str, pool: Pool):
        """Update connection pool metrics"""
        
        metrics = self.pool_metrics[pool_name]
        
        # Get pool statistics
        pool_stats = pool.get_stats()
        
        metrics.active_connections = pool_stats.get('active_connections', 0)
        metrics.idle_connections = pool_stats.get('idle_connections', 0)
        metrics.total_connections = metrics.active_connections + metrics.idle_connections
        metrics.timestamp = datetime.now(timezone.utc)
        
        # Store metrics in Redis for monitoring
        await self.redis.setex(
            f"pool_metrics:{pool_name}",
            300,  # 5 minute TTL
            json.dumps({
                "pool_name": pool_name,
                "role": metrics.role.value,
                "active_connections": metrics.active_connections,
                "idle_connections": metrics.idle_connections,
                "total_connections": metrics.total_connections,
                "max_connections": metrics.max_connections,
                "timestamp": metrics.timestamp.isoformat()
            })
        )
    
    def _select_pool(self, query_type: QueryType, prefer_role: Optional[ConnectionRole] = None) -> str:
        """Select optimal pool based on query type and load balancing"""
        
        # Determine target role
        target_role = prefer_role
        if not target_role:
            if query_type in [QueryType.SELECT]:
                target_role = ConnectionRole.REPLICA
            elif query_type == QueryType.ANALYTICAL:
                target_role = ConnectionRole.ANALYTICS
            else:
                target_role = ConnectionRole.PRIMARY
        
        # Get candidate pools
        candidates = []
        if target_role == ConnectionRole.PRIMARY and self.primary_pool:
            candidates = [self.primary_pool]
        elif target_role == ConnectionRole.REPLICA:
            candidates = self.replica_pools
        elif target_role == ConnectionRole.ANALYTICS:
            candidates = self.analytics_pools or self.replica_pools
        
        # Fallback to primary if no candidates
        if not candidates and self.primary_pool:
            candidates = [self.primary_pool]
        
        if not candidates:
            raise RuntimeError("No available database connections")
        
        # Weighted selection based on health
        if len(candidates) == 1:
            return candidates[0]
        
        # Select based on weights (higher weight = more likely to be selected)
        total_weight = sum(self.connection_weights.get(pool, 0.1) for pool in candidates)
        
        if total_weight <= 0:
            return candidates[0]  # Fallback
        
        import random
        selection_point = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for pool_name in candidates:
            cumulative_weight += self.connection_weights.get(pool_name, 0.1)
            if cumulative_weight >= selection_point:
                return pool_name
        
        return candidates[0]  # Fallback
    
    @asynccontextmanager
    async def get_connection(self, query_type: QueryType = QueryType.SELECT,
                           prefer_role: Optional[ConnectionRole] = None):
        """Get database connection with load balancing"""
        
        pool_name = self._select_pool(query_type, prefer_role)
        pool = self.pools[pool_name]
        
        start_time = time.time()
        
        try:
            async with pool.acquire() as connection:
                checkout_time = (time.time() - start_time) * 1000
                
                # Update metrics
                metrics = self.pool_metrics[pool_name]
                metrics.pool_hits += 1
                metrics.average_checkout_time_ms = (
                    (metrics.average_checkout_time_ms * (metrics.pool_hits - 1) + checkout_time) 
                    / metrics.pool_hits
                )
                
                yield connection, pool_name
                
        except Exception as e:
            metrics = self.pool_metrics[pool_name]
            metrics.pool_misses += 1
            logger.error(f"Connection acquisition failed from {pool_name}: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self, query_type: QueryType = QueryType.SELECT,
                         prefer_role: Optional[ConnectionRole] = None):
        """Get SQLAlchemy session with load balancing"""
        
        pool_name = self._select_pool(query_type, prefer_role)
        session_maker = self.session_makers[pool_name]
        
        async with session_maker() as session:
            try:
                yield session, pool_name
            except Exception as e:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def execute_optimized_query(self, query: str, parameters: Optional[Dict] = None,
                                    query_type: Optional[QueryType] = None,
                                    prefer_role: Optional[ConnectionRole] = None) -> Tuple[Any, QueryMetrics]:
        """Execute query with optimization and monitoring"""
        
        # Analyze query
        if not query_type:
            query_type = self.query_optimizer._classify_query(query)
        
        query_hash = self.query_optimizer._hash_query(query)
        
        start_time = time.time()
        error = None
        result = None
        rows_affected = 0
        
        try:
            async with self.get_connection(query_type, prefer_role) as (conn, pool_name):
                
                if parameters:
                    result = await conn.fetch(query, *parameters.values())
                else:
                    result = await conn.fetch(query)
                
                rows_affected = len(result) if result else 0
                
        except Exception as e:
            error = str(e)
            logger.error(f"Query execution failed: {e}")
            raise
        
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Create metrics
            metrics = QueryMetrics(
                query_hash=query_hash,
                query_type=query_type,
                execution_time_ms=execution_time_ms,
                rows_affected=rows_affected,
                connection_role=ConnectionRole.PRIMARY,  # Would be determined by pool selection
                timestamp=datetime.now(timezone.utc),
                database_name="prsm",
                table_names=self.query_optimizer._extract_table_names(query),
                error=error
            )
            
            # Record metrics
            await self.query_optimizer.record_query_execution(metrics)
        
        return result, metrics
    
    async def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        
        stats = {
            "pools": {},
            "total_connections": 0,
            "total_active": 0,
            "total_idle": 0,
            "health_weights": self.connection_weights.copy()
        }
        
        for pool_name, metrics in self.pool_metrics.items():
            stats["pools"][pool_name] = {
                "role": metrics.role.value,
                "active_connections": metrics.active_connections,
                "idle_connections": metrics.idle_connections,
                "total_connections": metrics.total_connections,
                "max_connections": metrics.max_connections,
                "pool_hits": metrics.pool_hits,
                "pool_misses": metrics.pool_misses,
                "average_checkout_time_ms": metrics.average_checkout_time_ms,
                "health_weight": self.connection_weights.get(pool_name, 0.0)
            }
            
            stats["total_connections"] += metrics.total_connections
            stats["total_active"] += metrics.active_connections
            stats["total_idle"] += metrics.idle_connections
        
        return stats
    
    async def close(self):
        """Close all connection pools"""
        
        # Stop health checking
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close pools
        for pool_name, pool in self.pools.items():
            try:
                await pool.close()
                logger.info(f"Closed pool: {pool_name}")
            except Exception as e:
                logger.error(f"Error closing pool {pool_name}: {e}")
        
        # Close engines
        for engine_name, engine in self.engines.items():
            try:
                await engine.dispose()
                logger.info(f"Disposed engine: {engine_name}")
            except Exception as e:
                logger.error(f"Error disposing engine {engine_name}: {e}")
        
        logger.info("All database connections closed")


# Global connection pool instance
connection_pool: Optional[AdvancedConnectionPool] = None


async def initialize_database_optimization(configs: List[DatabaseConfig], 
                                         redis_client: aioredis.Redis):
    """Initialize advanced database connection pooling"""
    global connection_pool
    
    connection_pool = AdvancedConnectionPool(configs, redis_client)
    await connection_pool.initialize()
    
    logger.info("✅ Advanced database optimization initialized")


def get_connection_pool() -> AdvancedConnectionPool:
    """Get the global connection pool instance"""
    if connection_pool is None:
        raise RuntimeError("Database optimization not initialized.")
    return connection_pool


async def shutdown_database_optimization():
    """Shutdown database optimization"""
    if connection_pool:
        await connection_pool.close()


# Convenience functions for common operations
async def execute_read_query(query: str, parameters: Optional[Dict] = None) -> Any:
    """Execute read-only query with automatic replica routing"""
    pool = get_connection_pool()
    result, _ = await pool.execute_optimized_query(
        query, parameters, QueryType.SELECT, ConnectionRole.REPLICA
    )
    return result


async def execute_write_query(query: str, parameters: Optional[Dict] = None) -> Any:
    """Execute write query on primary database"""
    pool = get_connection_pool()
    result, _ = await pool.execute_optimized_query(
        query, parameters, prefer_role=ConnectionRole.PRIMARY
    )
    return result


async def execute_analytical_query(query: str, parameters: Optional[Dict] = None) -> Any:
    """Execute analytical query on dedicated analytics database"""
    pool = get_connection_pool()
    result, _ = await pool.execute_optimized_query(
        query, parameters, QueryType.ANALYTICAL, ConnectionRole.ANALYTICS
    )
    return result


@asynccontextmanager
async def get_database_session(read_only: bool = False):
    """Get database session with automatic routing"""
    pool = get_connection_pool()
    
    query_type = QueryType.SELECT if read_only else QueryType.INSERT
    prefer_role = ConnectionRole.REPLICA if read_only else ConnectionRole.PRIMARY
    
    async with pool.get_session(query_type, prefer_role) as (session, pool_name):
        yield session