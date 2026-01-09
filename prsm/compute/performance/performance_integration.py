"""
PRSM Performance Integration System
Comprehensive integration and orchestration of all performance optimization components
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
import logging
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

from .caching_system import initialize_cache_system, get_cache_manager, shutdown_cache_system
from .cache_invalidation import initialize_invalidation_manager, get_invalidation_manager, shutdown_invalidation_manager
from .cache_monitoring import initialize_cache_monitoring, get_cache_monitor, shutdown_cache_monitoring
from .database_optimization import initialize_database_optimization, get_connection_pool, shutdown_database_optimization, DatabaseConfig
from .db_monitoring import initialize_database_monitoring, get_database_monitor, shutdown_database_monitoring
from .indexing_strategies import initialize_index_analyzer, get_index_analyzer, shutdown_index_analyzer
from .replica_manager import initialize_replica_manager, get_replica_manager, shutdown_replica_manager, LoadBalancingStrategy

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Comprehensive performance system configuration"""
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_cluster_nodes: Optional[List[Dict[str, Any]]] = None
    
    # Database Configuration
    database_configs: List[DatabaseConfig] = field(default_factory=list)
    replica_configs: List[DatabaseConfig] = field(default_factory=list)
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME
    
    # Cache Configuration
    memory_cache_config: Optional[Dict[str, Any]] = None
    redis_cache_config: Optional[Dict[str, Any]] = None
    
    # Monitoring Configuration
    alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None
    health_check_interval: int = 30
    monitoring_enabled: bool = True
    
    # Feature Flags
    enable_caching: bool = True
    enable_cache_invalidation: bool = True
    enable_cache_monitoring: bool = True
    enable_database_optimization: bool = True
    enable_database_monitoring: bool = True
    enable_index_analysis: bool = True
    enable_replica_management: bool = True


class PerformanceOrchestrator:
    """Central orchestrator for all performance optimization components"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.redis_client: Optional[aioredis.Redis] = None
        self.initialized_components: List[str] = []
        self.startup_time: Optional[datetime] = None
        
        # Component status tracking
        self.component_status: Dict[str, bool] = {
            "redis": False,
            "caching": False,
            "cache_invalidation": False,
            "cache_monitoring": False,
            "database_optimization": False,
            "database_monitoring": False,
            "index_analysis": False,
            "replica_management": False
        }
    
    async def initialize(self):
        """Initialize all performance optimization components"""
        self.startup_time = datetime.now(timezone.utc)
        logger.info("🚀 Initializing PRSM Performance Optimization System...")
        
        try:
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Initialize components in dependency order
            await self._initialize_components()
            
            # Verify system health
            await self._verify_system_health()
            
            logger.info(f"✅ Performance optimization system initialized in {len(self.initialized_components)} components")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize performance system: {e}")
            await self._cleanup_on_failure()
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if self.config.redis_cluster_nodes:
                # Redis Cluster mode
                from redis.asyncio.cluster import RedisCluster
                self.redis_client = RedisCluster(
                    startup_nodes=self.config.redis_cluster_nodes,
                    decode_responses=False,
                    skip_full_coverage_check=True
                )
            else:
                # Single Redis instance
                self.redis_client = aioredis.from_url(
                    self.config.redis_url,
                    decode_responses=False
                )
            
            # Test connection
            await self.redis_client.ping()
            self.component_status["redis"] = True
            self.initialized_components.append("redis")
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            raise
    
    async def _initialize_components(self):
        """Initialize all performance components"""
        
        # Initialize caching system
        if self.config.enable_caching:
            try:
                await initialize_cache_system(
                    memory_config=self.config.memory_cache_config,
                    redis_cluster_config=self.config.redis_cache_config
                )
                self.component_status["caching"] = True
                self.initialized_components.append("caching")
                logger.info("✅ Caching system initialized")
            except Exception as e:
                logger.error(f"❌ Caching system initialization failed: {e}")
        
        # Initialize cache invalidation
        if self.config.enable_cache_invalidation:
            try:
                await initialize_invalidation_manager(self.redis_client)
                self.component_status["cache_invalidation"] = True
                self.initialized_components.append("cache_invalidation")
                logger.info("✅ Cache invalidation system initialized")
            except Exception as e:
                logger.error(f"❌ Cache invalidation initialization failed: {e}")
        
        # Initialize cache monitoring
        if self.config.enable_cache_monitoring:
            try:
                await initialize_cache_monitoring(self.redis_client)
                self.component_status["cache_monitoring"] = True
                self.initialized_components.append("cache_monitoring")
                logger.info("✅ Cache monitoring initialized")
            except Exception as e:
                logger.error(f"❌ Cache monitoring initialization failed: {e}")
        
        # Initialize database optimization
        if self.config.enable_database_optimization and self.config.database_configs:
            try:
                await initialize_database_optimization(
                    self.config.database_configs,
                    self.redis_client
                )
                self.component_status["database_optimization"] = True
                self.initialized_components.append("database_optimization")
                logger.info("✅ Database optimization initialized")
            except Exception as e:
                logger.error(f"❌ Database optimization initialization failed: {e}")
        
        # Initialize database monitoring
        if self.config.enable_database_monitoring:
            try:
                await initialize_database_monitoring(
                    self.redis_client,
                    self.config.alert_thresholds
                )
                self.component_status["database_monitoring"] = True
                self.initialized_components.append("database_monitoring")
                logger.info("✅ Database monitoring initialized")
            except Exception as e:
                logger.error(f"❌ Database monitoring initialization failed: {e}")
        
        # Initialize index analysis
        if self.config.enable_index_analysis:
            try:
                await initialize_index_analyzer(self.redis_client)
                self.component_status["index_analysis"] = True
                self.initialized_components.append("index_analysis")
                logger.info("✅ Index analysis initialized")
            except Exception as e:
                logger.error(f"❌ Index analysis initialization failed: {e}")
        
        # Initialize replica management
        if self.config.enable_replica_management and self.config.replica_configs:
            try:
                await initialize_replica_manager(
                    self.redis_client,
                    self.config.replica_configs,
                    self.config.load_balancing_strategy
                )
                self.component_status["replica_management"] = True
                self.initialized_components.append("replica_management")
                logger.info("✅ Replica management initialized")
            except Exception as e:
                logger.error(f"❌ Replica management initialization failed: {e}")
    
    async def _verify_system_health(self):
        """Verify that all initialized components are healthy"""
        health_checks = []
        
        # Check caching system
        if self.component_status["caching"]:
            health_checks.append(self._check_cache_health())
        
        # Check database optimization
        if self.component_status["database_optimization"]:
            health_checks.append(self._check_database_health())
        
        # Check replica management
        if self.component_status["replica_management"]:
            health_checks.append(self._check_replica_health())
        
        # Run all health checks
        if health_checks:
            results = await asyncio.gather(*health_checks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Health check {i} failed: {result}")
    
    async def _check_cache_health(self):
        """Check caching system health"""
        try:
            cache_manager = get_cache_manager()
            stats = await cache_manager.get_comprehensive_stats()
            logger.debug(f"Cache system healthy: {len(stats)} metrics")
        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
            raise
    
    async def _check_database_health(self):
        """Check database optimization health"""
        try:
            connection_pool = get_connection_pool()
            stats = await connection_pool.get_pool_statistics()
            logger.debug(f"Database optimization healthy: {stats['total_connections']} connections")
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            raise
    
    async def _check_replica_health(self):
        """Check replica management health"""
        try:
            replica_manager = get_replica_manager()
            report = await replica_manager.get_replica_status_report()
            logger.debug(f"Replica management healthy: {report['summary']['total_replicas']} replicas")
        except Exception as e:
            logger.warning(f"Replica health check failed: {e}")
            raise
    
    async def _cleanup_on_failure(self):
        """Cleanup components on initialization failure"""
        logger.info("🧹 Cleaning up failed initialization...")
        await self.shutdown()
    
    async def shutdown(self):
        """Shutdown all performance optimization components"""
        logger.info("🛑 Shutting down performance optimization system...")
        
        # Shutdown components in reverse order
        shutdown_tasks = []
        
        if "replica_management" in self.initialized_components:
            shutdown_tasks.append(shutdown_replica_manager())
        
        if "index_analysis" in self.initialized_components:
            shutdown_tasks.append(shutdown_index_analyzer())
        
        if "database_monitoring" in self.initialized_components:
            shutdown_tasks.append(shutdown_database_monitoring())
        
        if "database_optimization" in self.initialized_components:
            shutdown_tasks.append(shutdown_database_optimization())
        
        if "cache_monitoring" in self.initialized_components:
            shutdown_tasks.append(shutdown_cache_monitoring())
        
        if "cache_invalidation" in self.initialized_components:
            shutdown_tasks.append(shutdown_invalidation_manager())
        
        if "caching" in self.initialized_components:
            shutdown_tasks.append(shutdown_cache_system())
        
        # Execute all shutdowns
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        # Reset state
        self.initialized_components.clear()
        self.component_status = {key: False for key in self.component_status.keys()}
        
        logger.info("✅ Performance optimization system shutdown complete")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "uptime_seconds": (
                (datetime.now(timezone.utc) - self.startup_time).total_seconds()
                if self.startup_time else 0
            ),
            "components": self.component_status.copy(),
            "initialized_components": self.initialized_components.copy(),
            "component_stats": {}
        }
        
        # Collect component statistics
        try:
            if self.component_status["caching"]:
                cache_manager = get_cache_manager()
                status["component_stats"]["caching"] = await cache_manager.get_comprehensive_stats()
        except Exception as e:
            logger.debug(f"Failed to get cache stats: {e}")
        
        try:
            if self.component_status["database_optimization"]:
                connection_pool = get_connection_pool()
                status["component_stats"]["database"] = await connection_pool.get_pool_statistics()
        except Exception as e:
            logger.debug(f"Failed to get database stats: {e}")
        
        try:
            if self.component_status["database_monitoring"]:
                db_monitor = get_database_monitor()
                status["component_stats"]["monitoring"] = await db_monitor.get_performance_summary()
        except Exception as e:
            logger.debug(f"Failed to get monitoring stats: {e}")
        
        try:
            if self.component_status["index_analysis"]:
                index_analyzer = get_index_analyzer()
                status["component_stats"]["indexing"] = await index_analyzer.get_index_health_report()
        except Exception as e:
            logger.debug(f"Failed to get index stats: {e}")
        
        try:
            if self.component_status["replica_management"]:
                replica_manager = get_replica_manager()
                status["component_stats"]["replicas"] = await replica_manager.get_replica_status_report()
        except Exception as e:
            logger.debug(f"Failed to get replica stats: {e}")
        
        return status
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "healthy": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {}
        }
        
        # Check Redis connectivity
        try:
            if self.redis_client:
                await self.redis_client.ping()
                health_status["checks"]["redis"] = {"status": "healthy", "response_time_ms": 0}
            else:
                health_status["checks"]["redis"] = {"status": "not_initialized"}
        except Exception as e:
            health_status["checks"]["redis"] = {"status": "failed", "error": str(e)}
            health_status["healthy"] = False
        
        # Check each component
        for component, enabled in self.component_status.items():
            if enabled:
                try:
                    if component == "caching":
                        cache_manager = get_cache_manager()
                        stats = await cache_manager.get_comprehensive_stats()
                        health_status["checks"][component] = {
                            "status": "healthy",
                            "memory_usage": stats.get("memory_cache", {}).get("memory_usage_bytes", 0)
                        }
                    
                    elif component == "database_optimization":
                        connection_pool = get_connection_pool()
                        stats = await connection_pool.get_pool_statistics()
                        health_status["checks"][component] = {
                            "status": "healthy",
                            "total_connections": stats["total_connections"],
                            "active_connections": stats["total_active"]
                        }
                    
                    elif component == "replica_management":
                        replica_manager = get_replica_manager()
                        report = await replica_manager.get_replica_status_report()
                        health_status["checks"][component] = {
                            "status": "healthy",
                            "healthy_replicas": report["summary"]["healthy_replicas"],
                            "failed_replicas": report["summary"]["failed_replicas"]
                        }
                        
                        if report["summary"]["failed_replicas"] > 0:
                            health_status["healthy"] = False
                    
                    else:
                        health_status["checks"][component] = {"status": "healthy"}
                
                except Exception as e:
                    health_status["checks"][component] = {"status": "failed", "error": str(e)}
                    health_status["healthy"] = False
            else:
                health_status["checks"][component] = {"status": "disabled"}
        
        return health_status


# Global performance orchestrator instance
performance_orchestrator: Optional[PerformanceOrchestrator] = None


async def initialize_performance_system(config: PerformanceConfig):
    """Initialize the comprehensive performance optimization system"""
    global performance_orchestrator
    
    performance_orchestrator = PerformanceOrchestrator(config)
    await performance_orchestrator.initialize()
    
    logger.info("🎉 PRSM Performance Optimization System fully initialized!")


def get_performance_orchestrator() -> PerformanceOrchestrator:
    """Get the global performance orchestrator instance"""
    if performance_orchestrator is None:
        raise RuntimeError("Performance system not initialized. Call initialize_performance_system() first.")
    return performance_orchestrator


async def shutdown_performance_system():
    """Shutdown the comprehensive performance optimization system"""
    if performance_orchestrator:
        await performance_orchestrator.shutdown()


@asynccontextmanager
async def performance_system_context(config: PerformanceConfig):
    """Context manager for performance system lifecycle"""
    try:
        await initialize_performance_system(config)
        yield get_performance_orchestrator()
    finally:
        await shutdown_performance_system()


# Convenience functions for common operations
async def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health status"""
    orchestrator = get_performance_orchestrator()
    return await orchestrator.perform_health_check()


async def get_performance_metrics() -> Dict[str, Any]:
    """Get comprehensive performance metrics"""
    orchestrator = get_performance_orchestrator()
    return await orchestrator.get_system_status()


# Example configuration factory
def create_default_config(
    redis_url: str = "redis://localhost:6379",
    database_host: str = "localhost",
    database_port: int = 5432,
    database_name: str = "prsm",
    database_user: str = "prsm_user",
    database_password: str = "prsm_password"
) -> PerformanceConfig:
    """Create default performance configuration"""
    
    primary_db_config = DatabaseConfig(
        host=database_host,
        port=database_port,
        database=database_name,
        username=database_user,
        password=database_password,
        role=DatabaseConfig.ConnectionRole.PRIMARY,
        min_connections=5,
        max_connections=20
    )
    
    return PerformanceConfig(
        redis_url=redis_url,
        database_configs=[primary_db_config],
        replica_configs=[],  # Add replica configs as needed
        memory_cache_config={
            "max_size": 10000,
            "max_memory_mb": 512
        },
        redis_cache_config={
            "nodes": [{"host": "localhost", "port": 6379}],
            "password": None,
            "max_connections_per_node": 20
        },
        enable_caching=True,
        enable_cache_invalidation=True,
        enable_cache_monitoring=True,
        enable_database_optimization=True,
        enable_database_monitoring=True,
        enable_index_analysis=True,
        enable_replica_management=False  # Disabled by default until replicas are configured
    )