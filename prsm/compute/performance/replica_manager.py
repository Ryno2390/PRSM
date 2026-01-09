"""
PRSM Database Replica Management
Advanced read replica management, load balancing, and failover handling
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import time
import statistics
import json
import logging
from collections import defaultdict, deque
import redis.asyncio as aioredis
from .database_optimization import DatabaseConfig, ConnectionRole, QueryType, get_connection_pool

logger = logging.getLogger(__name__)


class ReplicaStatus(Enum):
    """Read replica status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    LAGGING = "lagging"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for read replicas"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    LEAST_CONNECTIONS = "least_connections"
    GEOGRAPHIC = "geographic"
    RANDOM = "random"


@dataclass
class ReplicaHealth:
    """Read replica health metrics"""
    replica_id: str
    host: str
    port: int
    status: ReplicaStatus
    response_time_ms: float
    replication_lag_ms: float
    connection_count: int
    max_connections: int
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    last_check: datetime
    consecutive_failures: int = 0
    is_promoted: bool = False  # Promoted to primary
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailoverEvent:
    """Database failover event"""
    event_id: str
    event_type: str  # "planned", "automatic", "manual"
    old_primary: str
    new_primary: str
    affected_replicas: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    reason: str
    success: bool = False
    rollback_plan: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplicationTopology:
    """Database replication topology"""
    primary_node: str
    read_replicas: List[str]
    async_replicas: List[str]
    sync_replicas: List[str]
    cascade_replicas: Dict[str, List[str]]  # Parent -> children
    geographic_distribution: Dict[str, str]  # Replica -> region
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ReplicaManager:
    """Advanced database replica management system"""
    
    def __init__(self, redis_client: aioredis.Redis, 
                 replica_configs: List[DatabaseConfig],
                 load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME):
        self.redis = redis_client
        self.replica_configs = replica_configs
        self.load_balancing_strategy = load_balancing_strategy
        
        # Replica tracking
        self.replica_health: Dict[str, ReplicaHealth] = {}
        self.topology = ReplicationTopology(
            primary_node="",
            read_replicas=[],
            async_replicas=[],
            sync_replicas=[],
            cascade_replicas={},
            geographic_distribution={}
        )
        
        # Health monitoring
        self.health_check_interval = 15  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        self.lag_threshold_ms = 10000  # 10 seconds
        self.failure_threshold = 3  # consecutive failures before marking failed
        
        # Load balancing
        self.connection_weights: Dict[str, float] = {}
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.round_robin_index = 0
        
        # Failover management
        self.auto_failover_enabled = True
        self.failover_timeout_seconds = 300  # 5 minutes
        self.failover_history: List[FailoverEvent] = []
        self.failover_handlers: List[Callable] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Statistics
        self.stats = {
            "health_checks_performed": 0,
            "failovers_executed": 0,
            "replicas_promoted": 0,
            "load_balanced_queries": 0,
            "replica_recoveries": 0
        }
    
    async def initialize(self):
        """Initialize replica manager"""
        try:
            # Initialize replica health tracking
            await self._initialize_replica_health()
            
            # Discover replication topology
            await self._discover_topology()
            
            # Start health monitoring
            await self.start_health_monitoring()
            
            logger.info(f"✅ Replica manager initialized with {len(self.replica_configs)} replicas")
            
        except Exception as e:
            logger.error(f"Failed to initialize replica manager: {e}")
            raise
    
    async def _initialize_replica_health(self):
        """Initialize health tracking for all replicas"""
        for config in self.replica_configs:
            replica_id = f"{config.host}:{config.port}"
            
            self.replica_health[replica_id] = ReplicaHealth(
                replica_id=replica_id,
                host=config.host,
                port=config.port,
                status=ReplicaStatus.HEALTHY,
                response_time_ms=0.0,
                replication_lag_ms=0.0,
                connection_count=0,
                max_connections=config.max_connections,
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                last_check=datetime.now(timezone.utc)
            )
            
            # Initialize connection weight
            self.connection_weights[replica_id] = 1.0
    
    async def _discover_topology(self):
        """Discover current replication topology"""
        try:
            connection_pool = get_connection_pool()
            
            # Query replication information (PostgreSQL specific)
            replication_query = """
            SELECT 
                client_addr,
                client_port,
                state,
                sync_state,
                replay_lag
            FROM pg_stat_replication;
            """
            
            async with connection_pool.get_connection(QueryType.SELECT, ConnectionRole.PRIMARY) as (conn, pool_name):
                result = await conn.fetch(replication_query)
                
                for row in result:
                    client_addr = row.get('client_addr')
                    if client_addr:
                        replica_id = f"{client_addr}:{row.get('client_port', 5432)}"
                        
                        if replica_id in self.replica_health:
                            # Update topology based on sync state
                            sync_state = row.get('sync_state', 'async')
                            if sync_state == 'sync':
                                if replica_id not in self.topology.sync_replicas:
                                    self.topology.sync_replicas.append(replica_id)
                            else:
                                if replica_id not in self.topology.async_replicas:
                                    self.topology.async_replicas.append(replica_id)
                            
                            if replica_id not in self.topology.read_replicas:
                                self.topology.read_replicas.append(replica_id)
                
                self.topology.updated_at = datetime.now(timezone.utc)
                
        except Exception as e:
            logger.error(f"Error discovering replication topology: {e}")
    
    async def start_health_monitoring(self):
        """Start continuous health monitoring of replicas"""
        if self.health_check_task is None or self.health_check_task.done():
            self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
            logger.info("✅ Replica health monitoring started")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Replica health monitoring stopped")
    
    async def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        while True:
            try:
                # Check health of all replicas
                await self._check_all_replicas_health()
                
                # Update load balancing weights
                await self._update_load_balancing_weights()
                
                # Check for automatic failover conditions
                if self.auto_failover_enabled:
                    await self._check_failover_conditions()
                
                # Store health metrics
                await self._store_health_metrics()
                
                self.stats["health_checks_performed"] += 1
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in replica health monitoring: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_all_replicas_health(self):
        """Check health of all read replicas"""
        health_check_tasks = []
        
        for replica_id in self.replica_health.keys():
            task = asyncio.create_task(self._check_replica_health(replica_id))
            health_check_tasks.append(task)
        
        # Wait for all health checks to complete
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _check_replica_health(self, replica_id: str):
        """Check health of individual replica"""
        health = self.replica_health[replica_id]
        
        try:
            start_time = time.time()
            
            # Test connection and basic query
            connection_pool = get_connection_pool()
            async with connection_pool.get_connection(QueryType.SELECT) as (conn, pool_name):
                # Basic connectivity test
                await conn.fetchval("SELECT 1")
                
                # Get replication lag
                lag_result = await conn.fetchval("""
                    SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) * 1000
                """)
                health.replication_lag_ms = float(lag_result) if lag_result else 0.0
                
                # Get connection count
                conn_result = await conn.fetchval("""
                    SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()
                """)
                health.connection_count = int(conn_result) if conn_result else 0
            
            # Calculate response time
            health.response_time_ms = (time.time() - start_time) * 1000
            health.last_check = datetime.now(timezone.utc)
            
            # Update status based on health metrics
            new_status = self._determine_replica_status(health)
            
            if new_status != health.status:
                logger.info(f"Replica {replica_id} status changed: {health.status.value} -> {new_status.value}")
                health.status = new_status
                
                # Reset failure count on recovery
                if new_status == ReplicaStatus.HEALTHY:
                    health.consecutive_failures = 0
                    self.stats["replica_recoveries"] += 1
            
            # Record performance metrics
            self.performance_metrics[f"{replica_id}_response_time"].append(health.response_time_ms)
            self.performance_metrics[f"{replica_id}_lag"].append(health.replication_lag_ms)
            
        except Exception as e:
            logger.warning(f"Health check failed for replica {replica_id}: {e}")
            health.consecutive_failures += 1
            health.last_check = datetime.now(timezone.utc)
            
            # Mark as failed after consecutive failures
            if health.consecutive_failures >= self.failure_threshold:
                if health.status != ReplicaStatus.FAILED:
                    logger.error(f"Marking replica {replica_id} as FAILED after {health.consecutive_failures} failures")
                    health.status = ReplicaStatus.FAILED
    
    def _determine_replica_status(self, health: ReplicaHealth) -> ReplicaStatus:
        """Determine replica status based on health metrics"""
        
        # Check replication lag
        if health.replication_lag_ms > self.lag_threshold_ms:
            return ReplicaStatus.LAGGING
        
        # Check response time (degraded if > 100ms)
        if health.response_time_ms > 100:
            return ReplicaStatus.DEGRADED
        
        # Check connection utilization (degraded if > 80%)
        if health.connection_count / health.max_connections > 0.8:
            return ReplicaStatus.DEGRADED
        
        return ReplicaStatus.HEALTHY
    
    async def _update_load_balancing_weights(self):
        """Update load balancing weights based on replica health"""
        
        for replica_id, health in self.replica_health.items():
            if health.status == ReplicaStatus.FAILED:
                self.connection_weights[replica_id] = 0.0
            elif health.status == ReplicaStatus.LAGGING:
                self.connection_weights[replica_id] = 0.2
            elif health.status == ReplicaStatus.DEGRADED:
                self.connection_weights[replica_id] = 0.5
            else:  # HEALTHY
                # Weight based on response time (lower = better)
                if health.response_time_ms > 0:
                    weight = max(0.1, 1.0 - (health.response_time_ms / 1000))
                    self.connection_weights[replica_id] = weight
                else:
                    self.connection_weights[replica_id] = 1.0
    
    async def _check_failover_conditions(self):
        """Check if automatic failover should be triggered"""
        
        # Count healthy replicas
        healthy_replicas = [
            replica_id for replica_id, health in self.replica_health.items()
            if health.status in [ReplicaStatus.HEALTHY, ReplicaStatus.DEGRADED]
        ]
        
        failed_replicas = [
            replica_id for replica_id, health in self.replica_health.items()
            if health.status == ReplicaStatus.FAILED
        ]
        
        # Trigger failover if we have failed replicas and healthy alternatives
        if failed_replicas and healthy_replicas:
            for failed_replica in failed_replicas:
                if not self.replica_health[failed_replica].is_promoted:
                    await self._consider_replica_promotion(failed_replica, healthy_replicas)
    
    async def _consider_replica_promotion(self, failed_replica: str, healthy_replicas: List[str]):
        """Consider promoting a healthy replica to replace failed one"""
        
        # Select best replica for promotion
        best_replica = self._select_promotion_candidate(healthy_replicas)
        
        if best_replica:
            await self._execute_replica_promotion(failed_replica, best_replica, "automatic")
    
    def _select_promotion_candidate(self, candidates: List[str]) -> Optional[str]:
        """Select best replica for promotion based on health metrics"""
        
        if not candidates:
            return None
        
        # Score candidates based on multiple factors
        candidate_scores = {}
        
        for replica_id in candidates:
            health = self.replica_health[replica_id]
            
            # Scoring factors (lower is better for response time and lag)
            response_score = max(0, 100 - health.response_time_ms)  # 0-100
            lag_score = max(0, 100 - (health.replication_lag_ms / 100))  # 0-100
            connection_score = max(0, 100 - (health.connection_count / health.max_connections * 100))
            
            total_score = (response_score + lag_score + connection_score) / 3
            candidate_scores[replica_id] = total_score
        
        # Return replica with highest score
        return max(candidate_scores.items(), key=lambda x: x[1])[0]
    
    async def _execute_replica_promotion(self, failed_replica: str, promoted_replica: str, 
                                       failover_type: str):
        """Execute replica promotion/failover"""
        
        failover_event = FailoverEvent(
            event_id=f"failover_{int(datetime.now().timestamp())}",
            event_type=failover_type,
            old_primary=failed_replica,
            new_primary=promoted_replica,
            affected_replicas=[failed_replica, promoted_replica],
            start_time=datetime.now(timezone.utc),
            reason=f"Automatic failover from failed replica {failed_replica}"
        )
        
        try:
            logger.info(f"Starting failover: {failed_replica} -> {promoted_replica}")
            
            # Mark promoted replica
            self.replica_health[promoted_replica].is_promoted = True
            
            # Update topology
            if failed_replica in self.topology.read_replicas:
                self.topology.read_replicas.remove(failed_replica)
            
            # Call failover handlers
            for handler in self.failover_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(failover_event)
                    else:
                        handler(failover_event)
                except Exception as e:
                    logger.error(f"Error in failover handler: {e}")
            
            failover_event.end_time = datetime.now(timezone.utc)
            failover_event.duration_seconds = (
                failover_event.end_time - failover_event.start_time
            ).total_seconds()
            failover_event.success = True
            
            self.failover_history.append(failover_event)
            self.stats["failovers_executed"] += 1
            self.stats["replicas_promoted"] += 1
            
            logger.info(f"Failover completed successfully in {failover_event.duration_seconds:.2f}s")
            
        except Exception as e:
            logger.error(f"Failover failed: {e}")
            failover_event.success = False
            failover_event.end_time = datetime.now(timezone.utc)
            self.failover_history.append(failover_event)
    
    async def _store_health_metrics(self):
        """Store replica health metrics in Redis"""
        try:
            health_data = {}
            
            for replica_id, health in self.replica_health.items():
                health_data[replica_id] = {
                    "status": health.status.value,
                    "response_time_ms": health.response_time_ms,
                    "replication_lag_ms": health.replication_lag_ms,
                    "connection_count": health.connection_count,
                    "max_connections": health.max_connections,
                    "weight": self.connection_weights.get(replica_id, 0.0),
                    "last_check": health.last_check.isoformat()
                }
            
            await self.redis.setex(
                "replica_health",
                300,  # 5 minute TTL
                json.dumps(health_data)
            )
            
            # Store topology
            topology_data = {
                "primary_node": self.topology.primary_node,
                "read_replicas": self.topology.read_replicas,
                "sync_replicas": self.topology.sync_replicas,
                "async_replicas": self.topology.async_replicas,
                "updated_at": self.topology.updated_at.isoformat()
            }
            
            await self.redis.setex(
                "replication_topology",
                300,
                json.dumps(topology_data)
            )
            
        except Exception as e:
            logger.error(f"Error storing health metrics: {e}")
    
    def select_read_replica(self, query_type: QueryType = QueryType.SELECT) -> Optional[str]:
        """Select optimal read replica based on load balancing strategy"""
        
        # Get available healthy replicas
        available_replicas = [
            replica_id for replica_id, health in self.replica_health.items()
            if health.status in [ReplicaStatus.HEALTHY, ReplicaStatus.DEGRADED]
            and self.connection_weights.get(replica_id, 0) > 0
        ]
        
        if not available_replicas:
            logger.warning("No healthy read replicas available")
            return None
        
        selected_replica = None
        
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected_replica = self._round_robin_selection(available_replicas)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            selected_replica = self._weighted_selection(available_replicas)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected_replica = self._least_connections_selection(available_replicas)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            import random
            selected_replica = random.choice(available_replicas)
        
        else:
            # Default to weighted selection
            selected_replica = self._weighted_selection(available_replicas)
        
        if selected_replica:
            self.request_counts[selected_replica] += 1
            self.stats["load_balanced_queries"] += 1
        
        return selected_replica
    
    def _round_robin_selection(self, replicas: List[str]) -> str:
        """Round-robin replica selection"""
        if not replicas:
            return replicas[0]
        
        selected = replicas[self.round_robin_index % len(replicas)]
        self.round_robin_index += 1
        return selected
    
    def _weighted_selection(self, replicas: List[str]) -> str:
        """Weighted replica selection based on performance"""
        if len(replicas) == 1:
            return replicas[0]
        
        # Calculate total weight
        total_weight = sum(self.connection_weights.get(replica, 0.1) for replica in replicas)
        
        if total_weight <= 0:
            return replicas[0]
        
        # Random selection based on weights
        import random
        selection_point = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for replica in replicas:
            cumulative_weight += self.connection_weights.get(replica, 0.1)
            if cumulative_weight >= selection_point:
                return replica
        
        return replicas[0]  # Fallback
    
    def _least_connections_selection(self, replicas: List[str]) -> str:
        """Select replica with least connections"""
        if not replicas:
            return replicas[0]
        
        return min(replicas, key=lambda r: self.replica_health[r].connection_count)
    
    def add_failover_handler(self, handler: Callable[[FailoverEvent], Any]):
        """Add failover event handler"""
        self.failover_handlers.append(handler)
    
    async def manual_failover(self, from_replica: str, to_replica: str) -> bool:
        """Execute manual failover between replicas"""
        try:
            if from_replica not in self.replica_health or to_replica not in self.replica_health:
                logger.error("Invalid replica specified for manual failover")
                return False
            
            if self.replica_health[to_replica].status == ReplicaStatus.FAILED:
                logger.error(f"Cannot failover to failed replica: {to_replica}")
                return False
            
            await self._execute_replica_promotion(from_replica, to_replica, "manual")
            return True
            
        except Exception as e:
            logger.error(f"Manual failover failed: {e}")
            return False
    
    async def get_replica_status_report(self) -> Dict[str, Any]:
        """Get comprehensive replica status report"""
        
        status_counts = defaultdict(int)
        for health in self.replica_health.values():
            status_counts[health.status.value] += 1
        
        # Calculate average metrics
        avg_response_time = statistics.mean([
            h.response_time_ms for h in self.replica_health.values() 
            if h.response_time_ms > 0
        ]) if self.replica_health else 0
        
        avg_lag = statistics.mean([
            h.replication_lag_ms for h in self.replica_health.values()
            if h.replication_lag_ms > 0
        ]) if self.replica_health else 0
        
        return {
            "summary": {
                "total_replicas": len(self.replica_health),
                "healthy_replicas": status_counts["healthy"],
                "degraded_replicas": status_counts["degraded"],
                "lagging_replicas": status_counts["lagging"],
                "failed_replicas": status_counts["failed"],
                "avg_response_time_ms": avg_response_time,
                "avg_replication_lag_ms": avg_lag
            },
            "replicas": {
                replica_id: {
                    "status": health.status.value,
                    "response_time_ms": health.response_time_ms,
                    "replication_lag_ms": health.replication_lag_ms,
                    "connection_utilization": health.connection_count / health.max_connections,
                    "weight": self.connection_weights.get(replica_id, 0.0),
                    "request_count": self.request_counts.get(replica_id, 0),
                    "last_check": health.last_check.isoformat()
                }
                for replica_id, health in self.replica_health.items()
            },
            "topology": {
                "read_replicas": self.topology.read_replicas,
                "sync_replicas": self.topology.sync_replicas,
                "async_replicas": self.topology.async_replicas
            },
            "failover_history": [
                {
                    "event_id": event.event_id,
                    "type": event.event_type,
                    "from": event.old_primary,
                    "to": event.new_primary,
                    "duration_seconds": event.duration_seconds,
                    "success": event.success,
                    "reason": event.reason,
                    "timestamp": event.start_time.isoformat()
                }
                for event in self.failover_history[-10:]  # Last 10 failovers
            ],
            "statistics": self.stats
        }


# Global replica manager instance
replica_manager: Optional[ReplicaManager] = None


async def initialize_replica_manager(redis_client: aioredis.Redis,
                                   replica_configs: List[DatabaseConfig],
                                   load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME):
    """Initialize the global replica manager"""
    global replica_manager
    
    replica_manager = ReplicaManager(redis_client, replica_configs, load_balancing_strategy)
    await replica_manager.initialize()
    
    logger.info("✅ Replica manager initialized")


def get_replica_manager() -> ReplicaManager:
    """Get the global replica manager instance"""
    if replica_manager is None:
        raise RuntimeError("Replica manager not initialized.")
    return replica_manager


async def shutdown_replica_manager():
    """Shutdown the replica manager"""
    if replica_manager:
        await replica_manager.stop_health_monitoring()


# Convenience function for query routing
def get_optimal_read_replica(query_type: QueryType = QueryType.SELECT) -> Optional[str]:
    """Get optimal read replica for query execution"""
    if replica_manager:
        return replica_manager.select_read_replica(query_type)
    return None