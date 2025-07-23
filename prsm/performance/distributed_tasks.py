"""
PRSM Distributed Task Processing Integration
Complete distributed task processing system with queues, workers, and monitoring
"""

from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import logging
from contextlib import asynccontextmanager
import redis.asyncio as aioredis

from .task_queue import (
    TaskQueue, TaskDefinition, TaskResult, TaskStatus, TaskPriority, TaskRegistry,
    initialize_task_queues, get_task_queue, enqueue_task, get_all_queue_stats, task
)
from .task_worker import (
    TaskWorker, WorkerManager, WorkerStatus, 
    initialize_worker_system, get_worker_manager, start_worker_cluster, stop_worker_cluster
)
from .task_monitor import (
    TaskMonitor, AlertType, AlertSeverity, TaskAlert,
    initialize_task_monitoring, get_task_monitor, start_task_monitoring, stop_task_monitoring
)

logger = logging.getLogger(__name__)


@dataclass
class DistributedTaskConfig:
    """Configuration for distributed task processing system"""
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_cluster_nodes: Optional[List[Dict[str, Any]]] = None
    
    # Queue Configuration
    queue_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "default": {"max_queue_size": 10000},
        "high_priority": {"max_queue_size": 5000},
        "low_priority": {"max_queue_size": 20000},
        "background": {"max_queue_size": 50000}
    })
    
    # Worker Configuration
    worker_configs: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "queue_names": ["high_priority", "default"],
            "max_concurrent_tasks": 4
        },
        {
            "queue_names": ["default", "low_priority"],
            "max_concurrent_tasks": 2
        },
        {
            "queue_names": ["background"],
            "max_concurrent_tasks": 1
        }
    ])
    
    # Monitoring Configuration
    monitoring_enabled: bool = True
    alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None
    monitoring_interval: int = 30
    
    # Feature Flags
    enable_task_retries: bool = True
    enable_task_scheduling: bool = True
    enable_priority_queues: bool = True
    enable_worker_scaling: bool = False  # Future feature


class DistributedTaskOrchestrator:
    """Central orchestrator for distributed task processing"""
    
    def __init__(self, config: DistributedTaskConfig):
        self.config = config
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Component tracking
        self.task_queues: Dict[str, TaskQueue] = {}
        self.worker_manager: Optional[WorkerManager] = None
        self.task_monitor: Optional[TaskMonitor] = None
        self.task_registry = TaskRegistry()
        
        # System state
        self.system_initialized = False
        self.system_running = False
        self.startup_time: Optional[datetime] = None
        
        # Background tasks
        self.scheduler_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the complete distributed task system"""
        
        if self.system_initialized:
            logger.warning("Distributed task system is already initialized")
            return
        
        self.startup_time = datetime.now(timezone.utc)
        logger.info("🚀 Initializing PRSM Distributed Task Processing System...")
        
        try:
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Initialize task queues
            await self._initialize_task_queues()
            
            # Initialize worker system
            await self._initialize_workers()
            
            # Initialize monitoring
            if self.config.monitoring_enabled:
                await self._initialize_monitoring()
            
            self.system_initialized = True
            logger.info("✅ Distributed task processing system initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize distributed task system: {e}")
            await self._cleanup_on_failure()
            raise
    
    async def start(self):
        """Start the distributed task processing system"""
        
        if not self.system_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        if self.system_running:
            logger.warning("Distributed task system is already running")
            return
        
        try:
            # Start worker cluster
            await start_worker_cluster()
            
            # Start monitoring
            if self.config.monitoring_enabled and self.task_monitor:
                await start_task_monitoring(self.task_queues, self.worker_manager)
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.system_running = True
            logger.info("🎉 Distributed task processing system started successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to start distributed task system: {e}")
            raise
    
    async def stop(self):
        """Stop the distributed task processing system"""
        
        if not self.system_running:
            return
        
        logger.info("🛑 Stopping distributed task processing system...")
        
        try:
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Stop monitoring
            if self.config.monitoring_enabled:
                await stop_task_monitoring()
            
            # Stop worker cluster
            await stop_worker_cluster()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            self.system_running = False
            logger.info("✅ Distributed task processing system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping distributed task system: {e}")
    
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
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            raise
    
    async def _initialize_task_queues(self):
        """Initialize task queues"""
        
        try:
            await initialize_task_queues(self.redis_client, self.config.queue_configs)
            
            # Get queue instances
            for queue_name in self.config.queue_configs.keys():
                self.task_queues[queue_name] = get_task_queue(queue_name)
            
            logger.info(f"✅ Task queues initialized: {list(self.task_queues.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Task queue initialization failed: {e}")
            raise
    
    async def _initialize_workers(self):
        """Initialize worker system"""
        
        try:
            await initialize_worker_system(
                self.redis_client,
                self.task_queues,
                self.config.worker_configs
            )
            
            self.worker_manager = get_worker_manager()
            logger.info(f"✅ Worker system initialized with {len(self.worker_manager.workers)} workers")
            
        except Exception as e:
            logger.error(f"❌ Worker system initialization failed: {e}")
            raise
    
    async def _initialize_monitoring(self):
        """Initialize monitoring system"""
        
        try:
            await initialize_task_monitoring(
                self.redis_client,
                self.config.alert_thresholds
            )
            
            self.task_monitor = get_task_monitor()
            logger.info("✅ Task monitoring system initialized")
            
        except Exception as e:
            logger.error(f"❌ Monitoring system initialization failed: {e}")
            raise
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        # Start scheduled task processor
        if self.config.enable_task_scheduling:
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("✅ Background tasks started")
    
    async def _stop_background_tasks(self):
        """Stop background tasks"""
        
        for task in [self.scheduler_task, self.cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("✅ Background tasks stopped")
    
    async def _scheduler_loop(self):
        """Background loop to process scheduled tasks"""
        
        while True:
            try:
                # Process scheduled tasks for all queues
                total_processed = 0
                for task_queue in self.task_queues.values():
                    processed = await task_queue.process_scheduled_tasks()
                    total_processed += processed
                
                if total_processed > 0:
                    logger.debug(f"Processed {total_processed} scheduled tasks")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self):
        """Background loop for system cleanup"""
        
        while True:
            try:
                # Clean up expired tasks and results
                await self._cleanup_expired_data()
                
                # Clean up orphaned processing tasks
                await self._cleanup_orphaned_tasks()
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_expired_data(self):
        """Clean up expired task data"""
        
        try:
            # Remove expired task results (older than 24 hours)
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            cutoff_timestamp = cutoff_time.timestamp()
            
            for queue_name in self.task_queues.keys():
                # Clean up old results
                results_key = f"queue:{queue_name}:results"
                failed_key = f"queue:{queue_name}:failed"
                
                # This would require more complex cleanup logic in production
                # For now, we rely on Redis TTL for automatic cleanup
                
            logger.debug("Completed expired data cleanup")
            
        except Exception as e:
            logger.error(f"Error in expired data cleanup: {e}")
    
    async def _cleanup_orphaned_tasks(self):
        """Clean up orphaned processing tasks"""
        
        try:
            # Check for tasks stuck in processing state
            for queue_name, task_queue in self.task_queues.items():
                processing_key = f"queue:{queue_name}:processing"
                
                # Get all processing tasks
                processing_tasks = await self.redis_client.lrange(processing_key, 0, -1)
                
                for task_data in processing_tasks:
                    try:
                        task_info = json.loads(task_data)
                        task_id = task_info["task_id"]
                        
                        # Check if task is still being processed by checking worker heartbeats
                        # If no worker has this task and it's been processing for too long,
                        # move it back to pending or mark as failed
                        
                        # This is a simplified implementation
                        # Production would need more sophisticated orphan detection
                        
                    except json.JSONDecodeError:
                        # Remove invalid task data
                        await self.redis_client.lrem(processing_key, 1, task_data)
            
            logger.debug("Completed orphaned task cleanup")
            
        except Exception as e:
            logger.error(f"Error in orphaned task cleanup: {e}")
    
    async def _cleanup_on_failure(self):
        """Cleanup on initialization failure"""
        
        logger.info("🧹 Cleaning up after initialization failure...")
        
        try:
            if self.redis_client:
                await self.redis_client.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        self.system_initialized = False
        self.system_running = False
    
    # Task Management API
    
    def register_task(self, name: Optional[str] = None, **kwargs):
        """Decorator to register a task function"""
        def decorator(func: Callable) -> Callable:
            task_name = name or f"{func.__module__}.{func.__name__}"
            
            # Register with global registry
            self.task_registry.register(task_name, func, **kwargs)
            
            # Also register with each worker's registry
            if self.worker_manager:
                for worker in self.worker_manager.workers.values():
                    worker.register_task(func, task_name, **kwargs)
            
            return func
        
        return decorator
    
    async def submit_task(self, 
                         func_name: str,
                         *args,
                         queue: str = "default",
                         priority: TaskPriority = TaskPriority.NORMAL,
                         delay_seconds: Optional[int] = None,
                         expires_in: Optional[int] = None,
                         **kwargs) -> str:
        """Submit a task for processing"""
        
        if not self.system_running:
            raise RuntimeError("Task system is not running")
        
        # Create task definition
        task_def = TaskDefinition(
            task_id=f"task_{int(datetime.now().timestamp() * 1000000)}",
            task_name=func_name,
            func_name=func_name,
            args=list(args),
            kwargs=kwargs,
            priority=priority,
            expires_at=(
                datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                if expires_in else None
            )
        )
        
        # Get appropriate queue
        if queue not in self.task_queues:
            raise ValueError(f"Queue '{queue}' not found")
        
        task_queue = self.task_queues[queue]
        
        # Schedule or enqueue task
        if delay_seconds and delay_seconds > 0:
            execute_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
            return await task_queue.schedule_task(task_def, execute_at)
        else:
            return await task_queue.enqueue(task_def)
    
    async def get_task_result(self, task_id: str, queue: str = "default") -> Optional[TaskResult]:
        """Get task execution result"""
        
        if queue not in self.task_queues:
            raise ValueError(f"Queue '{queue}' not found")
        
        return await self.task_queues[queue].get_task_result(task_id)
    
    async def cancel_task(self, task_id: str, queue: str = "default") -> bool:
        """Cancel a pending task"""
        
        if queue not in self.task_queues:
            raise ValueError(f"Queue '{queue}' not found")
        
        return await self.task_queues[queue].cancel_task(task_id)
    
    # System Monitoring API
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            "system_info": {
                "initialized": self.system_initialized,
                "running": self.system_running,
                "startup_time": self.startup_time.isoformat() if self.startup_time else None,
                "uptime_seconds": (
                    (datetime.now(timezone.utc) - self.startup_time).total_seconds()
                    if self.startup_time else 0
                )
            },
            "queues": {},
            "workers": {},
            "monitoring": {}
        }
        
        # Get queue statistics
        try:
            status["queues"] = await get_all_queue_stats()
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            status["queues"] = {"error": str(e)}
        
        # Get worker cluster status
        try:
            if self.worker_manager:
                cluster_status = await self.worker_manager.get_cluster_status()
                status["workers"] = cluster_status
        except Exception as e:
            logger.error(f"Error getting worker stats: {e}")
            status["workers"] = {"error": str(e)}
        
        # Get monitoring dashboard
        try:
            if self.task_monitor:
                monitoring_data = await self.task_monitor.get_monitoring_dashboard()
                status["monitoring"] = monitoring_data
        except Exception as e:
            logger.error(f"Error getting monitoring stats: {e}")
            status["monitoring"] = {"error": str(e)}
        
        return status
    
    async def get_queue_analytics(self, queue_name: str, 
                                time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get detailed queue analytics"""
        
        if not self.task_monitor:
            return {"error": "Monitoring not enabled"}
        
        return await self.task_monitor.get_queue_analytics(queue_name, time_range_minutes)
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active system alerts"""
        
        if not self.task_monitor:
            return []
        
        dashboard = await self.task_monitor.get_monitoring_dashboard()
        return dashboard.get("alerts", {}).get("active_alerts", [])
    
    def add_alert_handler(self, handler: Callable[[TaskAlert], Any]):
        """Add custom alert handler"""
        if self.task_monitor:
            self.task_monitor.add_alert_handler(handler)


# Global distributed task orchestrator
task_orchestrator: Optional[DistributedTaskOrchestrator] = None


async def initialize_distributed_tasks(config: DistributedTaskConfig):
    """Initialize the distributed task processing system"""
    global task_orchestrator
    
    task_orchestrator = DistributedTaskOrchestrator(config)
    await task_orchestrator.initialize()
    
    logger.info("🎉 PRSM Distributed Task Processing System initialized!")


def get_task_orchestrator() -> DistributedTaskOrchestrator:
    """Get the global task orchestrator instance"""
    if task_orchestrator is None:
        raise RuntimeError("Distributed task system not initialized")
    return task_orchestrator


async def start_distributed_tasks():
    """Start the distributed task processing system"""
    if task_orchestrator:
        await task_orchestrator.start()


async def stop_distributed_tasks():
    """Stop the distributed task processing system"""
    if task_orchestrator:
        await task_orchestrator.stop()


@asynccontextmanager
async def distributed_task_context(config: DistributedTaskConfig):
    """Context manager for distributed task system lifecycle"""
    try:
        await initialize_distributed_tasks(config)
        await start_distributed_tasks()
        yield get_task_orchestrator()
    finally:
        await stop_distributed_tasks()


# Convenience functions

async def submit_task(func_name: str, *args, **kwargs) -> str:
    """Submit a task for processing"""
    orchestrator = get_task_orchestrator()
    return await orchestrator.submit_task(func_name, *args, **kwargs)


async def get_task_result(task_id: str, queue: str = "default") -> Optional[TaskResult]:
    """Get task execution result"""
    orchestrator = get_task_orchestrator()
    return await orchestrator.get_task_result(task_id, queue)


def register_task(name: Optional[str] = None, **kwargs):
    """Decorator to register a task function"""
    orchestrator = get_task_orchestrator()
    return orchestrator.register_task(name, **kwargs)


# Example usage and configuration factory

def create_default_config(redis_url: str = "redis://localhost:6379") -> DistributedTaskConfig:
    """Create default distributed task configuration"""
    
    return DistributedTaskConfig(
        redis_url=redis_url,
        queue_configs={
            "default": {"max_queue_size": 10000},
            "high_priority": {"max_queue_size": 5000},
            "low_priority": {"max_queue_size": 20000},
            "background": {"max_queue_size": 50000},
            "ai_processing": {"max_queue_size": 1000},  # For AI/ML tasks
            "data_processing": {"max_queue_size": 5000}  # For data processing
        },
        worker_configs=[
            {
                "queue_names": ["high_priority", "default"],
                "max_concurrent_tasks": 4
            },
            {
                "queue_names": ["default", "low_priority"],
                "max_concurrent_tasks": 2
            },
            {
                "queue_names": ["background", "data_processing"],
                "max_concurrent_tasks": 1
            },
            {
                "queue_names": ["ai_processing"],
                "max_concurrent_tasks": 1
            }
        ],
        monitoring_enabled=True,
        alert_thresholds={
            "queue_backlog": {
                "max_pending_tasks": 1000,
                "max_oldest_task_age_minutes": 30
            },
            "failure_rate": {
                "max_failure_rate_percent": 10.0
            },
            "processing_time": {
                "max_avg_processing_time_ms": 5000
            },
            "worker_health": {
                "min_active_workers": 1,
                "max_cpu_usage_percent": 90,
                "max_memory_usage_mb": 2048
            }
        }
    )