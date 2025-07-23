"""
PRSM Distributed Task Worker System
High-performance distributed workers with intelligent load balancing, failure handling, and monitoring
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import uuid
import time
import logging
import signal
import psutil
import traceback
from collections import defaultdict, deque
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

from .task_queue import TaskQueue, TaskDefinition, TaskResult, TaskStatus, WorkerInfo, WorkerStatus, TaskRegistry

logger = logging.getLogger(__name__)


class WorkerLoadBalancingStrategy(Enum):
    """Worker load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_WORKER = "fastest_worker"
    QUEUE_PRIORITY = "queue_priority"


@dataclass
class WorkerMetrics:
    """Worker performance metrics"""
    worker_id: str
    cpu_usage_percent: float
    memory_usage_mb: float
    active_tasks: int
    tasks_per_minute: float
    avg_task_duration_ms: float
    error_rate_percent: float
    queue_processing_rates: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WorkerHealth:
    """Worker health status"""
    worker_id: str
    is_healthy: bool
    last_heartbeat: datetime
    consecutive_failures: int
    resource_usage: Dict[str, float]
    processing_capacity: float  # 0.0 to 1.0
    error_details: Optional[str] = None


class TaskWorker:
    """Advanced distributed task worker with intelligent task processing"""
    
    def __init__(self, 
                 worker_id: Optional[str] = None,
                 redis_client: Optional[aioredis.Redis] = None,
                 queue_names: Optional[List[str]] = None,
                 max_concurrent_tasks: int = 4,
                 heartbeat_interval: int = 30,
                 task_timeout: int = 300,
                 shutdown_timeout: int = 60):
        
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.redis = redis_client
        self.queue_names = queue_names or ["default"]
        self.max_concurrent_tasks = max_concurrent_tasks
        self.heartbeat_interval = heartbeat_interval
        self.task_timeout = task_timeout
        self.shutdown_timeout = shutdown_timeout
        
        # Worker state
        self.status = WorkerStatus.STOPPED
        self.current_tasks: Dict[str, TaskDefinition] = {}
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Task queues
        self.task_queues: Dict[str, TaskQueue] = {}
        self.task_registry = TaskRegistry()
        
        # Worker info and metrics
        self.worker_info = WorkerInfo(
            worker_id=self.worker_id,
            worker_name=f"PRSM-Worker-{self.worker_id}",
            status=WorkerStatus.STOPPED,
            queue_names=self.queue_names
        )
        
        self.metrics = WorkerMetrics(
            worker_id=self.worker_id,
            cpu_usage_percent=0.0,
            memory_usage_mb=0.0,
            active_tasks=0,
            tasks_per_minute=0.0,
            avg_task_duration_ms=0.0,
            error_rate_percent=0.0
        )
        
        # Background tasks
        self.worker_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.task_execution_times: deque = deque(maxlen=1000)
        self.task_completion_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.total_tasks_processed = 0
        
        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self.graceful_shutdown = True
        
        # Register signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except Exception as e:
            logger.warning(f"Could not setup signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Worker {self.worker_id} received signal {signum}")
        self.graceful_shutdown = True
        asyncio.create_task(self.stop())
    
    async def initialize(self, redis_client: aioredis.Redis, 
                        task_queues: Dict[str, TaskQueue]):
        """Initialize worker with Redis client and task queues"""
        self.redis = redis_client
        self.task_queues = task_queues
        
        # Register worker
        await self._register_worker()
        
        logger.info(f"✅ Worker {self.worker_id} initialized for queues: {self.queue_names}")
    
    async def start(self):
        """Start the worker"""
        if self.status != WorkerStatus.STOPPED:
            logger.warning(f"Worker {self.worker_id} is already running")
            return
        
        self.status = WorkerStatus.IDLE
        self.worker_info.status = WorkerStatus.IDLE
        self.worker_info.started_at = datetime.now(timezone.utc)
        
        # Start background tasks
        self.worker_task = asyncio.create_task(self._worker_loop())
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.metrics_task = asyncio.create_task(self._metrics_loop())
        
        logger.info(f"🚀 Worker {self.worker_id} started")
    
    async def stop(self):
        """Stop the worker gracefully"""
        if self.status == WorkerStatus.STOPPED:
            return
        
        logger.info(f"🛑 Stopping worker {self.worker_id}...")
        self.status = WorkerStatus.STOPPING
        self.worker_info.status = WorkerStatus.STOPPING
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for current tasks to complete or timeout
        if self.current_tasks:
            logger.info(f"Waiting for {len(self.current_tasks)} tasks to complete...")
            await asyncio.sleep(min(self.shutdown_timeout, 30))
        
        # Cancel background tasks
        for task in [self.worker_task, self.heartbeat_task, self.metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cancel any remaining current tasks if not graceful
        if not self.graceful_shutdown:
            for task_id in list(self.current_tasks.keys()):
                await self._cancel_current_task(task_id)
        
        # Unregister worker
        await self._unregister_worker()
        
        self.status = WorkerStatus.STOPPED
        self.worker_info.status = WorkerStatus.STOPPED
        
        logger.info(f"✅ Worker {self.worker_id} stopped")
    
    async def _worker_loop(self):
        """Main worker processing loop"""
        logger.info(f"Worker {self.worker_id} processing loop started")
        
        while not self.shutdown_event.is_set():
            try:
                # Process scheduled tasks for all queues
                for queue_name in self.queue_names:
                    if queue_name in self.task_queues:
                        await self.task_queues[queue_name].process_scheduled_tasks()
                
                # Try to get a task from queues (priority order)
                task_def = await self._get_next_task()
                
                if task_def:
                    # Process task concurrently
                    task_coroutine = self._process_task(task_def)
                    asyncio.create_task(task_coroutine)
                else:
                    # No tasks available, short sleep
                    await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(5)
        
        logger.info(f"Worker {self.worker_id} processing loop stopped")
    
    async def _get_next_task(self) -> Optional[TaskDefinition]:
        """Get next task from queues based on priority"""
        
        # Check if we can take more tasks
        if len(self.current_tasks) >= self.max_concurrent_tasks:
            return None
        
        # Try each queue in priority order
        for queue_name in self.queue_names:
            if queue_name not in self.task_queues:
                continue
            
            task_queue = self.task_queues[queue_name]
            
            try:
                # Try to dequeue with short timeout
                task_def = await task_queue.dequeue(timeout=1)
                if task_def:
                    return task_def
            except Exception as e:
                logger.warning(f"Error dequeuing from {queue_name}: {e}")
                continue
        
        return None
    
    async def _process_task(self, task_def: TaskDefinition):
        """Process a single task"""
        
        async with self.task_semaphore:
            # Add to current tasks
            self.current_tasks[task_def.task_id] = task_def
            self.status = WorkerStatus.BUSY
            self.worker_info.status = WorkerStatus.BUSY
            self.worker_info.current_task = task_def.task_id
            
            start_time = time.time()
            task_result = TaskResult(
                task_id=task_def.task_id,
                status=TaskStatus.RUNNING,
                started_at=datetime.now(timezone.utc),
                worker_id=self.worker_id
            )
            
            try:
                logger.debug(f"Processing task {task_def.task_id}: {task_def.func_name}")
                
                # Get task function from registry
                task_info = self.task_registry.get_task(task_def.func_name)
                if not task_info:
                    raise RuntimeError(f"Task function '{task_def.func_name}' not registered")
                
                task_func = task_info["func"]
                is_async = task_info["is_async"]
                
                # Execute task with timeout
                try:
                    if is_async:
                        result = await asyncio.wait_for(
                            task_func(*task_def.args, **task_def.kwargs),
                            timeout=task_def.timeout
                        )
                    else:
                        # Run sync function in thread pool
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: task_func(*task_def.args, **task_def.kwargs)
                        )
                    
                    # Task completed successfully
                    task_result.status = TaskStatus.SUCCESS
                    task_result.result = result
                    
                except asyncio.TimeoutError:
                    raise RuntimeError(f"Task timed out after {task_def.timeout} seconds")
                
            except Exception as e:
                # Task failed
                task_result.status = TaskStatus.FAILURE
                task_result.error = str(e)
                task_result.traceback = traceback.format_exc()
                
                logger.error(f"Task {task_def.task_id} failed: {e}")
                self.error_count += 1
                
                # Try to retry the task
                queue = self.task_queues.get(task_def.metadata.get("queue", "default"))
                if queue and task_def.max_retries > 0:
                    retry_success = await queue.retry_task(task_def, str(e))
                    if retry_success:
                        task_result.status = TaskStatus.RETRY
                        logger.info(f"Task {task_def.task_id} scheduled for retry")
            
            finally:
                # Complete task processing
                execution_time = (time.time() - start_time) * 1000
                task_result.completed_at = datetime.now(timezone.utc)
                task_result.execution_time_ms = execution_time
                
                # Store result in appropriate queue
                for queue_name, queue in self.task_queues.items():
                    if task_def.metadata.get("queue") == queue_name:
                        await queue.complete_task(task_def.task_id, task_result)
                        break
                
                # Update metrics
                self.task_execution_times.append(execution_time)
                self.task_completion_times.append(time.time())
                self.total_tasks_processed += 1
                self.worker_info.tasks_processed += 1
                
                if task_result.status == TaskStatus.FAILURE:
                    self.worker_info.tasks_failed += 1
                
                # Remove from current tasks
                self.current_tasks.pop(task_def.task_id, None)
                
                # Update worker status
                if not self.current_tasks:
                    self.status = WorkerStatus.IDLE
                    self.worker_info.status = WorkerStatus.IDLE
                    self.worker_info.current_task = None
                
                logger.debug(f"Completed task {task_def.task_id} in {execution_time:.2f}ms")
    
    async def _cancel_current_task(self, task_id: str):
        """Cancel a currently running task"""
        if task_id in self.current_tasks:
            # Mark task as cancelled
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED,
                completed_at=datetime.now(timezone.utc),
                worker_id=self.worker_id,
                error="Task cancelled due to worker shutdown"
            )
            
            # Store result
            for queue in self.task_queues.values():
                await queue.complete_task(task_id, task_result)
            
            self.current_tasks.pop(task_id, None)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while not self.shutdown_event.is_set():
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self):
        """Send heartbeat to Redis"""
        if not self.redis:
            return
        
        self.worker_info.last_heartbeat = datetime.now(timezone.utc)
        
        heartbeat_data = {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "current_task": self.worker_info.current_task,
            "tasks_processed": self.worker_info.tasks_processed,
            "tasks_failed": self.worker_info.tasks_failed,
            "started_at": self.worker_info.started_at.isoformat(),
            "last_heartbeat": self.worker_info.last_heartbeat.isoformat(),
            "queue_names": self.queue_names,
            "active_tasks": len(self.current_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks
        }
        
        # Store heartbeat with expiration
        await self.redis.setex(
            f"worker:{self.worker_id}:heartbeat",
            self.heartbeat_interval * 3,  # 3x interval for grace period
            json.dumps(heartbeat_data)
        )
    
    async def _metrics_loop(self):
        """Collect and update worker metrics"""
        while not self.shutdown_event.is_set():
            try:
                await self._update_metrics()
                await asyncio.sleep(60)  # Update metrics every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_metrics(self):
        """Update worker performance metrics"""
        try:
            # System metrics
            process = psutil.Process()
            self.metrics.cpu_usage_percent = process.cpu_percent()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            
            # Task metrics
            self.metrics.active_tasks = len(self.current_tasks)
            
            # Calculate tasks per minute
            now = time.time()
            minute_ago = now - 60
            recent_completions = [t for t in self.task_completion_times if t > minute_ago]
            self.metrics.tasks_per_minute = len(recent_completions)
            
            # Calculate average task duration
            if self.task_execution_times:
                self.metrics.avg_task_duration_ms = sum(self.task_execution_times) / len(self.task_execution_times)
            
            # Calculate error rate
            if self.total_tasks_processed > 0:
                self.metrics.error_rate_percent = (self.error_count / self.total_tasks_processed) * 100
            
            self.metrics.last_updated = datetime.now(timezone.utc)
            
            # Store metrics in Redis
            if self.redis:
                metrics_data = {
                    "worker_id": self.metrics.worker_id,
                    "cpu_usage_percent": self.metrics.cpu_usage_percent,
                    "memory_usage_mb": self.metrics.memory_usage_mb,
                    "active_tasks": self.metrics.active_tasks,
                    "tasks_per_minute": self.metrics.tasks_per_minute,
                    "avg_task_duration_ms": self.metrics.avg_task_duration_ms,
                    "error_rate_percent": self.metrics.error_rate_percent,
                    "last_updated": self.metrics.last_updated.isoformat()
                }
                
                await self.redis.setex(
                    f"worker:{self.worker_id}:metrics",
                    300,  # 5 minute TTL
                    json.dumps(metrics_data)
                )
        
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _register_worker(self):
        """Register worker in Redis"""
        if not self.redis:
            return
        
        worker_data = {
            "worker_id": self.worker_id,
            "worker_name": self.worker_info.worker_name,
            "status": self.status.value,
            "queue_names": self.queue_names,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        
        await self.redis.hset("workers:registry", self.worker_id, json.dumps(worker_data))
        logger.debug(f"Registered worker {self.worker_id}")
    
    async def _unregister_worker(self):
        """Unregister worker from Redis"""
        if not self.redis:
            return
        
        await self.redis.hdel("workers:registry", self.worker_id)
        await self.redis.delete(f"worker:{self.worker_id}:heartbeat")
        await self.redis.delete(f"worker:{self.worker_id}:metrics")
        
        logger.debug(f"Unregistered worker {self.worker_id}")
    
    def register_task(self, func: Callable, name: Optional[str] = None, **kwargs):
        """Register a task function"""
        task_name = name or f"{func.__module__}.{func.__name__}"
        return self.task_registry.register(task_name, func, **kwargs)
    
    async def get_worker_health(self) -> WorkerHealth:
        """Get current worker health status"""
        
        # Determine if worker is healthy
        is_healthy = (
            self.status in [WorkerStatus.IDLE, WorkerStatus.BUSY] and
            self.metrics.error_rate_percent < 50 and  # Less than 50% error rate
            self.metrics.cpu_usage_percent < 90 and   # Less than 90% CPU
            self.metrics.memory_usage_mb < 2048       # Less than 2GB memory
        )
        
        # Calculate processing capacity (0.0 to 1.0)
        capacity = max(0.0, 1.0 - (len(self.current_tasks) / self.max_concurrent_tasks))
        
        return WorkerHealth(
            worker_id=self.worker_id,
            is_healthy=is_healthy,
            last_heartbeat=self.worker_info.last_heartbeat,
            consecutive_failures=0,  # Would track this separately
            resource_usage={
                "cpu_percent": self.metrics.cpu_usage_percent,
                "memory_mb": self.metrics.memory_usage_mb,
                "active_tasks": len(self.current_tasks)
            },
            processing_capacity=capacity
        )
    
    async def get_worker_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker statistics"""
        
        health = await self.get_worker_health()
        
        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "health": {
                "is_healthy": health.is_healthy,
                "processing_capacity": health.processing_capacity,
                "resource_usage": health.resource_usage
            },
            "performance": {
                "total_processed": self.total_tasks_processed,
                "total_failed": self.error_count,
                "success_rate": (
                    ((self.total_tasks_processed - self.error_count) / max(1, self.total_tasks_processed)) * 100
                ),
                "avg_duration_ms": self.metrics.avg_task_duration_ms,
                "tasks_per_minute": self.metrics.tasks_per_minute,
                "error_rate_percent": self.metrics.error_rate_percent
            },
            "current_state": {
                "active_tasks": len(self.current_tasks),
                "max_concurrent": self.max_concurrent_tasks,
                "current_task": self.worker_info.current_task,
                "queue_names": self.queue_names
            },
            "uptime": {
                "started_at": self.worker_info.started_at.isoformat(),
                "last_heartbeat": self.worker_info.last_heartbeat.isoformat(),
                "uptime_seconds": (
                    datetime.now(timezone.utc) - self.worker_info.started_at
                ).total_seconds()
            }
        }


class WorkerManager:
    """Manager for multiple distributed workers"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.workers: Dict[str, TaskWorker] = {}
        self.task_queues: Dict[str, TaskQueue] = {}
        self.load_balancing_strategy = WorkerLoadBalancingStrategy.LEAST_LOADED
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitor_interval = 60  # seconds
    
    async def create_worker(self, 
                          worker_id: Optional[str] = None,
                          queue_names: List[str] = None,
                          max_concurrent_tasks: int = 4) -> TaskWorker:
        """Create and initialize a new worker"""
        
        worker = TaskWorker(
            worker_id=worker_id,
            redis_client=self.redis,
            queue_names=queue_names or ["default"],
            max_concurrent_tasks=max_concurrent_tasks
        )
        
        await worker.initialize(self.redis, self.task_queues)
        self.workers[worker.worker_id] = worker
        
        logger.info(f"Created worker {worker.worker_id}")
        return worker
    
    async def start_worker(self, worker_id: str):
        """Start a specific worker"""
        if worker_id in self.workers:
            await self.workers[worker_id].start()
    
    async def stop_worker(self, worker_id: str):
        """Stop a specific worker"""
        if worker_id in self.workers:
            await self.workers[worker_id].stop()
            del self.workers[worker_id]
    
    async def start_all_workers(self):
        """Start all workers"""
        start_tasks = []
        for worker in self.workers.values():
            start_tasks.append(worker.start())
        
        if start_tasks:
            await asyncio.gather(*start_tasks)
        
        # Start monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"✅ Started {len(self.workers)} workers")
    
    async def stop_all_workers(self):
        """Stop all workers"""
        
        # Stop monitoring
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop all workers
        stop_tasks = []
        for worker in self.workers.values():
            stop_tasks.append(worker.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.workers.clear()
        logger.info("✅ Stopped all workers")
    
    async def _monitoring_loop(self):
        """Monitor worker health and performance"""
        while True:
            try:
                await self._check_worker_health()
                await self._collect_cluster_metrics()
                await asyncio.sleep(self.monitor_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker monitoring: {e}")
                await asyncio.sleep(self.monitor_interval)
    
    async def _check_worker_health(self):
        """Check health of all workers"""
        unhealthy_workers = []
        
        for worker_id, worker in self.workers.items():
            try:
                health = await worker.get_worker_health()
                if not health.is_healthy:
                    unhealthy_workers.append(worker_id)
                    logger.warning(f"Worker {worker_id} is unhealthy")
            except Exception as e:
                logger.error(f"Error checking health of worker {worker_id}: {e}")
                unhealthy_workers.append(worker_id)
        
        # Could implement automatic worker restart here
        if unhealthy_workers:
            logger.warning(f"Found {len(unhealthy_workers)} unhealthy workers")
    
    async def _collect_cluster_metrics(self):
        """Collect cluster-wide metrics"""
        cluster_stats = {
            "total_workers": len(self.workers),
            "active_workers": 0,
            "total_active_tasks": 0,
            "total_processed": 0,
            "total_failed": 0,
            "avg_cpu_usage": 0.0,
            "avg_memory_usage": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        cpu_usage_sum = 0.0
        memory_usage_sum = 0.0
        
        for worker in self.workers.values():
            if worker.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]:
                cluster_stats["active_workers"] += 1
            
            cluster_stats["total_active_tasks"] += len(worker.current_tasks)
            cluster_stats["total_processed"] += worker.total_tasks_processed
            cluster_stats["total_failed"] += worker.error_count
            
            cpu_usage_sum += worker.metrics.cpu_usage_percent
            memory_usage_sum += worker.metrics.memory_usage_mb
        
        if self.workers:
            cluster_stats["avg_cpu_usage"] = cpu_usage_sum / len(self.workers)
            cluster_stats["avg_memory_usage"] = memory_usage_sum / len(self.workers)
        
        # Store cluster metrics
        await self.redis.setex(
            "workers:cluster_metrics",
            300,  # 5 minute TTL
            json.dumps(cluster_stats)
        )
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        
        worker_stats = {}
        for worker_id, worker in self.workers.items():
            worker_stats[worker_id] = await worker.get_worker_stats()
        
        # Get cluster metrics from Redis
        cluster_metrics_data = await self.redis.get("workers:cluster_metrics")
        cluster_metrics = {}
        if cluster_metrics_data:
            cluster_metrics = json.loads(cluster_metrics_data)
        
        return {
            "cluster_metrics": cluster_metrics,
            "workers": worker_stats,
            "queue_stats": {
                name: await queue.get_queue_stats()
                for name, queue in self.task_queues.items()
            }
        }


# Global worker manager
worker_manager: Optional[WorkerManager] = None


async def initialize_worker_system(redis_client: aioredis.Redis,
                                 task_queues: Dict[str, TaskQueue],
                                 worker_configs: Optional[List[Dict[str, Any]]] = None):
    """Initialize the distributed worker system"""
    global worker_manager
    
    worker_manager = WorkerManager(redis_client)
    worker_manager.task_queues = task_queues
    
    # Create workers based on configuration
    worker_configs = worker_configs or [{"queue_names": ["default"], "max_concurrent_tasks": 4}]
    
    for config in worker_configs:
        await worker_manager.create_worker(
            queue_names=config.get("queue_names", ["default"]),
            max_concurrent_tasks=config.get("max_concurrent_tasks", 4)
        )
    
    logger.info(f"✅ Worker system initialized with {len(worker_manager.workers)} workers")


def get_worker_manager() -> WorkerManager:
    """Get the global worker manager"""
    if worker_manager is None:
        raise RuntimeError("Worker system not initialized")
    return worker_manager


async def start_worker_cluster():
    """Start all workers in the cluster"""
    if worker_manager:
        await worker_manager.start_all_workers()


async def stop_worker_cluster():
    """Stop all workers in the cluster"""
    if worker_manager:
        await worker_manager.stop_all_workers()


@asynccontextmanager
async def worker_cluster_context(redis_client: aioredis.Redis,
                                task_queues: Dict[str, TaskQueue],
                                worker_configs: Optional[List[Dict[str, Any]]] = None):
    """Context manager for worker cluster lifecycle"""
    try:
        await initialize_worker_system(redis_client, task_queues, worker_configs)
        await start_worker_cluster()
        yield get_worker_manager()
    finally:
        await stop_worker_cluster()