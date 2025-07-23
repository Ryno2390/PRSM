"""
PRSM Advanced Distributed Task Queue System
High-performance distributed task processing with Redis backend, intelligent routing, and comprehensive monitoring
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
import pickle
import traceback
from collections import defaultdict, deque
import redis.asyncio as aioredis
from functools import wraps
import inspect

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    URGENT = 4


class WorkerStatus(Enum):
    """Worker status"""
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class TaskDefinition:
    """Task definition and metadata"""
    task_id: str
    task_name: str
    func_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: int = 300  # 5 minutes
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None
    worker_id: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerInfo:
    """Worker information and statistics"""
    worker_id: str
    worker_name: str
    status: WorkerStatus
    current_task: Optional[str] = None
    tasks_processed: int = 0
    tasks_failed: int = 0
    avg_execution_time_ms: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    queue_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskRegistry:
    """Registry for task functions and their configurations"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_configs: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, func: Callable, 
                priority: TaskPriority = TaskPriority.NORMAL,
                max_retries: int = 3,
                timeout: int = 300,
                queue: str = "default") -> Callable:
        """Register a task function"""
        
        task_config = {
            "func": func,
            "priority": priority,
            "max_retries": max_retries,
            "timeout": timeout,
            "queue": queue,
            "is_async": inspect.iscoroutinefunction(func)
        }
        
        self.tasks[name] = task_config
        self.task_configs[name] = {
            "priority": priority,
            "max_retries": max_retries,
            "timeout": timeout,
            "queue": queue
        }
        
        logger.info(f"Registered task '{name}' with queue '{queue}'")
        return func
    
    def get_task(self, name: str) -> Optional[Dict[str, Any]]:
        """Get task configuration"""
        return self.tasks.get(name)
    
    def list_tasks(self) -> Dict[str, Dict[str, Any]]:
        """List all registered tasks"""
        return self.task_configs.copy()


class TaskQueue:
    """Advanced distributed task queue with Redis backend"""
    
    def __init__(self, redis_client: aioredis.Redis, 
                 queue_name: str = "default",
                 max_queue_size: int = 10000):
        self.redis = redis_client
        self.queue_name = queue_name
        self.max_queue_size = max_queue_size
        
        # Redis keys
        self.pending_key = f"queue:{queue_name}:pending"
        self.processing_key = f"queue:{queue_name}:processing"
        self.results_key = f"queue:{queue_name}:results"
        self.failed_key = f"queue:{queue_name}:failed"
        self.scheduled_key = f"queue:{queue_name}:scheduled"
        
        # Task registry
        self.registry = TaskRegistry()
        
        # Queue statistics
        self.stats = {
            "tasks_enqueued": 0,
            "tasks_processed": 0,
            "tasks_failed": 0,
            "avg_processing_time_ms": 0.0,
            "current_queue_size": 0
        }
    
    async def enqueue(self, task_def: TaskDefinition) -> str:
        """Enqueue a task for processing"""
        
        # Check queue size limit
        current_size = await self.redis.llen(self.pending_key)
        if current_size >= self.max_queue_size:
            raise RuntimeError(f"Queue '{self.queue_name}' is full ({current_size}/{self.max_queue_size})")
        
        # Serialize task
        task_data = {
            "task_id": task_def.task_id,
            "task_name": task_def.task_name,
            "func_name": task_def.func_name,
            "args": task_def.args,
            "kwargs": task_def.kwargs,
            "priority": task_def.priority.value,
            "max_retries": task_def.max_retries,
            "retry_delay": task_def.retry_delay,
            "timeout": task_def.timeout,
            "expires_at": task_def.expires_at.isoformat() if task_def.expires_at else None,
            "created_at": task_def.created_at.isoformat(),
            "metadata": task_def.metadata
        }
        
        serialized_task = json.dumps(task_data)
        
        # Add to appropriate queue based on priority
        if task_def.priority == TaskPriority.URGENT:
            await self.redis.lpush(self.pending_key, serialized_task)
        elif task_def.priority == TaskPriority.CRITICAL:
            await self.redis.lpush(self.pending_key, serialized_task)
        elif task_def.priority == TaskPriority.HIGH:
            # Insert at 25% from front for high priority
            queue_size = await self.redis.llen(self.pending_key)
            insert_pos = max(0, queue_size // 4)
            if insert_pos == 0:
                await self.redis.lpush(self.pending_key, serialized_task)
            else:
                await self.redis.linsert(self.pending_key, "BEFORE", 
                                       await self.redis.lindex(self.pending_key, insert_pos),
                                       serialized_task)
        else:
            # Normal and low priority go to the end
            await self.redis.rpush(self.pending_key, serialized_task)
        
        # Store task metadata
        await self.redis.hset(f"task:{task_def.task_id}", mapping={
            "status": TaskStatus.PENDING.value,
            "queue": self.queue_name,
            "created_at": task_def.created_at.isoformat(),
            "task_data": serialized_task
        })
        
        # Set expiration if specified
        if task_def.expires_at:
            expire_seconds = int((task_def.expires_at - datetime.now(timezone.utc)).total_seconds())
            if expire_seconds > 0:
                await self.redis.expire(f"task:{task_def.task_id}", expire_seconds)
        
        self.stats["tasks_enqueued"] += 1
        logger.debug(f"Enqueued task {task_def.task_id} to queue '{self.queue_name}'")
        
        return task_def.task_id
    
    async def dequeue(self, timeout: int = 0) -> Optional[TaskDefinition]:
        """Dequeue a task for processing"""
        
        # Use blocking pop with timeout
        result = await self.redis.blpop(self.pending_key, timeout=timeout)
        
        if not result:
            return None
        
        queue_name, serialized_task = result
        task_data = json.loads(serialized_task)
        
        # Check if task has expired
        if task_data.get("expires_at"):
            expires_at = datetime.fromisoformat(task_data["expires_at"])
            if datetime.now(timezone.utc) > expires_at:
                # Task expired, mark as such
                await self._mark_task_expired(task_data["task_id"])
                return None
        
        # Move to processing queue
        await self.redis.lpush(self.processing_key, serialized_task)
        
        # Create TaskDefinition object
        task_def = TaskDefinition(
            task_id=task_data["task_id"],
            task_name=task_data["task_name"],
            func_name=task_data["func_name"],
            args=task_data["args"],
            kwargs=task_data["kwargs"],
            priority=TaskPriority(task_data["priority"]),
            max_retries=task_data["max_retries"],
            retry_delay=task_data["retry_delay"],
            timeout=task_data["timeout"],
            expires_at=datetime.fromisoformat(task_data["expires_at"]) if task_data.get("expires_at") else None,
            created_at=datetime.fromisoformat(task_data["created_at"]),
            metadata=task_data["metadata"]
        )
        
        # Update task status
        await self.redis.hset(f"task:{task_def.task_id}", "status", TaskStatus.RUNNING.value)
        
        return task_def
    
    async def complete_task(self, task_id: str, result: TaskResult) -> bool:
        """Mark task as completed and store result"""
        
        # Remove from processing queue
        await self._remove_from_processing(task_id)
        
        # Store result
        result_data = {
            "task_id": result.task_id,
            "status": result.status.value,
            "result": pickle.dumps(result.result) if result.result is not None else None,
            "error": result.error,
            "traceback": result.traceback,
            "started_at": result.started_at.isoformat() if result.started_at else None,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "execution_time_ms": result.execution_time_ms,
            "worker_id": result.worker_id,
            "retry_count": result.retry_count,
            "metadata": json.dumps(result.metadata)
        }
        
        # Store in results or failed queue
        if result.status == TaskStatus.SUCCESS:
            await self.redis.hset(self.results_key, task_id, json.dumps(result_data))
            self.stats["tasks_processed"] += 1
        else:
            await self.redis.hset(self.failed_key, task_id, json.dumps(result_data))
            self.stats["tasks_failed"] += 1
        
        # Update task metadata
        await self.redis.hset(f"task:{task_id}", mapping={
            "status": result.status.value,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "execution_time_ms": result.execution_time_ms or 0
        })
        
        # Set result expiration (keep results for 24 hours by default)
        await self.redis.expire(f"task:{task_id}", 86400)
        
        # Update statistics
        if result.execution_time_ms:
            current_avg = self.stats["avg_processing_time_ms"]
            processed_count = self.stats["tasks_processed"]
            if processed_count > 0:
                self.stats["avg_processing_time_ms"] = (
                    (current_avg * (processed_count - 1) + result.execution_time_ms) / processed_count
                )
            else:
                self.stats["avg_processing_time_ms"] = result.execution_time_ms
        
        return True
    
    async def retry_task(self, task_def: TaskDefinition, error: str) -> bool:
        """Retry a failed task"""
        
        if task_def.max_retries <= 0:
            return False
        
        # Remove from processing queue
        await self._remove_from_processing(task_def.task_id)
        
        # Create retry task with decreased retry count
        retry_task = TaskDefinition(
            task_id=f"{task_def.task_id}_retry_{task_def.max_retries}",
            task_name=task_def.task_name,
            func_name=task_def.func_name,
            args=task_def.args,
            kwargs=task_def.kwargs,
            priority=task_def.priority,
            max_retries=task_def.max_retries - 1,
            retry_delay=task_def.retry_delay,
            timeout=task_def.timeout,
            expires_at=task_def.expires_at,
            metadata={**task_def.metadata, "retry_reason": error, "original_task_id": task_def.task_id}
        )
        
        # Schedule retry with delay
        retry_time = datetime.now(timezone.utc) + timedelta(seconds=task_def.retry_delay)
        await self.schedule_task(retry_task, retry_time)
        
        logger.info(f"Scheduled retry for task {task_def.task_id} at {retry_time}")
        return True
    
    async def schedule_task(self, task_def: TaskDefinition, execute_at: datetime) -> str:
        """Schedule a task for future execution"""
        
        # Store in scheduled tasks with timestamp as score
        timestamp = execute_at.timestamp()
        task_data = json.dumps({
            "task_id": task_def.task_id,
            "task_name": task_def.task_name,
            "func_name": task_def.func_name,
            "args": task_def.args,
            "kwargs": task_def.kwargs,
            "priority": task_def.priority.value,
            "max_retries": task_def.max_retries,
            "retry_delay": task_def.retry_delay,
            "timeout": task_def.timeout,
            "expires_at": task_def.expires_at.isoformat() if task_def.expires_at else None,
            "created_at": task_def.created_at.isoformat(),
            "metadata": task_def.metadata,
            "scheduled_for": execute_at.isoformat()
        })
        
        await self.redis.zadd(self.scheduled_key, {task_data: timestamp})
        
        # Store task metadata
        await self.redis.hset(f"task:{task_def.task_id}", mapping={
            "status": TaskStatus.PENDING.value,
            "queue": self.queue_name,
            "scheduled_for": execute_at.isoformat(),
            "created_at": task_def.created_at.isoformat()
        })
        
        logger.debug(f"Scheduled task {task_def.task_id} for {execute_at}")
        return task_def.task_id
    
    async def process_scheduled_tasks(self) -> int:
        """Move due scheduled tasks to pending queue"""
        
        current_time = time.time()
        
        # Get all tasks due for execution
        due_tasks = await self.redis.zrangebyscore(
            self.scheduled_key, 0, current_time, withscores=True
        )
        
        if not due_tasks:
            return 0
        
        processed_count = 0
        
        for task_data, score in due_tasks:
            try:
                # Parse task data
                task_info = json.loads(task_data)
                
                # Create task definition
                task_def = TaskDefinition(
                    task_id=task_info["task_id"],
                    task_name=task_info["task_name"],
                    func_name=task_info["func_name"],
                    args=task_info["args"],
                    kwargs=task_info["kwargs"],
                    priority=TaskPriority(task_info["priority"]),
                    max_retries=task_info["max_retries"],
                    retry_delay=task_info["retry_delay"],
                    timeout=task_info["timeout"],
                    expires_at=datetime.fromisoformat(task_info["expires_at"]) if task_info.get("expires_at") else None,
                    created_at=datetime.fromisoformat(task_info["created_at"]),
                    metadata=task_info["metadata"]
                )
                
                # Move to pending queue
                await self.enqueue(task_def)
                
                # Remove from scheduled queue
                await self.redis.zrem(self.scheduled_key, task_data)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing scheduled task: {e}")
                # Remove invalid task from scheduled queue
                await self.redis.zrem(self.scheduled_key, task_data)
        
        if processed_count > 0:
            logger.info(f"Moved {processed_count} scheduled tasks to pending queue")
        
        return processed_count
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task execution result"""
        
        # Check in results
        result_data = await self.redis.hget(self.results_key, task_id)
        if result_data:
            data = json.loads(result_data)
            return self._deserialize_task_result(data)
        
        # Check in failed tasks
        result_data = await self.redis.hget(self.failed_key, task_id)
        if result_data:
            data = json.loads(result_data)
            return self._deserialize_task_result(data)
        
        # Check if task is still running
        task_info = await self.redis.hgetall(f"task:{task_id}")
        if task_info:
            status = TaskStatus(task_info.get("status", "pending"))
            
            return TaskResult(
                task_id=task_id,
                status=status,
                started_at=datetime.fromisoformat(task_info["created_at"]) if task_info.get("created_at") else None,
                metadata={"queue": task_info.get("queue")}
            )
        
        return None
    
    def _deserialize_task_result(self, data: Dict[str, Any]) -> TaskResult:
        """Deserialize task result from Redis data"""
        
        result = None
        if data.get("result"):
            try:
                result = pickle.loads(data["result"])
            except Exception as e:
                logger.warning(f"Failed to deserialize task result: {e}")
        
        return TaskResult(
            task_id=data["task_id"],
            status=TaskStatus(data["status"]),
            result=result,
            error=data.get("error"),
            traceback=data.get("traceback"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            execution_time_ms=data.get("execution_time_ms"),
            worker_id=data.get("worker_id"),
            retry_count=data.get("retry_count", 0),
            metadata=json.loads(data.get("metadata", "{}"))
        )
    
    async def _remove_from_processing(self, task_id: str):
        """Remove task from processing queue"""
        
        # Get all items in processing queue
        processing_tasks = await self.redis.lrange(self.processing_key, 0, -1)
        
        for task_data in processing_tasks:
            try:
                task_info = json.loads(task_data)
                if task_info["task_id"] == task_id:
                    await self.redis.lrem(self.processing_key, 1, task_data)
                    break
            except json.JSONDecodeError:
                continue
    
    async def _mark_task_expired(self, task_id: str):
        """Mark task as expired"""
        
        await self.redis.hset(f"task:{task_id}", mapping={
            "status": TaskStatus.EXPIRED.value,
            "completed_at": datetime.now(timezone.utc).isoformat()
        })
        
        logger.debug(f"Task {task_id} marked as expired")
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""
        
        pending_count = await self.redis.llen(self.pending_key)
        processing_count = await self.redis.llen(self.processing_key)
        scheduled_count = await self.redis.zcard(self.scheduled_key)
        results_count = await self.redis.hlen(self.results_key)
        failed_count = await self.redis.hlen(self.failed_key)
        
        self.stats["current_queue_size"] = pending_count
        
        return {
            "queue_name": self.queue_name,
            "pending_tasks": pending_count,
            "processing_tasks": processing_count,
            "scheduled_tasks": scheduled_count,
            "completed_tasks": results_count,
            "failed_tasks": failed_count,
            "total_enqueued": self.stats["tasks_enqueued"],
            "total_processed": self.stats["tasks_processed"],
            "total_failed": self.stats["tasks_failed"],
            "avg_processing_time_ms": self.stats["avg_processing_time_ms"],
            "success_rate": (
                self.stats["tasks_processed"] / 
                max(1, self.stats["tasks_processed"] + self.stats["tasks_failed"])
            ) * 100
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or scheduled task"""
        
        # Remove from pending queue
        pending_tasks = await self.redis.lrange(self.pending_key, 0, -1)
        for task_data in pending_tasks:
            try:
                task_info = json.loads(task_data)
                if task_info["task_id"] == task_id:
                    await self.redis.lrem(self.pending_key, 1, task_data)
                    await self.redis.hset(f"task:{task_id}", "status", TaskStatus.CANCELLED.value)
                    return True
            except json.JSONDecodeError:
                continue
        
        # Remove from scheduled queue
        scheduled_tasks = await self.redis.zrange(self.scheduled_key, 0, -1)
        for task_data in scheduled_tasks:
            try:
                task_info = json.loads(task_data)
                if task_info["task_id"] == task_id:
                    await self.redis.zrem(self.scheduled_key, task_data)
                    await self.redis.hset(f"task:{task_id}", "status", TaskStatus.CANCELLED.value)
                    return True
            except json.JSONDecodeError:
                continue
        
        return False
    
    async def purge_queue(self, queue_type: str = "all") -> Dict[str, int]:
        """Purge queue contents"""
        
        purged_counts = {}
        
        if queue_type in ["all", "pending"]:
            count = await self.redis.delete(self.pending_key)
            purged_counts["pending"] = count
        
        if queue_type in ["all", "processing"]:
            count = await self.redis.delete(self.processing_key)
            purged_counts["processing"] = count
        
        if queue_type in ["all", "scheduled"]:
            count = await self.redis.delete(self.scheduled_key)
            purged_counts["scheduled"] = count
        
        if queue_type in ["all", "results"]:
            count = await self.redis.delete(self.results_key)
            purged_counts["results"] = count
        
        if queue_type in ["all", "failed"]:
            count = await self.redis.delete(self.failed_key)
            purged_counts["failed"] = count
        
        logger.info(f"Purged queue '{self.queue_name}': {purged_counts}")
        return purged_counts


# Decorator for registering tasks
def task(name: Optional[str] = None,
         priority: TaskPriority = TaskPriority.NORMAL,
         max_retries: int = 3,
         timeout: int = 300,
         queue: str = "default"):
    """Decorator to register a function as a task"""
    
    def decorator(func: Callable) -> Callable:
        task_name = name or f"{func.__module__}.{func.__name__}"
        
        # This would be registered with a global task registry
        # For now, we'll add metadata to the function
        func._task_name = task_name
        func._task_priority = priority
        func._task_max_retries = max_retries
        func._task_timeout = timeout
        func._task_queue = queue
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Global task queue instances
task_queues: Dict[str, TaskQueue] = {}


async def initialize_task_queues(redis_client: aioredis.Redis, 
                                queue_configs: Optional[Dict[str, Dict[str, Any]]] = None):
    """Initialize task queue system"""
    global task_queues
    
    default_queues = ["default", "high_priority", "low_priority", "background"]
    queue_configs = queue_configs or {}
    
    for queue_name in default_queues:
        config = queue_configs.get(queue_name, {})
        task_queue = TaskQueue(
            redis_client=redis_client,
            queue_name=queue_name,
            max_queue_size=config.get("max_queue_size", 10000)
        )
        task_queues[queue_name] = task_queue
    
    logger.info(f"✅ Task queue system initialized with {len(task_queues)} queues")


def get_task_queue(queue_name: str = "default") -> TaskQueue:
    """Get task queue instance"""
    if queue_name not in task_queues:
        raise RuntimeError(f"Task queue '{queue_name}' not initialized")
    return task_queues[queue_name]


async def enqueue_task(func_name: str, *args, 
                      queue: str = "default",
                      priority: TaskPriority = TaskPriority.NORMAL,
                      **kwargs) -> str:
    """Enqueue a task for processing"""
    
    task_queue = get_task_queue(queue)
    task_id = str(uuid.uuid4())
    
    task_def = TaskDefinition(
        task_id=task_id,
        task_name=func_name,
        func_name=func_name,
        args=list(args),
        kwargs=kwargs,
        priority=priority
    )
    
    await task_queue.enqueue(task_def)
    return task_id


async def get_all_queue_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all queues"""
    
    all_stats = {}
    for queue_name, task_queue in task_queues.items():
        all_stats[queue_name] = await task_queue.get_queue_stats()
    
    return all_stats