#!/usr/bin/env python3
"""
Advanced Task Distribution System
=================================

Intelligent task distribution engine for managing AI workloads across
multiple models with priority queuing, resource optimization, and adaptive scheduling.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
import uuid
from pathlib import Path
import heapq
import math

from .model_manager import ModelManager, ModelInstance, ModelCapability

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class DistributionStrategy(Enum):
    """Task distribution strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_MATCH = "capability_match"
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    HYBRID = "hybrid"


@dataclass
class TaskRequirements:
    """Task resource and capability requirements"""
    required_capabilities: Set[ModelCapability] = field(default_factory=set)
    preferred_capabilities: Set[ModelCapability] = field(default_factory=set)
    
    # Resource requirements
    max_tokens: Optional[int] = None
    estimated_compute_units: float = 1.0
    memory_requirement_mb: Optional[int] = None
    
    # Quality requirements
    min_quality_score: float = 0.0
    preferred_quality_score: float = 80.0
    
    # Performance requirements
    max_response_time_ms: Optional[int] = None
    deadline: Optional[datetime] = None
    
    # Cost constraints
    max_cost_per_token: Optional[float] = None
    budget_limit: Optional[float] = None
    
    # Model preferences
    preferred_providers: Set[str] = field(default_factory=set)
    excluded_models: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "required_capabilities": [cap.value for cap in self.required_capabilities],
            "preferred_capabilities": [cap.value for cap in self.preferred_capabilities],
            "max_tokens": self.max_tokens,
            "estimated_compute_units": self.estimated_compute_units,
            "memory_requirement_mb": self.memory_requirement_mb,
            "min_quality_score": self.min_quality_score,
            "preferred_quality_score": self.preferred_quality_score,
            "max_response_time_ms": self.max_response_time_ms,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "max_cost_per_token": self.max_cost_per_token,
            "budget_limit": self.budget_limit,
            "preferred_providers": list(self.preferred_providers),
            "excluded_models": list(self.excluded_models)
        }


@dataclass
class Task:
    """AI task definition"""
    task_id: str
    name: str
    task_type: str
    description: str = ""
    
    # Task data
    input_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Requirements and constraints
    requirements: TaskRequirements = field(default_factory=TaskRequirements)
    
    # Priority and scheduling
    priority: TaskPriority = TaskPriority.MEDIUM
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: Optional[datetime] = None
    
    # Execution tracking
    status: TaskStatus = TaskStatus.PENDING
    assigned_model_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results and metrics
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    cost: Optional[float] = None
    
    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    
    # Callback configuration
    callback_url: Optional[str] = None
    callback_headers: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        """For priority queue sorting"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.submitted_at < other.submitted_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "task_type": self.task_type,
            "description": self.description,
            "input_data": self.input_data,
            "context": self.context,
            "requirements": self.requirements.to_dict(),
            "priority": self.priority.value,
            "submitted_at": self.submitted_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "status": self.status.value,
            "assigned_model_id": self.assigned_model_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "cost": self.cost,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "callback_url": self.callback_url,
            "callback_headers": self.callback_headers,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tags": self.tags
        }


@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    success: bool
    result_data: Dict[str, Any]
    execution_time_ms: float
    model_used: str
    cost: float = 0.0
    quality_score: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result_data": self.result_data,
            "execution_time_ms": self.execution_time_ms,
            "model_used": self.model_used,
            "cost": self.cost,
            "quality_score": self.quality_score,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat()
        }


class TaskQueue:
    """Priority-based task queue with advanced scheduling"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.priority_queue: List[Task] = []
        self.tasks_by_id: Dict[str, Task] = {}
        self.lock = asyncio.Lock()
    
    async def enqueue(self, task: Task) -> bool:
        """Add task to queue"""
        async with self.lock:
            if len(self.priority_queue) >= self.max_size:
                return False
            
            task.status = TaskStatus.QUEUED
            heapq.heappush(self.priority_queue, task)
            self.tasks_by_id[task.task_id] = task
            
            logger.info(f"Enqueued task: {task.name} (Priority: {task.priority.name})")
            return True
    
    async def dequeue(self) -> Optional[Task]:
        """Get next highest priority task"""
        async with self.lock:
            if not self.priority_queue:
                return None
            
            task = heapq.heappop(self.priority_queue)
            del self.tasks_by_id[task.task_id]
            
            return task
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get specific task by ID"""
        return self.tasks_by_id.get(task_id)
    
    async def remove_task(self, task_id: str) -> bool:
        """Remove task from queue"""
        async with self.lock:
            if task_id not in self.tasks_by_id:
                return False
            
            task = self.tasks_by_id[task_id]
            task.status = TaskStatus.CANCELLED
            del self.tasks_by_id[task_id]
            
            # Remove from priority queue (requires rebuilding)
            self.priority_queue = [t for t in self.priority_queue if t.task_id != task_id]
            heapq.heapify(self.priority_queue)
            
            return True
    
    def size(self) -> int:
        """Get queue size"""
        return len(self.priority_queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.priority_queue) == 0
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        priority_counts = {}
        for task in self.priority_queue:
            priority = task.priority.name
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return {
            "total_tasks": len(self.priority_queue),
            "priority_breakdown": priority_counts,
            "oldest_task_age_seconds": self._get_oldest_task_age(),
            "average_wait_time_seconds": self._get_average_wait_time()
        }
    
    def _get_oldest_task_age(self) -> float:
        """Get age of oldest task in seconds"""
        if not self.priority_queue:
            return 0.0
        
        oldest_task = min(self.priority_queue, key=lambda t: t.submitted_at)
        return (datetime.now(timezone.utc) - oldest_task.submitted_at).total_seconds()
    
    def _get_average_wait_time(self) -> float:
        """Get average wait time for tasks in queue"""
        if not self.priority_queue:
            return 0.0
        
        now = datetime.now(timezone.utc)
        total_wait_time = sum(
            (now - task.submitted_at).total_seconds() 
            for task in self.priority_queue
        )
        
        return total_wait_time / len(self.priority_queue)


class TaskMatcher:
    """Intelligent task-to-model matching system"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
        # Matching algorithms
        self.matching_strategies = {
            DistributionStrategy.CAPABILITY_MATCH: self._capability_match,
            DistributionStrategy.COST_OPTIMIZED: self._cost_optimized_match,
            DistributionStrategy.LATENCY_OPTIMIZED: self._latency_optimized_match,
            DistributionStrategy.QUALITY_OPTIMIZED: self._quality_optimized_match,
            DistributionStrategy.LEAST_LOADED: self._least_loaded_match,
            DistributionStrategy.HYBRID: self._hybrid_match
        }
        
        # Statistics
        self.stats = {
            "total_matches": 0,
            "successful_matches": 0,
            "failed_matches": 0,
            "matches_by_strategy": {}
        }
    
    async def find_best_model(self, task: Task, strategy: DistributionStrategy) -> Optional[ModelInstance]:
        """Find the best model for a task using specified strategy"""
        
        try:
            # Get matching function
            if strategy not in self.matching_strategies:
                logger.error(f"Unknown matching strategy: {strategy}")
                return None
            
            matching_func = self.matching_strategies[strategy]
            
            # Find best model
            best_model = await matching_func(task)
            
            # Update statistics
            self.stats["total_matches"] += 1
            strategy_name = strategy.value
            self.stats["matches_by_strategy"][strategy_name] = \
                self.stats["matches_by_strategy"].get(strategy_name, 0) + 1
            
            if best_model:
                self.stats["successful_matches"] += 1
            else:
                self.stats["failed_matches"] += 1
            
            return best_model
            
        except Exception as e:
            logger.error(f"Task matching error: {e}")
            self.stats["failed_matches"] += 1
            return None
    
    async def _capability_match(self, task: Task) -> Optional[ModelInstance]:
        """Match based on model capabilities"""
        available_models = self.model_manager.list_models()
        
        # Filter by required capabilities
        compatible_models = []
        for model in available_models:
            if not task.requirements.required_capabilities.issubset(model.capabilities):
                continue
            
            if model.model_id in task.requirements.excluded_models:
                continue
            
            compatible_models.append(model)
        
        if not compatible_models:
            return None
        
        # Score models based on capability overlap
        def capability_score(model: ModelInstance) -> float:
            required_match = len(task.requirements.required_capabilities.intersection(model.capabilities))
            preferred_match = len(task.requirements.preferred_capabilities.intersection(model.capabilities))
            total_capabilities = len(model.capabilities)
            
            return (required_match * 2 + preferred_match) / max(1, total_capabilities)
        
        return max(compatible_models, key=capability_score)
    
    async def _cost_optimized_match(self, task: Task) -> Optional[ModelInstance]:
        """Match based on cost optimization"""
        available_models = self.model_manager.list_models()
        
        # Filter compatible models
        compatible_models = [
            model for model in available_models
            if (task.requirements.required_capabilities.issubset(model.capabilities) and
                model.model_id not in task.requirements.excluded_models and
                (not task.requirements.max_cost_per_token or 
                 model.metrics.cost_per_token <= task.requirements.max_cost_per_token))
        ]
        
        if not compatible_models:
            return None
        
        # Select model with lowest cost per token
        return min(compatible_models, key=lambda m: m.metrics.cost_per_token)
    
    async def _latency_optimized_match(self, task: Task) -> Optional[ModelInstance]:
        """Match based on latency optimization"""
        available_models = self.model_manager.list_models()
        
        # Filter compatible models
        compatible_models = [
            model for model in available_models
            if (task.requirements.required_capabilities.issubset(model.capabilities) and
                model.model_id not in task.requirements.excluded_models and
                (not task.requirements.max_response_time_ms or 
                 model.metrics.avg_response_time_ms <= task.requirements.max_response_time_ms))
        ]
        
        if not compatible_models:
            return None
        
        # Select model with fastest response time
        return min(compatible_models, key=lambda m: m.metrics.avg_response_time_ms)
    
    async def _quality_optimized_match(self, task: Task) -> Optional[ModelInstance]:
        """Match based on quality optimization"""
        available_models = self.model_manager.list_models()
        
        # Filter compatible models
        compatible_models = [
            model for model in available_models
            if (task.requirements.required_capabilities.issubset(model.capabilities) and
                model.model_id not in task.requirements.excluded_models and
                model.metrics.quality_score >= task.requirements.min_quality_score)
        ]
        
        if not compatible_models:
            return None
        
        # Select model with highest quality score
        return max(compatible_models, key=lambda m: m.metrics.quality_score)
    
    async def _least_loaded_match(self, task: Task) -> Optional[ModelInstance]:
        """Match based on model load"""
        available_models = self.model_manager.list_models()
        
        # Filter compatible models
        compatible_models = [
            model for model in available_models
            if (task.requirements.required_capabilities.issubset(model.capabilities) and
                model.model_id not in task.requirements.excluded_models)
        ]
        
        if not compatible_models:
            return None
        
        # Select model with lowest queue length
        return min(compatible_models, key=lambda m: m.metrics.queue_length)
    
    async def _hybrid_match(self, task: Task) -> Optional[ModelInstance]:
        """Hybrid matching considering multiple factors"""
        available_models = self.model_manager.list_models()
        
        # Filter compatible models
        compatible_models = [
            model for model in available_models
            if (task.requirements.required_capabilities.issubset(model.capabilities) and
                model.model_id not in task.requirements.excluded_models)
        ]
        
        if not compatible_models:
            return None
        
        # Calculate composite score
        def composite_score(model: ModelInstance) -> float:
            # Normalize individual scores (0-1)
            quality_score = min(model.metrics.quality_score / 100, 1.0)
            speed_score = max(0, 1.0 - (model.metrics.avg_response_time_ms / 10000))  # 10s max
            cost_score = max(0, 1.0 - (model.metrics.cost_per_token * 1000))  # Assume reasonable range
            load_score = max(0, 1.0 - (model.metrics.queue_length / 100))  # 100 max queue
            health_score = model.health_score / 100
            
            # Weighted combination
            weights = {
                'quality': 0.25,
                'speed': 0.25,
                'cost': 0.20,
                'load': 0.20,
                'health': 0.10
            }
            
            return (quality_score * weights['quality'] +
                   speed_score * weights['speed'] +
                   cost_score * weights['cost'] +
                   load_score * weights['load'] +
                   health_score * weights['health'])
        
        return max(compatible_models, key=composite_score)
    
    def get_matching_stats(self) -> Dict[str, Any]:
        """Get task matching statistics"""
        return self.stats


class TaskExecutor:
    """Task execution engine"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time_ms": 0.0,
            "avg_execution_time_ms": 0.0
        }
    
    async def execute_task(self, task: Task, model: ModelInstance) -> TaskResult:
        """Execute a task on a specific model"""
        
        task.status = TaskStatus.ASSIGNED
        task.assigned_model_id = model.model_id
        task.started_at = datetime.now(timezone.utc)
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            
            # Execute task on model
            start_time = datetime.now()
            
            # Prepare request data
            request_data = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "input_data": task.input_data,
                "context": task.context,
                "requirements": task.requirements.to_dict()
            }
            
            # Execute on model
            execution_result = await self.model_manager.execute_request(
                model.model_id, 
                request_data
            )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create task result
            task_result = TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=execution_result,
                execution_time_ms=execution_time,
                model_used=model.model_id,
                cost=self._calculate_cost(execution_result, model),
                quality_score=self._evaluate_quality(execution_result, task)
            )
            
            # Update task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            task.result = execution_result
            task.execution_time_ms = execution_time
            task.cost = task_result.cost
            
            # Update statistics
            self.stats["total_executions"] += 1
            self.stats["successful_executions"] += 1
            self.stats["total_execution_time_ms"] += execution_time
            self._update_avg_execution_time()
            
            logger.info(f"Task executed successfully: {task.task_id}")
            
            return task_result
            
        except Exception as e:
            # Handle execution failure
            execution_time = (datetime.now() - task.started_at).total_seconds() * 1000
            
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                result_data={},
                execution_time_ms=execution_time,
                model_used=model.model_id,
                error_message=str(e)
            )
            
            # Update task
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now(timezone.utc)
            task.error_message = str(e)
            task.execution_time_ms = execution_time
            
            # Update statistics
            self.stats["total_executions"] += 1
            self.stats["failed_executions"] += 1
            self.stats["total_execution_time_ms"] += execution_time
            self._update_avg_execution_time()
            
            logger.error(f"Task execution failed: {task.task_id} - {e}")
            
            return task_result
    
    def _calculate_cost(self, execution_result: Dict[str, Any], model: ModelInstance) -> float:
        """Calculate task execution cost"""
        tokens_used = execution_result.get("tokens_used", 0)
        cost_per_token = model.metrics.cost_per_token
        return tokens_used * cost_per_token
    
    def _evaluate_quality(self, execution_result: Dict[str, Any], task: Task) -> float:
        """Evaluate result quality (placeholder for actual quality assessment)"""
        # This would contain actual quality evaluation logic
        # For now, return a baseline score
        return 80.0
    
    def _update_avg_execution_time(self):
        """Update average execution time"""
        if self.stats["total_executions"] > 0:
            self.stats["avg_execution_time_ms"] = \
                self.stats["total_execution_time_ms"] / self.stats["total_executions"]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            **self.stats,
            "running_tasks": len(self.running_tasks)
        }


class TaskDistributor:
    """Main task distribution system"""
    
    def __init__(self, model_manager: ModelManager, max_concurrent_tasks: int = 100):
        self.model_manager = model_manager
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Core components
        self.task_queue = TaskQueue()
        self.task_matcher = TaskMatcher(model_manager)
        self.task_executor = TaskExecutor(model_manager)
        
        # Task registry
        self.tasks: Dict[str, Task] = {}
        
        # Configuration
        self.default_strategy = DistributionStrategy.HYBRID
        self.enable_auto_retry = True
        
        # Worker management
        self.worker_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Statistics
        self.stats = {
            "total_tasks_submitted": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            "current_queue_size": 0,
            "active_workers": 0
        }
        
        logger.info("Task Distributor initialized")
    
    async def submit_task(self, task: Task) -> bool:
        """Submit a task for execution"""
        
        # Store task
        self.tasks[task.task_id] = task
        
        # Add to queue
        success = await self.task_queue.enqueue(task)
        
        if success:
            self.stats["total_tasks_submitted"] += 1
            self.stats["current_queue_size"] = self.task_queue.size()
            
            logger.info(f"Task submitted: {task.task_id}")
            return True
        else:
            logger.error(f"Failed to submit task: {task.task_id} - Queue full")
            return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        
        # Remove from queue if pending
        if await self.task_queue.remove_task(task_id):
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.CANCELLED
            
            logger.info(f"Task cancelled: {task_id}")
            return True
        
        return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and details"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return task.to_dict()
    
    async def start_workers(self, num_workers: int = 5):
        """Start task processing workers"""
        if self.running:
            logger.warning("Workers already running")
            return
        
        self.running = True
        
        # Start worker tasks
        for i in range(num_workers):
            worker_task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.worker_tasks.append(worker_task)
        
        self.stats["active_workers"] = num_workers
        logger.info(f"Started {num_workers} task processing workers")
    
    async def stop_workers(self):
        """Stop all task processing workers"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all worker tasks
        for worker_task in self.worker_tasks:
            worker_task.cancel()
        
        # Wait for workers to stop
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        self.stats["active_workers"] = 0
        
        logger.info("Stopped all task processing workers")
    
    async def _worker_loop(self, worker_name: str):
        """Main worker processing loop"""
        logger.info(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get next task from queue
                task = await self.task_queue.dequeue()
                
                if not task:
                    # No tasks available, wait briefly
                    await asyncio.sleep(0.1)
                    continue
                
                # Find best model for task
                best_model = await self.task_matcher.find_best_model(
                    task, 
                    self.default_strategy
                )
                
                if not best_model:
                    # No suitable model found, mark as failed
                    task.status = TaskStatus.FAILED
                    task.error_message = "No suitable model available"
                    task.completed_at = datetime.now(timezone.utc)
                    
                    self.stats["total_tasks_failed"] += 1
                    
                    logger.error(f"No suitable model for task: {task.task_id}")
                    continue
                
                # Execute task
                task_result = await self.task_executor.execute_task(task, best_model)
                
                # Handle result
                if task_result.success:
                    self.stats["total_tasks_completed"] += 1
                    logger.info(f"Task completed: {task.task_id}")
                else:
                    # Check if we should retry
                    if (self.enable_auto_retry and 
                        task.retry_count < task.max_retries):
                        
                        task.retry_count += 1
                        task.status = TaskStatus.PENDING
                        
                        # Re-queue for retry
                        await self.task_queue.enqueue(task)
                        
                        logger.info(f"Task queued for retry: {task.task_id} (attempt {task.retry_count})")
                    else:
                        self.stats["total_tasks_failed"] += 1
                        logger.error(f"Task failed permanently: {task.task_id}")
                
                # Update queue size stat
                self.stats["current_queue_size"] = self.task_queue.size()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.info(f"Worker {worker_name} stopped")
    
    def set_distribution_strategy(self, strategy: DistributionStrategy):
        """Set the default task distribution strategy"""
        self.default_strategy = strategy
        logger.info(f"Distribution strategy set to: {strategy.value}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "distributor_stats": self.stats,
            "queue_stats": self.task_queue.get_queue_stats(),
            "matching_stats": self.task_matcher.get_matching_stats(),
            "execution_stats": self.task_executor.get_execution_stats(),
            "model_stats": self.model_manager.get_system_stats(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def shutdown(self):
        """Graceful shutdown of task distributor"""
        logger.info("Shutting down Task Distributor")
        
        # Stop workers
        await self.stop_workers()
        
        logger.info("Task Distributor shutdown complete")


# Export main classes
__all__ = [
    'TaskPriority',
    'TaskStatus',
    'DistributionStrategy',
    'TaskRequirements',
    'Task',
    'TaskResult',
    'TaskQueue',
    'TaskMatcher',
    'TaskExecutor',
    'TaskDistributor'
]