#!/usr/bin/env python3
"""
Advanced Workflow Management System
===================================

Comprehensive workflow orchestration for complex AI operations with
dynamic execution, conditional branching, and adaptive optimization.
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
import math

from .model_manager import ModelManager, ModelInstance
from .task_distributor import TaskDistributor, Task, TaskPriority
from .reasoning_engine import ReasoningEngine, ReasoningChain

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepType(Enum):
    """Workflow step types"""
    TASK = "task"
    REASONING_CHAIN = "reasoning_chain"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    WAIT = "wait"
    WEBHOOK = "webhook"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    DECISION = "decision"


class ExecutionMode(Enum):
    """Workflow execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ADAPTIVE = "adaptive"
    FAULT_TOLERANT = "fault_tolerant"


@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    step_id: str
    name: str
    step_type: StepType
    description: str = ""
    
    # Step configuration
    config: Dict[str, Any] = field(default_factory=dict)
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Dependencies and flow control
    depends_on: List[str] = field(default_factory=list)
    success_next: Optional[str] = None
    failure_next: Optional[str] = None
    condition_next: Dict[str, str] = field(default_factory=dict)
    
    # Execution settings
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 5
    
    # Status tracking
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: float = 0.0
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Quality metrics
    success_rate: float = 100.0
    avg_execution_time_ms: float = 0.0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "step_type": self.step_type.value,
            "description": self.description,
            "config": self.config,
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping,
            "depends_on": self.depends_on,
            "success_next": self.success_next,
            "failure_next": self.failure_next,
            "condition_next": self.condition_next,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_ms": self.execution_time_ms,
            "result": self.result,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "success_rate": self.success_rate,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "tags": self.tags,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Workflow:
    """Workflow definition and state"""
    workflow_id: str
    name: str
    description: str = ""
    version: str = "1.0"
    
    # Workflow configuration
    steps: List[WorkflowStep] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    
    # Global settings
    global_timeout_seconds: int = 3600
    max_parallel_steps: int = 10
    enable_checkpoints: bool = True
    enable_auto_retry: bool = True
    
    # Input/Output schema
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Execution state
    status: WorkflowStatus = WorkflowStatus.DRAFT
    current_step_id: Optional[str] = None
    completed_steps: int = 0
    failed_steps: int = 0
    
    # Context and data
    context: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    success_rate: float = 100.0
    avg_execution_time_ms: float = 0.0
    total_executions: int = 0
    
    # Metadata
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)
    
    def add_step(self, step: WorkflowStep):
        """Add step to workflow"""
        self.steps.append(step)
        self.updated_at = datetime.now(timezone.utc)
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_next_steps(self, current_step_id: Optional[str] = None) -> List[str]:
        """Get next executable steps"""
        if not current_step_id:
            # Find steps with no dependencies
            return [step.step_id for step in self.steps if not step.depends_on]
        
        current_step = self.get_step(current_step_id)
        if not current_step:
            return []
        
        # Determine next steps based on current step result
        if current_step.status == "completed":
            if current_step.success_next:
                return [current_step.success_next]
        elif current_step.status == "failed":
            if current_step.failure_next:
                return [current_step.failure_next]
        
        # Check condition-based next steps
        if current_step.condition_next and current_step.result:
            for condition, next_step_id in current_step.condition_next.items():
                if self._evaluate_condition(condition, current_step.result):
                    return [next_step_id]
        
        return []
    
    def _evaluate_condition(self, condition: str, result: Dict[str, Any]) -> bool:
        """Evaluate condition against step result"""
        # Simple condition evaluation (would be enhanced with expression parser)
        try:
            # Replace result variables in condition
            for key, value in result.items():
                condition = condition.replace(f"{{{key}}}", str(value))
            
            # Basic evaluation (unsafe - would use safe evaluator in production)
            return eval(condition)
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": [step.to_dict() for step in self.steps],
            "execution_mode": self.execution_mode.value,
            "global_timeout_seconds": self.global_timeout_seconds,
            "max_parallel_steps": self.max_parallel_steps,
            "enable_checkpoints": self.enable_checkpoints,
            "enable_auto_retry": self.enable_auto_retry,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "status": self.status.value,
            "current_step_id": self.current_step_id,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "context": self.context,
            "variables": self.variables,
            "success_rate": self.success_rate,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "total_executions": self.total_executions,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags
        }


@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str
    workflow_id: str
    workflow_name: str
    
    # Input data
    input_data: Dict[str, Any] = field(default_factory=dict)
    
    # Execution state
    status: WorkflowStatus = WorkflowStatus.READY
    current_step_id: Optional[str] = None
    
    # Progress tracking
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    
    # Results and outputs
    step_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    final_output: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Performance metrics
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_execution_time_ms: float = 0.0
    
    # Resource usage
    models_used: List[str] = field(default_factory=list)
    total_cost: float = 0.0
    tokens_consumed: int = 0
    
    # Context
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    triggered_by: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "input_data": self.input_data,
            "status": self.status.value,
            "current_step_id": self.current_step_id,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "steps_skipped": self.steps_skipped,
            "step_results": self.step_results,
            "final_output": self.final_output,
            "errors": self.errors,
            "warnings": self.warnings,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_execution_time_ms": self.total_execution_time_ms,
            "models_used": self.models_used,
            "total_cost": self.total_cost,
            "tokens_consumed": self.tokens_consumed,
            "execution_context": self.execution_context,
            "triggered_by": self.triggered_by,
            "priority": self.priority.value
        }


class WorkflowScheduler:
    """Workflow scheduling and execution engine"""
    
    def __init__(self, task_distributor: TaskDistributor, reasoning_engine: ReasoningEngine):
        self.task_distributor = task_distributor
        self.reasoning_engine = reasoning_engine
        
        # Active executions
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Scheduling configuration
        self.max_concurrent_workflows: int = 50
        self.execution_timeout_seconds: int = 3600
        
        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time_ms": 0.0,
            "active_workflows": 0
        }
    
    async def execute_workflow(self, workflow: Workflow, execution: WorkflowExecution) -> WorkflowExecution:
        """Execute a workflow"""
        
        execution.status = WorkflowStatus.RUNNING
        execution.started_at = datetime.now(timezone.utc)
        start_time = execution.started_at
        
        try:
            # Execute based on workflow mode
            if workflow.execution_mode == ExecutionMode.SEQUENTIAL:
                result = await self._execute_sequential(workflow, execution)
            elif workflow.execution_mode == ExecutionMode.PARALLEL:
                result = await self._execute_parallel(workflow, execution)
            elif workflow.execution_mode == ExecutionMode.CONDITIONAL:
                result = await self._execute_conditional(workflow, execution)
            elif workflow.execution_mode == ExecutionMode.ADAPTIVE:
                result = await self._execute_adaptive(workflow, execution)
            else:
                result = await self._execute_sequential(workflow, execution)
            
            # Calculate execution time
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            execution.total_execution_time_ms = execution_time
            execution.completed_at = datetime.now(timezone.utc)
            
            # Update statistics
            self.stats["total_executions"] += 1
            if execution.status == WorkflowStatus.COMPLETED:
                self.stats["successful_executions"] += 1
            else:
                self.stats["failed_executions"] += 1
            
            self._update_avg_execution_time(execution_time)
            
            return execution
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.errors.append({
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step_id": execution.current_step_id
            })
            execution.completed_at = datetime.now(timezone.utc)
            
            self.stats["total_executions"] += 1
            self.stats["failed_executions"] += 1
            
            logger.error(f"Workflow execution failed: {workflow.name} - {e}")
            
            return execution
    
    async def _execute_sequential(self, workflow: Workflow, execution: WorkflowExecution) -> WorkflowExecution:
        """Execute workflow steps sequentially"""
        
        # Get dependency-ordered execution list
        execution_order = self._resolve_step_dependencies(workflow)
        
        if not execution_order:
            execution.status = WorkflowStatus.FAILED
            execution.errors.append({
                "error": "Unable to resolve step dependencies",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return execution
        
        # Execute steps in order
        for step_id in execution_order:
            step = workflow.get_step(step_id)
            if not step:
                continue
            
            execution.current_step_id = step_id
            
            try:
                step_result = await self._execute_workflow_step(step, workflow, execution)
                
                execution.step_results[step_id] = step_result
                
                if step.status == "completed":
                    execution.steps_completed.append(step_id)
                    workflow.completed_steps += 1
                elif step.status == "failed":
                    execution.steps_failed.append(step_id)
                    workflow.failed_steps += 1
                    
                    # Check if workflow should continue on failure
                    if not step.failure_next:
                        execution.status = WorkflowStatus.FAILED
                        break
                
                # Update execution context with step outputs
                if step.result:
                    for output_key, context_key in step.output_mapping.items():
                        if output_key in step.result:
                            execution.execution_context[context_key] = step.result[output_key]
                
            except Exception as e:
                step.status = "failed"
                step.error_message = str(e)
                execution.steps_failed.append(step_id)
                workflow.failed_steps += 1
                
                execution.errors.append({
                    "error": str(e),
                    "step_id": step_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                # Stop on error unless fault tolerant
                if workflow.execution_mode != ExecutionMode.FAULT_TOLERANT:
                    execution.status = WorkflowStatus.FAILED
                    break
        
        # Set final status
        if execution.status == WorkflowStatus.RUNNING:
            if len(execution.steps_failed) == 0:
                execution.status = WorkflowStatus.COMPLETED
            else:
                execution.status = WorkflowStatus.FAILED
        
        # Compile final output
        execution.final_output = self._compile_workflow_output(workflow, execution)
        
        return execution
    
    async def _execute_parallel(self, workflow: Workflow, execution: WorkflowExecution) -> WorkflowExecution:
        """Execute workflow with parallel processing where possible"""
        
        # Group steps by dependency level
        dependency_levels = self._group_steps_by_dependency_level(workflow)
        
        # Execute each level in parallel
        for level_steps in dependency_levels:
            if not level_steps:
                continue
            
            # Execute all steps in this level concurrently
            step_tasks = []
            for step_id in level_steps:
                step = workflow.get_step(step_id)
                if step:
                    task = asyncio.create_task(
                        self._execute_workflow_step(step, workflow, execution)
                    )
                    step_tasks.append((step, task))
            
            # Wait for all steps in this level to complete
            for step, task in step_tasks:
                try:
                    step_result = await task
                    execution.step_results[step.step_id] = step_result
                    
                    if step.status == "completed":
                        execution.steps_completed.append(step.step_id)
                        workflow.completed_steps += 1
                    else:
                        execution.steps_failed.append(step.step_id)
                        workflow.failed_steps += 1
                    
                    # Update execution context
                    if step.result:
                        for output_key, context_key in step.output_mapping.items():
                            if output_key in step.result:
                                execution.execution_context[context_key] = step.result[output_key]
                
                except Exception as e:
                    step.status = "failed"
                    step.error_message = str(e)
                    execution.steps_failed.append(step.step_id)
                    workflow.failed_steps += 1
                    
                    execution.errors.append({
                        "error": str(e),
                        "step_id": step.step_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
        
        # Set final status
        if len(execution.steps_failed) == 0:
            execution.status = WorkflowStatus.COMPLETED
        else:
            execution.status = WorkflowStatus.FAILED
        
        # Compile final output
        execution.final_output = self._compile_workflow_output(workflow, execution)
        
        return execution
    
    async def _execute_conditional(self, workflow: Workflow, execution: WorkflowExecution) -> WorkflowExecution:
        """Execute workflow with conditional branching"""
        
        current_step_ids = workflow.get_next_steps()
        visited_steps = set()
        
        while current_step_ids and execution.status == WorkflowStatus.RUNNING:
            next_step_ids = []
            
            for step_id in current_step_ids:
                if step_id in visited_steps:
                    continue  # Avoid cycles
                
                visited_steps.add(step_id)
                step = workflow.get_step(step_id)
                
                if not step:
                    continue
                
                execution.current_step_id = step_id
                
                try:
                    step_result = await self._execute_workflow_step(step, workflow, execution)
                    execution.step_results[step_id] = step_result
                    
                    if step.status == "completed":
                        execution.steps_completed.append(step_id)
                        workflow.completed_steps += 1
                        
                        # Determine next steps based on result
                        next_steps = workflow.get_next_steps(step_id)
                        next_step_ids.extend(next_steps)
                        
                    elif step.status == "failed":
                        execution.steps_failed.append(step_id)
                        workflow.failed_steps += 1
                        
                        # Handle failure path
                        if step.failure_next:
                            next_step_ids.append(step.failure_next)
                        else:
                            execution.status = WorkflowStatus.FAILED
                            break
                    
                    # Update execution context
                    if step.result:
                        for output_key, context_key in step.output_mapping.items():
                            if output_key in step.result:
                                execution.execution_context[context_key] = step.result[output_key]
                
                except Exception as e:
                    step.status = "failed"
                    step.error_message = str(e)
                    execution.steps_failed.append(step_id)
                    workflow.failed_steps += 1
                    
                    execution.errors.append({
                        "error": str(e),
                        "step_id": step_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
                    execution.status = WorkflowStatus.FAILED
                    break
            
            current_step_ids = list(set(next_step_ids))  # Remove duplicates
        
        # Set final status if still running
        if execution.status == WorkflowStatus.RUNNING:
            execution.status = WorkflowStatus.COMPLETED
        
        # Compile final output
        execution.final_output = self._compile_workflow_output(workflow, execution)
        
        return execution
    
    async def _execute_adaptive(self, workflow: Workflow, execution: WorkflowExecution) -> WorkflowExecution:
        """Execute workflow with adaptive optimization"""
        # Placeholder for adaptive execution (would include dynamic optimization)
        logger.warning("Adaptive execution mode not fully implemented, falling back to conditional")
        return await self._execute_conditional(workflow, execution)
    
    async def _execute_workflow_step(self, step: WorkflowStep, workflow: Workflow, 
                                   execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute an individual workflow step"""
        
        step.status = "running"
        step.started_at = datetime.now(timezone.utc)
        start_time = step.started_at
        
        try:
            # Prepare step input from execution context
            step_input = self._prepare_step_input(step, execution)
            
            # Execute based on step type
            if step.step_type == StepType.TASK:
                result = await self._execute_task_step(step, step_input)
            elif step.step_type == StepType.REASONING_CHAIN:
                result = await self._execute_reasoning_step(step, step_input)
            elif step.step_type == StepType.CONDITION:
                result = await self._execute_condition_step(step, step_input)
            elif step.step_type == StepType.TRANSFORM:
                result = await self._execute_transform_step(step, step_input)
            elif step.step_type == StepType.WAIT:
                result = await self._execute_wait_step(step, step_input)
            else:
                result = {"error": f"Unsupported step type: {step.step_type}"}
            
            # Calculate execution time
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            step.execution_time_ms = execution_time
            step.completed_at = datetime.now(timezone.utc)
            
            # Update step statistics
            step.avg_execution_time_ms = \
                (step.avg_execution_time_ms + execution_time) / 2
            
            if "error" not in result:
                step.status = "completed"
                step.result = result
            else:
                step.status = "failed"
                step.error_message = result["error"]
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            step.status = "failed"
            step.error_message = str(e)
            step.execution_time_ms = execution_time
            step.completed_at = datetime.now(timezone.utc)
            
            return {"error": str(e)}
    
    async def _execute_task_step(self, step: WorkflowStep, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task step"""
        
        # Create task for execution
        task = Task(
            task_id=f"workflow_task_{uuid.uuid4().hex[:8]}",
            name=step.name,
            task_type=step.config.get("task_type", "general"),
            input_data=step_input,
            context=step.config
        )
        
        # Submit task to distributor
        success = await self.task_distributor.submit_task(task)
        
        if not success:
            return {"error": "Failed to submit task to distributor"}
        
        # Wait for completion (simplified - would use proper task monitoring)
        timeout = step.timeout_seconds
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            task_status = await self.task_distributor.get_task_status(task.task_id)
            
            if task_status:
                if task_status["status"] == "completed":
                    return task_status.get("result", {})
                elif task_status["status"] == "failed":
                    return {"error": task_status.get("error_message", "Task failed")}
            
            await asyncio.sleep(1)  # Poll interval
        
        return {"error": "Task execution timeout"}
    
    async def _execute_reasoning_step(self, step: WorkflowStep, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a reasoning chain step"""
        
        chain_id = step.config.get("chain_id")
        if not chain_id:
            return {"error": "No reasoning chain ID specified"}
        
        try:
            result = await self.reasoning_engine.execute_reasoning_chain(chain_id, step_input)
            return result.to_dict()
        except Exception as e:
            return {"error": f"Reasoning chain execution failed: {e}"}
    
    async def _execute_condition_step(self, step: WorkflowStep, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a conditional step"""
        
        condition = step.config.get("condition", "true")
        
        try:
            # Simple condition evaluation
            result = eval(condition, {"__builtins__": {}}, step_input)
            return {"condition_result": bool(result)}
        except Exception as e:
            return {"error": f"Condition evaluation failed: {e}"}
    
    async def _execute_transform_step(self, step: WorkflowStep, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data transformation step"""
        
        transformation = step.config.get("transformation", {})
        
        try:
            result = {}
            
            # Apply transformations
            for output_key, transform_config in transformation.items():
                if "source" in transform_config:
                    source_key = transform_config["source"]
                    source_value = step_input.get(source_key)
                    
                    # Apply transformation function
                    transform_func = transform_config.get("function", "identity")
                    
                    if transform_func == "identity":
                        result[output_key] = source_value
                    elif transform_func == "uppercase":
                        result[output_key] = str(source_value).upper()
                    elif transform_func == "lowercase":
                        result[output_key] = str(source_value).lower()
                    elif transform_func == "length":
                        result[output_key] = len(str(source_value))
                    else:
                        result[output_key] = source_value
            
            return result
            
        except Exception as e:
            return {"error": f"Transformation failed: {e}"}
    
    async def _execute_wait_step(self, step: WorkflowStep, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a wait step"""
        
        wait_seconds = step.config.get("wait_seconds", 1)
        
        try:
            await asyncio.sleep(wait_seconds)
            return {"waited_seconds": wait_seconds}
        except Exception as e:
            return {"error": f"Wait step failed: {e}"}
    
    def _prepare_step_input(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Prepare input data for step execution"""
        
        step_input = {}
        
        # Add execution input data
        step_input.update(execution.input_data)
        
        # Add execution context
        step_input.update(execution.execution_context)
        
        # Apply input mapping
        for context_key, input_key in step.input_mapping.items():
            if context_key in execution.execution_context:
                step_input[input_key] = execution.execution_context[context_key]
        
        # Add step configuration
        step_input.update(step.config.get("input_data", {}))
        
        return step_input
    
    def _resolve_step_dependencies(self, workflow: Workflow) -> List[str]:
        """Resolve step dependencies and return execution order"""
        
        # Topological sort
        in_degree = {}
        graph = {}
        
        # Build dependency graph
        for step in workflow.steps:
            step_id = step.step_id
            in_degree[step_id] = 0
            graph[step_id] = step.depends_on.copy()
        
        # Calculate in-degrees
        for step_id, dependencies in graph.items():
            for dep in dependencies:
                if dep in in_degree:
                    in_degree[step_id] += 1
        
        # Find steps with no dependencies
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            # Reduce in-degree for dependent steps
            for step_id, dependencies in graph.items():
                if current in dependencies:
                    in_degree[step_id] -= 1
                    if in_degree[step_id] == 0:
                        queue.append(step_id)
        
        # Check for circular dependencies
        if len(execution_order) != len(graph):
            logger.error("Circular dependencies detected in workflow")
            return []
        
        return execution_order
    
    def _group_steps_by_dependency_level(self, workflow: Workflow) -> List[List[str]]:
        """Group steps by dependency level for parallel execution"""
        
        levels = []
        remaining_steps = set(step.step_id for step in workflow.steps)
        completed_steps = set()
        
        while remaining_steps:
            current_level = []
            
            # Find steps that can be executed (all dependencies satisfied)
            for step_id in list(remaining_steps):
                step = workflow.get_step(step_id)
                if step and all(dep in completed_steps for dep in step.depends_on):
                    current_level.append(step_id)
            
            if not current_level:
                # No progress possible - circular dependency
                logger.error("Circular dependencies prevent parallel execution")
                break
            
            levels.append(current_level)
            
            # Remove current level steps from remaining
            for step_id in current_level:
                remaining_steps.remove(step_id)
                completed_steps.add(step_id)
        
        return levels
    
    def _compile_workflow_output(self, workflow: Workflow, execution: WorkflowExecution) -> Dict[str, Any]:
        """Compile final workflow output"""
        
        output = {}
        
        # Include all step results if no output schema defined
        if not workflow.output_schema:
            output["step_results"] = execution.step_results
            output["execution_context"] = execution.execution_context
        else:
            # Map outputs according to schema
            for output_key, mapping in workflow.output_schema.items():
                if "source" in mapping:
                    source = mapping["source"]
                    if source in execution.execution_context:
                        output[output_key] = execution.execution_context[source]
        
        # Add execution metadata
        output["execution_metadata"] = {
            "execution_id": execution.execution_id,
            "steps_completed": len(execution.steps_completed),
            "steps_failed": len(execution.steps_failed),
            "total_execution_time_ms": execution.total_execution_time_ms,
            "models_used": execution.models_used,
            "total_cost": execution.total_cost
        }
        
        return output
    
    def _update_avg_execution_time(self, execution_time_ms: float):
        """Update average execution time"""
        total_executions = self.stats["total_executions"]
        if total_executions > 0:
            current_avg = self.stats["avg_execution_time_ms"]
            self.stats["avg_execution_time_ms"] = \
                (current_avg * (total_executions - 1) + execution_time_ms) / total_executions
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            **self.stats,
            "active_workflows": len(self.active_executions),
            "max_concurrent_workflows": self.max_concurrent_workflows
        }


class WorkflowManager:
    """Main workflow management system"""
    
    def __init__(self, model_manager: ModelManager, task_distributor: TaskDistributor,
                 reasoning_engine: ReasoningEngine):
        self.model_manager = model_manager
        self.task_distributor = task_distributor
        self.reasoning_engine = reasoning_engine
        
        # Core components
        self.scheduler = WorkflowScheduler(task_distributor, reasoning_engine)
        
        # Workflow registry
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        
        # Templates and builders
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            "total_workflows": 0,
            "total_executions": 0,
            "active_executions": 0,
            "avg_workflow_complexity": 0.0
        }
        
        logger.info("Workflow Manager initialized")
    
    def create_workflow(self, name: str, description: str = "", 
                       execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> Workflow:
        """Create a new workflow"""
        
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            execution_mode=execution_mode
        )
        
        self.workflows[workflow_id] = workflow
        self.stats["total_workflows"] += 1
        
        logger.info(f"Created workflow: {name}")
        
        return workflow
    
    def add_workflow_step(self, workflow_id: str, step_name: str, step_type: StepType,
                         config: Optional[Dict[str, Any]] = None,
                         depends_on: Optional[List[str]] = None) -> Optional[WorkflowStep]:
        """Add a step to a workflow"""
        
        if workflow_id not in self.workflows:
            logger.error(f"Workflow not found: {workflow_id}")
            return None
        
        workflow = self.workflows[workflow_id]
        
        step_id = f"step_{uuid.uuid4().hex[:8]}"
        
        step = WorkflowStep(
            step_id=step_id,
            name=step_name,
            step_type=step_type,
            config=config or {},
            depends_on=depends_on or []
        )
        
        workflow.add_step(step)
        
        logger.info(f"Added step: {step_name} to workflow {workflow.name}")
        
        return step
    
    async def execute_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None,
                              triggered_by: Optional[str] = None) -> WorkflowExecution:
        """Execute a workflow"""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        
        # Create execution instance
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            workflow_name=workflow.name,
            input_data=input_data or {},
            triggered_by=triggered_by
        )
        
        self.executions[execution_id] = execution
        self.stats["total_executions"] += 1
        self.stats["active_executions"] += 1
        
        try:
            # Execute workflow
            result = await self.scheduler.execute_workflow(workflow, execution)
            
            # Update workflow statistics
            workflow.total_executions += 1
            if result.status == WorkflowStatus.COMPLETED:
                success_rate = ((workflow.success_rate * (workflow.total_executions - 1)) + 100) / workflow.total_executions
            else:
                success_rate = (workflow.success_rate * (workflow.total_executions - 1)) / workflow.total_executions
            
            workflow.success_rate = success_rate
            
            # Update average execution time
            workflow.avg_execution_time_ms = \
                ((workflow.avg_execution_time_ms * (workflow.total_executions - 1)) + 
                 result.total_execution_time_ms) / workflow.total_executions
            
            logger.info(f"Workflow execution completed: {workflow.name} (Status: {result.status.value})")
            
            return result
            
        finally:
            self.stats["active_executions"] -= 1
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution by ID"""
        return self.executions.get(execution_id)
    
    def list_workflows(self, status: Optional[WorkflowStatus] = None) -> List[Workflow]:
        """List workflows with optional filtering"""
        workflows = list(self.workflows.values())
        
        if status:
            workflows = [w for w in workflows if w.status == status]
        
        return workflows
    
    def list_executions(self, workflow_id: Optional[str] = None,
                       status: Optional[WorkflowStatus] = None) -> List[WorkflowExecution]:
        """List executions with optional filtering"""
        executions = list(self.executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        return executions
    
    def create_workflow_from_template(self, template_name: str, workflow_name: str,
                                    parameters: Optional[Dict[str, Any]] = None) -> Optional[Workflow]:
        """Create workflow from template"""
        
        if template_name not in self.workflow_templates:
            logger.error(f"Workflow template not found: {template_name}")
            return None
        
        template = self.workflow_templates[template_name]
        
        # Create workflow from template
        workflow = self.create_workflow(
            name=workflow_name,
            description=template.get("description", ""),
            execution_mode=ExecutionMode(template.get("execution_mode", "sequential"))
        )
        
        # Add steps from template
        for step_config in template.get("steps", []):
            step_type = StepType(step_config["step_type"])
            config = step_config.get("config", {})
            
            # Apply parameters
            if parameters:
                config = self._apply_template_parameters(config, parameters)
            
            self.add_workflow_step(
                workflow.workflow_id,
                step_config["name"],
                step_type,
                config,
                step_config.get("depends_on", [])
            )
        
        return workflow
    
    def _apply_template_parameters(self, config: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply template parameters to configuration"""
        
        # Simple parameter substitution (would be enhanced with proper templating)
        config_str = json.dumps(config)
        
        for param_name, param_value in parameters.items():
            placeholder = f"{{{param_name}}}"
            config_str = config_str.replace(placeholder, str(param_value))
        
        try:
            return json.loads(config_str)
        except json.JSONDecodeError:
            logger.warning("Failed to apply template parameters")
            return config
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        # Calculate average workflow complexity
        if self.workflows:
            total_steps = sum(len(w.steps) for w in self.workflows.values())
            avg_complexity = total_steps / len(self.workflows)
        else:
            avg_complexity = 0.0
        
        return {
            "manager_statistics": {
                **self.stats,
                "avg_workflow_complexity": avg_complexity
            },
            "scheduler_statistics": self.scheduler.get_scheduler_stats(),
            "workflow_breakdown": {
                status.value: len([w for w in self.workflows.values() if w.status == status])
                for status in WorkflowStatus
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def shutdown(self):
        """Graceful shutdown of workflow manager"""
        logger.info("Shutting down Workflow Manager")
        
        # Cancel active executions
        for execution_id, task in self.scheduler.active_executions.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.scheduler.active_executions.clear()
        
        logger.info("Workflow Manager shutdown complete")


# Export main classes
__all__ = [
    'WorkflowStatus',
    'StepType',
    'ExecutionMode',
    'WorkflowStep',
    'Workflow',
    'WorkflowExecution',
    'WorkflowScheduler',
    'WorkflowManager'
]