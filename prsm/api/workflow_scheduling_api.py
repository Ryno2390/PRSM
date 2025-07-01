"""
Workflow Scheduling API
======================

REST API endpoints for workflow scheduling, critical path analysis,
and execution optimization with dependency management.

Key Features:
- Workflow scheduling with critical path optimization
- Resource-aware scheduling and conflict resolution
- Cost optimization through strategic scheduling
- Real-time execution monitoring and adjustment
"""

import structlog
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field

from prsm.auth import get_current_user, require_auth
from prsm.scheduling.workflow_scheduler import (
    get_workflow_scheduler, ScheduledWorkflow, WorkflowStep, 
    ExecutionWindow, ResourceRequirement, ResourceType,
    SchedulingPriority, WorkflowStatus
)
from prsm.core.models import AgentType

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/scheduling", tags=["workflow-scheduling"])


class WorkflowStepRequest(BaseModel):
    """Request model for workflow step definition"""
    step_name: str = Field(..., description="Name of the workflow step")
    step_description: str = Field(..., description="Description of what this step does")
    agent_type: AgentType = Field(..., description="Type of agent to execute this step")
    prompt_template: str = Field(..., description="Prompt template for the step")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    depends_on: List[str] = Field(default_factory=list, description="List of step IDs this depends on")
    blocks: List[str] = Field(default_factory=list, description="List of step IDs this blocks")
    estimated_duration_minutes: int = Field(default=5, description="Estimated duration in minutes")
    resource_requirements: List[Dict[str, Any]] = Field(default_factory=list, description="Resource requirements")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout_minutes: Optional[int] = Field(None, description="Step timeout in minutes")


class ExecutionWindowRequest(BaseModel):
    """Request model for execution window specification"""
    earliest_start_time: datetime = Field(..., description="Earliest allowed start time")
    latest_start_time: datetime = Field(..., description="Latest allowed start time")
    preferred_start_time: Optional[datetime] = Field(None, description="Preferred start time")
    max_duration_hours: float = Field(..., description="Maximum duration in hours")
    allow_split_execution: bool = Field(default=False, description="Allow splitting execution")
    allow_preemption: bool = Field(default=False, description="Allow preemption")


class WorkflowSchedulingRequest(BaseModel):
    """Request model for workflow scheduling"""
    workflow_name: str = Field(..., description="Name of the workflow")
    description: str = Field(..., description="Description of the workflow")
    steps: List[WorkflowStepRequest] = Field(..., description="List of workflow steps")
    execution_window: ExecutionWindowRequest = Field(..., description="Execution time window")
    scheduling_priority: SchedulingPriority = Field(default=SchedulingPriority.NORMAL, description="Scheduling priority")
    max_ftns_cost: Optional[float] = Field(None, description="Maximum FTNS cost budget")
    cost_optimization_enabled: bool = Field(default=True, description="Enable cost optimization")
    preemption_allowed: bool = Field(default=False, description="Allow preemption for higher priority workflows")
    tags: List[str] = Field(default_factory=list, description="Workflow tags")


class CriticalPathOptimizationRequest(BaseModel):
    """Request model for critical path optimization"""
    workflow_id: str = Field(..., description="ID of workflow to optimize")
    optimization_goals: Dict[str, float] = Field(
        default_factory=lambda: {"minimize_duration": 0.4, "minimize_cost": 0.4, "balance_resources": 0.2},
        description="Optimization goal weights"
    )
    apply_optimizations: bool = Field(default=False, description="Apply optimizations to workflow")


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status"""
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    scheduled_start: Optional[datetime]
    actual_start: Optional[datetime]
    actual_end: Optional[datetime]
    execution_attempts: int
    critical_path_duration_seconds: float
    estimated_cost: float
    actual_cost: Optional[float]
    cost_savings: Optional[float]


@router.post("/schedule-workflow")
async def schedule_workflow(
    request: WorkflowSchedulingRequest,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Schedule a workflow for execution with critical path optimization
    
    🚀 FEATURES:
    - Dependency-aware scheduling with critical path calculation
    - Resource conflict detection and resolution
    - Cost optimization through strategic timing
    - Parallel execution opportunity analysis
    """
    try:
        scheduler = get_workflow_scheduler()
        
        # Convert request to internal format
        workflow_steps = []
        step_id_mapping = {}  # Map string IDs to UUIDs
        
        # First pass: create all steps and build ID mapping
        for step_req in request.steps:
            step_id = uuid4()
            step_id_mapping[step_req.step_name] = step_id
            
            # Convert resource requirements
            resource_reqs = []
            for req_data in step_req.resource_requirements:
                resource_req = ResourceRequirement(
                    resource_type=ResourceType(req_data.get("resource_type", "cpu_cores")),
                    amount=req_data.get("amount", 1.0),
                    min_amount=req_data.get("min_amount"),
                    max_amount=req_data.get("max_amount"),
                    is_burst_capable=req_data.get("is_burst_capable", False),
                    priority_multiplier=req_data.get("priority_multiplier", 1.0)
                )
                resource_reqs.append(resource_req)
            
            step = WorkflowStep(
                step_id=step_id,
                step_name=step_req.step_name,
                step_description=step_req.step_description,
                agent_type=step_req.agent_type,
                prompt_template=step_req.prompt_template,
                parameters=step_req.parameters,
                depends_on=[],  # Will be filled in second pass
                blocks=[],      # Will be filled in second pass
                resource_requirements=resource_reqs,
                estimated_duration=timedelta(minutes=step_req.estimated_duration_minutes),
                max_retries=step_req.max_retries,
                retry_delay=timedelta(minutes=1),
                timeout=timedelta(minutes=step_req.timeout_minutes) if step_req.timeout_minutes else None
            )
            workflow_steps.append(step)
        
        # Second pass: resolve dependencies
        for i, step_req in enumerate(request.steps):
            step = workflow_steps[i]
            
            # Resolve depends_on
            for dep_name in step_req.depends_on:
                if dep_name in step_id_mapping:
                    step.depends_on.append(step_id_mapping[dep_name])
            
            # Resolve blocks
            for block_name in step_req.blocks:
                if block_name in step_id_mapping:
                    step.blocks.append(step_id_mapping[block_name])
        
        # Create execution window
        execution_window = ExecutionWindow(
            earliest_start=request.execution_window.earliest_start_time,
            latest_start=request.execution_window.latest_start_time,
            preferred_start=request.execution_window.preferred_start_time,
            max_duration=timedelta(hours=request.execution_window.max_duration_hours),
            allow_split_execution=request.execution_window.allow_split_execution,
            allow_preemption=request.execution_window.allow_preemption
        )
        
        # Create scheduled workflow
        workflow = ScheduledWorkflow(
            workflow_id=uuid4(),
            user_id=current_user,
            workflow_name=request.workflow_name,
            description=request.description,
            steps=workflow_steps,
            scheduling_priority=request.scheduling_priority,
            execution_window=execution_window,
            max_ftns_cost=request.max_ftns_cost,
            cost_optimization_enabled=request.cost_optimization_enabled,
            preemption_allowed=request.preemption_allowed,
            created_by=current_user,
            tags=request.tags
        )
        
        # Schedule the workflow
        scheduling_result = await scheduler.schedule_workflow(workflow)
        
        if scheduling_result["success"]:
            # Perform critical path optimization if enabled
            if request.cost_optimization_enabled:
                optimization_result = await scheduler.optimize_workflow_with_critical_path(
                    workflow,
                    {"apply_optimizations": True}
                )
                scheduling_result["optimization_analysis"] = optimization_result
        
        logger.info("Workflow scheduled successfully",
                   workflow_id=str(workflow.workflow_id),
                   user_id=current_user,
                   step_count=len(workflow_steps))
        
        return scheduling_result
        
    except Exception as e:
        logger.error("Failed to schedule workflow",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to schedule workflow: {str(e)}"
        )


@router.post("/optimize-critical-path")
async def optimize_critical_path(
    request: CriticalPathOptimizationRequest,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Perform advanced critical path analysis and optimization
    
    🧠 ANALYTICS:
    - Identifies critical path through workflow dependencies
    - Finds parallelization opportunities for time reduction
    - Detects resource conflicts and suggests resolutions
    - Provides actionable optimization recommendations
    """
    try:
        scheduler = get_workflow_scheduler()
        
        # Get the workflow
        try:
            workflow_id = UUID(request.workflow_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid workflow ID format"
            )
        
        workflow = scheduler.scheduled_workflows.get(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail="Workflow not found"
            )
        
        # Check ownership
        if workflow.user_id != current_user:
            raise HTTPException(
                status_code=403,
                detail="Access denied: workflow belongs to another user"
            )
        
        # Perform critical path optimization
        optimization_result = await scheduler.optimize_workflow_with_critical_path(
            workflow,
            {
                **request.optimization_goals,
                "apply_optimizations": request.apply_optimizations
            }
        )
        
        logger.info("Critical path optimization completed",
                   workflow_id=request.workflow_id,
                   user_id=current_user)
        
        return optimization_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to optimize critical path",
                    workflow_id=request.workflow_id,
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize critical path: {str(e)}"
        )


@router.get("/workflow/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    current_user: str = Depends(get_current_user)
) -> WorkflowStatusResponse:
    """
    Get comprehensive status of a scheduled workflow
    
    📊 STATUS:
    - Current execution status and progress
    - Critical path metrics and timing
    - Cost tracking and optimization savings
    - Resource utilization and conflicts
    """
    try:
        scheduler = get_workflow_scheduler()
        
        # Parse workflow ID
        try:
            workflow_uuid = UUID(workflow_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid workflow ID format"
            )
        
        workflow = scheduler.scheduled_workflows.get(workflow_uuid)
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail="Workflow not found"
            )
        
        # Check ownership
        if workflow.user_id != current_user:
            raise HTTPException(
                status_code=403,
                detail="Access denied: workflow belongs to another user"
            )
        
        # Calculate critical path duration
        critical_path_duration = workflow.get_critical_path_duration()
        
        # Estimate current cost
        current_rates = scheduler._get_current_ftns_rates()
        estimated_cost = workflow.calculate_total_estimated_cost(current_rates)
        
        return WorkflowStatusResponse(
            workflow_id=str(workflow.workflow_id),
            workflow_name=workflow.workflow_name,
            status=workflow.status,
            scheduled_start=workflow.scheduled_start,
            actual_start=workflow.actual_start,
            actual_end=workflow.actual_end,
            execution_attempts=workflow.execution_attempts,
            critical_path_duration_seconds=critical_path_duration.total_seconds(),
            estimated_cost=estimated_cost,
            actual_cost=workflow.actual_ftns_cost,
            cost_savings=workflow.cost_savings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow status",
                    workflow_id=workflow_id,
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow status: {str(e)}"
        )


@router.get("/workflows")
async def list_user_workflows(
    current_user: str = Depends(get_current_user),
    status_filter: Optional[WorkflowStatus] = Query(None, description="Filter by workflow status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of workflows to return"),
    offset: int = Query(0, ge=0, description="Number of workflows to skip")
) -> Dict[str, Any]:
    """
    List workflows owned by the current user
    
    📋 LISTING:
    - User's scheduled and executed workflows
    - Filtering by status and time range
    - Pagination support for large lists
    - Summary statistics and insights
    """
    try:
        scheduler = get_workflow_scheduler()
        
        # Filter workflows by user
        user_workflows = [
            workflow for workflow in scheduler.scheduled_workflows.values()
            if workflow.user_id == current_user
        ]
        
        # Apply status filter
        if status_filter:
            user_workflows = [
                workflow for workflow in user_workflows
                if workflow.status == status_filter
            ]
        
        # Sort by creation time (newest first)
        user_workflows.sort(key=lambda w: w.created_at, reverse=True)
        
        # Apply pagination
        total_count = len(user_workflows)
        paginated_workflows = user_workflows[offset:offset + limit]
        
        # Build response
        workflows_data = []
        for workflow in paginated_workflows:
            critical_path_duration = workflow.get_critical_path_duration()
            current_rates = scheduler._get_current_ftns_rates()
            estimated_cost = workflow.calculate_total_estimated_cost(current_rates)
            
            workflows_data.append({
                "workflow_id": str(workflow.workflow_id),
                "workflow_name": workflow.workflow_name,
                "description": workflow.description,
                "status": workflow.status,
                "scheduling_priority": workflow.scheduling_priority,
                "created_at": workflow.created_at.isoformat(),
                "scheduled_start": workflow.scheduled_start.isoformat() if workflow.scheduled_start else None,
                "step_count": len(workflow.steps),
                "critical_path_duration_seconds": critical_path_duration.total_seconds(),
                "estimated_cost": estimated_cost,
                "actual_cost": workflow.actual_ftns_cost,
                "cost_savings": workflow.cost_savings,
                "tags": workflow.tags
            })
        
        # Calculate summary statistics
        active_workflows = sum(1 for w in user_workflows if w.status in [WorkflowStatus.SCHEDULED, WorkflowStatus.RUNNING])
        completed_workflows = sum(1 for w in user_workflows if w.status == WorkflowStatus.COMPLETED)
        total_cost_savings = sum(w.cost_savings or 0 for w in user_workflows)
        
        return {
            "success": True,
            "workflows": workflows_data,
            "pagination": {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            },
            "summary": {
                "total_workflows": total_count,
                "active_workflows": active_workflows,
                "completed_workflows": completed_workflows,
                "total_cost_savings": total_cost_savings
            }
        }
        
    except Exception as e:
        logger.error("Failed to list user workflows",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list workflows: {str(e)}"
        )


@router.post("/workflow/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Execute a scheduled workflow immediately
    
    ⚡ EXECUTION:
    - Immediate execution override for urgent workflows
    - Real-time resource allocation and monitoring
    - Critical path optimization during execution
    - Comprehensive execution tracking and reporting
    """
    try:
        scheduler = get_workflow_scheduler()
        
        # Parse workflow ID
        try:
            workflow_uuid = UUID(workflow_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid workflow ID format"
            )
        
        workflow = scheduler.scheduled_workflows.get(workflow_uuid)
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail="Workflow not found"
            )
        
        # Check ownership
        if workflow.user_id != current_user:
            raise HTTPException(
                status_code=403,
                detail="Access denied: workflow belongs to another user"
            )
        
        # Check if workflow can be executed
        if workflow.status not in [WorkflowStatus.SCHEDULED, WorkflowStatus.FAILED]:
            raise HTTPException(
                status_code=400,
                detail=f"Workflow cannot be executed in {workflow.status} state"
            )
        
        # Execute the workflow
        execution_result = await scheduler.execute_scheduled_workflow(workflow_uuid)
        
        logger.info("Workflow execution requested",
                   workflow_id=workflow_id,
                   user_id=current_user,
                   success=execution_result["success"])
        
        return execution_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to execute workflow",
                    workflow_id=workflow_id,
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute workflow: {str(e)}"
        )


@router.get("/system/statistics")
async def get_scheduling_statistics(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive scheduling system statistics
    
    📈 ANALYTICS:
    - System-wide scheduling performance metrics
    - Resource utilization and optimization insights
    - Cost savings and efficiency improvements
    - Queue status and capacity planning data
    """
    try:
        scheduler = get_workflow_scheduler()
        
        # Get scheduling statistics
        stats = scheduler.get_scheduling_statistics()
        
        # Add current timestamp
        stats["timestamp"] = datetime.now(timezone.utc).isoformat()
        stats["user_id"] = current_user
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error("Failed to get scheduling statistics",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get scheduling statistics: {str(e)}"
        )