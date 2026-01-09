"""
Workflow Scheduling Engine

💰 FTNS SCHEDULING & MARKETPLACE SYSTEM:
- Time-based workflow execution with economic optimization
- Queue management with priority levels and resource allocation
- Cost-aware scheduling for minimizing FTNS spend during peak times
- Dependency tracking and multi-step workflow coordination
- Integration with dynamic pricing for optimal execution timing

This module implements intelligent workflow scheduling that enables users to:
1. Schedule prompts/workflows for off-peak execution to minimize costs
2. Optimize resource utilization across time periods
3. Balance system load through economic incentives
4. Enable FTNS trading opportunities for enterprising users
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from uuid import UUID, uuid4
from collections import defaultdict, deque
import heapq
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import (
    PRSMBaseModel, TimestampMixin, AgentType, TaskStatus, SafetyLevel
)
from prsm.data.context.selective_parallelism_engine import (
    ExecutionStrategy, TaskDefinition, ParallelismDecision, SelectiveParallelismEngine, TaskComplexity
)
from prsm.compute.scheduling.critical_path_calculator import (
    CriticalPathCalculator, ResourceConstraint
)

logger = structlog.get_logger(__name__)


class SchedulingPriority(str, Enum):
    """Scheduling priority levels"""
    IMMEDIATE = "immediate"         # Execute immediately, highest cost
    HIGH = "high"                  # Execute within 1 hour
    NORMAL = "normal"              # Execute within 6 hours  
    LOW = "low"                    # Execute within 24 hours
    BACKGROUND = "background"      # Execute when most cost-effective
    FLEXIBLE = "flexible"          # No time constraints, optimize for cost


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    SCHEDULED = "scheduled"        # Waiting for execution time
    QUEUED = "queued"             # In execution queue
    RUNNING = "running"           # Currently executing
    COMPLETED = "completed"       # Successfully completed
    FAILED = "failed"             # Failed execution
    CANCELLED = "cancelled"       # Cancelled by user
    PAUSED = "paused"            # Execution paused
    RETRYING = "retrying"        # Retrying after failure


class ResourceType(str, Enum):
    """Types of computational resources"""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    GPU_UNITS = "gpu_units"
    STORAGE_GB = "storage_gb"
    NETWORK_MBPS = "network_mbps"
    MODEL_INFERENCE_TOKENS = "model_inference_tokens"
    FTNS_CREDITS = "ftns_credits"


class ExecutionWindow(PRSMBaseModel):
    """Time window for workflow execution"""
    earliest_start: datetime
    latest_start: datetime
    preferred_start: Optional[datetime] = None
    max_duration: timedelta
    
    # Flexibility settings
    allow_split_execution: bool = Field(default=False)
    allow_preemption: bool = Field(default=False)
    
    def is_valid_time(self, check_time: datetime) -> bool:
        """Check if time falls within execution window"""
        return self.earliest_start <= check_time <= self.latest_start
    
    def get_flexibility_score(self) -> float:
        """Calculate flexibility score (0.0-1.0) based on window size"""
        window_duration = self.latest_start - self.earliest_start
        if window_duration.total_seconds() == 0:
            return 0.0
        
        # More flexible if window is larger
        max_flexibility_hours = 168  # 1 week
        flexibility = min(1.0, window_duration.total_seconds() / (max_flexibility_hours * 3600))
        return flexibility


class ResourceRequirement(PRSMBaseModel):
    """Resource requirement specification"""
    resource_type: ResourceType
    amount: float
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    
    # Cost considerations
    is_burst_capable: bool = Field(default=False)  # Can temporarily exceed amount
    priority_multiplier: float = Field(ge=0.1, le=10.0, default=1.0)
    
    def get_effective_amount(self, availability_factor: float = 1.0) -> float:
        """Get effective resource amount considering availability"""
        if self.min_amount is not None and availability_factor < 1.0:
            return max(self.min_amount, self.amount * availability_factor)
        return self.amount


class WorkflowStep(PRSMBaseModel):
    """Individual step in a workflow"""
    step_id: UUID = Field(default_factory=uuid4)
    step_name: str
    step_description: str
    
    # Execution parameters
    agent_type: AgentType
    prompt_template: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Dependencies
    depends_on: List[UUID] = Field(default_factory=list)
    blocks: List[UUID] = Field(default_factory=list)
    
    # Resource requirements
    resource_requirements: List[ResourceRequirement] = Field(default_factory=list)
    estimated_duration: timedelta = Field(default=timedelta(minutes=5))
    
    # Retry and error handling
    max_retries: int = Field(default=3)
    retry_delay: timedelta = Field(default=timedelta(minutes=1))
    timeout: Optional[timedelta] = None
    
    # Output handling
    output_format: str = Field(default="json")
    store_intermediate_results: bool = Field(default=True)
    
    def calculate_resource_cost(self, ftns_rates: Dict[ResourceType, float]) -> float:
        """Calculate FTNS cost for this step"""
        total_cost = 0.0
        for req in self.resource_requirements:
            rate = ftns_rates.get(req.resource_type, 0.0)
            cost = req.get_effective_amount() * rate * req.priority_multiplier
            total_cost += cost
        return total_cost


class ScheduledWorkflow(PRSMBaseModel):
    """Complete scheduled workflow definition"""
    workflow_id: UUID = Field(default_factory=uuid4)
    user_id: str
    workflow_name: str
    description: str
    
    # Workflow composition
    steps: List[WorkflowStep] = Field(default_factory=list)
    execution_strategy: ExecutionStrategy = Field(default=ExecutionStrategy.SEQUENTIAL)
    
    # Scheduling parameters
    scheduling_priority: SchedulingPriority = Field(default=SchedulingPriority.NORMAL)
    execution_window: ExecutionWindow
    
    # Resource and cost management
    max_ftns_cost: Optional[float] = None
    cost_optimization_enabled: bool = Field(default=True)
    preemption_allowed: bool = Field(default=False)
    
    # Workflow metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str
    tags: List[str] = Field(default_factory=list)
    
    # Execution tracking
    status: WorkflowStatus = Field(default=WorkflowStatus.SCHEDULED)
    scheduled_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    execution_attempts: int = Field(default=0)
    
    # Results and performance
    execution_results: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    actual_ftns_cost: Optional[float] = None
    cost_savings: Optional[float] = None
    
    def calculate_total_estimated_cost(self, ftns_rates: Dict[ResourceType, float]) -> float:
        """Calculate total estimated FTNS cost"""
        return sum(step.calculate_resource_cost(ftns_rates) for step in self.steps)
    
    def get_critical_path_duration(self) -> timedelta:
        """Calculate critical path duration through workflow steps"""
        if not self.steps:
            return timedelta()
        
        # Use the critical path calculator for accurate dependency-aware calculation
        try:
            # Create a temporary calculator instance for this workflow
            from prsm.compute.scheduling.critical_path_calculator import CriticalPathCalculator
            calculator = CriticalPathCalculator()
            
            # Run critical path analysis synchronously (simplified for this context)
            import asyncio
            
            async def calculate_critical_path():
                result = await calculator.calculate_critical_path(self.steps)
                return result.critical_path_duration
            
            # Run in event loop if available, otherwise create one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, we can't await
                    # Fall back to simplified calculation
                    return self._calculate_simplified_critical_path()
                else:
                    return loop.run_until_complete(calculate_critical_path())
            except RuntimeError:
                # No event loop available, create one
                return asyncio.run(calculate_critical_path())
                
        except Exception as e:
            # Fallback to simplified calculation if critical path analysis fails
            logger.warning("Critical path calculation failed, using fallback", error=str(e))
            return self._calculate_simplified_critical_path()
    
    def _calculate_simplified_critical_path(self) -> timedelta:
        """Simplified critical path calculation for fallback scenarios"""
        if not self.steps:
            return timedelta()
        
        # Build a simple dependency graph
        step_map = {step.step_id: step for step in self.steps}
        
        # Find steps with no dependencies (start points)
        start_steps = [step for step in self.steps if not step.depends_on]
        
        if not start_steps:
            # If no clear start point, assume sequential execution
            return sum((step.estimated_duration for step in self.steps), timedelta())
        
        # Calculate longest path from each start step
        def calculate_longest_path(step_id: UUID, visited: set) -> timedelta:
            if step_id in visited:
                return timedelta()  # Cycle detection
            
            visited.add(step_id)
            step = step_map.get(step_id)
            if not step:
                return timedelta()
            
            # Find all steps that depend on this one
            dependent_steps = [s for s in self.steps if step_id in s.depends_on]
            
            if not dependent_steps:
                # Terminal step
                return step.estimated_duration
            
            # Calculate maximum path through dependents
            max_dependent_path = max(
                (calculate_longest_path(dep_step.step_id, visited.copy()) 
                 for dep_step in dependent_steps),
                default=timedelta()
            )
            
            return step.estimated_duration + max_dependent_path
        
        # Return the longest path from any start step
        return max(
            (calculate_longest_path(step.step_id, set()) for step in start_steps),
            default=timedelta()
        )
    
    def is_deadline_feasible(self) -> bool:
        """Check if workflow can complete within execution window"""
        critical_path = self.get_critical_path_duration()
        window_duration = self.execution_window.latest_start - self.execution_window.earliest_start
        return critical_path <= window_duration


@dataclass
class SchedulingEvent:
    """Event in the scheduling system"""
    event_time: datetime
    event_type: str  # "workflow_start", "workflow_end", "price_update", etc.
    workflow_id: UUID
    priority: int = 0
    
    def __lt__(self, other):
        # For heapq - earlier times and higher priorities first
        if self.event_time != other.event_time:
            return self.event_time < other.event_time
        return self.priority > other.priority


class ResourcePool(PRSMBaseModel):
    """Pool of available computational resources"""
    pool_id: UUID = Field(default_factory=uuid4)
    resource_type: ResourceType
    
    # Capacity management
    total_capacity: float
    available_capacity: float
    reserved_capacity: float = Field(default=0.0)
    
    # Pricing and economics
    base_cost_per_unit: float
    current_cost_per_unit: float
    demand_multiplier: float = Field(default=1.0)
    
    # Performance tracking
    utilization_history: List[Tuple[datetime, float]] = Field(default_factory=list)
    peak_demand_times: List[datetime] = Field(default_factory=list)
    
    # Configuration
    oversubscription_ratio: float = Field(default=1.2)  # Allow 120% allocation
    min_reserve_percentage: float = Field(default=0.1)  # Keep 10% in reserve
    
    def get_utilization_percentage(self) -> float:
        """Get current utilization percentage"""
        used = self.total_capacity - self.available_capacity
        return (used / self.total_capacity) * 100 if self.total_capacity > 0 else 0
    
    def can_allocate(self, amount: float) -> bool:
        """Check if amount can be allocated"""
        max_allocatable = self.total_capacity * self.oversubscription_ratio
        current_allocated = self.total_capacity - self.available_capacity
        return (current_allocated + amount) <= max_allocatable
    
    def allocate_resources(self, amount: float) -> bool:
        """Allocate resources if available"""
        if self.can_allocate(amount):
            self.available_capacity -= amount
            return True
        return False
    
    def deallocate_resources(self, amount: float):
        """Deallocate resources back to pool"""
        self.available_capacity = min(
            self.total_capacity, 
            self.available_capacity + amount
        )
    
    def update_pricing(self):
        """Update pricing based on current demand"""
        utilization = self.get_utilization_percentage()
        
        # Dynamic pricing based on utilization
        if utilization > 90:
            self.demand_multiplier = 3.0  # High demand premium
        elif utilization > 75:
            self.demand_multiplier = 2.0  # Moderate premium
        elif utilization > 50:
            self.demand_multiplier = 1.5  # Light premium
        elif utilization < 25:
            self.demand_multiplier = 0.7  # Off-peak discount
        else:
            self.demand_multiplier = 1.0  # Base pricing
        
        self.current_cost_per_unit = self.base_cost_per_unit * self.demand_multiplier


class WorkflowScheduler(TimestampMixin):
    """
    Workflow Scheduling Engine
    
    Intelligent scheduling system for cost optimization and load balancing
    through time-based workflow execution and resource management.
    """
    
    # Performance tracking fields
    scheduling_statistics: Dict[str, Any] = Field(default_factory=lambda: defaultdict(int), description="Scheduling statistics")
    cost_optimization_savings: float = Field(default=0.0, description="Cost optimization savings")
    peak_load_periods: List[Tuple[datetime, float]] = Field(default_factory=list, description="Peak load periods")
    
    # Configuration fields
    max_concurrent_workflows: int = Field(default=50, description="Maximum concurrent workflows")
    scheduling_interval: timedelta = Field(default_factory=lambda: timedelta(minutes=1), description="Scheduling interval")
    cost_optimization_threshold: float = Field(default=0.8, description="Cost optimization threshold")
    default_workflow_timeout: timedelta = Field(default_factory=lambda: timedelta(hours=24), description="Default workflow timeout")
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Initialize non-serializable components after Pydantic initialization
        # Scheduling infrastructure
        self._scheduled_workflows: Dict[UUID, ScheduledWorkflow] = {}
        self._execution_queue: deque = deque()
        self._event_queue: List[SchedulingEvent] = []  # Min-heap for time-based events
        
        # Resource management
        self._resource_pools: Dict[ResourceType, ResourcePool] = {}
        self._active_allocations: Dict[UUID, Dict[ResourceType, float]] = defaultdict(dict)
        
        # Scheduling algorithms
        self._parallelism_engine: Optional[SelectiveParallelismEngine] = None
        self._critical_path_calculator = CriticalPathCalculator()
        
        self._initialize_resource_pools()
        self._start_scheduling_loop()
        
        logger.info("WorkflowScheduler initialized")
    
    def _initialize_resource_pools(self):
        """Initialize default resource pools"""
        default_pools = {
            ResourceType.CPU_CORES: ResourcePool(
                resource_type=ResourceType.CPU_CORES,
                total_capacity=64.0,
                available_capacity=64.0,
                base_cost_per_unit=0.10,  # $0.10 per core-hour
                current_cost_per_unit=0.10
            ),
            ResourceType.MEMORY_GB: ResourcePool(
                resource_type=ResourceType.MEMORY_GB,
                total_capacity=512.0,
                available_capacity=512.0,
                base_cost_per_unit=0.05,  # $0.05 per GB-hour
                current_cost_per_unit=0.05
            ),
            ResourceType.GPU_UNITS: ResourcePool(
                resource_type=ResourceType.GPU_UNITS,
                total_capacity=8.0,
                available_capacity=8.0,
                base_cost_per_unit=2.50,  # $2.50 per GPU-hour
                current_cost_per_unit=2.50
            ),
            ResourceType.FTNS_CREDITS: ResourcePool(
                resource_type=ResourceType.FTNS_CREDITS,
                total_capacity=10000.0,
                available_capacity=10000.0,
                base_cost_per_unit=1.00,  # $1.00 per FTNS credit
                current_cost_per_unit=1.00
            )
        }
        
        for resource_type, pool in default_pools.items():
            self._resource_pools[resource_type] = pool
    
    def set_parallelism_engine(self, engine: SelectiveParallelismEngine):
        """Set selective parallelism engine for integration"""
        self._parallelism_engine = engine
        logger.info("Parallelism engine integrated with scheduler")
    
    def _start_scheduling_loop(self):
        """Start the main scheduling event loop"""
        # In a real implementation, this would run as a background task
        # For now, we'll simulate the scheduling process
        pass
    
    async def schedule_workflow(
        self,
        workflow: ScheduledWorkflow,
        optimization_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Schedule a workflow for execution
        
        Args:
            workflow: Workflow definition to schedule
            optimization_preferences: User preferences for optimization
            
        Returns:
            Scheduling result with cost estimates and execution plan
        """
        try:
            # Validate workflow
            if not workflow.is_deadline_feasible():
                return {
                    "success": False,
                    "error": "Workflow cannot complete within specified execution window",
                    "critical_path_duration": workflow.get_critical_path_duration().total_seconds(),
                    "available_window": (workflow.execution_window.latest_start - 
                                       workflow.execution_window.earliest_start).total_seconds()
                }
            
            # Calculate cost estimates
            current_rates = self._get_current_ftns_rates()
            immediate_cost = workflow.calculate_total_estimated_cost(current_rates)
            
            # Find optimal execution time
            optimal_schedule = await self._find_optimal_execution_time(
                workflow, optimization_preferences
            )
            
            # Reserve resources if needed
            if optimal_schedule["feasible"]:
                workflow.scheduled_start = optimal_schedule["recommended_start_time"]
                workflow.status = WorkflowStatus.SCHEDULED
                
                # Store workflow
                self._scheduled_workflows[workflow.workflow_id] = workflow
                
                # Add to event queue
                schedule_event = SchedulingEvent(
                    event_time=workflow.scheduled_start,
                    event_type="workflow_start",
                    workflow_id=workflow.workflow_id,
                    priority=self._get_priority_value(workflow.scheduling_priority)
                )
                heapq.heappush(self.event_queue, schedule_event)
                
                # Update statistics
                self.scheduling_statistics["workflows_scheduled"] += 1
                self.scheduling_statistics[f"priority_{workflow.scheduling_priority}"] += 1
                
                logger.info(
                    "Workflow scheduled successfully",
                    workflow_id=str(workflow.workflow_id),
                    scheduled_start=workflow.scheduled_start.isoformat(),
                    estimated_cost=optimal_schedule["estimated_cost"],
                    cost_savings=optimal_schedule.get("cost_savings", 0)
                )
                
                return {
                    "success": True,
                    "workflow_id": str(workflow.workflow_id),
                    "scheduled_start_time": workflow.scheduled_start.isoformat(),
                    "estimated_cost": optimal_schedule["estimated_cost"],
                    "immediate_cost": immediate_cost,
                    "cost_savings": optimal_schedule.get("cost_savings", 0),
                    "cost_savings_percentage": optimal_schedule.get("cost_savings_percentage", 0),
                    "execution_plan": optimal_schedule["execution_plan"]
                }
            else:
                return {
                    "success": False,
                    "error": "No feasible execution time found",
                    "resource_constraints": optimal_schedule.get("constraints", []),
                    "recommendations": optimal_schedule.get("recommendations", [])
                }
                
        except Exception as e:
            logger.error("Error scheduling workflow", error=str(e))
            return {"success": False, "error": str(e)}
    
    def _get_current_ftns_rates(self) -> Dict[ResourceType, float]:
        """Get current FTNS rates for all resource types"""
        rates = {}
        for resource_type, pool in self._resource_pools.items():
            pool.update_pricing()
            rates[resource_type] = pool.current_cost_per_unit
        return rates
    
    async def _find_optimal_execution_time(
        self,
        workflow: ScheduledWorkflow,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Find optimal execution time considering cost and resource availability
        
        Args:
            workflow: Workflow to schedule
            preferences: User optimization preferences
            
        Returns:
            Optimal scheduling recommendation
        """
        try:
            preferences = preferences or {}
            cost_weight = preferences.get("cost_weight", 0.7)
            time_weight = preferences.get("time_weight", 0.3)
            
            # Sample time slots within execution window
            start_time = workflow.execution_window.earliest_start
            end_time = workflow.execution_window.latest_start
            critical_path = workflow.get_critical_path_duration()
            
            # Generate candidate time slots (every 30 minutes)
            slot_interval = timedelta(minutes=30)
            current_time = start_time
            candidate_slots = []
            
            while current_time <= end_time:
                if current_time + critical_path <= workflow.execution_window.latest_start + critical_path:
                    candidate_slots.append(current_time)
                current_time += slot_interval
            
            if not candidate_slots:
                return {
                    "feasible": False,
                    "error": "No valid time slots found",
                    "recommendations": ["Extend execution window", "Reduce workflow duration"]
                }
            
            # Evaluate each candidate slot
            best_slot = None
            best_score = float('-inf')
            evaluations = []
            
            for slot_time in candidate_slots:
                evaluation = await self._evaluate_time_slot(workflow, slot_time)
                
                # Calculate composite score
                cost_score = 1.0 - (evaluation["estimated_cost"] / evaluation.get("max_possible_cost", evaluation["estimated_cost"]))
                resource_score = evaluation["resource_availability_score"]
                time_preference_score = self._calculate_time_preference_score(
                    slot_time, workflow.execution_window.preferred_start
                )
                
                composite_score = (
                    cost_weight * cost_score +
                    0.2 * resource_score +
                    time_weight * time_preference_score
                )
                
                evaluation["composite_score"] = composite_score
                evaluation["slot_time"] = slot_time
                evaluations.append(evaluation)
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_slot = evaluation
            
            if best_slot:
                # Calculate savings compared to immediate execution
                immediate_rates = self._get_current_ftns_rates()
                immediate_cost = workflow.calculate_total_estimated_cost(immediate_rates)
                cost_savings = immediate_cost - best_slot["estimated_cost"]
                cost_savings_percentage = (cost_savings / immediate_cost) * 100 if immediate_cost > 0 else 0
                
                return {
                    "feasible": True,
                    "recommended_start_time": best_slot["slot_time"],
                    "estimated_cost": best_slot["estimated_cost"],
                    "cost_savings": max(0, cost_savings),
                    "cost_savings_percentage": max(0, cost_savings_percentage),
                    "resource_availability_score": best_slot["resource_availability_score"],
                    "execution_plan": best_slot["execution_plan"],
                    "all_evaluations": evaluations
                }
            else:
                return {
                    "feasible": False,
                    "error": "No suitable execution time found",
                    "evaluations": evaluations
                }
                
        except Exception as e:
            logger.error("Error finding optimal execution time", error=str(e))
            return {"feasible": False, "error": str(e)}
    
    async def _evaluate_time_slot(
        self,
        workflow: ScheduledWorkflow,
        slot_time: datetime
    ) -> Dict[str, Any]:
        """
        Evaluate a specific time slot for workflow execution
        
        Args:
            workflow: Workflow to evaluate
            slot_time: Candidate execution start time
            
        Returns:
            Evaluation metrics for the time slot
        """
        try:
            # Project resource demand and pricing at this time
            projected_rates = self._project_ftns_rates_at_time(slot_time)
            estimated_cost = workflow.calculate_total_estimated_cost(projected_rates)
            
            # Check resource availability
            resource_availability = {}
            total_availability_score = 0.0
            
            for step in workflow.steps:
                for req in step.resource_requirements:
                    pool = self._resource_pools.get(req.resource_type)
                    if pool:
                        # Project availability at execution time
                        projected_availability = self._project_resource_availability(
                            req.resource_type, slot_time
                        )
                        
                        availability_ratio = min(1.0, projected_availability / req.get_effective_amount())
                        resource_availability[req.resource_type.value] = {
                            "required": req.get_effective_amount(),
                            "projected_available": projected_availability,
                            "availability_ratio": availability_ratio
                        }
                        total_availability_score += availability_ratio
            
            avg_availability_score = (
                total_availability_score / len([req for step in workflow.steps for req in step.resource_requirements])
                if any(step.resource_requirements for step in workflow.steps) else 1.0
            )
            
            # Create execution plan
            execution_plan = {
                "start_time": slot_time.isoformat(),
                "estimated_end_time": (slot_time + workflow.get_critical_path_duration()).isoformat(),
                "resource_rates": {rt.value: rate for rt, rate in projected_rates.items()},
                "execution_strategy": workflow.execution_strategy,
                "step_schedule": []
            }
            
            # Add step scheduling details
            current_step_time = slot_time
            for step in workflow.steps:
                step_plan = {
                    "step_id": str(step.step_id),
                    "step_name": step.step_name,
                    "scheduled_start": current_step_time.isoformat(),
                    "estimated_duration": step.estimated_duration.total_seconds(),
                    "estimated_cost": step.calculate_resource_cost(projected_rates)
                }
                execution_plan["step_schedule"].append(step_plan)
                current_step_time += step.estimated_duration
            
            return {
                "slot_time": slot_time,
                "estimated_cost": estimated_cost,
                "resource_availability_score": avg_availability_score,
                "resource_availability": resource_availability,
                "execution_plan": execution_plan,
                "projected_rates": {rt.value: rate for rt, rate in projected_rates.items()}
            }
            
        except Exception as e:
            logger.error("Error evaluating time slot", error=str(e))
            return {
                "slot_time": slot_time,
                "error": str(e),
                "estimated_cost": float('inf'),
                "resource_availability_score": 0.0
            }
    
    def _project_ftns_rates_at_time(self, target_time: datetime) -> Dict[ResourceType, float]:
        """Project FTNS rates at a future time based on historical patterns"""
        # Simplified projection - in production, this would use ML models
        current_rates = self._get_current_ftns_rates()
        
        # Apply time-of-day multipliers
        hour = target_time.hour
        day_of_week = target_time.weekday()
        
        # Peak hours: 9 AM - 5 PM weekdays
        if day_of_week < 5 and 9 <= hour <= 17:
            multiplier = 1.5  # 50% premium during peak
        # Evening hours: 6 PM - 10 PM
        elif 18 <= hour <= 22:
            multiplier = 1.2  # 20% premium during evening
        # Off-peak: late night and early morning
        elif hour < 6 or hour > 23:
            multiplier = 0.7  # 30% discount during off-peak
        # Weekend premium (some users work weekends)
        elif day_of_week >= 5:
            multiplier = 0.9  # 10% discount on weekends
        else:
            multiplier = 1.0  # Standard rates
        
        projected_rates = {}
        for resource_type, current_rate in current_rates.items():
            projected_rates[resource_type] = current_rate * multiplier
        
        return projected_rates
    
    def _project_resource_availability(
        self,
        resource_type: ResourceType,
        target_time: datetime
    ) -> float:
        """Project resource availability at a future time"""
        pool = self._resource_pools.get(resource_type)
        if not pool:
            return 0.0
        
        # Simple projection based on current availability
        # In production, this would consider scheduled workflows and patterns
        base_availability = pool.available_capacity
        
        # Apply utilization patterns
        hour = target_time.hour
        
        # Assume higher utilization during business hours
        if 9 <= hour <= 17:
            utilization_factor = 0.7  # 70% utilization during peak
        elif 18 <= hour <= 22:
            utilization_factor = 0.5  # 50% utilization in evening
        else:
            utilization_factor = 0.2  # 20% utilization off-peak
        
        projected_available = pool.total_capacity * (1.0 - utilization_factor)
        return max(0.0, projected_available)
    
    def _calculate_time_preference_score(
        self,
        slot_time: datetime,
        preferred_time: Optional[datetime]
    ) -> float:
        """Calculate score based on time preferences"""
        if not preferred_time:
            return 0.5  # Neutral score if no preference
        
        time_diff = abs((slot_time - preferred_time).total_seconds())
        max_diff = 24 * 3600  # 24 hours
        
        # Closer to preferred time = higher score
        score = max(0.0, 1.0 - (time_diff / max_diff))
        return score
    
    def _get_priority_value(self, priority: SchedulingPriority) -> int:
        """Get numeric value for priority comparison"""
        priority_values = {
            SchedulingPriority.IMMEDIATE: 100,
            SchedulingPriority.HIGH: 80,
            SchedulingPriority.NORMAL: 60,
            SchedulingPriority.LOW: 40,
            SchedulingPriority.BACKGROUND: 20,
            SchedulingPriority.FLEXIBLE: 10
        }
        return priority_values.get(priority, 60)
    
    async def execute_scheduled_workflow(self, workflow_id: UUID) -> Dict[str, Any]:
        """
        Execute a scheduled workflow
        
        Args:
            workflow_id: ID of workflow to execute
            
        Returns:
            Execution result
        """
        try:
            workflow = self._scheduled_workflows.get(workflow_id)
            if not workflow:
                return {"success": False, "error": "Workflow not found"}
            
            if workflow.status != WorkflowStatus.SCHEDULED:
                return {"success": False, "error": f"Workflow is in {workflow.status} state"}
            
            # Update status
            workflow.status = WorkflowStatus.RUNNING
            workflow.actual_start = datetime.now(timezone.utc)
            workflow.execution_attempts += 1
            
            # Allocate resources
            allocation_success = await self._allocate_workflow_resources(workflow)
            if not allocation_success:
                workflow.status = WorkflowStatus.FAILED
                return {"success": False, "error": "Resource allocation failed"}
            
            # Execute workflow steps
            execution_result = await self._execute_workflow_steps(workflow)
            
            # Update final status
            if execution_result["success"]:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.actual_end = datetime.now(timezone.utc)
                workflow.execution_results = execution_result["results"]
                workflow.actual_ftns_cost = execution_result.get("actual_cost", 0)
                
                # Calculate cost savings
                if workflow.scheduled_start != workflow.actual_start:
                    immediate_rates = self._get_current_ftns_rates()
                    immediate_cost = workflow.calculate_total_estimated_cost(immediate_rates)
                    workflow.cost_savings = max(0, immediate_cost - workflow.actual_ftns_cost)
                    self.cost_optimization_savings += workflow.cost_savings
            else:
                workflow.status = WorkflowStatus.FAILED
            
            # Deallocate resources
            await self._deallocate_workflow_resources(workflow)
            
            # Update statistics
            self.scheduling_statistics["workflows_executed"] += 1
            if execution_result["success"]:
                self.scheduling_statistics["workflows_completed"] += 1
            else:
                self.scheduling_statistics["workflows_failed"] += 1
            
            logger.info(
                "Workflow execution completed",
                workflow_id=str(workflow_id),
                status=workflow.status,
                actual_cost=workflow.actual_ftns_cost,
                cost_savings=workflow.cost_savings
            )
            
            return {
                "success": execution_result["success"],
                "workflow_id": str(workflow_id),
                "status": workflow.status,
                "execution_results": workflow.execution_results,
                "actual_cost": workflow.actual_ftns_cost,
                "cost_savings": workflow.cost_savings,
                "execution_duration": (
                    (workflow.actual_end - workflow.actual_start).total_seconds()
                    if workflow.actual_end else None
                )
            }
            
        except Exception as e:
            logger.error("Error executing workflow", workflow_id=str(workflow_id), error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _allocate_workflow_resources(self, workflow: ScheduledWorkflow) -> bool:
        """Allocate resources for workflow execution"""
        try:
            allocations = {}
            
            # Calculate total resource requirements
            for step in workflow.steps:
                for req in step.resource_requirements:
                    resource_type = req.resource_type
                    amount = req.get_effective_amount()
                    
                    if resource_type not in allocations:
                        allocations[resource_type] = 0.0
                    allocations[resource_type] = max(allocations[resource_type], amount)
            
            # Attempt allocation
            allocated_resources = {}
            for resource_type, amount in allocations.items():
                pool = self._resource_pools.get(resource_type)
                if pool and pool.allocate_resources(amount):
                    allocated_resources[resource_type] = amount
                else:
                    # Rollback previous allocations
                    for rollback_type, rollback_amount in allocated_resources.items():
                        rollback_pool = self._resource_pools[rollback_type]
                        rollback_pool.deallocate_resources(rollback_amount)
                    return False
            
            # Store allocation record
            self._active_allocations[workflow.workflow_id] = allocated_resources
            return True
            
        except Exception as e:
            logger.error("Error allocating workflow resources", error=str(e))
            return False
    
    async def _deallocate_workflow_resources(self, workflow: ScheduledWorkflow):
        """Deallocate resources after workflow completion"""
        try:
            allocations = self._active_allocations.get(workflow.workflow_id, {})
            
            for resource_type, amount in allocations.items():
                pool = self._resource_pools.get(resource_type)
                if pool:
                    pool.deallocate_resources(amount)
            
            # Remove allocation record
            if workflow.workflow_id in self._active_allocations:
                del self._active_allocations[workflow.workflow_id]
                
        except Exception as e:
            logger.error("Error deallocating workflow resources", error=str(e))
    
    def _estimate_task_complexity(self, estimated_duration_seconds: float) -> TaskComplexity:
        """Estimate task complexity based on estimated duration"""
        if estimated_duration_seconds < 1:
            return TaskComplexity.TRIVIAL
        elif estimated_duration_seconds < 10:
            return TaskComplexity.SIMPLE
        elif estimated_duration_seconds < 60:
            return TaskComplexity.MODERATE
        elif estimated_duration_seconds < 600:  # 10 minutes
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.INTENSIVE
    
    async def _execute_workflow_steps(self, workflow: ScheduledWorkflow) -> Dict[str, Any]:
        """Execute all steps in a workflow"""
        try:
            results = {}
            total_cost = 0.0
            
            # Use parallelism engine if available
            if self._parallelism_engine and len(workflow.steps) > 1:
                # Convert to TaskDefinitions for parallelism analysis
                task_definitions = []
                for step in workflow.steps:
                    duration_seconds = step.estimated_duration.total_seconds()
                    task_def = TaskDefinition(
                        task_name=step.step_name,
                        agent_type=step.agent_type,
                        complexity=self._estimate_task_complexity(duration_seconds),
                        estimated_duration=duration_seconds,
                        input_dependencies=step.depends_on,
                        output_dependents=step.blocks
                    )
                    task_definitions.append(task_def)
                
                # Get parallelism decision
                parallelism_decision = await self._parallelism_engine.make_parallelism_decision(
                    task_definitions
                )
                
                # Execute according to strategy
                if parallelism_decision.recommended_strategy == ExecutionStrategy.PARALLEL:
                    # Execute all steps in parallel
                    step_results = await asyncio.gather(
                        *[self._execute_single_step(step, workflow) for step in workflow.steps],
                        return_exceptions=True
                    )
                    
                    for i, result in enumerate(step_results):
                        if isinstance(result, Exception):
                            return {"success": False, "error": str(result)}
                        results[workflow.steps[i].step_id] = result
                        total_cost += result.get("cost", 0)
                
                else:
                    # Execute sequentially
                    for step in workflow.steps:
                        step_result = await self._execute_single_step(step, workflow)
                        if not step_result.get("success", False):
                            return {"success": False, "error": step_result.get("error", "Step failed")}
                        results[step.step_id] = step_result
                        total_cost += step_result.get("cost", 0)
            else:
                # Simple sequential execution
                for step in workflow.steps:
                    step_result = await self._execute_single_step(step, workflow)
                    if not step_result.get("success", False):
                        return {"success": False, "error": step_result.get("error", "Step failed")}
                    results[step.step_id] = step_result
                    total_cost += step_result.get("cost", 0)
            
            return {
                "success": True,
                "results": results,
                "actual_cost": total_cost,
                "steps_completed": len(results)
            }
            
        except Exception as e:
            logger.error("Error executing workflow steps", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _execute_single_step(
        self,
        step: WorkflowStep,
        workflow: ScheduledWorkflow
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        try:
            # Simulate step execution
            start_time = datetime.now(timezone.utc)
            
            # Calculate actual cost
            current_rates = self._get_current_ftns_rates()
            step_cost = step.calculate_resource_cost(current_rates)
            
            # Simulate processing time
            await asyncio.sleep(0.1)  # Simulate work
            
            end_time = datetime.now(timezone.utc)
            execution_duration = (end_time - start_time).total_seconds()
            
            # Generate mock result
            result = {
                "success": True,
                "step_id": str(step.step_id),
                "step_name": step.step_name,
                "execution_duration": execution_duration,
                "cost": step_cost,
                "output": f"Mock output for {step.step_name}",
                "timestamp": end_time.isoformat()
            }
            
            logger.info(
                "Workflow step completed",
                step_id=str(step.step_id),
                duration=execution_duration,
                cost=step_cost
            )
            
            return result
            
        except Exception as e:
            logger.error("Error executing workflow step", step_id=str(step.step_id), error=str(e))
            return {"success": False, "error": str(e)}
    
    async def optimize_workflow_with_critical_path(
        self,
        workflow: ScheduledWorkflow,
        optimization_goals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize workflow scheduling using advanced critical path analysis
        
        Args:
            workflow: Workflow to optimize
            optimization_goals: Goals like 'minimize_duration', 'minimize_cost', 'balance_resources'
            
        Returns:
            Comprehensive optimization result with critical path insights
        """
        try:
            self.logger.info("Starting critical path optimization",
                           workflow_id=str(workflow.workflow_id))
            
            # Set default optimization goals
            goals = optimization_goals or {
                "minimize_duration": 0.4,
                "minimize_cost": 0.4,
                "balance_resources": 0.2
            }
            
            # Build resource constraints from current pools
            resource_constraints = {}
            for resource_type, pool in self._resource_pools.items():
                resource_constraints[resource_type.value] = ResourceConstraint(
                    resource_type=resource_type.value,
                    max_concurrent_usage=pool.available_capacity,
                    availability_windows=[(timedelta(), timedelta(days=7))]  # Assume week availability
                )
            
            # Perform comprehensive critical path analysis
            critical_path_result = await self._critical_path_calculator.calculate_critical_path(
                workflow.steps,
                resource_constraints
            )
            
            # Generate optimization recommendations
            optimization_recommendations = []
            
            # Critical path optimization
            if critical_path_result.critical_path_steps:
                critical_steps = [
                    step for step in workflow.steps 
                    if step.step_id in critical_path_result.critical_path_steps
                ]
                
                optimization_recommendations.extend([
                    f"Focus optimization on {len(critical_steps)} critical path steps",
                    f"Critical path duration: {critical_path_result.critical_path_duration.total_seconds():.1f} seconds",
                    f"Project efficiency: {critical_path_result.project_efficiency:.2%}"
                ])
                
                # Suggest resource allocation for critical steps
                for step in critical_steps:
                    if step.resource_requirements:
                        optimization_recommendations.append(
                            f"Consider increasing resources for critical step '{step.step_name}'"
                        )
            
            # Parallelization opportunities
            if critical_path_result.parallelizable_groups:
                total_parallelizable = sum(len(group) for group in critical_path_result.parallelizable_groups)
                optimization_recommendations.extend([
                    f"Found {len(critical_path_result.parallelizable_groups)} parallelization opportunities",
                    f"{total_parallelizable} steps can be executed in parallel",
                    f"Potential time savings: {critical_path_result.parallelization_potential:.1f}%"
                ])
            
            # Resource conflict resolution
            if critical_path_result.resource_conflicts:
                high_severity_conflicts = [
                    c for c in critical_path_result.resource_conflicts 
                    if c.get("severity") == "high"
                ]
                
                if high_severity_conflicts:
                    optimization_recommendations.append(
                        f"Resolve {len(high_severity_conflicts)} high-severity resource conflicts"
                    )
                
                # Suggest resource reallocation
                for conflict in critical_path_result.resource_conflicts[:3]:  # Top 3 conflicts
                    optimization_recommendations.append(
                        f"Resource conflict: {conflict['resource_type']} needs "
                        f"{conflict['overflow']:.1f} additional units at "
                        f"{conflict['time']}"
                    )
            
            # Calculate optimized schedule
            optimized_schedule = await self._generate_optimized_schedule(
                workflow, critical_path_result, goals
            )
            
            # Calculate improvement metrics
            original_duration = workflow.get_critical_path_duration()
            optimized_duration = critical_path_result.critical_path_duration
            time_improvement = ((original_duration - optimized_duration).total_seconds() / 
                              original_duration.total_seconds()) * 100 if original_duration.total_seconds() > 0 else 0
            
            optimization_result = {
                "success": True,
                "workflow_id": str(workflow.workflow_id),
                "critical_path_analysis": {
                    "critical_path_duration": critical_path_result.critical_path_duration.total_seconds(),
                    "critical_path_steps": [str(step_id) for step_id in critical_path_result.critical_path_steps],
                    "total_critical_paths": len(critical_path_result.critical_paths),
                    "project_efficiency": critical_path_result.project_efficiency,
                    "parallelization_potential": critical_path_result.parallelization_potential,
                    "resource_utilization_score": critical_path_result.resource_utilization_score
                },
                "optimization_opportunities": {
                    "parallelizable_groups": [
                        [str(step_id) for step_id in group] 
                        for group in critical_path_result.parallelizable_groups
                    ],
                    "resource_conflicts": critical_path_result.resource_conflicts,
                    "recommendations": optimization_recommendations
                },
                "optimized_schedule": optimized_schedule,
                "improvement_metrics": {
                    "time_improvement_percentage": time_improvement,
                    "estimated_cost_reduction": optimized_schedule.get("cost_reduction", 0),
                    "resource_efficiency_gain": optimized_schedule.get("efficiency_gain", 0)
                },
                "scheduling_recommendations": critical_path_result.scheduling_recommendations
            }
            
            # Update workflow with optimized execution strategy if requested
            if optimization_goals and optimization_goals.get("apply_optimizations", False):
                await self._apply_optimizations_to_workflow(workflow, critical_path_result)
                optimization_result["optimizations_applied"] = True
            
            self.logger.info("Critical path optimization completed",
                           workflow_id=str(workflow.workflow_id),
                           time_improvement=time_improvement,
                           parallelizable_groups=len(critical_path_result.parallelizable_groups))
            
            return optimization_result
            
        except Exception as e:
            self.logger.error("Error in critical path optimization",
                            workflow_id=str(workflow.workflow_id),
                            error=str(e))
            return {
                "success": False,
                "error": str(e),
                "workflow_id": str(workflow.workflow_id)
            }
    
    async def _generate_optimized_schedule(
        self,
        workflow: ScheduledWorkflow,
        critical_path_result,
        optimization_goals: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate optimized execution schedule based on critical path analysis"""
        try:
            # Calculate current costs
            current_rates = self._get_current_ftns_rates()
            current_cost = workflow.calculate_total_estimated_cost(current_rates)
            
            # Optimize based on parallelizable groups
            optimized_duration = critical_path_result.critical_path_duration
            
            # Estimate cost reduction from parallel execution
            if critical_path_result.parallelizable_groups:
                # Assume parallel execution reduces overall cost due to shorter duration
                parallelization_factor = min(0.3, critical_path_result.parallelization_potential / 100)
                estimated_cost_reduction = current_cost * parallelization_factor
            else:
                estimated_cost_reduction = 0
            
            # Resource efficiency improvements
            efficiency_gain = critical_path_result.resource_utilization_score / 100 * 0.1  # Up to 10% gain
            
            # Generate step-by-step optimized schedule
            optimized_steps_schedule = []
            current_time = timedelta()
            
            # Process parallelizable groups
            processed_steps = set()
            
            for group in critical_path_result.parallelizable_groups:
                if len(group) > 1:
                    # Parallel execution group
                    group_steps = [
                        step for step in workflow.steps 
                        if step.step_id in group and step.step_id not in processed_steps
                    ]
                    
                    if group_steps:
                        max_duration = max(step.estimated_duration for step in group_steps)
                        
                        for step in group_steps:
                            optimized_steps_schedule.append({
                                "step_id": str(step.step_id),
                                "step_name": step.step_name,
                                "start_time": current_time.total_seconds(),
                                "duration": step.estimated_duration.total_seconds(),
                                "execution_mode": "parallel",
                                "parallel_group": len(optimized_steps_schedule)
                            })
                            processed_steps.add(step.step_id)
                        
                        current_time += max_duration
            
            # Process remaining steps sequentially
            remaining_steps = [
                step for step in workflow.steps 
                if step.step_id not in processed_steps
            ]
            
            for step in remaining_steps:
                optimized_steps_schedule.append({
                    "step_id": str(step.step_id),
                    "step_name": step.step_name,
                    "start_time": current_time.total_seconds(),
                    "duration": step.estimated_duration.total_seconds(),
                    "execution_mode": "sequential"
                })
                current_time += step.estimated_duration
            
            return {
                "optimized_duration": optimized_duration.total_seconds(),
                "cost_reduction": estimated_cost_reduction,
                "efficiency_gain": efficiency_gain,
                "optimized_steps_schedule": optimized_steps_schedule,
                "execution_strategy": "hybrid_parallel_sequential",
                "optimization_applied": True
            }
            
        except Exception as e:
            self.logger.error("Error generating optimized schedule", error=str(e))
            return {
                "optimized_duration": workflow.get_critical_path_duration().total_seconds(),
                "cost_reduction": 0,
                "efficiency_gain": 0,
                "optimization_applied": False,
                "error": str(e)
            }
    
    async def _apply_optimizations_to_workflow(
        self,
        workflow: ScheduledWorkflow,
        critical_path_result
    ):
        """Apply optimizations to the workflow configuration"""
        try:
            # Update execution strategy based on parallelization opportunities
            if critical_path_result.parallelizable_groups:
                workflow.execution_strategy = ExecutionStrategy.HYBRID
                
                # Store parallelization information in workflow metadata
                if not hasattr(workflow, 'optimization_metadata'):
                    workflow.optimization_metadata = {}
                
                workflow.optimization_metadata.update({
                    "critical_path_optimized": True,
                    "parallelizable_groups": [
                        [str(step_id) for step_id in group] 
                        for group in critical_path_result.parallelizable_groups
                    ],
                    "optimization_timestamp": datetime.now(timezone.utc).isoformat(),
                    "expected_improvement": critical_path_result.parallelization_potential
                })
            
            # Adjust resource requirements for critical path steps
            critical_step_ids = set(critical_path_result.critical_path_steps)
            for step in workflow.steps:
                if step.step_id in critical_step_ids:
                    # Increase priority multiplier for critical steps
                    for req in step.resource_requirements:
                        req.priority_multiplier = min(req.priority_multiplier * 1.2, 10.0)
            
            self.logger.info("Optimizations applied to workflow",
                           workflow_id=str(workflow.workflow_id))
            
        except Exception as e:
            self.logger.error("Error applying optimizations", error=str(e))
    
    def get_scheduling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scheduling statistics"""
        return {
            "scheduling_stats": dict(self.scheduling_statistics),
            "active_workflows": len([w for w in self._scheduled_workflows.values() 
                                   if w.status in [WorkflowStatus.SCHEDULED, WorkflowStatus.RUNNING]]),
            "total_scheduled_workflows": len(self._scheduled_workflows),
            "cost_optimization_savings": self.cost_optimization_savings,
            "resource_utilization": {
                resource_type.value: {
                    "utilization_percentage": pool.get_utilization_percentage(),
                    "current_cost_per_unit": pool.current_cost_per_unit,
                    "demand_multiplier": pool.demand_multiplier
                }
                for resource_type, pool in self._resource_pools.items()
            },
            "upcoming_events": len(self.event_queue),
            "active_allocations": len(self._active_allocations)
        }


# Global instance for easy access
_workflow_scheduler = None

def get_workflow_scheduler() -> WorkflowScheduler:
    """Get global workflow scheduler instance"""
    global _workflow_scheduler
    if _workflow_scheduler is None:
        _workflow_scheduler = WorkflowScheduler()
    return _workflow_scheduler