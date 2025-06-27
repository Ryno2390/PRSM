"""
Selective Parallelism Decision Engine

🧠 COGNITION.AI INSIGHTS INTEGRATION:
- Intelligent parallel vs sequential execution routing based on task dependencies
- Dynamic task dependency analysis for optimal execution strategy selection
- Agent coordination-aware parallelism with context sharing requirements
- Resource contention avoidance and load balancing for multi-agent tasks
- Execution strategy optimization based on task characteristics and agent capabilities

This module implements sophisticated parallelism decision-making that addresses
Cognition.AI's insight about naive multi-agent frameworks executing everything
in parallel without considering task dependencies and coordination requirements.
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
import networkx as nx

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import (
    PRSMBaseModel, TimestampMixin, AgentType, TaskStatus, SafetyLevel
)
from prsm.context.enhanced_context_compression import (
    ContextSegment, ContextType, ContextImportance
)
from prsm.context.reasoning_trace_sharing import (
    ReasoningTraceLevel, EnhancedReasoningStep
)

logger = structlog.get_logger(__name__)


class ExecutionStrategy(str, Enum):
    """Execution strategy types"""
    SEQUENTIAL = "sequential"           # Execute one by one in order
    PARALLEL = "parallel"              # Execute all simultaneously
    MIXED_PARALLEL = "mixed_parallel"  # Mix of parallel and sequential
    PIPELINE = "pipeline"              # Pipeline execution with stages
    CONDITIONAL = "conditional"        # Strategy depends on runtime conditions
    ADAPTIVE = "adaptive"              # Strategy adapts based on performance


class TaskDependencyType(str, Enum):
    """Types of task dependencies"""
    DATA_DEPENDENCY = "data_dependency"           # Task requires output from another
    RESOURCE_DEPENDENCY = "resource_dependency"   # Tasks compete for same resources
    ORDERING_DEPENDENCY = "ordering_dependency"   # Tasks must execute in specific order
    CONTEXT_DEPENDENCY = "context_dependency"     # Tasks share context requirements
    SAFETY_DEPENDENCY = "safety_dependency"      # Safety constraints require ordering
    COORDINATION_DEPENDENCY = "coordination_dependency"  # Agents need coordination


class ResourceType(str, Enum):
    """Types of computational resources"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    API_QUOTA = "api_quota"
    MODEL_INFERENCE = "model_inference"
    CONTEXT_WINDOW = "context_window"
    AGENT_ATTENTION = "agent_attention"


class TaskComplexity(str, Enum):
    """Task complexity levels for execution planning"""
    TRIVIAL = "trivial"       # <1 second execution
    SIMPLE = "simple"         # 1-10 seconds execution
    MODERATE = "moderate"     # 10-60 seconds execution
    COMPLEX = "complex"       # 1-10 minutes execution
    INTENSIVE = "intensive"   # >10 minutes execution


class ParallelismDecision(PRSMBaseModel):
    """Decision about how to execute a set of tasks"""
    decision_id: UUID = Field(default_factory=uuid4)
    task_ids: List[UUID]
    recommended_strategy: ExecutionStrategy
    confidence_score: float = Field(ge=0.0, le=1.0)
    
    # Strategy Details
    parallel_groups: List[List[UUID]] = Field(default_factory=list)
    sequential_order: List[UUID] = Field(default_factory=list)
    pipeline_stages: List[List[UUID]] = Field(default_factory=list)
    
    # Reasoning
    decision_rationale: str
    dependency_analysis: Dict[str, Any] = Field(default_factory=dict)
    resource_analysis: Dict[str, Any] = Field(default_factory=dict)
    coordination_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance Predictions
    estimated_sequential_time: float = Field(default=0.0)
    estimated_parallel_time: float = Field(default=0.0)
    resource_utilization_score: float = Field(ge=0.0, le=1.0, default=0.5)
    coordination_overhead: float = Field(default=0.0)
    
    # Risk Assessment
    failure_risk_parallel: float = Field(ge=0.0, le=1.0, default=0.0)
    failure_risk_sequential: float = Field(ge=0.0, le=1.0, default=0.0)
    data_consistency_risk: float = Field(ge=0.0, le=1.0, default=0.0)


class TaskDefinition(PRSMBaseModel):
    """Comprehensive task definition for parallelism analysis"""
    task_id: UUID = Field(default_factory=uuid4)
    task_name: str
    agent_type: AgentType
    
    # Task Characteristics
    complexity: TaskComplexity
    estimated_duration: float = Field(default=0.0)  # seconds
    priority: int = Field(ge=1, le=10, default=5)
    
    # Resource Requirements
    resource_requirements: Dict[ResourceType, float] = Field(default_factory=dict)
    max_concurrent_executions: int = Field(default=1)
    
    # Dependencies
    input_dependencies: List[UUID] = Field(default_factory=list)  # Tasks that must complete first
    output_dependents: List[UUID] = Field(default_factory=list)   # Tasks that depend on this one
    resource_conflicts: List[UUID] = Field(default_factory=list)  # Tasks that can't run simultaneously
    
    # Context Requirements
    requires_shared_context: bool = Field(default=False)
    context_dependencies: List[UUID] = Field(default_factory=list)
    reasoning_trace_dependencies: List[UUID] = Field(default_factory=list)
    
    # Safety and Coordination
    safety_level: SafetyLevel = Field(default=SafetyLevel.MEDIUM)
    requires_agent_coordination: bool = Field(default=False)
    coordination_agents: List[AgentType] = Field(default_factory=list)
    
    # Execution Metadata
    can_be_interrupted: bool = Field(default=True)
    supports_checkpointing: bool = Field(default=False)
    rollback_capable: bool = Field(default=False)


class ResourceUtilization(PRSMBaseModel):
    """Resource utilization tracking"""
    resource_type: ResourceType
    current_usage: float = Field(ge=0.0, default=0.0)
    max_capacity: float = Field(gt=0.0, default=1.0)
    reserved_usage: float = Field(ge=0.0, default=0.0)
    
    # Usage patterns
    historical_usage: List[float] = Field(default_factory=list)
    peak_usage_times: List[datetime] = Field(default_factory=list)
    
    @property
    def available_capacity(self) -> float:
        return max(0.0, self.max_capacity - self.current_usage - self.reserved_usage)
    
    @property
    def utilization_percentage(self) -> float:
        return (self.current_usage / self.max_capacity) * 100 if self.max_capacity > 0 else 0


class ExecutionPlan(PRSMBaseModel):
    """Comprehensive execution plan for task set"""
    plan_id: UUID = Field(default_factory=uuid4)
    task_definitions: List[TaskDefinition]
    parallelism_decision: ParallelismDecision
    
    # Execution Schedule
    execution_schedule: List[Dict[str, Any]] = Field(default_factory=list)
    resource_allocation: Dict[ResourceType, ResourceUtilization] = Field(default_factory=dict)
    
    # Monitoring and Adaptation
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    adaptation_triggers: List[str] = Field(default_factory=list)
    fallback_strategies: List[ExecutionStrategy] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    estimated_completion: Optional[datetime] = None
    actual_start_time: Optional[datetime] = None
    actual_completion_time: Optional[datetime] = None


class SelectiveParallelismEngine(TimestampMixin):
    """
    Selective Parallelism Decision Engine
    
    Intelligent decision-making for when to execute tasks in parallel vs sequential,
    addressing Cognition.AI's insights about naive multi-agent parallelism.
    """
    
    def __init__(self):
        super().__init__()
        self.dependency_graph = nx.DiGraph()
        self.resource_monitor = self._initialize_resource_monitor()
        self.execution_history: List[ExecutionPlan] = []
        self.performance_cache: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_parallel_tasks = 10
        self.resource_threshold = 0.8  # 80% utilization threshold
        self.coordination_overhead_factor = 0.1
        self.safety_buffer_factor = 0.2
        
        logger.info("SelectiveParallelismEngine initialized")
    
    def _initialize_resource_monitor(self) -> Dict[ResourceType, ResourceUtilization]:
        """Initialize resource monitoring"""
        resources = {}
        for resource_type in ResourceType:
            resources[resource_type] = ResourceUtilization(
                resource_type=resource_type,
                max_capacity=self._get_default_capacity(resource_type)
            )
        return resources
    
    def _get_default_capacity(self, resource_type: ResourceType) -> float:
        """Get default capacity for resource type"""
        defaults = {
            ResourceType.CPU: 8.0,  # 8 cores
            ResourceType.MEMORY: 32.0,  # 32 GB
            ResourceType.NETWORK: 1000.0,  # 1 Gbps
            ResourceType.STORAGE: 1000.0,  # 1 TB
            ResourceType.API_QUOTA: 1000.0,  # 1000 requests/minute
            ResourceType.MODEL_INFERENCE: 10.0,  # 10 concurrent inferences
            ResourceType.CONTEXT_WINDOW: 100000.0,  # 100k tokens
            ResourceType.AGENT_ATTENTION: 5.0,  # 5 concurrent agent tasks
        }
        return defaults.get(resource_type, 1.0)
    
    async def analyze_task_dependencies(
        self,
        tasks: List[TaskDefinition]
    ) -> Dict[str, Any]:
        """
        Analyze dependencies between tasks
        
        Args:
            tasks: List of task definitions to analyze
            
        Returns:
            Comprehensive dependency analysis
        """
        try:
            # Build dependency graph
            self.dependency_graph.clear()
            
            # Add nodes
            for task in tasks:
                self.dependency_graph.add_node(
                    task.task_id,
                    task_data=task.dict(),
                    complexity=task.complexity,
                    priority=task.priority
                )
            
            # Add edges for dependencies
            for task in tasks:
                for dep_id in task.input_dependencies:
                    if dep_id in [t.task_id for t in tasks]:
                        self.dependency_graph.add_edge(
                            dep_id, 
                            task.task_id,
                            dependency_type=TaskDependencyType.DATA_DEPENDENCY
                        )
                
                # Add resource conflict edges
                for conflict_id in task.resource_conflicts:
                    if conflict_id in [t.task_id for t in tasks]:
                        self.dependency_graph.add_edge(
                            task.task_id,
                            conflict_id,
                            dependency_type=TaskDependencyType.RESOURCE_DEPENDENCY
                        )
            
            # Analyze dependency patterns
            analysis = {
                "total_tasks": len(tasks),
                "dependency_count": self.dependency_graph.number_of_edges(),
                "strongly_connected_components": list(nx.strongly_connected_components(self.dependency_graph)),
                "topological_order": list(nx.topological_sort(self.dependency_graph)) if nx.is_directed_acyclic_graph(self.dependency_graph) else [],
                "has_cycles": not nx.is_directed_acyclic_graph(self.dependency_graph),
                "max_dependency_depth": 0,
                "parallelizable_groups": [],
                "critical_path": [],
                "bottleneck_tasks": []
            }
            
            # Find parallelizable groups
            if analysis["topological_order"]:
                analysis["parallelizable_groups"] = self._find_parallelizable_groups(tasks)
                analysis["critical_path"] = self._find_critical_path(tasks)
                analysis["max_dependency_depth"] = self._calculate_max_dependency_depth()
            
            # Identify bottlenecks
            analysis["bottleneck_tasks"] = self._identify_bottleneck_tasks(tasks)
            
            logger.info(
                "Task dependency analysis completed",
                task_count=len(tasks),
                dependency_count=analysis["dependency_count"],
                has_cycles=analysis["has_cycles"]
            )
            
            return analysis
            
        except Exception as e:
            logger.error("Error analyzing task dependencies", error=str(e))
            return {"error": str(e), "total_tasks": len(tasks)}
    
    def _find_parallelizable_groups(self, tasks: List[TaskDefinition]) -> List[List[UUID]]:
        """Find groups of tasks that can be executed in parallel"""
        groups = []
        remaining_tasks = set(task.task_id for task in tasks)
        
        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                predecessors = set(self.dependency_graph.predecessors(task_id))
                if not (predecessors & remaining_tasks):  # No unmet dependencies
                    ready_tasks.append(task_id)
            
            if ready_tasks:
                groups.append(ready_tasks)
                remaining_tasks -= set(ready_tasks)
            else:
                # Break cycles or handle deadlock
                remaining_tasks.clear()
        
        return groups
    
    def _find_critical_path(self, tasks: List[TaskDefinition]) -> List[UUID]:
        """Find the critical path through task dependencies"""
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            return []
        
        # Calculate longest path (critical path)
        task_durations = {task.task_id: task.estimated_duration for task in tasks}
        
        try:
            # Use networkx to find longest path
            longest_path = nx.dag_longest_path(
                self.dependency_graph,
                weight=lambda u, v, d: task_durations.get(v, 0)
            )
            return longest_path
        except:
            return []
    
    def _calculate_max_dependency_depth(self) -> int:
        """Calculate maximum dependency depth"""
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            return 0
        
        depths = {}
        for node in nx.topological_sort(self.dependency_graph):
            predecessors = list(self.dependency_graph.predecessors(node))
            if not predecessors:
                depths[node] = 0
            else:
                depths[node] = 1 + max(depths[pred] for pred in predecessors)
        
        return max(depths.values()) if depths else 0
    
    def _identify_bottleneck_tasks(self, tasks: List[TaskDefinition]) -> List[UUID]:
        """Identify tasks that are likely bottlenecks"""
        bottlenecks = []
        
        for task in tasks:
            # High resource usage tasks
            total_resource_usage = sum(task.resource_requirements.values())
            if total_resource_usage > 5.0:  # Arbitrary threshold
                bottlenecks.append(task.task_id)
            
            # Tasks with many dependents
            dependents = len(list(self.dependency_graph.successors(task.task_id)))
            if dependents > 3:
                bottlenecks.append(task.task_id)
            
            # Long duration tasks
            if task.estimated_duration > 300:  # 5 minutes
                bottlenecks.append(task.task_id)
        
        return list(set(bottlenecks))
    
    async def analyze_resource_requirements(
        self,
        tasks: List[TaskDefinition],
        execution_strategy: ExecutionStrategy
    ) -> Dict[str, Any]:
        """
        Analyze resource requirements for given execution strategy
        
        Args:
            tasks: List of task definitions
            execution_strategy: Proposed execution strategy
            
        Returns:
            Resource analysis results
        """
        try:
            analysis = {
                "strategy": execution_strategy,
                "resource_feasibility": True,
                "resource_utilization": {},
                "resource_conflicts": [],
                "peak_usage_periods": [],
                "optimization_suggestions": []
            }
            
            # Calculate resource requirements by strategy
            if execution_strategy == ExecutionStrategy.PARALLEL:
                # All tasks run simultaneously
                peak_requirements = defaultdict(float)
                for task in tasks:
                    for resource_type, requirement in task.resource_requirements.items():
                        peak_requirements[resource_type] += requirement
                
                # Check feasibility
                for resource_type, requirement in peak_requirements.items():
                    available = self.resource_monitor[resource_type].available_capacity
                    utilization = requirement / self.resource_monitor[resource_type].max_capacity
                    
                    analysis["resource_utilization"][resource_type.value] = {
                        "required": requirement,
                        "available": available,
                        "utilization_percentage": utilization * 100,
                        "feasible": requirement <= available
                    }
                    
                    if requirement > available:
                        analysis["resource_feasibility"] = False
                        analysis["resource_conflicts"].append({
                            "resource": resource_type.value,
                            "required": requirement,
                            "available": available,
                            "deficit": requirement - available
                        })
            
            elif execution_strategy == ExecutionStrategy.SEQUENTIAL:
                # Tasks run one after another
                max_requirements = defaultdict(float)
                for task in tasks:
                    for resource_type, requirement in task.resource_requirements.items():
                        max_requirements[resource_type] = max(
                            max_requirements[resource_type], 
                            requirement
                        )
                
                # Check feasibility (much easier for sequential)
                for resource_type, requirement in max_requirements.items():
                    available = self.resource_monitor[resource_type].available_capacity
                    utilization = requirement / self.resource_monitor[resource_type].max_capacity
                    
                    analysis["resource_utilization"][resource_type.value] = {
                        "required": requirement,
                        "available": available,
                        "utilization_percentage": utilization * 100,
                        "feasible": requirement <= available
                    }
                    
                    if requirement > available:
                        analysis["resource_feasibility"] = False
            
            # Generate optimization suggestions
            if not analysis["resource_feasibility"]:
                analysis["optimization_suggestions"].extend([
                    "Consider mixed parallel execution strategy",
                    "Implement task batching to reduce peak resource usage",
                    "Add resource scaling or queuing mechanisms",
                    "Prioritize tasks by resource efficiency"
                ])
            elif execution_strategy == ExecutionStrategy.SEQUENTIAL:
                analysis["optimization_suggestions"].append(
                    "Consider selective parallelism for independent tasks"
                )
            
            logger.info(
                "Resource analysis completed",
                strategy=execution_strategy,
                feasible=analysis["resource_feasibility"],
                task_count=len(tasks)
            )
            
            return analysis
            
        except Exception as e:
            logger.error("Error analyzing resource requirements", error=str(e))
            return {"error": str(e), "strategy": execution_strategy}
    
    async def make_parallelism_decision(
        self,
        tasks: List[TaskDefinition],
        context_requirements: Optional[Dict[str, Any]] = None
    ) -> ParallelismDecision:
        """
        Make intelligent decision about execution strategy
        
        Args:
            tasks: List of task definitions
            context_requirements: Context sharing requirements
            
        Returns:
            Parallelism decision with detailed reasoning
        """
        try:
            # Analyze dependencies and resources
            dependency_analysis = await self.analyze_task_dependencies(tasks)
            
            # Evaluate different strategies
            strategies_evaluated = []
            
            # 1. Evaluate Sequential Strategy
            sequential_analysis = await self.analyze_resource_requirements(
                tasks, ExecutionStrategy.SEQUENTIAL
            )
            sequential_time = sum(task.estimated_duration for task in tasks)
            
            strategies_evaluated.append({
                "strategy": ExecutionStrategy.SEQUENTIAL,
                "feasible": sequential_analysis["resource_feasibility"],
                "estimated_time": sequential_time,
                "resource_efficiency": 0.7,  # Generally resource efficient
                "coordination_overhead": 0.0,
                "failure_risk": 0.1
            })
            
            # 2. Evaluate Parallel Strategy
            parallel_analysis = await self.analyze_resource_requirements(
                tasks, ExecutionStrategy.PARALLEL
            )
            parallel_time = max(task.estimated_duration for task in tasks) if tasks else 0
            coordination_overhead = len(tasks) * self.coordination_overhead_factor
            
            strategies_evaluated.append({
                "strategy": ExecutionStrategy.PARALLEL,
                "feasible": parallel_analysis["resource_feasibility"] and not dependency_analysis["has_cycles"],
                "estimated_time": parallel_time + coordination_overhead,
                "resource_efficiency": 0.4 if parallel_analysis["resource_feasibility"] else 0.1,
                "coordination_overhead": coordination_overhead,
                "failure_risk": 0.3 if len(tasks) > 5 else 0.2
            })
            
            # 3. Evaluate Mixed Parallel Strategy
            mixed_feasible = len(dependency_analysis.get("parallelizable_groups", [])) > 1
            mixed_time = self._estimate_mixed_parallel_time(tasks, dependency_analysis)
            
            strategies_evaluated.append({
                "strategy": ExecutionStrategy.MIXED_PARALLEL,
                "feasible": mixed_feasible,
                "estimated_time": mixed_time,
                "resource_efficiency": 0.6,
                "coordination_overhead": coordination_overhead * 0.5,
                "failure_risk": 0.15
            })
            
            # Select best strategy
            feasible_strategies = [s for s in strategies_evaluated if s["feasible"]]
            
            if not feasible_strategies:
                # Fallback to sequential if nothing else works
                selected_strategy = strategies_evaluated[0]
                selected_strategy["feasible"] = True
            else:
                # Score strategies based on multiple criteria
                for strategy in feasible_strategies:
                    time_score = 1.0 / (1.0 + strategy["estimated_time"] / 60)  # Normalize by minutes
                    efficiency_score = strategy["resource_efficiency"]
                    risk_score = 1.0 - strategy["failure_risk"]
                    
                    # Weighted score
                    strategy["composite_score"] = (
                        0.4 * time_score +
                        0.3 * efficiency_score +
                        0.3 * risk_score
                    )
                
                selected_strategy = max(feasible_strategies, key=lambda s: s["composite_score"])
            
            # Build decision
            decision = ParallelismDecision(
                task_ids=[task.task_id for task in tasks],
                recommended_strategy=selected_strategy["strategy"],
                confidence_score=min(0.95, selected_strategy.get("composite_score", 0.5)),
                estimated_sequential_time=sequential_time,
                estimated_parallel_time=parallel_time,
                resource_utilization_score=selected_strategy["resource_efficiency"],
                coordination_overhead=selected_strategy["coordination_overhead"],
                failure_risk_parallel=0.3 if len(tasks) > 5 else 0.2,
                failure_risk_sequential=0.1,
                decision_rationale=self._generate_decision_rationale(
                    selected_strategy, dependency_analysis, len(tasks)
                ),
                dependency_analysis=dependency_analysis,
                resource_analysis={
                    "sequential": sequential_analysis,
                    "parallel": parallel_analysis
                }
            )
            
            # Set execution groups based on strategy
            if selected_strategy["strategy"] == ExecutionStrategy.PARALLEL:
                decision.parallel_groups = [[task.task_id for task in tasks]]
            elif selected_strategy["strategy"] == ExecutionStrategy.SEQUENTIAL:
                decision.sequential_order = dependency_analysis.get("topological_order", [task.task_id for task in tasks])
            elif selected_strategy["strategy"] == ExecutionStrategy.MIXED_PARALLEL:
                decision.parallel_groups = dependency_analysis.get("parallelizable_groups", [])
            
            logger.info(
                "Parallelism decision made",
                strategy=decision.recommended_strategy,
                confidence=decision.confidence_score,
                task_count=len(tasks)
            )
            
            return decision
            
        except Exception as e:
            logger.error("Error making parallelism decision", error=str(e))
            # Return safe fallback
            return ParallelismDecision(
                task_ids=[task.task_id for task in tasks],
                recommended_strategy=ExecutionStrategy.SEQUENTIAL,
                confidence_score=0.3,
                decision_rationale=f"Fallback to sequential due to error: {str(e)}"
            )
    
    def _estimate_mixed_parallel_time(
        self, 
        tasks: List[TaskDefinition], 
        dependency_analysis: Dict[str, Any]
    ) -> float:
        """Estimate execution time for mixed parallel strategy"""
        groups = dependency_analysis.get("parallelizable_groups", [])
        if not groups:
            return sum(task.estimated_duration for task in tasks)
        
        total_time = 0.0
        task_duration_map = {task.task_id: task.estimated_duration for task in tasks}
        
        for group in groups:
            # Time for this group is the maximum duration in the group
            group_time = max(task_duration_map.get(task_id, 0) for task_id in group)
            total_time += group_time
        
        return total_time
    
    def _generate_decision_rationale(
        self, 
        selected_strategy: Dict[str, Any], 
        dependency_analysis: Dict[str, Any], 
        task_count: int
    ) -> str:
        """Generate human-readable rationale for the decision"""
        strategy = selected_strategy["strategy"]
        
        if strategy == ExecutionStrategy.SEQUENTIAL:
            if dependency_analysis.get("has_cycles"):
                return f"Sequential execution required due to circular dependencies among {task_count} tasks."
            elif not selected_strategy["feasible"]:
                return f"Sequential execution chosen as resource-safe fallback for {task_count} tasks."
            else:
                return f"Sequential execution optimal for {task_count} tasks with high interdependency."
        
        elif strategy == ExecutionStrategy.PARALLEL:
            return f"Full parallel execution feasible and optimal for {task_count} independent tasks with sufficient resources."
        
        elif strategy == ExecutionStrategy.MIXED_PARALLEL:
            group_count = len(dependency_analysis.get("parallelizable_groups", []))
            return f"Mixed parallel execution with {group_count} parallel groups balances efficiency and dependencies for {task_count} tasks."
        
        else:
            return f"Strategy {strategy} selected based on task characteristics and resource analysis."
    
    async def create_execution_plan(
        self,
        tasks: List[TaskDefinition],
        parallelism_decision: ParallelismDecision,
        context_requirements: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create comprehensive execution plan
        
        Args:
            tasks: List of task definitions
            parallelism_decision: Parallelism decision
            context_requirements: Context sharing requirements
            
        Returns:
            Detailed execution plan
        """
        try:
            plan = ExecutionPlan(
                task_definitions=tasks,
                parallelism_decision=parallelism_decision,
                resource_allocation=dict(self.resource_monitor)
            )
            
            # Generate execution schedule
            schedule = []
            
            if parallelism_decision.recommended_strategy == ExecutionStrategy.SEQUENTIAL:
                for i, task_id in enumerate(parallelism_decision.sequential_order):
                    schedule.append({
                        "step": i + 1,
                        "type": "sequential",
                        "tasks": [task_id],
                        "estimated_duration": next(
                            task.estimated_duration for task in tasks 
                            if task.task_id == task_id
                        )
                    })
            
            elif parallelism_decision.recommended_strategy == ExecutionStrategy.PARALLEL:
                schedule.append({
                    "step": 1,
                    "type": "parallel",
                    "tasks": parallelism_decision.task_ids,
                    "estimated_duration": max(task.estimated_duration for task in tasks)
                })
            
            elif parallelism_decision.recommended_strategy == ExecutionStrategy.MIXED_PARALLEL:
                for i, group in enumerate(parallelism_decision.parallel_groups):
                    schedule.append({
                        "step": i + 1,
                        "type": "parallel_group",
                        "tasks": group,
                        "estimated_duration": max(
                            task.estimated_duration for task in tasks 
                            if task.task_id in group
                        )
                    })
            
            plan.execution_schedule = schedule
            
            # Calculate estimated completion time
            total_duration = sum(step["estimated_duration"] for step in schedule)
            plan.estimated_completion = plan.created_at + timedelta(seconds=total_duration)
            
            # Add performance metrics
            plan.performance_metrics = {
                "estimated_total_duration": total_duration,
                "parallelization_efficiency": (
                    sum(task.estimated_duration for task in tasks) / total_duration
                    if total_duration > 0 else 1.0
                ),
                "resource_efficiency": parallelism_decision.resource_utilization_score,
                "coordination_overhead": parallelism_decision.coordination_overhead
            }
            
            # Add adaptation triggers
            plan.adaptation_triggers = [
                "task_failure_rate > 0.2",
                "resource_utilization > 0.9",
                "execution_time > 1.5 * estimated_time",
                "coordination_overhead > 0.3"
            ]
            
            # Add fallback strategies
            fallbacks = [ExecutionStrategy.SEQUENTIAL]
            if parallelism_decision.recommended_strategy != ExecutionStrategy.MIXED_PARALLEL:
                fallbacks.append(ExecutionStrategy.MIXED_PARALLEL)
            plan.fallback_strategies = fallbacks
            
            logger.info(
                "Execution plan created",
                plan_id=str(plan.plan_id),
                strategy=parallelism_decision.recommended_strategy,
                estimated_duration=total_duration,
                task_count=len(tasks)
            )
            
            return plan
            
        except Exception as e:
            logger.error("Error creating execution plan", error=str(e))
            raise
    
    async def optimize_execution_strategy(
        self,
        execution_plan: ExecutionPlan,
        performance_feedback: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Optimize execution strategy based on performance feedback
        
        Args:
            execution_plan: Current execution plan
            performance_feedback: Performance data from execution
            
        Returns:
            Optimized execution plan
        """
        try:
            # Analyze performance
            actual_duration = performance_feedback.get("actual_duration", 0)
            estimated_duration = execution_plan.performance_metrics.get("estimated_total_duration", 0)
            
            # Check if adaptation is needed
            adaptation_needed = False
            
            if actual_duration > estimated_duration * 1.5:
                adaptation_needed = True
                logger.warning("Execution took significantly longer than estimated")
            
            failure_rate = performance_feedback.get("failure_rate", 0)
            if failure_rate > 0.2:
                adaptation_needed = True
                logger.warning("High failure rate detected")
            
            resource_issues = performance_feedback.get("resource_issues", [])
            if resource_issues:
                adaptation_needed = True
                logger.warning("Resource issues detected", issues=resource_issues)
            
            if not adaptation_needed:
                return execution_plan
            
            # Create optimized plan
            tasks = execution_plan.task_definitions
            
            # Try fallback strategy
            fallback_strategy = execution_plan.fallback_strategies[0]
            
            new_decision = ParallelismDecision(
                task_ids=[task.task_id for task in tasks],
                recommended_strategy=fallback_strategy,
                confidence_score=0.6,  # Lower confidence for fallback
                decision_rationale=f"Fallback to {fallback_strategy} due to performance issues"
            )
            
            # Create new execution plan
            optimized_plan = await self.create_execution_plan(
                tasks, new_decision
            )
            
            # Preserve history
            optimized_plan.performance_metrics.update({
                "optimization_trigger": "performance_adaptation",
                "previous_strategy": execution_plan.parallelism_decision.recommended_strategy,
                "previous_duration": actual_duration,
                "optimization_timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(
                "Execution strategy optimized",
                original_strategy=execution_plan.parallelism_decision.recommended_strategy,
                new_strategy=fallback_strategy,
                plan_id=str(optimized_plan.plan_id)
            )
            
            return optimized_plan
            
        except Exception as e:
            logger.error("Error optimizing execution strategy", error=str(e))
            return execution_plan  # Return original if optimization fails
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics and insights"""
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        strategies_used = [plan.parallelism_decision.recommended_strategy for plan in self.execution_history]
        strategy_counts = {strategy: strategies_used.count(strategy) for strategy in set(strategies_used)}
        
        avg_confidence = sum(
            plan.parallelism_decision.confidence_score for plan in self.execution_history
        ) / len(self.execution_history)
        
        completed_plans = [
            plan for plan in self.execution_history 
            if plan.actual_completion_time is not None
        ]
        
        accuracy_metrics = {}
        if completed_plans:
            estimation_errors = []
            for plan in completed_plans:
                if plan.actual_start_time and plan.actual_completion_time:
                    actual_duration = (
                        plan.actual_completion_time - plan.actual_start_time
                    ).total_seconds()
                    estimated_duration = plan.performance_metrics.get("estimated_total_duration", 0)
                    if estimated_duration > 0:
                        error = abs(actual_duration - estimated_duration) / estimated_duration
                        estimation_errors.append(error)
            
            if estimation_errors:
                accuracy_metrics = {
                    "avg_estimation_error": sum(estimation_errors) / len(estimation_errors),
                    "max_estimation_error": max(estimation_errors),
                    "min_estimation_error": min(estimation_errors)
                }
        
        return {
            "total_plans_created": len(self.execution_history),
            "strategy_distribution": strategy_counts,
            "average_confidence": avg_confidence,
            "completed_plans": len(completed_plans),
            "accuracy_metrics": accuracy_metrics,
            "resource_utilization": {
                resource_type.value: {
                    "current_usage": resource.current_usage,
                    "utilization_percentage": resource.utilization_percentage
                }
                for resource_type, resource in self.resource_monitor.items()
            }
        }


# Global instance for easy access
_selective_parallelism_engine = None

def get_selective_parallelism_engine() -> SelectiveParallelismEngine:
    """Get global selective parallelism engine instance"""
    global _selective_parallelism_engine
    if _selective_parallelism_engine is None:
        _selective_parallelism_engine = SelectiveParallelismEngine()
    return _selective_parallelism_engine