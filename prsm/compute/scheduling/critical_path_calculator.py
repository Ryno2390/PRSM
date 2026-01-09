"""
Critical Path Calculator for Workflow Scheduling
==============================================

Advanced critical path method (CPM) implementation for calculating optimal
workflow execution schedules considering dependencies, resource constraints,
and parallel execution opportunities.

Key Features:
- Topological sorting for dependency resolution
- Forward and backward pass calculations for critical path identification
- Parallel execution optimization with resource constraints
- Float time calculation for scheduling flexibility
- Critical path visualization and analysis
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from uuid import UUID
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WorkflowNode:
    """Node representing a workflow step in the critical path analysis"""
    step_id: UUID
    step_name: str
    duration: timedelta
    resource_requirements: List[Any]  # WorkflowStep.ResourceRequirement
    
    # Dependency relationships
    predecessors: Set[UUID] = field(default_factory=set)
    successors: Set[UUID] = field(default_factory=set)
    
    # Critical path calculations
    earliest_start: timedelta = field(default=timedelta())
    earliest_finish: timedelta = field(default=timedelta())
    latest_start: timedelta = field(default=timedelta())
    latest_finish: timedelta = field(default=timedelta())
    
    # Float calculations
    total_float: timedelta = field(default=timedelta())
    free_float: timedelta = field(default=timedelta())
    
    # Critical path status
    is_critical: bool = field(default=False)
    criticality_index: float = field(default=0.0)
    
    def calculate_float_times(self):
        """Calculate total and free float times"""
        self.total_float = self.latest_start - self.earliest_start
        # Free float calculation requires successor information
    
    def is_on_critical_path(self) -> bool:
        """Check if this node is on the critical path"""
        return self.total_float == timedelta() and self.is_critical


@dataclass
class CriticalPathResult:
    """Result of critical path analysis"""
    critical_path_duration: timedelta
    critical_path_steps: List[UUID]
    total_project_duration: timedelta
    
    # Detailed analysis
    all_nodes: Dict[UUID, WorkflowNode]
    critical_paths: List[List[UUID]]  # Multiple critical paths possible
    
    # Optimization opportunities
    parallelizable_groups: List[List[UUID]]
    resource_conflicts: List[Dict[str, Any]]
    scheduling_recommendations: List[str]
    
    # Performance metrics
    project_efficiency: float  # Critical path / Total work ratio
    parallelization_potential: float  # Percentage of work that can be parallelized
    resource_utilization_score: float


@dataclass
class ResourceConstraint:
    """Resource constraint for critical path calculation"""
    resource_type: str
    max_concurrent_usage: float
    availability_windows: List[Tuple[timedelta, timedelta]]  # (start, end) times when available


class CriticalPathCalculator:
    """
    Advanced Critical Path Method (CPM) calculator for workflow scheduling
    
    Features:
    - Dependency graph construction and validation
    - Forward and backward pass calculations
    - Critical path identification with multiple path support
    - Resource-constrained project scheduling (RCPS)
    - Parallel execution optimization
    - Float time analysis for scheduling flexibility
    """
    
    def __init__(self):
        self.logger = logger.bind(component="critical_path_calculator")
        
        # Analysis state
        self.nodes: Dict[UUID, WorkflowNode] = {}
        self.dependency_graph: Dict[UUID, Set[UUID]] = defaultdict(set)
        self.reverse_graph: Dict[UUID, Set[UUID]] = defaultdict(set)
        
        # Resource constraints
        self.resource_constraints: Dict[str, ResourceConstraint] = {}
        
        # Analysis results
        self.critical_paths: List[List[UUID]] = []
        self.project_duration: timedelta = timedelta()
        
        self.logger.info("Critical Path Calculator initialized")
    
    async def calculate_critical_path(
        self,
        workflow_steps: List[Any],  # List[WorkflowStep]
        resource_constraints: Optional[Dict[str, ResourceConstraint]] = None
    ) -> CriticalPathResult:
        """
        Calculate critical path for workflow with comprehensive analysis
        
        Args:
            workflow_steps: List of workflow steps with dependencies
            resource_constraints: Optional resource availability constraints
            
        Returns:
            Complete critical path analysis result
        """
        try:
            self.logger.info("Starting critical path calculation",
                           step_count=len(workflow_steps))
            
            # Reset state
            self._reset_analysis_state()
            
            # Store resource constraints
            if resource_constraints:
                self.resource_constraints = resource_constraints
            
            # Build dependency graph
            await self._build_dependency_graph(workflow_steps)
            
            # Validate graph for cycles and consistency
            validation_result = await self._validate_dependency_graph()
            if not validation_result["valid"]:
                raise ValueError(f"Invalid dependency graph: {validation_result['errors']}")
            
            # Perform forward pass (earliest times)
            await self._calculate_forward_pass()
            
            # Perform backward pass (latest times)
            await self._calculate_backward_pass()
            
            # Calculate float times
            await self._calculate_float_times()
            
            # Identify critical path(s)
            await self._identify_critical_paths()
            
            # Analyze parallelization opportunities
            parallelizable_groups = await self._analyze_parallelization_opportunities()
            
            # Check resource conflicts
            resource_conflicts = await self._analyze_resource_conflicts()
            
            # Generate optimization recommendations
            recommendations = await self._generate_scheduling_recommendations()
            
            # Calculate performance metrics
            metrics = await self._calculate_performance_metrics()
            
            # Create comprehensive result
            result = CriticalPathResult(
                critical_path_duration=self.project_duration,
                critical_path_steps=self.critical_paths[0] if self.critical_paths else [],
                total_project_duration=self.project_duration,
                all_nodes=self.nodes.copy(),
                critical_paths=self.critical_paths.copy(),
                parallelizable_groups=parallelizable_groups,
                resource_conflicts=resource_conflicts,
                scheduling_recommendations=recommendations,
                project_efficiency=metrics["project_efficiency"],
                parallelization_potential=metrics["parallelization_potential"],
                resource_utilization_score=metrics["resource_utilization_score"]
            )
            
            self.logger.info("Critical path calculation completed",
                           critical_path_duration=self.project_duration.total_seconds(),
                           critical_paths_found=len(self.critical_paths),
                           parallelizable_groups=len(parallelizable_groups))
            
            return result
            
        except Exception as e:
            self.logger.error("Error calculating critical path", error=str(e))
            raise
    
    def _reset_analysis_state(self):
        """Reset analysis state for new calculation"""
        self.nodes.clear()
        self.dependency_graph.clear()
        self.reverse_graph.clear()
        self.critical_paths.clear()
        self.project_duration = timedelta()
    
    async def _build_dependency_graph(self, workflow_steps: List[Any]):
        """Build dependency graph from workflow steps"""
        try:
            # Create nodes for all steps
            for step in workflow_steps:
                node = WorkflowNode(
                    step_id=step.step_id,
                    step_name=step.step_name,
                    duration=step.estimated_duration,
                    resource_requirements=step.resource_requirements
                )
                self.nodes[step.step_id] = node
            
            # Build dependency relationships
            for step in workflow_steps:
                step_id = step.step_id
                
                # Add predecessors (dependencies)
                for dep_id in step.depends_on:
                    if dep_id in self.nodes:
                        self.dependency_graph[dep_id].add(step_id)
                        self.reverse_graph[step_id].add(dep_id)
                        self.nodes[step_id].predecessors.add(dep_id)
                        self.nodes[dep_id].successors.add(step_id)
                
                # Add successors (blocks)
                for blocked_id in step.blocks:
                    if blocked_id in self.nodes:
                        self.dependency_graph[step_id].add(blocked_id)
                        self.reverse_graph[blocked_id].add(step_id)
                        self.nodes[blocked_id].predecessors.add(step_id)
                        self.nodes[step_id].successors.add(blocked_id)
            
            self.logger.debug("Dependency graph built",
                            nodes=len(self.nodes),
                            dependencies=sum(len(deps) for deps in self.dependency_graph.values()))
            
        except Exception as e:
            self.logger.error("Error building dependency graph", error=str(e))
            raise
    
    async def _validate_dependency_graph(self) -> Dict[str, Any]:
        """Validate dependency graph for cycles and consistency"""
        try:
            errors = []
            
            # Check for circular dependencies using DFS
            visited = set()
            rec_stack = set()
            
            def has_cycle(node_id: UUID) -> bool:
                visited.add(node_id)
                rec_stack.add(node_id)
                
                for successor in self.dependency_graph.get(node_id, set()):
                    if successor not in visited:
                        if has_cycle(successor):
                            return True
                    elif successor in rec_stack:
                        return True
                
                rec_stack.remove(node_id)
                return False
            
            # Check all nodes for cycles
            for node_id in self.nodes:
                if node_id not in visited:
                    if has_cycle(node_id):
                        errors.append(f"Circular dependency detected involving node {node_id}")
                        break
            
            # Check for orphaned dependencies
            for node_id, node in self.nodes.items():
                for pred_id in node.predecessors:
                    if pred_id not in self.nodes:
                        errors.append(f"Node {node_id} depends on non-existent node {pred_id}")
                
                for succ_id in node.successors:
                    if succ_id not in self.nodes:
                        errors.append(f"Node {node_id} blocks non-existent node {succ_id}")
            
            # Check for duplicate relationships
            for node_id in self.nodes:
                if len(self.dependency_graph[node_id]) != len(set(self.dependency_graph[node_id])):
                    errors.append(f"Duplicate dependencies found for node {node_id}")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "node_count": len(self.nodes),
                "dependency_count": sum(len(deps) for deps in self.dependency_graph.values())
            }
            
        except Exception as e:
            self.logger.error("Error validating dependency graph", error=str(e))
            return {"valid": False, "errors": [str(e)]}
    
    async def _calculate_forward_pass(self):
        """Calculate earliest start and finish times (forward pass)"""
        try:
            # Topological sort to process nodes in dependency order
            in_degree = {node_id: len(node.predecessors) for node_id, node in self.nodes.items()}
            queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
            
            processed_count = 0
            
            while queue:
                current_id = queue.popleft()
                current_node = self.nodes[current_id]
                processed_count += 1
                
                # Calculate earliest start time
                if current_node.predecessors:
                    # Max of all predecessor finish times
                    current_node.earliest_start = max(
                        self.nodes[pred_id].earliest_finish
                        for pred_id in current_node.predecessors
                    )
                else:
                    # Root node starts at time 0
                    current_node.earliest_start = timedelta()
                
                # Calculate earliest finish time
                current_node.earliest_finish = current_node.earliest_start + current_node.duration
                
                # Process successors
                for successor_id in current_node.successors:
                    in_degree[successor_id] -= 1
                    if in_degree[successor_id] == 0:
                        queue.append(successor_id)
            
            # Verify all nodes were processed (no cycles)
            if processed_count != len(self.nodes):
                raise ValueError("Circular dependency detected during forward pass")
            
            # Calculate project duration
            self.project_duration = max(
                node.earliest_finish for node in self.nodes.values()
            ) if self.nodes else timedelta()
            
            self.logger.debug("Forward pass completed",
                            project_duration=self.project_duration.total_seconds())
            
        except Exception as e:
            self.logger.error("Error in forward pass calculation", error=str(e))
            raise
    
    async def _calculate_backward_pass(self):
        """Calculate latest start and finish times (backward pass)"""
        try:
            # Start from nodes with no successors
            out_degree = {node_id: len(node.successors) for node_id, node in self.nodes.items()}
            queue = deque([node_id for node_id, degree in out_degree.items() if degree == 0])
            
            # Initialize terminal nodes with project duration
            for node_id in queue:
                node = self.nodes[node_id]
                node.latest_finish = self.project_duration
                node.latest_start = node.latest_finish - node.duration
            
            processed_count = len(queue)
            
            while queue:
                current_id = queue.popleft()
                current_node = self.nodes[current_id]
                
                # Process predecessors
                for predecessor_id in current_node.predecessors:
                    predecessor_node = self.nodes[predecessor_id]
                    
                    # Update latest finish time (min of all successor start times)
                    if not hasattr(predecessor_node, '_latest_finish_updated'):
                        predecessor_node.latest_finish = current_node.latest_start
                        predecessor_node._latest_finish_updated = True
                    else:
                        predecessor_node.latest_finish = min(
                            predecessor_node.latest_finish,
                            current_node.latest_start
                        )
                    
                    # Calculate latest start time
                    predecessor_node.latest_start = (
                        predecessor_node.latest_finish - predecessor_node.duration
                    )
                    
                    out_degree[predecessor_id] -= 1
                    if out_degree[predecessor_id] == 0:
                        queue.append(predecessor_id)
                        processed_count += 1
            
            # Clean up temporary attributes
            for node in self.nodes.values():
                if hasattr(node, '_latest_finish_updated'):
                    delattr(node, '_latest_finish_updated')
            
            # Verify all nodes were processed
            if processed_count != len(self.nodes):
                raise ValueError("Error in backward pass: not all nodes processed")
            
            self.logger.debug("Backward pass completed")
            
        except Exception as e:
            self.logger.error("Error in backward pass calculation", error=str(e))
            raise
    
    async def _calculate_float_times(self):
        """Calculate total and free float for all nodes"""
        try:
            for node_id, node in self.nodes.items():
                # Total float = Latest Start - Earliest Start
                node.total_float = node.latest_start - node.earliest_start
                
                # Free float = min(successor earliest start) - earliest finish
                if node.successors:
                    min_successor_start = min(
                        self.nodes[succ_id].earliest_start
                        for succ_id in node.successors
                    )
                    node.free_float = min_successor_start - node.earliest_finish
                else:
                    # Terminal node: free float = total float
                    node.free_float = node.total_float
                
                # Ensure non-negative floats
                node.free_float = max(timedelta(), node.free_float)
                
                # Mark critical nodes (zero total float)
                node.is_critical = node.total_float == timedelta()
            
            self.logger.debug("Float calculations completed")
            
        except Exception as e:
            self.logger.error("Error calculating float times", error=str(e))
            raise
    
    async def _identify_critical_paths(self):
        """Identify all critical paths through the project"""
        try:
            self.critical_paths.clear()
            
            # Find all critical nodes
            critical_nodes = {
                node_id for node_id, node in self.nodes.items()
                if node.is_critical
            }
            
            if not critical_nodes:
                self.logger.warning("No critical path found")
                return
            
            # Find start nodes (critical nodes with no critical predecessors)
            start_nodes = {
                node_id for node_id in critical_nodes
                if not any(pred_id in critical_nodes for pred_id in self.nodes[node_id].predecessors)
            }
            
            # Find all paths from start nodes to end nodes
            def find_critical_paths_from_node(node_id: UUID, current_path: List[UUID]):
                current_path = current_path + [node_id]
                
                # Check if this is an end node (no critical successors)
                critical_successors = [
                    succ_id for succ_id in self.nodes[node_id].successors
                    if succ_id in critical_nodes
                ]
                
                if not critical_successors:
                    # End of critical path
                    self.critical_paths.append(current_path)
                    return
                
                # Continue path through critical successors
                for successor_id in critical_successors:
                    find_critical_paths_from_node(successor_id, current_path)
            
            # Generate all critical paths
            for start_node_id in start_nodes:
                find_critical_paths_from_node(start_node_id, [])
            
            # Remove duplicate paths
            unique_paths = []
            for path in self.critical_paths:
                if path not in unique_paths:
                    unique_paths.append(path)
            self.critical_paths = unique_paths
            
            # Calculate criticality index for each node
            for node_id, node in self.nodes.items():
                if node.is_critical:
                    node.criticality_index = 1.0
                else:
                    # Criticality index based on float ratio
                    max_float = max(
                        n.total_float.total_seconds() for n in self.nodes.values()
                        if n.total_float > timedelta()
                    ) or 1.0
                    node.criticality_index = 1.0 - (node.total_float.total_seconds() / max_float)
            
            self.logger.info("Critical paths identified",
                           critical_paths_count=len(self.critical_paths),
                           critical_nodes_count=len(critical_nodes))
            
        except Exception as e:
            self.logger.error("Error identifying critical paths", error=str(e))
            raise
    
    async def _analyze_parallelization_opportunities(self) -> List[List[UUID]]:
        """Analyze opportunities for parallel execution"""
        try:
            parallelizable_groups = []
            
            # Group nodes by their earliest start time
            time_groups = defaultdict(list)
            for node_id, node in self.nodes.items():
                start_time_key = int(node.earliest_start.total_seconds())
                time_groups[start_time_key].append(node_id)
            
            # Identify groups that can run in parallel
            for start_time, node_ids in time_groups.items():
                if len(node_ids) > 1:
                    # Check for resource conflicts
                    conflict_free_groups = self._resolve_resource_conflicts(node_ids)
                    parallelizable_groups.extend(conflict_free_groups)
            
            # Also check nodes with overlapping time windows (considering float)
            for node_id, node in self.nodes.items():
                parallel_candidates = []
                
                for other_id, other_node in self.nodes.items():
                    if node_id != other_id:
                        # Check if time windows overlap
                        node_window_start = node.earliest_start
                        node_window_end = node.latest_finish
                        other_window_start = other_node.earliest_start
                        other_window_end = other_node.latest_finish
                        
                        # Check for overlap and no dependency relationship
                        if (node_window_start < other_window_end and 
                            other_window_start < node_window_end and
                            other_id not in node.predecessors and
                            other_id not in node.successors and
                            node_id not in other_node.predecessors and
                            node_id not in other_node.successors):
                            parallel_candidates.append(other_id)
                
                if parallel_candidates:
                    group = [node_id] + parallel_candidates
                    if group not in parallelizable_groups:
                        parallelizable_groups.append(group)
            
            # Remove duplicate and subset groups
            unique_groups = []
            for group in parallelizable_groups:
                group_set = set(group)
                is_subset = any(
                    group_set.issubset(set(existing_group))
                    for existing_group in unique_groups
                )
                if not is_subset:
                    unique_groups.append(group)
            
            self.logger.debug("Parallelization analysis completed",
                            parallelizable_groups=len(unique_groups))
            
            return unique_groups
            
        except Exception as e:
            self.logger.error("Error analyzing parallelization opportunities", error=str(e))
            return []
    
    def _resolve_resource_conflicts(self, node_ids: List[UUID]) -> List[List[UUID]]:
        """Resolve resource conflicts for parallel execution"""
        try:
            if not self.resource_constraints:
                return [node_ids]  # No constraints, all can run in parallel
            
            # Group nodes by resource requirements
            resource_usage = defaultdict(float)
            for node_id in node_ids:
                node = self.nodes[node_id]
                for req in node.resource_requirements:
                    resource_type = getattr(req, 'resource_type', str(req))
                    amount = getattr(req, 'amount', 1.0)
                    resource_usage[resource_type] += amount
            
            # Check against constraints
            conflicting_resources = []
            for resource_type, total_usage in resource_usage.items():
                constraint = self.resource_constraints.get(resource_type)
                if constraint and total_usage > constraint.max_concurrent_usage:
                    conflicting_resources.append(resource_type)
            
            if not conflicting_resources:
                return [node_ids]  # No conflicts
            
            # Partition nodes to resolve conflicts
            # Simple greedy approach - can be enhanced with optimization algorithms
            partitions = []
            remaining_nodes = node_ids.copy()
            
            while remaining_nodes:
                current_partition = []
                current_usage = defaultdict(float)
                
                for node_id in remaining_nodes.copy():
                    node = self.nodes[node_id]
                    can_add = True
                    
                    # Check if adding this node would violate constraints
                    for req in node.resource_requirements:
                        resource_type = getattr(req, 'resource_type', str(req))
                        amount = getattr(req, 'amount', 1.0)
                        
                        constraint = self.resource_constraints.get(resource_type)
                        if constraint:
                            new_usage = current_usage[resource_type] + amount
                            if new_usage > constraint.max_concurrent_usage:
                                can_add = False
                                break
                    
                    if can_add:
                        current_partition.append(node_id)
                        remaining_nodes.remove(node_id)
                        
                        # Update current usage
                        for req in node.resource_requirements:
                            resource_type = getattr(req, 'resource_type', str(req))
                            amount = getattr(req, 'amount', 1.0)
                            current_usage[resource_type] += amount
                
                if current_partition:
                    partitions.append(current_partition)
                else:
                    # Cannot place remaining nodes, add them individually
                    for node_id in remaining_nodes:
                        partitions.append([node_id])
                    break
            
            return partitions
            
        except Exception as e:
            self.logger.error("Error resolving resource conflicts", error=str(e))
            return [[node_id] for node_id in node_ids]  # Fallback to sequential
    
    async def _analyze_resource_conflicts(self) -> List[Dict[str, Any]]:
        """Analyze resource conflicts across the entire project"""
        try:
            conflicts = []
            
            if not self.resource_constraints:
                return conflicts
            
            # Simulate project execution timeline
            timeline_events = []
            
            for node_id, node in self.nodes.items():
                # Add start and end events
                timeline_events.append({
                    "time": node.earliest_start,
                    "type": "start",
                    "node_id": node_id,
                    "resource_requirements": node.resource_requirements
                })
                timeline_events.append({
                    "time": node.earliest_finish,
                    "type": "end",
                    "node_id": node_id,
                    "resource_requirements": node.resource_requirements
                })
            
            # Sort events by time
            timeline_events.sort(key=lambda x: x["time"])
            
            # Track resource usage over time
            current_usage = defaultdict(float)
            active_nodes = set()
            
            for event in timeline_events:
                if event["type"] == "start":
                    active_nodes.add(event["node_id"])
                    
                    # Add resource usage
                    for req in event["resource_requirements"]:
                        resource_type = getattr(req, 'resource_type', str(req))
                        amount = getattr(req, 'amount', 1.0)
                        current_usage[resource_type] += amount
                        
                        # Check for conflicts
                        constraint = self.resource_constraints.get(resource_type)
                        if constraint and current_usage[resource_type] > constraint.max_concurrent_usage:
                            conflicts.append({
                                "time": event["time"],
                                "resource_type": resource_type,
                                "required": current_usage[resource_type],
                                "available": constraint.max_concurrent_usage,
                                "overflow": current_usage[resource_type] - constraint.max_concurrent_usage,
                                "active_nodes": list(active_nodes),
                                "severity": "high" if current_usage[resource_type] > constraint.max_concurrent_usage * 1.5 else "medium"
                            })
                
                elif event["type"] == "end":
                    active_nodes.discard(event["node_id"])
                    
                    # Remove resource usage
                    for req in event["resource_requirements"]:
                        resource_type = getattr(req, 'resource_type', str(req))
                        amount = getattr(req, 'amount', 1.0)
                        current_usage[resource_type] = max(0, current_usage[resource_type] - amount)
            
            self.logger.debug("Resource conflict analysis completed",
                            conflicts_found=len(conflicts))
            
            return conflicts
            
        except Exception as e:
            self.logger.error("Error analyzing resource conflicts", error=str(e))
            return []
    
    async def _generate_scheduling_recommendations(self) -> List[str]:
        """Generate actionable scheduling recommendations"""
        try:
            recommendations = []
            
            # Critical path recommendations
            if self.critical_paths:
                longest_path = max(self.critical_paths, key=len, default=[])
                recommendations.append(
                    f"Focus on optimizing {len(longest_path)} critical steps to reduce project duration"
                )
                
                if len(self.critical_paths) > 1:
                    recommendations.append(
                        f"Multiple critical paths detected ({len(self.critical_paths)}). "
                        "Consider parallel optimization strategies"
                    )
            
            # Float time recommendations
            high_float_nodes = [
                node for node in self.nodes.values()
                if node.total_float > timedelta(hours=1) and not node.is_critical
            ]
            
            if high_float_nodes:
                recommendations.append(
                    f"{len(high_float_nodes)} tasks have significant scheduling flexibility. "
                    "Use for resource leveling and cost optimization"
                )
            
            # Resource utilization recommendations
            if self.resource_constraints:
                # Check for underutilized resources
                max_usage = defaultdict(float)
                for node in self.nodes.values():
                    for req in node.resource_requirements:
                        resource_type = getattr(req, 'resource_type', str(req))
                        amount = getattr(req, 'amount', 1.0)
                        max_usage[resource_type] = max(max_usage[resource_type], amount)
                
                for resource_type, constraint in self.resource_constraints.items():
                    if max_usage[resource_type] < constraint.max_concurrent_usage * 0.7:
                        recommendations.append(
                            f"Resource {resource_type} is underutilized. "
                            f"Consider increasing parallelism or reducing allocation"
                        )
            
            # Parallelization recommendations
            sequential_duration = sum(node.duration for node in self.nodes.values())
            if self.project_duration < sequential_duration * 0.8:
                recommendations.append(
                    "Good parallelization achieved. Consider further optimization of critical path"
                )
            elif self.project_duration > sequential_duration * 0.95:
                recommendations.append(
                    "Limited parallelization detected. Review dependencies and consider async execution"
                )
            
            # Dependency optimization
            high_dependency_nodes = [
                node for node in self.nodes.values()
                if len(node.predecessors) > 3 or len(node.successors) > 3
            ]
            
            if high_dependency_nodes:
                recommendations.append(
                    f"{len(high_dependency_nodes)} tasks have complex dependencies. "
                    "Consider breaking down or restructuring workflow"
                )
            
            if not recommendations:
                recommendations.append("Workflow is well-optimized for current constraints")
            
            return recommendations
            
        except Exception as e:
            self.logger.error("Error generating recommendations", error=str(e))
            return ["Error generating recommendations"]
    
    async def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the workflow"""
        try:
            # Project efficiency: Critical path duration / Sum of all durations
            total_work = sum(node.duration for node in self.nodes.values())
            project_efficiency = (
                self.project_duration.total_seconds() / total_work.total_seconds()
                if total_work.total_seconds() > 0 else 0.0
            )
            
            # Parallelization potential
            sequential_duration = total_work.total_seconds()
            parallel_duration = self.project_duration.total_seconds()
            parallelization_potential = (
                1.0 - (parallel_duration / sequential_duration)
                if sequential_duration > 0 else 0.0
            ) * 100
            
            # Resource utilization score
            if self.resource_constraints:
                utilization_scores = []
                for resource_type, constraint in self.resource_constraints.items():
                    max_usage = 0.0
                    for node in self.nodes.values():
                        for req in node.resource_requirements:
                            req_type = getattr(req, 'resource_type', str(req))
                            if req_type == resource_type:
                                amount = getattr(req, 'amount', 1.0)
                                max_usage = max(max_usage, amount)
                    
                    utilization = max_usage / constraint.max_concurrent_usage
                    utilization_scores.append(min(1.0, utilization))
                
                resource_utilization_score = (
                    sum(utilization_scores) / len(utilization_scores)
                    if utilization_scores else 0.0
                ) * 100
            else:
                resource_utilization_score = 100.0  # No constraints
            
            metrics = {
                "project_efficiency": project_efficiency,
                "parallelization_potential": parallelization_potential,
                "resource_utilization_score": resource_utilization_score
            }
            
            self.logger.debug("Performance metrics calculated", **metrics)
            return metrics
            
        except Exception as e:
            self.logger.error("Error calculating performance metrics", error=str(e))
            return {
                "project_efficiency": 0.0,
                "parallelization_potential": 0.0,
                "resource_utilization_score": 0.0
            }


# Factory function
def get_critical_path_calculator() -> CriticalPathCalculator:
    """Get critical path calculator instance"""
    return CriticalPathCalculator()