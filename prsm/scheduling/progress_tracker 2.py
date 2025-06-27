"""
Progress Tracking for Scheduled Jobs

📊 PROGRESS TRACKING SYSTEM:
- Real-time workflow execution monitoring with live updates
- Performance metrics tracking (execution time, resource usage, success rates)
- Progress visualization with completion percentage and ETA
- Historical analytics and trend analysis
- SLA monitoring and compliance tracking
- Bottleneck detection and optimization recommendations

This module implements comprehensive job monitoring that enables:
1. Real-time progress updates with granular step tracking
2. Performance analytics for optimization insights
3. SLA compliance monitoring and alerting
4. Resource utilization tracking and optimization
5. Historical trend analysis for predictive scheduling
6. Automated bottleneck detection and recommendations
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from uuid import UUID, uuid4
from collections import defaultdict, deque
import statistics
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import PRSMBaseModel, TimestampMixin, AgentType
from prsm.scheduling.workflow_scheduler import (
    ScheduledWorkflow, WorkflowStep, WorkflowStatus, ResourceType
)
from prsm.scheduling.workflow_persistence import WorkflowCheckpoint

logger = structlog.get_logger(__name__)


class ProgressStatus(str, Enum):
    """Progress tracking status"""
    NOT_STARTED = "not_started"       # Workflow not yet started
    INITIALIZING = "initializing"     # Setting up execution environment
    RUNNING = "running"               # Actively executing steps
    WAITING = "waiting"               # Waiting for resources or dependencies
    PAUSED = "paused"                 # Manually paused
    COMPLETING = "completing"         # Finalizing and cleanup
    COMPLETED = "completed"           # Successfully finished
    FAILED = "failed"                 # Failed with errors
    CANCELLED = "cancelled"           # Cancelled by user
    TIMEOUT = "timeout"               # Exceeded time limits


class StepProgressStatus(str, Enum):
    """Individual step progress status"""
    PENDING = "pending"               # Step not yet started
    STARTING = "starting"             # Step initialization
    EXECUTING = "executing"           # Step actively running
    WAITING_DEPENDENCY = "waiting_dependency"  # Waiting for prerequisite
    WAITING_RESOURCE = "waiting_resource"      # Waiting for resource allocation
    COMPLETING = "completing"         # Step finishing up
    COMPLETED = "completed"           # Step successfully finished
    FAILED = "failed"                 # Step failed
    SKIPPED = "skipped"              # Step skipped due to conditions
    RETRYING = "retrying"            # Step being retried after failure


class MetricType(str, Enum):
    """Types of performance metrics"""
    EXECUTION_TIME = "execution_time"
    RESOURCE_USAGE = "resource_usage"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    COST = "cost"
    EFFICIENCY = "efficiency"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"                     # Informational updates
    WARNING = "warning"               # Potential issues
    ERROR = "error"                   # Error conditions
    CRITICAL = "critical"             # Critical failures


class StepProgress(PRSMBaseModel):
    """Progress tracking for individual workflow step"""
    step_id: UUID
    step_name: str
    agent_type: AgentType
    
    # Progress status
    status: StepProgressStatus = Field(default=StepProgressStatus.PENDING)
    completion_percentage: float = Field(ge=0.0, le=100.0, default=0.0)
    
    # Timing information
    scheduled_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    estimated_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    
    # Resource tracking
    allocated_resources: Dict[ResourceType, float] = Field(default_factory=dict)
    resource_usage: Dict[ResourceType, float] = Field(default_factory=dict)
    peak_resource_usage: Dict[ResourceType, float] = Field(default_factory=dict)
    
    # Execution details
    attempts: int = Field(default=0)
    max_attempts: int = Field(default=3)
    current_operation: str = Field(default="")
    
    # Results and metrics
    output_size_bytes: int = Field(default=0)
    processing_rate: Optional[float] = None  # items/second, tokens/second, etc.
    quality_score: Optional[float] = None
    error_count: int = Field(default=0)
    warning_count: int = Field(default=0)
    
    # Cost tracking
    ftns_cost: float = Field(default=0.0)
    resource_cost: float = Field(default=0.0)
    
    # Progress updates
    progress_updates: List[Dict[str, Any]] = Field(default_factory=list)
    last_progress_update: Optional[datetime] = None
    
    def update_progress(
        self,
        percentage: float,
        current_operation: str = "",
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Update step progress"""
        self.completion_percentage = min(100.0, max(0.0, percentage))
        if current_operation:
            self.current_operation = current_operation
        
        # Record progress update
        update = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "percentage": self.completion_percentage,
            "operation": current_operation,
            "metrics": metrics or {}
        }
        self.progress_updates.append(update)
        self.last_progress_update = datetime.now(timezone.utc)
        
        # Keep only last 50 updates to manage memory
        if len(self.progress_updates) > 50:
            self.progress_updates = self.progress_updates[-50:]
    
    def calculate_execution_efficiency(self) -> Optional[float]:
        """Calculate execution efficiency score"""
        if not self.estimated_duration or not self.actual_duration:
            return None
        
        estimated_seconds = self.estimated_duration.total_seconds()
        actual_seconds = self.actual_duration.total_seconds()
        
        if actual_seconds == 0:
            return 1.0
        
        # Efficiency = estimated / actual (>1.0 is better than expected)
        return estimated_seconds / actual_seconds


class WorkflowProgress(PRSMBaseModel):
    """Comprehensive workflow progress tracking"""
    workflow_id: UUID
    workflow_name: str
    user_id: str
    
    # Overall progress
    status: ProgressStatus = Field(default=ProgressStatus.NOT_STARTED)
    overall_completion_percentage: float = Field(ge=0.0, le=100.0, default=0.0)
    
    # Step tracking
    step_progress: Dict[UUID, StepProgress] = Field(default_factory=dict)
    current_step_id: Optional[UUID] = None
    completed_steps: List[UUID] = Field(default_factory=list)
    failed_steps: List[UUID] = Field(default_factory=list)
    
    # Timing information
    scheduled_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    estimated_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    time_remaining: Optional[timedelta] = None
    
    # Performance metrics
    total_steps: int = Field(default=0)
    completed_step_count: int = Field(default=0)
    failed_step_count: int = Field(default=0)
    average_step_duration: Optional[float] = None
    throughput: Optional[float] = None  # steps per hour
    
    # Resource tracking
    total_resource_usage: Dict[ResourceType, float] = Field(default_factory=dict)
    peak_resource_usage: Dict[ResourceType, float] = Field(default_factory=dict)
    resource_efficiency: Dict[ResourceType, float] = Field(default_factory=dict)
    
    # Cost tracking
    estimated_cost: float = Field(default=0.0)
    actual_cost: float = Field(default=0.0)
    cost_efficiency: Optional[float] = None
    
    # SLA and quality metrics
    sla_target_completion: Optional[datetime] = None
    sla_compliance: Optional[bool] = None
    quality_score: Optional[float] = None
    
    # Progress milestones
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = Field(default_factory=list)
    optimization_opportunities: List[str] = Field(default_factory=list)
    
    # Real-time updates
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    update_frequency: timedelta = Field(default=timedelta(seconds=30))
    
    def calculate_overall_progress(self):
        """Calculate overall workflow completion percentage"""
        if not self.step_progress:
            self.overall_completion_percentage = 0.0
            return
        
        # Weight by estimated duration if available
        total_weight = 0.0
        weighted_progress = 0.0
        
        for step_prog in self.step_progress.values():
            weight = 1.0  # Default equal weight
            if step_prog.estimated_duration:
                weight = step_prog.estimated_duration.total_seconds()
            
            total_weight += weight
            weighted_progress += (step_prog.completion_percentage / 100.0) * weight
        
        if total_weight > 0:
            self.overall_completion_percentage = (weighted_progress / total_weight) * 100.0
        else:
            # Fallback to simple average
            progress_sum = sum(sp.completion_percentage for sp in self.step_progress.values())
            self.overall_completion_percentage = progress_sum / len(self.step_progress)
    
    def estimate_time_remaining(self):
        """Estimate time remaining for workflow completion"""
        if self.overall_completion_percentage >= 100.0:
            self.time_remaining = timedelta(0)
            return
        
        if not self.actual_start or self.overall_completion_percentage <= 0:
            self.time_remaining = self.estimated_duration
            return
        
        elapsed = datetime.now(timezone.utc) - self.actual_start
        progress_ratio = self.overall_completion_percentage / 100.0
        
        if progress_ratio > 0:
            estimated_total = elapsed / progress_ratio
            self.time_remaining = estimated_total - elapsed
        else:
            self.time_remaining = self.estimated_duration
    
    def check_sla_compliance(self):
        """Check if workflow is meeting SLA targets"""
        if not self.sla_target_completion:
            return
        
        now = datetime.now(timezone.utc)
        
        if self.status == ProgressStatus.COMPLETED:
            # Check if completed on time
            self.sla_compliance = (
                self.actual_completion and 
                self.actual_completion <= self.sla_target_completion
            )
        else:
            # Check if on track to complete on time
            if self.time_remaining:
                projected_completion = now + self.time_remaining
                self.sla_compliance = projected_completion <= self.sla_target_completion
    
    def detect_bottlenecks(self):
        """Detect performance bottlenecks in workflow execution"""
        self.bottlenecks.clear()
        
        if not self.step_progress:
            return
        
        # Find steps taking longer than expected
        for step_id, step_prog in self.step_progress.items():
            if (step_prog.estimated_duration and 
                step_prog.actual_duration and
                step_prog.actual_duration > step_prog.estimated_duration * 1.5):
                
                bottleneck = {
                    "step_id": str(step_id),
                    "step_name": step_prog.step_name,
                    "type": "slow_execution",
                    "severity": AlertSeverity.WARNING.value,
                    "estimated_duration": step_prog.estimated_duration.total_seconds(),
                    "actual_duration": step_prog.actual_duration.total_seconds(),
                    "delay_factor": step_prog.actual_duration / step_prog.estimated_duration
                }
                self.bottlenecks.append(bottleneck)
        
        # Find resource contention
        for resource_type, usage in self.peak_resource_usage.items():
            if usage > 0.9:  # >90% resource utilization
                bottleneck = {
                    "type": "resource_contention",
                    "resource_type": resource_type.value,
                    "severity": AlertSeverity.WARNING.value,
                    "peak_usage": usage,
                    "description": f"High {resource_type.value} utilization ({usage*100:.1f}%)"
                }
                self.bottlenecks.append(bottleneck)


class PerformanceMetric(PRSMBaseModel):
    """Performance metric tracking"""
    metric_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    step_id: Optional[UUID] = None
    
    # Metric details
    metric_type: MetricType
    metric_name: str
    value: float
    unit: str = Field(default="")
    
    # Context
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_type: Optional[AgentType] = None
    resource_type: Optional[ResourceType] = None
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)


class ProgressAlert(PRSMBaseModel):
    """Progress monitoring alert"""
    alert_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    step_id: Optional[UUID] = None
    
    # Alert details
    severity: AlertSeverity
    alert_type: str
    title: str
    message: str
    
    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Response tracking
    acknowledged_by: Optional[str] = None
    resolution_actions: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProgressTracker(TimestampMixin):
    """
    Progress Tracking System for Scheduled Jobs
    
    Comprehensive monitoring and analytics system for workflow execution
    with real-time updates, performance metrics, and SLA tracking.
    """
    
    def __init__(self):
        super().__init__()
        
        # Progress tracking storage
        self.workflow_progress: Dict[UUID, WorkflowProgress] = {}
        self.performance_metrics: List[PerformanceMetric] = []
        self.progress_alerts: List[ProgressAlert] = []
        
        # Analytics and aggregations
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.success_rates: Dict[str, float] = defaultdict(float)
        self.average_durations: Dict[str, float] = defaultdict(float)
        
        # Real-time monitoring
        self.active_subscriptions: Dict[UUID, List[Callable]] = defaultdict(list)
        self.update_intervals: Dict[UUID, timedelta] = {}
        
        # Configuration
        self.default_update_interval = timedelta(seconds=30)
        self.metrics_retention_days = 30
        self.alert_retention_days = 7
        self.performance_window_hours = 24
        
        # Statistics
        self.tracking_statistics: Dict[str, Any] = defaultdict(int)
        
        self._start_monitoring_tasks()
        
        logger.info("ProgressTracker initialized")
    
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        # In production, these would run as async background tasks
        pass
    
    async def start_workflow_tracking(
        self,
        workflow: ScheduledWorkflow,
        sla_target_completion: Optional[datetime] = None
    ) -> WorkflowProgress:
        """
        Start tracking progress for a workflow
        
        Args:
            workflow: Workflow to track
            sla_target_completion: SLA target completion time
            
        Returns:
            Workflow progress tracker
        """
        try:
            # Create workflow progress tracker
            progress = WorkflowProgress(
                workflow_id=workflow.workflow_id,
                workflow_name=workflow.workflow_name,
                user_id=workflow.user_id,
                scheduled_start=workflow.scheduled_start,
                estimated_duration=workflow.get_critical_path_duration(),
                sla_target_completion=sla_target_completion,
                total_steps=len(workflow.steps)
            )
            
            # Initialize step progress trackers
            for step in workflow.steps:
                step_progress = StepProgress(
                    step_id=step.step_id,
                    step_name=step.step_name,
                    agent_type=step.agent_type,
                    estimated_duration=step.estimated_duration,
                    max_attempts=step.max_retries + 1
                )
                progress.step_progress[step.step_id] = step_progress
            
            # Calculate estimated completion
            if progress.scheduled_start and progress.estimated_duration:
                progress.estimated_completion = progress.scheduled_start + progress.estimated_duration
            
            # Store progress tracker
            self.workflow_progress[workflow.workflow_id] = progress
            
            # Update statistics
            self.tracking_statistics["workflows_tracked"] += 1
            
            logger.info(
                "Started workflow tracking",
                workflow_id=str(workflow.workflow_id),
                workflow_name=workflow.workflow_name,
                total_steps=len(workflow.steps)
            )
            
            return progress
            
        except Exception as e:
            logger.error("Error starting workflow tracking", error=str(e))
            raise
    
    async def update_workflow_status(
        self,
        workflow_id: UUID,
        status: ProgressStatus,
        current_step_id: Optional[UUID] = None
    ):
        """Update overall workflow status"""
        try:
            if workflow_id not in self.workflow_progress:
                logger.warning("Workflow not tracked", workflow_id=str(workflow_id))
                return
            
            progress = self.workflow_progress[workflow_id]
            old_status = progress.status
            progress.status = status
            progress.current_step_id = current_step_id
            progress.last_update = datetime.now(timezone.utc)
            
            # Update timing based on status
            now = datetime.now(timezone.utc)
            
            if status == ProgressStatus.RUNNING and not progress.actual_start:
                progress.actual_start = now
            elif status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]:
                progress.actual_completion = now
                if progress.actual_start:
                    progress.actual_duration = now - progress.actual_start
            
            # Recalculate progress and estimates
            progress.calculate_overall_progress()
            progress.estimate_time_remaining()
            progress.check_sla_compliance()
            progress.detect_bottlenecks()
            
            # Generate alerts for status changes
            if old_status != status:
                await self._generate_status_alert(progress, old_status, status)
            
            # Notify subscribers
            await self._notify_subscribers(workflow_id, "status_change", {
                "old_status": old_status.value,
                "new_status": status.value,
                "progress": progress.overall_completion_percentage
            })
            
            logger.info(
                "Workflow status updated",
                workflow_id=str(workflow_id),
                old_status=old_status.value,
                new_status=status.value,
                progress=progress.overall_completion_percentage
            )
            
        except Exception as e:
            logger.error("Error updating workflow status", error=str(e))
    
    async def update_step_progress(
        self,
        workflow_id: UUID,
        step_id: UUID,
        status: Optional[StepProgressStatus] = None,
        completion_percentage: Optional[float] = None,
        current_operation: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        resource_usage: Optional[Dict[ResourceType, float]] = None
    ):
        """Update individual step progress"""
        try:
            if workflow_id not in self.workflow_progress:
                logger.warning("Workflow not tracked", workflow_id=str(workflow_id))
                return
            
            progress = self.workflow_progress[workflow_id]
            
            if step_id not in progress.step_progress:
                logger.warning("Step not tracked", step_id=str(step_id))
                return
            
            step_progress = progress.step_progress[step_id]
            old_status = step_progress.status
            
            # Update step status
            if status:
                step_progress.status = status
                
                # Update timing based on status
                now = datetime.now(timezone.utc)
                
                if status == StepProgressStatus.EXECUTING and not step_progress.actual_start:
                    step_progress.actual_start = now
                    step_progress.attempts += 1
                elif status in [StepProgressStatus.COMPLETED, StepProgressStatus.FAILED]:
                    step_progress.actual_completion = now
                    if step_progress.actual_start:
                        step_progress.actual_duration = now - step_progress.actual_start
                    
                    # Update workflow step counts
                    if status == StepProgressStatus.COMPLETED:
                        if step_id not in progress.completed_steps:
                            progress.completed_steps.append(step_id)
                            progress.completed_step_count += 1
                    elif status == StepProgressStatus.FAILED:
                        if step_id not in progress.failed_steps:
                            progress.failed_steps.append(step_id)
                            progress.failed_step_count += 1
            
            # Update progress percentage
            if completion_percentage is not None:
                step_progress.update_progress(completion_percentage, current_operation, metrics)
            
            # Update resource usage
            if resource_usage:
                step_progress.resource_usage.update(resource_usage)
                
                # Track peak usage
                for resource_type, usage in resource_usage.items():
                    current_peak = step_progress.peak_resource_usage.get(resource_type, 0.0)
                    step_progress.peak_resource_usage[resource_type] = max(current_peak, usage)
                    
                    # Update workflow peak usage
                    workflow_peak = progress.peak_resource_usage.get(resource_type, 0.0)
                    progress.peak_resource_usage[resource_type] = max(workflow_peak, usage)
            
            # Store performance metrics
            if metrics:
                await self._store_step_metrics(workflow_id, step_id, metrics)
            
            # Recalculate overall progress
            progress.calculate_overall_progress()
            progress.estimate_time_remaining()
            progress.detect_bottlenecks()
            
            # Generate alerts for significant events
            if old_status != step_progress.status:
                await self._generate_step_alert(progress, step_progress, old_status)
            
            # Notify subscribers
            await self._notify_subscribers(workflow_id, "step_progress", {
                "step_id": str(step_id),
                "step_name": step_progress.step_name,
                "status": step_progress.status.value,
                "completion_percentage": step_progress.completion_percentage,
                "current_operation": step_progress.current_operation
            })
            
            # Update workflow last update time
            progress.last_update = datetime.now(timezone.utc)
            
            logger.debug(
                "Step progress updated",
                workflow_id=str(workflow_id),
                step_id=str(step_id),
                status=step_progress.status.value,
                completion=step_progress.completion_percentage
            )
            
        except Exception as e:
            logger.error("Error updating step progress", error=str(e))
    
    async def _store_step_metrics(
        self,
        workflow_id: UUID,
        step_id: UUID,
        metrics: Dict[str, Any]
    ):
        """Store performance metrics for a step"""
        try:
            step_progress = self.workflow_progress[workflow_id].step_progress[step_id]
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Determine metric type
                    metric_type = MetricType.EXECUTION_TIME
                    unit = ""
                    
                    if "time" in metric_name.lower() or "duration" in metric_name.lower():
                        metric_type = MetricType.EXECUTION_TIME
                        unit = "seconds"
                    elif "cost" in metric_name.lower():
                        metric_type = MetricType.COST
                        unit = "ftns"
                    elif "rate" in metric_name.lower() or "throughput" in metric_name.lower():
                        metric_type = MetricType.THROUGHPUT
                        unit = "items/sec"
                    elif "usage" in metric_name.lower():
                        metric_type = MetricType.RESOURCE_USAGE
                        unit = "percent"
                    
                    # Create metric
                    metric = PerformanceMetric(
                        workflow_id=workflow_id,
                        step_id=step_id,
                        metric_type=metric_type,
                        metric_name=metric_name,
                        value=float(value),
                        unit=unit,
                        agent_type=step_progress.agent_type
                    )
                    
                    self.performance_metrics.append(metric)
                    
                    # Update historical data
                    key = f"{step_progress.agent_type.value}_{metric_name}"
                    self.performance_history[key].append(float(value))
                    
                    # Keep only recent history
                    if len(self.performance_history[key]) > 1000:
                        self.performance_history[key] = self.performance_history[key][-1000:]
            
        except Exception as e:
            logger.error("Error storing step metrics", error=str(e))
    
    async def _generate_status_alert(
        self,
        progress: WorkflowProgress,
        old_status: ProgressStatus,
        new_status: ProgressStatus
    ):
        """Generate alert for workflow status change"""
        try:
            severity = AlertSeverity.INFO
            alert_type = "status_change"
            
            if new_status == ProgressStatus.FAILED:
                severity = AlertSeverity.ERROR
                alert_type = "workflow_failed"
            elif new_status == ProgressStatus.TIMEOUT:
                severity = AlertSeverity.ERROR
                alert_type = "workflow_timeout"
            elif new_status == ProgressStatus.COMPLETED:
                severity = AlertSeverity.INFO
                alert_type = "workflow_completed"
                
                # Check SLA compliance
                if progress.sla_compliance is False:
                    severity = AlertSeverity.WARNING
                    alert_type = "sla_violation"
            
            alert = ProgressAlert(
                workflow_id=progress.workflow_id,
                severity=severity,
                alert_type=alert_type,
                title=f"Workflow {new_status.value.title()}",
                message=f"Workflow '{progress.workflow_name}' changed from {old_status.value} to {new_status.value}",
                metadata={
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "completion_percentage": progress.overall_completion_percentage,
                    "sla_compliance": progress.sla_compliance
                }
            )
            
            self.progress_alerts.append(alert)
            
        except Exception as e:
            logger.error("Error generating status alert", error=str(e))
    
    async def _generate_step_alert(
        self,
        workflow_progress: WorkflowProgress,
        step_progress: StepProgress,
        old_status: StepProgressStatus
    ):
        """Generate alert for step status change"""
        try:
            new_status = step_progress.status
            
            if new_status == StepProgressStatus.FAILED:
                alert = ProgressAlert(
                    workflow_id=workflow_progress.workflow_id,
                    step_id=step_progress.step_id,
                    severity=AlertSeverity.ERROR,
                    alert_type="step_failed",
                    title=f"Step Failed: {step_progress.step_name}",
                    message=f"Step '{step_progress.step_name}' failed after {step_progress.attempts} attempts",
                    metadata={
                        "step_name": step_progress.step_name,
                        "attempts": step_progress.attempts,
                        "agent_type": step_progress.agent_type.value
                    }
                )
                self.progress_alerts.append(alert)
            
            elif new_status == StepProgressStatus.RETRYING:
                alert = ProgressAlert(
                    workflow_id=workflow_progress.workflow_id,
                    step_id=step_progress.step_id,
                    severity=AlertSeverity.WARNING,
                    alert_type="step_retry",
                    title=f"Step Retrying: {step_progress.step_name}",
                    message=f"Step '{step_progress.step_name}' is retrying (attempt {step_progress.attempts})",
                    metadata={
                        "step_name": step_progress.step_name,
                        "attempts": step_progress.attempts,
                        "max_attempts": step_progress.max_attempts
                    }
                )
                self.progress_alerts.append(alert)
        
        except Exception as e:
            logger.error("Error generating step alert", error=str(e))
    
    async def _notify_subscribers(
        self,
        workflow_id: UUID,
        event_type: str,
        data: Dict[str, Any]
    ):
        """Notify progress subscribers of updates"""
        try:
            if workflow_id in self.active_subscriptions:
                event_data = {
                    "workflow_id": str(workflow_id),
                    "event_type": event_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": data
                }
                
                # Call all subscriber callbacks
                for callback in self.active_subscriptions[workflow_id]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event_data)
                        else:
                            callback(event_data)
                    except Exception as e:
                        logger.warning("Error notifying subscriber", error=str(e))
        
        except Exception as e:
            logger.error("Error notifying subscribers", error=str(e))
    
    def subscribe_to_progress(
        self,
        workflow_id: UUID,
        callback: Callable,
        update_interval: Optional[timedelta] = None
    ):
        """Subscribe to real-time progress updates"""
        self.active_subscriptions[workflow_id].append(callback)
        
        if update_interval:
            self.update_intervals[workflow_id] = update_interval
        
        logger.info(
            "Progress subscription added",
            workflow_id=str(workflow_id),
            subscriber_count=len(self.active_subscriptions[workflow_id])
        )
    
    def unsubscribe_from_progress(self, workflow_id: UUID, callback: Callable):
        """Unsubscribe from progress updates"""
        if workflow_id in self.active_subscriptions:
            try:
                self.active_subscriptions[workflow_id].remove(callback)
                if not self.active_subscriptions[workflow_id]:
                    del self.active_subscriptions[workflow_id]
                    if workflow_id in self.update_intervals:
                        del self.update_intervals[workflow_id]
            except ValueError:
                pass  # Callback not in list
    
    def get_workflow_progress(self, workflow_id: UUID) -> Optional[WorkflowProgress]:
        """Get current workflow progress"""
        return self.workflow_progress.get(workflow_id)
    
    def get_step_progress(self, workflow_id: UUID, step_id: UUID) -> Optional[StepProgress]:
        """Get current step progress"""
        workflow_progress = self.workflow_progress.get(workflow_id)
        if workflow_progress:
            return workflow_progress.step_progress.get(step_id)
        return None
    
    def get_workflow_analytics(
        self,
        workflow_id: Optional[UUID] = None,
        user_id: Optional[str] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get workflow analytics and performance insights"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
            
            # Filter workflows
            workflows = []
            if workflow_id:
                if workflow_id in self.workflow_progress:
                    workflows = [self.workflow_progress[workflow_id]]
            else:
                workflows = list(self.workflow_progress.values())
                if user_id:
                    workflows = [wp for wp in workflows if wp.user_id == user_id]
            
            # Calculate analytics
            total_workflows = len(workflows)
            completed_workflows = len([wp for wp in workflows if wp.status == ProgressStatus.COMPLETED])
            failed_workflows = len([wp for wp in workflows if wp.status == ProgressStatus.FAILED])
            
            success_rate = (completed_workflows / total_workflows) * 100 if total_workflows > 0 else 0
            
            # Duration statistics
            completed_durations = []
            for wp in workflows:
                if wp.actual_duration:
                    completed_durations.append(wp.actual_duration.total_seconds())
            
            avg_duration = statistics.mean(completed_durations) if completed_durations else 0
            
            # SLA compliance
            sla_compliant = len([wp for wp in workflows if wp.sla_compliance is True])
            sla_violations = len([wp for wp in workflows if wp.sla_compliance is False])
            sla_compliance_rate = (
                (sla_compliant / (sla_compliant + sla_violations)) * 100 
                if (sla_compliant + sla_violations) > 0 else 0
            )
            
            # Cost analytics
            total_cost = sum(wp.actual_cost for wp in workflows)
            avg_cost = total_cost / total_workflows if total_workflows > 0 else 0
            
            # Resource utilization
            resource_stats = {}
            for resource_type in ResourceType:
                usages = [
                    wp.peak_resource_usage.get(resource_type, 0) 
                    for wp in workflows 
                    if resource_type in wp.peak_resource_usage
                ]
                if usages:
                    resource_stats[resource_type.value] = {
                        "avg_usage": statistics.mean(usages),
                        "max_usage": max(usages),
                        "min_usage": min(usages)
                    }
            
            # Recent alerts
            recent_alerts = [
                alert for alert in self.progress_alerts
                if alert.created_at >= cutoff_time
            ]
            
            return {
                "time_window_hours": time_window_hours,
                "total_workflows": total_workflows,
                "completed_workflows": completed_workflows,
                "failed_workflows": failed_workflows,
                "success_rate_percentage": success_rate,
                "average_duration_seconds": avg_duration,
                "sla_compliance_rate_percentage": sla_compliance_rate,
                "sla_violations": sla_violations,
                "total_cost": total_cost,
                "average_cost": avg_cost,
                "resource_utilization": resource_stats,
                "recent_alerts": len(recent_alerts),
                "alert_breakdown": {
                    severity.value: len([a for a in recent_alerts if a.severity == severity])
                    for severity in AlertSeverity
                }
            }
            
        except Exception as e:
            logger.error("Error generating workflow analytics", error=str(e))
            return {"error": str(e)}
    
    def get_performance_trends(
        self,
        metric_type: MetricType,
        agent_type: Optional[AgentType] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance trends for specific metrics"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
            
            # Filter metrics
            filtered_metrics = [
                metric for metric in self.performance_metrics
                if (metric.metric_type == metric_type and
                    metric.timestamp >= cutoff_time and
                    (agent_type is None or metric.agent_type == agent_type))
            ]
            
            if not filtered_metrics:
                return {"message": "No metrics found for specified criteria"}
            
            # Calculate trends
            values = [metric.value for metric in filtered_metrics]
            timestamps = [metric.timestamp for metric in filtered_metrics]
            
            # Group by hour for trending
            hourly_data = defaultdict(list)
            for metric in filtered_metrics:
                hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
                hourly_data[hour_key].append(metric.value)
            
            # Calculate hourly averages
            hourly_averages = {}
            for hour, values_in_hour in hourly_data.items():
                hourly_averages[hour.isoformat()] = statistics.mean(values_in_hour)
            
            return {
                "metric_type": metric_type.value,
                "agent_type": agent_type.value if agent_type else "all",
                "time_window_hours": time_window_hours,
                "total_measurements": len(values),
                "average_value": statistics.mean(values),
                "min_value": min(values),
                "max_value": max(values),
                "median_value": statistics.median(values),
                "std_deviation": statistics.stdev(values) if len(values) > 1 else 0,
                "hourly_trends": hourly_averages,
                "unit": filtered_metrics[0].unit if filtered_metrics else ""
            }
            
        except Exception as e:
            logger.error("Error generating performance trends", error=str(e))
            return {"error": str(e)}
    
    def get_active_alerts(
        self,
        workflow_id: Optional[UUID] = None,
        severity_filter: Optional[AlertSeverity] = None,
        unacknowledged_only: bool = False
    ) -> List[ProgressAlert]:
        """Get active progress alerts"""
        alerts = self.progress_alerts
        
        # Apply filters
        if workflow_id:
            alerts = [alert for alert in alerts if alert.workflow_id == workflow_id]
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        if unacknowledged_only:
            alerts = [alert for alert in alerts if alert.acknowledged_at is None]
        
        # Sort by severity and creation time
        severity_order = {
            AlertSeverity.CRITICAL: 4,
            AlertSeverity.ERROR: 3,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 1
        }
        
        alerts.sort(
            key=lambda a: (severity_order.get(a.severity, 0), a.created_at),
            reverse=True
        )
        
        return alerts
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics"""
        active_workflows = len([
            wp for wp in self.workflow_progress.values()
            if wp.status in [ProgressStatus.RUNNING, ProgressStatus.INITIALIZING]
        ])
        
        total_alerts = len(self.progress_alerts)
        unresolved_alerts = len([
            alert for alert in self.progress_alerts
            if alert.resolved_at is None
        ])
        
        return {
            "tracking_statistics": dict(self.tracking_statistics),
            "active_workflows": active_workflows,
            "total_workflows_tracked": len(self.workflow_progress),
            "total_performance_metrics": len(self.performance_metrics),
            "total_alerts": total_alerts,
            "unresolved_alerts": unresolved_alerts,
            "active_subscriptions": len(self.active_subscriptions),
            "metrics_retention_days": self.metrics_retention_days,
            "alert_retention_days": self.alert_retention_days
        }


# Global instance for easy access
_progress_tracker = None

def get_progress_tracker() -> ProgressTracker:
    """Get global progress tracker instance"""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ProgressTracker()
    return _progress_tracker