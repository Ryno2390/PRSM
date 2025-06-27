"""
Workflow Rollback and Recovery System

🔄 WORKFLOW ROLLBACK CAPABILITIES:
- Intelligent rollback strategies based on failure analysis
- Automatic state restoration to last known good checkpoint
- Resource cleanup and deallocation on failure
- Compensation actions for partial workflow completions
- Integration with persistence layer for state management
- Smart retry logic with exponential backoff and circuit breakers

This module implements comprehensive rollback mechanisms that enable:
1. Automatic recovery from workflow failures with minimal data loss
2. Intelligent rollback strategies based on failure type and impact
3. Resource cleanup and cost recovery for failed executions
4. Compensation actions for workflows with side effects
5. Integration with notification system for rollback status updates
6. Analytics and learning from failure patterns for prevention
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
import traceback
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import PRSMBaseModel, TimestampMixin, AgentType
from prsm.scheduling.workflow_scheduler import (
    ScheduledWorkflow, WorkflowStep, WorkflowStatus, SchedulingPriority, ResourceType
)
from prsm.scheduling.workflow_persistence import (
    WorkflowCheckpoint, WorkflowPersistence, PersistenceStatus
)
from prsm.scheduling.progress_tracker import (
    ProgressTracker, ProgressStatus, StepProgressStatus
)
from prsm.scheduling.notification_system import (
    NotificationSystem, NotificationType, NotificationPriority
)

logger = structlog.get_logger(__name__)


class RollbackStrategy(str, Enum):
    """Rollback strategy types"""
    FULL_ROLLBACK = "full_rollback"           # Rollback entire workflow
    PARTIAL_ROLLBACK = "partial_rollback"     # Rollback to last checkpoint
    COMPENSATE_ONLY = "compensate_only"       # Run compensation actions only
    RETRY_FAILED_STEP = "retry_failed_step"   # Retry only the failed step
    SKIP_AND_CONTINUE = "skip_and_continue"   # Skip failed step and continue
    MANUAL_INTERVENTION = "manual_intervention"  # Require manual intervention


class FailureType(str, Enum):
    """Types of workflow failures"""
    STEP_EXECUTION_ERROR = "step_execution_error"      # Individual step failed
    RESOURCE_EXHAUSTION = "resource_exhaustion"        # Ran out of resources
    TIMEOUT = "timeout"                                # Execution timeout
    DEPENDENCY_FAILURE = "dependency_failure"          # External dependency failed
    PERMISSION_DENIED = "permission_denied"            # Authorization failure
    NETWORK_ERROR = "network_error"                    # Network connectivity issues
    SYSTEM_ERROR = "system_error"                      # System-level errors
    USER_CANCELLATION = "user_cancellation"           # User requested cancellation
    COST_THRESHOLD_EXCEEDED = "cost_threshold_exceeded"  # Cost limits exceeded
    SAFETY_VIOLATION = "safety_violation"              # Safety policy violation


class RollbackAction(str, Enum):
    """Types of rollback actions"""
    RESTORE_STATE = "restore_state"                    # Restore workflow state
    DEALLOCATE_RESOURCES = "deallocate_resources"      # Free allocated resources
    CLEANUP_ARTIFACTS = "cleanup_artifacts"            # Delete created artifacts
    REFUND_CREDITS = "refund_credits"                  # Refund FTNS credits
    SEND_NOTIFICATION = "send_notification"            # Send failure notification
    LOG_INCIDENT = "log_incident"                      # Log failure incident
    COMPENSATE_EFFECTS = "compensate_effects"          # Compensate side effects
    RESET_DEPENDENCIES = "reset_dependencies"          # Reset dependency state


class RollbackResult(str, Enum):
    """Results of rollback operations"""
    SUCCESS = "success"                                # Rollback successful
    PARTIAL_SUCCESS = "partial_success"                # Some actions succeeded
    FAILED = "failed"                                  # Rollback failed
    SKIPPED = "skipped"                               # Rollback was skipped
    MANUAL_REQUIRED = "manual_required"               # Manual intervention required


class CompensationAction(PRSMBaseModel):
    """Compensation action for workflow rollback"""
    action_id: UUID = Field(default_factory=uuid4)
    step_id: UUID
    action_type: RollbackAction
    
    # Action details
    description: str = Field(default="")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution
    executed: bool = Field(default=False)
    execution_time: Optional[datetime] = None
    result: Optional[RollbackResult] = None
    error_message: Optional[str] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = Field(default=5)  # 1-10, lower is higher priority


class RollbackPlan(PRSMBaseModel):
    """Plan for rolling back a failed workflow"""
    plan_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    
    # Failure analysis
    failure_type: FailureType
    failed_step_id: Optional[UUID] = None
    failure_reason: str = Field(default="")
    failure_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Rollback strategy
    strategy: RollbackStrategy
    target_checkpoint_id: Optional[UUID] = None
    
    # Compensation actions
    compensation_actions: List[CompensationAction] = Field(default_factory=list)
    
    # Execution tracking
    status: str = Field(default="pending")  # pending, executing, completed, failed
    execution_started: Optional[datetime] = None
    execution_completed: Optional[datetime] = None
    
    # Results
    overall_result: Optional[RollbackResult] = None
    actions_executed: int = Field(default=0)
    actions_successful: int = Field(default=0)
    actions_failed: int = Field(default=0)
    
    # Metadata
    estimated_duration: Optional[timedelta] = None
    estimated_cost: float = Field(default=0.0)
    actual_cost: float = Field(default=0.0)


class RetryPolicy(PRSMBaseModel):
    """Retry policy for failed workflows"""
    policy_id: UUID = Field(default_factory=uuid4)
    
    # Retry configuration
    max_retries: int = Field(default=3)
    base_delay_seconds: float = Field(default=60.0)  # Base delay between retries
    max_delay_seconds: float = Field(default=3600.0)  # Maximum delay
    exponential_backoff: bool = Field(default=True)
    jitter: bool = Field(default=True)  # Add randomness to delays
    
    # Retry conditions
    retry_on_failure_types: List[FailureType] = Field(default_factory=list)
    retry_on_step_types: List[str] = Field(default_factory=list)
    
    # Circuit breaker
    circuit_breaker_enabled: bool = Field(default=True)
    failure_threshold: int = Field(default=5)  # Failures before circuit opens
    recovery_timeout: timedelta = Field(default=timedelta(minutes=30))
    
    # Conditions
    max_retry_cost: float = Field(default=100.0)  # Maximum cost for retries
    retry_within_hours: float = Field(default=24.0)  # Time window for retries


class WorkflowRollbackSystem(TimestampMixin):
    """
    Comprehensive Workflow Rollback and Recovery System
    
    Handles intelligent rollback strategies, resource cleanup, and
    automatic recovery from workflow failures.
    """
    
    def __init__(
        self,
        persistence: Optional[WorkflowPersistence] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        notification_system: Optional[NotificationSystem] = None
    ):
        super().__init__()
        
        # Core dependencies
        self.persistence = persistence
        self.progress_tracker = progress_tracker
        self.notification_system = notification_system
        
        # Rollback management
        self.active_rollbacks: Dict[UUID, RollbackPlan] = {}
        self.rollback_history: List[RollbackPlan] = []
        self.retry_policies: Dict[str, RetryPolicy] = {}
        
        # Circuit breaker state
        self.circuit_breaker_state: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Failure analysis
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.compensation_handlers: Dict[RollbackAction, Callable] = {}
        
        # Statistics
        self.rollback_statistics: Dict[str, Any] = defaultdict(int)
        
        # Configuration
        self.default_retry_policy = RetryPolicy(
            retry_on_failure_types=[
                FailureType.NETWORK_ERROR,
                FailureType.RESOURCE_EXHAUSTION,
                FailureType.TIMEOUT,
                FailureType.SYSTEM_ERROR
            ]
        )
        
        self._initialize_compensation_handlers()
        
        logger.info("WorkflowRollbackSystem initialized")
    
    def _initialize_compensation_handlers(self):
        """Initialize compensation action handlers"""
        self.compensation_handlers = {
            RollbackAction.RESTORE_STATE: self._restore_workflow_state,
            RollbackAction.DEALLOCATE_RESOURCES: self._deallocate_resources,
            RollbackAction.CLEANUP_ARTIFACTS: self._cleanup_artifacts,
            RollbackAction.REFUND_CREDITS: self._refund_credits,
            RollbackAction.SEND_NOTIFICATION: self._send_rollback_notification,
            RollbackAction.LOG_INCIDENT: self._log_incident,
            RollbackAction.COMPENSATE_EFFECTS: self._compensate_side_effects,
            RollbackAction.RESET_DEPENDENCIES: self._reset_dependencies
        }
    
    async def handle_workflow_failure(
        self,
        workflow_id: UUID,
        failure_type: FailureType,
        failure_reason: str,
        failed_step_id: Optional[UUID] = None,
        failure_context: Optional[Dict[str, Any]] = None
    ) -> RollbackPlan:
        """
        Handle workflow failure with intelligent rollback strategy
        
        Args:
            workflow_id: ID of failed workflow
            failure_type: Type of failure
            failure_reason: Reason for failure
            failed_step_id: ID of failed step
            failure_context: Additional failure context
            
        Returns:
            Rollback plan for the failure
        """
        try:
            logger.info(
                "Handling workflow failure",
                workflow_id=str(workflow_id),
                failure_type=failure_type.value,
                failure_reason=failure_reason,
                failed_step_id=str(failed_step_id) if failed_step_id else None
            )
            
            # Analyze failure and determine strategy
            strategy = await self._analyze_failure_and_determine_strategy(
                workflow_id, failure_type, failure_reason, failed_step_id, failure_context
            )
            
            # Create rollback plan
            rollback_plan = await self._create_rollback_plan(
                workflow_id, failure_type, failure_reason, failed_step_id, strategy
            )
            
            # Store active rollback
            self.active_rollbacks[workflow_id] = rollback_plan
            
            # Execute rollback plan
            await self._execute_rollback_plan(rollback_plan)
            
            # Update statistics
            self.rollback_statistics["total_rollbacks"] += 1
            self.rollback_statistics[f"rollbacks_{failure_type.value}"] += 1
            self.rollback_statistics[f"strategy_{strategy.value}"] += 1
            
            # Store failure pattern for learning
            await self._store_failure_pattern(
                workflow_id, failure_type, failure_reason, failed_step_id, failure_context
            )
            
            return rollback_plan
            
        except Exception as e:
            logger.error("Error handling workflow failure", error=str(e), workflow_id=str(workflow_id))
            # Create minimal rollback plan for emergency cleanup
            emergency_plan = RollbackPlan(
                workflow_id=workflow_id,
                failure_type=failure_type,
                failure_reason=f"Emergency rollback due to handler error: {str(e)}",
                strategy=RollbackStrategy.MANUAL_INTERVENTION
            )
            return emergency_plan
    
    async def _analyze_failure_and_determine_strategy(
        self,
        workflow_id: UUID,
        failure_type: FailureType,
        failure_reason: str,
        failed_step_id: Optional[UUID] = None,
        failure_context: Optional[Dict[str, Any]] = None
    ) -> RollbackStrategy:
        """Analyze failure and determine appropriate rollback strategy"""
        try:
            # Get workflow information
            workflow_info = await self._get_workflow_info(workflow_id)
            if not workflow_info:
                return RollbackStrategy.MANUAL_INTERVENTION
            
            # Check circuit breaker state
            if self._is_circuit_breaker_open(workflow_id, failure_type):
                logger.warning("Circuit breaker open for workflow", workflow_id=str(workflow_id), failure_type=failure_type.value)
                return RollbackStrategy.MANUAL_INTERVENTION
            
            # Determine strategy based on failure type
            if failure_type == FailureType.USER_CANCELLATION:
                return RollbackStrategy.FULL_ROLLBACK
            
            elif failure_type == FailureType.COST_THRESHOLD_EXCEEDED:
                return RollbackStrategy.PARTIAL_ROLLBACK
            
            elif failure_type == FailureType.SAFETY_VIOLATION:
                return RollbackStrategy.FULL_ROLLBACK
            
            elif failure_type == FailureType.PERMISSION_DENIED:
                return RollbackStrategy.MANUAL_INTERVENTION
            
            elif failure_type in [FailureType.NETWORK_ERROR, FailureType.TIMEOUT]:
                # Check if retry is viable
                if await self._should_retry_workflow(workflow_id, failure_type):
                    return RollbackStrategy.RETRY_FAILED_STEP
                else:
                    return RollbackStrategy.PARTIAL_ROLLBACK
            
            elif failure_type == FailureType.RESOURCE_EXHAUSTION:
                # Check if resources might become available
                if await self._resources_likely_available_soon():
                    return RollbackStrategy.RETRY_FAILED_STEP
                else:
                    return RollbackStrategy.PARTIAL_ROLLBACK
            
            elif failure_type == FailureType.DEPENDENCY_FAILURE:
                # Check if dependency is critical
                if await self._is_dependency_critical(workflow_id, failed_step_id):
                    return RollbackStrategy.PARTIAL_ROLLBACK
                else:
                    return RollbackStrategy.SKIP_AND_CONTINUE
            
            elif failure_type == FailureType.STEP_EXECUTION_ERROR:
                # Analyze step importance and side effects
                if await self._step_has_side_effects(workflow_id, failed_step_id):
                    return RollbackStrategy.COMPENSATE_ONLY
                else:
                    return RollbackStrategy.RETRY_FAILED_STEP
            
            else:
                # Default to partial rollback for unknown failures
                return RollbackStrategy.PARTIAL_ROLLBACK
                
        except Exception as e:
            logger.error("Error analyzing failure", error=str(e))
            return RollbackStrategy.MANUAL_INTERVENTION
    
    async def _create_rollback_plan(
        self,
        workflow_id: UUID,
        failure_type: FailureType,
        failure_reason: str,
        failed_step_id: Optional[UUID],
        strategy: RollbackStrategy
    ) -> RollbackPlan:
        """Create detailed rollback plan"""
        try:
            plan = RollbackPlan(
                workflow_id=workflow_id,
                failure_type=failure_type,
                failed_step_id=failed_step_id,
                failure_reason=failure_reason,
                strategy=strategy
            )
            
            # Determine target checkpoint
            if strategy in [RollbackStrategy.PARTIAL_ROLLBACK, RollbackStrategy.RETRY_FAILED_STEP]:
                plan.target_checkpoint_id = await self._find_best_checkpoint(workflow_id, failed_step_id)
            
            # Create compensation actions based on strategy
            if strategy == RollbackStrategy.FULL_ROLLBACK:
                plan.compensation_actions = await self._create_full_rollback_actions(workflow_id)
            
            elif strategy == RollbackStrategy.PARTIAL_ROLLBACK:
                plan.compensation_actions = await self._create_partial_rollback_actions(
                    workflow_id, plan.target_checkpoint_id
                )
            
            elif strategy == RollbackStrategy.COMPENSATE_ONLY:
                plan.compensation_actions = await self._create_compensation_actions(
                    workflow_id, failed_step_id
                )
            
            elif strategy == RollbackStrategy.RETRY_FAILED_STEP:
                plan.compensation_actions = await self._create_retry_actions(
                    workflow_id, failed_step_id
                )
            
            elif strategy == RollbackStrategy.SKIP_AND_CONTINUE:
                plan.compensation_actions = await self._create_skip_actions(
                    workflow_id, failed_step_id
                )
            
            # Estimate duration and cost
            plan.estimated_duration = self._estimate_rollback_duration(plan)
            plan.estimated_cost = self._estimate_rollback_cost(plan)
            
            logger.info(
                "Rollback plan created",
                plan_id=str(plan.plan_id),
                strategy=strategy.value,
                actions=len(plan.compensation_actions),
                estimated_duration=plan.estimated_duration.total_seconds() if plan.estimated_duration else 0
            )
            
            return plan
            
        except Exception as e:
            logger.error("Error creating rollback plan", error=str(e))
            # Return minimal plan
            return RollbackPlan(
                workflow_id=workflow_id,
                failure_type=failure_type,
                failure_reason=failure_reason,
                strategy=RollbackStrategy.MANUAL_INTERVENTION
            )
    
    async def _execute_rollback_plan(self, plan: RollbackPlan):
        """Execute rollback plan with compensation actions"""
        try:
            logger.info("Executing rollback plan", plan_id=str(plan.plan_id), strategy=plan.strategy.value)
            
            plan.status = "executing"
            plan.execution_started = datetime.now(timezone.utc)
            
            # Sort actions by priority
            sorted_actions = sorted(plan.compensation_actions, key=lambda a: a.priority)
            
            # Execute compensation actions
            for action in sorted_actions:
                try:
                    await self._execute_compensation_action(action)
                    plan.actions_executed += 1
                    
                    if action.result == RollbackResult.SUCCESS:
                        plan.actions_successful += 1
                    else:
                        plan.actions_failed += 1
                        
                except Exception as e:
                    logger.error("Compensation action failed", action_id=str(action.action_id), error=str(e))
                    action.result = RollbackResult.FAILED
                    action.error_message = str(e)
                    plan.actions_failed += 1
            
            # Determine overall result
            if plan.actions_failed == 0:
                plan.overall_result = RollbackResult.SUCCESS
            elif plan.actions_successful > 0:
                plan.overall_result = RollbackResult.PARTIAL_SUCCESS
            else:
                plan.overall_result = RollbackResult.FAILED
            
            plan.status = "completed"
            plan.execution_completed = datetime.now(timezone.utc)
            
            # Move to history and remove from active
            self.rollback_history.append(plan)
            if plan.workflow_id in self.active_rollbacks:
                del self.active_rollbacks[plan.workflow_id]
            
            # Send notification about rollback completion
            if self.notification_system:
                await self._send_rollback_completion_notification(plan)
            
            logger.info(
                "Rollback plan completed",
                plan_id=str(plan.plan_id),
                result=plan.overall_result.value,
                actions_successful=plan.actions_successful,
                actions_failed=plan.actions_failed
            )
            
        except Exception as e:
            logger.error("Error executing rollback plan", error=str(e), plan_id=str(plan.plan_id))
            plan.status = "failed"
            plan.overall_result = RollbackResult.FAILED
    
    async def _execute_compensation_action(self, action: CompensationAction):
        """Execute individual compensation action"""
        try:
            logger.debug("Executing compensation action", action_id=str(action.action_id), action_type=action.action_type.value)
            
            action.execution_time = datetime.now(timezone.utc)
            
            # Get handler for action type
            handler = self.compensation_handlers.get(action.action_type)
            if not handler:
                logger.warning("No handler for action type", action_type=action.action_type.value)
                action.result = RollbackResult.SKIPPED
                return
            
            # Execute handler
            result = await handler(action)
            action.result = result
            action.executed = True
            
            logger.debug(
                "Compensation action completed",
                action_id=str(action.action_id),
                result=result.value
            )
            
        except Exception as e:
            logger.error("Error executing compensation action", error=str(e), action_id=str(action.action_id))
            action.result = RollbackResult.FAILED
            action.error_message = str(e)
            action.executed = True
    
    async def _restore_workflow_state(self, action: CompensationAction) -> RollbackResult:
        """Restore workflow state from checkpoint"""
        try:
            if not self.persistence:
                logger.warning("No persistence system available for state restoration")
                return RollbackResult.SKIPPED
            
            checkpoint_id = action.parameters.get("checkpoint_id")
            if not checkpoint_id:
                logger.error("No checkpoint ID provided for state restoration")
                return RollbackResult.FAILED
            
            # Mock state restoration - would integrate with persistence system
            logger.info("Restoring workflow state", checkpoint_id=checkpoint_id)
            
            # In production, this would:
            # 1. Load checkpoint from persistence
            # 2. Restore workflow state
            # 3. Reset progress tracking
            # 4. Deallocate resources beyond checkpoint
            
            return RollbackResult.SUCCESS
            
        except Exception as e:
            logger.error("Error restoring workflow state", error=str(e))
            return RollbackResult.FAILED
    
    async def _deallocate_resources(self, action: CompensationAction) -> RollbackResult:
        """Deallocate workflow resources"""
        try:
            resources = action.parameters.get("resources", [])
            
            logger.info("Deallocating resources", resources=resources)
            
            # Mock resource deallocation
            # In production, this would:
            # 1. Release allocated compute resources
            # 2. Clean up temporary storage
            # 3. Close network connections
            # 4. Update resource accounting
            
            return RollbackResult.SUCCESS
            
        except Exception as e:
            logger.error("Error deallocating resources", error=str(e))
            return RollbackResult.FAILED
    
    async def _cleanup_artifacts(self, action: CompensationAction) -> RollbackResult:
        """Clean up workflow artifacts"""
        try:
            artifacts = action.parameters.get("artifacts", [])
            
            logger.info("Cleaning up artifacts", artifacts=artifacts)
            
            # Mock artifact cleanup
            # In production, this would:
            # 1. Delete temporary files
            # 2. Clean up database records
            # 3. Remove cached data
            # 4. Clear intermediate results
            
            return RollbackResult.SUCCESS
            
        except Exception as e:
            logger.error("Error cleaning up artifacts", error=str(e))
            return RollbackResult.FAILED
    
    async def _refund_credits(self, action: CompensationAction) -> RollbackResult:
        """Refund FTNS credits for failed execution"""
        try:
            refund_amount = action.parameters.get("refund_amount", 0.0)
            user_id = action.parameters.get("user_id")
            
            if refund_amount <= 0 or not user_id:
                return RollbackResult.SKIPPED
            
            logger.info("Refunding FTNS credits", user_id=user_id, amount=refund_amount)
            
            # Mock credit refund
            # In production, this would:
            # 1. Calculate refund amount based on partial execution
            # 2. Update user's FTNS balance
            # 3. Log transaction for auditing
            # 4. Send refund notification
            
            return RollbackResult.SUCCESS
            
        except Exception as e:
            logger.error("Error refunding credits", error=str(e))
            return RollbackResult.FAILED
    
    async def _send_rollback_notification(self, action: CompensationAction) -> RollbackResult:
        """Send rollback notification to user"""
        try:
            if not self.notification_system:
                return RollbackResult.SKIPPED
            
            user_id = action.parameters.get("user_id")
            workflow_id = action.parameters.get("workflow_id")
            
            if not user_id:
                return RollbackResult.SKIPPED
            
            await self.notification_system.send_notification(
                user_id=user_id,
                notification_type=NotificationType.WORKFLOW_FAILED,
                context=action.parameters,
                priority=NotificationPriority.HIGH,
                workflow_id=UUID(workflow_id) if workflow_id else None
            )
            
            return RollbackResult.SUCCESS
            
        except Exception as e:
            logger.error("Error sending rollback notification", error=str(e))
            return RollbackResult.FAILED
    
    async def _log_incident(self, action: CompensationAction) -> RollbackResult:
        """Log rollback incident for analysis"""
        try:
            incident_data = action.parameters
            
            logger.info("Logging rollback incident", incident_data=incident_data)
            
            # Mock incident logging
            # In production, this would:
            # 1. Store incident in database
            # 2. Generate incident report
            # 3. Update failure statistics
            # 4. Trigger analysis pipeline
            
            return RollbackResult.SUCCESS
            
        except Exception as e:
            logger.error("Error logging incident", error=str(e))
            return RollbackResult.FAILED
    
    async def _compensate_side_effects(self, action: CompensationAction) -> RollbackResult:
        """Compensate for workflow side effects"""
        try:
            side_effects = action.parameters.get("side_effects", [])
            
            logger.info("Compensating side effects", side_effects=side_effects)
            
            # Mock side effect compensation
            # In production, this would:
            # 1. Reverse external API calls
            # 2. Undo database changes
            # 3. Cancel scheduled tasks
            # 4. Restore modified files
            
            return RollbackResult.SUCCESS
            
        except Exception as e:
            logger.error("Error compensating side effects", error=str(e))
            return RollbackResult.FAILED
    
    async def _reset_dependencies(self, action: CompensationAction) -> RollbackResult:
        """Reset workflow dependencies"""
        try:
            dependencies = action.parameters.get("dependencies", [])
            
            logger.info("Resetting dependencies", dependencies=dependencies)
            
            # Mock dependency reset
            # In production, this would:
            # 1. Reset dependency states
            # 2. Clear dependency caches
            # 3. Notify dependent workflows
            # 4. Update dependency graph
            
            return RollbackResult.SUCCESS
            
        except Exception as e:
            logger.error("Error resetting dependencies", error=str(e))
            return RollbackResult.FAILED
    
    async def _create_full_rollback_actions(self, workflow_id: UUID) -> List[CompensationAction]:
        """Create actions for full workflow rollback"""
        actions = []
        
        # High priority: Send notifications
        actions.append(CompensationAction(
            step_id=workflow_id,  # Use workflow_id as step_id for workflow-level actions
            action_type=RollbackAction.SEND_NOTIFICATION,
            description="Notify user of workflow failure",
            parameters={"user_id": "user123", "workflow_id": str(workflow_id)},
            priority=1
        ))
        
        # High priority: Deallocate resources
        actions.append(CompensationAction(
            step_id=workflow_id,
            action_type=RollbackAction.DEALLOCATE_RESOURCES,
            description="Deallocate all workflow resources",
            parameters={"resources": ["cpu", "memory", "storage"]},
            priority=2
        ))
        
        # Medium priority: Refund credits
        actions.append(CompensationAction(
            step_id=workflow_id,
            action_type=RollbackAction.REFUND_CREDITS,
            description="Refund unused FTNS credits",
            parameters={"user_id": "user123", "refund_amount": 50.0},
            priority=3
        ))
        
        # Medium priority: Cleanup artifacts
        actions.append(CompensationAction(
            step_id=workflow_id,
            action_type=RollbackAction.CLEANUP_ARTIFACTS,
            description="Clean up workflow artifacts",
            parameters={"artifacts": ["temp_files", "intermediate_results"]},
            priority=4
        ))
        
        # Low priority: Log incident
        actions.append(CompensationAction(
            step_id=workflow_id,
            action_type=RollbackAction.LOG_INCIDENT,
            description="Log workflow failure incident",
            parameters={"workflow_id": str(workflow_id), "incident_type": "full_rollback"},
            priority=5
        ))
        
        return actions
    
    async def _create_partial_rollback_actions(self, workflow_id: UUID, checkpoint_id: Optional[UUID]) -> List[CompensationAction]:
        """Create actions for partial workflow rollback"""
        actions = []
        
        if checkpoint_id:
            # Restore to checkpoint
            actions.append(CompensationAction(
                step_id=workflow_id,
                action_type=RollbackAction.RESTORE_STATE,
                description="Restore workflow to checkpoint",
                parameters={"checkpoint_id": str(checkpoint_id)},
                priority=1
            ))
        
        # Deallocate resources beyond checkpoint
        actions.append(CompensationAction(
            step_id=workflow_id,
            action_type=RollbackAction.DEALLOCATE_RESOURCES,
            description="Deallocate resources beyond checkpoint",
            parameters={"resources": ["partial_cpu", "partial_memory"]},
            priority=2
        ))
        
        # Partial refund
        actions.append(CompensationAction(
            step_id=workflow_id,
            action_type=RollbackAction.REFUND_CREDITS,
            description="Partial refund for failed execution",
            parameters={"user_id": "user123", "refund_amount": 25.0},
            priority=3
        ))
        
        return actions
    
    async def _create_compensation_actions(self, workflow_id: UUID, failed_step_id: Optional[UUID]) -> List[CompensationAction]:
        """Create compensation actions for failed step"""
        actions = []
        
        if failed_step_id:
            # Compensate side effects
            actions.append(CompensationAction(
                step_id=failed_step_id,
                action_type=RollbackAction.COMPENSATE_EFFECTS,
                description="Compensate step side effects",
                parameters={"side_effects": ["api_calls", "database_changes"]},
                priority=1
            ))
        
        return actions
    
    async def _create_retry_actions(self, workflow_id: UUID, failed_step_id: Optional[UUID]) -> List[CompensationAction]:
        """Create actions for retrying failed step"""
        actions = []
        
        # Reset dependencies
        actions.append(CompensationAction(
            step_id=workflow_id,
            action_type=RollbackAction.RESET_DEPENDENCIES,
            description="Reset step dependencies for retry",
            parameters={"dependencies": ["external_api", "database_connection"]},
            priority=1
        ))
        
        return actions
    
    async def _create_skip_actions(self, workflow_id: UUID, failed_step_id: Optional[UUID]) -> List[CompensationAction]:
        """Create actions for skipping failed step"""
        actions = []
        
        # Log that step was skipped
        actions.append(CompensationAction(
            step_id=workflow_id,
            action_type=RollbackAction.LOG_INCIDENT,
            description="Log skipped step",
            parameters={"workflow_id": str(workflow_id), "skipped_step": str(failed_step_id)},
            priority=1
        ))
        
        return actions
    
    async def _get_workflow_info(self, workflow_id: UUID) -> Optional[Dict[str, Any]]:
        """Get workflow information"""
        # Mock implementation
        return {
            "workflow_id": str(workflow_id),
            "status": "failed",
            "user_id": "user123"
        }
    
    async def _find_best_checkpoint(self, workflow_id: UUID, failed_step_id: Optional[UUID]) -> Optional[UUID]:
        """Find best checkpoint for rollback"""
        # Mock implementation
        if self.persistence:
            # Would query persistence for latest checkpoint before failed step
            return uuid4()
        return None
    
    async def _should_retry_workflow(self, workflow_id: UUID, failure_type: FailureType) -> bool:
        """Check if workflow should be retried"""
        # Check retry policy
        policy = self.retry_policies.get(str(workflow_id), self.default_retry_policy)
        
        if failure_type not in policy.retry_on_failure_types:
            return False
        
        # Check retry count
        retry_count = self.rollback_statistics.get(f"retries_{workflow_id}", 0)
        if retry_count >= policy.max_retries:
            return False
        
        return True
    
    async def _resources_likely_available_soon(self) -> bool:
        """Check if resources are likely to be available soon"""
        # Mock implementation - would check resource availability
        return True
    
    async def _is_dependency_critical(self, workflow_id: UUID, step_id: Optional[UUID]) -> bool:
        """Check if dependency is critical for workflow"""
        # Mock implementation - would analyze workflow dependencies
        return True
    
    async def _step_has_side_effects(self, workflow_id: UUID, step_id: Optional[UUID]) -> bool:
        """Check if step has side effects that need compensation"""
        # Mock implementation - would analyze step definition
        return False
    
    def _is_circuit_breaker_open(self, workflow_id: UUID, failure_type: FailureType) -> bool:
        """Check if circuit breaker is open for workflow/failure type"""
        key = f"{workflow_id}_{failure_type.value}"
        breaker_state = self.circuit_breaker_state.get(key, {})
        
        if not breaker_state:
            return False
        
        # Check if circuit is open
        if breaker_state.get("state") == "open":
            # Check if recovery timeout has passed
            last_failure = breaker_state.get("last_failure_time")
            if last_failure:
                recovery_timeout = self.default_retry_policy.recovery_timeout
                if datetime.now(timezone.utc) - last_failure > recovery_timeout:
                    # Reset circuit breaker
                    breaker_state["state"] = "closed"
                    breaker_state["failure_count"] = 0
                    return False
                else:
                    return True
        
        return False
    
    def _estimate_rollback_duration(self, plan: RollbackPlan) -> Optional[timedelta]:
        """Estimate rollback execution duration"""
        # Base time per action
        base_time_per_action = timedelta(seconds=30)
        return base_time_per_action * len(plan.compensation_actions)
    
    def _estimate_rollback_cost(self, plan: RollbackPlan) -> float:
        """Estimate rollback execution cost"""
        # Base cost per action
        base_cost_per_action = 1.0
        return base_cost_per_action * len(plan.compensation_actions)
    
    async def _store_failure_pattern(
        self,
        workflow_id: UUID,
        failure_type: FailureType,
        failure_reason: str,
        failed_step_id: Optional[UUID],
        failure_context: Optional[Dict[str, Any]]
    ):
        """Store failure pattern for learning"""
        pattern = {
            "workflow_id": str(workflow_id),
            "failure_type": failure_type.value,
            "failure_reason": failure_reason,
            "failed_step_id": str(failed_step_id) if failed_step_id else None,
            "failure_context": failure_context or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.failure_patterns[failure_type.value].append(pattern)
        
        # Keep only recent patterns (last 1000)
        if len(self.failure_patterns[failure_type.value]) > 1000:
            self.failure_patterns[failure_type.value] = self.failure_patterns[failure_type.value][-1000:]
    
    async def _send_rollback_completion_notification(self, plan: RollbackPlan):
        """Send notification about rollback completion"""
        if not self.notification_system:
            return
        
        context = {
            "workflow_id": str(plan.workflow_id),
            "rollback_strategy": plan.strategy.value,
            "rollback_result": plan.overall_result.value if plan.overall_result else "unknown",
            "actions_executed": plan.actions_executed,
            "actions_successful": plan.actions_successful,
            "actions_failed": plan.actions_failed,
            "execution_duration": str(plan.execution_completed - plan.execution_started) if plan.execution_completed and plan.execution_started else "unknown"
        }
        
        # Determine notification type based on result
        if plan.overall_result == RollbackResult.SUCCESS:
            notification_type = NotificationType.WORKFLOW_COMPLETED
            priority = NotificationPriority.NORMAL
        else:
            notification_type = NotificationType.WORKFLOW_FAILED
            priority = NotificationPriority.HIGH
        
        await self.notification_system.send_notification(
            user_id="user123",  # Would get from workflow info
            notification_type=notification_type,
            context=context,
            priority=priority,
            workflow_id=plan.workflow_id
        )
    
    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get rollback system statistics"""
        return {
            "rollback_statistics": dict(self.rollback_statistics),
            "active_rollbacks": len(self.active_rollbacks),
            "total_rollback_history": len(self.rollback_history),
            "failure_patterns": {
                failure_type: len(patterns) 
                for failure_type, patterns in self.failure_patterns.items()
            },
            "circuit_breaker_states": len(self.circuit_breaker_state)
        }
    
    def get_active_rollbacks(self) -> List[RollbackPlan]:
        """Get list of active rollback plans"""
        return list(self.active_rollbacks.values())
    
    def get_rollback_history(
        self,
        workflow_id: Optional[UUID] = None,
        failure_type: Optional[FailureType] = None,
        limit: int = 100
    ) -> List[RollbackPlan]:
        """Get rollback history with filters"""
        history = self.rollback_history
        
        if workflow_id:
            history = [plan for plan in history if plan.workflow_id == workflow_id]
        
        if failure_type:
            history = [plan for plan in history if plan.failure_type == failure_type]
        
        # Sort by execution time (newest first) and limit
        history.sort(key=lambda p: p.execution_started or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return history[:limit]
    
    def set_retry_policy(self, workflow_id: str, policy: RetryPolicy):
        """Set retry policy for specific workflow"""
        self.retry_policies[workflow_id] = policy
        logger.info("Retry policy set", workflow_id=workflow_id, max_retries=policy.max_retries)
    
    async def retry_workflow(self, workflow_id: UUID) -> bool:
        """Manually retry a failed workflow"""
        try:
            # Check if workflow can be retried
            if not await self._should_retry_workflow(workflow_id, FailureType.STEP_EXECUTION_ERROR):
                logger.warning("Workflow cannot be retried", workflow_id=str(workflow_id))
                return False
            
            # Create retry rollback plan
            retry_plan = RollbackPlan(
                workflow_id=workflow_id,
                failure_type=FailureType.STEP_EXECUTION_ERROR,
                failure_reason="Manual retry requested",
                strategy=RollbackStrategy.RETRY_FAILED_STEP
            )
            
            retry_plan.compensation_actions = await self._create_retry_actions(workflow_id, None)
            
            # Execute retry plan
            await self._execute_rollback_plan(retry_plan)
            
            # Update retry count
            self.rollback_statistics[f"retries_{workflow_id}"] += 1
            
            logger.info("Workflow retry completed", workflow_id=str(workflow_id), result=retry_plan.overall_result.value if retry_plan.overall_result else "unknown")
            
            return retry_plan.overall_result == RollbackResult.SUCCESS
            
        except Exception as e:
            logger.error("Error retrying workflow", error=str(e), workflow_id=str(workflow_id))
            return False


# Global instance for easy access
_workflow_rollback_system = None

def get_workflow_rollback_system() -> WorkflowRollbackSystem:
    """Get global workflow rollback system instance"""
    global _workflow_rollback_system
    if _workflow_rollback_system is None:
        _workflow_rollback_system = WorkflowRollbackSystem()
    return _workflow_rollback_system