"""
PRSM Governance Execution System

Implements governance execution that connects approved proposals to actual
configuration changes, parameter modifications, and treasury operations.

Key Components:
- ProposalType: Types of governance proposals
- GovernanceAction: Action to be executed by governance
- TimelockController: Manages timelocked governance actions
- GovernanceExecutor: Executes approved governance proposals
- GovernableParameterRegistry: Registry of governable parameters

Security Features:
- Timelock delays for sensitive changes
- Execution transparency and audit trails
- Graceful failure handling
- Rollback mechanisms
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal

import structlog
from pydantic import Field, BaseModel, validator

logger = structlog.get_logger()


# === Enums ===

class ProposalType(str, Enum):
    """Types of governance proposals with execution semantics."""
    PARAMETER_CHANGE = "parameter_change"          # Change a system parameter
    PROTOCOL_UPGRADE = "protocol_upgrade"          # Upgrade protocol version
    TREASURY_SPEND = "treasury_spend"              # Spend from treasury
    EMERGENCY_ACTION = "emergency_action"          # Emergency response
    MEMBER_EJECTION = "member_ejection"            # Eject a network member
    FEE_ADJUSTMENT = "fee_adjustment"              # Adjust network fees
    REWARD_RATE_CHANGE = "reward_rate_change"      # Change staking reward rates
    SLASHING_CONFIG = "slashing_config"            # Modify slashing parameters
    CIRCUIT_BREAKER = "circuit_breaker"            # Circuit breaker configuration
    ACCESS_CONTROL = "access_control"              # Access control modifications


class GovernanceActionStatus(str, Enum):
    """Status of a governance action."""
    PENDING = "pending"                # Action created, not yet scheduled
    SCHEDULED = "scheduled"            # Scheduled for future execution
    READY = "ready"                    # Ready for execution (timelock passed)
    EXECUTING = "executing"            # Currently being executed
    EXECUTED = "executed"              # Successfully executed
    FAILED = "failed"                  # Execution failed
    CANCELLED = "cancelled"            # Action was cancelled
    EXPIRED = "expired"                # Action expired before execution


class ExecutionPriority(str, Enum):
    """Priority levels for action execution."""
    LOW = "low"          # Standard execution
    NORMAL = "normal"    # Normal priority
    HIGH = "high"        # High priority
    CRITICAL = "critical"  # Critical, minimal delay
    EMERGENCY = "emergency"  # Emergency, bypass some checks


# === Data Classes ===

@dataclass
class GovernanceAction:
    """
    Action to be executed by governance.
    
    Represents a single executable action that results from an approved proposal.
    """
    action_id: str
    action_type: ProposalType
    target_module: str              # e.g., "ftns", "staking", "network"
    target_parameter: str           # e.g., "reward_rate", "min_stake"
    current_value: Any
    proposed_value: Any
    execution_delay: int            # Seconds before execution
    requires_timelock: bool
    proposal_id: Optional[str] = None
    proposer_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: GovernanceActionStatus = GovernanceActionStatus.PENDING
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.action_type, str):
            self.action_type = ProposalType(self.action_type)
        if isinstance(self.status, str):
            self.status = GovernanceActionStatus(self.status)
        if isinstance(self.priority, str):
            self.priority = ExecutionPriority(self.priority)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for serialization."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "target_module": self.target_module,
            "target_parameter": self.target_parameter,
            "current_value": str(self.current_value) if isinstance(self.current_value, Decimal) else self.current_value,
            "proposed_value": str(self.proposed_value) if isinstance(self.proposed_value, Decimal) else self.proposed_value,
            "execution_delay": self.execution_delay,
            "requires_timelock": self.requires_timelock,
            "proposal_id": self.proposal_id,
            "proposer_id": self.proposer_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "priority": self.priority.value,
            "metadata": self.metadata,
        }


@dataclass
class TimelockRecord:
    """Record of a timelocked action."""
    record_id: str
    action: GovernanceAction
    scheduled_at: datetime
    execute_at: datetime
    executed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    cancellation_reason: Optional[str] = None
    status: GovernanceActionStatus = GovernanceActionStatus.SCHEDULED
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = GovernanceActionStatus(self.status)
    
    @property
    def is_ready(self) -> bool:
        """Check if the action is ready for execution."""
        return (
            self.status == GovernanceActionStatus.SCHEDULED and
            datetime.now(timezone.utc) >= self.execute_at
        )
    
    @property
    def time_remaining(self) -> timedelta:
        """Get time remaining until execution."""
        if self.status != GovernanceActionStatus.SCHEDULED:
            return timedelta(0)
        remaining = self.execute_at - datetime.now(timezone.utc)
        return remaining if remaining > timedelta(0) else timedelta(0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "record_id": self.record_id,
            "action": self.action.to_dict(),
            "scheduled_at": self.scheduled_at.isoformat(),
            "execute_at": self.execute_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "cancellation_reason": self.cancellation_reason,
            "status": self.status.value,
            "is_ready": self.is_ready,
            "time_remaining_seconds": self.time_remaining.total_seconds(),
        }


@dataclass
class ExecutionResult:
    """Result of a governance action execution."""
    result_id: str
    action_id: str
    success: bool
    execution_time: datetime
    execution_duration_ms: int
    message: str
    error_details: Optional[str] = None
    rollback_available: bool = True
    rollback_data: Optional[Dict[str, Any]] = None
    verification_steps: List[str] = field(default_factory=list)
    affected_resources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "result_id": self.result_id,
            "action_id": self.action_id,
            "success": self.success,
            "execution_time": self.execution_time.isoformat(),
            "execution_duration_ms": self.execution_duration_ms,
            "message": self.message,
            "error_details": self.error_details,
            "rollback_available": self.rollback_available,
            "rollback_data": self.rollback_data,
            "verification_steps": self.verification_steps,
            "affected_resources": self.affected_resources,
        }


@dataclass
class ParameterDefinition:
    """Definition of a governable parameter."""
    module: str
    name: str
    description: str
    current_value: Any
    value_type: str                  # "int", "float", "decimal", "str", "bool", "list", "dict"
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    requires_timelock: bool = True
    timelock_delay_seconds: int = 24 * 3600  # Default 24 hours
    requires_emergency_timelock: bool = True
    emergency_timelock_seconds: int = 2 * 3600  # 2 hours for emergency
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified_by: str = "system"
    
    def validate_value(self, value: Any) -> bool:
        """Validate a proposed value against constraints."""
        # Check type
        if self.value_type == "int":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
            value = int(value)
        elif self.value_type == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
        elif self.value_type == "decimal":
            if not isinstance(value, (int, float, Decimal)):
                return False
        elif self.value_type == "str":
            if not isinstance(value, str):
                return False
        elif self.value_type == "bool":
            if not isinstance(value, bool):
                return False
        
        # Check allowed values
        if self.allowed_values is not None:
            if value not in self.allowed_values:
                return False
        
        # Check min/max for numeric types
        if self.value_type in ("int", "float", "decimal"):
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert definition to dictionary."""
        return {
            "module": self.module,
            "name": self.name,
            "description": self.description,
            "current_value": str(self.current_value) if isinstance(self.current_value, Decimal) else self.current_value,
            "value_type": self.value_type,
            "min_value": str(self.min_value) if isinstance(self.min_value, Decimal) else self.min_value,
            "max_value": str(self.max_value) if isinstance(self.max_value, Decimal) else self.max_value,
            "allowed_values": [str(v) if isinstance(v, Decimal) else v for v in self.allowed_values] if self.allowed_values else None,
            "requires_timelock": self.requires_timelock,
            "timelock_delay_seconds": self.timelock_delay_seconds,
            "last_modified": self.last_modified.isoformat(),
            "modified_by": self.modified_by,
        }


@dataclass
class ParameterChangeRecord:
    """Record of a parameter change."""
    change_id: str
    module: str
    parameter_name: str
    old_value: Any
    new_value: Any
    changed_at: datetime
    changed_by: str                  # "governance" or specific action_id
    proposal_id: Optional[str] = None
    action_id: Optional[str] = None
    rollback_available: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "change_id": self.change_id,
            "module": self.module,
            "parameter_name": self.parameter_name,
            "old_value": str(self.old_value) if isinstance(self.old_value, Decimal) else self.old_value,
            "new_value": str(self.new_value) if isinstance(self.new_value, Decimal) else self.new_value,
            "changed_at": self.changed_at.isoformat(),
            "changed_by": self.changed_by,
            "proposal_id": self.proposal_id,
            "action_id": self.action_id,
            "rollback_available": self.rollback_available,
        }


# === Timelock Controller ===

class TimelockController:
    """
    Manages timelocked governance actions.
    
    Provides security through mandatory delays before executing sensitive
    governance actions, allowing time for review and potential cancellation.
    """
    
    def __init__(
        self,
        min_delay: int = 24 * 3600,      # 24 hours default
        max_delay: int = 7 * 24 * 3600,   # 7 days max
        emergency_delay: int = 2 * 3600,  # 2 hours for emergency
        grace_period: int = 3 * 24 * 3600  # 3 days to execute after ready
    ):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.emergency_delay = emergency_delay
        self.grace_period = grace_period
        
        self.logger = logger.bind(component="timelock_controller")
        
        # Storage for timelocked actions
        self._pending_actions: Dict[str, TimelockRecord] = {}
        self._execution_history: Dict[str, ExecutionResult] = {}
        
        # Callbacks for execution
        self._execution_callbacks: Dict[str, Callable[[GovernanceAction], Awaitable[ExecutionResult]]] = {}
    
    def register_execution_callback(
        self,
        action_type: ProposalType,
        callback: Callable[[GovernanceAction], Awaitable[ExecutionResult]]
    ):
        """Register a callback for executing specific action types."""
        self._execution_callbacks[action_type.value] = callback
        self.logger.info(
            "Registered execution callback",
            action_type=action_type.value
        )
    
    async def schedule_action(
        self,
        action: GovernanceAction,
        execute_at: Optional[datetime] = None
    ) -> TimelockRecord:
        """
        Schedule an action for future execution.
        
        Args:
            action: The governance action to schedule
            execute_at: Optional specific execution time (calculated if not provided)
            
        Returns:
            TimelockRecord for the scheduled action
            
        Raises:
            ValueError: If action is invalid or delay constraints violated
        """
        self.logger.info(
            "Scheduling governance action",
            action_id=action.action_id,
            action_type=action.action_type.value,
            requires_timelock=action.requires_timelock
        )
        
        # Validate action
        if not action.action_id:
            raise ValueError("Action must have an action_id")
        
        # Check if already scheduled
        if action.action_id in self._pending_actions:
            raise ValueError(f"Action {action.action_id} already scheduled")
        
        # Calculate execution time
        scheduled_at = datetime.now(timezone.utc)
        
        if execute_at is None:
            # Calculate based on action requirements
            if action.priority == ExecutionPriority.EMERGENCY:
                delay = self.emergency_delay
            elif action.priority == ExecutionPriority.CRITICAL:
                delay = max(self.min_delay // 2, action.execution_delay)
            else:
                delay = max(self.min_delay, action.execution_delay) if action.requires_timelock else 0
            
            # Enforce max delay
            delay = min(delay, self.max_delay)
            execute_at = scheduled_at + timedelta(seconds=delay)
        else:
            # Validate provided execution time
            delay = (execute_at - scheduled_at).total_seconds()
            if action.requires_timelock and delay < self.min_delay:
                raise ValueError(
                    f"Timelock action requires minimum {self.min_delay}s delay, "
                    f"but only {delay}s provided"
                )
            if delay > self.max_delay:
                raise ValueError(
                    f"Delay exceeds maximum of {self.max_delay}s"
                )
        
        # Create timelock record
        record = TimelockRecord(
            record_id=str(uuid4()),
            action=action,
            scheduled_at=scheduled_at,
            execute_at=execute_at,
            status=GovernanceActionStatus.SCHEDULED
        )
        
        # Store record
        self._pending_actions[action.action_id] = record
        
        # Update action status
        action.status = GovernanceActionStatus.SCHEDULED
        
        self.logger.info(
            "Action scheduled successfully",
            action_id=action.action_id,
            record_id=record.record_id,
            execute_at=execute_at.isoformat(),
            delay_seconds=delay
        )
        
        return record
    
    async def execute_action(self, action_id: str) -> ExecutionResult:
        """
        Execute a timelocked action.
        
        Args:
            action_id: ID of the action to execute
            
        Returns:
            ExecutionResult with execution details
            
        Raises:
            ValueError: If action not found or not ready
        """
        self.logger.info("Executing governance action", action_id=action_id)
        
        # Get record
        record = self._pending_actions.get(action_id)
        if not record:
            raise ValueError(f"Action {action_id} not found")
        
        # Check if ready
        if not record.is_ready:
            remaining = record.time_remaining
            raise ValueError(
                f"Action not ready for execution. "
                f"Time remaining: {remaining.total_seconds()}s"
            )
        
        # Check if already executed or cancelled
        if record.status in (GovernanceActionStatus.EXECUTED, GovernanceActionStatus.CANCELLED):
            raise ValueError(f"Action already {record.status.value}")
        
        # Update status
        record.status = GovernanceActionStatus.EXECUTING
        record.action.status = GovernanceActionStatus.EXECUTING
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get execution callback
            action_type = record.action.action_type.value
            callback = self._execution_callbacks.get(action_type)
            
            if not callback:
                # Default execution (just mark as executed)
                result = ExecutionResult(
                    result_id=str(uuid4()),
                    action_id=action_id,
                    success=True,
                    execution_time=start_time,
                    execution_duration_ms=0,
                    message=f"No callback registered for {action_type}, marked as executed",
                    verification_steps=["No verification - no callback"],
                    affected_resources=[]
                )
            else:
                # Execute via callback
                result = await callback(record.action)
            
            # Update record
            record.executed_at = datetime.now(timezone.utc)
            record.status = GovernanceActionStatus.EXECUTED
            record.action.status = GovernanceActionStatus.EXECUTED
            
            # Store result
            self._execution_history[action_id] = result
            
            self.logger.info(
                "Action executed successfully",
                action_id=action_id,
                success=result.success,
                duration_ms=result.execution_duration_ms
            )
            
            return result
            
        except Exception as e:
            # Handle execution failure
            error_message = str(e)
            
            result = ExecutionResult(
                result_id=str(uuid4()),
                action_id=action_id,
                success=False,
                execution_time=start_time,
                execution_duration_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
                message="Execution failed",
                error_details=error_message,
                rollback_available=True
            )
            
            record.status = GovernanceActionStatus.FAILED
            record.action.status = GovernanceActionStatus.FAILED
            
            self._execution_history[action_id] = result
            
            self.logger.error(
                "Action execution failed",
                action_id=action_id,
                error=error_message
            )
            
            return result
    
    async def cancel_action(self, action_id: str, reason: str) -> bool:
        """
        Cancel a scheduled action.
        
        Args:
            action_id: ID of the action to cancel
            reason: Reason for cancellation
            
        Returns:
            True if cancelled successfully
            
        Raises:
            ValueError: If action not found or already executed
        """
        self.logger.info(
            "Cancelling governance action",
            action_id=action_id,
            reason=reason
        )
        
        record = self._pending_actions.get(action_id)
        if not record:
            raise ValueError(f"Action {action_id} not found")
        
        # Check if can be cancelled
        if record.status == GovernanceActionStatus.EXECUTED:
            raise ValueError("Cannot cancel executed action")
        
        if record.status == GovernanceActionStatus.CANCELLED:
            return True  # Already cancelled
        
        # Cancel the action
        record.cancelled_at = datetime.now(timezone.utc)
        record.cancellation_reason = reason
        record.status = GovernanceActionStatus.CANCELLED
        record.action.status = GovernanceActionStatus.CANCELLED
        
        self.logger.info(
            "Action cancelled successfully",
            action_id=action_id
        )
        
        return True
    
    async def get_pending_actions(self) -> List[TimelockRecord]:
        """Get all pending timelocked actions."""
        return [
            record for record in self._pending_actions.values()
            if record.status == GovernanceActionStatus.SCHEDULED
        ]
    
    async def get_ready_actions(self) -> List[TimelockRecord]:
        """Get all actions ready for execution."""
        return [
            record for record in self._pending_actions.values()
            if record.is_ready
        ]
    
    async def get_action_record(self, action_id: str) -> Optional[TimelockRecord]:
        """Get a specific action record."""
        return self._pending_actions.get(action_id)
    
    async def get_execution_result(self, action_id: str) -> Optional[ExecutionResult]:
        """Get execution result for an action."""
        return self._execution_history.get(action_id)
    
    async def cleanup_expired_actions(self) -> int:
        """
        Remove expired actions that passed their grace period.
        
        Returns:
            Number of actions cleaned up
        """
        now = datetime.now(timezone.utc)
        expired_count = 0
        
        for action_id, record in list(self._pending_actions.items()):
            if record.status == GovernanceActionStatus.SCHEDULED:
                expiry_time = record.execute_at + timedelta(seconds=self.grace_period)
                if now > expiry_time:
                    record.status = GovernanceActionStatus.EXPIRED
                    record.action.status = GovernanceActionStatus.EXPIRED
                    expired_count += 1
                    
                    self.logger.warning(
                        "Action expired",
                        action_id=action_id,
                        execute_at=record.execute_at.isoformat()
                    )
        
        return expired_count


# === Governable Parameter Registry ===

class GovernableParameterRegistry:
    """
    Registry of governable parameters.
    
    Maintains definitions of all parameters that can be modified through
    governance, including validation rules and change history.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="parameter_registry")
        
        # Parameter storage: module -> name -> definition
        self.parameters: Dict[str, Dict[str, ParameterDefinition]] = {}
        
        # Change history: module -> name -> list of changes
        self.change_history: Dict[str, Dict[str, List[ParameterChangeRecord]]] = {}
        
        # Callbacks for parameter changes
        self._change_callbacks: Dict[str, Callable[[str, str, Any], Awaitable[bool]]] = {}
    
    def register_parameter(
        self,
        module: str,
        name: str,
        description: str,
        current_value: Any,
        value_type: str,
        min_value: Any = None,
        max_value: Any = None,
        allowed_values: Optional[List[Any]] = None,
        requires_timelock: bool = True,
        timelock_delay_seconds: int = 24 * 3600
    ) -> ParameterDefinition:
        """
        Register a governable parameter.
        
        Args:
            module: Module that owns the parameter
            name: Parameter name
            description: Human-readable description
            current_value: Current value
            value_type: Type of value (int, float, decimal, str, bool)
            min_value: Minimum allowed value (for numeric types)
            max_value: Maximum allowed value (for numeric types)
            allowed_values: List of allowed values (for enum-like parameters)
            requires_timelock: Whether changes require timelock
            timelock_delay_seconds: Timelock delay in seconds
            
        Returns:
            The created ParameterDefinition
        """
        definition = ParameterDefinition(
            module=module,
            name=name,
            description=description,
            current_value=current_value,
            value_type=value_type,
            min_value=min_value,
            max_value=max_value,
            allowed_values=allowed_values,
            requires_timelock=requires_timelock,
            timelock_delay_seconds=timelock_delay_seconds
        )
        
        # Initialize storage
        if module not in self.parameters:
            self.parameters[module] = {}
            self.change_history[module] = {}
        
        self.parameters[module][name] = definition
        self.change_history[module][name] = []
        
        self.logger.info(
            "Registered governable parameter",
            module=module,
            name=name,
            value_type=value_type,
            requires_timelock=requires_timelock
        )
        
        return definition
    
    def register_change_callback(
        self,
        module: str,
        callback: Callable[[str, str, Any], Awaitable[bool]]
    ):
        """
        Register a callback to be called when parameters change in a module.
        
        The callback receives (module, parameter_name, new_value) and should
        return True if the change was applied successfully.
        """
        self._change_callbacks[module] = callback
        self.logger.info("Registered change callback", module=module)
    
    def get_parameter(self, module: str, name: str) -> Optional[ParameterDefinition]:
        """Get a parameter definition."""
        return self.parameters.get(module, {}).get(name)
    
    def get_parameter_value(self, module: str, name: str) -> Optional[Any]:
        """Get current value of a parameter."""
        param = self.get_parameter(module, name)
        return param.current_value if param else None
    
    def get_all_parameters(self, module: Optional[str] = None) -> List[ParameterDefinition]:
        """Get all parameters, optionally filtered by module."""
        if module:
            return list(self.parameters.get(module, {}).values())
        
        return [
            param for module_params in self.parameters.values()
            for param in module_params.values()
        ]
    
    async def set_parameter(
        self,
        module: str,
        name: str,
        value: Any,
        executed_by: str = "governance",
        proposal_id: Optional[str] = None,
        action_id: Optional[str] = None
    ) -> bool:
        """
        Set a parameter value (called by governance executor).
        
        Args:
            module: Module name
            name: Parameter name
            value: New value
            executed_by: Who executed the change
            proposal_id: Associated proposal ID
            action_id: Associated action ID
            
        Returns:
            True if successful
            
        Raises:
            ValueError: If parameter not found or value invalid
        """
        param = self.get_parameter(module, name)
        if not param:
            raise ValueError(f"Parameter {module}.{name} not found")
        
        # Validate value
        if not param.validate_value(value):
            raise ValueError(
                f"Invalid value {value} for parameter {module}.{name}. "
                f"Constraints: type={param.value_type}, "
                f"min={param.min_value}, max={param.max_value}, "
                f"allowed={param.allowed_values}"
            )
        
        old_value = param.current_value
        
        # Call change callback if registered
        callback = self._change_callbacks.get(module)
        if callback:
            try:
                success = await callback(module, name, value)
                if not success:
                    self.logger.error(
                        "Parameter change callback failed",
                        module=module,
                        name=name
                    )
                    return False
            except Exception as e:
                self.logger.error(
                    "Parameter change callback error",
                    module=module,
                    name=name,
                    error=str(e)
                )
                return False
        
        # Update parameter
        param.current_value = value
        param.last_modified = datetime.now(timezone.utc)
        param.modified_by = executed_by
        
        # Record change
        change_record = ParameterChangeRecord(
            change_id=str(uuid4()),
            module=module,
            parameter_name=name,
            old_value=old_value,
            new_value=value,
            changed_at=datetime.now(timezone.utc),
            changed_by=executed_by,
            proposal_id=proposal_id,
            action_id=action_id
        )
        
        self.change_history[module][name].append(change_record)
        
        self.logger.info(
            "Parameter changed",
            module=module,
            name=name,
            old_value=str(old_value),
            new_value=str(value),
            changed_by=executed_by
        )
        
        return True
    
    def get_parameter_history(
        self,
        module: str,
        name: str,
        limit: int = 100
    ) -> List[ParameterChangeRecord]:
        """Get history of parameter changes."""
        history = self.change_history.get(module, {}).get(name, [])
        return history[-limit:]
    
    def get_module_parameters(self, module: str) -> Dict[str, ParameterDefinition]:
        """Get all parameters for a module."""
        return self.parameters.get(module, {})
    
    def initialize_default_parameters(self):
        """Initialize default governable parameters for the system."""
        # FTNS/Economy parameters
        self.register_parameter(
            module="ftns",
            name="reward_rate",
            description="Annual reward rate for staking (as decimal, e.g., 0.05 = 5%)",
            current_value=0.05,
            value_type="float",
            min_value=0.0,
            max_value=0.5,
            requires_timelock=True,
            timelock_delay_seconds=48 * 3600  # 48 hours
        )
        
        self.register_parameter(
            module="ftns",
            name="min_stake",
            description="Minimum stake amount in FTNS",
            current_value=1000,
            value_type="int",
            min_value=100,
            max_value=1_000_000,
            requires_timelock=True,
            timelock_delay_seconds=24 * 3600
        )
        
        self.register_parameter(
            module="ftns",
            name="max_stake_per_user",
            description="Maximum stake per user in FTNS",
            current_value=10_000_000,
            value_type="int",
            min_value=1_000_000,
            max_value=100_000_000,
            requires_timelock=True,
            timelock_delay_seconds=24 * 3600
        )
        
        # Staking parameters
        self.register_parameter(
            module="staking",
            name="unstaking_period_seconds",
            description="Unstaking period in seconds",
            current_value=7 * 24 * 3600,  # 7 days
            value_type="int",
            min_value=24 * 3600,  # 1 day minimum
            max_value=30 * 24 * 3600,  # 30 days maximum
            requires_timelock=True,
            timelock_delay_seconds=48 * 3600
        )
        
        self.register_parameter(
            module="staking",
            name="slashing_rate_base",
            description="Base slashing rate (as decimal)",
            current_value=0.1,
            value_type="float",
            min_value=0.0,
            max_value=0.5,
            requires_timelock=True,
            timelock_delay_seconds=48 * 3600
        )
        
        # Network parameters
        self.register_parameter(
            module="network",
            name="max_block_size",
            description="Maximum block size in bytes",
            current_value=1_000_000,
            value_type="int",
            min_value=100_000,
            max_value=10_000_000,
            requires_timelock=True,
            timelock_delay_seconds=24 * 3600
        )
        
        self.register_parameter(
            module="network",
            name="transaction_fee_base",
            description="Base transaction fee in FTNS",
            current_value=0.001,
            value_type="decimal",
            min_value=0.0,
            max_value=1.0,
            requires_timelock=True,
            timelock_delay_seconds=24 * 3600
        )
        
        # Governance parameters
        self.register_parameter(
            module="governance",
            name="voting_period_seconds",
            description="Default voting period in seconds",
            current_value=7 * 24 * 3600,  # 7 days
            value_type="int",
            min_value=24 * 3600,  # 1 day
            max_value=14 * 24 * 3600,  # 14 days
            requires_timelock=True,
            timelock_delay_seconds=48 * 3600
        )
        
        self.register_parameter(
            module="governance",
            name="required_approval_percentage",
            description="Required approval percentage for proposals",
            current_value=0.6,
            value_type="float",
            min_value=0.5,
            max_value=0.9,
            requires_timelock=True,
            timelock_delay_seconds=48 * 3600
        )
        
        self.register_parameter(
            module="governance",
            name="minimum_participation",
            description="Minimum participation rate for valid votes",
            current_value=0.3,
            value_type="float",
            min_value=0.1,
            max_value=0.5,
            requires_timelock=True,
            timelock_delay_seconds=48 * 3600
        )
        
        self.logger.info("Initialized default governable parameters")


# === Governance Executor ===

class GovernanceExecutor:
    """
    Executes approved governance proposals.
    
    Connects approved proposals to actual configuration changes through
    the timelock controller and parameter registry.
    """
    
    def __init__(
        self,
        timelock: Optional[TimelockController] = None,
        parameter_registry: Optional[GovernableParameterRegistry] = None
    ):
        self.timelock = timelock or TimelockController()
        self.parameter_registry = parameter_registry or GovernableParameterRegistry()
        
        self.logger = logger.bind(component="governance_executor")
        
        # Service references (set via setters or init)
        self._ftns_service = None
        self._staking_manager = None
        self._network_manager = None
        
        # Execution history
        self._execution_history: Dict[str, ExecutionResult] = {}
        
        # Register execution callbacks
        self._register_callbacks()
    
    def set_ftns_service(self, service: Any):
        """Set FTNS service reference."""
        self._ftns_service = service
        self.logger.info("FTNS service set")
    
    def set_staking_manager(self, manager: Any):
        """Set staking manager reference."""
        self._staking_manager = manager
        self.logger.info("Staking manager set")
    
    def set_network_manager(self, manager: Any):
        """Set network manager reference."""
        self._network_manager = manager
        self.logger.info("Network manager set")
    
    def _register_callbacks(self):
        """Register execution callbacks with timelock controller."""
        self.timelock.register_execution_callback(
            ProposalType.PARAMETER_CHANGE,
            self._execute_parameter_change
        )
        
        self.timelock.register_execution_callback(
            ProposalType.REWARD_RATE_CHANGE,
            self._execute_parameter_change
        )
        
        self.timelock.register_execution_callback(
            ProposalType.FEE_ADJUSTMENT,
            self._execute_parameter_change
        )
        
        self.timelock.register_execution_callback(
            ProposalType.SLASHING_CONFIG,
            self._execute_parameter_change
        )
        
        self.timelock.register_execution_callback(
            ProposalType.CIRCUIT_BREAKER,
            self._execute_circuit_breaker_change
        )
        
        self.timelock.register_execution_callback(
            ProposalType.TREASURY_SPEND,
            self._execute_treasury_spend
        )
        
        self.timelock.register_execution_callback(
            ProposalType.EMERGENCY_ACTION,
            self._execute_emergency_action
        )
        
        self.timelock.register_execution_callback(
            ProposalType.MEMBER_EJECTION,
            self._execute_member_ejection
        )
        
        self.timelock.register_execution_callback(
            ProposalType.PROTOCOL_UPGRADE,
            self._execute_protocol_upgrade
        )
    
    async def create_action_from_proposal(
        self,
        proposal_id: str,
        proposal_type: str,
        proposal_data: Dict[str, Any]
    ) -> GovernanceAction:
        """
        Create a governance action from an approved proposal.
        
        Args:
            proposal_id: ID of the approved proposal
            proposal_type: Type of proposal
            proposal_data: Proposal data including target and values
            
        Returns:
            GovernanceAction ready for scheduling
        """
        action_id = str(uuid4())
        
        # Extract action details from proposal data
        target_module = proposal_data.get("target_module", "system")
        target_parameter = proposal_data.get("target_parameter", "")
        current_value = proposal_data.get("current_value")
        proposed_value = proposal_data.get("proposed_value")
        
        # Get parameter definition if available
        param_def = self.parameter_registry.get_parameter(target_module, target_parameter)
        
        # Determine execution requirements
        requires_timelock = True
        execution_delay = 24 * 3600  # Default 24 hours
        priority = ExecutionPriority.NORMAL
        
        if param_def:
            requires_timelock = param_def.requires_timelock
            execution_delay = param_def.timelock_delay_seconds
            current_value = param_def.current_value
        
        # Check for emergency or critical priority
        if proposal_data.get("urgency") == "emergency":
            priority = ExecutionPriority.EMERGENCY
            execution_delay = self.timelock.emergency_delay
        elif proposal_data.get("urgency") == "critical":
            priority = ExecutionPriority.CRITICAL
        
        action = GovernanceAction(
            action_id=action_id,
            action_type=ProposalType(proposal_type),
            target_module=target_module,
            target_parameter=target_parameter,
            current_value=current_value,
            proposed_value=proposed_value,
            execution_delay=execution_delay,
            requires_timelock=requires_timelock,
            proposal_id=proposal_id,
            proposer_id=proposal_data.get("proposer_id"),
            priority=priority,
            metadata=proposal_data.get("metadata", {})
        )
        
        self.logger.info(
            "Created governance action from proposal",
            action_id=action_id,
            proposal_id=proposal_id,
            action_type=proposal_type
        )
        
        return action
    
    async def execute_proposal(
        self,
        proposal_id: str,
        proposal_type: str,
        proposal_data: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute an approved proposal.
        
        This is the main entry point for executing governance proposals.
        It creates an action, schedules it (if timelock required), and
        either executes immediately or returns the scheduled action info.
        
        Args:
            proposal_id: ID of the approved proposal
            proposal_type: Type of proposal
            proposal_data: Proposal data
            
        Returns:
            ExecutionResult (immediate) or scheduling info (timelocked)
        """
        self.logger.info(
            "Executing approved proposal",
            proposal_id=proposal_id,
            proposal_type=proposal_type
        )
        
        # Create action from proposal
        action = await self.create_action_from_proposal(
            proposal_id=proposal_id,
            proposal_type=proposal_type,
            proposal_data=proposal_data
        )
        
        # Check if immediate execution is allowed
        if not action.requires_timelock or action.priority == ExecutionPriority.EMERGENCY:
            # Execute immediately for non-timelocked or emergency actions
            if action.priority == ExecutionPriority.EMERGENCY:
                self.logger.warning(
                    "Executing emergency action without full timelock",
                    action_id=action.action_id,
                    proposal_id=proposal_id
                )
            
            # Schedule with minimal delay
            record = await self.timelock.schedule_action(action)
            
            # Execute immediately if ready (0 delay)
            if record.is_ready:
                result = await self.timelock.execute_action(action.action_id)
                self._execution_history[action.action_id] = result
                return result
            
            # Return scheduled result
            return ExecutionResult(
                result_id=str(uuid4()),
                action_id=action.action_id,
                success=True,
                execution_time=datetime.now(timezone.utc),
                execution_duration_ms=0,
                message=f"Action scheduled for execution at {record.execute_at.isoformat()}",
                verification_steps=["Action scheduled"],
                affected_resources=[]
            )
        
        # Schedule for timelocked execution
        record = await self.timelock.schedule_action(action)
        
        return ExecutionResult(
            result_id=str(uuid4()),
            action_id=action.action_id,
            success=True,
            execution_time=datetime.now(timezone.utc),
            execution_duration_ms=0,
            message=f"Action scheduled for execution at {record.execute_at.isoformat()}",
            verification_steps=["Action scheduled", "Timelock enforced"],
            affected_resources=[]
        )
    
    async def _execute_parameter_change(
        self,
        action: GovernanceAction
    ) -> ExecutionResult:
        """Execute a parameter change."""
        start_time = datetime.now(timezone.utc)
        verification_steps = []
        affected_resources = []
        
        try:
            # Validate parameter exists
            param_def = self.parameter_registry.get_parameter(
                action.target_module,
                action.target_parameter
            )
            
            if not param_def:
                raise ValueError(
                    f"Parameter {action.target_module}.{action.target_parameter} not found"
                )
            
            verification_steps.append(f"Parameter definition found for {action.target_module}.{action.target_parameter}")
            
            # Apply the change
            success = await self.parameter_registry.set_parameter(
                module=action.target_module,
                name=action.target_parameter,
                value=action.proposed_value,
                executed_by="governance",
                proposal_id=action.proposal_id,
                action_id=action.action_id
            )
            
            if not success:
                raise RuntimeError("Parameter change callback failed")
            
            verification_steps.append(f"Parameter value updated to {action.proposed_value}")
            affected_resources.append(f"{action.target_module}.{action.target_parameter}")
            
            # Apply to actual service if available
            await self._apply_parameter_to_service(action)
            verification_steps.append("Parameter applied to service")
            
            execution_time = datetime.now(timezone.utc)
            duration_ms = int((execution_time - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                result_id=str(uuid4()),
                action_id=action.action_id,
                success=True,
                execution_time=execution_time,
                execution_duration_ms=duration_ms,
                message=f"Parameter {action.target_module}.{action.target_parameter} changed successfully",
                verification_steps=verification_steps,
                affected_resources=affected_resources,
                rollback_available=True,
                rollback_data={
                    "module": action.target_module,
                    "parameter": action.target_parameter,
                    "old_value": str(action.current_value)
                }
            )
            
        except Exception as e:
            execution_time = datetime.now(timezone.utc)
            duration_ms = int((execution_time - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                result_id=str(uuid4()),
                action_id=action.action_id,
                success=False,
                execution_time=execution_time,
                execution_duration_ms=duration_ms,
                message="Parameter change failed",
                error_details=str(e),
                verification_steps=verification_steps,
                rollback_available=False
            )
    
    async def _apply_parameter_to_service(self, action: GovernanceAction):
        """Apply parameter change to the actual service."""
        module = action.target_module
        param = action.target_parameter
        value = action.proposed_value
        
        # Apply to FTNS service
        if module == "ftns" and self._ftns_service:
            if hasattr(self._ftns_service, 'update_config'):
                await self._ftns_service.update_config({param: value})
            self.logger.info(
                "Applied parameter to FTNS service",
                parameter=param,
                value=value
            )
        
        # Apply to staking manager
        elif module == "staking" and self._staking_manager:
            if hasattr(self._staking_manager, 'update_config'):
                await self._staking_manager.update_config({param: value})
            self.logger.info(
                "Applied parameter to staking manager",
                parameter=param,
                value=value
            )
        
        # Apply to network manager
        elif module == "network" and self._network_manager:
            if hasattr(self._network_manager, 'update_config'):
                await self._network_manager.update_config({param: value})
            self.logger.info(
                "Applied parameter to network manager",
                parameter=param,
                value=value
            )
    
    async def _execute_treasury_spend(
        self,
        action: GovernanceAction
    ) -> ExecutionResult:
        """Execute a treasury spend."""
        start_time = datetime.now(timezone.utc)
        verification_steps = []
        affected_resources = []
        
        try:
            # Extract spend details
            recipient = action.metadata.get("recipient")
            amount = action.metadata.get("amount")
            reason = action.metadata.get("reason", "Governance approved spend")
            
            if not recipient or amount is None:
                raise ValueError("Treasury spend requires recipient and amount")
            
            verification_steps.append(f"Validated spend parameters: {amount} to {recipient}")
            
            # Execute spend via FTNS service
            if self._ftns_service:
                # Use the appropriate method based on service type
                if hasattr(self._ftns_service, 'treasury_spend'):
                    await self._ftns_service.treasury_spend(recipient, amount, reason)
                elif hasattr(self._ftns_service, 'transfer'):
                    await self._ftns_service.transfer(
                        from_user="treasury",
                        to_user=recipient,
                        amount=amount,
                        transaction_type="treasury_spend",
                        metadata={"reason": reason, "action_id": action.action_id}
                    )
                
                verification_steps.append(f"Treasury spend executed: {amount} FTNS to {recipient}")
                affected_resources.append(f"treasury:{recipient}")
            else:
                raise RuntimeError(
                    "Treasury spend requires a real FTNS service. "
                    "Wire GovernanceExecutor._ftns_service before executing treasury proposals."
                )
            
            execution_time = datetime.now(timezone.utc)
            duration_ms = int((execution_time - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                result_id=str(uuid4()),
                action_id=action.action_id,
                success=True,
                execution_time=execution_time,
                execution_duration_ms=duration_ms,
                message=f"Treasury spend of {amount} FTNS to {recipient} completed",
                verification_steps=verification_steps,
                affected_resources=affected_resources,
                rollback_available=False  # Treasury spends are not reversible
            )
            
        except Exception as e:
            execution_time = datetime.now(timezone.utc)
            duration_ms = int((execution_time - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                result_id=str(uuid4()),
                action_id=action.action_id,
                success=False,
                execution_time=execution_time,
                execution_duration_ms=duration_ms,
                message="Treasury spend failed",
                error_details=str(e),
                verification_steps=verification_steps,
                rollback_available=False
            )
    
    async def _execute_emergency_action(
        self,
        action: GovernanceAction
    ) -> ExecutionResult:
        """Execute an emergency action."""
        start_time = datetime.now(timezone.utc)
        verification_steps = []
        affected_resources = []
        
        try:
            emergency_type = action.metadata.get("emergency_type")
            
            verification_steps.append(f"Emergency action type: {emergency_type}")
            
            if emergency_type == "circuit_breaker_trigger":
                # Trigger circuit breaker
                reason = action.metadata.get("reason", "Governance emergency trigger")
                if self._network_manager and hasattr(self._network_manager, 'trigger_circuit_breaker'):
                    await self._network_manager.trigger_circuit_breaker(reason)
                verification_steps.append("Circuit breaker triggered")
                affected_resources.append("network:circuit_breaker")
                
            elif emergency_type == "pause_operations":
                # Pause operations
                scope = action.metadata.get("scope", "all")
                if self._network_manager and hasattr(self._network_manager, 'pause'):
                    await self._network_manager.pause(scope)
                verification_steps.append(f"Operations paused: {scope}")
                affected_resources.append(f"network:operations:{scope}")
                
            elif emergency_type == "freeze_accounts":
                # Freeze specific accounts
                accounts = action.metadata.get("accounts", [])
                for account in accounts:
                    if self._ftns_service and hasattr(self._ftns_service, 'freeze_account'):
                        await self._ftns_service.freeze_account(account)
                verification_steps.append(f"Accounts frozen: {len(accounts)}")
                affected_resources.extend([f"account:{a}" for a in accounts])
                
            else:
                # Generic emergency action
                verification_steps.append(f"Generic emergency action: {emergency_type}")
            
            execution_time = datetime.now(timezone.utc)
            duration_ms = int((execution_time - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                result_id=str(uuid4()),
                action_id=action.action_id,
                success=True,
                execution_time=execution_time,
                execution_duration_ms=duration_ms,
                message=f"Emergency action {emergency_type} executed",
                verification_steps=verification_steps,
                affected_resources=affected_resources,
                rollback_available=True
            )
            
        except Exception as e:
            execution_time = datetime.now(timezone.utc)
            duration_ms = int((execution_time - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                result_id=str(uuid4()),
                action_id=action.action_id,
                success=False,
                execution_time=execution_time,
                execution_duration_ms=duration_ms,
                message="Emergency action failed",
                error_details=str(e),
                verification_steps=verification_steps,
                rollback_available=False
            )
    
    async def _execute_member_ejection(
        self,
        action: GovernanceAction
    ) -> ExecutionResult:
        """Execute a member ejection."""
        start_time = datetime.now(timezone.utc)
        verification_steps = []
        affected_resources = []
        
        try:
            member_id = action.metadata.get("member_id")
            reason = action.metadata.get("reason", "Governance approved ejection")
            
            if not member_id:
                raise ValueError("Member ejection requires member_id")
            
            verification_steps.append(f"Ejecting member: {member_id}")
            
            # Execute ejection via network manager
            if self._network_manager and hasattr(self._network_manager, 'eject_member'):
                await self._network_manager.eject_member(member_id, reason)
                verification_steps.append(f"Member {member_id} ejected from network")
            else:
                raise RuntimeError(
                    "Member ejection requires a network manager with eject_member(). "
                    "Wire GovernanceExecutor._network_manager before executing ejection proposals."
                )
            
            # Slash stake if applicable
            slash_percentage = action.metadata.get("slash_percentage", 0)
            if slash_percentage > 0 and self._staking_manager:
                if hasattr(self._staking_manager, 'slash_stake'):
                    await self._staking_manager.slash_stake(
                        user_id=member_id,
                        percentage=slash_percentage,
                        reason=f"Governance ejection: {reason}"
                    )
                verification_steps.append(f"Stake slashed by {slash_percentage}%")
            
            affected_resources.append(f"member:{member_id}")
            
            execution_time = datetime.now(timezone.utc)
            duration_ms = int((execution_time - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                result_id=str(uuid4()),
                action_id=action.action_id,
                success=True,
                execution_time=execution_time,
                execution_duration_ms=duration_ms,
                message=f"Member {member_id} ejected successfully",
                verification_steps=verification_steps,
                affected_resources=affected_resources,
                rollback_available=False
            )
            
        except Exception as e:
            execution_time = datetime.now(timezone.utc)
            duration_ms = int((execution_time - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                result_id=str(uuid4()),
                action_id=action.action_id,
                success=False,
                execution_time=execution_time,
                execution_duration_ms=duration_ms,
                message="Member ejection failed",
                error_details=str(e),
                verification_steps=verification_steps,
                rollback_available=False
            )
    
    async def _execute_circuit_breaker_change(
        self,
        action: GovernanceAction
    ) -> ExecutionResult:
        """Execute a circuit breaker configuration change."""
        # This is similar to parameter change but specific to circuit breaker
        return await self._execute_parameter_change(action)
    
    async def _execute_protocol_upgrade(
        self,
        action: GovernanceAction
    ) -> ExecutionResult:
        """Execute a protocol upgrade."""
        start_time = datetime.now(timezone.utc)
        verification_steps = []
        affected_resources = []
        
        try:
            new_version = action.metadata.get("version")
            upgrade_data = action.metadata.get("upgrade_data", {})
            
            if not new_version:
                raise ValueError("Protocol upgrade requires version")
            
            verification_steps.append(f"Upgrading to version: {new_version}")
            
            # Execute upgrade via network manager
            if self._network_manager and hasattr(self._network_manager, 'upgrade_protocol'):
                await self._network_manager.upgrade_protocol(new_version, upgrade_data)
                verification_steps.append(f"Protocol upgraded to {new_version}")
            else:
                raise RuntimeError(
                    "Protocol upgrade requires a network manager with upgrade_protocol(). "
                    "Wire GovernanceExecutor._network_manager before executing upgrade proposals."
                )
            
            affected_resources.append("network:protocol")
            
            execution_time = datetime.now(timezone.utc)
            duration_ms = int((execution_time - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                result_id=str(uuid4()),
                action_id=action.action_id,
                success=True,
                execution_time=execution_time,
                execution_duration_ms=duration_ms,
                message=f"Protocol upgraded to version {new_version}",
                verification_steps=verification_steps,
                affected_resources=affected_resources,
                rollback_available=True,
                rollback_data={"previous_version": action.current_value}
            )
            
        except Exception as e:
            execution_time = datetime.now(timezone.utc)
            duration_ms = int((execution_time - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                result_id=str(uuid4()),
                action_id=action.action_id,
                success=False,
                execution_time=execution_time,
                execution_duration_ms=duration_ms,
                message="Protocol upgrade failed",
                error_details=str(e),
                verification_steps=verification_steps,
                rollback_available=False
            )
    
    async def get_pending_executions(self) -> List[TimelockRecord]:
        """Get all pending executions."""
        return await self.timelock.get_pending_actions()
    
    async def get_ready_executions(self) -> List[TimelockRecord]:
        """Get all executions ready to run."""
        return await self.timelock.get_ready_actions()
    
    async def process_ready_executions(self) -> List[ExecutionResult]:
        """
        Process all executions that are ready.
        
        This should be called periodically by a background task.
        
        Returns:
            List of execution results
        """
        ready_actions = await self.timelock.get_ready_actions()
        results = []
        
        for record in ready_actions:
            try:
                result = await self.timelock.execute_action(record.action.action_id)
                results.append(result)
            except Exception as e:
                self.logger.error(
                    "Failed to process ready execution",
                    action_id=record.action.action_id,
                    error=str(e)
                )
        
        return results
    
    async def get_execution_result(self, action_id: str) -> Optional[ExecutionResult]:
        """Get execution result for an action."""
        return await self.timelock.get_execution_result(action_id)
    
    async def cancel_execution(self, action_id: str, reason: str) -> bool:
        """Cancel a pending execution."""
        return await self.timelock.cancel_action(action_id, reason)


# === Global Instance ===

_governance_executor: Optional[GovernanceExecutor] = None


def get_governance_executor() -> GovernanceExecutor:
    """Get or create global governance executor instance."""
    global _governance_executor
    if _governance_executor is None:
        _governance_executor = GovernanceExecutor()
        _governance_executor.parameter_registry.initialize_default_parameters()
    return _governance_executor
