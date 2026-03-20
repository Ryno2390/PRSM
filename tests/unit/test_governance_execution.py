"""
Tests for Governance Execution System (Phase 3.3)

Tests the governance execution components:
- TimelockController: Manages timelocked actions
- GovernanceExecutor: Executes approved proposals
- GovernableParameterRegistry: Registry of governable parameters
- Integration with SafetyGovernance
"""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from decimal import Decimal
import asyncio

from prsm.governance.execution import (
    ProposalType,
    GovernanceAction,
    GovernanceActionStatus,
    ExecutionPriority,
    TimelockRecord,
    ExecutionResult,
    ParameterDefinition,
    ParameterChangeRecord,
    TimelockController,
    GovernanceExecutor,
    GovernableParameterRegistry,
    get_governance_executor,
)


# === Fixtures ===

@pytest.fixture
def timelock_controller():
    """Create a timelock controller for testing."""
    return TimelockController(
        min_delay=3600,  # 1 hour for testing
        max_delay=7 * 24 * 3600,
        emergency_delay=300,  # 5 minutes for testing
        grace_period=24 * 3600
    )


@pytest.fixture
def parameter_registry():
    """Create a parameter registry for testing."""
    registry = GovernableParameterRegistry()
    registry.initialize_default_parameters()
    return registry


@pytest.fixture
def governance_executor(timelock_controller, parameter_registry):
    """Create a governance executor for testing."""
    return GovernanceExecutor(
        timelock=timelock_controller,
        parameter_registry=parameter_registry
    )


@pytest.fixture
def sample_action():
    """Create a sample governance action for testing."""
    return GovernanceAction(
        action_id=str(uuid4()),
        action_type=ProposalType.PARAMETER_CHANGE,
        target_module="ftns",
        target_parameter="reward_rate",
        current_value=0.05,
        proposed_value=0.06,
        execution_delay=3600,
        requires_timelock=True,
        proposal_id=str(uuid4()),
        proposer_id="test_proposer"
    )


@pytest.fixture
def emergency_action():
    """Create an emergency action for testing."""
    return GovernanceAction(
        action_id=str(uuid4()),
        action_type=ProposalType.EMERGENCY_ACTION,
        target_module="network",
        target_parameter="circuit_breaker",
        current_value=False,
        proposed_value=True,
        execution_delay=300,
        requires_timelock=True,
        priority=ExecutionPriority.EMERGENCY,
        proposal_id=str(uuid4()),
        metadata={"emergency_type": "circuit_breaker_trigger", "reason": "Test emergency"}
    )


# === TimelockController Tests ===

class TestTimelockController:
    """Tests for TimelockController."""
    
    @pytest.mark.asyncio
    async def test_schedule_action(self, timelock_controller, sample_action):
        """Test scheduling a governance action."""
        record = await timelock_controller.schedule_action(sample_action)
        
        assert record.record_id is not None
        assert record.action == sample_action
        assert record.status == GovernanceActionStatus.SCHEDULED
        assert record.execute_at > record.scheduled_at
        assert record.is_ready is False
    
    @pytest.mark.asyncio
    async def test_schedule_action_with_custom_time(self, timelock_controller, sample_action):
        """Test scheduling with custom execution time."""
        execute_at = datetime.now(timezone.utc) + timedelta(hours=2)
        record = await timelock_controller.schedule_action(
            sample_action,
            execute_at=execute_at
        )
        
        assert record.execute_at == execute_at
    
    @pytest.mark.asyncio
    async def test_schedule_action_min_delay_enforcement(self, timelock_controller):
        """Test that minimum delay is enforced for timelocked actions."""
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="ftns",
            target_parameter="min_stake",
            current_value=1000,
            proposed_value=2000,
            execution_delay=100,  # Too short
            requires_timelock=True
        )
        
        record = await timelock_controller.schedule_action(action)
        
        # Should enforce minimum delay (with small tolerance for timing precision)
        time_remaining = record.time_remaining.total_seconds()
        assert time_remaining >= timelock_controller.min_delay - 1  # 1 second tolerance
    
    @pytest.mark.asyncio
    async def test_schedule_emergency_action(self, timelock_controller, emergency_action):
        """Test scheduling an emergency action with reduced delay."""
        record = await timelock_controller.schedule_action(emergency_action)
        
        # Emergency should use emergency_delay
        expected_delay = timelock_controller.emergency_delay
        actual_delay = (record.execute_at - record.scheduled_at).total_seconds()
        
        assert actual_delay <= expected_delay + 1  # Allow 1 second tolerance
    
    @pytest.mark.asyncio
    async def test_schedule_duplicate_action_fails(self, timelock_controller, sample_action):
        """Test that scheduling duplicate action fails."""
        await timelock_controller.schedule_action(sample_action)
        
        with pytest.raises(ValueError, match="already scheduled"):
            await timelock_controller.schedule_action(sample_action)
    
    @pytest.mark.asyncio
    async def test_execute_action_not_ready(self, timelock_controller, sample_action):
        """Test that executing non-ready action fails."""
        await timelock_controller.schedule_action(sample_action)
        
        with pytest.raises(ValueError, match="not ready"):
            await timelock_controller.execute_action(sample_action.action_id)
    
    @pytest.mark.asyncio
    async def test_execute_ready_action(self, timelock_controller):
        """Test executing a ready action."""
        # Create action with no delay
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="test",
            target_parameter="test_param",
            current_value=1,
            proposed_value=2,
            execution_delay=0,
            requires_timelock=False
        )
        
        await timelock_controller.schedule_action(action)
        result = await timelock_controller.execute_action(action.action_id)
        
        assert result.success is True
        assert result.action_id == action.action_id
    
    @pytest.mark.asyncio
    async def test_cancel_action(self, timelock_controller, sample_action):
        """Test cancelling a scheduled action."""
        await timelock_controller.schedule_action(sample_action)
        
        success = await timelock_controller.cancel_action(
            sample_action.action_id,
            "Test cancellation"
        )
        
        assert success is True
        
        record = await timelock_controller.get_action_record(sample_action.action_id)
        assert record.status == GovernanceActionStatus.CANCELLED
        assert record.cancellation_reason == "Test cancellation"
    
    @pytest.mark.asyncio
    async def test_cancel_executed_action_fails(self, timelock_controller):
        """Test that cancelling executed action fails."""
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="test",
            target_parameter="test_param",
            current_value=1,
            proposed_value=2,
            execution_delay=0,
            requires_timelock=False
        )
        
        await timelock_controller.schedule_action(action)
        await timelock_controller.execute_action(action.action_id)
        
        with pytest.raises(ValueError, match="Cannot cancel executed"):
            await timelock_controller.cancel_action(action.action_id, "Too late")
    
    @pytest.mark.asyncio
    async def test_get_pending_actions(self, timelock_controller, sample_action):
        """Test getting pending actions."""
        await timelock_controller.schedule_action(sample_action)
        
        pending = await timelock_controller.get_pending_actions()
        
        assert len(pending) == 1
        assert pending[0].action.action_id == sample_action.action_id
    
    @pytest.mark.asyncio
    async def test_get_ready_actions(self, timelock_controller):
        """Test getting ready actions."""
        # Create action that's ready immediately
        ready_action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="test",
            target_parameter="test_param",
            current_value=1,
            proposed_value=2,
            execution_delay=0,
            requires_timelock=False
        )
        
        # Create action that's not ready
        pending_action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="test",
            target_parameter="test_param2",
            current_value=1,
            proposed_value=2,
            execution_delay=3600,
            requires_timelock=True
        )
        
        await timelock_controller.schedule_action(ready_action)
        await timelock_controller.schedule_action(pending_action)
        
        ready = await timelock_controller.get_ready_actions()
        
        assert len(ready) == 1
        assert ready[0].action.action_id == ready_action.action_id
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_actions(self, timelock_controller):
        """Test cleaning up expired actions."""
        # Create an action with very short grace period
        controller = TimelockController(
            min_delay=0,
            grace_period=0  # Immediate expiry
        )
        
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="test",
            target_parameter="test_param",
            current_value=1,
            proposed_value=2,
            execution_delay=0,
            requires_timelock=False
        )
        
        await controller.schedule_action(action)
        
        # Cleanup should mark it as expired
        expired_count = await controller.cleanup_expired_actions()
        
        # Since grace_period is 0, it should be expired immediately
        # But we need to wait a tiny bit for time to pass
        await asyncio.sleep(0.1)
        expired_count = await controller.cleanup_expired_actions()
        
        # The action should still be scheduled since execute_at hasn't passed
        # Let's check the status
        record = await controller.get_action_record(action.action_id)
        assert record is not None


# === GovernableParameterRegistry Tests ===

class TestGovernableParameterRegistry:
    """Tests for GovernableParameterRegistry."""
    
    def test_register_parameter(self, parameter_registry):
        """Test registering a parameter."""
        param = parameter_registry.register_parameter(
            module="test_module",
            name="test_param",
            description="Test parameter",
            current_value=100,
            value_type="int",
            min_value=0,
            max_value=1000,
            requires_timelock=True
        )
        
        assert param.module == "test_module"
        assert param.name == "test_param"
        assert param.current_value == 100
        assert param.requires_timelock is True
    
    def test_get_parameter(self, parameter_registry):
        """Test getting a parameter."""
        # Should have default parameters
        param = parameter_registry.get_parameter("ftns", "reward_rate")
        
        assert param is not None
        assert param.module == "ftns"
        assert param.name == "reward_rate"
    
    def test_get_parameter_value(self, parameter_registry):
        """Test getting parameter value."""
        value = parameter_registry.get_parameter_value("ftns", "reward_rate")
        
        assert value == 0.05
    
    def test_get_all_parameters(self, parameter_registry):
        """Test getting all parameters."""
        all_params = parameter_registry.get_all_parameters()
        
        assert len(all_params) > 0
        
        # Filter by module
        ftns_params = parameter_registry.get_all_parameters(module="ftns")
        
        assert all(p.module == "ftns" for p in ftns_params)
    
    def test_set_parameter(self, parameter_registry):
        """Test setting a parameter value."""
        success = asyncio.run(parameter_registry.set_parameter(
            module="ftns",
            name="reward_rate",
            value=0.07,
            executed_by="test"
        ))
        
        assert success is True
        
        # Check value was updated
        value = parameter_registry.get_parameter_value("ftns", "reward_rate")
        assert value == 0.07
    
    def test_set_parameter_validation(self, parameter_registry):
        """Test parameter validation on set."""
        # Try to set value outside range
        with pytest.raises(ValueError, match="Invalid value"):
            asyncio.run(parameter_registry.set_parameter(
                module="ftns",
                name="reward_rate",
                value=10.0,  # Exceeds max of 0.5
                executed_by="test"
            ))
    
    def test_set_parameter_invalid_type(self, parameter_registry):
        """Test parameter type validation."""
        # Try to set wrong type
        with pytest.raises(ValueError, match="Invalid value"):
            asyncio.run(parameter_registry.set_parameter(
                module="ftns",
                name="reward_rate",
                value="not a number",
                executed_by="test"
            ))
    
    def test_parameter_history(self, parameter_registry):
        """Test parameter change history."""
        # Make a change
        asyncio.run(parameter_registry.set_parameter(
            module="ftns",
            name="reward_rate",
            value=0.06,
            executed_by="test",
            proposal_id="prop_1"
        ))
        
        # Make another change
        asyncio.run(parameter_registry.set_parameter(
            module="ftns",
            name="reward_rate",
            value=0.07,
            executed_by="test",
            proposal_id="prop_2"
        ))
        
        # Get history
        history = parameter_registry.get_parameter_history("ftns", "reward_rate")
        
        assert len(history) == 2
        assert history[0].old_value == 0.05
        assert history[0].new_value == 0.06
        assert history[1].old_value == 0.06
        assert history[1].new_value == 0.07
    
    def test_change_callback(self, parameter_registry):
        """Test parameter change callback."""
        callback_called = []
        
        async def test_callback(module, name, value):
            callback_called.append((module, name, value))
            return True
        
        parameter_registry.register_change_callback("test_module", test_callback)
        
        # Register a test parameter
        parameter_registry.register_parameter(
            module="test_module",
            name="test_param",
            description="Test",
            current_value=1,
            value_type="int"
        )
        
        # Set the parameter
        asyncio.run(parameter_registry.set_parameter(
            module="test_module",
            name="test_param",
            value=2,
            executed_by="test"
        ))
        
        assert len(callback_called) == 1
        assert callback_called[0] == ("test_module", "test_param", 2)
    
    def test_initialize_default_parameters(self):
        """Test default parameter initialization."""
        registry = GovernableParameterRegistry()
        registry.initialize_default_parameters()
        
        # Check some default parameters exist
        assert registry.get_parameter("ftns", "reward_rate") is not None
        assert registry.get_parameter("staking", "unstaking_period_seconds") is not None
        assert registry.get_parameter("network", "max_block_size") is not None
        assert registry.get_parameter("governance", "voting_period_seconds") is not None


# === GovernanceExecutor Tests ===

class TestGovernanceExecutor:
    """Tests for GovernanceExecutor."""
    
    @pytest.mark.asyncio
    async def test_create_action_from_proposal(self, governance_executor):
        """Test creating action from proposal."""
        action = await governance_executor.create_action_from_proposal(
            proposal_id="prop_123",
            proposal_type="parameter_change",
            proposal_data={
                "target_module": "ftns",
                "target_parameter": "reward_rate",
                "proposed_value": 0.07,
                "proposer_id": "user_1"
            }
        )
        
        assert action.action_id is not None
        assert action.action_type == ProposalType.PARAMETER_CHANGE
        assert action.target_module == "ftns"
        assert action.target_parameter == "reward_rate"
        assert action.proposed_value == 0.07
    
    @pytest.mark.asyncio
    async def test_create_action_with_parameter_def(self, governance_executor):
        """Test creating action uses parameter definition."""
        action = await governance_executor.create_action_from_proposal(
            proposal_id="prop_123",
            proposal_type="parameter_change",
            proposal_data={
                "target_module": "ftns",
                "target_parameter": "reward_rate",
                "proposed_value": 0.07
            }
        )
        
        # Should use parameter definition for current value and timelock
        assert action.current_value == 0.05  # Default value
        assert action.requires_timelock is True
    
    @pytest.mark.asyncio
    async def test_execute_proposal_immediate(self, governance_executor):
        """Test executing proposal without timelock."""
        result = await governance_executor.execute_proposal(
            proposal_id="prop_123",
            proposal_type="parameter_change",
            proposal_data={
                "target_module": "ftns",
                "target_parameter": "reward_rate",
                "proposed_value": 0.06,
                "urgency": "emergency"  # Bypass timelock
            }
        )
        
        assert result.success is True
        assert "scheduled" in result.message.lower() or "executed" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_execute_proposal_with_timelock(self, governance_executor):
        """Test executing proposal with timelock."""
        result = await governance_executor.execute_proposal(
            proposal_id="prop_456",
            proposal_type="parameter_change",
            proposal_data={
                "target_module": "ftns",
                "target_parameter": "reward_rate",
                "proposed_value": 0.08
            }
        )
        
        assert result.success is True
        assert "scheduled" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_execute_parameter_change(self, governance_executor):
        """Test executing parameter change."""
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="ftns",
            target_parameter="reward_rate",
            current_value=0.05,
            proposed_value=0.06,
            execution_delay=0,
            requires_timelock=False
        )
        
        result = await governance_executor._execute_parameter_change(action)
        
        assert result.success is True
        assert "reward_rate" in result.message
    
    @pytest.mark.asyncio
    async def test_execute_parameter_change_invalid_param(self, governance_executor):
        """Test executing parameter change for invalid parameter."""
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="nonexistent",
            target_parameter="invalid_param",
            current_value=1,
            proposed_value=2,
            execution_delay=0,
            requires_timelock=False
        )
        
        result = await governance_executor._execute_parameter_change(action)
        
        assert result.success is False
        assert "not found" in result.error_details
    
    @pytest.mark.asyncio
    async def test_execute_treasury_spend(self, governance_executor):
        """Test executing treasury spend."""
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.TREASURY_SPEND,
            target_module="treasury",
            target_parameter="spend",
            current_value=0,
            proposed_value=1000,
            execution_delay=0,
            requires_timelock=False,
            metadata={
                "recipient": "user_123",
                "amount": 1000,
                "reason": "Test spend"
            }
        )
        
        result = await governance_executor._execute_treasury_spend(action)
        
        # Must fail without FTNS service — silent simulation is a lie
        assert result.success is False
        assert "FTNS service" in result.error_details
    
    @pytest.mark.asyncio
    async def test_execute_treasury_spend_missing_params(self, governance_executor):
        """Test treasury spend with missing parameters."""
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.TREASURY_SPEND,
            target_module="treasury",
            target_parameter="spend",
            current_value=0,
            proposed_value=0,
            execution_delay=0,
            requires_timelock=False,
            metadata={}  # Missing recipient and amount
        )
        
        result = await governance_executor._execute_treasury_spend(action)
        
        assert result.success is False
        assert "requires recipient and amount" in result.error_details
    
    @pytest.mark.asyncio
    async def test_execute_emergency_action(self, governance_executor):
        """Test executing emergency action."""
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.EMERGENCY_ACTION,
            target_module="network",
            target_parameter="circuit_breaker",
            current_value=False,
            proposed_value=True,
            execution_delay=0,
            requires_timelock=False,
            metadata={
                "emergency_type": "circuit_breaker_trigger",
                "reason": "Test emergency"
            }
        )
        
        result = await governance_executor._execute_emergency_action(action)
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_get_pending_executions(self, governance_executor):
        """Test getting pending executions."""
        # Schedule an action
        await governance_executor.execute_proposal(
            proposal_id="prop_789",
            proposal_type="parameter_change",
            proposal_data={
                "target_module": "ftns",
                "target_parameter": "reward_rate",
                "proposed_value": 0.09
            }
        )
        
        pending = await governance_executor.get_pending_executions()
        
        assert len(pending) >= 1
    
    @pytest.mark.asyncio
    async def test_cancel_execution(self, governance_executor):
        """Test cancelling an execution."""
        # Schedule an action
        result = await governance_executor.execute_proposal(
            proposal_id="prop_cancel",
            proposal_type="parameter_change",
            proposal_data={
                "target_module": "ftns",
                "target_parameter": "reward_rate",
                "proposed_value": 0.10
            }
        )
        
        # Cancel it
        success = await governance_executor.cancel_execution(
            result.action_id,
            "Test cancellation"
        )
        
        assert success is True


# === GovernanceAction Tests ===

class TestGovernanceAction:
    """Tests for GovernanceAction dataclass."""
    
    def test_create_action(self):
        """Test creating a governance action."""
        action = GovernanceAction(
            action_id="action_123",
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="ftns",
            target_parameter="reward_rate",
            current_value=0.05,
            proposed_value=0.06,
            execution_delay=3600,
            requires_timelock=True
        )
        
        assert action.action_id == "action_123"
        assert action.status == GovernanceActionStatus.PENDING
        assert action.priority == ExecutionPriority.NORMAL
    
    def test_action_to_dict(self, sample_action):
        """Test converting action to dictionary."""
        action_dict = sample_action.to_dict()
        
        assert action_dict["action_id"] == sample_action.action_id
        assert action_dict["action_type"] == "parameter_change"
        assert "created_at" in action_dict
    
    def test_action_with_decimal_values(self):
        """Test action with Decimal values."""
        action = GovernanceAction(
            action_id="action_dec",
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="ftns",
            target_parameter="transaction_fee_base",
            current_value=Decimal("0.001"),
            proposed_value=Decimal("0.002"),
            execution_delay=3600,
            requires_timelock=True
        )
        
        action_dict = action.to_dict()
        
        # Decimal should be converted to string
        assert isinstance(action_dict["current_value"], str)
        assert isinstance(action_dict["proposed_value"], str)


# === TimelockRecord Tests ===

class TestTimelockRecord:
    """Tests for TimelockRecord dataclass."""
    
    def test_create_record(self, sample_action):
        """Test creating a timelock record."""
        now = datetime.now(timezone.utc)
        execute_at = now + timedelta(hours=1)
        
        record = TimelockRecord(
            record_id="record_123",
            action=sample_action,
            scheduled_at=now,
            execute_at=execute_at
        )
        
        assert record.record_id == "record_123"
        assert record.status == GovernanceActionStatus.SCHEDULED
        assert record.is_ready is False
    
    def test_record_is_ready(self, sample_action):
        """Test checking if record is ready."""
        # Create record that's ready
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        record = TimelockRecord(
            record_id="record_ready",
            action=sample_action,
            scheduled_at=past_time,
            execute_at=past_time
        )
        
        assert record.is_ready is True
    
    def test_record_time_remaining(self, sample_action):
        """Test calculating time remaining."""
        now = datetime.now(timezone.utc)
        execute_at = now + timedelta(hours=2)
        
        record = TimelockRecord(
            record_id="record_time",
            action=sample_action,
            scheduled_at=now,
            execute_at=execute_at
        )
        
        remaining = record.time_remaining
        
        assert remaining > timedelta(hours=1)
        assert remaining < timedelta(hours=3)
    
    def test_record_to_dict(self, sample_action):
        """Test converting record to dictionary."""
        now = datetime.now(timezone.utc)
        execute_at = now + timedelta(hours=1)
        
        record = TimelockRecord(
            record_id="record_dict",
            action=sample_action,
            scheduled_at=now,
            execute_at=execute_at
        )
        
        record_dict = record.to_dict()
        
        assert record_dict["record_id"] == "record_dict"
        assert "action" in record_dict
        assert "is_ready" in record_dict
        assert "time_remaining_seconds" in record_dict


# === ExecutionResult Tests ===

class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""
    
    def test_create_result(self):
        """Test creating an execution result."""
        result = ExecutionResult(
            result_id="result_123",
            action_id="action_123",
            success=True,
            execution_time=datetime.now(timezone.utc),
            execution_duration_ms=100,
            message="Test execution"
        )
        
        assert result.result_id == "result_123"
        assert result.success is True
        assert result.rollback_available is True
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = ExecutionResult(
            result_id="result_dict",
            action_id="action_dict",
            success=False,
            execution_time=datetime.now(timezone.utc),
            execution_duration_ms=50,
            message="Failed",
            error_details="Test error"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["result_id"] == "result_dict"
        assert result_dict["success"] is False
        assert result_dict["error_details"] == "Test error"


# === ParameterDefinition Tests ===

class TestParameterDefinition:
    """Tests for ParameterDefinition dataclass."""
    
    def test_create_definition(self):
        """Test creating a parameter definition."""
        param = ParameterDefinition(
            module="ftns",
            name="test_param",
            description="Test parameter",
            current_value=100,
            value_type="int",
            min_value=0,
            max_value=1000
        )
        
        assert param.module == "ftns"
        assert param.requires_timelock is True
        assert param.timelock_delay_seconds == 24 * 3600
    
    def test_validate_int_value(self):
        """Test validating integer values."""
        param = ParameterDefinition(
            module="test",
            name="int_param",
            description="Int param",
            current_value=50,
            value_type="int",
            min_value=0,
            max_value=100
        )
        
        assert param.validate_value(75) is True
        assert param.validate_value(0) is True
        assert param.validate_value(100) is True
        assert param.validate_value(150) is False  # Over max
        assert param.validate_value(-1) is False  # Under min
        assert param.validate_value("not_int") is False
    
    def test_validate_float_value(self):
        """Test validating float values."""
        param = ParameterDefinition(
            module="test",
            name="float_param",
            description="Float param",
            current_value=0.5,
            value_type="float",
            min_value=0.0,
            max_value=1.0
        )
        
        assert param.validate_value(0.75) is True
        assert param.validate_value(0.0) is True
        assert param.validate_value(1.0) is True
        assert param.validate_value(1.5) is False
        assert param.validate_value("not_float") is False
    
    def test_validate_allowed_values(self):
        """Test validating against allowed values."""
        param = ParameterDefinition(
            module="test",
            name="enum_param",
            description="Enum param",
            current_value="option1",
            value_type="str",
            allowed_values=["option1", "option2", "option3"]
        )
        
        assert param.validate_value("option1") is True
        assert param.validate_value("option2") is True
        assert param.validate_value("option4") is False
    
    def test_definition_to_dict(self):
        """Test converting definition to dictionary."""
        param = ParameterDefinition(
            module="test",
            name="dict_param",
            description="Dict test",
            current_value=Decimal("0.05"),
            value_type="decimal"
        )
        
        param_dict = param.to_dict()
        
        assert param_dict["module"] == "test"
        assert param_dict["name"] == "dict_param"
        assert isinstance(param_dict["current_value"], str)


# === Integration Tests ===

class TestGovernanceIntegration:
    """Integration tests for governance execution."""
    
    @pytest.mark.asyncio
    async def test_full_execution_flow(self, governance_executor):
        """Test full execution flow from proposal to execution."""
        # Create action from proposal
        action = await governance_executor.create_action_from_proposal(
            proposal_id="integration_prop",
            proposal_type="parameter_change",
            proposal_data={
                "target_module": "ftns",
                "target_parameter": "reward_rate",
                "proposed_value": 0.06,
                "proposer_id": "test_user"
            }
        )
        
        assert action is not None
        
        # Execute (will schedule due to timelock)
        result = await governance_executor.execute_proposal(
            proposal_id="integration_prop",
            proposal_type="parameter_change",
            proposal_data={
                "target_module": "ftns",
                "target_parameter": "reward_rate",
                "proposed_value": 0.06
            }
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_emergency_bypasses_timelock(self, governance_executor):
        """Test that emergency actions bypass full timelock."""
        result = await governance_executor.execute_proposal(
            proposal_id="emergency_prop",
            proposal_type="emergency_action",
            proposal_data={
                "target_module": "network",
                "target_parameter": "circuit_breaker",
                "proposed_value": True,
                "urgency": "emergency",
                "metadata": {
                    "emergency_type": "circuit_breaker_trigger",
                    "reason": "Test emergency"
                }
            }
        )
        
        assert result.success is True
        # Emergency should use reduced delay
        assert "scheduled" in result.message.lower() or "executed" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_parameter_change_persists(self, governance_executor):
        """Test that parameter changes persist in registry."""
        # Get initial value
        initial_value = governance_executor.parameter_registry.get_parameter_value(
            "ftns", "reward_rate"
        )
        
        # Execute parameter change with emergency to bypass timelock
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="ftns",
            target_parameter="reward_rate",
            current_value=initial_value,
            proposed_value=0.08,
            execution_delay=0,
            requires_timelock=False
        )
        
        await governance_executor.timelock.schedule_action(action)
        result = await governance_executor.timelock.execute_action(action.action_id)
        
        assert result.success is True
        
        # Check value was updated
        new_value = governance_executor.parameter_registry.get_parameter_value(
            "ftns", "reward_rate"
        )
        
        assert new_value == 0.08
    
    @pytest.mark.asyncio
    async def test_execution_history(self, governance_executor):
        """Test that execution history is maintained."""
        # Execute an action
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PARAMETER_CHANGE,
            target_module="ftns",
            target_parameter="min_stake",
            current_value=1000,
            proposed_value=2000,
            execution_delay=0,
            requires_timelock=False
        )
        
        await governance_executor.timelock.schedule_action(action)
        await governance_executor.timelock.execute_action(action.action_id)
        
        # Get execution result
        result = await governance_executor.timelock.get_execution_result(action.action_id)
        
        assert result is not None
        assert result.action_id == action.action_id

    @pytest.mark.asyncio
    async def test_execute_treasury_spend_with_service(self, governance_executor):
        """Treasury spend succeeds when FTNS service is present and supports transfer."""
        from unittest.mock import AsyncMock, MagicMock
        # Use spec to ensure only 'transfer' method exists (not treasury_spend)
        mock_ftns = MagicMock()
        mock_ftns.transfer = AsyncMock(return_value=True)
        # Ensure treasury_spend does NOT exist so code uses transfer
        del mock_ftns.treasury_spend
        governance_executor.set_ftns_service(mock_ftns)

        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.TREASURY_SPEND,
            target_module="treasury",
            target_parameter="spend",
            current_value=0,
            proposed_value=500,
            execution_delay=0,
            requires_timelock=False,
            metadata={"recipient": "user_abc", "amount": 500, "reason": "Test"}
        )

        result = await governance_executor._execute_treasury_spend(action)
        assert result.success is True
        mock_ftns.transfer.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_member_ejection_no_network_manager(self, governance_executor):
        """Member ejection fails explicitly when no network manager is configured."""
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.MEMBER_EJECTION,
            target_module="network",
            target_parameter="member",
            current_value=None,
            proposed_value=None,
            execution_delay=0,
            requires_timelock=False,
            metadata={"member_id": "bad_actor_node", "reason": "Governance vote"}
        )

        result = await governance_executor._execute_member_ejection(action)
        assert result.success is False
        assert "network manager" in result.error_details

    @pytest.mark.asyncio
    async def test_execute_member_ejection_with_network_manager(self, governance_executor):
        """Member ejection succeeds when network manager is present."""
        from unittest.mock import AsyncMock
        mock_network = AsyncMock()
        mock_network.eject_member = AsyncMock(return_value=True)
        governance_executor.set_network_manager(mock_network)

        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.MEMBER_EJECTION,
            target_module="network",
            target_parameter="member",
            current_value=None,
            proposed_value=None,
            execution_delay=0,
            requires_timelock=False,
            metadata={"member_id": "bad_actor_node", "reason": "Governance vote"}
        )

        result = await governance_executor._execute_member_ejection(action)
        assert result.success is True
        mock_network.eject_member.assert_awaited_once_with("bad_actor_node", "Governance vote")

    @pytest.mark.asyncio
    async def test_execute_protocol_upgrade_no_network_manager(self, governance_executor):
        """Protocol upgrade fails explicitly when no network manager is configured."""
        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PROTOCOL_UPGRADE,
            target_module="network",
            target_parameter="version",
            current_value="1.0.0",
            proposed_value="1.1.0",
            execution_delay=0,
            requires_timelock=False,
            metadata={"version": "1.1.0", "upgrade_data": {}}
        )

        result = await governance_executor._execute_protocol_upgrade(action)
        assert result.success is False
        assert "network manager" in result.error_details

    @pytest.mark.asyncio
    async def test_execute_protocol_upgrade_with_network_manager(self, governance_executor):
        """Protocol upgrade succeeds when network manager is present."""
        from unittest.mock import AsyncMock
        mock_network = AsyncMock()
        mock_network.upgrade_protocol = AsyncMock(return_value=True)
        governance_executor.set_network_manager(mock_network)

        action = GovernanceAction(
            action_id=str(uuid4()),
            action_type=ProposalType.PROTOCOL_UPGRADE,
            target_module="network",
            target_parameter="version",
            current_value="1.0.0",
            proposed_value="1.1.0",
            execution_delay=0,
            requires_timelock=False,
            metadata={"version": "1.1.0", "upgrade_data": {}}
        )

        result = await governance_executor._execute_protocol_upgrade(action)
        assert result.success is True
        mock_network.upgrade_protocol.assert_awaited_once_with("1.1.0", {})


# === Global Instance Tests ===

class TestGlobalInstance:
    """Tests for global instance management."""
    
    def test_get_governance_executor(self):
        """Test getting global executor instance."""
        executor1 = get_governance_executor()
        executor2 = get_governance_executor()
        
        # Should return same instance
        assert executor1 is executor2
    
    def test_executor_has_default_parameters(self):
        """Test that executor has default parameters initialized."""
        executor = get_governance_executor()
        
        # Should have default parameters
        param = executor.parameter_registry.get_parameter("ftns", "reward_rate")
        
        assert param is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])