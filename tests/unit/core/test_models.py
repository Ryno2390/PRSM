"""
Unit Tests for PRSM Core Models
Data validation and model constraint testing
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from uuid import uuid4, UUID
from typing import Dict, Any

from pydantic import ValidationError

from prsm.core.models import (
    PRSMSession, UserInput, FTNSTransaction, FTNSBalance, ArchitectTask,
    TeacherModel, CircuitBreakerEvent, GovernanceProposal, Vote,
    ContextUsage, PeerNode, PerformanceMetric, SafetyLevel,
    TaskStatus, SafetyFlag, ReasoningStep
)


class TestPRSMSession:
    """Unit tests for PRSMSession model"""
    
    def test_prsm_session_creation_valid(self):
        """Test creating a valid PRSM session"""
        session = PRSMSession(
            session_id=uuid4(),
            user_id="test_user_001",
            nwtn_context_allocation=100,
            status=TaskStatus.IN_PROGRESS
        )
        
        assert isinstance(session.session_id, UUID)
        assert session.user_id == "test_user_001"
        assert session.nwtn_context_allocation == 100
        assert session.status == TaskStatus.IN_PROGRESS
        assert session.context_used == 0  # Default value
    
    def test_prsm_session_required_fields(self):
        """Test that required fields are enforced"""
        with pytest.raises(ValidationError) as exc_info:
            PRSMSession()
        
        errors = exc_info.value.errors()
        required_fields = {"user_id"}  # Only user_id is required, others have defaults
        error_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
        
        assert required_fields.issubset(error_fields)
    
    def test_prsm_session_invalid_status(self):
        """Test session with invalid status"""
        with pytest.raises(ValidationError) as exc_info:
            PRSMSession(
                user_id="test_user",
                status="invalid_status"
            )
        
        assert any(
            "invalid_status" in str(error) 
            for error in exc_info.value.errors()
        )
    
    def test_prsm_session_defaults(self):
        """Test default values for optional fields"""
        session = PRSMSession(
            user_id="test_user"
        )
        
        assert isinstance(session.session_id, UUID)  # Generated automatically
        assert session.status == TaskStatus.PENDING
        assert session.nwtn_context_allocation == 0
        assert session.context_used == 0
        assert session.reasoning_trace == []
        assert session.safety_flags == []
        assert session.metadata == {}
    
    @pytest.mark.parametrize("status", [
        TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, 
        TaskStatus.FAILED, TaskStatus.CANCELLED
    ])
    def test_prsm_session_valid_statuses(self, status):
        """Test all valid session statuses"""
        session = PRSMSession(
            user_id="test_user",
            status=status
        )
        assert session.status == status


class TestUserInput:
    """Unit tests for UserInput model"""
    
    def test_user_input_creation_valid(self):
        """Test creating a valid user input"""
        user_input = UserInput(
            user_id="test_user_001",
            prompt="Test quantum field analysis",
            context_allocation=100,
            preferences={"max_budget": 100.0}
        )
        
        assert user_input.user_id == "test_user_001"
        assert user_input.prompt == "Test quantum field analysis"
        assert user_input.context_allocation == 100
        assert user_input.preferences["max_budget"] == 100.0
    
    def test_user_input_required_fields(self):
        """Test that required fields are enforced"""
        with pytest.raises(ValidationError) as exc_info:
            UserInput()
        
        errors = exc_info.value.errors()
        required_fields = {"user_id", "prompt"}
        error_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
        
        assert required_fields.issubset(error_fields)
    
    def test_user_input_defaults(self):
        """Test default values for optional fields"""
        user_input = UserInput(
            user_id="test_user",
            prompt="Test prompt"
        )
        
        assert user_input.context_allocation is None
        assert user_input.preferences == {}
        assert user_input.session_id is None


class TestFTNSTransaction:
    """Unit tests for FTNS Transaction model"""
    
    def test_ftns_transaction_creation_valid(self):
        """Test creating a valid FTNS transaction"""
        transaction = FTNSTransaction(
            to_user="test_user_001",
            amount=25.50,
            transaction_type="context_access",
            description="Context allocation for quantum analysis"
        )
        
        assert isinstance(transaction.transaction_id, UUID)
        assert transaction.to_user == "test_user_001"
        assert transaction.amount == 25.50
        assert transaction.transaction_type == "context_access"
        assert isinstance(transaction.created_at, datetime)
    
    def test_ftns_transaction_required_fields(self):
        """Test that required fields are enforced"""
        with pytest.raises(ValidationError) as exc_info:
            FTNSTransaction()
        
        errors = exc_info.value.errors()
        required_fields = {"to_user", "amount", "transaction_type", "description"}
        error_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
        
        assert required_fields.issubset(error_fields)
    
    def test_ftns_transaction_decimal_precision(self):
        """Test decimal precision for financial amounts"""
        # Test high precision float
        transaction = FTNSTransaction(
            to_user="test_user",
            amount=123.456789012345,
            transaction_type="reward",
            description="High precision reward"
        )
        
        assert isinstance(transaction.amount, float)
        assert transaction.amount == 123.456789012345
    
    def test_ftns_transaction_negative_amount(self):
        """Test transaction with negative amount (spending)"""
        transaction = FTNSTransaction(
            to_user="test_user",
            amount=-50.00,
            transaction_type="context_access",
            description="Context usage charge"
        )
        
        assert transaction.amount == -50.00
    
    @pytest.mark.parametrize("tx_type", [
        "reward", "charge", "transfer", "dividend"
    ])
    def test_ftns_transaction_valid_types(self, tx_type):
        """Test all valid transaction types"""
        transaction = FTNSTransaction(
            to_user="test_user",
            amount=10.00,
            transaction_type=tx_type,
            description=f"Test {tx_type} transaction"
        )
        assert transaction.transaction_type == tx_type
    
    def test_ftns_transaction_optional_fields(self):
        """Test transaction with optional fields"""
        transaction = FTNSTransaction(
            from_user="sender_user",
            to_user="test_user",
            amount=10.00,
            transaction_type="transfer",
            description="User to user transfer",
            context_units=100,
            ipfs_cid="QmTestHash123"
        )
        assert transaction.from_user == "sender_user"
        assert transaction.context_units == 100
        assert transaction.ipfs_cid == "QmTestHash123"


class TestFTNSBalance:
    """Unit tests for FTNS Balance model"""
    
    def test_ftns_balance_creation_valid(self):
        """Test creating a valid FTNS balance"""
        balance = FTNSBalance(
            user_id="test_user_001",
            balance=1000.0,
            locked_balance=50.0
        )
        
        assert balance.user_id == "test_user_001"
        assert balance.balance == 1000.0
        assert balance.locked_balance == 50.0
    
    def test_ftns_balance_defaults(self):
        """Test default values for FTNS balance"""
        balance = FTNSBalance(user_id="test_user")
        
        assert balance.balance == 0.0
        assert balance.locked_balance == 0.0
        assert balance.last_dividend is None
        assert isinstance(balance.created_at, datetime)
    
    def test_ftns_balance_available_calculation(self):
        """Test available balance calculation"""
        balance = FTNSBalance(
            user_id="test_user",
            balance=1000.0,
            locked_balance=200.0
        )
        
        # Available should be total - locked
        available = balance.balance - balance.locked_balance
        assert available == 800.0
    
    def test_ftns_balance_non_negative_constraints(self):
        """Test balance non-negative constraints"""
        # Both balance and locked_balance have ge=0.0 constraints
        balance = FTNSBalance(
            user_id="test_user",
            balance=100.0,
            locked_balance=50.0
        )
        
        assert balance.balance >= 0.0
        assert balance.locked_balance >= 0.0
        
        # Test that negative values should raise validation error
        with pytest.raises(ValidationError):
            FTNSBalance(
                user_id="test_user",
                balance=-10.0  # Should fail validation
            )


class TestArchitectTask:
    """Unit tests for ArchitectTask model"""
    
    def test_architect_task_creation_valid(self):
        """Test creating a valid architect task"""
        session_id = uuid4()
        task = ArchitectTask(
            session_id=session_id,
            instruction="Analyze quantum field interactions",
            complexity_score=0.75,
            status=TaskStatus.PENDING
        )
        
        assert isinstance(task.task_id, UUID)
        assert task.session_id == session_id
        assert task.instruction == "Analyze quantum field interactions"
        assert task.complexity_score == 0.75
        assert task.status == TaskStatus.PENDING
    
    def test_architect_task_complexity_bounds(self):
        """Test complexity bounds validation"""
        session_id = uuid4()
        # Valid complexity values
        task = ArchitectTask(
            session_id=session_id,
            instruction="Test task",
            complexity_score=0.5,
            status=TaskStatus.PENDING
        )
        assert 0.0 <= task.complexity_score <= 1.0
        
        # Test boundary values
        for complexity in [0.0, 0.25, 0.5, 0.75, 1.0]:
            task = ArchitectTask(
                session_id=session_id,
                instruction="Test task",
                complexity_score=complexity,
                status=TaskStatus.PENDING
            )
            assert task.complexity_score == complexity
    
    def test_architect_task_hierarchy_levels(self):
        """Test task hierarchy level validation"""
        session_id = uuid4()
        parent_id = uuid4()
        
        # Test different hierarchy levels
        for level in range(0, 5):
            task = ArchitectTask(
                session_id=session_id,
                parent_task_id=parent_id if level > 0 else None,
                level=level,
                instruction="Test task",
                status=TaskStatus.PENDING
            )
            assert task.level == level
    
    @pytest.mark.parametrize("status", [
        TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.FAILED
    ])
    def test_architect_task_valid_statuses(self, status):
        """Test valid task statuses"""
        task = ArchitectTask(
            session_id=uuid4(),
            instruction="Test task",
            status=status
        )
        assert task.status == status


class TestSafetyModels:
    """Unit tests for safety-related models"""
    
    def test_safety_flag_creation(self):
        """Test creating a safety flag"""
        flag = SafetyFlag(
            level=SafetyLevel.HIGH,
            category="content_safety",
            description="Potential harmful content detected",
            triggered_by="content_filter"
        )
        
        assert isinstance(flag.flag_id, UUID)
        assert flag.level == SafetyLevel.HIGH
        assert flag.description == "Potential harmful content detected"
        assert flag.triggered_by == "content_filter"
        assert flag.resolved == False
    
    def test_circuit_breaker_event_creation(self):
        """Test creating a circuit breaker event"""
        event = CircuitBreakerEvent(
            triggered_by="nwtn_orchestrator",
            reason="excessive_resource_usage",
            safety_level=SafetyLevel.CRITICAL,
            affected_components=["nwtn_orchestrator", "agent_framework"],
            resolution_action="halt_processing"
        )
        
        assert event.triggered_by == "nwtn_orchestrator"
        assert event.reason == "excessive_resource_usage"
        assert event.safety_level == SafetyLevel.CRITICAL
        assert event.resolution_action == "halt_processing"
        assert "nwtn_orchestrator" in event.affected_components
    
    @pytest.mark.parametrize("safety_level", [SafetyLevel.LOW, SafetyLevel.MEDIUM, SafetyLevel.HIGH, SafetyLevel.CRITICAL])
    def test_safety_level_enum_values(self, safety_level):
        """Test valid safety level enum values"""
        event = CircuitBreakerEvent(
            triggered_by="test_component",
            reason="test_trigger",
            safety_level=safety_level
        )
        assert event.safety_level == safety_level


class TestGovernanceModels:
    """Unit tests for governance-related models"""
    
    def test_governance_proposal_creation(self):
        """Test creating a governance proposal"""
        now = datetime.now(timezone.utc)
        proposal = GovernanceProposal(
            title="Increase context unit base cost",
            description="Proposal to adjust FTNS pricing",
            proposer_id="governance_user_001",
            proposal_type="economic",
            voting_starts=now,
            voting_ends=now + timedelta(days=7)
        )
        
        assert proposal.title == "Increase context unit base cost"
        assert proposal.proposer_id == "governance_user_001"
        assert proposal.proposal_type == "economic"
        assert isinstance(proposal.voting_starts, datetime)
        assert isinstance(proposal.voting_ends, datetime)
    
    def test_vote_creation(self):
        """Test creating a vote"""
        proposal_id = uuid4()
        vote = Vote(
            proposal_id=proposal_id,
            voter_id="voter_001",
            vote=True,
            voting_power=100.50
        )
        
        assert vote.voter_id == "voter_001"
        assert vote.vote == True
        assert vote.voting_power == 100.50
        assert vote.proposal_id == proposal_id
    
    @pytest.mark.parametrize("vote_choice", [True, False])
    def test_vote_valid_choices(self, vote_choice):
        """Test valid vote choices"""
        vote = Vote(
            proposal_id=uuid4(),
            voter_id="voter_001",
            vote=vote_choice,
            voting_power=50.00
        )
        assert vote.vote == vote_choice


class TestModelValidationEdgeCases:
    """Test edge cases and validation constraints"""
    
    def test_empty_string_validation(self):
        """Test handling of empty strings in required fields"""
        # Empty string is actually accepted for user_id
        session = PRSMSession(
            user_id=""  # Empty string is accepted
        )
        assert session.user_id == ""
        
        # Test that None is rejected
        with pytest.raises(ValidationError):
            PRSMSession(
                user_id=None  # None should fail validation
            )
    
    def test_none_values_in_optional_fields(self):
        """Test None values in optional fields"""
        session = PRSMSession(
            user_id="test_user"
        )
        # Optional fields should have their default values
        assert session.metadata == {}
        assert session.reasoning_trace == []
        assert session.safety_flags == []
    
    def test_large_float_values(self):
        """Test handling of very large float values"""
        large_amount = 999999999.999999999
        transaction = FTNSTransaction(
            to_user="test_user",
            amount=large_amount,
            transaction_type="reward",
            description="Large reward transaction"
        )
        assert transaction.amount == large_amount
    
    def test_uuid_generation_validation(self):
        """Test UUID generation and validation"""
        # Test that UUIDs are properly generated
        session = PRSMSession(
            user_id="test_user"
        )
        assert isinstance(session.session_id, UUID)
        
        # Test that different sessions get different UUIDs
        session2 = PRSMSession(
            user_id="test_user2"
        )
        assert session.session_id != session2.session_id
    
    def test_datetime_timezone_handling(self):
        """Test datetime timezone handling"""
        now = datetime.now(timezone.utc)
        balance = FTNSBalance(
            user_id="test_user",
            last_dividend=now
        )
        assert balance.last_dividend.tzinfo is not None
        assert balance.created_at.tzinfo is not None


class TestModelSerialization:
    """Test model serialization and deserialization"""
    
    def test_session_dict_conversion(self):
        """Test converting session to dict and back"""
        original_session = PRSMSession(
            user_id="test_user",
            nwtn_context_allocation=100,
            metadata={"max_budget": 100.0},
            status=TaskStatus.PENDING
        )
        
        # Convert to dict
        session_dict = original_session.model_dump()
        assert isinstance(session_dict, dict)
        assert session_dict["user_id"] == "test_user"
        assert session_dict["metadata"]["max_budget"] == 100.0
        
        # Convert back to model
        recreated_session = PRSMSession(**session_dict)
        assert recreated_session.user_id == original_session.user_id
        assert recreated_session.metadata == original_session.metadata
    
    def test_transaction_json_serialization(self):
        """Test transaction JSON serialization"""
        transaction = FTNSTransaction(
            to_user="test_user",
            amount=25.50,
            transaction_type="reward",
            description="Test reward"
        )
        
        # Should be serializable to JSON
        json_data = transaction.model_dump_json()
        assert isinstance(json_data, str)
        assert "25.5" in json_data
        
        # Should be deserializable from JSON
        recreated = FTNSTransaction.model_validate_json(json_data)
        assert recreated.amount == transaction.amount
        assert recreated.to_user == transaction.to_user