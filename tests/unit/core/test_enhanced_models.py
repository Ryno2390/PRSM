"""
Enhanced Core Models Tests
==========================

Comprehensive tests for PRSM core data models with validation,
serialization, edge cases, and performance testing.
"""

import pytest
import json
import uuid
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch

try:
    from pydantic import ValidationError
    from prsm.core.models import (
        PRSMSession, UserInput, FTNSTransaction, FTNSBalance,
        ArchitectTask, SafetyFlag, CircuitBreakerEvent, GovernanceProposal,
        Vote, PeerNode
    )
    from prsm.core.validation import validate_session_data, validate_user_input
except ImportError:
    # Create mock classes if imports fail
    ValidationError = Exception
    PRSMSession = Mock
    UserInput = Mock
    FTNSTransaction = Mock
    FTNSBalance = Mock
    ArchitectTask = Mock
    SafetyFlag = Mock
    CircuitBreakerEvent = Mock
    GovernanceProposal = Mock
    Vote = Mock
    PeerNode = Mock
    validate_session_data = Mock()
    validate_user_input = Mock()


class TestPRSMSessionEnhanced:
    """Enhanced tests for PRSM Session model"""
    
    def test_session_creation_with_all_fields(self):
        """Test session creation with all possible fields"""
        session_data = {
            "session_id": str(uuid.uuid4()),
            "user_id": "test_user_123",
            "status": "in_progress",
            "query_count": 5,
            "total_cost": Decimal("15.75"),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "metadata": {
                "client_version": "1.0.0",
                "user_agent": "PRSM-Client/1.0.0",
                "ip_address": "192.168.1.100"
            }
        }
        
        session = PRSMSession(**session_data)
        
        assert session.session_id == session_data["session_id"]
        assert session.user_id == session_data["user_id"]
        assert session.status == session_data["status"]
        assert session.query_count == session_data["query_count"]
        assert session.total_cost == session_data["total_cost"]
        assert session.metadata == session_data["metadata"]
    
    def test_session_status_validation(self):
        """Test session status validation"""
        valid_statuses = ["pending", "in_progress", "completed", "failed", "cancelled"]
        
        for status in valid_statuses:
            session = PRSMSession(
                session_id=str(uuid.uuid4()),
                user_id="test_user",
                status=status
            )
            assert session.status == status
        
        # Test invalid status
        with pytest.raises(ValidationError):
            PRSMSession(
                session_id=str(uuid.uuid4()),
                user_id="test_user",
                status="invalid_status"
            )
    
    def test_session_cost_calculation(self):
        """Test session cost calculations"""
        session = PRSMSession(
            session_id=str(uuid.uuid4()),
            user_id="test_user",
            status="completed",
            query_count=10,
            total_cost=Decimal("25.50")
        )
        
        # Calculate average cost per query
        avg_cost = session.total_cost / session.query_count
        assert avg_cost == Decimal("2.55")
    
    def test_session_duration_calculation(self):
        """Test session duration calculation"""
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=2, minutes=30)
        
        session = PRSMSession(
            session_id=str(uuid.uuid4()),
            user_id="test_user",
            status="completed",
            created_at=start_time,
            updated_at=end_time
        )
        
        duration = session.updated_at - session.created_at
        assert duration.total_seconds() == 9000  # 2.5 hours in seconds
    
    @pytest.mark.performance
    def test_session_bulk_creation_performance(self, performance_runner):
        """Test performance of bulk session creation"""
        def create_sessions():
            sessions = []
            for i in range(100):
                session = PRSMSession(
                    session_id=str(uuid.uuid4()),
                    user_id=f"user_{i}",
                    status="pending"
                )
                sessions.append(session)
            return sessions
        
        metrics = performance_runner.run_performance_test(
            create_sessions,
            iterations=10,
            warmup_iterations=2
        )
        
        # Assert performance is reasonable
        assert metrics.execution_time_ms < 1000  # Less than 1 second
        assert metrics.error_rate == 0.0
    
    def test_session_serialization(self):
        """Test session JSON serialization"""
        session = PRSMSession(
            session_id=str(uuid.uuid4()),
            user_id="test_user",
            status="completed",
            total_cost=Decimal("10.50"),
            metadata={"test": "data"}
        )
        
        # Test dict conversion
        session_dict = session.dict()
        assert isinstance(session_dict, dict)
        assert session_dict["user_id"] == "test_user"
        assert session_dict["status"] == "completed"
        
        # Test JSON serialization
        session_json = session.json()
        parsed = json.loads(session_json)
        assert parsed["user_id"] == "test_user"


class TestUserInputEnhanced:
    """Enhanced tests for User Input model"""
    
    def test_input_content_validation(self):
        """Test user input content validation"""
        # Test valid content
        valid_inputs = [
            "Simple question",
            "Complex question with multiple sentences. This should work fine.",
            "Question with numbers 123 and symbols !@#$%",
            "A" * 1000,  # Long input
        ]
        
        for content in valid_inputs:
            user_input = UserInput(
                input_id=str(uuid.uuid4()),
                user_id="test_user",
                content=content
            )
            assert user_input.content == content
    
    def test_input_content_limits(self):
        """Test input content length limits"""
        # Test extremely long input
        very_long_content = "A" * 10000
        
        user_input = UserInput(
            input_id=str(uuid.uuid4()),
            user_id="test_user",
            content=very_long_content
        )
        
        assert len(user_input.content) == 10000
    
    def test_input_metadata_handling(self):
        """Test user input metadata handling"""
        metadata = {
            "source": "web_ui",
            "context": "research",
            "priority": "high",
            "tags": ["science", "AI", "research"]
        }
        
        user_input = UserInput(
            input_id=str(uuid.uuid4()),
            user_id="test_user",
            content="Test question",
            metadata=metadata
        )
        
        assert user_input.metadata == metadata
        assert user_input.metadata["tags"] == ["science", "AI", "research"]
    
    def test_input_timestamp_handling(self):
        """Test input timestamp handling"""
        now = datetime.now(timezone.utc)
        
        user_input = UserInput(
            input_id=str(uuid.uuid4()),
            user_id="test_user",
            content="Test question",
            timestamp=now
        )
        
        assert user_input.timestamp == now
        assert user_input.timestamp.tzinfo == timezone.utc


class TestFTNSTransactionEnhanced:
    """Enhanced tests for FTNS Transaction model"""
    
    def test_transaction_amount_precision(self):
        """Test transaction amount decimal precision"""
        amounts = [
            Decimal("10.00"),
            Decimal("0.01"),
            Decimal("999999.99"),
            Decimal("0.001"),  # Very small amount
        ]
        
        for amount in amounts:
            transaction = FTNSTransaction(
                transaction_id=str(uuid.uuid4()),
                user_id="test_user",
                amount=amount,
                transaction_type="reward"
            )
            assert transaction.amount == amount
    
    def test_transaction_type_validation(self):
        """Test transaction type validation"""
        valid_types = ["reward", "charge", "transfer", "dividend", "fee", "refund"]
        
        for tx_type in valid_types:
            transaction = FTNSTransaction(
                transaction_id=str(uuid.uuid4()),
                user_id="test_user",
                amount=Decimal("10.00"),
                transaction_type=tx_type
            )
            assert transaction.transaction_type == tx_type
    
    def test_transaction_negative_amount_handling(self):
        """Test handling of negative transaction amounts"""
        # Negative amounts should be allowed for certain transaction types
        transaction = FTNSTransaction(
            transaction_id=str(uuid.uuid4()),
            user_id="test_user",
            amount=Decimal("-10.00"),
            transaction_type="charge"
        )
        
        assert transaction.amount == Decimal("-10.00")
    
    def test_transaction_fee_calculation(self):
        """Test transaction fee calculation"""
        base_amount = Decimal("100.00")
        fee_rate = Decimal("0.01")  # 1%
        
        transaction = FTNSTransaction(
            transaction_id=str(uuid.uuid4()),
            user_id="test_user",
            amount=base_amount,
            transaction_type="transfer",
            fee=base_amount * fee_rate
        )
        
        assert transaction.fee == Decimal("1.00")
        
        # Calculate net amount
        net_amount = transaction.amount - (transaction.fee or Decimal("0"))
        assert net_amount == Decimal("99.00")
    
    def test_transaction_status_tracking(self):
        """Test transaction status tracking"""
        statuses = ["pending", "processing", "confirmed", "failed", "cancelled"]
        
        for status in statuses:
            transaction = FTNSTransaction(
                transaction_id=str(uuid.uuid4()),
                user_id="test_user",
                amount=Decimal("10.00"),
                transaction_type="transfer",
                status=status
            )
            assert transaction.status == status
    
    @pytest.mark.performance
    def test_transaction_batch_processing(self, performance_runner):
        """Test performance of batch transaction processing"""
        def create_transactions():
            transactions = []
            for i in range(1000):
                tx = FTNSTransaction(
                    transaction_id=str(uuid.uuid4()),
                    user_id=f"user_{i % 10}",
                    amount=Decimal(f"{10 + i}.00"),
                    transaction_type=["reward", "charge", "transfer"][i % 3]
                )
                transactions.append(tx)
            return transactions
        
        metrics = performance_runner.run_performance_test(
            create_transactions,
            iterations=5,
            warmup_iterations=1
        )
        
        # Assert reasonable performance
        assert metrics.execution_time_ms < 2000  # Less than 2 seconds
        assert metrics.error_rate == 0.0


class TestFTNSBalanceEnhanced:
    """Enhanced tests for FTNS Balance model"""
    
    def test_balance_calculations(self):
        """Test balance calculations"""
        balance = FTNSBalance(
            user_id="test_user",
            total_balance=Decimal("100.00"),
            reserved_balance=Decimal("20.00"),
            pending_balance=Decimal("5.00")
        )
        
        # Available balance calculation
        available = balance.total_balance - balance.reserved_balance
        assert available == Decimal("80.00")
        
        # Total pending
        total_with_pending = balance.total_balance + balance.pending_balance
        assert total_with_pending == Decimal("105.00")
    
    def test_balance_constraints(self):
        """Test balance constraints and validation"""
        # Test non-negative constraints
        balance = FTNSBalance(
            user_id="test_user",
            total_balance=Decimal("0.00"),
            reserved_balance=Decimal("0.00"),
            pending_balance=Decimal("0.00")
        )
        
        assert balance.total_balance >= 0
        assert balance.reserved_balance >= 0
        assert balance.pending_balance >= 0
    
    def test_balance_updates(self):
        """Test balance update operations"""
        initial_balance = FTNSBalance(
            user_id="test_user",
            total_balance=Decimal("100.00"),
            reserved_balance=Decimal("10.00")
        )
        
        # Simulate credit operation
        credit_amount = Decimal("50.00")
        new_total = initial_balance.total_balance + credit_amount
        
        updated_balance = FTNSBalance(
            user_id="test_user",
            total_balance=new_total,
            reserved_balance=initial_balance.reserved_balance
        )
        
        assert updated_balance.total_balance == Decimal("150.00")
        assert updated_balance.reserved_balance == Decimal("10.00")


class TestArchitectTaskEnhanced:
    """Enhanced tests for Architect Task model"""
    
    def test_task_complexity_validation(self):
        """Test task complexity validation"""
        valid_complexities = [1, 2, 3, 4, 5]
        
        for complexity in valid_complexities:
            task = ArchitectTask(
                task_id=str(uuid.uuid4()),
                user_id="test_user",
                description="Test task",
                complexity=complexity,
                estimated_cost=Decimal("10.00")
            )
            assert task.complexity == complexity
    
    def test_task_hierarchy_levels(self):
        """Test task hierarchy levels"""
        parent_task = ArchitectTask(
            task_id=str(uuid.uuid4()),
            user_id="test_user",
            description="Parent task",
            complexity=3,
            estimated_cost=Decimal("50.00")
        )
        
        child_task = ArchitectTask(
            task_id=str(uuid.uuid4()),
            user_id="test_user",
            description="Child task",
            complexity=2,
            estimated_cost=Decimal("20.00"),
            parent_task_id=parent_task.task_id
        )
        
        assert child_task.parent_task_id == parent_task.task_id
    
    def test_task_status_workflow(self):
        """Test task status workflow"""
        task = ArchitectTask(
            task_id=str(uuid.uuid4()),
            user_id="test_user",
            description="Test task",
            complexity=2,
            estimated_cost=Decimal("15.00"),
            status="pending"
        )
        
        # Simulate status progression
        statuses = ["pending", "in_progress", "completed"]
        
        for status in statuses:
            task.status = status
            assert task.status == status
    
    def test_task_cost_estimation(self):
        """Test task cost estimation"""
        # Cost should scale with complexity
        complexities_and_costs = [
            (1, Decimal("5.00")),
            (2, Decimal("10.00")),
            (3, Decimal("20.00")),
            (4, Decimal("40.00")),
            (5, Decimal("80.00"))
        ]
        
        for complexity, expected_cost in complexities_and_costs:
            task = ArchitectTask(
                task_id=str(uuid.uuid4()),
                user_id="test_user",
                description=f"Task complexity {complexity}",
                complexity=complexity,
                estimated_cost=expected_cost
            )
            
            assert task.estimated_cost == expected_cost
            
            # Cost per complexity unit
            cost_per_unit = task.estimated_cost / complexity
            assert cost_per_unit == Decimal("5.00")


class TestSafetyModelsEnhanced:
    """Enhanced tests for Safety models"""
    
    def test_safety_flag_creation(self):
        """Test safety flag creation with all fields"""
        safety_flag = SafetyFlag(
            flag_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            flag_type="content_warning",
            severity="medium",
            description="Potentially sensitive content detected",
            metadata={
                "confidence": 0.75,
                "categories": ["violence", "adult_content"],
                "auto_generated": True
            }
        )
        
        assert safety_flag.flag_type == "content_warning"
        assert safety_flag.severity == "medium"
        assert safety_flag.metadata["confidence"] == 0.75
    
    def test_circuit_breaker_event(self):
        """Test circuit breaker event handling"""
        event = CircuitBreakerEvent(
            event_id=str(uuid.uuid4()),
            component="nwtn_engine",
            event_type="failure_threshold_exceeded",
            threshold_value=10,
            current_value=15,
            action_taken="service_disabled",
            recovery_time_seconds=300
        )
        
        assert event.component == "nwtn_engine"
        assert event.current_value > event.threshold_value
        assert event.action_taken == "service_disabled"
    
    def test_safety_level_enum_validation(self):
        """Test safety level enumeration values"""
        levels = ["low", "medium", "high", "critical"]
        
        for level in levels:
            flag = SafetyFlag(
                flag_id=str(uuid.uuid4()),
                session_id=str(uuid.uuid4()),
                flag_type="test_flag",
                severity=level,
                description="Test flag"
            )
            assert flag.severity == level


class TestGovernanceModelsEnhanced:
    """Enhanced tests for Governance models"""
    
    def test_governance_proposal_creation(self):
        """Test governance proposal creation"""
        proposal = GovernanceProposal(
            proposal_id=str(uuid.uuid4()),
            title="Increase NWTN processing timeout",
            description="Proposal to increase the NWTN processing timeout from 30s to 60s",
            proposer_id="user_123",
            proposal_type="parameter_change",
            voting_end_date=datetime.now(timezone.utc) + timedelta(days=7),
            required_quorum=Decimal("0.51"),  # 51%
            metadata={
                "parameter": "nwtn_timeout",
                "current_value": 30,
                "proposed_value": 60,
                "impact_assessment": "low"
            }
        )
        
        assert proposal.proposal_type == "parameter_change"
        assert proposal.required_quorum == Decimal("0.51")
        assert proposal.metadata["parameter"] == "nwtn_timeout"
    
    def test_vote_creation_and_validation(self):
        """Test vote creation and validation"""
        proposal_id = str(uuid.uuid4())
        
        # Test positive vote
        vote_yes = Vote(
            vote_id=str(uuid.uuid4()),
            proposal_id=proposal_id,
            voter_id="voter_123",
            vote_choice=True,
            voting_power=Decimal("100.0"),
            reason="I support this change for better performance"
        )
        
        assert vote_yes.vote_choice is True
        assert vote_yes.voting_power == Decimal("100.0")
        
        # Test negative vote
        vote_no = Vote(
            vote_id=str(uuid.uuid4()),
            proposal_id=proposal_id,
            voter_id="voter_456",
            vote_choice=False,
            voting_power=Decimal("50.0"),
            reason="I think the current timeout is sufficient"
        )
        
        assert vote_no.vote_choice is False
        assert vote_no.voting_power == Decimal("50.0")
    
    def test_voting_power_calculations(self):
        """Test voting power calculations"""
        votes = [
            Vote(vote_id=str(uuid.uuid4()), proposal_id=str(uuid.uuid4()), 
                 voter_id="v1", vote_choice=True, voting_power=Decimal("100")),
            Vote(vote_id=str(uuid.uuid4()), proposal_id=str(uuid.uuid4()), 
                 voter_id="v2", vote_choice=True, voting_power=Decimal("150")),
            Vote(vote_id=str(uuid.uuid4()), proposal_id=str(uuid.uuid4()), 
                 voter_id="v3", vote_choice=False, voting_power=Decimal("75")),
        ]
        
        total_yes = sum(v.voting_power for v in votes if v.vote_choice)
        total_no = sum(v.voting_power for v in votes if not v.vote_choice)
        total_power = total_yes + total_no
        
        assert total_yes == Decimal("250")
        assert total_no == Decimal("75")
        assert total_power == Decimal("325")
        
        # Calculate vote percentage
        yes_percentage = total_yes / total_power
        assert yes_percentage == Decimal("250") / Decimal("325")  # ~76.9%


class TestModelValidationEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_string_validation(self):
        """Test handling of empty strings"""
        # Some fields should accept empty strings, others shouldn't
        with pytest.raises(ValidationError):
            UserInput(
                input_id=str(uuid.uuid4()),
                user_id="test_user",
                content=""  # Empty content should be invalid
            )
    
    def test_none_values_in_optional_fields(self):
        """Test None values in optional fields"""
        session = PRSMSession(
            session_id=str(uuid.uuid4()),
            user_id="test_user",
            status="pending",
            metadata=None  # Optional field
        )
        
        assert session.metadata is None
    
    def test_large_decimal_values(self):
        """Test handling of large decimal values"""
        large_amount = Decimal("999999999.99")
        
        transaction = FTNSTransaction(
            transaction_id=str(uuid.uuid4()),
            user_id="test_user",
            amount=large_amount,
            transaction_type="reward"
        )
        
        assert transaction.amount == large_amount
    
    def test_uuid_generation_and_validation(self):
        """Test UUID generation and validation"""
        # Test valid UUID
        valid_uuid = str(uuid.uuid4())
        session = PRSMSession(
            session_id=valid_uuid,
            user_id="test_user",
            status="pending"
        )
        
        assert session.session_id == valid_uuid
        
        # Test invalid UUID format
        with pytest.raises(ValidationError):
            PRSMSession(
                session_id="invalid-uuid-format",
                user_id="test_user",
                status="pending"
            )
    
    def test_datetime_timezone_handling(self):
        """Test datetime timezone handling"""
        # Test UTC timezone
        utc_time = datetime.now(timezone.utc)
        session = PRSMSession(
            session_id=str(uuid.uuid4()),
            user_id="test_user",
            status="pending",
            created_at=utc_time
        )
        
        assert session.created_at.tzinfo == timezone.utc
        
        # Test naive datetime (should be handled appropriately)
        naive_time = datetime.now()
        session_naive = PRSMSession(
            session_id=str(uuid.uuid4()),
            user_id="test_user",
            status="pending",
            created_at=naive_time
        )
        
        # Should either convert to UTC or raise an error
        assert session_naive.created_at is not None


class TestModelSerialization:
    """Test model serialization and deserialization"""
    
    def test_session_dict_conversion(self):
        """Test session dictionary conversion"""
        session = PRSMSession(
            session_id=str(uuid.uuid4()),
            user_id="test_user",
            status="completed",
            query_count=5,
            total_cost=Decimal("25.50"),
            metadata={"test": "data"}
        )
        
        session_dict = session.dict()
        
        # Verify all fields are present
        expected_fields = [
            "session_id", "user_id", "status", "query_count", 
            "total_cost", "created_at", "updated_at", "metadata"
        ]
        
        for field in expected_fields:
            assert field in session_dict
        
        # Verify data types
        assert isinstance(session_dict["query_count"], int)
        assert isinstance(session_dict["metadata"], dict)
    
    def test_transaction_json_serialization(self):
        """Test transaction JSON serialization"""
        transaction = FTNSTransaction(
            transaction_id=str(uuid.uuid4()),
            user_id="test_user",
            amount=Decimal("10.50"),
            transaction_type="reward",
            metadata={"source": "api"}
        )
        
        # Test JSON conversion
        json_str = transaction.json()
        parsed = json.loads(json_str)
        
        assert parsed["user_id"] == "test_user"
        assert parsed["transaction_type"] == "reward"
        assert parsed["metadata"]["source"] == "api"
        
        # Decimal should be serialized as string or number
        assert "amount" in parsed
    
    def test_model_round_trip_serialization(self):
        """Test round-trip serialization (dict -> model -> dict)"""
        original_data = {
            "session_id": str(uuid.uuid4()),
            "user_id": "test_user",
            "status": "pending",
            "query_count": 0,
            "total_cost": Decimal("0.00")
        }
        
        # Create model from dict
        session = PRSMSession(**original_data)
        
        # Convert back to dict
        serialized = session.dict()
        
        # Create new model from serialized data
        session2 = PRSMSession(**serialized)
        
        # Verify equality
        assert session.session_id == session2.session_id
        assert session.user_id == session2.user_id
        assert session.status == session2.status


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for models working together"""
    
    def test_session_with_transactions(self, db_factory):
        """Test session with associated transactions"""
        session = db_factory.create_prsm_session(
            user_id="test_user",
            status="in_progress"
        )
        
        transactions = [
            db_factory.create_ftns_transaction(
                user_id="test_user",
                amount=Decimal("5.00"),
                transaction_type="charge"
            ),
            db_factory.create_ftns_transaction(
                user_id="test_user",
                amount=Decimal("10.00"),
                transaction_type="reward"
            )
        ]
        
        # Calculate total cost from transactions
        total_charges = sum(
            tx.amount for tx in transactions 
            if tx.transaction_type == "charge"
        )
        
        assert total_charges == Decimal("5.00")
    
    def test_user_balance_with_transactions(self, db_factory):
        """Test user balance calculations with transactions"""
        user_id = "test_user"
        
        # Create initial balance
        initial_balance = FTNSBalance(
            user_id=user_id,
            total_balance=Decimal("100.00"),
            reserved_balance=Decimal("0.00")
        )
        
        # Create transactions
        transactions = [
            db_factory.create_ftns_transaction(
                user_id=user_id,
                amount=Decimal("50.00"),
                transaction_type="reward"
            ),
            db_factory.create_ftns_transaction(
                user_id=user_id,
                amount=Decimal("-15.00"),
                transaction_type="charge"
            )
        ]
        
        # Calculate new balance
        balance_changes = sum(tx.amount for tx in transactions)
        new_balance = initial_balance.total_balance + balance_changes
        
        assert new_balance == Decimal("135.00")


@pytest.mark.performance
class TestModelPerformance:
    """Performance tests for model operations"""
    
    def test_large_batch_model_creation(self, performance_runner):
        """Test performance of creating many model instances"""
        def create_many_sessions():
            sessions = []
            for i in range(1000):
                session = PRSMSession(
                    session_id=str(uuid.uuid4()),
                    user_id=f"user_{i}",
                    status="pending"
                )
                sessions.append(session)
            return sessions
        
        metrics = performance_runner.run_performance_test(
            create_many_sessions,
            iterations=5,
            warmup_iterations=1
        )
        
        # Performance assertions
        assert metrics.execution_time_ms < 3000  # Less than 3 seconds
        assert metrics.error_rate == 0.0
        assert metrics.throughput_ops_per_sec > 1  # At least 1 op/sec
    
    def test_model_serialization_performance(self, performance_runner):
        """Test performance of model serialization"""
        # Create test models
        models = []
        for i in range(100):
            session = PRSMSession(
                session_id=str(uuid.uuid4()),
                user_id=f"user_{i}",
                status="pending",
                metadata={"index": i, "data": f"test_data_{i}"}
            )
            models.append(session)
        
        def serialize_models():
            return [model.dict() for model in models]
        
        metrics = performance_runner.run_performance_test(
            serialize_models,
            iterations=10,
            warmup_iterations=2
        )
        
        # Performance assertions
        assert metrics.execution_time_ms < 500  # Less than 500ms
        assert metrics.error_rate == 0.0