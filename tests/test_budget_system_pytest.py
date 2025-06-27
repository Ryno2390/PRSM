#!/usr/bin/env python3
"""
Budget System Test Suite

Comprehensive pytest tests for PRSM budget functionality including
FTNS budget management, cost prediction, and spending tracking.
Converted from test_budget_standalone.py to follow pytest conventions.
"""

import pytest
import asyncio
import sys
import os
from decimal import Decimal
from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict, Any
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import PRSM modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockUserInput:
    """Mock user input for testing"""
    def __init__(self, user_id: str, prompt: str, **kwargs):
        self.user_id = user_id
        self.prompt = prompt
        self.preferences = kwargs.get('preferences', {})
        self.context_allocation = kwargs.get('context_allocation')
        self.budget_config = kwargs.get('budget_config', {})


class MockPRSMSession:
    """Mock PRSM session for testing"""
    def __init__(self, user_id: str, **kwargs):
        self.session_id = uuid4()
        self.user_id = user_id
        self.metadata = kwargs.get('metadata', {})


class MockFTNSService:
    """Mock FTNS service for testing"""
    def __init__(self):
        self.balances = {}
    
    async def get_user_balance(self, user_id: str):
        class MockBalance:
            def __init__(self, balance: float):
                self.balance = balance
        return MockBalance(1000.0)  # Mock user has 1000 FTNS


class SpendingCategory:
    """Enum-like class for spending categories"""
    MODEL_INFERENCE = "model_inference"
    AGENT_COORDINATION = "agent_coordination"
    TOOL_EXECUTION = "tool_execution"
    DATA_ACCESS = "data_access"
    NETWORK_FEES = "network_fees"


class BudgetConstraint:
    """Budget constraint implementation for testing"""
    def __init__(self, category: str, limit: Decimal, period: str = "daily"):
        self.category = category
        self.limit = limit
        self.period = period
        self.current_spending = Decimal('0.0')
    
    def can_spend(self, amount: Decimal) -> bool:
        """Check if spending amount is within budget"""
        return (self.current_spending + amount) <= self.limit
    
    def record_spending(self, amount: Decimal):
        """Record spending against this constraint"""
        self.current_spending += amount
    
    def get_remaining_budget(self) -> Decimal:
        """Get remaining budget amount"""
        return max(Decimal('0.0'), self.limit - self.current_spending)


class FTNSBudgetManager:
    """Budget manager implementation for testing"""
    def __init__(self, user_id: str, ftns_service=None):
        self.user_id = user_id
        self.ftns_service = ftns_service or MockFTNSService()
        self.constraints: Dict[str, BudgetConstraint] = {}
        self.spending_history = []
    
    def add_constraint(self, constraint: BudgetConstraint):
        """Add budget constraint"""
        self.constraints[constraint.category] = constraint
    
    async def predict_cost(self, user_input: MockUserInput) -> Dict[str, Any]:
        """Predict cost for user input"""
        # Simple cost prediction logic for testing
        base_cost = len(user_input.prompt) * 0.01  # 1 cent per character
        
        return {
            "total_cost": Decimal(str(base_cost)),
            "breakdown": {
                SpendingCategory.MODEL_INFERENCE: Decimal(str(base_cost * 0.6)),
                SpendingCategory.AGENT_COORDINATION: Decimal(str(base_cost * 0.2)),
                SpendingCategory.TOOL_EXECUTION: Decimal(str(base_cost * 0.1)),
                SpendingCategory.DATA_ACCESS: Decimal(str(base_cost * 0.1))
            },
            "confidence": 0.85,
            "estimated_tokens": len(user_input.prompt.split()) * 1.3
        }
    
    async def check_budget_compliance(self, predicted_cost: Dict[str, Any]) -> Dict[str, Any]:
        """Check if predicted cost complies with budget constraints"""
        compliance_results = {}
        total_cost = predicted_cost["total_cost"]
        breakdown = predicted_cost["breakdown"]
        
        for category, amount in breakdown.items():
            if category in self.constraints:
                constraint = self.constraints[category]
                can_spend = constraint.can_spend(amount)
                compliance_results[category] = {
                    "can_spend": can_spend,
                    "requested": amount,
                    "remaining": constraint.get_remaining_budget(),
                    "limit": constraint.limit
                }
            else:
                compliance_results[category] = {
                    "can_spend": True,
                    "requested": amount,
                    "remaining": None,
                    "limit": None
                }
        
        overall_compliance = all(result["can_spend"] for result in compliance_results.values())
        
        return {
            "compliant": overall_compliance,
            "total_cost": total_cost,
            "category_results": compliance_results,
            "warnings": [] if overall_compliance else ["Budget constraints violated"]
        }
    
    async def record_spending(self, session: MockPRSMSession, amount: Decimal, category: str):
        """Record actual spending"""
        spending_record = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "amount": amount,
            "category": category,
            "timestamp": datetime.now(timezone.utc)
        }
        
        self.spending_history.append(spending_record)
        
        if category in self.constraints:
            self.constraints[category].record_spending(amount)
        
        return spending_record


class TestBudgetConstraints:
    """Test suite for budget constraint functionality"""
    
    @pytest.fixture
    def sample_constraint(self):
        """Fixture providing a sample budget constraint"""
        return BudgetConstraint(
            category=SpendingCategory.MODEL_INFERENCE,
            limit=Decimal('100.0'),
            period="daily"
        )
    
    def test_constraint_creation(self, sample_constraint):
        """Test budget constraint creation"""
        assert sample_constraint.category == SpendingCategory.MODEL_INFERENCE
        assert sample_constraint.limit == Decimal('100.0')
        assert sample_constraint.period == "daily"
        assert sample_constraint.current_spending == Decimal('0.0')
    
    def test_can_spend_within_budget(self, sample_constraint):
        """Test spending within budget constraints"""
        # Should be able to spend within limit
        assert sample_constraint.can_spend(Decimal('50.0')) is True
        assert sample_constraint.can_spend(Decimal('100.0')) is True
        
        # Should not be able to spend over limit
        assert sample_constraint.can_spend(Decimal('101.0')) is False
    
    def test_record_spending(self, sample_constraint):
        """Test recording spending against constraints"""
        initial_spending = sample_constraint.current_spending
        
        sample_constraint.record_spending(Decimal('25.0'))
        
        assert sample_constraint.current_spending == initial_spending + Decimal('25.0')
        assert sample_constraint.get_remaining_budget() == Decimal('75.0')
    
    def test_spending_progression(self, sample_constraint):
        """Test spending progression affects budget availability"""
        # Initially can spend full amount
        assert sample_constraint.can_spend(Decimal('100.0')) is True
        
        # Record some spending
        sample_constraint.record_spending(Decimal('60.0'))
        
        # Now can only spend remaining amount
        assert sample_constraint.can_spend(Decimal('40.0')) is True
        assert sample_constraint.can_spend(Decimal('41.0')) is False
        
        # Check remaining budget
        assert sample_constraint.get_remaining_budget() == Decimal('40.0')


class TestFTNSBudgetManager:
    """Test suite for FTNS budget manager functionality"""
    
    @pytest.fixture
    def budget_manager(self):
        """Fixture providing a budget manager instance"""
        return FTNSBudgetManager("test_user_123")
    
    @pytest.fixture
    def sample_user_input(self):
        """Fixture providing sample user input"""
        return MockUserInput(
            user_id="test_user_123",
            prompt="Explain quantum computing in simple terms",
            preferences={"model": "gpt-4", "max_tokens": 500}
        )
    
    @pytest.fixture
    def budget_manager_with_constraints(self, budget_manager):
        """Fixture providing budget manager with sample constraints"""
        constraints = [
            BudgetConstraint(SpendingCategory.MODEL_INFERENCE, Decimal('50.0')),
            BudgetConstraint(SpendingCategory.AGENT_COORDINATION, Decimal('20.0')),
            BudgetConstraint(SpendingCategory.TOOL_EXECUTION, Decimal('10.0'))
        ]
        
        for constraint in constraints:
            budget_manager.add_constraint(constraint)
        
        return budget_manager
    
    def test_budget_manager_creation(self, budget_manager):
        """Test budget manager creation and initialization"""
        assert budget_manager.user_id == "test_user_123"
        assert isinstance(budget_manager.ftns_service, MockFTNSService)
        assert len(budget_manager.constraints) == 0
        assert len(budget_manager.spending_history) == 0
    
    def test_add_constraint(self, budget_manager):
        """Test adding budget constraints"""
        constraint = BudgetConstraint(
            category=SpendingCategory.MODEL_INFERENCE,
            limit=Decimal('100.0')
        )
        
        budget_manager.add_constraint(constraint)
        
        assert SpendingCategory.MODEL_INFERENCE in budget_manager.constraints
        assert budget_manager.constraints[SpendingCategory.MODEL_INFERENCE] == constraint
    
    @pytest.mark.asyncio
    async def test_cost_prediction(self, budget_manager, sample_user_input):
        """Test cost prediction functionality"""
        predicted_cost = await budget_manager.predict_cost(sample_user_input)
        
        # Verify prediction structure
        assert "total_cost" in predicted_cost
        assert "breakdown" in predicted_cost
        assert "confidence" in predicted_cost
        assert "estimated_tokens" in predicted_cost
        
        # Verify breakdown categories
        breakdown = predicted_cost["breakdown"]
        expected_categories = [
            SpendingCategory.MODEL_INFERENCE,
            SpendingCategory.AGENT_COORDINATION,
            SpendingCategory.TOOL_EXECUTION,
            SpendingCategory.DATA_ACCESS
        ]
        
        for category in expected_categories:
            assert category in breakdown
            assert isinstance(breakdown[category], Decimal)
            assert breakdown[category] >= 0
        
        # Verify total equals sum of breakdown
        total_from_breakdown = sum(breakdown.values())
        assert abs(predicted_cost["total_cost"] - total_from_breakdown) < Decimal('0.01')
    
    @pytest.mark.asyncio
    async def test_budget_compliance_check(self, budget_manager_with_constraints, sample_user_input):
        """Test budget compliance checking"""
        predicted_cost = await budget_manager_with_constraints.predict_cost(sample_user_input)
        compliance_result = await budget_manager_with_constraints.check_budget_compliance(predicted_cost)
        
        # Verify compliance result structure
        assert "compliant" in compliance_result
        assert "total_cost" in compliance_result
        assert "category_results" in compliance_result
        assert "warnings" in compliance_result
        
        # Verify category results
        category_results = compliance_result["category_results"]
        for category, result in category_results.items():
            assert "can_spend" in result
            assert "requested" in result
            assert "remaining" in result
            assert "limit" in result
    
    @pytest.mark.asyncio
    async def test_spending_record(self, budget_manager, sample_user_input):
        """Test spending record functionality"""
        session = MockPRSMSession("test_user_123")
        amount = Decimal('25.0')
        category = SpendingCategory.MODEL_INFERENCE
        
        record = await budget_manager.record_spending(session, amount, category)
        
        # Verify spending record structure
        assert record["session_id"] == session.session_id
        assert record["user_id"] == session.user_id
        assert record["amount"] == amount
        assert record["category"] == category
        assert "timestamp" in record
        
        # Verify record is added to history
        assert len(budget_manager.spending_history) == 1
        assert budget_manager.spending_history[0] == record
    
    @pytest.mark.asyncio
    async def test_budget_enforcement(self, budget_manager_with_constraints):
        """Test budget enforcement over multiple spending events"""
        session = MockPRSMSession("test_user_123")
        
        # First spending should be allowed
        first_amount = Decimal('30.0')
        await budget_manager_with_constraints.record_spending(
            session, first_amount, SpendingCategory.MODEL_INFERENCE
        )
        
        # Check remaining budget
        constraint = budget_manager_with_constraints.constraints[SpendingCategory.MODEL_INFERENCE]
        assert constraint.get_remaining_budget() == Decimal('20.0')
        
        # Second spending should still be allowed
        second_amount = Decimal('15.0')
        await budget_manager_with_constraints.record_spending(
            session, second_amount, SpendingCategory.MODEL_INFERENCE
        )
        
        # Now budget should be nearly exhausted
        assert constraint.get_remaining_budget() == Decimal('5.0')
        
        # Large spending should be rejected
        assert constraint.can_spend(Decimal('10.0')) is False


class TestBudgetIntegration:
    """Integration tests for complete budget workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_budget_workflow(self):
        """Test complete budget workflow from prediction to spending"""
        # Setup
        budget_manager = FTNSBudgetManager("integration_user")
        budget_manager.add_constraint(
            BudgetConstraint(SpendingCategory.MODEL_INFERENCE, Decimal('100.0'))
        )
        
        user_input = MockUserInput(
            user_id="integration_user",
            prompt="Write a comprehensive analysis of machine learning algorithms",
            preferences={"model": "gpt-4", "max_tokens": 1000}
        )
        
        # Step 1: Predict cost
        predicted_cost = await budget_manager.predict_cost(user_input)
        assert predicted_cost["total_cost"] > 0
        
        # Step 2: Check compliance
        compliance = await budget_manager.check_budget_compliance(predicted_cost)
        assert compliance["compliant"] is True
        
        # Step 3: Record spending
        session = MockPRSMSession("integration_user")
        spending_amount = predicted_cost["breakdown"][SpendingCategory.MODEL_INFERENCE]
        
        record = await budget_manager.record_spending(
            session, spending_amount, SpendingCategory.MODEL_INFERENCE
        )
        
        # Verify complete workflow
        assert record["amount"] == spending_amount
        assert len(budget_manager.spending_history) == 1
        
        # Verify budget updated
        constraint = budget_manager.constraints[SpendingCategory.MODEL_INFERENCE]
        assert constraint.current_spending == spending_amount


if __name__ == "__main__":
    # Run the tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])