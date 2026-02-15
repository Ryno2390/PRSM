"""
Unit Tests for FTNS Service
Critical financial calculations and business logic testing
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal, getcontext
from uuid import uuid4
from datetime import datetime, timezone

# Set precision for financial calculations matching production
getcontext().prec = 18

from prsm.economy.tokenomics.ftns_service import (
    FTNSService,
    BASE_NWTN_FEE,
    CONTEXT_UNIT_COST,
    ARCHITECT_DECOMPOSITION_COST,
    COMPILER_SYNTHESIS_COST,
    AGENT_COSTS,
    REWARD_PER_MB,
    MODEL_CONTRIBUTION_REWARD,
    SUCCESSFUL_TEACHING_REWARD
)
from prsm.core.models import PRSMSession, FTNSBalance, FTNSTransaction


class TestFTNSService:
    """Unit tests for FTNS Service core functionality"""
    
    @pytest.fixture
    def ftns_service(self):
        """Create fresh FTNS service instance for each test"""
        return FTNSService()
    
    @pytest.fixture
    def mock_session(self):
        """Create mock PRSM session for testing"""
        return PRSMSession(
            user_id="test_user_001",
            nwtn_context_allocation=100,
            status="pending"
        )
    
    @pytest.fixture
    def mock_balance(self):
        """Create mock FTNS balance for testing"""
        return FTNSBalance(
            user_id="test_user_001",
            balance=1000.0,
            locked_balance=50.0,
            earned_total=2000.0,
            spent_total=1000.0
        )

    # === Context Cost Calculation Tests ===
    
    @pytest.mark.asyncio
    async def test_calculate_context_cost_basic(self, ftns_service, mock_session):
        """Test basic context cost calculation with no multipliers"""
        with patch.object(ftns_service, '_get_user_tier_multiplier', return_value=1.0):
            cost = await ftns_service.calculate_context_cost(mock_session, 100)
            
            expected_cost = 100 * CONTEXT_UNIT_COST  # 100 * 0.1 = 10.0
            assert cost == expected_cost
            assert isinstance(cost, float)
    
    @pytest.mark.asyncio
    async def test_calculate_context_cost_with_user_tier(self, ftns_service, mock_session):
        """Test context cost calculation with user tier discount"""
        # Mock premium user tier with 0.8 multiplier (20% discount)
        with patch.object(ftns_service, '_get_user_tier_multiplier', return_value=0.8):
            cost = await ftns_service.calculate_context_cost(mock_session, 100)
            
            expected_cost = 100 * CONTEXT_UNIT_COST * 0.8  # 10.0 * 0.8 = 8.0
            assert cost == expected_cost
    
    @pytest.mark.asyncio
    async def test_calculate_context_cost_with_complexity(self, ftns_service, mock_session):
        """Test context cost calculation with complexity multiplier using mock"""
        # Mock session with complexity estimate attribute
        mock_session_with_complexity = Mock()
        mock_session_with_complexity.user_id = "test_user_001"
        mock_session_with_complexity.complexity_estimate = 0.5  # 50% complexity increase
        
        with patch.object(ftns_service, '_get_user_tier_multiplier', return_value=1.0):
            cost = await ftns_service.calculate_context_cost(mock_session_with_complexity, 100)
            
            # Base cost * complexity multiplier: 10.0 * (1.0 + 0.5 * 0.5) = 10.0 * 1.25 = 12.5
            expected_cost = 100 * CONTEXT_UNIT_COST * 1.25
            assert cost == expected_cost
    
    @pytest.mark.parametrize("context_units,expected_base_cost", [
        (0, 0.0),
        (1, 0.1),
        (10, 1.0),
        (100, 10.0),
        (1000, 100.0),
    ])
    @pytest.mark.asyncio
    async def test_calculate_context_cost_parametrized(
        self, ftns_service, mock_session, context_units, expected_base_cost
    ):
        """Test context cost calculation with various input values"""
        with patch.object(ftns_service, '_get_user_tier_multiplier', return_value=1.0):
            cost = await ftns_service.calculate_context_cost(mock_session, context_units)
            assert cost == expected_base_cost
    
    @pytest.mark.asyncio
    async def test_calculate_context_cost_precision(self, ftns_service, mock_session):
        """Test financial precision in context cost calculation"""
        with patch.object(ftns_service, '_get_user_tier_multiplier', return_value=1.0):
            # Test with fractional context units that require precision
            cost = await ftns_service.calculate_context_cost(mock_session, 333)
            
            # Should maintain precision: 333 * 0.1 = 33.3 exactly
            expected_cost = 33.3
            assert cost == expected_cost
            assert len(str(cost).split('.')[-1]) <= 8  # Max 8 decimal places
    
    # === Context Charging Tests ===
    
    @pytest.mark.asyncio
    async def test_charge_context_access_sufficient_balance(self, ftns_service, mock_balance):
        """Test successful context charging with sufficient balance"""
        user_id = "test_user_001"
        ftns_service.balances[user_id] = mock_balance
        
        result = await ftns_service.charge_context_access(user_id, 100)
        
        assert result is True
        # Balance should be reduced by cost (100 * 0.1 = 10.0)
        assert ftns_service.balances[user_id].balance == 990.0
    
    @pytest.mark.asyncio
    async def test_charge_context_access_insufficient_balance(self, ftns_service, mock_balance):
        """Test context charging failure with insufficient balance"""
        user_id = "test_user_001"
        # Set balance lower than cost (100 * 0.1 = 10.0)
        mock_balance.balance = 5.0
        ftns_service.balances[user_id] = mock_balance
        
        result = await ftns_service.charge_context_access(user_id, 100)
        
        assert result is False
        # Balance should remain unchanged
        assert ftns_service.balances[user_id].balance == 5.0
    
    @pytest.mark.asyncio
    async def test_charge_context_access_creates_transaction(self, ftns_service, mock_balance):
        """Test that context charging creates proper transaction record"""
        user_id = "test_user_001"
        ftns_service.balances[user_id] = mock_balance
        
        await ftns_service.charge_context_access(user_id, 100)
        
        # Should create a transaction record
        assert len(ftns_service.transactions) == 1
        transaction = ftns_service.transactions[0]
        assert transaction.from_user == user_id
        assert transaction.to_user == "system"
        assert transaction.amount == 10.0  # Positive amount in transaction
        assert transaction.transaction_type == "charge"
    
    # === Agent Cost Reference Tests ===
    
    def test_agent_costs_constants(self):
        """Test that agent cost constants are properly defined"""
        assert AGENT_COSTS["architect"] > 0
        assert AGENT_COSTS["prompter"] > 0
        assert AGENT_COSTS["router"] > 0
        assert AGENT_COSTS["executor"] > 0
        assert AGENT_COSTS["compiler"] > 0
    
    def test_agent_costs_relative_pricing(self):
        """Test relative pricing relationships between agents"""
        # Compiler should be most expensive (complex synthesis)
        assert AGENT_COSTS["compiler"] >= AGENT_COSTS["executor"]
        assert AGENT_COSTS["executor"] >= AGENT_COSTS["architect"]
        assert AGENT_COSTS["architect"] >= AGENT_COSTS["prompter"]
        assert AGENT_COSTS["prompter"] >= AGENT_COSTS["router"]
    
    # === Reward Calculation Tests ===
    
    @pytest.mark.asyncio
    async def test_calculate_contribution_reward_data(self, ftns_service):
        """Test data contribution reward calculation"""
        # 100 MB upload
        reward = await ftns_service._calculate_contribution_reward("data", 100.0, None)
        expected_reward = 100 * REWARD_PER_MB
        assert reward == expected_reward
    
    @pytest.mark.asyncio
    async def test_calculate_contribution_reward_model(self, ftns_service):
        """Test model contribution reward calculation"""
        reward = await ftns_service._calculate_contribution_reward("model", 0.0, None)
        assert reward == MODEL_CONTRIBUTION_REWARD
    
    @pytest.mark.asyncio
    async def test_calculate_contribution_reward_teaching(self, ftns_service):
        """Test teaching reward calculation"""
        # Teaching rewards are passed as value directly
        reward_value = 25.0
        reward = await ftns_service._calculate_contribution_reward("teaching", reward_value, None)
        assert reward == reward_value
    
    # === Balance Management Tests ===
    
    @pytest.mark.asyncio
    async def test_get_user_balance_existing(self, ftns_service, mock_balance):
        """Test getting balance for existing user"""
        user_id = "test_user_001"
        ftns_service.balances[user_id] = mock_balance
        
        balance = await ftns_service.get_user_balance(user_id)
        assert balance == mock_balance
    
    @pytest.mark.asyncio
    async def test_get_user_balance_new_user(self, ftns_service):
        """Test getting balance for new user creates default balance"""
        user_id = "new_user_001"
        
        balance = await ftns_service.get_user_balance(user_id)
        assert balance.user_id == user_id
        assert balance.balance == 0.0
        assert balance.locked_balance == 0.0
        assert user_id in ftns_service.balances
    
    @pytest.mark.asyncio
    async def test_update_balance_add(self, ftns_service, mock_balance):
        """Test balance update - adding funds"""
        user_id = "test_user_001"
        ftns_service.balances[user_id] = mock_balance
        original_balance = mock_balance.balance
        
        await ftns_service._update_balance(user_id, 100.0)
        assert ftns_service.balances[user_id].balance == original_balance + 100.0
    
    @pytest.mark.asyncio
    async def test_update_balance_subtract(self, ftns_service, mock_balance):
        """Test balance update - subtracting funds"""
        user_id = "test_user_001"
        ftns_service.balances[user_id] = mock_balance
        original_balance = mock_balance.balance
        
        await ftns_service._update_balance(user_id, -50.0)
        assert ftns_service.balances[user_id].balance == original_balance - 50.0
    
    @pytest.mark.asyncio
    async def test_reward_contribution_creates_transaction(self, ftns_service):
        """Test that reward contribution creates transaction records"""
        user_id = "test_user_001"
        
        await ftns_service.reward_contribution(user_id, "data", 100.0)
        
        # Should create transaction
        assert len(ftns_service.transactions) == 1
        transaction = ftns_service.transactions[0]
        assert transaction.to_user == user_id
        assert transaction.from_user is None  # System mint
        assert transaction.amount == 100.0 * REWARD_PER_MB
        assert transaction.transaction_type == "reward"
    
    # === Financial Edge Cases ===
    
    @pytest.mark.asyncio
    async def test_zero_context_units_cost(self, ftns_service, mock_session):
        """Test cost calculation for zero context units"""
        with patch.object(ftns_service, '_get_user_tier_multiplier', return_value=1.0):
            cost = await ftns_service.calculate_context_cost(mock_session, 0)
            assert cost == 0.0
    
    @pytest.mark.asyncio
    async def test_large_context_units_cost(self, ftns_service, mock_session):
        """Test cost calculation for very large context units"""
        large_units = 1000000  # 1 million context units
        with patch.object(ftns_service, '_get_user_tier_multiplier', return_value=1.0):
            cost = await ftns_service.calculate_context_cost(mock_session, large_units)
            expected_cost = large_units * CONTEXT_UNIT_COST
            assert cost == expected_cost
    
    @pytest.mark.asyncio
    async def test_negative_balance_protection(self, ftns_service):
        """Test that balances cannot go negative through normal operations"""
        user_id = "test_user_001"
        balance = FTNSBalance(user_id=user_id, balance=10.0)
        ftns_service.balances[user_id] = balance
        
        # Try to charge more than available (200 * 0.1 = 20.0 > 10.0)
        result = await ftns_service.charge_context_access(user_id, 200)
        
        assert result is False
        assert ftns_service.balances[user_id].balance == 10.0  # Unchanged
    
    # === Integration with Core Models ===
    
    @pytest.mark.asyncio
    async def test_session_integration(self, ftns_service):
        """Test integration with PRSMSession model"""
        session = PRSMSession(
            user_id="integration_test_user",
            nwtn_context_allocation=50,
            status="pending"
        )
        
        with patch.object(ftns_service, '_get_user_tier_multiplier', return_value=1.0):
            cost = await ftns_service.calculate_context_cost(session, 100)
            assert isinstance(cost, float)
            assert cost > 0
    
    # === Error Handling Tests ===
    
    @pytest.mark.asyncio
    async def test_invalid_user_id_handling(self, ftns_service):
        """Test handling of invalid user IDs"""
        # None should fail
        with pytest.raises((ValueError, TypeError, AttributeError)):
            await ftns_service.get_user_balance(None)
        
        # Empty string is actually accepted and creates a balance
        balance = await ftns_service.get_user_balance("")
        assert balance.user_id == ""
        assert balance.balance == 0.0
    
    @pytest.mark.asyncio
    async def test_negative_context_units(self, ftns_service, mock_session):
        """Test handling of negative context units"""
        with patch.object(ftns_service, '_get_user_tier_multiplier', return_value=1.0):
            # Current implementation doesn't validate negative units, just calculates cost
            cost = await ftns_service.calculate_context_cost(mock_session, -10)
            # Should return negative cost for negative units
            expected_cost = -10 * CONTEXT_UNIT_COST
            assert cost == expected_cost


class TestFTNSServiceIntegration:
    """Integration tests for FTNS Service with external dependencies"""
    
    @pytest.fixture
    def ftns_service(self):
        return FTNSService()
    
    @pytest.mark.asyncio
    async def test_user_tier_multiplier_integration(self, ftns_service):
        """Test user tier multiplier calculation (currently returns default)"""
        # The current implementation returns 1.0 as default
        multiplier = await ftns_service._get_user_tier_multiplier("premium_user")
        assert isinstance(multiplier, float)
        assert multiplier == 1.0  # Current default implementation
    
    @pytest.mark.asyncio
    async def test_transaction_persistence(self, ftns_service):
        """Test that transactions are properly formatted for database persistence"""
        user_id = "persistence_test_user"
        balance = FTNSBalance(user_id=user_id, balance=100.0)
        ftns_service.balances[user_id] = balance
        
        # Use reward_contribution which creates transactions
        await ftns_service.reward_contribution(user_id, "data", 25.0)
        
        transaction = ftns_service.transactions[0]
        # Verify transaction has all required fields for database
        assert hasattr(transaction, 'transaction_id')
        assert hasattr(transaction, 'to_user')
        assert hasattr(transaction, 'amount')
        assert hasattr(transaction, 'transaction_type')
        assert hasattr(transaction, 'created_at')
        assert transaction.to_user == user_id
        assert transaction.amount == 25.0 * REWARD_PER_MB