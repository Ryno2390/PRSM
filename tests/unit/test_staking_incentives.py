"""
Tests for Staking and Incentive Mechanisms (Phase 3.2)

This module tests the staking system including:
- Staking operations (stake, unstake, withdraw)
- Slashing mechanism for misbehavior
- Reward calculations and distribution
- Supply adjustments
- Anti-hoarding decay integration
"""

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from prsm.economy.tokenomics.staking_manager import (
    StakingManager,
    StakingConfig,
    StakeRecord,
    UnstakeRequest,
    SlashRecord,
    RewardCalculation,
    StakeStatus,
    UnstakeRequestStatus,
    SlashReason,
    StakeType,
    get_staking_manager
)


# === Fixtures ===

@pytest.fixture
def staking_config():
    """Create a test staking configuration"""
    return StakingConfig(
        minimum_stake=1000,
        unstaking_period_seconds=60,  # 1 minute for testing
        reward_rate_annual=0.05,
        slashing_rate_base=0.1,
        max_stake_per_user=10_000_000,
        max_total_stake=1_000_000_000,
        reward_compounding=False,
        min_reward_claim_interval_seconds=60,
        governance_slash_multiplier=2.0,
        appeal_period_seconds=180,  # 3 minutes for testing
        max_stake_operations_per_day=5,
        max_unstake_operations_per_day=3,
        reward_calculation_interval_seconds=60,
        min_stake_age_for_rewards_seconds=60,
        downtime_slash_threshold_seconds=300,
        max_slash_per_event=0.5
    )


@pytest.fixture
def mock_ftns_service():
    """Create a mock FTNS service"""
    service = AsyncMock()
    service.get_available_balance = AsyncMock(return_value=Decimal('100000'))
    service.lock_tokens = AsyncMock(return_value=True)
    service.unlock_tokens = AsyncMock(return_value=True)
    service.burn_tokens = AsyncMock(return_value=True)
    service.mint_tokens = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_db_session():
    """Create a mock database session"""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
async def staking_manager(mock_db_session, mock_ftns_service, staking_config):
    """Create a staking manager instance"""
    return StakingManager(
        db_session=mock_db_session,
        ftns_service=mock_ftns_service,
        config=staking_config
    )


# === StakingConfig Tests ===

class TestStakingConfig:
    """Tests for StakingConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = StakingConfig()
        
        assert config.minimum_stake == 1000
        assert config.unstaking_period_seconds == 7 * 24 * 3600  # 7 days
        assert config.reward_rate_annual == 0.05  # 5%
        assert config.slashing_rate_base == 0.1  # 10%
        assert config.max_stake_per_user == 10_000_000
        assert config.max_total_stake == 1_000_000_000
        assert config.reward_compounding is False
    
    def test_custom_config(self, staking_config):
        """Test custom configuration values"""
        assert staking_config.minimum_stake == 1000
        assert staking_config.unstaking_period_seconds == 60
        assert staking_config.reward_rate_annual == 0.05
        assert staking_config.slashing_rate_base == 0.1


# === StakeRecord Tests ===

class TestStakeRecord:
    """Tests for StakeRecord dataclass"""
    
    def test_stake_record_creation(self):
        """Test creating a stake record"""
        now = datetime.now(timezone.utc)
        stake = StakeRecord(
            stake_id="test-stake-1",
            user_id="user-123",
            amount=Decimal('5000'),
            stake_type=StakeType.GOVERNANCE,
            staked_at=now,
            status=StakeStatus.ACTIVE
        )
        
        assert stake.stake_id == "test-stake-1"
        assert stake.user_id == "user-123"
        assert stake.amount == Decimal('5000')
        assert stake.stake_type == StakeType.GOVERNANCE
        assert stake.status == StakeStatus.ACTIVE
        assert stake.rewards_earned == Decimal('0')
    
    def test_stake_record_string_conversion(self):
        """Test stake record with string status/type"""
        stake = StakeRecord(
            stake_id="test-stake-1",
            user_id="user-123",
            amount=Decimal('5000'),
            stake_type="governance",  # String instead of enum
            staked_at=datetime.now(timezone.utc),
            status="active"  # String instead of enum
        )
        
        assert stake.stake_type == StakeType.GOVERNANCE
        assert stake.status == StakeStatus.ACTIVE


# === UnstakeRequest Tests ===

class TestUnstakeRequest:
    """Tests for UnstakeRequest dataclass"""
    
    def test_unstake_request_creation(self):
        """Test creating an unstake request"""
        now = datetime.now(timezone.utc)
        available = now + timedelta(hours=1)
        
        request = UnstakeRequest(
            request_id="request-1",
            stake_id="stake-1",
            user_id="user-123",
            amount=Decimal('5000'),
            requested_at=now,
            available_at=available,
            status=UnstakeRequestStatus.PENDING
        )
        
        assert request.request_id == "request-1"
        assert request.status == UnstakeRequestStatus.PENDING
        assert request.is_available is False
    
    def test_is_available_property(self):
        """Test is_available property"""
        # Request not yet available
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        request = UnstakeRequest(
            request_id="request-1",
            stake_id="stake-1",
            user_id="user-123",
            amount=Decimal('5000'),
            requested_at=datetime.now(timezone.utc),
            available_at=future,
            status=UnstakeRequestStatus.AVAILABLE
        )
        assert request.is_available is False
        
        # Request available
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        request.available_at = past
        assert request.is_available is True


# === SlashRecord Tests ===

class TestSlashRecord:
    """Tests for SlashRecord dataclass"""
    
    def test_slash_record_creation(self):
        """Test creating a slash record"""
        now = datetime.now(timezone.utc)
        
        slash = SlashRecord(
            slash_id="slash-1",
            stake_id="stake-1",
            user_id="user-123",
            amount_slashed=Decimal('500'),
            reason=SlashReason.MISCONDUCT,
            slash_rate=0.1,
            evidence={"description": "Test evidence"},
            slashed_at=now,
            slashed_by="governance-proposal-1"
        )
        
        assert slash.slash_id == "slash-1"
        assert slash.reason == SlashReason.MISCONDUCT
        assert slash.slash_rate == 0.1
        assert slash.appeal_deadline is not None
        assert slash.appeal_status is None
    
    def test_slash_record_appeal_deadline(self):
        """Test appeal deadline is set correctly"""
        now = datetime.now(timezone.utc)
        slash = SlashRecord(
            slash_id="slash-1",
            stake_id="stake-1",
            user_id="user-123",
            amount_slashed=Decimal('500'),
            reason=SlashReason.MISCONDUCT,
            slash_rate=0.1,
            evidence={},
            slashed_at=now,
            slashed_by="system"
        )
        
        # Default appeal period is 3 days
        expected_deadline = now + timedelta(days=3)
        assert abs((slash.appeal_deadline - expected_deadline).total_seconds()) < 1


# === StakingManager Tests ===

class TestStakingManager:
    """Tests for StakingManager class"""
    
    @pytest.mark.asyncio
    async def test_stake_creation(self, staking_manager, mock_ftns_service):
        """Test creating a stake"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000'),
            stake_type=StakeType.GOVERNANCE
        )
        
        assert stake is not None
        assert stake.user_id == "user-123"
        assert stake.amount == Decimal('5000')
        assert stake.stake_type == StakeType.GOVERNANCE
        assert stake.status == StakeStatus.ACTIVE
        
        # Verify FTNS service was called
        mock_ftns_service.lock_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stake_below_minimum(self, staking_manager):
        """Test staking below minimum amount"""
        with pytest.raises(ValueError, match="below minimum"):
            await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('100')  # Below minimum of 1000
            )
    
    @pytest.mark.asyncio
    async def test_stake_exceeds_user_max(self, staking_manager, mock_ftns_service):
        """Test staking exceeding user maximum"""
        mock_ftns_service.get_available_balance = AsyncMock(
            return_value=Decimal('20000000')
        )
        
        with pytest.raises(ValueError, match="exceed user maximum"):
            await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('15000000')  # Exceeds max of 10M
            )
    
    @pytest.mark.asyncio
    async def test_stake_insufficient_balance(self, staking_manager, mock_ftns_service):
        """Test staking with insufficient balance"""
        mock_ftns_service.get_available_balance = AsyncMock(
            return_value=Decimal('100')  # Not enough
        )
        
        with pytest.raises(ValueError, match="Insufficient balance"):
            await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('5000')
            )
    
    @pytest.mark.asyncio
    async def test_unstake_creation(self, staking_manager, mock_ftns_service):
        """Test creating an unstake request"""
        # First create a stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        # Then unstake
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        
        assert request is not None
        assert request.user_id == "user-123"
        assert request.amount == Decimal('5000')
        assert request.status == UnstakeRequestStatus.PENDING
        assert stake.status == StakeStatus.UNSTAKING
    
    @pytest.mark.asyncio
    async def test_partial_unstake(self, staking_manager, mock_ftns_service):
        """Test partial unstake"""
        # Create stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        # Partial unstake
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id,
            amount=Decimal('2000')
        )
        
        assert request.amount == Decimal('2000')
        assert stake.status == StakeStatus.ACTIVE  # Still active
        assert stake.amount == Decimal('3000')  # Reduced
    
    @pytest.mark.asyncio
    async def test_unstake_nonexistent_stake(self, staking_manager):
        """Test unstaking a non-existent stake"""
        with pytest.raises(ValueError, match="not found"):
            await staking_manager.unstake(
                user_id="user-123",
                stake_id="nonexistent"
            )
    
    @pytest.mark.asyncio
    async def test_unstake_wrong_user(self, staking_manager, mock_ftns_service):
        """Test unstaking another user's stake"""
        # Create stake for user-123
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        # Try to unstake as user-456
        with pytest.raises(ValueError, match="does not belong to user"):
            await staking_manager.unstake(
                user_id="user-456",
                stake_id=stake.stake_id
            )
    
    @pytest.mark.asyncio
    async def test_withdraw_success(self, staking_manager, mock_ftns_service):
        """Test successful withdrawal"""
        # Create and unstake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        
        # Make it available
        request.available_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        request.status = UnstakeRequestStatus.AVAILABLE
        
        # Withdraw
        success, amount = await staking_manager.withdraw(
            user_id="user-123",
            request_id=request.request_id
        )
        
        assert success is True
        assert amount == Decimal('5000')
        assert request.status == UnstakeRequestStatus.COMPLETED
        mock_ftns_service.unlock_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_withdraw_too_early(self, staking_manager, mock_ftns_service):
        """Test withdrawing before period ends"""
        # Create and unstake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        
        # Try to withdraw immediately (period not over)
        with pytest.raises(ValueError, match="not complete"):
            await staking_manager.withdraw(
                user_id="user-123",
                request_id=request.request_id
            )
    
    @pytest.mark.asyncio
    async def test_cancel_unstake(self, staking_manager, mock_ftns_service):
        """Test cancelling an unstake request"""
        # Create and unstake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        
        # Cancel
        result = await staking_manager.cancel_unstake(
            user_id="user-123",
            request_id=request.request_id,
            reason="Changed mind"
        )
        
        assert result is True
        assert request.status == UnstakeRequestStatus.CANCELLED
        assert stake.status == StakeStatus.ACTIVE  # Restored


# === Slashing Tests ===

class TestSlashing:
    """Tests for slashing mechanism"""
    
    @pytest.mark.asyncio
    async def test_slash_creation(self, staking_manager, mock_ftns_service):
        """Test creating a slash"""
        # Create stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        # Slash
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.MISCONDUCT,
            evidence={"description": "Test violation"},
            slashed_by="governance-proposal-1"
        )
        
        assert slash is not None
        assert slash.reason == SlashReason.MISCONDUCT
        assert slash.slash_rate == 0.1  # Default 10%
        assert slash.amount_slashed == Decimal('500')  # 10% of 5000
        assert stake.amount == Decimal('4500')  # Reduced
        
        mock_ftns_service.burn_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_slash_custom_rate(self, staking_manager, mock_ftns_service):
        """Test slashing with custom rate"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.FRAUD,
            evidence={"description": "Fraud detected"},
            slash_rate=0.3,  # 30%
            slashed_by="system"
        )
        
        assert slash.slash_rate == 0.3
        assert slash.amount_slashed == Decimal('1500')  # 30% of 5000
        assert stake.amount == Decimal('3500')
    
    @pytest.mark.asyncio
    async def test_slash_max_cap(self, staking_manager, mock_ftns_service):
        """Test slash is capped at maximum"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        # Try to slash 80% (over max of 50%)
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.SECURITY_BREACH,
            evidence={"description": "Security breach"},
            slash_rate=0.8,  # 80%
            slashed_by="system"
        )
        
        # Should be capped at 50%
        assert slash.slash_rate == 0.5
        assert slash.amount_slashed == Decimal('2500')  # 50% of 5000
    
    @pytest.mark.asyncio
    async def test_slash_depletes_stake(self, staking_manager, mock_ftns_service):
        """Test that slashing can deplete a stake"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('1000')
        )
        
        # Slash 50% (max)
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.FRAUD,
            evidence={"description": "Major fraud"},
            slash_rate=0.5,
            slashed_by="governance"
        )
        
        assert stake.amount == Decimal('500')
        assert stake.status == StakeStatus.ACTIVE  # Still active
        
        # Slash again to deplete
        slash2 = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.FRAUD,
            evidence={"description": "More fraud"},
            slash_rate=0.5,
            slashed_by="governance"
        )
        
        assert stake.amount == Decimal('250')
        
        # Slash to depletion
        slash3 = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.FRAUD,
            evidence={"description": "Final fraud"},
            slash_rate=1.0,  # 100% - but capped at 50%
            slashed_by="governance"
        )
        
        # Should be slashed but not fully depleted due to cap
        assert stake.status == StakeStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_appeal_slash(self, staking_manager, mock_ftns_service):
        """Test submitting a slash appeal"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.MISCONDUCT,
            evidence={"description": "Test"},
            slashed_by="system"
        )
        
        # Submit appeal
        result = await staking_manager.appeal_slash(
            user_id="user-123",
            slash_id=slash.slash_id,
            appeal_evidence={"description": "False accusation"}
        )
        
        assert result is True
        assert slash.appeal_status == "pending"
    
    @pytest.mark.asyncio
    async def test_resolve_appeal_approved(self, staking_manager, mock_ftns_service):
        """Test resolving an appeal (approved)"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.MISCONDUCT,
            evidence={"description": "Test"},
            slashed_by="system"
        )
        
        # Submit appeal
        await staking_manager.appeal_slash(
            user_id="user-123",
            slash_id=slash.slash_id,
            appeal_evidence={"description": "False accusation"}
        )
        
        # Resolve appeal (approved)
        result = await staking_manager.resolve_appeal(
            slash_id=slash.slash_id,
            approved=True,
            resolution_note="Evidence was incorrect"
        )
        
        assert result is True
        assert slash.appeal_status == "approved"
        assert stake.amount == Decimal('5000')  # Refunded
        mock_ftns_service.mint_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_resolve_appeal_rejected(self, staking_manager, mock_ftns_service):
        """Test resolving an appeal (rejected)"""
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.MISCONDUCT,
            evidence={"description": "Test"},
            slashed_by="system"
        )
        
        await staking_manager.appeal_slash(
            user_id="user-123",
            slash_id=slash.slash_id,
            appeal_evidence={"description": "Appeal"}
        )
        
        # Reject appeal
        result = await staking_manager.resolve_appeal(
            slash_id=slash.slash_id,
            approved=False,
            resolution_note="Evidence is valid"
        )
        
        assert result is True
        assert slash.appeal_status == "rejected"
        assert stake.amount == Decimal('4500')  # Not refunded


# === Reward Tests ===

class TestRewards:
    """Tests for reward calculations"""
    
    @pytest.mark.asyncio
    async def test_calculate_rewards(self, staking_manager, mock_ftns_service):
        """Test calculating rewards"""
        # Create stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('10000')
        )
        
        # Set stake age to be past minimum
        stake.staked_at = datetime.now(timezone.utc) - timedelta(hours=2)
        stake.last_reward_calculation = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Calculate rewards
        calculations = await staking_manager.calculate_rewards("user-123")
        
        assert len(calculations) == 1
        calc = calculations[0]
        assert calc.user_id == "user-123"
        assert calc.principal == Decimal('10000')
        assert calc.annual_rate == 0.05
        assert calc.reward_amount > 0
    
    @pytest.mark.asyncio
    async def test_calculate_rewards_too_new(self, staking_manager, mock_ftns_service):
        """Test calculating rewards for new stake"""
        # Create stake (too new)
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('10000')
        )
        
        # Try to calculate rewards immediately
        calculations = await staking_manager.calculate_rewards("user-123")
        
        # Should be empty because stake is too new
        assert len(calculations) == 0
    
    @pytest.mark.asyncio
    async def test_claim_rewards(self, staking_manager, mock_ftns_service):
        """Test claiming rewards"""
        # Create stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('10000')
        )
        
        # Set stake age
        stake.staked_at = datetime.now(timezone.utc) - timedelta(hours=2)
        stake.last_reward_calculation = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Claim rewards
        total_rewards = await staking_manager.claim_rewards("user-123")
        
        assert total_rewards > 0
        mock_ftns_service.mint_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reward_compounding(self, mock_db_session, mock_ftns_service):
        """Test reward compounding"""
        config = StakingConfig(
            minimum_stake=1000,
            reward_compounding=True,  # Enable compounding
            min_stake_age_for_rewards_seconds=60  # 1 minute for testing
        )
        manager = StakingManager(
            db_session=mock_db_session,
            ftns_service=mock_ftns_service,
            config=config
        )
        
        # Create stake
        stake = await manager.stake(
            user_id="user-123",
            amount=Decimal('10000')
        )
        
        # Set stake age to be past minimum for rewards
        stake.staked_at = datetime.now(timezone.utc) - timedelta(hours=2)
        stake.last_reward_calculation = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Claim rewards
        total_rewards = await manager.claim_rewards("user-123")
        
        # With compounding, stake amount should increase by the claimed rewards
        assert stake.amount == Decimal('10000') + total_rewards
        assert stake.amount > Decimal('10000')


# === Query Tests ===

class TestQueries:
    """Tests for query operations"""
    
    @pytest.mark.asyncio
    async def test_get_user_stakes(self, staking_manager, mock_ftns_service):
        """Test getting user stakes"""
        # Create multiple stakes
        await staking_manager.stake(user_id="user-123", amount=Decimal('5000'))
        await staking_manager.stake(user_id="user-123", amount=Decimal('3000'))
        await staking_manager.stake(user_id="user-456", amount=Decimal('2000'))
        
        # Get user-123's stakes
        stakes = await staking_manager.get_user_stakes("user-123")
        
        assert len(stakes) == 2
        for stake in stakes:
            assert stake.user_id == "user-123"
    
    @pytest.mark.asyncio
    async def test_get_user_total_stake(self, staking_manager, mock_ftns_service):
        """Test getting user total stake"""
        await staking_manager.stake(user_id="user-123", amount=Decimal('5000'))
        await staking_manager.stake(user_id="user-123", amount=Decimal('3000'))
        
        total = await staking_manager.get_user_total_stake("user-123")
        
        assert total == Decimal('8000')
    
    @pytest.mark.asyncio
    async def test_get_network_total_stake(self, staking_manager, mock_ftns_service):
        """Test getting network total stake"""
        await staking_manager.stake(user_id="user-1", amount=Decimal('5000'))
        await staking_manager.stake(user_id="user-2", amount=Decimal('3000'))
        await staking_manager.stake(user_id="user-3", amount=Decimal('2000'))
        
        total = await staking_manager.get_network_total_stake()
        
        assert total == Decimal('10000')
    
    @pytest.mark.asyncio
    async def test_get_pending_unstake_requests(self, staking_manager, mock_ftns_service):
        """Test getting pending unstake requests"""
        # Create and unstake
        stake1 = await staking_manager.stake(user_id="user-1", amount=Decimal('5000'))
        stake2 = await staking_manager.stake(user_id="user-2", amount=Decimal('3000'))
        
        await staking_manager.unstake(user_id="user-1", stake_id=stake1.stake_id)
        await staking_manager.unstake(user_id="user-2", stake_id=stake2.stake_id)
        
        # Get pending requests
        requests = await staking_manager.get_pending_unstake_requests()
        
        assert len(requests) == 2
    
    @pytest.mark.asyncio
    async def test_get_slash_history(self, staking_manager, mock_ftns_service):
        """Test getting slash history"""
        # Create stakes and slash them
        stake1 = await staking_manager.stake(user_id="user-1", amount=Decimal('5000'))
        stake2 = await staking_manager.stake(user_id="user-2", amount=Decimal('3000'))
        
        await staking_manager.slash(
            user_id="user-1",
            stake_id=stake1.stake_id,
            reason=SlashReason.MISCONDUCT,
            evidence={},
            slashed_by="system"
        )
        await staking_manager.slash(
            user_id="user-2",
            stake_id=stake2.stake_id,
            reason=SlashReason.DOWNTIME,
            evidence={},
            slashed_by="system"
        )
        
        # Get history
        history = await staking_manager.get_slash_history()
        
        assert len(history) == 2
        
        # Get user-specific history
        user_history = await staking_manager.get_slash_history(user_id="user-1")
        assert len(user_history) == 1


# === Rate Limiting Tests ===

class TestRateLimiting:
    """Tests for rate limiting"""
    
    @pytest.mark.asyncio
    async def test_stake_rate_limit(self, staking_manager, mock_ftns_service):
        """Test stake rate limiting"""
        mock_ftns_service.get_available_balance = AsyncMock(
            return_value=Decimal('100000000')
        )
        
        # Make max stakes (5 per day)
        for i in range(5):
            await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('1000')
            )
        
        # Next should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('1000')
            )
    
    @pytest.mark.asyncio
    async def test_unstake_rate_limit(self, staking_manager, mock_ftns_service):
        """Test unstake rate limiting"""
        # Create stakes
        stakes = []
        for i in range(5):
            stake = await staking_manager.stake(
                user_id="user-123",
                amount=Decimal('1000')
            )
            stakes.append(stake)
        
        # Make max unstakes (3 per day)
        for i in range(3):
            await staking_manager.unstake(
                user_id="user-123",
                stake_id=stakes[i].stake_id
            )
        
        # Next should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            await staking_manager.unstake(
                user_id="user-123",
                stake_id=stakes[3].stake_id
            )


# === Maintenance Tests ===

class TestMaintenance:
    """Tests for maintenance operations"""
    
    @pytest.mark.asyncio
    async def test_process_matured_unstakes(self, staking_manager, mock_ftns_service):
        """Test processing matured unstakes"""
        # Create and unstake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('5000')
        )
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        
        # Make it available
        request.available_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        
        # Process matured
        matured = await staking_manager.process_matured_unstakes()
        
        assert len(matured) == 1
        assert matured[0].request_id == request.request_id
        assert matured[0].status == UnstakeRequestStatus.AVAILABLE
    
    @pytest.mark.asyncio
    async def test_get_staking_stats(self, staking_manager, mock_ftns_service):
        """Test getting staking statistics"""
        # Create some stakes
        await staking_manager.stake(user_id="user-1", amount=Decimal('5000'))
        await staking_manager.stake(user_id="user-2", amount=Decimal('3000'))
        
        stats = await staking_manager.get_staking_stats()
        
        assert stats["total_stakes"] == 2
        assert stats["active_stakes"] == 2
        assert stats["total_staked"] == 8000.0
        assert stats["unique_stakers"] == 2
        assert "config" in stats


# === Factory Function Tests ===

class TestFactoryFunction:
    """Tests for get_staking_manager factory"""
    
    @pytest.mark.asyncio
    async def test_get_staking_manager_singleton(self, mock_db_session, mock_ftns_service):
        """Test that get_staking_manager returns singleton"""
        manager1 = await get_staking_manager(
            db_session=mock_db_session,
            ftns_service=mock_ftns_service
        )
        manager2 = await get_staking_manager(
            db_session=mock_db_session,
            ftns_service=mock_ftns_service
        )
        
        assert manager1 is manager2


# === Integration Tests ===

class TestIntegration:
    """Integration tests for staking system"""
    
    @pytest.mark.asyncio
    async def test_full_staking_lifecycle(self, staking_manager, mock_ftns_service):
        """Test complete staking lifecycle"""
        # 1. Stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('10000'),
            stake_type=StakeType.GOVERNANCE
        )
        assert stake.status == StakeStatus.ACTIVE
        
        # 2. Check balance
        total = await staking_manager.get_user_total_stake("user-123")
        assert total == Decimal('10000')
        
        # 3. Request unstake
        request = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id,
            amount=Decimal('5000')
        )
        assert request.status == UnstakeRequestStatus.PENDING
        assert stake.amount == Decimal('5000')  # Partial unstake
        
        # 4. Cancel unstake
        await staking_manager.cancel_unstake(
            user_id="user-123",
            request_id=request.request_id
        )
        assert stake.amount == Decimal('10000')  # Restored
        
        # 5. Full unstake
        request2 = await staking_manager.unstake(
            user_id="user-123",
            stake_id=stake.stake_id
        )
        assert stake.status == StakeStatus.UNSTAKING
        
        # 6. Make available and withdraw
        request2.available_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        request2.status = UnstakeRequestStatus.AVAILABLE
        
        success, amount = await staking_manager.withdraw(
            user_id="user-123",
            request_id=request2.request_id
        )
        assert success is True
        assert amount == Decimal('10000')
    
    @pytest.mark.asyncio
    async def test_stake_slash_and_appeal(self, staking_manager, mock_ftns_service):
        """Test stake, slash, and appeal flow"""
        # 1. Stake
        stake = await staking_manager.stake(
            user_id="user-123",
            amount=Decimal('10000')
        )
        
        # 2. Slash
        slash = await staking_manager.slash(
            user_id="user-123",
            stake_id=stake.stake_id,
            reason=SlashReason.VALIDATION_FAILURE,
            evidence={"description": "Failed validation"},
            slashed_by="validator"
        )
        assert stake.amount == Decimal('9000')  # 10% slashed
        
        # 3. Appeal
        await staking_manager.appeal_slash(
            user_id="user-123",
            slash_id=slash.slash_id,
            appeal_evidence={"description": "False positive"}
        )
        
        # 4. Resolve (approved)
        await staking_manager.resolve_appeal(
            slash_id=slash.slash_id,
            approved=True,
            resolution_note="Appeal approved"
        )
        
        # Stake should be restored
        assert stake.amount == Decimal('10000')