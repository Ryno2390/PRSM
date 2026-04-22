"""
Unit tests for PRSM Marketplace Analytics Service

Tests all stub implementations with mocked database sessions.
No real database connections - all DB operations are mocked.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import json

from prsm.economy.marketplace.analytics_service import (
    MarketplaceAnalyticsService,
    ActivityType,
    MetricPeriod,
    UserActivity,
    RevenueMetric,
    UserEngagementMetric,
    MarketplaceKPI,
)
from prsm.economy.marketplace.database_models import (
    MarketplaceActivityModel, MarketplaceOrder, MarketplaceResource
)
from prsm.economy.marketplace.database_models import UserReputationModel


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def service():
    """Create a fresh analytics service for each test."""
    return MarketplaceAnalyticsService()


@pytest.fixture
def mock_session():
    """Create a mock async database session."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.fixture
def sample_activity():
    """Create a sample user activity for testing."""
    return UserActivity(
        activity_id="activity_123",
        user_id="user_456",
        activity_type=ActivityType.PURCHASE,
        resource_id="resource_789",
        timestamp=datetime.now(timezone.utc),
        session_id="session_abc",
        metadata={"source": "web", "page": "/marketplace"},
        value=99.99
    )


@pytest.fixture
def sample_order():
    """Create a sample marketplace order for testing."""
    order = MagicMock(spec=MarketplaceOrder)
    order.id = uuid4()
    order.order_id = "order_123"
    order.buyer_id = "user_456"
    order.resource_id = "resource_789"
    order.status = "completed"
    order.total_price = 150.0
    order.created_at = datetime.now(timezone.utc) - timedelta(days=7)
    return order


@pytest.fixture
def sample_resource():
    """Create a sample marketplace resource for testing."""
    resource = MagicMock(spec=MarketplaceResource)
    resource.id = uuid4()
    resource.resource_id = "resource_789"
    resource.resource_type = "model"
    resource.title = "Test Model"
    resource.download_count = 500
    resource.usage_count = 1000
    return resource


# =============================================================================
# Tests for _store_activity
# =============================================================================

class TestStoreActivity:
    """Tests for _store_activity method."""

    @pytest.mark.asyncio
    async def test_store_activity_basic(self, service, mock_session, sample_activity):
        """Test storing a basic activity record."""
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            await service._store_activity(sample_activity)
        
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_activity_with_metadata(self, service, mock_session):
        """Test storing activity with complex metadata."""
        activity = UserActivity(
            activity_id="activity_meta",
            user_id="user_123",
            activity_type=ActivityType.SEARCH,
            resource_id=None,
            timestamp=datetime.now(timezone.utc),
            session_id="session_1",
            metadata={
                "query": "NLP model",
                "filters": {"type": "model", "price_max": 100},
                "results_count": 25
            }
        )
        
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            await service._store_activity(activity)
        
        mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_activity_purchase_with_value(self, service, mock_session):
        """Test storing purchase activity with monetary value."""
        activity = UserActivity(
            activity_id="purchase_1",
            user_id="buyer_123",
            activity_type=ActivityType.PURCHASE,
            resource_id="resource_456",
            timestamp=datetime.now(timezone.utc),
            session_id="session_2",
            metadata={},
            value=299.99
        )
        
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            await service._store_activity(activity)
        
        mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_activity_database_error(self, service, mock_session, sample_activity):
        """Test handling database errors during storage."""
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock(side_effect=Exception("DB Error"))
        mock_session.rollback = AsyncMock()
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            # Should not raise, just log error
            await service._store_activity(sample_activity)
        
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_activity_all_activity_types(self, service, mock_session):
        """Test storing all different activity types."""
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        
        activity_types = [
            ActivityType.PAGE_VIEW,
            ActivityType.SEARCH,
            ActivityType.RESOURCE_VIEW,
            ActivityType.PURCHASE,
            ActivityType.DOWNLOAD,
            ActivityType.RATING,
            ActivityType.REVIEW,
            ActivityType.SHARE,
            ActivityType.BOOKMARK,
            ActivityType.API_CALL,
        ]
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            for activity_type in activity_types:
                activity = UserActivity(
                    activity_id=f"activity_{activity_type.value}",
                    user_id="user_123",
                    activity_type=activity_type,
                    resource_id="resource_1",
                    timestamp=datetime.now(timezone.utc),
                    session_id="session_1",
                    metadata={}
                )
                await service._store_activity(activity)
        
        assert mock_session.add.call_count == len(activity_types)


# =============================================================================
# Tests for _update_realtime_metrics
# =============================================================================

class TestUpdateRealtimeMetrics:
    """Tests for _update_realtime_metrics method."""

    @pytest.mark.asyncio
    async def test_update_realtime_metrics_download(self, service, mock_session, sample_resource):
        """Test updating metrics for download activity."""
        valid_resource_uuid = str(uuid4())
        activity = UserActivity(
            activity_id="download_1",
            user_id="user_123",
            activity_type=ActivityType.DOWNLOAD,
            resource_id=valid_resource_uuid,
            timestamp=datetime.now(timezone.utc),
            session_id="session_1",
            metadata={}
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_resource
        mock_session.execute.return_value = mock_result
        mock_session.commit = AsyncMock()

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            await service._update_realtime_metrics(activity)

        mock_session.execute.assert_called()

    @pytest.mark.asyncio
    async def test_update_realtime_metrics_api_call(self, service, mock_session, sample_resource):
        """Test updating metrics for API call activity."""
        valid_resource_uuid = str(uuid4())
        activity = UserActivity(
            activity_id="api_1",
            user_id="user_123",
            activity_type=ActivityType.API_CALL,
            resource_id=valid_resource_uuid,
            timestamp=datetime.now(timezone.utc),
            session_id="session_1",
            metadata={}
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_resource
        mock_session.execute.return_value = mock_result
        mock_session.commit = AsyncMock()

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            await service._update_realtime_metrics(activity)

        mock_session.execute.assert_called()

    @pytest.mark.asyncio
    async def test_update_realtime_metrics_no_resource_id(self, service, mock_session):
        """Test handling activity without resource ID."""
        activity = UserActivity(
            activity_id="search_1",
            user_id="user_123",
            activity_type=ActivityType.SEARCH,
            resource_id=None,
            timestamp=datetime.now(timezone.utc),
            session_id="session_1",
            metadata={"query": "test"}
        )
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            await service._update_realtime_metrics(activity)
        
        # Should not execute any queries when no resource_id
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_realtime_metrics_resource_not_found(self, service, mock_session):
        """Test handling when resource not found."""
        valid_nonexistent_uuid = str(uuid4())
        activity = UserActivity(
            activity_id="download_1",
            user_id="user_123",
            activity_type=ActivityType.DOWNLOAD,
            resource_id=valid_nonexistent_uuid,
            timestamp=datetime.now(timezone.utc),
            session_id="session_1",
            metadata={}
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            await service._update_realtime_metrics(activity)

        mock_session.execute.assert_called()

    @pytest.mark.asyncio
    async def test_update_realtime_metrics_database_error(self, service, mock_session):
        """Test handling database errors during metric update."""
        valid_resource_uuid = str(uuid4())
        activity = UserActivity(
            activity_id="download_1",
            user_id="user_123",
            activity_type=ActivityType.DOWNLOAD,
            resource_id=valid_resource_uuid,
            timestamp=datetime.now(timezone.utc),
            session_id="session_1",
            metadata={}
        )

        mock_session.execute.side_effect = Exception("DB Error")
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            # Should not raise, just log error
            await service._update_realtime_metrics(activity)


# =============================================================================
# Tests for _categorize_users
# =============================================================================

class TestCategorizeUsers:
    """Tests for _categorize_users method."""

    @pytest.mark.asyncio
    async def test_categorize_users_basic(self, service, mock_session):
        """Test basic user categorization."""
        user_ids = {"user_1", "user_2", "user_3"}
        period_start = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Mock user reputation models
        user1 = MagicMock(spec=UserReputationModel)
        user1.user_id = "user_1"
        user1.created_at = datetime.now(timezone.utc) - timedelta(days=5)  # New user
        
        user2 = MagicMock(spec=UserReputationModel)
        user2.user_id = "user_2"
        user2.created_at = datetime.now(timezone.utc) - timedelta(days=60)  # Returning user
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [user1, user2]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            new_users, returning_users = await service._categorize_users(user_ids, period_start)
        
        assert isinstance(new_users, set)
        assert isinstance(returning_users, set)

    @pytest.mark.asyncio
    async def test_categorize_users_all_new(self, service, mock_session):
        """Test when all users are new."""
        user_ids = {"user_1", "user_2"}
        period_start = datetime.now(timezone.utc) - timedelta(days=30)
        
        user1 = MagicMock(spec=UserReputationModel)
        user1.user_id = "user_1"
        user1.created_at = datetime.now(timezone.utc) - timedelta(days=5)
        
        user2 = MagicMock(spec=UserReputationModel)
        user2.user_id = "user_2"
        user2.created_at = datetime.now(timezone.utc) - timedelta(days=10)
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [user1, user2]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            new_users, returning_users = await service._categorize_users(user_ids, period_start)
        
        assert isinstance(new_users, set)
        assert isinstance(returning_users, set)

    @pytest.mark.asyncio
    async def test_categorize_users_all_returning(self, service, mock_session):
        """Test when all users are returning."""
        user_ids = {"user_1", "user_2"}
        period_start = datetime.now(timezone.utc) - timedelta(days=30)
        
        user1 = MagicMock(spec=UserReputationModel)
        user1.user_id = "user_1"
        user1.created_at = datetime.now(timezone.utc) - timedelta(days=60)
        
        user2 = MagicMock(spec=UserReputationModel)
        user2.user_id = "user_2"
        user2.created_at = datetime.now(timezone.utc) - timedelta(days=90)
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [user1, user2]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            new_users, returning_users = await service._categorize_users(user_ids, period_start)
        
        assert isinstance(new_users, set)
        assert isinstance(returning_users, set)

    @pytest.mark.asyncio
    async def test_categorize_users_empty_set(self, service, mock_session):
        """Test with empty user set."""
        user_ids = set()
        period_start = datetime.now(timezone.utc) - timedelta(days=30)
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            new_users, returning_users = await service._categorize_users(user_ids, period_start)
        
        assert new_users == set()
        assert returning_users == set()

    @pytest.mark.asyncio
    async def test_categorize_users_database_error(self, service, mock_session):
        """Test handling database errors — uses hash-based fallback split."""
        user_ids = {"user_1", "user_2"}
        period_start = datetime.now(timezone.utc) - timedelta(days=30)

        mock_session.execute.side_effect = Exception("DB Error")

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            new_users, returning_users = await service._categorize_users(user_ids, period_start)

        # On DB error the impl uses hash-based fallback; all users are distributed
        assert isinstance(new_users, set)
        assert isinstance(returning_users, set)
        assert new_users | returning_users == user_ids


# =============================================================================
# Tests for _calculate_retention_rate
# =============================================================================

class TestCalculateRetentionRate:
    """Tests for _calculate_retention_rate method."""

    @pytest.mark.asyncio
    async def test_calculate_retention_rate_basic(self, service):
        """Test basic retention rate calculation."""
        # The method uses the activity buffer, not database
        result = await service._calculate_retention_rate(MetricPeriod.WEEKLY)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_retention_rate_daily(self, service):
        """Test retention rate for daily period."""
        result = await service._calculate_retention_rate(MetricPeriod.DAILY)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_retention_rate_monthly(self, service):
        """Test retention rate for monthly period."""
        result = await service._calculate_retention_rate(MetricPeriod.MONTHLY)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_retention_rate_empty_buffer(self, service):
        """Test retention rate with empty activity buffer."""
        # Clear the activity buffer
        service.activity_buffer = []
        
        result = await service._calculate_retention_rate(MetricPeriod.WEEKLY)
        
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_retention_rate_with_activities(self, service):
        """Test retention rate with activities in buffer."""
        # Add some activities to the buffer
        now = datetime.now(timezone.utc)
        service.activity_buffer = [
            UserActivity(
                activity_id=f"activity_{i}",
                user_id=f"user_{i % 5}",  # 5 unique users
                activity_type=ActivityType.PAGE_VIEW,
                resource_id="resource_1",
                timestamp=now - timedelta(days=i),
                session_id=f"session_{i}",
                metadata={}
            )
            for i in range(20)
        ]

        result = await service._calculate_retention_rate(MetricPeriod.WEEKLY)

        assert isinstance(result, float)
        # impl returns percentage (0-100), not fraction (0-1)
        assert 0.0 <= result <= 100.0


# =============================================================================
# Tests for _calculate_customer_lifetime_value
# =============================================================================

class TestCalculateCustomerLifetimeValue:
    """Tests for _calculate_customer_lifetime_value method."""

    @pytest.mark.asyncio
    async def test_calculate_clv_basic(self, service, mock_session):
        """Test basic CLV calculation."""
        mock_result = MagicMock()
        # rows of (buyer_user_id, total_spent) — 10 buyers each spending 150.0
        mock_result.fetchall.return_value = [(f"user_{i}", 150.0) for i in range(10)]
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service._calculate_customer_lifetime_value()

        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_calculate_clv_no_orders(self, service, mock_session):
        """Test CLV when no orders exist."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service._calculate_customer_lifetime_value()

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_clv_single_customer(self, service, mock_session):
        """Test CLV with single customer."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("user_1", 500.0)]
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service._calculate_customer_lifetime_value()

        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_calculate_clv_database_error(self, service, mock_session):
        """Test handling database errors."""
        mock_session.execute.side_effect = Exception("DB Error")

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service._calculate_customer_lifetime_value()

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_clv_high_value_customers(self, service, mock_session):
        """Test CLV with high-value customers."""
        mock_result = MagicMock()
        # 50 high-value customers spending 2000 each
        mock_result.fetchall.return_value = [(f"user_{i}", 2000.0) for i in range(50)]
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service._calculate_customer_lifetime_value()

        assert isinstance(result, float)


# =============================================================================
# Tests for _calculate_recommendation_click_rate
# =============================================================================

class TestCalculateRecommendationClickRate:
    """Tests for _calculate_recommendation_click_rate method."""

    @pytest.mark.asyncio
    async def test_calculate_click_rate_basic(self, service):
        """Test basic click rate calculation."""
        activities = [
            UserActivity(
                activity_id="impression_1",
                user_id="user_1",
                activity_type=ActivityType.PAGE_VIEW,
                resource_id="resource_1",
                timestamp=datetime.now(timezone.utc),
                session_id="session_1",
                metadata={"recommendation_impression": True}
            ),
            UserActivity(
                activity_id="click_1",
                user_id="user_1",
                activity_type=ActivityType.RESOURCE_VIEW,
                resource_id="resource_1",
                timestamp=datetime.now(timezone.utc),
                session_id="session_1",
                metadata={"recommendation_click": True}
            )
        ]
        
        result = await service._calculate_recommendation_click_rate(activities)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_click_rate_no_impressions(self, service):
        """Test click rate with no impressions."""
        activities = [
            UserActivity(
                activity_id="click_1",
                user_id="user_1",
                activity_type=ActivityType.RESOURCE_VIEW,
                resource_id="resource_1",
                timestamp=datetime.now(timezone.utc),
                session_id="session_1",
                metadata={"recommendation_click": True}
            )
        ]
        
        result = await service._calculate_recommendation_click_rate(activities)
        
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_click_rate_no_clicks(self, service):
        """Test click rate with impressions but no clicks."""
        activities = [
            UserActivity(
                activity_id="impression_1",
                user_id="user_1",
                activity_type=ActivityType.PAGE_VIEW,
                resource_id="resource_1",
                timestamp=datetime.now(timezone.utc),
                session_id="session_1",
                metadata={"recommendation_impression": True}
            )
        ]
        
        result = await service._calculate_recommendation_click_rate(activities)
        
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_click_rate_empty_activities(self, service):
        """Test click rate with empty activities list."""
        result = await service._calculate_recommendation_click_rate([])
        
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_click_rate_high_engagement(self, service):
        """Test click rate with high engagement."""
        activities = []
        # 10 impressions
        for i in range(10):
            activities.append(UserActivity(
                activity_id=f"impression_{i}",
                user_id=f"user_{i}",
                activity_type=ActivityType.PAGE_VIEW,
                resource_id="resource_1",
                timestamp=datetime.now(timezone.utc),
                session_id=f"session_{i}",
                metadata={"recommendation_impression": True}
            ))
        # 8 clicks (80% CTR)
        for i in range(8):
            activities.append(UserActivity(
                activity_id=f"click_{i}",
                user_id=f"user_{i}",
                activity_type=ActivityType.RESOURCE_VIEW,
                resource_id="resource_1",
                timestamp=datetime.now(timezone.utc),
                session_id=f"session_{i}",
                metadata={"recommendation_click": True}
            ))
        
        result = await service._calculate_recommendation_click_rate(activities)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Tests for _calculate_dispute_rate
# =============================================================================

class TestCalculateDisputeRate:
    """Tests for _calculate_dispute_rate method."""

    @pytest.mark.asyncio
    async def test_calculate_dispute_rate_basic(self, service, mock_session):
        """Test basic dispute rate calculation."""
        mock_result = MagicMock()
        # Return rows: (status, count) tuples — 95 completed, 5 disputed
        mock_result.fetchall.return_value = [("completed", 95), ("disputed", 5)]
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service._calculate_dispute_rate()

        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0

    @pytest.mark.asyncio
    async def test_calculate_dispute_rate_no_orders(self, service, mock_session):
        """Test dispute rate with no orders."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service._calculate_dispute_rate()

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_dispute_rate_no_disputes(self, service, mock_session):
        """Test dispute rate with no disputes."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("completed", 100)]
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service._calculate_dispute_rate()

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_dispute_rate_high_dispute(self, service, mock_session):
        """Test dispute rate with high dispute count."""
        mock_result = MagicMock()
        # 75 completed, 25 disputed = 25% dispute rate
        mock_result.fetchall.return_value = [("completed", 75), ("disputed", 25)]
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service._calculate_dispute_rate()

        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0

    @pytest.mark.asyncio
    async def test_calculate_dispute_rate_database_error(self, service, mock_session):
        """Test handling database errors."""
        mock_session.execute.side_effect = Exception("DB Error")
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service._calculate_dispute_rate()
        
        assert result == 0.0


# =============================================================================
# Tests for main service methods
# =============================================================================

class TestTrackUserActivity:
    """Tests for track_user_activity method."""

    @pytest.mark.asyncio
    async def test_track_user_activity_basic(self, service, mock_session):
        """Test basic user activity tracking."""
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service.track_user_activity(
                user_id="user_123",
                activity_type=ActivityType.PURCHASE,
                resource_id="resource_456",
                session_id="session_abc",
                metadata={"value": 99.99}
            )

        assert result is not None

    @pytest.mark.asyncio
    async def test_track_user_activity_with_value(self, service, mock_session):
        """Test tracking activity with monetary value."""
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service.track_user_activity(
                user_id="user_123",
                activity_type=ActivityType.PURCHASE,
                resource_id="resource_456",
                session_id="session_abc",
                metadata={},
                value=299.99
            )

        assert result is not None


class TestCalculateRevenueGrowth:
    """Tests for calculate_revenue_growth method."""

    @pytest.mark.asyncio
    async def test_calculate_revenue_growth_basic(self, service, mock_session):
        """Test basic revenue growth calculation."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 10000.0
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service.calculate_revenue_growth(period=MetricPeriod.MONTHLY)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_calculate_revenue_growth_weekly(self, service, mock_session):
        """Test weekly revenue growth."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 2500.0
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service.calculate_revenue_growth(period=MetricPeriod.WEEKLY)
        
        assert result is not None


class TestGetUserEngagementMetrics:
    """Tests for get_user_engagement_metrics method."""

    @pytest.mark.asyncio
    async def test_get_user_engagement_basic(self, service):
        """Test basic user engagement metrics."""
        # Add some activities to buffer
        service.activity_buffer = [
            UserActivity(
                activity_id=f"activity_{i}",
                user_id=f"user_{i % 3}",
                activity_type=ActivityType.PAGE_VIEW,
                resource_id="resource_1",
                timestamp=datetime.now(timezone.utc),
                session_id=f"session_{i}",
                metadata={}
            )
            for i in range(10)
        ]
        
        result = await service.get_user_engagement_metrics(period=MetricPeriod.DAILY)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_user_engagement_empty_buffer(self, service):
        """Test engagement metrics with empty buffer."""
        service.activity_buffer = []
        
        result = await service.get_user_engagement_metrics(period=MetricPeriod.DAILY)
        
        assert result is not None


class TestGetMarketplaceKPIs:
    """Tests for get_marketplace_kpis method."""

    @pytest.mark.asyncio
    async def test_get_kpis_basic(self, service, mock_session):
        """Test basic KPI retrieval."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 100
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service.get_marketplace_kpis()
        
        assert result is not None


# =============================================================================
# Tests for data classes and enums
# =============================================================================

class TestDataClasses:
    """Tests for data classes."""

    def test_activity_type_enum_values(self):
        """Test ActivityType enum has expected values."""
        assert ActivityType.PAGE_VIEW.value == "page_view"
        assert ActivityType.SEARCH.value == "search"
        assert ActivityType.RESOURCE_VIEW.value == "resource_view"
        assert ActivityType.PURCHASE.value == "purchase"
        assert ActivityType.DOWNLOAD.value == "download"
        assert ActivityType.RATING.value == "rating"
        assert ActivityType.REVIEW.value == "review"
        assert ActivityType.SHARE.value == "share"
        assert ActivityType.BOOKMARK.value == "bookmark"
        assert ActivityType.API_CALL.value == "api_call"

    def test_metric_period_enum_values(self):
        """Test MetricPeriod enum has expected values."""
        assert MetricPeriod.HOURLY.value == "hourly"
        assert MetricPeriod.DAILY.value == "daily"
        assert MetricPeriod.WEEKLY.value == "weekly"
        assert MetricPeriod.MONTHLY.value == "monthly"
        assert MetricPeriod.QUARTERLY.value == "quarterly"
        assert MetricPeriod.YEARLY.value == "yearly"

    def test_user_activity_dataclass(self):
        """Test UserActivity dataclass creation."""
        activity = UserActivity(
            activity_id="test_activity",
            user_id="test_user",
            activity_type=ActivityType.PURCHASE,
            resource_id="resource_123",
            timestamp=datetime.now(timezone.utc),
            session_id="session_1",
            metadata={"key": "value"},
            value=99.99
        )
        
        assert activity.activity_id == "test_activity"
        assert activity.user_id == "test_user"
        assert activity.activity_type == ActivityType.PURCHASE
        assert activity.resource_id == "resource_123"
        assert activity.value == 99.99

    def test_revenue_metric_dataclass(self):
        """Test RevenueMetric dataclass creation."""
        metric = RevenueMetric(
            period="monthly",
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            total_revenue=10000.0,
            transaction_count=100,
            average_transaction_value=100.0,
            growth_rate=0.15,
            new_customers=25,
            returning_customers=75
        )
        
        assert metric.period == "monthly"
        assert metric.total_revenue == 10000.0
        assert metric.growth_rate == 0.15

    def test_user_engagement_metric_dataclass(self):
        """Test UserEngagementMetric dataclass creation."""
        metric = UserEngagementMetric(
            period="weekly",
            active_users=500,
            new_users=100,
            returning_users=400,
            session_count=1500,
            average_session_duration=12.5,
            page_views=5000,
            bounce_rate=0.25,
            conversion_rate=0.05
        )

        assert metric.period == "weekly"
        assert metric.active_users == 500
        assert metric.average_session_duration == 12.5

    def test_marketplace_kpi_dataclass(self):
        """Test MarketplaceKPI dataclass creation."""
        kpi = MarketplaceKPI(
            period="daily",
            timestamp=datetime.now(timezone.utc),
            daily_active_users=200,
            monthly_active_users=1500,
            user_retention_rate=75.0,
            total_revenue=5000.0,
            revenue_growth_rate=10.5,
            average_revenue_per_user=25.0,
            customer_lifetime_value=150.0,
            resource_views=3000,
            resource_downloads=500,
            search_success_rate=80.0,
            recommendation_click_rate=5.0,
            average_rating=4.5,
            review_count=120,
            dispute_rate=2.0
        )

        assert kpi.period == "daily"
        assert kpi.total_revenue == 5000.0
        assert kpi.dispute_rate == 2.0


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_concurrent_activity_tracking(self, service, mock_session):
        """Test handling concurrent activity tracking."""
        import asyncio
        
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            tasks = [
                service.track_user_activity(
                    user_id=f"user_{i}",
                    activity_type=ActivityType.PAGE_VIEW,
                    resource_id="resource_1",
                    session_id=f"session_{i}",
                    metadata={}
                )
                for i in range(10)
            ]
            results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        for result in results:
            assert result is not None

    @pytest.mark.asyncio
    async def test_activity_with_special_metadata(self, service, mock_session):
        """Test activity with special characters in metadata."""
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        
        special_metadata = {
            "query": "NLP & ML <script>alert('xss')</script>",
            "unicode": "日本語 🤖",
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3, "four"]
        }
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service.track_user_activity(
                user_id="user_123",
                activity_type=ActivityType.SEARCH,
                resource_id=None,
                session_id="session_1",
                metadata=special_metadata
            )

        assert result is not None

    def test_service_initialization(self):
        """Test service initializes correctly."""
        service = MarketplaceAnalyticsService()

        assert service.activity_buffer is not None
        # activity_buffer is a deque (list-like)
        assert hasattr(service.activity_buffer, '__iter__')
        assert service.revenue_cache is not None
        assert isinstance(service.revenue_cache, dict)

    @pytest.mark.asyncio
    async def test_long_running_analytics(self, service, mock_session):
        """Test analytics with large activity buffer."""
        # Simulate large activity buffer
        service.activity_buffer = [
            UserActivity(
                activity_id=f"activity_{i}",
                user_id=f"user_{i % 100}",
                activity_type=ActivityType.PAGE_VIEW,
                resource_id=f"resource_{i % 50}",
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                session_id=f"session_{i % 20}",
                metadata={}
            )
            for i in range(1000)
        ]
        
        mock_result = MagicMock()
        mock_result.scalar.return_value = 100
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            result = await service.get_user_engagement_metrics(period=MetricPeriod.DAILY)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_zero_division_protection(self, service, mock_session):
        """Test protection against division by zero."""
        mock_result = MagicMock()
        # Both methods use fetchall() — return empty rows to trigger zero-division guards
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.analytics_service.get_async_session', return_value=mock_session):
            clv = await service._calculate_customer_lifetime_value()
            dispute_rate = await service._calculate_dispute_rate()

        assert clv == 0.0
        assert dispute_rate == 0.0
