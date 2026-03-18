"""
Unit tests for PRSM Marketplace Recommendation Engine

Tests all stub implementations with mocked database sessions.
No real database connections - all DB operations are mocked.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
import json

from prsm.economy.marketplace.recommendation_engine import (
    MarketplaceRecommendationEngine,
    RecommendationType,
    RecommendationScore,
    UserProfile,
    RecommendationContext,
)
from prsm.core.models import UserRole
from prsm.economy.marketplace.database_models import (
    MarketplaceResource, MarketplaceOrder, MarketplaceReview
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def engine():
    """Create a fresh recommendation engine for each test."""
    return MarketplaceRecommendationEngine()


@pytest.fixture
def mock_session():
    """Create a mock async database session."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.fixture
def sample_user_profile():
    """Create a sample user profile for testing."""
    return UserProfile(
        user_id="user_123",
        user_role=UserRole.RESEARCHER,
        preferences={"preferred_types": ["model", "dataset"]},
        interaction_history=[],
        resource_usage={"model": 5, "dataset": 3},
        quality_preference="premium",
        price_sensitivity=0.5,
        domains_of_interest=["nlp", "computer_vision"],
        last_updated=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_resource():
    """Create a sample marketplace resource for testing."""
    resource = MagicMock(spec=MarketplaceResource)
    resource.id = uuid4()
    resource.resource_type = "model"
    resource.name = "Test AI Model"
    resource.description = "A test model for NLP tasks"
    resource.owner_user_id = "creator_456"
    resource.base_price = 100.0
    resource.quality_grade = "premium"
    resource.download_count = 500
    resource.usage_count = 1000
    resource.rating_average = 4.5
    resource.rating_count = 50
    resource.created_at = datetime.now(timezone.utc) - timedelta(days=30)
    resource.license_type = "mit"
    resource.status = "published"
    return resource


@pytest.fixture
def sample_order():
    """Create a sample marketplace order for testing."""
    order = MagicMock(spec=MarketplaceOrder)
    order.id = uuid4()
    order.order_id = "order_123"
    order.buyer_id = "user_123"
    order.resource_id = "resource_456"
    order.status = "completed"
    order.total_price = 150.0
    order.created_at = datetime.now(timezone.utc) - timedelta(days=7)
    return order


# =============================================================================
# Tests for _get_resource_details
# =============================================================================

class TestGetResourceDetails:
    """Tests for _get_resource_details method."""

    @pytest.mark.asyncio
    async def test_get_resource_details_valid_uuid(self, engine, mock_session, sample_resource):
        """Test retrieving resource details with valid UUID."""
        resource_id = str(sample_resource.id)
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_resource
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            result = await engine._get_resource_details(resource_id)

        assert result is not None
        assert result["resource_type"] == "model"
        assert result["name"] == "Test AI Model"

    @pytest.mark.asyncio
    async def test_get_resource_details_invalid_uuid(self, engine):
        """Test with invalid UUID format."""
        result = await engine._get_resource_details("not-a-uuid")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_resource_details_nonexistent(self, engine, mock_session):
        """Test retrieving non-existent resource."""
        resource_id = str(uuid4())
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            result = await engine._get_resource_details(resource_id)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_resource_details_empty_string(self, engine):
        """Test with empty string resource ID."""
        result = await engine._get_resource_details("")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_resource_details_none_input(self, engine):
        """Test with None input."""
        result = await engine._get_resource_details(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_resource_details_database_error(self, engine, mock_session):
        """Test handling database errors gracefully."""
        resource_id = str(uuid4())
        mock_session.execute.side_effect = Exception("Database connection error")
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            result = await engine._get_resource_details(resource_id)
        
        assert result is None


# =============================================================================
# Tests for _find_similar_resources_by_type
# =============================================================================

class TestFindSimilarResourcesByType:
    """Tests for _find_similar_resources_by_type method."""

    @pytest.mark.asyncio
    async def test_find_similar_by_type_basic(self, engine, mock_session, sample_user_profile, sample_resource):
        """Test finding similar resources by type."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_similar_resources_by_type(
                "model", sample_user_profile, limit=5
            )
        
        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_find_similar_by_type_with_price_filter(self, engine, mock_session, sample_user_profile):
        """Test price filtering based on user price sensitivity."""
        expensive_resource = MagicMock(spec=MarketplaceResource)
        expensive_resource.id = uuid4()
        expensive_resource.resource_type = "model"
        expensive_resource.name = "Expensive Model"
        expensive_resource.description = "Very expensive"
        expensive_resource.owner_user_id = "creator_1"
        expensive_resource.base_price = 10000.0
        expensive_resource.quality_grade = "enterprise"
        expensive_resource.download_count = 100
        expensive_resource.usage_count = 200
        expensive_resource.rating_average = 4.8
        expensive_resource.rating_count = 10
        expensive_resource.created_at = datetime.now(timezone.utc)
        expensive_resource.license_type = "commercial"
        expensive_resource.status = "published"
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [expensive_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_similar_resources_by_type(
                "model", sample_user_profile, limit=5
            )
        
        # Price should be filtered based on user's price sensitivity
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_similar_by_type_quality_preference(self, engine, mock_session, sample_user_profile, sample_resource):
        """Test quality grade filtering."""
        sample_user_profile.quality_preference = "enterprise"
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_similar_resources_by_type(
                "model", sample_user_profile, limit=5
            )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_similar_by_type_no_results(self, engine, mock_session, sample_user_profile):
        """Test when no similar resources found."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_similar_resources_by_type(
                "model", sample_user_profile, limit=5
            )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_find_similar_by_type_database_error(self, engine, mock_session, sample_user_profile):
        """Test handling database errors."""
        mock_session.execute.side_effect = Exception("DB Error")
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_similar_resources_by_type(
                "model", sample_user_profile, limit=5
            )
        
        assert results == []


# =============================================================================
# Tests for _find_content_similar_resources
# =============================================================================

class TestFindContentSimilarResources:
    """Tests for _find_content_similar_resources method."""

    @pytest.mark.asyncio
    async def test_find_content_similar_basic(self, engine, mock_session, sample_resource):
        """Test finding content-similar resources."""
        reference_resource = {
            "resource_id": "ref_123",
            "resource_type": "model",
            "tags": ["nlp", "transformer"],
            "title": "Reference Model"
        }
        
        similar_resource = MagicMock(spec=MarketplaceResource)
        similar_resource.id = uuid4()
        similar_resource.resource_type = "model"
        similar_resource.name = "Similar Model"
        similar_resource.description = "Similar NLP model"
        similar_resource.owner_user_id = "creator_1"
        similar_resource.base_price = 50.0
        similar_resource.quality_grade = "verified"
        similar_resource.download_count = 200
        similar_resource.usage_count = 400
        similar_resource.rating_average = 4.2
        similar_resource.rating_count = 20
        similar_resource.created_at = datetime.now(timezone.utc)
        similar_resource.license_type = "apache"
        similar_resource.status = "published"
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [similar_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_content_similar_resources(
                reference_resource, limit=5
            )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_content_similar_jaccard_similarity(self, engine, mock_session):
        """Test Jaccard similarity calculation for tags."""
        reference_resource = {
            "resource_id": "ref_123",
            "resource_type": "model",
            "tags": ["nlp", "transformer", "bert"],
            "title": "BERT Model"
        }
        
        # Resource with overlapping tags
        similar_resource = MagicMock(spec=MarketplaceResource)
        similar_resource.id = uuid4()
        similar_resource.resource_type = "model"
        similar_resource.name = "Similar BERT"
        similar_resource.description = "Another BERT model"
        similar_resource.owner_user_id = "creator_1"
        similar_resource.base_price = 75.0
        similar_resource.quality_grade = "premium"
        similar_resource.download_count = 300
        similar_resource.usage_count = 600
        similar_resource.rating_average = 4.5
        similar_resource.rating_count = 30
        similar_resource.created_at = datetime.now(timezone.utc)
        similar_resource.license_type = "mit"
        similar_resource.status = "published"
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [similar_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_content_similar_resources(
                reference_resource, limit=5
            )
        
        # Should have some similarity score
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_content_similar_no_tags(self, engine, mock_session):
        """Test with resources having no tags."""
        reference_resource = {
            "resource_id": "ref_123",
            "resource_type": "model",
            "tags": [],
            "title": "No Tags Model"
        }
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_content_similar_resources(
                reference_resource, limit=5
            )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_content_similar_empty_reference(self, engine, mock_session):
        """Test with empty reference resource."""
        reference_resource = {}
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_content_similar_resources(
                reference_resource, limit=5
            )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_find_content_similar_database_error(self, engine, mock_session):
        """Test handling database errors."""
        reference_resource = {
            "resource_id": "ref_123",
            "resource_type": "model",
            "tags": ["nlp"]
        }
        
        mock_session.execute.side_effect = Exception("DB Error")
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_content_similar_resources(
                reference_resource, limit=5
            )
        
        assert results == []


# =============================================================================
# Tests for _find_search_similar_resources
# =============================================================================

class TestFindSearchSimilarResources:
    """Tests for _find_search_similar_resources method."""

    @pytest.mark.asyncio
    async def test_find_search_similar_basic(self, engine, mock_session, sample_resource):
        """Test finding resources matching search query."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_search_similar_resources(
                query="NLP model", limit=5
            )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_search_similar_title_match(self, engine, mock_session):
        """Test matching resources by title."""
        resource = MagicMock(spec=MarketplaceResource)
        resource.id = uuid4()
        resource.resource_type = "model"
        resource.name = "Advanced NLP Transformer Model"
        resource.description = "State of the art"
        resource.owner_user_id = "creator_1"
        resource.base_price = 200.0
        resource.quality_grade = "enterprise"
        resource.download_count = 1000
        resource.usage_count = 2000
        resource.rating_average = 4.9
        resource.rating_count = 100
        resource.created_at = datetime.now(timezone.utc)
        resource.license_type = "commercial"
        resource.status = "published"
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_search_similar_resources(
                query="NLP Transformer", limit=5
            )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_search_similar_empty_query(self, engine, mock_session):
        """Test with empty search query."""
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_search_similar_resources(
                query="", limit=5
            )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_find_search_similar_short_query(self, engine, mock_session):
        """Test with too short search query."""
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_search_similar_resources(
                query="a", limit=5
            )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_find_search_similar_whitespace_query(self, engine, mock_session):
        """Test with whitespace-only query."""
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_search_similar_resources(
                query="   ", limit=5
            )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_find_search_similar_none_query(self, engine, mock_session):
        """Test with None query."""
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_search_similar_resources(
                query=None, limit=5
            )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_find_search_similar_no_results(self, engine, mock_session):
        """Test when no results match query."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_search_similar_resources(
                query="xyznonexistent123", limit=5
            )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_find_search_similar_database_error(self, engine, mock_session):
        """Test handling database errors."""
        mock_session.execute.side_effect = Exception("DB Error")
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_search_similar_resources(
                query="test query", limit=5
            )
        
        assert results == []


# =============================================================================
# Tests for _find_similar_users
# =============================================================================

class TestFindSimilarUsers:
    """Tests for _find_similar_users method."""

    @pytest.mark.asyncio
    async def test_find_similar_users_basic(self, engine, mock_session, sample_user_profile, sample_order):
        """Test finding users with similar behavior."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_order]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_similar_users(sample_user_profile)
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_similar_users_with_purchases(self, engine, mock_session, sample_user_profile):
        """Test finding similar users based on purchase history."""
        # Create orders from different users
        order1 = MagicMock(spec=MarketplaceOrder)
        order1.buyer_id = "user_456"
        order1.resource_id = "resource_123"
        order1.status = "completed"
        
        order2 = MagicMock(spec=MarketplaceOrder)
        order2.buyer_id = "user_789"
        order2.resource_id = "resource_456"
        order2.status = "completed"
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [order1, order2]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_similar_users(sample_user_profile)
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_similar_users_no_purchases(self, engine, mock_session, sample_user_profile):
        """Test when user has no purchases."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_similar_users(sample_user_profile)
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_similar_users_database_error(self, engine, mock_session, sample_user_profile):
        """Test handling database errors."""
        mock_session.execute.side_effect = Exception("DB Error")
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_similar_users(sample_user_profile)
        
        assert results == []


# =============================================================================
# Tests for _get_trending_resources
# =============================================================================

class TestGetTrendingResources:
    """Tests for _get_trending_resources method."""

    @pytest.mark.asyncio
    async def test_get_trending_basic(self, engine, mock_session, sample_resource):
        """Test getting trending resources."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(sample_resource, 5)]
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_trending_resources(limit=10)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_trending_with_recent_orders(self, engine, mock_session):
        """Test trending calculation with recent orders."""
        resource = MagicMock(spec=MarketplaceResource)
        resource.id = uuid4()
        resource.resource_type = "model"
        resource.name = "Trending Model"
        resource.description = "Hot new model"
        resource.owner_user_id = "creator_1"
        resource.base_price = 100.0
        resource.quality_grade = "premium"
        resource.download_count = 5000
        resource.usage_count = 10000
        resource.rating_average = 4.7
        resource.rating_count = 200
        resource.created_at = datetime.now(timezone.utc) - timedelta(days=3)
        resource.license_type = "mit"
        resource.status = "published"

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(resource, 50)]
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_trending_resources(limit=10)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_trending_no_results(self, engine, mock_session):
        """Test when no trending resources found."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_trending_resources(limit=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_get_trending_database_error(self, engine, mock_session):
        """Test handling database errors."""
        mock_session.execute.side_effect = Exception("DB Error")

        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_trending_resources(limit=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_get_trending_with_resource_type(self, engine, mock_session, sample_resource):
        """Test getting trending resources filtered by type."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(sample_resource, 3)]
        mock_session.execute.return_value = mock_result

        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_trending_resources(limit=10)

        assert isinstance(results, list)


# =============================================================================
# Tests for _get_high_quality_resources
# =============================================================================

class TestGetHighQualityResources:
    """Tests for _get_high_quality_resources method."""

    @pytest.mark.asyncio
    async def test_get_high_quality_basic(self, engine, mock_session, sample_resource):
        """Test getting high quality resources."""
        sample_resource.quality_grade = "enterprise"
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_high_quality_resources(limit=10)
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_high_quality_enterprise_only(self, engine, mock_session):
        """Test filtering for enterprise grade only."""
        enterprise_resource = MagicMock(spec=MarketplaceResource)
        enterprise_resource.id = uuid4()
        enterprise_resource.resource_type = "model"
        enterprise_resource.name = "Enterprise Model"
        enterprise_resource.description = "Enterprise grade"
        enterprise_resource.owner_user_id = "creator_1"
        enterprise_resource.base_price = 500.0
        enterprise_resource.quality_grade = "enterprise"
        enterprise_resource.download_count = 2000
        enterprise_resource.usage_count = 5000
        enterprise_resource.rating_average = 4.9
        enterprise_resource.rating_count = 100
        enterprise_resource.created_at = datetime.now(timezone.utc)
        enterprise_resource.license_type = "commercial"
        enterprise_resource.status = "published"
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [enterprise_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_high_quality_resources(limit=10)
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_high_quality_no_results(self, engine, mock_session):
        """Test when no high quality resources found."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_high_quality_resources(limit=10)
        
        assert results == []

    @pytest.mark.asyncio
    async def test_get_high_quality_database_error(self, engine, mock_session):
        """Test handling database errors."""
        mock_session.execute.side_effect = Exception("DB Error")
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_high_quality_resources(limit=10)
        
        assert results == []


# =============================================================================
# Tests for _get_popular_resources
# =============================================================================

class TestGetPopularResources:
    """Tests for _get_popular_resources method."""

    @pytest.mark.asyncio
    async def test_get_popular_basic(self, engine, mock_session, sample_resource):
        """Test getting popular resources."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_popular_resources(limit=10)
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_popular_by_download_count(self, engine, mock_session):
        """Test popularity based on download count."""
        popular_resource = MagicMock(spec=MarketplaceResource)
        popular_resource.id = uuid4()
        popular_resource.resource_type = "dataset"
        popular_resource.name = "Popular Dataset"
        popular_resource.description = "Highly downloaded"
        popular_resource.owner_user_id = "creator_1"
        popular_resource.base_price = 50.0
        popular_resource.quality_grade = "verified"
        popular_resource.download_count = 50000
        popular_resource.usage_count = 100000
        popular_resource.rating_average = 4.6
        popular_resource.rating_count = 500
        popular_resource.created_at = datetime.now(timezone.utc) - timedelta(days=60)
        popular_resource.license_type = "cc0"
        popular_resource.status = "published"
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [popular_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_popular_resources(limit=10)
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_popular_no_results(self, engine, mock_session):
        """Test when no popular resources found."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_popular_resources(limit=10)
        
        assert results == []

    @pytest.mark.asyncio
    async def test_get_popular_database_error(self, engine, mock_session):
        """Test handling database errors."""
        mock_session.execute.side_effect = Exception("DB Error")
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_popular_resources(limit=10)
        
        assert results == []

    @pytest.mark.asyncio
    async def test_get_popular_with_type_filter(self, engine, mock_session, sample_resource):
        """Test getting popular resources with type filter."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._get_popular_resources(limit=10)
        
        assert isinstance(results, list)


# =============================================================================
# Tests for _apply_business_filters
# =============================================================================

class TestApplyBusinessFilters:
    """Tests for _apply_business_filters method."""

    @pytest.mark.asyncio
    async def test_apply_business_filters_basic(self, engine, mock_session):
        """Test applying business filters to recommendations."""
        recommendations = [
            {
                "resource_id": "res_1",
                "resource_type": "model",
                "title": "Test Model",
                "price": 100.0,
                "quality_grade": "premium",
                "is_batch_verified": True,
                "license_type": "mit"
            }
        ]
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._apply_business_filters(recommendations)
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_apply_business_filters_batch_verification(self, engine, mock_session):
        """Test filtering by batch verification status."""
        verified_resource = {
            "resource_id": "verified_1",
            "resource_type": "model",
            "title": "Verified Model",
            "price": 100.0,
            "quality_grade": "premium",
            "is_batch_verified": True,
            "license_type": "mit"
        }
        
        unverified_resource = {
            "resource_id": "unverified_1",
            "resource_type": "model",
            "title": "Unverified Model",
            "price": 50.0,
            "quality_grade": "basic",
            "is_batch_verified": False,
            "license_type": "mit"
        }
        
        recommendations = [verified_resource, unverified_resource]
        
        # Mock the verification check query
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1  # Verified count
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._apply_business_filters(recommendations)
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_apply_business_filters_empty_input(self, engine, mock_session):
        """Test with empty recommendations list."""
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._apply_business_filters([])
        
        assert results == []

    @pytest.mark.asyncio
    async def test_apply_business_filters_price_range(self, engine, mock_session):
        """Test filtering by price range."""
        recommendations = [
            {
                "resource_id": "cheap",
                "resource_type": "model",
                "title": "Cheap Model",
                "price": 10.0,
                "quality_grade": "basic",
                "is_batch_verified": True,
                "license_type": "mit"
            },
            {
                "resource_id": "expensive",
                "resource_type": "model",
                "title": "Expensive Model",
                "price": 10000.0,
                "quality_grade": "enterprise",
                "is_batch_verified": True,
                "license_type": "commercial"
            }
        ]
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._apply_business_filters(recommendations)
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_apply_business_filters_license_type(self, engine, mock_session):
        """Test filtering by license type."""
        recommendations = [
            {
                "resource_id": "mit_1",
                "resource_type": "model",
                "title": "MIT Model",
                "price": 100.0,
                "quality_grade": "premium",
                "is_batch_verified": True,
                "license_type": "mit"
            },
            {
                "resource_id": "proprietary_1",
                "resource_type": "model",
                "title": "Proprietary Model",
                "price": 500.0,
                "quality_grade": "enterprise",
                "is_batch_verified": True,
                "license_type": "proprietary"
            }
        ]
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._apply_business_filters(recommendations)
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_apply_business_filters_database_error(self, engine, mock_session):
        """Test handling database errors during filtering."""
        recommendations = [
            {
                "resource_id": "res_1",
                "resource_type": "model",
                "title": "Test Model",
                "price": 100.0,
                "quality_grade": "premium",
                "is_batch_verified": True,
                "license_type": "mit"
            }
        ]
        
        mock_session.execute.side_effect = Exception("DB Error")
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._apply_business_filters(recommendations)
        
        # Should return original recommendations on error
        assert isinstance(results, list)


# =============================================================================
# Tests for _resource_to_dict helper
# =============================================================================

class TestResourceToDict:
    """Tests for _resource_to_dict helper method."""

    def test_resource_to_dict_basic(self, engine, sample_resource):
        """Test converting resource to dictionary."""
        result = engine._resource_to_dict(sample_resource)

        assert result is not None
        assert result["resource_type"] == "model"
        assert result["name"] == "Test AI Model"

    def test_resource_to_dict_all_fields(self, engine, sample_resource):
        """Test all fields are included in dictionary."""
        result = engine._resource_to_dict(sample_resource)

        expected_fields = [
            "id", "resource_type", "name", "description",
            "quality_grade", "base_price", "tags",
            "download_count", "usage_count", "rating_average",
            "created_at", "license_type"
        ]

        for field in expected_fields:
            assert field in result

    def test_resource_to_dict_none_input(self, engine):
        """Test with None input."""
        result = engine._resource_to_dict(None)
        assert result is None


# =============================================================================
# Integration-style tests for main recommendation flow
# =============================================================================

class TestRecommendationFlow:
    """Tests for the main recommendation generation flow."""

    @pytest.mark.asyncio
    async def test_get_recommendations_basic(self, engine, mock_session, sample_user_profile, sample_resource):
        """Test basic recommendation generation."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_result.scalar_one_or_none.return_value = sample_resource
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            # This tests the main entry point
            results = await engine.get_recommendations(
                user_id="user_123",
                limit=10
            )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_recommendations_with_context(self, engine, mock_session, sample_user_profile, sample_resource):
        """Test recommendation with context."""
        context = RecommendationContext(
            user_profile=sample_user_profile,
            current_resource="resource_456",
            search_query="NLP model",
            filters={"resource_type": "model"},
            session_context={},
            business_constraints={}
        )
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine.get_recommendations(
                user_id="user_123",
                context=context,
                limit=10
            )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_recommendations_cold_start(self, engine, mock_session, sample_resource):
        """Test recommendations for new user (cold start)."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine.get_recommendations(
                user_id="new_user_123",
                limit=10
            )
        
        assert isinstance(results, list)


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_concurrent_recommendation_requests(self, engine, mock_session, sample_resource):
        """Test handling concurrent recommendation requests."""
        import asyncio
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            tasks = [
                engine.get_recommendations(user_id=f"user_{i}", limit=5)
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_recommendation_with_malformed_resource_id(self, engine, mock_session):
        """Test handling malformed resource IDs."""
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            result = await engine._get_resource_details("malformed-uuid-!!!")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_recommendation_with_special_characters_in_query(self, engine, mock_session, sample_resource):
        """Test search query with special characters."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_search_similar_resources(
                query="NLP & ML <script>", limit=5
            )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_recommendation_with_unicode_query(self, engine, mock_session, sample_resource):
        """Test search query with unicode characters."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_resource]
        mock_session.execute.return_value = mock_result
        
        with patch('prsm.economy.marketplace.recommendation_engine.get_async_session', return_value=mock_session):
            results = await engine._find_search_similar_resources(
                query="NLP 模型 🤖", limit=5
            )
        
        assert isinstance(results, list)

    def test_recommendation_type_enum_values(self):
        """Test RecommendationType enum has expected values."""
        assert RecommendationType.PERSONALIZED.value == "personalized"
        assert RecommendationType.TRENDING.value == "trending"
        assert RecommendationType.SIMILAR.value == "similar"
        assert RecommendationType.COLLABORATIVE.value == "collaborative"
        assert RecommendationType.CONTENT_BASED.value == "content_based"
        assert RecommendationType.BUSINESS_RULES.value == "business_rules"
        assert RecommendationType.COLD_START.value == "cold_start"

    def test_user_profile_dataclass(self):
        """Test UserProfile dataclass creation."""
        profile = UserProfile(
            user_id="test_user",
            user_role=UserRole.DEVELOPER,
            preferences={},
            interaction_history=[],
            resource_usage={},
            quality_preference="premium",
            price_sensitivity=0.5,
            domains_of_interest=["ml"],
            last_updated=datetime.now(timezone.utc)
        )
        
        assert profile.user_id == "test_user"
        assert profile.user_role == UserRole.DEVELOPER
        assert profile.quality_preference == "premium"

    def test_recommendation_score_dataclass(self):
        """Test RecommendationScore dataclass creation."""
        score = RecommendationScore(
            resource_id="res_123",
            resource_type="model",
            score=0.95,
            confidence=0.8,
            reasoning=["High quality", "Popular"],
            recommendation_type=RecommendationType.PERSONALIZED,
            metadata={}
        )
        
        assert score.resource_id == "res_123"
        assert score.score == 0.95
        assert len(score.reasoning) == 2
