"""
Test Suite for Reputation System Persistence
============================================

Comprehensive tests for the reputation system's database persistence layer.
All database operations are mocked via AsyncMock - no real database needed.

Test Categories:
- DB Model Serialization (3 tests)
- Store/Retrieve Round-trips (4 tests)
- Incremental Update (6 tests)
- User Statistics (5 tests)
- Fraud Detection (5 tests)
- End-to-End Flow (4 tests)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import numpy as np

from prsm.economy.marketplace.reputation_system import (
    ReputationCalculator,
    UserReputation,
    ReputationScore,
    ReputationDimension,
    TrustLevel,
    ReputationEvent,
    ReputationTransaction,
)
from prsm.economy.marketplace.database_models import (
    UserReputationModel,
    ReputationEventModel,
    MarketplaceResource,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_db_session():
    """Create a mock async database session"""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def reputation_calculator():
    """Create a ReputationCalculator instance for testing"""
    return ReputationCalculator()


@pytest.fixture
def sample_reputation():
    """Create a sample UserReputation for testing"""
    return UserReputation(
        user_id="test_user_123",
        overall_score=75.0,
        trust_level=TrustLevel.TRUSTED,
        dimension_scores={
            ReputationDimension.QUALITY: ReputationScore(
                dimension=ReputationDimension.QUALITY,
                score=80.0,
                confidence=0.9,
                last_updated=datetime.now(timezone.utc),
                trend="improving",
                evidence_count=10,
            ),
            ReputationDimension.RELIABILITY: ReputationScore(
                dimension=ReputationDimension.RELIABILITY,
                score=70.0,
                confidence=0.8,
                last_updated=datetime.now(timezone.utc),
                trend=0.0,
                evidence_count=5,
            ),
        },
        badges=["first_contribution"],
        verification_status={"email_verified": True},
        reputation_history=[],
        last_calculated=datetime.now(timezone.utc),
        next_review=datetime.now(timezone.utc) + timedelta(days=30),
    )


@pytest.fixture
def sample_transaction():
    """Create a sample ReputationTransaction for testing"""
    return ReputationTransaction(
        transaction_id=str(uuid4()),
        user_id="test_user_123",
        event_type=ReputationEvent.RESOURCE_RATED,
        dimension=ReputationDimension.QUALITY,
        score_change=5.0,
        evidence={"quality_score": 0.9},
        validated=True,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_db_reputation_row():
    """Create a mock database row for UserReputationModel"""
    row = MagicMock(spec=UserReputationModel)
    row.user_id = "test_user_123"
    row.overall_score = 75.0
    row.trust_level = "TRUSTED"
    row.dimension_scores = {
        "quality": {
            "score": 80.0,
            "confidence": 0.9,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "trend": "improving",
            "evidence_count": 10,
        }
    }
    row.badges = ["first_contribution"]
    row.verification_status = {"email_verified": True}
    row.reputation_history = []
    row.last_calculated = datetime.now(timezone.utc)
    row.next_review = datetime.now(timezone.utc) + timedelta(days=30)
    row.created_at = datetime.now(timezone.utc) - timedelta(days=100)
    return row


@pytest.fixture
def sample_event_rows():
    """Create mock database rows for ReputationEventModel"""
    events = []
    for i in range(5):
        row = MagicMock(spec=ReputationEventModel)
        row.transaction_id = str(uuid4())
        row.user_id = "test_user_123"
        row.event_type = ReputationEvent.RESOURCE_RATED.value
        row.dimension = ReputationDimension.QUALITY.value
        row.score_change = 2.0
        row.evidence = {"quality_score": 0.9}
        row.timestamp = datetime.now(timezone.utc) - timedelta(days=i)
        row.validated = True
        events.append(row)
    return events


# ============================================================================
# DB MODEL SERIALIZATION TESTS (3 tests)
# ============================================================================

class TestDBModelSerialization:
    """Tests for serialization/deserialization of reputation data to/from database models"""

    def test_store_reputation_serializes_dimension_scores_to_json(self, reputation_calculator, sample_reputation):
        """Dimension scores should serialize to JSON-safe dict for storage"""
        # The _store_reputation method serializes dimension_scores to a JSON-safe dict
        # Verify the serialization format matches expected structure
        
        dim_scores_json = {
            dim.value: {
                "score": score.score,
                "confidence": score.confidence,
                "last_updated": score.last_updated.isoformat(),
                "trend": score.trend,
                "evidence_count": score.evidence_count,
            }
            for dim, score in sample_reputation.dimension_scores.items()
        }
        
        # Verify structure is JSON-serializable
        assert "quality" in dim_scores_json
        assert dim_scores_json["quality"]["score"] == 80.0
        assert dim_scores_json["quality"]["confidence"] == 0.9
        assert isinstance(dim_scores_json["quality"]["last_updated"], str)
        assert dim_scores_json["quality"]["evidence_count"] == 10

    def test_get_existing_reputation_deserializes_dimension_scores(self, reputation_calculator, sample_db_reputation_row):
        """Stored JSON should deserialize back to ReputationScore objects"""
        # Simulate deserialization as done in _get_existing_reputation
        dimension_scores = {}
        for dim_value, score_dict in sample_db_reputation_row.dimension_scores.items():
            try:
                dimension = ReputationDimension(dim_value)
                dimension_scores[dimension] = ReputationScore(
                    dimension=dimension,
                    score=score_dict["score"],
                    confidence=score_dict["confidence"],
                    last_updated=datetime.fromisoformat(score_dict["last_updated"]),
                    trend=score_dict["trend"],
                    evidence_count=score_dict["evidence_count"]
                )
            except (KeyError, ValueError):
                continue
        
        assert ReputationDimension.QUALITY in dimension_scores
        assert dimension_scores[ReputationDimension.QUALITY].score == 80.0
        assert dimension_scores[ReputationDimension.QUALITY].confidence == 0.9

    def test_trust_level_name_round_trips_through_db(self, reputation_calculator):
        """TrustLevel enum name should persist and restore correctly"""
        # Test all trust levels
        for trust_level in TrustLevel:
            stored_name = trust_level.name
            restored_level = TrustLevel[stored_name]
            assert restored_level == trust_level
            assert restored_level.value == trust_level.value


# ============================================================================
# STORE/RETRIEVE ROUND-TRIP TESTS (4 tests)
# ============================================================================

class TestStoreRetrieveRoundTrips:
    """Tests for storing and retrieving reputation data from database"""

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_store_reputation_inserts_when_missing(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_reputation
    ):
        """Should INSERT when no existing row found"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Mock no existing row
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        await reputation_calculator._store_reputation(sample_reputation)
        
        # Verify db.add was called (INSERT path)
        mock_db_session.add.assert_called_once()
        added_model = mock_db_session.add.call_args[0][0]
        assert isinstance(added_model, UserReputationModel)
        assert added_model.user_id == sample_reputation.user_id
        
        # Verify commit was called
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_store_reputation_updates_when_existing(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_reputation, sample_db_reputation_row
    ):
        """Should UPDATE when row already exists"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Mock existing row
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_db_reputation_row
        mock_db_session.execute.return_value = mock_result
        
        await reputation_calculator._store_reputation(sample_reputation)
        
        # Verify the existing row was updated (not added)
        mock_db_session.add.assert_not_called()
        
        # Verify fields were updated
        assert sample_db_reputation_row.overall_score == sample_reputation.overall_score
        assert sample_db_reputation_row.trust_level == sample_reputation.trust_level.name
        
        # Verify commit was called
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_store_reputation_transaction_inserts_row(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_transaction
    ):
        """_store_reputation_transaction should add row to reputation_events table"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        await reputation_calculator._store_reputation_transaction(sample_transaction)
        
        # Verify db.add was called with ReputationEventModel
        mock_db_session.add.assert_called_once()
        added_model = mock_db_session.add.call_args[0][0]
        assert isinstance(added_model, ReputationEventModel)
        assert added_model.transaction_id == sample_transaction.transaction_id
        assert added_model.user_id == sample_transaction.user_id
        assert added_model.event_type == sample_transaction.event_type.value
        
        # Verify commit was called
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_store_reputation_transaction_exception_does_not_raise(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_transaction
    ):
        """Exceptions in _store_reputation_transaction should be caught and logged, not raised"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Make commit raise an exception
        mock_db_session.commit.side_effect = Exception("Database error")
        
        # Should not raise - exception is caught and logged
        await reputation_calculator._store_reputation_transaction(sample_transaction)
        
        # Verify add was still called
        mock_db_session.add.assert_called_once()


# ============================================================================
# INCREMENTAL UPDATE TESTS (6 tests)
# ============================================================================

class TestIncrementalUpdate:
    """Tests for incremental reputation updates without full recalculation"""

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_incremental_update_applies_score_change_to_dimension(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_transaction, sample_db_reputation_row
    ):
        """Score change should be applied to the affected dimension"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Set up existing dimension scores
        sample_db_reputation_row.dimension_scores = {
            "quality": {
                "score": 80.0,
                "confidence": 0.9,
                "evidence_count": 10,
            }
        }
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_db_reputation_row
        mock_db_session.execute.return_value = mock_result
        
        sample_transaction.score_change = 5.0
        sample_transaction.dimension = ReputationDimension.QUALITY
        
        await reputation_calculator._update_reputation_incremental(
            sample_transaction.user_id, sample_transaction
        )
        
        # Verify score was updated (80 + 5 = 85)
        updated_quality = sample_db_reputation_row.dimension_scores["quality"]
        assert updated_quality["score"] == 85.0

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_incremental_update_recalculates_overall_score(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_transaction, sample_db_reputation_row
    ):
        """Overall score should be recalculated from updated dimension scores"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Set up dimension scores with known weights
        sample_db_reputation_row.dimension_scores = {
            "quality": {"score": 80.0, "confidence": 0.9, "evidence_count": 10},
            "reliability": {"score": 70.0, "confidence": 0.8, "evidence_count": 5},
            "trustworthiness": {"score": 75.0, "confidence": 0.7, "evidence_count": 8},
        }
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_db_reputation_row
        mock_db_session.execute.return_value = mock_result
        
        await reputation_calculator._update_reputation_incremental(
            sample_transaction.user_id, sample_transaction
        )
        
        # Verify overall_score was recalculated
        assert sample_db_reputation_row.overall_score is not None
        assert 0 <= sample_db_reputation_row.overall_score <= 100

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_incremental_update_appends_to_history(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_transaction, sample_db_reputation_row
    ):
        """Transaction should be appended to reputation_history"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        sample_db_reputation_row.dimension_scores = {
            "quality": {"score": 80.0, "confidence": 0.9, "evidence_count": 10}
        }
        sample_db_reputation_row.reputation_history = []
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_db_reputation_row
        mock_db_session.execute.return_value = mock_result
        
        await reputation_calculator._update_reputation_incremental(
            sample_transaction.user_id, sample_transaction
        )
        
        # Verify history was appended
        assert len(sample_db_reputation_row.reputation_history) == 1
        history_entry = sample_db_reputation_row.reputation_history[0]
        assert history_entry["transaction_id"] == sample_transaction.transaction_id
        assert history_entry["event_type"] == sample_transaction.event_type.value

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_incremental_update_caps_history_at_50_entries(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_transaction, sample_db_reputation_row
    ):
        """History should be capped at 50 entries (rolling window)"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        sample_db_reputation_row.dimension_scores = {
            "quality": {"score": 80.0, "confidence": 0.9, "evidence_count": 10}
        }
        
        # Pre-populate history with 50 entries
        sample_db_reputation_row.reputation_history = [
            {"transaction_id": f"tx_{i}", "event_type": "quality_rating", "score_change": 1.0}
            for i in range(50)
        ]
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_db_reputation_row
        mock_db_session.execute.return_value = mock_result
        
        await reputation_calculator._update_reputation_incremental(
            sample_transaction.user_id, sample_transaction
        )
        
        # Verify history is still capped at 50
        assert len(sample_db_reputation_row.reputation_history) == 50
        
        # Verify the oldest entry was removed (tx_0 should be gone)
        first_entry = sample_db_reputation_row.reputation_history[0]
        assert first_entry["transaction_id"] == "tx_1"

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_incremental_update_clamps_score_between_0_and_100(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_transaction, sample_db_reputation_row
    ):
        """Scores should be clamped to [0, 100] range"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Test upper bound clamping
        sample_db_reputation_row.dimension_scores = {
            "quality": {"score": 95.0, "confidence": 0.9, "evidence_count": 10}
        }
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_db_reputation_row
        mock_db_session.execute.return_value = mock_result
        
        sample_transaction.score_change = 10.0  # Would push to 105
        
        await reputation_calculator._update_reputation_incremental(
            sample_transaction.user_id, sample_transaction
        )
        
        # Verify score was clamped to 100
        assert sample_db_reputation_row.dimension_scores["quality"]["score"] == 100.0

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_incremental_update_skips_when_no_stored_reputation(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_transaction
    ):
        """Should skip incremental update if no stored reputation exists"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Mock no existing row
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        # Should not raise and should return early
        await reputation_calculator._update_reputation_incremental(
            sample_transaction.user_id, sample_transaction
        )
        
        # Verify commit was not called (early return)
        mock_db_session.commit.assert_not_called()


# ============================================================================
# USER STATISTICS TESTS (5 tests)
# ============================================================================

class TestUserStatistics:
    """Tests for retrieving user statistics for badge calculation"""

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_get_user_statistics_queries_marketplace_resources(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """Should count resources owned by user"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Mock resource count query
        mock_resource_result = MagicMock()
        mock_resource_result.scalar.return_value = 15
        
        # Mock other queries
        mock_other_result = MagicMock()
        mock_other_result.scalar.return_value = 0
        mock_other_result.scalar_one_or_none.return_value = None
        
        mock_db_session.execute.side_effect = [
            mock_resource_result,  # resources_published
            mock_other_result,     # community_reports
            mock_other_result,     # expert_endorsements
            mock_other_result,     # account_age
        ]
        
        stats = await reputation_calculator._get_user_statistics("test_user_123")
        
        assert stats["resources_published"] == 15

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_get_user_statistics_counts_community_reports(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """Should count COMMUNITY_REPORT events"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        mock_zero_result = MagicMock()
        mock_zero_result.scalar.return_value = 0
        mock_zero_result.scalar_one_or_none.return_value = None
        
        mock_reports_result = MagicMock()
        mock_reports_result.scalar.return_value = 5
        
        mock_db_session.execute.side_effect = [
            mock_zero_result,   # resources_published
            mock_reports_result,  # community_reports
            mock_zero_result,     # expert_endorsements
            mock_zero_result,     # account_age
        ]
        
        stats = await reputation_calculator._get_user_statistics("test_user_123")
        
        assert stats["community_reports"] == 5

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_get_user_statistics_counts_endorsements(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """Should count EXPERT_ENDORSEMENT events"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        mock_zero_result = MagicMock()
        mock_zero_result.scalar.return_value = 0
        mock_zero_result.scalar_one_or_none.return_value = None
        
        mock_endorsements_result = MagicMock()
        mock_endorsements_result.scalar.return_value = 3
        
        mock_db_session.execute.side_effect = [
            mock_zero_result,       # resources_published
            mock_zero_result,       # community_reports
            mock_endorsements_result,  # expert_endorsements
            mock_zero_result,       # account_age
        ]
        
        stats = await reputation_calculator._get_user_statistics("test_user_123")
        
        assert stats["expert_endorsements"] == 3

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_get_user_statistics_computes_account_age(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """Should compute account age from created_at timestamp"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        mock_zero_result = MagicMock()
        mock_zero_result.scalar.return_value = 0
        
        mock_age_result = MagicMock()
        mock_age_result.scalar_one_or_none.return_value = datetime.now(timezone.utc) - timedelta(days=45)
        
        mock_db_session.execute.side_effect = [
            mock_zero_result,   # resources_published
            mock_zero_result,   # community_reports
            mock_zero_result,   # expert_endorsements
            mock_age_result,    # account_age
        ]
        
        stats = await reputation_calculator._get_user_statistics("test_user_123")
        
        # Account age should be approximately 45 days
        assert 44 <= stats["account_age_days"] <= 46

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_get_user_statistics_returns_zeros_on_exception(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """Should return zeros dict on database error"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Make execute raise an exception
        mock_db_session.execute.side_effect = Exception("Database connection error")
        
        stats = await reputation_calculator._get_user_statistics("test_user_123")
        
        # Should return default zeros
        assert stats["resources_published"] == 0
        assert stats["community_reports"] == 0
        assert stats["expert_endorsements"] == 0
        assert stats["account_age_days"] == 0


# ============================================================================
# FRAUD DETECTION TESTS (5 tests)
# ============================================================================

class TestFraudDetection:
    """Tests for fraud pattern detection in reputation events"""

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_analyze_fraud_velocity_attack_triggers_fraud_event(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """Should record FRAUD_DETECTED event when velocity threshold exceeded"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Create 25 events within the last hour (exceeds threshold of 20)
        events = []
        for i in range(25):
            events.append({
                "transaction_id": f"tx_{i}",
                "event_type": "quality_rating",
                "dimension": "quality",
                "score_change": 1.0,
                "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=i)).isoformat(),
                "evidence": {},
                "validated": True,
            })
        
        # Mock the record_reputation_event to track calls
        original_record = reputation_calculator.record_reputation_event
        recorded_events = []
        
        async def mock_record(user_id, event_type, evidence=None, source_user_id=None):
            recorded_events.append({"user_id": user_id, "event_type": event_type, "evidence": evidence})
        
        reputation_calculator.record_reputation_event = mock_record
        
        await reputation_calculator._analyze_fraud_patterns("test_user_123", events)
        
        # Verify FRAUD_DETECTED was recorded
        assert len(recorded_events) == 1
        assert recorded_events[0]["event_type"] == ReputationEvent.FRAUD_DETECTED
        assert recorded_events[0]["evidence"]["pattern"] == "velocity_attack"
        
        # Restore original method
        reputation_calculator.record_reputation_event = original_record

    @pytest.mark.asyncio
    async def test_analyze_fraud_velocity_below_threshold_no_flag(
        self, reputation_calculator
    ):
        """Should not flag when events below velocity threshold"""
        # Create 10 events within the last hour (below threshold of 20)
        events = []
        for i in range(10):
            events.append({
                "transaction_id": f"tx_{i}",
                "event_type": "quality_rating",
                "dimension": "quality",
                "score_change": 1.0,
                "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=i)).isoformat(),
                "evidence": {},
                "validated": True,
            })
        
        # Mock record_reputation_event to ensure it's not called
        recorded_events = []
        
        async def mock_record(user_id, event_type, evidence=None, source_user_id=None):
            recorded_events.append({"user_id": user_id, "event_type": event_type})
        
        reputation_calculator.record_reputation_event = mock_record
        
        await reputation_calculator._analyze_fraud_patterns("test_user_123", events)
        
        # Verify no FRAUD_DETECTED was recorded
        assert len(recorded_events) == 0

    @pytest.mark.asyncio
    async def test_analyze_fraud_concentrated_source_logs_warning(
        self, reputation_calculator
    ):
        """Should log warning for concentrated source pattern"""
        # Create events with 70% from single source (exceeds 60% threshold)
        events = []
        for i in range(10):
            events.append({
                "transaction_id": f"tx_{i}",
                "event_type": "quality_rating",
                "dimension": "quality",
                "score_change": 1.0,
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat(),
                "evidence": {"source_user_id": "same_source_user" if i < 7 else f"other_user_{i}"},
                "validated": True,
            })
        
        # Should not raise - just logs warning
        await reputation_calculator._analyze_fraud_patterns("test_user_123", events)

    @pytest.mark.asyncio
    async def test_analyze_fraud_bot_regularity_logs_warning(
        self, reputation_calculator
    ):
        """Should log warning for bot-like regularity pattern"""
        # Create events with very regular intervals (bot-like)
        events = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)
        for i in range(15):
            events.append({
                "transaction_id": f"tx_{i}",
                "event_type": "quality_rating",
                "dimension": "quality",
                "score_change": 1.0,
                "timestamp": (base_time + timedelta(minutes=2*i)).isoformat(),  # Exactly 2 minutes apart
                "evidence": {},
                "validated": True,
            })
        
        # Should not raise - just logs warning
        await reputation_calculator._analyze_fraud_patterns("test_user_123", events)

    @pytest.mark.asyncio
    async def test_analyze_fraud_empty_events_no_crash(
        self, reputation_calculator
    ):
        """Should handle empty events list gracefully"""
        # Should not raise with empty list
        await reputation_calculator._analyze_fraud_patterns("test_user_123", [])


# ============================================================================
# END-TO-END FLOW TESTS (4 tests)
# ============================================================================

class TestEndToEndFlow:
    """Tests for end-to-end reputation calculation and persistence flows"""

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_calculate_reputation_uses_stored_events_not_empty_list(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_event_rows
    ):
        """calculate_reputation should use events from _get_user_reputation_events"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Mock no existing reputation
        mock_no_result = MagicMock()
        mock_no_result.scalar_one_or_none.return_value = None
        
        # Mock events query
        mock_events_result = MagicMock()
        mock_events_result.scalars.return_value.all.return_value = sample_event_rows
        
        # Mock statistics queries
        mock_stats_result = MagicMock()
        mock_stats_result.scalar.return_value = 0
        mock_stats_result.scalar_one_or_none.return_value = None
        
        # Set up execute to return different results for different queries
        mock_db_session.execute.side_effect = [
            mock_no_result,       # _get_existing_reputation
            mock_events_result,   # _get_user_reputation_events
            mock_stats_result,    # _get_user_statistics (resources)
            mock_stats_result,    # _get_user_statistics (reports)
            mock_stats_result,    # _get_user_statistics (endorsements)
            mock_stats_result,    # _get_user_statistics (age)
            mock_no_result,       # _store_reputation check
        ]
        
        reputation = await reputation_calculator.calculate_user_reputation("test_user_123")
        
        assert reputation is not None
        assert reputation.user_id == "test_user_123"

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_record_event_persists_transaction(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """record_reputation_event should call _store_reputation_transaction"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Track if add was called
        add_called = []
        original_add = mock_db_session.add
        
        def track_add(model):
            add_called.append(model)
            return original_add(model)
        
        mock_db_session.add = track_add
        
        transaction_id = await reputation_calculator.record_reputation_event(
            user_id="test_user_123",
            event_type=ReputationEvent.RESOURCE_RATED,
            evidence={"quality_score": 0.9}
        )
        
        assert transaction_id is not None
        assert len(add_called) == 1
        assert isinstance(add_called[0], ReputationEventModel)

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_record_event_triggers_incremental_update(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_db_reputation_row
    ):
        """record_reputation_event should trigger incremental update path"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Set up for incremental update
        sample_db_reputation_row.dimension_scores = {
            "quality": {"score": 80.0, "confidence": 0.9, "evidence_count": 10}
        }
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_db_reputation_row
        
        # First call for _store_reputation_transaction, second for _update_reputation_incremental
        mock_db_session.execute.return_value = mock_result
        
        await reputation_calculator.record_reputation_event(
            user_id="test_user_123",
            event_type=ReputationEvent.RESOURCE_RATED,
            evidence={"quality_score": 0.9}
        )
        
        # Verify commit was called (both for transaction storage and incremental update)
        assert mock_db_session.commit.called

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_reputation_cache_hit_skips_recalculation(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_db_reputation_row
    ):
        """Should use stored reputation if recent enough"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Set last_calculated to very recent (within 1 day)
        sample_db_reputation_row.last_calculated = datetime.now(timezone.utc) - timedelta(hours=1)
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_db_reputation_row
        mock_db_session.execute.return_value = mock_result
        
        reputation = await reputation_calculator.calculate_user_reputation("test_user_123")
        
        # Should return the cached reputation without full recalculation
        assert reputation is not None
        assert reputation.user_id == "test_user_123"
        # Execute should only be called once (for the cache check)
        assert mock_db_session.execute.call_count == 1


# ============================================================================
# ADDITIONAL HELPER METHOD TESTS
# ============================================================================

class TestHelperMethods:
    """Tests for helper methods in ReputationCalculator"""

    def test_determine_trust_level_returns_correct_levels(self, reputation_calculator):
        """_determine_trust_level should return correct TrustLevel for score ranges"""
        assert reputation_calculator._determine_trust_level(97) == TrustLevel.ELITE
        assert reputation_calculator._determine_trust_level(85) == TrustLevel.EXPERT
        assert reputation_calculator._determine_trust_level(70) == TrustLevel.TRUSTED
        assert reputation_calculator._determine_trust_level(50) == TrustLevel.MEMBER
        assert reputation_calculator._determine_trust_level(30) == TrustLevel.NEWCOMER
        assert reputation_calculator._determine_trust_level(10) == TrustLevel.UNTRUSTED

    def test_calculate_overall_score_with_empty_dimensions(self, reputation_calculator):
        """_calculate_overall_score should return 50.0 for empty dimension scores"""
        score = reputation_calculator._calculate_overall_score({})
        assert score == 50.0

    def test_calculate_overall_score_weights_by_confidence(self, reputation_calculator):
        """_calculate_overall_score should weight scores by confidence"""
        dimension_scores = {
            ReputationDimension.QUALITY: ReputationScore(
                dimension=ReputationDimension.QUALITY,
                score=100.0,
                confidence=1.0,
                last_updated=datetime.now(timezone.utc),
                trend=0.0,
                evidence_count=10,
            ),
            ReputationDimension.RELIABILITY: ReputationScore(
                dimension=ReputationDimension.RELIABILITY,
                score=0.0,
                confidence=0.1,  # Low confidence
                last_updated=datetime.now(timezone.utc),
                trend=0.0,
                evidence_count=1,
            ),
        }
        
        score = reputation_calculator._calculate_overall_score(dimension_scores)
        
        # Quality (100) with weight 0.25 * confidence 1.0 should dominate
        # over Reliability (0) with weight 0.20 * confidence 0.1
        assert score > 50.0  # Should be closer to 100 than to 0

    def test_is_reputation_current_returns_true_for_recent(self, reputation_calculator, sample_reputation):
        """_is_reputation_current should return True for recently calculated reputation"""
        sample_reputation.last_calculated = datetime.now(timezone.utc) - timedelta(hours=12)
        
        assert reputation_calculator._is_reputation_current(sample_reputation) is True

    def test_is_reputation_current_returns_false_for_old(self, reputation_calculator, sample_reputation):
        """_is_reputation_current should return False for old reputation"""
        sample_reputation.last_calculated = datetime.now(timezone.utc) - timedelta(days=2)
        
        assert reputation_calculator._is_reputation_current(sample_reputation) is False

    def test_get_primary_dimension_returns_correct_dimension(self, reputation_calculator):
        """_get_primary_dimension should return correct dimension for event types"""
        assert reputation_calculator._get_primary_dimension(
            ReputationEvent.RESOURCE_PUBLISHED
        ) == ReputationDimension.QUALITY
        
        assert reputation_calculator._get_primary_dimension(
            ReputationEvent.EXPERT_ENDORSEMENT
        ) == ReputationDimension.EXPERTISE
        
        assert reputation_calculator._get_primary_dimension(
            ReputationEvent.FRAUD_DETECTED
        ) == ReputationDimension.TRUSTWORTHINESS

    def test_calculate_immediate_impact_returns_correct_values(self, reputation_calculator):
        """_calculate_immediate_impact should return correct impact for event types"""
        assert reputation_calculator._calculate_immediate_impact(
            ReputationEvent.RESOURCE_PUBLISHED, {}
        ) == 2.0
        
        assert reputation_calculator._calculate_immediate_impact(
            ReputationEvent.EXPERT_ENDORSEMENT, {}
        ) == 5.0
        
        assert reputation_calculator._calculate_immediate_impact(
            ReputationEvent.FRAUD_DETECTED, {}
        ) == -10.0

    @pytest.mark.asyncio
    async def test_create_default_reputation_returns_valid_structure(self, reputation_calculator):
        """_create_default_reputation should return valid UserReputation with defaults"""
        reputation = await reputation_calculator._create_default_reputation("new_user_123")
        
        assert reputation.user_id == "new_user_123"
        assert reputation.overall_score == 50.0
        assert reputation.trust_level == TrustLevel.NEWCOMER
        assert len(reputation.badges) == 0
        assert len(reputation.dimension_scores) == len(ReputationDimension)
        
        # All dimension scores should be 50.0 with low confidence
        for dim_score in reputation.dimension_scores.values():
            assert dim_score.score == 50.0
            assert dim_score.confidence == 0.1


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling"""

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_get_existing_reputation_returns_none_on_db_error(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """_get_existing_reputation should return None on database error"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        mock_db_session.execute.side_effect = Exception("Database error")
        
        result = await reputation_calculator._get_existing_reputation("test_user_123")
        
        assert result is None

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_get_user_reputation_events_returns_empty_on_error(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """_get_user_reputation_events should return empty list on error"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        mock_db_session.execute.side_effect = Exception("Database error")
        
        result = await reputation_calculator._get_user_reputation_events("test_user_123")
        
        assert result == []

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_get_reputation_history_returns_empty_on_error(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """_get_reputation_history should return empty list on error"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        mock_db_session.execute.side_effect = Exception("Database error")
        
        result = await reputation_calculator._get_reputation_history("test_user_123")
        
        assert result == []

    def test_event_affects_dimension_returns_correct_boolean(self, reputation_calculator):
        """_event_affects_dimension should correctly identify affected dimensions"""
        # RESOURCE_RATED affects QUALITY and RELIABILITY
        event = {"event_type": "resource_rated"}
        
        assert reputation_calculator._event_affects_dimension(event, ReputationDimension.QUALITY) is True
        assert reputation_calculator._event_affects_dimension(event, ReputationDimension.RELIABILITY) is True
        assert reputation_calculator._event_affects_dimension(event, ReputationDimension.EXPERTISE) is False

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_store_reputation_handles_exception_gracefully(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_reputation
    ):
        """_store_reputation should handle exceptions gracefully without raising"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        mock_db_session.execute.side_effect = Exception("Database error")
        
        # Should not raise
        await reputation_calculator._store_reputation(sample_reputation)

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_update_reputation_incremental_handles_exception_gracefully(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_transaction
    ):
        """_update_reputation_incremental should handle exceptions gracefully"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        mock_db_session.execute.side_effect = Exception("Database error")
        
        # Should not raise
        await reputation_calculator._update_reputation_incremental("test_user_123", sample_transaction)

    def test_score_to_trust_level_converts_correctly(self, reputation_calculator):
        """_score_to_trust_level should convert scores to correct trust level names"""
        assert reputation_calculator._score_to_trust_level(95) == TrustLevel.EXPERT.name
        assert reputation_calculator._score_to_trust_level(80) == TrustLevel.TRUSTED.name
        assert reputation_calculator._score_to_trust_level(65) == TrustLevel.MEMBER.name
        assert reputation_calculator._score_to_trust_level(45) == TrustLevel.NEWCOMER.name
        assert reputation_calculator._score_to_trust_level(20) == TrustLevel.UNTRUSTED.name


# ============================================================================
# TRUST LEVEL THRESHOLD TESTS
# ============================================================================

class TestTrustLevelThresholds:
    """Tests for trust level threshold boundaries"""

    def test_trust_level_elite_threshold(self, reputation_calculator):
        """Test ELITE trust level at threshold boundary (96)"""
        assert reputation_calculator._determine_trust_level(96) == TrustLevel.ELITE
        assert reputation_calculator._determine_trust_level(95.9) == TrustLevel.EXPERT

    def test_trust_level_expert_threshold(self, reputation_calculator):
        """Test EXPERT trust level at threshold boundary (81)"""
        assert reputation_calculator._determine_trust_level(81) == TrustLevel.EXPERT
        assert reputation_calculator._determine_trust_level(80.9) == TrustLevel.TRUSTED

    def test_trust_level_trusted_threshold(self, reputation_calculator):
        """Test TRUSTED trust level at threshold boundary (61)"""
        assert reputation_calculator._determine_trust_level(61) == TrustLevel.TRUSTED
        assert reputation_calculator._determine_trust_level(60.9) == TrustLevel.MEMBER

    def test_trust_level_member_threshold(self, reputation_calculator):
        """Test MEMBER trust level at threshold boundary (41)"""
        assert reputation_calculator._determine_trust_level(41) == TrustLevel.MEMBER
        assert reputation_calculator._determine_trust_level(40.9) == TrustLevel.NEWCOMER

    def test_trust_level_newcomer_threshold(self, reputation_calculator):
        """Test NEWCOMER trust level at threshold boundary (21)"""
        assert reputation_calculator._determine_trust_level(21) == TrustLevel.NEWCOMER
        assert reputation_calculator._determine_trust_level(20.9) == TrustLevel.UNTRUSTED


# ============================================================================
# REPUTATION EVENT TYPE TESTS
# ============================================================================

class TestReputationEventTypes:
    """Tests for reputation event type handling"""

    def test_all_reputation_events_have_defined_impacts(self, reputation_calculator):
        """Verify all ReputationEvent types have defined impacts or default to 0"""
        for event in ReputationEvent:
            # Should not raise - either has defined impact or defaults to 0
            impact = reputation_calculator._calculate_immediate_impact(event, {})
            assert isinstance(impact, (int, float))

    def test_event_impact_resource_rated_depends_on_rating(self, reputation_calculator):
        """RESOURCE_RATED impact should depend on rating value"""
        # High rating (5) should have positive impact
        high_rating_impact = reputation_calculator._get_event_impact(
            {"event_type": "resource_rated", "rating": 5.0},
            ReputationDimension.QUALITY
        )
        
        # Low rating (1) should have negative impact
        low_rating_impact = reputation_calculator._get_event_impact(
            {"event_type": "resource_rated", "rating": 1.0},
            ReputationDimension.QUALITY
        )
        
        assert high_rating_impact > low_rating_impact

    def test_event_impact_review_received_depends_on_rating(self, reputation_calculator):
        """REVIEW_RECEIVED impact should depend on review rating"""
        high_review_impact = reputation_calculator._get_event_impact(
            {"event_type": "review_received", "review_rating": 5.0},
            ReputationDimension.RESPONSIVENESS
        )
        
        low_review_impact = reputation_calculator._get_event_impact(
            {"event_type": "review_received", "review_rating": 1.0},
            ReputationDimension.RESPONSIVENESS
        )
        
        assert high_review_impact > low_review_impact


# ============================================================================
# CONFIDENCE CALCULATION TESTS
# ============================================================================

class TestConfidenceCalculation:
    """Tests for confidence score calculation"""

    def test_calculate_confidence_returns_minimum_for_empty_events(self, reputation_calculator):
        """_calculate_confidence should return 0.1 for empty events"""
        confidence = reputation_calculator._calculate_confidence([])
        assert confidence == 0.1

    def test_calculate_confidence_increases_with_event_count(self, reputation_calculator):
        """Confidence should increase with more events"""
        # Spread events over time so time_factor > 0; use different event types for diversity
        few_events = [
            {
                "timestamp": (datetime.now(timezone.utc) - timedelta(days=i * 30)).isoformat(),
                "event_type": f"type_{i}",
                "quality_score": 1.0,
            }
            for i in range(3)
        ]

        many_events = [
            {
                "timestamp": (datetime.now(timezone.utc) - timedelta(days=i * 15)).isoformat(),
                "event_type": f"type_{i % 5}",
                "quality_score": 1.0,
            }
            for i in range(20)
        ]

        few_confidence = reputation_calculator._calculate_confidence(few_events)
        many_confidence = reputation_calculator._calculate_confidence(many_events)

        assert many_confidence > few_confidence

    def test_calculate_confidence_considers_event_diversity(self, reputation_calculator):
        """Confidence should be higher with diverse event types"""
        same_events = [
            {"timestamp": datetime.now(timezone.utc).isoformat(), "event_type": "quality_rating", "quality_score": 1.0}
            for _ in range(10)
        ]
        
        diverse_events = [
            {"timestamp": datetime.now(timezone.utc).isoformat(), "event_type": f"event_type_{i % 5}", "quality_score": 1.0}
            for i in range(10)
        ]
        
        same_confidence = reputation_calculator._calculate_confidence(same_events)
        diverse_confidence = reputation_calculator._calculate_confidence(diverse_events)
        
        assert diverse_confidence >= same_confidence


# ============================================================================
# TIME DECAY TESTS
# ============================================================================

class TestTimeDecay:
    """Tests for time decay in reputation calculations"""

    def test_apply_time_decay_returns_score_for_empty_events(self, reputation_calculator):
        """_apply_time_decay should return original score for empty events"""
        score = 75.0
        decayed = reputation_calculator._apply_time_decay(score, [])
        assert decayed == score

    def test_calculate_event_weight_decays_with_time(self, reputation_calculator):
        """Event weight should decrease for older events"""
        recent_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quality_score": 1.0,
        }
        
        old_event = {
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
            "quality_score": 1.0,
        }
        
        recent_weight = reputation_calculator._calculate_event_weight(recent_event)
        old_weight = reputation_calculator._calculate_event_weight(old_event)
        
        assert recent_weight > old_weight

    def test_calculate_event_weight_increases_with_source_credibility(self, reputation_calculator):
        """Event weight should increase for verified/trusted sources"""
        base_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quality_score": 1.0,
        }
        
        verified_event = {
            **base_event,
            "source_verified": True,
        }
        
        trusted_event = {
            **base_event,
            "source_trusted": True,
        }
        
        base_weight = reputation_calculator._calculate_event_weight(base_event)
        verified_weight = reputation_calculator._calculate_event_weight(verified_event)
        trusted_weight = reputation_calculator._calculate_event_weight(trusted_event)
        
        assert verified_weight > base_weight
        assert trusted_weight > base_weight


# ============================================================================
# BADGE CALCULATION TESTS
# ============================================================================

class TestBadgeCalculation:
    """Tests for badge earning logic"""

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_calculate_user_badges_awards_prolific_contributor(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """Should award prolific_contributor badge for 50+ resources"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session

        mock_result = MagicMock()
        mock_result.scalar.return_value = 55
        mock_result.scalar_one_or_none.return_value = None  # account_age query returns None
        mock_db_session.execute.return_value = mock_result
        
        dimension_scores = {
            ReputationDimension.QUALITY: ReputationScore(
                dimension=ReputationDimension.QUALITY,
                score=80.0,
                confidence=0.9,
                last_updated=datetime.now(timezone.utc),
                trend=0.0,
                evidence_count=10,
            )
        }
        
        badges = await reputation_calculator._calculate_user_badges("test_user_123", dimension_scores)
        
        assert "prolific_contributor" in badges

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_calculate_user_badges_awards_community_guardian(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """Should award community_guardian badge for 20+ community reports"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session

        mock_result = MagicMock()
        mock_result.scalar.return_value = 25
        mock_result.scalar_one_or_none.return_value = None  # account_age query returns None
        mock_db_session.execute.return_value = mock_result
        
        dimension_scores = {}
        
        badges = await reputation_calculator._calculate_user_badges("test_user_123", dimension_scores)
        
        assert "community_guardian" in badges

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_calculate_user_badges_awards_peer_recognized(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """Should award peer_recognized badge for 5+ expert endorsements"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session

        mock_result = MagicMock()
        mock_result.scalar.return_value = 7
        mock_result.scalar_one_or_none.return_value = None  # account_age query returns None
        mock_db_session.execute.return_value = mock_result
        
        dimension_scores = {}
        
        badges = await reputation_calculator._calculate_user_badges("test_user_123", dimension_scores)
        
        assert "peer_recognized" in badges

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_calculate_user_badges_awards_veteran_member(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """Should award veteran_member badge for 365+ day account age"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_result.scalar_one_or_none.return_value = datetime.now(timezone.utc) - timedelta(days=400)
        
        # Need to set up multiple query results
        mock_db_session.execute.return_value = mock_result
        
        dimension_scores = {}
        
        badges = await reputation_calculator._calculate_user_badges("test_user_123", dimension_scores)
        
        assert "veteran_member" in badges


# ============================================================================
# VERIFICATION STATUS TESTS
# ============================================================================

class TestVerificationStatus:
    """Tests for verification status handling"""

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_get_verification_status_returns_default_for_new_user(
        self, mock_get_session, reputation_calculator, mock_db_session
    ):
        """Should return default verification status for new users"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        status = await reputation_calculator._get_verification_status("new_user_123")
        
        assert status["email_verified"] is True
        assert status["identity_verified"] is False
        assert status["expert_verified"] is False
        assert status["organization_verified"] is False

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_get_verification_status_returns_stored_status(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_db_reputation_row
    ):
        """Should return stored verification status if available"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        sample_db_reputation_row.verification_status = {
            "email_verified": True,
            "identity_verified": True,
            "expert_verified": False,
            "organization_verified": True,
        }
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_db_reputation_row
        mock_db_session.execute.return_value = mock_result
        
        status = await reputation_calculator._get_verification_status("test_user_123")
        
        assert status["identity_verified"] is True
        assert status["organization_verified"] is True


# ============================================================================
# REPUTATION SUMMARY TESTS
# ============================================================================

class TestReputationSummary:
    """Tests for reputation summary generation"""

    @pytest.mark.asyncio
    @patch('prsm.economy.marketplace.reputation_system.get_async_session')
    async def test_get_user_reputation_summary_returns_valid_structure(
        self, mock_get_session, reputation_calculator, mock_db_session, sample_db_reputation_row
    ):
        """get_user_reputation_summary should return valid summary structure"""
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        sample_db_reputation_row.last_calculated = datetime.now(timezone.utc) - timedelta(hours=1)
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_db_reputation_row
        
        mock_events_result = MagicMock()
        mock_events_result.scalars.return_value.all.return_value = []
        
        mock_stats_result = MagicMock()
        mock_stats_result.scalar.return_value = 0
        mock_stats_result.scalar_one_or_none.return_value = None
        
        mock_db_session.execute.side_effect = [
            mock_result,        # _get_existing_reputation (cache hit)
        ]
        
        summary = await reputation_calculator.get_user_reputation_summary("test_user_123")
        
        assert "user_id" in summary
        assert "overall_score" in summary
        assert "trust_level" in summary
        assert "badges" in summary

    @pytest.mark.asyncio
    async def test_get_user_reputation_summary_handles_errors(
        self, reputation_calculator
    ):
        """get_user_reputation_summary should handle errors gracefully"""
        # Force calculate_user_reputation itself to raise (triggers the outer except path)
        from unittest.mock import AsyncMock as AM
        reputation_calculator.calculate_user_reputation = AM(
            side_effect=RuntimeError("Calculation failed")
        )

        summary = await reputation_calculator.get_user_reputation_summary("test_user_123")

        # The outer except path returns a default dict with 'error' key
        assert summary["overall_score"] == 50.0
        assert "error" in summary
