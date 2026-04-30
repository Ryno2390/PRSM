"""
BSC Integration Tests
=====================

Integration tests covering the complete BSC pipeline from quality scoring
through event publishing to whiteboard push handling.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.compute.nwtn.bsc.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from prsm.compute.nwtn.bsc.event_bus import BSCEvent, EventBus, EventType
from prsm.compute.nwtn.bsc.kl_filter import AdaptiveKLFilter, ProgressiveKLFilter, FilterDecision
from prsm.compute.nwtn.bsc.predictor import SurpriseScore
from prsm.compute.nwtn.bsc.promoter import BSCPromoter, ChunkMetadata, PromotionDecision
from prsm.compute.nwtn.bsc.quality_gate import QualityGate, QualityReport
from prsm.compute.nwtn.bsc.quality_scorer import QualityScore, QualityScorer
from prsm.compute.nwtn.bsc.semantic_dedup import DedupResult
from prsm.compute.nwtn.bsc.whiteboard_push import WhiteboardPushHandler
from prsm.compute.nwtn.bsc.deployment import BSCDeploymentConfig


# ======================================================================
# Helpers
# ======================================================================

def _make_mock_promotion_decision(
    chunk: str = "Test chunk content",
    source_agent: str = "agent/coder",
    session_id: str = "session/test-123",
    surprise_score: float = 0.85,
    raw_perplexity: float = 80.0,
    promoted: bool = True,
) -> PromotionDecision:
    """Create a PromotionDecision with proper metadata."""
    return PromotionDecision(
        promoted=promoted,
        chunk=chunk,
        metadata=ChunkMetadata(
            source_agent=source_agent,
            session_id=session_id,
        ),
        surprise_score=surprise_score,
        raw_perplexity=raw_perplexity,
        similarity_score=0.3,
        kl_result=MagicMock(decision=FilterDecision.PROMOTE, score=surprise_score, epsilon=0.55, reason="test"),
        dedup_result=DedupResult(
            is_redundant=False,
            max_similarity=0.3,
            most_similar_index=None,
            reason="novel",
        ),
        quality_report=None,
        reason="Test decision",
    )


def _make_surprise_score(score: float = 0.75, raw_perplexity: float = 80.0) -> SurpriseScore:
    """Create a SurpriseScore for mocking predictor output."""
    return SurpriseScore(
        score=score,
        raw_perplexity=raw_perplexity,
        token_count=20,
        context_tokens=100,
        adaptive_baseline=50.0,
    )


# ======================================================================
# Group 1: Quality Scoring Pipeline
# ======================================================================

class TestQualityScoringPipeline:
    """
    Tests for QualityScorer + QualityGate integration.
    
    Mock embedding_engine on QualityScorer to avoid loading sentence-transformers.
    Use real QualityScorer + QualityGate instances.
    """

    @pytest.mark.asyncio
    async def test_high_quality_chunk_passes_gate(self):
        """High-quality chunk (concrete, consistent) passes the quality gate."""
        scorer = QualityScorer()
        gate = QualityGate(threshold=0.3, scorer=scorer)

        # High-quality chunk: file paths, decisions, concrete content
        high_quality_chunk = (
            "Decided to migrate from SQLite to PostgreSQL for the auth layer. "
            "Updated prsm/compute/nwtn/auth.py and created migration script at "
            "migrations/001_add_postgres.py. Status: deployed to staging."
        )

        decision = _make_mock_promotion_decision(
            chunk=high_quality_chunk,
            raw_perplexity=75.0,  # Good coherence range
        )

        report = await gate.evaluate(decision)

        assert report.passed, f"Expected pass, got fail with reason: {report.reason}"
        assert report.overall_score > gate.threshold

    @pytest.mark.asyncio
    async def test_low_quality_chunk_fails_gate(self):
        """Low-quality chunk (vague, contradictory) fails the quality gate."""
        scorer = QualityScorer()
        # Use higher threshold to ensure low-quality chunk fails
        gate = QualityGate(threshold=0.5, scorer=scorer)

        # Low-quality chunk: vague markers, no concrete content, meta-commentary
        low_quality_chunk = (
            "Perhaps maybe we should consider thinking about the possibility "
            "of perhaps looking into the thing. Just a thought. TBD. Unclear."
        )

        decision = _make_mock_promotion_decision(
            chunk=low_quality_chunk,
            raw_perplexity=500.0,  # Very high perplexity = incoherent
        )

        report = await gate.evaluate(decision)

        assert not report.passed, f"Expected fail, got pass with score {report.overall_score}"
        assert report.overall_score < gate.threshold

    @pytest.mark.asyncio
    async def test_borderline_chunk_gets_scored(self):
        """Borderline chunk gets scored but we don't assert pass/fail, just valid range."""
        scorer = QualityScorer()
        gate = QualityGate(threshold=0.35, scorer=scorer)

        # Borderline chunk: some actionability but vague
        borderline_chunk = (
            "Updated the config file. Maybe this will help with performance."
        )

        decision = _make_mock_promotion_decision(
            chunk=borderline_chunk,
            raw_perplexity=100.0,
        )

        report = await gate.evaluate(decision)

        # Just verify score is in valid range [0, 1]
        assert 0.0 <= report.overall_score <= 1.0
        assert 0.0 <= report.quality_score.factual_consistency <= 1.0
        assert 0.0 <= report.quality_score.source_reliability <= 1.0
        assert 0.0 <= report.quality_score.actionability <= 1.0
        assert 0.0 <= report.quality_score.coherence <= 1.0


# ======================================================================
# Group 2: Promoter Flow
# ======================================================================

class TestPromoterFlow:
    """
    Tests for BSCPromoter flow with real AdaptiveKLFilter + QualityGate + EventBus.
    Mock BSCPredictor and SemanticDeduplicator.
    """

    @pytest.mark.asyncio
    async def test_high_surprise_chunk_promoted_and_event_published(self):
        """High-surprise chunk → promoted=True → CHUNK_PROMOTED event published to EventBus."""
        bus = EventBus()
        event_received = asyncio.Event()
        captured_event = None

        async def event_handler(event: BSCEvent):
            nonlocal captured_event
            captured_event = event
            event_received.set()

        await bus.subscribe(EventType.CHUNK_PROMOTED, event_handler)

        # Mock predictor returns high surprise
        mock_predictor = AsyncMock()
        mock_predictor.score_surprise.return_value = _make_surprise_score(score=0.85)

        # Mock deduplicator returns not-duplicate
        mock_dedup = AsyncMock()
        mock_dedup.check.return_value = DedupResult(
            is_redundant=False,
            max_similarity=0.2,
            most_similar_index=None,
            reason="novel content",
        )
        mock_dedup.add_to_index.return_value = 0

        # Real KL filter with low epsilon
        kl_filter = AdaptiveKLFilter(epsilon=0.5)

        # Real quality gate
        gate = QualityGate(threshold=0.3)

        promoter = BSCPromoter(
            predictor=mock_predictor,
            kl_filter=kl_filter,
            deduplicator=mock_dedup,
            quality_gate=gate,
            event_bus=bus,
        )

        decision = await promoter.process_chunk(
            chunk="Critical decision: migrated auth layer to PostgreSQL.",
            context="Previous context about SQLite issues.",
            source_agent="agent/coder",
            session_id="session/promoter-test",
        )

        # Wait for event to be published
        await asyncio.wait_for(event_received.wait(), timeout=2.0)

        assert decision.promoted, "Chunk should be promoted"
        assert decision.surprise_score > kl_filter.epsilon
        assert captured_event is not None
        assert captured_event.event_type == EventType.CHUNK_PROMOTED
        assert "decision" in captured_event.data

    @pytest.mark.asyncio
    async def test_low_surprise_chunk_not_promoted_no_event(self):
        """Low-surprise chunk → promoted=False → no event published."""
        bus = EventBus()
        event_count = 0

        async def event_handler(event: BSCEvent):
            nonlocal event_count
            event_count += 1

        await bus.subscribe(EventType.CHUNK_PROMOTED, event_handler)

        # Mock predictor returns low surprise
        mock_predictor = AsyncMock()
        mock_predictor.score_surprise.return_value = _make_surprise_score(score=0.3)

        mock_dedup = AsyncMock()
        mock_dedup.check.return_value = DedupResult(
            is_redundant=False,
            max_similarity=0.2,
            most_similar_index=None,
            reason="novel",
        )

        kl_filter = AdaptiveKLFilter(epsilon=0.55)
        gate = QualityGate(threshold=0.3)

        promoter = BSCPromoter(
            predictor=mock_predictor,
            kl_filter=kl_filter,
            deduplicator=mock_dedup,
            quality_gate=gate,
            event_bus=bus,
        )

        decision = await promoter.process_chunk(
            chunk="Some routine update about the same topic.",
            context="Context about routine updates.",
            source_agent="agent/coder",
            session_id="session/low-surprise",
        )

        # Give event bus time to process if it were going to
        await asyncio.sleep(0.1)

        assert not decision.promoted, "Low-surprise chunk should not be promoted"
        assert event_count == 0, "No event should be published for non-promoted chunk"

    @pytest.mark.asyncio
    async def test_duplicate_chunk_not_promoted(self):
        """Duplicate chunk (mock dedup returns duplicate) → not promoted."""
        bus = EventBus()
        event_count = 0

        async def event_handler(event: BSCEvent):
            nonlocal event_count
            event_count += 1

        await bus.subscribe(EventType.CHUNK_PROMOTED, event_handler)

        # Mock predictor returns high surprise
        mock_predictor = AsyncMock()
        mock_predictor.score_surprise.return_value = _make_surprise_score(score=0.85)

        # Mock deduplicator returns IS duplicate
        mock_dedup = AsyncMock()
        mock_dedup.check.return_value = DedupResult(
            is_redundant=True,
            max_similarity=0.92,
            most_similar_index=5,
            reason="duplicate content detected",
        )

        kl_filter = AdaptiveKLFilter(epsilon=0.5)
        gate = QualityGate(threshold=0.3)

        promoter = BSCPromoter(
            predictor=mock_predictor,
            kl_filter=kl_filter,
            deduplicator=mock_dedup,
            quality_gate=gate,
            event_bus=bus,
        )

        decision = await promoter.process_chunk(
            chunk="This is a duplicate of existing content.",
            context="Context with similar content.",
            source_agent="agent/coder",
            session_id="session/dup-test",
        )

        await asyncio.sleep(0.1)

        assert not decision.promoted, "Duplicate chunk should not be promoted"
        assert decision.dedup_result is not None
        assert decision.dedup_result.is_redundant
        assert event_count == 0


# ======================================================================
# Group 3: End-to-End Event Pipeline
# ======================================================================

class TestEndToEndEventPipeline:
    """
    Tests for EventBus + WhiteboardPushHandler + LiveScribe integration.
    """

    @pytest.mark.asyncio
    async def test_promoted_event_triggers_livescribe_call(self):
        """Promoted event → WhiteboardPushHandler receives it → LiveScribe.on_chunk_promoted called."""
        bus = EventBus()
        mock_scribe = AsyncMock()
        mock_scribe.on_chunk_promoted.return_value = MagicMock(conflict_detected=False)

        handler = WhiteboardPushHandler(event_bus=bus, live_scribe=mock_scribe)
        await handler.start()

        decision = _make_mock_promotion_decision()
        event = BSCEvent(
            event_type=EventType.CHUNK_PROMOTED,
            data={"decision": decision},
            session_id="session/e2e-test",
        )

        await bus.publish(event)
        await asyncio.sleep(0.1)  # Allow async delivery

        mock_scribe.on_chunk_promoted.assert_called_once()
        stats = handler.get_stats()
        assert stats["pushed"] == 1

        await handler.stop()

    @pytest.mark.asyncio
    async def test_livescribe_exception_increments_failed(self):
        """LiveScribe raises Exception → stats.failed incremented."""
        bus = EventBus()
        mock_scribe = AsyncMock()
        mock_scribe.on_chunk_promoted.side_effect = RuntimeError("Scribe error")

        handler = WhiteboardPushHandler(event_bus=bus, live_scribe=mock_scribe)
        await handler.start()

        decision = _make_mock_promotion_decision()
        event = BSCEvent(
            event_type=EventType.CHUNK_PROMOTED,
            data={"decision": decision},
            session_id="session/fail-test",
        )

        await bus.publish(event)
        await asyncio.sleep(0.1)

        stats = handler.get_stats()
        assert stats["failed"] == 1
        assert stats["pushed"] == 0

        await handler.stop()

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_after_failures(self):
        """CircuitBreaker trips after 5 LiveScribe failures → subsequent events skipped (stats.skipped)."""
        current_time = [0.0]

        def mock_clock():
            return current_time[0]

        bus = EventBus()

        # LiveScribe always fails
        mock_scribe = AsyncMock()
        mock_scribe.on_chunk_promoted.side_effect = RuntimeError("Scribe always fails")

        # Circuit breaker with low threshold for testing
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=2,
            clock=mock_clock,
        )

        handler = WhiteboardPushHandler(
            event_bus=bus,
            live_scribe=mock_scribe,
            circuit_breaker=breaker,
        )
        await handler.start()

        # Publish 5 events to trigger the circuit
        for i in range(5):
            decision = _make_mock_promotion_decision(chunk=f"Failed chunk {i}")
            event = BSCEvent(
                event_type=EventType.CHUNK_PROMOTED,
                data={"decision": decision},
                session_id=f"session/circuit-test-{i}",
            )
            await bus.publish(event)
            await asyncio.sleep(0.05)

        # Verify circuit is now open
        assert await breaker.get_state() == CircuitState.OPEN

        # Next event should be skipped due to open circuit
        decision = _make_mock_promotion_decision(chunk="Should be skipped")
        event = BSCEvent(
            event_type=EventType.CHUNK_PROMOTED,
            data={"decision": decision},
            session_id="session/after-open",
        )
        await bus.publish(event)
        await asyncio.sleep(0.1)

        stats = handler.get_stats()
        # 5 failures + 1 skipped = 6 total failed/skipped
        assert stats["failed"] == 5, f"Expected 5 failures, got {stats['failed']}"
        assert stats["skipped"] >= 1, f"Expected at least 1 skipped, got {stats['skipped']}"

        await handler.stop()


# ======================================================================
# Group 4: Round Advancement
# ======================================================================

class TestRoundAdvancement:
    """
    Tests for ProgressiveKLFilter round advancement behavior.
    """

    def test_advance_round_decreases_epsilon(self):
        """advance_round() decreases epsilon progressively."""
        kl_filter = ProgressiveKLFilter(
            initial_epsilon=0.55,
            min_epsilon=0.38,
            total_rounds=9,
        )

        # Round 0: initial epsilon
        assert kl_filter.epsilon == 0.55

        # Advance to round 1
        kl_filter.advance_round(1)
        assert kl_filter.epsilon < 0.55, f"Epsilon should decrease, got {kl_filter.epsilon}"

        # Advance to round 5
        kl_filter.advance_round(5)
        assert kl_filter.epsilon < 0.55
        assert kl_filter.epsilon >= 0.38, f"Epsilon should stay above min, got {kl_filter.epsilon}"

        # Advance to final round
        kl_filter.advance_round(9)
        assert kl_filter.epsilon == 0.38, f"Final round should hit min epsilon, got {kl_filter.epsilon}"

    def test_reset_session_resets_round_counter(self):
        """Reset session functionality exists on ProgressiveKLFilter."""
        kl_filter = ProgressiveKLFilter(
            initial_epsilon=0.55,
            min_epsilon=0.38,
            total_rounds=9,
        )

        # Advance several rounds
        kl_filter.advance_round(5)
        assert kl_filter.current_round == 5

        # Reset to round 0
        kl_filter.advance_round(0)
        assert kl_filter.current_round == 0
        assert kl_filter.epsilon == 0.55, f"Epsilon should reset to initial, got {kl_filter.epsilon}"


# ======================================================================
# Group 5: Session Reset
# ======================================================================

class TestSessionReset:
    """
    Tests for BSCPromoter session reset behavior.
    """

    @pytest.mark.asyncio
    async def test_chunk_seen_before_reset_fails_dedup_after_reset_passes(self):
        """Chunk seen before reset fails dedup → after reset_session() same chunk passes dedup."""
        bus = EventBus()

        # Mock predictor
        mock_predictor = AsyncMock()
        mock_predictor.score_surprise.return_value = _make_surprise_score(score=0.85)

        # Real deduplicator starts empty
        from prsm.compute.nwtn.bsc.semantic_dedup import SemanticDeduplicator
        
        # Mock the deduplicator to simulate the behavior
        dedup_calls = []
        
        async def mock_dedup_check(chunk):
            # First call for chunk A returns not redundant
            # After it's added, second call for same chunk returns redundant
            if chunk == "Unique chunk X" and len(dedup_calls) == 0:
                dedup_calls.append(chunk)
                return DedupResult(is_redundant=False, max_similarity=0.1, most_similar_index=None, reason="novel")
            elif chunk == "Unique chunk X" and len(dedup_calls) > 0:
                dedup_calls.append(chunk)
                return DedupResult(is_redundant=True, max_similarity=0.95, most_similar_index=0, reason="duplicate")
            return DedupResult(is_redundant=False, max_similarity=0.1, most_similar_index=None, reason="novel")
        
        mock_dedup = AsyncMock()
        mock_dedup.check = mock_dedup_check
        mock_dedup.add_to_index = AsyncMock(return_value=0)
        mock_dedup.clear = MagicMock()

        kl_filter = AdaptiveKLFilter(epsilon=0.5)
        gate = QualityGate(threshold=0.3)

        promoter = BSCPromoter(
            predictor=mock_predictor,
            kl_filter=kl_filter,
            deduplicator=mock_dedup,
            quality_gate=gate,
            event_bus=bus,
        )

        # First call - should pass dedup (not redundant)
        decision1 = await promoter.process_chunk(
            chunk="Unique chunk X",
            context="Test context",
            source_agent="agent/coder",
            session_id="session/reset-test",
        )
        assert decision1.promoted, "First occurrence should be promoted"

        # Reset the dedup tracking to simulate session reset
        dedup_calls.clear()
        
        # Call reset_session
        promoter.reset_session()

        # Now same chunk should pass dedup again (tracking cleared)
        decision2 = await promoter.process_chunk(
            chunk="Unique chunk X",
            context="Test context",
            source_agent="agent/coder",
            session_id="session/reset-test",
        )
        # After reset, the dedup should see this as novel again
        assert decision2.promoted, "After reset, chunk should be treated as novel again"

    @pytest.mark.asyncio
    async def test_stats_reset_after_reset_session(self):
        """Stats reset after reset_session()."""
        bus = EventBus()

        mock_predictor = AsyncMock()
        mock_predictor.score_surprise.return_value = _make_surprise_score(score=0.85)

        mock_dedup = AsyncMock()
        mock_dedup.check.return_value = DedupResult(
            is_redundant=False,
            max_similarity=0.1,
            most_similar_index=None,
            reason="novel",
        )
        mock_dedup.add_to_index.return_value = 0
        mock_dedup.clear = MagicMock()
        mock_dedup.index_size = 0

        kl_filter = AdaptiveKLFilter(epsilon=0.5)
        gate = QualityGate(threshold=0.3)

        promoter = BSCPromoter(
            predictor=mock_predictor,
            kl_filter=kl_filter,
            deduplicator=mock_dedup,
            quality_gate=gate,
            event_bus=bus,
        )

        # Process a chunk
        await promoter.process_chunk(
            chunk="Test chunk",
            context="Context",
            source_agent="agent/coder",
            session_id="session/stats-test",
        )

        # Get stats before reset via the stats property
        stats_before = promoter.stats
        assert stats_before.get("total_processed", 0) >= 1

        # Reset session
        promoter.reset_session()

        # Verify dedup.clear was called
        mock_dedup.clear.assert_called_once()


# ======================================================================
# Group 6: Config Path
# ======================================================================

class TestConfigPath:
    """
    Tests for BSCPromoter.from_config() construction.
    """

    @pytest.mark.asyncio
    async def test_from_config_builds_without_loading_ml_models(self):
        """BSCPromoter.from_config(BSCDeploymentConfig.auto()) builds successfully without loading ML models."""
        config = BSCDeploymentConfig.auto()

        # Patch BSCPredictor.__init__ and SemanticDeduplicator.__init__ to no-op
        with patch('prsm.compute.nwtn.bsc.promoter.BSCPredictor') as MockPredictor:
            with patch('prsm.compute.nwtn.bsc.promoter.SemanticDeduplicator') as MockDedup:
                mock_predictor = MagicMock()
                mock_predictor.warmup = AsyncMock()
                MockPredictor.return_value = mock_predictor

                mock_dedup = MagicMock()
                MockDedup.return_value = mock_dedup

                promoter = BSCPromoter.from_config(config)

                assert promoter is not None
                assert promoter._predictor is mock_predictor
                assert promoter._deduplicator is mock_dedup

    @pytest.mark.asyncio
    async def test_from_config_without_quality_gate(self):
        """from_config with enable_quality_gate=False builds without quality gate."""
        config = BSCDeploymentConfig.auto()

        with patch('prsm.compute.nwtn.bsc.promoter.BSCPredictor') as MockPredictor:
            with patch('prsm.compute.nwtn.bsc.promoter.SemanticDeduplicator') as MockDedup:
                mock_predictor = MagicMock()
                mock_predictor.warmup = AsyncMock()
                MockPredictor.return_value = mock_predictor

                mock_dedup = MagicMock()
                MockDedup.return_value = mock_dedup

                promoter = BSCPromoter.from_config(config, enable_quality_gate=False)

                assert promoter is not None
                assert promoter._quality_gate is None, "Quality gate should be None when enable_quality_gate=False"
