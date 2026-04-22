"""
Tests for Sub-phase 10.1: BSC (Bayesian Surprise Compressor) Core

Tests are organised into four layers:
  1. Unit tests  — individual components, no model loading
  2. Integration tests — full pipeline with a real (small) model
  3. Threshold calibration tests — verify epsilon / similarity semantics
  4. Adaptive behaviour tests — burst mode, baseline drift

All tests that require torch/transformers are guarded so the suite passes
on machines without GPU, and the integration tests are skipped if the
default BSC model is not cached locally (to avoid downloading during CI).
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from prsm.compute.nwtn.bsc import (
    AdaptiveKLFilter,
    BSCDeploymentConfig,
    BSCPredictor,
    BSCPromoter,
    ChunkMetadata,
    DeploymentMode,
    DedupResult,
    FilterDecision,
    KLFilter,
    KLFilterResult,
    PromotionDecision,
    SemanticDeduplicator,
    SurpriseScore,
)


# ======================================================================
# Helpers
# ======================================================================

def _score(score: float, perplexity: float = 50.0, tokens: int = 20) -> SurpriseScore:
    return SurpriseScore(
        score=score,
        raw_perplexity=perplexity,
        token_count=tokens,
        context_tokens=100,
        adaptive_baseline=45.0,
    )


# ======================================================================
# 1. Deployment config
# ======================================================================

class TestBSCDeploymentConfig:
    def test_auto_returns_local_transformers(self):
        cfg = BSCDeploymentConfig.auto()
        assert cfg.mode == DeploymentMode.LOCAL_TRANSFORMERS
        assert cfg.device in ("mps", "cuda", "cpu")

    def test_defaults_are_reasonable(self):
        cfg = BSCDeploymentConfig()
        assert 0 < cfg.epsilon < 1
        assert 0 < cfg.similarity_threshold <= 1
        assert cfg.max_context_tokens > 0

    def test_validation_requires_endpoint_for_network_mode(self):
        cfg = BSCDeploymentConfig(mode=DeploymentMode.NETWORK_SERVICE)
        with pytest.raises(ValueError, match="network_endpoint"):
            cfg.validate()

    def test_validation_passes_with_endpoint(self):
        cfg = BSCDeploymentConfig(
            mode=DeploymentMode.NETWORK_SERVICE,
            network_endpoint="http://bsc.prsm-network.com:7890",
        )
        cfg.validate()

    def test_validation_rejects_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            BSCDeploymentConfig(epsilon=0.0).validate()
        with pytest.raises(ValueError, match="epsilon"):
            BSCDeploymentConfig(epsilon=1.0).validate()

    def test_validation_rejects_invalid_similarity_threshold(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            BSCDeploymentConfig(similarity_threshold=0.0).validate()


# ======================================================================
# 2. KL Filter — unit tests (no model)
# ======================================================================

class TestKLFilter:
    def test_promotes_high_surprise(self):
        kl = KLFilter(epsilon=0.55)
        result = kl.evaluate(_score(0.80))
        assert result.decision == FilterDecision.PROMOTE

    def test_discards_low_surprise(self):
        kl = KLFilter(epsilon=0.55)
        result = kl.evaluate(_score(0.30))
        assert result.decision == FilterDecision.DISCARD

    def test_discards_exactly_at_epsilon(self):
        kl = KLFilter(epsilon=0.55)
        result = kl.evaluate(_score(0.55))
        # score must be STRICTLY greater than epsilon to promote
        assert result.decision == FilterDecision.DISCARD

    def test_promotes_just_above_epsilon(self):
        kl = KLFilter(epsilon=0.55)
        result = kl.evaluate(_score(0.5501))
        assert result.decision == FilterDecision.PROMOTE

    def test_discards_short_chunks_regardless_of_score(self):
        kl = KLFilter(epsilon=0.55, min_token_count=8)
        result = kl.evaluate(_score(0.99, tokens=3))
        assert result.decision == FilterDecision.DISCARD
        assert "too short" in result.reason

    def test_result_contains_epsilon(self):
        kl = KLFilter(epsilon=0.65)
        result = kl.evaluate(_score(0.80))
        assert result.epsilon == 0.65

    def test_result_contains_score(self):
        kl = KLFilter(epsilon=0.55)
        result = kl.evaluate(_score(0.72))
        assert result.score == pytest.approx(0.72)

    def test_reason_string_is_non_empty(self):
        kl = KLFilter(epsilon=0.55)
        for score_val in [0.2, 0.55, 0.9]:
            result = kl.evaluate(_score(score_val))
            assert len(result.reason) > 0


class TestAdaptiveKLFilter:
    def test_promotes_under_normal_conditions(self):
        kl = AdaptiveKLFilter(epsilon=0.55)
        result = kl.evaluate(_score(0.80))
        assert result.decision == FilterDecision.PROMOTE

    def test_rate_limits_during_burst(self):
        """Fill the window with high-surprise chunks, then a moderate chunk should be rate-limited."""
        kl = AdaptiveKLFilter(
            epsilon=0.55,
            window_size=10,
            burst_threshold=0.65,
            burst_penalty=0.20,
        )
        # Fill window with high-surprise scores to trigger burst mode
        for _ in range(10):
            kl.evaluate(_score(0.85))

        # A moderate chunk that would normally promote is rate-limited
        moderate = _score(0.60)
        result = kl.evaluate(moderate)
        # In burst mode epsilon is raised to 0.75; 0.60 should be rate-limited/discarded
        assert result.decision in (FilterDecision.DISCARD, FilterDecision.RATE_LIMIT)

    def test_very_high_surprise_still_promotes_during_burst(self):
        kl = AdaptiveKLFilter(
            epsilon=0.55,
            window_size=10,
            burst_threshold=0.65,
            burst_penalty=0.20,
        )
        for _ in range(10):
            kl.evaluate(_score(0.85))

        # Even in burst mode, a genuinely very-high-surprise chunk should promote
        very_high = _score(0.95)
        result = kl.evaluate(very_high)
        assert result.decision == FilterDecision.PROMOTE


# ======================================================================
# 3. Semantic De-duplicator — unit tests (mocked encoder)
# ======================================================================

class TestSemanticDeduplicator:
    """Tests using a mocked encoder to avoid downloading models in unit tests."""

    def _make_dedup(self, threshold: float = 0.85) -> SemanticDeduplicator:
        dedup = SemanticDeduplicator(
            model_name="all-MiniLM-L6-v2",
            similarity_threshold=threshold,
        )
        dedup._loaded = True
        dedup._encoder_type = "sentence_transformers"
        return dedup

    def _set_mock_embed(self, dedup: SemanticDeduplicator, vector: np.ndarray):
        dedup._encoder = MagicMock()
        dedup._encoder.encode = MagicMock(return_value=vector)

    @pytest.mark.asyncio
    async def test_accepts_when_whiteboard_empty(self):
        dedup = self._make_dedup()
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._set_mock_embed(dedup, vec)

        result = await dedup.check("any text at all")
        assert not result.is_redundant
        assert result.max_similarity == 0.0
        assert "empty" in result.reason

    @pytest.mark.asyncio
    async def test_rejects_near_duplicate(self):
        dedup = self._make_dedup(threshold=0.85)
        # Store a vector
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dedup._embeddings.append(v1)

        # Query with an almost-identical vector (cos sim ≈ 0.9998)
        v2 = np.array([0.9999, 0.01, 0.0], dtype=np.float32)
        v2 /= np.linalg.norm(v2)
        self._set_mock_embed(dedup, v2)

        result = await dedup.check("near duplicate")
        assert result.is_redundant
        assert result.max_similarity > 0.85

    @pytest.mark.asyncio
    async def test_accepts_novel_vector(self):
        dedup = self._make_dedup(threshold=0.85)
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dedup._embeddings.append(v1)

        # Orthogonal vector — cos sim = 0
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._set_mock_embed(dedup, v2)

        result = await dedup.check("completely different topic")
        assert not result.is_redundant
        assert result.max_similarity == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.asyncio
    async def test_add_to_index_grows_index(self):
        dedup = self._make_dedup()
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._set_mock_embed(dedup, v)

        assert dedup.index_size == 0
        idx = await dedup.add_to_index("first entry")
        assert idx == 0
        assert dedup.index_size == 1

        idx2 = await dedup.add_to_index("second entry")
        assert idx2 == 1
        assert dedup.index_size == 2

    def test_clear_resets_index(self):
        dedup = self._make_dedup()
        dedup._embeddings = [np.ones(3) for _ in range(5)]
        assert dedup.index_size == 5
        dedup.clear()
        assert dedup.index_size == 0


# ======================================================================
# 4. BSC Promoter — unit tests (all stages mocked)
# ======================================================================

class TestBSCPromoter:
    """Full pipeline tests with mocked predictor, KL filter, and deduplicator."""

    def _make_promoter(self) -> BSCPromoter:
        predictor = MagicMock(spec=BSCPredictor)
        predictor.warmup = AsyncMock()
        predictor.baseline_perplexity = 45.0

        kl_filter = MagicMock(spec=KLFilter)
        kl_filter.epsilon = 0.55

        deduplicator = MagicMock(spec=SemanticDeduplicator)
        deduplicator.check = AsyncMock()
        deduplicator.add_to_index = AsyncMock(return_value=0)
        deduplicator.index_size = 0

        promoter = BSCPromoter(
            predictor=predictor,
            kl_filter=kl_filter,
            deduplicator=deduplicator,
        )
        return promoter

    @pytest.mark.asyncio
    async def test_promotes_high_surprise_novel_chunk(self):
        promoter = self._make_promoter()

        promoter._predictor.score_surprise = AsyncMock(
            return_value=_score(0.82)
        )
        promoter._kl_filter.evaluate = MagicMock(
            return_value=KLFilterResult(
                decision=FilterDecision.PROMOTE,
                score=0.82,
                epsilon=0.55,
                reason="high surprise",
            )
        )
        promoter._deduplicator.check = AsyncMock(
            return_value=DedupResult(
                is_redundant=False,
                max_similarity=0.12,
                most_similar_index=None,
                reason="novel",
            )
        )

        decision = await promoter.process_chunk(
            chunk="Auth layer now requires PostgreSQL.",
            context="We are using SQLite.",
            source_agent="agent/coder-20260326",
            session_id="sess-001",
        )

        assert decision.promoted is True
        assert decision.surprise_score == pytest.approx(0.82)
        assert decision.similarity_score == pytest.approx(0.12)
        assert decision.metadata.source_agent == "agent/coder-20260326"

    @pytest.mark.asyncio
    async def test_discards_low_surprise_chunk(self):
        promoter = self._make_promoter()

        promoter._predictor.score_surprise = AsyncMock(
            return_value=_score(0.25)
        )
        promoter._kl_filter.evaluate = MagicMock(
            return_value=KLFilterResult(
                decision=FilterDecision.DISCARD,
                score=0.25,
                epsilon=0.55,
                reason="below epsilon",
            )
        )

        decision = await promoter.process_chunk(
            chunk="The code is proceeding as planned.",
            context="We are building a database migration.",
            source_agent="agent/coder-20260326",
            session_id="sess-001",
        )

        assert decision.promoted is False
        assert decision.dedup_result is None  # never reached dedup
        promoter._deduplicator.check.assert_not_called()

    @pytest.mark.asyncio
    async def test_discards_semantic_duplicate(self):
        promoter = self._make_promoter()

        promoter._predictor.score_surprise = AsyncMock(
            return_value=_score(0.78)
        )
        promoter._kl_filter.evaluate = MagicMock(
            return_value=KLFilterResult(
                decision=FilterDecision.PROMOTE,
                score=0.78,
                epsilon=0.55,
                reason="high surprise",
            )
        )
        promoter._deduplicator.check = AsyncMock(
            return_value=DedupResult(
                is_redundant=True,
                max_similarity=0.91,
                most_similar_index=3,
                reason="semantic duplicate",
            )
        )

        decision = await promoter.process_chunk(
            chunk="The database needs Postgres, not SQLite.",
            context="Auth requires PostgreSQL.",
            source_agent="agent/coder-20260326",
            session_id="sess-001",
        )

        assert decision.promoted is False
        assert decision.similarity_score == pytest.approx(0.91)
        # add_to_index must NOT be called for duplicates
        promoter._deduplicator.add_to_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_to_index_called_on_promotion(self):
        promoter = self._make_promoter()

        promoter._predictor.score_surprise = AsyncMock(return_value=_score(0.82))
        promoter._kl_filter.evaluate = MagicMock(
            return_value=KLFilterResult(
                decision=FilterDecision.PROMOTE, score=0.82, epsilon=0.55, reason="ok"
            )
        )
        promoter._deduplicator.check = AsyncMock(
            return_value=DedupResult(
                is_redundant=False,
                max_similarity=0.10,
                most_similar_index=None,
                reason="novel",
            )
        )

        await promoter.process_chunk(
            chunk="New finding.",
            context="context",
            source_agent="agent/test",
            session_id="sess-001",
        )

        promoter._deduplicator.add_to_index.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_chunk_returns_not_promoted(self):
        promoter = self._make_promoter()
        promoter._predictor.score_surprise = AsyncMock(
            return_value=SurpriseScore(
                score=0.0, raw_perplexity=0.0, token_count=0,
                context_tokens=0, adaptive_baseline=45.0,
            )
        )
        promoter._kl_filter.evaluate = MagicMock(
            return_value=KLFilterResult(
                decision=FilterDecision.DISCARD, score=0.0, epsilon=0.55,
                reason="empty chunk"
            )
        )

        decision = await promoter.process_chunk(
            chunk="   ",
            context="context",
            source_agent="agent/test",
            session_id="sess-001",
        )
        assert decision.promoted is False

    @pytest.mark.asyncio
    async def test_stats_track_promotion_rate(self):
        promoter = self._make_promoter()

        def make_decision(promoted: bool):
            promoter._predictor.score_surprise = AsyncMock(return_value=_score(0.82 if promoted else 0.20))
            kl_decision = FilterDecision.PROMOTE if promoted else FilterDecision.DISCARD
            promoter._kl_filter.evaluate = MagicMock(
                return_value=KLFilterResult(
                    decision=kl_decision, score=0.82 if promoted else 0.20,
                    epsilon=0.55, reason="test"
                )
            )
            if promoted:
                promoter._deduplicator.check = AsyncMock(
                    return_value=DedupResult(
                        is_redundant=False, max_similarity=0.10,
                        most_similar_index=None, reason="novel"
                    )
                )

        async def run_chunks(n_promote: int, n_discard: int):
            for _ in range(n_promote):
                make_decision(promoted=True)
                await promoter.process_chunk("chunk", "ctx", "agent/test", "sess")
            for _ in range(n_discard):
                make_decision(promoted=False)
                await promoter.process_chunk("chunk", "ctx", "agent/test", "sess")

        await run_chunks(3, 7)
        stats = promoter.stats
        assert stats["total_processed"] == 10
        assert stats["total_promoted"] == 3
        assert stats["promotion_rate"] == pytest.approx(0.3)

    def test_reset_session_clears_dedup(self):
        promoter = self._make_promoter()
        promoter.reset_session()
        promoter._deduplicator.clear.assert_called_once()

    @classmethod
    def from_config_creates_wired_promoter(cls):
        cfg = BSCDeploymentConfig(
            mode=DeploymentMode.LOCAL_TRANSFORMERS,
            epsilon=0.60,
            similarity_threshold=0.80,
        )
        cfg.validate()
        promoter = BSCPromoter.from_config(cfg)
        assert isinstance(promoter, BSCPromoter)
        assert promoter._kl_filter.epsilon == pytest.approx(0.60)


# ======================================================================
# 5. Predictor — unit tests (mocked model, no download)
# ======================================================================

class TestBSCPredictorScoreNormalisation:
    """Test the score normalisation logic without loading a real model."""

    def test_build_score_above_baseline_gives_high_score(self):
        cfg = BSCDeploymentConfig.auto()
        predictor = BSCPredictor(cfg)
        predictor._baseline = 45.0

        score = predictor._build_score(
            perplexity=200.0,  # much higher than baseline
            n_chunk=20,
            n_ctx=100,
        )
        assert score.score > 0.7

    def test_build_score_at_baseline_gives_midpoint(self):
        cfg = BSCDeploymentConfig.auto()
        predictor = BSCPredictor(cfg)
        predictor._baseline = 45.0

        score = predictor._build_score(
            perplexity=45.0,  # exactly at baseline
            n_chunk=20,
            n_ctx=100,
        )
        # Sigmoid(0) = 0.5
        assert score.score == pytest.approx(0.5, abs=0.01)

    def test_build_score_below_baseline_gives_low_score(self):
        cfg = BSCDeploymentConfig.auto()
        predictor = BSCPredictor(cfg)
        predictor._baseline = 45.0

        score = predictor._build_score(
            perplexity=5.0,   # much lower than baseline
            n_chunk=20,
            n_ctx=100,
        )
        assert score.score < 0.3

    def test_adaptive_baseline_drifts_toward_recent_perplexity(self):
        cfg = BSCDeploymentConfig.auto()
        predictor = BSCPredictor(cfg)
        predictor._baseline = 45.0

        initial = predictor._baseline
        for _ in range(50):
            predictor._build_score(perplexity=20.0, n_chunk=20, n_ctx=100)

        # After many low-perplexity samples, baseline should drift down
        assert predictor._baseline < initial

    def test_score_is_clipped_to_unit_interval(self):
        cfg = BSCDeploymentConfig.auto()
        predictor = BSCPredictor(cfg)
        predictor._baseline = 1.0

        for pp in [0.0, 0.001, 1e6]:
            s = predictor._build_score(pp, 20, 100)
            assert 0.0 <= s.score <= 1.0

    def test_score_contains_correct_token_counts(self):
        cfg = BSCDeploymentConfig.auto()
        predictor = BSCPredictor(cfg)
        s = predictor._build_score(50.0, n_chunk=17, n_ctx=83)
        assert s.token_count == 17
        assert s.context_tokens == 83


# ======================================================================
# 6. Integration — full pipeline smoke test (skipped if model absent)
# ======================================================================

@pytest.mark.asyncio
@pytest.mark.integration
class TestBSCIntegration:
    """
    Full-pipeline tests using a real (small) HuggingFace model.

    Skipped automatically in CI unless the model is cached locally.
    Run manually with:
        pytest tests/test_bsc_core.py -m integration
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("transformers or torch not installed")

        # Skip if the default model is not already cached locally.
        # This prevents CI from blocking on a network download.
        from pathlib import Path
        import os
        cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        model_slug = "models--Qwen--Qwen2.5-0.5B"
        if not (cache_dir / "hub" / model_slug).exists():
            pytest.skip(
                "Qwen/Qwen2.5-0.5B not cached locally — "
                "run `huggingface-cli download Qwen/Qwen2.5-0.5B` to enable integration tests"
            )

    async def test_high_surprise_chunk_is_promoted(self):
        cfg = BSCDeploymentConfig(
            mode=DeploymentMode.LOCAL_TRANSFORMERS,
            model_name="Qwen/Qwen2.5-0.5B",
            epsilon=0.50,
            similarity_threshold=0.90,
        )
        promoter = BSCPromoter.from_config(cfg)
        await promoter.warmup()

        # Context: generic intro; chunk: a genuinely unexpected pivot
        context = (
            "We are building a distributed AI protocol called PRSM. "
            "The codebase uses Python and FastAPI."
        )
        chunk = (
            "CRITICAL: The auth layer's JWT secret has been leaked in the public "
            "git history. We must rotate credentials immediately and audit all "
            "recent commits for other sensitive data."
        )

        decision = await promoter.process_chunk(
            chunk=chunk,
            context=context,
            source_agent="agent/security-20260326",
            session_id="integration-test",
        )

        assert decision.promoted is True, (
            f"Expected promotion but got: {decision.reason}"
        )

    async def test_routine_update_is_discarded(self):
        cfg = BSCDeploymentConfig(
            mode=DeploymentMode.LOCAL_TRANSFORMERS,
            model_name="Qwen/Qwen2.5-0.5B",
            epsilon=0.50,
        )
        promoter = BSCPromoter.from_config(cfg)
        await promoter.warmup()

        context = (
            "We are refactoring the PRSM database layer from SQLite to PostgreSQL. "
            "Current task: update the ORM models."
        )
        # Very predictable continuation — exactly what the model would expect next
        chunk = "Continuing to update the ORM models as planned."

        decision = await promoter.process_chunk(
            chunk=chunk,
            context=context,
            source_agent="agent/coder-20260326",
            session_id="integration-test",
        )

        assert not decision.promoted, (
            f"Expected discard but got promotion: {decision.reason}"
        )

    async def test_semantic_duplicate_is_caught(self):
        cfg = BSCDeploymentConfig(
            mode=DeploymentMode.LOCAL_TRANSFORMERS,
            model_name="Qwen/Qwen2.5-0.5B",
            epsilon=0.45,
            similarity_threshold=0.80,
        )
        promoter = BSCPromoter.from_config(cfg)
        await promoter.warmup()

        context = "Building PRSM."
        original = "The database migration requires PostgreSQL 14 or later."
        rephrasing = "You need PostgreSQL version 14+ for the database migration to work."

        # First: promote the original
        d1 = await promoter.process_chunk(
            chunk=original,
            context=context,
            source_agent="agent/coder",
            session_id="s",
        )

        # Second: the rephrasing should be caught as a semantic duplicate
        d2 = await promoter.process_chunk(
            chunk=rephrasing,
            context=context,
            source_agent="agent/coder",
            session_id="s",
        )

        # d1 may or may not promote (model-dependent), but if it did, d2 must not
        if d1.promoted:
            assert not d2.promoted, (
                "Semantic duplicate was not caught. "
                f"Original promoted with score={d1.surprise_score:.3f}; "
                f"Rephrasing got score={d2.surprise_score:.3f}, "
                f"similarity={d2.similarity_score:.3f}"
            )
