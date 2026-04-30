"""
Tests for BSC Quality Gate
==========================

Covers:
  - QualityScorer dimension scoring (factual consistency, source reliability, actionability, coherence)
  - QualityGate pass/fail decisions with configurable threshold
  - Graceful degradation (scoring errors → fallback PASS)
  - QualityReport structure and logging output
  - Integration with PromotionDecision
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

from prsm.compute.nwtn.bsc.quality_scorer import QualityScore, QualityScorer
from prsm.compute.nwtn.bsc.quality_gate import (
    QualityGate,
    QualityReport,
    DEFAULT_QUALITY_THRESHOLD,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_mock_pd(
    chunk: str = "Test chunk",
    source_agent: str = "agent/coder",
    session_id: str = "session/test",
    surprise_score: float = 0.85,
    raw_perplexity: float = 80.0,
):
    """Create a mock PromotionDecision with metadata.source_agent structure."""
    pd = MagicMock()
    pd.chunk = chunk
    pd.source_agent = source_agent
    pd.session_id = session_id
    pd.surprise_score = surprise_score
    pd.raw_perplexity = raw_perplexity
    # QualityGate reads pd.metadata.source_agent / .session_id
    pd.metadata.source_agent = source_agent
    pd.metadata.session_id = session_id
    return pd


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def scorer():
    """Default QualityScorer with standard weights."""
    return QualityScorer()


@pytest.fixture
def gate():
    """Default QualityGate with default threshold."""
    return QualityGate()


# ======================================================================
# QualityScorer Tests
# ======================================================================

class TestQualityScorer:
    """Tests for the QualityScorer dimension scoring."""

    def test_weights_must_sum_to_one(self):
        """Weights that don't sum to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            QualityScorer(factual_weight=0.5, reliability_weight=0.5, actionability_weight=0.5, coherence_weight=0.5)

    def test_default_weights_valid(self):
        """Default weights sum to 1.0 and scorer constructs."""
        s = QualityScorer()
        assert s._weights["factual"] == 0.30
        assert s._weights["reliability"] == 0.25
        assert s._weights["actionability"] == 0.25
        assert s._weights["coherence"] == 0.20

    def test_actionability_with_file_path(self, scorer):
        """Chunks containing file paths score positively on actionability.
        Formula: 0.30 (file) + 0.25 (decision) = 0.55 with 1 decision keyword."""
        result = scorer.score(
            "Updated prsm/compute/nwtn/team/planner.py with decision: add milestone tracking.",
            source_agent="agent/coder",
            raw_perplexity=80.0,
        )
        assert result.actionability > 0.5  # file + decision keyword

    def test_actionability_with_vague_text(self, scorer):
        """Vague meta-commentary scores low on actionability."""
        result = scorer.score(
            "Perhaps we should consider looking into this. It seems interesting, maybe we could explore it.",
            source_agent="agent/coder",
            raw_perplexity=80.0,
        )
        assert result.actionability < 0.5

    def test_actionability_with_multiple_decision_keywords(self, scorer):
        """Multiple decision keywords stack up to +0.15 bonus."""
        result = scorer.score(
            "Decision approved: migrate from SQLite to PostgreSQL for the project ledger.",
            source_agent="agent/coder",
            raw_perplexity=80.0,
        )
        # 2 decision keywords: 0.25 (has_decision) + 0.05 (stacking) = 0.30
        assert result.actionability >= 0.3

    def test_actionability_with_file_and_url(self, scorer):
        """File path + URL combine for higher actionability."""
        result = scorer.score(
            "Documentation updated: see https://example.com/docs for details in docs/guide.md.",
            source_agent="agent/coder",
            raw_perplexity=80.0,
        )
        # 0.30 (file) + 0.20 (url) = 0.50
        assert result.actionability >= 0.5

    def test_source_reliability_unknown_agent(self, scorer):
        """Unknown agents get neutral reliability (0.5)."""
        result = scorer.score(
            "Some chunk text.",
            source_agent="agent/unknown",
            raw_perplexity=80.0,
        )
        assert result.source_reliability == 0.5

    def test_source_reliability_high_accuracy(self, scorer):
        """High-accuracy agents get high reliability via non-linear mapping."""
        result = scorer.score(
            "Some chunk text.",
            source_agent="agent/senior",
            raw_perplexity=80.0,
            agent_accuracy_rate=0.95,
        )
        # Mapping: rate > 0.5 → 0.3 + 1.4 * (0.95 - 0.5) = 0.93
        assert result.source_reliability == pytest.approx(0.93, abs=0.01)

    def test_source_reliability_low_accuracy(self, scorer):
        """Low-accuracy agents get low reliability via non-linear mapping."""
        result = scorer.score(
            "Some chunk text.",
            source_agent="agent/novice",
            raw_perplexity=80.0,
            agent_accuracy_rate=0.2,
        )
        # Mapping: rate <= 0.5 → 0.6 * 0.2 = 0.12
        assert result.source_reliability == pytest.approx(0.12, abs=0.01)

    def test_coherence_sweet_spot(self, scorer):
        """Perplexity in the sweet spot (20-200) gets high coherence."""
        result = scorer.score(
            "Some chunk text.",
            source_agent="agent/coder",
            raw_perplexity=100.0,
        )
        assert result.coherence == 1.0

    def test_coherence_at_floor(self, scorer):
        """Perplexity at floor (10) returns exactly 1.0."""
        result = scorer.score(
            "Some chunk text.",
            source_agent="agent/coder",
            raw_perplexity=10.0,
        )
        assert result.coherence == 1.0

    def test_coherence_below_floor(self, scorer):
        """Very low perplexity gets proportional low score (p / floor)."""
        result = scorer.score(
            "Some chunk text.",
            source_agent="agent/coder",
            raw_perplexity=2.0,  # 2/10 = 0.2
        )
        assert result.coherence == pytest.approx(0.2, abs=0.01)

    def test_coherence_above_ceiling(self, scorer):
        """Very high perplexity (>500) gets 0 coherence."""
        result = scorer.score(
            "Some chunk text.",
            source_agent="agent/coder",
            raw_perplexity=600.0,
        )
        assert result.coherence == 0.0

    def test_overall_is_weighted_sum(self, scorer):
        """Overall score is the weighted sum of dimensions."""
        result = scorer.score(
            "Updated planner.py with new milestone tracking.",
            source_agent="agent/coder",
            raw_perplexity=100.0,
            agent_accuracy_rate=0.8,
        )
        expected = (
            0.30 * result.factual_consistency
            + 0.25 * result.source_reliability
            + 0.25 * result.actionability
            + 0.20 * result.coherence
        )
        assert abs(result.overall - expected) < 1e-6

    def test_factual_consistency_no_whiteboard(self, scorer):
        """With no whiteboard entries, factual consistency is 1.0."""
        result = scorer.score(
            "Some text.",
            source_agent="agent/coder",
            raw_perplexity=100.0,
            whiteboard_entries=[],
        )
        assert result.factual_consistency == 1.0

    def test_score_returns_quality_score(self, scorer):
        """score() always returns a QualityScore instance."""
        result = scorer.score(
            "Test chunk.",
            source_agent="agent/coder",
            raw_perplexity=80.0,
        )
        assert isinstance(result, QualityScore)
        for dim in ["factual_consistency", "source_reliability", "actionability", "coherence", "overall"]:
            val = getattr(result, dim)
            assert 0.0 <= val <= 1.0, f"{dim} = {val} not in [0, 1]"


# ======================================================================
# QualityGate Tests
# ======================================================================

class TestQualityGate:
    """Tests for QualityGate pass/fail decisions."""

    @pytest.mark.asyncio
    async def test_high_quality_passes(self, gate):
        """A high-quality chunk passes the gate."""
        pd = _make_mock_pd(
            chunk="Fixed auth bug in prsm/core/auth.py. Decision: switch to bcrypt with cost factor 12. Deployed to staging.",
            raw_perplexity=100.0,
        )
        report = await gate.evaluate(pd)
        assert report.passed is True
        assert report.overall_score > DEFAULT_QUALITY_THRESHOLD

    @pytest.mark.asyncio
    async def test_low_quality_blocked(self):
        """A vague, low-quality chunk is blocked by high threshold."""
        gate = QualityGate(threshold=0.7)
        pd = _make_mock_pd(
            chunk="Hmm, maybe we should think about this perhaps.",
            source_agent="agent/novice",
            raw_perplexity=80.0,
        )
        report = await gate.evaluate(pd)
        assert report.passed is False
        assert report.overall_score < 0.7

    @pytest.mark.asyncio
    async def test_graceful_degradation_returns_pass(self):
        """If scoring fails, gate defaults to PASS (fallback)."""
        gate = QualityGate(threshold=0.99)
        broken_scorer = MagicMock()
        broken_scorer.score.side_effect = RuntimeError("embedding service down")
        gate._scorer = broken_scorer

        pd = _make_mock_pd()
        report = await gate.evaluate(pd)
        assert report.passed is True
        assert report.fallback_pass is True

    @pytest.mark.asyncio
    async def test_quality_report_structure(self, gate):
        """QualityReport has all expected fields."""
        pd = _make_mock_pd(
            chunk="Implemented the new caching layer in cache.py. Decision: use LRU with 10k max entries.",
            source_agent="agent/coder-20260326",
            session_id="session/test-001",
            raw_perplexity=100.0,
        )
        report = await gate.evaluate(pd)
        assert isinstance(report, QualityReport)
        assert isinstance(report.timestamp, datetime)
        assert report.chunk_preview  # non-empty
        assert report.source_agent == "agent/coder-20260326"
        assert report.session_id == "session/test-001"
        assert isinstance(report.quality_score, QualityScore)
        assert report.threshold == DEFAULT_QUALITY_THRESHOLD

    @pytest.mark.asyncio
    async def test_custom_threshold(self):
        """Custom threshold changes pass/fail boundary."""
        pd = _make_mock_pd(
            chunk="Updated prsm/compute/nwtn/team/planner.py. Decision: add new milestones.",
            raw_perplexity=100.0,
        )
        low_gate = QualityGate(threshold=0.1)
        high_gate = QualityGate(threshold=0.9)

        low_report = await low_gate.evaluate(pd)
        high_report = await high_gate.evaluate(pd)

        assert low_report.threshold == 0.1
        assert high_report.threshold == 0.9
        # Low threshold should pass more easily
        assert low_report.passed is True

    @pytest.mark.asyncio
    async def test_agent_reputation_callback(self):
        """Agent reputation callback feeds into source reliability."""
        gate = QualityGate(
            agent_reputation_callback=AsyncMock(return_value=0.9),
        )
        pd = _make_mock_pd(
            chunk="Deployed v2.1 to production.",
            source_agent="agent/senior",
            raw_perplexity=100.0,
        )
        report = await gate.evaluate(pd)
        # rate=0.9 → 0.3 + 1.4*(0.9-0.5) = 0.86
        assert report.quality_score.source_reliability == pytest.approx(0.86, abs=0.01)

    @pytest.mark.asyncio
    async def test_quality_report_reason(self, gate):
        """Report contains a human-readable reason."""
        pd = _make_mock_pd(raw_perplexity=100.0)
        report = await gate.evaluate(pd)
        assert report.reason  # non-empty
        assert "quality" in report.reason.lower() or "pass" in report.reason.lower()
