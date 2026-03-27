"""
Quality Gate — Combined Novelty + Quality Filter
=================================================

The QualityGate is the BSC pipeline's final gatekeeper.  It sits immediately
after semantic de-duplication and enforces a two-factor promotion criterion:

    promoted = novel (KL filter passed) AND quality (QualityGate passes)

Motivation
----------
The KL filter asks "is this *novel*?".  The QualityGate asks "is this
*correct and useful*?".  Neither is sufficient alone:

- Novel but wrong  → blocked by QualityGate
- Correct but stale → already blocked by semantic dedup
- Novel AND correct → promoted to the whiteboard

Design
------
The QualityGate wraps a :class:`~prsm.compute.nwtn.bsc.quality_scorer.QualityScorer`
and adds:

- Per-session configurable quality threshold (like KL epsilon).
- Optional agent reputation lookup via the DomainAuthority system.
- Optional whiteboard entry loading for contradiction detection.
- Quality score logging for the downstream tuning loop.
- Graceful degradation: any QualityGate failure defaults to PASS, so
  the BSC pipeline never blocks on quality-scorer errors.

Usage
-----
.. code-block:: python

    from prsm.compute.nwtn.bsc.quality_gate import QualityGate

    gate = QualityGate(threshold=0.35)
    report = await gate.evaluate(
        promotion_decision=kl_decision,
        whiteboard_entries=whiteboard.recent_entries(),
    )
    if report.passed:
        await whiteboard.write(promotion_decision)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Coroutine, List, Optional, Sequence

from .quality_scorer import QualityScore, QualityScorer

if TYPE_CHECKING:
    from .promoter import PromotionDecision

logger = logging.getLogger(__name__)

# Default quality threshold — overall score must exceed this to pass.
# This is tunable per-session, matching the KL epsilon pattern.
DEFAULT_QUALITY_THRESHOLD = 0.35

# Type alias for the optional async agent-reputation callback
AgentReputationCallback = Optional[
    Callable[[str], Coroutine[Any, Any, Optional[float]]]
]


@dataclass
class QualityReport:
    """
    The complete output of one QualityGate evaluation.

    Consumers (the Promoter) inspect ``passed`` and, on pass, propagate the
    original ``PromotionDecision`` to the whiteboard.  The per-dimension
    scores are retained for the evaluator tuning loop.
    """

    passed: bool
    """True if the chunk cleared the quality threshold."""

    overall_score: float
    """Overall quality score in [0, 1]."""

    threshold: float
    """The quality threshold that was applied."""

    quality_score: QualityScore
    """Full per-dimension breakdown."""

    chunk_preview: str
    """First 120 characters of the evaluated chunk (for log readability)."""

    source_agent: str
    """Agent that produced the chunk."""

    session_id: str
    """Session identifier."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    reason: str = ""
    """Human-readable explanation of the pass/fail decision."""

    fallback_pass: bool = False
    """
    True if the gate defaulted to PASS due to a scoring error.
    This field exists to distinguish genuine passes from fallback passes
    in the tuning loop metrics.
    """


class QualityGate:
    """
    Quality gate for the BSC pipeline.

    Evaluates a chunk that has already passed the KL novelty filter and
    semantic de-duplication, and decides whether its quality is sufficient
    for whiteboard promotion.

    Parameters
    ----------
    threshold : float
        Minimum overall quality score required for promotion.
        Must be in (0, 1).  Default 0.35.
    scorer : QualityScorer, optional
        Custom scorer instance.  If None, a default :class:`QualityScorer`
        is created.
    agent_reputation_callback : async callable, optional
        ``async (agent_id: str) -> float | None`` — returns the agent's
        historical accuracy rate in [0, 1], or None if unknown.  Used by
        the source reliability dimension.  If not provided, source reliability
        defaults to 0.5 (neutral) for all agents.
    whiteboard_entries_callback : async callable, optional
        ``async () -> list[str]`` — returns the current whiteboard text
        entries for contradiction detection.  If not provided, factual
        consistency defaults to 1.0 (no entries to contradict).
    whiteboard_embeddings_callback : async callable, optional
        ``async () -> list[list[float]]`` — returns pre-computed embeddings
        for the whiteboard entries.  If provided alongside
        ``whiteboard_entries_callback``, enables embedding-based contradiction
        detection.  Otherwise falls back to keyword heuristics.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_QUALITY_THRESHOLD,
        scorer: Optional[QualityScorer] = None,
        agent_reputation_callback: AgentReputationCallback = None,
        whiteboard_entries_callback: Optional[
            Callable[[], Coroutine[Any, Any, List[str]]]
        ] = None,
        whiteboard_embeddings_callback: Optional[
            Callable[[], Coroutine[Any, Any, List[List[float]]]]
        ] = None,
    ) -> None:
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"QualityGate: threshold must be in (0, 1), got {threshold}"
            )
        self._threshold = threshold
        self._scorer = scorer or QualityScorer()
        self._reputation_cb = agent_reputation_callback
        self._entries_cb = whiteboard_entries_callback
        self._embeddings_cb = whiteboard_embeddings_callback

        # Session-level accounting
        self._total_evaluated: int = 0
        self._total_passed: int = 0
        self._total_fallback_passed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        promotion_decision: PromotionDecision,
        whiteboard_entries: Optional[Sequence[str]] = None,
        whiteboard_embeddings: Optional[List[List[float]]] = None,
        chunk_embedding: Optional[List[float]] = None,
    ) -> QualityReport:
        """
        Evaluate the quality of a chunk that has already passed the KL filter
        and semantic de-duplication.

        Parameters
        ----------
        promotion_decision : PromotionDecision
            The decision object produced by the KL filter + dedup stages.
            The chunk text and provenance metadata are read from here.
        whiteboard_entries : sequence of str, optional
            Current whiteboard entries for contradiction detection.  If None
            and a ``whiteboard_entries_callback`` was registered, it is called
            automatically.  If neither is available, factual consistency
            defaults to 1.0.
        whiteboard_embeddings : list of list of float, optional
            Pre-computed embeddings for ``whiteboard_entries``.  If None and
            a ``whiteboard_embeddings_callback`` was registered, it is called
            automatically.
        chunk_embedding : list of float, optional
            Pre-computed embedding for the chunk.  Pass when the semantic
            deduplicator has already computed this to avoid redundant work.

        Returns
        -------
        QualityReport
        """
        self._total_evaluated += 1
        pd = promotion_decision
        source_agent = pd.metadata.source_agent
        session_id = pd.metadata.session_id
        chunk = pd.chunk
        chunk_preview = chunk[:120].replace("\n", " ")

        try:
            # ── Gather inputs ────────────────────────────────────────────
            agent_accuracy = await self._get_agent_accuracy(source_agent)

            wb_entries = await self._get_whiteboard_entries(whiteboard_entries)
            wb_embeddings = await self._get_whiteboard_embeddings(
                whiteboard_embeddings, len(wb_entries)
            )

            # ── Score ────────────────────────────────────────────────────
            quality: QualityScore = self._scorer.score(
                chunk,
                source_agent=source_agent,
                raw_perplexity=pd.raw_perplexity,
                whiteboard_entries=wb_entries,
                whiteboard_embeddings=wb_embeddings if wb_embeddings else None,
                chunk_embedding=chunk_embedding,
                agent_accuracy_rate=agent_accuracy,
            )

            # ── Decision ─────────────────────────────────────────────────
            passed = quality.overall >= self._threshold
            if passed:
                self._total_passed += 1

            reason = self._build_reason(quality, passed)

            logger.info(
                "QualityGate: agent=%s session=%s overall=%.3f "
                "(factual=%.3f reliability=%.3f action=%.3f coherence=%.3f) "
                "threshold=%.3f → %s",
                source_agent, session_id,
                quality.overall,
                quality.factual_consistency,
                quality.source_reliability,
                quality.actionability,
                quality.coherence,
                self._threshold,
                "PASS" if passed else "FAIL",
            )

            return QualityReport(
                passed=passed,
                overall_score=quality.overall,
                threshold=self._threshold,
                quality_score=quality,
                chunk_preview=chunk_preview,
                source_agent=source_agent,
                session_id=session_id,
                reason=reason,
                fallback_pass=False,
            )

        except Exception as exc:  # noqa: BLE001
            # Graceful degradation: default to PASS on any error
            self._total_passed += 1
            self._total_fallback_passed += 1
            logger.warning(
                "QualityGate.evaluate: error during scoring (%s) — defaulting to PASS "
                "(agent=%s session=%s)",
                exc, source_agent, session_id,
            )
            # Build a neutral score for the fallback case
            fallback_quality = QualityScore(
                factual_consistency=0.5,
                source_reliability=0.5,
                actionability=0.5,
                coherence=0.5,
                overall=0.5,
                dimension_weights=self._scorer.weights,
                diagnostics={"fallback": True, "error": str(exc)},
            )
            return QualityReport(
                passed=True,
                overall_score=0.5,
                threshold=self._threshold,
                quality_score=fallback_quality,
                chunk_preview=chunk_preview,
                source_agent=source_agent,
                session_id=session_id,
                reason=f"FALLBACK PASS — quality scoring error: {exc}",
                fallback_pass=True,
            )

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Current quality threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not (0.0 < value < 1.0):
            raise ValueError(f"QualityGate: threshold must be in (0, 1), got {value}")
        self._threshold = value
        logger.info("QualityGate: threshold updated to %.3f", value)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def pass_rate(self) -> float:
        """Fraction of evaluated chunks that passed quality gate (0–1)."""
        if self._total_evaluated == 0:
            return 0.0
        return self._total_passed / self._total_evaluated

    @property
    def stats(self) -> dict:
        """Session-level statistics for the quality gate."""
        return {
            "total_evaluated": self._total_evaluated,
            "total_passed": self._total_passed,
            "total_fallback_passed": self._total_fallback_passed,
            "pass_rate": round(self.pass_rate, 4),
            "threshold": self._threshold,
            "scorer_weights": self._scorer.weights,
        }

    def reset_session(self) -> None:
        """Reset session-level counters."""
        self._total_evaluated = 0
        self._total_passed = 0
        self._total_fallback_passed = 0
        logger.info("QualityGate: session counters reset")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_agent_accuracy(self, source_agent: str) -> Optional[float]:
        """Retrieve agent accuracy rate via callback, or return None."""
        if self._reputation_cb is None:
            return None
        try:
            return await self._reputation_cb(source_agent)
        except Exception as exc:  # noqa: BLE001
            logger.debug("QualityGate._get_agent_accuracy: failed (%s)", exc)
            return None

    async def _get_whiteboard_entries(
        self, override: Optional[Sequence[str]]
    ) -> List[str]:
        """Resolve whiteboard entries from override or callback."""
        if override is not None:
            return list(override)
        if self._entries_cb is not None:
            try:
                return await self._entries_cb()
            except Exception as exc:  # noqa: BLE001
                logger.debug("QualityGate._get_whiteboard_entries: failed (%s)", exc)
        return []

    async def _get_whiteboard_embeddings(
        self, override: Optional[List[List[float]]], expected_len: int
    ) -> Optional[List[List[float]]]:
        """Resolve whiteboard embeddings from override or callback."""
        if override is not None:
            return override
        if self._embeddings_cb is not None:
            try:
                embs = await self._embeddings_cb()
                if len(embs) == expected_len:
                    return embs
                logger.debug(
                    "QualityGate: whiteboard embeddings length mismatch "
                    "(got %d, expected %d) — skipping embedding-based contradiction check",
                    len(embs), expected_len,
                )
                return None
            except Exception as exc:  # noqa: BLE001
                logger.debug("QualityGate._get_whiteboard_embeddings: failed (%s)", exc)
        return None

    @staticmethod
    def _build_reason(quality: QualityScore, passed: bool) -> str:
        """Build a human-readable reason string from a QualityScore."""
        status = "PASS" if passed else "FAIL"
        dims = (
            f"factual={quality.factual_consistency:.3f} "
            f"reliability={quality.source_reliability:.3f} "
            f"actionability={quality.actionability:.3f} "
            f"coherence={quality.coherence:.3f}"
        )
        return f"Quality {status} — overall={quality.overall:.3f} ({dims})"
