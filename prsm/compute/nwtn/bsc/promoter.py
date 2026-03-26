"""
BSC Promoter — Full Pipeline Orchestrator
==========================================

The Promoter is the single entry point for external callers.  It wires
together all three BSC stages:

    chunk → Predictor  →  KL Filter  →  Semantic Dedup  →  promote / discard

Each call to ``process_chunk()`` makes a binary promote/discard decision and
returns a ``PromotionDecision`` that callers (the Active Whiteboard, the
OpenClaw adapter) can act on.

The Promoter is deliberately storage-agnostic: it does not write to any
database or file.  The downstream whiteboard (Sub-phase 10.2) reads the
``PromotionDecision`` objects and persists the accepted chunks.  This keeps
the BSC Core testable in isolation.

Typical usage
-------------
.. code-block:: python

    from prsm.compute.nwtn.bsc import BSCPromoter, BSCDeploymentConfig

    config = BSCDeploymentConfig.auto()
    promoter = BSCPromoter.from_config(config)
    await promoter.warmup()

    decision = await promoter.process_chunk(
        chunk="Auth layer now requires PostgreSQL — SQLite migration plan is void.",
        context=whiteboard.compressed_state(),
        source_agent="agent/coder-20260326",
        session_id="sess-abc123",
    )

    if decision.promoted:
        await whiteboard.write(decision)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .deployment import BSCDeploymentConfig
from .kl_filter import AdaptiveKLFilter, FilterDecision, KLFilter, KLFilterResult
from .predictor import BSCPredictor, SurpriseScore
from .semantic_dedup import DedupResult, SemanticDeduplicator

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Provenance metadata attached to every processed chunk."""

    source_agent: str
    """Identifier of the agent that produced this chunk.
    Convention: ``"agent/<role>-<YYYYMMDD>"``, e.g. ``"agent/security-20260326"``."""

    session_id: str
    """Identifier of the current working session."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    extra: dict = field(default_factory=dict)
    """Arbitrary metadata for future use (e.g. git branch, skill name)."""


@dataclass
class PromotionDecision:
    """
    The complete output of one BSC pipeline run.

    A ``PromotionDecision`` with ``promoted=True`` should be written to the
    Active Whiteboard.  One with ``promoted=False`` is simply discarded.
    """

    promoted: bool
    """True if the chunk cleared both the KL filter and semantic de-duplication."""

    chunk: str
    """The text that was evaluated."""

    metadata: ChunkMetadata

    # Stage scores (always populated, regardless of outcome)
    surprise_score: float
    """Normalised surprise score from the predictor (0–1)."""

    raw_perplexity: float
    """Raw perplexity from the predictor."""

    similarity_score: float
    """Max cosine similarity with existing whiteboard entries (0–1)."""

    # Stage decisions
    kl_result: KLFilterResult
    dedup_result: Optional[DedupResult]
    """None if the KL filter discarded the chunk before reaching semantic dedup."""

    reason: str
    """Final human-readable explanation of the outcome."""

    # Assigned after whiteboard write (Sub-phase 10.2 fills this in)
    whiteboard_index: Optional[int] = None


class BSCPromoter:
    """
    Full BSC pipeline: predictor → KL filter → semantic dedup.

    Parameters
    ----------
    predictor : BSCPredictor
    kl_filter : KLFilter
    deduplicator : SemanticDeduplicator
    """

    def __init__(
        self,
        predictor: BSCPredictor,
        kl_filter: KLFilter,
        deduplicator: SemanticDeduplicator,
    ) -> None:
        self._predictor = predictor
        self._kl_filter = kl_filter
        self._deduplicator = deduplicator

        self._total_processed: int = 0
        self._total_promoted: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Optional[BSCDeploymentConfig] = None) -> "BSCPromoter":
        """
        Convenience factory: build a fully-wired BSCPromoter from a single
        ``BSCDeploymentConfig`` object (or auto-detect defaults).
        """
        cfg = config or BSCDeploymentConfig.auto()
        cfg.validate()

        predictor = BSCPredictor(cfg)
        kl_filter = AdaptiveKLFilter(epsilon=cfg.epsilon)
        deduplicator = SemanticDeduplicator(
            model_name=cfg.embedding_model,
            similarity_threshold=cfg.similarity_threshold,
            device=cfg.device,
        )
        return cls(predictor=predictor, kl_filter=kl_filter, deduplicator=deduplicator)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Pre-load all models so the first real call has no cold-start latency."""
        await self._predictor.warmup()
        # The deduplicator loads lazily on first use; trigger it here with a
        # short dummy string so the first real chunk isn't slow.
        await self._deduplicator.check("warmup")
        logger.info("BSCPromoter: warmup complete")

    async def process_chunk(
        self,
        chunk: str,
        context: str,
        source_agent: str,
        session_id: str,
        extra_metadata: Optional[dict] = None,
    ) -> PromotionDecision:
        """
        Run the full BSC pipeline on a new agent output chunk.

        Parameters
        ----------
        chunk : str
            The raw agent output to evaluate.
        context : str
            Current compressed whiteboard state.  Pass ``""`` for the first
            chunk of a session.
        source_agent : str
            Agent identifier (e.g. ``"agent/coder-20260326"``).
        session_id : str
            Session identifier.
        extra_metadata : dict, optional
            Additional provenance data to attach to the decision.

        Returns
        -------
        PromotionDecision
        """
        self._total_processed += 1
        metadata = ChunkMetadata(
            source_agent=source_agent,
            session_id=session_id,
            extra=extra_metadata or {},
        )

        # ── Stage 1: Predictor ──────────────────────────────────────────
        ss: SurpriseScore = await self._predictor.score_surprise(context, chunk)

        # ── Stage 2: KL Filter ─────────────────────────────────────────
        kl_result: KLFilterResult = self._kl_filter.evaluate(ss)

        if kl_result.decision != FilterDecision.PROMOTE:
            logger.debug(
                "BSC discard [KL]: agent=%s score=%.3f reason=%s",
                source_agent, ss.score, kl_result.reason,
            )
            return PromotionDecision(
                promoted=False,
                chunk=chunk,
                metadata=metadata,
                surprise_score=ss.score,
                raw_perplexity=ss.raw_perplexity,
                similarity_score=0.0,
                kl_result=kl_result,
                dedup_result=None,
                reason=kl_result.reason,
            )

        # ── Stage 3: Semantic De-duplication ────────────────────────────
        dedup_result: DedupResult = await self._deduplicator.check(chunk)

        if dedup_result.is_redundant:
            logger.debug(
                "BSC discard [dedup]: agent=%s similarity=%.3f reason=%s",
                source_agent, dedup_result.max_similarity, dedup_result.reason,
            )
            return PromotionDecision(
                promoted=False,
                chunk=chunk,
                metadata=metadata,
                surprise_score=ss.score,
                raw_perplexity=ss.raw_perplexity,
                similarity_score=dedup_result.max_similarity,
                kl_result=kl_result,
                dedup_result=dedup_result,
                reason=dedup_result.reason,
            )

        # ── Promotion ───────────────────────────────────────────────────
        # Add to dedup index so future candidates are compared against this chunk
        await self._deduplicator.add_to_index(chunk)
        self._total_promoted += 1

        reason = (
            f"PROMOTED — surprise={ss.score:.3f} (perplexity={ss.raw_perplexity:.1f}), "
            f"similarity={dedup_result.max_similarity:.3f} — novel high-surprise chunk"
        )
        logger.info(
            "BSC promote: agent=%s score=%.3f similarity=%.3f tokens=%d",
            source_agent, ss.score, dedup_result.max_similarity, ss.token_count,
        )
        return PromotionDecision(
            promoted=True,
            chunk=chunk,
            metadata=metadata,
            surprise_score=ss.score,
            raw_perplexity=ss.raw_perplexity,
            similarity_score=dedup_result.max_similarity,
            kl_result=kl_result,
            dedup_result=dedup_result,
            reason=reason,
        )

    def reset_session(self) -> None:
        """
        Clear the deduplication index at the start of a new working session.
        The predictor's adaptive baseline is preserved so it does not need to
        re-warm across sessions.
        """
        self._deduplicator.clear()
        logger.info("BSCPromoter: session reset — dedup index cleared")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def promotion_rate(self) -> float:
        """Fraction of processed chunks that were promoted (0–1)."""
        if self._total_processed == 0:
            return 0.0
        return self._total_promoted / self._total_processed

    @property
    def stats(self) -> dict:
        """Summary statistics for the current session."""
        return {
            "total_processed": self._total_processed,
            "total_promoted": self._total_promoted,
            "promotion_rate": round(self.promotion_rate, 4),
            "whiteboard_entries": self._deduplicator.index_size,
            "predictor_baseline_perplexity": round(
                self._predictor.baseline_perplexity, 2
            ),
            "kl_epsilon": self._kl_filter.epsilon,
        }
