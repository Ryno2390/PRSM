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

Pipeline (updated)
------------------
.. code-block::

    Text Chunk → Predictor → KL Filter → Semantic Dedup → Quality Gate → Promotion

The Quality Gate sits after semantic de-duplication.  It evaluates four
dimensions — factual consistency, source reliability, actionability, and
coherence — and only allows promotion if BOTH novelty AND quality pass.

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
from .event_bus import BSCEvent, EventBus, EventType
from .kl_filter import AdaptiveKLFilter, FilterDecision, KLFilter, KLFilterResult
from .predictor import BSCPredictor, SurpriseScore
from .quality_gate import QualityGate, QualityReport
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

    quality_report: Optional["QualityReport"] = None
    """None if the chunk was discarded before reaching the quality gate."""

    reason: str = ""
    """Final human-readable explanation of the outcome."""

    # Assigned after whiteboard write (Sub-phase 10.2 fills this in)
    whiteboard_index: Optional[int] = None


class BSCPromoter:
    """
    Full BSC pipeline: predictor → KL filter → semantic dedup → quality gate.

    Parameters
    ----------
    predictor : BSCPredictor
    kl_filter : KLFilter
    deduplicator : SemanticDeduplicator
    quality_gate : QualityGate, optional
        Quality gate filter.  If None, quality checking is disabled and the
        pipeline falls back to the pre-gate behaviour (KL + dedup only).
        Pass a :class:`~prsm.compute.nwtn.bsc.quality_gate.QualityGate`
        instance to enable the full four-dimension quality check.
    """

    def __init__(
        self,
        predictor: BSCPredictor,
        kl_filter: KLFilter,
        deduplicator: SemanticDeduplicator,
        quality_gate: Optional[QualityGate] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._predictor = predictor
        self._kl_filter = kl_filter
        self._deduplicator = deduplicator
        self._quality_gate = quality_gate
        self._event_bus = event_bus

        self._total_processed: int = 0
        self._total_promoted: int = 0
        self._total_quality_failed: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Optional[BSCDeploymentConfig] = None,
        enable_quality_gate: bool = True,
        quality_threshold: Optional[float] = None,
        event_bus: Optional[EventBus] = None,
    ) -> "BSCPromoter":
        """
        Convenience factory: build a fully-wired BSCPromoter from a single
        ``BSCDeploymentConfig`` object (or auto-detect defaults).

        Parameters
        ----------
        config : BSCDeploymentConfig, optional
            Deployment configuration.  Auto-detected if None.
        enable_quality_gate : bool
            Whether to attach a :class:`QualityGate` to the pipeline.
            Default True.  Set False to disable quality gating (e.g., for
            benchmarking the novelty-only pipeline).
        quality_threshold : float, optional
            Override the quality gate threshold.  If None, uses
            ``QualityGate``'s default (0.35).
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

        gate: Optional[QualityGate] = None
        if enable_quality_gate:
            gate_kwargs: dict = {}
            if quality_threshold is not None:
                gate_kwargs["threshold"] = quality_threshold
            gate = QualityGate(**gate_kwargs)

        return cls(
            predictor=predictor,
            kl_filter=kl_filter,
            deduplicator=deduplicator,
            quality_gate=gate,
            event_bus=event_bus,
        )

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
            decision = PromotionDecision(
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
            await self._publish_event(
                EventType.CHUNK_REJECTED, decision, session_id, reason="kl_filter",
            )
            return decision

        # ── Stage 3: Semantic De-duplication ────────────────────────────
        dedup_result: DedupResult = await self._deduplicator.check(chunk)

        if dedup_result.is_redundant:
            logger.debug(
                "BSC discard [dedup]: agent=%s similarity=%.3f reason=%s",
                source_agent, dedup_result.max_similarity, dedup_result.reason,
            )
            decision = PromotionDecision(
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
            await self._publish_event(
                EventType.CHUNK_REJECTED, decision, session_id, reason="semantic_dedup",
            )
            return decision

        # ── Stage 4: Quality Gate ───────────────────────────────────────
        quality_report: Optional[QualityReport] = None
        if self._quality_gate is not None:
            # Build a partial PromotionDecision to pass metadata to the gate
            _partial = PromotionDecision(
                promoted=False,  # placeholder; gate doesn't rely on this
                chunk=chunk,
                metadata=metadata,
                surprise_score=ss.score,
                raw_perplexity=ss.raw_perplexity,
                similarity_score=dedup_result.max_similarity,
                kl_result=kl_result,
                dedup_result=dedup_result,
            )
            quality_report = await self._quality_gate.evaluate(_partial)

            if not quality_report.passed:
                self._total_quality_failed += 1
                logger.debug(
                    "BSC discard [quality]: agent=%s overall=%.3f threshold=%.3f reason=%s",
                    source_agent, quality_report.overall_score,
                    quality_report.threshold, quality_report.reason,
                )
                decision = PromotionDecision(
                    promoted=False,
                    chunk=chunk,
                    metadata=metadata,
                    surprise_score=ss.score,
                    raw_perplexity=ss.raw_perplexity,
                    similarity_score=dedup_result.max_similarity,
                    kl_result=kl_result,
                    dedup_result=dedup_result,
                    quality_report=quality_report,
                    reason=quality_report.reason,
                )
                await self._publish_event(
                    EventType.CHUNK_REJECTED, decision, session_id, reason="quality_gate",
                )
                return decision

        # ── Promotion ───────────────────────────────────────────────────
        # Add to dedup index so future candidates are compared against this chunk
        await self._deduplicator.add_to_index(chunk)
        self._total_promoted += 1

        quality_suffix = ""
        if quality_report is not None:
            quality_suffix = (
                f", quality={quality_report.overall_score:.3f}"
                f"{'[fallback]' if quality_report.fallback_pass else ''}"
            )

        reason = (
            f"PROMOTED — surprise={ss.score:.3f} (perplexity={ss.raw_perplexity:.1f}), "
            f"similarity={dedup_result.max_similarity:.3f}{quality_suffix}"
            f" — novel high-quality chunk"
        )
        logger.info(
            "BSC promote: agent=%s score=%.3f similarity=%.3f%s tokens=%d",
            source_agent, ss.score, dedup_result.max_similarity,
            f" quality={quality_report.overall_score:.3f}" if quality_report else "",
            ss.token_count,
        )
        decision = PromotionDecision(
            promoted=True,
            chunk=chunk,
            metadata=metadata,
            surprise_score=ss.score,
            raw_perplexity=ss.raw_perplexity,
            similarity_score=dedup_result.max_similarity,
            kl_result=kl_result,
            dedup_result=dedup_result,
            quality_report=quality_report,
            reason=reason,
        )
        await self._publish_event(
            EventType.CHUNK_PROMOTED, decision, session_id,
        )
        return decision

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    async def _publish_event(
        self,
        event_type: EventType,
        decision: "PromotionDecision",
        session_id: str,
        **extra_data: object,
    ) -> None:
        """Publish a BSC event if an EventBus is configured.  Fire-and-forget."""
        if self._event_bus is None:
            return
        try:
            await self._event_bus.publish(BSCEvent(
                event_type=event_type,
                data={"decision": decision, **extra_data},
                session_id=session_id,
            ))
        except Exception:
            logger.exception(
                "BSCPromoter: failed to publish %s event",
                event_type.value,
            )

    def advance_round(
        self,
        round_number: int,
        dedup_keep_last_n_rounds: int = 2,
        dedup_entries_per_round: int = 10,
        session_id: str = "",
    ) -> dict:
        """
        Advance the BSC pipeline to the next round of a multi-round session.

        Coordinates two round-aware components:

        1. ``ProgressiveKLFilter.advance_round(round_number)`` — decreases
           epsilon so later rounds are more permissive (genuine discoveries
           in rounds 6-9 still clear the threshold even as context density
           increases).

        2. ``SemanticDeduplicator.advance_round(...)`` — evicts the oldest
           embeddings from the dedup index so it compares against RECENT
           context only, preventing over-filtering of valid new discoveries
           that happen to be semantically adjacent to early-round entries.

        Parameters
        ----------
        round_number : int
            The round about to start (0-indexed).
        dedup_keep_last_n_rounds : int
            How many rounds of dedup embeddings to retain (default 2).
        dedup_entries_per_round : int
            Approximate promoted entries per round (used to compute window).

        Returns
        -------
        dict
            Diagnostics: ``{"round": n, "epsilon": e, "dedup_evicted": k,
            "dedup_index_size": s}``.
        """
        from .kl_filter import ProgressiveKLFilter

        new_epsilon = self._kl_filter.epsilon
        if isinstance(self._kl_filter, ProgressiveKLFilter):
            self._kl_filter.advance_round(round_number)
            new_epsilon = self._kl_filter.epsilon

        evicted = self._deduplicator.advance_round(
            keep_last_n_rounds=dedup_keep_last_n_rounds,
            entries_per_round=dedup_entries_per_round,
        )

        logger.info(
            "BSCPromoter.advance_round: round=%d epsilon=%.3f "
            "dedup_evicted=%d dedup_size=%d",
            round_number, new_epsilon, evicted, self._deduplicator.index_size,
        )
        result = {
            "round": round_number,
            "epsilon": new_epsilon,
            "dedup_evicted": evicted,
            "dedup_index_size": self._deduplicator.index_size,
        }
        # Publish ROUND_ADVANCED event (fire-and-forget)
        if self._event_bus is not None and session_id:
            # Use asyncio.create_task for fire-and-forget
            import asyncio
            asyncio.create_task(
                self._event_bus.publish(BSCEvent(
                    event_type=EventType.ROUND_ADVANCED,
                    data=result,
                    session_id=session_id,
                ))
            )
        return result

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
        result: dict = {
            "total_processed": self._total_processed,
            "total_promoted": self._total_promoted,
            "total_quality_failed": self._total_quality_failed,
            "promotion_rate": round(self.promotion_rate, 4),
            "whiteboard_entries": self._deduplicator.index_size,
            "predictor_baseline_perplexity": round(
                self._predictor.baseline_perplexity, 2
            ),
            "kl_epsilon": self._kl_filter.epsilon,
        }
        if self._quality_gate is not None:
            result["quality_gate"] = self._quality_gate.stats
        return result
