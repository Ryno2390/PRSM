"""
BSC Pipeline — Complete Orchestrator
======================================

Full BSC pipeline orchestrator that wires:

    Predictor → KL Filter → Dedup → Promoter → EventBus → WhiteboardPush → LiveScribe

This is the single entry point for the complete, event-driven pipeline.
It integrates the BSC novelty filter with the LiveScribe distribution system,
making promoted chunks immediately available to all agents via their inboxes.

Without this orchestrator, the BSC and LiveScribe are decoupled components.
With it, they form a real-time collaboration backbone.

Typical usage
-------------
.. code-block:: python

    pipeline = BSCPipeline(
        whiteboard_store=store,
        meta_plan=plan,
        team=team,
        convergence_tracker=tracker,
        narrative_synthesizer=narrator,
        project_ledger=ledger,
        backend_registry=registry,
    )

    await pipeline.start()                         # wire event bus, start handlers
    decision = await pipeline.process_chunk(
        chunk="Auth layer now requires PostgreSQL...",
        source_agent="agent/coder-20260326",
        session_id="sess-abc123",
    )
    # ^^ automatically pushes to LiveScribe if promoted

    status = await pipeline.status()               # push stats, scribe status
    await pipeline.stop()                          # clean shutdown
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from .deployment import BSCDeploymentConfig
from .event_bus import EventBus
from .kl_filter import AdaptiveKLFilter
from .predictor import BSCPredictor
from .promoter import BSCPromoter, ChunkMetadata, PromotionDecision
from .quality_gate import QualityGate
from .semantic_dedup import SemanticDeduplicator
from .whiteboard_push import WhiteboardPushHandler

if TYPE_CHECKING:
    from ..team.live_scribe import LiveScribe
    from ..whiteboard.schema import WhiteboardStore
    from ..whiteboard.query import MetaPlan, ConvergenceTracker, NarrativeSynthesizer
    from ..economy.ledger import ProjectLedger
    from ..core.backend import BackendRegistry
    from ..team.team import AgentTeam

logger = logging.getLogger(__name__)


class BSCPipeline:
    """
    Full BSC pipeline orchestrator.

    Wires all BSC components together and connects them to a LiveScribe,
    creating a complete event-driven pipeline that pushes promoted chunks
    to agent inboxes.

    Parameters
    ----------
    whiteboard_store : WhiteboardStore
        Whiteboard persistence layer.
    meta_plan : MetaPlan
        Current meta‑plan (goals, milestones).
    team : AgentTeam
        Team roster (agent identities, roles, inboxes).
    convergence_tracker : ConvergenceTracker
        Monitors consensus and staleness.
    narrative_synthesizer : NarrativeSynthesizer
        Generates narrative summaries from whiteboard entries.
    project_ledger : ProjectLedger
        Token‑based accountability and incentives ledger.
    backend_registry : BackendRegistry, optional
        Backend registry for inference, embeddings, etc.
    config : BSCDeploymentConfig, optional
        BSC hardware/software configuration.  Auto‑detects if None.
    quality_threshold : float
        Quality gate threshold (default 0.35).
    enable_quality_gate : bool
        Whether to include the quality gate (default True).
    """

    def __init__(
        self,
        whiteboard_store: "WhiteboardStore",
        meta_plan: "MetaPlan",
        team: "AgentTeam",
        convergence_tracker: "ConvergenceTracker",
        narrative_synthesizer: "NarrativeSynthesizer",
        project_ledger: "ProjectLedger",
        backend_registry: Optional["BackendRegistry"] = None,
        config: Optional[BSCDeploymentConfig] = None,
        quality_threshold: float = 0.35,
        enable_quality_gate: bool = True,
    ) -> None:
        self._config = config or BSCDeploymentConfig.auto()
        self._config.validate()

        # 1. Event bus (core of the push system)
        self._event_bus = EventBus()

        # 2. BSC components
        self._predictor = BSCPredictor(self._config)
        self._kl_filter = AdaptiveKLFilter(epsilon=self._config.epsilon)
        self._dedup = SemanticDeduplicator(
            model_name=self._config.embedding_model,
            similarity_threshold=self._config.similarity_threshold,
            device=self._config.device,
        )

        gate: Optional[QualityGate] = None
        if enable_quality_gate:
            gate = QualityGate(threshold=quality_threshold)

        # 3. Promoter with event bus
        self._promoter = BSCPromoter(
            predictor=self._predictor,
            kl_filter=self._kl_filter,
            deduplicator=self._dedup,
            quality_gate=gate,
            event_bus=self._event_bus,
        )

        # 4. LiveScribe
        # Import here to avoid circular dependency
        from ..team.live_scribe import LiveScribe as LiveScribeImpl

        self._scribe = LiveScribeImpl(
            whiteboard_store=whiteboard_store,
            meta_plan=meta_plan,
            team=team,
            convergence_tracker=convergence_tracker,
            narrative_synthesizer=narrative_synthesizer,
            project_ledger=project_ledger,
            backend_registry=backend_registry,
        )

        # 5. Push handler (bridge)
        self._push_handler = WhiteboardPushHandler(
            event_bus=self._event_bus,
            live_scribe=self._scribe,
        )

        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize all async components and start the push handler."""
        if self._running:
            logger.warning("BSCPipeline: already running")
            return

        # Warm up models (predictor, dedup embeddings)
        await self._promoter.warmup()

        # Set up LiveScribe (connect to whiteboard, load inboxes)
        await self._scribe.setup()

        # Start push handler (subscribe to events)
        await self._push_handler.start()

        self._running = True
        logger.info(
            "BSCPipeline: started — predictor warm, dedup ready, push handler active "
            "(epsilon=%.3f, quality=%s)",
            self._config.epsilon,
            "enabled" if self._promoter._quality_gate else "disabled",
        )

    async def stop(self) -> None:
        """Shut down cleanly (unsubscribe, flush pending writes)."""
        if not self._running:
            return

        await self._push_handler.stop()
        # Note: we don't stop the scribe; it remains alive for other uses
        self._running = False
        logger.info("BSCPipeline: stopped")

    @property
    def running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def process_chunk(
        self,
        chunk: str,
        source_agent: str,
        session_id: str,
        context: str = "",
        extra_metadata: Optional[dict] = None,
    ) -> PromotionDecision:
        """
        Process a chunk through the full pipeline.

        The pipeline:
            1. Scores novelty (predictor)
            2. Filters by surprise (KL filter)
            3. Checks semantic redundancy (dedup)
            4. Evaluates quality (quality gate)
            5. **If promoted** → publishes CHUNK_PROMOTED event →
               WhiteboardPushHandler pushes to LiveScribe → agents notified

        Parameters
        ----------
        chunk : str
            Agent output to evaluate.
        source_agent : str
            Agent identifier (e.g., ``"agent/coder-20260326"``).
        session_id : str
            Session identifier.
        context : str
            Current compressed whiteboard state (default empty).
        extra_metadata : dict, optional
            Additional provenance data.

        Returns
        -------
        PromotionDecision
            Complete decision with all intermediate scores.
        """
        if not self._running:
            logger.warning(
                "BSCPipeline.process_chunk: pipeline not started; "
                "calling start() automatically",
            )
            await self.start()

        decision = await self._promoter.process_chunk(
            chunk=chunk,
            context=context,
            source_agent=source_agent,
            session_id=session_id,
            extra_metadata=extra_metadata,
        )

        # The decision is already published via the promoter's _publish_event()
        # The WhiteboardPushHandler picks it up automatically and pushes to scribe.
        return decision

    def advance_round(
        self,
        round_number: int,
        session_id: str,
        dedup_keep_last_n_rounds: int = 2,
        dedup_entries_per_round: int = 10,
    ) -> Dict[str, Any]:
        """
        Advance the pipeline to the next round.

        Triggers:
            - ProgressiveKLFilter epsilon adjustment
            - Dedup index window sliding
            - ROUND_ADVANCED event (monitoring/analytics)

        Parameters
        ----------
        round_number : int
            The round about to start (0‑indexed).
        session_id : str
            Session identifier (required for event).
        dedup_keep_last_n_rounds : int
            Dedup embeddings to retain (default 2).
        dedup_entries_per_round : int
            Approximate promoted entries per round (for window sizing).

        Returns
        -------
        dict
            Diagnostics: ``{round, epsilon, dedup_evicted, dedup_index_size}``.
        """
        return self._promoter.advance_round(
            round_number=round_number,
            dedup_keep_last_n_rounds=dedup_keep_last_n_rounds,
            dedup_entries_per_round=dedup_entries_per_round,
            session_id=session_id,
        )

    async def status(self) -> Dict[str, Any]:
        """
        Return comprehensive pipeline status.

        Includes:
            - BSC promoter statistics (processed, promoted, quality fails)
            - Push handler statistics (pushed, failed, conflicts)
            - Event bus statistics (published, delivery errors, subscribers)
            - LiveScribe status (inbox counts, priority breakdown)
            - Whiteboard entry count
        """
        status: Dict[str, Any] = {
            "running": self._running,
            "config": {
                "epsilon": self._config.epsilon,
                "embedding_model": self._config.embedding_model,
                "similarity_threshold": self._config.similarity_threshold,
                "quality_gate_enabled": self._promoter._quality_gate is not None,
            },
            "promoter": self._promoter.stats,
            "push_handler": self._push_handler.get_stats(),
            "event_bus": self._event_bus.get_stats(),
        }

        # Add scribe status if available
        try:
            scribe_status = await self._scribe.status()
            status["scribe"] = scribe_status
        except Exception as e:
            logger.debug("BSCPipeline: could not get scribe status: %s", e)
            status["scribe"] = {"error": str(e)}

        return status

    async def reset_session(self) -> None:
        """Clear dedup index at the start of a new session."""
        self._promoter.reset_session()
        logger.info("BSCPipeline: session reset — dedup index cleared")

    # ------------------------------------------------------------------
    # Accessors for integration
    # ------------------------------------------------------------------

    @property
    def promoter(self) -> BSCPromoter:
        """Access the raw BSC promoter (for advanced use)."""
        return self._promoter

    @property
    def event_bus(self) -> EventBus:
        """Access the internal event bus (for custom subscribers)."""
        return self._event_bus

    @property
    def scribe(self) -> "LiveScribe":
        """Access the LiveScribe instance."""
        return self._scribe