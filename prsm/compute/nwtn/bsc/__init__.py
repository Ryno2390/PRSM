"""
prsm.compute.nwtn.bsc — Bayesian Surprise Compressor
=====================================================

The BSC is the core token-management innovation that makes flat Agent Team
architectures viable.  It monitors each agent's output stream and promotes
only high-information-density chunks to the shared Active Whiteboard,
keeping shared context lean regardless of team size or session length.

Pipeline
--------

    agent output chunk
          │
          ▼
    ┌─────────────┐
    │  Predictor  │  Evaluates cross-entropy loss of the chunk relative to
    │             │  the current compressed context (perplexity proxy for KL).
    └──────┬──────┘
           │  SurpriseScore (normalised 0–1, adaptive baseline)
           ▼
    ┌─────────────┐
    │  KL Filter  │  Threshold gate: score > epsilon → forward; else discard.
    │             │  AdaptiveKLFilter raises epsilon during burst periods.
    └──────┬──────┘
           │  (only high-surprise chunks reach here)
           ▼
    ┌──────────────────┐
    │ Semantic Dedup   │  Cosine-similarity check against whiteboard embeddings.
    │                  │  Redundant rephrasings are discarded despite high score.
    └──────┬───────────┘
           │
    PromotionDecision(promoted=True)
           │
           ▼
    Active Whiteboard  (Sub-phase 10.2)

Quick start
-----------
.. code-block:: python

    from prsm.compute.nwtn.bsc import BSCPromoter, BSCDeploymentConfig

    config = BSCDeploymentConfig.auto()          # auto-detect hardware
    promoter = BSCPromoter.from_config(config)
    await promoter.warmup()                      # pre-load models

    decision = await promoter.process_chunk(
        chunk="Auth layer now requires PostgreSQL — SQLite path is invalid.",
        context=whiteboard.compressed_state(),
        source_agent="agent/coder-20260326",
        session_id="sess-abc123",
    )

    if decision.promoted:
        await whiteboard.write(decision)         # Sub-phase 10.2
"""

from .deployment import BSCDeploymentConfig, DeploymentMode
from .kl_filter import AdaptiveKLFilter, FilterDecision, KLFilter, KLFilterResult, ProgressiveKLFilter
from .predictor import BSCPredictor, SurpriseScore
from .promoter import BSCPromoter, ChunkMetadata, PromotionDecision
from .semantic_dedup import DedupResult, SemanticDeduplicator

__all__ = [
    # Deployment
    "BSCDeploymentConfig",
    "DeploymentMode",
    # Predictor
    "BSCPredictor",
    "SurpriseScore",
    # KL Filter
    "KLFilter",
    "AdaptiveKLFilter",
    "ProgressiveKLFilter",
    "KLFilterResult",
    "FilterDecision",
    # Semantic De-duplication
    "SemanticDeduplicator",
    "DedupResult",
    # Promoter (main entry point)
    "BSCPromoter",
    "PromotionDecision",
    "ChunkMetadata",
]
