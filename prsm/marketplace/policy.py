"""Phase 3 Task 4: DispatchPolicy.

Requester-side policy that drives EligibilityFilter. All fields have
safe defaults so a caller who supplies `DispatchPolicy()` gets a
permissive-but-sane filter: any non-expired listing passes except the
anti-loss-leader price floor.

Defaults rationale (docs/2026-04-20-phase3-marketplace-design.md §8.4):
  - min_price_per_shard_ftns = 0.01 FTNS — a provider advertising less
    than this is either running a loss-leader attack (attract requests,
    degrade service) or misconfigured. The default rejects both.
  - min_reputation_score = 0.0 — allows new providers in (their score
    is 0.5 neutral per ReputationTracker); raises to 0.5 to gate on
    "proven decent" once you have history.
  - required_dtype = "float64" — matches Phase 2's executor contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DispatchPolicy:
    """Policy for marketplace provider selection.

    Consumed by EligibilityFilter.filter() to narrow a directory's
    listing set down to a policy-compliant subset.
    """
    max_price_per_shard_ftns: float = float("inf")
    min_price_per_shard_ftns: float = 0.01
    require_tee: bool = False
    min_stake_tier: str = "open"
    min_reputation_score: float = 0.0
    required_dtype: str = "float64"
    min_capacity_shards_per_sec: float = 0.0
    max_timeout_seconds: float = 30.0
    require_unique_providers: bool = False
    # Phase 7.1 Task 5: redundant-execution verification (Tier B). When
    # consensus_mode is None (the default), dispatch is single-provider
    # as in Phase 7. When set, the orchestrator routes the shard to k
    # providers in parallel and decides agreement by output-hash voting.
    #   - "majority": winning group >= (k // 2) + 1 (default)
    #   - "unanimous": all k must respond AND all must agree
    #   - "byzantine": reserved — raises NotImplementedError until 7.1x
    consensus_mode: Optional[str] = None
    consensus_k: int = 3
