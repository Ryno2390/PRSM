"""Phase 2.1: TopologyRandomizer.

Per-inference random node-to-shard assignment. Selected from a pool of
eligible providers for each inference, so consecutive inferences from
the same requester with the same prompt prefix do NOT reuse the same
node-to-shard mapping. This breaks the accumulation pattern needed for
activation-inversion attacks where a fixed subset of early-layer nodes
could reconstruct prompts over many observations.

Acceptance criterion (Vision doc §7 Line Item B): across 100 consecutive
inferences from the same requester, no single node appears in more than
10% of assignments for any given shard position, when the eligible pool
has >=10 nodes.
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


class InsufficientEligiblePoolError(RuntimeError):
    """Raised when the eligible pool is empty (can't assign any shard)."""


@dataclass(frozen=True)
class ShardAssignment:
    """One shard's assignment to a node for a specific inference."""
    shard_index: int
    node_id: str


class TopologyRandomizer:
    """Randomly assigns shard_index → node_id from an eligible pool.

    Uses `secrets.SystemRandom` for cryptographically-unpredictable
    selection. A downstream observer cannot predict future assignments
    even with knowledge of prior mappings — this is what defeats the
    activation-inversion attack that relies on repeat node observations.

    Sampling is WITH REPLACEMENT: a single node can serve multiple
    shards of one inference when the pool is small. For large pools the
    birthday-collision probability is low. Callers who require unique
    node-per-shard (e.g., tensor-parallel without collocation) should
    use `assign_unique()` — which raises if pool < num_shards.
    """

    def __init__(self, rng: Optional[secrets.SystemRandom] = None):
        self._rng = rng or secrets.SystemRandom()

    def assign(
        self,
        eligible_node_ids: Sequence[str],
        num_shards: int,
    ) -> List[ShardAssignment]:
        """Sample with replacement from the eligible pool.

        Returns one ShardAssignment per shard_index in [0, num_shards).
        Raises InsufficientEligiblePoolError on empty pool.
        """
        if not eligible_node_ids:
            raise InsufficientEligiblePoolError(
                f"cannot assign {num_shards} shards from empty eligible pool"
            )
        pool = list(eligible_node_ids)
        return [
            ShardAssignment(shard_index=i, node_id=self._rng.choice(pool))
            for i in range(num_shards)
        ]

    def assign_unique(
        self,
        eligible_node_ids: Sequence[str],
        num_shards: int,
    ) -> List[ShardAssignment]:
        """Sample WITHOUT replacement — one distinct node per shard.

        Raises InsufficientEligiblePoolError if pool < num_shards.
        """
        pool = list(eligible_node_ids)
        if len(pool) < num_shards:
            raise InsufficientEligiblePoolError(
                f"pool size {len(pool)} < requested shards {num_shards} "
                f"for unique assignment"
            )
        # Fisher-Yates on the first num_shards positions — unbiased and
        # doesn't build a full shuffled list when pool >> num_shards.
        selected: List[str] = []
        seen: set = set()
        while len(selected) < num_shards:
            cand = self._rng.choice(pool)
            if cand in seen:
                continue
            seen.add(cand)
            selected.append(cand)
        return [
            ShardAssignment(shard_index=i, node_id=selected[i])
            for i in range(num_shards)
        ]
