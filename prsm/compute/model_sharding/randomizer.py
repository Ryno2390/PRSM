"""
PipelineRandomizer — randomized node assignment with pool enforcement.

Assigns model shards to randomly selected nodes from a sufficiently
large pool, with optional TEE filtering, to resist collusion attacks.
"""

import random
from typing import Dict, List


class PipelineRandomizer:
    """Randomly assign shards to nodes from a validated pool."""

    def __init__(self, min_pool_size: int = 20):
        self.min_pool_size = min_pool_size

    def assign_pipeline(
        self,
        shard_count: int,
        available_nodes: List[Dict],
        require_tee: bool = False,
    ) -> List[Dict]:
        """Assign shards to randomly selected nodes.

        Args:
            shard_count: Number of shards that need node assignments.
            available_nodes: List of node dicts, each with at least
                ``"node_id"`` and optionally ``"tee_enabled"``.
            require_tee: If True, only nodes with ``tee_enabled=True``
                are eligible.

        Returns:
            List of ``{"node_id": ..., "shard_index": ...}`` assignments.

        Raises:
            ValueError: If the eligible pool is smaller than
                :attr:`min_pool_size` or smaller than *shard_count*.
        """
        pool = available_nodes
        if require_tee:
            pool = [n for n in pool if n.get("tee_enabled", False)]

        self._validate_pool(pool, shard_count)

        selected = random.sample(pool, shard_count)
        return [
            {"node_id": node["node_id"], "shard_index": idx}
            for idx, node in enumerate(selected)
        ]

    def _validate_pool(
        self, available_nodes: List[Dict], shard_count: int
    ) -> None:
        """Ensure the node pool meets minimum size requirements.

        Raises:
            ValueError: If pool is too small.
        """
        if len(available_nodes) < self.min_pool_size:
            raise ValueError(
                f"Pool size {len(available_nodes)} is below the minimum "
                f"required pool size of {self.min_pool_size}"
            )
        if len(available_nodes) < shard_count:
            raise ValueError(
                f"Pool size {len(available_nodes)} is smaller than the "
                f"requested shard count {shard_count}"
            )
