"""
Tensor Parallel Executor
========================

Coordinates parallel model shard execution across nodes
with ring all-reduce synchronization.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Any, Dict, List, Optional

from prsm.compute.model_sharding.models import ShardedModel, PipelineConfig

logger = logging.getLogger(__name__)


class TensorParallelExecutor:
    """Executes model shards in parallel across assigned nodes."""

    def __init__(
        self,
        confidential_executor=None,
        pipeline_config: Optional[PipelineConfig] = None,
    ):
        self._confidential_executor = confidential_executor
        self.config = pipeline_config or PipelineConfig()

    async def execute_parallel(
        self,
        sharded_model: ShardedModel,
        input_data: bytes,
        node_assignments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute all shards in parallel and aggregate results.

        Each shard is executed independently (tensor parallelism).
        Results are combined via all-reduce (sum then average).
        """
        started_at = time.time()
        shard_results = []
        errors = []

        # Fan out execution per shard
        tasks = []
        for assignment in node_assignments:
            shard_index = assignment.get("shard_index", 0)
            shard = sharded_model.get_shard_by_index(shard_index)
            if shard is None:
                errors.append(f"Shard {shard_index} not found")
                continue
            tasks.append(self._execute_shard(shard, input_data, assignment))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(str(result))
            elif result is not None:
                shard_results.append(result)

        # All-reduce aggregation
        aggregated = None
        if shard_results:
            arrays = []
            for r in shard_results:
                if isinstance(r, dict) and "output_array" in r:
                    arrays.append(np.array(r["output_array"]))
                elif isinstance(r, np.ndarray):
                    arrays.append(r)

            if arrays:
                aggregated = self.all_reduce(arrays)

        return {
            "status": "success" if shard_results else "failed",
            "shards_executed": len(shard_results),
            "shards_failed": len(errors),
            "errors": errors,
            "aggregated_output": aggregated.tolist() if aggregated is not None else None,
            "execution_time_seconds": time.time() - started_at,
        }

    async def _execute_shard(
        self,
        shard,
        input_data: bytes,
        assignment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single shard (placeholder for remote execution)."""
        # In production, this dispatches to the assigned node via Ring 2
        # For now, simulate local execution
        tensor = np.frombuffer(shard.tensor_data, dtype=np.float64).reshape(shard.tensor_shape)

        # Simple forward pass simulation: multiply input by shard weights
        try:
            input_array = np.frombuffer(input_data, dtype=np.float64)
            if input_array.size == 0:
                input_array = np.ones(tensor.shape[-1] if tensor.ndim > 1 else tensor.shape[0])

            if tensor.ndim == 2 and input_array.shape[0] == tensor.shape[1]:
                output = tensor @ input_array
            else:
                output = tensor.flatten()[:min(10, tensor.size)]
        except Exception:
            output = tensor.flatten()[:min(10, tensor.size)]

        return {
            "shard_index": shard.shard_index,
            "node_id": assignment.get("node_id", "local"),
            "output_array": output.tolist(),
        }

    @staticmethod
    def all_reduce(shard_outputs: List[np.ndarray]) -> np.ndarray:
        """Ring all-reduce: average across all shard outputs.

        In tensor parallelism, each shard produces a partial result.
        All-reduce combines them (typically sum for column-parallel,
        concatenate for row-parallel). We use average as a general default.
        """
        if not shard_outputs:
            return np.array([])

        # Pad to same length if needed
        max_len = max(a.size for a in shard_outputs)
        padded = []
        for a in shard_outputs:
            flat = a.flatten()
            if flat.size < max_len:
                flat = np.pad(flat, (0, max_len - flat.size))
            padded.append(flat)

        stacked = np.stack(padded)
        return np.mean(stacked, axis=0)
