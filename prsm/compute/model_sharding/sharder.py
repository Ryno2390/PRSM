"""
ModelSharder — tensor partitioning with checksum verification.

Splits model weight tensors into shards for distributed inference,
with SHA-256 integrity checksums on every shard.
"""

import hashlib
import uuid
from typing import Dict, List

import numpy as np

from prsm.compute.model_sharding.models import (
    ModelShard,
    PipelineStakeTier,
    ShardedModel,
)


class ModelSharder:
    """Partition model tensors into shards for tensor-parallel inference."""

    @staticmethod
    def shard_tensor(tensor: np.ndarray, n_shards: int) -> List[np.ndarray]:
        """Split a tensor along its largest dimension.

        Args:
            tensor: The numpy array to partition.
            n_shards: Number of shards to produce.

        Returns:
            List of sub-arrays.
        """
        if n_shards <= 0:
            raise ValueError("n_shards must be positive")
        split_axis = int(np.argmax(tensor.shape))
        return list(np.array_split(tensor, n_shards, axis=split_axis))

    @staticmethod
    def reassemble_tensor(
        shards: List[np.ndarray], axis: int = 0
    ) -> np.ndarray:
        """Concatenate shards back into a single tensor.

        Args:
            shards: List of sub-arrays to concatenate.
            axis: Axis along which to concatenate (default 0).

        Returns:
            Reassembled numpy array.
        """
        return np.concatenate(shards, axis=axis)

    @staticmethod
    def shard_model(
        model_id: str,
        model_name: str,
        weight_tensors: Dict[str, np.ndarray],
        n_shards: int,
        stake_tier: PipelineStakeTier = PipelineStakeTier.STANDARD,
    ) -> ShardedModel:
        """Shard all weight tensors of a model.

        For each tensor, splits into *n_shards* pieces and creates
        :class:`ModelShard` objects with serialized numpy data and
        SHA-256 checksums.

        Args:
            model_id: Unique model identifier.
            model_name: Human-readable model name.
            weight_tensors: Mapping of layer name to numpy weight array.
            n_shards: Number of shards per tensor.
            stake_tier: Required stake tier for pipeline participation.

        Returns:
            A :class:`ShardedModel` containing all shards.
        """
        all_shards: List[ModelShard] = []

        for layer_name, tensor in weight_tensors.items():
            split_axis = int(np.argmax(tensor.shape))
            pieces = list(np.array_split(tensor, n_shards, axis=split_axis))

            for idx, piece in enumerate(pieces):
                raw = piece.tobytes()
                checksum = hashlib.sha256(raw).hexdigest()
                shard = ModelShard(
                    shard_id=f"{model_id}-{layer_name}-{idx}-{uuid.uuid4().hex[:8]}",
                    model_id=model_id,
                    shard_index=idx,
                    total_shards=n_shards,
                    tensor_data=raw,
                    tensor_shape=piece.shape,
                    layer_range=(0, 0),
                    size_bytes=len(raw),
                    checksum=checksum,
                )
                all_shards.append(shard)

        return ShardedModel(
            model_id=model_id,
            model_name=model_name,
            total_shards=n_shards,
            shards=all_shards,
            stake_tier=stake_tier,
        )
