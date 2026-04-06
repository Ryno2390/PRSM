"""
Data models for tensor-parallel model sharding.

Defines shard metadata, sharded model containers, pipeline configuration,
and stake-tier enumerations for collusion-resistant inference pipelines.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PipelineStakeTier(Enum):
    """Stake tiers for pipeline participation.

    Each member stores (label, stake_required, slash_rate).
    """

    OPEN = ("open", 0, 0.0)
    STANDARD = ("standard", 5000, 0.5)
    PREMIUM = ("premium", 25000, 1.0)
    CRITICAL = ("critical", 50000, 1.0)

    def __init__(self, label: str, stake_required: int, slash_rate: float):
        self.label = label
        self.stake_required = stake_required
        self.slash_rate = slash_rate


@dataclass
class ModelShard:
    """A single shard of a partitioned model tensor."""

    shard_id: str
    model_id: str
    shard_index: int
    total_shards: int
    tensor_data: bytes
    tensor_shape: Tuple[int, ...]
    layer_range: Tuple[int, int] = (0, 0)
    size_bytes: int = 0
    checksum: str = ""

    def to_dict(self) -> Dict:
        """Serialize shard metadata to a dictionary."""
        return {
            "shard_id": self.shard_id,
            "model_id": self.model_id,
            "shard_index": self.shard_index,
            "total_shards": self.total_shards,
            "tensor_data": self.tensor_data.hex(),
            "tensor_shape": list(self.tensor_shape),
            "layer_range": list(self.layer_range),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelShard":
        """Deserialize a shard from a dictionary."""
        return cls(
            shard_id=data["shard_id"],
            model_id=data["model_id"],
            shard_index=data["shard_index"],
            total_shards=data["total_shards"],
            tensor_data=bytes.fromhex(data["tensor_data"]),
            tensor_shape=tuple(data["tensor_shape"]),
            layer_range=tuple(data["layer_range"]),
            size_bytes=data["size_bytes"],
            checksum=data["checksum"],
        )


@dataclass
class ShardedModel:
    """A model that has been partitioned into multiple shards."""

    model_id: str
    model_name: str
    total_shards: int
    shards: List[ModelShard] = field(default_factory=list)
    stake_tier: PipelineStakeTier = PipelineStakeTier.STANDARD
    created_at: float = field(default_factory=time.time)

    def get_shard_by_index(self, index: int) -> Optional[ModelShard]:
        """Return the shard at the given index, or None if not found."""
        for shard in self.shards:
            if shard.shard_index == index:
                return shard
        return None

    @property
    def total_size_bytes(self) -> int:
        """Total size across all shards in bytes."""
        return sum(s.size_bytes for s in self.shards)


@dataclass
class PipelineConfig:
    """Configuration for a sharded inference pipeline."""

    parallelism_degree: int = 4
    min_pool_size: int = 20
    require_tee: bool = False
    privacy_level: str = "standard"
    stake_tier: PipelineStakeTier = PipelineStakeTier.STANDARD
    enable_diversified_pipeline: bool = False
    max_latency_ms: int = 5000
