"""
Model Sharding
==============

Tensor-parallel model distribution with collusion resistance.
Ring 8 of the Sovereign-Edge AI architecture.
"""

from prsm.compute.model_sharding.models import (
    PipelineStakeTier,
    ModelShard,
    ShardedModel,
    PipelineConfig,
)
from prsm.compute.model_sharding.sharder import ModelSharder
from prsm.compute.model_sharding.randomizer import PipelineRandomizer
from prsm.compute.model_sharding.executor import TensorParallelExecutor
from prsm.compute.model_sharding.collision_detector import CollisionDetector

__all__ = [
    "PipelineStakeTier",
    "ModelShard",
    "ShardedModel",
    "PipelineConfig",
    "ModelSharder",
    "PipelineRandomizer",
    "TensorParallelExecutor",
    "CollisionDetector",
]
