"""
Swarm Compute
=============

Parallel map-reduce execution across semantically-sharded data.
Ring 3 of the Sovereign-Edge AI architecture.
"""

from prsm.compute.swarm.models import (
    SwarmJob,
    SwarmResult,
    SwarmStatus,
    MapReduceStrategy,
    ReduceLocation,
    ShardAssignment,
)
from prsm.compute.swarm.coordinator import SwarmCoordinator

__all__ = [
    "SwarmJob",
    "SwarmResult",
    "SwarmStatus",
    "MapReduceStrategy",
    "ReduceLocation",
    "ShardAssignment",
    "SwarmCoordinator",
]
