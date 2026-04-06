"""
Swarm Job Data Models
=====================

Data structures for parallel map-reduce across sharded datasets.
"""

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from prsm.compute.agents.models import AgentManifest


class SwarmStatus(str, Enum):
    PENDING = "pending"
    DISPATCHING = "dispatching"
    EXECUTING = "executing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    EXPIRED = "expired"


class ReduceLocation(str, Enum):
    ORIGIN = "origin"
    AGGREGATOR = "aggregator"


@dataclass
class MapReduceStrategy:
    reduce_location: ReduceLocation = ReduceLocation.ORIGIN
    quorum_pct: float = 0.8
    per_shard_timeout: int = 60
    global_timeout: int = 300
    budget_split: str = "equal"
    max_retries_per_shard: int = 2


@dataclass
class ShardAssignment:
    shard_cid: str
    agent_id: str = ""
    provider_id: str = ""
    status: str = "dispatched"
    result: Optional[Dict[str, Any]] = None
    pcu_used: float = 0.0
    ftns_spent: float = 0.0
    retries: int = 0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class SwarmJob:
    job_id: str
    query: str
    shard_cids: List[str]
    wasm_binary: bytes
    agent_manifest: AgentManifest
    budget_ftns: float
    strategy: MapReduceStrategy = field(default_factory=MapReduceStrategy)
    status: SwarmStatus = SwarmStatus.PENDING
    assignments: Dict[str, ShardAssignment] = field(default_factory=dict)
    completed_shards: List[str] = field(default_factory=list)
    failed_shards: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def budget_per_shard(self) -> float:
        if not self.shard_cids:
            return 0.0
        return self.budget_ftns / len(self.shard_cids)

    @property
    def quorum_count(self) -> int:
        return math.ceil(len(self.shard_cids) * self.strategy.quorum_pct)

    def is_quorum_met(self) -> bool:
        return len(self.completed_shards) >= self.quorum_count


@dataclass
class SwarmResult:
    job_id: str
    shard_results: Dict[str, Any]
    total_pcu: float = 0.0
    total_ftns_spent: float = 0.0
    total_shards: int = 0
    aggregation_time_seconds: float = 0.0
    aggregated_output: Optional[Dict[str, Any]] = None

    @property
    def shards_completed(self) -> int:
        return len(self.shard_results)

    @property
    def success_rate(self) -> float:
        if self.total_shards == 0:
            return 0.0
        return self.shards_completed / self.total_shards
