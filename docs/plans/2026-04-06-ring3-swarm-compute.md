# Ring 3 — "The Swarm" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A large dataset can be semantically sharded across the network by meaning, and a single query triggers parallel agent execution across all nodes holding relevant shards. Results are collected by an aggregator and returned as a single response.

**Architecture:** A `SemanticSharder` wraps the existing `ContentSharder` to add embedding-based clustering before byte-level sharding. A `SwarmJob` dataclass represents a parallel fan-out of Ring 2 dispatches. A `SwarmCoordinator` orchestrates the lifecycle: decompose query → find relevant shards → fan out agents → collect results → aggregate → settle. The existing settler registry provides aggregator node selection.

**Tech Stack:** Existing PRSM infrastructure (IPFS sharding, vector store, agent dispatcher, settler registry, content economy). `scikit-learn` for k-means clustering (already available via numpy dependency). No new external dependencies.

**Scope note:** This plan covers the core swarm pipeline. Edge caching (the "Netflix model" pre-staging) is deferred to Ring 6 hardening — the schema and replication tracking are in place, but the off-peak pre-fetch loop is not built here.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `prsm/data/semantic_shard.py` | `SemanticSharder` — embedding-based record clustering + shard manifest |
| Create | `prsm/data/shard_models.py` | `ShardManifest` (semantic), `SemanticShard`, `ShardQuery` dataclasses |
| Create | `prsm/compute/swarm/__init__.py` | Package exports |
| Create | `prsm/compute/swarm/models.py` | `SwarmJob`, `SwarmResult`, `MapReduceStrategy` |
| Create | `prsm/compute/swarm/coordinator.py` | `SwarmCoordinator` — parallel dispatch + aggregation lifecycle |
| Modify | `prsm/node/node.py` | Wire SwarmCoordinator into PRSMNode |
| Create | `tests/unit/test_semantic_shard.py` | Semantic sharding tests |
| Create | `tests/unit/test_swarm_models.py` | Swarm job data model tests |
| Create | `tests/unit/test_swarm_coordinator.py` | Coordinator lifecycle tests |
| Create | `tests/integration/test_ring3_swarm.py` | End-to-end swarm smoke test |

---

### Task 1: Semantic Shard Data Models

**Files:**
- Create: `prsm/data/shard_models.py`
- Test: `tests/unit/test_semantic_shard.py`

- [ ] **Step 1: Create test file**

Create `tests/unit/test_semantic_shard.py`:

```python
"""Tests for semantic sharding data models."""

import pytest
from prsm.data.shard_models import (
    SemanticShard,
    SemanticShardManifest,
    ShardQuery,
)


class TestSemanticShard:
    def test_shard_creation(self):
        shard = SemanticShard(
            shard_id="shard-001",
            parent_dataset="dataset-abc",
            cid="QmShard001",
            centroid=[0.1, 0.2, 0.3],
            record_count=500,
            size_bytes=1024 * 1024,
            keywords=["electric vehicles", "north carolina"],
        )
        assert shard.shard_id == "shard-001"
        assert shard.record_count == 500
        assert len(shard.centroid) == 3

    def test_shard_to_dict_roundtrip(self):
        shard = SemanticShard(
            shard_id="shard-002",
            parent_dataset="ds-1",
            cid="QmShard002",
            centroid=[0.5, 0.6],
            record_count=100,
            size_bytes=2048,
            keywords=["test"],
        )
        d = shard.to_dict()
        restored = SemanticShard.from_dict(d)
        assert restored.shard_id == shard.shard_id
        assert restored.centroid == shard.centroid
        assert restored.keywords == ["test"]


class TestSemanticShardManifest:
    def test_manifest_creation(self):
        shards = [
            SemanticShard(
                shard_id=f"s-{i}",
                parent_dataset="ds-1",
                cid=f"QmShard{i}",
                centroid=[float(i) * 0.1],
                record_count=100,
                size_bytes=1024,
                keywords=[f"topic-{i}"],
            )
            for i in range(5)
        ]
        manifest = SemanticShardManifest(
            dataset_id="ds-1",
            total_records=500,
            total_size_bytes=5120,
            shards=shards,
        )
        assert len(manifest.shards) == 5
        assert manifest.total_records == 500

    def test_manifest_find_relevant_shards(self):
        shards = [
            SemanticShard(
                shard_id="ev-shard",
                parent_dataset="ds-1",
                cid="QmEV",
                centroid=[1.0, 0.0, 0.0],
                record_count=100,
                size_bytes=1024,
                keywords=["electric vehicles"],
            ),
            SemanticShard(
                shard_id="gas-shard",
                parent_dataset="ds-1",
                cid="QmGas",
                centroid=[0.0, 1.0, 0.0],
                record_count=100,
                size_bytes=1024,
                keywords=["gasoline"],
            ),
            SemanticShard(
                shard_id="hybrid-shard",
                parent_dataset="ds-1",
                cid="QmHybrid",
                centroid=[0.7, 0.3, 0.0],
                record_count=100,
                size_bytes=1024,
                keywords=["hybrid"],
            ),
        ]
        manifest = SemanticShardManifest(
            dataset_id="ds-1",
            total_records=300,
            total_size_bytes=3072,
            shards=shards,
        )
        # Query embedding close to EV shard
        query_embedding = [0.9, 0.1, 0.0]
        relevant = manifest.find_relevant_shards(query_embedding, top_k=2)
        assert len(relevant) == 2
        # EV shard should be first (closest to query)
        assert relevant[0].shard_id == "ev-shard"

    def test_manifest_to_dict_roundtrip(self):
        shards = [
            SemanticShard(
                shard_id="s-0",
                parent_dataset="ds-1",
                cid="Qm0",
                centroid=[0.1],
                record_count=50,
                size_bytes=512,
                keywords=[],
            ),
        ]
        manifest = SemanticShardManifest(
            dataset_id="ds-1",
            total_records=50,
            total_size_bytes=512,
            shards=shards,
        )
        d = manifest.to_dict()
        restored = SemanticShardManifest.from_dict(d)
        assert restored.dataset_id == "ds-1"
        assert len(restored.shards) == 1


class TestShardQuery:
    def test_query_creation(self):
        query = ShardQuery(
            query_text="EV adoption trends",
            query_embedding=[0.9, 0.1, 0.0],
            top_k=5,
            min_similarity=0.7,
        )
        assert query.top_k == 5
        assert query.min_similarity == 0.7

    def test_query_defaults(self):
        query = ShardQuery(
            query_text="test",
            query_embedding=[0.5],
        )
        assert query.top_k == 10
        assert query.min_similarity == 0.5
```

- [ ] **Step 2: Run tests — verify fail**

Run: `python -m pytest tests/unit/test_semantic_shard.py::TestSemanticShard::test_shard_creation -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement shard_models.py**

Create `prsm/data/shard_models.py`:

```python
"""
Semantic Shard Data Models
==========================

Data structures for semantically-sharded datasets. Records are clustered
by embedding similarity into neighborhoods, each stored as an IPFS shard.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SemanticShard:
    """A single shard of a semantically-partitioned dataset."""

    shard_id: str
    parent_dataset: str
    cid: str  # IPFS content address
    centroid: List[float]  # Embedding centroid of this neighborhood
    record_count: int
    size_bytes: int
    keywords: List[str] = field(default_factory=list)
    providers: List[str] = field(default_factory=list)  # Node IDs hosting this shard
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "parent_dataset": self.parent_dataset,
            "cid": self.cid,
            "centroid": self.centroid,
            "record_count": self.record_count,
            "size_bytes": self.size_bytes,
            "keywords": self.keywords,
            "providers": self.providers,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SemanticShard":
        return cls(
            shard_id=d["shard_id"],
            parent_dataset=d["parent_dataset"],
            cid=d["cid"],
            centroid=d["centroid"],
            record_count=d.get("record_count", 0),
            size_bytes=d.get("size_bytes", 0),
            keywords=d.get("keywords", []),
            providers=d.get("providers", []),
            created_at=d.get("created_at", time.time()),
        )


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class SemanticShardManifest:
    """Manifest for a semantically-sharded dataset."""

    dataset_id: str
    total_records: int
    total_size_bytes: int
    shards: List[SemanticShard] = field(default_factory=list)
    embedding_dimension: int = 0
    created_at: float = field(default_factory=time.time)

    def find_relevant_shards(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[SemanticShard]:
        """Find shards whose centroids are closest to the query embedding."""
        scored = []
        for shard in self.shards:
            sim = _cosine_similarity(query_embedding, shard.centroid)
            if sim >= min_similarity:
                scored.append((sim, shard))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [shard for _, shard in scored[:top_k]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "total_records": self.total_records,
            "total_size_bytes": self.total_size_bytes,
            "shards": [s.to_dict() for s in self.shards],
            "embedding_dimension": self.embedding_dimension,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SemanticShardManifest":
        return cls(
            dataset_id=d["dataset_id"],
            total_records=d["total_records"],
            total_size_bytes=d["total_size_bytes"],
            shards=[SemanticShard.from_dict(s) for s in d.get("shards", [])],
            embedding_dimension=d.get("embedding_dimension", 0),
            created_at=d.get("created_at", time.time()),
        )


@dataclass
class ShardQuery:
    """A query against a semantically-sharded dataset."""

    query_text: str
    query_embedding: List[float]
    top_k: int = 10
    min_similarity: float = 0.5
    dataset_id: Optional[str] = None
```

- [ ] **Step 4: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_semantic_shard.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/data/shard_models.py tests/unit/test_semantic_shard.py
git commit -m "feat(ring3): SemanticShard + ShardManifest + cosine similarity search"
```

---

### Task 2: Swarm Job Data Models

**Files:**
- Create: `prsm/compute/swarm/__init__.py`
- Create: `prsm/compute/swarm/models.py`
- Test: `tests/unit/test_swarm_models.py`

- [ ] **Step 1: Create package + test file**

```bash
mkdir -p prsm/compute/swarm
```

Create `tests/unit/test_swarm_models.py`:

```python
"""Tests for Swarm Job data models."""

import pytest
from prsm.compute.swarm.models import (
    SwarmJob,
    SwarmResult,
    SwarmStatus,
    MapReduceStrategy,
    ReduceLocation,
    ShardAssignment,
)
from prsm.compute.agents.models import AgentManifest


class TestMapReduceStrategy:
    def test_default_strategy(self):
        strategy = MapReduceStrategy()
        assert strategy.reduce_location == ReduceLocation.ORIGIN
        assert strategy.quorum_pct == 0.8
        assert strategy.per_shard_timeout == 60

    def test_custom_strategy(self):
        strategy = MapReduceStrategy(
            reduce_location=ReduceLocation.AGGREGATOR,
            quorum_pct=0.5,
            per_shard_timeout=120,
            global_timeout=600,
        )
        assert strategy.reduce_location == ReduceLocation.AGGREGATOR
        assert strategy.quorum_pct == 0.5


class TestSwarmJob:
    def test_job_creation(self):
        job = SwarmJob(
            job_id="swarm-001",
            query="EV adoption trends",
            shard_cids=["QmA", "QmB", "QmC"],
            wasm_binary=b"\x00asm\x01\x00\x00\x00",
            agent_manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=10.0,
        )
        assert job.job_id == "swarm-001"
        assert len(job.shard_cids) == 3
        assert job.budget_ftns == 10.0
        assert job.status == SwarmStatus.PENDING

    def test_job_budget_per_shard(self):
        job = SwarmJob(
            job_id="swarm-002",
            query="test",
            shard_cids=["QmA", "QmB", "QmC", "QmD"],
            wasm_binary=b"\x00asm\x01\x00\x00\x00",
            agent_manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=20.0,
        )
        assert job.budget_per_shard == 5.0  # 20.0 / 4 shards

    def test_job_quorum_count(self):
        job = SwarmJob(
            job_id="swarm-003",
            query="test",
            shard_cids=[f"Qm{i}" for i in range(10)],
            wasm_binary=b"\x00asm\x01\x00\x00\x00",
            agent_manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=50.0,
            strategy=MapReduceStrategy(quorum_pct=0.8),
        )
        assert job.quorum_count == 8  # ceil(10 * 0.8)

    def test_job_is_quorum_met(self):
        job = SwarmJob(
            job_id="swarm-004",
            query="test",
            shard_cids=["QmA", "QmB"],
            wasm_binary=b"\x00asm\x01\x00\x00\x00",
            agent_manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=10.0,
            strategy=MapReduceStrategy(quorum_pct=0.5),
        )
        # Need 1 of 2 (ceil(2 * 0.5))
        job.completed_shards.append("QmA")
        assert job.is_quorum_met()


class TestSwarmResult:
    def test_result_creation(self):
        result = SwarmResult(
            job_id="swarm-001",
            shard_results={"QmA": {"count": 42}, "QmB": {"count": 58}},
            total_pcu=1.5,
            total_ftns_spent=5.0,
        )
        assert len(result.shard_results) == 2
        assert result.total_pcu == 1.5
        assert result.shards_completed == 2

    def test_result_success_rate(self):
        result = SwarmResult(
            job_id="swarm-002",
            shard_results={"QmA": {"v": 1}, "QmB": {"v": 2}},
            total_shards=5,
            total_pcu=2.0,
            total_ftns_spent=8.0,
        )
        assert result.success_rate == 0.4  # 2/5


class TestShardAssignment:
    def test_assignment_creation(self):
        assignment = ShardAssignment(
            shard_cid="QmTestShard",
            agent_id="agent-001",
            provider_id="provider-abc",
        )
        assert assignment.shard_cid == "QmTestShard"
        assert assignment.status == "dispatched"
```

- [ ] **Step 2: Run tests — verify fail**

Run: `python -m pytest tests/unit/test_swarm_models.py::TestMapReduceStrategy -v`
Expected: FAIL

- [ ] **Step 3: Implement swarm models**

Create `prsm/compute/swarm/__init__.py`:

```python
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

__all__ = [
    "SwarmJob",
    "SwarmResult",
    "SwarmStatus",
    "MapReduceStrategy",
    "ReduceLocation",
    "ShardAssignment",
]
```

Create `prsm/compute/swarm/models.py`:

```python
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
    """Lifecycle status of a swarm job."""
    PENDING = "pending"
    DISPATCHING = "dispatching"
    EXECUTING = "executing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    PARTIAL = "partial"  # Quorum met but not all shards completed
    FAILED = "failed"
    EXPIRED = "expired"


class ReduceLocation(str, Enum):
    """Where aggregation happens."""
    ORIGIN = "origin"          # Requester node aggregates
    AGGREGATOR = "aggregator"  # Staked settler node aggregates


@dataclass
class MapReduceStrategy:
    """Configuration for how a swarm job executes."""
    reduce_location: ReduceLocation = ReduceLocation.ORIGIN
    quorum_pct: float = 0.8       # Min % of shards that must complete
    per_shard_timeout: int = 60   # Seconds per shard
    global_timeout: int = 300     # Seconds for entire job
    budget_split: str = "equal"   # "equal" or "weighted" (by shard size)
    max_retries_per_shard: int = 2


@dataclass
class ShardAssignment:
    """Tracks a single shard's dispatch within a swarm job."""
    shard_cid: str
    agent_id: str = ""
    provider_id: str = ""
    status: str = "dispatched"  # dispatched, executing, completed, failed
    result: Optional[Dict[str, Any]] = None
    pcu_used: float = 0.0
    ftns_spent: float = 0.0
    retries: int = 0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class SwarmJob:
    """A parallel map-reduce job across multiple shards."""
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
    """Aggregated result from a swarm job."""
    job_id: str
    shard_results: Dict[str, Any]  # CID -> result dict
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
```

- [ ] **Step 4: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_swarm_models.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/compute/swarm/__init__.py prsm/compute/swarm/models.py tests/unit/test_swarm_models.py
git commit -m "feat(ring3): SwarmJob + MapReduceStrategy + SwarmResult data models"
```

---

### Task 3: Swarm Coordinator

**Files:**
- Create: `prsm/compute/swarm/coordinator.py`
- Test: `tests/unit/test_swarm_coordinator.py`

The coordinator orchestrates the full swarm lifecycle: query → find shards → fan out agents → collect results → aggregate.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_swarm_coordinator.py`:

```python
"""Tests for SwarmCoordinator — parallel dispatch + aggregation."""

import pytest
import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.compute.swarm.coordinator import SwarmCoordinator
from prsm.compute.swarm.models import (
    SwarmJob,
    SwarmStatus,
    MapReduceStrategy,
    ReduceLocation,
)
from prsm.compute.agents.models import AgentManifest, DispatchStatus
from prsm.data.shard_models import SemanticShard, SemanticShardManifest


MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


@pytest.fixture
def mock_dispatcher():
    dispatcher = AsyncMock()

    mock_record = MagicMock()
    mock_record.status = DispatchStatus.COMPLETED
    mock_record.result = {
        "status": "success",
        "output_b64": base64.b64encode(b'{"count": 42}').decode(),
        "pcu": 0.1,
        "execution_time_seconds": 1.0,
    }
    mock_record.bids = [{"provider_id": "prov-1", "price_ftns": 0.5}]
    mock_record._result_event = asyncio.Event()
    mock_record._result_event.set()

    def create_agent_side_effect(wasm_binary, manifest, ftns_budget, ttl=120):
        agent = MagicMock()
        agent.agent_id = f"agent-{id(manifest)}"
        agent.manifest = manifest
        return agent

    dispatcher.create_agent = MagicMock(side_effect=create_agent_side_effect)
    dispatcher.dispatch = AsyncMock(return_value=mock_record)
    dispatcher.select_and_transfer = AsyncMock(return_value=True)
    dispatcher.wait_for_result = AsyncMock(return_value=mock_record.result)
    dispatcher.get_record = MagicMock(return_value=mock_record)

    return dispatcher


@pytest.fixture
def coordinator(mock_dispatcher):
    return SwarmCoordinator(dispatcher=mock_dispatcher)


class TestSwarmCoordinator:
    def test_create_swarm_job(self, coordinator):
        job = coordinator.create_swarm_job(
            query="EV trends",
            shard_cids=["QmA", "QmB", "QmC"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=15.0,
        )
        assert job.status == SwarmStatus.PENDING
        assert len(job.shard_cids) == 3
        assert job.budget_per_shard == 5.0

    @pytest.mark.asyncio
    async def test_execute_dispatches_per_shard(self, coordinator, mock_dispatcher):
        job = coordinator.create_swarm_job(
            query="test query",
            shard_cids=["QmA", "QmB"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=10.0,
        )

        result = await coordinator.execute(job)

        # Should have dispatched one agent per shard
        assert mock_dispatcher.create_agent.call_count == 2
        assert mock_dispatcher.dispatch.call_count == 2
        assert result is not None
        assert result.shards_completed >= 0

    @pytest.mark.asyncio
    async def test_execute_collects_results(self, coordinator, mock_dispatcher):
        job = coordinator.create_swarm_job(
            query="test",
            shard_cids=["QmA", "QmB", "QmC"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=15.0,
        )

        result = await coordinator.execute(job)

        assert result.shards_completed == 3
        assert result.total_pcu > 0

    @pytest.mark.asyncio
    async def test_execute_handles_partial_failure(self, coordinator, mock_dispatcher):
        # Make the second dispatch return a failed result
        call_count = {"n": 0}
        original_wait = mock_dispatcher.wait_for_result

        async def failing_wait(agent_id, timeout=None):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return None  # Simulate timeout/failure
            return {
                "status": "success",
                "output_b64": base64.b64encode(b'{"v": 1}').decode(),
                "pcu": 0.1,
                "execution_time_seconds": 0.5,
            }

        mock_dispatcher.wait_for_result = AsyncMock(side_effect=failing_wait)

        job = coordinator.create_swarm_job(
            query="test",
            shard_cids=["QmA", "QmB", "QmC"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=15.0,
            strategy=MapReduceStrategy(quorum_pct=0.5),
        )

        result = await coordinator.execute(job)

        # 2 of 3 succeeded, quorum_pct=0.5 → quorum met
        assert result.shards_completed >= 2
        assert job.status in (SwarmStatus.COMPLETED, SwarmStatus.PARTIAL)

    @pytest.mark.asyncio
    async def test_execute_with_aggregation(self, coordinator, mock_dispatcher):
        job = coordinator.create_swarm_job(
            query="aggregate test",
            shard_cids=["QmA", "QmB"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=10.0,
        )

        result = await coordinator.execute(job)

        # Aggregated output should combine shard results
        assert result.aggregated_output is not None
        assert "shard_count" in result.aggregated_output

    def test_get_job(self, coordinator):
        job = coordinator.create_swarm_job(
            query="test",
            shard_cids=["QmA"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=5.0,
        )
        found = coordinator.get_job(job.job_id)
        assert found is not None
        assert found.job_id == job.job_id

    def test_get_nonexistent_job(self, coordinator):
        assert coordinator.get_job("nonexistent") is None
```

- [ ] **Step 2: Run tests — verify fail**

Run: `python -m pytest tests/unit/test_swarm_coordinator.py::TestSwarmCoordinator::test_create_swarm_job -v`
Expected: FAIL

- [ ] **Step 3: Implement the coordinator**

Create `prsm/compute/swarm/coordinator.py`:

```python
"""
Swarm Coordinator
=================

Orchestrates parallel map-reduce across semantically-sharded datasets.
Fans out Ring 2 agent dispatches, collects results, aggregates.
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from prsm.compute.agents.models import AgentManifest
from prsm.compute.swarm.models import (
    MapReduceStrategy,
    ShardAssignment,
    SwarmJob,
    SwarmResult,
    SwarmStatus,
)

logger = logging.getLogger(__name__)


class SwarmCoordinator:
    """Coordinates parallel agent execution across data shards."""

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self._jobs: Dict[str, SwarmJob] = {}

    def create_swarm_job(
        self,
        query: str,
        shard_cids: List[str],
        wasm_binary: bytes,
        manifest: AgentManifest,
        budget_ftns: float,
        strategy: Optional[MapReduceStrategy] = None,
        reduce_wasm: Optional[bytes] = None,
    ) -> SwarmJob:
        """Create a swarm job for parallel shard processing."""
        job_id = f"swarm-{uuid.uuid4().hex[:12]}"
        job = SwarmJob(
            job_id=job_id,
            query=query,
            shard_cids=shard_cids,
            wasm_binary=wasm_binary,
            agent_manifest=manifest,
            budget_ftns=budget_ftns,
            strategy=strategy or MapReduceStrategy(),
        )
        self._jobs[job_id] = job
        return job

    async def execute(self, job: SwarmJob) -> SwarmResult:
        """Execute a swarm job: fan out, collect, aggregate.

        1. For each shard, create and dispatch a MobileAgent
        2. Collect results as they arrive
        3. Aggregate when quorum is met
        """
        job.status = SwarmStatus.DISPATCHING
        started_at = time.time()

        # Phase 1: Fan out — dispatch one agent per shard
        dispatch_tasks = []
        for shard_cid in job.shard_cids:
            shard_manifest = AgentManifest(
                required_cids=[shard_cid],
                min_hardware_tier=job.agent_manifest.min_hardware_tier,
                max_memory_bytes=job.agent_manifest.max_memory_bytes,
                max_execution_seconds=job.agent_manifest.max_execution_seconds,
                max_output_bytes=job.agent_manifest.max_output_bytes,
                required_capabilities=job.agent_manifest.required_capabilities,
            )

            agent = self.dispatcher.create_agent(
                wasm_binary=job.wasm_binary,
                manifest=shard_manifest,
                ftns_budget=job.budget_per_shard,
                ttl=job.strategy.per_shard_timeout,
            )

            assignment = ShardAssignment(
                shard_cid=shard_cid,
                agent_id=agent.agent_id,
            )
            job.assignments[shard_cid] = assignment

            dispatch_tasks.append(self._dispatch_shard(job, agent, shard_cid))

        job.status = SwarmStatus.EXECUTING

        # Dispatch all shards in parallel
        dispatch_results = await asyncio.gather(*dispatch_tasks, return_exceptions=True)

        # Phase 2: Collect results
        shard_results = {}
        total_pcu = 0.0
        total_ftns = 0.0

        for shard_cid, result in zip(job.shard_cids, dispatch_results):
            assignment = job.assignments.get(shard_cid)
            if isinstance(result, Exception):
                if assignment:
                    assignment.status = "failed"
                job.failed_shards.append(shard_cid)
                logger.warning(f"Shard {shard_cid[:12]} dispatch error: {result}")
                continue

            if result and result.get("status") == "success":
                shard_results[shard_cid] = result
                job.completed_shards.append(shard_cid)
                pcu = result.get("pcu", 0)
                total_pcu += pcu
                total_ftns += job.budget_per_shard
                if assignment:
                    assignment.status = "completed"
                    assignment.result = result
                    assignment.pcu_used = pcu
                    assignment.completed_at = time.time()
            else:
                job.failed_shards.append(shard_cid)
                if assignment:
                    assignment.status = "failed"

        # Phase 3: Determine status
        if job.is_quorum_met():
            job.status = SwarmStatus.COMPLETED if not job.failed_shards else SwarmStatus.PARTIAL
        else:
            job.status = SwarmStatus.FAILED
            job.error = (
                f"Quorum not met: {len(job.completed_shards)}/{job.quorum_count} required "
                f"({len(job.failed_shards)} shards failed)"
            )

        job.completed_at = time.time()

        # Phase 4: Aggregate
        job.status = SwarmStatus.AGGREGATING if job.status != SwarmStatus.FAILED else job.status
        aggregation_start = time.time()

        aggregated = self._aggregate_results(shard_results, job.query)

        if job.status == SwarmStatus.AGGREGATING:
            job.status = SwarmStatus.COMPLETED if not job.failed_shards else SwarmStatus.PARTIAL

        swarm_result = SwarmResult(
            job_id=job.job_id,
            shard_results=shard_results,
            total_pcu=total_pcu,
            total_ftns_spent=total_ftns,
            total_shards=len(job.shard_cids),
            aggregation_time_seconds=time.time() - aggregation_start,
            aggregated_output=aggregated,
        )

        logger.info(
            f"Swarm {job.job_id[:12]}: {len(job.completed_shards)}/{len(job.shard_cids)} shards, "
            f"{total_pcu:.3f} PCU, {job.status.value}"
        )

        return swarm_result

    async def _dispatch_shard(
        self,
        job: SwarmJob,
        agent,
        shard_cid: str,
    ) -> Optional[Dict[str, Any]]:
        """Dispatch a single shard agent and wait for result."""
        try:
            record = await self.dispatcher.dispatch(agent)

            # Wait for bids + select + transfer
            await asyncio.sleep(0.1)  # Brief window for bids in tests
            await self.dispatcher.select_and_transfer(agent.agent_id)

            # Wait for result
            result = await self.dispatcher.wait_for_result(
                agent.agent_id,
                timeout=job.strategy.per_shard_timeout,
            )
            return result

        except Exception as e:
            logger.error(f"Shard {shard_cid[:12]} dispatch failed: {e}")
            return None

    def _aggregate_results(
        self,
        shard_results: Dict[str, Any],
        query: str,
    ) -> Dict[str, Any]:
        """Aggregate shard results into a single output.

        Default aggregation: collect all shard outputs into a summary.
        Custom reduce agents (Ring 5) will replace this.
        """
        outputs = []
        for cid, result in shard_results.items():
            output_b64 = result.get("output_b64", "")
            if output_b64:
                try:
                    raw = base64.b64decode(output_b64)
                    parsed = json.loads(raw)
                    outputs.append({"shard_cid": cid, "data": parsed})
                except (json.JSONDecodeError, Exception):
                    outputs.append({"shard_cid": cid, "data": output_b64})

        return {
            "query": query,
            "shard_count": len(shard_results),
            "shard_outputs": outputs,
        }

    def get_job(self, job_id: str) -> Optional[SwarmJob]:
        return self._jobs.get(job_id)
```

- [ ] **Step 4: Update swarm `__init__.py`**

Add coordinator exports to `prsm/compute/swarm/__init__.py`:

```python
from prsm.compute.swarm.coordinator import SwarmCoordinator

# Add to __all__:
    "SwarmCoordinator",
```

- [ ] **Step 5: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_swarm_coordinator.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add prsm/compute/swarm/coordinator.py prsm/compute/swarm/__init__.py tests/unit/test_swarm_coordinator.py
git commit -m "feat(ring3): SwarmCoordinator — parallel dispatch + aggregation lifecycle"
```

---

### Task 4: Node Integration + Integration Smoke Test

**Files:**
- Modify: `prsm/node/node.py`
- Create: `tests/integration/test_ring3_swarm.py`

- [ ] **Step 1: Wire SwarmCoordinator into node.py**

In `prsm/node/node.py`, find where `agent_dispatcher` is initialized (the Ring 2 block) and add after it:

```python
        # ── Swarm Compute (Ring 3) ────────────────────────────────────
        try:
            from prsm.compute.swarm.coordinator import SwarmCoordinator

            self.swarm_coordinator = SwarmCoordinator(
                dispatcher=self.agent_dispatcher,
            )
            logger.info("Swarm compute (Ring 3) initialized")
        except (ImportError, AttributeError):
            self.swarm_coordinator = None
            logger.debug("Swarm compute not available")
```

- [ ] **Step 2: Write the integration test**

Create `tests/integration/test_ring3_swarm.py`:

```python
"""
Ring 3 Smoke Test
=================

End-to-end: create semantic shards, dispatch swarm job, collect and aggregate.
"""

import pytest
import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

from prsm.data.shard_models import SemanticShard, SemanticShardManifest
from prsm.compute.swarm import SwarmCoordinator, SwarmStatus, MapReduceStrategy
from prsm.compute.agents.models import AgentManifest, DispatchStatus


MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


class TestRing3Smoke:
    @pytest.mark.asyncio
    async def test_semantic_shard_to_swarm_pipeline(self):
        """Full Ring 3 flow: find relevant shards → dispatch swarm → aggregate."""
        # Step 1: Create semantic shard manifest
        shards = [
            SemanticShard(
                shard_id=f"shard-{i}",
                parent_dataset="nada-nc",
                cid=f"QmShard{i:03d}",
                centroid=[float(i) * 0.2, 1.0 - float(i) * 0.2, 0.5],
                record_count=1000,
                size_bytes=1024 * 1024,
                keywords=[f"topic-{i}"],
            )
            for i in range(5)
        ]
        manifest = SemanticShardManifest(
            dataset_id="nada-nc-2025",
            total_records=5000,
            total_size_bytes=5 * 1024 * 1024,
            shards=shards,
        )

        # Step 2: Find relevant shards for query
        query_embedding = [0.1, 0.9, 0.5]  # Close to shard-0
        relevant = manifest.find_relevant_shards(query_embedding, top_k=3)
        assert len(relevant) == 3

        # Step 3: Set up mock dispatcher
        mock_dispatcher = AsyncMock()

        def create_agent_fn(wasm_binary, manifest, ftns_budget, ttl=120):
            agent = MagicMock()
            agent.agent_id = f"agent-{id(manifest)}"
            agent.manifest = manifest
            return agent

        mock_record = MagicMock()
        mock_record.status = DispatchStatus.COMPLETED
        mock_record.bids = [{"provider_id": "p1"}]
        mock_record._result_event = asyncio.Event()
        mock_record._result_event.set()

        mock_dispatcher.create_agent = MagicMock(side_effect=create_agent_fn)
        mock_dispatcher.dispatch = AsyncMock(return_value=mock_record)
        mock_dispatcher.select_and_transfer = AsyncMock(return_value=True)
        mock_dispatcher.wait_for_result = AsyncMock(return_value={
            "status": "success",
            "output_b64": base64.b64encode(b'{"ev_count": 142}').decode(),
            "pcu": 0.2,
            "execution_time_seconds": 1.5,
        })
        mock_dispatcher.get_record = MagicMock(return_value=mock_record)

        # Step 4: Execute swarm job
        coordinator = SwarmCoordinator(dispatcher=mock_dispatcher)

        job = coordinator.create_swarm_job(
            query="EV adoption in NC",
            shard_cids=[s.cid for s in relevant],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=15.0,
        )

        result = await coordinator.execute(job)

        # Verify
        assert result.shards_completed == 3
        assert result.total_pcu > 0
        assert result.aggregated_output is not None
        assert result.aggregated_output["shard_count"] == 3
        assert job.status in (SwarmStatus.COMPLETED, SwarmStatus.PARTIAL)

    @pytest.mark.asyncio
    async def test_swarm_with_quorum_failure(self):
        """Swarm job fails when quorum not met."""
        mock_dispatcher = AsyncMock()

        def create_agent_fn(wasm_binary, manifest, ftns_budget, ttl=120):
            agent = MagicMock()
            agent.agent_id = f"agent-{id(manifest)}"
            return agent

        mock_dispatcher.create_agent = MagicMock(side_effect=create_agent_fn)
        mock_dispatcher.dispatch = AsyncMock(return_value=MagicMock(
            status=DispatchStatus.FAILED
        ))
        mock_dispatcher.select_and_transfer = AsyncMock(return_value=False)
        mock_dispatcher.wait_for_result = AsyncMock(return_value=None)  # All fail
        mock_dispatcher.get_record = MagicMock(return_value=None)

        coordinator = SwarmCoordinator(dispatcher=mock_dispatcher)

        job = coordinator.create_swarm_job(
            query="failing query",
            shard_cids=["QmA", "QmB", "QmC"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=15.0,
            strategy=MapReduceStrategy(quorum_pct=0.8),
        )

        result = await coordinator.execute(job)

        assert job.status == SwarmStatus.FAILED
        assert result.shards_completed == 0
        assert "quorum" in (job.error or "").lower()
```

- [ ] **Step 3: Run all Ring 3 tests**

Run: `python -m pytest tests/unit/test_semantic_shard.py tests/unit/test_swarm_models.py tests/unit/test_swarm_coordinator.py tests/integration/test_ring3_swarm.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 4: Run full Ring 1+2+3 regression**

Run: `python -m pytest tests/unit/test_wasm_runtime.py tests/unit/test_hardware_profiler.py tests/unit/test_wasm_compute_provider.py tests/unit/test_mobile_agent_models.py tests/unit/test_agent_executor.py tests/unit/test_agent_dispatcher.py tests/unit/test_semantic_shard.py tests/unit/test_swarm_models.py tests/unit/test_swarm_coordinator.py tests/integration/test_ring1_smoke.py tests/integration/test_ring2_dispatch.py tests/integration/test_ring3_swarm.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/node/node.py tests/integration/test_ring3_swarm.py
git commit -m "feat(ring3): wire SwarmCoordinator into PRSMNode + integration smoke test"
```

---

### Task 5: Version Bump + Push + PyPI

**Files:**
- Modify: `prsm/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump version to 0.28.0**

Change `__version__` in `prsm/__init__.py` to `"0.28.0"`.
Change `version` in `pyproject.toml` to `"0.28.0"`.

- [ ] **Step 2: Final regression test**

Run: `python -m pytest tests/unit/test_semantic_shard.py tests/unit/test_swarm_models.py tests/unit/test_swarm_coordinator.py tests/integration/test_ring3_swarm.py -v --timeout=30`
Expected: All Ring 3 tests PASS

- [ ] **Step 3: Commit and push**

```bash
git add prsm/__init__.py pyproject.toml
git commit -m "chore: bump version to 0.28.0 for Ring 3 — Swarm Compute"
git push origin main
```

- [ ] **Step 4: Build and publish**

```bash
rm -rf build/ dist/ prsm_network.egg-info/
python3 -m build
python3 -m twine upload dist/prsm_network-0.28.0*
```
