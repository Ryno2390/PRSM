"""
Semantic Shard Data Models
==========================

Data structures for content-addressable dataset shards with semantic
embeddings.  Each shard carries a centroid vector so the network can
route queries to the most relevant slice of a dataset without
downloading everything.

Classes:
    SemanticShard         - A single shard with centroid embedding
    SemanticShardManifest - Collection of shards for a dataset
    ShardQuery            - Incoming query targeting relevant shards
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Cosine Similarity ────────────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Returns 0.0 for empty vectors or when either vector has zero norm.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# ── SemanticShard ────────────────────────────────────────────────────────

@dataclass
class SemanticShard:
    """A single content-addressable shard with a centroid embedding."""

    shard_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_dataset: str = ""
    cid: str = ""
    centroid: List[float] = field(default_factory=list)
    record_count: int = 0
    size_bytes: int = 0
    keywords: List[str] = field(default_factory=list)
    providers: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "parent_dataset": self.parent_dataset,
            "cid": self.cid,
            "centroid": list(self.centroid),
            "record_count": self.record_count,
            "size_bytes": self.size_bytes,
            "keywords": list(self.keywords),
            "providers": list(self.providers),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SemanticShard:
        return cls(
            shard_id=data["shard_id"],
            parent_dataset=data["parent_dataset"],
            cid=data["cid"],
            centroid=data["centroid"],
            record_count=data["record_count"],
            size_bytes=data["size_bytes"],
            keywords=data.get("keywords", []),
            providers=data.get("providers", []),
            created_at=data.get("created_at", time.time()),
        )


# ── SemanticShardManifest ────────────────────────────────────────────────

@dataclass
class SemanticShardManifest:
    """Collection of shards for a dataset, supporting similarity search."""

    dataset_id: str = ""
    total_records: int = 0
    total_size_bytes: int = 0
    shards: List[SemanticShard] = field(default_factory=list)
    embedding_dimension: int = 0
    created_at: float = field(default_factory=time.time)

    def find_relevant_shards(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[SemanticShard, float]]:
        """Return shards sorted by cosine similarity to *query_embedding*.

        Each entry is ``(shard, similarity)``.  Only shards meeting
        *min_similarity* are included, capped at *top_k*.
        """
        scored: List[Tuple[SemanticShard, float]] = []
        for shard in self.shards:
            sim = _cosine_similarity(query_embedding, shard.centroid)
            if sim >= min_similarity:
                scored.append((shard, sim))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

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
    def from_dict(cls, data: Dict[str, Any]) -> SemanticShardManifest:
        return cls(
            dataset_id=data["dataset_id"],
            total_records=data["total_records"],
            total_size_bytes=data["total_size_bytes"],
            shards=[SemanticShard.from_dict(s) for s in data.get("shards", [])],
            embedding_dimension=data.get("embedding_dimension", 0),
            created_at=data.get("created_at", time.time()),
        )


# ── ShardQuery ───────────────────────────────────────────────────────────

@dataclass
class ShardQuery:
    """Incoming query targeting the most relevant shards."""

    query_text: str = ""
    query_embedding: List[float] = field(default_factory=list)
    top_k: int = 10
    min_similarity: float = 0.5
    dataset_id: Optional[str] = None
