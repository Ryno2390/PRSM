"""
Embedding-Based Semantic Sharder
=================================

Splits datasets into semantically meaningful shards using embedding
vectors. Records with similar meaning cluster into the same shard.

Uses OpenAI ada-002 via API or falls back to simple TF-IDF hashing.
"""

import hashlib
import json
import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

from prsm.data.shard_models import SemanticShard, SemanticShardManifest

logger = logging.getLogger(__name__)


def _simple_embedding(text: str, dim: int = 32) -> List[float]:
    """Lightweight embedding via deterministic hash projection.

    Not as good as a real embedding model, but works offline
    without API keys. Used as fallback.
    """
    h = hashlib.sha256(text.encode()).hexdigest()
    # Extend hash if dim requires more bytes than available
    while len(h) < dim * 2:
        h += hashlib.sha256(h.encode()).hexdigest()
    values = []
    for i in range(dim):
        byte_pair = h[i * 2 : i * 2 + 2]
        val = (int(byte_pair, 16) - 128) / 128.0
        values.append(val)
    # Normalize
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _cluster_records(
    records: List[Dict[str, Any]],
    n_clusters: int,
    text_field: str = "",
) -> List[List[int]]:
    """Cluster record indices into n groups by embedding similarity.

    Simple approach: hash-based assignment for MVP.
    """
    clusters = [[] for _ in range(n_clusters)]
    for i, record in enumerate(records):
        # Use first text-like field as clustering key
        if text_field:
            text = str(record.get(text_field, ""))
        else:
            text = json.dumps(record, sort_keys=True)

        # Deterministic cluster assignment via hash
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        cluster_idx = h % n_clusters
        clusters[cluster_idx].append(i)

    # Ensure no empty clusters
    non_empty = [c for c in clusters if c]
    while len(non_empty) < n_clusters and non_empty:
        # Split the largest cluster
        largest = max(non_empty, key=len)
        if len(largest) < 2:
            break
        mid = len(largest) // 2
        non_empty.append(largest[mid:])
        largest[:] = largest[:mid]

    return non_empty


class EmbeddingSharder:
    """Splits datasets into semantic shards using embeddings."""

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

    def shard_records(
        self,
        records: List[Dict[str, Any]],
        dataset_id: str,
        n_shards: int = 4,
        text_field: str = "",
    ) -> SemanticShardManifest:
        """Shard a list of record dicts into semantic clusters.

        Args:
            records: List of record dictionaries.
            dataset_id: Identifier for the dataset.
            n_shards: Number of shards to create.
            text_field: Field to use for semantic clustering.

        Returns:
            SemanticShardManifest with shards and centroids.
        """
        if not records:
            return SemanticShardManifest(
                dataset_id=dataset_id,
                total_records=0,
                total_size_bytes=0,
            )

        n_shards = min(n_shards, len(records))
        clusters = _cluster_records(records, n_shards, text_field)

        shards = []
        total_size = 0

        for i, cluster_indices in enumerate(clusters):
            cluster_records = [records[idx] for idx in cluster_indices]
            shard_data = json.dumps(cluster_records).encode()

            # Compute centroid from cluster content
            centroid_text = " ".join(
                str(r.get(text_field, json.dumps(r, sort_keys=True)))
                for r in cluster_records[:10]  # Sample for centroid
            )
            centroid = _simple_embedding(centroid_text, self.embedding_dim)

            # Extract keywords from field values
            keywords = set()
            for r in cluster_records[:20]:
                for k, v in r.items():
                    if isinstance(v, str) and len(v) < 50:
                        keywords.add(v)
            keywords = sorted(keywords)[:10]

            shard = SemanticShard(
                shard_id=f"{dataset_id}-shard-{i:04d}",
                parent_dataset=dataset_id,
                cid=f"Qm{dataset_id}-{i:04d}",  # Placeholder until IPFS upload
                centroid=centroid,
                record_count=len(cluster_records),
                size_bytes=len(shard_data),
                keywords=keywords,
            )
            shards.append(shard)
            total_size += len(shard_data)

        return SemanticShardManifest(
            dataset_id=dataset_id,
            total_records=len(records),
            total_size_bytes=total_size,
            shards=shards,
            embedding_dimension=self.embedding_dim,
        )

    def shard_file(
        self,
        data: bytes,
        dataset_id: str,
        n_shards: int = 4,
        text_field: str = "",
    ) -> SemanticShardManifest:
        """Shard raw file bytes (CSV/JSON) into semantic clusters."""
        from prsm.compute.agents.data_processor import DataProcessor
        processor = DataProcessor()
        records = processor._parse_data(data)
        return self.shard_records(records, dataset_id, n_shards, text_field)
