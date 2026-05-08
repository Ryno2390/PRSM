"""SemanticIndexAdapter — bridges shard_finder's string-input
`SemanticIndex` Protocol to `_SemanticIndex.find_top_k`'s vector-input
API.

Per `docs/2026-05-08-query-orchestrator-wiring-readiness.md` the
shard_finder Protocol takes a query string; the existing
`_SemanticIndex` (text-vector lane) takes a numpy embedding. This
module owns the single string→embedding step + delegates lookup.

Production wiring: in `node.py`, the adapter is constructed with
the ContentUploader's `_semantic_index` and an Embedder backed by
the node's local sentence-transformer model (per Phase 3.x.2 model
registry). The same adapter shape works for the binary lane
(`_FingerprintIndex`) once a per-content-type embedder lands —
Tier B/C fingerprint kinds = R&D follow-on.

Threat-model note: the embedding model itself is part of the
node's trusted compute base — a malicious local model could bias
relevance scores. R3 (data-poisoning) covers the cross-node
adversary; the local-model-integrity case is covered by Phase
3.x.2 ModelRegistry's signed manifests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Pluggable string→vector embedder.

    Production wiring: a wrapper around the node's local
    sentence-transformer model. Tests inject a deterministic stub.

    The contract is sync because embedding is CPU-bound; async would
    add complexity for no benefit. If batch embedding becomes
    important later, add a separate `encode_batch(queries: list[str])
    -> list[np.ndarray]` method without breaking this one.
    """

    def encode(self, query: str) -> np.ndarray: ...


class _IndexLike(Protocol):
    """Internal — minimum surface the adapter needs from the wrapped
    index. `_SemanticIndex.find_top_k` already satisfies this."""

    def find_top_k(self, embedding: np.ndarray, k: int) -> list: ...


@dataclass
class SemanticIndexAdapter:
    """Adapts `_IndexLike.find_top_k(embedding, k)` to the shard_finder
    `SemanticIndex.find_top_k(query: str, k: int)` Protocol.

    Owns the single string→embedding step. The embedder runs once per
    lookup — embedding is the expensive part of the path; the index
    scan is sub-millisecond by comparison.
    """
    embedder: Embedder
    index: Any  # IndexLike — duck-typed because _SemanticIndex isn't a Protocol export

    def find_top_k(self, query: str, k: int) -> list[tuple[str, float, str]]:
        """Encode the query, then run a top-k cosine-similarity scan
        on the underlying index. Returns descending-similarity triples."""
        embedding = self.embedder.encode(query)
        return self.index.find_top_k(embedding, k)
