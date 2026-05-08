"""QueryOrchestrator — shard discovery.

Bridge between query decomposition and swarm dispatch: given a query
string + a semantic index, returns the shard CIDs the swarm runner
should fan WASM agents out to.

Composes:
  - The local-node `SemanticIndex` (text-vector lane, currently
    `prsm/node/content_uploader.py::_SemanticIndex`). The
    `_FingerprintIndex` (binary lane) is a sibling with the same
    Protocol shape — either lane drives lookup.

  - The PRSM-PROV-1 Item 4 cross-node escalation that landed
    2026-05-07. When the local index is below the derivative
    threshold, the underlying index transparently escalates to peer
    nodes via `EmbeddingDHT` — shard_finder doesn't have to know.

This module's contribution is the orchestration layer:
  - dedup by CID (collapsing same-CID-different-score collisions)
  - similarity threshold filtering
  - count cap
  - descending sort by similarity (best first)

It does NOT implement vector search or DHT escalation — those are
the index's job.

Threat-model note: NOT in the aggregator-selector A1–A10 catalog.
Shard selection happens BEFORE aggregator selection — relevance has
no per-prompter trust binding. Data-poisoning at the embedding lane
(R3) is covered by ProvenanceRegistry v2 + slashing, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# Default similarity threshold below which a shard isn't a "relevant"
# match. Cosine-similarity space — 0.30 is a generous floor; the
# actual production threshold should be calibrated against real corpus
# data (PRSM-PROV-1 T4.8 / T6.3). Until calibrated, this default is
# intentionally permissive — the swarm runner will further filter
# downstream by tier eligibility.
DEFAULT_MIN_SIMILARITY = 0.30

# Hard cap on returned shard count. Prevents a malformed/over-broad
# query from fanning out to thousands of WASM agents.
DEFAULT_LIMIT = 32
MAX_LIMIT = 1024


@dataclass(frozen=True)
class ShardCandidate:
    """A shard ranked relevant to a query.

    Attributes
    ----------
    cid:
        Content identifier — `prsm:` scheme + content hash. The WASM
        runner uses this to fetch the shard via BitTorrent /
        ContentStore.
    similarity:
        Cosine similarity in [-1.0, 1.0]. Positive = related;
        negative = anti-correlated. The swarm runner uses this as a
        weight-or-skip signal.
    creator_id:
        The original publisher's identifier — flows to RoyaltyDistributor
        at settlement time.
    holder_node_ids:
        Nodes known to hold this shard, from ManifestDHT enrichment.
        Empty until the orchestrator's wiring fills this in. Frozen
        tuple so the dataclass stays hashable.
    """
    cid: str
    similarity: float
    creator_id: str
    holder_node_ids: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class SemanticIndex(Protocol):
    """Pluggable lookup contract.

    Production wiring: a small wrapper around
    `ContentUploader._semantic_index` (or `._fingerprint_index` for
    binary content) that exposes `find_top_k`. The underlying index's
    DHT escalation kicks in transparently when the local result is
    weak — shard_finder doesn't have to drive that.

    The triple shape `(cid, similarity, creator_id)` matches the
    existing index's return type for `find_nearest`; `find_top_k` is
    just the multi-result generalization.
    """

    def find_top_k(self, query: str, k: int) -> list[tuple[str, float, str]]: ...


def find_relevant_shards(
    query: str,
    *,
    semantic_index: SemanticIndex,
    limit: int = DEFAULT_LIMIT,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[ShardCandidate]:
    """Resolve a query string to its relevant shards.

    Walks `semantic_index.find_top_k`, applies the orchestration layer
    (dedup + threshold + cap + sort), returns descending-similarity
    `ShardCandidate` list.

    Raises:
        ValueError: query is empty / whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("query is empty")

    effective_limit = max(1, min(int(limit), MAX_LIMIT))

    # Pull more than `limit` initially — dedup + threshold may shrink
    # the result, and we still want `limit` worth of shards if
    # available.
    raw = semantic_index.find_top_k(query, effective_limit * 2)

    # Dedup-collapse by CID — keep the highest similarity per CID.
    best_per_cid: dict[str, tuple[str, float, str]] = {}
    for cid, sim, creator in raw:
        prior = best_per_cid.get(cid)
        if prior is None or sim > prior[1]:
            best_per_cid[cid] = (cid, sim, creator)

    # Threshold + sort + cap.
    filtered = [
        ShardCandidate(cid=cid, similarity=sim, creator_id=creator)
        for cid, sim, creator in best_per_cid.values()
        if sim >= min_similarity
    ]
    filtered.sort(key=lambda c: c.similarity, reverse=True)
    return filtered[:effective_limit]
