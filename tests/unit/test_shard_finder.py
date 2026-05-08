"""QueryOrchestrator shard_finder — query → shard CIDs.

Shard discovery is the bridge between decomposition (a `query` string +
DSL manifest) and the swarm runner (a `tuple[ShardCandidate, ...]` to
fan out WASM agents to). It composes:

  - The local-node `SemanticIndex` (text-vector lane —
    `prsm/node/content_uploader.py::_SemanticIndex`). The
    `_FingerprintIndex` (binary lane) is a sibling; this module exposes
    a pluggable Protocol so either lane can drive lookup.

  - The PRSM-PROV-1 Item 4 cross-node escalation that landed
    2026-05-07 — when the local index is below the derivative
    threshold, the underlying index transparently escalates to peer
    nodes via `EmbeddingDHT`.

shard_finder's contribution is the orchestration layer: dedup,
similarity threshold, count cap, descending sort by similarity. It
does NOT itself implement vector search — that's the index's job.

Threat-model note: NOT in the aggregator-selector A1–A10 catalog.
Shard selection happens BEFORE aggregator selection — the relevance of
shards to a query has no per-prompter trust binding. If a malicious
indexer poisons relevance scores (Sybil-stuffing the embedding lane),
that's R3 territory (data-poisoning) covered by ProvenanceRegistry v2
+ slashing, not this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from prsm.compute.query_orchestrator import (
    SemanticIndex,
    ShardCandidate,
    find_relevant_shards,
)


# ──────────────────────────────────────────────────────────────────────
# Test fixture — stub semantic index
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _StubIndex:
    """Test-only SemanticIndex. Returns whatever is in `_canned`."""
    _canned: list[tuple[str, float, str]]

    def find_top_k(self, query: str, k: int) -> list[tuple[str, float, str]]:
        return self._canned[:k]


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_returns_shard_candidates(self):
        idx = _StubIndex(_canned=[
            ("prsm:cid-a", 0.95, "alice"),
            ("prsm:cid-b", 0.87, "bob"),
        ])
        out = find_relevant_shards("anything", semantic_index=idx)
        assert len(out) == 2
        for c in out:
            assert isinstance(c, ShardCandidate)
        # Fields threaded through.
        assert out[0].cid == "prsm:cid-a"
        assert out[0].creator_id == "alice"
        assert out[0].similarity == pytest.approx(0.95)

    def test_descending_similarity_sort(self):
        # Even when the index returns out-of-order, shard_finder sorts.
        idx = _StubIndex(_canned=[
            ("c", 0.40, "x"),
            ("a", 0.95, "y"),
            ("b", 0.80, "z"),
        ])
        out = find_relevant_shards("q", semantic_index=idx, min_similarity=0.0)
        sims = [c.similarity for c in out]
        assert sims == sorted(sims, reverse=True)


# ──────────────────────────────────────────────────────────────────────
# Validation + thresholds
# ──────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_empty_query_raises(self):
        idx = _StubIndex(_canned=[])
        with pytest.raises(ValueError, match="empty"):
            find_relevant_shards("", semantic_index=idx)

    def test_min_similarity_filter(self):
        idx = _StubIndex(_canned=[
            ("a", 0.95, "x"),
            ("b", 0.45, "y"),
            ("c", 0.20, "z"),  # below 0.30 threshold
        ])
        out = find_relevant_shards("q", semantic_index=idx, min_similarity=0.30)
        cids = [c.cid for c in out]
        assert cids == ["a", "b"]

    def test_limit_caps_output(self):
        idx = _StubIndex(_canned=[
            (f"cid-{i}", 0.9 - i * 0.01, f"creator-{i}") for i in range(50)
        ])
        out = find_relevant_shards("q", semantic_index=idx, limit=10)
        assert len(out) == 10

    def test_empty_index_returns_empty(self):
        idx = _StubIndex(_canned=[])
        out = find_relevant_shards("q", semantic_index=idx)
        assert out == []

    def test_negative_similarity_scrubbed(self):
        # Cosine similarity can be negative for orthogonal/opposite
        # vectors. Treat anything below min_similarity as filtered —
        # don't return shards the prompter would never want.
        idx = _StubIndex(_canned=[
            ("a", 0.95, "x"),
            ("b", -0.3, "y"),
        ])
        out = find_relevant_shards("q", semantic_index=idx, min_similarity=0.0)
        cids = [c.cid for c in out]
        # 0.95 included, -0.3 excluded by min_similarity=0.0.
        assert cids == ["a"]

    def test_dedup_by_cid(self):
        # The index might (incorrectly) return the same CID twice with
        # slightly different scores — collapse to the highest score.
        idx = _StubIndex(_canned=[
            ("dup-cid", 0.95, "alice"),
            ("dup-cid", 0.85, "alice"),  # duplicate
            ("other", 0.70, "bob"),
        ])
        out = find_relevant_shards("q", semantic_index=idx, min_similarity=0.0)
        cids = [c.cid for c in out]
        assert cids.count("dup-cid") == 1


# ──────────────────────────────────────────────────────────────────────
# Protocol pin
# ──────────────────────────────────────────────────────────────────────


class TestSemanticIndexProtocolPin:
    """Pin the SemanticIndex Protocol surface. ContentUploader's
    `_semantic_index` will satisfy this once a `find_top_k` method is
    added (small follow-on); until then, production deployments inject
    a wrapper. Pin the contract so both sides update together."""

    def test_protocol_method_signature(self):
        idx = _StubIndex(_canned=[("a", 0.5, "x")])
        # Must accept (query: str, k: int) and return list of triples.
        result = idx.find_top_k("anything", 5)
        assert isinstance(result, list)
        for tup in result:
            assert len(tup) == 3
            cid, sim, creator = tup
            assert isinstance(cid, str)
            assert isinstance(sim, float)
            assert isinstance(creator, str)

    def test_shard_candidate_has_holder_node_ids_field(self):
        # ManifestDHT enrichment slot — orchestrator's swarm_runner
        # consumes this to dispatch agents. Default empty for now;
        # production wiring fills it.
        c = ShardCandidate(
            cid="x", similarity=0.5, creator_id="y", holder_node_ids=("n1", "n2")
        )
        assert c.holder_node_ids == ("n1", "n2")
