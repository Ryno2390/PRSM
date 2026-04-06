"""
Tests for Semantic Shard Data Models
=====================================

Verifies SemanticShard, SemanticShardManifest, ShardQuery,
and cosine-similarity-based shard retrieval.
"""

import time
import uuid
import pytest

from prsm.data.shard_models import (
    SemanticShard,
    SemanticShardManifest,
    ShardQuery,
    _cosine_similarity,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_shard(centroid, keywords=None, record_count=100, size_bytes=4096):
    return SemanticShard(
        shard_id=str(uuid.uuid4()),
        parent_dataset="ds-001",
        cid=f"Qm{uuid.uuid4().hex[:40]}",
        centroid=centroid,
        record_count=record_count,
        size_bytes=size_bytes,
        keywords=keywords or [],
        providers=["provider-a"],
    )


# ── Test: SemanticShard creation and roundtrip ───────────────────────────

class TestSemanticShard:
    def test_creation_and_roundtrip(self):
        shard = _make_shard([1.0, 0.0, 0.0], keywords=["electric", "vehicles"])
        d = shard.to_dict()

        assert d["parent_dataset"] == "ds-001"
        assert d["centroid"] == [1.0, 0.0, 0.0]
        assert d["keywords"] == ["electric", "vehicles"]
        assert "created_at" in d

        restored = SemanticShard.from_dict(d)
        assert restored.shard_id == shard.shard_id
        assert restored.centroid == shard.centroid
        assert restored.record_count == shard.record_count


# ── Test: cosine similarity edge cases ───────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_zero_norm_vector(self):
        assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


# ── Test: SemanticShardManifest ──────────────────────────────────────────

class TestSemanticShardManifest:
    def test_creation(self):
        shards = [_make_shard([1, 0, 0]), _make_shard([0, 1, 0])]
        manifest = SemanticShardManifest(
            dataset_id="ds-001",
            total_records=200,
            total_size_bytes=8192,
            shards=shards,
            embedding_dimension=3,
        )
        assert manifest.dataset_id == "ds-001"
        assert len(manifest.shards) == 2

    def test_find_relevant_shards_order(self):
        """EV shard ([1,0,0]) should rank highest for an EV query ([0.9, 0.1, 0])."""
        ev_shard = _make_shard([1.0, 0.0, 0.0], keywords=["electric", "vehicles"])
        bio_shard = _make_shard([0.0, 1.0, 0.0], keywords=["biology"])
        astro_shard = _make_shard([0.0, 0.0, 1.0], keywords=["astronomy"])

        manifest = SemanticShardManifest(
            dataset_id="ds-001",
            total_records=300,
            total_size_bytes=12288,
            shards=[bio_shard, astro_shard, ev_shard],
            embedding_dimension=3,
        )

        results = manifest.find_relevant_shards([0.9, 0.1, 0.0], top_k=3)
        # First result should be the EV shard (highest cosine similarity)
        assert results[0][0].shard_id == ev_shard.shard_id
        # Similarities should be descending
        sims = [sim for _, sim in results]
        assert sims == sorted(sims, reverse=True)

    def test_manifest_roundtrip(self):
        shards = [_make_shard([1, 0, 0]), _make_shard([0, 1, 0])]
        manifest = SemanticShardManifest(
            dataset_id="ds-001",
            total_records=200,
            total_size_bytes=8192,
            shards=shards,
            embedding_dimension=3,
        )
        d = manifest.to_dict()
        restored = SemanticShardManifest.from_dict(d)
        assert restored.dataset_id == manifest.dataset_id
        assert len(restored.shards) == 2
        assert restored.shards[0].shard_id == shards[0].shard_id


# ── Test: ShardQuery ─────────────────────────────────────────────────────

class TestShardQuery:
    def test_creation_and_defaults(self):
        q = ShardQuery(query_text="electric cars", query_embedding=[0.9, 0.1, 0.0])
        assert q.top_k == 10
        assert q.min_similarity == 0.5
        assert q.dataset_id is None
        assert q.query_text == "electric cars"
