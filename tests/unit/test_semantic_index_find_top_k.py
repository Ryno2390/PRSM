"""_SemanticIndex.find_top_k — top-k variant of find_nearest.

Per the QueryOrchestrator wiring-readiness assessment
(docs/2026-05-08-query-orchestrator-wiring-readiness.md, Blocker 4):
shard_finder.py consumes a `SemanticIndex.find_top_k(...)` Protocol.
The single-best `find_nearest` already exists; top-k is the
generalization.

Tests stay vector-native — the string→vector embedding step belongs
to the orchestrator wiring adapter (separate task). This file tests
only the lookup primitive.
"""
from __future__ import annotations

import numpy as np

from prsm.node.content_uploader import _SemanticIndex


def _vec(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def _make_index(*entries) -> _SemanticIndex:
    """entries: iterable of (cid, vector, creator_id)."""
    idx = _SemanticIndex()
    for cid, vec, creator in entries:
        idx.store(cid, vec, creator)
    return idx


# ──────────────────────────────────────────────────────────────────────
# Empty / degenerate
# ──────────────────────────────────────────────────────────────────────


class TestDegenerate:
    def test_empty_index_returns_empty(self):
        idx = _SemanticIndex()
        assert idx.find_top_k(_vec(1.0, 0.0), k=5) == []

    def test_zero_norm_query_returns_empty(self):
        idx = _make_index(("a", _vec(1.0, 0.0), "alice"))
        assert idx.find_top_k(_vec(0.0, 0.0), k=5) == []

    def test_k_zero_returns_empty(self):
        idx = _make_index(("a", _vec(1.0, 0.0), "alice"))
        assert idx.find_top_k(_vec(1.0, 0.0), k=0) == []

    def test_negative_k_returns_empty(self):
        idx = _make_index(("a", _vec(1.0, 0.0), "alice"))
        assert idx.find_top_k(_vec(1.0, 0.0), k=-3) == []


# ──────────────────────────────────────────────────────────────────────
# Top-k semantics
# ──────────────────────────────────────────────────────────────────────


class TestTopKSemantics:
    def test_returns_at_most_k(self):
        idx = _make_index(
            ("a", _vec(1.0, 0.0), "alice"),
            ("b", _vec(0.95, 0.05), "bob"),
            ("c", _vec(0.0, 1.0), "carol"),
            ("d", _vec(-1.0, 0.0), "dave"),
        )
        out = idx.find_top_k(_vec(1.0, 0.0), k=2)
        assert len(out) == 2

    def test_descending_similarity_sort(self):
        idx = _make_index(
            ("a", _vec(0.0, 1.0), "alice"),    # cos = 0 to query
            ("b", _vec(0.95, 0.05), "bob"),    # cos ~ 0.998
            ("c", _vec(1.0, 0.0), "carol"),    # cos = 1.0
        )
        out = idx.find_top_k(_vec(1.0, 0.0), k=3)
        sims = [s for _, s, _ in out]
        assert sims == sorted(sims, reverse=True)
        # First entry = exact match.
        assert out[0][0] == "c"

    def test_k_larger_than_index_returns_full_sorted(self):
        idx = _make_index(
            ("a", _vec(1.0, 0.0), "alice"),
            ("b", _vec(0.95, 0.05), "bob"),
        )
        out = idx.find_top_k(_vec(1.0, 0.0), k=999)
        assert len(out) == 2
        sims = [s for _, s, _ in out]
        assert sims == sorted(sims, reverse=True)

    def test_returns_triples_cid_sim_creator(self):
        idx = _make_index(
            ("the-cid", _vec(1.0, 0.0), "the-creator"),
        )
        [(cid, sim, creator)] = idx.find_top_k(_vec(1.0, 0.0), k=5)
        assert cid == "the-cid"
        assert creator == "the-creator"
        assert isinstance(sim, float)
        # Self-match has cosine ~ 1.0.
        assert sim > 0.99


# ──────────────────────────────────────────────────────────────────────
# Consistency with find_nearest (same answer at k=1)
# ──────────────────────────────────────────────────────────────────────


class TestParityWithFindNearest:
    """At k=1, find_top_k MUST return the same single result that
    find_nearest does. Pin the contract — if the two diverge a future
    bug could land where shard_finder gets a different answer than
    the upload-path dedup logic."""

    def test_k_one_matches_find_nearest(self):
        idx = _make_index(
            ("a", _vec(1.0, 0.0), "alice"),
            ("b", _vec(0.5, 0.5), "bob"),
            ("c", _vec(0.0, 1.0), "carol"),
        )
        query = _vec(0.9, 0.1)
        nearest = idx.find_nearest(query)
        top1 = idx.find_top_k(query, k=1)
        assert nearest is not None
        assert len(top1) == 1
        assert top1[0][0] == nearest[0]
        assert top1[0][1] == nearest[1]
        assert top1[0][2] == nearest[2]
