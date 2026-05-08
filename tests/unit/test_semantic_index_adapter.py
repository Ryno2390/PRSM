"""SemanticIndexAdapter — bridges shard_finder's string-input
SemanticIndex Protocol to _SemanticIndex's vector-input find_top_k.

The shard_finder Protocol (`find_top_k(query: str, k: int)`) takes
a query string. _SemanticIndex.find_top_k takes a vector. The
adapter owns the embedding step + delegates lookup.

Production wiring: in node.py, the adapter is constructed with the
ContentUploader's _semantic_index and an Embedder backed by the
node's local sentence-transformer model (per Phase 3.x.2 model
registry).

Tests use a stub Embedder that produces deterministic vectors so
the lookup behavior is the only variable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prsm.compute.query_orchestrator.semantic_index_adapter import (
    Embedder,
    SemanticIndexAdapter,
)
from prsm.node.content_uploader import _SemanticIndex


# ──────────────────────────────────────────────────────────────────────
# Stubs
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _StubEmbedder:
    """Test-only Embedder. Returns a fixed vector regardless of input
    by default; deterministic per query if _vector_for is set."""
    canned: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))
    queries_seen: list = field(default_factory=list)

    def encode(self, query: str) -> np.ndarray:
        self.queries_seen.append(query)
        return self.canned


def _vec(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def _make_index(*entries) -> _SemanticIndex:
    idx = _SemanticIndex()
    for cid, vec, creator in entries:
        idx.store(cid, vec, creator)
    return idx


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_adapter_threads_query_through_embedder_to_index(self):
        index = _make_index(
            ("a", _vec(1.0, 0.0), "alice"),
            ("b", _vec(0.0, 1.0), "bob"),
        )
        embedder = _StubEmbedder(canned=_vec(1.0, 0.0))
        adapter = SemanticIndexAdapter(embedder=embedder, index=index)

        out = adapter.find_top_k("query about a", k=2)

        # The query string was passed to the embedder.
        assert embedder.queries_seen == ["query about a"]
        # Top-1 = the closest stored vector.
        assert out[0][0] == "a"

    def test_returns_list_of_triples(self):
        index = _make_index(
            ("a", _vec(1.0, 0.0), "alice"),
        )
        adapter = SemanticIndexAdapter(
            embedder=_StubEmbedder(canned=_vec(1.0, 0.0)),
            index=index,
        )
        out = adapter.find_top_k("any query", k=1)
        assert len(out) == 1
        cid, sim, creator = out[0]
        assert cid == "a"
        assert isinstance(sim, float)
        assert creator == "alice"

    def test_satisfies_shard_finder_protocol_runtime_check(self):
        from prsm.compute.query_orchestrator import SemanticIndex
        adapter = SemanticIndexAdapter(
            embedder=_StubEmbedder(),
            index=_SemanticIndex(),
        )
        # The adapter MUST be Protocol-compatible — shard_finder will
        # accept it as a SemanticIndex.
        assert isinstance(adapter, SemanticIndex)


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_index_returns_empty(self):
        adapter = SemanticIndexAdapter(
            embedder=_StubEmbedder(canned=_vec(1.0, 0.0)),
            index=_SemanticIndex(),
        )
        assert adapter.find_top_k("anything", k=5) == []

    def test_k_larger_than_index_returns_full(self):
        index = _make_index(
            ("a", _vec(1.0, 0.0), "alice"),
            ("b", _vec(0.95, 0.05), "bob"),
        )
        adapter = SemanticIndexAdapter(
            embedder=_StubEmbedder(canned=_vec(1.0, 0.0)),
            index=index,
        )
        out = adapter.find_top_k("q", k=999)
        assert len(out) == 2

    def test_embedder_called_once_per_lookup(self):
        # Don't accidentally call the embedder twice — embedding is
        # the expensive step.
        embedder = _StubEmbedder(canned=_vec(1.0, 0.0))
        adapter = SemanticIndexAdapter(
            embedder=embedder, index=_make_index(("a", _vec(1.0, 0.0), "x")),
        )
        adapter.find_top_k("q", k=5)
        assert len(embedder.queries_seen) == 1


# ──────────────────────────────────────────────────────────────────────
# Embedder Protocol pin
# ──────────────────────────────────────────────────────────────────────


class TestEmbedderProtocolPin:
    """Pin the Embedder shape — production wiring must satisfy
    `encode(query: str) -> np.ndarray`. If a future change moves to
    async or batch encoding, this test blows."""

    def test_encode_takes_string_returns_ndarray(self):
        e = _StubEmbedder()
        result = e.encode("anything")
        assert isinstance(result, np.ndarray)

    def test_embedder_protocol_runtime_checkable(self):
        e = _StubEmbedder()
        assert isinstance(e, Embedder)
