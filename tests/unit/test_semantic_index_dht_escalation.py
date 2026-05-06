"""
PRSM-PROV-1 Item 3 Task 5 — _SemanticIndex DHT escalation.

Covers:
  - Local-only path unchanged when DHT not wired (regression guard).
  - find_nearest skips DHT when local match is already strong.
  - find_nearest escalates to DHT when local match is weak.
  - DHT-pulled embedding can become the new best match.
  - Cross-model isolation: model_id passed through unchanged.
  - All failure paths degrade silently to local-only — never raise.
  - Bounded by max_remote_pulls_per_query.
  - Already-cached CIDs are not re-fetched (idempotent).
  - _make_peer_candidates_fn filters records correctly.

Tests use real numpy + a stubbed dht_client (duck-typed) — the
dht_client.py contract has its own paired-with-server unit test
suite (test_embedding_dht_server_client.py).
"""

from __future__ import annotations

import base64
import struct
from typing import Iterable, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from prsm.network.embedding_dht.protocol import (
    EmbeddingResponse,
    ProviderInfo,
)
from prsm.node.content_uploader import ContentUploader, _SemanticIndex


# ---- helpers --------------------------------------------------------


def _vec(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def _normalised(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _make_response(
    vector: np.ndarray,
    *,
    content_hash: str = "0xdead",
    model_id: str = "m1",
    creator_id: str = "creator-X",
) -> EmbeddingResponse:
    """Build an EmbeddingResponse around a raw vector. Signature is
    fixed bogus base64 — _SemanticIndex never re-verifies (the
    EmbeddingDHTClient is responsible for that, and we stub it here)."""
    raw = struct.pack(f"<{vector.shape[0]}f", *vector.tolist())
    return EmbeddingResponse(
        request_id="r" * 32,
        content_hash=content_hash,
        model_id=model_id,
        dimension=vector.shape[0],
        dtype="float32",
        vector_b64=base64.b64encode(raw).decode("ascii"),
        creator_id=creator_id,
        created_at=1234567890.0,
        signature_b64=base64.b64encode(b"\x00" * 64).decode("ascii"),
    )


class _StubDht:
    """Duck-typed EmbeddingDHTClient.

    Configurable per-test. Records calls so tests can assert what was
    asked for and how many times.
    """

    def __init__(
        self,
        *,
        providers: Optional[List[ProviderInfo]] = None,
        find_providers_raises: Optional[Exception] = None,
        fetch_responses: Optional[dict] = None,
    ) -> None:
        self.providers_to_return = providers or []
        self.find_providers_raises = find_providers_raises
        # Map (content_hash, model_id, provider_node_id) → response or
        # exception. None means "raise EmbeddingNotFoundError".
        self.fetch_responses = fetch_responses or {}
        self.find_providers_calls: List[Tuple[str, str]] = []
        self.fetch_calls: List[Tuple[str, str, str]] = []

    def find_providers(self, content_hash: str, model_id: str):
        self.find_providers_calls.append((content_hash, model_id))
        if self.find_providers_raises is not None:
            raise self.find_providers_raises
        return list(self.providers_to_return)

    def fetch_embedding(
        self,
        provider: ProviderInfo,
        content_hash: str,
        model_id: str,
    ):
        self.fetch_calls.append((provider.node_id, content_hash, model_id))
        key = (content_hash, model_id, provider.node_id)
        if key not in self.fetch_responses:
            raise RuntimeError(f"unconfigured fetch: {key}")
        outcome = self.fetch_responses[key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# ---- tests: dht_enabled flag ----------------------------------------


def test_dht_disabled_when_no_args():
    idx = _SemanticIndex()
    assert idx.dht_enabled is False


def test_dht_disabled_when_only_model_id():
    idx = _SemanticIndex(model_id="m1")
    assert idx.dht_enabled is False


def test_dht_disabled_when_only_dht_client():
    idx = _SemanticIndex(dht_client=_StubDht())
    assert idx.dht_enabled is False


def test_dht_disabled_when_only_peer_fn():
    idx = _SemanticIndex(peer_candidates_fn=lambda: [])
    assert idx.dht_enabled is False


def test_dht_enabled_when_all_three_set():
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=_StubDht(),
        peer_candidates_fn=lambda: [],
    )
    assert idx.dht_enabled is True


# ---- tests: local-only regression guards ----------------------------


def test_find_nearest_returns_none_on_empty_index():
    idx = _SemanticIndex()
    assert idx.find_nearest(_vec(1.0, 0.0)) is None


def test_find_nearest_returns_none_for_zero_norm_query():
    idx = _SemanticIndex()
    idx.store("a", _vec(1.0, 0.0), "ca")
    assert idx.find_nearest(_vec(0.0, 0.0)) is None


def test_find_nearest_local_only_match():
    idx = _SemanticIndex()
    idx.store("a", _vec(1.0, 0.0), "ca")
    idx.store("b", _vec(0.0, 1.0), "cb")
    match = idx.find_nearest(_vec(0.99, 0.05))
    assert match is not None
    assert match[0] == "a"
    assert match[2] == "ca"


# ---- tests: short-circuit on strong local match ---------------------


def test_strong_local_match_skips_dht():
    """A local match >= DERIVATIVE_THRESHOLD must NOT trigger DHT
    escalation — saves the network round-trip on the common case."""
    dht = _StubDht()
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=dht,
        peer_candidates_fn=lambda: [("peer-cid", "0xpeer")],
    )
    idx.store("local-cid", _vec(1.0, 0.0), "local-creator")

    match = idx.find_nearest(_vec(1.0, 0.0))
    assert match is not None
    assert match[0] == "local-cid"
    assert match[1] >= _SemanticIndex.DERIVATIVE_THRESHOLD
    # Critical: DHT must not have been called.
    assert dht.find_providers_calls == []
    assert dht.fetch_calls == []


# ---- tests: escalation triggers + cross-model isolation -------------


def test_weak_local_match_triggers_dht_pull():
    """When local best similarity is below DERIVATIVE_THRESHOLD, the
    DHT is consulted, the pulled vector is added to the local index,
    and find_nearest re-scans."""
    peer_cid = "peer-cid"
    peer_hash = "0xpeerhash"
    peer_vec = _normalised(_vec(0.99, 0.10, 0.05))  # close to query

    dht = _StubDht(
        providers=[ProviderInfo(node_id="peer1", address="h1:8000")],
        fetch_responses={
            (peer_hash, "m1", "peer1"): _make_response(
                peer_vec, content_hash=peer_hash, creator_id="peer-creator",
            )
        },
    )
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=dht,
        peer_candidates_fn=lambda: [(peer_cid, peer_hash)],
    )
    # Local index has one bad candidate so best_local is weak.
    idx.store("orthogonal", _vec(0.0, 1.0, 0.0), "creator-Y")

    query = _vec(1.0, 0.0, 0.0)
    match = idx.find_nearest(query)

    # Pulled vector should be the new best match.
    assert match is not None
    assert match[0] == peer_cid
    assert match[2] == "peer-creator"
    # Cross-model isolation: DHT was called with our model_id only.
    assert dht.find_providers_calls == [(peer_hash, "m1")]


def test_dht_call_includes_correct_model_id():
    dht = _StubDht(
        providers=[ProviderInfo(node_id="p1", address="h:1")],
        fetch_responses={
            ("0xhash", "openai/text-embedding-3-small", "p1"):
                _make_response(
                    _normalised(_vec(1.0, 0.0)),
                    content_hash="0xhash",
                    model_id="openai/text-embedding-3-small",
                )
        },
    )
    idx = _SemanticIndex(
        model_id="openai/text-embedding-3-small",
        dht_client=dht,
        peer_candidates_fn=lambda: [("cid", "0xhash")],
    )
    idx.find_nearest(_vec(1.0, 0.0))
    assert dht.find_providers_calls == [
        ("0xhash", "openai/text-embedding-3-small")
    ]


# ---- tests: failure paths degrade silently --------------------------


def test_peer_candidates_fn_raising_does_not_propagate():
    def boom():
        raise RuntimeError("gossip layer broken")

    idx = _SemanticIndex(
        model_id="m1",
        dht_client=_StubDht(),
        peer_candidates_fn=boom,
    )
    idx.store("a", _vec(1.0, 0.0), "ca")
    # Should not raise; falls back to local-only result.
    match = idx.find_nearest(_vec(0.0, 1.0))
    assert match is not None
    assert match[0] == "a"


def test_find_providers_raising_does_not_propagate():
    dht = _StubDht(
        find_providers_raises=ConnectionError("network down"),
    )
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=dht,
        peer_candidates_fn=lambda: [("peer-cid", "0xhash")],
    )
    idx.store("a", _vec(1.0, 0.0), "ca")
    match = idx.find_nearest(_vec(0.0, 1.0))
    assert match is not None
    assert match[0] == "a"
    assert len(dht.find_providers_calls) == 1


def test_fetch_failure_falls_through_to_next_provider():
    """When one provider's fetch_embedding raises, the next provider
    in the list is tried. This is the eclipse / sybil defense — a
    single malicious peer cannot block honest peers."""
    good_vec = _normalised(_vec(0.95, 0.30))
    dht = _StubDht(
        providers=[
            ProviderInfo(node_id="bad", address="h:1"),
            ProviderInfo(node_id="good", address="h:2"),
        ],
        fetch_responses={
            ("0xh", "m1", "bad"): RuntimeError("poisoned!"),
            ("0xh", "m1", "good"): _make_response(
                good_vec, content_hash="0xh", creator_id="real-creator",
            ),
        },
    )
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=dht,
        peer_candidates_fn=lambda: [("peer-cid", "0xh")],
    )
    idx.store("orthogonal", _vec(0.0, 1.0), "ca")

    match = idx.find_nearest(_vec(1.0, 0.0))
    assert match is not None
    assert match[0] == "peer-cid"
    assert match[2] == "real-creator"
    # Both providers should have been tried in order.
    assert [c[0] for c in dht.fetch_calls] == ["bad", "good"]


def test_all_providers_failing_returns_local_best():
    dht = _StubDht(
        providers=[ProviderInfo(node_id="bad1", address="h:1")],
        fetch_responses={
            ("0xh", "m1", "bad1"): RuntimeError("poisoned!"),
        },
    )
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=dht,
        peer_candidates_fn=lambda: [("peer-cid", "0xh")],
    )
    idx.store("a", _vec(1.0, 0.0), "ca")

    match = idx.find_nearest(_vec(0.0, 1.0))
    assert match is not None
    assert match[0] == "a"  # local-best, since DHT pull failed


def test_empty_providers_list_returns_local_best():
    dht = _StubDht(providers=[])  # nobody knows the embedding
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=dht,
        peer_candidates_fn=lambda: [("peer-cid", "0xh")],
    )
    idx.store("a", _vec(1.0, 0.0), "ca")
    match = idx.find_nearest(_vec(0.0, 1.0))
    assert match is not None
    assert match[0] == "a"


def test_malformed_vector_dimension_dropped():
    """If a returned vector_b64 decodes to the wrong byte length for
    its declared dimension, we MUST NOT cache it. Defensive against
    a peer whose record passed local validation but became corrupt
    in transit."""
    raw_vec = _normalised(_vec(0.99, 0.10))  # dim=2
    # Build a malformed response — claim dim=4 but pack 2 floats.
    raw = struct.pack(f"<{raw_vec.shape[0]}f", *raw_vec.tolist())
    bad_resp = EmbeddingResponse(
        request_id="r" * 32,
        content_hash="0xh",
        model_id="m1",
        dimension=4,  # LIE
        dtype="float32",
        vector_b64=base64.b64encode(raw).decode("ascii"),
        creator_id="c",
        created_at=1.0,
        signature_b64=base64.b64encode(b"\x00" * 64).decode("ascii"),
    )
    dht = _StubDht(
        providers=[ProviderInfo(node_id="p", address="h:1")],
        fetch_responses={("0xh", "m1", "p"): bad_resp},
    )
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=dht,
        peer_candidates_fn=lambda: [("peer-cid", "0xh")],
    )
    idx.store("a", _vec(1.0, 0.0), "ca")
    match = idx.find_nearest(_vec(0.0, 1.0))
    # Bad vector dropped; falls back to local-only best.
    assert match is not None
    assert match[0] == "a"


# ---- tests: idempotency + bounding ---------------------------------


def test_already_cached_cid_not_refetched():
    """If peer_candidates_fn yields a CID already in the local index,
    we MUST NOT re-fetch it from the DHT. Bounds work on warm cache."""
    dht = _StubDht()  # would raise if called: no fetch_responses
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=dht,
        peer_candidates_fn=lambda: [("peer-cid", "0xh")],
    )
    idx.store("peer-cid", _vec(1.0, 0.0), "creator-X")
    # Local match for query vec (0,1) is weak (sim=0), so DHT
    # would be consulted — but peer-cid is already cached, so the
    # candidate should be skipped entirely.
    idx.find_nearest(_vec(0.0, 1.0))
    # Critical: dht.find_providers should NOT have been called for
    # an already-cached CID.
    assert dht.find_providers_calls == []


def test_max_remote_pulls_per_query_bounds_work():
    """The cap on remote pulls per call must be respected."""
    candidates = [(f"cid-{i}", f"0xh{i}") for i in range(20)]
    fetch_responses = {
        (f"0xh{i}", "m1", "p"): _make_response(
            _normalised(_vec(0.001, 1.0)),  # all close to (0,1)
            content_hash=f"0xh{i}",
        )
        for i in range(20)
    }
    dht = _StubDht(
        providers=[ProviderInfo(node_id="p", address="h:1")],
        fetch_responses=fetch_responses,
    )
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=dht,
        peer_candidates_fn=lambda: candidates,
        max_remote_pulls_per_query=5,
    )
    idx.store("anchor", _vec(1.0, 0.0), "ca")
    idx.find_nearest(_vec(0.0, 1.0))  # weak local match → escalate
    # Exactly max_remote_pulls_per_query=5 fetches, no more.
    assert len(dht.fetch_calls) == 5


def test_pulled_embedding_persists_for_next_query():
    """After a DHT pull, a subsequent find_nearest with the same
    query MUST hit the local cache without re-fetching."""
    peer_vec = _normalised(_vec(1.0, 0.0))
    dht = _StubDht(
        providers=[ProviderInfo(node_id="p", address="h:1")],
        fetch_responses={
            ("0xh", "m1", "p"): _make_response(
                peer_vec, content_hash="0xh", creator_id="cp",
            ),
        },
    )
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=dht,
        peer_candidates_fn=lambda: [("peer-cid", "0xh")],
    )
    idx.store("orthogonal", _vec(0.0, 1.0), "co")

    # First call: pulls from DHT.
    idx.find_nearest(_vec(1.0, 0.0))
    assert len(dht.fetch_calls) == 1

    # Second call: peer-cid is now cached locally. No new fetch.
    idx.find_nearest(_vec(1.0, 0.0))
    assert len(dht.fetch_calls) == 1


# ---- tests: filtering inputs ---------------------------------------


def test_invalid_candidate_rows_skipped():
    """Empty cid, empty hash, non-string types must be silently skipped
    instead of crashing — peer_candidates_fn is uploader-supplied
    code that we don't trust."""
    dht = _StubDht(
        providers=[ProviderInfo(node_id="p", address="h:1")],
        fetch_responses={
            ("0xgood", "m1", "p"): _make_response(
                _normalised(_vec(1.0, 0.0)), content_hash="0xgood",
            ),
        },
    )
    idx = _SemanticIndex(
        model_id="m1",
        dht_client=dht,
        peer_candidates_fn=lambda: [
            ("", "0xempty-cid"),
            ("c1", ""),
            (None, "0xnone-cid"),
            ("c2", None),
            (123, "0xtypeerror"),
            ("good-cid", "0xgood"),
        ],
    )
    idx.store("anchor", _vec(0.0, 1.0), "ca")
    match = idx.find_nearest(_vec(1.0, 0.0))
    assert match is not None
    assert match[0] == "good-cid"
    # Only the good row should have triggered a fetch.
    assert dht.fetch_calls == [("p", "0xgood", "m1")]


# ---- tests: ContentUploader._make_peer_candidates_fn ----------------


def _record(
    *,
    embedding_id: Optional[str] = None,
    provenance_hash: Optional[str] = None,
):
    """Build a minimal ContentRecord-shaped object."""
    rec = MagicMock()
    rec.embedding_id = embedding_id
    rec.provenance_hash = provenance_hash
    return rec


def test_make_peer_candidates_fn_yields_only_qualified_records():
    content_index = MagicMock()
    content_index._records = {
        "good-cid": _record(
            embedding_id="emb:good", provenance_hash="0xgood",
        ),
        "no-prov": _record(
            embedding_id="emb:np", provenance_hash=None,
        ),
        "no-embed": _record(
            embedding_id=None, provenance_hash="0xne",
        ),
        "no-both": _record(
            embedding_id=None, provenance_hash=None,
        ),
        "empty-prov": _record(
            embedding_id="emb:ep", provenance_hash="",
        ),
    }
    supplier = ContentUploader._make_peer_candidates_fn(content_index)
    out = list(supplier())
    assert out == [("good-cid", "0xgood")]


def test_make_peer_candidates_fn_handles_missing_records_attr():
    """ContentIndex with no _records attr (or None) must yield empty
    list, not crash."""
    bogus = MagicMock(spec=[])  # no _records attribute
    supplier = ContentUploader._make_peer_candidates_fn(bogus)
    assert list(supplier()) == []

    none_idx = MagicMock()
    none_idx._records = None
    supplier = ContentUploader._make_peer_candidates_fn(none_idx)
    assert list(supplier()) == []


def test_make_peer_candidates_fn_lazily_re_evaluates():
    """A new advertisement landing in the index between two uploads
    must be picked up on the second find_nearest without any
    re-wiring."""
    content_index = MagicMock()
    content_index._records = {}
    supplier = ContentUploader._make_peer_candidates_fn(content_index)

    assert list(supplier()) == []

    # Simulate gossip arriving.
    content_index._records["new-cid"] = _record(
        embedding_id="emb:n", provenance_hash="0xn",
    )
    assert list(supplier()) == [("new-cid", "0xn")]
