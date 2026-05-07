"""PRSM-PROV-1 Item 4 T4.9.next2 — FingerprintIndex DHT escalation.

Mirrors the ``_SemanticIndex._pull_remote_embeddings`` test surface for
the binary-fingerprint lane. Verifies:

- ``dht_enabled`` requires both client and candidates wired
- escalation NO-OP when local best already ≥ derivative threshold
- escalation triggers when local result is None
- escalation triggers when local best < derivative threshold
- pulled fingerprints are scanned + can become the new best match
- per-kind partitioning extends to escalation: a request for kind A
  ignores stored kind-B fingerprints on the same content_hash
- already-cached candidates are skipped (idempotent)
- ``max_remote_pulls_per_query`` caps blocking time
- malformed wire-shape responses (bad base64, empty payload) are
  silently dropped
- peer_candidates_fn raises → silent skip, NEVER raise out
- find_fingerprint_providers raises → silent skip on that hash, others
  still pulled
- fetch_fingerprint signature failure → try next provider, keep going
- persistence: pulled fingerprints land on disk via _save()
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from prsm.data.fingerprints import (
    FingerprintIndex,
    FingerprintKind,
    FingerprintRecord,
)
from prsm.data.fingerprints.base import BinaryFingerprint


# ──────────────────────────────────────────────────────────────────────
# Test infrastructure
# ──────────────────────────────────────────────────────────────────────


class _StubImageBackend(BinaryFingerprint):
    """Hamming-style similarity over fixed 8-byte payloads."""

    KIND = FingerprintKind.IMAGE_PHASH
    PAYLOAD_LEN = 8

    def compute(self, content, *, filename=None):  # pragma: no cover
        if not content:
            return None
        h = (hash(content) & ((1 << 64) - 1)).to_bytes(self.PAYLOAD_LEN, "big")
        return FingerprintRecord(kind=self.KIND, payload=h)

    def similarity(self, a, b):
        if len(a) != self.PAYLOAD_LEN or len(b) != self.PAYLOAD_LEN:
            return 0.0
        diff_bits = sum((x ^ y).bit_count() for x, y in zip(a, b))
        return 1.0 - diff_bits / (self.PAYLOAD_LEN * 8)


class _StubVideoBackend(BinaryFingerprint):
    KIND = FingerprintKind.VIDEO_MULTIHASH
    PAYLOAD_LEN = 64

    def compute(self, content, *, filename=None):  # pragma: no cover
        return None

    def similarity(self, a, b):
        if len(a) != self.PAYLOAD_LEN or len(b) != self.PAYLOAD_LEN:
            return 0.0
        return 1.0 if a == b else 0.0


class _FakeFingerprintResponse:
    """Duck-typed FingerprintResponse — what the real DHT client returns."""

    def __init__(
        self, *, payload_b64: str, creator_id: str = "creator-remote",
    ) -> None:
        self.payload_b64 = payload_b64
        self.creator_id = creator_id


class _FakeProvider:
    def __init__(self, node_id: str = "peer-1") -> None:
        self.node_id = node_id


def _build_index(
    *,
    persist_path: Optional[Path] = None,
    dht_client=None,
    peer_candidates_fn=None,
    max_remote_pulls_per_query: Optional[int] = None,
    derivative_thresholds=None,
) -> FingerprintIndex:
    return FingerprintIndex(
        backends={
            FingerprintKind.IMAGE_PHASH: _StubImageBackend(),
            FingerprintKind.VIDEO_MULTIHASH: _StubVideoBackend(),
        },
        persist_path=persist_path,
        derivative_thresholds=derivative_thresholds,
        dht_client=dht_client,
        peer_candidates_fn=peer_candidates_fn,
        max_remote_pulls_per_query=max_remote_pulls_per_query,
    )


# ──────────────────────────────────────────────────────────────────────
# Wiring
# ──────────────────────────────────────────────────────────────────────


class TestWiring:
    def test_dht_disabled_without_both_kwargs(self):
        idx = _build_index()
        assert idx.dht_enabled is False

        idx_only_client = _build_index(dht_client=MagicMock())
        assert idx_only_client.dht_enabled is False

        idx_only_fn = _build_index(peer_candidates_fn=lambda: [])
        assert idx_only_fn.dht_enabled is False

    def test_dht_enabled_when_both_wired(self):
        idx = _build_index(
            dht_client=MagicMock(),
            peer_candidates_fn=lambda: [],
        )
        assert idx.dht_enabled is True


# ──────────────────────────────────────────────────────────────────────
# Escalation gating
# ──────────────────────────────────────────────────────────────────────


class TestEscalationGate:
    def test_strong_local_match_skips_dht(self):
        """Local already ≥ derivative threshold → no remote pull."""
        client = MagicMock()
        candidates_called = MagicMock(return_value=[])
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=candidates_called,
        )
        # Store payload that matches the query exactly.
        anchor = b"\x42" * 8
        idx.store(
            "cid-local",
            FingerprintRecord(kind=FingerprintKind.IMAGE_PHASH, payload=anchor),
            "creator-local",
        )
        result = idx.find_nearest(
            FingerprintRecord(kind=FingerprintKind.IMAGE_PHASH, payload=anchor),
        )
        assert result is not None
        assert result.similarity == pytest.approx(1.0)
        # Strong local match → DHT path never invoked.
        client.find_fingerprint_providers.assert_not_called()
        candidates_called.assert_not_called()

    def test_no_local_index_triggers_escalation(self):
        client = MagicMock()
        client.find_fingerprint_providers.return_value = []
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=lambda: [("cid-X", "0xabc")],
        )
        idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
            ),
        )
        # No providers → still attempted to consult the DHT.
        client.find_fingerprint_providers.assert_called_once_with(
            "0xabc", "image-phash",
        )

    def test_weak_local_triggers_escalation(self):
        """Local sim < derivative_threshold → pull remote."""
        client = MagicMock()
        client.find_fingerprint_providers.return_value = []
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=lambda: [("cid-X", "0xabc")],
        )
        # Stored payload is far from the query.
        idx.store(
            "cid-far",
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\xff" * 8,
            ),
            "creator-far",
        )
        idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
            ),
        )
        client.find_fingerprint_providers.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# Successful pull + re-scan
# ──────────────────────────────────────────────────────────────────────


class TestPullSuccess:
    def test_remote_payload_becomes_new_best(self):
        """A pulled remote fingerprint that's a better match than every
        local entry must become the new ``find_nearest`` result."""
        anchor = b"\x00" * 8
        remote_payload = b"\x00" * 8  # exact match to anchor
        client = MagicMock()
        client.find_fingerprint_providers.return_value = [_FakeProvider()]
        client.fetch_fingerprint.return_value = _FakeFingerprintResponse(
            payload_b64=base64.b64encode(remote_payload).decode(),
            creator_id="creator-remote",
        )
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=lambda: [("cid-remote", "0xabc")],
        )
        # Local has only a far entry.
        idx.store(
            "cid-far",
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\xff" * 8,
            ),
            "creator-far",
        )
        result = idx.find_nearest(
            FingerprintRecord(kind=FingerprintKind.IMAGE_PHASH, payload=anchor),
        )
        assert result is not None
        assert result.content_id == "cid-remote"
        assert result.creator_id == "creator-remote"
        assert result.similarity == pytest.approx(1.0)
        # The pull populated the local cache.
        assert idx.size(FingerprintKind.IMAGE_PHASH) == 2

    def test_already_cached_candidate_skipped(self):
        """If candidate's content_id is already in the local index for
        this kind, do NOT call fetch_fingerprint for it."""
        client = MagicMock()
        client.find_fingerprint_providers.return_value = [_FakeProvider()]
        client.fetch_fingerprint.return_value = _FakeFingerprintResponse(
            payload_b64=base64.b64encode(b"\x00" * 8).decode(),
        )
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=lambda: [
                ("cid-cached", "0xabc"),  # already local
                ("cid-new", "0xdef"),
            ],
        )
        # Pre-cache cid-cached with a far payload, so escalation still
        # fires but the candidate itself should be skipped.
        idx.store(
            "cid-cached",
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\xff" * 8,
            ),
            "creator-cached",
        )
        idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
            ),
        )
        # Only one provider lookup — for cid-new.
        client.find_fingerprint_providers.assert_called_once_with(
            "0xdef", "image-phash",
        )

    def test_max_pulls_per_query_caps_calls(self):
        """``max_remote_pulls_per_query`` bounds the network blocking."""
        client = MagicMock()
        client.find_fingerprint_providers.return_value = [_FakeProvider()]
        client.fetch_fingerprint.return_value = _FakeFingerprintResponse(
            payload_b64=base64.b64encode(b"\x55" * 8).decode(),
        )
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=lambda: [
                (f"cid-{i}", f"0x{i:040x}") for i in range(10)
            ],
            max_remote_pulls_per_query=3,
        )
        idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
            ),
        )
        # Only 3 provider lookups happened despite 10 candidates.
        assert client.find_fingerprint_providers.call_count == 3
        assert idx.size(FingerprintKind.IMAGE_PHASH) == 3


# ──────────────────────────────────────────────────────────────────────
# Per-kind partitioning extends to escalation
# ──────────────────────────────────────────────────────────────────────


class TestPerKindEscalation:
    def test_different_kind_query_uses_correct_kind_in_dht_call(self):
        client = MagicMock()
        client.find_fingerprint_providers.return_value = []
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=lambda: [("cid-X", "0xabc")],
        )
        # Query with VIDEO kind; must propagate kind="video-multihash"
        # to the DHT, NOT the image-phash kind.
        idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.VIDEO_MULTIHASH, payload=b"\x00" * 64,
            ),
        )
        client.find_fingerprint_providers.assert_called_once_with(
            "0xabc", "video-multihash",
        )


# ──────────────────────────────────────────────────────────────────────
# Failure-mode silent skip
# ──────────────────────────────────────────────────────────────────────


class TestFailureModes:
    def test_peer_candidates_fn_raises_silently_skipped(self):
        client = MagicMock()

        def boom():
            raise RuntimeError("gossip layer unavailable")

        idx = _build_index(dht_client=client, peer_candidates_fn=boom)
        # Must not raise out of find_nearest.
        result = idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
            ),
        )
        assert result is None
        client.find_fingerprint_providers.assert_not_called()

    def test_find_providers_raises_per_hash_skipped(self):
        """One bad hash doesn't poison the whole batch."""
        client = MagicMock()

        def find_providers(content_hash, kind):
            if content_hash == "0xbad":
                raise ConnectionError("network glitch")
            return [_FakeProvider()]

        client.find_fingerprint_providers.side_effect = find_providers
        client.fetch_fingerprint.return_value = _FakeFingerprintResponse(
            payload_b64=base64.b64encode(b"\x00" * 8).decode(),
        )
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=lambda: [
                ("cid-bad", "0xbad"),
                ("cid-good", "0xgood"),
            ],
        )
        idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
            ),
        )
        # cid-good still landed in the index; cid-bad did not.
        assert idx.size(FingerprintKind.IMAGE_PHASH) == 1

    def test_fetch_failure_tries_next_provider(self):
        """A poisoning peer cannot block fetching from honest peers."""
        client = MagicMock()
        bad_provider = _FakeProvider("peer-bad")
        good_provider = _FakeProvider("peer-good")
        client.find_fingerprint_providers.return_value = [
            bad_provider, good_provider,
        ]

        good_response = _FakeFingerprintResponse(
            payload_b64=base64.b64encode(b"\x00" * 8).decode(),
        )

        def fetch(provider, content_hash, kind):
            if provider.node_id == "peer-bad":
                raise ValueError("signature verification failed")
            return good_response

        client.fetch_fingerprint.side_effect = fetch
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=lambda: [("cid-X", "0xabc")],
        )
        idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
            ),
        )
        assert client.fetch_fingerprint.call_count == 2
        assert idx.size(FingerprintKind.IMAGE_PHASH) == 1

    def test_malformed_payload_skipped(self):
        """Bad base64 → drop the record, don't crash."""
        client = MagicMock()
        client.find_fingerprint_providers.return_value = [_FakeProvider()]
        client.fetch_fingerprint.return_value = _FakeFingerprintResponse(
            payload_b64="!!!not-valid-base64!!!",
        )
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=lambda: [("cid-X", "0xabc")],
        )
        idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
            ),
        )
        assert idx.size(FingerprintKind.IMAGE_PHASH) == 0

    def test_empty_payload_skipped(self):
        """Empty bytes after b64 decode → drop the record."""
        client = MagicMock()
        client.find_fingerprint_providers.return_value = [_FakeProvider()]
        client.fetch_fingerprint.return_value = _FakeFingerprintResponse(
            payload_b64=base64.b64encode(b"").decode(),
        )
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=lambda: [("cid-X", "0xabc")],
        )
        idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
            ),
        )
        assert idx.size(FingerprintKind.IMAGE_PHASH) == 0

    def test_invalid_candidate_tuple_silently_skipped(self):
        """peer_candidates_fn yielding garbage doesn't crash."""
        client = MagicMock()
        client.find_fingerprint_providers.return_value = [_FakeProvider()]
        client.fetch_fingerprint.return_value = _FakeFingerprintResponse(
            payload_b64=base64.b64encode(b"\x00" * 8).decode(),
        )
        idx = _build_index(
            dht_client=client,
            peer_candidates_fn=lambda: [
                (None, "0xabc"),       # cid not str
                ("cid-X", None),        # hash not str
                ("", "0xabc"),          # empty cid
                ("cid-Y", ""),           # empty hash
                ("cid-good", "0xgood"),  # the only valid one
            ],
        )
        idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
            ),
        )
        assert idx.size(FingerprintKind.IMAGE_PHASH) == 1


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_pulled_fingerprints_persist_to_disk(self, tmp_path: Path):
        persist = tmp_path / "fp_index.json"
        client = MagicMock()
        client.find_fingerprint_providers.return_value = [_FakeProvider()]
        client.fetch_fingerprint.return_value = _FakeFingerprintResponse(
            payload_b64=base64.b64encode(b"\x00" * 8).decode(),
            creator_id="creator-remote",
        )
        idx = _build_index(
            persist_path=persist,
            dht_client=client,
            peer_candidates_fn=lambda: [("cid-remote", "0xabc")],
        )
        idx.find_nearest(
            FingerprintRecord(
                kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
            ),
        )
        assert persist.exists()

        # Reload — DHT fields not needed for replay.
        reloaded = FingerprintIndex(
            backends={FingerprintKind.IMAGE_PHASH: _StubImageBackend()},
            persist_path=persist,
        )
        assert reloaded.size(FingerprintKind.IMAGE_PHASH) == 1
