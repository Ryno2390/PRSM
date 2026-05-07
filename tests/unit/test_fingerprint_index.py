"""
PRSM-PROV-1 Item 4 T4.6 — FingerprintIndex tests.

Covers:
  - Empty-index behavior: find_nearest returns None.
  - Per-kind dispatch: image record never matches a structural-kind
    stored entry and vice versa.
  - Single-store + find: record matches itself with similarity 1.0.
  - Multiple stored records: find_nearest picks the highest-similarity
    one (per the backend's similarity function).
  - Backends without registration: find_nearest returns None when no
    backend is configured for the kind.
  - Backends without registration: store still accepts the record
    (forgiving for cross-version persistence).
  - Persistence: round-trip through a JSON file preserves index state.
  - threshold accessors return defaults / overrides correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from prsm.data.fingerprints import (
    FingerprintIndex,
    FingerprintKind,
    FingerprintMatch,
    FingerprintRecord,
)
from prsm.data.fingerprints.base import BinaryFingerprint


# ──────────────────────────────────────────────────────────────────────
# Stub backend — Hamming-style similarity over fixed-length payloads.
# Lets the index tests run without any optional fingerprint deps.
# ──────────────────────────────────────────────────────────────────────


class _StubImageBackend(BinaryFingerprint):
    KIND = FingerprintKind.IMAGE_PHASH
    PAYLOAD_LEN = 8

    def compute(
        self, content: bytes, *, filename: Optional[str] = None,
    ) -> Optional[FingerprintRecord]:
        if not content:
            return None
        # Simple synthetic hash: bytes 0..7 modulo content length, no
        # collisions for our tests.
        h = (hash(content) & ((1 << 64) - 1)).to_bytes(self.PAYLOAD_LEN, "big")
        return FingerprintRecord(kind=self.KIND, payload=h)

    def similarity(self, a: bytes, b: bytes) -> float:
        if len(a) != self.PAYLOAD_LEN or len(b) != self.PAYLOAD_LEN:
            return 0.0
        diff_bits = sum((x ^ y).bit_count() for x, y in zip(a, b))
        return 1.0 - diff_bits / (self.PAYLOAD_LEN * 8)


class _StubStructuralBackend(BinaryFingerprint):
    KIND = FingerprintKind.STRUCTURAL
    PAYLOAD_LEN = 32

    def compute(
        self, content: bytes, *, filename: Optional[str] = None,
    ) -> Optional[FingerprintRecord]:
        if not content:
            return None
        import hashlib
        return FingerprintRecord(
            kind=self.KIND,
            payload=hashlib.sha256(content).digest(),
        )

    def similarity(self, a: bytes, b: bytes) -> float:
        if len(a) != self.PAYLOAD_LEN or len(b) != self.PAYLOAD_LEN:
            return 0.0
        return 1.0 if a == b else 0.0


@pytest.fixture
def index() -> FingerprintIndex:
    return FingerprintIndex(
        backends={
            FingerprintKind.IMAGE_PHASH: _StubImageBackend(),
            FingerprintKind.STRUCTURAL: _StubStructuralBackend(),
        }
    )


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


class TestFingerprintIndex:
    def test_empty_find_nearest_returns_none(self, index):
        record = FingerprintRecord(
            kind=FingerprintKind.IMAGE_PHASH, payload=b"\x00" * 8,
        )
        assert index.find_nearest(record) is None
        assert len(index) == 0
        assert index.size(FingerprintKind.IMAGE_PHASH) == 0

    def test_store_then_find_self_match(self, index):
        record = FingerprintRecord(
            kind=FingerprintKind.IMAGE_PHASH, payload=b"\x01" * 8,
        )
        index.store("cid-A", record, "creator-A")
        result = index.find_nearest(record)
        assert isinstance(result, FingerprintMatch)
        assert result.content_id == "cid-A"
        assert result.similarity == pytest.approx(1.0)
        assert result.creator_id == "creator-A"
        assert result.kind == FingerprintKind.IMAGE_PHASH

    def test_per_kind_isolation(self, index):
        """An image record should never match a structural-stored entry."""
        image_record = FingerprintRecord(
            kind=FingerprintKind.IMAGE_PHASH, payload=b"\x42" * 8,
        )
        structural_record = FingerprintRecord(
            kind=FingerprintKind.STRUCTURAL, payload=b"\x00" * 32,
        )
        index.store("structural-cid", structural_record, "creator-S")
        # Querying with image kind must NOT see the structural entry.
        assert index.find_nearest(image_record) is None

    def test_finds_highest_similarity_among_many(self, index):
        anchor = b"\x00" * 8
        # Stored payloads with varying Hamming distances from anchor.
        stored = {
            "cid-distant": b"\xff" * 8,  # 64 bits flipped — sim 0.0
            "cid-close": b"\x01" + b"\x00" * 7,  # 1 bit flipped — sim 63/64
            "cid-medium": b"\x0f" + b"\x00" * 7,  # 4 bits — sim 60/64
        }
        for cid, payload in stored.items():
            index.store(
                cid,
                FingerprintRecord(kind=FingerprintKind.IMAGE_PHASH, payload=payload),
                f"creator-{cid}",
            )

        anchor_record = FingerprintRecord(
            kind=FingerprintKind.IMAGE_PHASH, payload=anchor,
        )
        result = index.find_nearest(anchor_record)
        assert result is not None
        assert result.content_id == "cid-close"
        assert result.similarity == pytest.approx(63 / 64)

    def test_no_backend_registered_returns_none(self):
        """find_nearest for a kind with no backend returns None."""
        index = FingerprintIndex(
            backends={FingerprintKind.IMAGE_PHASH: _StubImageBackend()}
        )
        # Audio backend not registered.
        record = FingerprintRecord(
            kind=FingerprintKind.AUDIO_CHROMAPRINT, payload=b"\x00" * 32,
        )
        assert index.find_nearest(record) is None
        assert not index.has_backend(FingerprintKind.AUDIO_CHROMAPRINT)

    def test_store_accepts_unknown_kind(self):
        """Forgiving: callers can store records of kinds we don't have a
        backend for. find_nearest will just return None for them — but
        the data isn't lost (lets us load a persisted file written by a
        node with more backends than we currently have)."""
        index = FingerprintIndex(
            backends={FingerprintKind.IMAGE_PHASH: _StubImageBackend()}
        )
        record = FingerprintRecord(
            kind=FingerprintKind.AUDIO_CHROMAPRINT, payload=b"\x00" * 32,
        )
        index.store("audio-cid", record, "creator-X")
        assert index.size(FingerprintKind.AUDIO_CHROMAPRINT) == 1
        # find_nearest still returns None — no backend.
        assert index.find_nearest(record) is None

    def test_threshold_accessors(self):
        index = FingerprintIndex(
            backends={FingerprintKind.IMAGE_PHASH: _StubImageBackend()},
            duplicate_thresholds={FingerprintKind.IMAGE_PHASH: 0.99},
            derivative_thresholds={FingerprintKind.IMAGE_PHASH: 0.85},
        )
        assert index.duplicate_threshold(FingerprintKind.IMAGE_PHASH) == 0.99
        assert index.derivative_threshold(FingerprintKind.IMAGE_PHASH) == 0.85
        # Defaults for kinds without explicit overrides → 1.0.
        assert index.duplicate_threshold(FingerprintKind.AUDIO_CHROMAPRINT) == 1.0

    def test_persistence_round_trip(self, tmp_path: Path):
        persist = tmp_path / "fingerprints.json"
        backends = {
            FingerprintKind.IMAGE_PHASH: _StubImageBackend(),
            FingerprintKind.STRUCTURAL: _StubStructuralBackend(),
        }
        first = FingerprintIndex(backends=backends, persist_path=persist)

        record_a = FingerprintRecord(
            kind=FingerprintKind.IMAGE_PHASH, payload=b"\xaa" * 8,
        )
        record_b = FingerprintRecord(
            kind=FingerprintKind.STRUCTURAL, payload=b"\xbb" * 32,
        )
        first.store("cid-a", record_a, "creator-1")
        first.store("cid-b", record_b, "creator-2")

        assert persist.exists(), "expected JSON persist file to be written"

        # Build a second index from the same file — state should match.
        second = FingerprintIndex(backends=backends, persist_path=persist)
        assert len(second) == 2
        assert second.size(FingerprintKind.IMAGE_PHASH) == 1
        assert second.size(FingerprintKind.STRUCTURAL) == 1

        match_a = second.find_nearest(record_a)
        assert match_a is not None
        assert match_a.content_id == "cid-a"
        assert match_a.similarity == pytest.approx(1.0)
        assert match_a.creator_id == "creator-1"

        match_b = second.find_nearest(record_b)
        assert match_b is not None
        assert match_b.content_id == "cid-b"

    def test_compute_dispatches_to_backend(self, index):
        record = index.compute(b"hello world", FingerprintKind.IMAGE_PHASH)
        assert record is not None
        assert record.kind == FingerprintKind.IMAGE_PHASH
        assert len(record.payload) == 8

    def test_compute_returns_none_for_unregistered_kind(self):
        index = FingerprintIndex(
            backends={FingerprintKind.IMAGE_PHASH: _StubImageBackend()}
        )
        # No video backend registered.
        assert (
            index.compute(b"hello", FingerprintKind.VIDEO_MULTIHASH) is None
        )
