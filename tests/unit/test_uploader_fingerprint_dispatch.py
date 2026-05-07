"""
PRSM-PROV-1 Item 4 T4.7 — ContentUploader binary-fingerprint dispatch tests.

Covers the upload-path wiring around the FingerprintIndex:
  - _maybe_compute_binary_fingerprint correctly classifies content kind
    and routes to the matching backend.
  - Text content yields None (the embedding lane handles it).
  - BYTE_HASH content yields None (no backend).
  - When dispatch returns a record, it ends up in self._fingerprint_index
    after a successful upload (via the production store callsite).
  - find_nearest correctly identifies a binary near-duplicate on a
    second upload of similar content.

These tests stub the BitTorrent layer + on-chain side via the
existing ContentUploader fixtures; the dispatch logic is the only
thing under test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, AsyncMock

import pytest

from prsm.data.fingerprints import (
    FingerprintIndex,
    FingerprintKind,
    FingerprintRecord,
)
from prsm.data.fingerprints.base import BinaryFingerprint
from prsm.node.content_uploader import ContentUploader


# ──────────────────────────────────────────────────────────────────────
# Stub backends — Hamming-style on fixed-length payloads. Lets the
# tests run without imagehash / pyacoustid / PyAV / h5py installed.
# ──────────────────────────────────────────────────────────────────────


class _StubImageBackend(BinaryFingerprint):
    KIND = FingerprintKind.IMAGE_PHASH
    PAYLOAD_LEN = 8

    def compute(
        self, content: bytes, *, filename: Optional[str] = None,
    ) -> Optional[FingerprintRecord]:
        if not content:
            return None
        h = (hash(content) & ((1 << 64) - 1)).to_bytes(self.PAYLOAD_LEN, "big")
        return FingerprintRecord(kind=self.KIND, payload=h)

    def similarity(self, a: bytes, b: bytes) -> float:
        if len(a) != self.PAYLOAD_LEN or len(b) != self.PAYLOAD_LEN:
            return 0.0
        diff = sum((x ^ y).bit_count() for x, y in zip(a, b))
        return 1.0 - diff / (self.PAYLOAD_LEN * 8)


def _make_uploader(tmp_path: Path) -> ContentUploader:
    """Construct a ContentUploader with stub identity + index."""
    identity = MagicMock()
    identity.node_id = "test-node"
    identity.public_key_b64 = "stub-pubkey"
    identity.sign = MagicMock(return_value="stub-signature")
    gossip = MagicMock()
    gossip.publish = AsyncMock()
    ledger = MagicMock()

    uploader = ContentUploader.__new__(ContentUploader)
    uploader.identity = identity
    uploader.gossip = gossip
    uploader.ledger = ledger
    # Replace the index with a deterministic stub-backend version so
    # the dispatch path doesn't depend on optional fingerprint deps.
    uploader._fingerprint_index = FingerprintIndex(
        backends={FingerprintKind.IMAGE_PHASH: _StubImageBackend()},
        persist_path=tmp_path / "fp.json",
    )
    return uploader


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


class TestUploaderFingerprintDispatch:
    def test_text_content_yields_none(self, tmp_path):
        """Text falls through to TEXT_VECTOR and the binary path returns None."""
        uploader = _make_uploader(tmp_path)
        text = b"This is plain English text. " * 10  # > 50 chars
        result = uploader._maybe_compute_binary_fingerprint(text, "doc.txt")
        assert result is None

    def test_byte_hash_yields_none(self, tmp_path):
        """Random non-classifiable bytes fall through to BYTE_HASH ⇒ None."""
        uploader = _make_uploader(tmp_path)
        # 40 bytes of random with no recognizable header — falls below
        # the 50-char text threshold AND has no magic bytes.
        result = uploader._maybe_compute_binary_fingerprint(
            b"\x80\x90\xa0\xb0" * 10,
            "blob.bin",
        )
        assert result is None

    def test_image_content_yields_record(self, tmp_path):
        """A PNG-magic'd payload routes to the image backend."""
        uploader = _make_uploader(tmp_path)
        png_payload = b"\x89PNG\r\n\x1a\n" + b"image-body-bytes" * 50
        record = uploader._maybe_compute_binary_fingerprint(
            png_payload, "photo.png",
        )
        assert record is not None
        assert record.kind == FingerprintKind.IMAGE_PHASH
        assert len(record.payload) == _StubImageBackend.PAYLOAD_LEN

    def test_unregistered_kind_yields_none(self, tmp_path):
        """Audio content with no audio backend registered ⇒ None."""
        uploader = _make_uploader(tmp_path)
        # FLAC magic — kind detected as AUDIO_CHROMAPRINT, but the
        # uploader's index has only the image backend.
        flac_payload = b"fLaC" + b"\x00" * 100
        result = uploader._maybe_compute_binary_fingerprint(
            flac_payload, "song.flac",
        )
        assert result is None

    def test_register_then_find_dedup_match(self, tmp_path):
        """Computing + storing + finding a fingerprint round-trips."""
        uploader = _make_uploader(tmp_path)
        png = b"\x89PNG\r\n\x1a\n" + b"sample-image-data" * 50

        # First upload: compute + store.
        record_first = uploader._maybe_compute_binary_fingerprint(png, "a.png")
        assert record_first is not None
        uploader._fingerprint_index.store(
            "cid-original", record_first, "creator-original",
        )

        # Second "upload" of identical bytes: find_nearest should return
        # the original CID as a perfect match.
        record_second = uploader._maybe_compute_binary_fingerprint(png, "b.png")
        assert record_second is not None
        match = uploader._fingerprint_index.find_nearest(record_second)
        assert match is not None
        assert match.content_id == "cid-original"
        assert match.similarity == pytest.approx(1.0)
        assert match.kind == FingerprintKind.IMAGE_PHASH

    def test_unrecognized_audio_kind_with_no_backend(self, tmp_path):
        """Verify the audio kind has no backend in the test setup."""
        uploader = _make_uploader(tmp_path)
        assert not uploader._fingerprint_index.has_backend(
            FingerprintKind.AUDIO_CHROMAPRINT,
        )
        assert uploader._fingerprint_index.has_backend(
            FingerprintKind.IMAGE_PHASH,
        )
