"""
PRSM-PROV-1 Item 4 T4.9 — LocalFingerprintIndex tests.

Covers:
  - Empty-index lookup returns None.
  - Round-trip register + lookup + reload-from-disk.
  - Per-kind partitioning: same content_hash under different kinds
    are distinct entries.
  - Unknown / corrupt kind values are rejected at construction.
  - Oversized payload rejected.
  - Index file is JSON-deserializable + diff-stable across writes.
  - lookup_creator_by_content_hash convenience accessor.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from prsm.network.embedding_dht.local_fingerprint_index import (
    LocalFingerprintIndex,
    LocalFingerprintRecord,
)
from prsm.network.embedding_dht.protocol import (
    ALLOWED_FINGERPRINT_KINDS,
    MAX_FINGERPRINT_PAYLOAD_BYTES,
)


VALID_HASH_A = "0x" + "ab" * 32
VALID_HASH_B = "0x" + "cd" * 32
VALID_SIG = base64.b64encode(b"x" * 64).decode()


def _make_record(
    content_hash: str = VALID_HASH_A,
    fingerprint_kind: str = "image-phash",
    payload: bytes = b"\xde\xad\xbe\xef" * 2,
    creator_id: str = "node-1",
    created_at: float = 1700000000.0,
) -> LocalFingerprintRecord:
    return LocalFingerprintRecord(
        content_hash=content_hash,
        fingerprint_kind=fingerprint_kind,
        payload_b64=base64.b64encode(payload).decode(),
        creator_id=creator_id,
        created_at=created_at,
        signature_b64=VALID_SIG,
    )


class TestLocalFingerprintRecord:
    def test_construction_round_trips_through_dict(self):
        rec = _make_record()
        again = LocalFingerprintRecord.from_dict(rec.to_dict())
        assert rec == again

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="fingerprint_kind"):
            LocalFingerprintRecord(
                content_hash=VALID_HASH_A,
                fingerprint_kind="text-vector",
                payload_b64=base64.b64encode(b"abcd").decode(),
                creator_id="x",
                created_at=1.0,
                signature_b64=VALID_SIG,
            )

    def test_empty_payload_rejected(self):
        # Empty string fails the non-empty guard; an empty-but-non-empty
        # base64 (e.g. "AA==" decodes to b"\x00") would still be one
        # byte and pass — only literally empty payload_b64 rejects here.
        with pytest.raises(ValueError, match="non-empty"):
            LocalFingerprintRecord(
                content_hash=VALID_HASH_A,
                fingerprint_kind="image-phash",
                payload_b64="",
                creator_id="x",
                created_at=1.0,
                signature_b64=VALID_SIG,
            )

    def test_oversized_payload_rejected(self):
        too_big = b"x" * (MAX_FINGERPRINT_PAYLOAD_BYTES + 1)
        with pytest.raises(ValueError, match="MAX_FINGERPRINT"):
            LocalFingerprintRecord(
                content_hash=VALID_HASH_A,
                fingerprint_kind="audio-chromaprint",
                payload_b64=base64.b64encode(too_big).decode(),
                creator_id="x",
                created_at=1.0,
                signature_b64=VALID_SIG,
            )


class TestLocalFingerprintIndex:
    def test_empty_lookup_returns_none(self, tmp_path: Path):
        idx = LocalFingerprintIndex(tmp_path)
        assert idx.lookup(VALID_HASH_A, "image-phash") is None
        assert len(idx) == 0

    def test_register_then_lookup(self, tmp_path: Path):
        idx = LocalFingerprintIndex(tmp_path)
        rec = _make_record()
        idx.register(rec)
        assert idx.lookup(rec.content_hash, "image-phash") == rec
        assert idx.has(rec.content_hash, "image-phash")
        assert (rec.content_hash, "image-phash") in idx
        assert len(idx) == 1

    def test_per_kind_partitioning(self, tmp_path: Path):
        """Same content_hash under different kinds = distinct entries."""
        idx = LocalFingerprintIndex(tmp_path)
        rec_image = _make_record(fingerprint_kind="image-phash")
        rec_video = _make_record(
            fingerprint_kind="video-multihash",
            payload=b"\xab" * 64,  # 64-byte video payload
        )
        idx.register(rec_image)
        idx.register(rec_video)
        assert len(idx) == 2
        assert idx.lookup(VALID_HASH_A, "image-phash") == rec_image
        assert idx.lookup(VALID_HASH_A, "video-multihash") == rec_video
        # And neither bleeds into a third kind.
        assert idx.lookup(VALID_HASH_A, "audio-chromaprint") is None

    def test_persist_and_reload(self, tmp_path: Path):
        idx = LocalFingerprintIndex(tmp_path)
        rec_a = _make_record(content_hash=VALID_HASH_A)
        rec_b = _make_record(content_hash=VALID_HASH_B, creator_id="node-2")
        idx.register(rec_a)
        idx.register(rec_b)

        # Reload from disk and verify both entries survived.
        reloaded = LocalFingerprintIndex(tmp_path)
        assert len(reloaded) == 2
        assert reloaded.lookup(VALID_HASH_A, "image-phash") == rec_a
        assert reloaded.lookup(VALID_HASH_B, "image-phash") == rec_b

    def test_index_file_is_json_list(self, tmp_path: Path):
        """The on-disk file shape is a list of record dicts."""
        idx = LocalFingerprintIndex(tmp_path)
        idx.register(_make_record())
        index_file = tmp_path / "fingerprint_index.json"
        assert index_file.exists()
        data = json.loads(index_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["fingerprint_kind"] == "image-phash"
        assert data[0]["content_hash"] == VALID_HASH_A

    def test_index_file_diff_stable_across_writes(self, tmp_path: Path):
        """Same set of records ⇒ same on-disk bytes (sort key invariant)."""
        idx1 = LocalFingerprintIndex(tmp_path)
        idx1.register(_make_record(fingerprint_kind="video-multihash",
                                    payload=b"\xab" * 64))
        idx1.register(_make_record(fingerprint_kind="image-phash"))
        first = (tmp_path / "fingerprint_index.json").read_bytes()

        # Re-register in different order — file should byte-match.
        for f in tmp_path.iterdir():
            f.unlink()
        idx2 = LocalFingerprintIndex(tmp_path)
        idx2.register(_make_record(fingerprint_kind="image-phash"))
        idx2.register(_make_record(fingerprint_kind="video-multihash",
                                    payload=b"\xab" * 64))
        second = (tmp_path / "fingerprint_index.json").read_bytes()
        assert first == second

    def test_unregister(self, tmp_path: Path):
        idx = LocalFingerprintIndex(tmp_path)
        rec = _make_record()
        idx.register(rec)
        assert idx.unregister(rec.content_hash, "image-phash") is True
        assert idx.lookup(rec.content_hash, "image-phash") is None
        # Idempotent: a second unregister returns False.
        assert idx.unregister(rec.content_hash, "image-phash") is False

    def test_unregister_unknown_kind_returns_false(self, tmp_path: Path):
        idx = LocalFingerprintIndex(tmp_path)
        idx.register(_make_record())
        # text-vector isn't in ALLOWED_FINGERPRINT_KINDS — defensive
        # path returns False without raising.
        assert idx.unregister(VALID_HASH_A, "text-vector") is False
        # The legitimate entry still exists.
        assert idx.has(VALID_HASH_A, "image-phash")

    def test_lookup_creator_by_content_hash(self, tmp_path: Path):
        idx = LocalFingerprintIndex(tmp_path)
        rec_image = _make_record(
            fingerprint_kind="image-phash", creator_id="creator-A",
        )
        idx.register(rec_image)
        assert idx.lookup_creator_by_content_hash(VALID_HASH_A) == "creator-A"
        # Unknown hash → None.
        assert idx.lookup_creator_by_content_hash(VALID_HASH_B) is None

    def test_list_kinds_for(self, tmp_path: Path):
        idx = LocalFingerprintIndex(tmp_path)
        idx.register(_make_record(fingerprint_kind="image-phash"))
        idx.register(_make_record(
            fingerprint_kind="video-multihash", payload=b"\xab" * 64,
        ))
        kinds = idx.list_kinds_for(VALID_HASH_A)
        assert kinds == ["image-phash", "video-multihash"]

    def test_constructor_rejects_missing_directory(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            LocalFingerprintIndex(tmp_path / "does-not-exist")

    def test_load_drops_corrupt_entries_keeps_valid_ones(self, tmp_path: Path):
        """One corrupt record must not DoS the entire index."""
        # Hand-craft an index file with one valid and one bad entry.
        valid_dict = _make_record().to_dict()
        bad_dict = {"this_is_not": "a fingerprint record"}
        index_file = tmp_path / "fingerprint_index.json"
        index_file.write_text(json.dumps([valid_dict, bad_dict]))

        idx = LocalFingerprintIndex(tmp_path)
        # Valid record loaded; bad one dropped silently with a warning.
        assert len(idx) == 1
        assert idx.lookup(VALID_HASH_A, "image-phash") is not None
