"""PRSM-PROV-1 Item 3 Task 2 — LocalEmbeddingIndex tests.

Verifies the on-disk JSON-backed map of (content_hash, model_id) →
embedding record: round-trip, persistence, key partitioning,
corruption tolerance, and key validation."""
from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import pytest

from prsm.network.embedding_dht.local_index import (
    LocalEmbeddingIndex,
    LocalEmbeddingRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    content_hash: str = "0xabc",
    model_id: str = "openai/text-embedding-ada-002",
    dimension: int = 4,
    creator_id: str = "creator-A",
    created_at: float = 1715000000.0,
) -> LocalEmbeddingRecord:
    raw = struct.pack(f"<{dimension}f", *[0.1 * i for i in range(dimension)])
    return LocalEmbeddingRecord(
        content_hash=content_hash,
        model_id=model_id,
        dimension=dimension,
        dtype="float32",
        vector_b64=base64.b64encode(raw).decode("ascii"),
        creator_id=creator_id,
        created_at=created_at,
        signature_b64=base64.b64encode(b"\x00" * 64).decode("ascii"),
    )


# ---------------------------------------------------------------------------
# LocalEmbeddingRecord
# ---------------------------------------------------------------------------


def test_record_round_trip_dict():
    rec = _make_record()
    out = LocalEmbeddingRecord.from_dict(rec.to_dict())
    assert out == rec


def test_record_rejects_dim_mismatch():
    raw = struct.pack("<2f", 0.1, 0.2)  # 8 bytes
    with pytest.raises(ValueError, match="decodes to"):
        LocalEmbeddingRecord(
            content_hash="0xabc",
            model_id="m",
            dimension=4,  # would need 16 bytes
            dtype="float32",
            vector_b64=base64.b64encode(raw).decode("ascii"),
            creator_id="c",
            created_at=1.0,
            signature_b64=base64.b64encode(b"\x00" * 64).decode("ascii"),
        )


def test_record_rejects_non_float32_dtype():
    raw = struct.pack("<4f", 0, 0, 0, 0)
    with pytest.raises(ValueError, match="dtype"):
        LocalEmbeddingRecord(
            content_hash="0xabc",
            model_id="m",
            dimension=4,
            dtype="float16",
            vector_b64=base64.b64encode(raw).decode("ascii"),
            creator_id="c",
            created_at=1.0,
            signature_b64=base64.b64encode(b"\x00" * 64).decode("ascii"),
        )


def test_record_rejects_unsafe_content_hash():
    with pytest.raises(ValueError, match="content_hash"):
        _make_record(content_hash="bad hash with spaces")


def test_record_rejects_unsafe_model_id():
    # Spaces and special chars (#, !, etc.) are outside the safe-key
    # regex character class. Slashes and dots are permitted because
    # legitimate model_ids look like "openai/text-embedding-ada-002".
    with pytest.raises(ValueError, match="model_id"):
        _make_record(model_id="bad model with spaces")


def test_record_rejects_reserved_name_as_id():
    with pytest.raises(ValueError, match="content_hash"):
        _make_record(content_hash="..")


def test_record_rejects_zero_dimension():
    with pytest.raises(ValueError, match="dimension"):
        LocalEmbeddingRecord(
            content_hash="0xabc",
            model_id="m",
            dimension=0,
            dtype="float32",
            vector_b64="AAAA",
            creator_id="c",
            created_at=1.0,
            signature_b64="BBBB",
        )


# ---------------------------------------------------------------------------
# LocalEmbeddingIndex — basic CRUD
# ---------------------------------------------------------------------------


def test_index_register_and_lookup(tmp_path: Path):
    idx = LocalEmbeddingIndex(tmp_path)
    rec = _make_record()
    idx.register(rec)
    assert idx.lookup(rec.content_hash, rec.model_id) == rec
    assert idx.has(rec.content_hash, rec.model_id)
    assert len(idx) == 1


def test_index_lookup_returns_none_for_unknown(tmp_path: Path):
    idx = LocalEmbeddingIndex(tmp_path)
    assert idx.lookup("0xnope", "m") is None
    assert not idx.has("0xnope", "m")


def test_index_unregister(tmp_path: Path):
    idx = LocalEmbeddingIndex(tmp_path)
    rec = _make_record()
    idx.register(rec)
    assert idx.unregister(rec.content_hash, rec.model_id) is True
    assert idx.lookup(rec.content_hash, rec.model_id) is None
    # Second unregister is a no-op returning False
    assert idx.unregister(rec.content_hash, rec.model_id) is False


def test_index_overwrite_silent(tmp_path: Path):
    idx = LocalEmbeddingIndex(tmp_path)
    a = _make_record(creator_id="creator-A", created_at=1.0)
    b = _make_record(creator_id="creator-B", created_at=2.0)
    idx.register(a)
    idx.register(b)
    found = idx.lookup(a.content_hash, a.model_id)
    assert found is not None
    assert found.creator_id == "creator-B"
    assert found.created_at == 2.0


# ---------------------------------------------------------------------------
# Cross-model partitioning
# ---------------------------------------------------------------------------


def test_index_partitions_by_model_id(tmp_path: Path):
    """The whole reason this index exists with a tuple key — same
    content_hash under two different model_ids must coexist."""
    idx = LocalEmbeddingIndex(tmp_path)
    a = _make_record(
        content_hash="0xabc",
        model_id="openai/text-embedding-ada-002",
        dimension=4,
    )
    b = _make_record(
        content_hash="0xabc",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dimension=4,
    )
    idx.register(a)
    idx.register(b)
    assert len(idx) == 2
    assert idx.lookup("0xabc", "openai/text-embedding-ada-002") == a
    assert idx.lookup(
        "0xabc", "sentence-transformers/all-MiniLM-L6-v2"
    ) == b
    assert idx.list_models_for("0xabc") == [
        "openai/text-embedding-ada-002",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]


def test_index_list_content_hashes_dedupes_across_models(tmp_path: Path):
    idx = LocalEmbeddingIndex(tmp_path)
    idx.register(_make_record(content_hash="0xa", model_id="m1"))
    idx.register(_make_record(content_hash="0xa", model_id="m2"))
    idx.register(_make_record(content_hash="0xb", model_id="m1"))
    assert idx.list_content_hashes() == ["0xa", "0xb"]


def test_index_list_keys_sorted(tmp_path: Path):
    idx = LocalEmbeddingIndex(tmp_path)
    idx.register(_make_record(content_hash="0xb", model_id="m2"))
    idx.register(_make_record(content_hash="0xa", model_id="m1"))
    idx.register(_make_record(content_hash="0xa", model_id="m2"))
    assert idx.list_keys() == [
        ("0xa", "m1"),
        ("0xa", "m2"),
        ("0xb", "m2"),
    ]


def test_index_contains_tuple_key(tmp_path: Path):
    idx = LocalEmbeddingIndex(tmp_path)
    rec = _make_record()
    idx.register(rec)
    assert (rec.content_hash, rec.model_id) in idx
    assert ("0xnope", "m") not in idx
    # Wrong shape — must not raise
    assert "not-a-tuple" not in idx
    assert (rec.content_hash,) not in idx


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_index_persists_across_instances(tmp_path: Path):
    idx1 = LocalEmbeddingIndex(tmp_path)
    rec = _make_record()
    idx1.register(rec)

    idx2 = LocalEmbeddingIndex(tmp_path)
    assert len(idx2) == 1
    assert idx2.lookup(rec.content_hash, rec.model_id) == rec


def test_index_atomic_write_uses_tmp(tmp_path: Path):
    idx = LocalEmbeddingIndex(tmp_path)
    idx.register(_make_record())
    # After register, no .tmp file should remain (os.replace consumed it).
    assert not (tmp_path / "embedding_index.json.tmp").exists()
    assert (tmp_path / "embedding_index.json").exists()


def test_index_persistence_is_deterministic(tmp_path: Path):
    """Same set of registrations should produce byte-identical
    embedding_index.json regardless of insertion order. Important
    for git-friendliness if an operator commits the file."""
    idx_a = LocalEmbeddingIndex(tmp_path)
    idx_a.register(_make_record(content_hash="0xa", model_id="m1"))
    idx_a.register(_make_record(content_hash="0xb", model_id="m2"))
    bytes_a = (tmp_path / "embedding_index.json").read_bytes()

    # Different tmp dir, opposite insertion order
    other = tmp_path / "other"
    other.mkdir()
    idx_b = LocalEmbeddingIndex(other)
    idx_b.register(_make_record(content_hash="0xb", model_id="m2"))
    idx_b.register(_make_record(content_hash="0xa", model_id="m1"))
    bytes_b = (other / "embedding_index.json").read_bytes()

    assert bytes_a == bytes_b


# ---------------------------------------------------------------------------
# Corruption tolerance
# ---------------------------------------------------------------------------


def test_index_recovers_from_corrupt_json(tmp_path: Path):
    (tmp_path / "embedding_index.json").write_text("this is not json {{{")
    idx = LocalEmbeddingIndex(tmp_path)
    assert len(idx) == 0
    # Should be usable after recovery
    rec = _make_record()
    idx.register(rec)
    assert idx.lookup(rec.content_hash, rec.model_id) == rec


def test_index_recovers_from_wrong_top_level_shape(tmp_path: Path):
    (tmp_path / "embedding_index.json").write_text(
        json.dumps({"not": "a list"})
    )
    idx = LocalEmbeddingIndex(tmp_path)
    assert len(idx) == 0


def test_index_drops_individual_invalid_entries(tmp_path: Path):
    """One bad apple shouldn't deny-of-service the whole index."""
    valid = _make_record()
    (tmp_path / "embedding_index.json").write_text(json.dumps([
        valid.to_dict(),
        {"content_hash": "0xbroken"},  # missing fields
        "not even a dict",
        {**valid.to_dict(), "dimension": "should_be_int"},  # bad dim type
    ]))
    idx = LocalEmbeddingIndex(tmp_path)
    assert len(idx) == 1
    assert idx.lookup(valid.content_hash, valid.model_id) == valid


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_index_rejects_nonexistent_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LocalEmbeddingIndex(tmp_path / "does-not-exist")


def test_index_rejects_root_that_is_a_file(tmp_path: Path):
    f = tmp_path / "a-file"
    f.write_text("hi")
    with pytest.raises(NotADirectoryError):
        LocalEmbeddingIndex(f)


# ---------------------------------------------------------------------------
# Defensive lookups
# ---------------------------------------------------------------------------


def test_lookup_returns_none_for_unsafe_key(tmp_path: Path):
    """Caller passing an unsafe key gets None instead of a stack trace."""
    idx = LocalEmbeddingIndex(tmp_path)
    assert idx.lookup("bad space", "m") is None
    assert idx.lookup("0xabc", "..") is None
    assert idx.lookup("", "m") is None


def test_unregister_returns_false_for_unsafe_key(tmp_path: Path):
    idx = LocalEmbeddingIndex(tmp_path)
    assert idx.unregister("bad space", "m") is False


def test_register_rejects_non_record(tmp_path: Path):
    idx = LocalEmbeddingIndex(tmp_path)
    with pytest.raises(TypeError, match="LocalEmbeddingRecord"):
        idx.register({"not": "a record"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Vector encoding sanity
# ---------------------------------------------------------------------------


def test_record_preserves_vector_bytes_round_trip(tmp_path: Path):
    """End-to-end: register a known float vector, persist, reload,
    confirm bytes are byte-identical (no precision loss)."""
    raw = struct.pack("<8f", 1.5, 2.25, -3.125, 0.0, 7.875, -0.5, 100.0, 1e-3)
    rec = LocalEmbeddingRecord(
        content_hash="0xexact",
        model_id="m",
        dimension=8,
        dtype="float32",
        vector_b64=base64.b64encode(raw).decode("ascii"),
        creator_id="c",
        created_at=1.0,
        signature_b64=base64.b64encode(b"\x00" * 64).decode("ascii"),
    )
    idx = LocalEmbeddingIndex(tmp_path)
    idx.register(rec)
    fresh = LocalEmbeddingIndex(tmp_path)
    found = fresh.lookup("0xexact", "m")
    assert found is not None
    assert base64.b64decode(found.vector_b64) == raw
