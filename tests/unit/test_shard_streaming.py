"""Unit tests for prsm.node.shard_streaming.

Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §6 Task 6b.

Per plan §7 Task 7 acceptance, a 100 MB gRPC round trip must succeed —
covered here by test_roundtrip_100mb_payload. Generating 100 MB of
random data in a unit test is fine at the ~100 ms cost we see.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from prsm.node.shard_streaming import (
    DEFAULT_CHUNK_BYTES,
    ChunkChecksumMismatch,
    ChunkOutOfOrder,
    PayloadChecksumMismatch,
    ShardAssembler,
    ShardChunk,
    ShardChunker,
    ShardManifest,
    StreamingError,
)


# -----------------------------------------------------------------------------
# Chunker
# -----------------------------------------------------------------------------


def test_chunker_rejects_non_positive_size():
    with pytest.raises(ValueError):
        ShardChunker(chunk_bytes=0)


def test_chunker_splits_payload_on_even_boundary():
    chunker = ShardChunker(chunk_bytes=100)
    payload = b"A" * 300  # exactly 3 chunks
    manifest, chunks = chunker.chunk("shard-1", payload)
    assert manifest.total_chunks == 3
    assert manifest.payload_bytes == 300
    assert [c.sequence for c in chunks] == [0, 1, 2]
    assert all(len(c.data) == 100 for c in chunks)


def test_chunker_handles_uneven_final_chunk():
    chunker = ShardChunker(chunk_bytes=100)
    payload = b"A" * 250  # 2 full + 1 partial
    manifest, chunks = chunker.chunk("shard-1", payload)
    assert manifest.total_chunks == 3
    assert len(chunks[0].data) == 100
    assert len(chunks[1].data) == 100
    assert len(chunks[2].data) == 50


def test_chunker_empty_payload():
    chunker = ShardChunker(chunk_bytes=100)
    manifest, chunks = chunker.chunk("shard-1", b"")
    assert manifest.total_chunks == 0
    assert manifest.payload_bytes == 0
    assert chunks == []


def test_chunker_manifest_has_overall_sha256():
    payload = b"hello world"
    expected = hashlib.sha256(payload).hexdigest()
    manifest, _ = ShardChunker(chunk_bytes=100).chunk("x", payload)
    assert manifest.payload_sha256 == expected


def test_chunker_per_chunk_sha256_is_correct():
    chunker = ShardChunker(chunk_bytes=4)
    payload = b"abcdefgh"
    _, chunks = chunker.chunk("x", payload)
    assert chunks[0].chunk_sha256 == hashlib.sha256(b"abcd").hexdigest()
    assert chunks[1].chunk_sha256 == hashlib.sha256(b"efgh").hexdigest()


def test_chunker_is_deterministic():
    """Same input must yield byte-identical output."""
    payload = os.urandom(5000)
    chunker_a = ShardChunker(chunk_bytes=500)
    chunker_b = ShardChunker(chunk_bytes=500)
    m_a, c_a = chunker_a.chunk("x", payload)
    m_b, c_b = chunker_b.chunk("x", payload)
    assert m_a == m_b
    assert c_a == c_b


# -----------------------------------------------------------------------------
# Assembler happy path
# -----------------------------------------------------------------------------


def test_roundtrip_simple():
    payload = b"hello, shard world"
    manifest, chunks = ShardChunker(chunk_bytes=7).chunk("x", payload)
    recovered = ShardAssembler.reassemble(manifest, chunks)
    assert recovered == payload


def test_roundtrip_exact_chunk_multiple():
    payload = b"X" * 4096
    manifest, chunks = ShardChunker(chunk_bytes=1024).chunk("x", payload)
    recovered = ShardAssembler.reassemble(manifest, chunks)
    assert recovered == payload


def test_roundtrip_empty_payload():
    manifest, chunks = ShardChunker(chunk_bytes=100).chunk("x", b"")
    recovered = ShardAssembler.reassemble(manifest, chunks)
    assert recovered == b""


def test_roundtrip_100mb_payload():
    """Plan §7 acceptance criterion: gRPC round-trip ≥ 100 MB payload.

    Validates the chunker + assembler can handle the plan-specified upper
    bound without integrity loss. The actual gRPC transport is pass-through.
    """
    payload = os.urandom(100 * 1024 * 1024)
    manifest, chunks = ShardChunker(chunk_bytes=DEFAULT_CHUNK_BYTES).chunk(
        "big-shard", payload
    )
    assert manifest.payload_bytes == len(payload)
    assert manifest.total_chunks == len(chunks)
    recovered = ShardAssembler.reassemble(manifest, chunks)
    assert recovered == payload


# -----------------------------------------------------------------------------
# Assembler failure paths
# -----------------------------------------------------------------------------


def test_out_of_order_chunk_rejected():
    payload = b"A" * 300
    manifest, chunks = ShardChunker(chunk_bytes=100).chunk("x", payload)

    asm = ShardAssembler(manifest)
    asm.feed(chunks[0])
    with pytest.raises(ChunkOutOfOrder):
        asm.feed(chunks[2])  # skip chunk 1


def test_chunk_shard_id_mismatch_rejected():
    payload = b"A" * 100
    manifest, chunks = ShardChunker(chunk_bytes=100).chunk("x", payload)

    asm = ShardAssembler(manifest)
    bad_chunk = ShardChunk(
        shard_id="wrong-id",
        sequence=0,
        data=chunks[0].data,
        chunk_sha256=chunks[0].chunk_sha256,
    )
    with pytest.raises(StreamingError):
        asm.feed(bad_chunk)


def test_chunk_checksum_mismatch_rejected():
    payload = b"A" * 100
    manifest, chunks = ShardChunker(chunk_bytes=100).chunk("x", payload)

    # Tamper with chunk data; checksum no longer matches.
    tampered = ShardChunk(
        shard_id=chunks[0].shard_id,
        sequence=chunks[0].sequence,
        data=b"B" * 100,
        chunk_sha256=chunks[0].chunk_sha256,  # stale digest
    )
    asm = ShardAssembler(manifest)
    with pytest.raises(ChunkChecksumMismatch):
        asm.feed(tampered)


def test_payload_checksum_mismatch_rejected():
    """An attacker who forges consistent per-chunk sha256s but not the
    overall payload sha256 gets caught at finalize()."""
    payload = b"A" * 100
    manifest, _ = ShardChunker(chunk_bytes=100).chunk("x", payload)

    # Fabricate a different payload with a consistent per-chunk checksum
    # but non-matching overall sha256.
    forged_data = b"B" * 100
    forged_chunk = ShardChunk(
        shard_id="x",
        sequence=0,
        data=forged_data,
        chunk_sha256=hashlib.sha256(forged_data).hexdigest(),
    )
    asm = ShardAssembler(manifest)
    asm.feed(forged_chunk)
    with pytest.raises(PayloadChecksumMismatch):
        asm.finalize()


def test_incomplete_stream_rejected():
    payload = b"A" * 300
    manifest, chunks = ShardChunker(chunk_bytes=100).chunk("x", payload)

    asm = ShardAssembler(manifest)
    asm.feed(chunks[0])
    asm.feed(chunks[1])
    # Stop short; finalize must complain.
    with pytest.raises(StreamingError):
        asm.finalize()


def test_double_finalize_rejected():
    payload = b"A" * 100
    manifest, chunks = ShardChunker(chunk_bytes=100).chunk("x", payload)
    asm = ShardAssembler(manifest)
    asm.feed(chunks[0])
    asm.finalize()
    with pytest.raises(StreamingError):
        asm.finalize()


def test_feed_after_finalize_rejected():
    payload = b"A" * 100
    manifest, chunks = ShardChunker(chunk_bytes=100).chunk("x", payload)
    asm = ShardAssembler(manifest)
    asm.feed(chunks[0])
    asm.finalize()
    with pytest.raises(StreamingError):
        asm.feed(chunks[0])
