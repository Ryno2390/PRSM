"""Unit tests for prsm.storage.erasure.

Per docs/2026-04-22-phase7-storage-design-plan.md §6 Task 1.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from prsm.storage.erasure import (
    DEFAULT_K,
    DEFAULT_N,
    CorruptShardError,
    DuplicateShardError,
    ErasureError,
    ErasureMetadata,
    ErasureShard,
    InsufficientShardsError,
    PayloadChecksumError,
    decode,
    encode,
)


# -----------------------------------------------------------------------------
# encode
# -----------------------------------------------------------------------------


def test_encode_produces_n_shards_at_default_params():
    payload = b"PRSM erasure test payload" * 10
    meta, shards = encode(payload)
    assert meta.k == DEFAULT_K
    assert meta.n == DEFAULT_N
    assert len(shards) == DEFAULT_N
    assert all(isinstance(s, ErasureShard) for s in shards)


def test_encode_shard_indices_are_0_to_n_minus_1():
    meta, shards = encode(b"x" * 60)
    assert [s.index for s in shards] == list(range(DEFAULT_N))


def test_encode_all_shards_are_equal_size():
    meta, shards = encode(b"x" * 120)
    sizes = {len(s.data) for s in shards}
    assert len(sizes) == 1
    assert meta.shard_bytes == sizes.pop()


def test_encode_metadata_payload_sha256_matches_input():
    payload = b"canary"
    meta, _ = encode(payload)
    assert meta.payload_sha256 == hashlib.sha256(payload).hexdigest()


def test_encode_each_shard_has_correct_self_sha256():
    _, shards = encode(b"data" * 100)
    for s in shards:
        assert s.sha256 == hashlib.sha256(s.data).hexdigest()


def test_encode_empty_payload_produces_zero_shards():
    meta, shards = encode(b"")
    assert shards == []
    assert meta.payload_bytes == 0


def test_encode_rejects_invalid_k():
    with pytest.raises(ErasureError):
        encode(b"x", k=0, n=4)


def test_encode_rejects_n_less_than_k():
    with pytest.raises(ErasureError):
        encode(b"x", k=6, n=5)


def test_encode_custom_params_k3_n5():
    """k=3, n=5 matches Tier C Shamir parameters per plan §2."""
    payload = b"y" * 30
    meta, shards = encode(payload, k=3, n=5)
    assert len(shards) == 5
    assert meta.k == 3


# -----------------------------------------------------------------------------
# round-trip
# -----------------------------------------------------------------------------


def test_roundtrip_from_all_n_shards():
    payload = b"hello reed solomon"
    meta, shards = encode(payload)
    assert decode(meta, shards) == payload


def test_roundtrip_from_first_k_shards():
    payload = os.urandom(1024)
    meta, shards = encode(payload)
    recovered = decode(meta, shards[: DEFAULT_K])
    assert recovered == payload


def test_roundtrip_from_last_k_shards():
    """Shards 4..9 are a mix of primary + parity; decode must handle."""
    payload = os.urandom(2048)
    meta, shards = encode(payload)
    recovered = decode(meta, shards[-DEFAULT_K:])
    assert recovered == payload


def test_roundtrip_from_mixed_shard_indices():
    payload = os.urandom(3000)
    meta, shards = encode(payload)
    mixed = [shards[i] for i in (0, 2, 4, 6, 8, 9)]
    assert decode(meta, mixed) == payload


def test_roundtrip_uneven_payload_size():
    """Payload not evenly divisible by k → padding must be stripped."""
    payload = b"A" * 37  # 37 / 6 = 7 with remainder
    meta, shards = encode(payload)
    assert decode(meta, shards[: DEFAULT_K]) == payload
    assert decode(meta, shards[: DEFAULT_K]) != b"A" * 42  # padding not leaked


def test_roundtrip_short_payload():
    """Payload shorter than k bytes still works — each block ends up 1 byte."""
    payload = b"xyz"  # 3 bytes
    meta, shards = encode(payload)
    assert decode(meta, shards[: DEFAULT_K]) == payload


def test_roundtrip_large_payload():
    payload = os.urandom(1 * 1024 * 1024)  # 1 MiB
    meta, shards = encode(payload)
    assert decode(meta, shards[: DEFAULT_K]) == payload


def test_roundtrip_empty_payload():
    meta, shards = encode(b"")
    assert decode(meta, shards) == b""


# -----------------------------------------------------------------------------
# decode failure paths
# -----------------------------------------------------------------------------


def test_decode_with_fewer_than_k_shards_raises():
    payload = b"x" * 60
    meta, shards = encode(payload)
    with pytest.raises(InsufficientShardsError):
        decode(meta, shards[: DEFAULT_K - 1])


def test_decode_rejects_duplicate_indices():
    payload = b"x" * 60
    meta, shards = encode(payload)
    # Duplicate shard 0.
    dup = [shards[0], shards[0], shards[1], shards[2], shards[3], shards[4]]
    with pytest.raises(DuplicateShardError):
        decode(meta, dup)


def test_decode_rejects_out_of_range_index():
    payload = b"x" * 60
    meta, shards = encode(payload)
    bogus = ErasureShard(
        index=99,
        data=shards[0].data,
        sha256=shards[0].sha256,
    )
    # Supply k-1 valid + 1 out-of-range.
    with pytest.raises(ErasureError):
        decode(meta, list(shards[:5]) + [bogus])


def test_decode_rejects_tampered_shard():
    """Shard's data was modified but the stated sha256 is stale."""
    payload = b"x" * 60
    meta, shards = encode(payload)
    tampered = ErasureShard(
        index=0,
        data=b"\x00" * len(shards[0].data),
        sha256=shards[0].sha256,  # stale
    )
    with pytest.raises(CorruptShardError):
        decode(meta, [tampered] + list(shards[1:6]))


def test_decode_rejects_payload_checksum_mismatch():
    """Forged shards with consistent per-shard hashes but wrong overall
    payload sha256 → PayloadChecksumError at finalize."""
    payload = b"x" * 60
    meta, shards = encode(payload)
    # Fake new metadata with wrong payload_sha256.
    bad_meta = ErasureMetadata(
        k=meta.k,
        n=meta.n,
        payload_bytes=meta.payload_bytes,
        shard_bytes=meta.shard_bytes,
        payload_sha256="0" * 64,  # wrong
    )
    with pytest.raises(PayloadChecksumError):
        decode(bad_meta, shards[: DEFAULT_K])


# -----------------------------------------------------------------------------
# Shard.verify direct test
# -----------------------------------------------------------------------------


def test_shard_verify_passes_for_consistent_sha():
    s = ErasureShard(
        index=0,
        data=b"hello",
        sha256=hashlib.sha256(b"hello").hexdigest(),
    )
    s.verify()  # no raise


def test_shard_verify_raises_on_inconsistent_sha():
    s = ErasureShard(index=0, data=b"hello", sha256="0" * 64)
    with pytest.raises(CorruptShardError):
        s.verify()


# -----------------------------------------------------------------------------
# Plan-acceptance scenario
# -----------------------------------------------------------------------------


def test_plan_acceptance_40_percent_shard_loss_recoverable():
    """Plan §7 acceptance criterion: 40% provider loss recoverable.

    k=6, n=10 → tolerates n-k = 4 shard losses. Killing shards 0, 2, 5, 9
    still leaves 6 shards (0.6 of total) — decode must succeed.
    """
    payload = os.urandom(4096)
    meta, shards = encode(payload)
    killed = {0, 2, 5, 9}
    survivors = [s for s in shards if s.index not in killed]
    assert len(survivors) == DEFAULT_N - len(killed) == 6
    assert decode(meta, survivors) == payload


def test_five_shard_loss_is_unrecoverable():
    """Killing 5 of 10 shards leaves only 5 survivors (below k=6). Decode
    MUST raise — we do not want silent partial recovery."""
    payload = os.urandom(1024)
    meta, shards = encode(payload)
    survivors = shards[:5]  # 5 shards, one short
    with pytest.raises(InsufficientShardsError):
        decode(meta, survivors)
