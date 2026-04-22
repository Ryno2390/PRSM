"""Reed-Solomon erasure coding for PRSM shard durability.

Per docs/2026-04-22-phase7-storage-design-plan.md §2.1 + §6 Task 1.

Uses zfec (Tahoe-LAFS lineage) for the underlying Reed-Solomon math. Plan
explicitly requires a third-party RS implementation — no roll-your-own —
because the arithmetic is easy to get subtly wrong and the consequence
(silent corruption of stored content) is unrecoverable.

Default parameters `k=6, n=10`:
  * 10 shards produced per payload.
  * Any 6 suffice for reconstruction.
  * Tolerates up to 4 provider losses (40% shard loss recoverable per
    plan §7 acceptance criterion).

Integrity model:

  * Per-shard SHA-256 catches single-shard corruption on decode.
  * Overall payload SHA-256 (stored in metadata) catches cross-shard
    integrity violations that consistent per-shard checksums would
    miss (i.e., a coordinated forgery across multiple shards).
  * Decode refuses the shards it was given rather than silently
    skipping corrupt ones — the caller picks the K-of-N subset.

Scope boundary — what this module does NOT do:
  * Shard placement / provider selection (Task 2 ShardEngine).
  * Proof-of-retrievability challenge/response (Task 4).
  * Encryption (Task 5 — Tier B encryption wraps erasure output).
  * Key-share distribution (Task 7 Shamir for Tier C).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple

import zfec


__all__ = [
    "CorruptShardError",
    "DEFAULT_K",
    "DEFAULT_N",
    "DuplicateShardError",
    "ErasureMetadata",
    "ErasureShard",
    "ErasureError",
    "InsufficientShardsError",
    "PayloadChecksumError",
    "decode",
    "encode",
]


DEFAULT_K = 6
DEFAULT_N = 10


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class ErasureError(Exception):
    """Base for erasure-coding failures."""


class InsufficientShardsError(ErasureError):
    """Fewer than K distinct shards were supplied for decode."""


class DuplicateShardError(ErasureError):
    """Decode received the same shard index twice."""


class CorruptShardError(ErasureError):
    """A shard's per-shard SHA-256 did not match its stated digest."""


class PayloadChecksumError(ErasureError):
    """Reconstructed payload did not match metadata.payload_sha256."""


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ErasureMetadata:
    k: int
    n: int
    payload_bytes: int   # original plaintext length; used to trim padding
    shard_bytes: int     # length of each shard's data blob
    payload_sha256: str  # hex digest over the full plaintext


@dataclass(frozen=True)
class ErasureShard:
    index: int           # 0..n-1
    data: bytes
    sha256: str          # hex digest of `data`

    def verify(self) -> None:
        actual = hashlib.sha256(self.data).hexdigest()
        if actual != self.sha256:
            raise CorruptShardError(
                f"shard {self.index}: digest mismatch"
            )


# -----------------------------------------------------------------------------
# Encode
# -----------------------------------------------------------------------------


def encode(
    payload: bytes,
    *,
    k: int = DEFAULT_K,
    n: int = DEFAULT_N,
) -> Tuple[ErasureMetadata, List[ErasureShard]]:
    """Encode `payload` into `n` Reed-Solomon shards; any `k` of them
    are sufficient for reconstruction.

    Block-size rule: zfec requires k equal-sized input blocks. We split
    the payload into k blocks of `ceil(len(payload)/k)` bytes each, padding
    the last block with zero bytes. `metadata.payload_bytes` captures the
    original length so decode() strips the padding deterministically.
    """
    _validate_params(k, n)
    if len(payload) == 0:
        # Empty payload → metadata captures the fact; no shards to distribute.
        return (
            ErasureMetadata(
                k=k,
                n=n,
                payload_bytes=0,
                shard_bytes=0,
                payload_sha256=hashlib.sha256(b"").hexdigest(),
            ),
            [],
        )

    block_bytes = (len(payload) + k - 1) // k
    total_padded_bytes = block_bytes * k
    padded = payload + b"\x00" * (total_padded_bytes - len(payload))

    blocks = [padded[i * block_bytes : (i + 1) * block_bytes] for i in range(k)]

    encoder = zfec.Encoder(k, n)
    shares = encoder.encode(blocks)

    shards = [
        ErasureShard(
            index=i,
            data=bytes(shares[i]),
            sha256=hashlib.sha256(shares[i]).hexdigest(),
        )
        for i in range(n)
    ]

    metadata = ErasureMetadata(
        k=k,
        n=n,
        payload_bytes=len(payload),
        shard_bytes=block_bytes,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )
    return metadata, shards


# -----------------------------------------------------------------------------
# Decode
# -----------------------------------------------------------------------------


def decode(
    metadata: ErasureMetadata,
    shards: List[ErasureShard],
) -> bytes:
    """Reconstruct the original payload from `shards`.

    Requires at least `metadata.k` distinct shards. Shards beyond that
    minimum are NOT additionally verified against each other — the
    first `k` distinct-index shards are used for reconstruction, and
    the final payload-level SHA-256 is the cross-shard integrity check.
    """
    if metadata.payload_bytes == 0:
        # Empty-payload round-trip.
        return b""

    _validate_params(metadata.k, metadata.n)

    # Deduplicate by index; verify each shard's own SHA-256 before use.
    by_index: dict[int, ErasureShard] = {}
    for s in shards:
        if s.index < 0 or s.index >= metadata.n:
            raise ErasureError(
                f"shard index {s.index} out of range [0, {metadata.n})"
            )
        if s.index in by_index:
            raise DuplicateShardError(
                f"duplicate shard index {s.index}"
            )
        s.verify()  # raises CorruptShardError
        by_index[s.index] = s

    if len(by_index) < metadata.k:
        raise InsufficientShardsError(
            f"need {metadata.k} shards, got {len(by_index)}"
        )

    # Pick any k distinct shards. Prefer low-index primaries for
    # determinism — zfec returns the primary blocks untouched when the
    # inputs are the first k shares, which makes the test output
    # predictable.
    picked = sorted(by_index.values(), key=lambda s: s.index)[: metadata.k]

    decoder = zfec.Decoder(metadata.k, metadata.n)
    blocks = decoder.decode(
        [s.data for s in picked],
        [s.index for s in picked],
    )

    # Concatenate primary blocks in their original order (decoder returns
    # them in ascending original-index order) and trim padding.
    padded_payload = b"".join(bytes(b) for b in blocks)
    payload = padded_payload[: metadata.payload_bytes]

    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != metadata.payload_sha256:
        raise PayloadChecksumError(
            f"reconstructed sha256 {actual_sha} != metadata {metadata.payload_sha256}"
        )
    return payload


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _validate_params(k: int, n: int) -> None:
    if k <= 0:
        raise ErasureError(f"k must be > 0, got {k}")
    if n < k:
        raise ErasureError(f"n ({n}) must be >= k ({k})")
    if n > 256:
        # zfec uses GF(2^8); n ≤ 256 is a hard limit of the math.
        raise ErasureError(f"n must be <= 256, got {n}")
