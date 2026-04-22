"""Shamir Secret Sharing for Tier C AES-256 key distribution.

Per docs/2026-04-22-phase7-storage-design-plan.md §2.1, §6 Task 7.

Tier C splits the content-encryption key across `n` holders such that
any `m` of them can reconstruct it, but fewer than `m` learn nothing
information-theoretically. Combined with Reed-Solomon erasure over the
ciphertext (Task 1) and AES-256-GCM encryption (Task 5), Tier C
reconstruction requires crossing BOTH the K-of-N shard-reconstruction
threshold AND the M-of-N key-share threshold.

Default parameters per plan §2.1: `m=3, n=5`. Share holders should be
disjoint from storage providers (Task 7 §2 requirement) so a single
colluder cannot obtain both.

Implementation:

  * PyCryptodome's Shamir operates over 16-byte secrets in GF(2^128).
  * AES-256 keys are 32 bytes, so we split into two halves and share
    each half independently. A KeyShare is a triple
    (index, first_half_share, second_half_share). Reconstruction
    combines both halves for each of `m` distinct indices.
  * Share indices are 1-indexed (PyCryptodome convention) but exposed
    as 0-indexed externally for consistency with the erasure shard
    numbering in Task 1. Translation is internal.

Scope boundary — what this module does NOT do:
  * Key-holder selection / geography (Task 7 §2 governance; out of
    scope for the primitive).
  * On-chain coordination of share release (Task 6 KeyDistribution.sol).
  * AES encryption itself (Task 5 prsm.storage.encryption).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from Crypto.Protocol.SecretSharing import Shamir

from prsm.storage.encryption import AES_KEY_BYTES, AESKey


__all__ = [
    "DEFAULT_M",
    "DEFAULT_N",
    "InsufficientSharesError",
    "KeyShare",
    "ShamirError",
    "combine_shares",
    "split_key",
]


DEFAULT_M = 3
DEFAULT_N = 5

_HALF_BYTES = AES_KEY_BYTES // 2  # 16 bytes per Shamir block


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class ShamirError(Exception):
    """Base class for Shamir key-share failures."""


class InsufficientSharesError(ShamirError):
    """Fewer than m distinct shares were supplied for reconstruction."""


# -----------------------------------------------------------------------------
# Data class
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyShare:
    """One share of a split AES-256 key.

    `index` is 0-indexed externally (matching erasure-shard numbering).
    `first_half` / `second_half` each hold a 16-byte Shamir share of
    the corresponding key half.
    """

    index: int
    first_half: bytes
    second_half: bytes
    key_id: str
    m: int
    n: int

    def __post_init__(self) -> None:
        if len(self.first_half) != _HALF_BYTES or len(self.second_half) != _HALF_BYTES:
            raise ShamirError(
                f"each half-share must be {_HALF_BYTES} bytes"
            )


# -----------------------------------------------------------------------------
# Split
# -----------------------------------------------------------------------------


def split_key(
    key: AESKey,
    *,
    m: int = DEFAULT_M,
    n: int = DEFAULT_N,
) -> List[KeyShare]:
    """Split `key` into `n` KeyShares such that any `m` of them suffice
    to reconstruct the key.

    Information-theoretic: fewer than `m` shares yield zero information
    about the key. The split's randomness comes from PyCryptodome's
    internal coefficient generation.
    """
    _validate_params(m, n)

    first_half = key.key_bytes[:_HALF_BYTES]
    second_half = key.key_bytes[_HALF_BYTES:]

    first_shares = Shamir.split(m, n, first_half)
    second_shares = Shamir.split(m, n, second_half)

    shares: List[KeyShare] = []
    for i in range(n):
        # PyCryptodome indexes 1..n; we present 0..n-1 externally.
        first_idx, first_blob = first_shares[i]
        second_idx, second_blob = second_shares[i]
        assert first_idx == second_idx == i + 1, (
            f"Shamir index drift: {first_idx} vs {second_idx} at slot {i}"
        )
        shares.append(
            KeyShare(
                index=i,
                first_half=bytes(first_blob),
                second_half=bytes(second_blob),
                key_id=key.key_id,
                m=m,
                n=n,
            )
        )
    return shares


# -----------------------------------------------------------------------------
# Combine
# -----------------------------------------------------------------------------


def combine_shares(shares: Sequence[KeyShare]) -> AESKey:
    """Reconstruct the AESKey from `m`-of-`n` shares.

    Requires at least `shares[0].m` distinct shares. Duplicate indices
    raise; mismatched key_id / m / n across shares raises (all shares
    must come from the same split operation).
    """
    if not shares:
        raise InsufficientSharesError("no shares supplied")

    first = shares[0]
    m, n, key_id = first.m, first.n, first.key_id

    seen_indices: set[int] = set()
    for s in shares:
        if s.m != m or s.n != n or s.key_id != key_id:
            raise ShamirError(
                "shares come from different splits "
                f"(key_id/m/n mismatch at index {s.index})"
            )
        if s.index < 0 or s.index >= n:
            raise ShamirError(
                f"share index {s.index} out of range [0, {n})"
            )
        if s.index in seen_indices:
            raise ShamirError(f"duplicate share index {s.index}")
        seen_indices.add(s.index)

    if len(shares) < m:
        raise InsufficientSharesError(
            f"need {m} shares, got {len(shares)}"
        )

    # Take any m. Prefer low indices for determinism.
    picked = sorted(shares, key=lambda s: s.index)[:m]

    # PyCryptodome combine expects (1-indexed_idx, 16-byte-share) tuples.
    first_tuples = [(s.index + 1, s.first_half) for s in picked]
    second_tuples = [(s.index + 1, s.second_half) for s in picked]

    first_half = Shamir.combine(first_tuples)
    second_half = Shamir.combine(second_tuples)

    return AESKey(
        key_id=key_id,
        key_bytes=first_half + second_half,
    )


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _validate_params(m: int, n: int) -> None:
    if m <= 0:
        raise ShamirError(f"m must be > 0, got {m}")
    if n < m:
        raise ShamirError(f"n ({n}) must be >= m ({m})")
    if n > 255:
        raise ShamirError(f"n must be <= 255, got {n}")
