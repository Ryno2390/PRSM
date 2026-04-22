"""Unit tests for prsm.storage.key_sharing.

Per docs/2026-04-22-phase7-storage-design-plan.md §6 Task 7.
"""

from __future__ import annotations

import pytest

from prsm.storage.encryption import AESKey, generate_key
from prsm.storage.key_sharing import (
    DEFAULT_M,
    DEFAULT_N,
    InsufficientSharesError,
    KeyShare,
    ShamirError,
    combine_shares,
    split_key,
)


# -----------------------------------------------------------------------------
# split_key
# -----------------------------------------------------------------------------


def test_split_produces_n_shares_at_default_params():
    k = generate_key()
    shares = split_key(k)
    assert len(shares) == DEFAULT_N
    assert all(isinstance(s, KeyShare) for s in shares)


def test_split_share_indices_are_0_to_n_minus_1():
    k = generate_key()
    shares = split_key(k)
    assert [s.index for s in shares] == list(range(DEFAULT_N))


def test_split_each_share_has_key_id_mn_metadata():
    k = generate_key()
    shares = split_key(k)
    for s in shares:
        assert s.key_id == k.key_id
        assert s.m == DEFAULT_M
        assert s.n == DEFAULT_N


def test_split_share_halves_are_16_bytes_each():
    k = generate_key()
    shares = split_key(k)
    for s in shares:
        assert len(s.first_half) == 16
        assert len(s.second_half) == 16


def test_split_rejects_invalid_m():
    k = generate_key()
    with pytest.raises(ShamirError):
        split_key(k, m=0, n=5)


def test_split_rejects_n_less_than_m():
    k = generate_key()
    with pytest.raises(ShamirError):
        split_key(k, m=3, n=2)


def test_split_custom_params():
    k = generate_key()
    shares = split_key(k, m=2, n=3)
    assert len(shares) == 3
    assert shares[0].m == 2


# -----------------------------------------------------------------------------
# combine_shares — happy path
# -----------------------------------------------------------------------------


def test_combine_from_all_shares_recovers_key():
    k = generate_key()
    shares = split_key(k)
    recovered = combine_shares(shares)
    assert recovered.key_bytes == k.key_bytes
    assert recovered.key_id == k.key_id


def test_combine_from_first_m_shares_recovers_key():
    k = generate_key()
    shares = split_key(k)
    recovered = combine_shares(shares[:DEFAULT_M])
    assert recovered.key_bytes == k.key_bytes


def test_combine_from_last_m_shares_recovers_key():
    """Trailing m shares (indices n-m..n-1) must also reconstruct."""
    k = generate_key()
    shares = split_key(k)
    recovered = combine_shares(shares[-DEFAULT_M:])
    assert recovered.key_bytes == k.key_bytes


def test_combine_from_mixed_indices_recovers_key():
    k = generate_key()
    shares = split_key(k)
    # Pick shares 0, 2, 4 (non-contiguous).
    recovered = combine_shares([shares[0], shares[2], shares[4]])
    assert recovered.key_bytes == k.key_bytes


def test_combine_with_more_than_m_shares_works():
    k = generate_key()
    shares = split_key(k)
    recovered = combine_shares(shares[: DEFAULT_M + 1])
    assert recovered.key_bytes == k.key_bytes


# -----------------------------------------------------------------------------
# Collusion resistance + insufficient-share handling
# -----------------------------------------------------------------------------


def test_combine_with_m_minus_one_raises_insufficient():
    """Plan §7 acceptance: M-of-N threshold enforced. m-1 shares must fail."""
    k = generate_key()
    shares = split_key(k)
    with pytest.raises(InsufficientSharesError):
        combine_shares(shares[: DEFAULT_M - 1])


def test_combine_with_zero_shares_raises():
    with pytest.raises(InsufficientSharesError):
        combine_shares([])


def test_collusion_at_m_minus_one_gives_no_key():
    """Even with 2 shares (m-1 for default m=3), the combined result
    should NOT match the real key. We verify the negative: the function
    either raises OR (if it did return, which it doesn't) the returned
    key would not match.
    """
    k = generate_key()
    shares = split_key(k)
    with pytest.raises(InsufficientSharesError):
        combine_shares(shares[:2])


# -----------------------------------------------------------------------------
# Error paths
# -----------------------------------------------------------------------------


def test_combine_rejects_duplicate_share_indices():
    k = generate_key()
    shares = split_key(k)
    dup = [shares[0], shares[0], shares[1]]
    with pytest.raises(ShamirError):
        combine_shares(dup)


def test_combine_rejects_mismatched_key_ids():
    k1 = generate_key()
    k2 = generate_key()
    shares1 = split_key(k1)
    shares2 = split_key(k2)
    # Mix one from each.
    mixed = [shares1[0], shares1[1], shares2[2]]
    with pytest.raises(ShamirError):
        combine_shares(mixed)


def test_combine_rejects_out_of_range_index():
    k = generate_key()
    shares = split_key(k)
    bogus = KeyShare(
        index=99,
        first_half=shares[0].first_half,
        second_half=shares[0].second_half,
        key_id=shares[0].key_id,
        m=shares[0].m,
        n=shares[0].n,
    )
    with pytest.raises(ShamirError):
        combine_shares([shares[0], shares[1], bogus])


def test_keyshare_rejects_wrong_half_size():
    with pytest.raises(ShamirError):
        KeyShare(
            index=0,
            first_half=b"too-short",
            second_half=b"0" * 16,
            key_id="k",
            m=3,
            n=5,
        )


# -----------------------------------------------------------------------------
# Plan acceptance criterion
# -----------------------------------------------------------------------------


def test_plan_acceptance_m3_n5_threshold_enforced():
    """Plan §2.1 Tier C: m=3, n=5.

    * All 5 → recovers.
    * Any 3 → recovers.
    * Exactly 2 → raises (collusion resistance at m-1).
    """
    k = generate_key()
    shares = split_key(k, m=3, n=5)

    assert combine_shares(shares).key_bytes == k.key_bytes
    assert combine_shares(shares[:3]).key_bytes == k.key_bytes
    # Every combination of 3 indices works (sanity on a few).
    assert combine_shares([shares[0], shares[2], shares[4]]).key_bytes == k.key_bytes
    assert combine_shares([shares[1], shares[3], shares[4]]).key_bytes == k.key_bytes

    with pytest.raises(InsufficientSharesError):
        combine_shares(shares[:2])
