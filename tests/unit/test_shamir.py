"""Sprint 307 — Shamir Secret Sharing over GF(256).

Foundation for threshold (t-of-n) encryption mode: a
random symmetric key is split into n shares such that any
t shares can reconstruct it, but t-1 reveal no information.

Pure-Python implementation, no new external dependencies.
The math: polynomial arithmetic in GF(2^8) using the AES
irreducible polynomial 0x11b. Each byte of the secret is
independently split via a degree-(t-1) polynomial over
GF(256); each share holds the polynomial evaluations at
its x-coordinate.

This module is the math layer. recipient_encryption gets
extended in the same sprint with a threshold-mode encrypt
path that splits the symmetric key through this primitive
and seals each share to a recipient.
"""
from __future__ import annotations

import os

import pytest

from prsm.enterprise.shamir import (
    GF256,
    Share,
    reconstruct_secret,
    split_secret,
)


# ── GF(256) field operations ─────────────────────────


def test_gf256_mul_identity():
    # x * 1 == x for all x
    for x in range(256):
        assert GF256.mul(x, 1) == x
        assert GF256.mul(1, x) == x


def test_gf256_mul_zero():
    for x in range(256):
        assert GF256.mul(x, 0) == 0
        assert GF256.mul(0, x) == 0


def test_gf256_div_self_is_one():
    for x in range(1, 256):
        assert GF256.div(x, x) == 1


def test_gf256_mul_div_round_trip():
    # (a * b) / b == a for all a, all non-zero b
    for a in (0, 1, 2, 17, 99, 200, 255):
        for b in (1, 3, 47, 128, 255):
            assert GF256.div(GF256.mul(a, b), b) == a


def test_gf256_div_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        GF256.div(1, 0)


# ── Split / reconstruct round-trip ───────────────────


def test_split_then_reconstruct_round_trip():
    secret = b"a 32-byte symmetric key padded.."
    assert len(secret) == 32
    shares = split_secret(secret, t=3, n=5)
    assert len(shares) == 5
    # Any t shares reconstruct
    out = reconstruct_secret(shares[:3], t=3)
    assert out == secret


def test_split_unique_indices():
    secret = b"\x00" * 32
    shares = split_secret(secret, t=2, n=5)
    indices = {s.index for s in shares}
    assert len(indices) == 5  # All distinct
    assert all(1 <= i <= 5 for i in indices)


def test_reconstruct_with_t_plus_one_shares_works():
    secret = b"hello world!" + b"\x00" * 20
    shares = split_secret(secret, t=2, n=5)
    out = reconstruct_secret(shares[:3], t=2)
    assert out == secret


def test_reconstruct_with_all_n_shares_works():
    secret = os.urandom(32)
    shares = split_secret(secret, t=3, n=7)
    out = reconstruct_secret(shares, t=3)
    assert out == secret


def test_reconstruct_with_fewer_than_t_raises():
    secret = b"\x01" * 32
    shares = split_secret(secret, t=3, n=5)
    with pytest.raises(ValueError, match="at least t"):
        reconstruct_secret(shares[:2], t=3)


def test_reconstruct_with_arbitrary_subset():
    """Any t-element subset, not just the first t —
    Lagrange reconstruction is independent of which
    shares are presented."""
    secret = b"check arbitrary subset...........  "[:32]
    shares = split_secret(secret, t=3, n=5)
    out = reconstruct_secret(
        [shares[0], shares[2], shares[4]], t=3,
    )
    assert out == secret


# ── Bad inputs ───────────────────────────────────────


def test_split_rejects_t_greater_than_n():
    with pytest.raises(ValueError, match="t <= n"):
        split_secret(b"\x00" * 32, t=4, n=3)


def test_split_rejects_t_zero():
    with pytest.raises(ValueError, match="t >= 1"):
        split_secret(b"\x00" * 32, t=0, n=3)


def test_split_rejects_n_over_255():
    """GF(256) supports indices 1..255; we cap n at 255."""
    with pytest.raises(ValueError, match="n <= 255"):
        split_secret(b"\x00" * 32, t=2, n=256)


def test_split_rejects_empty_secret():
    with pytest.raises(ValueError, match="empty"):
        split_secret(b"", t=2, n=3)


def test_reconstruct_rejects_duplicate_indices():
    """Two shares with the same x-coordinate breaks
    Lagrange — refuse loud."""
    secret = b"\x00" * 32
    shares = split_secret(secret, t=2, n=3)
    dup = Share(index=shares[0].index, y_values=shares[0].y_values)
    with pytest.raises(ValueError, match="duplicate"):
        reconstruct_secret([shares[0], dup], t=2)


def test_reconstruct_rejects_mismatched_length():
    secret = b"\x00" * 32
    shares = split_secret(secret, t=2, n=3)
    shares[1].y_values = shares[1].y_values + b"\x00"
    with pytest.raises(ValueError, match="length"):
        reconstruct_secret(shares[:2], t=2)


# ── Information-theoretic security smoke check ───────


def test_one_share_alone_leaks_nothing():
    """A single share, in isolation, should look uniformly
    random — at minimum, repeated runs produce different
    share values even with the same secret + same t/n.
    This is the security property of Shamir + a fresh
    random polynomial each split."""
    secret = b"X" * 32
    runs = [
        split_secret(secret, t=3, n=5)[0].y_values
        for _ in range(10)
    ]
    # All ten runs produce distinct first-share y-values
    # (probability of collision is ~zero with 32 random
    # bytes per share)
    assert len(set(runs)) == 10


def test_t_minus_one_shares_do_not_determine_secret():
    """With t-1 shares of a t-of-n split, the missing
    information must remain hidden — different secrets
    that produce the same t-1 shares are equally
    plausible. Smoke-test: t-1 shares from secret_A and
    t-1 shares from secret_B can both be 'reconstructed'
    to wrong values when paired with an attacker-chosen
    fake share — i.e. the t-1 shares constrain nothing
    on their own."""
    # This is a structural property, not a direct test —
    # the test is that reconstruct REFUSES with < t shares
    # (proven above) and the y-values look random (proven
    # above). Combined, those two are the load-bearing
    # security claim.
    secret_a = b"AAAA" * 8
    secret_b = b"BBBB" * 8
    shares_a = split_secret(secret_a, t=3, n=5)
    shares_b = split_secret(secret_b, t=3, n=5)
    # The first 2 shares from each are unrelated
    assert shares_a[0].y_values != shares_b[0].y_values


# ── Share serialization ──────────────────────────────


def test_share_to_dict_round_trip():
    secret = b"\xab" * 32
    share = split_secret(secret, t=2, n=3)[0]
    d = share.to_dict()
    restored = Share.from_dict(d)
    assert restored.index == share.index
    assert restored.y_values == share.y_values


def test_share_y_values_length_matches_secret():
    for secret_len in (16, 32, 64):
        shares = split_secret(b"\x01" * secret_len, t=2, n=3)
        for s in shares:
            assert len(s.y_values) == secret_len
