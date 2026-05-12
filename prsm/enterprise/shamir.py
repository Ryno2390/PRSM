"""Sprint 307 — Shamir Secret Sharing over GF(256).

Foundation for threshold (t-of-n) encryption mode (Vision
§7 Enterprise Confidentiality Mode follow-on layer):
splits a symmetric key into n shares such that any t
shares reconstruct it, but any t-1 reveal NOTHING about
the secret (information-theoretic security under the
assumption that the polynomial coefficients are uniformly
random).

The math: polynomial arithmetic in GF(2^8) using the
standard AES irreducible polynomial 0x11b. Each byte of
the secret is independently split via a degree-(t-1)
polynomial. The polynomial coefficients are random; only
the constant term carries the secret byte. Each share is
the polynomial evaluated at a unique non-zero x-coordinate
in 1..255.

Pure Python, no new dependencies. The log/exp tables are
precomputed once at import for fast multiplication.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence


# ── GF(256) arithmetic ───────────────────────────────


_AES_IRREDUCIBLE = 0x11B  # x^8 + x^4 + x^3 + x + 1


def _build_gf256_tables() -> tuple[list[int], list[int]]:
    """Precompute log/exp tables for fast multiplication.
    The generator 0x03 produces all 255 non-zero elements
    when raised to powers 0..254 in GF(2^8)."""
    exp_table = [0] * 512
    log_table = [0] * 256
    x = 1
    for i in range(255):
        exp_table[i] = x
        log_table[x] = i
        # Multiply x by the generator (0x03) in GF(256):
        # x' = (x << 1) ^ (carry-conditional 0x11b)
        carry = x & 0x80
        x = ((x << 1) ^ (0x1B if carry else 0)) & 0xFF
        # 0x1B = AES irreducible minus the high bit
        # we strip in the same step
        x ^= exp_table[i]  # multiply-by-3 = (x*2) ^ x
    # Duplicate exp_table so we can index past 255 without
    # branching during multiplication
    for i in range(255, 512):
        exp_table[i] = exp_table[i - 255]
    return exp_table, log_table


_EXP, _LOG = _build_gf256_tables()


class GF256:
    """Static-method namespace for GF(2^8) operations."""

    @staticmethod
    def mul(a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return _EXP[_LOG[a] + _LOG[b]]

    @staticmethod
    def div(a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError(
                "GF256 division by zero"
            )
        if a == 0:
            return 0
        # log_a - log_b mod 255; the +255 keeps it positive
        return _EXP[_LOG[a] + 255 - _LOG[b]]


def _eval_poly(coeffs: Sequence[int], x: int) -> int:
    """Horner's method: evaluate polynomial whose
    coefficients are coeffs[0] + coeffs[1]*x + ... at the
    point x, all over GF(256)."""
    result = 0
    for c in reversed(coeffs):
        result = GF256.mul(result, x) ^ c
    return result


# ── Share dataclass ──────────────────────────────────


@dataclass
class Share:
    """A single Shamir share. `index` is the x-coordinate
    (1..n, non-zero); `y_values` holds one polynomial
    evaluation per byte of the original secret."""

    index: int
    y_values: bytes

    def to_dict(self) -> dict:
        import base64
        return {
            "index": int(self.index),
            "y_values_b64": base64.b64encode(
                self.y_values,
            ).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Share":
        import base64
        return cls(
            index=int(d["index"]),
            y_values=base64.b64decode(
                d["y_values_b64"],
            ),
        )


# ── Split ────────────────────────────────────────────


def split_secret(
    secret: bytes, *, t: int, n: int,
) -> List[Share]:
    """Split `secret` into n shares, any t of which
    reconstruct it.

    t in [1, n]; n in [1, 255]. The polynomial degree is
    t-1; share x-coordinates are 1..n."""
    if t < 1:
        raise ValueError("t >= 1 required")
    if n < t:
        raise ValueError("t <= n required")
    if n > 255:
        raise ValueError(
            "n <= 255 required (GF(256) index limit)"
        )
    if not secret:
        raise ValueError("secret must be non-empty")

    # For each byte of the secret, build a polynomial whose
    # constant term IS the secret byte and whose t-1 higher
    # coefficients are uniformly random.
    shares: List[bytearray] = [
        bytearray(len(secret)) for _ in range(n)
    ]
    random_bytes = (
        os.urandom(len(secret) * (t - 1)) if t > 1 else b""
    )
    for byte_idx, secret_byte in enumerate(secret):
        coeffs = [secret_byte]
        for j in range(1, t):
            coeffs.append(
                random_bytes[
                    byte_idx * (t - 1) + (j - 1)
                ],
            )
        for share_idx in range(n):
            x = share_idx + 1  # x in 1..n
            shares[share_idx][byte_idx] = _eval_poly(
                coeffs, x,
            )
    return [
        Share(index=i + 1, y_values=bytes(shares[i]))
        for i in range(n)
    ]


# ── Reconstruct ──────────────────────────────────────


def reconstruct_secret(
    shares: Sequence[Share], *, t: int,
) -> bytes:
    """Reconstruct the original secret from at least t
    shares using Lagrange interpolation at x=0."""
    if len(shares) < t:
        raise ValueError(
            f"reconstruct requires at least t={t} shares; "
            f"got {len(shares)}"
        )
    used = list(shares)[:t]
    indices = [s.index for s in used]
    if len(set(indices)) != len(indices):
        raise ValueError(
            "duplicate share indices not allowed"
        )
    secret_len = len(used[0].y_values)
    for s in used[1:]:
        if len(s.y_values) != secret_len:
            raise ValueError(
                f"share length mismatch: expected "
                f"{secret_len}, got {len(s.y_values)} "
                f"on share index={s.index}"
            )

    # Precompute Lagrange basis denominators at x=0:
    #   L_i(0) = prod_{j != i} (-x_j) / (x_i - x_j)
    # Since we're in GF(256), subtraction = XOR.
    basis = [0] * len(used)
    for i, s_i in enumerate(used):
        num = 1
        den = 1
        for j, s_j in enumerate(used):
            if i == j:
                continue
            num = GF256.mul(num, s_j.index)
            den = GF256.mul(
                den, s_i.index ^ s_j.index,
            )
        basis[i] = GF256.div(num, den)

    out = bytearray(secret_len)
    for byte_idx in range(secret_len):
        acc = 0
        for i, s in enumerate(used):
            acc ^= GF256.mul(
                basis[i], s.y_values[byte_idx],
            )
        out[byte_idx] = acc
    return bytes(out)
