"""
Key Manager — AES-256-GCM encryption and Shamir's Secret Sharing over GF(2^8).

Provides content-encryption key generation, authenticated encryption, and
threshold key splitting via Shamir's scheme using Galois Field (256) arithmetic
with the AES irreducible polynomial.
"""

from __future__ import annotations

import os
import secrets
from typing import List, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from prsm.storage.exceptions import KeyReconstructionError
from prsm.storage.models import ContentHash, KeyShare


# ---------------------------------------------------------------------------
# GF(256) arithmetic with AES irreducible polynomial x^8+x^4+x^3+x+1
# ---------------------------------------------------------------------------

_EXP: list[int] = [0] * 256
_LOG: list[int] = [0] * 256

# Precompute log/exp tables with generator 3
x = 1
for i in range(255):
    _EXP[i] = x
    _LOG[x] = i
    x = x ^ (x << 1)
    if x & 0x100:
        x ^= 0x11B
_EXP[255] = _EXP[0]  # wrap-around for convenience


def _gf256_mul(a: int, b: int) -> int:
    """Multiply two elements in GF(256) using log/exp tables."""
    if a == 0 or b == 0:
        return 0
    return _EXP[(_LOG[a] + _LOG[b]) % 255]


def _gf256_inv(a: int) -> int:
    """Multiplicative inverse in GF(256)."""
    if a == 0:
        raise ZeroDivisionError("No inverse for 0 in GF(256)")
    return _EXP[255 - _LOG[a]]


def _evaluate_polynomial(coeffs: list[int], x: int) -> int:
    """Evaluate polynomial at *x* in GF(256) using Horner's method.

    coeffs[0] is the constant term (the secret byte).
    """
    # Horner: start from highest degree coefficient
    result = 0
    for coeff in reversed(coeffs):
        result = _gf256_mul(result, x) ^ coeff
    return result


def _lagrange_interpolate(shares: list[tuple[int, int]], x: int = 0) -> int:
    """Lagrange interpolation at *x* in GF(256).

    Each share is (x_i, y_i) where x_i != 0.
    """
    result = 0
    for i, (xi, yi) in enumerate(shares):
        basis = yi
        for j, (xj, _) in enumerate(shares):
            if i == j:
                continue
            num = x ^ xj
            den = xi ^ xj
            basis = _gf256_mul(basis, _gf256_mul(num, _gf256_inv(den)))
        result ^= basis
    return result


# ---------------------------------------------------------------------------
# KeyManager
# ---------------------------------------------------------------------------

class KeyManager:
    """AES-256-GCM encryption and Shamir's Secret Sharing key management."""

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def generate_key(self) -> bytes:
        """Generate a random 256-bit (32-byte) AES key."""
        return os.urandom(32)

    # ------------------------------------------------------------------
    # AES-256-GCM
    # ------------------------------------------------------------------

    def encrypt(self, key: bytes, plaintext: bytes) -> bytes:
        """AES-256-GCM encrypt.

        Returns ``nonce (12 bytes) || ciphertext || tag (16 bytes)``.
        """
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ct_with_tag = aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ct_with_tag

    def decrypt(self, key: bytes, data: bytes) -> bytes:
        """AES-256-GCM decrypt.

        *data* is ``nonce (12) || ciphertext || tag (16)``.
        """
        if len(data) < 12:
            raise ValueError("Ciphertext too short — missing nonce")
        nonce = data[:12]
        ct_with_tag = data[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct_with_tag, None)

    # ------------------------------------------------------------------
    # Shamir's Secret Sharing
    # ------------------------------------------------------------------

    def split_secret(
        self,
        secret: bytes,
        threshold: int,
        num_shares: int,
    ) -> List[Tuple[int, bytes]]:
        """Split *secret* into *num_shares* shares with the given *threshold*.

        Each byte of the secret is shared independently using a random
        polynomial of degree ``threshold - 1`` over GF(256).

        Returns a list of ``(share_index, share_data)`` tuples.
        ``share_index`` is 1-based.
        """
        if threshold < 2:
            raise ValueError("threshold must be >= 2")
        if threshold > num_shares:
            raise ValueError("threshold must be <= num_shares")

        # For each byte position, create a random polynomial and evaluate
        shares_data: list[bytearray] = [bytearray() for _ in range(num_shares)]

        for byte_val in secret:
            # coeffs[0] = secret byte, coeffs[1..threshold-1] = random
            coeffs = [byte_val] + [secrets.randbelow(256) for _ in range(threshold - 1)]
            for share_idx in range(num_shares):
                x = share_idx + 1  # 1-based evaluation point
                y = _evaluate_polynomial(coeffs, x)
                shares_data[share_idx].append(y)

        return [(i + 1, bytes(shares_data[i])) for i in range(num_shares)]

    def reconstruct_secret(
        self,
        shares: List[Tuple[int, bytes]],
        threshold: int,
    ) -> bytes:
        """Reconstruct the secret from *threshold*-or-more shares.

        Uses Lagrange interpolation at x=0 in GF(256).
        If fewer than *threshold* shares are given the result will be
        incorrect (by design — no error is raised).
        """
        if not shares:
            raise KeyReconstructionError("No shares provided")

        secret_len = len(shares[0][1])
        result = bytearray()

        for byte_pos in range(secret_len):
            points = [(idx, data[byte_pos]) for idx, data in shares]
            result.append(_lagrange_interpolate(points, x=0))

        return bytes(result)

    # ------------------------------------------------------------------
    # Manifest-level helpers
    # ------------------------------------------------------------------

    def encrypt_manifest(
        self,
        manifest_data: bytes,
        content_hash: ContentHash,
        threshold: int,
        num_shares: int,
    ) -> Tuple[bytes, List[KeyShare]]:
        """Generate key, encrypt *manifest_data*, split key into KeyShares.

        Returns ``(ciphertext, key_shares)``.
        """
        key = self.generate_key()
        ciphertext = self.encrypt(key, manifest_data)
        raw_shares = self.split_secret(key, threshold, num_shares)

        key_shares = [
            KeyShare(
                content_hash=content_hash,
                share_index=idx,
                share_data=data,
                threshold=threshold,
                total_shares=num_shares,
                algorithm_id=0x01,  # AES-256-GCM
            )
            for idx, data in raw_shares
        ]
        return ciphertext, key_shares

    def decrypt_manifest(
        self,
        ciphertext: bytes,
        key_shares: List[KeyShare],
    ) -> bytes:
        """Reconstruct the encryption key from *key_shares* and decrypt."""
        if not key_shares:
            raise KeyReconstructionError("No key shares provided")

        threshold = key_shares[0].threshold
        raw_shares = [(ks.share_index, ks.share_data) for ks in key_shares]
        key = self.reconstruct_secret(raw_shares, threshold)
        return self.decrypt(key, ciphertext)
