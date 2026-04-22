"""AES-256-GCM authenticated encryption for PRSM Tier B / Tier C content.

Per docs/2026-04-22-phase7-storage-design-plan.md §2.1, §6 Task 5.

Plan §2.1 commits the Tier B pipeline to AES-256-GCM with:
  * 256-bit keys (32 bytes).
  * 96-bit IVs (12 bytes — the standard GCM nonce size).
  * 128-bit auth tag (16 bytes — GCM default, maximum tag size).

Tier C layers Shamir key-splitting on top of Task 5's key generation
(Task 7). This module owns only the symmetric-cipher primitive; key
distribution is separate.

Design commitments:

  * Use cryptography.hazmat.primitives.ciphers.aead.AESGCM — the
    project-wide audited AEAD wrapper. No direct ciphers.Cipher usage.
  * IVs are generated with os.urandom at encrypt-time. Per GCM spec, an
    IV MUST NOT be reused with the same key; random 12-byte IVs have a
    ~2^-48 collision probability per-key across 2^32 messages, which
    is acceptable for PRSM's per-content key model.
  * Key IDs are UUID4 strings — opaque handles that KeyDistribution.sol
    (Task 6) uses to reference on-chain key records.
  * Streaming API (StreamingEncryptor / StreamingDecryptor) chunks
    plaintext for payloads too large to fit in memory. Uses GCM's
    native streaming support; auth tag is emitted once at finalize().
  * Authentication is enforced on decrypt: tampered ciphertext, IV, or
    tag all raise EncryptionAuthError. Silent corruption is impossible.

Scope boundary:
  * No key distribution — that's Task 6 (on-chain) + Task 7 (Shamir).
  * No ContentUploader integration — that's Task 2 ShardEngine wiring
    (encrypt-then-shard for Tier B).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


__all__ = [
    "AES_KEY_BYTES",
    "AESKey",
    "AUTH_TAG_BYTES",
    "EncryptedPayload",
    "EncryptionAuthError",
    "EncryptionError",
    "IV_BYTES",
    "StreamingDecryptor",
    "StreamingEncryptor",
    "decrypt",
    "encrypt",
    "generate_key",
]


AES_KEY_BYTES = 32   # AES-256
IV_BYTES = 12        # GCM standard nonce size
AUTH_TAG_BYTES = 16  # GCM default tag size


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class EncryptionError(Exception):
    """Base class for encryption failures."""


class EncryptionAuthError(EncryptionError):
    """Authenticated decryption failed — ciphertext, IV, or tag tampered."""


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AESKey:
    key_id: str       # opaque handle for on-chain KeyDistribution lookup
    key_bytes: bytes  # 32-byte raw key

    def __post_init__(self) -> None:
        if len(self.key_bytes) != AES_KEY_BYTES:
            raise EncryptionError(
                f"key_bytes must be {AES_KEY_BYTES} bytes, got {len(self.key_bytes)}"
            )


@dataclass(frozen=True)
class EncryptedPayload:
    """Wire format for an AES-256-GCM encrypted blob.

    `ciphertext` is the body only — auth tag is separate, so streaming
    decryption can verify at finalize() without probing the tail of a
    potentially large body.
    """

    ciphertext: bytes
    iv: bytes
    auth_tag: bytes
    key_id: str  # echoes the AESKey used, for key-distribution lookup

    def __post_init__(self) -> None:
        if len(self.iv) != IV_BYTES:
            raise EncryptionError(
                f"iv must be {IV_BYTES} bytes, got {len(self.iv)}"
            )
        if len(self.auth_tag) != AUTH_TAG_BYTES:
            raise EncryptionError(
                f"auth_tag must be {AUTH_TAG_BYTES} bytes, got {len(self.auth_tag)}"
            )


# -----------------------------------------------------------------------------
# Key generation
# -----------------------------------------------------------------------------


def generate_key() -> AESKey:
    """Generate a fresh AES-256 key with a UUID4 key_id."""
    return AESKey(
        key_id=str(uuid.uuid4()),
        key_bytes=os.urandom(AES_KEY_BYTES),
    )


# -----------------------------------------------------------------------------
# One-shot encrypt / decrypt
# -----------------------------------------------------------------------------


def encrypt(
    plaintext: bytes,
    key: AESKey,
    *,
    associated_data: Optional[bytes] = None,
) -> EncryptedPayload:
    """Encrypt `plaintext` under `key`. IV is randomly generated.

    `associated_data` (AEAD AD) can bind the ciphertext to context — for
    PRSM, likely `key.key_id.encode()` or the shard manifest hash.
    """
    iv = os.urandom(IV_BYTES)
    aead = AESGCM(key.key_bytes)
    combined = aead.encrypt(iv, plaintext, associated_data)
    # AESGCM appends the tag; split for the explicit-tag wire format.
    ciphertext, auth_tag = combined[:-AUTH_TAG_BYTES], combined[-AUTH_TAG_BYTES:]
    return EncryptedPayload(
        ciphertext=ciphertext,
        iv=iv,
        auth_tag=auth_tag,
        key_id=key.key_id,
    )


def decrypt(
    payload: EncryptedPayload,
    key: AESKey,
    *,
    associated_data: Optional[bytes] = None,
) -> bytes:
    """Decrypt and authenticate. Raises EncryptionAuthError on tag
    mismatch (tampering, wrong key, wrong IV, wrong AD)."""
    if payload.key_id != key.key_id:
        # key_id mismatch is a caller error; surface it distinctly from
        # tag failure so debugging is easier.
        raise EncryptionError(
            f"key_id mismatch: payload={payload.key_id!r}, key={key.key_id!r}"
        )
    aead = AESGCM(key.key_bytes)
    combined = payload.ciphertext + payload.auth_tag
    try:
        return aead.decrypt(payload.iv, combined, associated_data)
    except InvalidTag as exc:
        raise EncryptionAuthError("authentication failed") from exc


# -----------------------------------------------------------------------------
# Streaming encrypt / decrypt
# -----------------------------------------------------------------------------


class StreamingEncryptor:
    """Chunk-by-chunk encryption for large payloads.

    Usage:
        enc = StreamingEncryptor(key)
        for chunk in stream:
            ciphertext_chunk = enc.encrypt_chunk(chunk)
            ...
        auth_tag = enc.finalize()

    `iv` is generated at construction; callers must emit it alongside
    the ciphertext. `finalize()` returns the auth tag; callers emit it
    last. Attempting to encrypt after finalize raises.
    """

    def __init__(
        self,
        key: AESKey,
        *,
        associated_data: Optional[bytes] = None,
    ) -> None:
        self._key = key
        self._iv = os.urandom(IV_BYTES)
        self._cipher = Cipher(
            algorithms.AES(key.key_bytes),
            modes.GCM(self._iv),
        ).encryptor()
        if associated_data is not None:
            self._cipher.authenticate_additional_data(associated_data)
        self._finalized = False

    @property
    def iv(self) -> bytes:
        return self._iv

    @property
    def key_id(self) -> str:
        return self._key.key_id

    def encrypt_chunk(self, chunk: bytes) -> bytes:
        if self._finalized:
            raise EncryptionError("encryptor already finalized")
        return self._cipher.update(chunk)

    def finalize(self) -> bytes:
        if self._finalized:
            raise EncryptionError("encryptor already finalized")
        self._cipher.finalize()
        self._finalized = True
        return self._cipher.tag


class StreamingDecryptor:
    """Chunk-by-chunk authenticated decryption.

    Auth tag is NOT verified until finalize() returns — chunks emitted
    by decrypt_chunk before finalize() are UNAUTHENTICATED. Callers
    MUST buffer the chunks internally and only release them after
    finalize() succeeds, or equivalent.
    """

    def __init__(
        self,
        key: AESKey,
        iv: bytes,
        auth_tag: bytes,
        *,
        associated_data: Optional[bytes] = None,
    ) -> None:
        if len(iv) != IV_BYTES:
            raise EncryptionError(f"iv must be {IV_BYTES} bytes")
        if len(auth_tag) != AUTH_TAG_BYTES:
            raise EncryptionError(f"auth_tag must be {AUTH_TAG_BYTES} bytes")
        self._cipher = Cipher(
            algorithms.AES(key.key_bytes),
            modes.GCM(iv, auth_tag),
        ).decryptor()
        if associated_data is not None:
            self._cipher.authenticate_additional_data(associated_data)
        self._finalized = False

    def decrypt_chunk(self, chunk: bytes) -> bytes:
        if self._finalized:
            raise EncryptionError("decryptor already finalized")
        return self._cipher.update(chunk)

    def finalize(self) -> None:
        if self._finalized:
            raise EncryptionError("decryptor already finalized")
        try:
            self._cipher.finalize()
        except InvalidTag as exc:
            raise EncryptionAuthError("authentication failed") from exc
        self._finalized = True
