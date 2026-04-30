"""Phase 3.x.11.q.y — AES-GCM cipher for `proposed_token_probs`.

The chain-level constant-time decorators in Phase 3.x.11.q mask
the executor → caller wire only. The per-stage wire (executor →
each chain stage) continues to carry per-token timing AND
content-correlated payload (notably `proposed_token_probs` which
ships K floats per VERIFY round under v2 stochastic speculation).

Phase 3.x.11.q.y composes the chain-level decorators with three
new per-stage-wire mitigations to lift the "speculation + Tier C
denied" gate from threat-model addendum §3.7. This module
implements the FIRST mitigation: AES-GCM encryption of the
plaintext `proposed_token_probs` bytes.

**Scope.** Pure crypto helper — a `ProbsCipher` class with
`encrypt` / `decrypt` methods. The cipher takes a 32-byte
pre-shared key at construction; key distribution is the
operator's responsibility (deployment-wide PSK per
executor/tail pair, distributed out-of-band).

**Out of scope** for this slice:
- ECDH on-wire key negotiation (would extend HandoffToken; defer
  to Phase 3.x.11.q.y' if PSK distribution becomes operationally
  burdensome).
- Per-request key rotation (operator can rotate PSKs at deployment
  cadence; the AAD `(request_id || stage_index)` already binds each
  ciphertext to its dispatch).
- Hardware-backed key storage (TEE-resident PSK material is the
  R5 Tier C hardening direction; orthogonal).

**AAD binding.** Each ciphertext is bound to the dispatch via
`AAD = request_id_utf8 || b"|" || stage_index_byte`. A relay
adversary that swaps a ciphertext from one (request, stage)
to another causes decryption to fail at the tail (AES-GCM tag
mismatch). This is the in-band integrity check that compensates
for the request-body-not-being-end-to-end-signed honest scope
carry-forward from Phase 3.x.11.y.x M2.

**Encoding.** Plaintext probs are encoded as a tightly-packed
sequence of `K * 8` little-endian float64 bytes. The K count
isn't carried in the ciphertext — the recipient knows K from
the co-set `proposed_token_ids` field (length must match).
"""

from __future__ import annotations

import struct
from typing import List, Sequence

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


__all__ = [
    "ProbsCipher",
    "ProbsCipherError",
    "ProbsDecryptionError",
]


_AES_KEY_LEN = 32      # AES-256
_GCM_NONCE_LEN = 12    # 96-bit nonce, AES-GCM standard
_FLOAT_FORMAT = "<d"   # little-endian double-precision float
_FLOAT_SIZE = 8


class ProbsCipherError(RuntimeError):
    """Base exception for ProbsCipher errors."""


class ProbsDecryptionError(ProbsCipherError):
    """Raised when AES-GCM decryption fails — wrong key, AAD mismatch
    (relay adversary swapped ciphertext across (request, stage)),
    or tampered ciphertext. Tail surfaces this as MALFORMED_REQUEST
    so the executor can surface a clear error to the caller."""


def _aad(request_id: str, stage_index: int) -> bytes:
    """Canonical AAD encoding. Bind ciphertext to its (request,
    stage) dispatch — prevents replay across stages and across
    requests."""
    if not isinstance(request_id, str) or not request_id:
        raise ProbsCipherError(
            f"request_id must be non-empty str, got {request_id!r}"
        )
    if (
        not isinstance(stage_index, int)
        or isinstance(stage_index, bool)
        or stage_index < 0
        or stage_index > 255
    ):
        raise ProbsCipherError(
            f"stage_index must be int in [0, 255], got {stage_index!r}"
        )
    return request_id.encode("utf-8") + b"|" + bytes([stage_index])


def _encode_probs(probs: Sequence[float]) -> bytes:
    """Pack K probs as `K * 8` little-endian float64 bytes."""
    if not isinstance(probs, (list, tuple)):
        raise ProbsCipherError(
            f"probs must be list/tuple, got {type(probs).__name__}"
        )
    out = bytearray()
    for i, p in enumerate(probs):
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ProbsCipherError(
                f"probs[{i}] must be number, got {type(p).__name__}"
            )
        if not (0.0 <= float(p) <= 1.0):
            raise ProbsCipherError(
                f"probs[{i}] must be in [0, 1], got {p}"
            )
        out += struct.pack(_FLOAT_FORMAT, float(p))
    return bytes(out)


def _decode_probs(plaintext: bytes, expected_k: int) -> List[float]:
    """Unpack `expected_k * 8` bytes → list of K floats. Validates
    length matches the caller-supplied K (from co-set
    proposed_token_ids)."""
    expected_bytes = expected_k * _FLOAT_SIZE
    if len(plaintext) != expected_bytes:
        raise ProbsCipherError(
            f"decrypted plaintext length {len(plaintext)} does not "
            f"match expected K * 8 = {expected_bytes} (K={expected_k})"
        )
    out: List[float] = []
    for i in range(expected_k):
        offset = i * _FLOAT_SIZE
        chunk = plaintext[offset:offset + _FLOAT_SIZE]
        (p,) = struct.unpack(_FLOAT_FORMAT, chunk)
        if not (0.0 <= p <= 1.0):
            raise ProbsCipherError(
                f"decrypted probs[{i}] = {p} out of [0, 1] range "
                f"(corrupted plaintext or wrong key)"
            )
        out.append(float(p))
    return out


class ProbsCipher:
    """AES-256-GCM cipher for `proposed_token_probs` round-trip.

    Both executor and tail construct a ProbsCipher with the same
    32-byte pre-shared key. Operator wires the key into both
    deployments out-of-band (e.g., environment variable read at
    server startup). The cipher is stateless after construction —
    each `encrypt`/`decrypt` call generates a fresh random nonce
    bound to the AAD.

    Wire format of the ciphertext:
      ``ciphertext = nonce (12 bytes) || aes_gcm_output``

    where ``aes_gcm_output`` is the standard library output
    (encrypted plaintext + 16-byte authentication tag).
    """

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, (bytes, bytearray)):
            raise ProbsCipherError(
                f"key must be bytes, got {type(key).__name__}"
            )
        if len(key) != _AES_KEY_LEN:
            raise ProbsCipherError(
                f"key must be {_AES_KEY_LEN} bytes (AES-256), got "
                f"{len(key)} bytes"
            )
        self._aesgcm = AESGCM(bytes(key))

    def encrypt(
        self,
        *,
        probs: Sequence[float],
        request_id: str,
        stage_index: int,
    ) -> bytes:
        """Encrypt K probs to wire-format ciphertext bytes.

        Plaintext: K floats in [0, 1] → ``K * 8`` little-endian
        float64 bytes. Ciphertext: ``nonce (12B) || AES-GCM output``.
        AAD: ``request_id_utf8 || b"|" || stage_index_byte``.

        The fresh random nonce makes the ciphertext non-deterministic
        across calls with the same plaintext — defends against
        deterministic-encryption attacks where a passive observer
        could recognize repeated proposals.
        """
        plaintext = _encode_probs(probs)
        aad = _aad(request_id, stage_index)
        # AESGCM.encrypt requires a 12-byte nonce; we generate via
        # the library's random for cryptographic strength.
        nonce = AESGCM.generate_key(bit_length=128)[:_GCM_NONCE_LEN]
        try:
            ct = self._aesgcm.encrypt(nonce, plaintext, aad)
        except Exception as exc:  # noqa: BLE001
            raise ProbsCipherError(
                f"AES-GCM encrypt failed: {exc}"
            ) from exc
        return nonce + ct

    def decrypt(
        self,
        *,
        ciphertext: bytes,
        request_id: str,
        stage_index: int,
        expected_k: int,
    ) -> List[float]:
        """Decrypt wire-format ciphertext bytes → K floats.

        Validates: total ciphertext length ≥ nonce + tag (28 bytes
        minimum); AAD matches the binding; plaintext length matches
        ``expected_k * 8`` (the caller MUST supply K from the co-set
        ``proposed_token_ids`` field — defends against length-mismatch
        attacks where a tampered ciphertext successfully decrypts to
        a wrong-K plaintext).

        Raises ``ProbsDecryptionError`` on tag mismatch (wrong key,
        AAD mismatch from cross-(request, stage) replay, or tampered
        bytes). Raises ``ProbsCipherError`` on length / range
        validation failures.
        """
        if not isinstance(ciphertext, (bytes, bytearray)):
            raise ProbsCipherError(
                f"ciphertext must be bytes, got "
                f"{type(ciphertext).__name__}"
            )
        if len(ciphertext) < _GCM_NONCE_LEN + 16:
            raise ProbsCipherError(
                f"ciphertext too short: {len(ciphertext)} bytes "
                f"(minimum {_GCM_NONCE_LEN + 16} = nonce + GCM tag)"
            )
        nonce = bytes(ciphertext[:_GCM_NONCE_LEN])
        ct = bytes(ciphertext[_GCM_NONCE_LEN:])
        aad = _aad(request_id, stage_index)
        try:
            plaintext = self._aesgcm.decrypt(nonce, ct, aad)
        except InvalidTag as exc:
            raise ProbsDecryptionError(
                "AES-GCM tag mismatch — ciphertext was tampered, the "
                "wrong key was used, OR the ciphertext was replayed "
                "across (request_id, stage_index) pairs (relay "
                "adversary). Caller should surface "
                "MALFORMED_REQUEST."
            ) from exc
        return _decode_probs(plaintext, expected_k)


def derive_key_from_psk(
    psk_bytes: bytes, *, salt: bytes = b"prsm-3.x.11.q.y-probs-cipher",
) -> bytes:
    """Operator helper: derive a 32-byte AES key from a pre-shared
    secret of arbitrary length via HKDF-SHA256.

    The salt is constant across deployments — operators rotate keys
    by rotating the PSK, not the salt. Per-request derivation is
    NOT done here (the same derived key is reused across all
    requests served by a (executor, tail) pair); the AES-GCM nonce
    + AAD binding provide per-request uniqueness.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    if not isinstance(psk_bytes, (bytes, bytearray)):
        raise ProbsCipherError(
            f"psk_bytes must be bytes, got {type(psk_bytes).__name__}"
        )
    if len(psk_bytes) < 16:
        raise ProbsCipherError(
            f"psk_bytes too short: {len(psk_bytes)} bytes (minimum "
            f"16 — operator should provision a 32+ byte PSK)"
        )
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_AES_KEY_LEN,
        salt=salt,
        info=b"probs-cipher-v1",
    )
    return hkdf.derive(bytes(psk_bytes))
