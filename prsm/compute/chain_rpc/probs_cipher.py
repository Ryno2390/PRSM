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

import os
import struct
from typing import List, Optional, Sequence, Tuple

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


__all__ = [
    "ProbsCipher",
    "ProbsCipherError",
    "ProbsDecryptionError",
    "X25519AnchoredCipher",
    "derive_key_from_psk",
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


def _aad_rollback(request_id: str, stage_index: int) -> bytes:
    """Phase 3.x.11.q.y' — distinct AAD for the rollback-prefix
    channel. Adds a ``|rollback`` suffix to the existing AAD so a
    relay adversary CANNOT replay a probs ciphertext into a
    rollback envelope (or vice versa). The two AAD spaces are
    disjoint by construction."""
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
    return (
        request_id.encode("utf-8")
        + b"|"
        + bytes([stage_index])
        + b"|rollback"
    )


def _encode_prefix(prefix: Sequence[int]) -> bytes:
    """Pack K token IDs as `K * 8` little-endian int64 bytes."""
    if not isinstance(prefix, (list, tuple)):
        raise ProbsCipherError(
            f"prefix must be list/tuple, got {type(prefix).__name__}"
        )
    out = bytearray()
    for i, t in enumerate(prefix):
        if isinstance(t, bool) or not isinstance(t, int):
            raise ProbsCipherError(
                f"prefix[{i}] must be int, got {type(t).__name__}"
            )
        if t < 0:
            raise ProbsCipherError(
                f"prefix[{i}] must be non-negative, got {t}"
            )
        out += struct.pack("<q", int(t))
    return bytes(out)


def _decode_prefix(plaintext: bytes, expected_k: int) -> List[int]:
    """Unpack `expected_k * 8` bytes → list of K token IDs."""
    expected_bytes = expected_k * 8
    if len(plaintext) != expected_bytes:
        raise ProbsCipherError(
            f"decrypted prefix length {len(plaintext)} does not "
            f"match expected K * 8 = {expected_bytes} (K={expected_k})"
        )
    out: List[int] = []
    for i in range(expected_k):
        offset = i * 8
        chunk = plaintext[offset:offset + 8]
        (t,) = struct.unpack("<q", chunk)
        if t < 0:
            raise ProbsCipherError(
                f"decrypted prefix[{i}] = {t} negative "
                f"(corrupted plaintext or wrong key)"
            )
        out.append(int(t))
    return out


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
        # AES-GCM requires a 12-byte nonce sourced from a CSPRNG.
        # ``os.urandom`` reads directly from the OS CSPRNG (the same
        # source ``cryptography`` uses internally for key
        # generation). Round-1 review L1: previously this used
        # ``AESGCM.generate_key(bit_length=128)[:_GCM_NONCE_LEN]``,
        # which is functionally equivalent but idiomatically wrong
        # — ``generate_key`` is intended for key generation, not
        # nonces. The reader-confusion cost outweighs any micro-
        # convenience.
        nonce = os.urandom(_GCM_NONCE_LEN)
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

    def encrypt_prefix(
        self,
        *,
        prefix: Sequence[int],
        request_id: str,
        stage_index: int,
    ) -> bytes:
        """Phase 3.x.11.q.y' — encrypt the accepted-prefix tokens
        for the always-rollback-K + replay-prefix protocol.

        Plaintext: K token IDs → ``K * 8`` little-endian int64 bytes.
        Ciphertext: ``nonce (12B) || AES-GCM output``.
        AAD: ``request_id_utf8 || b"|" || stage_index_byte || b"|rollback"``
        — distinct from the ``encrypt`` AAD so a relay adversary
        cannot replay a probs ciphertext into a rollback envelope
        (the AAD spaces are disjoint by construction).
        """
        plaintext = _encode_prefix(prefix)
        aad = _aad_rollback(request_id, stage_index)
        nonce = os.urandom(_GCM_NONCE_LEN)
        try:
            ct = self._aesgcm.encrypt(nonce, plaintext, aad)
        except Exception as exc:  # noqa: BLE001
            raise ProbsCipherError(
                f"AES-GCM encrypt_prefix failed: {exc}"
            ) from exc
        return nonce + ct

    def decrypt_prefix(
        self,
        *,
        ciphertext: bytes,
        request_id: str,
        stage_index: int,
        expected_k: int,
    ) -> List[int]:
        """Phase 3.x.11.q.y' — decrypt the accepted-prefix tokens.

        Validates total ciphertext length, AAD match (distinct
        from probs AAD), and plaintext length matches
        ``expected_k * 8``. Caller MUST pre-bound ``expected_k``
        per the rollback-protocol contract: prefix length is at
        most ``MAX_VERIFY_BATCH_TOKENS`` (the cap is enforced by
        the wire validator on ``RollbackCacheRequest``).
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
        aad = _aad_rollback(request_id, stage_index)
        try:
            plaintext = self._aesgcm.decrypt(nonce, ct, aad)
        except InvalidTag as exc:
            raise ProbsDecryptionError(
                "AES-GCM tag mismatch on rollback prefix — "
                "ciphertext tampered, wrong key, OR ciphertext "
                "replayed across (request_id, stage_index) pairs "
                "or across the probs/rollback AAD boundary. "
                "Caller should surface MALFORMED_REQUEST."
            ) from exc
        return _decode_prefix(plaintext, expected_k)


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


# ──────────────────────────────────────────────────────────────────────────
# X25519AnchoredCipher — Phase 3.x.11.q.y'
# ──────────────────────────────────────────────────────────────────────────


class X25519AnchoredCipher:
    """Phase 3.x.11.q.y' — per-request ECDH key negotiation cipher.

    Drop-in alternative to ``ProbsCipher`` for executors and tail
    stages that want fresh-per-request AES keys without operator-
    managed PSK distribution. The cipher derives a per-request
    AES-256 key by:

      1. Reading the executor's ephemeral X25519 public key from
         the request's HandoffToken (``request.upstream_token.
         ephemeral_pubkey``); the executor mints a fresh keypair
         per request and ships the public half on the (signed)
         token. Token signature commits the ephemeral pubkey, so
         a relay can't substitute it.
      2. Performing ECDH against this stage's long-term X25519
         private key → 32-byte shared secret.
      3. HKDF-SHA256 over the shared secret with
         ``info = request_id || stage_index_byte || chain_total_stages_byte``
         → AES-256 key. Per-request uniqueness comes from the
         ephemeral half (item 1); per-stage uniqueness comes from
         the HKDF info input.

    Surface-compatible with ``ProbsCipher``: implements ``encrypt``
    / ``decrypt`` / ``encrypt_prefix`` / ``decrypt_prefix`` with
    the same signatures, so swapping is a drop-in change at the
    ``RpcChainExecutor`` / ``LayerStageServer`` level. Operators
    pick PSK or ECDH based on deployment posture; the runtime
    integration is unchanged.

    **Operational model.**
    - Long-term X25519 keypair per stage, signed by the anchor
      under the existing publisher-key infrastructure (so the
      executor can resolve + verify the public half).
    - Executor mints + caches a fresh ephemeral X25519 keypair
      per request. Carries the public half on every HandoffToken
      it issues (one per stage in the chain). Caches the private
      half in-memory for the request's lifetime.
    - Receiving stage performs ECDH against its long-term private
      key + the on-token ephemeral public; derives the cipher key.
      The cipher caches derived keys per-request to avoid
      recomputing on every encrypt/decrypt call within the same
      request.

    **Security model.**
    - Forward secrecy: per-request ephemeral keys mean compromise
      of one request's traffic does not compromise other requests'.
      Compromise of the long-term key still permits decryption of
      future traffic but NOT past traffic (since the executor's
      ephemeral private was discarded after the request completed).
    - The anchor + token signature chain authenticates the public
      keys. A relay cannot substitute a different ephemeral
      pubkey without invalidating the settler's signature on
      HandoffToken.

    **Honest scope (v1).**
    - Replay-attack window inside ``deadline_unix`` is NOT closed
      here — a relay could replay an entire request envelope
      (including the ephemeral pubkey + ciphertexts). Mitigation
      via per-stage nonce cache is orthogonal; defer to follow-up.
    - Post-quantum resistance not addressed. R6 trigger-watch.
    """

    def __init__(self, local_x25519_privkey_bytes: bytes) -> None:
        """``local_x25519_privkey_bytes`` is the 32-byte raw private
        key (X25519 format per RFC 7748). Operators load this from
        secure storage at server startup; see ``generate_keypair``
        for fresh key minting."""
        # Lazy import to keep the module loadable when X25519 isn't
        # available (e.g., older cryptography versions). Fail loudly
        # at construction-time rather than at first encrypt.
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PrivateKey,
        )
        if not isinstance(
            local_x25519_privkey_bytes, (bytes, bytearray),
        ):
            raise ProbsCipherError(
                f"local_x25519_privkey_bytes must be bytes, got "
                f"{type(local_x25519_privkey_bytes).__name__}"
            )
        if len(local_x25519_privkey_bytes) != 32:
            raise ProbsCipherError(
                f"local_x25519_privkey_bytes must be exactly 32 "
                f"bytes (X25519 raw private key), got "
                f"{len(local_x25519_privkey_bytes)}"
            )
        self._privkey = X25519PrivateKey.from_private_bytes(
            bytes(local_x25519_privkey_bytes),
        )
        # Per-(request_id, ephemeral_pubkey) → AESGCM cache. Bounded
        # by the request lifecycle: the operator's TTL sweeper +
        # eviction broadcast bounds the cache size. We don't add
        # an internal LRU here because the cache key set is
        # already bounded by concurrent in-flight requests.
        self._key_cache: dict = {}

    def _derive_key(
        self,
        *,
        ephemeral_pubkey_bytes: bytes,
        request_id: str,
        stage_index: int,
        chain_total_stages: Optional[int] = None,
    ) -> "AESGCM":
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PublicKey,
        )
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        cache_key = (
            request_id,
            bytes(ephemeral_pubkey_bytes),
            int(stage_index),
            int(chain_total_stages or 0),
        )
        cached = self._key_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            peer_pub = X25519PublicKey.from_public_bytes(
                bytes(ephemeral_pubkey_bytes),
            )
            shared_secret = self._privkey.exchange(peer_pub)
        except Exception as exc:  # noqa: BLE001
            raise ProbsCipherError(
                f"X25519 ECDH failed: {exc}"
            ) from exc
        # info input bound to (request_id, stage_index,
        # chain_total_stages). chain_total_stages defends against
        # chain-length forgery (a relay can't lift an envelope
        # from a 2-stage chain into a 3-stage chain even at the
        # same stage_index).
        info = (
            request_id.encode("utf-8")
            + b"|"
            + bytes([int(stage_index) & 0xFF])
            + b"|"
            + bytes([int(chain_total_stages or 0) & 0xFF])
        )
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=_AES_KEY_LEN,
            salt=b"prsm-3.x.11.q.y-prime-x25519-aead",
            info=info,
        )
        derived = hkdf.derive(shared_secret)
        aesgcm = AESGCM(derived)
        self._key_cache[cache_key] = aesgcm
        return aesgcm

    @staticmethod
    def generate_keypair() -> Tuple[bytes, bytes]:
        """Mint a fresh X25519 keypair. Returns
        ``(private_bytes, public_bytes)`` (each 32 raw bytes).
        Operators use this to provision long-term server keys
        AND the executor uses it to mint per-request ephemeral
        keys."""
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PrivateKey,
        )
        from cryptography.hazmat.primitives import serialization
        priv = X25519PrivateKey.generate()
        priv_bytes = priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub = priv.public_key()
        pub_bytes = pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return bytes(priv_bytes), bytes(pub_bytes)

    # ─── Surface-compatible API ──────────────────────────────────────
    #
    # The methods below mirror ProbsCipher's surface. They take
    # the same kwargs (request_id, stage_index, etc.) plus a new
    # ``ephemeral_pubkey`` kwarg that the caller (executor or
    # server) extracts from the request's HandoffToken. The
    # ProbsCipher integration in client.py / server.py duck-types
    # via Protocol; the executor-side wrapper in client.py
    # detects which cipher class is in use and threads the
    # appropriate kwargs.

    def encrypt(
        self,
        *,
        probs: Sequence[float],
        request_id: str,
        stage_index: int,
        ephemeral_pubkey: bytes,
        chain_total_stages: Optional[int] = None,
    ) -> bytes:
        plaintext = _encode_probs(probs)
        aad = _aad(request_id, stage_index)
        nonce = os.urandom(_GCM_NONCE_LEN)
        aesgcm = self._derive_key(
            ephemeral_pubkey_bytes=ephemeral_pubkey,
            request_id=request_id,
            stage_index=stage_index,
            chain_total_stages=chain_total_stages,
        )
        try:
            ct = aesgcm.encrypt(nonce, plaintext, aad)
        except Exception as exc:  # noqa: BLE001
            raise ProbsCipherError(
                f"X25519AnchoredCipher.encrypt failed: {exc}"
            ) from exc
        return nonce + ct

    def decrypt(
        self,
        *,
        ciphertext: bytes,
        request_id: str,
        stage_index: int,
        expected_k: int,
        ephemeral_pubkey: bytes,
        chain_total_stages: Optional[int] = None,
    ) -> List[float]:
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
        aesgcm = self._derive_key(
            ephemeral_pubkey_bytes=ephemeral_pubkey,
            request_id=request_id,
            stage_index=stage_index,
            chain_total_stages=chain_total_stages,
        )
        try:
            plaintext = aesgcm.decrypt(nonce, ct, aad)
        except InvalidTag as exc:
            raise ProbsDecryptionError(
                "X25519AnchoredCipher: AES-GCM tag mismatch — "
                "ciphertext tampered, ephemeral_pubkey "
                "substituted, OR replayed across (request_id, "
                "stage_index) pairs. Caller should surface "
                "MALFORMED_REQUEST."
            ) from exc
        return _decode_probs(plaintext, expected_k)

    def encrypt_prefix(
        self,
        *,
        prefix: Sequence[int],
        request_id: str,
        stage_index: int,
        ephemeral_pubkey: bytes,
        chain_total_stages: Optional[int] = None,
    ) -> bytes:
        plaintext = _encode_prefix(prefix)
        aad = _aad_rollback(request_id, stage_index)
        nonce = os.urandom(_GCM_NONCE_LEN)
        aesgcm = self._derive_key(
            ephemeral_pubkey_bytes=ephemeral_pubkey,
            request_id=request_id,
            stage_index=stage_index,
            chain_total_stages=chain_total_stages,
        )
        try:
            ct = aesgcm.encrypt(nonce, plaintext, aad)
        except Exception as exc:  # noqa: BLE001
            raise ProbsCipherError(
                f"X25519AnchoredCipher.encrypt_prefix failed: {exc}"
            ) from exc
        return nonce + ct

    def decrypt_prefix(
        self,
        *,
        ciphertext: bytes,
        request_id: str,
        stage_index: int,
        expected_k: int,
        ephemeral_pubkey: bytes,
        chain_total_stages: Optional[int] = None,
    ) -> List[int]:
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
        aad = _aad_rollback(request_id, stage_index)
        aesgcm = self._derive_key(
            ephemeral_pubkey_bytes=ephemeral_pubkey,
            request_id=request_id,
            stage_index=stage_index,
            chain_total_stages=chain_total_stages,
        )
        try:
            plaintext = aesgcm.decrypt(nonce, ct, aad)
        except InvalidTag as exc:
            raise ProbsDecryptionError(
                "X25519AnchoredCipher: AES-GCM tag mismatch on "
                "rollback prefix — ciphertext tampered, "
                "ephemeral_pubkey substituted, OR cross-AAD "
                "replay (probs ↔ rollback). Caller should "
                "surface MALFORMED_REQUEST."
            ) from exc
        return _decode_prefix(plaintext, expected_k)

    def evict_request(self, request_id: str) -> int:
        """Drop all cached AESGCM instances for ``request_id``.
        Called by the operator at request terminal. Returns the
        number of (ephemeral, stage) cache entries dropped."""
        keys_to_drop = [
            k for k in self._key_cache.keys() if k[0] == request_id
        ]
        for k in keys_to_drop:
            self._key_cache.pop(k, None)
        return len(keys_to_drop)
