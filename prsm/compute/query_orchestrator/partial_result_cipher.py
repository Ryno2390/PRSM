"""Partial-result cipher for AggregateResponse.encrypted_plaintext.

Closes the placeholder in
``prsm/compute/query_orchestrator/aggregate_server.py:287-294``
and the symmetric prompter-side decrypt at
``aggregator_client_adapter.py:388-392``.

Construction:
  1. Convert each side's existing long-term Ed25519 keypair to its
     X25519 (Curve25519) equivalent via libsodium's
     ``crypto_sign_ed25519_{pk,sk}_to_curve25519``. This avoids a
     wire-format extension to ship per-request X25519 pubkeys —
     the existing ``prompter_pubkey`` (on the signed request) and
     ``aggregator_pubkey`` (on the response) already authenticate
     the participants, and we derive the encryption keys from
     those same trusted bits.
  2. ECDH on X25519 → 32-byte shared secret.
  3. HKDF-SHA256(shared_secret,
                 info = b"prsm:aggregate-cipher:v1\\n" || request_id,
                 salt = b"prsm:aggregate-cipher:hkdf-salt:v1")
     → 32-byte XChaCha20-Poly1305 key. The ``request_id`` in info
     gives per-call domain separation; the version prefix lets us
     bump cipher schemes without protocol re-negotiation.
  4. XChaCha20-Poly1305 with 24-byte random nonce (matches the
     existing ``AggregateResponse.nonce`` field's width). AAD binds
     the AggregationCommit's signing payload so the ciphertext is
     rejected if the commit is tampered with.

Honest scope (v1):
  - **No forward secrecy.** We reuse long-term Ed25519 keys; if
    either side's private key is compromised, all past traffic is
    decryptable. A separate ephemeral-keys follow-on would extend
    AggregateRequest/Response with a fresh per-request X25519
    public half on each side. The wire format already has 32-byte
    pubkey fields we'd reuse.
  - **Marginal value over TLS.** ``HttpAggregateTransport`` runs
    on httpx with TLS, so passive observers between aggregator and
    prompter are already excluded. This in-band cipher hardens
    against TLS-MITM proxies and on-disk caches; the practical
    threat model is narrow.
  - **No replay defense.** A relay could replay a complete
    response envelope (commit_signature + ciphertext + nonce). The
    AggregationCommit's ``query_id`` already serves as the
    application-layer freshness binding; the cipher itself does
    not add nonce-cache-style replay protection.
"""
from __future__ import annotations

from os import urandom

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from nacl.bindings import (
    crypto_aead_xchacha20poly1305_ietf_NPUBBYTES,
    crypto_aead_xchacha20poly1305_ietf_decrypt,
    crypto_aead_xchacha20poly1305_ietf_encrypt,
    crypto_scalarmult,
    crypto_sign_ed25519_pk_to_curve25519,
    crypto_sign_ed25519_sk_to_curve25519,
)
from nacl.exceptions import CryptoError


# ──────────────────────────────────────────────────────────────────────
# Constants — version prefix locks the cipher scheme. Bumping this
# string breaks compat by design (forces ratification + redeploy).
# ──────────────────────────────────────────────────────────────────────


_HKDF_INFO_PREFIX = b"prsm:aggregate-cipher:v1\n"
_HKDF_SALT = b"prsm:aggregate-cipher:hkdf-salt:v1"
_NONCE_LEN = crypto_aead_xchacha20poly1305_ietf_NPUBBYTES  # 24
_AEAD_KEY_LEN = 32


class PartialResultCipherError(Exception):
    """Raised on any cipher-side failure — bad input shape, AEAD
    MAC mismatch, key conversion error. Callers map this to their
    own typed exception (``AggregationCommitMismatchError`` on the
    prompter, ``AggregateServerError`` on the aggregator)."""


def _validate_32(name: str, value: bytes) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise PartialResultCipherError(
            f"{name} must be bytes, got {type(value).__name__}"
        )
    if len(value) != 32:
        raise PartialResultCipherError(
            f"{name} must be 32 bytes (Ed25519 raw), got {len(value)}"
        )
    return bytes(value)


def _ed25519_priv_to_x25519(privkey_32: bytes) -> bytes:
    """Convert a 32-byte raw Ed25519 private key to its X25519 (Curve25519)
    equivalent via libsodium's official derivation.

    libsodium expects a 64-byte Ed25519 secret key (seed + pubkey concat)
    in the standard NaCl convention. We derive the matching pubkey from
    the seed and concat to satisfy the API.
    """
    # libsodium's sk-to-curve takes the 64-byte secret (seed||pk).
    # We can compute pk from sk via signing-key derivation — the
    # cryptography lib gives us exactly that.
    from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed
    pubkey = _ed.Ed25519PrivateKey.from_private_bytes(
        privkey_32,
    ).public_key().public_bytes_raw()
    nacl_sk = privkey_32 + pubkey  # 64 bytes
    try:
        return crypto_sign_ed25519_sk_to_curve25519(nacl_sk)
    except CryptoError as exc:
        raise PartialResultCipherError(
            f"Ed25519 → X25519 private-key conversion failed: {exc}"
        ) from exc


def _ed25519_pub_to_x25519(pubkey_32: bytes) -> bytes:
    try:
        return crypto_sign_ed25519_pk_to_curve25519(pubkey_32)
    except CryptoError as exc:
        raise PartialResultCipherError(
            f"Ed25519 → X25519 public-key conversion failed: {exc}"
        ) from exc


def _derive_aead_key(
    *,
    my_x25519_priv: bytes,
    their_x25519_pub: bytes,
    request_id: bytes,
) -> bytes:
    try:
        shared = crypto_scalarmult(my_x25519_priv, their_x25519_pub)
    except CryptoError as exc:
        raise PartialResultCipherError(
            f"X25519 ECDH failed: {exc}"
        ) from exc
    info = _HKDF_INFO_PREFIX + bytes(request_id)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_AEAD_KEY_LEN,
        salt=_HKDF_SALT,
        info=info,
    )
    return hkdf.derive(shared)


def encrypt_aggregate_response(
    *,
    aggregator_ed25519_privkey: bytes,
    prompter_ed25519_pubkey: bytes,
    plaintext: bytes,
    request_id: bytes,
    commit_aad: bytes,
) -> tuple[bytes, bytes]:
    """Encrypt the aggregator's plaintext partial-result combination
    for delivery on AggregateResponse.

    Returns (ciphertext, nonce). ``nonce`` is 24 bytes — assign
    directly to ``AggregateResponse.nonce``.
    """
    aggregator_priv = _validate_32(
        "aggregator_ed25519_privkey", aggregator_ed25519_privkey,
    )
    prompter_pub = _validate_32(
        "prompter_ed25519_pubkey", prompter_ed25519_pubkey,
    )
    if not isinstance(plaintext, (bytes, bytearray)):
        raise PartialResultCipherError(
            f"plaintext must be bytes, got {type(plaintext).__name__}"
        )

    my_x25519 = _ed25519_priv_to_x25519(aggregator_priv)
    their_x25519 = _ed25519_pub_to_x25519(prompter_pub)
    key = _derive_aead_key(
        my_x25519_priv=my_x25519,
        their_x25519_pub=their_x25519,
        request_id=request_id,
    )

    nonce = urandom(_NONCE_LEN)
    try:
        ciphertext = crypto_aead_xchacha20poly1305_ietf_encrypt(
            bytes(plaintext), bytes(commit_aad), nonce, key,
        )
    except CryptoError as exc:
        raise PartialResultCipherError(
            f"XChaCha20-Poly1305 encrypt failed: {exc}"
        ) from exc
    return ciphertext, nonce


def decrypt_aggregate_response(
    *,
    prompter_ed25519_privkey: bytes,
    aggregator_ed25519_pubkey: bytes,
    ciphertext: bytes,
    nonce: bytes,
    request_id: bytes,
    commit_aad: bytes,
) -> bytes:
    """Decrypt and authenticate the AggregateResponse.encrypted_plaintext.

    Raises ``PartialResultCipherError`` on AAD mismatch, MAC failure,
    short nonce, or any key-conversion error.
    """
    prompter_priv = _validate_32(
        "prompter_ed25519_privkey", prompter_ed25519_privkey,
    )
    aggregator_pub = _validate_32(
        "aggregator_ed25519_pubkey", aggregator_ed25519_pubkey,
    )
    if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != _NONCE_LEN:
        raise PartialResultCipherError(
            f"nonce must be {_NONCE_LEN} bytes (XChaCha20-Poly1305 IETF), "
            f"got {len(nonce) if isinstance(nonce, (bytes, bytearray)) else type(nonce).__name__}"
        )

    my_x25519 = _ed25519_priv_to_x25519(prompter_priv)
    their_x25519 = _ed25519_pub_to_x25519(aggregator_pub)
    key = _derive_aead_key(
        my_x25519_priv=my_x25519,
        their_x25519_pub=their_x25519,
        request_id=request_id,
    )

    try:
        return crypto_aead_xchacha20poly1305_ietf_decrypt(
            bytes(ciphertext), bytes(commit_aad), bytes(nonce), key,
        )
    except CryptoError as exc:
        raise PartialResultCipherError(
            f"XChaCha20-Poly1305 decrypt/MAC failed: {exc}"
        ) from exc
