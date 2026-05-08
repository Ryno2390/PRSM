"""Tests for the AggregateResponse partial-result cipher.

Closes the placeholder in
``prsm/compute/query_orchestrator/aggregate_server.py:287-294`` and
the symmetric placeholder in
``aggregator_client_adapter.py:388-392``.

Design:
  - Both prompter (signer) and aggregator (signer) already carry a
    long-term Ed25519 keypair on the wire (request signing /
    response commit signing). We DERIVE per-side X25519 keys from
    the existing Ed25519 keys via libsodium's
    ``crypto_sign_ed25519_{pk,sk}_to_curve25519`` so the wire format
    needs no extension to ship the X25519 public halves.
  - ECDH(my-X25519-priv, their-X25519-pub) → 32-byte shared.
  - HKDF-SHA256 over shared with
    ``info = b"prsm:aggregate-cipher:v1\\n" || request_id_bytes``
    → 32-byte XChaCha20-Poly1305 key. Per-request domain separation
    via request_id; protocol versioning in the prefix.
  - XChaCha20-Poly1305 with 24-byte random nonce (matches the
    existing ``AggregateResponse.nonce`` field's 24-byte width).
  - AAD binds the AggregationCommit's signing payload so the
    ciphertext is rejected if the commit was tampered with.

Honest scope:
  - Forward secrecy is NOT achieved here — we reuse long-term Ed25519
    keys. A separate ephemeral-keys follow-on would add a wire-format
    extension for fresh per-request X25519 pubkeys. Documented in
    docstring §"Honest scope".
"""
from __future__ import annotations

import os

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from prsm.compute.query_orchestrator.partial_result_cipher import (
    PartialResultCipherError,
    decrypt_aggregate_response,
    encrypt_aggregate_response,
)


# ──────────────────────────────────────────────────────────────────────
# Fixture helpers
# ──────────────────────────────────────────────────────────────────────


def _ed25519_keypair():
    """Generate a fresh Ed25519 keypair, return (priv_bytes, pub_bytes).

    Uses 32-byte raw private + 32-byte raw public (matches the wire
    format used elsewhere in PRSM).
    """
    priv = ed25519.Ed25519PrivateKey.generate()
    return (
        priv.private_bytes_raw(),
        priv.public_key().public_bytes_raw(),
    )


# ──────────────────────────────────────────────────────────────────────
# Round-trip
# ──────────────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_encrypt_then_decrypt_recovers_plaintext(self):
        prompter_priv, prompter_pub = _ed25519_keypair()
        aggregator_priv, aggregator_pub = _ed25519_keypair()

        plaintext = b"COUNT=42 and other tabular bytes"
        request_id = b"req-0001"
        commit_aad = b"prsm:aggregation-commit:v1\nq:hash:digest"

        ciphertext, nonce = encrypt_aggregate_response(
            aggregator_ed25519_privkey=aggregator_priv,
            prompter_ed25519_pubkey=prompter_pub,
            plaintext=plaintext,
            request_id=request_id,
            commit_aad=commit_aad,
        )
        assert len(nonce) == 24
        assert ciphertext != plaintext  # actually encrypted

        recovered = decrypt_aggregate_response(
            prompter_ed25519_privkey=prompter_priv,
            aggregator_ed25519_pubkey=aggregator_pub,
            ciphertext=ciphertext,
            nonce=nonce,
            request_id=request_id,
            commit_aad=commit_aad,
        )
        assert recovered == plaintext

    def test_empty_plaintext_round_trips(self):
        prompter_priv, prompter_pub = _ed25519_keypair()
        aggregator_priv, aggregator_pub = _ed25519_keypair()

        ciphertext, nonce = encrypt_aggregate_response(
            aggregator_ed25519_privkey=aggregator_priv,
            prompter_ed25519_pubkey=prompter_pub,
            plaintext=b"",
            request_id=b"req-empty",
            commit_aad=b"aad",
        )
        recovered = decrypt_aggregate_response(
            prompter_ed25519_privkey=prompter_priv,
            aggregator_ed25519_pubkey=aggregator_pub,
            ciphertext=ciphertext,
            nonce=nonce,
            request_id=b"req-empty",
            commit_aad=b"aad",
        )
        assert recovered == b""

    def test_distinct_keys_per_call_produce_distinct_ciphertexts(self):
        prompter_priv, prompter_pub = _ed25519_keypair()
        aggregator_priv, aggregator_pub = _ed25519_keypair()

        plaintext = b"same payload"
        ct1, _ = encrypt_aggregate_response(
            aggregator_ed25519_privkey=aggregator_priv,
            prompter_ed25519_pubkey=prompter_pub,
            plaintext=plaintext,
            request_id=b"req-1",
            commit_aad=b"aad",
        )
        ct2, _ = encrypt_aggregate_response(
            aggregator_ed25519_privkey=aggregator_priv,
            prompter_ed25519_pubkey=prompter_pub,
            plaintext=plaintext,
            request_id=b"req-1",
            commit_aad=b"aad",
        )
        # Random nonce per call → distinct ciphertexts.
        assert ct1 != ct2


# ──────────────────────────────────────────────────────────────────────
# AAD + nonce + key tampering
# ──────────────────────────────────────────────────────────────────────


class TestTamperRejection:
    def test_aad_mismatch_rejects(self):
        prompter_priv, prompter_pub = _ed25519_keypair()
        aggregator_priv, aggregator_pub = _ed25519_keypair()

        ciphertext, nonce = encrypt_aggregate_response(
            aggregator_ed25519_privkey=aggregator_priv,
            prompter_ed25519_pubkey=prompter_pub,
            plaintext=b"data",
            request_id=b"req-1",
            commit_aad=b"correct-aad",
        )
        with pytest.raises(PartialResultCipherError):
            decrypt_aggregate_response(
                prompter_ed25519_privkey=prompter_priv,
                aggregator_ed25519_pubkey=aggregator_pub,
                ciphertext=ciphertext,
                nonce=nonce,
                request_id=b"req-1",
                commit_aad=b"WRONG-aad",
            )

    def test_request_id_mismatch_rejects(self):
        # request_id participates in HKDF info — different request_id
        # → different key → MAC fails.
        prompter_priv, prompter_pub = _ed25519_keypair()
        aggregator_priv, aggregator_pub = _ed25519_keypair()

        ciphertext, nonce = encrypt_aggregate_response(
            aggregator_ed25519_privkey=aggregator_priv,
            prompter_ed25519_pubkey=prompter_pub,
            plaintext=b"data",
            request_id=b"req-A",
            commit_aad=b"aad",
        )
        with pytest.raises(PartialResultCipherError):
            decrypt_aggregate_response(
                prompter_ed25519_privkey=prompter_priv,
                aggregator_ed25519_pubkey=aggregator_pub,
                ciphertext=ciphertext,
                nonce=nonce,
                request_id=b"req-B",
                commit_aad=b"aad",
            )

    def test_wrong_aggregator_pubkey_rejects(self):
        prompter_priv, prompter_pub = _ed25519_keypair()
        aggregator_priv, _ = _ed25519_keypair()
        _, attacker_pub = _ed25519_keypair()

        ciphertext, nonce = encrypt_aggregate_response(
            aggregator_ed25519_privkey=aggregator_priv,
            prompter_ed25519_pubkey=prompter_pub,
            plaintext=b"data",
            request_id=b"req-1",
            commit_aad=b"aad",
        )
        with pytest.raises(PartialResultCipherError):
            decrypt_aggregate_response(
                prompter_ed25519_privkey=prompter_priv,
                aggregator_ed25519_pubkey=attacker_pub,  # wrong
                ciphertext=ciphertext,
                nonce=nonce,
                request_id=b"req-1",
                commit_aad=b"aad",
            )

    def test_ciphertext_corruption_rejects(self):
        prompter_priv, prompter_pub = _ed25519_keypair()
        aggregator_priv, aggregator_pub = _ed25519_keypair()

        ciphertext, nonce = encrypt_aggregate_response(
            aggregator_ed25519_privkey=aggregator_priv,
            prompter_ed25519_pubkey=prompter_pub,
            plaintext=b"data",
            request_id=b"req-1",
            commit_aad=b"aad",
        )
        # Flip one bit in the middle of the ciphertext.
        flipped = bytearray(ciphertext)
        flipped[len(flipped) // 2] ^= 0x01
        with pytest.raises(PartialResultCipherError):
            decrypt_aggregate_response(
                prompter_ed25519_privkey=prompter_priv,
                aggregator_ed25519_pubkey=aggregator_pub,
                ciphertext=bytes(flipped),
                nonce=nonce,
                request_id=b"req-1",
                commit_aad=b"aad",
            )

    def test_nonce_corruption_rejects(self):
        prompter_priv, prompter_pub = _ed25519_keypair()
        aggregator_priv, aggregator_pub = _ed25519_keypair()

        ciphertext, nonce = encrypt_aggregate_response(
            aggregator_ed25519_privkey=aggregator_priv,
            prompter_ed25519_pubkey=prompter_pub,
            plaintext=b"data",
            request_id=b"req-1",
            commit_aad=b"aad",
        )
        wrong_nonce = bytes(b ^ 0xFF for b in nonce)
        with pytest.raises(PartialResultCipherError):
            decrypt_aggregate_response(
                prompter_ed25519_privkey=prompter_priv,
                aggregator_ed25519_pubkey=aggregator_pub,
                ciphertext=ciphertext,
                nonce=wrong_nonce,
                request_id=b"req-1",
                commit_aad=b"aad",
            )


# ──────────────────────────────────────────────────────────────────────
# Validation — input shape
# ──────────────────────────────────────────────────────────────────────


class TestInputValidation:
    def test_short_aggregator_priv_rejected(self):
        with pytest.raises(PartialResultCipherError, match="32 bytes"):
            encrypt_aggregate_response(
                aggregator_ed25519_privkey=b"\x00" * 16,
                prompter_ed25519_pubkey=b"\x00" * 32,
                plaintext=b"",
                request_id=b"r",
                commit_aad=b"a",
            )

    def test_short_prompter_pub_rejected(self):
        with pytest.raises(PartialResultCipherError, match="32 bytes"):
            encrypt_aggregate_response(
                aggregator_ed25519_privkey=b"\x00" * 32,
                prompter_ed25519_pubkey=b"\x00" * 16,
                plaintext=b"",
                request_id=b"r",
                commit_aad=b"a",
            )

    def test_decrypt_short_nonce_rejected(self):
        prompter_priv, prompter_pub = _ed25519_keypair()
        aggregator_priv, aggregator_pub = _ed25519_keypair()

        ciphertext, _ = encrypt_aggregate_response(
            aggregator_ed25519_privkey=aggregator_priv,
            prompter_ed25519_pubkey=prompter_pub,
            plaintext=b"x",
            request_id=b"r",
            commit_aad=b"a",
        )
        with pytest.raises(PartialResultCipherError, match="24"):
            decrypt_aggregate_response(
                prompter_ed25519_privkey=prompter_priv,
                aggregator_ed25519_pubkey=aggregator_pub,
                ciphertext=ciphertext,
                nonce=b"\x00" * 12,  # ChaCha20-Poly1305 IETF nonce, not XChaCha
                request_id=b"r",
                commit_aad=b"a",
            )
