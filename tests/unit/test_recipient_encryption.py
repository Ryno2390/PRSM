"""Sprint 304 — recipient-encrypted upload primitive.

Vision §7 Enterprise Confidentiality Mode foundation:
hybrid X25519 + ChaCha20-Poly1305 encryption. Random
symmetric key encrypts the content; per-recipient sealed
copies of the symmetric key live in a manifest. Any one
designated recipient can decrypt (OR-decrypt semantics);
no one outside the recipient set can, regardless of FTNS
balance.

The cryptographic primitive is what stops a hacker —
not a token, not a payment gate.
"""
from __future__ import annotations

import base64
import json
import os

import pytest

from prsm.enterprise.recipient_encryption import (
    EncryptedPayload,
    EnterpriseRecipient,
    MANIFEST_VERSION,
    RecipientEntry,
    RecipientManifest,
    decrypt_for_recipient,
    encrypt_for_recipients,
    generate_recipient_keypair,
)


# ── Keypair generation ───────────────────────────────


def test_generate_keypair_returns_b64_strings():
    priv, pub = generate_recipient_keypair()
    assert isinstance(priv, str)
    assert isinstance(pub, str)
    # X25519 keys are 32 bytes → 44-char b64 (with padding)
    assert len(base64.b64decode(priv)) == 32
    assert len(base64.b64decode(pub)) == 32


def test_generate_keypair_unique():
    p1 = generate_recipient_keypair()
    p2 = generate_recipient_keypair()
    assert p1 != p2


# ── Single-recipient round-trip ──────────────────────


def test_encrypt_decrypt_round_trip_single_recipient():
    priv, pub = generate_recipient_keypair()
    plaintext = b"sensitive enterprise data"
    payload = encrypt_for_recipients(
        plaintext,
        [EnterpriseRecipient(
            identifier="alice@corp.com",
            x25519_pubkey_b64=pub,
        )],
    )
    out = decrypt_for_recipient(payload, priv)
    assert out == plaintext


def test_encrypt_empty_plaintext():
    priv, pub = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"",
        [EnterpriseRecipient(
            identifier="alice", x25519_pubkey_b64=pub,
        )],
    )
    assert decrypt_for_recipient(payload, priv) == b""


def test_encrypt_large_plaintext():
    """1 MiB sanity — the symmetric layer is ChaCha20, so
    this must work without stack pressure."""
    priv, pub = generate_recipient_keypair()
    plaintext = os.urandom(1024 * 1024)
    payload = encrypt_for_recipients(
        plaintext,
        [EnterpriseRecipient(
            identifier="alice", x25519_pubkey_b64=pub,
        )],
    )
    assert decrypt_for_recipient(payload, priv) == plaintext


# ── Multi-recipient OR-decrypt ───────────────────────


def test_each_recipient_can_decrypt_independently():
    """OR-decrypt — any one designated recipient suffices."""
    priv_a, pub_a = generate_recipient_keypair()
    priv_b, pub_b = generate_recipient_keypair()
    priv_c, pub_c = generate_recipient_keypair()
    plaintext = b"audit-shared data"
    payload = encrypt_for_recipients(
        plaintext,
        [
            EnterpriseRecipient("alice", pub_a),
            EnterpriseRecipient("bob", pub_b),
            EnterpriseRecipient("carol", pub_c),
        ],
    )
    assert decrypt_for_recipient(payload, priv_a) == plaintext
    assert decrypt_for_recipient(payload, priv_b) == plaintext
    assert decrypt_for_recipient(payload, priv_c) == plaintext


def test_manifest_has_one_entry_per_recipient():
    priv_a, pub_a = generate_recipient_keypair()
    _, pub_b = generate_recipient_keypair()
    _, pub_c = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"x",
        [
            EnterpriseRecipient("alice", pub_a),
            EnterpriseRecipient("bob", pub_b),
            EnterpriseRecipient("carol", pub_c),
        ],
    )
    assert len(payload.manifest.entries) == 3
    idents = {e.identifier for e in payload.manifest.entries}
    assert idents == {"alice", "bob", "carol"}


def test_manifest_version_pinned():
    _, pub = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"x",
        [EnterpriseRecipient("alice", pub)],
    )
    assert payload.manifest.version == MANIFEST_VERSION


# ── Outsider rejection ───────────────────────────────


def test_outsider_cannot_decrypt():
    """Someone NOT in the recipient set — even with valid
    keypair format — cannot decrypt. This is the load-
    bearing claim: FTNS balance is irrelevant."""
    _, pub = generate_recipient_keypair()
    outsider_priv, _ = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"secret",
        [EnterpriseRecipient("alice", pub)],
    )
    with pytest.raises(ValueError, match="no entry"):
        decrypt_for_recipient(payload, outsider_priv)


def test_malformed_recipient_privkey_rejected():
    _, pub = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"x",
        [EnterpriseRecipient("alice", pub)],
    )
    with pytest.raises(ValueError):
        decrypt_for_recipient(payload, "not-base64!")


def test_zero_recipients_rejected():
    with pytest.raises(ValueError, match="at least one"):
        encrypt_for_recipients(b"x", [])


def test_duplicate_recipient_identifier_rejected():
    _, pub_a = generate_recipient_keypair()
    _, pub_b = generate_recipient_keypair()
    with pytest.raises(ValueError, match="duplicate"):
        encrypt_for_recipients(
            b"x",
            [
                EnterpriseRecipient("alice", pub_a),
                EnterpriseRecipient("alice", pub_b),
            ],
        )


def test_malformed_recipient_pubkey_rejected():
    with pytest.raises(ValueError):
        encrypt_for_recipients(
            b"x",
            [EnterpriseRecipient(
                "alice", x25519_pubkey_b64="not-a-key",
            )],
        )


# ── Tamper detection ─────────────────────────────────


def test_ciphertext_tamper_rejected():
    priv, pub = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"sensitive",
        [EnterpriseRecipient("alice", pub)],
    )
    # Flip one byte in the ciphertext
    raw = base64.b64decode(payload.ciphertext_b64)
    tampered = bytearray(raw)
    tampered[20] ^= 0x01  # past the nonce prefix
    payload.ciphertext_b64 = base64.b64encode(
        bytes(tampered),
    ).decode()
    with pytest.raises(Exception):
        decrypt_for_recipient(payload, priv)


def test_sealed_key_tamper_rejected():
    priv, pub = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"sensitive",
        [EnterpriseRecipient("alice", pub)],
    )
    raw = base64.b64decode(
        payload.manifest.entries[0].sealed_symmetric_key_b64,
    )
    tampered = bytearray(raw)
    tampered[0] ^= 0x01
    payload.manifest.entries[
        0
    ].sealed_symmetric_key_b64 = base64.b64encode(
        bytes(tampered),
    ).decode()
    with pytest.raises(ValueError, match="no entry"):
        decrypt_for_recipient(payload, priv)


def test_manifest_identifier_swap_rejected():
    """Identifier-swap tamper detection — the AEAD AAD
    binding pins each sealed key to its original identifier.
    After swap, NO recipient can decrypt their own entry,
    because the entry their key matches now carries a
    mismatched identifier in the AAD. This is a stronger
    invariant than just "outsider can't read": even an
    authorized recipient is denied access if the manifest
    has been tampered with."""
    priv_a, pub_a = generate_recipient_keypair()
    priv_b, pub_b = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"x",
        [
            EnterpriseRecipient("alice", pub_a),
            EnterpriseRecipient("bob", pub_b),
        ],
    )
    # Swap identifiers
    payload.manifest.entries[0].identifier = "bob"
    payload.manifest.entries[1].identifier = "alice"
    with pytest.raises(ValueError, match="no entry"):
        decrypt_for_recipient(payload, priv_a)
    with pytest.raises(ValueError, match="no entry"):
        decrypt_for_recipient(payload, priv_b)


# ── Serialization round-trip ─────────────────────────


def test_payload_to_dict_round_trip():
    priv, pub = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"plaintext here",
        [EnterpriseRecipient("alice", pub)],
    )
    as_dict = payload.to_dict()
    # Must be JSON-serializable end-to-end
    blob = json.dumps(as_dict)
    restored = EncryptedPayload.from_dict(json.loads(blob))
    assert decrypt_for_recipient(restored, priv) == (
        b"plaintext here"
    )


def test_dict_shape_documented():
    """The shape of to_dict() is part of the wire protocol —
    pin it explicitly so future refactors don't silently
    break the SDK."""
    _, pub = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"x",
        [EnterpriseRecipient("alice", pub)],
    )
    d = payload.to_dict()
    assert set(d.keys()) >= {"ciphertext_b64", "manifest"}
    assert d["manifest"]["version"] == MANIFEST_VERSION
    e = d["manifest"]["entries"][0]
    assert set(e.keys()) >= {
        "identifier",
        "ephemeral_pubkey_b64",
        "nonce_b64",
        "sealed_symmetric_key_b64",
    }


def test_recipient_entry_round_trip():
    e = RecipientEntry(
        identifier="alice",
        ephemeral_pubkey_b64="AA==",
        nonce_b64="BB==",
        sealed_symmetric_key_b64="CC==",
    )
    assert RecipientEntry.from_dict(e.to_dict()) == e


def test_manifest_from_dict_unknown_version_rejected():
    with pytest.raises(ValueError, match="version"):
        RecipientManifest.from_dict({
            "version": "v999",
            "entries": [],
        })
