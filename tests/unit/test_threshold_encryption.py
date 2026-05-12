"""Sprint 307 — threshold (t-of-n) encryption mode.

Vision §7 Enterprise Confidentiality Mode follow-on:
alternative to sprint 304's OR-decrypt for orgs that want
NO single decryption-capable party. Composes onto the
existing EncryptedPayload shape — same ciphertext, same
per-recipient sealing, but the sealed payload is a Shamir
SHARE of the symmetric key, not the full key.

Decryption is a two-phase protocol:
  1. Each recipient unseals their share locally
     (no plaintext exposed yet)
  2. Any t recipients pool their unsealed shares to
     reconstruct the symmetric key and decrypt the
     content
"""
from __future__ import annotations

import base64
import json
import os

import pytest

from prsm.enterprise.recipient_encryption import (
    EncryptedPayload,
    EnterpriseRecipient,
    ShareContribution,
    ThresholdParams,
    combine_shares_and_decrypt,
    decrypt_for_recipient,
    encrypt_for_threshold,
    generate_recipient_keypair,
    unseal_share_for_recipient,
)


# ── ThresholdParams ──────────────────────────────────


def test_threshold_params_round_trip():
    p = ThresholdParams(t=3, n=5)
    assert p.to_dict() == {"t": 3, "n": 5}
    assert ThresholdParams.from_dict(p.to_dict()) == p


def test_threshold_params_validation():
    with pytest.raises(ValueError):
        ThresholdParams(t=0, n=3)
    with pytest.raises(ValueError):
        ThresholdParams(t=4, n=3)


# ── encrypt_for_threshold ────────────────────────────


def _gen_recipients(n: int):
    """Return (recipients, privkeys) for n fresh keys."""
    recipients = []
    privkeys = []
    for i in range(n):
        priv, pub = generate_recipient_keypair()
        privkeys.append(priv)
        recipients.append(EnterpriseRecipient(
            identifier=f"recipient-{i}",
            x25519_pubkey_b64=pub,
        ))
    return recipients, privkeys


def test_encrypt_threshold_writes_threshold_field():
    recipients, _ = _gen_recipients(3)
    payload = encrypt_for_threshold(
        b"x", recipients, threshold=2,
    )
    assert payload.manifest.threshold == (
        ThresholdParams(t=2, n=3)
    )


def test_encrypt_threshold_each_entry_has_share_index():
    recipients, _ = _gen_recipients(5)
    payload = encrypt_for_threshold(
        b"x", recipients, threshold=3,
    )
    # Share indices are 1..n distinct
    indices = {
        e.share_index for e in payload.manifest.entries
    }
    assert indices == {1, 2, 3, 4, 5}


def test_encrypt_threshold_rejects_threshold_over_n():
    recipients, _ = _gen_recipients(3)
    with pytest.raises(ValueError, match="threshold"):
        encrypt_for_threshold(
            b"x", recipients, threshold=4,
        )


def test_encrypt_threshold_rejects_threshold_zero():
    recipients, _ = _gen_recipients(3)
    with pytest.raises(ValueError, match="threshold"):
        encrypt_for_threshold(
            b"x", recipients, threshold=0,
        )


def test_encrypt_threshold_rejects_empty_recipients():
    with pytest.raises(ValueError, match="recipient"):
        encrypt_for_threshold(
            b"x", [], threshold=1,
        )


# ── Two-phase decryption ─────────────────────────────


def test_threshold_decrypt_3_of_5_happy_path():
    recipients, privkeys = _gen_recipients(5)
    plaintext = b"high-stakes enterprise data" * 4
    payload = encrypt_for_threshold(
        plaintext, recipients, threshold=3,
    )
    # Three recipients unseal their shares independently
    contributions = [
        unseal_share_for_recipient(payload, privkeys[i])
        for i in (0, 2, 4)
    ]
    out = combine_shares_and_decrypt(
        payload, contributions,
    )
    assert out == plaintext


def test_threshold_decrypt_t_minus_1_shares_refused():
    recipients, privkeys = _gen_recipients(5)
    payload = encrypt_for_threshold(
        b"x" * 100, recipients, threshold=3,
    )
    contributions = [
        unseal_share_for_recipient(payload, privkeys[i])
        for i in (0, 1)  # only 2 of required 3
    ]
    with pytest.raises(ValueError, match="at least"):
        combine_shares_and_decrypt(
            payload, contributions,
        )


def test_threshold_decrypt_all_n_shares_works():
    recipients, privkeys = _gen_recipients(5)
    plaintext = b"x" * 64
    payload = encrypt_for_threshold(
        plaintext, recipients, threshold=3,
    )
    contributions = [
        unseal_share_for_recipient(payload, k)
        for k in privkeys
    ]
    assert combine_shares_and_decrypt(
        payload, contributions,
    ) == plaintext


def test_unseal_share_with_outsider_privkey_fails():
    recipients, _ = _gen_recipients(3)
    payload = encrypt_for_threshold(
        b"x", recipients, threshold=2,
    )
    outsider_priv, _ = generate_recipient_keypair()
    with pytest.raises(ValueError, match="no entry"):
        unseal_share_for_recipient(payload, outsider_priv)


def test_or_decrypt_refused_on_threshold_payload():
    """decrypt_for_recipient (sprint 304's OR-decrypt path)
    must refuse a threshold-mode payload — would otherwise
    silently produce garbled output."""
    recipients, privkeys = _gen_recipients(3)
    payload = encrypt_for_threshold(
        b"x", recipients, threshold=2,
    )
    with pytest.raises(ValueError, match="threshold"):
        decrypt_for_recipient(payload, privkeys[0])


def test_threshold_payload_round_trips_through_json():
    recipients, privkeys = _gen_recipients(4)
    plaintext = b"json round trip plaintext       "
    payload = encrypt_for_threshold(
        plaintext, recipients, threshold=2,
    )
    blob = json.dumps(payload.to_dict())
    restored = EncryptedPayload.from_dict(
        json.loads(blob),
    )
    assert restored.manifest.threshold == (
        ThresholdParams(t=2, n=4)
    )
    contributions = [
        unseal_share_for_recipient(restored, privkeys[i])
        for i in (1, 3)
    ]
    assert combine_shares_and_decrypt(
        restored, contributions,
    ) == plaintext


# ── Tamper detection ─────────────────────────────────


def test_threshold_ciphertext_tamper_detected():
    recipients, privkeys = _gen_recipients(3)
    payload = encrypt_for_threshold(
        b"secret", recipients, threshold=2,
    )
    # Tamper the ciphertext
    raw = base64.b64decode(payload.ciphertext_b64)
    bad = bytearray(raw)
    bad[20] ^= 0x01
    payload.ciphertext_b64 = base64.b64encode(
        bytes(bad),
    ).decode()
    contributions = [
        unseal_share_for_recipient(payload, privkeys[i])
        for i in (0, 1)
    ]
    with pytest.raises(Exception):
        combine_shares_and_decrypt(
            payload, contributions,
        )


def test_threshold_share_tamper_propagates_to_wrong_key():
    """Tampering a share's y-values means the reconstructed
    symmetric key will be WRONG, which surfaces at the AEAD
    decrypt as InvalidTag (no silent corruption)."""
    recipients, privkeys = _gen_recipients(3)
    payload = encrypt_for_threshold(
        b"secret data", recipients, threshold=2,
    )
    contributions = [
        unseal_share_for_recipient(payload, privkeys[i])
        for i in (0, 1)
    ]
    # Flip one byte of share[0] y_values
    tampered = bytearray(contributions[0].share_y_values)
    tampered[0] ^= 0x01
    contributions[0] = ShareContribution(
        share_index=contributions[0].share_index,
        share_y_values=bytes(tampered),
    )
    with pytest.raises(Exception):
        combine_shares_and_decrypt(
            payload, contributions,
        )


def test_threshold_combine_rejects_duplicate_share_index():
    recipients, privkeys = _gen_recipients(3)
    payload = encrypt_for_threshold(
        b"x", recipients, threshold=2,
    )
    contributions = [
        unseal_share_for_recipient(payload, privkeys[0]),
        # Same recipient twice — same share_index
        unseal_share_for_recipient(payload, privkeys[0]),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        combine_shares_and_decrypt(
            payload, contributions,
        )


# ── Backwards compat: OR-decrypt unaffected ─────────


def test_or_decrypt_payload_has_no_threshold():
    """Sprint 304 OR-decrypt mode must continue producing
    payloads with manifest.threshold == None."""
    from prsm.enterprise.recipient_encryption import (
        encrypt_for_recipients,
    )
    recipients, _ = _gen_recipients(2)
    payload = encrypt_for_recipients(b"x", recipients)
    assert payload.manifest.threshold is None
    for e in payload.manifest.entries:
        assert e.share_index is None


def test_or_decrypt_still_works_round_trip():
    """Belt-and-suspenders: confirm sprint 304's path
    works identically after the schema extension."""
    from prsm.enterprise.recipient_encryption import (
        encrypt_for_recipients,
    )
    priv, pub = generate_recipient_keypair()
    payload = encrypt_for_recipients(
        b"or-decrypt",
        [EnterpriseRecipient("alice", pub)],
    )
    assert decrypt_for_recipient(payload, priv) == (
        b"or-decrypt"
    )
