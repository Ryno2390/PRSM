"""Unit tests for prsm.storage.encryption.

Per docs/2026-04-22-phase7-storage-design-plan.md §6 Task 5.
"""

from __future__ import annotations

import os

import pytest

from prsm.storage.encryption import (
    AES_KEY_BYTES,
    AUTH_TAG_BYTES,
    AESKey,
    EncryptedPayload,
    EncryptionAuthError,
    EncryptionError,
    IV_BYTES,
    StreamingDecryptor,
    StreamingEncryptor,
    decrypt,
    encrypt,
    generate_key,
)


# -----------------------------------------------------------------------------
# Key generation
# -----------------------------------------------------------------------------


def test_generate_key_returns_32_byte_key():
    k = generate_key()
    assert len(k.key_bytes) == AES_KEY_BYTES == 32


def test_generate_key_ids_are_unique():
    keys = [generate_key() for _ in range(50)]
    assert len({k.key_id for k in keys}) == 50


def test_generate_keys_are_random():
    keys = [generate_key().key_bytes for _ in range(50)]
    assert len({k for k in keys}) == 50


def test_aeskey_rejects_wrong_size():
    with pytest.raises(EncryptionError):
        AESKey(key_id="abc", key_bytes=b"too-short")


# -----------------------------------------------------------------------------
# One-shot encrypt / decrypt
# -----------------------------------------------------------------------------


def test_encrypt_produces_iv_of_correct_length():
    k = generate_key()
    payload = encrypt(b"hello", k)
    assert len(payload.iv) == IV_BYTES == 12


def test_encrypt_produces_auth_tag_of_correct_length():
    k = generate_key()
    payload = encrypt(b"hello", k)
    assert len(payload.auth_tag) == AUTH_TAG_BYTES == 16


def test_encrypt_iv_is_random_across_calls():
    k = generate_key()
    ivs = {encrypt(b"x", k).iv for _ in range(50)}
    assert len(ivs) == 50  # random IVs should not collide


def test_encrypt_ciphertext_differs_from_plaintext():
    k = generate_key()
    payload = encrypt(b"sensitive", k)
    assert payload.ciphertext != b"sensitive"


def test_encrypt_propagates_key_id():
    k = generate_key()
    payload = encrypt(b"x", k)
    assert payload.key_id == k.key_id


def test_roundtrip_basic():
    k = generate_key()
    plaintext = b"The quick brown fox jumps over the lazy dog"
    payload = encrypt(plaintext, k)
    assert decrypt(payload, k) == plaintext


def test_roundtrip_empty_plaintext():
    k = generate_key()
    payload = encrypt(b"", k)
    assert decrypt(payload, k) == b""


def test_roundtrip_large_plaintext():
    k = generate_key()
    plaintext = os.urandom(10 * 1024 * 1024)  # 10 MiB
    payload = encrypt(plaintext, k)
    assert decrypt(payload, k) == plaintext


def test_roundtrip_with_associated_data():
    k = generate_key()
    ad = b"shard-manifest-sha256=deadbeef"
    payload = encrypt(b"sensitive", k, associated_data=ad)
    assert decrypt(payload, k, associated_data=ad) == b"sensitive"


# -----------------------------------------------------------------------------
# Authentication failures
# -----------------------------------------------------------------------------


def test_decrypt_with_wrong_key_raises():
    k1 = generate_key()
    k2 = AESKey(key_id=k1.key_id, key_bytes=os.urandom(32))  # same id, wrong bytes
    payload = encrypt(b"x", k1)
    with pytest.raises(EncryptionAuthError):
        decrypt(payload, k2)


def test_decrypt_with_mismatched_key_id_raises():
    k1 = generate_key()
    k2 = generate_key()
    payload = encrypt(b"x", k1)
    with pytest.raises(EncryptionError):
        decrypt(payload, k2)


def test_decrypt_with_tampered_ciphertext_raises():
    k = generate_key()
    payload = encrypt(b"sensitive payload for tamper test", k)
    tampered = EncryptedPayload(
        ciphertext=bytes(b ^ 0x01 for b in payload.ciphertext),
        iv=payload.iv,
        auth_tag=payload.auth_tag,
        key_id=payload.key_id,
    )
    with pytest.raises(EncryptionAuthError):
        decrypt(tampered, k)


def test_decrypt_with_tampered_iv_raises():
    k = generate_key()
    payload = encrypt(b"sensitive", k)
    tampered = EncryptedPayload(
        ciphertext=payload.ciphertext,
        iv=bytes(b ^ 0x01 for b in payload.iv),
        auth_tag=payload.auth_tag,
        key_id=payload.key_id,
    )
    with pytest.raises(EncryptionAuthError):
        decrypt(tampered, k)


def test_decrypt_with_tampered_auth_tag_raises():
    k = generate_key()
    payload = encrypt(b"sensitive", k)
    tampered = EncryptedPayload(
        ciphertext=payload.ciphertext,
        iv=payload.iv,
        auth_tag=bytes(b ^ 0x01 for b in payload.auth_tag),
        key_id=payload.key_id,
    )
    with pytest.raises(EncryptionAuthError):
        decrypt(tampered, k)


def test_decrypt_with_wrong_associated_data_raises():
    k = generate_key()
    payload = encrypt(b"x", k, associated_data=b"original-ad")
    with pytest.raises(EncryptionAuthError):
        decrypt(payload, k, associated_data=b"tampered-ad")


# -----------------------------------------------------------------------------
# Streaming
# -----------------------------------------------------------------------------


def test_streaming_roundtrip_small():
    k = generate_key()
    plaintext = b"A" * 128

    enc = StreamingEncryptor(k)
    ciphertext = enc.encrypt_chunk(plaintext)
    tag = enc.finalize()

    dec = StreamingDecryptor(k, enc.iv, tag)
    recovered = dec.decrypt_chunk(ciphertext)
    dec.finalize()

    assert recovered == plaintext


def test_streaming_roundtrip_chunked_large_payload():
    k = generate_key()
    plaintext = os.urandom(4 * 1024 * 1024)  # 4 MiB

    enc = StreamingEncryptor(k)
    iv = enc.iv
    ciphertext_chunks = []
    chunk_size = 64 * 1024
    for i in range(0, len(plaintext), chunk_size):
        ciphertext_chunks.append(
            enc.encrypt_chunk(plaintext[i : i + chunk_size])
        )
    tag = enc.finalize()

    dec = StreamingDecryptor(k, iv, tag)
    recovered = b"".join(dec.decrypt_chunk(c) for c in ciphertext_chunks)
    dec.finalize()

    assert recovered == plaintext


def test_streaming_decrypt_raises_on_tampered_final():
    k = generate_key()
    enc = StreamingEncryptor(k)
    ciphertext = enc.encrypt_chunk(b"payload")
    tag = enc.finalize()

    # Flip one tag bit.
    bad_tag = bytes(b ^ 0x01 for b in tag)

    dec = StreamingDecryptor(k, enc.iv, bad_tag)
    dec.decrypt_chunk(ciphertext)  # unauthenticated chunk
    with pytest.raises(EncryptionAuthError):
        dec.finalize()


def test_streaming_encryptor_rejects_double_finalize():
    k = generate_key()
    enc = StreamingEncryptor(k)
    enc.finalize()
    with pytest.raises(EncryptionError):
        enc.finalize()


def test_streaming_encryptor_rejects_post_finalize_chunk():
    k = generate_key()
    enc = StreamingEncryptor(k)
    enc.finalize()
    with pytest.raises(EncryptionError):
        enc.encrypt_chunk(b"late")
