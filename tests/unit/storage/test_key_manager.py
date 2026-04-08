"""Tests for KeyManager — AES-256-GCM encryption and Shamir's Secret Sharing."""

from __future__ import annotations

import itertools
import os

import pytest

from prsm.storage.exceptions import KeyReconstructionError
from prsm.storage.key_manager import KeyManager
from prsm.storage.models import ContentHash, KeyShare


@pytest.fixture
def km() -> KeyManager:
    return KeyManager()


# -----------------------------------------------------------------------
# TestAESEncryption
# -----------------------------------------------------------------------

class TestAESEncryption:
    """AES-256-GCM encryption tests."""

    def test_encrypt_decrypt_roundtrip(self, km: KeyManager) -> None:
        key = km.generate_key()
        plaintext = b"hello world"
        ciphertext = km.encrypt(key, plaintext)
        assert km.decrypt(key, ciphertext) == plaintext

    def test_wrong_key_fails(self, km: KeyManager) -> None:
        key = km.generate_key()
        wrong_key = km.generate_key()
        ciphertext = km.encrypt(key, b"secret")
        with pytest.raises(Exception):
            km.decrypt(wrong_key, ciphertext)

    def test_ciphertext_includes_nonce(self, km: KeyManager) -> None:
        key = km.generate_key()
        plaintext = b"same input"
        ct1 = km.encrypt(key, plaintext)
        ct2 = km.encrypt(key, plaintext)
        # Random nonce means different ciphertexts each time
        assert ct1 != ct2
        # Both decrypt to the same plaintext
        assert km.decrypt(key, ct1) == plaintext
        assert km.decrypt(key, ct2) == plaintext

    def test_empty_plaintext(self, km: KeyManager) -> None:
        key = km.generate_key()
        ciphertext = km.encrypt(key, b"")
        assert km.decrypt(key, ciphertext) == b""

    def test_large_plaintext(self, km: KeyManager) -> None:
        key = km.generate_key()
        plaintext = os.urandom(100 * 1024)  # 100 KB
        ciphertext = km.encrypt(key, plaintext)
        assert km.decrypt(key, ciphertext) == plaintext


# -----------------------------------------------------------------------
# TestShamirSecretSharing
# -----------------------------------------------------------------------

class TestShamirSecretSharing:
    """Shamir's Secret Sharing over GF(256) tests."""

    def test_split_produces_n_shares(self, km: KeyManager) -> None:
        secret = os.urandom(32)
        shares = km.split_secret(secret, threshold=3, num_shares=5)
        assert len(shares) == 5
        # share_index is 1-based
        indices = [idx for idx, _ in shares]
        assert indices == [1, 2, 3, 4, 5]
        # Each share has same length as secret
        for _, data in shares:
            assert len(data) == len(secret)

    def test_threshold_shares_reconstruct(self, km: KeyManager) -> None:
        secret = b"a 32-byte secret for testing!..."
        assert len(secret) == 32
        shares = km.split_secret(secret, threshold=3, num_shares=5)
        # Exactly K=3 shares
        recovered = km.reconstruct_secret(shares[:3], threshold=3)
        assert recovered == secret

    def test_more_than_threshold_shares_reconstruct(self, km: KeyManager) -> None:
        secret = os.urandom(32)
        shares = km.split_secret(secret, threshold=3, num_shares=5)
        # 4 shares (more than threshold)
        recovered = km.reconstruct_secret(shares[:4], threshold=3)
        assert recovered == secret

    def test_fewer_than_threshold_fails(self, km: KeyManager) -> None:
        secret = os.urandom(32)
        shares = km.split_secret(secret, threshold=3, num_shares=5)
        # Only 2 shares when threshold=3 => wrong result, NOT an error
        recovered = km.reconstruct_secret(shares[:2], threshold=3)
        assert recovered != secret

    def test_any_k_of_n_combination(self, km: KeyManager) -> None:
        secret = os.urandom(32)
        shares = km.split_secret(secret, threshold=3, num_shares=5)
        # All C(5,3) = 10 combinations must reconstruct correctly
        for combo in itertools.combinations(shares, 3):
            recovered = km.reconstruct_secret(list(combo), threshold=3)
            assert recovered == secret, f"Failed for indices {[s[0] for s in combo]}"

    def test_different_threshold_sizes(self, km: KeyManager) -> None:
        secret = os.urandom(16)
        for threshold, num_shares in [(2, 3), (3, 5), (5, 8), (7, 12)]:
            shares = km.split_secret(secret, threshold=threshold, num_shares=num_shares)
            assert len(shares) == num_shares
            # Use first threshold shares
            recovered = km.reconstruct_secret(shares[:threshold], threshold=threshold)
            assert recovered == secret, f"Failed for ({threshold},{num_shares})"

    def test_single_byte_secret(self, km: KeyManager) -> None:
        secret = b"\x42"
        shares = km.split_secret(secret, threshold=2, num_shares=3)
        recovered = km.reconstruct_secret(shares[:2], threshold=2)
        assert recovered == secret


# -----------------------------------------------------------------------
# TestKeyManagerIntegration
# -----------------------------------------------------------------------

class TestKeyManagerIntegration:
    """Full pipeline integration tests."""

    def test_generate_encrypt_split_reconstruct_decrypt(self, km: KeyManager) -> None:
        # Generate key
        key = km.generate_key()
        assert len(key) == 32

        # Encrypt
        plaintext = b"integration test payload"
        ciphertext = km.encrypt(key, plaintext)

        # Split key
        shares = km.split_secret(key, threshold=3, num_shares=5)
        assert len(shares) == 5

        # Reconstruct key from 3 shares
        recovered_key = km.reconstruct_secret(shares[:3], threshold=3)
        assert recovered_key == key

        # Decrypt
        result = km.decrypt(recovered_key, ciphertext)
        assert result == plaintext

    def test_create_key_shares_for_content(self, km: KeyManager) -> None:
        manifest_data = b'{"content_hash": "01abcdef", "shards": []}'
        content_hash = ContentHash.from_data(manifest_data)

        ciphertext, key_shares = km.encrypt_manifest(
            manifest_data, content_hash, threshold=3, num_shares=5,
        )

        # Verify KeyShare structure
        assert len(key_shares) == 5
        for i, ks in enumerate(key_shares):
            assert isinstance(ks, KeyShare)
            assert ks.content_hash == content_hash
            assert ks.share_index == i + 1
            assert ks.threshold == 3
            assert ks.total_shares == 5
            assert ks.algorithm_id == 0x01  # AES-256-GCM

        # Decrypt with key shares
        recovered = km.decrypt_manifest(ciphertext, key_shares[:3])
        assert recovered == manifest_data

    def test_key_refresh(self, km: KeyManager) -> None:
        # Encrypt
        key = km.generate_key()
        plaintext = b"key refresh test"
        ciphertext = km.encrypt(key, plaintext)

        # Split
        shares = km.split_secret(key, threshold=3, num_shares=5)

        # Reconstruct
        recovered_key = km.reconstruct_secret(shares[:3], threshold=3)
        assert recovered_key == key

        # Re-split the same key with different parameters
        new_shares = km.split_secret(recovered_key, threshold=2, num_shares=4)
        assert len(new_shares) == 4

        # Reconstruct again from new shares
        recovered_key_2 = km.reconstruct_secret(new_shares[:2], threshold=2)
        assert recovered_key_2 == key

        # Decrypt with the re-reconstructed key
        result = km.decrypt(recovered_key_2, ciphertext)
        assert result == plaintext
