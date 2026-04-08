"""
Integration tests for the PRSM native storage module.

Exercises the full storage pipeline end-to-end, crossing multiple subsystems:
  BlobStore <-> ShardEngine <-> KeyManager <-> ContentStore <-> DistributionManager

Test classes
------------
TestFullLifecycle
    1. small_content_roundtrip          — store small content, retrieve, verify
    2. large_content_sharded_roundtrip  — store 500 bytes, retrieve, verify hash
    3. sharding_produces_correct_count  — 275 bytes / 50-byte shards = 6 shards
    4. tampered_shard_detected          — corrupt shard file -> ShardIntegrityError

TestEncryptedManifestPipeline
    5. manifest_survives_encrypt_decrypt        — full encrypt/decrypt round-trip
    6. any_k_shares_decrypt_manifest            — all C(threshold, num_shares) combos

TestContentHashAlgorithmAgility
    7. sha256_prefix                     — hex starts with "01", len 66
    8. roundtrip_preserves_algorithm     — from_data -> from_hex -> algorithm preserved

TestDescriptorSigning
    9. owner_sign_and_verify             — Ed25519 sign, verify, tamper fails
"""

from __future__ import annotations

import itertools
import math
import os

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from prsm.storage import ContentStore, ContentHash
from prsm.storage.blob_store import BlobStore
from prsm.storage.shard_engine import ShardEngine
from prsm.storage.key_manager import KeyManager
from prsm.storage.distribution import DistributionManager
from prsm.storage.models import ShardManifest, AlgorithmID, ReplicationPolicy
from prsm.storage.exceptions import ShardIntegrityError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeDiscovery:
    """Stub peer-discovery that returns no peers."""

    def find_peers_by_capability(self, *a, **kw):
        return []


def _make_store(tmp_path, node_id="node-A"):
    """Create a ContentStore with low shard thresholds for testing."""
    return ContentStore(
        data_dir=str(tmp_path / "storage"),
        node_id=node_id,
        shard_threshold=100,
        shard_size=50,
    )


# ---------------------------------------------------------------------------
# TestFullLifecycle
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    """End-to-end store/retrieve pipeline tests."""

    # ------------------------------------------------------------------
    # 1. small_content_roundtrip
    # ------------------------------------------------------------------
    async def test_small_content_roundtrip(self, tmp_path):
        """Content below shard threshold is stored and retrieved correctly."""
        store = _make_store(tmp_path)
        data = os.urandom(40)  # well below 100-byte threshold

        content_hash = await store.store_local(data)
        retrieved = await store.retrieve_local(content_hash)

        assert retrieved == data

    # ------------------------------------------------------------------
    # 2. large_content_sharded_roundtrip
    # ------------------------------------------------------------------
    async def test_large_content_sharded_roundtrip(self, tmp_path):
        """500 bytes (above threshold) is sharded, retrieved, and hash verified."""
        store = _make_store(tmp_path)
        data = os.urandom(500)
        expected_hash = ContentHash.from_data(data)

        stored_hash = await store.store_local(data, replication_factor=3)
        assert stored_hash == expected_hash

        retrieved = await store.retrieve_local(stored_hash)
        assert retrieved == data
        assert ContentHash.from_data(retrieved) == expected_hash

    # ------------------------------------------------------------------
    # 3. sharding_produces_correct_shard_count
    # ------------------------------------------------------------------
    async def test_sharding_produces_correct_shard_count(self, tmp_path):
        """275 bytes / 50-byte shards = ceil(275/50) = 6 shards in the manifest."""
        store = _make_store(tmp_path)
        data = os.urandom(275)

        content_hash, manifest, _ciphertext, _key_shares = await store._store_and_encrypt(
            data,
            owner_node_id="node-A",
            replication_factor=1,
        )

        expected_shards = math.ceil(275 / 50)  # = 6
        assert len(manifest.shard_hashes) == expected_shards

    # ------------------------------------------------------------------
    # 4. tampered_shard_detected
    # ------------------------------------------------------------------
    async def test_tampered_shard_detected(self, tmp_path):
        """Corrupting a shard file on disk causes retrieve_local to raise ShardIntegrityError."""
        store = _make_store(tmp_path)
        data = os.urandom(200)  # above threshold -> multiple shards

        content_hash = await store.store_local(data)

        # Access the manifest to find the first shard's disk path.
        cache_entry = store._manifest_cache[content_hash.hex()]
        _ciphertext, _key_shares, manifest = cache_entry
        first_shard_hash = manifest.shard_hashes[0]

        shard_path = store.blob_store._path_for(first_shard_hash)
        assert os.path.exists(shard_path), "Shard file must exist before tampering"

        # Overwrite with garbage bytes.
        with open(shard_path, "wb") as fh:
            fh.write(b"\xff" * 50)

        with pytest.raises(ShardIntegrityError):
            await store.retrieve_local(content_hash)


# ---------------------------------------------------------------------------
# TestEncryptedManifestPipeline
# ---------------------------------------------------------------------------

class TestEncryptedManifestPipeline:
    """Tests that exercise the manifest encryption/decryption pipeline."""

    # ------------------------------------------------------------------
    # 5. manifest_survives_encrypt_decrypt
    # ------------------------------------------------------------------
    async def test_manifest_survives_encrypt_decrypt(self, tmp_path):
        """Full round-trip: store -> _store_and_encrypt -> decrypt -> reassemble."""
        store = _make_store(tmp_path)
        data = os.urandom(300)

        content_hash, manifest, ciphertext, key_shares = await store._store_and_encrypt(
            data,
            owner_node_id="node-A",
            replication_factor=3,
        )

        # ciphertext must differ from plaintext manifest JSON.
        assert isinstance(ciphertext, bytes) and len(ciphertext) > 0
        assert ciphertext != manifest.to_json().encode("utf-8")

        # Decrypt with ALL key shares (threshold satisfied).
        recovered_bytes = store.key_manager.decrypt_manifest(ciphertext, key_shares)
        recovered_manifest = ShardManifest.from_json(recovered_bytes.decode("utf-8"))

        assert recovered_manifest.content_hash == manifest.content_hash
        assert recovered_manifest.shard_hashes == manifest.shard_hashes
        assert recovered_manifest.total_size == manifest.total_size
        assert recovered_manifest.owner_node_id == "node-A"

        # Reassemble from recovered manifest and verify data matches.
        reassembled = await store.shard_engine.reassemble(recovered_manifest)
        assert reassembled == data

    # ------------------------------------------------------------------
    # 6. any_k_shares_decrypt_manifest
    # ------------------------------------------------------------------
    async def test_any_k_shares_decrypt_manifest(self, tmp_path):
        """All C(threshold, num_shares) subsets of key shares decrypt the manifest correctly."""
        store = _make_store(tmp_path)
        data = os.urandom(150)  # above threshold -> few shards -> 3-of-5 policy

        content_hash, manifest, ciphertext, key_shares = await store._store_and_encrypt(
            data,
            owner_node_id="node-A",
            replication_factor=1,
        )

        # Policy for < 10 shards is 3-of-5.
        threshold = key_shares[0].threshold
        total_shares = key_shares[0].total_shares
        assert total_shares == len(key_shares)

        original_json = manifest.to_json()

        # Test every combination of exactly `threshold` shares.
        for combo in itertools.combinations(key_shares, threshold):
            combo_shares = list(combo)
            recovered_bytes = store.key_manager.decrypt_manifest(ciphertext, combo_shares)
            recovered_manifest = ShardManifest.from_json(recovered_bytes.decode("utf-8"))
            assert recovered_manifest.content_hash == manifest.content_hash, (
                f"Combination of shares {[s.share_index for s in combo_shares]} "
                f"failed to decrypt correctly"
            )


# ---------------------------------------------------------------------------
# TestContentHashAlgorithmAgility
# ---------------------------------------------------------------------------

class TestContentHashAlgorithmAgility:
    """Tests for ContentHash algorithm-agility properties."""

    # ------------------------------------------------------------------
    # 7. sha256_prefix
    # ------------------------------------------------------------------
    def test_sha256_prefix(self):
        """ContentHash.from_data hex starts with '01' and is 66 characters long."""
        data = os.urandom(64)
        ch = ContentHash.from_data(data, AlgorithmID.SHA256)
        hex_str = ch.hex()

        assert hex_str.startswith("01"), f"Expected prefix '01', got {hex_str[:2]!r}"
        assert len(hex_str) == 66, f"Expected 66 chars, got {len(hex_str)}"

    # ------------------------------------------------------------------
    # 8. roundtrip_preserves_algorithm
    # ------------------------------------------------------------------
    def test_roundtrip_preserves_algorithm(self):
        """from_data -> hex -> from_hex preserves algorithm_id."""
        data = os.urandom(32)
        original = ContentHash.from_data(data, AlgorithmID.SHA256)
        roundtripped = ContentHash.from_hex(original.hex())

        assert roundtripped.algorithm_id == original.algorithm_id
        assert roundtripped.digest == original.digest
        assert roundtripped == original


# ---------------------------------------------------------------------------
# TestDescriptorSigning
# ---------------------------------------------------------------------------

class TestDescriptorSigning:
    """Tests for Ed25519-based ContentDescriptor signing via DistributionManager."""

    # ------------------------------------------------------------------
    # 9. owner_sign_and_verify
    # ------------------------------------------------------------------
    async def test_owner_sign_and_verify(self, tmp_path):
        """Sign a descriptor with an Ed25519 private key, verify with public key; tamper fails."""
        blob_store = BlobStore(data_dir=str(tmp_path / "blobs"))
        key_manager = KeyManager()
        discovery = FakeDiscovery()

        dm = DistributionManager(
            node_id="node-A",
            discovery=discovery,
            transport=None,
            blob_store=blob_store,
            key_manager=key_manager,
        )

        # Generate a real Ed25519 keypair.
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        contract_pubkey = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

        # Build a descriptor stub.
        content_hash = ContentHash.from_data(os.urandom(64))
        replication_policy = ReplicationPolicy(replication_factor=3)
        descriptor = dm._create_descriptor_stub(
            content_hash=content_hash,
            owner_node_id="node-A",
            visibility="public",
            replication_policy=replication_policy,
            contract_pubkey=contract_pubkey,
        )

        # Sign with the private key.
        signed = dm._sign_descriptor(descriptor, private_key, signer_type="owner")
        assert signed.signature != b"", "Signature must be non-empty after signing"
        assert signed.signer_type == "owner"

        # Verify with the matching public key — must succeed.
        assert dm._verify_descriptor_signature(signed, public_key) is True

        # Tamper: flip a byte in the signature.
        tampered_sig = bytearray(signed.signature)
        tampered_sig[0] ^= 0xFF
        signed.signature = bytes(tampered_sig)

        assert dm._verify_descriptor_signature(signed, public_key) is False
