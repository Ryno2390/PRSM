"""
Unit tests for prsm.storage.models and prsm.storage.exceptions.

Test matrix
-----------
TestContentHash
    - from_data produces correct SHA-256 hash
    - hex() is 66 characters and starts with "01" for SHA-256
    - from_hex() round-trips perfectly
    - from_hex() with unknown algorithm byte raises ValueError
    - identical data deduplicates (same hash)
    - different data yields different hash
    - equality is based on algorithm_id + digest
    - ContentHash instances are hashable and usable as dict keys
    - str(content_hash) returns the hex representation

TestShardManifest
    - basic construction
    - to_json() / from_json() round-trip
    - from_json() with invalid JSON raises ManifestError

TestReplicationPolicy
    - default field values
    - custom values are preserved

TestKeyShare
    - construction with explicit fields

TestExceptions
    - ContentNotFoundError stores content_hash attribute
    - ShardIntegrityError stores expected/actual attributes
    - PlacementError stores reason and min_nodes_needed
    - All exceptions are subclasses of StorageError
"""

from __future__ import annotations

import hashlib
import json
import time

import pytest

from prsm.storage.exceptions import (
    ContentNotFoundError,
    KeyReconstructionError,
    ManifestError,
    PlacementError,
    ShardIntegrityError,
    StorageError,
)
from prsm.storage.models import (
    AlgorithmID,
    ContentHash,
    KeyShare,
    ReplicationPolicy,
    ShardManifest,
)


# ============================================================
# Helpers
# ============================================================

def _sample_manifest(content_hash: ContentHash | None = None) -> ShardManifest:
    """Return a fully-populated ShardManifest for use in multiple tests."""
    if content_hash is None:
        content_hash = ContentHash.from_data(b"hello world")
    shard_data = [b"shard0", b"shard1", b"shard2"]
    shard_hashes = [ContentHash.from_data(sd) for sd in shard_data]
    return ShardManifest(
        content_hash=content_hash,
        shard_hashes=shard_hashes,
        total_size=len(b"hello world"),
        shard_size=4,
        algorithm_id=AlgorithmID.SHA256,
        created_at=1_700_000_000.0,
        replication_factor=3,
        owner_node_id="node-abc",
        visibility="public",
    )


# ============================================================
# TestContentHash
# ============================================================

class TestContentHash:
    def test_from_data_sha256_digest(self):
        """from_data should produce the same digest as hashlib.sha256."""
        data = b"PRSM native storage"
        ch = ContentHash.from_data(data)
        expected_digest = hashlib.sha256(data).digest()
        assert ch.digest == expected_digest
        assert ch.algorithm_id == AlgorithmID.SHA256

    def test_hex_length_and_prefix_sha256(self):
        """SHA-256 hex should be 66 characters and start with '01'."""
        ch = ContentHash.from_data(b"test data")
        h = ch.hex()
        assert len(h) == 66
        assert h[:2] == "01"

    def test_from_hex_roundtrip(self):
        """from_hex(ch.hex()) should reproduce the original ContentHash."""
        original = ContentHash.from_data(b"round-trip test")
        restored = ContentHash.from_hex(original.hex())
        assert restored == original
        assert restored.algorithm_id == original.algorithm_id
        assert restored.digest == original.digest

    def test_from_hex_unknown_algorithm_raises(self):
        """from_hex with an unregistered algorithm byte must raise ValueError."""
        # Prefix byte 0xFF is not in AlgorithmID
        bad_hex = "ff" + "a" * 64
        with pytest.raises(ValueError, match="Unknown algorithm"):
            ContentHash.from_hex(bad_hex)

    def test_from_hex_too_short_raises(self):
        """from_hex with a string shorter than 4 chars must raise ValueError."""
        with pytest.raises(ValueError):
            ContentHash.from_hex("01")

    def test_same_content_same_hash(self):
        """Hashing the same bytes twice must yield equal ContentHash objects."""
        data = b"deterministic"
        assert ContentHash.from_data(data) == ContentHash.from_data(data)

    def test_different_content_different_hash(self):
        """Different byte strings must produce different ContentHash objects."""
        assert ContentHash.from_data(b"foo") != ContentHash.from_data(b"bar")

    def test_equality(self):
        """ContentHash equality is determined by algorithm_id and digest."""
        ch1 = ContentHash.from_data(b"eq test")
        ch2 = ContentHash.from_data(b"eq test")
        assert ch1 == ch2

    def test_not_equal_to_non_content_hash(self):
        """Comparing ContentHash to an unrelated type should return NotImplemented."""
        ch = ContentHash.from_data(b"x")
        assert ch != "not a content hash"
        assert ch != 42

    def test_hashable_for_dict_key(self):
        """ContentHash must be usable as a dict key and in sets."""
        ch1 = ContentHash.from_data(b"key1")
        ch2 = ContentHash.from_data(b"key2")
        d = {ch1: "value1", ch2: "value2"}
        assert d[ch1] == "value1"
        assert d[ch2] == "value2"
        # Dedup in a set
        s = {ContentHash.from_data(b"key1"), ContentHash.from_data(b"key1")}
        assert len(s) == 1

    def test_str_returns_hex(self):
        """str(content_hash) must equal content_hash.hex()."""
        ch = ContentHash.from_data(b"string repr")
        assert str(ch) == ch.hex()

    def test_frozen_immutable(self):
        """ContentHash is a frozen dataclass; attribute assignment must fail."""
        ch = ContentHash.from_data(b"immutable")
        with pytest.raises((AttributeError, TypeError)):
            ch.digest = b"tampered"  # type: ignore[misc]


# ============================================================
# TestShardManifest
# ============================================================

class TestShardManifest:
    def test_basic_construction(self):
        """ShardManifest can be constructed with all required fields."""
        m = _sample_manifest()
        assert m.owner_node_id == "node-abc"
        assert m.replication_factor == 3
        assert len(m.shard_hashes) == 3
        assert m.visibility == "public"

    def test_to_json_from_json_roundtrip(self):
        """to_json() / from_json() must reproduce the original manifest."""
        original = _sample_manifest()
        json_str = original.to_json()

        # Must be valid JSON
        raw = json.loads(json_str)
        assert isinstance(raw, dict)

        restored = ShardManifest.from_json(json_str)
        assert restored.content_hash == original.content_hash
        assert restored.shard_hashes == original.shard_hashes
        assert restored.total_size == original.total_size
        assert restored.shard_size == original.shard_size
        assert restored.algorithm_id == original.algorithm_id
        assert restored.created_at == original.created_at
        assert restored.replication_factor == original.replication_factor
        assert restored.owner_node_id == original.owner_node_id
        assert restored.visibility == original.visibility

    def test_from_json_invalid_raises_manifest_error(self):
        """from_json with non-JSON input must raise ManifestError."""
        with pytest.raises(ManifestError):
            ShardManifest.from_json("{not valid json[[[")

    def test_default_visibility(self):
        """Visibility defaults to 'public' when omitted from JSON."""
        m = _sample_manifest()
        raw = json.loads(m.to_json())
        del raw["visibility"]  # remove so from_json uses default
        restored = ShardManifest.from_json(json.dumps(raw))
        assert restored.visibility == "public"


# ============================================================
# TestReplicationPolicy
# ============================================================

class TestReplicationPolicy:
    def test_default_values(self):
        """Default fields match the spec."""
        p = ReplicationPolicy(replication_factor=3)
        assert p.min_asn_diversity == 2
        assert p.owner_excluded is True
        assert p.key_shard_separation is True
        assert p.degraded_constraints == []

    def test_custom_values(self):
        """Custom field values are preserved."""
        p = ReplicationPolicy(
            replication_factor=5,
            min_asn_diversity=3,
            owner_excluded=False,
            key_shard_separation=False,
            degraded_constraints=["allow_same_asn"],
        )
        assert p.replication_factor == 5
        assert p.min_asn_diversity == 3
        assert p.owner_excluded is False
        assert p.key_shard_separation is False
        assert p.degraded_constraints == ["allow_same_asn"]

    def test_degraded_constraints_independent(self):
        """Each instance gets its own degraded_constraints list."""
        p1 = ReplicationPolicy(replication_factor=3)
        p2 = ReplicationPolicy(replication_factor=3)
        p1.degraded_constraints.append("x")
        assert p2.degraded_constraints == []


# ============================================================
# TestKeyShare
# ============================================================

class TestKeyShare:
    def test_basic_construction(self):
        """KeyShare stores all provided fields correctly."""
        ch = ContentHash.from_data(b"encrypted content")
        ks = KeyShare(
            content_hash=ch,
            share_index=1,
            share_data=b"\xde\xad\xbe\xef",
            threshold=3,
            total_shares=5,
            algorithm_id=0x01,  # AES-256-GCM
        )
        assert ks.content_hash == ch
        assert ks.share_index == 1
        assert ks.share_data == b"\xde\xad\xbe\xef"
        assert ks.threshold == 3
        assert ks.total_shares == 5
        assert ks.algorithm_id == 0x01


# ============================================================
# TestExceptions
# ============================================================

class TestExceptions:
    def test_storage_error_is_base_exception(self):
        with pytest.raises(StorageError):
            raise StorageError("base")

    def test_content_not_found_stores_hash(self):
        exc = ContentNotFoundError("01abc123")
        assert exc.content_hash == "01abc123"
        assert "01abc123" in str(exc)

    def test_content_not_found_is_storage_error(self):
        with pytest.raises(StorageError):
            raise ContentNotFoundError("01abc")

    def test_shard_integrity_error_stores_expected_actual(self):
        exc = ShardIntegrityError("expected_hex", "actual_hex")
        assert exc.expected == "expected_hex"
        assert exc.actual == "actual_hex"
        assert "expected_hex" in str(exc)
        assert "actual_hex" in str(exc)

    def test_shard_integrity_is_storage_error(self):
        with pytest.raises(StorageError):
            raise ShardIntegrityError("e", "a")

    def test_manifest_error_is_storage_error(self):
        with pytest.raises(StorageError):
            raise ManifestError("bad manifest")

    def test_key_reconstruction_error_is_storage_error(self):
        with pytest.raises(StorageError):
            raise KeyReconstructionError("not enough shares")

    def test_placement_error_stores_reason_and_min_nodes(self):
        exc = PlacementError("not enough peers", 5)
        assert exc.reason == "not enough peers"
        assert exc.min_nodes_needed == 5
        assert "not enough peers" in str(exc)
        assert "5" in str(exc)

    def test_placement_error_is_storage_error(self):
        with pytest.raises(StorageError):
            raise PlacementError("reason", 3)
