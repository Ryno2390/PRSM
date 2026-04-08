"""
Unit tests for prsm.storage.shard_engine.ShardEngine.

Test matrix
-----------
TestShardEngine
    - small_content_single_shard        (5 bytes -> 1 shard)
    - large_content_multiple_shards     (200 bytes -> 4 shards)
    - shard_count_ceil                  (130 bytes / 50 = 3 shards via ceil)
    - reassemble_produces_original      (200 bytes roundtrip)
    - reassemble_small_content          (small roundtrip)
    - content_hash_matches              (manifest.content_hash == ContentHash.from_data(data))
    - each_shard_stored_in_blob_store   (blob_store.exists for each shard_hash)
    - tampered_shard_detected           (corrupt first shard -> ShardIntegrityError)
    - manifest_json_roundtrip           (split -> to_json -> from_json -> hashes match)
    - manifest_metadata                 (owner_node_id, replication_factor, algorithm_id,
                                         visibility, created_at)
    - empty_content                     (b"" -> 1 shard, reassembles to b"")
"""

from __future__ import annotations

import math
import pathlib

import pytest

from prsm.storage.blob_store import BlobStore
from prsm.storage.exceptions import ShardIntegrityError
from prsm.storage.models import AlgorithmID, ContentHash, ShardManifest
from prsm.storage.shard_engine import ShardEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Use low thresholds so tests stay fast and exercise multi-shard paths easily.
SHARD_THRESHOLD = 100
SHARD_SIZE = 50


@pytest.fixture
def blob_store(tmp_path: pathlib.Path) -> BlobStore:
    """BlobStore backed by a fresh temporary directory."""
    return BlobStore(str(tmp_path / "blobs"))


@pytest.fixture
def engine(blob_store: BlobStore) -> ShardEngine:
    """ShardEngine with low thresholds for easy multi-shard testing."""
    return ShardEngine(
        blob_store=blob_store,
        shard_threshold=SHARD_THRESHOLD,
        shard_size=SHARD_SIZE,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n: int) -> bytes:
    """Return *n* bytes of deterministic content."""
    return bytes(i % 256 for i in range(n))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShardEngine:

    # ------------------------------------------------------------------
    # split() — shard count
    # ------------------------------------------------------------------

    async def test_small_content_single_shard(self, engine: ShardEngine) -> None:
        """Content <= threshold must be stored as a single shard."""
        data = b"hello"  # 5 bytes < 100
        manifest = await engine.split(data, owner_node_id="node-1", replication_factor=1)
        assert len(manifest.shard_hashes) == 1

    async def test_large_content_multiple_shards(self, engine: ShardEngine) -> None:
        """200 bytes > 100-byte threshold; with shard_size=50 -> 4 shards."""
        data = _make_data(200)
        manifest = await engine.split(data, owner_node_id="node-1", replication_factor=3)
        expected_shards = math.ceil(len(data) / SHARD_SIZE)  # ceil(200/50) = 4
        assert len(manifest.shard_hashes) == expected_shards

    async def test_shard_count_ceil(self, engine: ShardEngine) -> None:
        """130 bytes / 50 bytes per shard = 2.6, ceil = 3 shards."""
        data = _make_data(130)
        manifest = await engine.split(data, owner_node_id="node-1", replication_factor=2)
        expected_shards = math.ceil(len(data) / SHARD_SIZE)  # ceil(130/50) = 3
        assert len(manifest.shard_hashes) == expected_shards

    # ------------------------------------------------------------------
    # reassemble() — correctness
    # ------------------------------------------------------------------

    async def test_reassemble_produces_original(self, engine: ShardEngine) -> None:
        """200-byte roundtrip: split then reassemble must return original bytes."""
        data = _make_data(200)
        manifest = await engine.split(data, owner_node_id="node-1", replication_factor=3)
        result = await engine.reassemble(manifest)
        assert result == data

    async def test_reassemble_small_content(self, engine: ShardEngine) -> None:
        """Small content (single shard) roundtrip."""
        data = b"short"
        manifest = await engine.split(data, owner_node_id="node-a", replication_factor=1)
        result = await engine.reassemble(manifest)
        assert result == data

    # ------------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------------

    async def test_content_hash_matches(self, engine: ShardEngine) -> None:
        """manifest.content_hash must equal ContentHash.from_data(original)."""
        data = _make_data(200)
        manifest = await engine.split(data, owner_node_id="node-1", replication_factor=3)
        expected = ContentHash.from_data(data)
        assert manifest.content_hash == expected

    async def test_each_shard_stored_in_blob_store(
        self, engine: ShardEngine, blob_store: BlobStore
    ) -> None:
        """Every shard_hash in the manifest must be present in the blob store."""
        data = _make_data(200)
        manifest = await engine.split(data, owner_node_id="node-1", replication_factor=2)
        for shard_hash in manifest.shard_hashes:
            assert await blob_store.exists(shard_hash), (
                f"Shard {shard_hash.hex()} not found in blob store"
            )

    async def test_tampered_shard_detected(
        self, engine: ShardEngine, blob_store: BlobStore
    ) -> None:
        """Corrupting a shard file on disk must cause reassemble to raise ShardIntegrityError."""
        data = _make_data(200)
        manifest = await engine.split(data, owner_node_id="node-1", replication_factor=2)

        # Corrupt the first shard by writing garbage bytes directly to its file.
        first_hash = manifest.shard_hashes[0]
        shard_path = blob_store._path_for(first_hash)
        with open(shard_path, "wb") as f:
            f.write(b"\xff" * 50)

        with pytest.raises(ShardIntegrityError):
            await engine.reassemble(manifest)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    async def test_manifest_json_roundtrip(self, engine: ShardEngine) -> None:
        """split -> to_json -> from_json -> content_hash and shard_hashes preserved."""
        data = _make_data(200)
        manifest = await engine.split(data, owner_node_id="node-1", replication_factor=3)
        json_str = manifest.to_json()
        restored = ShardManifest.from_json(json_str)

        assert restored.content_hash == manifest.content_hash
        assert restored.shard_hashes == manifest.shard_hashes
        assert restored.total_size == manifest.total_size
        assert restored.shard_size == manifest.shard_size

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    async def test_manifest_metadata(self, engine: ShardEngine) -> None:
        """Verify owner_node_id, replication_factor, algorithm_id, visibility, created_at."""
        data = _make_data(50)
        manifest = await engine.split(
            data,
            owner_node_id="node-xyz",
            replication_factor=5,
            visibility="private",
        )
        assert manifest.owner_node_id == "node-xyz"
        assert manifest.replication_factor == 5
        assert manifest.algorithm_id == AlgorithmID.SHA256
        assert manifest.visibility == "private"
        assert isinstance(manifest.created_at, float)
        assert manifest.created_at > 0.0

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    async def test_empty_content(self, engine: ShardEngine) -> None:
        """Empty bytes -> 1 shard, reassembles back to b""."""
        data = b""
        manifest = await engine.split(data, owner_node_id="node-1", replication_factor=1)
        assert len(manifest.shard_hashes) == 1
        result = await engine.reassemble(manifest)
        assert result == b""
