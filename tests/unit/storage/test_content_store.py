"""
Unit tests for prsm.storage.content_store.ContentStore.

Test matrix
-----------
TestContentStoreLocal
    1. store_and_retrieve_small        (below shard threshold)
    2. store_and_retrieve_large        (above shard threshold — exercises sharding)
    3. content_hash_returned           (returned hash matches ContentHash.from_data)
    4. exists_local_true               (exists_local returns True after store)
    5. exists_local_false              (exists_local returns False before store)
    6. delete_local                    (shards and manifest removed; exists False after)
    7. retrieve_nonexistent_raises     (ContentNotFoundError for unknown hash)
    8. manifest_encrypted_recoverable  (_store_and_encrypt -> decrypt -> manifest roundtrip)
    9. full_local_pipeline             (500-byte store -> retrieve, content hash verified)
"""

from __future__ import annotations

import pathlib

import pytest

from prsm.storage.content_store import ContentStore, _select_threshold_params
from prsm.storage.exceptions import ContentNotFoundError
from prsm.storage.models import ContentHash, ShardManifest


# ---------------------------------------------------------------------------
# Low thresholds so multi-shard paths are exercised with small test data.
# ---------------------------------------------------------------------------
SHARD_THRESHOLD = 100
SHARD_SIZE = 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path: pathlib.Path) -> ContentStore:
    """ContentStore backed by a fresh temporary directory, low shard thresholds."""
    return ContentStore(
        data_dir=str(tmp_path / "content_store"),
        node_id="test-node",
        shard_threshold=SHARD_THRESHOLD,
        shard_size=SHARD_SIZE,
    )


def _make_data(n: int) -> bytes:
    """Return *n* deterministic bytes."""
    return bytes(i % 256 for i in range(n))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestContentStoreLocal:

    # ------------------------------------------------------------------
    # 1. store_and_retrieve_small
    # ------------------------------------------------------------------
    async def test_store_and_retrieve_small(self, store: ContentStore) -> None:
        """Content below the shard threshold is stored as a single shard and retrieved correctly."""
        data = b"hello, PRSM"  # 11 bytes < SHARD_THRESHOLD (100)
        content_hash = await store.store_local(data, replication_factor=1)
        retrieved = await store.retrieve_local(content_hash)
        assert retrieved == data

    # ------------------------------------------------------------------
    # 2. store_and_retrieve_large
    # ------------------------------------------------------------------
    async def test_store_and_retrieve_large(self, store: ContentStore) -> None:
        """Content above the shard threshold is split into multiple shards and reassembled."""
        data = _make_data(300)  # 300 bytes >> SHARD_THRESHOLD (100); yields 6 shards
        content_hash = await store.store_local(data, replication_factor=3)
        retrieved = await store.retrieve_local(content_hash)
        assert retrieved == data

    # ------------------------------------------------------------------
    # 3. content_hash_returned
    # ------------------------------------------------------------------
    async def test_content_hash_returned(self, store: ContentStore) -> None:
        """Hash returned by store_local must equal ContentHash.from_data(original)."""
        data = _make_data(50)
        content_hash = await store.store_local(data)
        expected = ContentHash.from_data(data)
        assert content_hash == expected

    # ------------------------------------------------------------------
    # 4. exists_local — true branch
    # ------------------------------------------------------------------
    async def test_exists_local_true(self, store: ContentStore) -> None:
        """exists_local returns True for a hash that has been stored."""
        data = b"presence check"
        content_hash = await store.store_local(data)
        assert await store.exists_local(content_hash) is True

    # ------------------------------------------------------------------
    # 5. exists_local — false branch
    # ------------------------------------------------------------------
    async def test_exists_local_false(self, store: ContentStore) -> None:
        """exists_local returns False for a hash that has never been stored."""
        unknown_hash = ContentHash.from_data(b"not stored")
        assert await store.exists_local(unknown_hash) is False

    # ------------------------------------------------------------------
    # 6. delete_local
    # ------------------------------------------------------------------
    async def test_delete_local(self, store: ContentStore) -> None:
        """After delete_local, exists_local returns False and retrieve raises."""
        data = _make_data(200)
        content_hash = await store.store_local(data)
        assert await store.exists_local(content_hash)

        await store.delete_local(content_hash)

        assert not await store.exists_local(content_hash)
        with pytest.raises(ContentNotFoundError):
            await store.retrieve_local(content_hash)

    # ------------------------------------------------------------------
    # 7. retrieve_nonexistent_raises ContentNotFoundError
    # ------------------------------------------------------------------
    async def test_retrieve_nonexistent_raises(self, store: ContentStore) -> None:
        """retrieve_local raises ContentNotFoundError for an unknown content hash."""
        ghost_hash = ContentHash.from_data(b"ghost content")
        with pytest.raises(ContentNotFoundError):
            await store.retrieve_local(ghost_hash)

    # ------------------------------------------------------------------
    # 8. manifest_encrypted_and_recoverable
    # ------------------------------------------------------------------
    async def test_manifest_encrypted_and_recoverable(self, store: ContentStore) -> None:
        """_store_and_encrypt returns an encrypted manifest that can be decrypted
        using the provided key_shares, and the recovered manifest round-trips correctly."""
        data = _make_data(150)  # above threshold -> multiple shards
        content_hash, manifest, ciphertext, key_shares = await store._store_and_encrypt(
            data,
            owner_node_id="test-node",
            replication_factor=3,
        )

        # ciphertext must be non-empty and differ from the plaintext manifest JSON.
        assert isinstance(ciphertext, bytes)
        assert len(ciphertext) > 0
        manifest_json = manifest.to_json().encode("utf-8")
        assert ciphertext != manifest_json

        # Decrypt using the key_shares and verify manifest round-trips.
        recovered_manifest_bytes = store.key_manager.decrypt_manifest(ciphertext, key_shares)
        recovered_manifest = ShardManifest.from_json(recovered_manifest_bytes.decode("utf-8"))

        assert recovered_manifest.content_hash == manifest.content_hash
        assert recovered_manifest.shard_hashes == manifest.shard_hashes
        assert recovered_manifest.total_size == manifest.total_size
        assert recovered_manifest.owner_node_id == "test-node"
        assert recovered_manifest.replication_factor == 3

    # ------------------------------------------------------------------
    # 9. full_local_pipeline
    # ------------------------------------------------------------------
    async def test_full_local_pipeline(self, store: ContentStore) -> None:
        """500-byte end-to-end: store, retrieve, and verify content hash matches."""
        data = _make_data(500)
        expected_hash = ContentHash.from_data(data)

        stored_hash = await store.store_local(data, replication_factor=3)
        assert stored_hash == expected_hash

        retrieved = await store.retrieve_local(stored_hash)
        assert retrieved == data
        assert ContentHash.from_data(retrieved) == expected_hash


# ---------------------------------------------------------------------------
# Threshold selection helper tests
# ---------------------------------------------------------------------------

class TestSelectThresholdParams:
    """Unit tests for the _select_threshold_params helper."""

    def test_fewer_than_10_shards(self) -> None:
        """<10 shards -> 3-of-5."""
        assert _select_threshold_params(1) == (3, 5)
        assert _select_threshold_params(9) == (3, 5)

    def test_10_to_99_shards(self) -> None:
        """10-99 shards -> 5-of-8."""
        assert _select_threshold_params(10) == (5, 8)
        assert _select_threshold_params(99) == (5, 8)

    def test_100_plus_shards(self) -> None:
        """100+ shards -> 7-of-12."""
        assert _select_threshold_params(100) == (7, 12)
        assert _select_threshold_params(1000) == (7, 12)


# ---------------------------------------------------------------------------
# Delete idempotency / edge cases
# ---------------------------------------------------------------------------

class TestDeleteEdgeCases:

    async def test_delete_nonexistent_is_noop(self, store: ContentStore) -> None:
        """delete_local on an unknown hash must not raise."""
        ghost = ContentHash.from_data(b"never stored")
        await store.delete_local(ghost)  # should not raise

    async def test_double_store_idempotent(self, store: ContentStore) -> None:
        """Storing the same data twice must succeed and return the same hash."""
        data = b"duplicate content"
        h1 = await store.store_local(data)
        h2 = await store.store_local(data)
        assert h1 == h2
        retrieved = await store.retrieve_local(h1)
        assert retrieved == data
