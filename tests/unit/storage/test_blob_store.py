"""
Unit tests for prsm.storage.blob_store.BlobStore.

Test matrix
-----------
TestBlobStore
    - store_and_retrieve_roundtrip
    - content_hash_is_sha256
    - deduplication_single_file (same data stored twice = one file on disk)
    - hash_prefix_directory_created
    - exists_true
    - exists_false
    - delete_removes_file
    - retrieve_nonexistent_raises_ContentNotFoundError
    - delete_nonexistent_is_noop
    - large_content (1 MB)
    - empty_content
    - data_dir_created_on_first_store (nested dir that doesn't exist yet)
"""

from __future__ import annotations

import os
import pathlib

import pytest

from prsm.storage.blob_store import BlobStore
from prsm.storage.exceptions import ContentNotFoundError
from prsm.storage.models import AlgorithmID, ContentHash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: pathlib.Path) -> BlobStore:
    """BlobStore backed by a fresh temporary directory."""
    return BlobStore(str(tmp_path / "blobs"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBlobStore:
    async def test_store_and_retrieve_roundtrip(self, store: BlobStore) -> None:
        data = b"hello, PRSM blob store!"
        content_hash = await store.store(data)
        retrieved = await store.retrieve(content_hash)
        assert retrieved == data

    async def test_content_hash_is_sha256(self, store: BlobStore) -> None:
        data = b"deterministic content"
        content_hash = await store.store(data)
        expected = ContentHash.from_data(data, AlgorithmID.SHA256)
        assert content_hash == expected
        assert content_hash.algorithm_id == AlgorithmID.SHA256

    async def test_deduplication_single_file(self, store: BlobStore) -> None:
        """Storing the same data twice must not create duplicate files."""
        data = b"deduplicate me"
        h1 = await store.store(data)
        h2 = await store.store(data)
        assert h1 == h2
        path = store._path_for(h1)
        # Only one file exists (the original)
        assert os.path.isfile(path)
        # Parent directory has exactly one entry
        parent = os.path.dirname(path)
        entries = os.listdir(parent)
        assert len(entries) == 1

    async def test_hash_prefix_directory_created(self, store: BlobStore) -> None:
        data = b"directory structure test"
        content_hash = await store.store(data)
        path = store._path_for(content_hash)
        # Bucket prefix is the first 2 hex chars of the DIGEST (not the algo-prefixed hex)
        digest_hex = content_hash.digest.hex()
        expected_prefix_dir = os.path.join(store.data_dir, digest_hex[:2])
        assert os.path.isdir(expected_prefix_dir)
        assert os.path.isfile(path)

    async def test_exists_true(self, store: BlobStore) -> None:
        data = b"exists check"
        content_hash = await store.store(data)
        assert await store.exists(content_hash) is True

    async def test_exists_false(self, store: BlobStore) -> None:
        content_hash = ContentHash.from_data(b"not stored", AlgorithmID.SHA256)
        assert await store.exists(content_hash) is False

    async def test_delete_removes_file(self, store: BlobStore) -> None:
        data = b"delete me"
        content_hash = await store.store(data)
        assert await store.exists(content_hash) is True
        await store.delete(content_hash)
        assert await store.exists(content_hash) is False
        assert not os.path.exists(store._path_for(content_hash))

    async def test_retrieve_nonexistent_raises_content_not_found_error(
        self, store: BlobStore
    ) -> None:
        content_hash = ContentHash.from_data(b"ghost data", AlgorithmID.SHA256)
        with pytest.raises(ContentNotFoundError) as exc_info:
            await store.retrieve(content_hash)
        assert content_hash.hex() in str(exc_info.value)

    async def test_delete_nonexistent_is_noop(self, store: BlobStore) -> None:
        content_hash = ContentHash.from_data(b"never stored", AlgorithmID.SHA256)
        # Should complete without raising
        await store.delete(content_hash)

    async def test_large_content(self, store: BlobStore) -> None:
        data = os.urandom(1024 * 1024)  # 1 MB
        content_hash = await store.store(data)
        retrieved = await store.retrieve(content_hash)
        assert retrieved == data

    async def test_empty_content(self, store: BlobStore) -> None:
        data = b""
        content_hash = await store.store(data)
        retrieved = await store.retrieve(content_hash)
        assert retrieved == data
        assert content_hash == ContentHash.from_data(b"", AlgorithmID.SHA256)

    async def test_data_dir_created_on_first_store(self, tmp_path: pathlib.Path) -> None:
        """BlobStore should create deeply-nested data_dir on first store."""
        nested_dir = str(tmp_path / "a" / "b" / "c" / "blobs")
        assert not os.path.exists(nested_dir)
        store = BlobStore(nested_dir)
        data = b"nested directory creation"
        content_hash = await store.store(data)
        assert os.path.isdir(nested_dir)
        retrieved = await store.retrieve(content_hash)
        assert retrieved == data
