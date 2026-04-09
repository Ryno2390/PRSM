"""
ContentStore wiring regression tests for node providers.

Tightens the v1.5.0 IPFS -> ContentStore migration for the three TODO
sites in prsm/node/storage_provider.py and prsm/node/content_provider.py:

  * StorageProvider._get_content_size must read ContentStore, not return 0.
  * StorageProvider.pin_content must mark pinned content with the real size.
  * StorageProvider.verify_pin must match ContentStore state.
  * ContentProvider._ipfs_cat must round-trip content through ContentStore.

These tests exercise the real ContentStore (no IPFS, no mocks).
"""

import pytest
import pytest_asyncio

from prsm.storage import (
    ContentHash,
    close_content_store,
    get_content_store,
    init_content_store,
)


@pytest_asyncio.fixture
async def content_store(tmp_path):
    """Provide an isolated ContentStore instance backed by a tmp dir."""
    # Guarantee a clean slate even if a previous test leaked the singleton.
    close_content_store()
    store = init_content_store(
        data_dir=str(tmp_path / "storage"),
        node_id="test-node-1",
    )
    try:
        yield store
    finally:
        close_content_store()


class TestContentStoreSizeLookup:
    """ContentStore exposes a public size_local() method for pinned content."""

    async def test_size_local_reports_total_size(self, content_store):
        payload = b"contentstore-integration-size-check" * 32
        expected_size = len(payload)

        content_hash = await content_store.store_local(payload)

        assert await content_store.size_local(content_hash) == expected_size

    async def test_size_local_returns_zero_for_missing(self, content_store):
        # A valid-format content hash (sha256 prefix 0x01) that was never stored.
        fake = ContentHash.from_hex("01" + "ff" * 32)
        assert await content_store.size_local(fake) == 0


class TestStorageProviderContentStoreWiring:
    """StorageProvider delegates pin/size/verify to ContentStore."""

    async def test_get_content_size_reads_contentstore(self, content_store):
        """_get_content_size() must return the real size from ContentStore."""
        from prsm.node.storage_provider import StorageProvider

        payload = b"storage-provider-size-check" * 16
        expected_size = len(payload)
        content_hash = await content_store.store_local(payload)
        cid_hex = content_hash.hex()

        # Construct a minimal StorageProvider that does not need network/
        # identity wiring. Only the _get_content_size path is exercised.
        provider = StorageProvider.__new__(StorageProvider)
        size = await provider._get_content_size(cid_hex)

        assert size == expected_size

    async def test_pin_content_records_real_size(self, content_store):
        """pin_content() must record the real content size, not 0."""
        from prsm.node.storage_provider import PinnedContent, StorageProvider

        payload = b"storage-provider-pin-check" * 16
        expected_size = len(payload)
        content_hash = await content_store.store_local(payload)
        cid_hex = content_hash.hex()

        provider = StorageProvider.__new__(StorageProvider)
        provider.ipfs_available = True
        provider.pinned_content = {}

        assert await provider.pin_content(cid_hex) is True
        pinned: PinnedContent = provider.pinned_content[cid_hex]
        assert pinned.size_bytes == expected_size

    async def test_verify_pin_matches_contentstore(self, content_store):
        """verify_pin() must reflect ContentStore presence."""
        from prsm.node.storage_provider import StorageProvider

        payload = b"storage-provider-verify-check"
        content_hash = await content_store.store_local(payload)
        cid_hex = content_hash.hex()

        provider = StorageProvider.__new__(StorageProvider)
        provider.ipfs_available = True

        assert await provider.verify_pin(cid_hex) is True
        # A valid-format hex digest that was never stored must not be found.
        assert await provider.verify_pin("01" + "ab" * 32) is False


class TestContentProviderContentStoreWiring:
    """ContentProvider retrieves content through the real ContentStore."""

    async def test_ipfs_cat_round_trips_through_contentstore(self, content_store):
        """_ipfs_cat() retrieves bytes stored via ContentStore.store_local."""
        from prsm.node.content_provider import ContentProvider

        payload = b"content-provider-roundtrip" * 8
        content_hash = await content_store.store_local(payload)
        cid_hex = content_hash.hex()

        provider = ContentProvider.__new__(ContentProvider)

        retrieved = await provider._ipfs_cat(cid_hex)
        assert retrieved == payload

    async def test_ipfs_cat_returns_none_for_missing(self, content_store):
        """_ipfs_cat() returns None when the content is absent."""
        from prsm.node.content_provider import ContentProvider

        provider = ContentProvider.__new__(ContentProvider)
        # Valid-format hex (sha256 prefix 0x01) for content never stored.
        retrieved = await provider._ipfs_cat("01" + "ab" * 32)
        assert retrieved is None

    async def test_ipfs_cat_returns_none_for_malformed_id(self, content_store):
        """_ipfs_cat() returns None instead of raising for malformed hex."""
        from prsm.node.content_provider import ContentProvider

        provider = ContentProvider.__new__(ContentProvider)
        # Unknown algorithm byte 0xff must not raise ValueError out of the method.
        retrieved = await provider._ipfs_cat("ff" * 32)
        assert retrieved is None
