"""
Tests for IPFS Content Sharding

This module tests the content sharding functionality for large files,
including splitting, uploading, downloading, and reassembling shards.
"""

import asyncio
import hashlib
import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from prsm.core.ipfs_sharding import (
    ShardingConfig,
    ShardInfo,
    ShardManifest,
    ShardIndex,
    ContentSharder,
    ShardingError,
    ShardVerificationError,
    ShardMissingError,
    ManifestError,
    calculate_optimal_shard_size,
    estimate_shard_count,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_sharding_config():
    """Default sharding configuration for tests."""
    return ShardingConfig(
        shard_size=1024,  # 1KB for faster tests
        max_shards=100,
        parallel_uploads=2,
        parallel_downloads=2,
        retry_attempts=2,
        auto_shard_threshold=512  # 512 bytes
    )


@pytest.fixture
def large_sharding_config():
    """Sharding configuration for larger content tests."""
    return ShardingConfig(
        shard_size=10 * 1024,  # 10KB
        max_shards=1000,
        parallel_uploads=3,
        parallel_downloads=5,
        retry_attempts=3,
        auto_shard_threshold=10 * 1024  # 10KB
    )


@pytest.fixture
def mock_ipfs_client():
    """Mock IPFS client for testing."""
    client = MagicMock()
    client.connected = True
    
    # Track uploaded content
    uploaded_content: Dict[str, bytes] = {}
    cid_counter = [0]
    
    def generate_cid():
        cid_counter[0] += 1
        return f"QmTestCID{cid_counter[0]:04d}"
    
    async def mock_upload_content(content, filename=None, pin=True, progress_callback=None):
        cid = generate_cid()
        if isinstance(content, bytes):
            uploaded_content[cid] = content
        else:
            uploaded_content[cid] = content.encode('utf-8') if isinstance(content, str) else content
        
        result = MagicMock()
        result.success = True
        result.cid = cid
        result.size = len(uploaded_content[cid])
        result.error = None
        result.metadata = {"uploaded": True}
        return result
    
    async def mock_download_content(cid, output_path=None, progress_callback=None, verify_integrity=True):
        result = MagicMock()
        if cid in uploaded_content:
            result.success = True
            result.cid = cid
            result.size = len(uploaded_content[cid])
            result.error = None
            result.metadata = {"content": uploaded_content[cid]}
        else:
            result.success = False
            result.cid = cid
            result.error = f"Content not found: {cid}"
            result.metadata = {}
        return result
    
    client.upload_content = AsyncMock(side_effect=mock_upload_content)
    client.download_content = AsyncMock(side_effect=mock_download_content)
    client.uploaded_content = uploaded_content
    
    return client


@pytest.fixture
def sample_content_small():
    """Small content that doesn't need sharding."""
    return b"Small content for testing"


@pytest.fixture
def sample_content_medium():
    """Medium content that will be sharded into a few pieces."""
    # Create content that's exactly 2KB
    return b"X" * 2048


@pytest.fixture
def sample_content_large():
    """Large content that will be sharded into many pieces."""
    # Create content that's 50KB
    return b"Y" * (50 * 1024)


# =============================================================================
# ShardingConfig Tests
# =============================================================================

class TestShardingConfig:
    """Tests for ShardingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ShardingConfig()
        
        assert config.shard_size == 10 * 1024 * 1024  # 10MB
        assert config.max_shards == 1000
        assert config.parallel_uploads == 3
        assert config.parallel_downloads == 5
        assert config.retry_attempts == 3
        assert config.auto_shard_threshold == 10 * 1024 * 1024
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ShardingConfig(
            shard_size=20 * 1024 * 1024,
            max_shards=500,
            parallel_uploads=5
        )
        
        assert config.shard_size == 20 * 1024 * 1024
        assert config.max_shards == 500
        assert config.parallel_uploads == 5
    
    def test_invalid_shard_size(self):
        """Test that invalid shard size raises error."""
        with pytest.raises(ValueError, match="shard_size must be positive"):
            ShardingConfig(shard_size=0)
        
        with pytest.raises(ValueError, match="shard_size must be positive"):
            ShardingConfig(shard_size=-100)
    
    def test_invalid_max_shards(self):
        """Test that invalid max_shards raises error."""
        with pytest.raises(ValueError, match="max_shards must be positive"):
            ShardingConfig(max_shards=0)
    
    def test_invalid_parallel_uploads(self):
        """Test that invalid parallel_uploads raises error."""
        with pytest.raises(ValueError, match="parallel_uploads must be positive"):
            ShardingConfig(parallel_uploads=0)
    
    def test_invalid_parallel_downloads(self):
        """Test that invalid parallel_downloads raises error."""
        with pytest.raises(ValueError, match="parallel_downloads must be positive"):
            ShardingConfig(parallel_downloads=-1)
    
    def test_invalid_retry_attempts(self):
        """Test that invalid retry_attempts raises error."""
        with pytest.raises(ValueError, match="retry_attempts must be non-negative"):
            ShardingConfig(retry_attempts=-1)


# =============================================================================
# ShardInfo Tests
# =============================================================================

class TestShardInfo:
    """Tests for ShardInfo dataclass."""
    
    def test_shard_info_creation(self):
        """Test creating a ShardInfo instance."""
        shard = ShardInfo(
            index=0,
            cid="QmTestCID0001",
            size=1024,
            hash="abc123def456"
        )
        
        assert shard.index == 0
        assert shard.cid == "QmTestCID0001"
        assert shard.size == 1024
        assert shard.hash == "abc123def456"
    
    def test_shard_info_to_dict(self):
        """Test converting ShardInfo to dictionary."""
        shard = ShardInfo(
            index=5,
            cid="QmTestCID0005",
            size=2048,
            hash="hash123"
        )
        
        data = shard.to_dict()
        
        assert data["index"] == 5
        assert data["cid"] == "QmTestCID0005"
        assert data["size"] == 2048
        assert data["hash"] == "hash123"
    
    def test_shard_info_from_dict(self):
        """Test creating ShardInfo from dictionary."""
        data = {
            "index": 10,
            "cid": "QmTestCID0010",
            "size": 4096,
            "hash": "hash456"
        }
        
        shard = ShardInfo.from_dict(data)
        
        assert shard.index == 10
        assert shard.cid == "QmTestCID0010"
        assert shard.size == 4096
        assert shard.hash == "hash456"
    
    def test_shard_info_roundtrip(self):
        """Test ShardInfo serialization roundtrip."""
        original = ShardInfo(
            index=3,
            cid="QmTestCID0003",
            size=512,
            hash="testhash"
        )
        
        data = original.to_dict()
        restored = ShardInfo.from_dict(data)
        
        assert restored.index == original.index
        assert restored.cid == original.cid
        assert restored.size == original.size
        assert restored.hash == original.hash


# =============================================================================
# ShardManifest Tests
# =============================================================================

class TestShardManifest:
    """Tests for ShardManifest dataclass."""
    
    def test_manifest_creation(self):
        """Test creating a ShardManifest instance."""
        shards = [
            ShardInfo(index=0, cid="cid1", size=1024, hash="h1"),
            ShardInfo(index=1, cid="cid2", size=1024, hash="h2"),
        ]
        
        manifest = ShardManifest(
            original_cid="QmOriginal",
            original_size=2048,
            original_hash="originalhash",
            shard_size=1024,
            shards=shards
        )
        
        assert manifest.original_cid == "QmOriginal"
        assert manifest.original_size == 2048
        assert manifest.original_hash == "originalhash"
        assert manifest.shard_size == 1024
        assert len(manifest.shards) == 2
        assert manifest.total_shards == 2
    
    def test_manifest_total_shards(self):
        """Test total_shards property."""
        manifest = ShardManifest()
        assert manifest.total_shards == 0
        
        manifest.shards = [
            ShardInfo(index=i, cid=f"cid{i}", size=100, hash=f"h{i}")
            for i in range(5)
        ]
        assert manifest.total_shards == 5
    
    def test_manifest_cid_property(self):
        """Test manifest_cid property."""
        manifest = ShardManifest()
        assert manifest.manifest_cid is None
        
        manifest.manifest_cid = "QmManifestCID"
        assert manifest.manifest_cid == "QmManifestCID"
        assert manifest.metadata["manifest_cid"] == "QmManifestCID"
    
    def test_manifest_to_dict(self):
        """Test converting ShardManifest to dictionary."""
        shards = [
            ShardInfo(index=0, cid="cid1", size=100, hash="h1"),
        ]
        
        manifest = ShardManifest(
            original_size=100,
            original_hash="hash",
            shard_size=100,
            shards=shards
        )
        
        data = manifest.to_dict()
        
        assert data["original_size"] == 100
        assert data["original_hash"] == "hash"
        assert data["shard_size"] == 100
        assert len(data["shards"]) == 1
        assert "created_at" in data
        assert data["manifest_version"] == "1.0"
    
    def test_manifest_from_dict(self):
        """Test creating ShardManifest from dictionary."""
        data = {
            "original_cid": "QmOrig",
            "original_size": 500,
            "original_hash": "orghash",
            "shard_size": 250,
            "shards": [
                {"index": 0, "cid": "cid1", "size": 250, "hash": "h1"},
                {"index": 1, "cid": "cid2", "size": 250, "hash": "h2"},
            ],
            "created_at": "2024-01-01T00:00:00",
            "manifest_version": "1.0",
            "metadata": {"key": "value"}
        }
        
        manifest = ShardManifest.from_dict(data)
        
        assert manifest.original_cid == "QmOrig"
        assert manifest.original_size == 500
        assert manifest.total_shards == 2
        assert manifest.shards[0].cid == "cid1"
        assert manifest.shards[1].cid == "cid2"
    
    def test_manifest_json_roundtrip(self):
        """Test ShardManifest JSON serialization roundtrip."""
        original = ShardManifest(
            original_cid="QmOrig",
            original_size=1024,
            original_hash="testhash",
            shard_size=512,
            shards=[
                ShardInfo(index=0, cid="cid1", size=512, hash="h1"),
                ShardInfo(index=1, cid="cid2", size=512, hash="h2"),
            ],
            metadata={"custom": "data"}
        )
        
        json_str = original.to_json()
        restored = ShardManifest.from_json(json_str)
        
        assert restored.original_cid == original.original_cid
        assert restored.original_size == original.original_size
        assert restored.original_hash == original.original_hash
        assert restored.shard_size == original.shard_size
        assert restored.total_shards == original.total_shards
        assert restored.metadata["custom"] == "data"


# =============================================================================
# ShardIndex Tests
# =============================================================================

class TestShardIndex:
    """Tests for ShardIndex class."""
    
    def test_index_creation(self):
        """Test creating a ShardIndex instance."""
        index = ShardIndex()
        
        assert len(index) == 0
        assert len(index.manifests) == 0
        assert len(index.content_to_manifest) == 0
        assert len(index.cid_to_manifest) == 0
    
    def test_add_manifest(self):
        """Test adding a manifest to the index."""
        index = ShardIndex()
        manifest = ShardManifest(
            original_cid="QmOriginal",
            original_size=100,
            original_hash="hash123",
            shards=[]
        )
        
        index.add_manifest(manifest, "QmManifestCID")
        
        assert len(index) == 1
        assert "QmManifestCID" in index
        assert index.content_to_manifest["hash123"] == "QmManifestCID"
        assert index.cid_to_manifest["QmOriginal"] == "QmManifestCID"
    
    def test_remove_manifest(self):
        """Test removing a manifest from the index."""
        index = ShardIndex()
        manifest = ShardManifest(
            original_cid="QmOriginal",
            original_size=100,
            original_hash="hash123",
            shards=[]
        )
        
        index.add_manifest(manifest, "QmManifestCID")
        assert len(index) == 1
        
        result = index.remove_manifest("QmManifestCID")
        
        assert result is True
        assert len(index) == 0
        assert "hash123" not in index.content_to_manifest
        assert "QmOriginal" not in index.cid_to_manifest
    
    def test_remove_nonexistent_manifest(self):
        """Test removing a manifest that doesn't exist."""
        index = ShardIndex()
        
        result = index.remove_manifest("nonexistent")
        
        assert result is False
    
    def test_find_manifest_by_manifest_cid(self):
        """Test finding a manifest by its CID."""
        index = ShardIndex()
        manifest = ShardManifest(
            original_size=100,
            original_hash="hash123",
            shards=[]
        )
        
        index.add_manifest(manifest, "QmManifestCID")
        
        found = index.find_manifest(manifest_cid="QmManifestCID")
        
        assert found is not None
        assert found.original_size == 100
    
    def test_find_manifest_by_content_hash(self):
        """Test finding a manifest by content hash."""
        index = ShardIndex()
        manifest = ShardManifest(
            original_size=200,
            original_hash="contenthash",
            shards=[]
        )
        
        index.add_manifest(manifest, "QmManifestCID")
        
        found = index.find_manifest(content_hash="contenthash")
        
        assert found is not None
        assert found.original_size == 200
    
    def test_find_manifest_by_original_cid(self):
        """Test finding a manifest by original CID."""
        index = ShardIndex()
        manifest = ShardManifest(
            original_cid="QmOriginalCID",
            original_size=300,
            original_hash="hash",
            shards=[]
        )
        
        index.add_manifest(manifest, "QmManifestCID")
        
        found = index.find_manifest(original_cid="QmOriginalCID")
        
        assert found is not None
        assert found.original_size == 300
    
    def test_find_manifest_not_found(self):
        """Test finding a manifest that doesn't exist."""
        index = ShardIndex()
        
        found = index.find_manifest(content_hash="nonexistent")
        
        assert found is None
    
    def test_get_all_manifests(self):
        """Test getting all manifests."""
        index = ShardIndex()
        
        for i in range(3):
            manifest = ShardManifest(
                original_size=100 * (i + 1),
                original_hash=f"hash{i}",
                shards=[]
            )
            index.add_manifest(manifest, f"QmManifest{i}")
        
        all_manifests = index.get_all_manifests()
        
        assert len(all_manifests) == 3
    
    def test_clear_index(self):
        """Test clearing the index."""
        index = ShardIndex()
        manifest = ShardManifest(
            original_size=100,
            original_hash="hash",
            shards=[]
        )
        
        index.add_manifest(manifest, "QmManifestCID")
        assert len(index) == 1
        
        index.clear()
        
        assert len(index) == 0
        assert len(index.manifests) == 0
        assert len(index.content_to_manifest) == 0


# =============================================================================
# ContentSharder Tests
# =============================================================================

class TestContentSharder:
    """Tests for ContentSharder class."""
    
    @pytest.mark.asyncio
    async def test_sharder_creation(self, mock_ipfs_client, default_sharding_config):
        """Test creating a ContentSharder instance."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        assert sharder.ipfs_client == mock_ipfs_client
        assert sharder.config == default_sharding_config
        assert sharder.index is not None
    
    @pytest.mark.asyncio
    async def test_calculate_hash(self, mock_ipfs_client, default_sharding_config):
        """Test content hash calculation."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        content = b"test content"
        expected_hash = hashlib.sha256(content).hexdigest()
        
        assert sharder._calculate_hash(content) == expected_hash
    
    @pytest.mark.asyncio
    async def test_split_content(self, mock_ipfs_client, default_sharding_config):
        """Test content splitting."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # Content that's exactly 2 shards
        content = b"X" * 2048
        shards = sharder._split_content(content)
        
        assert len(shards) == 2
        assert len(shards[0]) == 1024
        assert len(shards[1]) == 1024
        assert b"".join(shards) == content
    
    @pytest.mark.asyncio
    async def test_split_content_partial_shard(self, mock_ipfs_client, default_sharding_config):
        """Test content splitting with partial last shard."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # Content that's 1.5 shards
        content = b"X" * 1536
        shards = sharder._split_content(content)
        
        assert len(shards) == 2
        assert len(shards[0]) == 1024
        assert len(shards[1]) == 512
        assert b"".join(shards) == content
    
    @pytest.mark.asyncio
    async def test_split_content_exceeds_max_shards(self, mock_ipfs_client):
        """Test that exceeding max_shards raises error."""
        config = ShardingConfig(
            shard_size=100,
            max_shards=2
        )
        sharder = ContentSharder(mock_ipfs_client, config)
        
        # Content that would need 3 shards
        content = b"X" * 300
        
        with pytest.raises(ShardingError, match="exceeding maximum"):
            sharder._split_content(content)
    
    @pytest.mark.asyncio
    async def test_shard_content_small(self, mock_ipfs_client, default_sharding_config, sample_content_small):
        """Test sharding small content (single shard)."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        manifest, manifest_cid = await sharder.shard_content(sample_content_small)
        
        assert manifest is not None
        assert manifest.original_size == len(sample_content_small)
        assert manifest.total_shards == 1
        assert manifest.original_hash == hashlib.sha256(sample_content_small).hexdigest()
        assert manifest_cid is not None
    
    @pytest.mark.asyncio
    async def test_shard_content_medium(self, mock_ipfs_client, default_sharding_config, sample_content_medium):
        """Test sharding medium content (multiple shards)."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        manifest, manifest_cid = await sharder.shard_content(sample_content_medium)
        
        assert manifest is not None
        assert manifest.original_size == len(sample_content_medium)
        assert manifest.total_shards == 2  # 2048 bytes / 1024 shard_size
        assert manifest_cid is not None
        
        # Verify all shards were uploaded
        for shard in manifest.shards:
            assert shard.cid in mock_ipfs_client.uploaded_content
    
    @pytest.mark.asyncio
    async def test_shard_content_large(self, mock_ipfs_client, large_sharding_config, sample_content_large):
        """Test sharding large content."""
        sharder = ContentSharder(mock_ipfs_client, large_sharding_config)
        
        manifest, manifest_cid = await sharder.shard_content(sample_content_large)
        
        assert manifest is not None
        assert manifest.original_size == len(sample_content_large)
        assert manifest.total_shards == 5  # 50KB / 10KB shard_size
        assert manifest_cid is not None
    
    @pytest.mark.asyncio
    async def test_shard_content_empty(self, mock_ipfs_client, default_sharding_config):
        """Test that sharding empty content raises error."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        with pytest.raises(ShardingError, match="Cannot shard empty content"):
            await sharder.shard_content(b"")
    
    @pytest.mark.asyncio
    async def test_shard_content_with_metadata(self, mock_ipfs_client, default_sharding_config, sample_content_medium):
        """Test sharding content with custom metadata."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        custom_metadata = {
            "filename": "test.bin",
            "content_type": "application/octet-stream"
        }
        
        manifest, _ = await sharder.shard_content(
            content=sample_content_medium,
            metadata=custom_metadata
        )
        
        assert manifest.metadata["filename"] == "test.bin"
        assert manifest.metadata["content_type"] == "application/octet-stream"
    
    @pytest.mark.asyncio
    async def test_shard_content_with_progress_callback(self, mock_ipfs_client, default_sharding_config, sample_content_medium):
        """Test sharding content with progress callback."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        progress_calls = []
        
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        await sharder.shard_content(
            content=sample_content_medium,
            progress_callback=progress_callback
        )
        
        # Should have progress updates
        assert len(progress_calls) > 0
    
    @pytest.mark.asyncio
    async def test_reassemble_content(self, mock_ipfs_client, default_sharding_config, sample_content_medium):
        """Test reassembling sharded content."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # First shard the content
        manifest, _ = await sharder.shard_content(sample_content_medium)
        
        # Then reassemble
        reassembled = await sharder.reassemble_content(manifest)
        
        assert reassembled == sample_content_medium
    
    @pytest.mark.asyncio
    async def test_reassemble_content_with_verification(self, mock_ipfs_client, default_sharding_config, sample_content_medium):
        """Test reassembling with hash verification."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        manifest, _ = await sharder.shard_content(sample_content_medium)
        
        reassembled = await sharder.reassemble_content(manifest, verify=True)
        
        assert reassembled == sample_content_medium
    
    @pytest.mark.asyncio
    async def test_reassemble_content_empty_manifest(self, mock_ipfs_client, default_sharding_config):
        """Test that reassembling with empty manifest raises error."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        manifest = ShardManifest()  # Empty manifest
        
        with pytest.raises(ManifestError, match="has no shards"):
            await sharder.reassemble_content(manifest)
    
    @pytest.mark.asyncio
    async def test_get_shard_manifest(self, mock_ipfs_client, default_sharding_config, sample_content_medium):
        """Test loading a shard manifest from IPFS."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # First create and store a manifest
        original_manifest, manifest_cid = await sharder.shard_content(sample_content_medium)
        
        # Clear the index to force a download
        sharder.index.clear()
        
        # Load the manifest
        loaded_manifest = await sharder.get_shard_manifest(manifest_cid)
        
        assert loaded_manifest is not None
        assert loaded_manifest.original_size == original_manifest.original_size
        assert loaded_manifest.original_hash == original_manifest.original_hash
        assert loaded_manifest.total_shards == original_manifest.total_shards
    
    @pytest.mark.asyncio
    async def test_should_shard(self, mock_ipfs_client, default_sharding_config):
        """Test should_shard method."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # Below threshold
        assert not sharder.should_shard(100)
        assert not sharder.should_shard(512)
        
        # Above threshold
        assert sharder.should_shard(513)
        assert sharder.should_shard(1024)
    
    @pytest.mark.asyncio
    async def test_roundtrip_integration(self, mock_ipfs_client, default_sharding_config, sample_content_large):
        """Test full roundtrip: shard -> store -> load -> reassemble."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # Shard and store
        manifest, manifest_cid = await sharder.shard_content(sample_content_large)
        
        # Clear index to simulate fresh load
        sharder.index.clear()
        
        # Load manifest
        loaded_manifest = await sharder.get_shard_manifest(manifest_cid)
        
        # Reassemble
        reassembled = await sharder.reassemble_content(loaded_manifest)
        
        assert reassembled == sample_content_large


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestShardingErrorHandling:
    """Tests for error handling in sharding operations."""
    
    @pytest.mark.asyncio
    async def test_upload_failure(self, mock_ipfs_client, default_sharding_config, sample_content_medium):
        """Test handling of upload failures."""
        # Make uploads fail
        mock_ipfs_client.upload_content = AsyncMock(
            return_value=MagicMock(success=False, error="Upload failed")
        )
        
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        with pytest.raises(ShardingError, match="Failed to upload shard"):
            await sharder.shard_content(sample_content_medium)
    
    @pytest.mark.asyncio
    async def test_download_missing_shard(self, mock_ipfs_client, default_sharding_config, sample_content_medium):
        """Test handling of missing shard during download."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # Shard content
        manifest, _ = await sharder.shard_content(sample_content_medium)
        
        # Clear uploaded content to simulate missing shard
        mock_ipfs_client.uploaded_content.clear()
        
        with pytest.raises(ShardMissingError):
            await sharder.reassemble_content(manifest)
    
    @pytest.mark.asyncio
    async def test_hash_verification_failure(self, mock_ipfs_client, default_sharding_config, sample_content_medium):
        """Test handling of hash verification failure."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # Shard content
        manifest, _ = await sharder.shard_content(sample_content_medium)
        
        # Corrupt one of the shards
        first_shard_cid = manifest.shards[0].cid
        mock_ipfs_client.uploaded_content[first_shard_cid] = b"CORRUPTED"
        
        with pytest.raises(ShardVerificationError, match="hash mismatch"):
            await sharder.reassemble_content(manifest, verify=True)
    
    @pytest.mark.asyncio
    async def test_manifest_parse_error(self, mock_ipfs_client, default_sharding_config):
        """Test handling of manifest parsing errors."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # Store invalid JSON as a manifest
        mock_ipfs_client.uploaded_content["QmInvalidManifest"] = b"not valid json"
        
        with pytest.raises(ManifestError, match="Failed to parse manifest"):
            await sharder.get_shard_manifest("QmInvalidManifest")


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_calculate_optimal_shard_size(self):
        """Test optimal shard size calculation."""
        # 100KB content, target 10 shards - use custom bounds for this test
        size = calculate_optimal_shard_size(
            100 * 1024,
            target_shards=10,
            min_shard_size=1 * 1024,  # 1KB min
            max_shard_size=100 * 1024 * 1024  # 100MB max
        )
        assert size == 10 * 1024  # 10KB per shard
        
        # Very small content - should hit minimum
        size = calculate_optimal_shard_size(
            100,
            target_shards=10,
            min_shard_size=1 * 1024,
            max_shard_size=100 * 1024 * 1024
        )
        assert size == 1 * 1024  # Minimum 1KB
        
        # Very large content - should hit maximum
        size = calculate_optimal_shard_size(
            10 * 1024 * 1024 * 1024,  # 10GB
            target_shards=10,
            min_shard_size=1 * 1024,
            max_shard_size=100 * 1024 * 1024
        )
        assert size == 100 * 1024 * 1024  # Maximum 100MB
    
    def test_calculate_optimal_shard_size_bounds(self):
        """Test that optimal shard size respects bounds."""
        # Test minimum bound
        size = calculate_optimal_shard_size(
            100,
            target_shards=10,
            min_shard_size=512,
            max_shard_size=1024
        )
        assert size == 512  # Should hit minimum
        
        # Test maximum bound
        size = calculate_optimal_shard_size(
            10 * 1024 * 1024,
            target_shards=10,
            min_shard_size=1024,
            max_shard_size=100 * 1024
        )
        assert size == 100 * 1024  # Should hit maximum
    
    def test_estimate_shard_count(self):
        """Test shard count estimation."""
        # Exact multiple
        count = estimate_shard_count(10 * 1024, 1024)
        assert count == 10
        
        # Partial last shard
        count = estimate_shard_count(10 * 1024 + 1, 1024)
        assert count == 11
        
        # Smaller than shard size
        count = estimate_shard_count(100, 1024)
        assert count == 1
    
    def test_estimate_shard_count_invalid(self):
        """Test that invalid shard size raises error."""
        with pytest.raises(ValueError, match="shard_size must be positive"):
            estimate_shard_count(1024, 0)
        
        with pytest.raises(ValueError, match="shard_size must be positive"):
            estimate_shard_count(1024, -1)


# =============================================================================
# Parallel Operations Tests
# =============================================================================

class TestParallelOperations:
    """Tests for parallel upload/download operations."""
    
    @pytest.mark.asyncio
    async def test_parallel_uploads(self, mock_ipfs_client, large_sharding_config, sample_content_large):
        """Test that shards are uploaded in parallel."""
        sharder = ContentSharder(mock_ipfs_client, large_sharding_config)
        
        upload_times = []
        original_upload = mock_ipfs_client.upload_content
        
        async def tracked_upload(*args, **kwargs):
            import time
            start = time.time()
            result = await original_upload(*args, **kwargs)
            upload_times.append(time.time() - start)
            return result
        
        mock_ipfs_client.upload_content = AsyncMock(side_effect=tracked_upload)
        
        manifest, _ = await sharder.shard_content(sample_content_large)
        
        # All shards should have been uploaded
        assert manifest.total_shards > 0
    
    @pytest.mark.asyncio
    async def test_parallel_downloads(self, mock_ipfs_client, large_sharding_config, sample_content_large):
        """Test that shards are downloaded in parallel."""
        sharder = ContentSharder(mock_ipfs_client, large_sharding_config)
        
        # First shard the content
        manifest, _ = await sharder.shard_content(sample_content_large)
        
        # Then reassemble (which downloads in parallel)
        reassembled = await sharder.reassemble_content(manifest)
        
        assert reassembled == sample_content_large
    
    @pytest.mark.asyncio
    async def test_concurrency_limit(self, mock_ipfs_client):
        """Test that concurrency is limited to configured value."""
        config = ShardingConfig(
            shard_size=100,
            parallel_uploads=2,  # Only 2 parallel uploads
            max_shards=100
        )
        sharder = ContentSharder(mock_ipfs_client, config)
        
        # Content that needs 5 shards
        content = b"X" * 500
        
        manifest, _ = await sharder.shard_content(content)
        
        # All shards should be uploaded despite limited concurrency
        assert manifest.total_shards == 5


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_content_exactly_shard_size(self, mock_ipfs_client, default_sharding_config):
        """Test content that's exactly one shard size."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # Content exactly 1KB
        content = b"X" * 1024
        
        manifest, _ = await sharder.shard_content(content)
        
        assert manifest.total_shards == 1
        assert manifest.shards[0].size == 1024
    
    @pytest.mark.asyncio
    async def test_content_one_byte_over_shard_size(self, mock_ipfs_client, default_sharding_config):
        """Test content that's one byte over one shard size."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # Content 1KB + 1 byte
        content = b"X" * 1025
        
        manifest, _ = await sharder.shard_content(content)
        
        assert manifest.total_shards == 2
        assert manifest.shards[0].size == 1024
        assert manifest.shards[1].size == 1
    
    @pytest.mark.asyncio
    async def test_single_byte_content(self, mock_ipfs_client, default_sharding_config):
        """Test content that's a single byte."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        content = b"X"
        
        manifest, _ = await sharder.shard_content(content)
        
        assert manifest.total_shards == 1
        assert manifest.shards[0].size == 1
    
    @pytest.mark.asyncio
    async def test_binary_content(self, mock_ipfs_client, default_sharding_config):
        """Test binary content with all byte values."""
        sharder = ContentSharder(mock_ipfs_client, default_sharding_config)
        
        # All possible byte values
        content = bytes(range(256)) * 4  # 1024 bytes
        
        manifest, _ = await sharder.shard_content(content)
        
        assert manifest.total_shards == 1
        
        reassembled = await sharder.reassemble_content(manifest)
        assert reassembled == content
    
    @pytest.mark.asyncio
    async def test_zero_threshold(self, mock_ipfs_client):
        """Test that zero threshold disables auto-sharding."""
        config = ShardingConfig(
            shard_size=1024,
            auto_shard_threshold=0  # Disabled
        )
        sharder = ContentSharder(mock_ipfs_client, config)
        
        # Large content
        content = b"X" * 100 * 1024
        
        # Should not trigger auto-sharding
        assert not sharder.should_shard(len(content))
