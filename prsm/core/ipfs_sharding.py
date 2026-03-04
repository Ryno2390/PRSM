"""
PRSM IPFS Content Sharding - Large File Support

🎯 PURPOSE IN PRSM:
This module provides content sharding capabilities for IPFS, enabling efficient
storage and retrieval of large files by splitting them into manageable chunks
that can be uploaded, downloaded, and verified in parallel.

🔧 INTEGRATION POINTS:
- IPFS Client: Automatic sharding for large file uploads
- Content Provider: Parallel shard retrieval across nodes
- Shard Index: Track and discover sharded content

🚀 KEY FEATURES:
- Configurable shard sizes (default 10MB)
- Parallel upload/download of shards
- SHA-256 hash verification for each shard
- Automatic reassembly of original content
- Manifest-based tracking for integrity
- Memory-efficient streaming for large files
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class ShardingError(Exception):
    """Base exception for sharding operations."""
    pass


class ShardVerificationError(ShardingError):
    """Exception raised when shard hash verification fails."""
    pass


class ShardMissingError(ShardingError):
    """Exception raised when a required shard is missing."""
    pass


class ManifestError(ShardingError):
    """Exception raised for manifest-related issues."""
    pass


# =============================================================================
# Configuration and Data Classes
# =============================================================================

@dataclass
class ShardingConfig:
    """
    Configuration for content sharding.
    
    This dataclass provides configurable settings for the sharding system,
    allowing tuning based on network conditions and file sizes.
    
    Attributes:
        shard_size: Size of each shard in bytes (default: 10MB)
        max_shards: Maximum number of shards per file
        parallel_uploads: Number of parallel upload workers
        parallel_downloads: Number of parallel download workers
        retry_attempts: Retry attempts for failed operations
        auto_shard_threshold: File size threshold for automatic sharding
    
    Example:
        config = ShardingConfig(
            shard_size=20 * 1024 * 1024,  # 20MB shards
            parallel_uploads=5
        )
    """
    shard_size: int = 10 * 1024 * 1024  # 10MB default shard size
    max_shards: int = 1000  # Maximum number of shards per file
    parallel_uploads: int = 3  # Number of parallel upload workers
    parallel_downloads: int = 5  # Number of parallel download workers
    retry_attempts: int = 3  # Retry attempts for failed operations
    auto_shard_threshold: int = 10 * 1024 * 1024  # Auto-shard files > 10MB
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if self.max_shards <= 0:
            raise ValueError("max_shards must be positive")
        if self.parallel_uploads <= 0:
            raise ValueError("parallel_uploads must be positive")
        if self.parallel_downloads <= 0:
            raise ValueError("parallel_downloads must be positive")
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")


@dataclass
class ShardInfo:
    """
    Information about a single shard.
    
    This dataclass holds metadata about an individual shard,
    including its position, IPFS identifier, size, and hash.
    
    Attributes:
        index: Shard index (0-based)
        cid: IPFS CID of the shard
        size: Actual size of this shard in bytes
        hash: SHA-256 hash of shard content
    
    Example:
        shard = ShardInfo(
            index=0,
            cid="QmXxx...",
            size=10 * 1024 * 1024,
            hash="abc123..."
        )
    """
    index: int  # Shard index (0-based)
    cid: str  # IPFS CID of the shard
    size: int  # Actual size of this shard
    hash: str  # SHA-256 hash of shard content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "index": self.index,
            "cid": self.cid,
            "size": self.size,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShardInfo':
        """Create from dictionary."""
        return cls(
            index=data["index"],
            cid=data["cid"],
            size=data["size"],
            hash=data["hash"]
        )


@dataclass
class ShardManifest:
    """
    Manifest for a sharded file.
    
    This dataclass holds comprehensive metadata about a sharded file,
    including information about the original file and all its shards.
    The manifest itself can be stored in IPFS for discovery.
    
    Attributes:
        original_cid: CID of the original file (if known, before sharding)
        original_size: Original file size in bytes
        original_hash: SHA-256 hash of original content
        shard_size: Configured size of each shard
        shards: List of shard information
        created_at: When the manifest was created
        manifest_version: Version of the manifest format
        metadata: Additional metadata dictionary
    
    Example:
        manifest = ShardManifest(
            original_size=100 * 1024 * 1024,
            original_hash="def456...",
            shard_size=10 * 1024 * 1024,
            shards=[shard1, shard2, ...]
        )
    """
    original_cid: Optional[str] = None  # CID of the original file (if known)
    original_size: int = 0  # Original file size in bytes
    original_hash: str = ""  # SHA-256 hash of original content
    shard_size: int = 10 * 1024 * 1024  # Size of each shard
    shards: List[ShardInfo] = field(default_factory=list)  # List of shard information
    created_at: datetime = field(default_factory=datetime.utcnow)  # When the manifest was created
    manifest_version: str = "1.0"  # Version of the manifest format
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    @property
    def total_shards(self) -> int:
        """Get total number of shards."""
        return len(self.shards)
    
    @property
    def manifest_cid(self) -> Optional[str]:
        """Get the manifest CID if stored."""
        return self.metadata.get("manifest_cid")
    
    @manifest_cid.setter
    def manifest_cid(self, value: str):
        """Set the manifest CID."""
        self.metadata["manifest_cid"] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_cid": self.original_cid,
            "original_size": self.original_size,
            "original_hash": self.original_hash,
            "shard_size": self.shard_size,
            "shards": [s.to_dict() for s in self.shards],
            "created_at": self.created_at.isoformat(),
            "manifest_version": self.manifest_version,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShardManifest':
        """Create from dictionary."""
        return cls(
            original_cid=data.get("original_cid"),
            original_size=data["original_size"],
            original_hash=data["original_hash"],
            shard_size=data["shard_size"],
            shards=[ShardInfo.from_dict(s) for s in data["shards"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            manifest_version=data.get("manifest_version", "1.0"),
            metadata=data.get("metadata", {})
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ShardManifest':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Shard Index
# =============================================================================

class ShardIndex:
    """
    Index of all sharded content.
    
    This class maintains an in-memory index of shard manifests,
    enabling quick lookup of sharded content by various keys.
    
    Example:
        index = ShardIndex()
        index.add_manifest(manifest, manifest_cid)
        found = index.find_manifest(content_hash)
    """
    
    def __init__(self):
        """Initialize the shard index."""
        self.manifests: Dict[str, ShardManifest] = {}  # manifest_cid -> manifest
        self.content_to_manifest: Dict[str, str] = {}  # original_hash -> manifest_cid
        self.cid_to_manifest: Dict[str, str] = {}  # original_cid -> manifest_cid
    
    def add_manifest(self, manifest: ShardManifest, manifest_cid: str) -> None:
        """
        Add a manifest to the index.
        
        Args:
            manifest: The shard manifest to index
            manifest_cid: The IPFS CID of the manifest
        """
        self.manifests[manifest_cid] = manifest
        
        # Index by original content hash
        if manifest.original_hash:
            self.content_to_manifest[manifest.original_hash] = manifest_cid
        
        # Index by original CID if available
        if manifest.original_cid:
            self.cid_to_manifest[manifest.original_cid] = manifest_cid
        
        logger.debug("Manifest added to index",
                    manifest_cid=manifest_cid,
                    total_shards=manifest.total_shards,
                    original_size=manifest.original_size)
    
    def remove_manifest(self, manifest_cid: str) -> bool:
        """
        Remove a manifest from the index.
        
        Args:
            manifest_cid: The CID of the manifest to remove
            
        Returns:
            True if the manifest was removed, False if not found
        """
        if manifest_cid not in self.manifests:
            return False
        
        manifest = self.manifests[manifest_cid]
        
        # Remove from all indexes
        if manifest.original_hash and manifest.original_hash in self.content_to_manifest:
            del self.content_to_manifest[manifest.original_hash]
        
        if manifest.original_cid and manifest.original_cid in self.cid_to_manifest:
            del self.cid_to_manifest[manifest.original_cid]
        
        del self.manifests[manifest_cid]
        
        logger.debug("Manifest removed from index", manifest_cid=manifest_cid)
        return True
    
    def find_manifest(self, content_hash: Optional[str] = None, 
                      original_cid: Optional[str] = None,
                      manifest_cid: Optional[str] = None) -> Optional[ShardManifest]:
        """
        Find a manifest by various lookup keys.
        
        Args:
            content_hash: SHA-256 hash of original content
            original_cid: CID of the original file
            manifest_cid: CID of the manifest itself
            
        Returns:
            The found manifest or None
        """
        if manifest_cid:
            return self.manifests.get(manifest_cid)
        
        if content_hash:
            found_cid = self.content_to_manifest.get(content_hash)
            if found_cid:
                return self.manifests.get(found_cid)
        
        if original_cid:
            found_cid = self.cid_to_manifest.get(original_cid)
            if found_cid:
                return self.manifests.get(found_cid)
        
        return None
    
    def get_all_manifests(self) -> List[Tuple[str, ShardManifest]]:
        """Get all manifests with their CIDs."""
        return list(self.manifests.items())
    
    def clear(self) -> None:
        """Clear all entries from the index."""
        self.manifests.clear()
        self.content_to_manifest.clear()
        self.cid_to_manifest.clear()
    
    def __len__(self) -> int:
        """Return number of manifests in the index."""
        return len(self.manifests)
    
    def __contains__(self, manifest_cid: str) -> bool:
        """Check if a manifest CID is in the index."""
        return manifest_cid in self.manifests


# =============================================================================
# Content Sharder
# =============================================================================

class ContentSharder:
    """
    Handles sharding of large content for IPFS storage.
    
    This class provides methods to split large content into shards,
    upload them to IPFS, and reassemble them when needed.
    
    Example:
        sharder = ContentSharder(ipfs_client, config)
        manifest = await sharder.shard_content(large_content)
        reassembled = await sharder.reassemble_content(manifest)
    """
    
    def __init__(self, ipfs_client, config: Optional[ShardingConfig] = None):
        """
        Initialize the content sharder.
        
        Args:
            ipfs_client: IPFSClient instance for IPFS operations
            config: Sharding configuration (uses defaults if not provided)
        """
        self.ipfs_client = ipfs_client
        self.config = config or ShardingConfig()
        self.index = ShardIndex()
    
    def _calculate_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()
    
    def _split_content(self, content: bytes) -> List[bytes]:
        """
        Split content into shards.
        
        Args:
            content: The content to split
            
        Returns:
            List of content shards
        """
        total_size = len(content)
        shard_size = self.config.shard_size
        
        # Calculate number of shards
        num_shards = (total_size + shard_size - 1) // shard_size
        
        if num_shards > self.config.max_shards:
            raise ShardingError(
                f"Content requires {num_shards} shards, "
                f"exceeding maximum of {self.config.max_shards}"
            )
        
        shards = []
        for i in range(num_shards):
            start = i * shard_size
            end = min(start + shard_size, total_size)
            shards.append(content[start:end])
        
        logger.debug("Content split into shards",
                    total_size=total_size,
                    num_shards=num_shards,
                    shard_size=shard_size)
        
        return shards
    
    async def _upload_shard(
        self,
        shard: bytes,
        index: int,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> ShardInfo:
        """
        Upload a single shard to IPFS.
        
        Args:
            shard: The shard content
            index: The shard index
            progress_callback: Optional callback for progress updates
            
        Returns:
            ShardInfo for the uploaded shard
        """
        # Calculate hash before upload
        shard_hash = self._calculate_hash(shard)
        
        # Upload to IPFS
        result = await self.ipfs_client.upload_content(
            content=shard,
            filename=f"shard_{index}.bin",
            pin=True
        )
        
        if not result.success:
            raise ShardingError(f"Failed to upload shard {index}: {result.error}")
        
        shard_info = ShardInfo(
            index=index,
            cid=result.cid,
            size=len(shard),
            hash=shard_hash
        )
        
        if progress_callback:
            progress_callback(index, -1)  # -1 indicates single shard completion
        
        logger.debug("Shard uploaded",
                    index=index,
                    cid=result.cid,
                    size=len(shard))
        
        return shard_info
    
    async def _upload_shards_parallel(
        self,
        shards: List[bytes],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ShardInfo]:
        """
        Upload shards in parallel with controlled concurrency.
        
        Args:
            shards: List of shard contents
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of ShardInfo for all uploaded shards
        """
        total_shards = len(shards)
        shard_infos: List[Optional[ShardInfo]] = [None] * total_shards
        completed_count = 0
        lock = asyncio.Lock()
        
        async def upload_with_retry(shard: bytes, index: int) -> None:
            nonlocal completed_count
            
            for attempt in range(self.config.retry_attempts):
                try:
                    shard_info = await self._upload_shard(shard, index)
                    shard_infos[index] = shard_info
                    
                    async with lock:
                        completed_count += 1
                        if progress_callback:
                            progress_callback(completed_count, total_shards)
                    
                    return
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise ShardingError(
                            f"Failed to upload shard {index} after "
                            f"{self.config.retry_attempts} attempts: {e}"
                        )
                    await asyncio.sleep(1 * (attempt + 1))  # Simple backoff
        
        # Create semaphore for controlled parallelism
        semaphore = asyncio.Semaphore(self.config.parallel_uploads)
        
        async def bounded_upload(shard: bytes, index: int) -> None:
            async with semaphore:
                await upload_with_retry(shard, index)
        
        # Upload all shards with controlled parallelism
        tasks = [
            bounded_upload(shard, index)
            for index, shard in enumerate(shards)
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify all shards were uploaded
        if None in shard_infos:
            raise ShardingError("Some shards failed to upload")
        
        return shard_infos  # type: ignore
    
    async def shard_content(
        self,
        content: bytes,
        original_cid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[ShardManifest, str]:
        """
        Split content into shards and upload to IPFS.
        
        This method:
        1. Calculates the hash of the original content
        2. Splits content into appropriately sized shards
        3. Uploads shards in parallel to IPFS
        4. Creates and stores a manifest
        5. Returns the manifest and its CID
        
        Args:
            content: The content to shard
            original_cid: Optional CID of the original file (if known)
            metadata: Additional metadata to include in manifest
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (ShardManifest, manifest_cid)
        """
        if not content:
            raise ShardingError("Cannot shard empty content")
        
        # Calculate original content hash
        original_hash = self._calculate_hash(content)
        original_size = len(content)
        
        logger.info("Starting content sharding",
                   original_size=original_size,
                   original_hash=original_hash[:16] + "...")
        
        # Split content into shards
        shards = self._split_content(content)
        
        if len(shards) == 0:
            raise ShardingError("No shards created from content")
        
        # Handle single shard case (file smaller than shard size)
        if len(shards) == 1:
            logger.info("Content fits in single shard, uploading directly")
            shard_info = await self._upload_shard(shards[0], 0, progress_callback)
            
            manifest = ShardManifest(
                original_cid=original_cid,
                original_size=original_size,
                original_hash=original_hash,
                shard_size=self.config.shard_size,
                shards=[shard_info],
                metadata=metadata or {}
            )
        else:
            # Upload shards in parallel
            shard_infos = await self._upload_shards_parallel(shards, progress_callback)
            
            manifest = ShardManifest(
                original_cid=original_cid,
                original_size=original_size,
                original_hash=original_hash,
                shard_size=self.config.shard_size,
                shards=shard_infos,
                metadata=metadata or {}
            )
        
        # Upload manifest to IPFS
        manifest_json = manifest.to_json()
        manifest_result = await self.ipfs_client.upload_content(
            content=manifest_json,
            filename="shard_manifest.json",
            pin=True
        )
        
        if not manifest_result.success:
            raise ShardingError(f"Failed to upload manifest: {manifest_result.error}")
        
        manifest_cid = manifest_result.cid
        manifest.manifest_cid = manifest_cid
        
        # Add to index
        self.index.add_manifest(manifest, manifest_cid)
        
        logger.info("Content sharding complete",
                   manifest_cid=manifest_cid,
                   total_shards=manifest.total_shards,
                   original_size=original_size)
        
        return manifest, manifest_cid
    
    async def _download_shard(
        self,
        shard_info: ShardInfo,
        verify: bool = True
    ) -> bytes:
        """
        Download a single shard from IPFS.
        
        Args:
            shard_info: Information about the shard to download
            verify: Whether to verify the shard hash
            
        Returns:
            The shard content
        """
        result = await self.ipfs_client.download_content(shard_info.cid)
        
        if not result.success:
            raise ShardMissingError(
                f"Failed to download shard {shard_info.index}: {result.error}"
            )
        
        content = result.metadata.get("content", b"") if result.metadata else b""
        
        if not content:
            raise ShardMissingError(
                f"Shard {shard_info.index} returned empty content"
            )
        
        # Verify hash if requested
        if verify:
            actual_hash = self._calculate_hash(content)
            if actual_hash != shard_info.hash:
                raise ShardVerificationError(
                    f"Shard {shard_info.index} hash mismatch: "
                    f"expected {shard_info.hash[:16]}..., "
                    f"got {actual_hash[:16]}..."
                )
        
        return content
    
    async def _download_shards_parallel(
        self,
        shards: List[ShardInfo],
        verify: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[bytes]:
        """
        Download shards in parallel with controlled concurrency.
        
        Args:
            shards: List of shard information
            verify: Whether to verify each shard hash
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of shard contents in order
        """
        total_shards = len(shards)
        contents: List[Optional[bytes]] = [None] * total_shards
        completed_count = 0
        lock = asyncio.Lock()
        
        async def download_with_retry(shard_info: ShardInfo) -> None:
            nonlocal completed_count
            
            for attempt in range(self.config.retry_attempts):
                try:
                    content = await self._download_shard(shard_info, verify)
                    contents[shard_info.index] = content
                    
                    async with lock:
                        completed_count += 1
                        if progress_callback:
                            progress_callback(completed_count, total_shards)
                    
                    return
                except ShardVerificationError:
                    # Don't retry verification failures
                    raise
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise ShardMissingError(
                            f"Failed to download shard {shard_info.index} after "
                            f"{self.config.retry_attempts} attempts: {e}"
                        )
                    await asyncio.sleep(1 * (attempt + 1))
        
        # Create semaphore for controlled parallelism
        semaphore = asyncio.Semaphore(self.config.parallel_downloads)
        
        async def bounded_download(shard_info: ShardInfo) -> None:
            async with semaphore:
                await download_with_retry(shard_info)
        
        # Download all shards with controlled parallelism
        tasks = [bounded_download(shard_info) for shard_info in shards]
        await asyncio.gather(*tasks)
        
        # Verify all shards were downloaded
        if None in contents:
            raise ShardMissingError("Some shards failed to download")
        
        return contents  # type: ignore
    
    async def reassemble_content(
        self,
        manifest: ShardManifest,
        verify: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bytes:
        """
        Download shards and reassemble original content.
        
        This method:
        1. Downloads all shards in parallel
        2. Verifies each shard's hash
        3. Reassembles shards in order
        4. Verifies the final content hash
        
        Args:
            manifest: The shard manifest
            verify: Whether to verify hashes
            progress_callback: Optional callback for progress updates
            
        Returns:
            The reassembled original content
        """
        if not manifest.shards:
            raise ManifestError("Manifest has no shards")
        
        logger.info("Starting content reassembly",
                   manifest_cid=manifest.manifest_cid,
                   total_shards=manifest.total_shards,
                   original_size=manifest.original_size)
        
        # Download all shards in parallel
        shard_contents = await self._download_shards_parallel(
            manifest.shards, verify, progress_callback
        )
        
        # Reassemble in order
        reassembled = b"".join(shard_contents)
        
        # Verify final size
        if len(reassembled) != manifest.original_size:
            raise ShardVerificationError(
                f"Reassembled size mismatch: "
                f"expected {manifest.original_size}, "
                f"got {len(reassembled)}"
            )
        
        # Verify final hash
        if verify:
            final_hash = self._calculate_hash(reassembled)
            if final_hash != manifest.original_hash:
                raise ShardVerificationError(
                    f"Final content hash mismatch: "
                    f"expected {manifest.original_hash[:16]}..., "
                    f"got {final_hash[:16]}..."
                )
        
        logger.info("Content reassembly complete",
                   original_size=len(reassembled),
                   total_shards=manifest.total_shards)
        
        return reassembled
    
    async def get_shard_manifest(self, manifest_cid: str) -> ShardManifest:
        """
        Load a shard manifest from IPFS.
        
        Args:
            manifest_cid: The CID of the manifest
            
        Returns:
            The loaded ShardManifest
        """
        # Check index first
        cached = self.index.find_manifest(manifest_cid=manifest_cid)
        if cached:
            return cached
        
        # Download from IPFS
        result = await self.ipfs_client.download_content(manifest_cid)
        
        if not result.success:
            raise ManifestError(
                f"Failed to download manifest: {result.error}"
            )
        
        content = result.metadata.get("content", b"") if result.metadata else b""
        
        if not content:
            raise ManifestError("Manifest returned empty content")
        
        try:
            manifest = ShardManifest.from_json(content.decode('utf-8'))
            manifest.manifest_cid = manifest_cid
            
            # Add to index
            self.index.add_manifest(manifest, manifest_cid)
            
            return manifest
        except (json.JSONDecodeError, KeyError) as e:
            raise ManifestError(f"Failed to parse manifest: {e}")
    
    def should_shard(self, content_size: int) -> bool:
        """
        Determine if content should be sharded based on size.
        
        Args:
            content_size: Size of the content in bytes
            
        Returns:
            True if content should be sharded
        """
        # Zero or negative threshold means sharding is disabled
        if self.config.auto_shard_threshold <= 0:
            return False
        return content_size > self.config.auto_shard_threshold
    
    def get_index(self) -> ShardIndex:
        """Get the shard index."""
        return self.index


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_optimal_shard_size(
    content_size: int,
    target_shards: int = 10,
    min_shard_size: int = 1 * 1024 * 1024,  # 1MB
    max_shard_size: int = 100 * 1024 * 1024  # 100MB
) -> int:
    """
    Calculate optimal shard size for given content.
    
    Args:
        content_size: Total size of content in bytes
        target_shards: Target number of shards
        min_shard_size: Minimum shard size
        max_shard_size: Maximum shard size
        
    Returns:
        Optimal shard size in bytes
    """
    if content_size <= 0:
        return min_shard_size
    
    calculated_size = content_size // target_shards
    
    # Clamp to bounds
    return max(min_shard_size, min(calculated_size, max_shard_size))


def estimate_shard_count(content_size: int, shard_size: int) -> int:
    """
    Estimate the number of shards for given content.
    
    Args:
        content_size: Total size of content in bytes
        shard_size: Size of each shard in bytes
        
    Returns:
        Estimated number of shards
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    
    return (content_size + shard_size - 1) // shard_size
