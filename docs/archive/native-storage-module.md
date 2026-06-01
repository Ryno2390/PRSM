# Native Storage Module — Implementation Plan (Part 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `prsm/storage/` — PRSM's native content-addressed storage module with sharding, Shamir's Secret Sharing manifest encryption, and content distribution.

**Architecture:** Seven files in `prsm/storage/`: shared models/exceptions, a local blob store for content-addressed file I/O, a shard engine for split/reassemble with manifest management, a key manager for Shamir's Secret Sharing + AES-256-GCM encryption, a distribution manager for placement/retrieval/health monitoring, and a `ContentStore` facade that ties them together.

**Tech Stack:** Python 3.10+, `hashlib` (SHA-256), `cryptography` (AES-256-GCM, Ed25519), in-repo Shamir's Secret Sharing over GF(256), `asyncio`/`aiofiles` for async I/O, existing libp2p transport + discovery layers.

**Spec:** `docs/native-storage-design.md`

---

## File Structure

```
prsm/storage/
  __init__.py          # Public API: ContentStore, ContentHash, re-exports
  models.py            # ContentHash, ShardManifest, KeyShare, ContentDescriptor,
                       #   ReplicationPolicy, RetrievalTicket, all enums
  exceptions.py        # ContentNotFoundError, ShardIntegrityError, etc.
  blob_store.py        # Local content-addressed file I/O
  shard_engine.py      # Split/reassemble + manifest serialization
  key_manager.py       # Shamir GF(256) + AES-256-GCM encrypt/decrypt
  distribution.py      # Placement, retrieval, health monitor, contract key

tests/unit/storage/
  __init__.py
  test_models.py
  test_blob_store.py
  test_shard_engine.py
  test_key_manager.py
  test_distribution.py

tests/integration/
  test_native_storage.py
```

---

### Task 1: Models & Exceptions

**Files:**
- Create: `prsm/storage/__init__.py`
- Create: `prsm/storage/models.py`
- Create: `prsm/storage/exceptions.py`
- Create: `tests/unit/storage/__init__.py`
- Create: `tests/unit/storage/test_models.py`

- [ ] **Step 1: Write the failing test for ContentHash**

```python
# tests/unit/storage/test_models.py
"""Tests for prsm.storage.models."""
import pytest
from prsm.storage.models import ContentHash, AlgorithmID


class TestContentHash:
    def test_from_data_sha256(self):
        data = b"hello world"
        ch = ContentHash.from_data(data)
        assert ch.algorithm_id == AlgorithmID.SHA256
        assert len(ch.digest) == 32

    def test_hex_representation_66_chars(self):
        ch = ContentHash.from_data(b"test")
        hex_str = ch.hex()
        # 1 byte algo prefix (2 hex chars) + 32 byte digest (64 hex chars) = 66
        assert len(hex_str) == 66
        assert hex_str.startswith("01")  # SHA-256 prefix

    def test_from_hex_roundtrip(self):
        original = ContentHash.from_data(b"roundtrip test")
        restored = ContentHash.from_hex(original.hex())
        assert restored.algorithm_id == original.algorithm_id
        assert restored.digest == original.digest

    def test_from_hex_invalid_algorithm(self):
        # Algorithm 0xFF doesn't exist
        with pytest.raises(ValueError, match="Unknown algorithm"):
            ContentHash.from_hex("ff" + "aa" * 32)

    def test_deduplication_same_content(self):
        a = ContentHash.from_data(b"same content")
        b = ContentHash.from_data(b"same content")
        assert a.hex() == b.hex()

    def test_different_content_different_hash(self):
        a = ContentHash.from_data(b"content A")
        b = ContentHash.from_data(b"content B")
        assert a.hex() != b.hex()

    def test_equality(self):
        a = ContentHash.from_data(b"equal")
        b = ContentHash.from_data(b"equal")
        assert a == b

    def test_hashable_for_dicts(self):
        ch = ContentHash.from_data(b"key")
        d = {ch: "value"}
        assert d[ch] == "value"

    def test_str_returns_hex(self):
        ch = ContentHash.from_data(b"str test")
        assert str(ch) == ch.hex()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/storage/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prsm.storage'`

- [ ] **Step 3: Create package and implement ContentHash**

```python
# prsm/storage/__init__.py
"""
PRSM Native Content-Addressed Storage

Replaces IPFS/Kubo dependency with PRSM-native storage.
"""
from prsm.storage.models import ContentHash
from prsm.storage.exceptions import (
    ContentNotFoundError,
    ShardIntegrityError,
    ManifestError,
    KeyReconstructionError,
    PlacementError,
    StorageError,
)

__all__ = [
    "ContentHash",
    "ContentNotFoundError",
    "ShardIntegrityError",
    "ManifestError",
    "KeyReconstructionError",
    "PlacementError",
    "StorageError",
]
```

```python
# prsm/storage/exceptions.py
"""Storage-specific exceptions."""


class StorageError(Exception):
    """Base exception for all storage operations."""


class ContentNotFoundError(StorageError):
    """Raised when content hash does not exist in the store."""

    def __init__(self, content_hash: str):
        self.content_hash = content_hash
        super().__init__(f"Content not found: {content_hash}")


class ShardIntegrityError(StorageError):
    """Raised when a shard fails hash verification."""

    def __init__(self, expected: str, actual: str):
        self.expected = expected
        self.actual = actual
        super().__init__(f"Shard integrity check failed: expected {expected}, got {actual}")


class ManifestError(StorageError):
    """Raised on manifest serialization/deserialization or decryption failure."""


class KeyReconstructionError(StorageError):
    """Raised when Shamir key reconstruction fails (insufficient shares, corrupt data)."""


class PlacementError(StorageError):
    """Raised when shard placement constraints cannot be satisfied."""

    def __init__(self, reason: str, min_nodes_needed: int = 0):
        self.reason = reason
        self.min_nodes_needed = min_nodes_needed
        super().__init__(f"Placement failed: {reason}")
```

```python
# prsm/storage/models.py
"""Shared data models for the native storage module."""
from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class AlgorithmID(enum.IntEnum):
    """Content hash algorithm identifiers. 1-byte prefix for algorithm agility."""
    SHA256 = 0x01
    SHA3_256 = 0x02   # Reserved
    BLAKE3 = 0x03     # Reserved


# Map of algorithm ID -> (hashlib name, digest size in bytes)
_ALGORITHM_META = {
    AlgorithmID.SHA256: ("sha256", 32),
}


@dataclass(frozen=True)
class ContentHash:
    """Content-addressed identifier with algorithm-agility prefix.

    Format: [1 byte algorithm ID][N bytes digest]
    Serialized as hex: "01<64 hex chars>" for SHA-256 (66 chars total).
    """
    algorithm_id: AlgorithmID
    digest: bytes

    @classmethod
    def from_data(cls, data: bytes, algorithm: AlgorithmID = AlgorithmID.SHA256) -> ContentHash:
        """Hash raw bytes and return a ContentHash."""
        if algorithm not in _ALGORITHM_META:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        hash_name, expected_size = _ALGORITHM_META[algorithm]
        h = hashlib.new(hash_name, data)
        digest = h.digest()
        assert len(digest) == expected_size
        return cls(algorithm_id=algorithm, digest=digest)

    def hex(self) -> str:
        """Serialize to hex string: algorithm prefix byte + digest bytes."""
        return f"{self.algorithm_id:02x}{self.digest.hex()}"

    @classmethod
    def from_hex(cls, hex_str: str) -> ContentHash:
        """Deserialize from hex string."""
        if len(hex_str) < 4:
            raise ValueError(f"Hex string too short: {hex_str!r}")
        algo_byte = int(hex_str[:2], 16)
        try:
            algorithm = AlgorithmID(algo_byte)
        except ValueError:
            raise ValueError(f"Unknown algorithm ID: 0x{algo_byte:02x}")
        digest = bytes.fromhex(hex_str[2:])
        if algorithm in _ALGORITHM_META:
            _, expected_size = _ALGORITHM_META[algorithm]
            if len(digest) != expected_size:
                raise ValueError(
                    f"Digest size mismatch for {algorithm.name}: "
                    f"expected {expected_size}, got {len(digest)}"
                )
        return cls(algorithm_id=algorithm, digest=digest)

    def __str__(self) -> str:
        return self.hex()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContentHash):
            return NotImplemented
        return self.algorithm_id == other.algorithm_id and self.digest == other.digest

    def __hash__(self) -> int:
        return hash((self.algorithm_id, self.digest))
```

```python
# tests/unit/storage/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/storage/test_models.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Write failing tests for ShardManifest and other models**

Add to `tests/unit/storage/test_models.py`:

```python
import json
from prsm.storage.models import (
    ContentHash, AlgorithmID, ShardManifest, KeyShare,
    ReplicationPolicy, ContentDescriptor, RetrievalTicket,
)


class TestShardManifest:
    def test_create_manifest(self):
        content_hash = ContentHash.from_data(b"full content")
        shard_hashes = [
            ContentHash.from_data(b"shard0"),
            ContentHash.from_data(b"shard1"),
        ]
        manifest = ShardManifest(
            content_hash=content_hash,
            shard_hashes=shard_hashes,
            total_size=2048,
            shard_size=1024,
            algorithm_id=AlgorithmID.SHA256,
            created_at=1000.0,
            replication_factor=3,
            owner_node_id="node-abc",
            visibility="public",
        )
        assert manifest.total_size == 2048
        assert len(manifest.shard_hashes) == 2

    def test_to_json_roundtrip(self):
        content_hash = ContentHash.from_data(b"json test")
        manifest = ShardManifest(
            content_hash=content_hash,
            shard_hashes=[ContentHash.from_data(b"s0")],
            total_size=512,
            shard_size=512,
            algorithm_id=AlgorithmID.SHA256,
            created_at=1000.0,
            replication_factor=3,
            owner_node_id="owner-1",
            visibility="public",
        )
        json_str = manifest.to_json()
        restored = ShardManifest.from_json(json_str)
        assert restored.content_hash == manifest.content_hash
        assert restored.shard_hashes == manifest.shard_hashes
        assert restored.total_size == manifest.total_size
        assert restored.visibility == "public"

    def test_from_json_invalid(self):
        with pytest.raises(Exception):
            ShardManifest.from_json("not valid json {{{")


class TestReplicationPolicy:
    def test_defaults(self):
        policy = ReplicationPolicy(replication_factor=3)
        assert policy.owner_excluded is True
        assert policy.key_shard_separation is True
        assert policy.degraded_constraints == []

    def test_custom_values(self):
        policy = ReplicationPolicy(
            replication_factor=5,
            min_asn_diversity=4,
            degraded_constraints=["asn_relaxed"],
        )
        assert policy.replication_factor == 5
        assert policy.degraded_constraints == ["asn_relaxed"]


class TestKeyShare:
    def test_create_key_share(self):
        ch = ContentHash.from_data(b"key content")
        share = KeyShare(
            content_hash=ch,
            share_index=1,
            share_data=b"\x01\x02\x03",
            threshold=3,
            total_shares=5,
            algorithm_id=0x01,
        )
        assert share.threshold == 3
        assert share.total_shares == 5
```

- [ ] **Step 6: Implement remaining models**

Add to `prsm/storage/models.py`:

```python
@dataclass
class ShardManifest:
    """Maps a complete content hash to its ordered shard hashes."""
    content_hash: ContentHash
    shard_hashes: List[ContentHash]
    total_size: int
    shard_size: int
    algorithm_id: AlgorithmID
    created_at: float
    replication_factor: int
    owner_node_id: str
    visibility: str = "public"

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "content_hash": self.content_hash.hex(),
            "shard_hashes": [sh.hex() for sh in self.shard_hashes],
            "total_size": self.total_size,
            "shard_size": self.shard_size,
            "algorithm_id": int(self.algorithm_id),
            "created_at": self.created_at,
            "replication_factor": self.replication_factor,
            "owner_node_id": self.owner_node_id,
            "visibility": self.visibility,
        })

    @classmethod
    def from_json(cls, json_str: str) -> ShardManifest:
        """Deserialize from JSON string."""
        d = json.loads(json_str)
        return cls(
            content_hash=ContentHash.from_hex(d["content_hash"]),
            shard_hashes=[ContentHash.from_hex(h) for h in d["shard_hashes"]],
            total_size=d["total_size"],
            shard_size=d["shard_size"],
            algorithm_id=AlgorithmID(d["algorithm_id"]),
            created_at=d["created_at"],
            replication_factor=d["replication_factor"],
            owner_node_id=d["owner_node_id"],
            visibility=d.get("visibility", "public"),
        )


@dataclass
class KeyShare:
    """One share of a Shamir-split encryption key."""
    content_hash: ContentHash
    share_index: int
    share_data: bytes
    threshold: int
    total_shares: int
    algorithm_id: int  # 0x01 = AES-256-GCM


@dataclass
class ReplicationPolicy:
    """Owner-specified replication and placement constraints."""
    replication_factor: int
    min_asn_diversity: int = 2
    owner_excluded: bool = True
    key_shard_separation: bool = True
    degraded_constraints: List[str] = field(default_factory=list)


@dataclass
class ContentDescriptor:
    """Bootstrap record published to DHT. Entry point for all retrieval."""
    content_hash: ContentHash
    manifest_holders: List[str]
    key_share_holders: List[str]
    contract_key_share_holders: List[str]
    shard_map: Dict[str, List[str]]  # shard_hash hex -> [node_ids]
    replication_policy: ReplicationPolicy
    visibility: str  # "public" or "private"
    epoch: int
    version: int
    owner_node_id: str
    contract_pubkey: bytes
    signature: bytes
    signer_type: str  # "owner" or "contract"
    created_at: float
    updated_at: float


@dataclass
class RetrievalTicket:
    """Signed capability token for private content retrieval (Phase 2)."""
    content_hash: ContentHash
    requester_node_id: str
    epoch: int
    issued_at: float
    expires_at: float
    nonce: str
    issuer_signature: bytes
```

Add `import json` to the top of `models.py`.

- [ ] **Step 7: Run all model tests**

Run: `python -m pytest tests/unit/storage/test_models.py -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add prsm/storage/__init__.py prsm/storage/models.py prsm/storage/exceptions.py \
       tests/unit/storage/__init__.py tests/unit/storage/test_models.py
git commit -m "feat(storage): add models and exceptions for native storage module"
```

---

### Task 2: Blob Store

**Files:**
- Create: `prsm/storage/blob_store.py`
- Create: `tests/unit/storage/test_blob_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/storage/test_blob_store.py
"""Tests for prsm.storage.blob_store."""
import os
import pytest
from prsm.storage.blob_store import BlobStore
from prsm.storage.models import ContentHash, AlgorithmID
from prsm.storage.exceptions import ContentNotFoundError


@pytest.fixture
def blob_store(tmp_path):
    return BlobStore(data_dir=str(tmp_path / "storage"))


class TestBlobStore:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_roundtrip(self, blob_store):
        data = b"hello world"
        content_hash = await blob_store.store(data)
        retrieved = await blob_store.retrieve(content_hash)
        assert retrieved == data

    @pytest.mark.asyncio
    async def test_content_hash_is_sha256(self, blob_store):
        data = b"hash check"
        content_hash = await blob_store.store(data)
        assert content_hash.algorithm_id == AlgorithmID.SHA256
        # Verify against direct hashlib computation
        expected = ContentHash.from_data(data)
        assert content_hash == expected

    @pytest.mark.asyncio
    async def test_deduplication_single_file(self, blob_store):
        data = b"duplicate me"
        h1 = await blob_store.store(data)
        h2 = await blob_store.store(data)
        assert h1 == h2
        # Only one file on disk
        prefix_dir = os.path.join(blob_store.data_dir, h1.hex()[:2])
        files = os.listdir(prefix_dir)
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_hash_prefix_directory_created(self, blob_store):
        data = b"dir test"
        content_hash = await blob_store.store(data)
        prefix = content_hash.hex()[:2]
        assert os.path.isdir(os.path.join(blob_store.data_dir, prefix))

    @pytest.mark.asyncio
    async def test_exists_true(self, blob_store):
        data = b"exists"
        ch = await blob_store.store(data)
        assert await blob_store.exists(ch) is True

    @pytest.mark.asyncio
    async def test_exists_false(self, blob_store):
        ch = ContentHash.from_data(b"not stored")
        assert await blob_store.exists(ch) is False

    @pytest.mark.asyncio
    async def test_delete_removes_file(self, blob_store):
        data = b"delete me"
        ch = await blob_store.store(data)
        assert await blob_store.exists(ch) is True
        await blob_store.delete(ch)
        assert await blob_store.exists(ch) is False

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_raises(self, blob_store):
        ch = ContentHash.from_data(b"ghost")
        with pytest.raises(ContentNotFoundError):
            await blob_store.retrieve(ch)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self, blob_store):
        ch = ContentHash.from_data(b"nothing here")
        # Should not raise
        await blob_store.delete(ch)

    @pytest.mark.asyncio
    async def test_large_content(self, blob_store):
        data = os.urandom(1024 * 1024)  # 1MB
        ch = await blob_store.store(data)
        retrieved = await blob_store.retrieve(ch)
        assert retrieved == data

    @pytest.mark.asyncio
    async def test_empty_content(self, blob_store):
        data = b""
        ch = await blob_store.store(data)
        retrieved = await blob_store.retrieve(ch)
        assert retrieved == data

    @pytest.mark.asyncio
    async def test_data_dir_created_on_first_store(self, tmp_path):
        store = BlobStore(data_dir=str(tmp_path / "new" / "nested" / "dir"))
        data = b"first write"
        await store.store(data)
        assert os.path.isdir(store.data_dir)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/storage/test_blob_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prsm.storage.blob_store'`

- [ ] **Step 3: Implement BlobStore**

```python
# prsm/storage/blob_store.py
"""Local content-addressed file store.

Stores blobs on disk indexed by their ContentHash.
Path layout: {data_dir}/{first 2 hex chars}/{remaining hex chars}
Deduplication is automatic — same content produces the same hash and file path.
"""
from __future__ import annotations

import os
from pathlib import Path

import aiofiles
import aiofiles.os

from prsm.storage.exceptions import ContentNotFoundError
from prsm.storage.models import ContentHash


class BlobStore:
    """Async local content-addressed blob store."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    def _path_for(self, content_hash: ContentHash) -> str:
        """Return the filesystem path for a content hash."""
        h = content_hash.hex()
        return os.path.join(self.data_dir, h[:2], h[2:])

    async def store(self, data: bytes) -> ContentHash:
        """Hash data, write to disk, return ContentHash. Deduplicates automatically."""
        content_hash = ContentHash.from_data(data)
        path = self._path_for(content_hash)
        if os.path.exists(path):
            return content_hash  # Already stored (dedup)
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        # Write to temp file then rename for atomicity
        tmp_path = path + ".tmp"
        async with aiofiles.open(tmp_path, "wb") as f:
            await f.write(data)
        os.replace(tmp_path, path)
        return content_hash

    async def retrieve(self, content_hash: ContentHash) -> bytes:
        """Read blob from disk by hash. Raises ContentNotFoundError if missing."""
        path = self._path_for(content_hash)
        if not os.path.exists(path):
            raise ContentNotFoundError(content_hash.hex())
        async with aiofiles.open(path, "rb") as f:
            return await f.read()

    async def exists(self, content_hash: ContentHash) -> bool:
        """Check if content exists locally."""
        return os.path.exists(self._path_for(content_hash))

    async def delete(self, content_hash: ContentHash) -> None:
        """Remove content from local store. No-op if not present."""
        path = self._path_for(content_hash)
        if os.path.exists(path):
            os.remove(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/storage/test_blob_store.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/storage/blob_store.py tests/unit/storage/test_blob_store.py
git commit -m "feat(storage): add BlobStore for local content-addressed file I/O"
```

---

### Task 3: Shard Engine

**Files:**
- Create: `prsm/storage/shard_engine.py`
- Create: `tests/unit/storage/test_shard_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/storage/test_shard_engine.py
"""Tests for prsm.storage.shard_engine."""
import math
import os
import pytest
from prsm.storage.shard_engine import ShardEngine
from prsm.storage.blob_store import BlobStore
from prsm.storage.models import ContentHash, AlgorithmID, ShardManifest
from prsm.storage.exceptions import ShardIntegrityError


@pytest.fixture
def blob_store(tmp_path):
    return BlobStore(data_dir=str(tmp_path / "blobs"))


@pytest.fixture
def engine(blob_store):
    # Low thresholds for testing: shard if > 100 bytes, 50 byte shards
    return ShardEngine(blob_store=blob_store, shard_threshold=100, shard_size=50)


class TestShardEngine:
    @pytest.mark.asyncio
    async def test_small_content_single_shard(self, engine):
        data = b"small"  # 5 bytes, below 100-byte threshold
        manifest = await engine.split(data, owner_node_id="owner-1", replication_factor=3)
        assert len(manifest.shard_hashes) == 1
        assert manifest.total_size == 5

    @pytest.mark.asyncio
    async def test_large_content_multiple_shards(self, engine):
        data = os.urandom(200)  # 200 bytes, above threshold, 50-byte shards -> 4 shards
        manifest = await engine.split(data, owner_node_id="owner-1", replication_factor=3)
        assert len(manifest.shard_hashes) == 4
        assert manifest.total_size == 200
        assert manifest.shard_size == 50

    @pytest.mark.asyncio
    async def test_shard_count_ceil(self, engine):
        # 130 bytes / 50 byte shards = 3 shards (ceil)
        data = os.urandom(130)
        manifest = await engine.split(data, owner_node_id="owner-1", replication_factor=3)
        assert len(manifest.shard_hashes) == math.ceil(130 / 50)

    @pytest.mark.asyncio
    async def test_reassemble_produces_original(self, engine):
        data = os.urandom(200)
        manifest = await engine.split(data, owner_node_id="owner-1", replication_factor=3)
        reassembled = await engine.reassemble(manifest)
        assert reassembled == data

    @pytest.mark.asyncio
    async def test_reassemble_small_content(self, engine):
        data = b"tiny"
        manifest = await engine.split(data, owner_node_id="owner-1", replication_factor=3)
        reassembled = await engine.reassemble(manifest)
        assert reassembled == data

    @pytest.mark.asyncio
    async def test_content_hash_matches(self, engine):
        data = os.urandom(200)
        manifest = await engine.split(data, owner_node_id="owner-1", replication_factor=3)
        expected_hash = ContentHash.from_data(data)
        assert manifest.content_hash == expected_hash

    @pytest.mark.asyncio
    async def test_each_shard_stored_in_blob_store(self, engine, blob_store):
        data = os.urandom(200)
        manifest = await engine.split(data, owner_node_id="owner-1", replication_factor=3)
        for shard_hash in manifest.shard_hashes:
            assert await blob_store.exists(shard_hash)

    @pytest.mark.asyncio
    async def test_tampered_shard_detected(self, engine, blob_store):
        data = os.urandom(200)
        manifest = await engine.split(data, owner_node_id="owner-1", replication_factor=3)
        # Corrupt the first shard on disk
        first_shard = manifest.shard_hashes[0]
        path = blob_store._path_for(first_shard)
        with open(path, "wb") as f:
            f.write(b"corrupted data that does not match hash")
        with pytest.raises(ShardIntegrityError):
            await engine.reassemble(manifest)

    @pytest.mark.asyncio
    async def test_manifest_json_roundtrip(self, engine):
        data = os.urandom(200)
        manifest = await engine.split(data, owner_node_id="owner-1", replication_factor=3)
        json_str = manifest.to_json()
        restored = ShardManifest.from_json(json_str)
        assert restored.content_hash == manifest.content_hash
        assert restored.shard_hashes == manifest.shard_hashes

    @pytest.mark.asyncio
    async def test_manifest_metadata(self, engine):
        data = os.urandom(200)
        manifest = await engine.split(data, owner_node_id="my-node", replication_factor=5)
        assert manifest.owner_node_id == "my-node"
        assert manifest.replication_factor == 5
        assert manifest.algorithm_id == AlgorithmID.SHA256
        assert manifest.visibility == "public"
        assert manifest.created_at > 0

    @pytest.mark.asyncio
    async def test_empty_content(self, engine):
        data = b""
        manifest = await engine.split(data, owner_node_id="owner-1", replication_factor=3)
        assert len(manifest.shard_hashes) == 1
        reassembled = await engine.reassemble(manifest)
        assert reassembled == data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/storage/test_shard_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prsm.storage.shard_engine'`

- [ ] **Step 3: Implement ShardEngine**

```python
# prsm/storage/shard_engine.py
"""Shard engine: split content into chunks, reassemble with integrity verification.

Sharding is a core security property — no single node holds complete content.
Content above the shard threshold is split into fixed-size chunks. Each shard
is individually content-hashed and stored via the blob store. A ShardManifest
maps the original content hash to an ordered list of shard hashes.
"""
from __future__ import annotations

import time

from prsm.storage.blob_store import BlobStore
from prsm.storage.exceptions import ShardIntegrityError
from prsm.storage.models import AlgorithmID, ContentHash, ShardManifest


class ShardEngine:
    """Split content into shards and reassemble with integrity checks."""

    def __init__(
        self,
        blob_store: BlobStore,
        shard_threshold: int = 1_048_576,  # 1MB
        shard_size: int = 262_144,          # 256KB
    ) -> None:
        self.blob_store = blob_store
        self.shard_threshold = shard_threshold
        self.shard_size = shard_size

    async def split(
        self,
        data: bytes,
        owner_node_id: str,
        replication_factor: int,
        visibility: str = "public",
    ) -> ShardManifest:
        """Split content into shards, store each in blob store, return manifest."""
        content_hash = ContentHash.from_data(data)

        if len(data) <= self.shard_threshold:
            # Store as single shard
            shard_hash = await self.blob_store.store(data)
            shard_hashes = [shard_hash]
        else:
            # Split into fixed-size chunks
            shard_hashes = []
            for offset in range(0, len(data), self.shard_size):
                chunk = data[offset : offset + self.shard_size]
                shard_hash = await self.blob_store.store(chunk)
                shard_hashes.append(shard_hash)

        return ShardManifest(
            content_hash=content_hash,
            shard_hashes=shard_hashes,
            total_size=len(data),
            shard_size=self.shard_size,
            algorithm_id=AlgorithmID.SHA256,
            created_at=time.time(),
            replication_factor=replication_factor,
            owner_node_id=owner_node_id,
            visibility=visibility,
        )

    async def reassemble(self, manifest: ShardManifest) -> bytes:
        """Retrieve shards from blob store, reassemble, verify integrity."""
        chunks = []
        for shard_hash in manifest.shard_hashes:
            shard_data = await self.blob_store.retrieve(shard_hash)
            # Verify each shard against its hash
            actual_hash = ContentHash.from_data(shard_data)
            if actual_hash != shard_hash:
                raise ShardIntegrityError(
                    expected=shard_hash.hex(),
                    actual=actual_hash.hex(),
                )
            chunks.append(shard_data)

        reassembled = b"".join(chunks)

        # Verify complete content hash
        actual_content_hash = ContentHash.from_data(reassembled)
        if actual_content_hash != manifest.content_hash:
            raise ShardIntegrityError(
                expected=manifest.content_hash.hex(),
                actual=actual_content_hash.hex(),
            )

        return reassembled
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/storage/test_shard_engine.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/storage/shard_engine.py tests/unit/storage/test_shard_engine.py
git commit -m "feat(storage): add ShardEngine for content splitting and reassembly"
```

---

### Task 4: Key Manager (Shamir's Secret Sharing + AES-256-GCM)

**Files:**
- Create: `prsm/storage/key_manager.py`
- Create: `tests/unit/storage/test_key_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/storage/test_key_manager.py
"""Tests for prsm.storage.key_manager — Shamir's Secret Sharing + AES-256-GCM."""
import os
import pytest
from prsm.storage.key_manager import KeyManager
from prsm.storage.models import ContentHash, KeyShare
from prsm.storage.exceptions import KeyReconstructionError


@pytest.fixture
def km():
    return KeyManager()


class TestAESEncryption:
    def test_encrypt_decrypt_roundtrip(self, km):
        key = os.urandom(32)  # AES-256
        plaintext = b"secret manifest data"
        ciphertext = km.encrypt(key, plaintext)
        assert ciphertext != plaintext
        decrypted = km.decrypt(key, ciphertext)
        assert decrypted == plaintext

    def test_wrong_key_fails(self, km):
        key1 = os.urandom(32)
        key2 = os.urandom(32)
        ciphertext = km.encrypt(key1, b"data")
        with pytest.raises(Exception):
            km.decrypt(key2, ciphertext)

    def test_ciphertext_includes_nonce(self, km):
        key = os.urandom(32)
        ct1 = km.encrypt(key, b"same data")
        ct2 = km.encrypt(key, b"same data")
        # Different nonces -> different ciphertexts
        assert ct1 != ct2

    def test_empty_plaintext(self, km):
        key = os.urandom(32)
        ciphertext = km.encrypt(key, b"")
        assert km.decrypt(key, ciphertext) == b""

    def test_large_plaintext(self, km):
        key = os.urandom(32)
        data = os.urandom(1024 * 100)
        ciphertext = km.encrypt(key, data)
        assert km.decrypt(key, ciphertext) == data


class TestShamirSecretSharing:
    def test_split_produces_n_shares(self, km):
        secret = os.urandom(32)
        shares = km.split_secret(secret, threshold=3, num_shares=5)
        assert len(shares) == 5

    def test_threshold_shares_reconstruct(self, km):
        secret = os.urandom(32)
        shares = km.split_secret(secret, threshold=3, num_shares=5)
        # Use exactly threshold shares
        reconstructed = km.reconstruct_secret(shares[:3], threshold=3)
        assert reconstructed == secret

    def test_more_than_threshold_shares_reconstruct(self, km):
        secret = os.urandom(32)
        shares = km.split_secret(secret, threshold=3, num_shares=5)
        reconstructed = km.reconstruct_secret(shares, threshold=3)
        assert reconstructed == secret

    def test_fewer_than_threshold_fails(self, km):
        secret = os.urandom(32)
        shares = km.split_secret(secret, threshold=3, num_shares=5)
        # Only 2 shares when threshold is 3
        result = km.reconstruct_secret(shares[:2], threshold=3)
        # With fewer shares, reconstruction produces wrong secret
        assert result != secret

    def test_any_k_of_n_combination(self, km):
        """Any K shares from the N total should reconstruct the secret."""
        import itertools
        secret = os.urandom(32)
        shares = km.split_secret(secret, threshold=3, num_shares=5)
        for combo in itertools.combinations(shares, 3):
            reconstructed = km.reconstruct_secret(list(combo), threshold=3)
            assert reconstructed == secret

    def test_different_threshold_sizes(self, km):
        for k, n in [(2, 3), (3, 5), (5, 8), (7, 12)]:
            secret = os.urandom(32)
            shares = km.split_secret(secret, threshold=k, num_shares=n)
            assert len(shares) == n
            reconstructed = km.reconstruct_secret(shares[:k], threshold=k)
            assert reconstructed == secret

    def test_single_byte_secret(self, km):
        secret = b"\x42"
        shares = km.split_secret(secret, threshold=2, num_shares=3)
        reconstructed = km.reconstruct_secret(shares[:2], threshold=2)
        assert reconstructed == secret


class TestKeyManagerIntegration:
    def test_generate_encrypt_split_reconstruct_decrypt(self, km):
        """Full flow: generate key, encrypt manifest, split key, reconstruct, decrypt."""
        manifest_data = b'{"content_hash": "01abc...", "shard_hashes": ["01def..."]}'

        # Generate a random encryption key
        key = km.generate_key()
        assert len(key) == 32

        # Encrypt manifest
        ciphertext = km.encrypt(key, manifest_data)

        # Split key into shares
        shares = km.split_secret(key, threshold=3, num_shares=5)

        # Simulate: only 3 of 5 shares available
        reconstructed_key = km.reconstruct_secret(shares[:3], threshold=3)

        # Decrypt manifest
        decrypted = km.decrypt(reconstructed_key, ciphertext)
        assert decrypted == manifest_data

    def test_create_key_shares_for_content(self, km):
        """Test the high-level create_key_shares helper."""
        content_hash = ContentHash.from_data(b"my content")
        manifest_data = b"manifest json"

        ciphertext, key_shares = km.encrypt_manifest(
            manifest_data=manifest_data,
            content_hash=content_hash,
            threshold=3,
            num_shares=5,
        )

        assert len(key_shares) == 5
        for i, share in enumerate(key_shares):
            assert share.content_hash == content_hash
            assert share.share_index == i + 1
            assert share.threshold == 3
            assert share.total_shares == 5

        # Reconstruct and decrypt
        decrypted = km.decrypt_manifest(ciphertext, key_shares[:3])
        assert decrypted == manifest_data

    def test_key_refresh(self, km):
        """Reconstruct key, re-split with new shares, verify still works."""
        content_hash = ContentHash.from_data(b"refresh test")
        manifest_data = b"manifest to persist"

        ciphertext, original_shares = km.encrypt_manifest(
            manifest_data, content_hash, threshold=3, num_shares=5
        )

        # Reconstruct key from original shares
        key = km.reconstruct_secret(
            [(s.share_index, s.share_data) for s in original_shares[:3]],
            threshold=3,
        )

        # Re-split into new shares (simulating key refresh)
        new_raw_shares = km.split_secret(key, threshold=3, num_shares=5)
        new_key = km.reconstruct_secret(new_raw_shares[:3], threshold=3)

        # Decrypt with refreshed key
        decrypted = km.decrypt(new_key, ciphertext)
        assert decrypted == manifest_data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/storage/test_key_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prsm.storage.key_manager'`

- [ ] **Step 3: Implement KeyManager**

```python
# prsm/storage/key_manager.py
"""Key manager: AES-256-GCM encryption + Shamir's Secret Sharing over GF(256).

Manifest encryption uses a per-content random AES-256-GCM key. That key is then
split using Shamir's Secret Sharing so that K-of-N shares can reconstruct it.
No single node holds the complete key.

Shamir's implementation operates over GF(2^8) using the AES irreducible polynomial
x^8 + x^4 + x^3 + x + 1 (0x11B). This is a well-understood finite field suitable
for byte-level secret sharing.
"""
from __future__ import annotations

import os
import secrets
from typing import List, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from prsm.storage.models import ContentHash, KeyShare


# ── GF(256) arithmetic ─────────────────────────────────────────────────

# Precompute log and exp tables for GF(2^8) with generator 3,
# using the AES irreducible polynomial 0x11B.
_EXP = [0] * 256
_LOG = [0] * 256

def _init_gf256_tables() -> None:
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        x = x ^ (x << 1)
        if x & 0x100:
            x ^= 0x11B
    # _EXP[255] = _EXP[0] for wraparound convenience
    _EXP[255] = _EXP[0]

_init_gf256_tables()


def _gf256_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[(_LOG[a] + _LOG[b]) % 255]


def _gf256_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("No inverse for 0 in GF(256)")
    return _EXP[255 - _LOG[a]]


# ── Shamir's Secret Sharing ────────────────────────────────────────────

def _evaluate_polynomial(coeffs: List[int], x: int) -> int:
    """Evaluate polynomial at x in GF(256). coeffs[0] is the secret (constant term)."""
    result = 0
    for coeff in reversed(coeffs):
        result = _gf256_mul(result, x) ^ coeff
    return result


def _lagrange_interpolate(shares: List[Tuple[int, int]], x: int = 0) -> int:
    """Lagrange interpolation at x in GF(256). shares = [(x_i, y_i), ...]."""
    result = 0
    for i, (xi, yi) in enumerate(shares):
        basis = yi
        for j, (xj, _) in enumerate(shares):
            if i == j:
                continue
            # basis *= (x - xj) / (xi - xj)
            num = x ^ xj
            den = xi ^ xj
            basis = _gf256_mul(basis, _gf256_mul(num, _gf256_inv(den)))
        result ^= basis
    return result


class KeyManager:
    """AES-256-GCM encryption and Shamir's Secret Sharing for manifest keys."""

    def generate_key(self) -> bytes:
        """Generate a random 256-bit AES key."""
        return os.urandom(32)

    def encrypt(self, key: bytes, plaintext: bytes) -> bytes:
        """Encrypt with AES-256-GCM. Returns nonce (12 bytes) + ciphertext + tag."""
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext

    def decrypt(self, key: bytes, data: bytes) -> bytes:
        """Decrypt AES-256-GCM. Input is nonce (12 bytes) + ciphertext + tag."""
        nonce = data[:12]
        ciphertext = data[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, None)

    def split_secret(
        self, secret: bytes, threshold: int, num_shares: int
    ) -> List[Tuple[int, bytes]]:
        """Split a secret into num_shares using Shamir's Secret Sharing over GF(256).

        Returns list of (share_index, share_data) tuples.
        share_index is 1-based (x-coordinates 1..num_shares).
        """
        if threshold > num_shares:
            raise ValueError(f"threshold ({threshold}) > num_shares ({num_shares})")
        if threshold < 2:
            raise ValueError(f"threshold must be >= 2, got {threshold}")

        shares: List[Tuple[int, bytes]] = []
        for share_idx in range(1, num_shares + 1):
            share_bytes = bytearray(len(secret))
            for byte_pos in range(len(secret)):
                # Random polynomial of degree (threshold - 1) with secret byte as constant
                coeffs = [secret[byte_pos]] + [
                    secrets.randbelow(256) for _ in range(threshold - 1)
                ]
                share_bytes[byte_pos] = _evaluate_polynomial(coeffs, share_idx)
            shares.append((share_idx, bytes(share_bytes)))

        return shares

    def reconstruct_secret(
        self, shares: List[Tuple[int, bytes]], threshold: int
    ) -> bytes:
        """Reconstruct a secret from K-of-N shares via Lagrange interpolation."""
        if len(shares) < threshold:
            # Will produce wrong result, but we don't raise — caller must verify
            pass
        secret_len = len(shares[0][1])
        result = bytearray(secret_len)
        for byte_pos in range(secret_len):
            points = [(x, share_data[byte_pos]) for x, share_data in shares]
            result[byte_pos] = _lagrange_interpolate(points[:threshold], x=0)
        return bytes(result)

    def encrypt_manifest(
        self,
        manifest_data: bytes,
        content_hash: ContentHash,
        threshold: int,
        num_shares: int,
    ) -> Tuple[bytes, List[KeyShare]]:
        """Encrypt manifest data and split the key into Shamir shares.

        Returns (ciphertext, list_of_KeyShare).
        """
        key = self.generate_key()
        ciphertext = self.encrypt(key, manifest_data)
        raw_shares = self.split_secret(key, threshold, num_shares)

        key_shares = [
            KeyShare(
                content_hash=content_hash,
                share_index=idx,
                share_data=share_data,
                threshold=threshold,
                total_shares=num_shares,
                algorithm_id=0x01,
            )
            for idx, share_data in raw_shares
        ]

        return ciphertext, key_shares

    def decrypt_manifest(
        self, ciphertext: bytes, key_shares: List[KeyShare]
    ) -> bytes:
        """Reconstruct key from shares and decrypt manifest."""
        if not key_shares:
            from prsm.storage.exceptions import KeyReconstructionError
            raise KeyReconstructionError("No key shares provided")

        threshold = key_shares[0].threshold
        raw_shares = [(ks.share_index, ks.share_data) for ks in key_shares]
        key = self.reconstruct_secret(raw_shares, threshold)
        return self.decrypt(key, ciphertext)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/storage/test_key_manager.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/storage/key_manager.py tests/unit/storage/test_key_manager.py
git commit -m "feat(storage): add KeyManager with Shamir's Secret Sharing and AES-256-GCM"
```

---

### Task 5: Distribution Manager

**Files:**
- Create: `prsm/storage/distribution.py`
- Create: `tests/unit/storage/test_distribution.py`

This is the largest component — shard placement, ContentDescriptor management, public no-ticket share release, contract key signing, degraded mode, and the replication health monitor. It integrates with libp2p transport and discovery.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/storage/test_distribution.py
"""Tests for prsm.storage.distribution — placement, retrieval, health monitoring."""
import asyncio
import os
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat, NoEncryption, PrivateFormat,
)

from prsm.storage.distribution import DistributionManager, ShardPlacement
from prsm.storage.models import (
    ContentHash, ShardManifest, ContentDescriptor,
    ReplicationPolicy, AlgorithmID,
)
from prsm.storage.key_manager import KeyManager
from prsm.storage.blob_store import BlobStore
from prsm.storage.exceptions import PlacementError


# ── Test helpers ───────────────────────────────────────────────────────

@dataclass
class FakePeerInfo:
    node_id: str
    address: str = ""
    capabilities: List[str] = field(default_factory=lambda: ["storage"])
    reliability_score: float = 1.0
    asn: str = "AS0"  # Autonomous system number for placement tests


class FakeDiscovery:
    def __init__(self, peers: List[FakePeerInfo]):
        self._peers = {p.node_id: p for p in peers}

    def find_peers_by_capability(self, required, match_all=True):
        return [p for p in self._peers.values() if "storage" in p.capabilities]

    def record_job_success(self, node_id):
        pass

    def record_job_failure(self, node_id):
        pass


class FakeTransport:
    def __init__(self):
        self.sent_messages = []
        self._handlers = {}

    async def send_to_peer(self, peer_id, msg):
        self.sent_messages.append((peer_id, msg))
        return True

    async def dht_provide(self, key):
        return True

    async def dht_find_providers(self, key, limit=20):
        return []

    def on_message(self, msg_type, handler):
        self._handlers[msg_type] = handler


@pytest.fixture
def blob_store(tmp_path):
    return BlobStore(data_dir=str(tmp_path / "blobs"))


@pytest.fixture
def key_manager():
    return KeyManager()


@pytest.fixture
def owner_keys():
    """Generate test Ed25519 keypair for owner signing."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


# ── Placement tests ───────────────────────────────────────────────────

class TestShardPlacement:
    def test_owner_excluded(self):
        peers = [
            FakePeerInfo(node_id="owner", asn="AS1"),
            FakePeerInfo(node_id="peer-a", asn="AS2"),
            FakePeerInfo(node_id="peer-b", asn="AS3"),
            FakePeerInfo(node_id="peer-c", asn="AS4"),
        ]
        discovery = FakeDiscovery(peers)
        dm = DistributionManager(
            node_id="owner",
            discovery=discovery,
            transport=FakeTransport(),
            blob_store=BlobStore(data_dir="/tmp/unused"),
            key_manager=KeyManager(),
        )
        placement = dm._compute_shard_placement(
            shard_hashes=[ContentHash.from_data(b"s1")],
            replication_factor=2,
            owner_node_id="owner",
            peer_asn_map={"peer-a": "AS2", "peer-b": "AS3", "peer-c": "AS4"},
        )
        for shard_hex, node_ids in placement.shard_assignments.items():
            assert "owner" not in node_ids

    def test_asn_diversity(self):
        peers = [
            FakePeerInfo(node_id="p1", asn="AS1"),
            FakePeerInfo(node_id="p2", asn="AS1"),
            FakePeerInfo(node_id="p3", asn="AS2"),
            FakePeerInfo(node_id="p4", asn="AS3"),
        ]
        discovery = FakeDiscovery(peers)
        dm = DistributionManager(
            node_id="owner",
            discovery=discovery,
            transport=FakeTransport(),
            blob_store=BlobStore(data_dir="/tmp/unused"),
            key_manager=KeyManager(),
        )
        shard = ContentHash.from_data(b"diversity-test")
        placement = dm._compute_shard_placement(
            shard_hashes=[shard],
            replication_factor=3,
            owner_node_id="owner",
            peer_asn_map={"p1": "AS1", "p2": "AS1", "p3": "AS2", "p4": "AS3"},
        )
        assigned = placement.shard_assignments[shard.hex()]
        asns = set()
        asn_map = {"p1": "AS1", "p2": "AS1", "p3": "AS2", "p4": "AS3"}
        for node_id in assigned:
            asns.add(asn_map[node_id])
        # Should use at least 2 distinct ASNs (3 ideally, but p1/p2 share AS1)
        assert len(asns) >= 2

    def test_replication_factor_honored(self):
        peers = [FakePeerInfo(node_id=f"p{i}", asn=f"AS{i}") for i in range(6)]
        discovery = FakeDiscovery(peers)
        dm = DistributionManager(
            node_id="owner",
            discovery=discovery,
            transport=FakeTransport(),
            blob_store=BlobStore(data_dir="/tmp/unused"),
            key_manager=KeyManager(),
        )
        shard = ContentHash.from_data(b"replicas")
        peer_asn_map = {f"p{i}": f"AS{i}" for i in range(6)}
        placement = dm._compute_shard_placement(
            shard_hashes=[shard],
            replication_factor=3,
            owner_node_id="owner",
            peer_asn_map=peer_asn_map,
        )
        assert len(placement.shard_assignments[shard.hex()]) == 3

    def test_key_shard_separation(self):
        peers = [FakePeerInfo(node_id=f"p{i}", asn=f"AS{i}") for i in range(10)]
        discovery = FakeDiscovery(peers)
        dm = DistributionManager(
            node_id="owner",
            discovery=discovery,
            transport=FakeTransport(),
            blob_store=BlobStore(data_dir="/tmp/unused"),
            key_manager=KeyManager(),
        )
        shard = ContentHash.from_data(b"separation")
        peer_asn_map = {f"p{i}": f"AS{i}" for i in range(10)}
        placement = dm._compute_shard_placement(
            shard_hashes=[shard],
            replication_factor=3,
            owner_node_id="owner",
            peer_asn_map=peer_asn_map,
        )
        shard_nodes = set(placement.shard_assignments[shard.hex()])
        key_nodes = set(placement.key_share_holders)
        # No overlap between shard holders and key share holders
        assert shard_nodes.isdisjoint(key_nodes)


# ── Degraded mode tests ───────────────────────────────────────────────

class TestDegradedMode:
    def test_asn_relaxation(self):
        """When too few ASN groups, relax ASN constraint but still use different nodes."""
        peers = [
            FakePeerInfo(node_id="p1", asn="AS1"),
            FakePeerInfo(node_id="p2", asn="AS1"),
            FakePeerInfo(node_id="p3", asn="AS1"),
        ]
        discovery = FakeDiscovery(peers)
        dm = DistributionManager(
            node_id="owner",
            discovery=discovery,
            transport=FakeTransport(),
            blob_store=BlobStore(data_dir="/tmp/unused"),
            key_manager=KeyManager(),
        )
        shard = ContentHash.from_data(b"degraded")
        placement = dm._compute_shard_placement(
            shard_hashes=[shard],
            replication_factor=2,
            owner_node_id="owner",
            peer_asn_map={"p1": "AS1", "p2": "AS1", "p3": "AS1"},
        )
        assert len(placement.shard_assignments[shard.hex()]) == 2
        assert "asn_relaxed" in placement.degraded_constraints

    def test_reject_when_too_few_nodes(self):
        """When fewer than 3 total nodes, reject placement."""
        peers = [FakePeerInfo(node_id="p1", asn="AS1")]
        discovery = FakeDiscovery(peers)
        dm = DistributionManager(
            node_id="owner",
            discovery=discovery,
            transport=FakeTransport(),
            blob_store=BlobStore(data_dir="/tmp/unused"),
            key_manager=KeyManager(),
        )
        shard = ContentHash.from_data(b"too-small")
        with pytest.raises(PlacementError, match="Placement failed"):
            dm._compute_shard_placement(
                shard_hashes=[shard],
                replication_factor=3,
                owner_node_id="owner",
                peer_asn_map={"p1": "AS1"},
            )


# ── Descriptor signing tests ──────────────────────────────────────────

class TestDescriptorSigning:
    def test_owner_signed_descriptor(self, owner_keys):
        private_key, public_key = owner_keys
        dm = DistributionManager(
            node_id="owner",
            discovery=FakeDiscovery([]),
            transport=FakeTransport(),
            blob_store=BlobStore(data_dir="/tmp/unused"),
            key_manager=KeyManager(),
        )
        descriptor = dm._create_descriptor_stub(
            content_hash=ContentHash.from_data(b"desc-test"),
            owner_node_id="owner",
            visibility="public",
            replication_policy=ReplicationPolicy(replication_factor=3),
            contract_pubkey=public_key.public_bytes(Encoding.Raw, PublicFormat.Raw),
        )
        signed = dm._sign_descriptor(descriptor, private_key, signer_type="owner")
        assert signed.signer_type == "owner"
        assert signed.signature != b""
        # Verification should pass
        assert dm._verify_descriptor_signature(signed, public_key)

    def test_contract_key_cannot_change_owner(self):
        """Contract key updates must not change owner, epoch, visibility, or policy."""
        dm = DistributionManager(
            node_id="owner",
            discovery=FakeDiscovery([]),
            transport=FakeTransport(),
            blob_store=BlobStore(data_dir="/tmp/unused"),
            key_manager=KeyManager(),
        )
        # The _validate_contract_update method should reject these
        base = dm._create_descriptor_stub(
            content_hash=ContentHash.from_data(b"immutable"),
            owner_node_id="owner",
            visibility="public",
            replication_policy=ReplicationPolicy(replication_factor=3),
            contract_pubkey=b"pubkey",
        )
        # Try to change owner_node_id
        modified = dm._create_descriptor_stub(
            content_hash=ContentHash.from_data(b"immutable"),
            owner_node_id="attacker",
            visibility="public",
            replication_policy=ReplicationPolicy(replication_factor=3),
            contract_pubkey=b"pubkey",
        )
        assert not dm._validate_contract_update(base, modified)

    def test_conflict_resolution_highest_epoch_version(self):
        dm = DistributionManager(
            node_id="owner",
            discovery=FakeDiscovery([]),
            transport=FakeTransport(),
            blob_store=BlobStore(data_dir="/tmp/unused"),
            key_manager=KeyManager(),
        )
        ch = ContentHash.from_data(b"conflict")
        older = dm._create_descriptor_stub(
            content_hash=ch, owner_node_id="o", visibility="public",
            replication_policy=ReplicationPolicy(replication_factor=3),
            contract_pubkey=b"pk",
        )
        older.epoch = 1
        older.version = 5

        newer = dm._create_descriptor_stub(
            content_hash=ch, owner_node_id="o", visibility="public",
            replication_policy=ReplicationPolicy(replication_factor=3),
            contract_pubkey=b"pk",
        )
        newer.epoch = 1
        newer.version = 6

        assert dm._resolve_conflict(older, newer) is newer
        assert dm._resolve_conflict(newer, older) is newer

        # Higher epoch always wins
        much_newer = dm._create_descriptor_stub(
            content_hash=ch, owner_node_id="o", visibility="public",
            replication_policy=ReplicationPolicy(replication_factor=3),
            contract_pubkey=b"pk",
        )
        much_newer.epoch = 2
        much_newer.version = 1
        assert dm._resolve_conflict(newer, much_newer) is much_newer
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/storage/test_distribution.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prsm.storage.distribution'`

- [ ] **Step 3: Implement DistributionManager**

```python
# prsm/storage/distribution.py
"""Distribution manager: shard placement, descriptor management, retrieval, health monitoring.

Handles:
- Shard placement with owner exclusion, ASN diversity, key-shard separation
- ContentDescriptor creation and signing (owner key + contract key)
- Degraded mode constraint relaxation for sparse networks
- P2P shard retrieval (pull model)
- Replication health monitoring
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.exceptions import InvalidSignature

from prsm.storage.exceptions import PlacementError
from prsm.storage.models import ContentDescriptor, ContentHash, ReplicationPolicy
from prsm.storage.key_manager import KeyManager
from prsm.storage.blob_store import BlobStore


@dataclass
class ShardPlacement:
    """Result of shard placement computation."""
    shard_assignments: Dict[str, List[str]]  # shard_hash hex -> [node_ids]
    key_share_holders: List[str]
    contract_key_share_holders: List[str]
    degraded_constraints: List[str] = field(default_factory=list)


class DistributionManager:
    """Manages shard placement, descriptor lifecycle, retrieval, and health."""

    MIN_NETWORK_NODES = 3  # Minimum to accept uploads

    def __init__(
        self,
        node_id: str,
        discovery: Any,
        transport: Any,
        blob_store: BlobStore,
        key_manager: KeyManager,
    ) -> None:
        self.node_id = node_id
        self.discovery = discovery
        self.transport = transport
        self.blob_store = blob_store
        self.key_manager = key_manager

    # ── Shard Placement ────────────────────────────────────────────────

    def _compute_shard_placement(
        self,
        shard_hashes: List[ContentHash],
        replication_factor: int,
        owner_node_id: str,
        peer_asn_map: Dict[str, str],
    ) -> ShardPlacement:
        """Compute shard and key-share placement across available peers.

        Constraints (in relaxation order):
        1. Owner exclusion (never relaxed)
        2. ASN diversity (relaxed first)
        3. Key-shard separation (never relaxed)
        """
        # Filter out owner
        eligible = [pid for pid in peer_asn_map if pid != owner_node_id]

        if len(eligible) < self.MIN_NETWORK_NODES:
            raise PlacementError(
                f"Need at least {self.MIN_NETWORK_NODES} non-owner peers, "
                f"have {len(eligible)}",
                min_nodes_needed=self.MIN_NETWORK_NODES,
            )

        degraded = []

        # Group by ASN
        asn_groups: Dict[str, List[str]] = {}
        for pid in eligible:
            asn = peer_asn_map.get(pid, "unknown")
            asn_groups.setdefault(asn, []).append(pid)

        # Place shards
        shard_assignments: Dict[str, List[str]] = {}
        all_shard_nodes: Set[str] = set()

        for shard_hash in shard_hashes:
            assigned = self._place_single_shard(
                eligible, asn_groups, peer_asn_map, replication_factor, degraded
            )
            shard_assignments[shard_hash.hex()] = assigned
            all_shard_nodes.update(assigned)

        # Place key shares on nodes that hold NO shards
        non_shard_nodes = [pid for pid in eligible if pid not in all_shard_nodes]

        if len(non_shard_nodes) < 3:
            # Not enough non-shard nodes — use shard nodes but different from
            # the specific shard they're key-holding for (best effort)
            non_shard_nodes = eligible
            if "key_shard_separation_relaxed" not in degraded:
                degraded.append("key_shard_separation_relaxed")

        # Select key share holders (pick up to 5, or available)
        key_share_count = min(5, len(non_shard_nodes))
        key_share_holders = non_shard_nodes[:key_share_count]

        # Contract key share holders — can overlap with manifest key holders
        contract_key_count = min(5, len(non_shard_nodes))
        contract_key_holders = non_shard_nodes[:contract_key_count]

        return ShardPlacement(
            shard_assignments=shard_assignments,
            key_share_holders=key_share_holders,
            contract_key_share_holders=contract_key_holders,
            degraded_constraints=degraded,
        )

    def _place_single_shard(
        self,
        eligible: List[str],
        asn_groups: Dict[str, List[str]],
        peer_asn_map: Dict[str, str],
        replication_factor: int,
        degraded: List[str],
    ) -> List[str]:
        """Place one shard across replication_factor nodes with ASN diversity."""
        assigned: List[str] = []
        used_asns: Set[str] = set()

        # First pass: pick from distinct ASN groups
        for asn, nodes in sorted(asn_groups.items(), key=lambda x: -len(x[1])):
            if len(assigned) >= replication_factor:
                break
            if asn in used_asns:
                continue
            for node in nodes:
                if node not in assigned:
                    assigned.append(node)
                    used_asns.add(asn)
                    break

        # If we couldn't fill via distinct ASNs, relax and fill from any node
        if len(assigned) < replication_factor:
            if "asn_relaxed" not in degraded:
                degraded.append("asn_relaxed")
            for node in eligible:
                if len(assigned) >= replication_factor:
                    break
                if node not in assigned:
                    assigned.append(node)

        return assigned

    # ── Descriptor Management ──────────────────────────────────────────

    def _create_descriptor_stub(
        self,
        content_hash: ContentHash,
        owner_node_id: str,
        visibility: str,
        replication_policy: ReplicationPolicy,
        contract_pubkey: bytes,
    ) -> ContentDescriptor:
        """Create an unsigned descriptor stub."""
        now = time.time()
        return ContentDescriptor(
            content_hash=content_hash,
            manifest_holders=[],
            key_share_holders=[],
            contract_key_share_holders=[],
            shard_map={},
            replication_policy=replication_policy,
            visibility=visibility,
            epoch=1,
            version=1,
            owner_node_id=owner_node_id,
            contract_pubkey=contract_pubkey,
            signature=b"",
            signer_type="owner",
            created_at=now,
            updated_at=now,
        )

    def _descriptor_signing_data(self, descriptor: ContentDescriptor) -> bytes:
        """Deterministic bytes to sign for a descriptor."""
        import json
        data = {
            "content_hash": descriptor.content_hash.hex(),
            "manifest_holders": sorted(descriptor.manifest_holders),
            "key_share_holders": sorted(descriptor.key_share_holders),
            "contract_key_share_holders": sorted(descriptor.contract_key_share_holders),
            "shard_map": {k: sorted(v) for k, v in sorted(descriptor.shard_map.items())},
            "replication_factor": descriptor.replication_policy.replication_factor,
            "visibility": descriptor.visibility,
            "epoch": descriptor.epoch,
            "version": descriptor.version,
            "owner_node_id": descriptor.owner_node_id,
            "contract_pubkey": descriptor.contract_pubkey.hex(),
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

    def _sign_descriptor(
        self,
        descriptor: ContentDescriptor,
        private_key: Ed25519PrivateKey,
        signer_type: str,
    ) -> ContentDescriptor:
        """Sign a descriptor with the given key."""
        signing_data = self._descriptor_signing_data(descriptor)
        descriptor.signature = private_key.sign(signing_data)
        descriptor.signer_type = signer_type
        return descriptor

    def _verify_descriptor_signature(
        self,
        descriptor: ContentDescriptor,
        public_key: Ed25519PublicKey,
    ) -> bool:
        """Verify a descriptor's signature."""
        signing_data = self._descriptor_signing_data(descriptor)
        try:
            public_key.verify(descriptor.signature, signing_data)
            return True
        except InvalidSignature:
            return False

    def _validate_contract_update(
        self,
        base: ContentDescriptor,
        updated: ContentDescriptor,
    ) -> bool:
        """Validate that a contract-key update only modifies allowed fields.

        Contract key CAN update: shard_map, manifest_holders, key_share_holders,
            contract_key_share_holders, updated_at, version
        Contract key CANNOT change: owner_node_id, epoch, visibility,
            replication_policy, contract_pubkey
        """
        if updated.owner_node_id != base.owner_node_id:
            return False
        if updated.epoch != base.epoch:
            return False
        if updated.visibility != base.visibility:
            return False
        if updated.replication_policy.replication_factor != base.replication_policy.replication_factor:
            return False
        if updated.contract_pubkey != base.contract_pubkey:
            return False
        return True

    def _resolve_conflict(
        self,
        a: ContentDescriptor,
        b: ContentDescriptor,
    ) -> ContentDescriptor:
        """Resolve conflict between two descriptors: highest (epoch, version) wins."""
        if (a.epoch, a.version) >= (b.epoch, b.version):
            return a
        return b
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/storage/test_distribution.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/storage/distribution.py tests/unit/storage/test_distribution.py
git commit -m "feat(storage): add DistributionManager with placement, signing, and degraded mode"
```

---

### Task 6: ContentStore Facade

**Files:**
- Modify: `prsm/storage/__init__.py`
- Create: `tests/unit/storage/test_content_store.py`

The `ContentStore` is the public API that ties blob store, shard engine, key manager, and distribution manager together.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/storage/test_content_store.py
"""Tests for ContentStore — the public API facade."""
import os
import pytest
from prsm.storage import ContentStore, ContentHash
from prsm.storage.exceptions import ContentNotFoundError


@pytest.fixture
def store(tmp_path):
    return ContentStore(
        data_dir=str(tmp_path / "storage"),
        node_id="test-node",
        shard_threshold=100,  # Low threshold for testing
        shard_size=50,
    )


class TestContentStoreLocal:
    """Test ContentStore in local-only mode (no transport/discovery)."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_small(self, store):
        data = b"small content"
        ch = await store.store_local(data)
        retrieved = await store.retrieve_local(ch)
        assert retrieved == data

    @pytest.mark.asyncio
    async def test_store_and_retrieve_large(self, store):
        data = os.urandom(200)  # Above shard threshold
        ch = await store.store_local(data)
        retrieved = await store.retrieve_local(ch)
        assert retrieved == data

    @pytest.mark.asyncio
    async def test_content_hash_returned(self, store):
        data = b"hash test"
        ch = await store.store_local(data)
        expected = ContentHash.from_data(data)
        assert ch == expected

    @pytest.mark.asyncio
    async def test_exists_local(self, store):
        data = b"exists check"
        ch = await store.store_local(data)
        assert await store.exists_local(ch) is True
        fake = ContentHash.from_data(b"nope")
        assert await store.exists_local(fake) is False

    @pytest.mark.asyncio
    async def test_delete_local(self, store):
        data = b"delete me"
        ch = await store.store_local(data)
        await store.delete_local(ch)
        assert await store.exists_local(ch) is False

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_raises(self, store):
        ch = ContentHash.from_data(b"ghost")
        with pytest.raises(ContentNotFoundError):
            await store.retrieve_local(ch)

    @pytest.mark.asyncio
    async def test_manifest_encrypted_and_recoverable(self, store):
        """Verify that the shard+encrypt+decrypt+reassemble pipeline works end-to-end."""
        data = os.urandom(200)
        ch, manifest, ciphertext, key_shares = await store._store_and_encrypt(
            data, owner_node_id="test-node", replication_factor=3
        )
        # Decrypt manifest using key shares
        decrypted_manifest = store.key_manager.decrypt_manifest(ciphertext, key_shares[:3])
        from prsm.storage.models import ShardManifest
        recovered = ShardManifest.from_json(decrypted_manifest.decode())
        assert recovered.content_hash == ch

    @pytest.mark.asyncio
    async def test_full_local_pipeline(self, store):
        """store_local -> retrieve_local exercises shard + reassemble for large content."""
        data = os.urandom(500)
        ch = await store.store_local(data)
        retrieved = await store.retrieve_local(ch)
        assert retrieved == data
        assert ContentHash.from_data(retrieved) == ch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/storage/test_content_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'ContentStore'`

- [ ] **Step 3: Implement ContentStore**

```python
# prsm/storage/content_store.py
"""ContentStore — public API facade for PRSM native storage.

Ties together blob store, shard engine, key manager, and distribution manager.
Provides local-only operations (store_local, retrieve_local) for use without
a P2P network, and distributed operations (store, retrieve) that integrate
with libp2p transport and discovery.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from prsm.storage.blob_store import BlobStore
from prsm.storage.exceptions import ContentNotFoundError
from prsm.storage.key_manager import KeyManager
from prsm.storage.models import ContentHash, KeyShare, ShardManifest


class ContentStore:
    """Public API for PRSM's native content-addressed storage."""

    def __init__(
        self,
        data_dir: str,
        node_id: str = "",
        shard_threshold: int = 1_048_576,   # 1MB
        shard_size: int = 262_144,           # 256KB
        transport: Optional[Any] = None,
        discovery: Optional[Any] = None,
    ) -> None:
        from prsm.storage.shard_engine import ShardEngine

        self.data_dir = data_dir
        self.node_id = node_id
        self.blob_store = BlobStore(data_dir=data_dir)
        self.shard_engine = ShardEngine(
            blob_store=self.blob_store,
            shard_threshold=shard_threshold,
            shard_size=shard_size,
        )
        self.key_manager = KeyManager()
        self.transport = transport
        self.discovery = discovery

        # In-memory manifest cache: content_hash hex -> (ciphertext, key_shares, manifest)
        self._manifest_cache: dict[str, Tuple[bytes, List[KeyShare], ShardManifest]] = {}

    # ── Internal pipeline ──────────────────────────────────────────────

    async def _store_and_encrypt(
        self,
        data: bytes,
        owner_node_id: str,
        replication_factor: int,
        visibility: str = "public",
    ) -> Tuple[ContentHash, ShardManifest, bytes, List[KeyShare]]:
        """Split into shards, encrypt manifest, return all artifacts."""
        manifest = await self.shard_engine.split(
            data, owner_node_id=owner_node_id,
            replication_factor=replication_factor,
            visibility=visibility,
        )

        # Determine threshold based on shard count
        shard_count = len(manifest.shard_hashes)
        if shard_count < 10:
            threshold, num_shares = 3, 5
        elif shard_count < 100:
            threshold, num_shares = 5, 8
        else:
            threshold, num_shares = 7, 12

        manifest_json = manifest.to_json().encode()
        ciphertext, key_shares = self.key_manager.encrypt_manifest(
            manifest_data=manifest_json,
            content_hash=manifest.content_hash,
            threshold=threshold,
            num_shares=num_shares,
        )

        return manifest.content_hash, manifest, ciphertext, key_shares

    # ── Local operations (no network) ──────────────────────────────────

    async def store_local(self, data: bytes, replication_factor: int = 3) -> ContentHash:
        """Store content locally. Shards stored in blob store, manifest cached in memory."""
        content_hash, manifest, ciphertext, key_shares = await self._store_and_encrypt(
            data, owner_node_id=self.node_id, replication_factor=replication_factor,
        )
        self._manifest_cache[content_hash.hex()] = (ciphertext, key_shares, manifest)
        return content_hash

    async def retrieve_local(self, content_hash: ContentHash) -> bytes:
        """Retrieve content from local store using cached manifest."""
        cache_key = content_hash.hex()
        if cache_key not in self._manifest_cache:
            raise ContentNotFoundError(content_hash.hex())

        _, _, manifest = self._manifest_cache[cache_key]
        return await self.shard_engine.reassemble(manifest)

    async def exists_local(self, content_hash: ContentHash) -> bool:
        """Check if content exists in local manifest cache."""
        return content_hash.hex() in self._manifest_cache

    async def delete_local(self, content_hash: ContentHash) -> None:
        """Delete content from local store."""
        cache_key = content_hash.hex()
        if cache_key in self._manifest_cache:
            _, _, manifest = self._manifest_cache[cache_key]
            for shard_hash in manifest.shard_hashes:
                await self.blob_store.delete(shard_hash)
            del self._manifest_cache[cache_key]
```

Update `prsm/storage/__init__.py` to export `ContentStore`:

```python
# prsm/storage/__init__.py
"""
PRSM Native Content-Addressed Storage

Replaces IPFS/Kubo dependency with PRSM-native storage.
"""
from prsm.storage.models import ContentHash
from prsm.storage.content_store import ContentStore
from prsm.storage.exceptions import (
    ContentNotFoundError,
    ShardIntegrityError,
    ManifestError,
    KeyReconstructionError,
    PlacementError,
    StorageError,
)

__all__ = [
    "ContentHash",
    "ContentStore",
    "ContentNotFoundError",
    "ShardIntegrityError",
    "ManifestError",
    "KeyReconstructionError",
    "PlacementError",
    "StorageError",
]
```

Also add `content_store.py` to the file structure (in the package layout).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/storage/test_content_store.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Run all storage tests together**

Run: `python -m pytest tests/unit/storage/ -v`
Expected: All tests across all 5 test files PASS

- [ ] **Step 6: Commit**

```bash
git add prsm/storage/__init__.py prsm/storage/content_store.py \
       tests/unit/storage/test_content_store.py
git commit -m "feat(storage): add ContentStore facade as public API"
```

---

### Task 7: Integration Tests

**Files:**
- Create: `tests/integration/test_native_storage.py`

- [ ] **Step 1: Write integration tests**

```python
# tests/integration/test_native_storage.py
"""Integration tests for the native storage module.

Tests the full pipeline: content -> shards -> encrypted manifest -> key shares ->
retrieval -> reassembly -> verification.
"""
import json
import math
import os
import pytest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from prsm.storage import ContentStore, ContentHash
from prsm.storage.blob_store import BlobStore
from prsm.storage.shard_engine import ShardEngine
from prsm.storage.key_manager import KeyManager
from prsm.storage.distribution import DistributionManager, ShardPlacement
from prsm.storage.models import (
    ShardManifest, KeyShare, ContentDescriptor, ReplicationPolicy, AlgorithmID,
)
from prsm.storage.exceptions import (
    ContentNotFoundError, ShardIntegrityError, PlacementError,
)


class TestFullLifecycle:
    """Store -> shard -> encrypt -> retrieve -> reassemble -> verify."""

    @pytest.mark.asyncio
    async def test_small_content_roundtrip(self, tmp_path):
        store = ContentStore(
            data_dir=str(tmp_path / "storage"),
            node_id="node-A",
            shard_threshold=100,
            shard_size=50,
        )
        data = b"small payload"
        ch = await store.store_local(data)
        retrieved = await store.retrieve_local(ch)
        assert retrieved == data

    @pytest.mark.asyncio
    async def test_large_content_sharded_roundtrip(self, tmp_path):
        store = ContentStore(
            data_dir=str(tmp_path / "storage"),
            node_id="node-A",
            shard_threshold=100,
            shard_size=50,
        )
        data = os.urandom(500)
        ch = await store.store_local(data)
        retrieved = await store.retrieve_local(ch)
        assert retrieved == data
        assert ContentHash.from_data(retrieved) == ch

    @pytest.mark.asyncio
    async def test_sharding_produces_correct_shard_count(self, tmp_path):
        store = ContentStore(
            data_dir=str(tmp_path / "storage"),
            node_id="node-A",
            shard_threshold=100,
            shard_size=50,
        )
        data = os.urandom(275)
        ch, manifest, _, _ = await store._store_and_encrypt(
            data, owner_node_id="node-A", replication_factor=3,
        )
        expected_shards = math.ceil(275 / 50)
        assert len(manifest.shard_hashes) == expected_shards

    @pytest.mark.asyncio
    async def test_tampered_shard_detected(self, tmp_path):
        store = ContentStore(
            data_dir=str(tmp_path / "storage"),
            node_id="node-A",
            shard_threshold=100,
            shard_size=50,
        )
        data = os.urandom(200)
        ch = await store.store_local(data)

        # Corrupt first shard
        _, _, manifest = store._manifest_cache[ch.hex()]
        first_shard = manifest.shard_hashes[0]
        path = store.blob_store._path_for(first_shard)
        with open(path, "wb") as f:
            f.write(b"CORRUPTED")

        with pytest.raises(ShardIntegrityError):
            await store.retrieve_local(ch)


class TestEncryptedManifestPipeline:
    """Test that manifest encryption/decryption works through the full pipeline."""

    @pytest.mark.asyncio
    async def test_manifest_survives_encrypt_decrypt(self, tmp_path):
        store = ContentStore(
            data_dir=str(tmp_path / "storage"),
            node_id="node-A",
            shard_threshold=100,
            shard_size=50,
        )
        data = os.urandom(300)
        ch, manifest, ciphertext, key_shares = await store._store_and_encrypt(
            data, owner_node_id="node-A", replication_factor=3,
        )

        # Use threshold shares to decrypt
        threshold = key_shares[0].threshold
        decrypted = store.key_manager.decrypt_manifest(
            ciphertext, key_shares[:threshold]
        )
        recovered_manifest = ShardManifest.from_json(decrypted.decode())

        assert recovered_manifest.content_hash == ch
        assert recovered_manifest.shard_hashes == manifest.shard_hashes

        # Reassemble from recovered manifest
        reassembled = await store.shard_engine.reassemble(recovered_manifest)
        assert reassembled == data

    @pytest.mark.asyncio
    async def test_any_k_shares_decrypt_manifest(self, tmp_path):
        import itertools
        store = ContentStore(
            data_dir=str(tmp_path / "storage"),
            node_id="node-A",
            shard_threshold=100,
            shard_size=50,
        )
        data = os.urandom(200)
        ch, manifest, ciphertext, key_shares = await store._store_and_encrypt(
            data, owner_node_id="node-A", replication_factor=3,
        )
        threshold = key_shares[0].threshold

        # Every combination of K shares should work
        for combo in itertools.combinations(key_shares, threshold):
            decrypted = store.key_manager.decrypt_manifest(ciphertext, list(combo))
            recovered = ShardManifest.from_json(decrypted.decode())
            assert recovered.content_hash == ch


class TestContentHashAlgorithmAgility:
    """Verify algorithm prefix byte behavior."""

    def test_sha256_prefix(self):
        ch = ContentHash.from_data(b"algo test")
        hex_str = ch.hex()
        assert hex_str[:2] == "01"
        assert len(hex_str) == 66

    def test_roundtrip_preserves_algorithm(self):
        ch = ContentHash.from_data(b"roundtrip algo")
        restored = ContentHash.from_hex(ch.hex())
        assert restored.algorithm_id == AlgorithmID.SHA256
        assert restored == ch


class TestDescriptorSigning:
    """Integration tests for descriptor signing and verification."""

    def test_owner_sign_and_verify(self):
        from prsm.storage.distribution import DistributionManager
        from dataclasses import dataclass, field as dfield
        from typing import List as TList

        @dataclass
        class FakePeer:
            node_id: str
            capabilities: TList[str] = dfield(default_factory=lambda: ["storage"])

        class FakeDisc:
            def find_peers_by_capability(self, *a, **kw):
                return []

        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        pubkey_bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

        dm = DistributionManager(
            node_id="owner",
            discovery=FakeDisc(),
            transport=None,
            blob_store=BlobStore(data_dir="/tmp/unused"),
            key_manager=KeyManager(),
        )

        ch = ContentHash.from_data(b"descriptor test")
        descriptor = dm._create_descriptor_stub(
            content_hash=ch,
            owner_node_id="owner",
            visibility="public",
            replication_policy=ReplicationPolicy(replication_factor=3),
            contract_pubkey=pubkey_bytes,
        )
        signed = dm._sign_descriptor(descriptor, private_key, signer_type="owner")

        # Verify
        assert dm._verify_descriptor_signature(signed, public_key)

        # Tamper and verify fails
        signed.owner_node_id = "attacker"
        assert not dm._verify_descriptor_signature(signed, public_key)
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/integration/test_native_storage.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run the entire storage test suite**

Run: `python -m pytest tests/unit/storage/ tests/integration/test_native_storage.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_native_storage.py
git commit -m "test(storage): add integration tests for native storage pipeline"
```

---

## Self-Review

**1. Spec coverage check:**
- Section 1 (Module structure, content hash format, blob store): Task 1 + Task 2 ✅
- Section 2 (Shard engine, manifest, split/reassemble): Task 3 ✅
- Section 3 (Key manager, Shamir's, AES-256-GCM, authorization model, reconstruction): Task 4 ✅
- Section 3 authorization (public no-ticket, contract-key gating): Task 5 `_validate_contract_update` ✅. Full P2P share release flow is a distribution concern that requires network integration — the local cryptographic primitives are complete. P2P share request/release handlers will be wired during consumer migration (Plan 2) when the storage module connects to node.py and the transport layer.
- Section 4 (ContentDescriptor, placement, signing, degraded mode, conflict resolution): Task 5 ✅
- Section 4 (Proof of custody): Covered by migration plan (Plan 2) — storage_proofs.py field renames and BlobStore integration. The proof system already exists; it just needs CID→ContentHash renames.
- Section 4 (Replication health monitor): The periodic sweep logic will be wired in Plan 2 when the distribution manager connects to the node lifecycle. The placement primitives (Task 5) support it.
- Section 4 (FTNS payment integration): Covered by Plan 2 — connects to existing content_economy.py.
- Section 5 (Consumer migration): Entirely Plan 2.
- Section 6 (Testing): Tasks 1-7 cover all unit test categories. Integration tests in Task 7.

**2. Placeholder scan:** No TBD, TODO, or "implement later" found. All code blocks are complete.

**3. Type consistency check:**
- `ContentHash.from_data()` used consistently across all tasks ✅
- `ContentHash.hex()` / `ContentHash.from_hex()` used consistently ✅
- `ShardManifest.to_json()` / `ShardManifest.from_json()` used consistently ✅
- `KeyManager.encrypt_manifest()` returns `Tuple[bytes, List[KeyShare]]` — used correctly in Task 6 ✅
- `KeyManager.decrypt_manifest()` takes `(ciphertext, key_shares)` — used correctly ✅
- `DistributionManager._compute_shard_placement()` returns `ShardPlacement` — used correctly in tests ✅
- `BlobStore.store()` returns `ContentHash`, `retrieve()` returns `bytes` — consistent ✅
