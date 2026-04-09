"""
ContentStore — public API facade for the PRSM native storage module.

Ties together BlobStore (local file I/O), ShardEngine (split/reassemble),
and KeyManager (AES-256-GCM + Shamir) into a single high-level interface.

Usage::

    store = ContentStore(data_dir="/var/prsm/storage", node_id="node-1")
    content_hash = await store.store_local(raw_bytes, replication_factor=3)
    data = await store.retrieve_local(content_hash)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from prsm.storage.blob_store import BlobStore
from prsm.storage.exceptions import ContentNotFoundError
from prsm.storage.key_manager import KeyManager
from prsm.storage.models import ContentHash, KeyShare, ShardManifest
from prsm.storage.shard_engine import ShardEngine


# ---------------------------------------------------------------------------
# Threshold policy helpers
# ---------------------------------------------------------------------------

def _select_threshold_params(num_shards: int) -> Tuple[int, int]:
    """Return (threshold, num_shares) based on the number of shards.

    Policy:
        <10 shards  -> 3-of-5
        10-99 shards -> 5-of-8
        100+ shards -> 7-of-12
    """
    if num_shards < 10:
        return 3, 5
    elif num_shards < 100:
        return 5, 8
    else:
        return 7, 12


# ---------------------------------------------------------------------------
# ContentStore
# ---------------------------------------------------------------------------

class ContentStore:
    """
    High-level content storage facade.

    Orchestrates sharding, encryption, and manifest management for locally
    stored content.  Network transport and discovery are accepted as optional
    parameters for future use but are not exercised in the local-only methods.

    Parameters
    ----------
    data_dir:
        Root directory used by the underlying BlobStore.
    node_id:
        Logical identifier for this storage node (used in manifest metadata).
    shard_threshold:
        Content <= this many bytes is stored as a single shard.
        Default: 1 MiB (1_048_576 bytes).
    shard_size:
        Target shard size in bytes for content that exceeds *shard_threshold*.
        Default: 256 KiB (262_144 bytes).
    transport:
        Optional network transport handle (reserved for future P2P use).
    discovery:
        Optional peer-discovery handle (reserved for future P2P use).
    """

    def __init__(
        self,
        data_dir: str,
        node_id: str = "",
        shard_threshold: int = 1_048_576,
        shard_size: int = 262_144,
        transport: Optional[Any] = None,
        discovery: Optional[Any] = None,
    ) -> None:
        self.data_dir = data_dir
        self.node_id = node_id
        self.transport = transport
        self.discovery = discovery

        self.blob_store = BlobStore(data_dir=data_dir)
        self.shard_engine = ShardEngine(
            blob_store=self.blob_store,
            shard_threshold=shard_threshold,
            shard_size=shard_size,
        )
        self.key_manager = KeyManager()

        # content_hash hex -> (ciphertext, key_shares, manifest)
        self._manifest_cache: Dict[str, Tuple[bytes, List[KeyShare], ShardManifest]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _store_and_encrypt(
        self,
        data: bytes,
        owner_node_id: str,
        replication_factor: int,
        visibility: str = "public",
    ) -> Tuple[ContentHash, ShardManifest, bytes, List[KeyShare]]:
        """Shard *data*, encrypt the manifest, and return all artefacts.

        Returns
        -------
        (content_hash, manifest, ciphertext, key_shares)
            *content_hash*  — content-addressed identifier of the original data.
            *manifest*      — ShardManifest describing the stored shards.
            *ciphertext*    — AES-256-GCM encrypted serialised manifest.
            *key_shares*    — Shamir key shares for the encryption key.
        """
        manifest = await self.shard_engine.split(
            data,
            owner_node_id=owner_node_id,
            replication_factor=replication_factor,
            visibility=visibility,
        )
        content_hash = manifest.content_hash

        # Determine threshold parameters from number of shards.
        num_shards = len(manifest.shard_hashes)
        threshold, num_shares = _select_threshold_params(num_shards)

        manifest_bytes = manifest.to_json().encode("utf-8")
        ciphertext, key_shares = self.key_manager.encrypt_manifest(
            manifest_bytes,
            content_hash,
            threshold=threshold,
            num_shares=num_shares,
        )

        return content_hash, manifest, ciphertext, key_shares

    def _decode_manifest(self, ciphertext: bytes, key_shares: List[KeyShare]) -> ShardManifest:
        """Reconstruct key from *key_shares* and decrypt the manifest ciphertext."""
        manifest_bytes = self.key_manager.decrypt_manifest(ciphertext, key_shares)
        return ShardManifest.from_json(manifest_bytes.decode("utf-8"))

    # ------------------------------------------------------------------
    # Public local API
    # ------------------------------------------------------------------

    async def store_local(
        self,
        data: bytes,
        replication_factor: int = 3,
    ) -> ContentHash:
        """Store *data* locally and cache the encrypted manifest in memory.

        Shards are written to the BlobStore; the manifest is encrypted and
        kept in :attr:`_manifest_cache` keyed by the content hash hex string.

        Parameters
        ----------
        data:
            Raw bytes to store.
        replication_factor:
            Desired network replication factor embedded in the manifest metadata.

        Returns
        -------
        ContentHash
            Content-addressed identifier for the stored data.
        """
        content_hash, manifest, ciphertext, key_shares = await self._store_and_encrypt(
            data,
            owner_node_id=self.node_id,
            replication_factor=replication_factor,
        )
        self._manifest_cache[content_hash.hex()] = (ciphertext, key_shares, manifest)
        return content_hash

    async def retrieve_local(self, content_hash: ContentHash) -> bytes:
        """Retrieve content by *content_hash* from the local store.

        Uses the cached manifest to locate and reassemble shards.

        Raises
        ------
        ContentNotFoundError
            If the content hash is not present in :attr:`_manifest_cache`.
        """
        cache_key = content_hash.hex()
        if cache_key not in self._manifest_cache:
            raise ContentNotFoundError(cache_key)

        ciphertext, key_shares, manifest = self._manifest_cache[cache_key]
        return await self.shard_engine.reassemble(manifest)

    async def exists_local(self, content_hash: ContentHash) -> bool:
        """Return ``True`` if *content_hash* is present in the manifest cache."""
        return content_hash.hex() in self._manifest_cache

    async def size_local(self, content_hash: ContentHash) -> int:
        """Return the total content size in bytes for *content_hash*.

        Reads the cached manifest's ``total_size`` field (set when the
        content was sharded and stored). Returns 0 if the content is not
        present in the local manifest cache — callers that care about the
        distinction between "absent" and "empty" should first check
        :meth:`exists_local`.
        """
        cache_key = content_hash.hex()
        entry = self._manifest_cache.get(cache_key)
        if entry is None:
            return 0
        _ciphertext, _key_shares, manifest = entry
        return manifest.total_size

    async def delete_local(self, content_hash: ContentHash) -> None:
        """Delete all shards and the manifest cache entry for *content_hash*.

        Shard blobs are removed from the BlobStore; the manifest cache entry
        is dropped.  No-op if *content_hash* is not present.
        """
        cache_key = content_hash.hex()
        if cache_key not in self._manifest_cache:
            return

        _ciphertext, _key_shares, manifest = self._manifest_cache.pop(cache_key)
        for shard_hash in manifest.shard_hashes:
            await self.blob_store.delete(shard_hash)
