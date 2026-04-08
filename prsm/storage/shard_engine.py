"""
Shard engine — splits content into fixed-size chunks, stores each chunk in
a BlobStore, and reassembles them with integrity verification.

Usage::

    engine = ShardEngine(blob_store)
    manifest = await engine.split(data, owner_node_id="node-1", replication_factor=3)
    original  = await engine.reassemble(manifest)
"""

from __future__ import annotations

import math
import time

from prsm.storage.blob_store import BlobStore
from prsm.storage.exceptions import ShardIntegrityError
from prsm.storage.models import AlgorithmID, ContentHash, ShardManifest


class ShardEngine:
    """
    Split content into shards and reassemble with integrity verification.

    Parameters
    ----------
    blob_store:
        Underlying content-addressed store used to persist individual shards.
    shard_threshold:
        Content whose byte-length is **at most** this value is stored as a
        single shard (no splitting needed).  Default: 1 MiB.
    shard_size:
        Target byte-length of each shard when content exceeds the threshold.
        The final shard may be smaller.  Default: 256 KiB.
    """

    def __init__(
        self,
        blob_store: BlobStore,
        shard_threshold: int = 1_048_576,  # 1 MiB
        shard_size: int = 262_144,          # 256 KiB
    ) -> None:
        self.blob_store = blob_store
        self.shard_threshold = shard_threshold
        self.shard_size = shard_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def split(
        self,
        data: bytes,
        owner_node_id: str,
        replication_factor: int,
        visibility: str = "public",
    ) -> ShardManifest:
        """Split *data* into shards, store each via the blob store, return manifest.

        Content whose length is **<= shard_threshold** is stored as a single
        shard.  Larger content is split into ``shard_size``-byte chunks (the
        last chunk may be smaller).  All chunks — including the single-shard
        case — are stored via :meth:`BlobStore.store`.

        Parameters
        ----------
        data:
            Raw bytes to be sharded.
        owner_node_id:
            Identifier of the node that owns this content.
        replication_factor:
            Number of network replicas desired (stored in manifest metadata).
        visibility:
            Access-control label; default ``"public"``.

        Returns
        -------
        ShardManifest
            Manifest describing the content hash, ordered shard hashes, and
            all associated metadata.
        """
        # Hash the complete original content first.
        content_hash = ContentHash.from_data(data, AlgorithmID.SHA256)
        total_size = len(data)

        # Determine chunks.
        if total_size <= self.shard_threshold:
            chunks = [data]
        else:
            num_shards = math.ceil(total_size / self.shard_size)
            chunks = [
                data[i * self.shard_size : (i + 1) * self.shard_size]
                for i in range(num_shards)
            ]

        # Store each chunk and collect shard hashes.
        shard_hashes: list[ContentHash] = []
        for chunk in chunks:
            shard_hash = await self.blob_store.store(chunk)
            shard_hashes.append(shard_hash)

        return ShardManifest(
            content_hash=content_hash,
            shard_hashes=shard_hashes,
            total_size=total_size,
            shard_size=self.shard_size,
            algorithm_id=AlgorithmID.SHA256,
            created_at=time.time(),
            replication_factor=replication_factor,
            owner_node_id=owner_node_id,
            visibility=visibility,
        )

    async def reassemble(self, manifest: ShardManifest) -> bytes:
        """Retrieve shards, verify integrity, and concatenate.

        Each shard is retrieved from the blob store and its hash is checked
        against the corresponding entry in *manifest.shard_hashes*.  After
        all shards pass their per-shard checks, the reassembled content is
        verified against *manifest.content_hash*.

        Parameters
        ----------
        manifest:
            The :class:`~prsm.storage.models.ShardManifest` produced by
            :meth:`split`.

        Returns
        -------
        bytes
            The original content, byte-for-byte identical to what was passed
            to :meth:`split`.

        Raises
        ------
        ShardIntegrityError
            If any individual shard or the final reassembled content fails its
            hash check.
        ContentNotFoundError
            If a shard is missing from the blob store (propagated from
            :meth:`BlobStore.retrieve`).
        """
        parts: list[bytes] = []

        for expected_hash in manifest.shard_hashes:
            raw = await self.blob_store.retrieve(expected_hash)

            # Verify the retrieved bytes match the recorded shard hash.
            actual_hash = ContentHash.from_data(raw, expected_hash.algorithm_id)
            if actual_hash != expected_hash:
                raise ShardIntegrityError(
                    expected=expected_hash.hex(),
                    actual=actual_hash.hex(),
                )

            parts.append(raw)

        reassembled = b"".join(parts)

        # Verify the reassembled content against the manifest's content hash.
        actual_content_hash = ContentHash.from_data(
            reassembled, manifest.content_hash.algorithm_id
        )
        if actual_content_hash != manifest.content_hash:
            raise ShardIntegrityError(
                expected=manifest.content_hash.hex(),
                actual=actual_content_hash.hex(),
            )

        return reassembled
