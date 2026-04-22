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
from prsm.storage import erasure
from prsm.storage.exceptions import ShardIntegrityError
from prsm.storage.models import (
    AlgorithmID,
    ContentHash,
    ErasureParams,
    ShardingMode,
    ShardManifest,
)


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
        *,
        sharding_mode: ShardingMode = ShardingMode.REPLICATION,
        erasure_k: int = erasure.DEFAULT_K,
        erasure_n: int = erasure.DEFAULT_N,
    ) -> ShardManifest:
        """Split *data* into shards, store each via the blob store, return manifest.

        Two sharding modes:

        - **REPLICATION** (default, Tier A): content ≤ shard_threshold is
          stored as a single shard; larger content is chunked into
          ``shard_size``-byte pieces stored independently. Reassembly
          concatenates every shard in order.
        - **ERASURE** (Phase 7-storage Tier B/C): content is Reed-Solomon
          encoded into ``erasure_n`` shards (default 10); any ``erasure_k``
          (default 6) suffice for reassembly. Tolerates
          ``erasure_n - erasure_k`` shard losses (default 40%).

        All shards — including the single-shard replication case — are
        stored via :meth:`BlobStore.store`.
        """
        # Hash the complete original content first.
        content_hash = ContentHash.from_data(data, AlgorithmID.SHA256)
        total_size = len(data)

        if sharding_mode is ShardingMode.ERASURE:
            return await self._split_erasure(
                data=data,
                content_hash=content_hash,
                total_size=total_size,
                owner_node_id=owner_node_id,
                replication_factor=replication_factor,
                visibility=visibility,
                k=erasure_k,
                n=erasure_n,
            )

        # Replication path — unchanged from the legacy behaviour.
        if total_size <= self.shard_threshold:
            chunks = [data]
        else:
            num_shards = math.ceil(total_size / self.shard_size)
            chunks = [
                data[i * self.shard_size : (i + 1) * self.shard_size]
                for i in range(num_shards)
            ]

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
            sharding_mode=ShardingMode.REPLICATION,
            erasure_params=None,
        )

    async def _split_erasure(
        self,
        *,
        data: bytes,
        content_hash: ContentHash,
        total_size: int,
        owner_node_id: str,
        replication_factor: int,
        visibility: str,
        k: int,
        n: int,
    ) -> ShardManifest:
        """Erasure-encode + store. Produces one blob_store entry per
        Reed-Solomon shard; manifest.shard_hashes is ordered by shard
        index (0..n-1)."""
        meta, shards = erasure.encode(data, k=k, n=n)

        shard_hashes: list[ContentHash] = []
        for shard in shards:
            stored_hash = await self.blob_store.store(shard.data)
            # Sanity: blob-store content-addressing must match the
            # erasure shard's own sha256. The erasure module produces
            # a raw 64-char hex digest; ContentHash.hex() prepends an
            # algorithm byte — compare only the digest portion.
            if stored_hash.digest.hex() != shard.sha256:
                raise ShardIntegrityError(
                    expected=shard.sha256,
                    actual=stored_hash.digest.hex(),
                )
            shard_hashes.append(stored_hash)

        return ShardManifest(
            content_hash=content_hash,
            shard_hashes=shard_hashes,
            total_size=total_size,
            shard_size=meta.shard_bytes,
            algorithm_id=AlgorithmID.SHA256,
            created_at=time.time(),
            replication_factor=replication_factor,
            owner_node_id=owner_node_id,
            visibility=visibility,
            sharding_mode=ShardingMode.ERASURE,
            erasure_params=ErasureParams(
                k=meta.k,
                n=meta.n,
                payload_bytes=meta.payload_bytes,
                shard_bytes=meta.shard_bytes,
                payload_sha256=meta.payload_sha256,
            ),
        )

    async def reassemble(self, manifest: ShardManifest) -> bytes:
        """Retrieve shards, verify integrity, and return the original content.

        Replication mode: concatenates every shard in order after
        per-shard + overall hash checks.

        Erasure mode: retrieves as many shards as the blob_store can
        supply (up to n), passes them to ``prsm.storage.erasure.decode``
        which requires at least k distinct shards and verifies the
        overall payload sha256 against the manifest's erasure_params.

        Raises
        ------
        ShardIntegrityError
            If any retrieved replication shard or the reassembled
            replication payload fails its hash check.
        InsufficientShardsError
            Erasure mode only — fewer than k shards were retrievable.
        """
        if manifest.sharding_mode is ShardingMode.ERASURE:
            return await self._reassemble_erasure(manifest)

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

    async def _reassemble_erasure(self, manifest: ShardManifest) -> bytes:
        """Erasure-mode reassembly. Gathers surviving shards (skipping
        any the blob_store cannot retrieve) and passes them to the
        erasure decoder, which enforces the k-of-n threshold and the
        overall payload sha256."""
        if manifest.erasure_params is None:
            raise ShardIntegrityError(
                expected="erasure_params",
                actual="None (manifest marked ERASURE but has no params)",
            )
        params = manifest.erasure_params

        survivors: list[erasure.ErasureShard] = []
        for index, expected_hash in enumerate(manifest.shard_hashes):
            try:
                raw = await self.blob_store.retrieve(expected_hash)
            except Exception:
                # Lost shard — skip. decode() will raise
                # InsufficientShardsError if too many are missing.
                continue

            survivors.append(
                erasure.ErasureShard(
                    index=index,
                    data=raw,
                    # erasure.decode expects a bare hex digest, not the
                    # algorithm-prefixed ContentHash.hex() serialisation.
                    sha256=expected_hash.digest.hex(),
                )
            )

        recovered = erasure.decode(
            erasure.ErasureMetadata(
                k=params.k,
                n=params.n,
                payload_bytes=params.payload_bytes,
                shard_bytes=params.shard_bytes,
                payload_sha256=params.payload_sha256,
            ),
            survivors,
        )

        # Cross-check against the manifest's content_hash (belt +
        # suspenders on top of erasure.decode's payload_sha256 check).
        actual_content_hash = ContentHash.from_data(
            recovered, manifest.content_hash.algorithm_id
        )
        if actual_content_hash != manifest.content_hash:
            raise ShardIntegrityError(
                expected=manifest.content_hash.hex(),
                actual=actual_content_hash.hex(),
            )
        return recovered
