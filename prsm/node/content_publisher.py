"""
ContentPublisher / ContentRetriever — unified publish/fetch surface for the
PRSM proprietary content-distribution layer.

Routes upload/download calls in ``content_uploader.py`` through the
BitTorrent provider/requester layer that is the canonical PRSM
data-distribution mechanism (see canonical scope memory and
``prsm/core/bittorrent_*.py``).

Three content tiers (see ``prsm/compute/inference/models.py:ContentTier``):

* **Tier A** (public) — bytes are written to a staging file (deduplicated
  by SHA-256) and seeded directly by ``BitTorrentProvider.seed_content``.
  Single-file torrent. The infohash becomes the content identifier used
  downstream (provenance records, on-chain
  ``RoyaltyDistributor.distributeRoyalty(contentHash, ...)``).
* **Tier B** (encrypted) — bytes are passed through
  ``ContentStore.store_local_with_artifacts`` which AES-256-GCM-encrypts
  each shard and Shamir-splits the encryption key. The publisher stages
  every artefact (encrypted shards + ciphertext manifest + key shares)
  into a directory and seeds it as a multi-file torrent.
* **Tier C** (encrypted + erasure) — same as Tier B; Tier C-specific
  Reed-Solomon erasure coding is selected inside
  ``ShardEngine.split`` based on content size / replication factor.
  From the publisher's perspective, B and C share the same artefact
  layout; the manifest's ``sharding_mode`` distinguishes them.

For Tier B/C the multi-file torrent layout is::

    <torrent root>/
        manifest.bin           — ciphertext (encrypted_manifest)
        keyshares.json         — JSON-serialised KeyShare list
        shard-0000.bin         — first shard ciphertext
        shard-0001.bin         — …
        …

Key-share distribution caveat: PR 2b ships with the Shamir shares
*colocated* in the torrent (alongside the encrypted manifest). This
makes the publish/fetch round-trip work end-to-end but negates the
per-share confidentiality benefit Shamir is meant to provide. Proper
distribution-of-shares lives behind ``KeyDistribution.sol`` and is the
follow-on workstream — see ``contracts/storage/KeyDistribution.sol``
and Phase 7-storage Task 6 in MEMORY.md. Until that ships, treat
Tier B/C content as "encrypted at rest in a torrent" but not "secret
from a node operator who downloads the torrent".
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from prsm.compute.inference.models import ContentTier

if TYPE_CHECKING:
    from prsm.core.bittorrent_manifest import TorrentManifest
    from prsm.node.bittorrent_provider import BitTorrentProvider
    from prsm.node.bittorrent_requester import BitTorrentRequester
    from prsm.storage import ContentStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier B/C torrent-directory layout helpers (used by both publisher and
# retriever to keep the wire format in one place).
# ---------------------------------------------------------------------------

_MANIFEST_FILENAME = "manifest.bin"
_KEYSHARES_FILENAME = "keyshares.json"
_SHARD_FILENAME_FMT = "shard-{:04d}.bin"


def _serialize_key_shares(shares: list) -> bytes:
    """Encode KeyShare list to JSON bytes (used in keyshares.json)."""
    serialised = [
        {
            "content_hash": share.content_hash.hex(),
            "share_index": share.share_index,
            "share_data": share.share_data.hex(),
            "threshold": share.threshold,
            "total_shares": share.total_shares,
            "algorithm_id": share.algorithm_id,
        }
        for share in shares
    ]
    return json.dumps(serialised).encode("utf-8")


def _deserialize_key_shares(payload: bytes) -> list:
    """Decode keyshares.json bytes back into KeyShare instances."""
    from prsm.storage.models import ContentHash, KeyShare

    raw = json.loads(payload.decode("utf-8"))
    return [
        KeyShare(
            content_hash=ContentHash.from_hex(entry["content_hash"]),
            share_index=entry["share_index"],
            share_data=bytes.fromhex(entry["share_data"]),
            threshold=entry["threshold"],
            total_shares=entry["total_shares"],
            algorithm_id=entry["algorithm_id"],
        )
        for entry in raw
    ]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishedContent:
    """Outcome of a successful :meth:`ContentPublisher.publish` call.

    ``torrent_infohash`` is the canonical content identifier used in
    downstream provenance + on-chain royalty calls. ``staged_path`` is
    the local file path where the content was staged before BitTorrent
    seeding (the node continues to seed from this path).
    """

    torrent_infohash: str
    staged_path: Path
    manifest: "TorrentManifest"


# ---------------------------------------------------------------------------
# ContentPublisher
# ---------------------------------------------------------------------------


class ContentPublisher:
    """Stage raw bytes to disk and publish them to the PRSM network.

    Tier A: bytes are written to a content-addressed staging file and
    seeded by BitTorrent as a single-file torrent.

    Tier B/C: bytes go through ``ContentStore.store_local_with_artifacts``
    (encrypt + shard + Shamir-split), then every artefact is staged as a
    multi-file torrent directory and seeded.

    Re-publishing the same bytes is a no-op for Tier A (content-addressed
    staging path). For Tier B/C the per-publish AES key is regenerated, so
    the same plaintext yields a fresh encrypted torrent each time — this
    is the intended behaviour (forward-secrecy at the share-distribution
    layer); callers that want deduplication should cache the
    :class:`PublishedContent` result keyed by the plaintext hash.
    """

    def __init__(
        self,
        bt_provider: "BitTorrentProvider",
        staging_dir: Path,
        content_store: Optional["ContentStore"] = None,
    ) -> None:
        self.bt_provider = bt_provider
        self.staging_dir = Path(staging_dir).expanduser()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        # Lazy-resolve from the global singleton if not supplied — keeps
        # the Tier-A construction site backwards compatible with PR 2a.
        self.content_store = content_store

    def _resolve_content_store(self) -> "ContentStore":
        """Return the configured ContentStore or raise if unavailable."""
        if self.content_store is not None:
            return self.content_store
        from prsm.storage import get_content_store
        store = get_content_store()
        if store is None:
            raise RuntimeError(
                "ContentPublisher: Tier B/C publish requires a ContentStore. "
                "Initialise via prsm.storage.init_content_store() or pass "
                "content_store= to ContentPublisher.__init__()."
            )
        return store

    async def publish(
        self,
        data: bytes,
        *,
        provenance_id: str,
        tier: ContentTier = ContentTier.A,
        name: Optional[str] = None,
        replication_factor: int = 3,
    ) -> PublishedContent:
        """Publish *data* to the PRSM network and return the resulting handle.

        Parameters
        ----------
        data:
            Raw content bytes.
        provenance_id:
            PRSM provenance identifier (typically the on-chain
            ``provenance_hash_hex`` set by the uploader). Embedded in the
            torrent manifest for downstream royalty resolution.
        tier:
            Content tier. ``A`` (public) seeds raw bytes as a single-file
            torrent. ``B`` (encrypted) and ``C`` (encrypted + erasure)
            run the bytes through :class:`ContentStore` first and seed
            the encrypted artefacts as a multi-file torrent.
        name:
            Optional human-readable torrent name. Defaults to the staging
            filename / directory name.
        replication_factor:
            Replication factor passed through to
            :meth:`ContentStore.store_local_with_artifacts` for Tier B/C.
            Ignored for Tier A.

        Returns
        -------
        PublishedContent
            Handle containing the torrent infohash and the staged path.

        Raises
        ------
        RuntimeError
            If BitTorrent seeding fails. The staged file/directory is
            left in place for retry.
        """
        if tier is ContentTier.A:
            return await self._publish_tier_a(data, provenance_id=provenance_id, name=name)

        # Tier B / C — same artefact layout, ContentStore picks erasure
        # mode internally based on size + replication_factor.
        return await self._publish_tier_bc(
            data,
            provenance_id=provenance_id,
            name=name,
            tier=tier,
            replication_factor=replication_factor,
        )

    async def _publish_tier_a(
        self,
        data: bytes,
        *,
        provenance_id: str,
        name: Optional[str],
    ) -> PublishedContent:
        """Tier A: stage raw bytes, seed as a single-file torrent."""
        staged_filename = hashlib.sha256(data).hexdigest()
        staged_path = self.staging_dir / staged_filename

        if not staged_path.exists():
            # Write atomically via tmp+rename so a concurrent reader never
            # sees a partial file.
            tmp_path = staged_path.with_suffix(".tmp")
            tmp_path.write_bytes(data)
            tmp_path.replace(staged_path)

        manifest = await self.bt_provider.seed_content(
            path=staged_path,
            name=name or staged_filename,
            provenance_id=provenance_id,
        )

        if manifest is None:
            raise RuntimeError(
                f"BitTorrent seed_content returned None for staged path "
                f"{staged_path} (provenance_id={provenance_id}). The staged "
                f"file is preserved; retry is safe."
            )

        logger.info(
            "Content published (tier=A): infohash=%s name=%s size=%d provenance_id=%s",
            manifest.infohash,
            manifest.name,
            len(data),
            provenance_id,
        )

        return PublishedContent(
            torrent_infohash=manifest.infohash,
            staged_path=staged_path,
            manifest=manifest,
        )

    async def _publish_tier_bc(
        self,
        data: bytes,
        *,
        provenance_id: str,
        name: Optional[str],
        tier: ContentTier,
        replication_factor: int,
    ) -> PublishedContent:
        """Tier B/C: encrypt+shard via ContentStore, seed as a multi-file torrent.

        Layout (see module docstring): ``manifest.bin``, ``keyshares.json``,
        and one ``shard-NNNN.bin`` per shard. The torrent root directory
        is named after the plaintext SHA-256 so re-publishing the same
        bytes lands in a deterministic directory (helps debuggability;
        BitTorrent infohash is still per-publish because the AES key
        rotates).
        """
        store = self._resolve_content_store()
        artefacts = await store.store_local_with_artifacts(
            data, replication_factor=replication_factor
        )

        torrent_root_name = hashlib.sha256(data).hexdigest()
        torrent_root = self.staging_dir / f"{torrent_root_name}-{tier.value}"
        torrent_root.mkdir(parents=True, exist_ok=True)

        # 1. Encrypted manifest.
        (torrent_root / _MANIFEST_FILENAME).write_bytes(artefacts.encrypted_manifest)

        # 2. Key shares (JSON).
        (torrent_root / _KEYSHARES_FILENAME).write_bytes(
            _serialize_key_shares(artefacts.key_shares)
        )

        # 3. Encrypted shards — read each from the BlobStore path that
        # store_local_with_artifacts produced and copy into the torrent
        # directory under a deterministic filename. We copy rather than
        # symlink so the BitTorrent client can mmap a stable path that
        # doesn't depend on the BlobStore's bucket layout.
        for shard_index, shard_path in enumerate(artefacts.shard_paths):
            shard_bytes = Path(shard_path).read_bytes()
            shard_filename = _SHARD_FILENAME_FMT.format(shard_index)
            (torrent_root / shard_filename).write_bytes(shard_bytes)

        bt_manifest = await self.bt_provider.seed_content(
            path=torrent_root,
            name=name or torrent_root.name,
            provenance_id=provenance_id,
        )

        if bt_manifest is None:
            raise RuntimeError(
                f"BitTorrent seed_content returned None for staged dir "
                f"{torrent_root} (provenance_id={provenance_id}). The staged "
                f"directory is preserved; retry is safe."
            )

        logger.info(
            "Content published (tier=%s): infohash=%s name=%s shards=%d "
            "shamir=%d-of-%d size=%d provenance_id=%s",
            tier.value,
            bt_manifest.infohash,
            bt_manifest.name,
            len(artefacts.shard_paths),
            artefacts.key_shares[0].threshold,
            artefacts.key_shares[0].total_shares,
            len(data),
            provenance_id,
        )

        return PublishedContent(
            torrent_infohash=bt_manifest.infohash,
            staged_path=torrent_root,
            manifest=bt_manifest,
        )


# ---------------------------------------------------------------------------
# ContentRetriever
# ---------------------------------------------------------------------------


class ContentRetriever:
    """Mirror of :class:`ContentPublisher` for the read path.

    Downloads content by infohash via the BitTorrent requester layer and
    returns the original plaintext bytes. Auto-detects Tier A (single
    file in the torrent root) vs Tier B/C (manifest.bin + keyshares.json
    + shard-NNNN.bin layout) and routes through ContentStore for the
    encrypted case.
    """

    def __init__(
        self,
        bt_requester: "BitTorrentRequester",
        cache_dir: Path,
        content_store: Optional["ContentStore"] = None,
    ) -> None:
        self.bt_requester = bt_requester
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.content_store = content_store

    def _resolve_content_store(self) -> "ContentStore":
        """Return the configured ContentStore or raise if unavailable."""
        if self.content_store is not None:
            return self.content_store
        from prsm.storage import get_content_store
        store = get_content_store()
        if store is None:
            raise RuntimeError(
                "ContentRetriever: Tier B/C fetch requires a ContentStore. "
                "Initialise via prsm.storage.init_content_store() or pass "
                "content_store= to ContentRetriever.__init__()."
            )
        return store

    async def fetch(
        self,
        torrent_infohash: str,
        *,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Download content by *torrent_infohash* and return the plaintext.

        Auto-routes between Tier A (single-file torrent) and Tier B/C
        (multi-file torrent containing ``manifest.bin``,
        ``keyshares.json``, and ``shard-NNNN.bin`` files). For Tier B/C,
        the encrypted shards are reassembled and decrypted via
        :meth:`ContentStore.retrieve_with_artifacts` before returning.

        Parameters
        ----------
        torrent_infohash:
            BitTorrent infohash returned by a prior
            :meth:`ContentPublisher.publish` call (or received via gossip).
        timeout:
            Maximum seconds to wait for the download. Defaults to the
            requester's configured default.

        Raises
        ------
        FileNotFoundError
            If the requester reports success but no file lands in the
            cache directory (indicates BitTorrent client misconfiguration).
        RuntimeError
            If the requester returns ``success=False``, or for Tier B/C
            if a required artefact (manifest, keyshares) is missing.
        """
        save_path = self.cache_dir / torrent_infohash
        save_path.mkdir(parents=True, exist_ok=True)

        result = await self.bt_requester.request_content(
            infohash=torrent_infohash,
            save_path=save_path,
            timeout=timeout,
        )

        if not result.success:
            raise RuntimeError(
                f"BitTorrent fetch failed for infohash={torrent_infohash}: "
                f"{result.error or 'no error reported'}"
            )

        # BitTorrent multi-file torrents land everything inside a single
        # subdirectory named after the torrent. Resolve that root before
        # classifying.
        contents = sorted(save_path.iterdir())
        if len(contents) == 0:
            raise FileNotFoundError(
                f"BitTorrent download reported success but no files found at "
                f"{save_path} for infohash={torrent_infohash}"
            )

        # If the torrent is wrapped in a single subdirectory, descend.
        if len(contents) == 1 and contents[0].is_dir():
            torrent_root = contents[0]
            files = sorted(p for p in torrent_root.iterdir() if p.is_file())
        else:
            torrent_root = save_path
            files = [p for p in contents if p.is_file()]

        # Tier A: single file, return its bytes directly.
        if len(files) == 1 and files[0].name not in (
            _MANIFEST_FILENAME, _KEYSHARES_FILENAME
        ):
            return files[0].read_bytes()

        # Tier B/C: expect manifest.bin + keyshares.json + N shard files.
        return await self._fetch_tier_bc(torrent_root, infohash=torrent_infohash)

    async def _fetch_tier_bc(self, torrent_root: Path, *, infohash: str) -> bytes:
        """Reassemble + decrypt a multi-file Tier-B/C torrent."""
        manifest_path = torrent_root / _MANIFEST_FILENAME
        keyshares_path = torrent_root / _KEYSHARES_FILENAME

        if not manifest_path.exists():
            raise RuntimeError(
                f"Tier B/C fetch failed for infohash={infohash}: "
                f"missing {_MANIFEST_FILENAME} in {torrent_root}"
            )
        if not keyshares_path.exists():
            raise RuntimeError(
                f"Tier B/C fetch failed for infohash={infohash}: "
                f"missing {_KEYSHARES_FILENAME} in {torrent_root}"
            )

        encrypted_manifest = manifest_path.read_bytes()
        key_shares = _deserialize_key_shares(keyshares_path.read_bytes())

        # Shards land as shard-NNNN.bin — sort by index to preserve order.
        shard_files = sorted(
            p for p in torrent_root.iterdir()
            if p.is_file() and p.name.startswith("shard-") and p.name.endswith(".bin")
        )
        shard_blobs: List[bytes] = [p.read_bytes() for p in shard_files]

        store = self._resolve_content_store()
        return await store.retrieve_with_artifacts(
            encrypted_manifest=encrypted_manifest,
            key_shares=key_shares,
            shard_blobs=shard_blobs,
        )
