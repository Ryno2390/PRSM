"""
ContentPublisher / ContentRetriever — unified publish/fetch surface for the
PRSM proprietary content-distribution layer.

Routes upload/download calls in ``content_uploader.py`` through the
BitTorrent provider/requester layer that is the canonical PRSM
data-distribution mechanism (see canonical scope memory and
``prsm/core/bittorrent_*.py``).

PR 2a scope — **Tier A only.**
    Tier A (public) content is the launch tier. Bytes are written to a
    staging file (deduplicated by SHA-256) and seeded by
    ``BitTorrentProvider.seed_content``. The returned torrent infohash
    becomes the content identifier used downstream (provenance records,
    on-chain ``RoyaltyDistributor.distributeRoyalty(contentHash, ...)``).

Tier B/C — deferred to PR 2b. Calls to ``publish`` with ``tier=B`` or
``tier=C`` raise :class:`NotImplementedError` with a pointer to the
follow-up PR. The ContentStore↔BitTorrent integration for encrypted
sharded content is a separate design problem (BlobStore writes shards
to scattered bucket directories; BitTorrent wants a single-file or
single-directory seed target — the staging strategy needs explicit
design rather than ad-hoc resolution).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from prsm.compute.inference.models import ContentTier

if TYPE_CHECKING:
    from prsm.core.bittorrent_manifest import TorrentManifest
    from prsm.node.bittorrent_provider import BitTorrentProvider
    from prsm.node.bittorrent_requester import BitTorrentRequester

logger = logging.getLogger(__name__)


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

    Tier A only in PR 2a. The staging strategy uses content-addressed
    filenames (SHA-256 hex) so that re-publishing the same bytes is a
    no-op at the staging layer (BitTorrent will likewise recognize the
    existing torrent in the provider's active set and return the cached
    manifest without re-seeding).
    """

    def __init__(
        self,
        bt_provider: "BitTorrentProvider",
        staging_dir: Path,
    ) -> None:
        self.bt_provider = bt_provider
        self.staging_dir = Path(staging_dir).expanduser()
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    async def publish(
        self,
        data: bytes,
        *,
        provenance_id: str,
        tier: ContentTier = ContentTier.A,
        name: Optional[str] = None,
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
            Content tier. PR 2a accepts only ``ContentTier.A``; B/C raise
            :class:`NotImplementedError`.
        name:
            Optional human-readable torrent name. Defaults to the staging
            filename (the SHA-256 hex of *data*).

        Returns
        -------
        PublishedContent
            Handle containing the torrent infohash and the staged path.

        Raises
        ------
        NotImplementedError
            If *tier* is ``B`` or ``C`` — the ContentStore↔BitTorrent
            integration for encrypted sharded content is deferred to
            PR 2b.
        RuntimeError
            If BitTorrent seeding fails. The staged file is left in
            place for retry.
        """
        if tier is not ContentTier.A:
            raise NotImplementedError(
                f"ContentPublisher.publish does not yet support tier={tier.value}. "
                "Tier B/C distribution requires ContentStore↔BitTorrent integration "
                "deferred to PR 2b — see "
                "docs/plans/native-storage-migration-status-2026-05-07.md."
            )

        # Content-addressed staging: same bytes → same file → BT dedupes.
        staged_filename = hashlib.sha256(data).hexdigest()
        staged_path = self.staging_dir / staged_filename

        if not staged_path.exists():
            # Write atomically via tmp+rename so a concurrent reader never
            # sees a partial file. We don't bother with aiofiles here —
            # the staging write is bounded by content size and the BT
            # seed_content call is the dominant cost.
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
            "Content published: infohash=%s name=%s size=%d provenance_id=%s",
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


# ---------------------------------------------------------------------------
# ContentRetriever
# ---------------------------------------------------------------------------


class ContentRetriever:
    """Mirror of :class:`ContentPublisher` for the read path.

    Downloads content by infohash via the BitTorrent requester layer and
    returns the raw bytes. Single-file torrents only (Tier A); B/C
    decryption will be wired alongside the publish-side B/C support in
    PR 2b.
    """

    def __init__(
        self,
        bt_requester: "BitTorrentRequester",
        cache_dir: Path,
    ) -> None:
        self.bt_requester = bt_requester
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch(
        self,
        torrent_infohash: str,
        *,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Download content by *torrent_infohash* and return the raw bytes.

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
            If the requester returns ``success=False``.
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

        # Tier A is single-file. Pick up the only file in save_path. If
        # save_path contains multiple files, we have a Tier-B/C-shaped
        # torrent that this PR doesn't yet support — surface it loudly.
        files = sorted(p for p in save_path.iterdir() if p.is_file())
        if len(files) == 0:
            raise FileNotFoundError(
                f"BitTorrent download reported success but no file found at "
                f"{save_path} for infohash={torrent_infohash}"
            )
        if len(files) > 1:
            raise NotImplementedError(
                f"ContentRetriever.fetch received multi-file torrent "
                f"(infohash={torrent_infohash}, {len(files)} files at "
                f"{save_path}) — multi-file (Tier B/C) decoding deferred to "
                f"PR 2b. See docs/plans/native-storage-migration-status-2026-05-07.md."
            )

        return files[0].read_bytes()
