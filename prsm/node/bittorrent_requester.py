"""
BitTorrent Requester
====================

Downloader/consumer side of BitTorrent integration for PRSM.
Mirrors compute_requester.py in structure.

A BitTorrentRequester:
- Discovers available torrents via gossip
- Downloads content from the swarm
- Pays FTNS for bandwidth consumed
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from prsm.node.identity import NodeIdentity
from prsm.node.gossip import GossipProtocol
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.core.bittorrent_client import (
    BitTorrentClient,
    TorrentInfo,
)
from prsm.core.bittorrent_manifest import (
    TorrentManifest,
    TorrentManifestStore,
    TorrentManifestIndex,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_DOWNLOAD_COST_PER_GB = Decimal("0.05")
DEFAULT_DOWNLOAD_TIMEOUT = 3600.0


@dataclass
class BitTorrentRequesterConfig:
    """Configuration for the BitTorrent requester."""
    max_concurrent_downloads: int = 10
    data_dir: str = "~/.prsm/torrents"
    download_cost_per_gb: Decimal = DEFAULT_DOWNLOAD_COST_PER_GB
    default_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT


@dataclass
class DownloadRequest:
    """Tracks an active download request."""
    request_id: str
    infohash: str
    name: str
    requester_node_id: str
    save_path: Path
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"
    bytes_downloaded: int = 0
    total_bytes: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "infohash": self.infohash,
            "name": self.name,
            "status": self.status,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "error": self.error,
        }


@dataclass
class DownloadResult:
    """Result of a download operation."""
    request_id: str
    infohash: str
    success: bool
    path: Optional[Path] = None
    bytes_downloaded: int = 0
    duration_secs: float = 0.0
    error: Optional[str] = None
    ftns_paid: Decimal = Decimal("0")


class BitTorrentRequester:
    """
    Discovers and downloads torrents from the PRSM network.

    Maintains a local index of known torrents (populated from gossip),
    and handles the complete download lifecycle including FTNS payments.
    """

    def __init__(
        self,
        identity: NodeIdentity,
        gossip: GossipProtocol,
        bt_client: BitTorrentClient,
        manifest_store: TorrentManifestStore,
        ledger: LocalLedger,
        config: Optional[BitTorrentRequesterConfig] = None,
    ):
        self.identity = identity
        self.gossip = gossip
        self.bt_client = bt_client
        self.manifest_store = manifest_store
        self.ledger = ledger
        self.config = config or BitTorrentRequesterConfig()

        # In-memory index of known torrents (from gossip announcements)
        self._discovery_index = TorrentManifestIndex()

        # Active downloads
        self._downloads: Dict[str, DownloadRequest] = {}

        self._running = False

    async def start(self) -> None:
        """Start the requester and begin listening for announcements."""
        if not self.bt_client.available:
            logger.warning("BitTorrent client not available - requester disabled")
            return

        self._running = True

        # Subscribe to BitTorrent gossip messages
        self.gossip.subscribe("bittorrent_announce", self._on_announce)
        self.gossip.subscribe("bittorrent_withdraw", self._on_withdraw)

        # Load previously known torrents from store
        await self._load_known_torrents()

        logger.info(
            f"BitTorrent requester started: {self._discovery_index.count()} known torrents"
        )

    async def stop(self) -> None:
        """Stop the requester."""
        self._running = False
        logger.info("BitTorrent requester stopped")

    async def request_content(
        self,
        infohash: str,
        save_path: Path,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[str, float, Dict], None]] = None,
    ) -> DownloadResult:
        """
        Download content from the swarm.

        Args:
            infohash: Infohash of torrent to download
            save_path: Where to save the downloaded files
            timeout: Maximum time to wait (default from config)
            progress_callback: Optional callback(request_id, progress, stats)

        Returns:
            DownloadResult with outcome details
        """
        if not self.bt_client.available:
            return DownloadResult(
                request_id="",
                infohash=infohash,
                success=False,
                error="BitTorrent client not available",
            )

        timeout = timeout or self.config.default_timeout
        save_path = Path(save_path).expanduser()
        save_path.mkdir(parents=True, exist_ok=True)

        # Find manifest
        manifest = await self.find_torrent(infohash)
        if not manifest:
            return DownloadResult(
                request_id="",
                infohash=infohash,
                success=False,
                error="Torrent not found",
            )

        # Create download request
        request_id = str(uuid.uuid4())
        request = DownloadRequest(
            request_id=request_id,
            infohash=infohash,
            name=manifest.name,
            requester_node_id=self.identity.node_id,
            save_path=save_path,
            total_bytes=manifest.total_size,
        )
        self._downloads[request_id] = request

        # Start download
        request.started_at = time.time()
        request.status = "downloading"

        try:
            # Add torrent to client
            add_result = await self.bt_client.add_torrent(
                source=manifest.magnet_uri or manifest.torrent_bytes,
                save_path=save_path,
            )

            if not add_result.success:
                request.status = "failed"
                request.error = add_result.error
                return DownloadResult(
                    request_id=request_id,
                    infohash=infohash,
                    success=False,
                    error=add_result.error,
                )

            # Wait for completion
            completion_result = await self.bt_client.wait_for_completion(
                infohash=infohash,
                timeout=timeout,
                progress_callback=lambda ih, prog, stats: self._update_progress(
                    request_id, prog, stats, progress_callback
                ),
            )

            if completion_result.success:
                status = await self.bt_client.get_status(infohash)
                if isinstance(status, TorrentInfo):
                    request.bytes_downloaded = status.bytes_downloaded

                request.status = "completed"
                request.completed_at = time.time()

                duration = request.completed_at - request.started_at
                ftns_paid = await self._charge_download(request)

                return DownloadResult(
                    request_id=request_id,
                    infohash=infohash,
                    success=True,
                    path=save_path,
                    bytes_downloaded=request.bytes_downloaded,
                    duration_secs=duration,
                    ftns_paid=ftns_paid,
                )
            else:
                request.status = "failed"
                request.error = completion_result.error or "Download timed out"
                return DownloadResult(
                    request_id=request_id,
                    infohash=infohash,
                    success=False,
                    error=request.error,
                )

        except Exception as e:
            request.status = "failed"
            request.error = str(e)
            logger.error(f"Download error for {infohash[:16]}...: {e}")
            return DownloadResult(
                request_id=request_id,
                infohash=infohash,
                success=False,
                error=str(e),
            )

    async def find_torrent(self, infohash: str) -> Optional[TorrentManifest]:
        """Look up a torrent manifest by infohash."""
        manifest = self._discovery_index.get_by_infohash(infohash)
        if manifest:
            return manifest
        return await self.manifest_store.load(infohash)

    async def list_available(self, query: Optional[str] = None) -> List[TorrentManifest]:
        """List all known available torrents."""
        if query:
            return self._discovery_index.search(query)
        return self._discovery_index.list_all()

    def get_download_status(self, request_id: str) -> Optional[DownloadRequest]:
        """Get status of a download by request ID."""
        return self._downloads.get(request_id)

    def get_stats(self) -> Dict[str, Any]:
        """Return requester statistics."""
        active = [d for d in self._downloads.values() if d.status in ("pending", "downloading")]
        completed = [d for d in self._downloads.values() if d.status == "completed"]
        total_bytes = sum(d.bytes_downloaded for d in completed)

        return {
            "known_torrents": self._discovery_index.count(),
            "active_downloads": len(active),
            "completed_downloads": len(completed),
            "total_bytes_downloaded": total_bytes,
            "download_cost_per_gb": float(self.config.download_cost_per_gb),
        }

    async def _on_announce(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Handle a bittorrent_announce message."""
        infohash = data.get("infohash", "")
        if not infohash:
            return

        manifest = TorrentManifest(
            infohash=infohash,
            name=data.get("name", "Unknown"),
            total_size=data.get("size_bytes", 0),
            piece_length=data.get("piece_length", 262144),
            magnet_uri=data.get("magnet_uri", ""),
            created_by_node_id=data.get("seeder_node_id", origin),
        )

        self._discovery_index.add(manifest)
        logger.debug(f"Discovered torrent: {manifest.name} ({infohash[:16]}...)")

    async def _on_withdraw(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Handle a bittorrent_withdraw message."""
        infohash = data.get("infohash", "")
        if infohash:
            self._discovery_index.remove(infohash)
            logger.debug(f"Torrent withdrawn: {infohash[:16]}...")

    async def _load_known_torrents(self) -> None:
        """Load previously known torrents from persistent store."""
        try:
            manifests = await self.manifest_store.list_all()
            for manifest in manifests:
                self._discovery_index.add(manifest)
        except Exception as e:
            logger.error(f"Error loading known torrents: {e}")

    def _update_progress(
        self,
        request_id: str,
        progress: float,
        stats: Dict[str, Any],
        callback: Optional[Callable],
    ) -> None:
        """Update download progress."""
        request = self._downloads.get(request_id)
        if request:
            request.bytes_downloaded = stats.get("bytes_downloaded", 0)

        if callback:
            try:
                callback(request_id, progress, stats)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

    async def _charge_download(self, request: DownloadRequest) -> Decimal:
        """Charge FTNS for downloaded content."""
        gb_downloaded = request.bytes_downloaded / (1024**3)
        amount = Decimal(str(gb_downloaded)) * self.config.download_cost_per_gb

        if amount > 0:
            try:
                await self.ledger.debit(
                    wallet_id=self.identity.node_id,
                    amount=float(amount),
                    tx_type=TransactionType.PAYMENT,
                    description=f"BitTorrent download: {request.name}",
                )
                logger.info(f"Charged {amount:.6f} FTNS for {gb_downloaded:.4f}GB download")
            except Exception as e:
                logger.error(f"Failed to charge for download: {e}")
                return Decimal("0")

        return amount
