"""
BitTorrent Provider
===================

Seeder component for the PRSM BitTorrent integration.
A node that holds data and offers it to the swarm.

Mirrors storage_provider.py for consistency with existing patterns.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from prsm.node.config import NodeConfig

from prsm.node.identity import NodeIdentity
from prsm.node.transport import MSG_DIRECT, P2PMessage, WebSocketTransport
from prsm.node.gossip import GossipProtocol
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.core.bittorrent_client import (
    BitTorrentClient,
    BitTorrentResult,
    TorrentInfo,
    TorrentState,
)
from prsm.core.bittorrent_manifest import (
    TorrentManifest,
    TorrentManifestStore,
    create_manifest_from_torrent,
)

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MAX_TORRENTS = 50
DEFAULT_REWARD_INTERVAL_SECS = 3600
DEFAULT_MIN_SEED_TIME_SECS = 3600
DEFAULT_SEEDER_REWARD_PER_GB = 0.10  # FTNS


@dataclass
class BitTorrentProviderConfig:
    """Configuration for the BitTorrent provider."""
    max_torrents: int = DEFAULT_MAX_TORRENTS
    data_dir: str = "~/.prsm/torrents"
    reward_interval_secs: float = DEFAULT_REWARD_INTERVAL_SECS
    min_seed_time_secs: float = DEFAULT_MIN_SEED_TIME_SECS
    seeder_reward_per_gb: Decimal = Decimal(str(DEFAULT_SEEDER_REWARD_PER_GB))
    announce_interval_secs: float = 1800.0


@dataclass
class ActiveTorrent:
    """Tracks a torrent currently being seeded."""
    infohash: str
    manifest: TorrentManifest
    started_at: float = field(default_factory=time.time)
    bytes_uploaded: int = 0
    last_reward_at: float = 0.0
    peer_count: int = 0


class BitTorrentProvider:
    """
    Contributes BitTorrent seeding bandwidth to the PRSM network.

    Handles:
    - Creating and seeding new torrents
    - Tracking upload statistics for reward calculation
    - Announcing availability via gossip protocol
    - Earning FTNS rewards for seeding
    """

    def __init__(
        self,
        identity: NodeIdentity,
        transport: WebSocketTransport,
        gossip: GossipProtocol,
        ledger: LocalLedger,
        bt_client: BitTorrentClient,
        manifest_store: TorrentManifestStore,
        config: Optional[BitTorrentProviderConfig] = None,
        node_config: Optional["NodeConfig"] = None,
    ):
        self.identity = identity
        self.transport = transport
        self.gossip = gossip
        self.ledger = ledger
        self.bt_client = bt_client
        self.manifest_store = manifest_store
        self.config = config or BitTorrentProviderConfig()
        self.node_config = node_config

        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._active_torrents: Dict[str, ActiveTorrent] = {}
        self.ledger_sync = None  # Set by node.py

    async def start(self) -> None:
        """Start the provider and begin seeding."""
        if not self.bt_client.available:
            logger.warning("BitTorrent client not available - provider disabled")
            return

        self._running = True

        # Register gossip handlers
        self.gossip.subscribe("bittorrent_request", self._on_torrent_request)

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._reward_loop()))
        self._tasks.append(asyncio.create_task(self._announce_loop()))

        logger.info(
            f"BitTorrent provider started: {len(self._active_torrents)} active torrents"
        )

    async def stop(self) -> None:
        """Stop the provider and all background tasks."""
        self._running = False

        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        logger.info("BitTorrent provider stopped")

    async def seed_content(
        self,
        path: Path,
        name: Optional[str] = None,
        provenance_id: Optional[str] = None,
        piece_length: int = 262144,
    ) -> Optional[TorrentManifest]:
        """
        Create a torrent from local content and begin seeding.

        Args:
            path: Path to file or directory to seed
            name: Optional name for the torrent (defaults to path name)
            provenance_id: Optional PRSM provenance ID
            piece_length: Size of each piece in bytes

        Returns:
            TorrentManifest on success, None on failure
        """
        path = Path(path).expanduser()
        if not path.exists():
            logger.error(f"Path does not exist: {path}")
            return None

        if not self.bt_client.available:
            logger.error("BitTorrent client not available")
            return None

        # Check max torrents limit
        if len(self._active_torrents) >= self.config.max_torrents:
            logger.warning(f"Max torrents reached ({self.config.max_torrents})")
            return None

        # Create the torrent
        result = await self.bt_client.create_torrent(
            path=path,
            piece_length=piece_length,
            comment=f"Seeded by PRSM node {self.identity.node_id[:8]}",
        )

        if not result.success:
            logger.error(f"Failed to create torrent: {result.error}")
            return None

        torrent_bytes = result.metadata.get("torrent_bytes", b"")
        infohash = result.infohash

        if infohash in self._active_torrents:
            logger.info(f"Already seeding: {infohash[:16]}...")
            return self._active_torrents[infohash].manifest

        # Create manifest
        manifest = create_manifest_from_torrent(
            torrent_bytes=torrent_bytes,
            node_id=self.identity.node_id,
            provenance_id=provenance_id,
        )

        if not manifest:
            logger.error("Failed to create manifest from torrent")
            return None

        manifest.name = name or manifest.name or path.name

        # Add to BitTorrent client in seed mode
        add_result = await self.bt_client.add_torrent(
            source=torrent_bytes,
            save_path=path.parent if path.is_file() else path,
            seed_mode=True,
        )

        if not add_result.success:
            logger.error(f"Failed to add torrent: {add_result.error}")
            return None

        # Store manifest
        await self.manifest_store.save(manifest)

        # Track as active
        self._active_torrents[infohash] = ActiveTorrent(
            infohash=infohash,
            manifest=manifest,
            started_at=time.time(),
        )

        # Announce to network
        await self._announce(manifest)

        logger.info(f"Now seeding: {manifest.name} ({infohash[:16]}...)")
        return manifest

    async def stop_seeding(self, infohash: str) -> bool:
        """Stop seeding a torrent."""
        if infohash not in self._active_torrents:
            return False

        active = self._active_torrents[infohash]

        # Check minimum seed time
        seed_time = time.time() - active.started_at
        if seed_time < self.config.min_seed_time_secs:
            logger.warning(
                f"Cannot stop seeding {infohash[:16]}...: "
                f"minimum seed time not met"
            )
            return False

        # Remove from BitTorrent client
        await self.bt_client.remove_torrent(infohash, delete_files=False)

        # Announce withdrawal
        await self._announce_withdraw(infohash)

        del self._active_torrents[infohash]
        logger.info(f"Stopped seeding: {infohash[:16]}...")
        return True

    def get_active_torrents(self) -> List[ActiveTorrent]:
        """Return list of currently active (seeding) torrents."""
        return list(self._active_torrents.values())

    def get_stats(self) -> Dict[str, Any]:
        """Return provider statistics."""
        total_uploaded = sum(t.bytes_uploaded for t in self._active_torrents.values())
        total_peers = sum(t.peer_count for t in self._active_torrents.values())

        return {
            "active_torrents": len(self._active_torrents),
            "max_torrents": self.config.max_torrents,
            "total_uploaded_bytes": total_uploaded,
            "total_uploaded_gb": total_uploaded / (1024**3),
            "total_peers": total_peers,
            "reward_rate_per_gb": float(self.config.seeder_reward_per_gb),
        }

    async def _on_torrent_request(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Handle a request for torrent metadata from another node."""
        infohash = data.get("infohash", "")
        if not infohash:
            return

        active = self._active_torrents.get(infohash)
        if not active:
            return

        # Send the manifest back via direct message
        response = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=self.identity.node_id,
            payload={
                "subtype": "bittorrent_manifest_response",
                "infohash": infohash,
                "manifest": active.manifest.to_dict(),
            },
        )
        await self.transport.send_to_peer(origin, response)

    async def _reward_loop(self) -> None:
        """Periodically calculate and claim rewards for seeding."""
        while self._running:
            await asyncio.sleep(self.config.reward_interval_secs)

            if not self._active_torrents:
                continue

            try:
                total_uploaded_this_interval = 0

                for infohash, active in list(self._active_torrents.items()):
                    status = await self.bt_client.get_status(infohash)
                    if isinstance(status, TorrentInfo):
                        uploaded_delta = status.bytes_uploaded - active.bytes_uploaded
                        active.bytes_uploaded = status.bytes_uploaded
                        active.peer_count = status.seeders + status.leechers
                        total_uploaded_this_interval += max(0, uploaded_delta)

                    active.last_reward_at = time.time()

                if total_uploaded_this_interval > 0:
                    gb_uploaded = total_uploaded_this_interval / (1024**3)
                    reward = Decimal(str(gb_uploaded)) * self.config.seeder_reward_per_gb

                    if reward > 0:
                        tx = await self.ledger.credit(
                            wallet_id=self.identity.node_id,
                            amount=float(reward),
                            tx_type=TransactionType.REWARD,
                            description=f"BitTorrent seeding: {gb_uploaded:.4f}GB uploaded",
                        )

                        if self.ledger_sync:
                            try:
                                await self.ledger_sync.broadcast_transaction(tx)
                            except Exception:
                                pass

                        logger.info(f"Seeder reward: {reward:.6f} FTNS for {gb_uploaded:.4f}GB")

            except Exception as e:
                logger.error(f"Reward loop error: {e}")

    async def _announce_loop(self) -> None:
        """Periodically announce active torrents to the network."""
        while self._running:
            await asyncio.sleep(self.config.announce_interval_secs)

            if not self._active_torrents:
                continue

            try:
                for active in self._active_torrents.values():
                    await self._announce(active.manifest)

            except Exception as e:
                logger.error(f"Announce loop error: {e}")

    async def _announce(self, manifest: TorrentManifest) -> None:
        """Publish bittorrent_announce gossip message."""
        await self.gossip.publish("bittorrent_announce", {
            "infohash": manifest.infohash,
            "name": manifest.name,
            "size_bytes": manifest.total_size,
            "magnet_uri": manifest.magnet_uri,
            "piece_length": manifest.piece_length,
            "num_pieces": manifest.num_pieces,
            "seeder_node_id": self.identity.node_id,
            "timestamp": time.time(),
        })

    async def _announce_withdraw(self, infohash: str) -> None:
        """Publish bittorrent_withdraw gossip message."""
        await self.gossip.publish("bittorrent_withdraw", {
            "infohash": infohash,
            "seeder_node_id": self.identity.node_id,
        })
