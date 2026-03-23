"""
Tests for BitTorrent Provider

Unit tests for the BitTorrent provider module, covering seeding,
reward calculation, and gossip announcements.
"""

import asyncio
import pytest
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from prsm.node.bittorrent_provider import (
    BitTorrentProviderConfig,
    ActiveTorrent,
    BitTorrentProvider,
    DEFAULT_MAX_TORRENTS,
    DEFAULT_REWARD_INTERVAL_SECS,
    DEFAULT_MIN_SEED_TIME_SECS,
    DEFAULT_SEEDER_REWARD_PER_GB,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def provider_config():
    """Default provider configuration for tests."""
    return BitTorrentProviderConfig(
        max_torrents=50,
        data_dir="/tmp/test_torrents",
        reward_interval_secs=3600,
        min_seed_time_secs=3600,
        seeder_reward_per_gb=Decimal("0.10"),
        announce_interval_secs=1800.0,
    )


@pytest.fixture
def mock_identity():
    """Mock node identity."""
    identity = MagicMock()
    identity.node_id = "test_node_123"
    return identity


@pytest.fixture
def mock_transport():
    """Mock WebSocket transport."""
    transport = MagicMock()
    transport.send = AsyncMock()
    return transport


@pytest.fixture
def mock_gossip():
    """Mock gossip protocol."""
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock()
    return gossip


@pytest.fixture
def mock_ledger():
    """Mock local ledger."""
    ledger = MagicMock()
    ledger.credit = AsyncMock(return_value=True)
    ledger.debit = AsyncMock(return_value=True)
    return ledger


@pytest.fixture
def mock_bt_client():
    """Mock BitTorrent client."""
    client = MagicMock()
    client.available = True

    # Mock create_torrent result
    create_result = MagicMock()
    create_result.success = True
    create_result.infohash = "abc123def456789"
    create_result.metadata = {
        "torrent_bytes": b"fake torrent",
        "name": "test_torrent",
        "total_size": 1024 * 1024,
        "piece_length": 262144,
        "num_pieces": 4,
    }
    client.create_torrent = AsyncMock(return_value=create_result)

    # Mock add_torrent result
    add_result = MagicMock()
    add_result.success = True
    add_result.infohash = "abc123def456789"
    client.add_torrent = AsyncMock(return_value=add_result)

    # Mock get_status
    from prsm.core.bittorrent_client import TorrentInfo, TorrentState
    status = TorrentInfo(
        infohash="abc123def456789",
        name="test_torrent",
        size_bytes=1024 * 1024,
        piece_length=262144,
        num_pieces=4,
        progress=1.0,
        state=TorrentState.SEEDING,
        seeders=1,
        leechers=0,
        bytes_uploaded=2 * 1024 * 1024,
    )
    client.get_status = AsyncMock(return_value=status)

    # Mock remove_torrent
    remove_result = MagicMock()
    remove_result.success = True
    client.remove_torrent = AsyncMock(return_value=remove_result)

    return client


@pytest.fixture
def mock_manifest_store():
    """Mock torrent manifest store."""
    store = MagicMock()
    store.save = AsyncMock()
    store.load = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_torrent_manifest():
    """Mock torrent manifest."""
    from prsm.core.bittorrent_manifest import TorrentManifest, PieceInfo

    manifest = MagicMock(spec=TorrentManifest)
    manifest.infohash = "abc123def456789"
    manifest.name = "test_torrent"
    manifest.total_size = 1024 * 1024
    manifest.piece_length = 262144
    manifest.num_pieces = 4
    manifest.magnet_uri = "magnet:?xt=urn:btih:abc123def456789"
    manifest.pieces = [
        PieceInfo(index=0, hash="piece1", size=262144, verified=True),
        PieceInfo(index=1, hash="piece2", size=262144, verified=True),
        PieceInfo(index=2, hash="piece3", size=262144, verified=True),
        PieceInfo(index=3, hash="piece4", size=262144, verified=True),
    ]
    manifest.to_dict = MagicMock(return_value={
        "infohash": "abc123def456789",
        "name": "test_torrent",
    })
    return manifest


# =============================================================================
# Configuration Tests
# =============================================================================

class TestBitTorrentProviderConfig:
    """Tests for BitTorrentProviderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BitTorrentProviderConfig()

        assert config.max_torrents == DEFAULT_MAX_TORRENTS
        assert config.reward_interval_secs == DEFAULT_REWARD_INTERVAL_SECS
        assert config.min_seed_time_secs == DEFAULT_MIN_SEED_TIME_SECS
        assert config.seeder_reward_per_gb == Decimal(str(DEFAULT_SEEDER_REWARD_PER_GB))

    def test_custom_config(self, provider_config):
        """Test custom configuration values."""
        assert provider_config.max_torrents == 50
        assert provider_config.data_dir == "/tmp/test_torrents"
        assert provider_config.seeder_reward_per_gb == Decimal("0.10")


# =============================================================================
# ActiveTorrent Tests
# =============================================================================

class TestActiveTorrent:
    """Tests for ActiveTorrent dataclass."""

    def test_active_torrent_defaults(self, mock_torrent_manifest):
        """Test ActiveTorrent with default values."""
        torrent = ActiveTorrent(
            infohash="abc123",
            manifest=mock_torrent_manifest,
        )

        assert torrent.infohash == "abc123"
        assert torrent.bytes_uploaded == 0
        assert torrent.peer_count == 0
        assert torrent.started_at > 0

    def test_active_torrent_with_values(self, mock_torrent_manifest):
        """Test ActiveTorrent with custom values."""
        torrent = ActiveTorrent(
            infohash="abc123",
            manifest=mock_torrent_manifest,
            started_at=1000.0,
            bytes_uploaded=1024 * 1024,
            last_reward_at=1100.0,
            peer_count=5,
        )

        assert torrent.started_at == 1000.0
        assert torrent.bytes_uploaded == 1024 * 1024
        assert torrent.last_reward_at == 1100.0
        assert torrent.peer_count == 5


# =============================================================================
# BitTorrentProvider Tests
# =============================================================================

class TestBitTorrentProvider:
    """Tests for BitTorrentProvider class."""

    def test_provider_initialization(
        self,
        mock_identity,
        mock_transport,
        mock_gossip,
        mock_ledger,
        mock_bt_client,
        mock_manifest_store,
        provider_config,
    ):
        """Test provider initialization."""
        provider = BitTorrentProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
            bt_client=mock_bt_client,
            manifest_store=mock_manifest_store,
            config=provider_config,
        )

        assert provider.identity == mock_identity
        assert provider.gossip == mock_gossip
        assert provider.ledger == mock_ledger
        assert provider.bt_client == mock_bt_client
        assert provider._running is False
        assert len(provider._active_torrents) == 0

    @pytest.mark.asyncio
    async def test_provider_start(
        self,
        mock_identity,
        mock_transport,
        mock_gossip,
        mock_ledger,
        mock_bt_client,
        mock_manifest_store,
        provider_config,
    ):
        """Test provider start method."""
        provider = BitTorrentProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
            bt_client=mock_bt_client,
            manifest_store=mock_manifest_store,
            config=provider_config,
        )

        await provider.start()

        assert provider._running is True
        mock_gossip.subscribe.assert_called_once()

        # Clean up
        await provider.stop()

    @pytest.mark.asyncio
    async def test_provider_stop(
        self,
        mock_identity,
        mock_transport,
        mock_gossip,
        mock_ledger,
        mock_bt_client,
        mock_manifest_store,
        provider_config,
    ):
        """Test provider stop method."""
        provider = BitTorrentProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
            bt_client=mock_bt_client,
            manifest_store=mock_manifest_store,
            config=provider_config,
        )

        await provider.start()
        await provider.stop()

        assert provider._running is False
        assert len(provider._tasks) == 0

    @pytest.mark.asyncio
    async def test_provider_start_without_client(
        self,
        mock_identity,
        mock_transport,
        mock_gossip,
        mock_ledger,
        mock_bt_client,
        mock_manifest_store,
        provider_config,
    ):
        """Test provider start when BitTorrent client is not available."""
        mock_bt_client.available = False

        provider = BitTorrentProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
            bt_client=mock_bt_client,
            manifest_store=mock_manifest_store,
            config=provider_config,
        )

        await provider.start()

        # Should not have started background tasks
        assert len(provider._tasks) == 0

    @pytest.mark.asyncio
    async def test_seed_content(
        self,
        mock_identity,
        mock_transport,
        mock_gossip,
        mock_ledger,
        mock_bt_client,
        mock_manifest_store,
        provider_config,
        tmp_path,
    ):
        """Test seed_content method."""
        # Create test file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"x" * 1024)

        provider = BitTorrentProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
            bt_client=mock_bt_client,
            manifest_store=mock_manifest_store,
            config=provider_config,
        )

        # Mock create_manifest_from_torrent
        with patch('prsm.node.bittorrent_provider.create_manifest_from_torrent') as mock_create:
            mock_create.return_value = MagicMock(
                infohash="abc123def456789",
                name="test_torrent",
            )

            result = await provider.seed_content(test_file)

            assert result is not None
            mock_bt_client.create_torrent.assert_called_once()
            mock_bt_client.add_torrent.assert_called_once()

    @pytest.mark.asyncio
    async def test_seed_content_client_unavailable(
        self,
        mock_identity,
        mock_transport,
        mock_gossip,
        mock_ledger,
        mock_bt_client,
        mock_manifest_store,
        provider_config,
        tmp_path,
    ):
        """Test seed_content when client is unavailable."""
        mock_bt_client.available = False

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"x" * 1024)

        provider = BitTorrentProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
            bt_client=mock_bt_client,
            manifest_store=mock_manifest_store,
            config=provider_config,
        )

        result = await provider.seed_content(test_file)

        assert result is None

    @pytest.mark.asyncio
    async def test_stop_seeding(
        self,
        mock_identity,
        mock_transport,
        mock_gossip,
        mock_ledger,
        mock_bt_client,
        mock_manifest_store,
        provider_config,
        mock_torrent_manifest,
    ):
        """Test stop_seeding method."""
        provider = BitTorrentProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
            bt_client=mock_bt_client,
            manifest_store=mock_manifest_store,
            config=provider_config,
        )

        # Add a torrent to active list
        provider._active_torrents["abc123"] = ActiveTorrent(
            infohash="abc123",
            manifest=mock_torrent_manifest,
            started_at=0,  # Old enough to stop
        )

        result = await provider.stop_seeding("abc123")

        assert result is True
        assert "abc123" not in provider._active_torrents
        mock_bt_client.remove_torrent.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_seeding_not_found(
        self,
        mock_identity,
        mock_transport,
        mock_gossip,
        mock_ledger,
        mock_bt_client,
        mock_manifest_store,
        provider_config,
    ):
        """Test stop_seeding when torrent not found."""
        provider = BitTorrentProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
            bt_client=mock_bt_client,
            manifest_store=mock_manifest_store,
            config=provider_config,
        )

        result = await provider.stop_seeding("nonexistent")

        assert result is False

    def test_get_active_torrents(
        self,
        mock_identity,
        mock_transport,
        mock_gossip,
        mock_ledger,
        mock_bt_client,
        mock_manifest_store,
        provider_config,
        mock_torrent_manifest,
    ):
        """Test get_active_torrents method."""
        provider = BitTorrentProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
            bt_client=mock_bt_client,
            manifest_store=mock_manifest_store,
            config=provider_config,
        )

        # Add a torrent
        provider._active_torrents["abc123"] = ActiveTorrent(
            infohash="abc123",
            manifest=mock_torrent_manifest,
        )

        result = provider.get_active_torrents()

        assert len(result) == 1
        assert result[0].infohash == "abc123"

    def test_get_stats(
        self,
        mock_identity,
        mock_transport,
        mock_gossip,
        mock_ledger,
        mock_bt_client,
        mock_manifest_store,
        provider_config,
        mock_torrent_manifest,
    ):
        """Test get_stats method."""
        provider = BitTorrentProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
            bt_client=mock_bt_client,
            manifest_store=mock_manifest_store,
            config=provider_config,
        )

        # Add a torrent with some upload
        provider._active_torrents["abc123"] = ActiveTorrent(
            infohash="abc123",
            manifest=mock_torrent_manifest,
            bytes_uploaded=1024 * 1024,
        )

        stats = provider.get_stats()

        assert stats["active_torrents"] == 1
        assert stats["total_uploaded_bytes"] == 1024 * 1024


# =============================================================================
# Reward Calculation Tests
# =============================================================================

class TestRewardCalculation:
    """Tests for reward calculation logic."""

    def test_reward_per_gb_calculation(self):
        """Test FTNS reward calculation per GB uploaded."""
        reward_per_gb = Decimal("0.10")
        bytes_uploaded = 1024 * 1024 * 1024  # 1 GB

        reward = Decimal(str(bytes_uploaded / (1024 * 1024 * 1024))) * reward_per_gb

        assert reward == Decimal("0.10")

    def test_partial_gb_reward(self):
        """Test reward for partial GB upload."""
        reward_per_gb = Decimal("0.10")
        bytes_uploaded = 512 * 1024 * 1024  # 0.5 GB

        reward = Decimal(str(bytes_uploaded / (1024 * 1024 * 1024))) * reward_per_gb

        assert reward == Decimal("0.05")

    def test_multiple_gb_reward(self):
        """Test reward for multiple GB upload."""
        reward_per_gb = Decimal("0.10")
        bytes_uploaded = 5 * 1024 * 1024 * 1024  # 5 GB

        reward = Decimal(str(bytes_uploaded / (1024 * 1024 * 1024))) * reward_per_gb

        assert reward == Decimal("0.50")
