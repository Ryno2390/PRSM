"""
Tests for BitTorrent Client

Unit tests for the BitTorrent client module, covering configuration,
torrent creation, seeding, downloading, and status monitoring.
"""

import asyncio
import hashlib
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Dict, Any, List

from prsm.core.bittorrent_client import (
    BitTorrentConfig,
    BitTorrentResult,
    BitTorrentClient,
    TorrentInfo,
    TorrentState,
    FileEntry,
    PeerInfo,
    get_bittorrent_client,
    LT_AVAILABLE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_bt_config():
    """Default BitTorrent configuration for tests."""
    return BitTorrentConfig(
        port_range_start=6881,
        port_range_end=6891,
        dht_enabled=True,
        download_dir="/tmp/test_torrents",
        max_uploads=4,
        max_connections=50,
        upload_rate_limit=0,
        download_rate_limit=0,
        alert_poll_interval=0.1,
        piece_length=262144,
    )


@pytest.fixture
def mock_libtorrent():
    """Mock libtorrent module for testing without the actual library."""
    with patch('prsm.core.bittorrent_client.lt') as mock_lt:
        # Mock settings_pack
        mock_settings = MagicMock()
        mock_settings.set_int = MagicMock()
        mock_settings.set_bool = MagicMock()
        mock_lt.settings_pack.return_value = mock_settings

        # Mock session
        mock_session = MagicMock()
        mock_session.add_torrent = MagicMock(return_value=MagicMock())
        mock_session.remove_torrent = MagicMock()
        mock_session.pop_alerts = MagicMock(return_value=[])
        mock_lt.session.return_value = mock_session

        # Mock torrent_status enum
        mock_lt.torrent_status.downloading = 3
        mock_lt.torrent_status.seeding = 5
        mock_lt.torrent_status.finished = 5
        mock_lt.torrent_status.checking_files = 1
        mock_lt.torrent_status.checking_resume_data = 2
        mock_lt.torrent_status.allocating = 0
        mock_lt.torrent_status.queued_for_checking = 0

        # Mock torrent_flags
        mock_lt.torrent_flags.seed_mode = 1

        yield mock_lt


@pytest.fixture
def mock_torrent_handle():
    """Mock torrent handle for testing."""
    handle = MagicMock()
    handle.is_valid = MagicMock(return_value=True)
    handle.has_metadata = MagicMock(return_value=True)
    handle.info_hash = MagicMock(return_value="abc123def456")
    handle.pause = MagicMock()
    handle.save_resume_data = MagicMock()

    # Mock status
    status = MagicMock()
    status.name = "test_torrent"
    status.state = 5  # seeding
    status.progress = 1.0
    status.total_wanted = 1024 * 1024  # 1MB
    status.total_wanted_done = 1024 * 1024
    status.total_done = 1024 * 1024
    status.total_upload = 2 * 1024 * 1024  # 2MB uploaded
    status.download_rate = 0.0
    status.upload_rate = 100.0
    status.num_seeds = 5
    status.num_peers = 10
    status.paused = False
    status.error = None
    handle.status = MagicMock(return_value=status)

    # Mock torrent_info
    torrent_info = MagicMock()
    torrent_info.name = MagicMock(return_value="test_torrent")
    torrent_info.total_size = MagicMock(return_value=1024 * 1024)
    torrent_info.piece_length = MagicMock(return_value=262144)
    torrent_info.num_pieces = MagicMock(return_value=4)
    torrent_info.num_files = MagicMock(return_value=1)
    torrent_info.file_at = MagicMock(return_value=MagicMock(
        path="test_file.bin",
        size=1024 * 1024,
        offset=0,
    ))
    handle.get_torrent_info = MagicMock(return_value=torrent_info)

    return handle


# =============================================================================
# Configuration Tests
# =============================================================================

class TestBitTorrentConfig:
    """Tests for BitTorrentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BitTorrentConfig()

        assert config.port_range_start == 6881
        assert config.port_range_end == 6891
        assert config.dht_enabled is True
        assert config.download_dir == "~/.prsm/torrents"
        assert config.max_uploads == 4
        assert config.max_connections == 50
        assert config.upload_rate_limit == 0
        assert config.download_rate_limit == 0
        assert config.alert_poll_interval == 0.1
        assert config.piece_length == 262144

    def test_custom_config(self, default_bt_config):
        """Test custom configuration values."""
        assert default_bt_config.port_range_start == 6881
        assert default_bt_config.download_dir == "/tmp/test_torrents"

    def test_get_download_dir_expands_home(self, tmp_path):
        """Test that get_download_dir expands ~ to home directory."""
        config = BitTorrentConfig(download_dir="~/test_torrents")
        result = config.get_download_dir()
        assert str(result).startswith(str(Path.home()))

    def test_dht_bootstrap_nodes_default(self):
        """Test default DHT bootstrap nodes."""
        config = BitTorrentConfig()
        assert len(config.dht_bootstrap_nodes) == 3
        assert "router.bittorrent.com:6881" in config.dht_bootstrap_nodes


# =============================================================================
# Dataclass Tests
# =============================================================================

class TestDataclasses:
    """Tests for dataclasses used by BitTorrent client."""

    def test_bit_torrent_result_success(self):
        """Test BitTorrentResult for success case."""
        result = BitTorrentResult(
            success=True,
            infohash="abc123def456",
            metadata={"name": "test"}
        )
        assert result.success is True
        assert result.infohash == "abc123def456"
        assert result.error is None
        assert result.metadata == {"name": "test"}

    def test_bit_torrent_result_failure(self):
        """Test BitTorrentResult for failure case."""
        result = BitTorrentResult(
            success=False,
            error="Connection failed"
        )
        assert result.success is False
        assert result.error == "Connection failed"

    def test_file_entry(self):
        """Test FileEntry dataclass."""
        entry = FileEntry(
            path="test/file.bin",
            size_bytes=1024,
            offset_in_torrent=0
        )
        assert entry.path == "test/file.bin"
        assert entry.size_bytes == 1024

    def test_peer_info(self):
        """Test PeerInfo dataclass."""
        peer = PeerInfo(
            peer_id="peer123",
            ip="192.168.1.1",
            port=6881,
            client="uTorrent",
            downloaded=1024,
            uploaded=2048,
            is_seed=True
        )
        assert peer.peer_id == "peer123"
        assert peer.ip == "192.168.1.1"
        assert peer.port == 6881
        assert peer.is_seed is True

    def test_torrent_info_to_dict(self):
        """Test TorrentInfo serialization."""
        info = TorrentInfo(
            infohash="abc123",
            name="test_torrent",
            size_bytes=1024,
            piece_length=262144,
            num_pieces=4,
            files=[FileEntry(path="test.bin", size_bytes=1024)],
            seeders=5,
            leechers=2,
            progress=0.5,
            state=TorrentState.DOWNLOADING,
        )
        d = info.to_dict()

        assert d["infohash"] == "abc123"
        assert d["name"] == "test_torrent"
        assert d["progress"] == 0.5
        assert d["state"] == "downloading"
        assert len(d["files"]) == 1


# =============================================================================
# TorrentState Tests
# =============================================================================

class TestTorrentState:
    """Tests for TorrentState enum."""

    def test_all_states_exist(self):
        """Test that all expected states are defined."""
        assert TorrentState.QUEUED.value == "queued"
        assert TorrentState.CHECKING.value == "checking"
        assert TorrentState.DOWNLOADING.value == "downloading"
        assert TorrentState.SEEDING.value == "seeding"
        assert TorrentState.PAUSED.value == "paused"
        assert TorrentState.ERROR.value == "error"
        assert TorrentState.UNKNOWN.value == "unknown"

    def test_state_string_comparison(self):
        """Test that states can be compared as strings."""
        assert TorrentState.SEEDING == "seeding"
        assert TorrentState.DOWNLOADING != "seeding"


# =============================================================================
# BitTorrentClient Tests
# =============================================================================

class TestBitTorrentClient:
    """Tests for BitTorrentClient class."""

    def test_client_initialization(self, default_bt_config):
        """Test client initialization with config."""
        client = BitTorrentClient(config=default_bt_config)

        assert client.config == default_bt_config
        assert client._session is None
        assert client._torrents == {}
        assert client._initialized is False

    def test_available_property_not_initialized(self, default_bt_config):
        """Test available property when not initialized."""
        client = BitTorrentClient(config=default_bt_config)

        # Without libtorrent or initialization, should be False
        assert client.available is False

    @pytest.mark.asyncio
    async def test_initialize_without_libtorrent(self, default_bt_config):
        """Test initialization when libtorrent is not available."""
        with patch('prsm.core.bittorrent_client.LT_AVAILABLE', False):
            client = BitTorrentClient(config=default_bt_config)
            result = await client.initialize()

            assert result is False
            assert client._initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_without_initialization(self, default_bt_config):
        """Test shutdown when client was never initialized."""
        client = BitTorrentClient(config=default_bt_config)

        # Should not raise any errors
        await client.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self, default_bt_config, mock_libtorrent):
        """Test that shutdown cleans up resources properly."""
        with patch('prsm.core.bittorrent_client.LT_AVAILABLE', True), \
             patch('prsm.core.bittorrent_client.lt', mock_libtorrent):
            client = BitTorrentClient(config=default_bt_config)
            await client.initialize()

            # Add a mock torrent
            mock_handle = MagicMock()
            mock_handle.is_valid = MagicMock(return_value=True)
            mock_handle.has_metadata = MagicMock(return_value=True)
            mock_handle.pause = MagicMock()
            client._torrents["test_infohash"] = mock_handle

            await client.shutdown()

            assert client._running is False
            assert client._initialized is False
            assert len(client._torrents) == 0

    @pytest.mark.asyncio
    async def test_create_torrent_not_initialized(self, default_bt_config, tmp_path):
        """Test create_torrent when client not initialized."""
        client = BitTorrentClient(config=default_bt_config)

        # Create a test file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"x" * 1024)

        result = await client.create_torrent(test_file)

        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_create_torrent_path_not_exists(self, default_bt_config, mock_libtorrent):
        """Test create_torrent with non-existent path."""
        with patch('prsm.core.bittorrent_client.LT_AVAILABLE', True), \
             patch('prsm.core.bittorrent_client.lt', mock_libtorrent):
            client = BitTorrentClient(config=default_bt_config)
            await client.initialize()

            result = await client.create_torrent(Path("/nonexistent/path"))

            assert result.success is False
            assert "does not exist" in result.error.lower()

    @pytest.mark.asyncio
    async def test_add_torrent_not_initialized(self, default_bt_config):
        """Test add_torrent when client not initialized."""
        client = BitTorrentClient(config=default_bt_config)

        result = await client.add_torrent(b"fake torrent bytes")

        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_remove_torrent_not_found(self, default_bt_config, mock_libtorrent):
        """Test remove_torrent when torrent doesn't exist."""
        with patch('prsm.core.bittorrent_client.LT_AVAILABLE', True), \
             patch('prsm.core.bittorrent_client.lt', mock_libtorrent):
            client = BitTorrentClient(config=default_bt_config)
            await client.initialize()

            result = await client.remove_torrent("nonexistent_infohash")

            assert result.success is False
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_status_empty(self, default_bt_config, mock_libtorrent):
        """Test get_status when no torrents are active."""
        with patch('prsm.core.bittorrent_client.LT_AVAILABLE', True), \
             patch('prsm.core.bittorrent_client.lt', mock_libtorrent):
            client = BitTorrentClient(config=default_bt_config)
            await client.initialize()

            result = await client.get_status()

            assert result == []

    @pytest.mark.asyncio
    async def test_get_status_single_torrent(self, default_bt_config, mock_libtorrent, mock_torrent_handle):
        """Test get_status for a single torrent."""
        with patch('prsm.core.bittorrent_client.LT_AVAILABLE', True), \
             patch('prsm.core.bittorrent_client.lt', mock_libtorrent):
            client = BitTorrentClient(config=default_bt_config)
            await client.initialize()

            # Add mock torrent
            client._torrents["abc123"] = mock_torrent_handle

            result = await client.get_status("abc123")

            assert isinstance(result, TorrentInfo)
            assert result.infohash == "abc123"
            assert result.state == TorrentState.SEEDING

    @pytest.mark.asyncio
    async def test_wait_for_completion_not_found(self, default_bt_config, mock_libtorrent):
        """Test wait_for_completion when torrent doesn't exist."""
        with patch('prsm.core.bittorrent_client.LT_AVAILABLE', True), \
             patch('prsm.core.bittorrent_client.lt', mock_libtorrent):
            client = BitTorrentClient(config=default_bt_config)
            await client.initialize()

            result = await client.wait_for_completion("nonexistent", timeout=1.0)

            assert result.success is False
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_wait_for_completion_already_complete(
        self, default_bt_config, mock_libtorrent, mock_torrent_handle
    ):
        """Test wait_for_completion when torrent is already complete."""
        with patch('prsm.core.bittorrent_client.LT_AVAILABLE', True), \
             patch('prsm.core.bittorrent_client.lt', mock_libtorrent):
            client = BitTorrentClient(config=default_bt_config)
            await client.initialize()

            # Add mock torrent (already seeding with progress=1.0)
            client._torrents["abc123"] = mock_torrent_handle

            result = await client.wait_for_completion("abc123", timeout=5.0)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_get_peers_empty(self, default_bt_config, mock_libtorrent):
        """Test get_peers when no peers are connected."""
        with patch('prsm.core.bittorrent_client.LT_AVAILABLE', True), \
             patch('prsm.core.bittorrent_client.lt', mock_libtorrent):
            client = BitTorrentClient(config=default_bt_config)
            await client.initialize()

            result = await client.get_peers("nonexistent")

            assert result == []

    def test_on_alert_registration(self, default_bt_config):
        """Test alert callback registration."""
        client = BitTorrentClient(config=default_bt_config)

        callback = MagicMock()
        client.on_alert("torrent_finished", callback)

        assert "torrent_finished" in client._alert_callbacks
        assert callback in client._alert_callbacks["torrent_finished"]


# =============================================================================
# Global Client Tests
# =============================================================================

class TestGetBitTorrentClient:
    """Tests for the global client getter."""

    def test_get_client_creates_instance(self):
        """Test that get_bittorrent_client creates a client."""
        # Reset the global client
        import prsm.core.bittorrent_client as bt_module
        bt_module._bittorrent_client = None

        client = get_bittorrent_client()
        assert client is not None
        assert isinstance(client, BitTorrentClient)

    def test_get_client_returns_same_instance(self):
        """Test that get_bittorrent_client returns the same instance."""
        # Reset the global client
        import prsm.core.bittorrent_client as bt_module
        bt_module._bittorrent_client = None

        client1 = get_bittorrent_client()
        client2 = get_bittorrent_client()

        assert client1 is client2


# =============================================================================
# Integration-style Tests (with mocked libtorrent)
# =============================================================================

class TestBitTorrentClientIntegration:
    """Integration-style tests with mocked libtorrent."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, default_bt_config, mock_libtorrent, tmp_path):
        """Test a complete workflow: init, add torrent, check status, shutdown."""
        with patch('prsm.core.bittorrent_client.LT_AVAILABLE', True), \
             patch('prsm.core.bittorrent_client.lt', mock_libtorrent):
            # Initialize
            client = BitTorrentClient(config=default_bt_config)
            init_result = await client.initialize()
            assert init_result is True

            # Check available
            assert client.available is True

            # Get status (empty)
            status = await client.get_status()
            assert status == []

            # Shutdown
            await client.shutdown()
            assert client.available is False

    @pytest.mark.asyncio
    async def test_progress_callback(self, default_bt_config, mock_libtorrent, mock_torrent_handle):
        """Test wait_for_completion with progress callback."""
        with patch('prsm.core.bittorrent_client.LT_AVAILABLE', True), \
             patch('prsm.core.bittorrent_client.lt', mock_libtorrent):
            client = BitTorrentClient(config=default_bt_config)
            await client.initialize()

            # Add mock torrent
            client._torrents["abc123"] = mock_torrent_handle

            # Track callback invocations
            callback_calls = []

            def progress_callback(infohash, progress, stats):
                callback_calls.append({
                    "infohash": infohash,
                    "progress": progress,
                    "stats": stats,
                })

            result = await client.wait_for_completion(
                "abc123",
                timeout=5.0,
                progress_callback=progress_callback,
            )

            # Since the mock torrent is already at 100%, callback might not be called
            # or might be called once. Either is fine for this test.
            assert result.success is True
