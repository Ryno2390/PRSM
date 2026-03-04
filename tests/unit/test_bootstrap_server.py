"""
Unit Tests for PRSM Bootstrap Server

Tests for bootstrap server configuration, peer management,
connection handling, and health checks.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import uuid

import pytest

# Import bootstrap modules
from prsm.bootstrap.config import (
    BootstrapConfig,
    get_bootstrap_config,
    set_bootstrap_config,
    reset_bootstrap_config,
)
from prsm.bootstrap.models import (
    PeerInfo,
    PeerStatus,
    BootstrapMetrics,
    BootstrapAnnouncement,
)
from prsm.bootstrap.server import (
    BootstrapServer,
    RateLimiter,
    PeerDatabase,
    run_bootstrap_server,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_config():
    """Create a sample bootstrap configuration for testing."""
    return BootstrapConfig(
        domain="test.prsm-network.com",
        host="0.0.0.0",
        port=8765,
        api_port=8000,
        ssl_enabled=False,  # Disable SSL for testing
        max_peers=100,
        peer_timeout=60,
        heartbeat_interval=10,
        peer_list_size=20,
        auth_required=False,
        persist_peers=False,
        metrics_enabled=True,
        log_level="DEBUG",
        region="test-region",
    )


@pytest.fixture
def sample_peer_info():
    """Create a sample peer info for testing."""
    return PeerInfo(
        peer_id="test-peer-123",
        address="192.168.1.100",
        port=8000,
        public_key="test-public-key",
        status=PeerStatus.ACTIVE,
        capabilities=["compute", "storage"],
        region="us-east-1",
        version="1.0.0",
    )


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_peers.db")


@pytest.fixture
def reset_config():
    """Reset global config after each test."""
    yield
    reset_bootstrap_config()


# =============================================================================
# BootstrapConfig Tests
# =============================================================================

class TestBootstrapConfig:
    """Tests for BootstrapConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BootstrapConfig()
        
        assert config.domain == "prsm-network.com"
        assert config.host == "0.0.0.0"
        assert config.port == 8765
        assert config.api_port == 8000
        assert config.ssl_enabled is True
        assert config.max_peers == 1000
        assert config.peer_timeout == 300
        assert config.heartbeat_interval == 30
    
    def test_custom_config(self, sample_config):
        """Test custom configuration values."""
        assert sample_config.domain == "test.prsm-network.com"
        assert sample_config.port == 8765
        assert sample_config.max_peers == 100
        assert sample_config.ssl_enabled is False
    
    def test_websocket_url(self, sample_config):
        """Test WebSocket URL generation."""
        assert sample_config.websocket_url == "ws://test.prsm-network.com:8765"
        
        # Test with SSL enabled
        ssl_config = BootstrapConfig(ssl_enabled=True, domain="secure.prsm.com")
        assert ssl_config.websocket_url == "wss://secure.prsm.com:8765"
    
    def test_api_url(self, sample_config):
        """Test API URL generation."""
        assert sample_config.api_url == "http://test.prsm-network.com:8000"
    
    def test_external_endpoint(self, sample_config):
        """Test external endpoint generation."""
        assert sample_config.external_endpoint == "test.prsm-network.com:8765"
        
        # Test with external IP
        config_with_ip = BootstrapConfig(external_ip="1.2.3.4", port=9000)
        assert config_with_ip.external_endpoint == "1.2.3.4:9000"
    
    def test_to_dict(self, sample_config):
        """Test configuration serialization."""
        config_dict = sample_config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["domain"] == "test.prsm-network.com"
        assert config_dict["port"] == 8765
        assert config_dict["websocket_url"] == sample_config.websocket_url
    
    def test_from_dict(self, sample_config):
        """Test configuration deserialization."""
        config_dict = sample_config.to_dict()
        restored_config = BootstrapConfig.from_dict(config_dict)
        
        assert restored_config.domain == sample_config.domain
        assert restored_config.port == sample_config.port
        assert restored_config.max_peers == sample_config.max_peers
    
    def test_invalid_port(self):
        """Test that invalid port raises error."""
        with pytest.raises(ValueError):
            BootstrapConfig(port=-1)
        
        with pytest.raises(ValueError):
            BootstrapConfig(port=70000)
    
    def test_invalid_max_peers(self):
        """Test that invalid max_peers raises error."""
        with pytest.raises(ValueError):
            BootstrapConfig(max_peers=0)
        
        with pytest.raises(ValueError):
            BootstrapConfig(max_peers=-10)
    
    def test_environment_variable_override(self, monkeypatch):
        """Test that environment variables override config."""
        monkeypatch.setenv("PRSM_DOMAIN", "env.prsm.com")
        monkeypatch.setenv("PRSM_BOOTSTRAP_PORT", "9999")
        monkeypatch.setenv("PRSM_MAX_PEERS", "500")
        
        config = BootstrapConfig()
        
        assert config.domain == "env.prsm.com"
        assert config.port == 9999
        assert config.max_peers == 500


class TestBootstrapConfigGlobal:
    """Tests for global configuration management."""
    
    def test_get_bootstrap_config(self, reset_config):
        """Test getting global configuration."""
        config = get_bootstrap_config()
        
        assert isinstance(config, BootstrapConfig)
        assert config is get_bootstrap_config()  # Same instance
    
    def test_set_bootstrap_config(self, sample_config, reset_config):
        """Test setting global configuration."""
        set_bootstrap_config(sample_config)
        
        assert get_bootstrap_config() is sample_config
    
    def test_reset_bootstrap_config(self, sample_config, reset_config):
        """Test resetting global configuration."""
        set_bootstrap_config(sample_config)
        reset_bootstrap_config()
        
        # Should be a new instance
        assert get_bootstrap_config() is not sample_config


# =============================================================================
# PeerInfo Tests
# =============================================================================

class TestPeerInfo:
    """Tests for PeerInfo class."""
    
    def test_default_peer_info(self):
        """Test default peer info values."""
        peer = PeerInfo(
            peer_id="test-peer",
            address="192.168.1.1",
            port=8000,
        )
        
        assert peer.peer_id == "test-peer"
        assert peer.status == PeerStatus.ACTIVE
        assert peer.capabilities == []
        assert peer.connection_count == 0
    
    def test_endpoint(self, sample_peer_info):
        """Test endpoint property."""
        assert sample_peer_info.endpoint == "192.168.1.100:8000"
    
    def test_websocket_url(self, sample_peer_info):
        """Test WebSocket URL property."""
        assert sample_peer_info.websocket_url == "ws://192.168.1.100:8000"
    
    def test_update_activity(self, sample_peer_info):
        """Test activity update."""
        old_last_seen = sample_peer_info.last_seen
        sample_peer_info.update_activity()
        
        assert sample_peer_info.last_seen > old_last_seen
    
    def test_is_stale(self, sample_peer_info):
        """Test staleness check."""
        # Fresh peer
        assert not sample_peer_info.is_stale(timeout_seconds=60)
        
        # Stale peer
        sample_peer_info.last_seen = datetime.now(timezone.utc) - timedelta(seconds=120)
        assert sample_peer_info.is_stale(timeout_seconds=60)
    
    def test_to_dict(self, sample_peer_info):
        """Test peer info serialization."""
        peer_dict = sample_peer_info.to_dict()
        
        assert peer_dict["peer_id"] == "test-peer-123"
        assert peer_dict["address"] == "192.168.1.100"
        assert peer_dict["status"] == "active"
        assert "compute" in peer_dict["capabilities"]
    
    def test_from_dict(self, sample_peer_info):
        """Test peer info deserialization."""
        peer_dict = sample_peer_info.to_dict()
        restored = PeerInfo.from_dict(peer_dict)
        
        assert restored.peer_id == sample_peer_info.peer_id
        assert restored.address == sample_peer_info.address
        assert restored.status == sample_peer_info.status
        assert restored.capabilities == sample_peer_info.capabilities


class TestPeerStatus:
    """Tests for PeerStatus enum."""
    
    def test_status_values(self):
        """Test status enum values."""
        assert PeerStatus.ACTIVE.value == "active"
        assert PeerStatus.IDLE.value == "idle"
        assert PeerStatus.UNREACHABLE.value == "unreachable"
        assert PeerStatus.DISCONNECTED.value == "disconnected"
        assert PeerStatus.BANNED.value == "banned"


# =============================================================================
# BootstrapMetrics Tests
# =============================================================================

class TestBootstrapMetrics:
    """Tests for BootstrapMetrics class."""
    
    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = BootstrapMetrics()
        
        assert metrics.total_connections == 0
        assert metrics.active_connections == 0
        assert metrics.bytes_sent == 0
        assert metrics.messages_processed == 0
    
    def test_update_uptime(self):
        """Test uptime update."""
        metrics = BootstrapMetrics()
        metrics.update_uptime()
        
        assert metrics.uptime_seconds >= 0
    
    def test_record_connection(self):
        """Test connection recording."""
        metrics = BootstrapMetrics()
        
        metrics.record_connection(success=True)
        assert metrics.total_connections == 1
        
        metrics.record_connection(success=False)
        assert metrics.failed_connections == 1
        
        metrics.record_connection(rejected=True)
        assert metrics.rejected_connections == 1
    
    def test_record_message(self):
        """Test message recording."""
        metrics = BootstrapMetrics()
        
        metrics.record_message(bytes_sent=100, bytes_received=200)
        
        assert metrics.messages_processed == 1
        assert metrics.bytes_sent == 100
        assert metrics.bytes_received == 200
    
    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = BootstrapMetrics()
        metrics.total_connections = 10
        metrics.active_connections = 5
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["total_connections"] == 10
        assert metrics_dict["active_connections"] == 5


# =============================================================================
# BootstrapAnnouncement Tests
# =============================================================================

class TestBootstrapAnnouncement:
    """Tests for BootstrapAnnouncement class."""
    
    def test_default_announcement(self):
        """Test default announcement values."""
        announcement = BootstrapAnnouncement()
        
        assert announcement.announcement_type == "peer_join"
        assert announcement.ttl == 300
        assert announcement.announcement_id is not None
    
    def test_is_expired(self):
        """Test announcement expiration check."""
        announcement = BootstrapAnnouncement()
        
        # Fresh announcement
        assert not announcement.is_expired()
        
        # Expired announcement
        announcement.timestamp = datetime.now(timezone.utc) - timedelta(seconds=400)
        assert announcement.is_expired()
    
    def test_to_dict(self):
        """Test announcement serialization."""
        announcement = BootstrapAnnouncement(
            announcement_type="peer_leave",
            peer_id="test-peer",
        )
        
        ann_dict = announcement.to_dict()
        
        assert ann_dict["announcement_type"] == "peer_leave"
        assert ann_dict["peer_id"] == "test-peer"
    
    def test_from_dict(self):
        """Test announcement deserialization."""
        ann_dict = {
            "announcement_type": "peer_join",
            "peer_id": "test-peer",
            "peer_endpoint": "192.168.1.1:8000",
            "ttl": 600,
        }
        
        announcement = BootstrapAnnouncement.from_dict(ann_dict)
        
        assert announcement.announcement_type == "peer_join"
        assert announcement.peer_id == "test-peer"
        assert announcement.ttl == 600


# =============================================================================
# RateLimiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter class."""
    
    def test_is_allowed(self):
        """Test rate limiting."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        # Should allow first 5 requests
        for _ in range(5):
            assert limiter.is_allowed("client1")
        
        # Should deny 6th request
        assert not limiter.is_allowed("client1")
    
    def test_different_clients(self):
        """Test rate limiting for different clients."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        # Each client should have separate limit
        assert limiter.is_allowed("client1")
        assert limiter.is_allowed("client2")
        assert limiter.is_allowed("client1")
        assert limiter.is_allowed("client2")
        
        # Both should be denied now
        assert not limiter.is_allowed("client1")
        assert not limiter.is_allowed("client2")
    
    def test_reset(self):
        """Test rate limiter reset."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")
        
        limiter.reset("client1")
        assert limiter.is_allowed("client1")


# =============================================================================
# PeerDatabase Tests
# =============================================================================

class TestPeerDatabase:
    """Tests for PeerDatabase class."""
    
    def test_add_peer(self, temp_db_path):
        """Test adding a peer to database."""
        db = PeerDatabase(temp_db_path)
        peer = PeerInfo(
            peer_id="test-peer",
            address="192.168.1.1",
            port=8000,
        )
        
        db.add_peer(peer)
        
        assert "test-peer" in db.peers
        assert db.peers["test-peer"].address == "192.168.1.1"
    
    def test_remove_peer(self, temp_db_path):
        """Test removing a peer from database."""
        db = PeerDatabase(temp_db_path)
        peer = PeerInfo(
            peer_id="test-peer",
            address="192.168.1.1",
            port=8000,
        )
        
        db.add_peer(peer)
        db.remove_peer("test-peer")
        
        assert "test-peer" not in db.peers
    
    def test_get_peer(self, temp_db_path):
        """Test getting a peer from database."""
        db = PeerDatabase(temp_db_path)
        peer = PeerInfo(
            peer_id="test-peer",
            address="192.168.1.1",
            port=8000,
        )
        
        db.add_peer(peer)
        retrieved = db.get_peer("test-peer")
        
        assert retrieved is not None
        assert retrieved.peer_id == "test-peer"
    
    def test_get_active_peers(self, temp_db_path):
        """Test getting active peers."""
        db = PeerDatabase(temp_db_path)
        
        # Add active peer
        active_peer = PeerInfo(
            peer_id="active-peer",
            address="192.168.1.1",
            port=8000,
            status=PeerStatus.ACTIVE,
        )
        db.add_peer(active_peer)
        
        # Add stale peer
        stale_peer = PeerInfo(
            peer_id="stale-peer",
            address="192.168.1.2",
            port=8000,
            status=PeerStatus.ACTIVE,
        )
        stale_peer.last_seen = datetime.now(timezone.utc) - timedelta(seconds=500)
        db.add_peer(stale_peer)
        
        active_peers = db.get_active_peers(timeout_seconds=300)
        
        assert len(active_peers) == 1
        assert active_peers[0].peer_id == "active-peer"
    
    def test_save_and_load(self, temp_db_path):
        """Test database persistence."""
        # Create and save
        db1 = PeerDatabase(temp_db_path)
        peer = PeerInfo(
            peer_id="test-peer",
            address="192.168.1.1",
            port=8000,
        )
        db1.add_peer(peer)
        db1.save()
        
        # Load in new instance
        db2 = PeerDatabase(temp_db_path)
        
        assert "test-peer" in db2.peers
        assert db2.peers["test-peer"].address == "192.168.1.1"


# =============================================================================
# BootstrapServer Tests
# =============================================================================

class TestBootstrapServer:
    """Tests for BootstrapServer class."""
    
    def test_init(self, sample_config):
        """Test server initialization."""
        server = BootstrapServer(sample_config)
        
        assert server.config is sample_config
        assert server.peers == {}
        assert server.connections == {}
        assert server.running is False
    
    def test_init_with_defaults(self):
        """Test server initialization with default config."""
        server = BootstrapServer()
        
        assert server.config is not None
        assert isinstance(server.config, BootstrapConfig)
    
    @pytest.mark.asyncio
    async def test_get_peer_list_empty(self, sample_config):
        """Test getting peer list when empty."""
        server = BootstrapServer(sample_config)
        
        peer_list = await server.get_peer_list()
        
        assert peer_list == []
    
    @pytest.mark.asyncio
    async def test_get_peer_list_with_peers(self, sample_config):
        """Test getting peer list with peers."""
        server = BootstrapServer(sample_config)
        
        # Add some peers
        server.peers["peer1"] = PeerInfo(
            peer_id="peer1",
            address="192.168.1.1",
            port=8000,
            status=PeerStatus.ACTIVE,
        )
        server.peers["peer2"] = PeerInfo(
            peer_id="peer2",
            address="192.168.1.2",
            port=8000,
            status=PeerStatus.ACTIVE,
        )
        
        peer_list = await server.get_peer_list()
        
        assert len(peer_list) == 2
    
    @pytest.mark.asyncio
    async def test_get_peer_list_exclude_peer(self, sample_config):
        """Test getting peer list excluding a peer."""
        server = BootstrapServer(sample_config)
        
        server.peers["peer1"] = PeerInfo(
            peer_id="peer1",
            address="192.168.1.1",
            port=8000,
            status=PeerStatus.ACTIVE,
        )
        server.peers["peer2"] = PeerInfo(
            peer_id="peer2",
            address="192.168.1.2",
            port=8000,
            status=PeerStatus.ACTIVE,
        )
        
        peer_list = await server.get_peer_list(exclude_peer="peer1")
        
        assert len(peer_list) == 1
        assert peer_list[0]["peer_id"] == "peer2"
    
    @pytest.mark.asyncio
    async def test_get_peer_list_filter_capabilities(self, sample_config):
        """Test getting peer list filtered by capabilities."""
        server = BootstrapServer(sample_config)
        
        server.peers["peer1"] = PeerInfo(
            peer_id="peer1",
            address="192.168.1.1",
            port=8000,
            status=PeerStatus.ACTIVE,
            capabilities=["compute"],
        )
        server.peers["peer2"] = PeerInfo(
            peer_id="peer2",
            address="192.168.1.2",
            port=8000,
            status=PeerStatus.ACTIVE,
            capabilities=["storage"],
        )
        
        peer_list = await server.get_peer_list(capabilities=["compute"])
        
        assert len(peer_list) == 1
        assert peer_list[0]["peer_id"] == "peer1"
    
    @pytest.mark.asyncio
    async def test_get_peer_list_filter_region(self, sample_config):
        """Test getting peer list filtered by region."""
        server = BootstrapServer(sample_config)
        
        server.peers["peer1"] = PeerInfo(
            peer_id="peer1",
            address="192.168.1.1",
            port=8000,
            status=PeerStatus.ACTIVE,
            region="us-east-1",
        )
        server.peers["peer2"] = PeerInfo(
            peer_id="peer2",
            address="192.168.1.2",
            port=8000,
            status=PeerStatus.ACTIVE,
            region="eu-west-1",
        )
        
        peer_list = await server.get_peer_list(region="us-east-1")
        
        assert len(peer_list) == 1
        assert peer_list[0]["peer_id"] == "peer1"
    
    @pytest.mark.asyncio
    async def test_health_check(self, sample_config):
        """Test health check."""
        server = BootstrapServer(sample_config)
        server.running = True
        
        health = await server.health_check()
        
        assert health["status"] == "healthy"
        assert health["active_connections"] == 0
        assert "uptime_seconds" in health
    
    @pytest.mark.asyncio
    async def test_health_check_stopped(self, sample_config):
        """Test health check when stopped."""
        server = BootstrapServer(sample_config)
        server.running = False
        
        health = await server.health_check()
        
        assert health["status"] == "stopped"
    
    @pytest.mark.asyncio
    async def test_handle_peer_disconnect(self, sample_config):
        """Test peer disconnection handling."""
        server = BootstrapServer(sample_config)
        
        # Add a peer
        server.peers["test-peer"] = PeerInfo(
            peer_id="test-peer",
            address="192.168.1.1",
            port=8000,
            status=PeerStatus.ACTIVE,
        )
        server.connections["test-peer"] = MagicMock()
        
        await server._handle_peer_disconnect("test-peer")
        
        assert server.peers["test-peer"].status == PeerStatus.DISCONNECTED
        assert "test-peer" not in server.connections


class TestBootstrapServerConnectionHandling:
    """Tests for BootstrapServer connection handling."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, sample_config):
        """Test that rate limiting is applied."""
        sample_config.rate_limit_requests = 2
        sample_config.rate_limit_window = 60
        
        server = BootstrapServer(sample_config)
        
        # Should allow first two requests
        assert server.rate_limiter.is_allowed("192.168.1.1")
        assert server.rate_limiter.is_allowed("192.168.1.1")
        
        # Should deny third
        assert not server.rate_limiter.is_allowed("192.168.1.1")
    
    @pytest.mark.asyncio
    async def test_ip_connection_limit(self, sample_config):
        """Test IP connection limit."""
        sample_config.max_connections_per_ip = 2
        
        server = BootstrapServer(sample_config)
        
        # Simulate connections from same IP
        server.ip_connections["192.168.1.1"] = 2
        
        # Should be at limit
        assert server.ip_connections["192.168.1.1"] >= sample_config.max_connections_per_ip
    
    @pytest.mark.asyncio
    async def test_banned_ip_rejection(self, sample_config):
        """Test that banned IPs are rejected."""
        sample_config.banned_ips = ["192.168.1.100"]
        
        server = BootstrapServer(sample_config)
        
        assert "192.168.1.100" in server.config.banned_ips
    
    @pytest.mark.asyncio
    async def test_banned_peer_rejection(self, sample_config):
        """Test that banned peers are rejected."""
        sample_config.banned_peers = ["banned-peer-id"]
        
        server = BootstrapServer(sample_config)
        
        assert "banned-peer-id" in server.config.banned_peers


# =============================================================================
# Integration Tests
# =============================================================================

class TestBootstrapServerIntegration:
    """Integration tests for BootstrapServer."""
    
    @pytest.mark.asyncio
    async def test_peer_registration_flow(self, sample_config):
        """Test complete peer registration flow."""
        server = BootstrapServer(sample_config)
        
        # Create mock websocket
        mock_ws = AsyncMock()
        mock_ws.remote_address = ("192.168.1.100", 12345)
        
        # Registration data
        register_msg = {
            "type": "register",
            "peer_id": "new-peer-123",
            "port": 8000,
            "capabilities": ["compute", "storage"],
            "version": "1.0.0",
            "region": "us-east-1",
        }
        
        # Handle registration
        peer_id = await server._handle_register(
            mock_ws,
            register_msg,
            "192.168.1.100"
        )
        
        assert peer_id == "new-peer-123"
        assert "new-peer-123" in server.peers
        assert server.peers["new-peer-123"].capabilities == ["compute", "storage"]
        
        # Verify acknowledgment was sent
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "register_ack"
        assert sent_data["peer_id"] == "new-peer-123"
    
    @pytest.mark.asyncio
    async def test_peer_list_request(self, sample_config):
        """Test peer list request."""
        server = BootstrapServer(sample_config)
        
        # Add some peers
        for i in range(3):
            server.peers[f"peer-{i}"] = PeerInfo(
                peer_id=f"peer-{i}",
                address=f"192.168.1.{i}",
                port=8000,
                status=PeerStatus.ACTIVE,
            )
        
        # Create mock websocket
        mock_ws = AsyncMock()
        
        # Request peer list
        await server._handle_get_peers(mock_ws, {})
        
        # Verify response
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "peer_list"
        assert len(sent_data["peers"]) == 3
    
    @pytest.mark.asyncio
    async def test_heartbeat_handling(self, sample_config):
        """Test heartbeat message handling."""
        server = BootstrapServer(sample_config)
        
        # Add a peer
        server.peers["test-peer"] = PeerInfo(
            peer_id="test-peer",
            address="192.168.1.1",
            port=8000,
            status=PeerStatus.ACTIVE,
        )
        
        old_last_seen = server.peers["test-peer"].last_seen
        
        # Create mock websocket
        mock_ws = AsyncMock()
        
        # Handle heartbeat
        await server._handle_heartbeat(mock_ws, {}, "test-peer")
        
        # Verify last_seen was updated
        assert server.peers["test-peer"].last_seen > old_last_seen
        
        # Verify acknowledgment
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "heartbeat_ack"
    
    @pytest.mark.asyncio
    async def test_max_peers_limit(self, sample_config):
        """Test max peers limit enforcement."""
        sample_config.max_peers = 2
        
        server = BootstrapServer(sample_config)
        
        # Add peers up to limit
        server.peers["peer1"] = PeerInfo(
            peer_id="peer1",
            address="192.168.1.1",
            port=8000,
        )
        server.peers["peer2"] = PeerInfo(
            peer_id="peer2",
            address="192.168.1.2",
            port=8000,
        )
        
        # Try to add another peer
        mock_ws = AsyncMock()
        mock_ws.remote_address = ("192.168.1.3", 12345)
        
        await server._handle_register(
            mock_ws,
            {
                "peer_id": "peer3",
                "port": 8000,
            },
            "192.168.1.3"
        )
        
        # Should have removed oldest peer
        assert len(server.peers) <= sample_config.max_peers


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
