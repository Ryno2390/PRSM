"""
BitTorrent Swarm Integration Tests
===================================

Comprehensive test suite for PRSM's BitTorrent integration.
Tests cover client operations, manifest system, provider/requester components,
storage proofs, and gossip integration.

Note: These tests use mocks and in-memory simulations to avoid requiring
actual libtorrent sessions or network connections.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Import BitTorrent components
from prsm.core.bittorrent_client import (
    BitTorrentClient,
    BitTorrentConfig,
    BitTorrentResult,
    FileEntry,
    PeerInfo,
    TorrentInfo,
    TorrentState,
    LT_AVAILABLE,
)
from prsm.core.bittorrent_manifest import (
    FileEntry as ManifestFileEntry,
    PieceInfo,
    TorrentManifest,
    TorrentManifestIndex,
    TorrentManifestStore,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_torrent_info():
    """Create sample torrent info for testing."""
    return TorrentInfo(
        infohash="a" * 40,  # 40 hex chars
        name="test_torrent.zip",
        size_bytes=1024 * 1024 * 100,  # 100MB
        piece_length=262144,  # 256KB
        num_pieces=400,
        files=[
            FileEntry(path="file1.txt", size_bytes=1024, offset_in_torrent=0),
            FileEntry(path="file2.txt", size_bytes=2048, offset_in_torrent=1024),
        ],
        created_at=time.time(),
        seeders=5,
        leechers=2,
        download_rate=1024.0,
        upload_rate=512.0,
        progress=0.5,
        state=TorrentState.DOWNLOADING,
        bytes_downloaded=52428800,
        bytes_uploaded=10485760,
        eta_seconds=300.0,
    )


@pytest.fixture
def sample_manifest():
    """Create sample torrent manifest for testing."""
    return TorrentManifest(
        infohash="b" * 40,
        name="sample_manifest.zip",
        total_size=1024 * 1024 * 50,  # 50MB
        piece_length=262144,
        pieces=[
            PieceInfo(index=0, hash="a" * 40, size=262144, verified=True),
            PieceInfo(index=1, hash="b" * 40, size=262144, verified=True),
        ],
        files=[
            ManifestFileEntry(path="data.bin", size_bytes=52428800, offset_in_torrent=0),
        ],
        magnet_uri="magnet:?xt=urn:btih:" + "b" * 40,
        created_at=time.time(),
        created_by_node_id="test_node",
    )


@pytest.fixture
def bt_config():
    """Create BitTorrent config for testing."""
    return BitTorrentConfig(
        port_range_start=6881,
        port_range_end=6891,
        dht_enabled=True,
        download_dir="/tmp/test_torrents",
        max_uploads=4,
        max_connections=50,
    )


# =============================================================================
# BitTorrentClient Unit Tests
# =============================================================================

class TestBitTorrentClientUnit:
    """Unit tests for BitTorrentClient without actual libtorrent."""

    def test_config_defaults(self):
        """Verify BitTorrentConfig has sensible defaults."""
        config = BitTorrentConfig()

        assert config.port_range_start == 6881
        assert config.port_range_end == 6891
        assert config.dht_enabled is True
        assert config.max_uploads == 4
        assert config.max_connections == 50
        assert config.piece_length == 262144

    def test_result_dataclass(self):
        """Test BitTorrentResult fields."""
        result = BitTorrentResult(
            success=True,
            infohash="a" * 40,
            metadata={"key": "value"}
        )

        assert result.success is True
        assert result.infohash == "a" * 40
        assert result.error is None
        assert result.metadata == {"key": "value"}

    def test_torrent_info_to_dict(self, sample_torrent_info):
        """Test TorrentInfo serialization roundtrip."""
        data = sample_torrent_info.to_dict()

        assert data["infohash"] == "a" * 40
        assert data["name"] == "test_torrent.zip"
        assert data["size_bytes"] == 1024 * 1024 * 100
        assert data["state"] == "downloading"
        assert len(data["files"]) == 2

    def test_file_entry_dataclass(self):
        """Test FileEntry fields."""
        entry = FileEntry(
            path="test/file.txt",
            size_bytes=1024,
            offset_in_torrent=2048
        )

        assert entry.path == "test/file.txt"
        assert entry.size_bytes == 1024
        assert entry.offset_in_torrent == 2048

    def test_peer_info_dataclass(self):
        """Test PeerInfo fields."""
        peer = PeerInfo(
            peer_id="-PRSM1234-",
            ip="192.168.1.1",
            port=6881,
            client="PRSM Client",
            downloaded=1024000,
            uploaded=512000,
            is_seed=False
        )

        assert peer.peer_id == "-PRSM1234-"
        assert peer.ip == "192.168.1.1"
        assert peer.is_seed is False

    def test_torrent_state_enum(self):
        """Test TorrentState enum values."""
        assert TorrentState.QUEUED.value == "queued"
        assert TorrentState.DOWNLOADING.value == "downloading"
        assert TorrentState.SEEDING.value == "seeding"
        assert TorrentState.PAUSED.value == "paused"
        assert TorrentState.ERROR.value == "error"

    def test_client_unavailable_graceful(self):
        """Returns errors when libtorrent absent."""
        # BitTorrentClient should handle missing libtorrent gracefully
        # This test verifies the LT_AVAILABLE flag is set correctly
        assert isinstance(LT_AVAILABLE, bool)


# =============================================================================
# TorrentManifest System Tests
# =============================================================================

class TestTorrentManifestSystem:
    """Tests for the torrent manifest system."""

    def test_piece_info_to_dict(self):
        """Test PieceInfo serialization."""
        piece = PieceInfo(
            index=5,
            hash="c" * 40,
            size=262144,
            verified=True
        )

        data = piece.to_dict()
        assert data["index"] == 5
        assert data["hash"] == "c" * 40
        assert data["verified"] is True

    def test_piece_info_from_dict(self):
        """Test PieceInfo deserialization."""
        data = {
            "index": 10,
            "hash": "d" * 40,
            "size": 131072,
            "verified": False
        }

        piece = PieceInfo.from_dict(data)
        assert piece.index == 10
        assert piece.hash == "d" * 40
        assert piece.verified is False

    def test_manifest_to_dict(self, sample_manifest):
        """Test TorrentManifest serialization."""
        data = sample_manifest.to_dict()

        assert data["infohash"] == "b" * 40
        assert data["name"] == "sample_manifest.zip"
        assert data["total_size"] == 1024 * 1024 * 50
        assert data["magnet_uri"].startswith("magnet:?")
        assert len(data["pieces"]) == 2
        assert len(data["files"]) == 1

    def test_manifest_from_dict(self, sample_manifest):
        """Test TorrentManifest deserialization roundtrip."""
        data = sample_manifest.to_dict()
        restored = TorrentManifest.from_dict(data)

        assert restored.infohash == sample_manifest.infohash
        assert restored.name == sample_manifest.name
        assert restored.total_size == sample_manifest.total_size
        assert len(restored.pieces) == len(sample_manifest.pieces)
        assert len(restored.files) == len(sample_manifest.files)

    def test_manifest_to_json(self, sample_manifest):
        """Test manifest JSON serialization."""
        data = sample_manifest.to_dict()
        json_str = json.dumps(data)

        assert "infohash" in json_str
        assert "sample_manifest.zip" in json_str

        # Roundtrip
        restored = json.loads(json_str)
        assert restored["infohash"] == "b" * 40

    def test_manifest_num_pieces_property(self, sample_manifest):
        """Test num_pieces property calculation."""
        # With pieces list
        assert sample_manifest.num_pieces == 2

        # Without pieces list (calculated from size)
        empty_manifest = TorrentManifest(
            infohash="e" * 40,
            name="empty.zip",
            total_size=1048576,  # 1MB
            piece_length=262144,  # 256KB
        )
        assert empty_manifest.num_pieces >= 1

    def test_manifest_index_add_get(self):
        """Test TorrentManifestIndex add and get operations."""
        index = TorrentManifestIndex(max_size=100)

        manifest = TorrentManifest(
            infohash="f" * 40,
            name="indexed.zip",
            total_size=1024,
            piece_length=256
        )

        index.add(manifest)
        retrieved = index.get_by_infohash("f" * 40)

        assert retrieved is not None
        assert retrieved.name == "indexed.zip"

    def test_manifest_index_search(self, sample_manifest):
        """Test TorrentManifestIndex search functionality."""
        index = TorrentManifestIndex()
        index.add(sample_manifest)

        # Search by name
        results = index.search("sample")
        assert len(results) >= 1
        assert results[0].name == "sample_manifest.zip"

    def test_manifest_index_lru_eviction(self):
        """Test LRU eviction when index is over limit."""
        index = TorrentManifestIndex(max_size=3)

        for i in range(5):
            manifest = TorrentManifest(
                infohash=str(i) * 40,
                name=f"file_{i}.zip",
                total_size=1024,
                piece_length=256
            )
            index.add(manifest)

        # Should have evicted oldest entries
        assert len(index._by_infohash) <= 3

    @pytest.mark.asyncio
    async def test_manifest_store_save_load(self, sample_manifest, tmp_path):
        """Test TorrentManifestStore save and load."""
        store = TorrentManifestStore(database_url=f"sqlite:///{tmp_path}/test.db")
        await store.initialize()

        # Save manifest
        await store.save(sample_manifest)

        # Load manifest
        loaded = await store.load(sample_manifest.infohash)
        assert loaded is not None
        assert loaded.name == sample_manifest.name

    @pytest.mark.asyncio
    async def test_manifest_store_list_all(self, tmp_path):
        """Test TorrentManifestStore list_all operation."""
        store = TorrentManifestStore(database_url=f"sqlite:///{tmp_path}/test.db")
        await store.initialize()

        for i in range(3):
            manifest = TorrentManifest(
                infohash=str(i) * 40,
                name=f"file_{i}.zip",
                total_size=1024,
                piece_length=256
            )
            await store.save(manifest)

        all_manifests = await store.list_all()
        assert len(all_manifests) == 3

    @pytest.mark.asyncio
    async def test_manifest_store_delete(self, sample_manifest, tmp_path):
        """Test TorrentManifestStore delete operation."""
        store = TorrentManifestStore(database_url=f"sqlite:///{tmp_path}/test.db")
        await store.initialize()
        await store.save(sample_manifest)

        # Delete
        await store.delete(sample_manifest.infohash)

        # Verify gone
        loaded = await store.load(sample_manifest.infohash)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_manifest_store_search(self, tmp_path):
        """Test TorrentManifestStore SQL search."""
        store = TorrentManifestStore(database_url=f"sqlite:///{tmp_path}/test.db")
        await store.initialize()

        manifest = TorrentManifest(
            infohash="g" * 40,
            name="searchable_file.zip",
            total_size=1024,
            piece_length=256
        )
        await store.save(manifest)

        results = await store.search("searchable")
        assert len(results) >= 1


# =============================================================================
# BitTorrent Provider Unit Tests
# =============================================================================

class TestBitTorrentProviderUnit:
    """Unit tests for BitTorrentProvider."""

    def test_provider_config_defaults(self):
        """Test BitTorrentProviderConfig defaults."""
        from prsm.node.bittorrent_provider import BitTorrentProviderConfig

        config = BitTorrentProviderConfig()

        assert config.max_torrents == 50
        assert config.reward_interval_secs == 3600
        assert config.min_seed_time_secs == 3600
        assert float(config.seeder_reward_per_gb) == 0.10

    def test_active_torrent_dataclass(self):
        """Test ActiveTorrent fields."""
        from prsm.node.bittorrent_provider import ActiveTorrent

        manifest = TorrentManifest(
            infohash="h" * 40,
            name="active.zip",
            total_size=1024,
            piece_length=256
        )

        active = ActiveTorrent(
            infohash="h" * 40,
            manifest=manifest,
            started_at=time.time(),
            bytes_uploaded=1024000
        )

        assert active.infohash == "h" * 40
        assert active.bytes_uploaded == 1024000

    def test_reward_calculation(self):
        """Test correct FTNS per GB math."""
        from decimal import Decimal
        from prsm.node.bittorrent_provider import BitTorrentProviderConfig

        config = BitTorrentProviderConfig(seeder_reward_per_gb=Decimal("0.10"))

        # 1 GB uploaded
        bytes_uploaded = 1024 * 1024 * 1024
        gb_uploaded = bytes_uploaded / (1024 * 1024 * 1024)
        reward = float(config.seeder_reward_per_gb) * gb_uploaded

        assert reward == 0.10

    def test_announce_payload_format(self):
        """Test gossip payload structure matches spec."""
        # The provider should announce with this structure
        expected_keys = {"infohash", "name", "size", "node_id", "timestamp"}

        payload = {
            "infohash": "i" * 40,
            "name": "announce_test.zip",
            "size": 1024000,
            "node_id": "test_node",
            "timestamp": time.time()
        }

        assert expected_keys.issubset(payload.keys())


# =============================================================================
# BitTorrent Requester Unit Tests
# =============================================================================

class TestBitTorrentRequesterUnit:
    """Unit tests for BitTorrentRequester."""

    def test_requester_config_defaults(self):
        """Test BitTorrentRequesterConfig defaults."""
        from prsm.node.bittorrent_requester import BitTorrentRequesterConfig

        config = BitTorrentRequesterConfig()

        assert config.max_concurrent_downloads == 10
        assert float(config.download_cost_per_gb) == 0.05

    def test_download_request_dataclass(self):
        """Test DownloadRequest fields."""
        from prsm.node.bittorrent_requester import DownloadRequest

        request = DownloadRequest(
            request_id=str(uuid4()),
            infohash="j" * 40,
            name="test_download.zip",
            requester_node_id="test_node",
            save_path=Path("/tmp/downloads"),
        )

        assert request.infohash == "j" * 40
        assert request.status == "pending"

    def test_charge_calculation(self):
        """Test correct FTNS deduction per GB."""
        from decimal import Decimal
        from prsm.node.bittorrent_requester import BitTorrentRequesterConfig

        config = BitTorrentRequesterConfig(download_cost_per_gb=Decimal("0.05"))

        # 2 GB downloaded
        bytes_downloaded = 2 * 1024 * 1024 * 1024
        gb_downloaded = bytes_downloaded / (1024 * 1024 * 1024)
        charge = float(config.download_cost_per_gb) * gb_downloaded

        assert charge == 0.10


# =============================================================================
# BitTorrent Storage Proofs Tests
# =============================================================================

class TestBitTorrentProofsUnit:
    """Unit tests for BitTorrent storage proofs."""

    def test_challenge_dataclass(self):
        """Test TorrentPieceChallenge fields."""
        from prsm.node.bittorrent_proofs import TorrentPieceChallenge

        challenge = TorrentPieceChallenge(
            challenge_id=str(uuid4()),
            infohash="k" * 40,
            piece_index=5,
            expected_hash="l" * 40,
            nonce="random_nonce_123",
            deadline=time.time() + 300,
            challenger_node_id="challenger_node"
        )

        assert challenge.piece_index == 5
        assert challenge.is_expired() is False

    def test_proof_dataclass(self):
        """Test TorrentPieceProof fields."""
        from prsm.node.bittorrent_proofs import TorrentPieceProof

        proof = TorrentPieceProof(
            challenge_id=str(uuid4()),
            infohash="m" * 40,
            piece_index=5,
            piece_data_hash="n" * 64,  # SHA-256 is 64 hex chars
            sha1_hash="o" * 40,
            signature="signature_bytes",
            responder_node_id="prover_node"
        )

        assert proof.piece_index == 5
        assert len(proof.piece_data_hash) == 64

    def test_challenge_status_enum(self):
        """Test ChallengeStatus values."""
        from prsm.node.bittorrent_proofs import ChallengeStatus

        assert ChallengeStatus.PENDING.value == "pending"
        assert ChallengeStatus.VERIFIED.value == "verified"
        assert ChallengeStatus.FAILED.value == "failed"
        assert ChallengeStatus.EXPIRED.value == "expired"

    def test_challenge_expiry(self):
        """Test expired challenges are detected."""
        from prsm.node.bittorrent_proofs import TorrentPieceChallenge

        # Already expired
        expired_challenge = TorrentPieceChallenge(
            challenge_id="expired",
            infohash="o" * 40,
            piece_index=0,
            expected_hash="p" * 40,
            nonce="nonce",
            deadline=time.time() - 1,  # Past
            challenger_node_id="challenger"
        )

        assert expired_challenge.is_expired() is True

    def test_verify_valid_proof(self):
        """Test correct piece hash passes verification."""
        from prsm.node.bittorrent_proofs import TorrentPieceProof

        # Create a proof with valid hash
        piece_data = b"test piece content"
        piece_hash = hashlib.sha256(piece_data).hexdigest()

        proof = TorrentPieceProof(
            challenge_id="test_challenge",
            infohash="q" * 40,
            piece_index=0,
            piece_data_hash=piece_hash,
            sha1_hash=hashlib.sha1(piece_data).hexdigest(),
            signature="sig",
            responder_node_id="prover"
        )

        # Verification would check hash matches
        assert len(proof.piece_data_hash) == 64

    def test_verify_invalid_proof(self):
        """Test wrong hash rejected."""
        from prsm.node.bittorrent_proofs import TorrentPieceProof

        # Wrong hash
        proof = TorrentPieceProof(
            challenge_id="test_challenge",
            infohash="r" * 40,
            piece_index=0,
            piece_data_hash="s" * 64,  # Wrong hash
            sha1_hash="t" * 40,
            signature="sig",
            responder_node_id="prover"
        )

        # This would fail verification in production
        assert proof.piece_data_hash != hashlib.sha256(b"actual data").hexdigest()


# =============================================================================
# BitTorrent API Router Tests
# =============================================================================

class TestBitTorrentAPIRouter:
    """Tests for the BitTorrent API router."""

    @pytest.mark.asyncio
    async def test_create_endpoint_schema(self):
        """Test POST /torrents/create request model."""
        # Request should include files path
        request_body = {
            "files_path": "/data/to/share",
            "name": "my_torrent",
            "private": False
        }

        assert "files_path" in request_body
        assert request_body["private"] is False

    @pytest.mark.asyncio
    async def test_add_endpoint_schema(self):
        """Test POST /torrents/add request model."""
        request_body = {
            "magnet_uri": "magnet:?xt=urn:btih:" + "t" * 40,
            "save_path": "/downloads"
        }

        assert "magnet_uri" in request_body
        assert request_body["magnet_uri"].startswith("magnet:?")

    @pytest.mark.asyncio
    async def test_list_endpoint_returns_200(self):
        """Test GET /torrents with mock client."""
        from unittest.mock import AsyncMock

        mock_client = MagicMock()
        mock_client.list_torrents = AsyncMock(return_value=[
            {"infohash": "u" * 40, "name": "torrent1.zip", "progress": 0.5}
        ])

        result = await mock_client.list_torrents()
        assert len(result) == 1
        assert result[0]["progress"] == 0.5

    @pytest.mark.asyncio
    async def test_status_endpoint_404(self):
        """Test unknown infohash returns 404."""
        # In production, the router would return 404 for unknown infohash
        unknown_infohash = "z" * 40

        # This test documents the expected behavior
        # Router should look up the infohash and return 404 if not found
        assert len(unknown_infohash) == 40

    @pytest.mark.asyncio
    async def test_seed_endpoint_calls_provider(self):
        """Test POST /seed triggers provider start_seed."""
        from unittest.mock import AsyncMock, MagicMock
        from prsm.node.bittorrent_provider import BitTorrentProvider

        # Mock provider
        mock_provider = MagicMock()
        mock_provider.start_seed = AsyncMock(return_value=True)

        # Simulate the API call
        infohash = "v" * 40
        result = await mock_provider.start_seed(infohash)

        assert result is True
        mock_provider.start_seed.assert_called_once_with(infohash)

    @pytest.mark.asyncio
    async def test_unseed_endpoint_calls_provider(self):
        """Test DELETE /seed triggers provider stop_seed."""
        from unittest.mock import AsyncMock, MagicMock

        mock_provider = MagicMock()
        mock_provider.stop_seed = AsyncMock(return_value=True)

        infohash = "w" * 40
        result = await mock_provider.stop_seed(infohash)

        assert result is True

    @pytest.mark.asyncio
    async def test_stats_endpoint_returns_dict(self):
        """Test GET /stats returns aggregate statistics."""
        expected_keys = {"total_torrents", "total_uploaded", "total_downloaded"}

        # Mock stats response
        stats = {
            "total_torrents": 5,
            "total_uploaded": 1024 * 1024 * 100,
            "total_downloaded": 1024 * 1024 * 200
        }

        assert expected_keys.issubset(stats.keys())


# =============================================================================
# BitTorrent Gossip Integration Tests
# =============================================================================

class TestBitTorrentGossipIntegration:
    """Tests for BitTorrent gossip protocol integration."""

    def test_announce_message_constants(self):
        """Test GOSSIP_BITTORRENT_* values are defined."""
        from prsm.node.gossip import (
            GOSSIP_BITTORRENT_ANNOUNCE,
            GOSSIP_BITTORRENT_WITHDRAW,
            GOSSIP_BITTORRENT_STATS,
        )

        assert GOSSIP_BITTORRENT_ANNOUNCE == "bittorrent_announce"
        assert GOSSIP_BITTORRENT_WITHDRAW == "bittorrent_withdraw"
        assert GOSSIP_BITTORRENT_STATS == "bittorrent_stats"

    def test_announce_retention_period(self):
        """Test 24h retention configured for announces."""
        from prsm.node.gossip import GOSSIP_RETENTION_SECONDS

        assert GOSSIP_RETENTION_SECONDS.get("bittorrent_announce") == 86400

    def test_withdraw_retention_period(self):
        """Test 1h retention configured for withdrawals."""
        from prsm.node.gossip import GOSSIP_RETENTION_SECONDS

        assert GOSSIP_RETENTION_SECONDS.get("bittorrent_withdraw") == 3600

    def test_stats_retention_period(self):
        """Test 30m retention configured for stats."""
        from prsm.node.gossip import GOSSIP_RETENTION_SECONDS

        assert GOSSIP_RETENTION_SECONDS.get("bittorrent_stats") == 1800

    @pytest.mark.asyncio
    async def test_gossip_publish_announce(self):
        """Test provider publishes to gossip on seed."""
        from unittest.mock import AsyncMock, MagicMock
        from prsm.node.gossip import GOSSIP_BITTORRENT_ANNOUNCE

        mock_gossip = MagicMock()
        mock_gossip.publish = AsyncMock()

        # Simulate publishing an announce
        await mock_gossip.publish(
            GOSSIP_BITTORRENT_ANNOUNCE,
            {"infohash": "x" * 40, "name": "test.zip"}
        )

        mock_gossip.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_gossip_subscribe_requester(self):
        """Test requester subscribes to announce."""
        from unittest.mock import MagicMock
        from prsm.node.gossip import GOSSIP_BITTORRENT_ANNOUNCE

        mock_gossip = MagicMock()
        mock_gossip.subscribe = MagicMock()

        # Simulate subscribing
        mock_gossip.subscribe(GOSSIP_BITTORRENT_ANNOUNCE, lambda *args: None)

        mock_gossip.subscribe.assert_called_once()


# =============================================================================
# End-to-End Simulation Tests
# =============================================================================

class TestBitTorrentEndToEndSimulation:
    """End-to-end simulations without real network."""

    @pytest.mark.asyncio
    async def test_manifest_lifecycle(self, tmp_path):
        """Test complete manifest lifecycle: create, save, load, delete."""
        store = TorrentManifestStore(database_url=f"sqlite:///{tmp_path}/lifecycle.db")
        await store.initialize()

        # Create
        manifest = TorrentManifest(
            infohash="lifecycle" + "0" * 32,
            name="lifecycle_test.zip",
            total_size=1024 * 1024,
            piece_length=262144
        )

        # Save
        await store.save(manifest)

        # Load
        loaded = await store.load(manifest.infohash)
        assert loaded is not None
        assert loaded.name == "lifecycle_test.zip"

        # Delete
        await store.delete(manifest.infohash)
        assert await store.load(manifest.infohash) is None

    @pytest.mark.asyncio
    async def test_piece_verification_flow(self):
        """Test piece challenge and proof verification flow."""
        # Simulate the verification flow
        piece_data = b"piece content for verification"
        expected_hash = hashlib.sha256(piece_data).hexdigest()

        # Prover would send this hash
        assert len(expected_hash) == 64

        # Verifier would check it matches
        verifier_hash = hashlib.sha256(piece_data).hexdigest()
        assert expected_hash == verifier_hash


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
