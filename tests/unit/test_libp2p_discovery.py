"""
Unit tests for prsm.node.libp2p_discovery.Libp2pDiscovery.

All tests use mocked transport / gossip objects — no Go library required.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.discovery import PeerInfo
from prsm.node.libp2p_discovery import Libp2pDiscovery


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_transport(node_id: str = "local-node-001") -> MagicMock:
    transport = MagicMock()
    transport.identity = MagicMock()
    transport.identity.node_id = node_id
    transport._handle = 0
    transport._lib = MagicMock()
    transport.connect_to_peer = AsyncMock(return_value=None)
    transport.dht_provide = AsyncMock(return_value=True)
    transport.dht_find_providers = AsyncMock(return_value=[])
    return transport


def _make_gossip() -> MagicMock:
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    return gossip


def _peer_info(
    node_id: str = "peer-001",
    gpu: bool = False,
    caps: list = None,
    backends: list = None,
) -> PeerInfo:
    return PeerInfo(
        node_id=node_id,
        address="127.0.0.1:9001",
        display_name="",
        roles=[],
        capabilities=caps or [],
        supported_backends=backends or [],
        gpu_available=gpu,
        last_seen=time.time(),
        last_capability_update=time.time(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCapabilityIndex:
    """_on_capability updates the index and find_peers_with_gpu works."""

    @pytest.mark.asyncio
    async def test_capability_index(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        data = {
            "node_id": "peer-gpu-001",
            "capabilities": ["inference"],
            "supported_backends": ["local"],
            "gpu_available": True,
        }
        await discovery._on_capability("capability_announce", data, "peer-gpu-001")

        gpu_peers = discovery.find_peers_with_gpu()
        assert len(gpu_peers) == 1
        assert gpu_peers[0].node_id == "peer-gpu-001"
        assert gpu_peers[0].gpu_available is True

    @pytest.mark.asyncio
    async def test_capability_index_update_existing(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        # First announcement
        await discovery._on_capability(
            "capability_announce",
            {"node_id": "peer-001", "capabilities": ["inference"],
             "supported_backends": [], "gpu_available": False},
            "peer-001",
        )
        assert not discovery._capability_index["peer-001"].gpu_available

        # Second announcement with GPU now available
        await discovery._on_capability(
            "capability_announce",
            {"node_id": "peer-001", "capabilities": ["inference"],
             "supported_backends": [], "gpu_available": True},
            "peer-001",
        )
        assert discovery._capability_index["peer-001"].gpu_available is True

    @pytest.mark.asyncio
    async def test_non_gpu_peer_not_in_gpu_list(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        await discovery._on_capability(
            "capability_announce",
            {"node_id": "cpu-peer", "capabilities": [], "supported_backends": [],
             "gpu_available": False},
            "cpu-peer",
        )

        assert discovery.find_peers_with_gpu() == []


class TestShardCache:
    """_on_shard_available populates the shard cache."""

    @pytest.mark.asyncio
    async def test_shard_cache(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        await discovery._on_shard_available(
            "shard_available",
            {"cid": "QmTest123", "node_id": "shard-peer-001"},
            "shard-peer-001",
        )

        assert "QmTest123" in discovery._shard_cache
        assert "shard-peer-001" in discovery._shard_cache["QmTest123"]

    @pytest.mark.asyncio
    async def test_shard_cache_multiple_providers(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        cid = "QmMultiProvider"
        for i in range(3):
            await discovery._on_shard_available(
                "shard_available",
                {"cid": cid, "node_id": f"peer-{i}"},
                f"peer-{i}",
            )

        assert len(discovery._shard_cache[cid]) == 3

    @pytest.mark.asyncio
    async def test_shard_cache_no_duplicates(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        cid = "QmDup"
        for _ in range(5):
            await discovery._on_shard_available(
                "shard_available",
                {"cid": cid, "node_id": "shard-peer"},
                "shard-peer",
            )

        assert len(discovery._shard_cache[cid]) == 1


class TestBootstrapDegraded:
    """When all bootstrap connections return None, status shows degraded."""

    @pytest.mark.asyncio
    async def test_bootstrap_degraded(self) -> None:
        transport = _make_transport()
        transport.connect_to_peer = AsyncMock(return_value=None)

        discovery = Libp2pDiscovery(
            transport,
            bootstrap_nodes=["ws://bootstrap1.example.com:8765",
                             "ws://bootstrap2.example.com:8765"],
        )

        connected = await discovery.bootstrap()

        assert connected == 0
        status = discovery.get_bootstrap_status()
        assert status["degraded"] is True
        assert status["connected"] == 0
        assert status["attempted"] == 2

    @pytest.mark.asyncio
    async def test_bootstrap_success(self) -> None:
        transport = _make_transport()
        mock_peer = MagicMock()
        mock_peer.peer_id = "remote-bootstrap-001"
        transport.connect_to_peer = AsyncMock(return_value=mock_peer)

        discovery = Libp2pDiscovery(
            transport,
            bootstrap_nodes=["ws://bootstrap1.example.com:8765"],
        )

        connected = await discovery.bootstrap()

        assert connected == 1
        status = discovery.get_bootstrap_status()
        assert status["degraded"] is False
        assert status["connected"] == 1

    @pytest.mark.asyncio
    async def test_no_bootstrap_nodes_not_degraded(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport, bootstrap_nodes=[])

        connected = await discovery.bootstrap()

        assert connected == 0
        assert discovery.get_bootstrap_status()["degraded"] is False


class TestFindPeersByCapability:
    """find_peers_by_capability with match_all=True requires all capabilities."""

    def test_find_peers_by_capability_match_all(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        # Peer with both capabilities
        discovery._capability_index["peer-both"] = _peer_info(
            "peer-both", caps=["inference", "embedding"]
        )
        # Peer with only one capability
        discovery._capability_index["peer-one"] = _peer_info(
            "peer-one", caps=["inference"]
        )

        results = discovery.find_peers_by_capability(
            ["inference", "embedding"], match_all=True
        )

        ids = [p.node_id for p in results]
        assert "peer-both" in ids
        assert "peer-one" not in ids

    def test_find_peers_by_capability_match_any(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        discovery._capability_index["peer-a"] = _peer_info("peer-a", caps=["inference"])
        discovery._capability_index["peer-b"] = _peer_info("peer-b", caps=["embedding"])
        discovery._capability_index["peer-c"] = _peer_info("peer-c", caps=["benchmark"])

        results = discovery.find_peers_by_capability(
            ["inference", "embedding"], match_all=False
        )

        ids = {p.node_id for p in results}
        assert "peer-a" in ids
        assert "peer-b" in ids
        assert "peer-c" not in ids

    def test_find_peers_case_insensitive(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        discovery._capability_index["peer-upper"] = _peer_info(
            "peer-upper", caps=["Inference", "EMBEDDING"]
        )

        results = discovery.find_peers_by_capability(["inference", "embedding"])
        assert len(results) == 1

    def test_find_peers_with_backend(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)

        discovery._capability_index["anthropic-peer"] = _peer_info(
            "anthropic-peer", backends=["anthropic", "openai"]
        )
        discovery._capability_index["local-peer"] = _peer_info(
            "local-peer", backends=["local"]
        )

        results = discovery.find_peers_with_backend("anthropic")
        assert len(results) == 1
        assert results[0].node_id == "anthropic-peer"


class TestStartSubscribes:
    """On start(), gossip.subscribe is called for capability_announce and shard_available."""

    @pytest.mark.asyncio
    async def test_start_subscribes_to_gossip(self) -> None:
        transport = _make_transport()
        # Make connect_to_peer succeed so bootstrap doesn't degrade
        transport.connect_to_peer = AsyncMock(return_value=MagicMock())
        gossip = _make_gossip()

        discovery = Libp2pDiscovery(
            transport,
            bootstrap_nodes=["ws://b1.example.com:8765"],
            gossip=gossip,
        )
        await discovery.start()

        subscribed_subtypes = {
            call.args[0] for call in gossip.subscribe.call_args_list
        }
        assert "capability_announce" in subscribed_subtypes
        assert "shard_available" in subscribed_subtypes

    @pytest.mark.asyncio
    async def test_start_without_gossip_no_error(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport, bootstrap_nodes=[])
        # Should not raise even with no gossip
        await discovery.start()


class TestAnnounceCapabilities:
    """announce_capabilities publishes the local capability payload via gossip."""

    @pytest.mark.asyncio
    async def test_announce_capabilities(self) -> None:
        transport = _make_transport()
        gossip = _make_gossip()

        discovery = Libp2pDiscovery(transport, gossip=gossip)
        discovery.set_local_capabilities(["inference"], ["anthropic"], gpu_available=True)

        result = await discovery.announce_capabilities()

        assert result == 1
        gossip.publish.assert_called_once()
        call_args = gossip.publish.call_args
        assert call_args.args[0] == "capability_announce"
        payload = call_args.args[1]
        assert payload["capabilities"] == ["inference"]
        assert payload["gpu_available"] is True

    @pytest.mark.asyncio
    async def test_announce_capabilities_no_gossip(self) -> None:
        transport = _make_transport()
        discovery = Libp2pDiscovery(transport)  # no gossip

        result = await discovery.announce_capabilities()
        assert result == 0


class TestPeerInfoReliability:
    """Tests for PeerInfo reliability tracking fields."""

    def test_reliability_score_new_peer(self):
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        assert peer.reliability_score == 1.0

    def test_reliability_score_all_successes(self):
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        peer.job_success_count = 10
        assert peer.reliability_score == 1.0

    def test_reliability_score_mixed(self):
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        peer.job_success_count = 2
        peer.job_failure_count = 1
        assert abs(peer.reliability_score - 0.6667) < 0.01

    def test_reliability_score_all_failures(self):
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        peer.job_failure_count = 5
        assert peer.reliability_score == 0.0

    def test_startup_timestamp_default(self):
        peer = PeerInfo(node_id="peer1", address="1.2.3.4:9000")
        assert peer.startup_timestamp == 0.0
