"""
Unit Tests for Capability-Based Peer Discovery
================================================

Tests for:
- PeerInfo capability fields
- PeerDiscovery capability query methods
- Capability gossip handling
- Smart job routing in ComputeRequester
"""

import pytest
import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.node.discovery import (
    PeerInfo,
    PeerDiscovery,
    DISCOVERY_CAPABILITY_ANNOUNCE,
)
from prsm.node.gossip import GOSSIP_CAPABILITY_ANNOUNCE
from prsm.node.compute_requester import ComputeRequester, JOB_TYPE_CAPABILITIES, JOB_TYPE_PREFERRED_BACKENDS
from prsm.node.compute_provider import JobType, JobStatus
from prsm.node.capability_detection import (
    detect_gpu_availability,
    detect_available_backends,
    detect_node_capabilities,
    get_capabilities_for_discovery,
)


class TestPeerInfo:
    """Tests for PeerInfo dataclass with capability fields."""

    def test_peer_info_defaults(self):
        """Test PeerInfo default values."""
        peer = PeerInfo(
            node_id="test-node",
            address="localhost:8080",
        )
        assert peer.capabilities == []
        assert peer.supported_backends == []
        assert peer.gpu_available is False
        assert peer.last_seen > 0
        assert peer.last_capability_update > 0

    def test_peer_info_with_capabilities(self):
        """Test PeerInfo with explicit capabilities."""
        peer = PeerInfo(
            node_id="test-node",
            address="localhost:8080",
            capabilities=["inference", "embedding"],
            supported_backends=["anthropic", "openai"],
            gpu_available=True,
        )
        assert "inference" in peer.capabilities
        assert "embedding" in peer.capabilities
        assert "anthropic" in peer.supported_backends
        assert "openai" in peer.supported_backends
        assert peer.gpu_available is True

    def test_peer_info_last_capability_update(self):
        """Test that last_capability_update is set correctly."""
        before = time.time()
        peer = PeerInfo(
            node_id="test-node",
            address="localhost:8080",
        )
        after = time.time()
        assert before <= peer.last_capability_update <= after


class TestPeerDiscoveryCapabilities:
    """Tests for PeerDiscovery capability query methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_transport = MagicMock()
        self.mock_transport.identity = MagicMock()
        self.mock_transport.identity.node_id = "local-node"
        self.mock_transport.host = "localhost"
        self.mock_transport.port = 8080
        self.mock_transport.peers = {}
        self.mock_transport.on_message = MagicMock()

        self.discovery = PeerDiscovery(
            transport=self.mock_transport,
            local_capabilities=["inference", "embedding"],
            local_backends=["anthropic", "openai"],
            local_gpu_available=True,
        )

    def test_find_peers_with_capability(self):
        """Test finding peers by capability."""
        # Add some known peers
        self.discovery.known_peers = {
            "peer1": PeerInfo(
                node_id="peer1",
                address="localhost:8001",
                capabilities=["inference", "embedding"],
            ),
            "peer2": PeerInfo(
                node_id="peer2",
                address="localhost:8002",
                capabilities=["benchmark"],
            ),
            "peer3": PeerInfo(
                node_id="peer3",
                address="localhost:8003",
                capabilities=["inference", "training"],
            ),
        }

        # Find peers with inference capability
        inference_peers = self.discovery.find_peers_with_capability("inference")
        assert len(inference_peers) == 2
        peer_ids = {p.node_id for p in inference_peers}
        assert "peer1" in peer_ids
        assert "peer3" in peer_ids
        assert "peer2" not in peer_ids

        # Find peers with embedding capability
        embedding_peers = self.discovery.find_peers_with_capability("embedding")
        assert len(embedding_peers) == 1
        assert embedding_peers[0].node_id == "peer1"

    def test_find_peers_with_capability_case_insensitive(self):
        """Test that capability search is case insensitive."""
        self.discovery.known_peers = {
            "peer1": PeerInfo(
                node_id="peer1",
                address="localhost:8001",
                capabilities=["Inference", "EMBEDDING"],
            ),
        }

        peers = self.discovery.find_peers_with_capability("INFERENCE")
        assert len(peers) == 1
        assert peers[0].node_id == "peer1"

    def test_find_peers_with_backend(self):
        """Test finding peers by backend."""
        self.discovery.known_peers = {
            "peer1": PeerInfo(
                node_id="peer1",
                address="localhost:8001",
                supported_backends=["anthropic", "openai"],
            ),
            "peer2": PeerInfo(
                node_id="peer2",
                address="localhost:8002",
                supported_backends=["local"],
            ),
            "peer3": PeerInfo(
                node_id="peer3",
                address="localhost:8003",
                supported_backends=["anthropic"],
            ),
        }

        anthropic_peers = self.discovery.find_peers_with_backend("anthropic")
        assert len(anthropic_peers) == 2
        peer_ids = {p.node_id for p in anthropic_peers}
        assert "peer1" in peer_ids
        assert "peer3" in peer_ids

        local_peers = self.discovery.find_peers_with_backend("local")
        assert len(local_peers) == 1
        assert local_peers[0].node_id == "peer2"

    def test_find_peers_with_backend_case_insensitive(self):
        """Test that backend search is case insensitive."""
        self.discovery.known_peers = {
            "peer1": PeerInfo(
                node_id="peer1",
                address="localhost:8001",
                supported_backends=["Anthropic", "OpenAI"],
            ),
        }

        peers = self.discovery.find_peers_with_backend("ANTHROPIC")
        assert len(peers) == 1

    def test_find_peers_with_gpu(self):
        """Test finding peers with GPU."""
        self.discovery.known_peers = {
            "peer1": PeerInfo(
                node_id="peer1",
                address="localhost:8001",
                gpu_available=True,
            ),
            "peer2": PeerInfo(
                node_id="peer2",
                address="localhost:8002",
                gpu_available=False,
            ),
            "peer3": PeerInfo(
                node_id="peer3",
                address="localhost:8003",
                gpu_available=True,
            ),
        }

        gpu_peers = self.discovery.find_peers_with_gpu()
        assert len(gpu_peers) == 2
        peer_ids = {p.node_id for p in gpu_peers}
        assert "peer1" in peer_ids
        assert "peer3" in peer_ids

    def test_set_local_capabilities(self):
        """Test setting local capabilities."""
        self.discovery.set_local_capabilities(
            capabilities=["training", "fine_tuning"],
            backends=["local"],
            gpu_available=True,
        )
        assert self.discovery._local_capabilities == ["training", "fine_tuning"]
        assert self.discovery._local_backends == ["local"]
        assert self.discovery._local_gpu_available is True

    @pytest.mark.asyncio
    async def test_handle_capability_announce_new_peer(self):
        """Test handling capability announcement for new peer."""
        mock_peer = MagicMock()
        mock_peer.peer_id = "remote-peer"
        mock_peer.address = "localhost:9000"

        msg = MagicMock()
        msg.sender_id = "remote-peer"
        msg.ttl = 3
        msg.nonce = "test-nonce"
        msg.payload = {
            "subtype": DISCOVERY_CAPABILITY_ANNOUNCE,
            "node_id": "remote-peer",
            "capabilities": ["inference", "embedding"],
            "supported_backends": ["anthropic", "openai"],
            "gpu_available": True,
        }

        self.mock_transport.gossip = AsyncMock(return_value=1)

        await self.discovery._handle_capability_announce(msg, mock_peer)

        assert "remote-peer" in self.discovery.known_peers
        peer = self.discovery.known_peers["remote-peer"]
        assert "inference" in peer.capabilities
        assert "anthropic" in peer.supported_backends
        assert peer.gpu_available is True

    @pytest.mark.asyncio
    async def test_handle_capability_announce_existing_peer(self):
        """Test handling capability announcement for existing peer."""
        # Add existing peer
        self.discovery.known_peers["remote-peer"] = PeerInfo(
            node_id="remote-peer",
            address="localhost:9000",
            capabilities=["old_capability"],
            supported_backends=["old_backend"],
            gpu_available=False,
        )

        mock_peer = MagicMock()
        mock_peer.peer_id = "remote-peer"
        mock_peer.address = "localhost:9000"

        msg = MagicMock()
        msg.sender_id = "remote-peer"
        msg.ttl = 3
        msg.nonce = "test-nonce"
        msg.payload = {
            "subtype": DISCOVERY_CAPABILITY_ANNOUNCE,
            "node_id": "remote-peer",
            "capabilities": ["inference", "embedding"],
            "supported_backends": ["anthropic"],
            "gpu_available": True,
        }

        self.mock_transport.gossip = AsyncMock(return_value=1)

        await self.discovery._handle_capability_announce(msg, mock_peer)

        peer = self.discovery.known_peers["remote-peer"]
        assert "inference" in peer.capabilities
        assert "old_capability" not in peer.capabilities
        assert "anthropic" in peer.supported_backends
        assert peer.gpu_available is True


class TestComputeRequesterSmartRouting:
    """Tests for ComputeRequester smart routing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_identity = MagicMock()
        self.mock_identity.node_id = "requester-node"

        self.mock_transport = MagicMock()
        self.mock_transport.identity = self.mock_identity

        self.mock_gossip = MagicMock()
        self.mock_gossip.subscribe = MagicMock()
        self.mock_gossip.publish = AsyncMock(return_value=1)

        self.mock_ledger = MagicMock()
        self.mock_ledger.get_balance = AsyncMock(return_value=100.0)

        self.discovery = MagicMock()
        self.discovery.find_peers_with_capability = MagicMock()
        self.discovery.find_peers_with_backend = MagicMock()

    @pytest.mark.asyncio
    async def test_smart_routing_targets_capable_peers(self):
        """Test that smart routing targets capable peers."""
        # Set up discovery to return capable peers
        mock_peer = MagicMock()
        mock_peer.node_id = "capable-peer"
        self.discovery.find_peers_with_capability.return_value = [mock_peer]
        self.discovery.find_peers_with_backend.return_value = [mock_peer]

        requester = ComputeRequester(
            identity=self.mock_identity,
            transport=self.mock_transport,
            gossip=self.mock_gossip,
            ledger=self.mock_ledger,
            discovery=self.discovery,
            smart_routing=True,
        )

        await requester.start()

        # Submit a job
        job = await requester.submit_job(
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=10.0,
        )

        # Verify the job offer was published
        self.mock_gossip.publish.assert_called_once()
        call_args = self.mock_gossip.publish.call_args

        # Check that target_peers was included in the job offer
        job_offer = call_args[0][1]
        assert "target_peers" in job_offer
        assert "capable-peer" in job_offer["target_peers"]

    @pytest.mark.asyncio
    async def test_smart_routing_fallback_to_broadcast(self):
        """Test that smart routing falls back to broadcast when no capable peers."""
        # Set up discovery to return no capable peers
        self.discovery.find_peers_with_capability.return_value = []
        self.discovery.find_peers_with_backend.return_value = []

        requester = ComputeRequester(
            identity=self.mock_identity,
            transport=self.mock_transport,
            gossip=self.mock_gossip,
            ledger=self.mock_ledger,
            discovery=self.discovery,
            smart_routing=True,
        )

        await requester.start()

        # Submit a job
        job = await requester.submit_job(
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=10.0,
        )

        # Verify the job offer was published
        self.mock_gossip.publish.assert_called_once()

        # Check that target_peers was NOT included (broadcast mode)
        call_args = self.mock_gossip.publish.call_args
        job_offer = call_args[0][1]
        # When no capable peers found, target_peers should not be in the offer
        # or should be empty
        assert "target_peers" not in job_offer or len(job_offer.get("target_peers", [])) == 0

    @pytest.mark.asyncio
    async def test_smart_routing_disabled(self):
        """Test that smart routing can be disabled."""
        requester = ComputeRequester(
            identity=self.mock_identity,
            transport=self.mock_transport,
            gossip=self.mock_gossip,
            ledger=self.mock_ledger,
            discovery=self.discovery,
            smart_routing=False,
        )

        await requester.start()

        # Submit a job
        job = await requester.submit_job(
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=10.0,
        )

        # Discovery should not have been called
        self.discovery.find_peers_with_capability.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_target_peers(self):
        """Test that explicit target peers override smart routing."""
        # Set up discovery to return different peers
        mock_peer = MagicMock()
        mock_peer.node_id = "capable-peer"
        self.discovery.find_peers_with_capability.return_value = [mock_peer]

        requester = ComputeRequester(
            identity=self.mock_identity,
            transport=self.mock_transport,
            gossip=self.mock_gossip,
            ledger=self.mock_ledger,
            discovery=self.discovery,
            smart_routing=True,
        )

        await requester.start()

        # Submit a job with explicit target peers
        job = await requester.submit_job(
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=10.0,
            target_peers=["explicit-peer-1", "explicit-peer-2"],
        )

        # Verify the job offer was published with explicit targets
        call_args = self.mock_gossip.publish.call_args
        job_offer = call_args[0][1]
        assert "target_peers" in job_offer
        assert "explicit-peer-1" in job_offer["target_peers"]
        assert "explicit-peer-2" in job_offer["target_peers"]
        # Smart routing should not have been called
        self.discovery.find_peers_with_capability.assert_not_called()

    def test_job_type_capability_mapping(self):
        """Test that job types are mapped to correct capabilities."""
        assert JOB_TYPE_CAPABILITIES[JobType.INFERENCE] == "inference"
        assert JOB_TYPE_CAPABILITIES[JobType.EMBEDDING] == "embedding"
        assert JOB_TYPE_CAPABILITIES[JobType.BENCHMARK] == "benchmark"

    def test_job_type_backend_mapping(self):
        """Test that LLM-backed job types have preferred backends defined.

        WASM_EXECUTE is intentionally excluded from JOB_TYPE_PREFERRED_BACKENDS
        because WASM jobs run in a local sandbox and do not depend on an
        LLM backend (anthropic / openai / local).
        """
        assert JobType.INFERENCE in JOB_TYPE_PREFERRED_BACKENDS
        assert JobType.EMBEDDING in JOB_TYPE_PREFERRED_BACKENDS
        assert JobType.BENCHMARK in JOB_TYPE_PREFERRED_BACKENDS
        # WASM_EXECUTE does not require an LLM backend — it is intentionally
        # absent from JOB_TYPE_PREFERRED_BACKENDS.
        assert JobType.WASM_EXECUTE not in JOB_TYPE_PREFERRED_BACKENDS
        # JobType.TRAINING was removed in v1.6.x along with the
        # distillation/teacher subsystem (PRSM is no longer an AGI framework).
        assert not hasattr(JobType, "TRAINING")


class TestCapabilityDetection:
    """Tests for capability detection functions."""

    def test_detect_available_backends_no_keys(self):
        """Test backend detection with no API keys."""
        with patch.dict("os.environ", {}, clear=True):
            backends, any_real = detect_available_backends()
            # Should return empty list when no keys are set
            assert isinstance(backends, list)
            assert isinstance(any_real, bool)

    def test_detect_available_backends_with_anthropic(self):
        """Test backend detection with Anthropic API key."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            backends, any_real = detect_available_backends()
            assert "anthropic" in backends
            assert any_real is True

    def test_detect_available_backends_with_openai(self):
        """Test backend detection with OpenAI API key."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            backends, any_real = detect_available_backends()
            assert "openai" in backends
            assert any_real is True

    def test_detect_node_capabilities_structure(self):
        """Test that detect_node_capabilities returns correct structure."""
        with patch("prsm.node.capability_detection.detect_available_backends") as mock_backends:
            with patch("prsm.node.capability_detection.detect_gpu_availability") as mock_gpu:
                mock_backends.return_value = (["anthropic", "openai"], True)
                mock_gpu.return_value = True

                caps = detect_node_capabilities()

                assert "capabilities" in caps
                assert "supported_backends" in caps
                assert "gpu_available" in caps
                assert "any_real_backend" in caps

                assert isinstance(caps["capabilities"], list)
                assert isinstance(caps["supported_backends"], list)
                assert isinstance(caps["gpu_available"], bool)
                assert isinstance(caps["any_real_backend"], bool)

    def test_get_capabilities_for_discovery(self):
        """Test get_capabilities_for_discovery returns correct format."""
        with patch("prsm.node.capability_detection.detect_node_capabilities") as mock_detect:
            mock_detect.return_value = {
                "capabilities": ["inference", "embedding"],
                "supported_backends": ["anthropic", "openai"],
                "gpu_available": True,
                "any_real_backend": True,
            }

            caps = get_capabilities_for_discovery()

            assert "capabilities" in caps
            assert "supported_backends" in caps
            assert "gpu_available" in caps
            assert "any_real_backend" not in caps  # Not included in discovery format

            assert caps["capabilities"] == ["inference", "embedding"]
            assert caps["supported_backends"] == ["anthropic", "openai"]
            assert caps["gpu_available"] is True


class TestGossipCapabilityAnnounce:
    """Tests for GOSSIP_CAPABILITY_ANNOUNCE constant."""

    def test_gossip_capability_announce_defined(self):
        """Test that GOSSIP_CAPABILITY_ANNOUNCE is defined in gossip module."""
        from prsm.node.gossip import GOSSIP_CAPABILITY_ANNOUNCE
        assert GOSSIP_CAPABILITY_ANNOUNCE == "capability_announce"

    def test_discovery_capability_announce_defined(self):
        """Test that DISCOVERY_CAPABILITY_ANNOUNCE is defined in discovery module."""
        from prsm.node.discovery import DISCOVERY_CAPABILITY_ANNOUNCE
        assert DISCOVERY_CAPABILITY_ANNOUNCE == "capability_announce"