"""
Tests for DistributionManager — shard placement, descriptor signing, and conflict resolution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from prsm.storage.distribution import DistributionManager, ShardPlacement
from prsm.storage.exceptions import PlacementError
from prsm.storage.models import (
    ContentDescriptor,
    ContentHash,
    ReplicationPolicy,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

@dataclass
class FakePeerInfo:
    node_id: str
    address: str = ""
    capabilities: List[str] = field(default_factory=lambda: ["storage"])
    reliability_score: float = 1.0
    asn: str = "AS0"


class FakeDiscovery:
    def __init__(self, peers: List[FakePeerInfo]):
        self._peers = {p.node_id: p for p in peers}

    def find_peers_by_capability(self, required, match_all=True):
        return [p for p in self._peers.values() if "storage" in p.capabilities]


class FakeTransport:
    def __init__(self):
        self.sent_messages: list = []
        self._handlers: dict = {}

    async def send_to_peer(self, peer_id, msg):
        self.sent_messages.append((peer_id, msg))
        return True

    async def dht_provide(self, key):
        return True

    def on_message(self, msg_type, handler):
        self._handlers[msg_type] = handler


def _make_manager(peers: List[FakePeerInfo], owner_id: str = "owner-node"):
    """Create a DistributionManager wired to fake collaborators."""
    return DistributionManager(
        node_id=owner_id,
        discovery=FakeDiscovery(peers),
        transport=FakeTransport(),
        blob_store=None,
        key_manager=None,
    )


def _shard_hashes(n: int = 4) -> List[str]:
    """Return *n* deterministic hex shard hashes."""
    return [ContentHash.from_data(f"shard-{i}".encode()).hex() for i in range(n)]


# ---------------------------------------------------------------------------
# TestShardPlacement
# ---------------------------------------------------------------------------

class TestShardPlacement:
    """Core placement constraint tests."""

    def _peers_multi_asn(self, count: int = 6) -> List[FakePeerInfo]:
        """Peers spread across distinct ASNs."""
        return [
            FakePeerInfo(node_id=f"peer-{i}", asn=f"AS{i}")
            for i in range(count)
        ]

    # 1. owner_excluded
    def test_owner_excluded(self):
        owner = "owner-node"
        peers = self._peers_multi_asn(6)
        mgr = _make_manager(peers, owner_id=owner)
        hashes = _shard_hashes(3)
        peer_asn = {p.node_id: p.asn for p in peers}

        placement = mgr._compute_shard_placement(
            shard_hashes=hashes,
            replication_factor=2,
            owner_node_id=owner,
            peer_asn_map=peer_asn,
        )

        all_assigned = set()
        for nodes in placement.shard_assignments.values():
            all_assigned.update(nodes)
        assert owner not in all_assigned

    # 2. asn_diversity
    def test_asn_diversity(self):
        peers = self._peers_multi_asn(6)
        mgr = _make_manager(peers)
        hashes = _shard_hashes(2)
        peer_asn = {p.node_id: p.asn for p in peers}

        placement = mgr._compute_shard_placement(
            shard_hashes=hashes,
            replication_factor=3,
            owner_node_id="owner-node",
            peer_asn_map=peer_asn,
        )

        for _shard_hash, nodes in placement.shard_assignments.items():
            asns = {peer_asn[n] for n in nodes}
            # With 6 distinct ASNs and rf=3, every shard should land on 3 ASNs
            assert len(asns) == 3

    # 3. replication_factor_honored
    def test_replication_factor_honored(self):
        peers = self._peers_multi_asn(6)
        mgr = _make_manager(peers)
        hashes = _shard_hashes(4)
        peer_asn = {p.node_id: p.asn for p in peers}

        placement = mgr._compute_shard_placement(
            shard_hashes=hashes,
            replication_factor=3,
            owner_node_id="owner-node",
            peer_asn_map=peer_asn,
        )

        for nodes in placement.shard_assignments.values():
            assert len(nodes) == 3

    # 4. key_shard_separation
    def test_key_shard_separation(self):
        # 8 peers, rf=2, 2 shards => 4 shard nodes, 4 left for keys
        peers = self._peers_multi_asn(8)
        mgr = _make_manager(peers)
        hashes = _shard_hashes(2)
        peer_asn = {p.node_id: p.asn for p in peers}

        placement = mgr._compute_shard_placement(
            shard_hashes=hashes,
            replication_factor=2,
            owner_node_id="owner-node",
            peer_asn_map=peer_asn,
        )

        shard_nodes = set()
        for nodes in placement.shard_assignments.values():
            shard_nodes.update(nodes)

        key_nodes = set(placement.key_share_holders)
        # Key holders should be disjoint from shard holders
        assert shard_nodes.isdisjoint(key_nodes)


# ---------------------------------------------------------------------------
# TestDegradedMode
# ---------------------------------------------------------------------------

class TestDegradedMode:
    """Tests for degraded/relaxed placement constraints."""

    # 5. asn_relaxation
    def test_asn_relaxation(self):
        # All peers share the same ASN
        peers = [
            FakePeerInfo(node_id=f"peer-{i}", asn="AS999")
            for i in range(5)
        ]
        mgr = _make_manager(peers)
        hashes = _shard_hashes(2)
        peer_asn = {p.node_id: p.asn for p in peers}

        placement = mgr._compute_shard_placement(
            shard_hashes=hashes,
            replication_factor=2,
            owner_node_id="owner-node",
            peer_asn_map=peer_asn,
        )

        assert "asn_relaxed" in placement.degraded_constraints
        # Placement still succeeds
        for nodes in placement.shard_assignments.values():
            assert len(nodes) == 2

    # 6. reject_when_too_few_nodes
    def test_reject_when_too_few_nodes(self):
        # Only 1 non-owner peer  (< MIN_NETWORK_NODES = 3)
        peers = [FakePeerInfo(node_id="peer-0", asn="AS0")]
        mgr = _make_manager(peers)
        hashes = _shard_hashes(1)
        peer_asn = {p.node_id: p.asn for p in peers}

        with pytest.raises(PlacementError):
            mgr._compute_shard_placement(
                shard_hashes=hashes,
                replication_factor=1,
                owner_node_id="owner-node",
                peer_asn_map=peer_asn,
            )


# ---------------------------------------------------------------------------
# TestDescriptorSigning
# ---------------------------------------------------------------------------

class TestDescriptorSigning:
    """Descriptor creation, signing, verification, and conflict resolution."""

    @staticmethod
    def _key_pair():
        priv = Ed25519PrivateKey.generate()
        pub = priv.public_key()
        return priv, pub

    @staticmethod
    def _pub_bytes(pub):
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
        )
        return pub.public_bytes(Encoding.Raw, PublicFormat.Raw)

    def _make_descriptor(self, mgr, pub_bytes, **overrides):
        defaults = dict(
            content_hash=ContentHash.from_data(b"hello"),
            owner_node_id="owner-node",
            visibility="public",
            replication_policy=ReplicationPolicy(replication_factor=3),
            contract_pubkey=pub_bytes,
        )
        defaults.update(overrides)
        return mgr._create_descriptor_stub(**defaults)

    # 7. owner_signed_descriptor — sign + verify roundtrip
    def test_owner_signed_descriptor(self):
        priv, pub = self._key_pair()
        pub_bytes = self._pub_bytes(pub)

        peers = [
            FakePeerInfo(node_id=f"peer-{i}", asn=f"AS{i}")
            for i in range(4)
        ]
        mgr = _make_manager(peers)
        desc = self._make_descriptor(mgr, pub_bytes)

        signed = mgr._sign_descriptor(desc, priv, signer_type="owner")
        assert signed.signature != b""
        assert signed.signer_type == "owner"

        ok = mgr._verify_descriptor_signature(signed, pub)
        assert ok is True

    # 8. contract_key_cannot_change_owner
    def test_contract_key_cannot_change_owner(self):
        priv, pub = self._key_pair()
        pub_bytes = self._pub_bytes(pub)

        peers = [
            FakePeerInfo(node_id=f"peer-{i}", asn=f"AS{i}")
            for i in range(4)
        ]
        mgr = _make_manager(peers)

        base = self._make_descriptor(mgr, pub_bytes)
        updated = self._make_descriptor(mgr, pub_bytes, owner_node_id="evil-node")

        assert mgr._validate_contract_update(base, updated) is False

    # 9. conflict_resolution_highest_epoch_version
    def test_conflict_resolution_highest_epoch_version(self):
        priv, pub = self._key_pair()
        pub_bytes = self._pub_bytes(pub)

        peers = [
            FakePeerInfo(node_id=f"peer-{i}", asn=f"AS{i}")
            for i in range(4)
        ]
        mgr = _make_manager(peers)

        a = self._make_descriptor(mgr, pub_bytes)
        b = self._make_descriptor(mgr, pub_bytes)

        # Same epoch, different version
        a.epoch = 1
        a.version = 2
        b.epoch = 1
        b.version = 5
        winner = mgr._resolve_conflict(a, b)
        assert winner is b

        # Different epoch
        a.epoch = 3
        a.version = 1
        b.epoch = 2
        b.version = 99
        winner = mgr._resolve_conflict(a, b)
        assert winner is a
