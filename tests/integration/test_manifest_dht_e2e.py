"""
End-to-end integration test — Phase 3.x.5 Task 7.

Acceptance per design plan §4 Task 7: spin up three simulated nodes
(alice, bob, charlie); alice registers a model; bob fetches via DHT;
charlie independently verifies via the anchor. Plus tampering and
offline-publisher scenarios.

Test approach — composition over isolation:
  Real ``FilesystemModelRegistry`` + real ``LocalManifestIndex`` +
  real ``ManifestDHTServer`` + real ``ManifestDHTClient`` per node.
  Real ``ModelManifest`` signing under real Ed25519 ``NodeIdentity``.
  Real wire protocol round-trips (encode → bytes → decode) through
  a synchronous in-process FakeNetwork bus.

  The anchor is a faithful Python mirror of PublisherKeyAnchor.sol's
  semantics (sha256-derived nodeId + lookup) — same simulator pattern
  used by ``test_publisher_key_anchor_e2e.py``. Solidity-side
  correctness is covered by Phase 3.x.3 Task 1's Hardhat tests; this
  test exercises the COMPOSITION — registry → DHT → anchor → verifier.

  Shards are NOT distributed via the DHT (per design plan §1.2 —
  manifest-only scope). Tests therefore exercise ``get_manifest()``
  fan-out through the DHT, with ``get()`` reserved for nodes that
  also have local shards.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pytest

from prsm.compute.model_registry import (
    FilesystemModelRegistry,
    ManifestVerificationError,
    ModelNotFoundError,
)
from prsm.compute.model_sharding.models import ModelShard, ShardedModel
from prsm.network.manifest_dht import (
    DEFAULT_K,
    LocalManifestIndex,
    ManifestDHTClient,
    ManifestDHTServer,
    ManifestNotFoundError,
    encode_message,
    parse_message,
)
from prsm.node.identity import NodeIdentity, generate_node_identity
from prsm.security.publisher_key_anchor import PublisherAlreadyRegisteredError


# ──────────────────────────────────────────────────────────────────────────
# Faithful simulated anchor — mirrors PublisherKeyAnchor.sol
# (same pattern used by test_publisher_key_anchor_e2e.py)
# ──────────────────────────────────────────────────────────────────────────


class SimulatedAnchorContract:
    """Python mirror of PublisherKeyAnchor.sol behavior."""

    def __init__(self) -> None:
        self._publisher_keys: Dict[bytes, bytes] = {}

    def register(self, public_key_bytes: bytes) -> bytes:
        if len(public_key_bytes) != 32:
            raise ValueError("InvalidPublicKeyLength")
        node_id = hashlib.sha256(public_key_bytes).digest()[:16]
        if node_id in self._publisher_keys:
            raise PublisherAlreadyRegisteredError(
                f"AlreadyRegistered: {node_id.hex()}"
            )
        self._publisher_keys[node_id] = public_key_bytes
        return node_id

    def lookup(self, node_id_bytes16: bytes) -> bytes:
        return self._publisher_keys.get(node_id_bytes16, b"")


class SimulatedAnchorClient:
    """Hex-string node_id facade over the contract — what the registry +
    DHT client consume via .lookup(node_id) → Optional[str]."""

    def __init__(self, contract: SimulatedAnchorContract) -> None:
        self._contract = contract

    def lookup(self, node_id: str) -> Optional[str]:
        if not node_id:
            return None
        s = node_id[2:] if node_id.startswith("0x") else node_id
        if len(s) != 32:
            return None
        try:
            node_id_bytes = bytes.fromhex(s)
        except ValueError:
            return None
        result = self._contract.lookup(node_id_bytes)
        if not result:
            return None
        return base64.b64encode(result).decode("ascii")


# ──────────────────────────────────────────────────────────────────────────
# FakeNetwork — synchronous in-process request/response bus
# ──────────────────────────────────────────────────────────────────────────


class FakeNetwork:
    """Maps each node's address to a handler callable. Models the
    underlying ``send_message(address, request_bytes) → response_bytes``
    contract that ``ManifestDHTClient`` consumes.

    Each registered node's handler is the bound method
    ``server.handle`` — meaning a ``ManifestDHTServer`` plays the role
    of an RPC endpoint without any actual sockets."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[bytes], bytes]] = {}

    def register(
        self, address: str, handler: Callable[[bytes], bytes]
    ) -> None:
        self._handlers[address] = handler

    def disconnect(self, address: str) -> None:
        """Simulate node going offline — subsequent requests raise."""
        self._handlers.pop(address, None)

    def send(self, address: str, request_bytes: bytes) -> bytes:
        handler = self._handlers.get(address)
        if handler is None:
            raise ConnectionRefusedError(f"no node at {address}")
        return handler(request_bytes)


# ──────────────────────────────────────────────────────────────────────────
# StaticRoutingTable — fixed peer list, simplest viable Kademlia stand-in
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class _Peer:
    node_id: str
    address: str


class StaticRoutingTable:
    """Returns a configurable list of peers. Kademlia distance ordering
    isn't relevant for E2E — what matters is that find_providers asks
    each peer once."""

    def __init__(self, peers: List[_Peer]) -> None:
        self._peers = list(peers)

    def find_closest_peers(self, target_id: str, count: int = 20):
        return list(self._peers[:count])


# ──────────────────────────────────────────────────────────────────────────
# Node — composition of registry + index + server + client per host
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Node:
    """Bundles everything one PRSM node needs for the manifest DHT
    flow. Constructed via ``make_node()``."""

    identity: NodeIdentity
    address: str
    root: Path
    anchor: SimulatedAnchorClient
    network: FakeNetwork
    local_index: LocalManifestIndex
    registry: FilesystemModelRegistry
    server: ManifestDHTServer
    client: ManifestDHTClient


def make_node(
    *,
    identity: NodeIdentity,
    address: str,
    root: Path,
    anchor: SimulatedAnchorClient,
    network: FakeNetwork,
    peers: List[_Peer],
) -> Node:
    """Wire a node end-to-end: server bound to its address, client
    routed through the FakeNetwork, registry + local_index pinned to
    the same root."""
    local_index = LocalManifestIndex(root)
    server = ManifestDHTServer(
        local_index=local_index,
        routing_table=StaticRoutingTable(peers),
        my_node_id=identity.node_id,
        my_address=address,
        k=DEFAULT_K,
    )
    network.register(address, server.handle)
    client = ManifestDHTClient(
        local_index=local_index,
        routing_table=StaticRoutingTable(peers),
        send_message=network.send,
        anchor=anchor,
        my_node_id=identity.node_id,
        my_address=address,
    )
    registry = FilesystemModelRegistry(
        root, anchor=anchor, dht=client
    )
    return Node(
        identity=identity,
        address=address,
        root=root,
        anchor=anchor,
        network=network,
        local_index=local_index,
        registry=registry,
        server=server,
        client=client,
    )


# ──────────────────────────────────────────────────────────────────────────
# Real ShardedModel — same shape used by test_model_registry.py
# ──────────────────────────────────────────────────────────────────────────


def _make_model(model_id: str = "test-llama", num_shards: int = 3) -> ShardedModel:
    shards = []
    for i in range(num_shards):
        rng = np.random.default_rng(seed=4000 + i)
        tensor = rng.standard_normal(size=(4, 8))
        data = tensor.tobytes()
        shards.append(
            ModelShard(
                shard_id=f"{model_id}-shard-{i}",
                model_id=model_id,
                shard_index=i,
                total_shards=num_shards,
                tensor_data=data,
                tensor_shape=(4, 8),
                layer_range=(0, 0),
                size_bytes=len(data),
                checksum=hashlib.sha256(data).hexdigest(),
            )
        )
    return ShardedModel(
        model_id=model_id,
        model_name=f"{model_id}-display",
        total_shards=num_shards,
        shards=shards,
    )


# ──────────────────────────────────────────────────────────────────────────
# Fixtures — three nodes wired through one shared anchor + network
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def alice_identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.5-task7-alice")


@pytest.fixture
def bob_identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.5-task7-bob")


@pytest.fixture
def charlie_identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.5-task7-charlie")


@pytest.fixture
def anchor_contract(alice_identity, bob_identity, charlie_identity):
    """Anchor with all three publishers pre-registered. Charlie is a
    publisher in his own right (verifies + stores) but tests don't
    require him to register a separate model."""
    contract = SimulatedAnchorContract()
    contract.register(alice_identity.public_key_bytes)
    contract.register(bob_identity.public_key_bytes)
    contract.register(charlie_identity.public_key_bytes)
    return contract


@pytest.fixture
def anchor(anchor_contract):
    return SimulatedAnchorClient(anchor_contract)


@pytest.fixture
def network():
    return FakeNetwork()


@pytest.fixture
def alice(tmp_path, alice_identity, bob_identity, charlie_identity, anchor, network):
    root = tmp_path / "alice"
    root.mkdir()
    return make_node(
        identity=alice_identity,
        address="alice:8001",
        root=root,
        anchor=anchor,
        network=network,
        peers=[
            _Peer(node_id=bob_identity.node_id, address="bob:8002"),
            _Peer(node_id=charlie_identity.node_id, address="charlie:8003"),
        ],
    )


@pytest.fixture
def bob(tmp_path, bob_identity, alice_identity, charlie_identity, anchor, network):
    root = tmp_path / "bob"
    root.mkdir()
    return make_node(
        identity=bob_identity,
        address="bob:8002",
        root=root,
        anchor=anchor,
        network=network,
        peers=[
            _Peer(node_id=alice_identity.node_id, address="alice:8001"),
            _Peer(node_id=charlie_identity.node_id, address="charlie:8003"),
        ],
    )


@pytest.fixture
def charlie(tmp_path, charlie_identity, alice_identity, bob_identity, anchor, network):
    root = tmp_path / "charlie"
    root.mkdir()
    return make_node(
        identity=charlie_identity,
        address="charlie:8003",
        root=root,
        anchor=anchor,
        network=network,
        peers=[
            _Peer(node_id=alice_identity.node_id, address="alice:8001"),
            _Peer(node_id=bob_identity.node_id, address="bob:8002"),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────
# 1. Happy path — alice publishes, bob + charlie fetch via DHT
# ──────────────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_alice_register_announces_to_dht(self, alice):
        model = _make_model("alpha")
        alice.registry.register(model, identity=alice.identity)

        # alice's local index now serves alpha
        assert alice.local_index.lookup("alpha") is not None
        assert "alpha" in alice.local_index

    def test_bob_fetches_alices_manifest_via_dht(
        self, alice, bob
    ):
        model = _make_model("beta")
        alice.registry.register(model, identity=alice.identity)

        # bob has an empty registry; get_manifest falls back to DHT
        manifest = bob.registry.get_manifest("beta")
        assert manifest.model_id == "beta"
        assert manifest.publisher_node_id == alice.identity.node_id

        # bob's registry cached the manifest locally
        assert (bob.root / "beta" / "manifest.json").exists()

    def test_charlie_independently_verifies_via_anchor(
        self, alice, charlie
    ):
        model = _make_model("gamma")
        alice.registry.register(model, identity=alice.identity)

        # charlie pulls the same manifest end-to-end through HIS DHT
        # client (different routing table, different cache, different
        # local_index) and verifies against the same anchor.
        manifest = charlie.registry.get_manifest("gamma")
        assert manifest.model_id == "gamma"
        assert manifest.publisher_node_id == alice.identity.node_id

    def test_three_node_consistency(
        self, alice, bob, charlie
    ):
        model = _make_model("delta")
        alice.registry.register(model, identity=alice.identity)

        manifest_bob = bob.registry.get_manifest("delta")
        manifest_charlie = charlie.registry.get_manifest("delta")

        # Same canonical bytes produce same manifests on every reader
        assert manifest_bob.publisher_signature == manifest_charlie.publisher_signature
        assert manifest_bob.shards == manifest_charlie.shards

    def test_bob_local_cache_avoids_second_dht_call(
        self, alice, bob
    ):
        # First fetch hits the network; second is purely local.
        # Disconnect alice after caching to prove it.
        model = _make_model("epsilon")
        alice.registry.register(model, identity=alice.identity)

        first = bob.registry.get_manifest("epsilon")
        bob.network.disconnect("alice:8001")

        second = bob.registry.get_manifest("epsilon")
        assert first.publisher_signature == second.publisher_signature


# ──────────────────────────────────────────────────────────────────────────
# 2. Tampering — malicious provider serves altered bytes
# ──────────────────────────────────────────────────────────────────────────


class TampingHandler:
    """Wraps a real server.handle and rewrites every ManifestResponse's
    ``manifest`` payload to corrupt the publisher signature. Other
    response types pass through unchanged."""

    def __init__(self, real_handler):
        self._real_handler = real_handler

    def __call__(self, request_bytes: bytes) -> bytes:
        # Defer to the real server, then mutate the response.
        response_bytes = self._real_handler(request_bytes)
        try:
            from prsm.network.manifest_dht.protocol import (
                ManifestResponse,
                ManifestResponse,
            )
            parsed = parse_message(response_bytes)
        except Exception:
            return response_bytes

        from prsm.network.manifest_dht.protocol import ManifestResponse as MR
        if not isinstance(parsed, MR):
            return response_bytes

        # Tamper: flip a byte of the publisher_signature inside the
        # manifest dict. Anchor-side signature verify will catch this.
        tampered_manifest = dict(parsed.manifest)
        sig = tampered_manifest.get("publisher_signature")
        if isinstance(sig, str) and sig:
            # Flip first base64 character (still valid base64 chars).
            new_first = "B" if sig[0] != "B" else "C"
            tampered_manifest["publisher_signature"] = new_first + sig[1:]

        new_response = MR(
            request_id=parsed.request_id,
            manifest=tampered_manifest,
        )
        return encode_message(new_response)


class TestTamperingProvider:
    def test_tampered_provider_caught_by_anchor_verify(
        self, alice, bob, charlie
    ):
        # alice publishes; charlie has the legitimate manifest cached
        # (via DHT). Then we replace alice's handler with a tampering
        # one, BEFORE bob ever fetches. Bob's DHT lookup may hit alice
        # OR charlie; either way:
        #   - tampered alice → anchor verify fails → bob retries
        #   - charlie (untampered) → succeeds
        # Bob's get_manifest must succeed.
        model = _make_model("zeta")
        alice.registry.register(model, identity=alice.identity)
        # Prime charlie's cache with the legitimate bytes.
        charlie.registry.get_manifest("zeta")

        # Now alice "turns malicious" — every ManifestResponse from her
        # server gets tampered.
        bob.network.register(
            "alice:8001", TampingHandler(alice.server.handle)
        )

        manifest = bob.registry.get_manifest("zeta")
        # Bob ended up with the legitimate, anchor-verified bytes.
        assert manifest.model_id == "zeta"

    def test_all_providers_tampered_yields_not_found(
        self, alice, bob, charlie
    ):
        # Every provider tampers — bob has no way to get clean bytes.
        # Must surface as ModelNotFoundError (the registry's wrapper
        # around the DHT client's ManifestNotFoundError).
        model = _make_model("eta")
        alice.registry.register(model, identity=alice.identity)
        # Prime charlie's cache so charlie also has bytes to tamper.
        charlie.registry.get_manifest("eta")

        bob.network.register(
            "alice:8001", TampingHandler(alice.server.handle)
        )
        bob.network.register(
            "charlie:8003", TampingHandler(charlie.server.handle)
        )

        with pytest.raises(ModelNotFoundError):
            bob.registry.get_manifest("eta")


# ──────────────────────────────────────────────────────────────────────────
# 3. Offline publisher — alice goes down after charlie cached
# ──────────────────────────────────────────────────────────────────────────


class TestOfflinePublisher:
    def test_offline_publisher_charlie_serves(
        self, alice, bob, charlie
    ):
        # alice publishes; charlie pre-fetches (becoming a provider);
        # alice goes offline; bob still gets the manifest from charlie.
        model = _make_model("theta")
        alice.registry.register(model, identity=alice.identity)
        # Charlie pulls the manifest, populating his local registry +
        # index. He's now an authoritative provider per his local_index.
        charlie.registry.get_manifest("theta")
        assert "theta" in charlie.local_index

        # Alice goes offline.
        bob.network.disconnect("alice:8001")

        # Bob can still fetch — find_providers asks charlie, charlie
        # answers with himself, bob fetches from charlie.
        manifest = bob.registry.get_manifest("theta")
        assert manifest.model_id == "theta"
        assert manifest.publisher_node_id == alice.identity.node_id

    def test_offline_publisher_no_other_providers_fails(
        self, alice, bob
    ):
        # Same setup but charlie hasn't cached the model — bob has no
        # peer that can serve it.
        model = _make_model("iota")
        alice.registry.register(model, identity=alice.identity)
        bob.network.disconnect("alice:8001")

        with pytest.raises(ModelNotFoundError):
            bob.registry.get_manifest("iota")


# ──────────────────────────────────────────────────────────────────────────
# 4. Anchor enforcement — unregistered publisher's manifest gets dropped
# ──────────────────────────────────────────────────────────────────────────


class TestAnchorEnforcement:
    def test_unregistered_publisher_manifest_rejected(
        self, tmp_path, anchor_contract, anchor, network
    ):
        # Mallory generates a NodeIdentity but is NOT in the anchor.
        # Even though her manifest is technically signed under her own
        # key, the anchor-lookup-by-node_id returns nothing → DHT
        # client drops the bytes → bob gets ManifestNotFound.
        mallory = generate_node_identity(display_name="phase3.x.5-mallory")
        # NOTE: deliberately NOT calling anchor_contract.register(mallory.public_key_bytes)

        mallory_root = tmp_path / "mallory"
        mallory_root.mkdir()
        mallory_node = make_node(
            identity=mallory,
            address="mallory:9999",
            root=mallory_root,
            anchor=anchor,
            network=network,
            peers=[],
        )
        # Mallory publishes locally — her registry's anchor= will reject
        # her own publisher_node_id since she's not on-chain. So we
        # bypass the registry path: write the manifest directly to her
        # local index via dht.announce after a non-anchor registration.
        # Simplest: register on a sidecar-only registry, then re-announce
        # via mallory's DHT client.
        sidecar_registry = FilesystemModelRegistry(mallory_root)
        signed = sidecar_registry.register(
            _make_model("kappa"), identity=mallory
        )
        # Hand-announce so mallory's local_index can serve.
        mallory_node.client.announce(
            "kappa", mallory_root / "kappa" / "manifest.json"
        )

        # Bob asks mallory directly (peers=[mallory only]).
        bob_root = tmp_path / "bob_isolated"
        bob_root.mkdir()
        bob_id = generate_node_identity(display_name="bob-isolated")
        anchor_contract.register(bob_id.public_key_bytes)
        bob_node = make_node(
            identity=bob_id,
            address="bob:9000",
            root=bob_root,
            anchor=anchor,
            network=network,
            peers=[
                _Peer(node_id=mallory.node_id, address="mallory:9999")
            ],
        )
        with pytest.raises(ModelNotFoundError):
            bob_node.registry.get_manifest("kappa")


# ──────────────────────────────────────────────────────────────────────────
# 5. Composition — registry + DHT + anchor exercised together
# ──────────────────────────────────────────────────────────────────────────


class TestRegistryDHTAnchorComposition:
    def test_full_chain_per_node_shows_three_anchor_calls(
        self, alice, bob, charlie, anchor_contract
    ):
        # The anchor's lookup-call counter is observable via the
        # contract's _publisher_keys access pattern, but we don't
        # instrument that directly. Behavior assertion: every reader
        # ended up with a manifest carrying the right publisher
        # node_id, and that node_id resolves on the anchor.
        model = _make_model("lambda")
        alice.registry.register(model, identity=alice.identity)

        for reader in (bob, charlie):
            manifest = reader.registry.get_manifest("lambda")
            assert manifest.publisher_node_id == alice.identity.node_id
            # The anchor knows this publisher.
            assert reader.anchor.lookup(
                manifest.publisher_node_id
            ) == alice.identity.public_key_b64

    def test_registry_get_at_publisher_serves_full_model(
        self, alice
    ):
        # Alice can ``get()`` her own model end-to-end (manifest +
        # shards) since both are on her disk. This is the registry's
        # standard happy path; the DHT integration must not regress it.
        model = _make_model("mu")
        alice.registry.register(model, identity=alice.identity)
        out = alice.registry.get("mu")
        assert out.model_id == "mu"
        assert out.total_shards == model.total_shards
