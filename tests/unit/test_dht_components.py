"""Unit tests for DHTNodeComponents — full DHT-stack aggregator.

Lifecycle, validation, peer-discovery hook, and a 2-node ManifestDHT
E2E roundtrip exercising the full stack: KademliaDHT routing →
SyncDHTTransport → DHTListener → DHTRequestRouter → ManifestDHTServer
→ ManifestDHTClient anchor verification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import pytest

from prsm.compute.model_registry.models import (
    ManifestShardEntry, ModelManifest,
)
from prsm.compute.model_registry.signing import sign_manifest
from prsm.network.dht_components import DHTNodeComponents
from prsm.network.embedding_dht.local_index import LocalEmbeddingIndex
from prsm.network.manifest_dht.local_index import LocalManifestIndex
from prsm.network.manifest_dht.dht_client import (
    ManifestNotFoundError,
)
from prsm.node.identity import generate_node_identity
from prsm.node.transport_adapter import DirectAdapter


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


class _FakeAnchor:
    """In-memory anchor: node_id → pubkey_b64. Mirrors what the
    on-chain PublisherKeyAnchor exposes so ManifestDHTClient's
    verify_manifest_with_anchor() works."""

    def __init__(self):
        self._registrations: dict[str, str] = {}

    def register(self, node_id: str, pubkey_b64: str) -> None:
        self._registrations[node_id] = pubkey_b64

    def lookup(self, node_id: str) -> Optional[str]:
        return self._registrations.get(node_id)


def _build_signed_manifest(
    identity, model_id: str = "llama-3-8b",
) -> ModelManifest:
    shards = [
        ManifestShardEntry(
            shard_id=f"sid-{i}",
            shard_index=i,
            tensor_shape=(8, 16),
            sha256=hashlib.sha256(f"shard-{i}".encode()).hexdigest(),
            size_bytes=7,
        )
        for i in range(2)
    ]
    unsigned = ModelManifest(
        model_id=model_id,
        model_name=model_id,
        publisher_node_id="placeholder",
        total_shards=len(shards),
        shards=tuple(shards),
        published_at=1714000000.0,
    )
    return sign_manifest(unsigned, identity)


def _hex_node_id(seed: str) -> str:
    """Produce a 32-char hex node_id matching NodeIdentity's
    convention (16 bytes / 128 bits). KademliaDHT's 160-bucket
    routing table accepts any hex of <=160 bits — using 32 chars
    here keeps the synthetic test ids in the same shape as
    production NodeIdentity.node_id values."""
    return hashlib.sha256(seed.encode()).hexdigest()[:32]


def _make_manifest_index(tmp_path: Path, sub: str) -> LocalManifestIndex:
    """LocalManifestIndex requires an existing directory root."""
    root = tmp_path / sub
    root.mkdir(parents=True, exist_ok=True)
    return LocalManifestIndex(root)


def _make_embedding_index(tmp_path: Path, sub: str) -> LocalEmbeddingIndex:
    root = tmp_path / sub
    root.mkdir(parents=True, exist_ok=True)
    return LocalEmbeddingIndex(root)


def _persist_manifest_in_root(
    index_root: Path, model_id: str, manifest: ModelManifest,
) -> Path:
    """Write a signed manifest under the index root so register()
    accepts it (the index rejects paths outside its root)."""
    path = index_root / f"{model_id}.json"
    path.write_text(json.dumps(manifest.to_dict()))
    return path


# ──────────────────────────────────────────────────────────────────────
# constructor / validation
# ──────────────────────────────────────────────────────────────────────


class TestBuild:
    def test_rejects_no_indexes(self, tmp_path):
        with pytest.raises(ValueError, match="at least one of"):
            DHTNodeComponents.build(
                my_node_id=_hex_node_id("a"),
                my_host="127.0.0.1",
                dht_listen_port=0,
                transport_adapter=DirectAdapter(),
            )

    def test_rejects_manifest_index_without_anchor(self, tmp_path):
        idx = _make_manifest_index(tmp_path, "m_idx.json")
        with pytest.raises(ValueError, match="anchor"):
            DHTNodeComponents.build(
                my_node_id=_hex_node_id("a"),
                my_host="127.0.0.1",
                dht_listen_port=0,
                transport_adapter=DirectAdapter(),
                local_manifest_index=idx,
                # anchor=... missing
            )

    def test_rejects_embedding_index_without_verifier(self, tmp_path):
        idx = _make_embedding_index(tmp_path, "e_idx")
        with pytest.raises(ValueError, match="creator_pubkey_for"):
            DHTNodeComponents.build(
                my_node_id=_hex_node_id("a"),
                my_host="127.0.0.1",
                dht_listen_port=0,
                transport_adapter=DirectAdapter(),
                local_embedding_index=idx,
                # creator_pubkey_for / verify_signature missing
            )

    def test_rejects_empty_node_id(self, tmp_path):
        idx = _make_manifest_index(tmp_path, "m_idx.json")
        with pytest.raises(ValueError, match="my_node_id"):
            DHTNodeComponents.build(
                my_node_id="",
                my_host="127.0.0.1",
                dht_listen_port=0,
                transport_adapter=DirectAdapter(),
                local_manifest_index=idx,
                anchor=_FakeAnchor(),
            )

    def test_rejects_empty_host(self, tmp_path):
        idx = _make_manifest_index(tmp_path, "m_idx.json")
        with pytest.raises(ValueError, match="my_host"):
            DHTNodeComponents.build(
                my_node_id=_hex_node_id("a"),
                my_host="",
                dht_listen_port=0,
                transport_adapter=DirectAdapter(),
                local_manifest_index=idx,
                anchor=_FakeAnchor(),
            )

    def test_manifest_only_build(self, tmp_path):
        idx = _make_manifest_index(tmp_path, "m_idx.json")
        comp = DHTNodeComponents.build(
            my_node_id=_hex_node_id("a"),
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            local_manifest_index=idx,
            anchor=_FakeAnchor(),
        )
        assert comp.manifest_server is not None
        assert comp.embedding_server is None

    def test_embedding_only_build(self, tmp_path):
        idx = _make_embedding_index(tmp_path, "e_idx")
        comp = DHTNodeComponents.build(
            my_node_id=_hex_node_id("a"),
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            local_embedding_index=idx,
            creator_pubkey_for=lambda _: None,
            verify_signature=lambda *_: True,
        )
        assert comp.manifest_server is None
        assert comp.embedding_server is not None

    def test_both_built(self, tmp_path):
        comp = DHTNodeComponents.build(
            my_node_id=_hex_node_id("a"),
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            local_manifest_index=_make_manifest_index(tmp_path, "m.json"),
            anchor=_FakeAnchor(),
            local_embedding_index=_make_embedding_index(tmp_path, "e_idx"),
            creator_pubkey_for=lambda _: None,
            verify_signature=lambda *_: True,
        )
        assert comp.manifest_server is not None
        assert comp.embedding_server is not None


# ──────────────────────────────────────────────────────────────────────
# lifecycle
# ──────────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_start_stop(self, tmp_path):
        comp = DHTNodeComponents.build(
            my_node_id=_hex_node_id("a"),
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_manifest_index=_make_manifest_index(tmp_path, "m.json"),
            anchor=_FakeAnchor(),
        )
        assert comp.is_running is False
        try:
            port = comp.start(anchor=_FakeAnchor())
            assert port > 0
            assert comp.listen_port == port
            assert comp.is_running is True
            assert comp.transport is not None
            assert comp.manifest_client is not None
        finally:
            comp.stop()
        assert comp.is_running is False

    def test_start_idempotent(self, tmp_path):
        anchor = _FakeAnchor()
        comp = DHTNodeComponents.build(
            my_node_id=_hex_node_id("a"),
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_manifest_index=_make_manifest_index(tmp_path, "m.json"),
            anchor=anchor,
        )
        try:
            port1 = comp.start(anchor=anchor)
            port2 = comp.start(anchor=anchor)
            assert port1 == port2
        finally:
            comp.stop()

    def test_stop_idempotent(self, tmp_path):
        comp = DHTNodeComponents.build(
            my_node_id=_hex_node_id("a"),
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_manifest_index=_make_manifest_index(tmp_path, "m.json"),
            anchor=_FakeAnchor(),
        )
        comp.start(anchor=_FakeAnchor())
        comp.stop()
        comp.stop()  # no raise

    def test_start_requires_anchor_for_manifest(self, tmp_path):
        comp = DHTNodeComponents.build(
            my_node_id=_hex_node_id("a"),
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_manifest_index=_make_manifest_index(tmp_path, "m.json"),
            anchor=_FakeAnchor(),
        )
        try:
            with pytest.raises(RuntimeError, match="anchor"):
                comp.start()  # anchor not passed
        finally:
            comp.stop()


# ──────────────────────────────────────────────────────────────────────
# add_peer
# ──────────────────────────────────────────────────────────────────────


class TestAddPeer:
    def test_adds_peer_to_routing_table(self, tmp_path):
        comp = DHTNodeComponents.build(
            my_node_id=_hex_node_id("a"),
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_manifest_index=_make_manifest_index(tmp_path, "m.json"),
            anchor=_FakeAnchor(),
        )
        peer_id = _hex_node_id("b")
        added = comp.add_peer(peer_id, "10.0.0.2", 9000)
        assert added is True
        # Routing table should now know about the peer.
        peers = comp.kademlia.find_closest_peers(peer_id, count=10)
        assert any(p.node_id == peer_id for p in peers)

    def test_rejects_self(self, tmp_path):
        my_id = _hex_node_id("a")
        comp = DHTNodeComponents.build(
            my_node_id=my_id,
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_manifest_index=_make_manifest_index(tmp_path, "m.json"),
            anchor=_FakeAnchor(),
        )
        # Adding self returns False per KademliaDHT contract.
        assert comp.add_peer(my_id, "127.0.0.1", 9000) is False

    def test_rejects_invalid_node_id(self, tmp_path):
        comp = DHTNodeComponents.build(
            my_node_id=_hex_node_id("a"),
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_manifest_index=_make_manifest_index(tmp_path, "m.json"),
            anchor=_FakeAnchor(),
        )
        with pytest.raises(ValueError, match="node_id"):
            comp.add_peer("", "10.0.0.2", 9000)

    def test_rejects_invalid_port(self, tmp_path):
        comp = DHTNodeComponents.build(
            my_node_id=_hex_node_id("a"),
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_manifest_index=_make_manifest_index(tmp_path, "m.json"),
            anchor=_FakeAnchor(),
        )
        with pytest.raises(ValueError, match="port"):
            comp.add_peer(_hex_node_id("b"), "10.0.0.2", 0)


# ──────────────────────────────────────────────────────────────────────
# 2-node ManifestDHT E2E — exercises the full stack
# ──────────────────────────────────────────────────────────────────────


class TestTwoNodeManifestE2E:
    def test_node_b_fetches_manifest_from_node_a(self, tmp_path):
        """Node A serves a signed manifest. Node B knows A as a peer
        via add_peer. Node B calls manifest_client.get_manifest() →
        DHT routes to A → A returns provider info → B fetches the
        manifest → anchor-verifies → returns it.
        """
        # ── Node A: publisher; has the manifest in its local index ──
        a_identity = generate_node_identity("node-a")
        a_node_id = a_identity.node_id
        manifest = _build_signed_manifest(a_identity, model_id="m1")
        a_index = _make_manifest_index(tmp_path, "a_idx")
        a_manifest_path = _persist_manifest_in_root(
            tmp_path / "a_idx", "m1", manifest,
        )
        a_index.register("m1", a_manifest_path)

        # Both nodes share an anchor that knows A's pubkey. (In
        # production, anchor.lookup queries the on-chain contract.)
        anchor = _FakeAnchor()
        anchor.register(a_node_id, a_identity.public_key_b64)

        node_a = DHTNodeComponents.build(
            my_node_id=a_node_id,
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_manifest_index=a_index,
            anchor=anchor,
        )

        # ── Node B: empty index; will fetch from A ──
        b_identity = generate_node_identity("node-b")
        b_index = _make_manifest_index(tmp_path, "b_idx.json")
        node_b = DHTNodeComponents.build(
            my_node_id=b_identity.node_id,
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_manifest_index=b_index,
            anchor=anchor,
        )

        try:
            port_a = node_a.start(anchor=anchor)
            node_b.start(anchor=anchor)

            # Tell B about A so the routing table can hand A to the
            # client when B queries find_closest_peers(model_id_hash).
            assert node_b.add_peer(a_node_id, "127.0.0.1", port_a)

            # B fetches the manifest cross-node + anchor-verifies.
            assert node_b.manifest_client is not None
            fetched = node_b.manifest_client.get_manifest("m1")
            assert fetched.model_id == "m1"
            assert fetched.publisher_signature == manifest.publisher_signature
        finally:
            node_b.stop()
            node_a.stop()

    def test_node_b_fails_when_no_provider(self, tmp_path):
        """B has no peers in its routing table → manifest_client
        raises ManifestNotFoundError, NOT a transport error. Confirms
        the contract distinguishes 'no providers' from 'transport
        failed'."""
        b_identity = generate_node_identity("node-b")
        anchor = _FakeAnchor()
        b_index = _make_manifest_index(tmp_path, "b_idx.json")
        node_b = DHTNodeComponents.build(
            my_node_id=b_identity.node_id,
            my_host="127.0.0.1",
            dht_listen_port=0,
            transport_adapter=DirectAdapter(),
            listen_host="127.0.0.1",
            local_manifest_index=b_index,
            anchor=anchor,
        )
        try:
            node_b.start(anchor=anchor)
            assert node_b.manifest_client is not None
            with pytest.raises(ManifestNotFoundError):
                node_b.manifest_client.get_manifest("m-unknown")
        finally:
            node_b.stop()
