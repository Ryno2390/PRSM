"""
Unit tests — Phase 3.x.5 Task 3 — ManifestDHTClient.

Acceptance per design plan §4 Task 3: client behavior pinned by
tests; the no-anchor-no-trust invariant is enforced + tested.

Real ModelManifest construction + signing — only the transport
(``send_message``) and routing table are mocked. Tests exercise the
full encode/decode cycle.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from prsm.compute.model_registry.models import (
    ManifestShardEntry,
    ModelManifest,
)
from prsm.compute.model_registry.signing import sign_manifest
from prsm.node.identity import NodeIdentity, generate_node_identity
from prsm.network.manifest_dht import (
    DEFAULT_K,
    LocalManifestIndex,
    ManifestDHTClient,
    ManifestNotFoundError,
    TransportFailureError,
)
from prsm.network.manifest_dht.protocol import (
    ErrorCode,
    ErrorResponse,
    FetchManifestRequest,
    FindProvidersRequest,
    ManifestResponse,
    ProviderInfo,
    ProvidersResponse,
    encode_message,
    parse_message,
)


# ──────────────────────────────────────────────────────────────────────────
# Fakes — minimal stand-ins for routing table + transport + anchor
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class FakePeer:
    """Minimal duck-typed PeerNode for routing-table fixtures."""
    node_id: str
    address: str


class FakeRoutingTable:
    """Routing table that returns a fixed list of peers regardless
    of target_id. Tests inject the peer list at construction."""

    def __init__(self, peers: List[FakePeer]):
        self._peers = peers

    def find_closest_peers(self, target_id: str, count: int = 20):
        return list(self._peers[:count])


class FakeAnchor:
    """In-memory anchor — node_id → public_key_b64."""

    def __init__(self, registrations: dict[str, str] | None = None):
        self._registrations = dict(registrations or {})

    def lookup(self, node_id: str) -> str | None:
        return self._registrations.get(node_id)

    def register(self, node_id: str, pubkey_b64: str) -> None:
        self._registrations[node_id] = pubkey_b64


class FakeNetwork:
    """Models a synchronous request/response transport. Each peer
    address routes to a server-style callable that takes raw request
    bytes and returns raw response bytes.

    Tests register handlers per address; the client's send_message
    is wired to FakeNetwork.send."""

    def __init__(self):
        self._handlers: dict[str, callable] = {}

    def register_handler(self, address: str, handler):
        self._handlers[address] = handler

    def remove_handler(self, address: str):
        self._handlers.pop(address, None)

    def send(self, address: str, request_bytes: bytes) -> bytes:
        handler = self._handlers.get(address)
        if handler is None:
            raise ConnectionRefusedError(f"no handler at {address}")
        return handler(request_bytes)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _build_signed_manifest(identity: NodeIdentity, model_id: str = "llama-3-8b") -> ModelManifest:
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


def _make_handler_returning(response_obj):
    """Build a fake server handler that ignores the request and
    returns ``response_obj`` (encoded)."""
    encoded = encode_message(response_obj)

    def handler(request_bytes: bytes) -> bytes:
        # Parse the request to echo back the request_id correctly.
        request = parse_message(request_bytes)
        # Replace the response's request_id with the request's so
        # correlation succeeds.
        replaced = dataclasses.replace(response_obj, request_id=request.request_id)
        return encode_message(replaced)

    return handler


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def alice() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.5-task3-alice")


@pytest.fixture
def bob() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.5-task3-bob")


@pytest.fixture
def network() -> FakeNetwork:
    return FakeNetwork()


@pytest.fixture
def empty_index(tmp_path) -> LocalManifestIndex:
    return LocalManifestIndex(tmp_path)


@pytest.fixture
def anchor(alice) -> FakeAnchor:
    return FakeAnchor(registrations={alice.node_id: alice.public_key_b64})


def _new_client(
    network: FakeNetwork,
    anchor: FakeAnchor,
    local_index: LocalManifestIndex,
    peers: List[FakePeer] | None = None,
    *,
    my_node_id: str = "node-self",
    my_address: str = "self:8765",
) -> ManifestDHTClient:
    return ManifestDHTClient(
        local_index=local_index,
        routing_table=FakeRoutingTable(peers or []),
        send_message=network.send,
        anchor=anchor,
        my_node_id=my_node_id,
        my_address=my_address,
    )


# ──────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_anchor_required(self, network, empty_index):
        with pytest.raises(RuntimeError, match="anchor"):
            ManifestDHTClient(
                local_index=empty_index,
                routing_table=FakeRoutingTable([]),
                send_message=network.send,
                anchor=None,
                my_node_id="self",
                my_address="self:8765",
            )

    def test_anchor_must_have_lookup(self, network, empty_index):
        with pytest.raises(RuntimeError, match="lookup"):
            ManifestDHTClient(
                local_index=empty_index,
                routing_table=FakeRoutingTable([]),
                send_message=network.send,
                anchor=object(),  # no .lookup
                my_node_id="self",
                my_address="self:8765",
            )

    def test_invalid_my_node_id(self, network, empty_index, anchor):
        with pytest.raises(ValueError, match="my_node_id"):
            ManifestDHTClient(
                local_index=empty_index,
                routing_table=FakeRoutingTable([]),
                send_message=network.send,
                anchor=anchor,
                my_node_id="",
                my_address="self:8765",
            )

    def test_invalid_my_address(self, network, empty_index, anchor):
        with pytest.raises(ValueError, match="host:port"):
            ManifestDHTClient(
                local_index=empty_index,
                routing_table=FakeRoutingTable([]),
                send_message=network.send,
                anchor=anchor,
                my_node_id="self",
                my_address="no-port-here",
            )


# ──────────────────────────────────────────────────────────────────────────
# announce — local-only
# ──────────────────────────────────────────────────────────────────────────


class TestAnnounce:
    def test_announce_registers_in_local_index(
        self, network, anchor, empty_index, tmp_path
    ):
        client = _new_client(network, anchor, empty_index)
        # Build a real manifest path so the index validates it.
        model_dir = tmp_path / "alpha"
        model_dir.mkdir()
        manifest_path = model_dir / "manifest.json"
        manifest_path.write_text("{}")

        client.announce("alpha", manifest_path)
        assert empty_index.lookup("alpha") == manifest_path.resolve()

    def test_announce_does_not_send_network_message(
        self, network, anchor, empty_index, tmp_path
    ):
        # No handlers registered on the network; if announce broadcast,
        # send would raise. v1 explicitly does NOT broadcast.
        client = _new_client(network, anchor, empty_index)
        model_dir = tmp_path / "beta"
        model_dir.mkdir()
        manifest_path = model_dir / "manifest.json"
        manifest_path.write_text("{}")
        # No exception → no broadcast happened.
        client.announce("beta", manifest_path)


# ──────────────────────────────────────────────────────────────────────────
# find_providers — single-round Kademlia query
# ──────────────────────────────────────────────────────────────────────────


class TestFindProviders:
    def test_includes_self_when_local_index_has_model(
        self, network, anchor, empty_index, tmp_path
    ):
        # Seed the local index so this node is a provider.
        model_dir = tmp_path / "alpha"
        model_dir.mkdir()
        manifest_path = model_dir / "manifest.json"
        manifest_path.write_text("{}")
        empty_index.register("alpha", manifest_path)

        client = _new_client(network, anchor, empty_index)
        providers = client.find_providers("alpha")

        assert len(providers) == 1
        assert providers[0].node_id == "node-self"
        assert providers[0].address == "self:8765"

    def test_queries_routing_table_peers(
        self, network, anchor, empty_index
    ):
        # No local provider; one peer responds with itself as provider.
        peer = FakePeer(node_id="bob", address="bob:8765")
        peer_self_response = ProvidersResponse(
            request_id="placeholder",
            providers=(ProviderInfo(node_id="bob", address="bob:8765"),),
        )
        network.register_handler(
            "bob:8765", _make_handler_returning(peer_self_response)
        )

        client = _new_client(network, anchor, empty_index, peers=[peer])
        providers = client.find_providers("alpha")

        assert len(providers) == 1
        assert providers[0].node_id == "bob"

    def test_skips_self_in_routing_table(
        self, network, anchor, empty_index
    ):
        # Routing table includes self — should be skipped.
        self_peer = FakePeer(node_id="node-self", address="self:8765")
        # If the client tried to query self, FakeNetwork would raise
        # (no handler registered). Test passes iff no exception.
        client = _new_client(network, anchor, empty_index, peers=[self_peer])
        providers = client.find_providers("alpha")
        assert providers == []  # nothing local, nothing from peers

    def test_dedupes_providers(self, network, anchor, empty_index):
        # Two peers both report bob as a provider — bob should appear
        # only once in the result.
        peer_a = FakePeer(node_id="charlie", address="charlie:1")
        peer_b = FakePeer(node_id="dave", address="dave:1")
        bob_resp = ProvidersResponse(
            request_id="x",
            providers=(ProviderInfo(node_id="bob", address="bob:8765"),),
        )
        network.register_handler("charlie:1", _make_handler_returning(bob_resp))
        network.register_handler("dave:1", _make_handler_returning(bob_resp))

        client = _new_client(network, anchor, empty_index, peers=[peer_a, peer_b])
        providers = client.find_providers("m")
        bob_count = sum(1 for p in providers if p.node_id == "bob")
        assert bob_count == 1

    def test_unreachable_peer_does_not_fail_lookup(
        self, network, anchor, empty_index
    ):
        # peer_a unreachable, peer_b responds — find_providers
        # returns peer_b's providers without erroring on peer_a.
        peer_a = FakePeer(node_id="dead", address="dead:1")
        peer_b = FakePeer(node_id="alive", address="alive:1")
        # Only peer_b has a handler. Sending to peer_a raises.
        network.register_handler(
            "alive:1",
            _make_handler_returning(
                ProvidersResponse(
                    request_id="x",
                    providers=(ProviderInfo(node_id="bob", address="bob:1"),),
                )
            ),
        )

        client = _new_client(network, anchor, empty_index, peers=[peer_a, peer_b])
        providers = client.find_providers("m")
        assert any(p.node_id == "bob" for p in providers)

    def test_peer_returning_wrong_message_type_is_skipped(
        self, network, anchor, empty_index
    ):
        # Peer returns ManifestResponse instead of ProvidersResponse —
        # we should skip it and continue.
        peer = FakePeer(node_id="confused", address="confused:1")
        wrong_response = ManifestResponse(
            request_id="x", manifest={"model_id": "m"}
        )
        network.register_handler(
            "confused:1", _make_handler_returning(wrong_response)
        )

        client = _new_client(network, anchor, empty_index, peers=[peer])
        providers = client.find_providers("m")
        assert providers == []

    def test_no_peers_returns_empty(self, network, anchor, empty_index):
        client = _new_client(network, anchor, empty_index, peers=[])
        assert client.find_providers("m") == []


# ──────────────────────────────────────────────────────────────────────────
# fetch_manifest — raw RPC, no verify
# ──────────────────────────────────────────────────────────────────────────


class TestFetchManifest:
    def test_fetch_happy_path(self, network, anchor, empty_index, alice):
        signed = _build_signed_manifest(alice)
        provider = ProviderInfo(node_id="bob", address="bob:1")
        network.register_handler(
            "bob:1",
            _make_handler_returning(
                ManifestResponse(request_id="x", manifest=signed.to_dict())
            ),
        )

        client = _new_client(network, anchor, empty_index)
        result = client.fetch_manifest(provider, "llama-3-8b")
        assert result == signed

    def test_fetch_not_found(self, network, anchor, empty_index):
        provider = ProviderInfo(node_id="bob", address="bob:1")
        network.register_handler(
            "bob:1",
            _make_handler_returning(
                ErrorResponse(
                    request_id="x", code="NOT_FOUND", message="don't have it"
                )
            ),
        )

        client = _new_client(network, anchor, empty_index)
        with pytest.raises(ManifestNotFoundError, match="not found"):
            client.fetch_manifest(provider, "missing-model")

    def test_fetch_other_error_is_transport_failure(
        self, network, anchor, empty_index
    ):
        provider = ProviderInfo(node_id="bob", address="bob:1")
        network.register_handler(
            "bob:1",
            _make_handler_returning(
                ErrorResponse(
                    request_id="x", code="INTERNAL_ERROR", message="kaboom"
                )
            ),
        )

        client = _new_client(network, anchor, empty_index)
        with pytest.raises(TransportFailureError, match="INTERNAL_ERROR"):
            client.fetch_manifest(provider, "m")

    def test_fetch_wrong_response_type(
        self, network, anchor, empty_index
    ):
        provider = ProviderInfo(node_id="bob", address="bob:1")
        network.register_handler(
            "bob:1",
            _make_handler_returning(
                ProvidersResponse(request_id="x", providers=())
            ),
        )

        client = _new_client(network, anchor, empty_index)
        with pytest.raises(TransportFailureError, match="ProvidersResponse"):
            client.fetch_manifest(provider, "m")

    def test_fetch_malformed_manifest_dict(
        self, network, anchor, empty_index
    ):
        provider = ProviderInfo(node_id="bob", address="bob:1")
        # ManifestResponse with a manifest dict that's not a valid
        # ModelManifest payload (missing required fields).
        network.register_handler(
            "bob:1",
            _make_handler_returning(
                ManifestResponse(
                    request_id="x", manifest={"only": "garbage"}
                )
            ),
        )

        client = _new_client(network, anchor, empty_index)
        with pytest.raises(ManifestNotFoundError, match="malformed"):
            client.fetch_manifest(provider, "m")

    def test_fetch_transport_raises(self, network, anchor, empty_index):
        # Provider's address has no handler → ConnectionRefusedError.
        provider = ProviderInfo(node_id="bob", address="dead:1")
        client = _new_client(network, anchor, empty_index)
        with pytest.raises(TransportFailureError, match="dead:1"):
            client.fetch_manifest(provider, "m")

    def test_response_request_id_must_match(
        self, network, anchor, empty_index
    ):
        # Build a handler that returns a hardcoded request_id, NOT the
        # one from the request — out-of-order / stale response.
        def stale_handler(request_bytes):
            stale = ManifestResponse(
                request_id="totally-different-request",
                manifest={"model_id": "m"},
            )
            return encode_message(stale)

        provider = ProviderInfo(node_id="bob", address="bob:1")
        network.register_handler("bob:1", stale_handler)

        client = _new_client(network, anchor, empty_index)
        with pytest.raises(TransportFailureError, match="request_id"):
            client.fetch_manifest(provider, "m")


# ──────────────────────────────────────────────────────────────────────────
# get_manifest — find + fetch + anchor verify
# ──────────────────────────────────────────────────────────────────────────


class TestGetManifest:
    def test_happy_path(self, network, anchor, empty_index, alice):
        signed = _build_signed_manifest(alice)
        # bob is a peer that lists himself as provider and serves the manifest.
        bob_peer = FakePeer(node_id="bob", address="bob:1")

        def bob_handler(request_bytes):
            request = parse_message(request_bytes)
            if isinstance(request, FindProvidersRequest):
                resp = ProvidersResponse(
                    request_id=request.request_id,
                    providers=(ProviderInfo(node_id="bob", address="bob:1"),),
                )
            elif isinstance(request, FetchManifestRequest):
                resp = ManifestResponse(
                    request_id=request.request_id,
                    manifest=signed.to_dict(),
                )
            else:
                raise AssertionError(f"unexpected request: {type(request).__name__}")
            return encode_message(resp)

        network.register_handler("bob:1", bob_handler)
        client = _new_client(network, anchor, empty_index, peers=[bob_peer])

        result = client.get_manifest("llama-3-8b")
        assert result == signed

    def test_no_providers_raises(self, network, anchor, empty_index):
        client = _new_client(network, anchor, empty_index, peers=[])
        with pytest.raises(ManifestNotFoundError, match="no providers"):
            client.get_manifest("m")

    def test_anchor_verify_fail_tries_next_provider(
        self, network, anchor, empty_index, alice, bob
    ):
        # bob is registered on anchor; alice is NOT. Build a manifest
        # signed by alice; serve it from bob's address — anchor lookup
        # for alice's node_id returns None → verify fails → drop.
        # Then a SECOND provider serves a manifest signed by bob;
        # anchor verify succeeds.
        anchor_with_bob_only = FakeAnchor(
            registrations={bob.node_id: bob.public_key_b64}
        )

        alice_signed = _build_signed_manifest(alice)
        bob_signed = _build_signed_manifest(bob)

        # peer_a serves alice-signed; peer_b serves bob-signed
        peer_a = FakePeer(node_id="peer-a", address="peer-a:1")
        peer_b = FakePeer(node_id="peer-b", address="peer-b:1")

        # Both peers list themselves as providers in find_providers.
        # Then on fetch, peer_a returns alice's manifest; peer_b returns
        # bob's manifest.
        def make_handler(signed_manifest, my_node_id, my_addr):
            def h(request_bytes):
                request = parse_message(request_bytes)
                if isinstance(request, FindProvidersRequest):
                    resp = ProvidersResponse(
                        request_id=request.request_id,
                        providers=(ProviderInfo(node_id=my_node_id, address=my_addr),),
                    )
                else:
                    resp = ManifestResponse(
                        request_id=request.request_id,
                        manifest=signed_manifest.to_dict(),
                    )
                return encode_message(resp)
            return h

        network.register_handler(
            "peer-a:1", make_handler(alice_signed, "peer-a", "peer-a:1")
        )
        network.register_handler(
            "peer-b:1", make_handler(bob_signed, "peer-b", "peer-b:1")
        )
        client = _new_client(
            network, anchor_with_bob_only, empty_index, peers=[peer_a, peer_b]
        )

        # find_providers returns BOTH peer-a and peer-b; client tries
        # peer-a first → anchor verify fails (alice not registered);
        # tries peer-b → anchor verify succeeds → return.
        result = client.get_manifest("llama-3-8b")
        assert result == bob_signed

    def test_all_providers_fail_verify_raises(
        self, network, anchor, empty_index, alice, bob
    ):
        # Anchor knows nobody. Every provider returns a manifest, but
        # NONE verify. Result: ManifestNotFoundError.
        empty_anchor = FakeAnchor(registrations={})
        signed = _build_signed_manifest(alice)
        peer = FakePeer(node_id="bob", address="bob:1")

        def h(request_bytes):
            request = parse_message(request_bytes)
            if isinstance(request, FindProvidersRequest):
                resp = ProvidersResponse(
                    request_id=request.request_id,
                    providers=(ProviderInfo(node_id="bob", address="bob:1"),),
                )
            else:
                resp = ManifestResponse(
                    request_id=request.request_id, manifest=signed.to_dict()
                )
            return encode_message(resp)

        network.register_handler("bob:1", h)
        client = _new_client(network, empty_anchor, empty_index, peers=[peer])

        with pytest.raises(ManifestNotFoundError, match="anchor verify|verified"):
            client.get_manifest("m")

    def test_self_provider_uses_local_fetch(
        self, network, anchor, empty_index, alice, tmp_path
    ):
        # If self is a provider, get_manifest reads from the local
        # index without making an RPC.
        signed = _build_signed_manifest(alice)
        model_dir = tmp_path / "llama-3-8b"
        model_dir.mkdir()
        manifest_path = model_dir / "manifest.json"
        import json
        manifest_path.write_text(json.dumps(signed.to_dict()))
        empty_index.register("llama-3-8b", manifest_path)

        # No peers in routing table; no handlers. If the client tried
        # to RPC, it would fail. Test passes iff local fetch path used.
        client = _new_client(network, anchor, empty_index, peers=[])
        result = client.get_manifest("llama-3-8b")
        assert result == signed

    def test_tampered_manifest_caught_by_anchor(
        self, network, anchor, empty_index, alice
    ):
        # Build a signed manifest, then tamper a field post-signing.
        # Anchor verify catches it → next provider tried → fails.
        signed = _build_signed_manifest(alice)
        tampered = dataclasses.replace(signed, model_name="EVIL")

        peer = FakePeer(node_id="bob", address="bob:1")

        def h(request_bytes):
            request = parse_message(request_bytes)
            if isinstance(request, FindProvidersRequest):
                resp = ProvidersResponse(
                    request_id=request.request_id,
                    providers=(ProviderInfo(node_id="bob", address="bob:1"),),
                )
            else:
                resp = ManifestResponse(
                    request_id=request.request_id, manifest=tampered.to_dict()
                )
            return encode_message(resp)

        network.register_handler("bob:1", h)
        client = _new_client(network, anchor, empty_index, peers=[peer])

        with pytest.raises(ManifestNotFoundError):
            client.get_manifest("llama-3-8b")


# ──────────────────────────────────────────────────────────────────────────
# Round 1 review remediations — HIGH-2 / HIGH-3
# ──────────────────────────────────────────────────────────────────────────


class TestSubstitutionDefense:
    """HIGH-2 from Phase 3.x.5 round 1 review: a malicious provider
    can return a *validly-anchor-signed* manifest under a DIFFERENT
    model_id than what the caller asked for. Anchor verify alone
    accepts it (signature is genuine). The DHT client must reject
    the substitution.
    """

    def test_validly_signed_wrong_model_id_rejected(
        self, network, anchor, empty_index, alice
    ):
        # Alice signs a manifest for "evil-model" with her real key.
        # A malicious provider serves these bytes in response to a
        # request for "target-model". The signature is genuine, but
        # the model_id mismatch must trigger rejection.
        evil_signed = _build_signed_manifest(alice, model_id="evil-model")
        peer = FakePeer(node_id="bob", address="bob:1")

        def h(request_bytes):
            request = parse_message(request_bytes)
            if isinstance(request, FindProvidersRequest):
                resp = ProvidersResponse(
                    request_id=request.request_id,
                    providers=(ProviderInfo(node_id="bob", address="bob:1"),),
                )
            else:
                # Server returns evil-model bytes to a target-model request
                resp = ManifestResponse(
                    request_id=request.request_id,
                    manifest=evil_signed.to_dict(),
                )
            return encode_message(resp)

        network.register_handler("bob:1", h)
        client = _new_client(network, anchor, empty_index, peers=[peer])

        with pytest.raises(ManifestNotFoundError, match="mismatched model_id"):
            client.get_manifest("target-model")

    def test_substitution_rejected_then_clean_provider_succeeds(
        self, network, anchor, empty_index, alice
    ):
        # Two providers: one substitutes, one serves correctly.
        # Client must reject the substitution and try the next, ending
        # with the legitimate manifest.
        target_signed = _build_signed_manifest(alice, model_id="target")
        evil_signed = _build_signed_manifest(alice, model_id="evil")

        peer_evil = FakePeer(node_id="evil-peer", address="evil:1")
        peer_clean = FakePeer(node_id="clean-peer", address="clean:1")

        def make_handler(node_id, address, payload):
            def h(request_bytes):
                request = parse_message(request_bytes)
                if isinstance(request, FindProvidersRequest):
                    resp = ProvidersResponse(
                        request_id=request.request_id,
                        providers=(ProviderInfo(node_id=node_id, address=address),),
                    )
                else:
                    resp = ManifestResponse(
                        request_id=request.request_id,
                        manifest=payload.to_dict(),
                    )
                return encode_message(resp)
            return h

        network.register_handler(
            "evil:1", make_handler("evil-peer", "evil:1", evil_signed)
        )
        network.register_handler(
            "clean:1", make_handler("clean-peer", "clean:1", target_signed)
        )
        client = _new_client(
            network, anchor, empty_index,
            peers=[peer_evil, peer_clean],
        )

        out = client.get_manifest("target")
        assert out.model_id == "target"


class TestAnchorRPCErrorHandling:
    """HIGH-3 from Phase 3.x.5 round 1 review: AnchorRPCError raised
    by the verifier during one provider's check must not abort the
    whole get_manifest. Treat as transient; continue to next provider.
    """

    def test_transient_anchor_rpc_failure_continues_to_next_provider(
        self, network, empty_index, alice, monkeypatch
    ):
        from prsm.security.publisher_key_anchor.exceptions import (
            AnchorRPCError,
        )
        from prsm.network.manifest_dht import dht_client as _dc

        signed = _build_signed_manifest(alice, model_id="theta")

        peer_flaky = FakePeer(node_id="flaky", address="flaky:1")
        peer_good = FakePeer(node_id="good", address="good:1")

        def make_handler(node_id, address):
            def h(request_bytes):
                request = parse_message(request_bytes)
                if isinstance(request, FindProvidersRequest):
                    resp = ProvidersResponse(
                        request_id=request.request_id,
                        providers=(ProviderInfo(node_id=node_id, address=address),),
                    )
                else:
                    resp = ManifestResponse(
                        request_id=request.request_id,
                        manifest=signed.to_dict(),
                    )
                return encode_message(resp)
            return h

        network.register_handler(
            "flaky:1", make_handler("flaky", "flaky:1")
        )
        network.register_handler(
            "good:1", make_handler("good", "good:1")
        )

        # Patch the verifier so the FIRST call raises AnchorRPCError
        # (simulating a transient anchor RPC blip during flaky's
        # verify) and subsequent calls succeed.
        call_count = {"n": 0}
        from prsm.security.publisher_key_anchor import verifiers as _ver

        def fake_verify(manifest, anchor):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise AnchorRPCError("transient blip")
            return True

        monkeypatch.setattr(
            _ver, "verify_manifest_with_anchor", fake_verify
        )

        anchor = type("A", (), {"lookup": staticmethod(lambda nid: "x")})()
        client = ManifestDHTClient(
            local_index=empty_index,
            routing_table=FakeRoutingTable([peer_flaky, peer_good]),
            send_message=network.send,
            anchor=anchor,
            my_node_id="self",
            my_address="self:0",
        )

        out = client.get_manifest("theta")
        assert out.model_id == "theta"
        # Confirms we tried both providers' verifiers
        assert call_count["n"] == 2

    def test_all_providers_anchor_rpc_failures_yields_not_found(
        self, network, empty_index, alice, monkeypatch
    ):
        from prsm.security.publisher_key_anchor.exceptions import (
            AnchorRPCError,
        )
        from prsm.security.publisher_key_anchor import verifiers as _ver

        signed = _build_signed_manifest(alice, model_id="iota")

        peer = FakePeer(node_id="p", address="p:1")

        def h(request_bytes):
            request = parse_message(request_bytes)
            if isinstance(request, FindProvidersRequest):
                resp = ProvidersResponse(
                    request_id=request.request_id,
                    providers=(ProviderInfo(node_id="p", address="p:1"),),
                )
            else:
                resp = ManifestResponse(
                    request_id=request.request_id,
                    manifest=signed.to_dict(),
                )
            return encode_message(resp)

        network.register_handler("p:1", h)

        # Every verify raises — no provider verifies; not-found.
        def always_raises(*args, **kwargs):
            raise AnchorRPCError("anchor down")

        monkeypatch.setattr(
            _ver, "verify_manifest_with_anchor", always_raises
        )

        anchor = type("A", (), {"lookup": staticmethod(lambda nid: "x")})()
        client = ManifestDHTClient(
            local_index=empty_index,
            routing_table=FakeRoutingTable([peer]),
            send_message=network.send,
            anchor=anchor,
            my_node_id="self",
            my_address="self:0",
        )

        with pytest.raises(ManifestNotFoundError):
            client.get_manifest("iota")
