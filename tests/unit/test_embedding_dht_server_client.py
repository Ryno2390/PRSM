"""PRSM-PROV-1 Item 3 Tasks 3+4 — EmbeddingDHTServer + EmbeddingDHTClient.

Paired tests covering the full request/reply round-trip:
- find_providers: the in-process server is wired to a fake routing
  table; the client asks for providers; the server replies with self
  (when it has the embedding) plus closest peers from the routing
  table. Verifies the dedup, the +k cap, and the dropping of malformed
  peers.
- fetch_embedding: a real Ed25519 keypair signs a real vector, the
  server stores it, the client fetches and verifies. Then poisoning
  scenarios: tampered vector, wrong creator pubkey, signature for a
  different content_hash, server returns a different content_hash than
  asked for, etc. — all must raise SignatureVerificationError.
"""
from __future__ import annotations

import base64
import struct
from typing import List, Optional

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
)

from prsm.network.embedding_dht.dht_client import (
    EmbeddingDHTClient,
    EmbeddingNotFoundError,
    SignatureVerificationError,
    TransportFailureError,
)
from prsm.network.embedding_dht.dht_server import (
    EmbeddingDHTServer,
)
from prsm.network.embedding_dht.local_index import (
    LocalEmbeddingIndex,
    LocalEmbeddingRecord,
)
from prsm.network.embedding_dht.protocol import (
    EmbeddingResponse,
    ErrorCode,
    ErrorResponse,
    FindEmbeddingRequest,
    FetchEmbeddingRequest,
    ProviderInfo,
    canonical_signing_payload,
    encode_message,
    parse_message,
)


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class FakePeer:
    """Minimal shape ``find_closest_peers`` returns: .node_id + .address."""

    def __init__(self, node_id: str, address: str) -> None:
        self.node_id = node_id
        self.address = address


class FakeRoutingTable:
    def __init__(self, peers: List[FakePeer]) -> None:
        self._peers = peers
        self.last_target: Optional[str] = None
        self.last_count: Optional[int] = None

    def find_closest_peers(self, target_id: str, count: int = 20):
        self.last_target = target_id
        self.last_count = count
        return list(self._peers[:count])


def _real_verify_signature(
    pubkey_bytes: bytes, message: bytes, signature: bytes,
) -> bool:
    """Real Ed25519 verification — the function shape the production
    EmbeddingDHTClient expects."""
    try:
        Ed25519PublicKey.from_public_bytes(pubkey_bytes).verify(
            signature, message,
        )
    except InvalidSignature:
        return False
    return True


def _make_signed_record(
    *,
    content_hash: str = "0xabc123",
    model_id: str = "openai/text-embedding-ada-002",
    dimension: int = 8,
    creator_id: str = "creator-A",
    created_at: float = 1715000000.0,
) -> tuple[LocalEmbeddingRecord, bytes]:
    """Build a record signed by a freshly-generated Ed25519 key.
    Returns (record, creator_pubkey_bytes)."""
    privkey = Ed25519PrivateKey.generate()
    pubkey_bytes = privkey.public_key().public_bytes(
        encoding=Encoding.Raw, format=PublicFormat.Raw,
    )

    raw_vec = struct.pack(
        f"<{dimension}f", *[0.1 * (i + 1) for i in range(dimension)],
    )
    payload = canonical_signing_payload(
        content_hash=content_hash,
        model_id=model_id,
        dimension=dimension,
        dtype="float32",
        vector_bytes=raw_vec,
        created_at=created_at,
    )
    sig = privkey.sign(payload)
    rec = LocalEmbeddingRecord(
        content_hash=content_hash,
        model_id=model_id,
        dimension=dimension,
        dtype="float32",
        vector_b64=base64.b64encode(raw_vec).decode("ascii"),
        creator_id=creator_id,
        created_at=created_at,
        signature_b64=base64.b64encode(sig).decode("ascii"),
    )
    return rec, pubkey_bytes


# ---------------------------------------------------------------------------
# EmbeddingDHTServer — find_providers
# ---------------------------------------------------------------------------


def test_server_find_returns_self_when_we_have_embedding(tmp_path):
    idx = LocalEmbeddingIndex(tmp_path)
    rec, _pk = _make_signed_record()
    idx.register(rec)

    server = EmbeddingDHTServer(
        local_index=idx,
        routing_table=FakeRoutingTable(peers=[]),
        my_node_id="node-self",
        my_address="127.0.0.1:9000",
    )
    request = FindEmbeddingRequest(
        content_hash=rec.content_hash,
        model_id=rec.model_id,
        request_id="r1",
    )
    resp_bytes = server.handle(encode_message(request))
    resp = parse_message(resp_bytes)
    assert len(resp.providers) == 1
    assert resp.providers[0].node_id == "node-self"
    assert resp.providers[0].address == "127.0.0.1:9000"


def test_server_find_returns_closest_peers_when_we_dont_have_it(tmp_path):
    idx = LocalEmbeddingIndex(tmp_path)
    server = EmbeddingDHTServer(
        local_index=idx,
        routing_table=FakeRoutingTable(peers=[
            FakePeer("peer-A", "10.0.0.1:9000"),
            FakePeer("peer-B", "10.0.0.2:9000"),
        ]),
        my_node_id="node-self",
        my_address="127.0.0.1:9000",
    )
    request = FindEmbeddingRequest(
        content_hash="0xnope", model_id="m", request_id="r1",
    )
    resp = parse_message(server.handle(encode_message(request)))
    nodes = sorted(p.node_id for p in resp.providers)
    assert nodes == ["peer-A", "peer-B"]


def test_server_find_dedupes_self_against_routing_table(tmp_path):
    idx = LocalEmbeddingIndex(tmp_path)
    rec, _pk = _make_signed_record()
    idx.register(rec)
    server = EmbeddingDHTServer(
        local_index=idx,
        # Routing table claims node-self is also a closest peer (e.g.
        # gossip echo) — must not double-count.
        routing_table=FakeRoutingTable(peers=[
            FakePeer("node-self", "127.0.0.1:9000"),
            FakePeer("peer-A", "10.0.0.1:9000"),
        ]),
        my_node_id="node-self",
        my_address="127.0.0.1:9000",
    )
    resp = parse_message(server.handle(encode_message(FindEmbeddingRequest(
        content_hash=rec.content_hash, model_id=rec.model_id, request_id="r",
    ))))
    nodes = sorted(p.node_id for p in resp.providers)
    assert nodes == ["node-self", "peer-A"]


def test_server_find_drops_malformed_peers(tmp_path):
    idx = LocalEmbeddingIndex(tmp_path)
    server = EmbeddingDHTServer(
        local_index=idx,
        routing_table=FakeRoutingTable(peers=[
            FakePeer("good-peer", "10.0.0.1:9000"),
            FakePeer("bad-peer", "no-port-here"),  # no port
            FakePeer("", "10.0.0.2:9000"),  # empty node_id
        ]),
        my_node_id="node-self",
        my_address="127.0.0.1:9000",
    )
    resp = parse_message(server.handle(encode_message(FindEmbeddingRequest(
        content_hash="0xabc", model_id="m", request_id="r",
    ))))
    nodes = [p.node_id for p in resp.providers]
    assert nodes == ["good-peer"]


# ---------------------------------------------------------------------------
# EmbeddingDHTServer — fetch_embedding
# ---------------------------------------------------------------------------


def test_server_fetch_returns_stored_record(tmp_path):
    idx = LocalEmbeddingIndex(tmp_path)
    rec, _pk = _make_signed_record()
    idx.register(rec)
    server = EmbeddingDHTServer(
        local_index=idx,
        routing_table=FakeRoutingTable(peers=[]),
        my_node_id="node-self",
        my_address="127.0.0.1:9000",
    )
    resp = parse_message(server.handle(encode_message(FetchEmbeddingRequest(
        content_hash=rec.content_hash, model_id=rec.model_id, request_id="r",
    ))))
    assert isinstance(resp, EmbeddingResponse)
    assert resp.content_hash == rec.content_hash
    assert resp.signature_b64 == rec.signature_b64
    assert resp.vector_b64 == rec.vector_b64
    assert resp.creator_id == rec.creator_id


def test_server_fetch_returns_not_found_when_missing(tmp_path):
    idx = LocalEmbeddingIndex(tmp_path)
    server = EmbeddingDHTServer(
        local_index=idx,
        routing_table=FakeRoutingTable(peers=[]),
        my_node_id="node-self",
        my_address="127.0.0.1:9000",
    )
    resp = parse_message(server.handle(encode_message(FetchEmbeddingRequest(
        content_hash="0xnope", model_id="m", request_id="r",
    ))))
    assert isinstance(resp, ErrorResponse)
    assert resp.code == ErrorCode.NOT_FOUND.value


def test_server_handle_returns_malformed_on_garbage_bytes(tmp_path):
    idx = LocalEmbeddingIndex(tmp_path)
    server = EmbeddingDHTServer(
        local_index=idx,
        routing_table=FakeRoutingTable(peers=[]),
        my_node_id="node-self",
        my_address="127.0.0.1:9000",
    )
    resp = parse_message(server.handle(b"not json at all"))
    assert isinstance(resp, ErrorResponse)
    assert resp.code == ErrorCode.MALFORMED_REQUEST.value


def test_server_handle_returns_malformed_on_response_shape(tmp_path):
    """Sending a response message TO the server (instead of a request)
    is misuse and must surface as MALFORMED_REQUEST."""
    idx = LocalEmbeddingIndex(tmp_path)
    server = EmbeddingDHTServer(
        local_index=idx,
        routing_table=FakeRoutingTable(peers=[]),
        my_node_id="node-self",
        my_address="127.0.0.1:9000",
    )
    misuse = ErrorResponse(
        request_id="x", code="any", message="any",
    )
    resp = parse_message(server.handle(encode_message(misuse)))
    assert isinstance(resp, ErrorResponse)
    assert resp.code == ErrorCode.MALFORMED_REQUEST.value


# ---------------------------------------------------------------------------
# EmbeddingDHTClient — constructor
# ---------------------------------------------------------------------------


def test_client_refuses_no_verifier():
    with pytest.raises(RuntimeError, match="verifier"):
        EmbeddingDHTClient(
            routing_table=FakeRoutingTable(peers=[]),
            send_message=lambda addr, b: b"",
            creator_pubkey_for=None,  # type: ignore[arg-type]
            verify_signature=lambda *a: True,  # type: ignore[arg-type]
            my_node_id="node-self",
            my_address="127.0.0.1:9000",
        )


def test_client_refuses_no_verify_callable():
    with pytest.raises(RuntimeError, match="verifier"):
        EmbeddingDHTClient(
            routing_table=FakeRoutingTable(peers=[]),
            send_message=lambda addr, b: b"",
            creator_pubkey_for=lambda h: None,
            verify_signature=None,  # type: ignore[arg-type]
            my_node_id="node-self",
            my_address="127.0.0.1:9000",
        )


# ---------------------------------------------------------------------------
# Server + Client round-trip
# ---------------------------------------------------------------------------


def _make_paired_server_client(tmp_path, peers=None):
    idx = LocalEmbeddingIndex(tmp_path)
    server = EmbeddingDHTServer(
        local_index=idx,
        routing_table=FakeRoutingTable(peers=peers or []),
        my_node_id="server-node",
        my_address="server:9000",
    )

    # In-process transport: route bytes back to server.handle.
    def send(address, request_bytes):
        assert address == "server:9000"
        return server.handle(request_bytes)

    return idx, server, send


def test_round_trip_find_then_fetch_with_real_signature(tmp_path):
    idx, _server, send = _make_paired_server_client(tmp_path)
    rec, pubkey = _make_signed_record()
    idx.register(rec)

    client = EmbeddingDHTClient(
        routing_table=FakeRoutingTable(peers=[
            FakePeer("server-node", "server:9000"),
        ]),
        send_message=send,
        creator_pubkey_for=lambda h: pubkey if h == rec.content_hash else None,
        verify_signature=_real_verify_signature,
        my_node_id="client-node",
        my_address="client:9000",
    )

    providers = client.find_providers(rec.content_hash, rec.model_id)
    assert len(providers) == 1
    assert providers[0].node_id == "server-node"

    fetched = client.fetch_embedding(
        providers[0], rec.content_hash, rec.model_id,
    )
    assert fetched.vector_b64 == rec.vector_b64
    assert fetched.signature_b64 == rec.signature_b64


def test_round_trip_not_found_raises_named_exception(tmp_path):
    idx, _server, send = _make_paired_server_client(tmp_path)
    # Note: deliberately NOT registering the embedding.
    _rec, pubkey = _make_signed_record()

    client = EmbeddingDHTClient(
        routing_table=FakeRoutingTable(peers=[]),
        send_message=send,
        creator_pubkey_for=lambda h: pubkey,
        verify_signature=_real_verify_signature,
        my_node_id="client-node",
        my_address="client:9000",
    )

    with pytest.raises(EmbeddingNotFoundError):
        client.fetch_embedding(
            ProviderInfo(node_id="server-node", address="server:9000"),
            "0xnope", "m",
        )


def test_round_trip_transport_failure_propagates(tmp_path):
    def broken_send(address, request_bytes):
        raise ConnectionError("boom")

    _rec, pubkey = _make_signed_record()
    client = EmbeddingDHTClient(
        routing_table=FakeRoutingTable(peers=[]),
        send_message=broken_send,
        creator_pubkey_for=lambda h: pubkey,
        verify_signature=_real_verify_signature,
        my_node_id="client-node",
        my_address="client:9000",
    )

    with pytest.raises(TransportFailureError, match="boom"):
        client.fetch_embedding(
            ProviderInfo(node_id="server-node", address="server:9000"),
            "0xabc", "m",
        )


# ---------------------------------------------------------------------------
# Poisoning defenses (the whole point of the signature)
# ---------------------------------------------------------------------------


def test_poisoning_tampered_vector_rejected(tmp_path):
    """Server stores valid record, then a malicious server returns a
    response with a different vector but the original signature.
    Client must reject."""
    idx, server, _send = _make_paired_server_client(tmp_path)
    rec, pubkey = _make_signed_record()
    idx.register(rec)

    # Build a tampered response with a different vector
    tampered_vec = struct.pack(f"<{rec.dimension}f", *([99.99] * rec.dimension))
    tampered_resp = EmbeddingResponse(
        request_id="will-be-overwritten",
        content_hash=rec.content_hash,
        model_id=rec.model_id,
        dimension=rec.dimension,
        dtype=rec.dtype,
        vector_b64=base64.b64encode(tampered_vec).decode("ascii"),
        creator_id=rec.creator_id,
        created_at=rec.created_at,
        signature_b64=rec.signature_b64,  # original signature, wrong vector
    )

    def malicious_send(address, request_bytes):
        # Always return the tampered response, ignoring the request.
        return encode_message(EmbeddingResponse(
            request_id="r",  # request_id matching not enforced at this layer
            content_hash=tampered_resp.content_hash,
            model_id=tampered_resp.model_id,
            dimension=tampered_resp.dimension,
            dtype=tampered_resp.dtype,
            vector_b64=tampered_resp.vector_b64,
            creator_id=tampered_resp.creator_id,
            created_at=tampered_resp.created_at,
            signature_b64=tampered_resp.signature_b64,
        ))

    client = EmbeddingDHTClient(
        routing_table=FakeRoutingTable(peers=[]),
        send_message=malicious_send,
        creator_pubkey_for=lambda h: pubkey,
        verify_signature=_real_verify_signature,
        my_node_id="client", my_address="client:9000",
    )

    with pytest.raises(SignatureVerificationError, match="FAILED"):
        client.fetch_embedding(
            ProviderInfo(node_id="m", address="m:9000"),
            rec.content_hash, rec.model_id,
        )


def test_poisoning_no_anchor_rejected(tmp_path):
    """Creator pubkey lookup returns None — refuse to trust unanchored
    embedding."""
    idx, _server, send = _make_paired_server_client(tmp_path)
    rec, _pk = _make_signed_record()
    idx.register(rec)

    client = EmbeddingDHTClient(
        routing_table=FakeRoutingTable(peers=[]),
        send_message=send,
        creator_pubkey_for=lambda h: None,  # no anchor
        verify_signature=_real_verify_signature,
        my_node_id="client", my_address="client:9000",
    )

    with pytest.raises(SignatureVerificationError, match="no on-chain"):
        client.fetch_embedding(
            ProviderInfo(node_id="server-node", address="server:9000"),
            rec.content_hash, rec.model_id,
        )


def test_poisoning_wrong_pubkey_rejected(tmp_path):
    """Creator pubkey lookup returns a DIFFERENT key than the one that
    signed — rejection."""
    idx, _server, send = _make_paired_server_client(tmp_path)
    rec, _real_pk = _make_signed_record()
    idx.register(rec)

    # Generate an unrelated key as the on-chain anchor
    other_pubkey = Ed25519PrivateKey.generate().public_key().public_bytes(
        encoding=Encoding.Raw, format=PublicFormat.Raw,
    )

    client = EmbeddingDHTClient(
        routing_table=FakeRoutingTable(peers=[]),
        send_message=send,
        creator_pubkey_for=lambda h: other_pubkey,
        verify_signature=_real_verify_signature,
        my_node_id="client", my_address="client:9000",
    )

    with pytest.raises(SignatureVerificationError, match="FAILED"):
        client.fetch_embedding(
            ProviderInfo(node_id="server-node", address="server:9000"),
            rec.content_hash, rec.model_id,
        )


def test_poisoning_server_swaps_content_hash_rejected(tmp_path):
    """Server returns a response whose content_hash != requested one.
    A signature might still verify against the swapped hash, but the
    client must refuse a swap."""
    rec, pubkey = _make_signed_record(content_hash="0xRESPONSE-HASH")
    # Don't register — we'll fake the server response directly.

    def swap_server(address, request_bytes):
        return encode_message(EmbeddingResponse(
            request_id="r",
            content_hash=rec.content_hash,  # what server returns
            model_id=rec.model_id,
            dimension=rec.dimension,
            dtype=rec.dtype,
            vector_b64=rec.vector_b64,
            creator_id=rec.creator_id,
            created_at=rec.created_at,
            signature_b64=rec.signature_b64,
        ))

    client = EmbeddingDHTClient(
        routing_table=FakeRoutingTable(peers=[]),
        send_message=swap_server,
        creator_pubkey_for=lambda h: pubkey,
        verify_signature=_real_verify_signature,
        my_node_id="client", my_address="client:9000",
    )

    with pytest.raises(SignatureVerificationError, match="content_hash"):
        client.fetch_embedding(
            ProviderInfo(node_id="m", address="m:9000"),
            "0xREQUESTED-HASH", rec.model_id,
        )


def test_poisoning_server_swaps_model_id_rejected(tmp_path):
    """Same swap defense for model_id."""
    rec, pubkey = _make_signed_record(model_id="m-actual")

    def swap_server(address, request_bytes):
        return encode_message(EmbeddingResponse(
            request_id="r",
            content_hash=rec.content_hash,
            model_id=rec.model_id,
            dimension=rec.dimension,
            dtype=rec.dtype,
            vector_b64=rec.vector_b64,
            creator_id=rec.creator_id,
            created_at=rec.created_at,
            signature_b64=rec.signature_b64,
        ))

    client = EmbeddingDHTClient(
        routing_table=FakeRoutingTable(peers=[]),
        send_message=swap_server,
        creator_pubkey_for=lambda h: pubkey,
        verify_signature=_real_verify_signature,
        my_node_id="client", my_address="client:9000",
    )

    with pytest.raises(SignatureVerificationError, match="model_id"):
        client.fetch_embedding(
            ProviderInfo(node_id="m", address="m:9000"),
            rec.content_hash, "m-asked-for",
        )


# ---------------------------------------------------------------------------
# find_providers — multi-peer dedup
# ---------------------------------------------------------------------------


def test_find_providers_dedupes_across_peers(tmp_path):
    """Two peers each return overlapping provider lists; the client
    deduplicates by node_id."""
    (tmp_path / "a").mkdir()
    idx_a = LocalEmbeddingIndex(tmp_path / "a")
    server_a = EmbeddingDHTServer(
        local_index=idx_a,
        routing_table=FakeRoutingTable(peers=[
            FakePeer("X", "10.0.0.1:9000"),
            FakePeer("Y", "10.0.0.2:9000"),
        ]),
        my_node_id="A", my_address="A:9000",
    )
    (tmp_path / "b").mkdir()
    idx_b = LocalEmbeddingIndex(tmp_path / "b")
    server_b = EmbeddingDHTServer(
        local_index=idx_b,
        routing_table=FakeRoutingTable(peers=[
            FakePeer("Y", "10.0.0.2:9000"),  # overlaps with A's list
            FakePeer("Z", "10.0.0.3:9000"),
        ]),
        my_node_id="B", my_address="B:9000",
    )

    def send(addr, b):
        if addr == "A:9000":
            return server_a.handle(b)
        if addr == "B:9000":
            return server_b.handle(b)
        raise AssertionError(f"unexpected addr {addr}")

    _rec, pubkey = _make_signed_record()
    client = EmbeddingDHTClient(
        routing_table=FakeRoutingTable(peers=[
            FakePeer("A", "A:9000"),
            FakePeer("B", "B:9000"),
        ]),
        send_message=send,
        creator_pubkey_for=lambda h: pubkey,
        verify_signature=_real_verify_signature,
        my_node_id="client", my_address="client:9000",
    )

    providers = client.find_providers("0xabc", "m")
    nodes = sorted(p.node_id for p in providers)
    # X from A; Y from both A and B (deduped); Z from B.
    assert nodes == ["X", "Y", "Z"]
