"""
Unit tests — Phase 3.x.5 Task 4 — ManifestDHTServer.

Acceptance per design plan §4 Task 4: each message type → correct
response shape; unknown type / malformed input → ErrorResponse;
concurrent requests don't corrupt state; the entry point NEVER raises.

Real protocol encode/decode — server is constructed against fake
LocalManifestIndex + RoutingTable fixtures (same shapes as the
client tests use).
"""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from typing import List

import pytest

from prsm.network.manifest_dht import (
    DEFAULT_K,
    LocalManifestIndex,
)
from prsm.network.manifest_dht.dht_server import (
    UNKNOWN_REQUEST_ID,
    ManifestDHTServer,
)
from prsm.network.manifest_dht.protocol import (
    DHT_PROTOCOL_VERSION,
    ErrorCode,
    ErrorResponse,
    FetchManifestRequest,
    FindProvidersRequest,
    ManifestResponse,
    MessageType,
    ProviderInfo,
    ProvidersResponse,
    encode_message,
    parse_message,
)


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class FakePeer:
    node_id: str
    address: str


class FakeRoutingTable:
    """Returns a fixed list of peers regardless of target_id."""

    def __init__(self, peers: List[FakePeer]):
        self._peers = list(peers)

    def find_closest_peers(self, target_id: str, count: int = 20):
        return list(self._peers[:count])


class RaisingRoutingTable:
    """Routing table that raises on lookup. Used to test the
    INTERNAL_ERROR fallback path on routing-table failures."""

    def find_closest_peers(self, target_id: str, count: int = 20):
        raise RuntimeError("kademlia exploded")


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _seed_manifest(root: Path, model_id: str, payload: dict) -> Path:
    """Create <root>/<model_id>/manifest.json with ``payload``.
    Returns the manifest path for direct use with index.register()."""
    model_dir = root / model_id
    model_dir.mkdir()
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload))
    return manifest_path


def _new_server(
    local_index: LocalManifestIndex,
    routing_table=None,
    *,
    my_node_id: str = "node-self",
    my_address: str = "self:8765",
    k: int = DEFAULT_K,
) -> ManifestDHTServer:
    return ManifestDHTServer(
        local_index=local_index,
        routing_table=routing_table or FakeRoutingTable([]),
        my_node_id=my_node_id,
        my_address=my_address,
        k=k,
    )


def _decode(response_bytes: bytes):
    """Convenience: parse server response bytes."""
    return parse_message(response_bytes)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def empty_index(tmp_path) -> LocalManifestIndex:
    return LocalManifestIndex(tmp_path)


# ──────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_invalid_my_node_id(self, empty_index):
        with pytest.raises(ValueError, match="my_node_id"):
            ManifestDHTServer(
                local_index=empty_index,
                routing_table=FakeRoutingTable([]),
                my_node_id="",
                my_address="self:8765",
            )

    def test_invalid_my_address(self, empty_index):
        with pytest.raises(ValueError, match="host:port"):
            ManifestDHTServer(
                local_index=empty_index,
                routing_table=FakeRoutingTable([]),
                my_node_id="self",
                my_address="no-port",
            )

    def test_construction_succeeds(self, empty_index):
        server = _new_server(empty_index)
        assert isinstance(server, ManifestDHTServer)


# ──────────────────────────────────────────────────────────────────────────
# find_providers handler
# ──────────────────────────────────────────────────────────────────────────


class TestFindProvidersHandler:
    def test_returns_self_when_local_has_model(self, empty_index, tmp_path):
        manifest_path = _seed_manifest(tmp_path, "alpha", {"k": "v"})
        empty_index.register("alpha", manifest_path)

        server = _new_server(empty_index)
        request = FindProvidersRequest(model_id="alpha", request_id="rid-1")
        response = _decode(server.handle(encode_message(request)))

        assert isinstance(response, ProvidersResponse)
        assert response.request_id == "rid-1"
        assert len(response.providers) == 1
        assert response.providers[0].node_id == "node-self"
        assert response.providers[0].address == "self:8765"

    def test_returns_closest_peers_when_local_missing(self, empty_index):
        peers = [
            FakePeer(node_id="peer-a", address="hosta:1"),
            FakePeer(node_id="peer-b", address="hostb:2"),
        ]
        server = _new_server(
            empty_index, routing_table=FakeRoutingTable(peers)
        )
        request = FindProvidersRequest(model_id="absent", request_id="rid-2")
        response = _decode(server.handle(encode_message(request)))

        assert isinstance(response, ProvidersResponse)
        node_ids = [p.node_id for p in response.providers]
        assert node_ids == ["peer-a", "peer-b"]

    def test_returns_self_plus_peers_when_local_has_model(
        self, empty_index, tmp_path
    ):
        manifest_path = _seed_manifest(tmp_path, "beta", {"k": "v"})
        empty_index.register("beta", manifest_path)

        peers = [FakePeer(node_id="peer-a", address="hosta:1")]
        server = _new_server(
            empty_index, routing_table=FakeRoutingTable(peers)
        )
        request = FindProvidersRequest(model_id="beta", request_id="rid-3")
        response = _decode(server.handle(encode_message(request)))

        assert isinstance(response, ProvidersResponse)
        node_ids = [p.node_id for p in response.providers]
        assert node_ids == ["node-self", "peer-a"]

    def test_excludes_self_from_routing_table(self, empty_index):
        # If routing table includes self (some Kademlia impls do),
        # the server must dedupe rather than serving self twice.
        peers = [
            FakePeer(node_id="node-self", address="self:8765"),
            FakePeer(node_id="peer-a", address="hosta:1"),
        ]
        server = _new_server(
            empty_index, routing_table=FakeRoutingTable(peers)
        )
        request = FindProvidersRequest(model_id="absent", request_id="rid-4")
        response = _decode(server.handle(encode_message(request)))

        node_ids = [p.node_id for p in response.providers]
        assert "node-self" not in node_ids
        assert node_ids == ["peer-a"]

    def test_dedupes_self_when_routing_includes_self(
        self, empty_index, tmp_path
    ):
        # Local has the model AND routing table includes self → still
        # only one self-entry in the response.
        manifest_path = _seed_manifest(tmp_path, "gamma", {"k": "v"})
        empty_index.register("gamma", manifest_path)
        peers = [FakePeer(node_id="node-self", address="self:8765")]
        server = _new_server(
            empty_index, routing_table=FakeRoutingTable(peers)
        )
        request = FindProvidersRequest(model_id="gamma", request_id="rid-5")
        response = _decode(server.handle(encode_message(request)))

        assert len(response.providers) == 1
        assert response.providers[0].node_id == "node-self"

    def test_skips_malformed_peer_entries(self, empty_index):
        # Routing tables with bad entries shouldn't fail the request —
        # malformed peers are dropped, the rest are served.
        peers = [
            FakePeer(node_id="", address="hosta:1"),  # empty node_id
            FakePeer(node_id="peer-b", address=""),  # empty address
            FakePeer(node_id="peer-c", address="no-port"),  # no colon
            FakePeer(node_id="peer-d", address="hostd:4"),  # valid
        ]
        server = _new_server(
            empty_index, routing_table=FakeRoutingTable(peers), k=10
        )
        request = FindProvidersRequest(
            model_id="absent", request_id="rid-6"
        )
        response = _decode(server.handle(encode_message(request)))

        node_ids = [p.node_id for p in response.providers]
        assert node_ids == ["peer-d"]

    def test_empty_when_no_local_no_peers(self, empty_index):
        server = _new_server(empty_index)
        request = FindProvidersRequest(model_id="absent", request_id="rid-7")
        response = _decode(server.handle(encode_message(request)))

        assert isinstance(response, ProvidersResponse)
        assert response.providers == ()

    def test_routing_table_failure_returns_internal_error(self, empty_index):
        server = _new_server(
            empty_index, routing_table=RaisingRoutingTable()
        )
        request = FindProvidersRequest(model_id="absent", request_id="rid-8")
        response = _decode(server.handle(encode_message(request)))

        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.INTERNAL_ERROR.value
        assert response.request_id == "rid-8"

    def test_respects_k_count(self, empty_index):
        # K=2: server should pass count=2 to find_closest_peers.
        peers = [
            FakePeer(node_id=f"peer-{i}", address=f"host{i}:1")
            for i in range(5)
        ]
        server = _new_server(
            empty_index, routing_table=FakeRoutingTable(peers), k=2
        )
        request = FindProvidersRequest(model_id="absent", request_id="rid-9")
        response = _decode(server.handle(encode_message(request)))
        assert len(response.providers) == 2


# ──────────────────────────────────────────────────────────────────────────
# fetch_manifest handler
# ──────────────────────────────────────────────────────────────────────────


class TestFetchManifestHandler:
    def test_returns_manifest_when_present(self, empty_index, tmp_path):
        payload = {"model_id": "alpha", "shards": []}
        manifest_path = _seed_manifest(tmp_path, "alpha", payload)
        empty_index.register("alpha", manifest_path)

        server = _new_server(empty_index)
        request = FetchManifestRequest(model_id="alpha", request_id="frid-1")
        response = _decode(server.handle(encode_message(request)))

        assert isinstance(response, ManifestResponse)
        assert response.request_id == "frid-1"
        assert response.manifest == payload

    def test_not_found_when_absent(self, empty_index):
        server = _new_server(empty_index)
        request = FetchManifestRequest(
            model_id="missing", request_id="frid-2"
        )
        response = _decode(server.handle(encode_message(request)))

        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.NOT_FOUND.value
        assert response.request_id == "frid-2"

    def test_internal_error_when_file_unreadable(self, empty_index, tmp_path):
        # Index claims to serve the model, but the file was deleted out
        # from under us. The handler must return INTERNAL_ERROR rather
        # than letting OSError escape.
        manifest_path = _seed_manifest(tmp_path, "alpha", {"k": "v"})
        empty_index.register("alpha", manifest_path)
        manifest_path.unlink()

        server = _new_server(empty_index)
        request = FetchManifestRequest(model_id="alpha", request_id="frid-3")
        response = _decode(server.handle(encode_message(request)))

        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.INTERNAL_ERROR.value

    def test_internal_error_when_file_not_json(self, empty_index, tmp_path):
        model_dir = tmp_path / "alpha"
        model_dir.mkdir()
        manifest_path = model_dir / "manifest.json"
        manifest_path.write_text("not valid json {")
        empty_index.register("alpha", manifest_path)

        server = _new_server(empty_index)
        request = FetchManifestRequest(model_id="alpha", request_id="frid-4")
        response = _decode(server.handle(encode_message(request)))

        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.INTERNAL_ERROR.value

    def test_internal_error_when_json_not_dict(self, empty_index, tmp_path):
        # JSON parses to a list — server must not let that propagate
        # to the client as a malformed ManifestResponse.
        model_dir = tmp_path / "alpha"
        model_dir.mkdir()
        manifest_path = model_dir / "manifest.json"
        manifest_path.write_text(json.dumps(["not", "a", "dict"]))
        empty_index.register("alpha", manifest_path)

        server = _new_server(empty_index)
        request = FetchManifestRequest(model_id="alpha", request_id="frid-5")
        response = _decode(server.handle(encode_message(request)))

        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.INTERNAL_ERROR.value


# ──────────────────────────────────────────────────────────────────────────
# Parse failures + protocol-level errors
# ──────────────────────────────────────────────────────────────────────────


class TestParseFailures:
    def test_non_json_bytes_yields_malformed(self, empty_index):
        server = _new_server(empty_index)
        response = _decode(server.handle(b"not json {{{"))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.MALFORMED_REQUEST.value
        assert response.request_id == UNKNOWN_REQUEST_ID

    def test_empty_bytes_yields_malformed(self, empty_index):
        server = _new_server(empty_index)
        response = _decode(server.handle(b""))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.MALFORMED_REQUEST.value

    def test_missing_type_field_yields_malformed(self, empty_index):
        server = _new_server(empty_index)
        response = _decode(server.handle(b'{"protocol_version": 1}'))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.MALFORMED_REQUEST.value

    def test_unknown_type_yields_malformed(self, empty_index):
        server = _new_server(empty_index)
        bytes_in = json.dumps({
            "type": "unknown_type",
            "protocol_version": DHT_PROTOCOL_VERSION,
            "request_id": "rid",
        }).encode("utf-8")
        response = _decode(server.handle(bytes_in))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.MALFORMED_REQUEST.value

    def test_version_mismatch_yields_unsupported_version(self, empty_index):
        server = _new_server(empty_index)
        # Build a wire-format payload with the WRONG protocol_version,
        # bypassing the dataclass which would itself reject.
        bytes_in = json.dumps({
            "type": MessageType.FIND_PROVIDERS.value,
            "protocol_version": DHT_PROTOCOL_VERSION + 1,
            "model_id": "alpha",
            "request_id": "rid",
        }).encode("utf-8")
        response = _decode(server.handle(bytes_in))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.UNSUPPORTED_VERSION.value

    def test_missing_required_field_yields_malformed(self, empty_index):
        server = _new_server(empty_index)
        bytes_in = json.dumps({
            "type": MessageType.FIND_PROVIDERS.value,
            "protocol_version": DHT_PROTOCOL_VERSION,
            # missing model_id
            "request_id": "rid",
        }).encode("utf-8")
        response = _decode(server.handle(bytes_in))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.MALFORMED_REQUEST.value


# ──────────────────────────────────────────────────────────────────────────
# Response-type messages arriving at server (misuse)
# ──────────────────────────────────────────────────────────────────────────


class TestResponseTypeRejection:
    def test_providers_response_at_server_rejected(self, empty_index):
        server = _new_server(empty_index)
        misuse = ProvidersResponse(
            request_id="misused",
            providers=(ProviderInfo(node_id="x", address="h:1"),),
        )
        response = _decode(server.handle(encode_message(misuse)))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.MALFORMED_REQUEST.value
        # request_id is preserved for correlation when possible
        assert response.request_id == "misused"

    def test_manifest_response_at_server_rejected(self, empty_index):
        server = _new_server(empty_index)
        misuse = ManifestResponse(
            request_id="misused-2",
            manifest={"k": "v"},
        )
        response = _decode(server.handle(encode_message(misuse)))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.MALFORMED_REQUEST.value
        assert response.request_id == "misused-2"

    def test_error_response_at_server_rejected(self, empty_index):
        server = _new_server(empty_index)
        misuse = ErrorResponse(
            request_id="misused-3",
            code=ErrorCode.NOT_FOUND.value,
            message="x",
        )
        response = _decode(server.handle(encode_message(misuse)))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.MALFORMED_REQUEST.value
        assert response.request_id == "misused-3"


# ──────────────────────────────────────────────────────────────────────────
# Never-raises invariant
# ──────────────────────────────────────────────────────────────────────────


class TestNeverRaises:
    @pytest.mark.parametrize("payload", [
        b"",
        b"\x00\x01\x02",
        b"{",
        b"null",
        b"42",
        b'"just a string"',
        b'{"type": null}',
        b'{"type": 42}',
        b'{"type": ""}',
        b'{"type": "find_providers"}',  # missing other required fields
    ])
    def test_handle_never_raises(self, empty_index, payload):
        server = _new_server(empty_index)
        # Must not raise — every bad input becomes encoded ErrorResponse.
        out = server.handle(payload)
        assert isinstance(out, bytes)
        response = _decode(out)
        assert isinstance(response, ErrorResponse)


# ──────────────────────────────────────────────────────────────────────────
# Concurrent safety
# ──────────────────────────────────────────────────────────────────────────


class TestConcurrentSafety:
    def test_parallel_handle_calls_all_succeed(self, empty_index, tmp_path):
        # Seed several models, fire many handle() calls in parallel.
        # Server has no shared mutable state, so all calls should
        # complete without exceptions and return correctly-correlated
        # responses.
        for i in range(5):
            mid = f"model-{i}"
            payload = {"model_id": mid, "n": i}
            path = _seed_manifest(tmp_path, mid, payload)
            empty_index.register(mid, path)

        server = _new_server(empty_index)
        results: dict[str, bytes] = {}
        errors: list[Exception] = []

        def worker(mid: str, rid: str):
            try:
                req = FetchManifestRequest(model_id=mid, request_id=rid)
                results[rid] = server.handle(encode_message(req))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(
                target=worker, args=(f"model-{i % 5}", f"rid-{i}")
            )
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 50
        for rid, raw in results.items():
            response = _decode(raw)
            assert isinstance(response, ManifestResponse)
            assert response.request_id == rid


# ──────────────────────────────────────────────────────────────────────────
# request_id correlation
# ──────────────────────────────────────────────────────────────────────────


class TestNeverRaisesUnderHandlerFailure:
    """HIGH-1 from Phase 3.x.5 round 1 review: the dispatch must wrap
    handler calls so a future regression in handler internals can't
    escape ``handle()``. We inject a broken LocalManifestIndex whose
    ``lookup`` raises and confirm handle() returns INTERNAL_ERROR
    rather than letting the exception propagate."""

    def test_handle_recovers_from_lookup_raising(self, tmp_path):
        class ExplodingIndex:
            def lookup(self, model_id):
                raise RuntimeError("index corrupt")

        server = ManifestDHTServer(
            local_index=ExplodingIndex(),
            routing_table=FakeRoutingTable([]),
            my_node_id="self",
            my_address="self:0",
        )
        request = FindProvidersRequest(
            model_id="alpha", request_id="rid-x"
        )
        # Must NOT raise — must encode an INTERNAL_ERROR response.
        response = _decode(server.handle(encode_message(request)))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.INTERNAL_ERROR.value
        assert response.request_id == "rid-x"

    def test_handle_recovers_from_unicode_error_on_manifest_read(
        self, empty_index, tmp_path
    ):
        # Operator drops a binary file at manifest.json; read_text
        # raises UnicodeDecodeError. Must surface as INTERNAL_ERROR
        # (handled via the explicit catch added in HIGH-1 fix), NOT
        # propagate.
        model_dir = tmp_path / "alpha"
        model_dir.mkdir()
        manifest_path = model_dir / "manifest.json"
        # Invalid UTF-8 byte sequence
        manifest_path.write_bytes(b"\xff\xfe\xfd binary garbage")
        empty_index.register("alpha", manifest_path)

        server = _new_server(empty_index)
        request = FetchManifestRequest(model_id="alpha", request_id="frid")
        response = _decode(server.handle(encode_message(request)))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.INTERNAL_ERROR.value


class TestSizeCaps:
    """MEDIUM-1 from Phase 3.x.5 round 1 review: oversized payloads
    must be rejected at parse time before json.loads allocates."""

    def test_oversized_payload_rejected(self, empty_index):
        from prsm.network.manifest_dht.protocol import MAX_MESSAGE_BYTES

        server = _new_server(empty_index)
        # Build a payload one byte over the cap.
        oversized = b"a" * (MAX_MESSAGE_BYTES + 1)
        response = _decode(server.handle(oversized))
        assert isinstance(response, ErrorResponse)
        assert response.code == ErrorCode.MALFORMED_REQUEST.value

    def test_at_cap_payload_processed_normally(self, empty_index):
        # A payload exactly at the cap must NOT be rejected by the
        # size guard (off-by-one would block legitimate large
        # manifests). It will still fail JSON parsing since "a"*N
        # isn't valid JSON, but that's the parse path, not the cap.
        from prsm.network.manifest_dht.protocol import MAX_MESSAGE_BYTES

        server = _new_server(empty_index)
        at_cap = b"a" * MAX_MESSAGE_BYTES
        response = _decode(server.handle(at_cap))
        assert isinstance(response, ErrorResponse)
        # Code is MALFORMED_REQUEST either way; the discriminating
        # message confirms which gate fired.
        assert "MAX_MESSAGE_BYTES" not in response.message


class TestRequestIdCorrelation:
    def test_find_providers_correlation(self, empty_index):
        server = _new_server(empty_index)
        request = FindProvidersRequest(
            model_id="x", request_id="unique-corr-1"
        )
        response = _decode(server.handle(encode_message(request)))
        assert response.request_id == "unique-corr-1"

    def test_fetch_manifest_correlation(self, empty_index):
        server = _new_server(empty_index)
        request = FetchManifestRequest(
            model_id="missing", request_id="unique-corr-2"
        )
        response = _decode(server.handle(encode_message(request)))
        assert response.request_id == "unique-corr-2"

    def test_parse_failure_returns_sentinel_request_id(self, empty_index):
        # Server can't extract a request_id from unparseable bytes,
        # so it returns the UNKNOWN_REQUEST_ID sentinel — client-side
        # correlation will see this as a mismatch and treat it as
        # transport failure (per dht_client._send_request).
        server = _new_server(empty_index)
        response = _decode(server.handle(b"not json"))
        assert response.request_id == UNKNOWN_REQUEST_ID
