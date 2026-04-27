"""
Unit tests — Phase 3.x.5 Task 1 — manifest DHT wire-format protocol.

Acceptance per design plan §4 Task 1: wire format pinned; tests cover
each message type's encode/decode + unknown-type rejection +
missing-field rejection + version mismatch.
"""

from __future__ import annotations

import json

import pytest

from prsm.network.manifest_dht.protocol import (
    DHT_PROTOCOL_VERSION,
    ErrorCode,
    ErrorResponse,
    FetchManifestRequest,
    FindProvidersRequest,
    IncompatibleProtocolVersionError,
    MalformedMessageError,
    ManifestResponse,
    MESSAGE_TYPE_REGISTRY,
    MessageType,
    ProtocolError,
    ProviderInfo,
    ProvidersResponse,
    UnknownMessageTypeError,
    encode_message,
    parse_message,
)


# ──────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────


class TestModuleConstants:
    def test_protocol_version_is_one(self):
        # Bumping requires a deliberate test update — pinned so it
        # can't drift silently.
        assert DHT_PROTOCOL_VERSION == 1

    def test_message_type_registry_covers_all_types(self):
        # Every MessageType has a registered constructor for parse_message.
        registered = set(MESSAGE_TYPE_REGISTRY.keys())
        expected = {t.value for t in MessageType}
        assert registered == expected

    def test_error_code_values_are_stable(self):
        # Operational tooling may switch on these values — pinned.
        assert ErrorCode.NOT_FOUND.value == "NOT_FOUND"
        assert ErrorCode.MALFORMED_REQUEST.value == "MALFORMED_REQUEST"
        assert ErrorCode.UNSUPPORTED_VERSION.value == "UNSUPPORTED_VERSION"
        assert ErrorCode.INTERNAL_ERROR.value == "INTERNAL_ERROR"


# ──────────────────────────────────────────────────────────────────────────
# FindProvidersRequest
# ──────────────────────────────────────────────────────────────────────────


class TestFindProvidersRequest:
    def test_basic_construction(self):
        req = FindProvidersRequest(model_id="llama-3-8b", request_id="req-1")
        assert req.model_id == "llama-3-8b"
        assert req.request_id == "req-1"
        assert req.protocol_version == DHT_PROTOCOL_VERSION

    def test_to_dict_shape(self):
        req = FindProvidersRequest(model_id="m1", request_id="r1")
        d = req.to_dict()
        assert d == {
            "type": "find_providers",
            "protocol_version": 1,
            "model_id": "m1",
            "request_id": "r1",
        }

    def test_from_dict_roundtrip(self):
        req = FindProvidersRequest(model_id="m1", request_id="r1")
        round_tripped = FindProvidersRequest.from_dict(req.to_dict())
        assert req == round_tripped

    def test_empty_model_id_rejected(self):
        with pytest.raises(MalformedMessageError, match="model_id"):
            FindProvidersRequest(model_id="", request_id="r1")

    def test_empty_request_id_rejected(self):
        with pytest.raises(MalformedMessageError, match="request_id"):
            FindProvidersRequest(model_id="m1", request_id="")

    def test_non_string_model_id_rejected(self):
        with pytest.raises(MalformedMessageError, match="model_id"):
            FindProvidersRequest(model_id=42, request_id="r1")  # type: ignore[arg-type]

    def test_from_dict_wrong_type_rejected(self):
        with pytest.raises(MalformedMessageError, match="type"):
            FindProvidersRequest.from_dict(
                {"type": "fetch_manifest", "model_id": "m", "request_id": "r",
                 "protocol_version": 1}
            )

    def test_from_dict_missing_field_rejected(self):
        with pytest.raises(MalformedMessageError, match="model_id"):
            FindProvidersRequest.from_dict(
                {"type": "find_providers", "request_id": "r",
                 "protocol_version": 1}
            )


# ──────────────────────────────────────────────────────────────────────────
# FetchManifestRequest — same contract as FindProvidersRequest
# ──────────────────────────────────────────────────────────────────────────


class TestFetchManifestRequest:
    def test_to_dict_shape(self):
        req = FetchManifestRequest(model_id="m1", request_id="r1")
        d = req.to_dict()
        assert d["type"] == "fetch_manifest"
        assert d["model_id"] == "m1"

    def test_from_dict_roundtrip(self):
        req = FetchManifestRequest(model_id="m1", request_id="r1")
        round_tripped = FetchManifestRequest.from_dict(req.to_dict())
        assert req == round_tripped

    def test_from_dict_wrong_type_rejected(self):
        with pytest.raises(MalformedMessageError, match="type"):
            FetchManifestRequest.from_dict(
                {"type": "find_providers", "model_id": "m", "request_id": "r",
                 "protocol_version": 1}
            )


# ──────────────────────────────────────────────────────────────────────────
# ProviderInfo + ProvidersResponse
# ──────────────────────────────────────────────────────────────────────────


class TestProviderInfo:
    def test_basic(self):
        p = ProviderInfo(node_id="abc123", address="host:8765")
        assert p.node_id == "abc123"
        assert p.address == "host:8765"

    def test_address_must_have_colon(self):
        with pytest.raises(MalformedMessageError, match="host:port"):
            ProviderInfo(node_id="abc", address="no-port")

    def test_to_from_dict(self):
        p = ProviderInfo(node_id="abc", address="h:1")
        assert ProviderInfo.from_dict(p.to_dict()) == p


class TestProvidersResponse:
    def test_construction_with_providers(self):
        providers = (
            ProviderInfo(node_id="a", address="h:1"),
            ProviderInfo(node_id="b", address="h:2"),
        )
        resp = ProvidersResponse(request_id="r1", providers=providers)
        assert resp.providers == providers
        assert resp.request_id == "r1"

    def test_empty_providers_allowed(self):
        # Server can legitimately respond "I have no providers" —
        # different from NOT_FOUND (which is for fetch_manifest, not
        # find_providers).
        resp = ProvidersResponse(request_id="r1", providers=())
        assert resp.providers == ()

    def test_list_coerced_to_tuple(self):
        # JSON load path passes lists; __post_init__ coerces.
        resp = ProvidersResponse(
            request_id="r1",
            providers=[ProviderInfo(node_id="a", address="h:1")],  # type: ignore[arg-type]
        )
        assert isinstance(resp.providers, tuple)

    def test_to_dict_serializes_providers(self):
        resp = ProvidersResponse(
            request_id="r1",
            providers=(ProviderInfo(node_id="a", address="h:1"),),
        )
        d = resp.to_dict()
        assert d["providers"] == [{"node_id": "a", "address": "h:1"}]

    def test_from_dict_roundtrip(self):
        resp = ProvidersResponse(
            request_id="r1",
            providers=(
                ProviderInfo(node_id="a", address="h:1"),
                ProviderInfo(node_id="b", address="h:2"),
            ),
        )
        assert ProvidersResponse.from_dict(resp.to_dict()) == resp

    def test_from_dict_providers_must_be_list(self):
        with pytest.raises(MalformedMessageError, match="providers must be a list"):
            ProvidersResponse.from_dict(
                {"type": "providers_response", "request_id": "r",
                 "protocol_version": 1, "providers": "not-a-list"}
            )


# ──────────────────────────────────────────────────────────────────────────
# ManifestResponse
# ──────────────────────────────────────────────────────────────────────────


class TestManifestResponse:
    def test_basic_construction(self):
        # The manifest field is a dict (ModelManifest.to_dict result).
        # The DHT layer doesn't validate its internal shape.
        manifest_dict = {"model_id": "m1", "publisher_node_id": "node-A",
                         "shards": []}
        resp = ManifestResponse(request_id="r1", manifest=manifest_dict)
        assert resp.manifest == manifest_dict

    def test_non_dict_manifest_rejected(self):
        with pytest.raises(MalformedMessageError, match="manifest"):
            ManifestResponse(request_id="r1", manifest="not-a-dict")  # type: ignore[arg-type]

    def test_from_dict_roundtrip(self):
        manifest_dict = {"model_id": "m1", "shards": []}
        resp = ManifestResponse(request_id="r1", manifest=manifest_dict)
        assert ManifestResponse.from_dict(resp.to_dict()) == resp

    def test_from_dict_missing_manifest_rejected(self):
        with pytest.raises(MalformedMessageError, match="manifest"):
            ManifestResponse.from_dict(
                {"type": "manifest_response", "request_id": "r",
                 "protocol_version": 1}
            )


# ──────────────────────────────────────────────────────────────────────────
# ErrorResponse
# ──────────────────────────────────────────────────────────────────────────


class TestErrorResponse:
    def test_basic(self):
        err = ErrorResponse(
            request_id="r1",
            code=ErrorCode.NOT_FOUND.value,
            message="manifest llama-3-8b not in local index",
        )
        assert err.code == "NOT_FOUND"
        assert "llama-3-8b" in err.message

    def test_empty_message_allowed(self):
        # Message can be empty (rare but valid).
        err = ErrorResponse(request_id="r1", code="NOT_FOUND", message="")
        assert err.message == ""

    def test_non_string_message_rejected(self):
        with pytest.raises(MalformedMessageError, match="message"):
            ErrorResponse(request_id="r1", code="X", message=42)  # type: ignore[arg-type]

    def test_to_from_dict_roundtrip(self):
        err = ErrorResponse(
            request_id="r1", code="NOT_FOUND", message="x"
        )
        assert ErrorResponse.from_dict(err.to_dict()) == err

    def test_from_dict_missing_message_defaults_to_empty(self):
        # Tolerant: if `message` is absent from the wire, treat as empty.
        # (Some peers may omit it for brevity.)
        err = ErrorResponse.from_dict(
            {"type": "error", "request_id": "r1", "code": "NOT_FOUND",
             "protocol_version": 1}
        )
        assert err.message == ""


# ──────────────────────────────────────────────────────────────────────────
# parse_message + encode_message — top-level dispatch
# ──────────────────────────────────────────────────────────────────────────


class TestParseEncode:
    def test_encode_decode_roundtrip_for_each_type(self):
        messages = [
            FindProvidersRequest(model_id="m", request_id="r"),
            FetchManifestRequest(model_id="m", request_id="r"),
            ProvidersResponse(
                request_id="r",
                providers=(ProviderInfo(node_id="a", address="h:1"),),
            ),
            ManifestResponse(request_id="r", manifest={"model_id": "m"}),
            ErrorResponse(request_id="r", code="NOT_FOUND", message="x"),
        ]
        for msg in messages:
            encoded = encode_message(msg)
            assert isinstance(encoded, bytes)
            decoded = parse_message(encoded)
            assert decoded == msg

    def test_encode_message_uses_canonical_sort_keys(self):
        # Two equivalent messages must encode to byte-equal payloads
        # (sort_keys ensures dict-insertion-order doesn't leak in).
        m1 = FindProvidersRequest(model_id="m", request_id="r")
        m2 = FindProvidersRequest(model_id="m", request_id="r")
        assert encode_message(m1) == encode_message(m2)

    def test_parse_unknown_type_raises_unknown_message_type_error(self):
        bad = json.dumps(
            {"type": "totally_bogus", "request_id": "r",
             "protocol_version": 1}
        ).encode()
        with pytest.raises(UnknownMessageTypeError, match="totally_bogus"):
            parse_message(bad)

    def test_parse_missing_type_raises_malformed(self):
        bad = json.dumps({"request_id": "r", "protocol_version": 1}).encode()
        with pytest.raises(MalformedMessageError, match="type"):
            parse_message(bad)

    def test_parse_non_string_type_raises_malformed(self):
        bad = json.dumps({"type": 42, "request_id": "r"}).encode()
        with pytest.raises(MalformedMessageError, match="type"):
            parse_message(bad)

    def test_parse_invalid_json_raises_malformed(self):
        with pytest.raises(MalformedMessageError, match="JSON parse failed"):
            parse_message(b"this is not json {{{")

    def test_parse_top_level_array_rejected(self):
        with pytest.raises(MalformedMessageError, match="top-level"):
            parse_message(b'[1, 2, 3]')

    def test_parse_non_bytes_payload_rejected(self):
        with pytest.raises(MalformedMessageError, match="bytes"):
            parse_message("not-bytes")  # type: ignore[arg-type]

    def test_parse_version_mismatch_raises_dedicated_exception(self):
        # A peer running a future protocol version should be rejected
        # with the dedicated exception type so callers can react
        # differently (skip peer / cool-off / etc.) than they would
        # for a malformed-message hostile peer.
        future = json.dumps(
            {"type": "find_providers", "model_id": "m", "request_id": "r",
             "protocol_version": 999}
        ).encode()
        with pytest.raises(IncompatibleProtocolVersionError, match="999"):
            parse_message(future)

    def test_encode_message_rejects_non_dataclass(self):
        with pytest.raises(MalformedMessageError, match="to_dict"):
            encode_message({"raw": "dict"})  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────
# Exception hierarchy
# ──────────────────────────────────────────────────────────────────────────


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        assert issubclass(MalformedMessageError, ProtocolError)
        assert issubclass(UnknownMessageTypeError, ProtocolError)
        assert issubclass(IncompatibleProtocolVersionError, ProtocolError)

    def test_distinct_types(self):
        # Callers that need to distinguish "hostile peer" from "version
        # skew" can catch the specific types; same base lets the
        # generic catch-all keep working.
        assert MalformedMessageError is not UnknownMessageTypeError
        assert UnknownMessageTypeError is not IncompatibleProtocolVersionError


# ──────────────────────────────────────────────────────────────────────────
# Cross-message: request_ids correlate request → response
# ──────────────────────────────────────────────────────────────────────────


class TestRequestIDCorrelation:
    """Documents the protocol's correlation pattern: each response
    carries the same request_id as the originating request, so the
    client can match responses to outstanding requests when multiple
    are in flight."""

    def test_request_ids_match(self):
        req = FindProvidersRequest(model_id="m", request_id="abc-123")
        resp = ProvidersResponse(request_id="abc-123", providers=())
        assert req.request_id == resp.request_id

    def test_error_carries_originating_request_id(self):
        req = FetchManifestRequest(model_id="m", request_id="def-456")
        err = ErrorResponse(
            request_id="def-456", code="NOT_FOUND", message=""
        )
        assert req.request_id == err.request_id
