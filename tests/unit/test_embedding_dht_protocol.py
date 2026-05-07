"""PRSM-PROV-1 Item 3 Task 1 — EmbeddingDHT wire-format protocol tests.

Verifies the JSON-over-transport messages exchanged between
``EmbeddingDHTClient`` and ``EmbeddingDHTServer``: round-trip encoding,
required-field validation, version-mismatch handling, size caps, and
the canonical signing payload format used to defend against malicious
peers serving forged vectors.
"""
from __future__ import annotations

import base64
import json
import struct
import time

import pytest

from prsm.network.embedding_dht.protocol import (
    ALLOWED_DTYPES,
    EMBEDDING_DHT_PROTOCOL_VERSION,
    MAX_MESSAGE_BYTES,
    MAX_PROVIDERS_PER_RESPONSE,
    MAX_VECTOR_DIMENSION,
    SIGNING_DOMAIN_TAG,
    EmbeddingProvidersResponse,
    EmbeddingResponse,
    ErrorCode,
    ErrorResponse,
    FetchEmbeddingRequest,
    FindEmbeddingRequest,
    IncompatibleProtocolVersionError,
    MalformedMessageError,
    MessageType,
    ProviderInfo,
    UnknownMessageTypeError,
    canonical_signing_payload,
    encode_message,
    parse_message,
)


# ---------------------------------------------------------------------------
# FindEmbeddingRequest
# ---------------------------------------------------------------------------


def test_find_embedding_round_trip():
    msg = FindEmbeddingRequest(
        content_hash="0xabc123",
        model_id="openai/text-embedding-ada-002",
        request_id="req-1",
    )
    wire = encode_message(msg)
    decoded = parse_message(wire)
    assert isinstance(decoded, FindEmbeddingRequest)
    assert decoded.content_hash == "0xabc123"
    assert decoded.model_id == "openai/text-embedding-ada-002"
    assert decoded.request_id == "req-1"
    assert decoded.protocol_version == EMBEDDING_DHT_PROTOCOL_VERSION


def test_find_embedding_rejects_empty_content_hash():
    with pytest.raises(MalformedMessageError, match="content_hash"):
        FindEmbeddingRequest(content_hash="", model_id="m", request_id="r")


def test_find_embedding_rejects_empty_model_id():
    with pytest.raises(MalformedMessageError, match="model_id"):
        FindEmbeddingRequest(content_hash="0xabc", model_id="", request_id="r")


def test_find_embedding_rejects_non_string_request_id():
    with pytest.raises(MalformedMessageError, match="request_id"):
        FindEmbeddingRequest(
            content_hash="0xabc", model_id="m", request_id=123,  # type: ignore
        )


def test_find_embedding_from_dict_rejects_wrong_type():
    bad = {
        "type": MessageType.FETCH_EMBEDDING.value,
        "protocol_version": 1,
        "content_hash": "0xabc",
        "model_id": "m",
        "request_id": "r",
    }
    with pytest.raises(MalformedMessageError, match="expected type"):
        FindEmbeddingRequest.from_dict(bad)


# ---------------------------------------------------------------------------
# FetchEmbeddingRequest
# ---------------------------------------------------------------------------


def test_fetch_embedding_round_trip():
    msg = FetchEmbeddingRequest(
        content_hash="0xdef456",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        request_id="req-2",
    )
    wire = encode_message(msg)
    decoded = parse_message(wire)
    assert isinstance(decoded, FetchEmbeddingRequest)
    assert decoded == msg


# ---------------------------------------------------------------------------
# ProviderInfo + EmbeddingProvidersResponse
# ---------------------------------------------------------------------------


def test_provider_info_round_trip():
    p = ProviderInfo(node_id="node-A", address="10.0.0.1:9000")
    out = ProviderInfo.from_dict(p.to_dict())
    assert out == p


def test_provider_info_rejects_address_without_port():
    with pytest.raises(MalformedMessageError, match="host:port"):
        ProviderInfo(node_id="node-A", address="10.0.0.1")


def test_embedding_providers_response_round_trip():
    msg = EmbeddingProvidersResponse(
        request_id="req-3",
        providers=(
            ProviderInfo(node_id="A", address="1.1.1.1:9000"),
            ProviderInfo(node_id="B", address="2.2.2.2:9000"),
        ),
    )
    wire = encode_message(msg)
    decoded = parse_message(wire)
    assert isinstance(decoded, EmbeddingProvidersResponse)
    assert decoded.request_id == "req-3"
    assert len(decoded.providers) == 2
    assert decoded.providers[0].node_id == "A"


def test_embedding_providers_accepts_list_input_coerces_tuple():
    msg = EmbeddingProvidersResponse(
        request_id="req-4",
        providers=[ProviderInfo(node_id="A", address="1:1")],  # type: ignore[arg-type]
    )
    assert isinstance(msg.providers, tuple)


def test_embedding_providers_response_rejects_too_many():
    too_many = [{"node_id": f"n{i}", "address": "1.1.1.1:9000"} for i in range(MAX_PROVIDERS_PER_RESPONSE + 1)]
    bad = {
        "type": MessageType.EMBEDDING_PROVIDERS.value,
        "protocol_version": 1,
        "request_id": "r",
        "providers": too_many,
    }
    with pytest.raises(MalformedMessageError, match="MAX_PROVIDERS"):
        EmbeddingProvidersResponse.from_dict(bad)


def test_embedding_providers_response_rejects_non_list():
    bad = {
        "type": MessageType.EMBEDDING_PROVIDERS.value,
        "protocol_version": 1,
        "request_id": "r",
        "providers": "not-a-list",
    }
    with pytest.raises(MalformedMessageError, match="providers must be a list"):
        EmbeddingProvidersResponse.from_dict(bad)


# ---------------------------------------------------------------------------
# EmbeddingResponse
# ---------------------------------------------------------------------------


def _make_embedding_response(
    *,
    dimension: int = 4,
    dtype: str = "float32",
    vector_b64: str | None = None,
    signature_b64: str = "AA" * 32,  # 32 bytes base64'd → 44 chars
    created_at: float | None = None,
) -> EmbeddingResponse:
    if vector_b64 is None:
        # Build a real float32 byte payload of correct length.
        # For invalid dimensions (<=0), use empty bytes so the
        # validator in __post_init__ is what raises, not struct.
        if dimension > 0:
            raw = struct.pack(f"<{dimension}f", *[0.1 * i for i in range(dimension)])
        else:
            raw = b""
        vector_b64 = base64.b64encode(raw).decode("ascii") or "AA=="
    if created_at is None:
        created_at = time.time()
    # Ed25519 signatures are 64 bytes; encode 64 zeros as a placeholder.
    sig_b64 = base64.b64encode(b"\x00" * 64).decode("ascii") if signature_b64 == "AA" * 32 else signature_b64
    return EmbeddingResponse(
        request_id="req-emb-1",
        content_hash="0xabc",
        model_id="openai/text-embedding-ada-002",
        dimension=dimension,
        dtype=dtype,
        vector_b64=vector_b64,
        creator_id="creator-node",
        created_at=created_at,
        signature_b64=sig_b64,
    )


def test_embedding_response_round_trip():
    msg = _make_embedding_response(dimension=384)
    wire = encode_message(msg)
    decoded = parse_message(wire)
    assert isinstance(decoded, EmbeddingResponse)
    assert decoded.dimension == 384
    assert decoded.dtype == "float32"
    assert decoded.vector_b64 == msg.vector_b64
    assert decoded.signature_b64 == msg.signature_b64
    assert decoded.content_hash == "0xabc"
    assert decoded.model_id == "openai/text-embedding-ada-002"


def test_embedding_response_rejects_negative_dimension():
    with pytest.raises(MalformedMessageError, match="dimension"):
        _make_embedding_response(dimension=-1)


def test_embedding_response_rejects_zero_dimension():
    with pytest.raises(MalformedMessageError, match="dimension"):
        _make_embedding_response(dimension=0)


def test_embedding_response_rejects_oversized_dimension():
    huge = MAX_VECTOR_DIMENSION + 1
    with pytest.raises(MalformedMessageError, match="MAX_VECTOR_DIMENSION"):
        _make_embedding_response(dimension=huge)


def test_embedding_response_rejects_disallowed_dtype():
    with pytest.raises(MalformedMessageError, match="dtype"):
        _make_embedding_response(dtype="float16")


def test_embedding_response_allowed_dtypes_includes_float32():
    assert "float32" in ALLOWED_DTYPES


def test_embedding_response_rejects_non_numeric_created_at():
    with pytest.raises(MalformedMessageError, match="created_at"):
        EmbeddingResponse(
            request_id="r",
            content_hash="0xabc",
            model_id="m",
            dimension=4,
            dtype="float32",
            vector_b64="AAAA",
            creator_id="c",
            created_at="not-a-number",  # type: ignore
            signature_b64="BBBB",
        )


# ---------------------------------------------------------------------------
# Canonical signing payload
# ---------------------------------------------------------------------------


def test_canonical_signing_payload_includes_domain_tag():
    raw = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    payload = canonical_signing_payload(
        content_hash="0xabc",
        model_id="m",
        dimension=4,
        dtype="float32",
        vector_bytes=raw,
        created_at=1715000000.0,
    )
    assert payload.startswith(SIGNING_DOMAIN_TAG)
    # Domain tag is padded to 32 bytes.
    assert len(SIGNING_DOMAIN_TAG.ljust(32, b"\x00")) == 32


def test_canonical_signing_payload_deterministic_for_same_input():
    raw = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    p1 = canonical_signing_payload(
        content_hash="0xabc",
        model_id="m",
        dimension=4,
        dtype="float32",
        vector_bytes=raw,
        created_at=1715000000.0,
    )
    p2 = canonical_signing_payload(
        content_hash="0xabc",
        model_id="m",
        dimension=4,
        dtype="float32",
        vector_bytes=raw,
        created_at=1715000000.0,
    )
    assert p1 == p2


def test_canonical_signing_payload_changes_with_content_hash():
    raw = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    p1 = canonical_signing_payload(
        content_hash="0xabc",
        model_id="m",
        dimension=4,
        dtype="float32",
        vector_bytes=raw,
        created_at=1715000000.0,
    )
    p2 = canonical_signing_payload(
        content_hash="0xdef",
        model_id="m",
        dimension=4,
        dtype="float32",
        vector_bytes=raw,
        created_at=1715000000.0,
    )
    assert p1 != p2


def test_canonical_signing_payload_changes_with_model_id():
    raw = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    p1 = canonical_signing_payload(
        content_hash="0xabc",
        model_id="m1",
        dimension=4,
        dtype="float32",
        vector_bytes=raw,
        created_at=1715000000.0,
    )
    p2 = canonical_signing_payload(
        content_hash="0xabc",
        model_id="m2",
        dimension=4,
        dtype="float32",
        vector_bytes=raw,
        created_at=1715000000.0,
    )
    assert p1 != p2


def test_canonical_signing_payload_changes_with_vector():
    raw1 = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    raw2 = struct.pack("<4f", 0.1, 0.2, 0.3, 0.5)
    p1 = canonical_signing_payload(
        content_hash="0xabc",
        model_id="m",
        dimension=4,
        dtype="float32",
        vector_bytes=raw1,
        created_at=1715000000.0,
    )
    p2 = canonical_signing_payload(
        content_hash="0xabc",
        model_id="m",
        dimension=4,
        dtype="float32",
        vector_bytes=raw2,
        created_at=1715000000.0,
    )
    assert p1 != p2


def test_canonical_signing_payload_changes_with_created_at():
    raw = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    p1 = canonical_signing_payload(
        content_hash="0xabc",
        model_id="m",
        dimension=4,
        dtype="float32",
        vector_bytes=raw,
        created_at=1715000000.0,
    )
    p2 = canonical_signing_payload(
        content_hash="0xabc",
        model_id="m",
        dimension=4,
        dtype="float32",
        vector_bytes=raw,
        created_at=1715000001.0,
    )
    assert p1 != p2


def test_canonical_signing_payload_rejects_mismatched_dim_vs_bytes():
    raw = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)  # 16 bytes, dim 4
    with pytest.raises(MalformedMessageError, match="length"):
        canonical_signing_payload(
            content_hash="0xabc",
            model_id="m",
            dimension=8,  # wrong: would need 32 bytes
            dtype="float32",
            vector_bytes=raw,
            created_at=1715000000.0,
        )


def test_canonical_signing_payload_rejects_disallowed_dtype():
    raw = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    with pytest.raises(MalformedMessageError, match="dtype"):
        canonical_signing_payload(
            content_hash="0xabc",
            model_id="m",
            dimension=4,
            dtype="float16",
            vector_bytes=raw,
            created_at=1715000000.0,
        )


def test_canonical_signing_payload_rejects_oversized_dimension():
    raw = b"\x00" * (MAX_VECTOR_DIMENSION + 1) * 4
    with pytest.raises(MalformedMessageError, match="MAX_VECTOR_DIMENSION"):
        canonical_signing_payload(
            content_hash="0xabc",
            model_id="m",
            dimension=MAX_VECTOR_DIMENSION + 1,
            dtype="float32",
            vector_bytes=raw,
            created_at=1715000000.0,
        )


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------


def test_error_response_round_trip():
    msg = ErrorResponse(
        request_id="r",
        code=ErrorCode.NOT_FOUND.value,
        message="nope",
    )
    decoded = parse_message(encode_message(msg))
    assert isinstance(decoded, ErrorResponse)
    assert decoded.code == ErrorCode.NOT_FOUND.value
    assert decoded.message == "nope"


def test_error_response_allows_empty_message():
    msg = ErrorResponse(
        request_id="r",
        code=ErrorCode.INTERNAL_ERROR.value,
        message="",
    )
    decoded = parse_message(encode_message(msg))
    assert decoded.message == ""


def test_error_codes_documented():
    # T4.9 — UNSUPPORTED_FINGERPRINT_KIND added alongside the
    # fingerprint-lane wire format extension. Previously: 5 codes;
    # post-T4.9: 6 codes.
    expected = {
        "NOT_FOUND",
        "MALFORMED_REQUEST",
        "UNSUPPORTED_VERSION",
        "UNSUPPORTED_MODEL",
        "UNSUPPORTED_FINGERPRINT_KIND",
        "INTERNAL_ERROR",
    }
    assert {e.value for e in ErrorCode} == expected


# ---------------------------------------------------------------------------
# parse_message dispatcher
# ---------------------------------------------------------------------------


def test_parse_rejects_non_bytes():
    with pytest.raises(MalformedMessageError, match="bytes"):
        parse_message("not-bytes")  # type: ignore


def test_parse_rejects_oversized_payload():
    big = b"x" * (MAX_MESSAGE_BYTES + 1)
    with pytest.raises(MalformedMessageError, match="MAX_MESSAGE_BYTES"):
        parse_message(big)


def test_parse_rejects_invalid_json():
    with pytest.raises(MalformedMessageError, match="JSON parse failed"):
        parse_message(b"this is not json")


def test_parse_rejects_non_dict_top_level():
    with pytest.raises(MalformedMessageError, match="top-level"):
        parse_message(b'["a list"]')


def test_parse_rejects_unknown_type():
    bad = json.dumps({"type": "unknown_msg", "protocol_version": 1}).encode()
    with pytest.raises(UnknownMessageTypeError, match="unknown_msg"):
        parse_message(bad)


def test_parse_rejects_missing_type():
    bad = json.dumps({"protocol_version": 1}).encode()
    with pytest.raises(MalformedMessageError, match="type"):
        parse_message(bad)


def test_parse_rejects_protocol_version_mismatch():
    bad = json.dumps({
        "type": MessageType.FIND_EMBEDDING.value,
        "protocol_version": EMBEDDING_DHT_PROTOCOL_VERSION + 99,
        "content_hash": "0xabc",
        "model_id": "m",
        "request_id": "r",
    }).encode()
    with pytest.raises(IncompatibleProtocolVersionError):
        parse_message(bad)


def test_encode_rejects_object_without_to_dict():
    class NotAMessage:
        pass
    with pytest.raises(MalformedMessageError, match="to_dict"):
        encode_message(NotAMessage())


# ---------------------------------------------------------------------------
# Cross-model partition (the design constraint that motivates this DHT)
# ---------------------------------------------------------------------------


def test_messages_carry_model_id_so_keyspace_is_partitioned():
    """A FIND_EMBEDDING for the same content_hash under two different
    model_ids must produce two distinct wire messages — confirming
    the keyspace partition that protects against cross-model
    contamination."""
    a = FindEmbeddingRequest(
        content_hash="0xabc",
        model_id="openai/text-embedding-ada-002",
        request_id="r",
    )
    b = FindEmbeddingRequest(
        content_hash="0xabc",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        request_id="r",
    )
    assert encode_message(a) != encode_message(b)


# ---------------------------------------------------------------------------
# Anti-injection: signing payload domain tag
# ---------------------------------------------------------------------------


def test_signing_payload_domain_tag_present_first_32_bytes():
    """The 32-byte domain prefix exists to prevent a creator's
    Ed25519 key being tricked into signing something later
    interpretable as a different protocol message (e.g. an
    InferenceReceipt). Verify the prefix is in fact the first
    32 bytes of the canonical payload."""
    raw = struct.pack("<2f", 0.1, 0.2)
    payload = canonical_signing_payload(
        content_hash="0xabc",
        model_id="m",
        dimension=2,
        dtype="float32",
        vector_bytes=raw,
        created_at=1.0,
    )
    expected_prefix = SIGNING_DOMAIN_TAG.ljust(32, b"\x00")
    assert payload[:32] == expected_prefix
