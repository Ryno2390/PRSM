"""
PRSM-PROV-1 Item 4 T4.9 — fingerprint DHT wire-format tests.

Covers the additive extension to the EmbeddingDHT protocol:
  - FindFingerprintRequest / FetchFingerprintRequest round-trip
  - FingerprintProvidersResponse round-trip
  - FingerprintResponse round-trip + payload-size cap
  - fingerprint_signing_payload — domain-tag separation,
    length-prefix discipline, payload size cap
  - Dispatcher recognizes the four new MessageType values
  - Existing embedding-lane messages still parse unchanged
    (additive extension proof)
  - Unknown fingerprint_kind values rejected at parse time
"""

from __future__ import annotations

import base64

import pytest

from prsm.network.embedding_dht.protocol import (
    ALLOWED_FINGERPRINT_KINDS,
    FINGERPRINT_SIGNING_DOMAIN_TAG,
    MAX_FINGERPRINT_PAYLOAD_BYTES,
    MAX_PROVIDERS_PER_RESPONSE,
    MESSAGE_TYPE_REGISTRY,
    MalformedMessageError,
    MessageType,
    SIGNING_DOMAIN_TAG,
    FetchFingerprintRequest,
    FindFingerprintRequest,
    FingerprintProvidersResponse,
    FingerprintResponse,
    FindEmbeddingRequest,
    ProviderInfo,
    encode_message,
    fingerprint_signing_payload,
    parse_message,
)


VALID_HASH = "0x" + "ab" * 32
VALID_SIG = base64.b64encode(b"x" * 64).decode()


# ──────────────────────────────────────────────────────────────────────
# FindFingerprintRequest / FetchFingerprintRequest
# ──────────────────────────────────────────────────────────────────────


class TestFindFingerprintRequest:
    def test_round_trip(self):
        req = FindFingerprintRequest(
            content_hash=VALID_HASH,
            fingerprint_kind="image-phash",
            request_id="req-1",
        )
        wire = encode_message(req)
        parsed = parse_message(wire)
        assert parsed == req

    def test_unknown_kind_rejected(self):
        with pytest.raises(MalformedMessageError, match="fingerprint_kind"):
            FindFingerprintRequest(
                content_hash=VALID_HASH,
                fingerprint_kind="text-vector",  # text uses embedding lane
                request_id="r",
            )

    def test_byte_hash_kind_rejected(self):
        """BYTE_HASH has no DHT participation — rejected at parse time."""
        with pytest.raises(MalformedMessageError):
            FindFingerprintRequest(
                content_hash=VALID_HASH,
                fingerprint_kind="byte-hash",
                request_id="r",
            )

    def test_each_allowed_kind_round_trips(self):
        for kind in sorted(ALLOWED_FINGERPRINT_KINDS):
            req = FindFingerprintRequest(
                content_hash=VALID_HASH,
                fingerprint_kind=kind,
                request_id=f"r-{kind}",
            )
            wire = encode_message(req)
            parsed = parse_message(wire)
            assert parsed.fingerprint_kind == kind


class TestFetchFingerprintRequest:
    def test_round_trip(self):
        req = FetchFingerprintRequest(
            content_hash=VALID_HASH,
            fingerprint_kind="audio-chromaprint",
            request_id="req-2",
        )
        parsed = parse_message(encode_message(req))
        assert parsed == req


# ──────────────────────────────────────────────────────────────────────
# FingerprintProvidersResponse
# ──────────────────────────────────────────────────────────────────────


class TestFingerprintProvidersResponse:
    def test_round_trip(self):
        resp = FingerprintProvidersResponse(
            request_id="req-3",
            providers=(
                ProviderInfo(node_id="node-a", address="10.0.0.1:9000"),
                ProviderInfo(node_id="node-b", address="10.0.0.2:9001"),
            ),
        )
        parsed = parse_message(encode_message(resp))
        assert parsed == resp

    def test_empty_providers_list_ok(self):
        """Server says "I know no peers serving this." Allowed."""
        resp = FingerprintProvidersResponse(
            request_id="req-empty",
            providers=(),
        )
        parsed = parse_message(encode_message(resp))
        assert parsed == resp
        assert parsed.providers == ()

    def test_oversized_providers_list_rejected(self):
        """A hostile peer that ships too many providers gets rejected."""
        too_many = [
            {"node_id": f"node-{i}", "address": f"10.0.0.{i % 256}:9000"}
            for i in range(MAX_PROVIDERS_PER_RESPONSE + 1)
        ]
        wire_dict = {
            "type": MessageType.FINGERPRINT_PROVIDERS.value,
            "protocol_version": 1,
            "request_id": "r",
            "providers": too_many,
        }
        with pytest.raises(MalformedMessageError, match="MAX_PROVIDERS"):
            FingerprintProvidersResponse.from_dict(wire_dict)


# ──────────────────────────────────────────────────────────────────────
# FingerprintResponse
# ──────────────────────────────────────────────────────────────────────


class TestFingerprintResponse:
    def _make(self, **overrides):
        defaults = dict(
            request_id="req-4",
            content_hash=VALID_HASH,
            fingerprint_kind="image-phash",
            payload_b64=base64.b64encode(b"\xde\xad\xbe\xef" * 2).decode(),
            creator_id="creator-node-1",
            created_at=1700000000.0,
            signature_b64=VALID_SIG,
        )
        defaults.update(overrides)
        return FingerprintResponse(**defaults)

    def test_round_trip(self):
        resp = self._make()
        parsed = parse_message(encode_message(resp))
        assert parsed == resp

    def test_oversized_payload_rejected_at_construction(self):
        """payload_b64 longer than the base64-of-MAX-raw cap is rejected."""
        # 200 KB raw → 273 KB encoded — well above 128 KB raw cap.
        oversized = base64.b64encode(b"x" * 200_000).decode()
        with pytest.raises(MalformedMessageError, match="payload_b64"):
            self._make(payload_b64=oversized)

    def test_unknown_kind_rejected(self):
        with pytest.raises(MalformedMessageError):
            self._make(fingerprint_kind="banana")


# ──────────────────────────────────────────────────────────────────────
# fingerprint_signing_payload
# ──────────────────────────────────────────────────────────────────────


class TestFingerprintSigningPayload:
    def test_starts_with_domain_tag(self):
        payload = fingerprint_signing_payload(
            content_hash=VALID_HASH,
            fingerprint_kind="image-phash",
            payload_bytes=b"abcd",
            created_at=1700000000.0,
        )
        # Domain tag is 32 bytes, zero-padded.
        assert payload[:32] == FINGERPRINT_SIGNING_DOMAIN_TAG.ljust(32, b"\x00")

    def test_distinct_from_embedding_domain_tag(self):
        """Embedding + fingerprint signing payloads must NOT be confusable."""
        assert FINGERPRINT_SIGNING_DOMAIN_TAG != SIGNING_DOMAIN_TAG

    def test_oversized_payload_rejected(self):
        too_big = b"x" * (MAX_FINGERPRINT_PAYLOAD_BYTES + 1)
        with pytest.raises(MalformedMessageError, match="MAX_FINGERPRINT"):
            fingerprint_signing_payload(
                content_hash=VALID_HASH,
                fingerprint_kind="audio-chromaprint",
                payload_bytes=too_big,
                created_at=1700000000.0,
            )

    def test_different_content_hash_different_payload(self):
        kwargs = dict(
            fingerprint_kind="image-phash",
            payload_bytes=b"abcd",
            created_at=1700000000.0,
        )
        a = fingerprint_signing_payload(content_hash="0x" + "11" * 32, **kwargs)
        b = fingerprint_signing_payload(content_hash="0x" + "22" * 32, **kwargs)
        assert a != b

    def test_different_kind_different_payload(self):
        """Same payload bytes under different kinds → distinct sig payloads.

        This is the load-bearing guarantee — without it, an
        attacker could reuse a signature obtained for image-phash on
        an audio-chromaprint record (or vice versa) of the same bytes.
        """
        kwargs = dict(
            content_hash=VALID_HASH,
            payload_bytes=b"abcd",
            created_at=1700000000.0,
        )
        a = fingerprint_signing_payload(fingerprint_kind="image-phash", **kwargs)
        b = fingerprint_signing_payload(fingerprint_kind="video-multihash", **kwargs)
        assert a != b

    def test_unknown_kind_rejected(self):
        with pytest.raises(MalformedMessageError):
            fingerprint_signing_payload(
                content_hash=VALID_HASH,
                fingerprint_kind="text-vector",
                payload_bytes=b"abcd",
                created_at=1700000000.0,
            )


# ──────────────────────────────────────────────────────────────────────
# Additive-extension proof: existing embedding-lane messages still work
# ──────────────────────────────────────────────────────────────────────


class TestAdditiveExtension:
    def test_existing_message_types_untouched(self):
        """The 5 original embedding-lane message types still dispatch."""
        for tag in (
            MessageType.FIND_EMBEDDING.value,
            MessageType.FETCH_EMBEDDING.value,
            MessageType.EMBEDDING_PROVIDERS.value,
            MessageType.EMBEDDING_RESPONSE.value,
            MessageType.ERROR.value,
        ):
            assert tag in MESSAGE_TYPE_REGISTRY

    def test_new_message_types_registered(self):
        """The 4 new fingerprint-lane types dispatch via the same registry."""
        for tag in (
            MessageType.FIND_FINGERPRINT.value,
            MessageType.FETCH_FINGERPRINT.value,
            MessageType.FINGERPRINT_PROVIDERS.value,
            MessageType.FINGERPRINT_RESPONSE.value,
        ):
            assert tag in MESSAGE_TYPE_REGISTRY

    def test_existing_embedding_request_round_trips(self):
        """Existing embedding-lane wire format unchanged."""
        req = FindEmbeddingRequest(
            content_hash=VALID_HASH,
            model_id="openai/text-embedding-ada-002",
            request_id="r",
        )
        parsed = parse_message(encode_message(req))
        assert parsed == req
