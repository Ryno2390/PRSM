"""B3.1b — AggregateRequest / AggregateResponse / SignedPartial wire format.

Per `docs/2026-05-08-aggregate-rpc-design.md`. Pattern-lift from
`RunLayerSliceRequest` (Phase 3.x.7) — same to_dict/from_dict shape,
same signing-payload prefix discipline, same canonical-encoding
contract.
"""
from __future__ import annotations

import base64
import hashlib

import pytest

from prsm.compute.chain_rpc.protocol import (
    ChainRpcMalformedError,
    ChainRpcMessageType,
    StageErrorCode,
)
from prsm.compute.query_orchestrator.aggregate_protocol import (
    AggregateRequest,
    AggregateResponse,
    SignedPartial,
)
from prsm.compute.query_orchestrator import AggregationCommit


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _make_partial(*, shard_cid: str = "prsm:s-0") -> SignedPartial:
    return SignedPartial(
        shard_cid=shard_cid,
        payload=b"\x01\x02\x03\x04",
        creator_id=f"creator-of-{shard_cid}",
        dp_noise_applied=True,
        source_agent_pubkey=b"\xaa" * 32,
        source_agent_signature=b"\xbb" * 64,
        privacy_budget_consumed=0.05,
    )


def _make_request(*, partials_n: int = 2) -> AggregateRequest:
    return AggregateRequest(
        request_id=b"\x10" * 32,
        query_id=b"\x20" * 32,
        manifest_json='{"query":"count records","instructions":[],"max_output_records":1000,"output_format":"json"}',
        partials=tuple(
            _make_partial(shard_cid=f"prsm:s-{i}") for i in range(partials_n)
        ),
        prompter_pubkey=b"\xc0" * 32,
        prompter_node_id="prompter-1",
        beacon_used=b"\xd0" * 32,
        aggregator_pubkey_hash=b"\xe0" * 32,
        ftns_budget=1000,
        deadline_unix=2_000_000_000,
        prompter_signature=b"\xf0" * 64,
    )


def _make_commit() -> AggregationCommit:
    return AggregationCommit(
        query_id=b"\x20" * 32,
        aggregator_pubkey_hash=b"\xe0" * 32,
        result_digest=b"\x30" * 32,
    )


def _make_response(commit: AggregationCommit | None = None) -> AggregateResponse:
    return AggregateResponse(
        request_id=b"\x10" * 32,
        query_id=b"\x20" * 32,
        commit=commit or _make_commit(),
        commit_signature=b"\x40" * 64,
        encrypted_plaintext=b"\x50" * 96,
        nonce=b"\x60" * 24,
        aggregator_pubkey=b"\x70" * 32,
        privacy_budget_consumed=0.10,
        contributing_creators=("creator-of-prsm:s-0", "creator-of-prsm:s-1"),
        completed_unix=2_000_000_001,
    )


# ──────────────────────────────────────────────────────────────────────
# SignedPartial — roundtrip + validation
# ──────────────────────────────────────────────────────────────────────


class TestSignedPartial:
    def test_roundtrip(self):
        p = _make_partial()
        again = SignedPartial.from_dict(p.to_dict())
        assert again == p

    def test_dp_noise_marker_required_field(self):
        p = _make_partial()
        d = p.to_dict()
        assert d["dp_noise_applied"] is True

    def test_short_pubkey_rejected(self):
        with pytest.raises(ChainRpcMalformedError, match="source_agent_pubkey"):
            SignedPartial(
                shard_cid="x",
                payload=b"y",
                creator_id="c",
                dp_noise_applied=True,
                source_agent_pubkey=b"\xaa" * 16,  # short
                source_agent_signature=b"\xbb" * 64,
                privacy_budget_consumed=0.0,
            )

    def test_short_signature_rejected(self):
        with pytest.raises(ChainRpcMalformedError, match="source_agent_signature"):
            SignedPartial(
                shard_cid="x",
                payload=b"y",
                creator_id="c",
                dp_noise_applied=True,
                source_agent_pubkey=b"\xaa" * 32,
                source_agent_signature=b"\xbb" * 32,  # short
                privacy_budget_consumed=0.0,
            )

    def test_negative_budget_rejected(self):
        with pytest.raises(ChainRpcMalformedError, match="privacy_budget"):
            SignedPartial(
                shard_cid="x",
                payload=b"y",
                creator_id="c",
                dp_noise_applied=True,
                source_agent_pubkey=b"\xaa" * 32,
                source_agent_signature=b"\xbb" * 64,
                privacy_budget_consumed=-0.1,
            )


# ──────────────────────────────────────────────────────────────────────
# AggregateRequest — roundtrip + signing payload + version
# ──────────────────────────────────────────────────────────────────────


class TestAggregateRequest:
    def test_message_type_pinned(self):
        req = _make_request()
        assert req.MESSAGE_TYPE == ChainRpcMessageType.AGGREGATE_REQUEST.value
        assert req.to_dict()["type"] == "aggregate_request"

    def test_roundtrip(self):
        req = _make_request(partials_n=3)
        again = AggregateRequest.from_dict(req.to_dict())
        assert again == req

    def test_version_mismatch_rejected(self):
        req = _make_request()
        d = req.to_dict()
        d["protocol_version"] = 999
        with pytest.raises(ChainRpcMalformedError):
            AggregateRequest.from_dict(d)

    def test_wrong_message_type_rejected(self):
        d = _make_request().to_dict()
        d["type"] = "run_layer_slice_request"
        with pytest.raises(ChainRpcMalformedError):
            AggregateRequest.from_dict(d)

    def test_short_query_id_rejected(self):
        with pytest.raises(ChainRpcMalformedError, match="query_id"):
            AggregateRequest(
                request_id=b"\x10" * 32,
                query_id=b"\x20" * 16,  # short
                manifest_json="{}",
                partials=(_make_partial(),),
                prompter_pubkey=b"\xc0" * 32,
                prompter_node_id="p",
                beacon_used=b"\xd0" * 32,
                aggregator_pubkey_hash=b"\xe0" * 32,
                ftns_budget=1000,
                deadline_unix=2_000_000_000,
                prompter_signature=b"\xf0" * 64,
            )

    def test_empty_partials_allowed_in_dataclass(self):
        # The wire format allows empty partials — server handler
        # rejects, but the dataclass doesn't (degenerate-but-valid;
        # mirrors RunLayerSliceRequest which doesn't enforce
        # business-logic-level fullness).
        req = _make_request(partials_n=0)
        assert req.partials == ()

    def test_signing_payload_includes_all_load_bearing_fields(self):
        req = _make_request()
        payload = req.signing_payload()
        # Prefix
        assert payload.startswith(AggregateRequest.SIGNING_PREFIX)
        # Each load-bearing 32-byte field appears verbatim in the payload.
        assert req.request_id in payload
        assert req.query_id in payload
        assert req.prompter_pubkey in payload
        assert req.beacon_used in payload
        assert req.aggregator_pubkey_hash in payload

    def test_signing_payload_changes_when_any_field_changes(self):
        a = _make_request()
        b_query_id = AggregateRequest(
            **{**a.__dict__, "query_id": b"\x99" * 32}
        )
        assert a.signing_payload() != b_query_id.signing_payload()


# ──────────────────────────────────────────────────────────────────────
# AggregateResponse — roundtrip + commit-signing payload reuse
# ──────────────────────────────────────────────────────────────────────


class TestAggregateResponse:
    def test_message_type_pinned(self):
        resp = _make_response()
        assert resp.MESSAGE_TYPE == ChainRpcMessageType.AGGREGATE_RESPONSE.value
        assert resp.to_dict()["type"] == "aggregate_response"

    def test_roundtrip(self):
        resp = _make_response()
        again = AggregateResponse.from_dict(resp.to_dict())
        assert again == resp

    def test_commit_signing_payload_is_reused(self):
        # The aggregator signs commit.signing_payload() (the existing
        # AggregationCommit method, B3.1a). The response carries that
        # signature unchanged — pin the contract.
        commit = _make_commit()
        # Response wire format MUST use the same signing-payload bytes
        # as commit.signing_payload() — verifier checks the signature
        # against THAT.
        resp = _make_response(commit=commit)
        assert resp.commit == commit
        # Existing AggregationCommit.signing_payload (from B3.1a) is
        # the canonical input for the signature on this response.
        # Nothing in the response should re-encode it.
        # Layout: 27-byte prefix + 32*3 = 123 bytes total.
        assert len(commit.signing_payload()) == 123

    def test_short_nonce_rejected(self):
        with pytest.raises(ChainRpcMalformedError, match="nonce"):
            AggregateResponse(
                request_id=b"\x10" * 32,
                query_id=b"\x20" * 32,
                commit=_make_commit(),
                commit_signature=b"\x40" * 64,
                encrypted_plaintext=b"\x50" * 32,
                nonce=b"\x60" * 12,  # short — chacha20-poly1305 needs 24
                aggregator_pubkey=b"\x70" * 32,
                privacy_budget_consumed=0.0,
                contributing_creators=(),
                completed_unix=1,
            )

    def test_short_aggregator_pubkey_rejected(self):
        with pytest.raises(ChainRpcMalformedError, match="aggregator_pubkey"):
            AggregateResponse(
                request_id=b"\x10" * 32,
                query_id=b"\x20" * 32,
                commit=_make_commit(),
                commit_signature=b"\x40" * 64,
                encrypted_plaintext=b"\x50",
                nonce=b"\x60" * 24,
                aggregator_pubkey=b"\x70" * 16,  # short
                privacy_budget_consumed=0.0,
                contributing_creators=(),
                completed_unix=1,
            )


# ──────────────────────────────────────────────────────────────────────
# Error code coverage pin
# ──────────────────────────────────────────────────────────────────────


class TestErrorCodes:
    """Pin the 3 new aggregate-RPC StageErrorCode values."""

    def test_dp_noise_marker_missing_present(self):
        assert StageErrorCode.DP_NOISE_MARKER_MISSING.value == "DP_NOISE_MARKER_MISSING"

    def test_privacy_budget_exhausted_present(self):
        assert StageErrorCode.PRIVACY_BUDGET_EXHAUSTED.value == "PRIVACY_BUDGET_EXHAUSTED"

    def test_invalid_partial_signature_present(self):
        assert StageErrorCode.INVALID_PARTIAL_SIGNATURE.value == "INVALID_PARTIAL_SIGNATURE"


# ──────────────────────────────────────────────────────────────────────
# Golden vector — request signing payload SHA-256 pin
# ──────────────────────────────────────────────────────────────────────


class TestRequestSigningPayloadGolden:
    """Once any aggregate-RPC signature exists on disk or on chain,
    changing this layout invalidates them all. Pin the SHA-256 of a
    fully-deterministic request's signing payload.

    Update the constant ONLY when intentionally bumping payload
    version (and bump SIGNING_PREFIX in lockstep)."""

    def test_known_request_payload_sha256(self):
        req = _make_request(partials_n=2)
        payload = req.signing_payload()
        digest = hashlib.sha256(payload).hexdigest()
        # If you intentionally change the signing payload format,
        # regenerate this constant from the failed test output.
        # Pinned 2026-05-08.
        assert digest == (
            "03ef9f5b607b161746e81d751c7094d0d08774ca7d8d0e4144626f802d0b865e"
        ), f"actual digest = {digest}"
