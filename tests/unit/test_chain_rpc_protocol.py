"""Phase 3.x.7 Task 1 — chain-RPC wire-protocol unit tests.

Coverage matches design plan §4 Task 1 acceptance:
  - Round-trip encode/decode for each wire type
  - HandoffToken signing-payload determinism + anchor-verify
  - RunLayerSliceResponse signing-payload determinism + anchor-verify
  - Cross-field consistency (token.request_id ↔ request.request_id)
  - Malformed-input rejection at parse time + per-field validation
  - Size cap fires before json.loads allocates
  - Version mismatch raises distinct exception class
"""

from __future__ import annotations

import base64
import json
from typing import Dict, Optional

import pytest

from prsm.compute.chain_rpc.protocol import (
    CHAIN_RPC_PROTOCOL_VERSION,
    MAX_HANDSHAKE_BYTES,
    ChainRpcMalformedError,
    ChainRpcMessageType,
    ChainRpcUnknownTypeError,
    ChainRpcVersionMismatchError,
    HandoffToken,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    StageError,
    StageErrorCode,
    encode_message,
    parse_message,
)
from prsm.compute.inference.models import ContentTier
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


class FakeAnchor:
    """In-memory AnchorLookup mirroring the Phase 3.x.3 simulator pattern."""

    def __init__(self, registered: Optional[Dict[str, str]] = None):
        self.registered: Dict[str, str] = dict(registered or {})

    def lookup(self, node_id: str) -> Optional[str]:
        return self.registered.get(node_id)


def _register(anchor: FakeAnchor, identity) -> None:
    anchor.registered[identity.node_id] = identity.public_key_b64


# ──────────────────────────────────────────────────────────────────────────
# HandoffToken
# ──────────────────────────────────────────────────────────────────────────


class TestHandoffTokenConstruction:
    def test_rejects_empty_request_id(self):
        with pytest.raises(ChainRpcMalformedError, match="request_id"):
            HandoffToken(
                request_id="",
                settler_node_id="abc",
                chain_stage_index=0,
                chain_total_stages=2,
                deadline_unix=100.0,
                signature_b64="sig",
            )

    def test_rejects_empty_settler_node_id(self):
        with pytest.raises(ChainRpcMalformedError, match="settler_node_id"):
            HandoffToken(
                request_id="r",
                settler_node_id="",
                chain_stage_index=0,
                chain_total_stages=2,
                deadline_unix=100.0,
                signature_b64="sig",
            )

    def test_rejects_negative_stage_index(self):
        with pytest.raises(ChainRpcMalformedError, match="chain_stage_index"):
            HandoffToken(
                request_id="r",
                settler_node_id="s",
                chain_stage_index=-1,
                chain_total_stages=2,
                deadline_unix=100.0,
                signature_b64="sig",
            )

    def test_rejects_zero_total_stages(self):
        with pytest.raises(ChainRpcMalformedError, match="chain_total_stages"):
            HandoffToken(
                request_id="r",
                settler_node_id="s",
                chain_stage_index=0,
                chain_total_stages=0,
                deadline_unix=100.0,
                signature_b64="sig",
            )

    def test_rejects_index_at_or_above_total(self):
        with pytest.raises(ChainRpcMalformedError, match="< chain_total_stages"):
            HandoffToken(
                request_id="r",
                settler_node_id="s",
                chain_stage_index=2,
                chain_total_stages=2,
                deadline_unix=100.0,
                signature_b64="sig",
            )

    def test_rejects_non_numeric_deadline(self):
        with pytest.raises(ChainRpcMalformedError, match="deadline_unix"):
            HandoffToken(
                request_id="r",
                settler_node_id="s",
                chain_stage_index=0,
                chain_total_stages=2,
                deadline_unix="soon",  # type: ignore[arg-type]
                signature_b64="sig",
            )


class TestHandoffTokenSigningPayload:
    def test_payload_deterministic(self):
        a = HandoffToken.signing_payload(
            "req-1", "settler", 0, 3, 100.5
        )
        b = HandoffToken.signing_payload(
            "req-1", "settler", 0, 3, 100.5
        )
        assert a == b

    def test_payload_changes_on_any_field_change(self):
        base = HandoffToken.signing_payload("r", "s", 0, 3, 100.0)
        assert base != HandoffToken.signing_payload("r2", "s", 0, 3, 100.0)
        assert base != HandoffToken.signing_payload("r", "s2", 0, 3, 100.0)
        assert base != HandoffToken.signing_payload("r", "s", 1, 3, 100.0)
        assert base != HandoffToken.signing_payload("r", "s", 0, 4, 100.0)
        assert base != HandoffToken.signing_payload("r", "s", 0, 3, 101.0)

    def test_payload_excludes_signature(self):
        # The static method takes no signature parameter; verify the
        # produced JSON does not contain a "signature_b64" key (which
        # would make signing self-referential).
        payload = HandoffToken.signing_payload("r", "s", 0, 3, 100.0)
        decoded = json.loads(payload)
        assert "signature_b64" not in decoded


class TestHandoffTokenSignAndVerify:
    def test_sign_returns_valid_token(self):
        identity = generate_node_identity("settler")
        token = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=3,
            deadline_unix=100.0,
        )
        assert token.settler_node_id == identity.node_id
        assert token.request_id == "req-1"
        assert token.chain_stage_index == 0
        assert token.chain_total_stages == 3
        assert token.deadline_unix == 100.0
        assert len(token.signature_b64) > 0

    def test_verify_with_anchor_succeeds_for_valid_token(self):
        identity = generate_node_identity("settler")
        anchor = FakeAnchor()
        _register(anchor, identity)
        token = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=3,
            deadline_unix=100.0,
        )
        assert token.verify_with_anchor(anchor) is True

    def test_verify_fails_for_unregistered_settler(self):
        identity = generate_node_identity("settler")
        token = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=3,
            deadline_unix=100.0,
        )
        # Anchor is empty — settler is not registered.
        assert token.verify_with_anchor(FakeAnchor()) is False

    def test_verify_fails_when_anchor_returns_wrong_pubkey(self):
        signer = generate_node_identity("settler")
        attacker = generate_node_identity("attacker")
        token = HandoffToken.sign(
            identity=signer,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=3,
            deadline_unix=100.0,
        )
        # Anchor maps signer's node_id to attacker's pubkey.
        anchor = FakeAnchor({signer.node_id: attacker.public_key_b64})
        assert token.verify_with_anchor(anchor) is False

    def test_verify_fails_for_tampered_field(self):
        identity = generate_node_identity("settler")
        anchor = FakeAnchor()
        _register(anchor, identity)
        token = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=3,
            deadline_unix=100.0,
        )
        # Tamper the deadline. Construct a new token with the same
        # signature but a different deadline.
        tampered = HandoffToken(
            request_id=token.request_id,
            settler_node_id=token.settler_node_id,
            chain_stage_index=token.chain_stage_index,
            chain_total_stages=token.chain_total_stages,
            deadline_unix=token.deadline_unix + 1000.0,
            signature_b64=token.signature_b64,
        )
        assert tampered.verify_with_anchor(anchor) is False

    def test_verify_with_none_anchor_returns_false(self):
        identity = generate_node_identity("settler")
        token = HandoffToken.sign(
            identity=identity,
            request_id="r",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=100.0,
        )
        assert token.verify_with_anchor(None) is False

    def test_verify_with_anchor_missing_lookup_returns_false(self):
        identity = generate_node_identity("settler")
        token = HandoffToken.sign(
            identity=identity,
            request_id="r",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=100.0,
        )

        class BadAnchor:
            pass

        assert token.verify_with_anchor(BadAnchor()) is False


class TestHandoffTokenDictRoundTrip:
    def test_round_trip_preserves_all_fields(self):
        identity = generate_node_identity("settler")
        original = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=2,
            chain_total_stages=5,
            deadline_unix=12345.678,
        )
        recovered = HandoffToken.from_dict(original.to_dict())
        assert recovered == original

    def test_from_dict_rejects_missing_fields(self):
        with pytest.raises(ChainRpcMalformedError):
            HandoffToken.from_dict({})


# ──────────────────────────────────────────────────────────────────────────
# RunLayerSliceRequest
# ──────────────────────────────────────────────────────────────────────────


def _valid_token(identity, request_id="req-1", stage_index=0, total=2):
    return HandoffToken.sign(
        identity=identity,
        request_id=request_id,
        chain_stage_index=stage_index,
        chain_total_stages=total,
        deadline_unix=1000.0,
    )


def _valid_request(*, identity, request_id="req-1", stage_index=0, total=2):
    return RunLayerSliceRequest(
        request_id=request_id,
        model_id="phase3x7-test-model",
        layer_range=(0, 4),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        activation_blob=b"activation-bytes",
        activation_shape=(1, 16),
        activation_dtype="float32",
        upstream_token=_valid_token(
            identity, request_id, stage_index, total
        ),
        deadline_unix=1000.0,
    )


class TestRunLayerSliceRequestConstruction:
    def test_happy_path(self):
        identity = generate_node_identity("settler")
        req = _valid_request(identity=identity)
        assert req.protocol_version == CHAIN_RPC_PROTOCOL_VERSION
        assert req.layer_range == (0, 4)
        assert req.privacy_tier == PrivacyLevel.NONE

    def test_rejects_invalid_layer_range_tuple(self):
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="layer_range"):
            RunLayerSliceRequest(
                request_id="r",
                model_id="m",
                layer_range=(0,),  # type: ignore[arg-type]
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"x",
                activation_shape=(1,),
                activation_dtype="f32",
                upstream_token=_valid_token(identity),
                deadline_unix=1.0,
            )

    def test_rejects_layer_range_with_end_le_start(self):
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="0 <= start < end"):
            RunLayerSliceRequest(
                request_id="r",
                model_id="m",
                layer_range=(4, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"x",
                activation_shape=(1,),
                activation_dtype="f32",
                upstream_token=_valid_token(identity),
                deadline_unix=1.0,
            )

    def test_rejects_negative_start(self):
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="0 <= start < end"):
            RunLayerSliceRequest(
                request_id="r",
                model_id="m",
                layer_range=(-1, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"x",
                activation_shape=(1,),
                activation_dtype="f32",
                upstream_token=_valid_token(identity),
                deadline_unix=1.0,
            )

    def test_rejects_non_bytes_activation_blob(self):
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="activation_blob"):
            RunLayerSliceRequest(
                request_id="r",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob="not-bytes",  # type: ignore[arg-type]
                activation_shape=(1,),
                activation_dtype="f32",
                upstream_token=_valid_token(identity),
                deadline_unix=1.0,
            )

    def test_rejects_non_positive_shape_dim(self):
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="positive int"):
            RunLayerSliceRequest(
                request_id="r",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"x",
                activation_shape=(0,),
                activation_dtype="f32",
                upstream_token=_valid_token(identity),
                deadline_unix=1.0,
            )

    def test_rejects_token_request_id_mismatch(self):
        identity = generate_node_identity("s")
        token = _valid_token(identity, request_id="OTHER")
        with pytest.raises(ChainRpcMalformedError, match="must match"):
            RunLayerSliceRequest(
                request_id="MINE",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"x",
                activation_shape=(1,),
                activation_dtype="f32",
                upstream_token=token,
                deadline_unix=1.0,
            )

    def test_rejects_non_handoff_token(self):
        with pytest.raises(ChainRpcMalformedError, match="HandoffToken"):
            RunLayerSliceRequest(
                request_id="r",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"x",
                activation_shape=(1,),
                activation_dtype="f32",
                upstream_token={"fake": "token"},  # type: ignore[arg-type]
                deadline_unix=1.0,
            )


class TestRunLayerSliceRequestRoundTrip:
    def test_round_trip(self):
        identity = generate_node_identity("settler")
        original = _valid_request(identity=identity)
        encoded = encode_message(original)
        recovered = parse_message(encoded)
        assert isinstance(recovered, RunLayerSliceRequest)
        assert recovered.request_id == original.request_id
        assert recovered.model_id == original.model_id
        assert recovered.layer_range == original.layer_range
        assert recovered.privacy_tier == original.privacy_tier
        assert recovered.content_tier == original.content_tier
        assert recovered.activation_blob == original.activation_blob
        assert recovered.activation_shape == original.activation_shape
        assert recovered.activation_dtype == original.activation_dtype
        assert recovered.upstream_token == original.upstream_token
        assert recovered.deadline_unix == original.deadline_unix

    def test_round_trip_preserves_token_signature(self):
        identity = generate_node_identity("settler")
        anchor = FakeAnchor()
        _register(anchor, identity)
        original = _valid_request(identity=identity)
        recovered = parse_message(encode_message(original))
        assert recovered.upstream_token.verify_with_anchor(anchor) is True


# ──────────────────────────────────────────────────────────────────────────
# RunLayerSliceResponse
# ──────────────────────────────────────────────────────────────────────────


class TestRunLayerSliceResponseSign:
    def test_sign_and_verify_round_trip(self):
        stage = generate_node_identity("alice")
        anchor = FakeAnchor()
        _register(anchor, stage)
        response = RunLayerSliceResponse.sign(
            identity=stage,
            request_id="req-1",
            activation_blob=b"output-bytes",
            activation_shape=(1, 32),
            activation_dtype="float16",
            duration_seconds=0.123,
            tee_attestation=b"\x01" * 32,
            tee_type=TEEType.SGX,
            epsilon_spent=0.5,
        )
        assert response.stage_node_id == stage.node_id
        assert response.verify_with_anchor(anchor) is True

    def test_verify_fails_for_tampered_activation(self):
        stage = generate_node_identity("alice")
        anchor = FakeAnchor()
        _register(anchor, stage)
        response = RunLayerSliceResponse.sign(
            identity=stage,
            request_id="req-1",
            activation_blob=b"honest",
            activation_shape=(1, 4),
            activation_dtype="float32",
            duration_seconds=0.05,
            tee_attestation=b"\x01" * 32,
            tee_type=TEEType.SGX,
            epsilon_spent=0.0,
        )
        # Reconstruct with swapped activation but same signature.
        tampered = RunLayerSliceResponse(
            request_id=response.request_id,
            activation_blob=b"forged",
            activation_shape=response.activation_shape,
            activation_dtype=response.activation_dtype,
            duration_seconds=response.duration_seconds,
            tee_attestation=response.tee_attestation,
            tee_type=response.tee_type,
            epsilon_spent=response.epsilon_spent,
            stage_signature_b64=response.stage_signature_b64,
            stage_node_id=response.stage_node_id,
        )
        assert tampered.verify_with_anchor(anchor) is False

    def test_verify_fails_for_tampered_tee_attestation(self):
        stage = generate_node_identity("alice")
        anchor = FakeAnchor()
        _register(anchor, stage)
        response = RunLayerSliceResponse.sign(
            identity=stage,
            request_id="r",
            activation_blob=b"o",
            activation_shape=(1,),
            activation_dtype="f32",
            duration_seconds=0.01,
            tee_attestation=b"\xaa" * 32,
            tee_type=TEEType.SGX,
            epsilon_spent=0.0,
        )
        tampered = RunLayerSliceResponse(
            request_id=response.request_id,
            activation_blob=response.activation_blob,
            activation_shape=response.activation_shape,
            activation_dtype=response.activation_dtype,
            duration_seconds=response.duration_seconds,
            tee_attestation=b"\xbb" * 32,  # different attestation
            tee_type=response.tee_type,
            epsilon_spent=response.epsilon_spent,
            stage_signature_b64=response.stage_signature_b64,
            stage_node_id=response.stage_node_id,
        )
        assert tampered.verify_with_anchor(anchor) is False


class TestRunLayerSliceResponseRoundTrip:
    def test_round_trip(self):
        stage = generate_node_identity("alice")
        original = RunLayerSliceResponse.sign(
            identity=stage,
            request_id="req-1",
            activation_blob=b"output",
            activation_shape=(1, 16),
            activation_dtype="float32",
            duration_seconds=0.1,
            tee_attestation=b"\x02" * 16,
            tee_type=TEEType.TDX,
            epsilon_spent=0.25,
        )
        recovered = parse_message(encode_message(original))
        assert isinstance(recovered, RunLayerSliceResponse)
        assert recovered == original


# ──────────────────────────────────────────────────────────────────────────
# StageError
# ──────────────────────────────────────────────────────────────────────────


class TestStageError:
    @pytest.mark.parametrize("code", list(StageErrorCode))
    def test_round_trip_for_each_code(self, code):
        original = StageError(
            request_id="req-1",
            code=code.value,
            message=f"stage failed with {code.value}",
        )
        recovered = parse_message(encode_message(original))
        assert isinstance(recovered, StageError)
        assert recovered == original

    def test_unknown_code_round_trips(self):
        """Future protocol versions may add codes; existing parsers
        should round-trip cleanly without rejecting."""
        original = StageError(
            request_id="r",
            code="FUTURE_CODE_NOT_IN_ENUM",
            message="from a newer peer",
        )
        recovered = parse_message(encode_message(original))
        assert recovered.code == "FUTURE_CODE_NOT_IN_ENUM"

    def test_rejects_empty_code(self):
        with pytest.raises(ChainRpcMalformedError, match="code"):
            StageError(request_id="r", code="", message="")

    def test_rejects_non_string_message(self):
        with pytest.raises(ChainRpcMalformedError, match="message"):
            StageError(
                request_id="r",
                code="X",
                message=123,  # type: ignore[arg-type]
            )


# ──────────────────────────────────────────────────────────────────────────
# Codec — parse_message / encode_message
# ──────────────────────────────────────────────────────────────────────────


class TestCodec:
    def test_parse_rejects_non_bytes(self):
        with pytest.raises(ChainRpcMalformedError, match="bytes"):
            parse_message("a string")  # type: ignore[arg-type]

    def test_parse_rejects_oversize_payload(self):
        big = b"x" * (MAX_HANDSHAKE_BYTES + 1)
        with pytest.raises(ChainRpcMalformedError, match="MAX_HANDSHAKE_BYTES"):
            parse_message(big)

    def test_parse_rejects_invalid_json(self):
        with pytest.raises(ChainRpcMalformedError, match="JSON parse"):
            parse_message(b"not-json")

    def test_parse_rejects_non_dict_top_level(self):
        with pytest.raises(ChainRpcMalformedError, match="dict"):
            parse_message(b"[1, 2, 3]")

    def test_parse_rejects_missing_type(self):
        with pytest.raises(ChainRpcMalformedError, match="type"):
            parse_message(b'{"protocol_version": 1}')

    def test_parse_unknown_type(self):
        payload = b'{"type": "made_up_type", "protocol_version": 1}'
        with pytest.raises(ChainRpcUnknownTypeError, match="unknown message type"):
            parse_message(payload)

    def test_parse_version_mismatch(self):
        payload = json.dumps({
            "type": ChainRpcMessageType.STAGE_ERROR.value,
            "protocol_version": CHAIN_RPC_PROTOCOL_VERSION + 99,
            "request_id": "r",
            "code": "X",
            "message": "from the future",
        }).encode("utf-8")
        with pytest.raises(ChainRpcVersionMismatchError):
            parse_message(payload)

    def test_encode_rejects_non_message(self):
        with pytest.raises(ChainRpcMalformedError, match="to_dict"):
            encode_message("not-a-message")

    def test_size_cap_fires_before_json_loads(self):
        """Cap check fires synchronously before json.loads gets the
        whole payload — protects against pathological-allocator DoS."""
        # 2× cap of repeating `{` would take noticeable time + memory
        # to json.loads. The cap rejects it without ever calling loads.
        payload = b"{" * (MAX_HANDSHAKE_BYTES + 100)
        with pytest.raises(ChainRpcMalformedError, match="MAX_HANDSHAKE_BYTES"):
            parse_message(payload)


# ──────────────────────────────────────────────────────────────────────────
# Constants sanity
# ──────────────────────────────────────────────────────────────────────────


class TestConstants:
    def test_protocol_version_is_one(self):
        assert CHAIN_RPC_PROTOCOL_VERSION == 1

    def test_max_handshake_bytes_reasonable(self):
        # Big enough to hold a small inline activation, small enough
        # to short-circuit DoS attempts.
        assert 1024 <= MAX_HANDSHAKE_BYTES <= 100 * 1024 * 1024

    def test_all_stage_error_codes_uppercase(self):
        for code in StageErrorCode:
            assert code.value.isupper() or "_" in code.value
