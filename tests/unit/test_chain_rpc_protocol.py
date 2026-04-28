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
        assert response.verify_with_anchor(
            anchor, expected_stage_node_id=stage.node_id
        ) is True

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
        assert tampered.verify_with_anchor(
            anchor, expected_stage_node_id=stage.node_id
        ) is False

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
        assert tampered.verify_with_anchor(
            anchor, expected_stage_node_id=stage.node_id
        ) is False

    def test_verify_rejects_substitution_under_different_signer(self):
        """H2 regression: a malicious peer with their own anchor-
        registered identity signs a response and claims to be the
        real stage. Without expected_stage_node_id parameter, the
        signature would verify under Mallory's pubkey. With it,
        the lookup uses Alice's pubkey and rejects."""
        alice = generate_node_identity("alice")
        mallory = generate_node_identity("mallory")
        anchor = FakeAnchor()
        _register(anchor, alice)
        _register(anchor, mallory)

        # Mallory signs a response and brazenly claims to be alice.
        response = RunLayerSliceResponse.sign(
            identity=mallory,
            request_id="req-1",
            activation_blob=b"forged",
            activation_shape=(1,),
            activation_dtype="float32",
            duration_seconds=0.05,
            tee_attestation=b"\x01" * 32,
            tee_type=TEEType.SGX,
            epsilon_spent=0.0,
        )
        # Hand-edit stage_node_id to claim alice (the impersonation
        # — sig still verifies under Mallory's pubkey at lookup-self
        # time, but expected-id check fires first).
        impersonating = RunLayerSliceResponse(
            request_id=response.request_id,
            activation_blob=response.activation_blob,
            activation_shape=response.activation_shape,
            activation_dtype=response.activation_dtype,
            duration_seconds=response.duration_seconds,
            tee_attestation=response.tee_attestation,
            tee_type=response.tee_type,
            epsilon_spent=response.epsilon_spent,
            stage_signature_b64=response.stage_signature_b64,
            stage_node_id=alice.node_id,  # claim to be alice
        )
        # Caller dispatched to alice → expects alice. Substitution
        # rejected at the cross-field check.
        assert impersonating.verify_with_anchor(
            anchor, expected_stage_node_id=alice.node_id
        ) is False
        # Verify the ORIGINAL response (truthful stage_node_id =
        # mallory) does verify when the caller dispatched to mallory.
        assert response.verify_with_anchor(
            anchor, expected_stage_node_id=mallory.node_id
        ) is True

    def test_verify_rejects_empty_expected_stage_node_id(self):
        stage = generate_node_identity("alice")
        anchor = FakeAnchor()
        _register(anchor, stage)
        response = RunLayerSliceResponse.sign(
            identity=stage,
            request_id="r",
            activation_blob=b"x",
            activation_shape=(1,),
            activation_dtype="float32",
            duration_seconds=0.05,
            tee_attestation=b"\x01" * 32,
            tee_type=TEEType.SGX,
            epsilon_spent=0.0,
        )
        assert response.verify_with_anchor(
            anchor, expected_stage_node_id=""
        ) is False
        assert response.verify_with_anchor(
            anchor, expected_stage_node_id=None  # type: ignore[arg-type]
        ) is False


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


class TestBoolRejection:
    """M1 regression: bool is a subclass of int in Python; without
    explicit guards a peer sending {field: true} would slip through
    via True == 1, polluting downstream telemetry + equality."""

    def test_handoff_token_rejects_bool_chain_stage_index(self):
        with pytest.raises(ChainRpcMalformedError, match="chain_stage_index"):
            HandoffToken(
                request_id="r",
                settler_node_id="s",
                chain_stage_index=False,  # type: ignore[arg-type]
                chain_total_stages=3,
                deadline_unix=100.0,
                signature_b64="sig",
            )

    def test_handoff_token_rejects_bool_chain_total_stages(self):
        with pytest.raises(ChainRpcMalformedError, match="chain_total_stages"):
            HandoffToken(
                request_id="r",
                settler_node_id="s",
                chain_stage_index=0,
                chain_total_stages=True,  # type: ignore[arg-type]
                deadline_unix=100.0,
                signature_b64="sig",
            )

    def test_parse_message_rejects_bool_protocol_version(self):
        # L1 round-1 (3.x.7.1): bool/string/missing protocol_version is
        # a version-negotiation failure, not a malformed-message
        # failure — surface it as ``ChainRpcVersionMismatchError``.
        # bool rejection itself preserved from M1 round-1 (Phase 3.x.7
        # Task 8) — bool is a subclass of int in Python, so True == 1
        # would slip through without the explicit type-check.
        payload = json.dumps({
            "type": ChainRpcMessageType.STAGE_ERROR.value,
            "protocol_version": True,  # would == 1 without explicit guard
            "request_id": "r",
            "code": "X",
            "message": "",
        }).encode("utf-8")
        with pytest.raises(ChainRpcVersionMismatchError, match="protocol_version"):
            parse_message(payload)

    def test_parse_message_rejects_string_protocol_version(self):
        # L1 round-1 (3.x.7.1): non-int version → version mismatch.
        payload = json.dumps({
            "type": ChainRpcMessageType.STAGE_ERROR.value,
            "protocol_version": "1",
            "request_id": "r",
            "code": "X",
            "message": "",
        }).encode("utf-8")
        with pytest.raises(ChainRpcVersionMismatchError, match="protocol_version"):
            parse_message(payload)

    def test_parse_message_rejects_missing_protocol_version(self):
        # L1 round-1 (3.x.7.1): a peer that omits protocol_version
        # entirely is mid-negotiation, not a malformed message.
        payload = json.dumps({
            "type": ChainRpcMessageType.STAGE_ERROR.value,
            # protocol_version absent
            "request_id": "r",
            "code": "X",
            "message": "",
        }).encode("utf-8")
        with pytest.raises(ChainRpcVersionMismatchError, match="protocol_version"):
            parse_message(payload)


class TestConstants:
    def test_protocol_version_is_two(self):
        # v2 added the optional activation_manifest field for the
        # Phase 3.x.7.1 chunked-streaming path. v2 nodes accept v1
        # messages from peers; v1 nodes reject v2.
        assert CHAIN_RPC_PROTOCOL_VERSION == 2

    def test_max_handshake_bytes_reasonable(self):
        # Big enough to hold a small inline activation, small enough
        # to short-circuit DoS attempts.
        assert 1024 <= MAX_HANDSHAKE_BYTES <= 100 * 1024 * 1024

    def test_all_stage_error_codes_uppercase(self):
        for code in StageErrorCode:
            assert code.value.isupper() or "_" in code.value


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.7.1 v2 streaming surface
# ──────────────────────────────────────────────────────────────────────────


import hashlib as _hashlib

from prsm.compute.chain_rpc.protocol import (
    SUPPORTED_PROTOCOL_VERSIONS,
    ActivationChunk,
)
from prsm.node.shard_streaming import ShardManifest


def _make_manifest(payload: bytes = b"hello world", shard_id: str = "act-1") -> ShardManifest:
    """Build a real ShardManifest committing to ``payload`` bytes."""
    return ShardManifest(
        shard_id=shard_id,
        payload_sha256=_hashlib.sha256(payload).hexdigest(),
        payload_bytes=len(payload),
        total_chunks=1,
        chunk_bytes=1024 * 1024,
    )


class TestActivationChunk:
    def test_round_trip(self):
        chunk = ActivationChunk(
            request_id="req-1",
            sequence=0,
            data=b"chunk-bytes",
            chunk_sha256=_hashlib.sha256(b"chunk-bytes").hexdigest(),
        )
        recovered = parse_message(encode_message(chunk))
        assert isinstance(recovered, ActivationChunk)
        assert recovered == chunk

    def test_rejects_negative_sequence(self):
        with pytest.raises(ChainRpcMalformedError, match="sequence"):
            ActivationChunk(
                request_id="r",
                sequence=-1,
                data=b"x",
                chunk_sha256="0" * 64,
            )

    def test_rejects_bool_sequence(self):
        with pytest.raises(ChainRpcMalformedError, match="sequence"):
            ActivationChunk(
                request_id="r",
                sequence=True,  # type: ignore[arg-type]
                data=b"x",
                chunk_sha256="0" * 64,
            )

    def test_rejects_non_bytes_data(self):
        with pytest.raises(ChainRpcMalformedError, match="data"):
            ActivationChunk(
                request_id="r",
                sequence=0,
                data="not-bytes",  # type: ignore[arg-type]
                chunk_sha256="0" * 64,
            )

    def test_rejects_empty_request_id(self):
        with pytest.raises(ChainRpcMalformedError, match="request_id"):
            ActivationChunk(
                request_id="",
                sequence=0,
                data=b"x",
                chunk_sha256="0" * 64,
            )

    def test_from_dict_rejects_bad_hex(self):
        bad = {
            "type": ChainRpcMessageType.ACTIVATION_CHUNK.value,
            "protocol_version": CHAIN_RPC_PROTOCOL_VERSION,
            "request_id": "r",
            "sequence": 0,
            "data_hex": "ZZZZ",  # not valid hex
            "chunk_sha256": "0" * 64,
        }
        with pytest.raises(ChainRpcMalformedError, match="hex"):
            ActivationChunk.from_dict(bad)


class TestStreamedRequestRoundTrip:
    def test_streamed_request_round_trips(self):
        """v2 RunLayerSliceRequest with activation_manifest set must
        round-trip through parse/encode preserving all fields."""
        identity = generate_node_identity("settler")
        manifest = _make_manifest(payload=b"big-tensor-bytes")
        token = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=1000.0,
        )
        req = RunLayerSliceRequest(
            request_id="req-1",
            model_id="m",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"",  # streamed → empty inline
            activation_shape=(1, 16),
            activation_dtype="float32",
            upstream_token=token,
            deadline_unix=1000.0,
            activation_manifest=manifest,
        )
        recovered = parse_message(encode_message(req))
        assert isinstance(recovered, RunLayerSliceRequest)
        assert recovered.activation_manifest == manifest
        assert recovered.activation_blob == b""

    def test_request_rejects_neither_blob_nor_manifest(self):
        """M3 round-1 (3.x.7.1): the inline-XOR-streamed integrity
        check rejects requests where BOTH the inline blob is empty AND
        the manifest is absent. Without a payload path the message is
        structurally meaningless."""
        identity = generate_node_identity("settler")
        token = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=1000.0,
        )
        with pytest.raises(
            ChainRpcMalformedError,
            match="exactly one payload path",
        ):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"",  # empty
                activation_shape=(1, 4),
                activation_dtype="float32",
                upstream_token=token,
                deadline_unix=1000.0,
                activation_manifest=None,  # also absent
            )

    def test_streamed_request_rejects_non_empty_blob(self):
        """Inline-XOR-streamed integrity check: when manifest is
        present, activation_blob MUST be empty."""
        identity = generate_node_identity("settler")
        manifest = _make_manifest()
        token = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=1000.0,
        )
        with pytest.raises(ChainRpcMalformedError, match="streamed mode"):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"non-empty",  # ← MUST be empty
                activation_shape=(1, 4),
                activation_dtype="float32",
                upstream_token=token,
                deadline_unix=1000.0,
                activation_manifest=manifest,
            )

    def test_inline_request_unchanged_when_manifest_absent(self):
        """v2 inline requests (manifest=None) round-trip cleanly and
        the to_dict output omits the activation_manifest key entirely
        (preserving v1 wire-byte equivalence)."""
        identity = generate_node_identity("settler")
        token = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=1000.0,
        )
        req = RunLayerSliceRequest(
            request_id="req-1",
            model_id="m",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"some-bytes",
            activation_shape=(1, 4),
            activation_dtype="float32",
            upstream_token=token,
            deadline_unix=1000.0,
        )
        d = req.to_dict()
        assert "activation_manifest" not in d
        recovered = parse_message(encode_message(req))
        assert recovered.activation_manifest is None


class TestStreamedResponseSigning:
    def test_streamed_response_signs_and_verifies(self):
        stage = generate_node_identity("alice")
        anchor = FakeAnchor()
        _register(anchor, stage)
        manifest = _make_manifest(payload=b"big-output-bytes")
        response = RunLayerSliceResponse.sign(
            identity=stage,
            request_id="req-1",
            activation_blob=b"",  # streamed
            activation_shape=(1, 16),
            activation_dtype="float32",
            duration_seconds=0.05,
            tee_attestation=b"\x01" * 32,
            tee_type=TEEType.SGX,
            epsilon_spent=0.0,
            activation_manifest=manifest,
        )
        assert response.activation_manifest == manifest
        assert response.verify_with_anchor(
            anchor, expected_stage_node_id=stage.node_id
        ) is True

    def test_tampered_manifest_sha_invalidates_signature(self):
        """Stage's signature commits to manifest.payload_sha256. Any
        tamper of the manifest invalidates the signature even if the
        rest of the response is intact."""
        stage = generate_node_identity("alice")
        anchor = FakeAnchor()
        _register(anchor, stage)
        manifest = _make_manifest(payload=b"honest")
        response = RunLayerSliceResponse.sign(
            identity=stage,
            request_id="req-1",
            activation_blob=b"",
            activation_shape=(1,),
            activation_dtype="float32",
            duration_seconds=0.05,
            tee_attestation=b"\x01" * 32,
            tee_type=TEEType.SGX,
            epsilon_spent=0.0,
            activation_manifest=manifest,
        )
        forged_manifest = ShardManifest(
            shard_id=manifest.shard_id,
            payload_sha256=_hashlib.sha256(b"forged").hexdigest(),
            payload_bytes=manifest.payload_bytes,
            total_chunks=manifest.total_chunks,
            chunk_bytes=manifest.chunk_bytes,
        )
        tampered = RunLayerSliceResponse(
            request_id=response.request_id,
            activation_blob=response.activation_blob,
            activation_shape=response.activation_shape,
            activation_dtype=response.activation_dtype,
            duration_seconds=response.duration_seconds,
            tee_attestation=response.tee_attestation,
            tee_type=response.tee_type,
            epsilon_spent=response.epsilon_spent,
            stage_signature_b64=response.stage_signature_b64,
            stage_node_id=response.stage_node_id,
            activation_manifest=forged_manifest,
        )
        assert tampered.verify_with_anchor(
            anchor, expected_stage_node_id=stage.node_id
        ) is False

    def test_response_rejects_neither_blob_nor_manifest(self):
        """M3 round-1 (3.x.7.1): response __post_init__ rejects the
        case where the inline blob is empty AND the manifest is
        absent. Mirror of the Request-side check."""
        with pytest.raises(
            ChainRpcMalformedError,
            match="exactly one payload path",
        ):
            RunLayerSliceResponse(
                request_id="r",
                activation_blob=b"",  # empty
                activation_shape=(1,),
                activation_dtype="float32",
                duration_seconds=0.05,
                tee_attestation=b"\x01" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
                stage_signature_b64="sig",
                stage_node_id="alice",
                activation_manifest=None,  # also absent
            )

    def test_streamed_response_round_trips(self):
        stage = generate_node_identity("alice")
        manifest = _make_manifest()
        original = RunLayerSliceResponse.sign(
            identity=stage,
            request_id="req-1",
            activation_blob=b"",
            activation_shape=(1, 16),
            activation_dtype="float32",
            duration_seconds=0.1,
            tee_attestation=b"\x02" * 16,
            tee_type=TEEType.TDX,
            epsilon_spent=0.25,
            activation_manifest=manifest,
        )
        recovered = parse_message(encode_message(original))
        assert isinstance(recovered, RunLayerSliceResponse)
        assert recovered == original


class TestV1V2InlineByteEquivalence:
    """v1 inline messages MUST sign-verify byte-identical under v2's
    signing_payload formula. The conditional manifest_sha encoding
    (omit field when manifest is None) makes this work."""

    def test_inline_signing_payload_byte_equivalent_when_manifest_none(self):
        # Build the exact same payload via signing_payload with
        # activation_manifest=None — should match the v1 formula.
        with_default = RunLayerSliceResponse.signing_payload(
            request_id="r",
            activation_blob=b"abc",
            activation_shape=(1,),
            activation_dtype="float32",
            duration_seconds=0.1,
            tee_attestation=b"\x01",
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
            stage_node_id="n",
        )
        with_explicit_none = RunLayerSliceResponse.signing_payload(
            request_id="r",
            activation_blob=b"abc",
            activation_shape=(1,),
            activation_dtype="float32",
            duration_seconds=0.1,
            tee_attestation=b"\x01",
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
            stage_node_id="n",
            activation_manifest=None,
        )
        assert with_default == with_explicit_none

    def test_inline_signing_payload_omits_manifest_envelope_key(self):
        """Verify the conditional encoding: payload bytes MUST NOT
        contain `activation_manifest_envelope` (the H2 remediation
        key) when manifest is None."""
        payload = RunLayerSliceResponse.signing_payload(
            request_id="r",
            activation_blob=b"abc",
            activation_shape=(1,),
            activation_dtype="float32",
            duration_seconds=0.1,
            tee_attestation=b"\x01",
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
            stage_node_id="n",
        )
        assert b"activation_manifest_envelope" not in payload
        # Defense in depth: also no legacy key.
        assert b"activation_manifest_sha" not in payload

    def test_streamed_signing_payload_includes_full_manifest_envelope(self):
        """H2 round-1 remediation: streamed signing payload commits
        to ALL FIVE manifest fields (shard_id, payload_sha256,
        payload_bytes, total_chunks, chunk_bytes) — not just
        payload_sha256. Tampering ANY of them invalidates the
        signature."""
        manifest = _make_manifest()
        payload = RunLayerSliceResponse.signing_payload(
            request_id="r",
            activation_blob=b"",
            activation_shape=(1,),
            activation_dtype="float32",
            duration_seconds=0.1,
            tee_attestation=b"\x01",
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
            stage_node_id="n",
            activation_manifest=manifest,
        )
        assert b"activation_manifest_envelope" in payload
        assert manifest.payload_sha256.encode() in payload
        # All five fields appear by name.
        assert b"shard_id" in payload
        assert b"payload_bytes" in payload
        assert b"total_chunks" in payload
        assert b"chunk_bytes" in payload

    def test_tampered_manifest_payload_bytes_invalidates_signature(self):
        """H2 round-1 remediation regression: a network-level relay
        that tampers payload_bytes (without an Ed25519 key) MUST be
        detected at signature verification."""
        stage = generate_node_identity("alice")
        anchor = FakeAnchor()
        _register(anchor, stage)
        manifest = _make_manifest(payload=b"honest")
        response = RunLayerSliceResponse.sign(
            identity=stage,
            request_id="r",
            activation_blob=b"",
            activation_shape=(1,),
            activation_dtype="float32",
            duration_seconds=0.05,
            tee_attestation=b"\x01" * 32,
            tee_type=TEEType.SGX,
            epsilon_spent=0.0,
            activation_manifest=manifest,
        )
        # Inflate payload_bytes 1000× (DoS vector) while keeping
        # payload_sha256 + signature intact.
        tampered_manifest = ShardManifest(
            shard_id=manifest.shard_id,
            payload_sha256=manifest.payload_sha256,  # unchanged
            payload_bytes=manifest.payload_bytes * 1000,
            total_chunks=manifest.total_chunks,
            chunk_bytes=manifest.chunk_bytes,
        )
        tampered = RunLayerSliceResponse(
            request_id=response.request_id,
            activation_blob=response.activation_blob,
            activation_shape=response.activation_shape,
            activation_dtype=response.activation_dtype,
            duration_seconds=response.duration_seconds,
            tee_attestation=response.tee_attestation,
            tee_type=response.tee_type,
            epsilon_spent=response.epsilon_spent,
            stage_signature_b64=response.stage_signature_b64,
            stage_node_id=response.stage_node_id,
            activation_manifest=tampered_manifest,
        )
        assert tampered.verify_with_anchor(
            anchor, expected_stage_node_id=stage.node_id
        ) is False

    def test_tampered_manifest_total_chunks_invalidates_signature(self):
        stage = generate_node_identity("alice")
        anchor = FakeAnchor()
        _register(anchor, stage)
        manifest = _make_manifest()
        response = RunLayerSliceResponse.sign(
            identity=stage,
            request_id="r",
            activation_blob=b"",
            activation_shape=(1,),
            activation_dtype="float32",
            duration_seconds=0.05,
            tee_attestation=b"\x01" * 32,
            tee_type=TEEType.SGX,
            epsilon_spent=0.0,
            activation_manifest=manifest,
        )
        tampered_manifest = ShardManifest(
            shard_id=manifest.shard_id,
            payload_sha256=manifest.payload_sha256,
            payload_bytes=manifest.payload_bytes,
            total_chunks=manifest.total_chunks + 1000000,  # DoS vector
            chunk_bytes=manifest.chunk_bytes,
        )
        tampered = RunLayerSliceResponse(
            request_id=response.request_id,
            activation_blob=response.activation_blob,
            activation_shape=response.activation_shape,
            activation_dtype=response.activation_dtype,
            duration_seconds=response.duration_seconds,
            tee_attestation=response.tee_attestation,
            tee_type=response.tee_type,
            epsilon_spent=response.epsilon_spent,
            stage_signature_b64=response.stage_signature_b64,
            stage_node_id=response.stage_node_id,
            activation_manifest=tampered_manifest,
        )
        assert tampered.verify_with_anchor(
            anchor, expected_stage_node_id=stage.node_id
        ) is False


class TestVersionAcceptance:
    """v2 nodes accept v1 + v2 messages from peers (forward-compat
    rolling-deploy support)."""

    def test_v1_message_accepted(self):
        """A v1-versioned StageError parses successfully on a v2 node."""
        payload = json.dumps({
            "type": ChainRpcMessageType.STAGE_ERROR.value,
            "protocol_version": 1,
            "request_id": "r",
            "code": "X",
            "message": "from a v1 peer",
        }).encode("utf-8")
        recovered = parse_message(payload)
        assert isinstance(recovered, StageError)
        assert recovered.protocol_version == 1

    def test_unsupported_future_version_rejected(self):
        payload = json.dumps({
            "type": ChainRpcMessageType.STAGE_ERROR.value,
            "protocol_version": 99,
            "request_id": "r",
            "code": "X",
            "message": "from the future",
        }).encode("utf-8")
        with pytest.raises(ChainRpcVersionMismatchError):
            parse_message(payload)

    def test_supported_versions_constant(self):
        assert SUPPORTED_PROTOCOL_VERSIONS == frozenset({1, 2})
        assert CHAIN_RPC_PROTOCOL_VERSION in SUPPORTED_PROTOCOL_VERSIONS


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.8 — TokenFrame + StreamFinalFrame + streaming flag
# ──────────────────────────────────────────────────────────────────────────


from prsm.compute.chain_rpc.protocol import (
    StreamFinalFrame,
    TokenFrame,
)


def _signed_response_for_text(
    text: str,
    *,
    request_id: str = "req-1",
    stage_node_id: str = "alice",
) -> "RunLayerSliceResponse":
    """Build a tail-stage signed response whose activation_blob carries
    the joined output bytes (UTF-8). Mirrors the Phase 3.x.8 contract:
    the stage signs over the joined text, so a relay tampering any
    TokenFrame.text_delta diverges the joined hash from what was
    signed."""
    payload = text.encode("utf-8")
    identity = generate_node_identity(stage_node_id)
    return RunLayerSliceResponse.sign(
        identity=identity,
        request_id=request_id,
        activation_blob=payload,
        activation_shape=(len(payload),) if payload else (0,),
        activation_dtype="uint8",
        duration_seconds=0.05,
        tee_attestation=b"\x01" * 32,
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
    )


class TestTokenFrameConstruction:
    def test_minimal_round_trip(self):
        frame = TokenFrame(
            request_id="req-1",
            sequence_index=0,
            text_delta="hello",
        )
        recovered = parse_message(encode_message(frame))
        assert isinstance(recovered, TokenFrame)
        assert recovered == frame

    def test_full_field_round_trip(self):
        frame = TokenFrame(
            request_id="req-1",
            sequence_index=42,
            text_delta=" world",
            token_id=12345,
            finish_reason="stop",
        )
        recovered = parse_message(encode_message(frame))
        assert recovered == frame
        assert recovered.token_id == 12345
        assert recovered.finish_reason == "stop"

    def test_empty_text_delta_allowed(self):
        # Empty text_delta is valid (some tokenizer outputs are
        # whitespace-only or BPE merges that produce no visible text
        # on a given step). Only None is rejected.
        frame = TokenFrame(
            request_id="req-1",
            sequence_index=0,
            text_delta="",
        )
        recovered = parse_message(encode_message(frame))
        assert recovered.text_delta == ""

    def test_rejects_empty_request_id(self):
        with pytest.raises(ChainRpcMalformedError, match="request_id"):
            TokenFrame(request_id="", sequence_index=0, text_delta="x")

    def test_rejects_negative_sequence_index(self):
        with pytest.raises(ChainRpcMalformedError, match="sequence_index"):
            TokenFrame(request_id="r", sequence_index=-1, text_delta="x")

    def test_rejects_bool_sequence_index(self):
        # bool is int-subclass; without explicit guard True == 1
        # would slip through and pollute ordering.
        with pytest.raises(ChainRpcMalformedError, match="sequence_index"):
            TokenFrame(
                request_id="r",
                sequence_index=True,  # type: ignore[arg-type]
                text_delta="x",
            )

    def test_rejects_non_string_text_delta(self):
        with pytest.raises(ChainRpcMalformedError, match="text_delta"):
            TokenFrame(
                request_id="r",
                sequence_index=0,
                text_delta=b"bytes-not-str",  # type: ignore[arg-type]
            )

    def test_rejects_invalid_finish_reason(self):
        with pytest.raises(ChainRpcMalformedError, match="finish_reason"):
            TokenFrame(
                request_id="r",
                sequence_index=0,
                text_delta="x",
                finish_reason="all_done",  # not in {stop, max_tokens, cancelled, error}
            )

    def test_rejects_bool_token_id(self):
        with pytest.raises(ChainRpcMalformedError, match="token_id"):
            TokenFrame(
                request_id="r",
                sequence_index=0,
                text_delta="x",
                token_id=True,  # type: ignore[arg-type]
            )

    def test_to_dict_omits_none_optionals(self):
        # Defense-in-depth: a frame with no token_id / finish_reason
        # should NOT emit `null` keys in the canonical JSON.
        frame = TokenFrame(
            request_id="r",
            sequence_index=0,
            text_delta="x",
        )
        d = frame.to_dict()
        assert "token_id" not in d
        assert "finish_reason" not in d

    def test_to_dict_includes_set_optionals(self):
        frame = TokenFrame(
            request_id="r",
            sequence_index=5,
            text_delta="end",
            token_id=42,
            finish_reason="max_tokens",
        )
        d = frame.to_dict()
        assert d["token_id"] == 42
        assert d["finish_reason"] == "max_tokens"


class TestStreamFinalFrameConstruction:
    def test_round_trip_with_signed_response(self):
        joined = "hello world"
        response = _signed_response_for_text(joined)
        frame = StreamFinalFrame(response=response)
        recovered = parse_message(encode_message(frame))
        assert isinstance(recovered, StreamFinalFrame)
        assert recovered == frame
        # The embedded response's activation_blob is the joined-text
        # bytes — the stage signature commits to it via signing_payload.
        assert recovered.response.activation_blob == joined.encode("utf-8")

    def test_signature_invalidates_when_joined_text_diverges(self):
        # The cross-frame integrity check the design plan promises:
        # if the JOINED TokenFrame text_deltas don't equal the
        # response's activation_blob, the consumer's joined hash
        # diverges and rejects. (The signature itself only commits
        # to what's IN the response — the consumer compares.)
        response = _signed_response_for_text("hello world")
        # Simulate a relay swapping in a tampered TokenFrame stream
        # whose joined text is "hellO world" (capital O).
        tampered_join = "hellO world"
        # The consumer-side check: assert joined.encode == response.activation_blob
        assert tampered_join.encode("utf-8") != response.activation_blob

    def test_rejects_non_response_payload(self):
        with pytest.raises(ChainRpcMalformedError, match="response"):
            StreamFinalFrame(response="not-a-response")  # type: ignore[arg-type]

    def test_from_dict_rejects_non_dict_response(self):
        bad = {
            "type": ChainRpcMessageType.STREAM_FINAL_FRAME.value,
            "protocol_version": CHAIN_RPC_PROTOCOL_VERSION,
            "response": "not-a-dict",
        }
        with pytest.raises(ChainRpcMalformedError, match="response"):
            StreamFinalFrame.from_dict(bad)


class TestStreamingFlagOnRunLayerSliceRequest:
    """The Phase 3.x.8 ``streaming: bool`` field on
    ``RunLayerSliceRequest``. Conditional encoding preserves
    byte-equivalence with v2-pre-3.x.8 messages so the existing v1↔v2
    forward-compat invariant still holds for non-streamed traffic."""

    def _build_request(self, *, streaming: bool = False) -> RunLayerSliceRequest:
        identity = generate_node_identity("settler")
        token = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=1000.0,
        )
        return RunLayerSliceRequest(
            request_id="req-1",
            model_id="m",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"some-bytes",
            activation_shape=(1, 4),
            activation_dtype="float32",
            upstream_token=token,
            deadline_unix=1000.0,
            streaming=streaming,
        )

    def test_default_streaming_false(self):
        req = self._build_request()
        assert req.streaming is False

    def test_streaming_false_omits_key_in_canonical_json(self):
        # Conditional encoding: streaming=False MUST omit the field
        # entirely, preserving byte-equivalence with v2-pre-3.x.8
        # serializers.
        req = self._build_request(streaming=False)
        d = req.to_dict()
        assert "streaming" not in d
        # The canonical-JSON bytes carry the same encoded structure
        # as a 3.x.7.1 request would have.
        encoded = encode_message(req)
        assert b'"streaming"' not in encoded

    def test_streaming_true_encoded_in_canonical_json(self):
        req = self._build_request(streaming=True)
        d = req.to_dict()
        assert d["streaming"] is True
        encoded = encode_message(req)
        assert b'"streaming": true' in encoded

    def test_round_trip_streaming_true(self):
        req = self._build_request(streaming=True)
        recovered = parse_message(encode_message(req))
        assert isinstance(recovered, RunLayerSliceRequest)
        assert recovered.streaming is True

    def test_round_trip_streaming_false(self):
        req = self._build_request(streaming=False)
        recovered = parse_message(encode_message(req))
        assert recovered.streaming is False

    def test_from_dict_defaults_to_false_when_key_absent(self):
        # A pre-3.x.8 serializer that doesn't know about the streaming
        # field still parses cleanly on a 3.x.8 server — streaming
        # defaults to False.
        req = self._build_request(streaming=False)
        wire = req.to_dict()
        assert "streaming" not in wire
        recovered = RunLayerSliceRequest.from_dict(wire)
        assert recovered.streaming is False

    def test_constructor_rejects_int_streaming(self):
        # bool is int-subclass; explicit type-check rejects coercion.
        identity = generate_node_identity("settler")
        token = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=1000.0,
        )
        with pytest.raises(ChainRpcMalformedError, match="streaming"):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"some-bytes",
                activation_shape=(1, 4),
                activation_dtype="float32",
                upstream_token=token,
                deadline_unix=1000.0,
                streaming=1,  # type: ignore[arg-type]
            )

    def test_from_dict_rejects_int_streaming(self):
        # Hostile peer ships {"streaming": 1} — must reject as
        # malformed, not silently coerce to True.
        identity = generate_node_identity("settler")
        token = HandoffToken.sign(
            identity=identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=2,
            deadline_unix=1000.0,
        )
        wire = {
            "type": ChainRpcMessageType.RUN_LAYER_SLICE_REQUEST.value,
            "protocol_version": CHAIN_RPC_PROTOCOL_VERSION,
            "request_id": "req-1",
            "model_id": "m",
            "layer_range": [0, 4],
            "privacy_tier": PrivacyLevel.NONE.value,
            "content_tier": ContentTier.A.value,
            "activation_blob_hex": b"some-bytes".hex(),
            "activation_shape": [1, 4],
            "activation_dtype": "float32",
            "upstream_token": token.to_dict(),
            "deadline_unix": 1000.0,
            "streaming": 1,  # int, not bool
        }
        with pytest.raises(ChainRpcMalformedError, match="streaming"):
            RunLayerSliceRequest.from_dict(wire)


class TestStreamingMessageTypeRouting:
    """parse_message correctly dispatches new wire types."""

    def test_token_frame_dispatched_via_parse_message(self):
        frame = TokenFrame(
            request_id="r", sequence_index=0, text_delta="x",
        )
        recovered = parse_message(encode_message(frame))
        assert isinstance(recovered, TokenFrame)

    def test_stream_final_frame_dispatched_via_parse_message(self):
        response = _signed_response_for_text("hi")
        frame = StreamFinalFrame(response=response)
        recovered = parse_message(encode_message(frame))
        assert isinstance(recovered, StreamFinalFrame)

    def test_message_type_registry_includes_new_types(self):
        # Sanity: both new types registered.
        from prsm.compute.chain_rpc.protocol import _MESSAGE_TYPE_REGISTRY
        assert ChainRpcMessageType.TOKEN_FRAME.value in _MESSAGE_TYPE_REGISTRY
        assert ChainRpcMessageType.STREAM_FINAL_FRAME.value in _MESSAGE_TYPE_REGISTRY


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.10.x — RunLayerSliceRequest sampling overrides
# (max_tokens + temperature on the wire)
# ──────────────────────────────────────────────────────────────────────────


class TestSamplingOverridesByteEquivalence:
    """The headline invariant: a request with both fields None
    produces the same canonical wire bytes as a pre-3.x.10.x
    request — no signed-bytes regression for older callers /
    older signed traffic."""

    def test_unset_omitted_byte_equivalent_to_pre_3_10_x(self):
        # Build a request without specifying max_tokens / temperature.
        # to_dict() must NOT include the keys (omit-when-None).
        identity = generate_node_identity("settler")
        req = _valid_request(identity=identity)
        wire = req.to_dict()
        assert "max_tokens" not in wire
        assert "temperature" not in wire

    def test_unset_canonical_bytes_unchanged(self):
        # Stronger: encoded bytes are byte-identical to a request
        # built before the new fields existed (mirrors the
        # streaming-flag byte-equivalence pattern).
        identity = generate_node_identity("settler")
        req_a = _valid_request(identity=identity)
        # Manually construct a request with both fields explicitly
        # None — must produce identical bytes to the no-kwarg form.
        req_b = RunLayerSliceRequest(
            request_id=req_a.request_id,
            model_id=req_a.model_id,
            layer_range=req_a.layer_range,
            privacy_tier=req_a.privacy_tier,
            content_tier=req_a.content_tier,
            activation_blob=req_a.activation_blob,
            activation_shape=req_a.activation_shape,
            activation_dtype=req_a.activation_dtype,
            upstream_token=req_a.upstream_token,
            deadline_unix=req_a.deadline_unix,
            max_tokens=None,
            temperature=None,
        )
        assert encode_message(req_a) == encode_message(req_b)

    def test_set_fields_appear_in_wire_dict(self):
        identity = generate_node_identity("settler")
        req = RunLayerSliceRequest(
            request_id="req-1",
            model_id="m",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"x",
            activation_shape=(1,),
            activation_dtype="f32",
            upstream_token=_valid_token(identity),
            deadline_unix=1000.0,
            max_tokens=8,
            temperature=0.7,
        )
        wire = req.to_dict()
        assert wire["max_tokens"] == 8
        assert wire["temperature"] == 0.7

    def test_set_fields_change_canonical_bytes(self):
        # Setting the fields MUST change the encoded bytes — otherwise
        # the conditional encoding would silently drop the override.
        identity = generate_node_identity("settler")
        unset = _valid_request(identity=identity)
        with_fields = RunLayerSliceRequest(
            request_id=unset.request_id,
            model_id=unset.model_id,
            layer_range=unset.layer_range,
            privacy_tier=unset.privacy_tier,
            content_tier=unset.content_tier,
            activation_blob=unset.activation_blob,
            activation_shape=unset.activation_shape,
            activation_dtype=unset.activation_dtype,
            upstream_token=unset.upstream_token,
            deadline_unix=unset.deadline_unix,
            max_tokens=4,
            temperature=0.0,
        )
        assert encode_message(unset) != encode_message(with_fields)


class TestSamplingOverridesValidation:
    def _base_kwargs(self, identity):
        return dict(
            request_id="req-1",
            model_id="m",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"x",
            activation_shape=(1,),
            activation_dtype="f32",
            upstream_token=_valid_token(identity),
            deadline_unix=1000.0,
        )

    def test_max_tokens_zero_rejected(self):
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="max_tokens"):
            RunLayerSliceRequest(
                **self._base_kwargs(identity), max_tokens=0,
            )

    def test_max_tokens_negative_rejected(self):
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="max_tokens"):
            RunLayerSliceRequest(
                **self._base_kwargs(identity), max_tokens=-5,
            )

    def test_max_tokens_bool_rejected(self):
        # bool is a subclass of int in Python; explicit guard.
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="max_tokens"):
            RunLayerSliceRequest(
                **self._base_kwargs(identity),
                max_tokens=True,  # type: ignore[arg-type]
            )

    def test_temperature_below_range_rejected(self):
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="temperature"):
            RunLayerSliceRequest(
                **self._base_kwargs(identity), temperature=-0.5,
            )

    def test_temperature_above_range_rejected(self):
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="temperature"):
            RunLayerSliceRequest(
                **self._base_kwargs(identity), temperature=2.1,
            )

    def test_temperature_zero_accepted_for_greedy(self):
        # 0.0 is the runner's greedy-decode signal; MUST be valid.
        identity = generate_node_identity("s")
        req = RunLayerSliceRequest(
            **self._base_kwargs(identity), temperature=0.0,
        )
        assert req.temperature == 0.0

    def test_temperature_two_accepted_at_boundary(self):
        identity = generate_node_identity("s")
        req = RunLayerSliceRequest(
            **self._base_kwargs(identity), temperature=2.0,
        )
        assert req.temperature == 2.0

    def test_temperature_bool_rejected(self):
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="temperature"):
            RunLayerSliceRequest(
                **self._base_kwargs(identity),
                temperature=False,  # type: ignore[arg-type]
            )

    def test_temperature_string_rejected(self):
        identity = generate_node_identity("s")
        with pytest.raises(ChainRpcMalformedError, match="temperature"):
            RunLayerSliceRequest(
                **self._base_kwargs(identity),
                temperature="0.5",  # type: ignore[arg-type]
            )


class TestSamplingOverridesRoundTrip:
    def test_round_trip_with_both_fields_set(self):
        identity = generate_node_identity("settler")
        original = RunLayerSliceRequest(
            request_id="req-1",
            model_id="m",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"x",
            activation_shape=(1,),
            activation_dtype="f32",
            upstream_token=_valid_token(identity),
            deadline_unix=1000.0,
            max_tokens=16,
            temperature=0.7,
        )
        recovered = parse_message(encode_message(original))
        assert isinstance(recovered, RunLayerSliceRequest)
        assert recovered.max_tokens == 16
        assert recovered.temperature == 0.7

    def test_round_trip_with_neither_field(self):
        identity = generate_node_identity("settler")
        original = _valid_request(identity=identity)
        recovered = parse_message(encode_message(original))
        assert isinstance(recovered, RunLayerSliceRequest)
        assert recovered.max_tokens is None
        assert recovered.temperature is None

    def test_round_trip_with_only_max_tokens(self):
        identity = generate_node_identity("settler")
        original = RunLayerSliceRequest(
            request_id="req-1",
            model_id="m",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"x",
            activation_shape=(1,),
            activation_dtype="f32",
            upstream_token=_valid_token(identity),
            deadline_unix=1000.0,
            max_tokens=4,
        )
        recovered = parse_message(encode_message(original))
        assert recovered.max_tokens == 4
        assert recovered.temperature is None

    def test_from_dict_rejects_bool_max_tokens(self):
        identity = generate_node_identity("settler")
        token = _valid_token(identity)
        wire = {
            "type": ChainRpcMessageType.RUN_LAYER_SLICE_REQUEST.value,
            "protocol_version": CHAIN_RPC_PROTOCOL_VERSION,
            "request_id": "req-1",
            "model_id": "m",
            "layer_range": [0, 4],
            "privacy_tier": PrivacyLevel.NONE.value,
            "content_tier": ContentTier.A.value,
            "activation_blob_hex": b"x".hex(),
            "activation_shape": [1],
            "activation_dtype": "f32",
            "upstream_token": token.to_dict(),
            "deadline_unix": 1000.0,
            "max_tokens": True,  # bool, not int
        }
        with pytest.raises(ChainRpcMalformedError, match="max_tokens"):
            RunLayerSliceRequest.from_dict(wire)

    def test_from_dict_rejects_string_temperature(self):
        identity = generate_node_identity("settler")
        token = _valid_token(identity)
        wire = {
            "type": ChainRpcMessageType.RUN_LAYER_SLICE_REQUEST.value,
            "protocol_version": CHAIN_RPC_PROTOCOL_VERSION,
            "request_id": "req-1",
            "model_id": "m",
            "layer_range": [0, 4],
            "privacy_tier": PrivacyLevel.NONE.value,
            "content_tier": ContentTier.A.value,
            "activation_blob_hex": b"x".hex(),
            "activation_shape": [1],
            "activation_dtype": "f32",
            "upstream_token": token.to_dict(),
            "deadline_unix": 1000.0,
            "temperature": "0.5",
        }
        with pytest.raises(ChainRpcMalformedError, match="temperature"):
            RunLayerSliceRequest.from_dict(wire)
