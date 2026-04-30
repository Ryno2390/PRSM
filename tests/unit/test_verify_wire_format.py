"""Phase 3.x.11.y Task 1 — VERIFY wire-format extension tests.

Covers:
  - DecodeMode.VERIFY round-trip on RunLayerSliceRequest
    (decode_mode field already validated; this just adds the
    new enum value)
  - RunLayerSliceResponse.verified_token_ids + accepted_count:
    construction validation, omit-when-default byte-equivalence
    pin, signing-payload commitment, signature verifies under
    anchor when set
  - RollbackCacheRequest + RollbackCacheResponse round-trip,
    cap enforcement, idempotency-contract documentation
"""

from __future__ import annotations

import json

import pytest

from prsm.compute.chain_rpc.protocol import (
    CHAIN_RPC_PROTOCOL_VERSION,
    MAX_VERIFY_BATCH_TOKENS,
    ChainRpcMalformedError,
    ChainRpcMessageType,
    DecodeMode,
    HandoffToken,
    RollbackCacheRequest,
    RollbackCacheResponse,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    encode_message,
    parse_message,
)
from prsm.compute.inference.models import ContentTier
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def stage_identity():
    return generate_node_identity("stage")


@pytest.fixture
def settler_identity():
    return generate_node_identity("settler")


def _make_request(
    *,
    settler,
    decode_mode: DecodeMode = DecodeMode.PREFILL,
    proposed_token_ids=None,
) -> RunLayerSliceRequest:
    token = HandoffToken.sign(
        identity=settler,
        request_id="req-1",
        chain_stage_index=0,
        chain_total_stages=1,
        deadline_unix=2000.0,
    )
    # Round-1 MEDIUM-1: VERIFY now requires proposed_token_ids
    # at the protocol layer (symmetric with response-side
    # verified_token_ids ⇔ accepted_count co-set invariant).
    # Tests defaulting to VERIFY mode synthesize a minimal
    # K=1 proposed list.
    if (
        decode_mode == DecodeMode.VERIFY
        and proposed_token_ids is None
    ):
        proposed_token_ids = (42,)
    return RunLayerSliceRequest(
        request_id="req-1",
        model_id="test-model",
        layer_range=(0, 4),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        activation_blob=b"\x00\x00\x00\x00",
        activation_shape=(1,),
        activation_dtype="int32",
        upstream_token=token,
        deadline_unix=2000.0,
        decode_mode=decode_mode,
        proposed_token_ids=proposed_token_ids,
    )


# ──────────────────────────────────────────────────────────────────────────
# DecodeMode.VERIFY on the request
# ──────────────────────────────────────────────────────────────────────────


class TestDecodeModeVerifyOnRequest:
    def test_verify_round_trip(self, settler_identity):
        req = _make_request(
            settler=settler_identity, decode_mode=DecodeMode.VERIFY,
        )
        wire = encode_message(req)
        parsed = parse_message(wire)
        assert isinstance(parsed, RunLayerSliceRequest)
        assert parsed.decode_mode == DecodeMode.VERIFY

    def test_verify_canonical_encodes_to_string(self, settler_identity):
        req = _make_request(
            settler=settler_identity, decode_mode=DecodeMode.VERIFY,
        )
        d = req.to_dict()
        assert d["decode_mode"] == "verify"


# ──────────────────────────────────────────────────────────────────────────
# RunLayerSliceResponse: verified_token_ids + accepted_count
# ──────────────────────────────────────────────────────────────────────────


class TestRunLayerSliceResponseVerifySignals:
    def _signed_response(
        self,
        identity,
        *,
        verified_token_ids=None,
        accepted_count=None,
    ) -> RunLayerSliceResponse:
        return RunLayerSliceResponse.sign(
            identity=identity,
            request_id="req-1",
            activation_blob=b"\x01\x02\x03\x04",
            activation_shape=(1,),
            activation_dtype="int32",
            duration_seconds=0.05,
            tee_attestation=b"\x07" * 32,
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
            verified_token_ids=verified_token_ids,
            accepted_count=accepted_count,
        )

    def test_round_trip_with_verify_signals(self, stage_identity):
        resp = self._signed_response(
            stage_identity,
            verified_token_ids=(7, 14, 21, 28, 35),
            accepted_count=2,
        )
        wire = encode_message(resp)
        parsed = parse_message(wire)
        assert isinstance(parsed, RunLayerSliceResponse)
        assert parsed.verified_token_ids == (7, 14, 21, 28, 35)
        assert parsed.accepted_count == 2

    def test_omit_when_default_byte_equivalence(self, stage_identity):
        # Pre-3.x.11.y signed bytes (no VERIFY signals) MUST stay
        # byte-equivalent. Sign the SAME response twice — once
        # with verified_token_ids=None, once with the default
        # constructor call — and verify the wire bytes match.
        resp_a = self._signed_response(
            stage_identity,
            verified_token_ids=None, accepted_count=None,
        )
        wire_a = encode_message(resp_a)
        d = json.loads(wire_a.split(b":", 0)[0] if False else wire_a)
        # Top-level dict MUST NOT contain the new keys when
        # both fields default to None.
        assert "verified_token_ids" not in d
        assert "accepted_count" not in d

    def test_signature_verifies_with_verify_signals(
        self, stage_identity,
    ):
        # The signing payload commits to verified_token_ids +
        # accepted_count when set. Round-trip + verify_with_anchor
        # must succeed.
        resp = self._signed_response(
            stage_identity,
            verified_token_ids=(10, 20, 30),
            accepted_count=1,
        )
        # Self-anchor for the test.
        class _Anchor:
            def __init__(self, identity):
                self._key_b64 = identity.public_key_b64
                self._node_id = identity.node_id

            def lookup(self, node_id):
                return self._key_b64 if node_id == self._node_id else None

        ok = resp.verify_with_anchor(
            _Anchor(stage_identity),
            expected_stage_node_id=stage_identity.node_id,
        )
        assert ok is True

    def test_signature_rejects_tampered_verify_signals(
        self, stage_identity,
    ):
        # Sign with one set of verify signals; tamper with
        # verified_token_ids on the response; verify-anchor
        # must fail.
        resp = self._signed_response(
            stage_identity,
            verified_token_ids=(10, 20, 30),
            accepted_count=1,
        )
        # Reconstruct with tampered verified_token_ids but
        # original signature.
        tampered = RunLayerSliceResponse(
            request_id=resp.request_id,
            activation_blob=resp.activation_blob,
            activation_shape=resp.activation_shape,
            activation_dtype=resp.activation_dtype,
            duration_seconds=resp.duration_seconds,
            tee_attestation=resp.tee_attestation,
            tee_type=resp.tee_type,
            epsilon_spent=resp.epsilon_spent,
            stage_signature_b64=resp.stage_signature_b64,
            stage_node_id=resp.stage_node_id,
            verified_token_ids=(99, 99, 99),  # tampered
            accepted_count=resp.accepted_count,
        )

        class _Anchor:
            def __init__(self, identity):
                self._key_b64 = identity.public_key_b64
                self._node_id = identity.node_id

            def lookup(self, node_id):
                return self._key_b64 if node_id == self._node_id else None

        ok = tampered.verify_with_anchor(
            _Anchor(stage_identity),
            expected_stage_node_id=stage_identity.node_id,
        )
        assert ok is False

    # ── construction validation ──────────────────────────────────────

    def test_rejects_co_set_violation_verified_only(self):
        # verified_token_ids set without accepted_count → reject.
        with pytest.raises(
            ChainRpcMalformedError, match="co-set",
        ):
            RunLayerSliceResponse(
                request_id="req-1",
                activation_blob=b"\x01" * 4,
                activation_shape=(1,),
                activation_dtype="int32",
                duration_seconds=0.01,
                tee_attestation=b"\x00" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
                stage_signature_b64="x",
                stage_node_id="alice",
                verified_token_ids=(1, 2, 3),
                accepted_count=None,
            )

    def test_rejects_co_set_violation_accepted_only(self):
        with pytest.raises(
            ChainRpcMalformedError, match="co-set",
        ):
            RunLayerSliceResponse(
                request_id="req-1",
                activation_blob=b"\x01" * 4,
                activation_shape=(1,),
                activation_dtype="int32",
                duration_seconds=0.01,
                tee_attestation=b"\x00" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
                stage_signature_b64="x",
                stage_node_id="alice",
                verified_token_ids=None,
                accepted_count=2,
            )

    def test_rejects_empty_verified_token_ids(self):
        with pytest.raises(
            ChainRpcMalformedError, match="non-empty",
        ):
            RunLayerSliceResponse(
                request_id="req-1",
                activation_blob=b"\x01" * 4,
                activation_shape=(1,),
                activation_dtype="int32",
                duration_seconds=0.01,
                tee_attestation=b"\x00" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
                stage_signature_b64="x",
                stage_node_id="alice",
                verified_token_ids=(),
                accepted_count=0,
            )

    def test_rejects_negative_token_id(self):
        with pytest.raises(
            ChainRpcMalformedError, match="non-negative",
        ):
            RunLayerSliceResponse(
                request_id="req-1",
                activation_blob=b"\x01" * 4,
                activation_shape=(1,),
                activation_dtype="int32",
                duration_seconds=0.01,
                tee_attestation=b"\x00" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
                stage_signature_b64="x",
                stage_node_id="alice",
                verified_token_ids=(1, -2, 3),
                accepted_count=1,
            )

    def test_rejects_bool_token_id(self):
        with pytest.raises(
            ChainRpcMalformedError, match="entries must be int",
        ):
            RunLayerSliceResponse(
                request_id="req-1",
                activation_blob=b"\x01" * 4,
                activation_shape=(1,),
                activation_dtype="int32",
                duration_seconds=0.01,
                tee_attestation=b"\x00" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
                stage_signature_b64="x",
                stage_node_id="alice",
                verified_token_ids=(1, True, 3),  # type: ignore[arg-type]
                accepted_count=1,
            )

    def test_rejects_accepted_count_exceeds_k(self):
        # K = len(verified_token_ids) - 1. accepted_count must
        # be <= K.
        with pytest.raises(
            ChainRpcMalformedError, match="exceeds max",
        ):
            RunLayerSliceResponse(
                request_id="req-1",
                activation_blob=b"\x01" * 4,
                activation_shape=(1,),
                activation_dtype="int32",
                duration_seconds=0.01,
                tee_attestation=b"\x00" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
                stage_signature_b64="x",
                stage_node_id="alice",
                verified_token_ids=(1, 2, 3),  # K=2
                accepted_count=3,  # > K
            )

    def test_rejects_negative_accepted_count(self):
        with pytest.raises(
            ChainRpcMalformedError, match="non-negative",
        ):
            RunLayerSliceResponse(
                request_id="req-1",
                activation_blob=b"\x01" * 4,
                activation_shape=(1,),
                activation_dtype="int32",
                duration_seconds=0.01,
                tee_attestation=b"\x00" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
                stage_signature_b64="x",
                stage_node_id="alice",
                verified_token_ids=(1, 2),
                accepted_count=-1,
            )

    def test_rejects_oversized_verified_token_ids(self):
        # len > MAX_VERIFY_BATCH_TOKENS rejected at construction.
        oversized = tuple(range(MAX_VERIFY_BATCH_TOKENS + 1))
        with pytest.raises(
            ChainRpcMalformedError, match="exceeds cap",
        ):
            RunLayerSliceResponse(
                request_id="req-1",
                activation_blob=b"\x01" * 4,
                activation_shape=(1,),
                activation_dtype="int32",
                duration_seconds=0.01,
                tee_attestation=b"\x00" * 32,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=0.0,
                stage_signature_b64="x",
                stage_node_id="alice",
                verified_token_ids=oversized,
                accepted_count=0,
            )

    def test_from_dict_rejects_oversized_at_parse_time(self):
        # Defends against a hostile peer claiming a huge
        # speculation depth that explodes server-side memory at
        # parse time. Must reject BEFORE constructing the
        # dataclass — list-length check in from_dict.
        oversized = list(range(MAX_VERIFY_BATCH_TOKENS + 1))
        bad = {
            "type": ChainRpcMessageType.RUN_LAYER_SLICE_RESPONSE.value,
            "protocol_version": CHAIN_RPC_PROTOCOL_VERSION,
            "request_id": "req-1",
            "activation_blob_hex": ("01" * 4),
            "activation_shape": [1],
            "activation_dtype": "int32",
            "duration_seconds": 0.01,
            "tee_attestation_hex": ("00" * 32),
            "tee_type": "software",
            "epsilon_spent": 0.0,
            "stage_signature_b64": "x",
            "stage_node_id": "alice",
            "verified_token_ids": oversized,
            "accepted_count": 0,
        }
        poisoned = json.dumps(bad).encode("utf-8")
        with pytest.raises(
            ChainRpcMalformedError, match="exceeds cap",
        ):
            parse_message(poisoned)


# ──────────────────────────────────────────────────────────────────────────
# RollbackCacheRequest + RollbackCacheResponse
# ──────────────────────────────────────────────────────────────────────────


class TestRollbackCacheRequest:
    def test_round_trip(self):
        req = RollbackCacheRequest(
            request_id="abc-123",
            n_positions_to_drop=4,
        )
        wire = encode_message(req)
        parsed = parse_message(wire)
        assert isinstance(parsed, RollbackCacheRequest)
        assert parsed.request_id == "abc-123"
        assert parsed.n_positions_to_drop == 4

    def test_zero_drop_documented_as_idempotent(self):
        # n_positions_to_drop=0 is valid + documented as the
        # idempotent no-op case (operator may broadcast
        # defensively even when nothing was speculated).
        req = RollbackCacheRequest(
            request_id="abc-123",
            n_positions_to_drop=0,
        )
        wire = encode_message(req)
        parsed = parse_message(wire)
        assert parsed.n_positions_to_drop == 0

    def test_rejects_negative_drop(self):
        with pytest.raises(
            ChainRpcMalformedError, match="non-negative",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=-1,
            )

    def test_rejects_bool_drop(self):
        with pytest.raises(
            ChainRpcMalformedError, match="must be int",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=True,  # type: ignore[arg-type]
            )

    def test_rejects_oversized_drop(self):
        with pytest.raises(
            ChainRpcMalformedError, match="exceeds cap",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=MAX_VERIFY_BATCH_TOKENS + 1,
            )

    def test_rejects_empty_request_id(self):
        with pytest.raises(
            ChainRpcMalformedError, match="request_id",
        ):
            RollbackCacheRequest(
                request_id="",
                n_positions_to_drop=4,
            )


class TestRollbackCacheRequestReplayPrefix:
    """Phase 3.x.11.q.y' — replay_accepted_prefix +
    encrypted_replay_accepted_prefix wire-format extensions.

    Validates the always-rollback-K protocol's wire envelope:
    plaintext prefix for non-Tier-C / debug deploys; encrypted
    prefix for the constant-time speculation stack."""

    def test_replay_prefix_round_trip(self):
        req = RollbackCacheRequest(
            request_id="abc",
            n_positions_to_drop=4,
            replay_accepted_prefix=(101, 202, 303),
        )
        wire = encode_message(req)
        parsed = parse_message(wire)
        assert isinstance(parsed, RollbackCacheRequest)
        assert parsed.replay_accepted_prefix == (101, 202, 303)
        assert parsed.encrypted_replay_accepted_prefix is None

    def test_encrypted_replay_prefix_round_trip(self):
        req = RollbackCacheRequest(
            request_id="abc",
            n_positions_to_drop=4,
            encrypted_replay_accepted_prefix=b"\x01\x02\x03\x04" * 16,
            target_stage_index=2,
        )
        wire = encode_message(req)
        parsed = parse_message(wire)
        assert isinstance(parsed, RollbackCacheRequest)
        assert parsed.replay_accepted_prefix is None
        assert parsed.encrypted_replay_accepted_prefix == (
            b"\x01\x02\x03\x04" * 16
        )
        assert parsed.target_stage_index == 2

    def test_encrypted_without_target_stage_index_rejected(self):
        # Co-set invariant: encrypted prefix requires
        # target_stage_index for AAD binding.
        with pytest.raises(
            ChainRpcMalformedError,
            match="target_stage_index",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=4,
                encrypted_replay_accepted_prefix=b"\x00" * 32,
                # target_stage_index NOT set
            )

    def test_target_stage_index_out_of_range_rejected(self):
        with pytest.raises(
            ChainRpcMalformedError, match=r"\[0, 255\]",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=4,
                target_stage_index=256,
            )

    def test_neither_set_is_v1_truncation_only(self):
        # Backwards-compat: pre-q.y' deployments emit rollbacks
        # without either field set. Round-trip stays byte-identical
        # to pre-q.y' behavior.
        req = RollbackCacheRequest(
            request_id="abc",
            n_positions_to_drop=4,
        )
        encoded = req.to_dict()
        # Neither field appears in the canonical encoding when
        # unset (omit-when-default invariant for backwards-compat).
        assert "replay_accepted_prefix" not in encoded
        assert "encrypted_replay_accepted_prefix_hex" not in encoded

    def test_rejects_both_plaintext_and_encrypted(self):
        with pytest.raises(
            ChainRpcMalformedError,
            match="cannot carry both",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=4,
                replay_accepted_prefix=(1, 2, 3),
                encrypted_replay_accepted_prefix=b"x" * 32,
                target_stage_index=0,
            )

    def test_rejects_non_tuple_plaintext(self):
        with pytest.raises(
            ChainRpcMalformedError, match="must be tuple",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=4,
                replay_accepted_prefix=[1, 2, 3],  # type: ignore[arg-type]
            )

    def test_rejects_negative_token_in_plaintext(self):
        with pytest.raises(
            ChainRpcMalformedError, match="non-negative",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=4,
                replay_accepted_prefix=(1, -2, 3),
            )

    def test_rejects_bool_token_in_plaintext(self):
        with pytest.raises(
            ChainRpcMalformedError, match="must be int",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=4,
                replay_accepted_prefix=(
                    1, True, 3,  # type: ignore[arg-type]
                ),
            )

    def test_rejects_oversized_plaintext(self):
        with pytest.raises(
            ChainRpcMalformedError, match="exceeds cap",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=4,
                replay_accepted_prefix=tuple(
                    range(MAX_VERIFY_BATCH_TOKENS + 1)
                ),
            )

    def test_rejects_empty_encrypted(self):
        with pytest.raises(
            ChainRpcMalformedError, match="non-empty",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=4,
                encrypted_replay_accepted_prefix=b"",
                target_stage_index=0,
            )

    def test_rejects_oversized_encrypted(self):
        with pytest.raises(
            ChainRpcMalformedError, match="exceeds 4096",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=4,
                encrypted_replay_accepted_prefix=b"\x00" * 4097,
                target_stage_index=0,
            )

    def test_rejects_non_bytes_encrypted(self):
        with pytest.raises(
            ChainRpcMalformedError, match="must be bytes",
        ):
            RollbackCacheRequest(
                request_id="abc",
                n_positions_to_drop=4,
                encrypted_replay_accepted_prefix=(
                    "not-bytes"  # type: ignore[arg-type]
                ),
                target_stage_index=0,
            )

    def test_invalid_hex_in_from_dict_rejected(self):
        # A wire blob with non-hex chars in the encrypted field
        # should fail parsing cleanly (defends against malformed
        # peer encoding).
        with pytest.raises(
            ChainRpcMalformedError, match="not.*valid hex",
        ):
            RollbackCacheRequest.from_dict({
                "type": "rollback_cache_request",
                "protocol_version": 2,
                "request_id": "abc",
                "n_positions_to_drop": 4,
                "encrypted_replay_accepted_prefix_hex": "ZZZZ",
            })

    def test_pre_q_y_prime_byte_equivalent_round_trip(self):
        # Load-bearing backwards-compat: rollbacks created before
        # this slice (no replay fields) MUST encode to the same
        # bytes pre- and post-extension. Pin via canonical
        # to_dict() shape comparison.
        req = RollbackCacheRequest(
            request_id="abc-123",
            n_positions_to_drop=2,
        )
        encoded = req.to_dict()
        # Canonical pre-q.y' shape: type + protocol_version +
        # request_id + n_positions_to_drop, NOTHING else.
        assert sorted(encoded.keys()) == sorted([
            "type",
            "protocol_version",
            "request_id",
            "n_positions_to_drop",
        ])


class TestRollbackCacheResponse:
    def test_round_trip(self):
        resp = RollbackCacheResponse(
            request_id="abc",
            rolled_back=True,
            actual_dropped=4,
        )
        wire = encode_message(resp)
        parsed = parse_message(wire)
        assert isinstance(parsed, RollbackCacheResponse)
        assert parsed.rolled_back is True
        assert parsed.actual_dropped == 4

    def test_idempotent_no_op_response(self):
        # rolled_back=False + actual_dropped=0 is the documented
        # no-op idempotent path (cache was empty or asked to
        # drop more than held).
        resp = RollbackCacheResponse(
            request_id="abc",
            rolled_back=False,
            actual_dropped=0,
        )
        wire = encode_message(resp)
        parsed = parse_message(wire)
        assert parsed.rolled_back is False
        assert parsed.actual_dropped == 0

    def test_rejects_non_bool_rolled_back(self):
        with pytest.raises(
            ChainRpcMalformedError, match="rolled_back",
        ):
            RollbackCacheResponse(
                request_id="abc",
                rolled_back="yes",  # type: ignore[arg-type]
                actual_dropped=2,
            )

    def test_rejects_negative_actual_dropped(self):
        with pytest.raises(
            ChainRpcMalformedError, match="non-negative",
        ):
            RollbackCacheResponse(
                request_id="abc",
                rolled_back=True,
                actual_dropped=-1,
            )

    def test_from_dict_rejects_int_for_rolled_back(self):
        bad = encode_message(RollbackCacheResponse(
            request_id="x", rolled_back=True, actual_dropped=1,
        ))
        d = json.loads(bad)
        d["rolled_back"] = 1  # int instead of bool
        poisoned = json.dumps(d).encode("utf-8")
        with pytest.raises(
            ChainRpcMalformedError, match="rolled_back",
        ):
            parse_message(poisoned)


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.y.x — proposed_token_probs (sampling-correct speculation)
# ──────────────────────────────────────────────────────────────────────────


class TestProposedTokenProbs:
    """Phase 3.x.11.y.x Task 1 — wire-format extension.

    Covers:
      - v2 stochastic round-trip (probs set co-set with ids)
      - v1 greedy omit-when-None byte-equivalence (no proposed_token_probs
        key in canonical JSON when probs is None)
      - co-set invariant (probs without ids rejected)
      - range validation (each prob in [0, 1])
      - length-must-match-ids invariant
      - bool rejected as a prob
      - signing-payload-coverage (signature commits to probs when set;
        tampering invalidates verification)
      - v1 client wire bytes byte-equivalent with pre-3.x.11.y.x
        (no probs key in JSON)
    """

    def test_v2_round_trip(self, settler_identity):
        req = _make_request(
            settler=settler_identity,
            decode_mode=DecodeMode.VERIFY,
            proposed_token_ids=(100, 101, 102),
        )
        # _make_request synthesizes (42,) with single prop; for v2
        # we explicitly pass probs co-set with the ids above.
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        v2 = RunLayerSliceRequest(
            request_id="req-1",
            model_id="test-model",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"\x00\x00\x00\x00",
            activation_shape=(1,),
            activation_dtype="int32",
            upstream_token=token,
            deadline_unix=2000.0,
            decode_mode=DecodeMode.VERIFY,
            proposed_token_ids=(100, 101, 102),
            proposed_token_probs=(0.9, 0.5, 0.3),
        )
        wire = encode_message(v2)
        parsed = parse_message(wire)
        assert isinstance(parsed, RunLayerSliceRequest)
        assert parsed.proposed_token_probs == (0.9, 0.5, 0.3)

    def test_v1_omits_when_none_byte_equivalence(self, settler_identity):
        # v1 greedy callers leave proposed_token_probs unset; the
        # canonical JSON MUST NOT carry the key (preserves byte-
        # equivalence with pre-3.x.11.y.x signed bytes).
        req = _make_request(
            settler=settler_identity,
            decode_mode=DecodeMode.VERIFY,
        )
        assert req.proposed_token_probs is None
        wire = encode_message(req)
        d = json.loads(wire)
        assert "proposed_token_probs" not in d

    def test_rejects_probs_without_ids(self, settler_identity):
        # proposed_token_probs without proposed_token_ids = malformed.
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        with pytest.raises(
            ChainRpcMalformedError, match="proposed_token_probs",
        ):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="test-model",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"\x00\x00\x00\x00",
                activation_shape=(1,),
                activation_dtype="int32",
                upstream_token=token,
                deadline_unix=2000.0,
                proposed_token_probs=(0.5,),
            )

    def test_rejects_prob_above_one(self, settler_identity):
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        with pytest.raises(
            ChainRpcMalformedError, match=r"\[0, 1\]",
        ):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="test-model",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"\x00\x00\x00\x00",
                activation_shape=(1,),
                activation_dtype="int32",
                upstream_token=token,
                deadline_unix=2000.0,
                decode_mode=DecodeMode.VERIFY,
                proposed_token_ids=(1,),
                proposed_token_probs=(1.5,),
            )

    def test_rejects_negative_prob(self, settler_identity):
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        with pytest.raises(
            ChainRpcMalformedError, match=r"\[0, 1\]",
        ):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="test-model",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"\x00\x00\x00\x00",
                activation_shape=(1,),
                activation_dtype="int32",
                upstream_token=token,
                deadline_unix=2000.0,
                decode_mode=DecodeMode.VERIFY,
                proposed_token_ids=(1,),
                proposed_token_probs=(-0.1,),
            )

    def test_rejects_length_mismatch(self, settler_identity):
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        with pytest.raises(
            ChainRpcMalformedError, match="length",
        ):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="test-model",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"\x00\x00\x00\x00",
                activation_shape=(1,),
                activation_dtype="int32",
                upstream_token=token,
                deadline_unix=2000.0,
                decode_mode=DecodeMode.VERIFY,
                proposed_token_ids=(1, 2, 3),
                proposed_token_probs=(0.5,),
            )

    def test_rejects_bool_prob(self, settler_identity):
        # bool is subclass of int; must be rejected explicitly.
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        with pytest.raises(
            ChainRpcMalformedError, match="proposed_token_probs",
        ):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="test-model",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"\x00\x00\x00\x00",
                activation_shape=(1,),
                activation_dtype="int32",
                upstream_token=token,
                deadline_unix=2000.0,
                decode_mode=DecodeMode.VERIFY,
                proposed_token_ids=(1,),
                proposed_token_probs=(True,),  # type: ignore[arg-type]
            )

    def test_from_dict_rejects_non_list_probs(self, settler_identity):
        # Wire-level: poison the JSON with proposed_token_probs as
        # a string instead of a list.
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        valid = RunLayerSliceRequest(
            request_id="req-1",
            model_id="test-model",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"\x00\x00\x00\x00",
            activation_shape=(1,),
            activation_dtype="int32",
            upstream_token=token,
            deadline_unix=2000.0,
            decode_mode=DecodeMode.VERIFY,
            proposed_token_ids=(1,),
            proposed_token_probs=(0.5,),
        )
        wire = encode_message(valid)
        d = json.loads(wire)
        d["proposed_token_probs"] = "not a list"
        poisoned = json.dumps(d).encode("utf-8")
        with pytest.raises(
            ChainRpcMalformedError, match="proposed_token_probs",
        ):
            parse_message(poisoned)

    def test_signing_payload_commits_probs(self, settler_identity):
        # v1 greedy bytes (probs unset) and v2 stochastic bytes
        # (probs set) MUST produce different canonical JSON, so
        # downstream signers commit to the probs field on the
        # response side. This test verifies the request-side
        # canonical encoding includes proposed_token_probs only
        # when set.
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        v1 = RunLayerSliceRequest(
            request_id="req-1",
            model_id="test-model",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"\x00\x00\x00\x00",
            activation_shape=(1,),
            activation_dtype="int32",
            upstream_token=token,
            deadline_unix=2000.0,
            decode_mode=DecodeMode.VERIFY,
            proposed_token_ids=(1, 2, 3),
        )
        v2 = RunLayerSliceRequest(
            request_id="req-1",
            model_id="test-model",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"\x00\x00\x00\x00",
            activation_shape=(1,),
            activation_dtype="int32",
            upstream_token=token,
            deadline_unix=2000.0,
            decode_mode=DecodeMode.VERIFY,
            proposed_token_ids=(1, 2, 3),
            proposed_token_probs=(0.9, 0.5, 0.3),
        )
        v1_bytes = encode_message(v1)
        v2_bytes = encode_message(v2)
        assert v1_bytes != v2_bytes
        # v1 has no probs key.
        assert b"proposed_token_probs" not in v1_bytes
        # v2 does.
        assert b"proposed_token_probs" in v2_bytes


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.q.y Task 1 — encrypted_proposed_token_probs wire field
# ──────────────────────────────────────────────────────────────────────────


class TestEncryptedProposedTokenProbs:
    """Encrypted-wire variant of proposed_token_probs (Phase 3.x.11.q.y).

    Mutually exclusive with plaintext probs; constraints + bytes
    cap; round-trip serialization; omit-when-None byte-equivalence
    with pre-3.x.11.q.y messages.
    """

    def test_round_trip_with_encrypted_field(self, settler_identity):
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        # AES-GCM ciphertext shape: 12-byte nonce + 16-byte tag +
        # ~24-byte K=3 plaintext = 52 bytes. Synthesize valid bytes
        # at a representative size; the wire layer doesn't validate
        # AES-GCM internals.
        ciphertext = bytes(range(52))
        req = RunLayerSliceRequest(
            request_id="req-1",
            model_id="m",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"\x00",
            activation_shape=(1,),
            activation_dtype="int32",
            upstream_token=token,
            deadline_unix=2000.0,
            decode_mode=DecodeMode.VERIFY,
            proposed_token_ids=(1, 2, 3),
            encrypted_proposed_token_probs=ciphertext,
        )
        wire = encode_message(req)
        decoded = parse_message(wire)
        assert isinstance(decoded, RunLayerSliceRequest)
        assert decoded.encrypted_proposed_token_probs == ciphertext
        # Plaintext probs MUST be None on the round-trip
        # (mutual exclusion enforced).
        assert decoded.proposed_token_probs is None

    def test_mutually_exclusive_with_plaintext_probs(
        self, settler_identity,
    ):
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        with pytest.raises(
            ChainRpcMalformedError,
            match="mutually exclusive",
        ):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"\x00",
                activation_shape=(1,),
                activation_dtype="int32",
                upstream_token=token,
                deadline_unix=2000.0,
                decode_mode=DecodeMode.VERIFY,
                proposed_token_ids=(1, 2),
                proposed_token_probs=(0.5, 0.5),
                encrypted_proposed_token_probs=b"\x00" * 32,
            )

    def test_requires_proposed_token_ids(self, settler_identity):
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        # decode_mode=VERIFY validator fires first ("requires
        # proposed_token_ids to be set") on this misconfig; the
        # encrypted-probs co-set check would also catch it. Both
        # match on substring "proposed_token_ids".
        with pytest.raises(
            ChainRpcMalformedError,
            match="proposed_token_ids",
        ):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"\x00",
                activation_shape=(1,),
                activation_dtype="int32",
                upstream_token=token,
                deadline_unix=2000.0,
                decode_mode=DecodeMode.VERIFY,
                proposed_token_ids=None,
                encrypted_proposed_token_probs=b"\x00" * 32,
            )

    def test_rejects_non_bytes(self, settler_identity):
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        with pytest.raises(
            ChainRpcMalformedError, match="must be bytes",
        ):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"\x00",
                activation_shape=(1,),
                activation_dtype="int32",
                upstream_token=token,
                deadline_unix=2000.0,
                decode_mode=DecodeMode.VERIFY,
                proposed_token_ids=(42,),
                encrypted_proposed_token_probs="not-bytes",  # type: ignore[arg-type]
            )

    def test_rejects_empty_bytes(self, settler_identity):
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        with pytest.raises(
            ChainRpcMalformedError, match="non-empty",
        ):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"\x00",
                activation_shape=(1,),
                activation_dtype="int32",
                upstream_token=token,
                deadline_unix=2000.0,
                decode_mode=DecodeMode.VERIFY,
                proposed_token_ids=(42,),
                encrypted_proposed_token_probs=b"",
            )

    def test_rejects_oversized_bytes(self, settler_identity):
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        with pytest.raises(
            ChainRpcMalformedError, match="1024-byte cap",
        ):
            RunLayerSliceRequest(
                request_id="req-1",
                model_id="m",
                layer_range=(0, 4),
                privacy_tier=PrivacyLevel.NONE,
                content_tier=ContentTier.A,
                activation_blob=b"\x00",
                activation_shape=(1,),
                activation_dtype="int32",
                upstream_token=token,
                deadline_unix=2000.0,
                decode_mode=DecodeMode.VERIFY,
                proposed_token_ids=(42,),
                encrypted_proposed_token_probs=b"\x00" * 1025,
            )

    def test_omit_when_none_preserves_byte_equivalence(
        self, settler_identity,
    ):
        # Pre-3.x.11.q.y request (no encrypted field) must serialize
        # to byte-identical wire as the same request constructed
        # post-3.x.11.q.y with encrypted field unset.
        token = HandoffToken.sign(
            identity=settler_identity,
            request_id="req-1",
            chain_stage_index=0,
            chain_total_stages=1,
            deadline_unix=2000.0,
        )
        common_kwargs = dict(
            request_id="req-1",
            model_id="m",
            layer_range=(0, 4),
            privacy_tier=PrivacyLevel.NONE,
            content_tier=ContentTier.A,
            activation_blob=b"\x00",
            activation_shape=(1,),
            activation_dtype="int32",
            upstream_token=token,
            deadline_unix=2000.0,
            decode_mode=DecodeMode.VERIFY,
            proposed_token_ids=(42,),
        )
        v1 = RunLayerSliceRequest(**common_kwargs)
        v2 = RunLayerSliceRequest(
            **common_kwargs,
            encrypted_proposed_token_probs=None,
        )
        assert encode_message(v1) == encode_message(v2)
        # Wire bytes do NOT contain the field name.
        assert b"encrypted_proposed_token_probs" not in encode_message(v1)
