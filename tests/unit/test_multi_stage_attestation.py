"""Phase 3.x.7 Task 5 — multi_stage_attestation unit tests.

Coverage matches design plan §4 Task 5 acceptance:
  - Multi-stage envelope round-trip
  - Worst-case TEE type policy (SOFTWARE drags hardware down)
  - Single-stage opaque bytes treated as back-compat (no envelope)
  - Magic-prefix detection
  - Malformed envelope (magic-prefixed but corrupt) raises cleanly
  - verify_stage_attestations API for single + multi-stage
  - Receipt-level integration: receipt with multi-stage envelope
    round-trips + signature still verifies (back-compat with the
    Phase 3.x.1 signing payload)
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

import pytest

from prsm.compute.inference.models import (
    ContentTier,
    InferenceReceipt,
)
from prsm.compute.inference.multi_stage_attestation import (
    MULTI_STAGE_ATTESTATION_VERSION,
    MULTI_STAGE_MAGIC_PREFIX,
    MultiStageAttestationError,
    MultiStageMalformedError,
    StageAttestation,
    StageVerificationResult,
    decode_multi_stage_attestation,
    encode_multi_stage_attestation,
    is_hardware_tee,
    is_multi_stage_attestation,
    verify_stage_attestations,
    worst_case_tee_type,
)
from prsm.compute.inference.receipt import (
    sign_receipt,
    verify_receipt,
    verify_stage_attestations as receipt_verify_stages,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# StageAttestation construction
# ──────────────────────────────────────────────────────────────────────────


class TestStageAttestationConstruction:
    def test_happy_path(self):
        s = StageAttestation(
            stage_index=0,
            stage_node_id="alice",
            tee_type=TEEType.SGX,
            attestation=b"\x01" * 32,
        )
        assert s.stage_index == 0
        assert s.tee_type == TEEType.SGX

    def test_rejects_negative_stage_index(self):
        with pytest.raises(MultiStageMalformedError, match="stage_index"):
            StageAttestation(
                stage_index=-1,
                stage_node_id="x",
                tee_type=TEEType.SGX,
                attestation=b"x",
            )

    def test_rejects_empty_node_id(self):
        with pytest.raises(MultiStageMalformedError, match="stage_node_id"):
            StageAttestation(
                stage_index=0,
                stage_node_id="",
                tee_type=TEEType.SGX,
                attestation=b"x",
            )

    def test_rejects_non_tee_type_enum(self):
        with pytest.raises(MultiStageMalformedError, match="tee_type"):
            StageAttestation(
                stage_index=0,
                stage_node_id="x",
                tee_type="sgx",  # type: ignore[arg-type]
                attestation=b"x",
            )

    def test_rejects_non_bytes_attestation(self):
        with pytest.raises(MultiStageMalformedError, match="attestation"):
            StageAttestation(
                stage_index=0,
                stage_node_id="x",
                tee_type=TEEType.SGX,
                attestation="not-bytes",  # type: ignore[arg-type]
            )

    def test_dict_round_trip(self):
        s = StageAttestation(
            stage_index=2,
            stage_node_id="alice",
            tee_type=TEEType.TDX,
            attestation=b"\xaa\xbb",
        )
        recovered = StageAttestation.from_dict(s.to_dict())
        assert recovered == s

    def test_from_dict_rejects_missing_fields(self):
        with pytest.raises(MultiStageMalformedError):
            StageAttestation.from_dict({})

    def test_from_dict_rejects_bad_hex(self):
        with pytest.raises(MultiStageMalformedError):
            StageAttestation.from_dict({
                "stage_index": 0,
                "stage_node_id": "x",
                "tee_type": "sgx",
                "attestation_hex": "ZZZZ",
            })

    def test_from_dict_rejects_bad_tee_type(self):
        with pytest.raises(MultiStageMalformedError):
            StageAttestation.from_dict({
                "stage_index": 0,
                "stage_node_id": "x",
                "tee_type": "bogus_enum",
                "attestation_hex": "00",
            })


# ──────────────────────────────────────────────────────────────────────────
# Envelope codec
# ──────────────────────────────────────────────────────────────────────────


def _stages(*specs):
    """Helper: builds a list of StageAttestation from (idx, name, tee, bytes)."""
    return [
        StageAttestation(
            stage_index=idx,
            stage_node_id=name,
            tee_type=tee,
            attestation=blob,
        )
        for idx, name, tee, blob in specs
    ]


class TestEnvelopeRoundTrip:
    def test_single_stage_round_trip(self):
        stages = _stages((0, "alice", TEEType.SGX, b"\x01" * 32))
        blob = encode_multi_stage_attestation(stages)
        recovered = decode_multi_stage_attestation(blob)
        assert recovered == stages

    def test_multi_stage_round_trip(self):
        stages = _stages(
            (0, "alice", TEEType.SGX, b"\x01" * 32),
            (1, "bob", TEEType.TDX, b"\x02" * 32),
            (2, "charlie", TEEType.SEV, b"\x03" * 32),
        )
        blob = encode_multi_stage_attestation(stages)
        recovered = decode_multi_stage_attestation(blob)
        assert recovered == stages

    def test_envelope_starts_with_magic_prefix(self):
        stages = _stages((0, "alice", TEEType.SGX, b"\x01"))
        blob = encode_multi_stage_attestation(stages)
        assert blob.startswith(MULTI_STAGE_MAGIC_PREFIX)

    def test_envelope_body_is_canonical_json(self):
        stages = _stages(
            (1, "bob", TEEType.TDX, b"\xff"),
            (0, "alice", TEEType.SGX, b"\xee"),
        )
        blob = encode_multi_stage_attestation(stages)
        body = blob[len(MULTI_STAGE_MAGIC_PREFIX):]
        decoded = json.loads(body)
        assert decoded["version"] == MULTI_STAGE_ATTESTATION_VERSION
        # Keys are sorted alphabetically (canonical JSON contract).
        keys = list(decoded.keys())
        assert keys == sorted(keys)

    def test_encode_rejects_empty_stage_list(self):
        with pytest.raises(MultiStageAttestationError, match="at least one"):
            encode_multi_stage_attestation([])

    def test_decode_returns_none_for_unprefixed_bytes(self):
        # Single-stage opaque attestation — no envelope.
        opaque = b"\x01" * 32
        assert decode_multi_stage_attestation(opaque) is None

    def test_decode_returns_none_for_empty_bytes(self):
        assert decode_multi_stage_attestation(b"") is None

    def test_decode_raises_on_malformed_envelope(self):
        # Magic prefix present but body is not JSON.
        bad = MULTI_STAGE_MAGIC_PREFIX + b"not-json"
        with pytest.raises(MultiStageMalformedError, match="JSON parse"):
            decode_multi_stage_attestation(bad)

    def test_decode_raises_on_wrong_version(self):
        body = json.dumps({
            "version": 999,
            "stages": [{"stage_index": 0, "stage_node_id": "x",
                        "tee_type": "sgx", "attestation_hex": "00"}],
        }).encode("utf-8")
        bad = MULTI_STAGE_MAGIC_PREFIX + body
        with pytest.raises(MultiStageMalformedError, match="version"):
            decode_multi_stage_attestation(bad)

    def test_decode_raises_on_missing_stages(self):
        body = json.dumps({"version": MULTI_STAGE_ATTESTATION_VERSION}).encode("utf-8")
        bad = MULTI_STAGE_MAGIC_PREFIX + body
        with pytest.raises(MultiStageMalformedError, match="stages"):
            decode_multi_stage_attestation(bad)

    def test_decode_raises_on_empty_stages(self):
        body = json.dumps({
            "version": MULTI_STAGE_ATTESTATION_VERSION,
            "stages": [],
        }).encode("utf-8")
        bad = MULTI_STAGE_MAGIC_PREFIX + body
        with pytest.raises(MultiStageMalformedError, match="non-empty"):
            decode_multi_stage_attestation(bad)

    def test_decode_rejects_duplicate_stage_index(self):
        """M2 regression: settler injecting duplicate stage_index
        could mask a SOFTWARE stage's presence."""
        body = json.dumps({
            "version": MULTI_STAGE_ATTESTATION_VERSION,
            "stages": [
                {"stage_index": 0, "stage_node_id": "alice",
                 "tee_type": "sgx", "attestation_hex": "01"},
                {"stage_index": 0, "stage_node_id": "bob",
                 "tee_type": "sgx", "attestation_hex": "02"},
            ],
        }).encode("utf-8")
        bad = MULTI_STAGE_MAGIC_PREFIX + body
        with pytest.raises(MultiStageMalformedError, match="duplicate"):
            decode_multi_stage_attestation(bad)

    def test_decode_rejects_non_contiguous_indices(self):
        """M2 regression: gap in stage_index sequence (e.g., 0 + 2
        but no 1) → settler omitted a stage."""
        body = json.dumps({
            "version": MULTI_STAGE_ATTESTATION_VERSION,
            "stages": [
                {"stage_index": 0, "stage_node_id": "alice",
                 "tee_type": "sgx", "attestation_hex": "01"},
                {"stage_index": 2, "stage_node_id": "charlie",
                 "tee_type": "sgx", "attestation_hex": "03"},
            ],
        }).encode("utf-8")
        bad = MULTI_STAGE_MAGIC_PREFIX + body
        with pytest.raises(MultiStageMalformedError, match="contiguous"):
            decode_multi_stage_attestation(bad)

    def test_decode_rejects_out_of_range_index(self):
        body = json.dumps({
            "version": MULTI_STAGE_ATTESTATION_VERSION,
            "stages": [
                {"stage_index": 5, "stage_node_id": "x",
                 "tee_type": "sgx", "attestation_hex": "01"},
            ],
        }).encode("utf-8")
        bad = MULTI_STAGE_MAGIC_PREFIX + body
        with pytest.raises(MultiStageMalformedError, match="contiguous"):
            decode_multi_stage_attestation(bad)

    def test_decode_returns_sorted_by_stage_index(self):
        """Even with valid input arriving out-of-order, decode returns
        stages sorted for deterministic iteration."""
        stages = _stages(
            (1, "bob", TEEType.TDX, b"\x02"),
            (0, "alice", TEEType.SGX, b"\x01"),
            (2, "charlie", TEEType.SEV, b"\x03"),
        )
        blob = encode_multi_stage_attestation(stages)
        recovered = decode_multi_stage_attestation(blob)
        assert recovered is not None
        assert [s.stage_index for s in recovered] == [0, 1, 2]
        assert [s.stage_node_id for s in recovered] == [
            "alice", "bob", "charlie",
        ]

    def test_decode_expected_stage_count_match(self):
        stages = _stages(
            (0, "alice", TEEType.SGX, b"\x01"),
            (1, "bob", TEEType.SGX, b"\x02"),
        )
        blob = encode_multi_stage_attestation(stages)
        recovered = decode_multi_stage_attestation(
            blob, expected_stage_count=2
        )
        assert recovered is not None
        assert len(recovered) == 2

    def test_decode_expected_stage_count_mismatch_raises(self):
        """M2 regression: caller knows the chain length and asserts
        the envelope has the same count — defends against omission."""
        stages = _stages(
            (0, "alice", TEEType.SGX, b"\x01"),
            (1, "bob", TEEType.SGX, b"\x02"),
        )
        blob = encode_multi_stage_attestation(stages)
        with pytest.raises(MultiStageMalformedError, match="3"):
            decode_multi_stage_attestation(blob, expected_stage_count=3)


# ──────────────────────────────────────────────────────────────────────────
# Magic-prefix detection
# ──────────────────────────────────────────────────────────────────────────


class TestIsMultiStage:
    def test_recognizes_envelope(self):
        stages = _stages((0, "alice", TEEType.SGX, b"\x01"))
        assert is_multi_stage_attestation(encode_multi_stage_attestation(stages))

    def test_rejects_unprefixed(self):
        assert is_multi_stage_attestation(b"\x01" * 32) is False

    def test_rejects_empty(self):
        assert is_multi_stage_attestation(b"") is False

    def test_rejects_non_bytes(self):
        assert is_multi_stage_attestation("string") is False  # type: ignore[arg-type]

    def test_rejects_partial_magic(self):
        # Less than full prefix length.
        assert is_multi_stage_attestation(MULTI_STAGE_MAGIC_PREFIX[:5]) is False


# ──────────────────────────────────────────────────────────────────────────
# Worst-case TEE policy
# ──────────────────────────────────────────────────────────────────────────


class TestWorstCaseTEE:
    def test_empty_returns_software(self):
        assert worst_case_tee_type([]) == TEEType.SOFTWARE

    def test_all_software_returns_software(self):
        stages = _stages(
            (0, "a", TEEType.SOFTWARE, b"\x00"),
            (1, "b", TEEType.SOFTWARE, b"\x00"),
        )
        assert worst_case_tee_type(stages) == TEEType.SOFTWARE

    def test_all_hardware_keeps_first(self):
        stages = _stages(
            (0, "a", TEEType.SGX, b"\x00"),
            (1, "b", TEEType.TDX, b"\x00"),
        )
        # First stage's hardware type wins by tie-break.
        assert worst_case_tee_type(stages) == TEEType.SGX

    def test_software_drags_hardware_down(self):
        # SGX, then SOFTWARE, then TDX → SOFTWARE wins.
        stages = _stages(
            (0, "a", TEEType.SGX, b"\x00"),
            (1, "b", TEEType.SOFTWARE, b"\x00"),
            (2, "c", TEEType.TDX, b"\x00"),
        )
        assert worst_case_tee_type(stages) == TEEType.SOFTWARE

    def test_software_at_tail_drags_down(self):
        stages = _stages(
            (0, "a", TEEType.SGX, b"\x00"),
            (1, "b", TEEType.SGX, b"\x00"),
            (2, "c", TEEType.SOFTWARE, b"\x00"),
        )
        assert worst_case_tee_type(stages) == TEEType.SOFTWARE


class TestIsHardwareTee:
    def test_software_is_not_hardware(self):
        assert is_hardware_tee(TEEType.SOFTWARE) is False

    @pytest.mark.parametrize("tee", [
        TEEType.SGX,
        TEEType.TDX,
        TEEType.SEV,
        TEEType.TRUSTZONE,
        TEEType.SECURE_ENCLAVE,
    ])
    def test_hardware_types_are_hardware(self, tee):
        assert is_hardware_tee(tee) is True


# ──────────────────────────────────────────────────────────────────────────
# verify_stage_attestations
# ──────────────────────────────────────────────────────────────────────────


class TestVerifyStageAttestations:
    def test_single_stage_opaque_returns_placeholder(self):
        ok, results = verify_stage_attestations(b"\x01" * 32)
        assert ok is True
        assert len(results) == 1
        r = results[0]
        assert r.structurally_ok is True
        assert r.vendor_verified is None
        assert r.is_placeholder is True
        # stage_index sentinel for single-stage back-compat.
        assert r.stage_index == -1
        assert "single-stage" in r.message.lower()

    def test_multi_stage_results_not_marked_placeholder(self):
        """L7 regression: real multi-stage entries must NOT have
        is_placeholder=True. Monitoring layers depend on this flag
        to skip back-compat results when aggregating tee_type counts."""
        stages = _stages(
            (0, "alice", TEEType.SGX, b"\x01"),
            (1, "bob", TEEType.TDX, b"\x02"),
        )
        blob = encode_multi_stage_attestation(stages)
        ok, results = verify_stage_attestations(blob)
        assert ok is True
        assert all(not r.is_placeholder for r in results)

    def test_empty_bytes_treated_as_single_stage(self):
        # No envelope → opaque single-stage.
        ok, results = verify_stage_attestations(b"")
        assert ok is True
        assert len(results) == 1

    def test_multi_stage_returns_per_stage_results(self):
        stages = _stages(
            (0, "alice", TEEType.SGX, b"\x01"),
            (1, "bob", TEEType.TDX, b"\x02"),
            (2, "charlie", TEEType.SEV, b"\x03"),
        )
        blob = encode_multi_stage_attestation(stages)
        ok, results = verify_stage_attestations(blob)

        assert ok is True
        assert len(results) == 3
        assert [r.stage_index for r in results] == [0, 1, 2]
        assert [r.stage_node_id for r in results] == ["alice", "bob", "charlie"]
        assert [r.tee_type for r in results] == [
            TEEType.SGX, TEEType.TDX, TEEType.SEV,
        ]
        assert all(r.structurally_ok for r in results)
        # v1 doesn't wire vendor verification.
        assert all(r.vendor_verified is None for r in results)

    def test_malformed_envelope_returns_failure_result(self):
        # Magic prefix present but JSON body is broken.
        bad = MULTI_STAGE_MAGIC_PREFIX + b"not-json"
        ok, results = verify_stage_attestations(bad)
        assert ok is False
        assert len(results) == 1
        assert results[0].structurally_ok is False
        assert "decode failed" in results[0].message.lower()

    def test_never_raises_on_corrupt_input(self):
        # The verify API is non-throwing by contract.
        for garbage in [
            MULTI_STAGE_MAGIC_PREFIX + b"\x00" * 10,
            MULTI_STAGE_MAGIC_PREFIX + b'{"version": "wrong"}',
            MULTI_STAGE_MAGIC_PREFIX + b'[]',
            MULTI_STAGE_MAGIC_PREFIX + b'null',
        ]:
            ok, results = verify_stage_attestations(garbage)
            # Either bad → ok=False, or valid edge case → ok=True.
            assert isinstance(ok, bool)
            assert len(results) >= 1


# ──────────────────────────────────────────────────────────────────────────
# Receipt-level integration: back-compat + multi-stage
# ──────────────────────────────────────────────────────────────────────────


def _make_receipt(*, tee_attestation: bytes, tee_type: TEEType) -> InferenceReceipt:
    return InferenceReceipt(
        job_id="job-1",
        request_id="req-1",
        model_id="test-model",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=tee_type,
        tee_attestation=tee_attestation,
        output_hash=hashlib.sha256(b"output").digest(),
        duration_seconds=0.5,
        cost_ftns=Decimal("0.10"),
    )


class TestReceiptBackCompat:
    def test_single_stage_receipt_signs_and_verifies(self):
        identity = generate_node_identity("settler")
        receipt = _make_receipt(
            tee_attestation=b"\x01" * 32,  # single-stage opaque bytes
            tee_type=TEEType.SGX,
        )
        signed = sign_receipt(receipt, identity)
        # Existing Phase 3.x.1 verification path still works.
        assert verify_receipt(signed, identity=identity) is True

    def test_single_stage_receipt_treated_as_back_compat_in_verify(self):
        identity = generate_node_identity("settler")
        receipt = _make_receipt(
            tee_attestation=b"\x01" * 32,
            tee_type=TEEType.SGX,
        )
        signed = sign_receipt(receipt, identity)
        ok, results = receipt_verify_stages(signed)
        assert ok is True
        assert len(results) == 1
        assert "single-stage" in results[0].message.lower()


class TestReceiptMultiStage:
    def test_multi_stage_receipt_round_trips_and_verifies(self):
        identity = generate_node_identity("settler")
        stages = _stages(
            (0, "alice", TEEType.SGX, b"\x01" * 32),
            (1, "bob", TEEType.TDX, b"\x02" * 32),
        )
        envelope = encode_multi_stage_attestation(stages)
        receipt = _make_receipt(
            tee_attestation=envelope,
            tee_type=worst_case_tee_type(stages),
        )
        signed = sign_receipt(receipt, identity)
        # Settler signature verifies — the signing payload includes
        # the entire envelope hex so the signature commits to all
        # per-stage attestations.
        assert verify_receipt(signed, identity=identity) is True

        # Per-stage verification helper iterates the envelope.
        ok, results = receipt_verify_stages(signed)
        assert ok is True
        assert [r.stage_node_id for r in results] == ["alice", "bob"]

    def test_tampered_envelope_invalidates_signature(self):
        identity = generate_node_identity("settler")
        stages = _stages(
            (0, "alice", TEEType.SGX, b"\x01" * 32),
            (1, "bob", TEEType.SGX, b"\x02" * 32),
        )
        envelope = encode_multi_stage_attestation(stages)
        receipt = _make_receipt(
            tee_attestation=envelope,
            tee_type=TEEType.SGX,
        )
        signed = sign_receipt(receipt, identity)

        # Tamper one stage's attestation by swapping bytes in the
        # envelope. Any byte-level change to tee_attestation should
        # invalidate the settler signature.
        tampered_envelope = envelope[:-2] + b"\xff\xff"
        tampered = InferenceReceipt(
            job_id=signed.job_id,
            request_id=signed.request_id,
            model_id=signed.model_id,
            content_tier=signed.content_tier,
            privacy_tier=signed.privacy_tier,
            epsilon_spent=signed.epsilon_spent,
            tee_type=signed.tee_type,
            tee_attestation=tampered_envelope,
            output_hash=signed.output_hash,
            duration_seconds=signed.duration_seconds,
            cost_ftns=signed.cost_ftns,
            settler_signature=signed.settler_signature,
            settler_node_id=signed.settler_node_id,
        )
        # Verify will return False rather than raise.
        assert verify_receipt(tampered, identity=identity) is False

    def test_multi_stage_with_software_records_software_top_level(self):
        # Receipt-level convention: top-level tee_type reflects worst-
        # case, set by the writer (typically RpcChainExecutor). This
        # test verifies the write-side helper produces the right
        # value.
        stages = _stages(
            (0, "alice", TEEType.SGX, b"\x01"),
            (1, "bob", TEEType.SOFTWARE, b"\x02"),
        )
        assert worst_case_tee_type(stages) == TEEType.SOFTWARE
