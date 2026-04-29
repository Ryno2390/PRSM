"""Phase 3.x.11.x Task 1 — multi_iteration_attestation unit tests.

Covers the per-token attestation envelope used by sharded
autoregressive decode receipts:

  - IterationAttestation construction validation (iteration_index,
    decode_mode, stage_records, PREFILL/INCREMENTAL coupling)
  - encode_multi_iteration_attestation round-trip
  - Magic-prefix detection (is_multi_iteration_attestation)
  - Discriminator vs the existing multi-stage envelope (an
    iteration envelope must NOT be detected as a stage envelope
    and vice-versa — back-compat preserved)
  - decode_multi_iteration_attestation structural validation
    (iteration_index gaps, duplicate iteration_index, non-uniform
    stage counts, malformed JSON)
  - worst_case_tee_type_across_iterations
  - Golden-bytes byte-equivalence pin for the existing
    encode_multi_stage_attestation shape (non-sharded receipts
    unchanged)
"""

from __future__ import annotations

import json

import pytest

from prsm.compute.chain_rpc.protocol import DecodeMode
from prsm.compute.inference.multi_stage_attestation import (
    MULTI_ITERATION_ATTESTATION_VERSION,
    MULTI_ITERATION_MAGIC_PREFIX,
    MULTI_STAGE_ATTESTATION_VERSION,
    MULTI_STAGE_MAGIC_PREFIX,
    IterationAttestation,
    MultiStageAttestationError,
    MultiStageMalformedError,
    StageAttestation,
    decode_multi_iteration_attestation,
    decode_multi_stage_attestation,
    encode_multi_iteration_attestation,
    encode_multi_stage_attestation,
    is_multi_iteration_attestation,
    is_multi_stage_attestation,
    worst_case_tee_type_across_iterations,
)
from prsm.compute.tee.models import TEEType


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _stage(idx: int, node_id: str = "alice", tee: TEEType = TEEType.SGX) -> StageAttestation:
    return StageAttestation(
        stage_index=idx,
        stage_node_id=f"{node_id}-{idx}",
        tee_type=tee,
        attestation=bytes([idx]) * 32,
    )


def _iteration(
    idx: int,
    n_stages: int = 2,
    decode_mode: DecodeMode = None,
) -> IterationAttestation:
    if decode_mode is None:
        decode_mode = DecodeMode.PREFILL if idx == 0 else DecodeMode.INCREMENTAL
    return IterationAttestation(
        iteration_index=idx,
        decode_mode=decode_mode,
        stage_records=[_stage(s) for s in range(n_stages)],
    )


# ──────────────────────────────────────────────────────────────────────────
# IterationAttestation construction
# ──────────────────────────────────────────────────────────────────────────


class TestIterationAttestationConstruction:
    def test_happy_path_prefill(self):
        it = _iteration(0)
        assert it.iteration_index == 0
        assert it.decode_mode == DecodeMode.PREFILL
        assert len(it.stage_records) == 2

    def test_happy_path_incremental(self):
        it = _iteration(1)
        assert it.iteration_index == 1
        assert it.decode_mode == DecodeMode.INCREMENTAL

    def test_rejects_negative_iteration_index(self):
        with pytest.raises(MultiStageMalformedError, match="iteration_index"):
            IterationAttestation(
                iteration_index=-1,
                decode_mode=DecodeMode.PREFILL,
                stage_records=[_stage(0)],
            )

    def test_rejects_bool_iteration_index(self):
        with pytest.raises(MultiStageMalformedError, match="iteration_index"):
            IterationAttestation(
                iteration_index=True,  # type: ignore[arg-type]
                decode_mode=DecodeMode.PREFILL,
                stage_records=[_stage(0)],
            )

    def test_rejects_non_decode_mode(self):
        with pytest.raises(MultiStageMalformedError, match="decode_mode"):
            IterationAttestation(
                iteration_index=0,
                decode_mode="prefill",  # type: ignore[arg-type]
                stage_records=[_stage(0)],
            )

    def test_rejects_empty_stage_records(self):
        with pytest.raises(MultiStageMalformedError, match="stage_records"):
            IterationAttestation(
                iteration_index=0,
                decode_mode=DecodeMode.PREFILL,
                stage_records=[],
            )

    def test_rejects_non_stage_attestation_in_records(self):
        with pytest.raises(MultiStageMalformedError, match="StageAttestation"):
            IterationAttestation(
                iteration_index=0,
                decode_mode=DecodeMode.PREFILL,
                stage_records=["not a StageAttestation"],  # type: ignore[list-item]
            )

    def test_rejects_iteration_zero_with_incremental(self):
        # iteration_index=0 must be PREFILL.
        with pytest.raises(
            MultiStageMalformedError, match="iteration_index=0",
        ):
            IterationAttestation(
                iteration_index=0,
                decode_mode=DecodeMode.INCREMENTAL,
                stage_records=[_stage(0)],
            )

    def test_rejects_iteration_nonzero_with_prefill(self):
        with pytest.raises(
            MultiStageMalformedError, match="INCREMENTAL",
        ):
            IterationAttestation(
                iteration_index=1,
                decode_mode=DecodeMode.PREFILL,
                stage_records=[_stage(0)],
            )


# ──────────────────────────────────────────────────────────────────────────
# Envelope round-trip
# ──────────────────────────────────────────────────────────────────────────


class TestEnvelopeRoundTrip:
    def test_round_trip_single_iteration(self):
        iterations = [_iteration(0, n_stages=2)]
        blob = encode_multi_iteration_attestation(iterations)
        decoded = decode_multi_iteration_attestation(blob)
        assert decoded is not None
        assert len(decoded) == 1
        assert decoded[0].iteration_index == 0
        assert decoded[0].decode_mode == DecodeMode.PREFILL
        assert len(decoded[0].stage_records) == 2

    def test_round_trip_four_iterations(self):
        # Mirrors a 4-token sharded decode (PREFILL + 3 INCREMENTALs).
        iterations = [_iteration(i, n_stages=2) for i in range(4)]
        blob = encode_multi_iteration_attestation(iterations)
        decoded = decode_multi_iteration_attestation(blob)
        assert decoded is not None
        assert len(decoded) == 4
        assert [it.iteration_index for it in decoded] == [0, 1, 2, 3]
        assert decoded[0].decode_mode == DecodeMode.PREFILL
        for it in decoded[1:]:
            assert it.decode_mode == DecodeMode.INCREMENTAL

    def test_envelope_carries_magic_prefix(self):
        blob = encode_multi_iteration_attestation([_iteration(0)])
        assert blob.startswith(MULTI_ITERATION_MAGIC_PREFIX)

    def test_envelope_top_level_keys(self):
        blob = encode_multi_iteration_attestation([_iteration(0)])
        body = blob[len(MULTI_ITERATION_MAGIC_PREFIX):]
        data = json.loads(body)
        assert set(data.keys()) == {"version", "iterations"}
        assert data["version"] == MULTI_ITERATION_ATTESTATION_VERSION
        assert isinstance(data["iterations"], list)

    def test_empty_iterations_raises_at_encode(self):
        with pytest.raises(
            MultiStageAttestationError, match="at least one iteration"
        ):
            encode_multi_iteration_attestation([])

    def test_decode_returns_iterations_sorted_by_index(self):
        # Encoder preserves order; encode an OUT-OF-ORDER input
        # and verify the decoder sorts.
        iterations = [_iteration(2), _iteration(0), _iteration(1)]
        blob = encode_multi_iteration_attestation(iterations)
        decoded = decode_multi_iteration_attestation(blob)
        assert [it.iteration_index for it in decoded] == [0, 1, 2]


# ──────────────────────────────────────────────────────────────────────────
# Discriminator: iteration vs stage envelope back-compat
# ──────────────────────────────────────────────────────────────────────────


class TestEnvelopeDiscriminator:
    def test_iteration_envelope_not_detected_as_stage(self):
        # The new multi-iteration envelope MUST NOT be detected as
        # a multi-stage envelope. Non-sharded receipt verifiers
        # (decode_multi_stage_attestation) get None for sharded
        # receipts — they fall back to opaque-single-stage handling
        # (which is wrong for sharded but at least non-corrupt).
        blob = encode_multi_iteration_attestation([_iteration(0)])
        assert is_multi_stage_attestation(blob) is False
        assert is_multi_iteration_attestation(blob) is True
        assert decode_multi_stage_attestation(blob) is None

    def test_stage_envelope_not_detected_as_iteration(self):
        # The existing multi-stage envelope MUST NOT be detected
        # as a multi-iteration envelope.
        blob = encode_multi_stage_attestation([_stage(0), _stage(1)])
        assert is_multi_iteration_attestation(blob) is False
        assert is_multi_stage_attestation(blob) is True
        assert decode_multi_iteration_attestation(blob) is None

    def test_random_bytes_neither_envelope(self):
        blob = b"random opaque bytes that aren't either envelope"
        assert is_multi_stage_attestation(blob) is False
        assert is_multi_iteration_attestation(blob) is False
        assert decode_multi_stage_attestation(blob) is None
        assert decode_multi_iteration_attestation(blob) is None


# ──────────────────────────────────────────────────────────────────────────
# Structural validation
# ──────────────────────────────────────────────────────────────────────────


class TestStructuralValidation:
    def test_decode_rejects_non_json_body(self):
        bad = MULTI_ITERATION_MAGIC_PREFIX + b"not-json"
        with pytest.raises(
            MultiStageMalformedError, match="JSON parse failed",
        ):
            decode_multi_iteration_attestation(bad)

    def test_decode_rejects_wrong_version(self):
        body = json.dumps({
            "version": MULTI_ITERATION_ATTESTATION_VERSION + 1,
            "iterations": [_iteration(0).to_dict()],
        }).encode("utf-8")
        bad = MULTI_ITERATION_MAGIC_PREFIX + body
        with pytest.raises(
            MultiStageMalformedError, match="envelope version",
        ):
            decode_multi_iteration_attestation(bad)

    def test_decode_rejects_empty_iterations(self):
        body = json.dumps({
            "version": MULTI_ITERATION_ATTESTATION_VERSION,
            "iterations": [],
        }).encode("utf-8")
        bad = MULTI_ITERATION_MAGIC_PREFIX + body
        with pytest.raises(
            MultiStageMalformedError, match="non-empty 'iterations'",
        ):
            decode_multi_iteration_attestation(bad)

    def test_decode_rejects_duplicate_iteration_index(self):
        body = json.dumps({
            "version": MULTI_ITERATION_ATTESTATION_VERSION,
            "iterations": [
                _iteration(0).to_dict(),
                _iteration(0).to_dict(),  # duplicate
            ],
        }).encode("utf-8")
        bad = MULTI_ITERATION_MAGIC_PREFIX + body
        with pytest.raises(
            MultiStageMalformedError, match="duplicate iteration_index",
        ):
            decode_multi_iteration_attestation(bad)

    def test_decode_rejects_gappy_iteration_index(self):
        body = json.dumps({
            "version": MULTI_ITERATION_ATTESTATION_VERSION,
            "iterations": [
                _iteration(0).to_dict(),
                _iteration(2).to_dict(),  # missing 1
            ],
        }).encode("utf-8")
        bad = MULTI_ITERATION_MAGIC_PREFIX + body
        with pytest.raises(
            MultiStageMalformedError, match="contiguous",
        ):
            decode_multi_iteration_attestation(bad)

    def test_decode_rejects_non_uniform_stage_counts(self):
        body = json.dumps({
            "version": MULTI_ITERATION_ATTESTATION_VERSION,
            "iterations": [
                _iteration(0, n_stages=2).to_dict(),
                _iteration(1, n_stages=3).to_dict(),  # different count
            ],
        }).encode("utf-8")
        bad = MULTI_ITERATION_MAGIC_PREFIX + body
        with pytest.raises(
            MultiStageMalformedError, match="uniform across",
        ):
            decode_multi_iteration_attestation(bad)

    def test_decode_rejects_gappy_stage_index_within_iteration(self):
        body = json.dumps({
            "version": MULTI_ITERATION_ATTESTATION_VERSION,
            "iterations": [{
                "iteration_index": 0,
                "decode_mode": "prefill",
                "stage_records": [
                    _stage(0).to_dict(),
                    _stage(2).to_dict(),  # missing 1
                ],
            }],
        }).encode("utf-8")
        bad = MULTI_ITERATION_MAGIC_PREFIX + body
        with pytest.raises(
            MultiStageMalformedError,
            match="stage_index values must be 0..1",
        ):
            decode_multi_iteration_attestation(bad)

    def test_decode_expected_iteration_count_mismatch(self):
        iterations = [_iteration(0), _iteration(1)]
        blob = encode_multi_iteration_attestation(iterations)
        with pytest.raises(
            MultiStageMalformedError, match="expected 5",
        ):
            decode_multi_iteration_attestation(
                blob, expected_iteration_count=5,
            )

    def test_decode_expected_stage_count_mismatch(self):
        iterations = [_iteration(0, n_stages=2)]
        blob = encode_multi_iteration_attestation(iterations)
        with pytest.raises(
            MultiStageMalformedError, match="expected 5",
        ):
            decode_multi_iteration_attestation(
                blob, expected_stage_count=5,
            )

    def test_decode_expected_counts_match_passes(self):
        iterations = [_iteration(0, n_stages=2), _iteration(1, n_stages=2)]
        blob = encode_multi_iteration_attestation(iterations)
        decoded = decode_multi_iteration_attestation(
            blob, expected_iteration_count=2, expected_stage_count=2,
        )
        assert len(decoded) == 2


# ──────────────────────────────────────────────────────────────────────────
# Worst-case TEE across iterations
# ──────────────────────────────────────────────────────────────────────────


class TestWorstCaseAcrossIterations:
    def test_all_hardware_returns_first_seen(self):
        iterations = [
            _iteration(0, n_stages=2),  # both SGX
            _iteration(1, n_stages=2),  # both SGX
        ]
        assert worst_case_tee_type_across_iterations(iterations) == TEEType.SGX

    def test_one_software_stage_drags_to_software(self):
        # PREFILL all SGX; INCREMENTAL has a SOFTWARE stage.
        iterations = [
            _iteration(0, n_stages=2),  # SGX both
            IterationAttestation(
                iteration_index=1,
                decode_mode=DecodeMode.INCREMENTAL,
                stage_records=[
                    _stage(0, tee=TEEType.SGX),
                    _stage(1, tee=TEEType.SOFTWARE),
                ],
            ),
        ]
        assert worst_case_tee_type_across_iterations(iterations) == TEEType.SOFTWARE

    def test_empty_returns_software(self):
        assert worst_case_tee_type_across_iterations([]) == TEEType.SOFTWARE


# ──────────────────────────────────────────────────────────────────────────
# Byte-equivalence pin for the existing multi-stage envelope
# ──────────────────────────────────────────────────────────────────────────


class TestExistingEnvelopeUnchanged:
    """Phase 3.x.11.x MUST NOT change the existing multi-stage
    envelope's byte output. Non-sharded receipts (Phase 3.x.7+
    unary, Phase 3.x.8+ streaming-tail) keep their pre-3.x.11.x
    canonical bytes — load-bearing for receipt-verification
    back-compat."""

    def test_two_stage_envelope_golden_bytes(self):
        # Pin the exact bytes for a deterministic 2-stage input.
        # Any change here means we broke pre-3.x.11.x receipts.
        stages = [
            StageAttestation(
                stage_index=0,
                stage_node_id="alice-fixed",
                tee_type=TEEType.SGX,
                attestation=b"\x01" * 16,
            ),
            StageAttestation(
                stage_index=1,
                stage_node_id="bob-fixed",
                tee_type=TEEType.SOFTWARE,
                attestation=b"\x02" * 16,
            ),
        ]
        blob = encode_multi_stage_attestation(stages)
        # Reconstruct the expected bytes from first principles to
        # avoid hardcoding a brittle hex string. Sort_keys=True
        # canonical encoding.
        expected_payload = {
            "stages": [
                {
                    "attestation_hex": "01" * 16,
                    "stage_index": 0,
                    "stage_node_id": "alice-fixed",
                    "tee_type": "sgx",
                },
                {
                    "attestation_hex": "02" * 16,
                    "stage_index": 1,
                    "stage_node_id": "bob-fixed",
                    "tee_type": "software",
                },
            ],
            "version": MULTI_STAGE_ATTESTATION_VERSION,
        }
        expected = (
            MULTI_STAGE_MAGIC_PREFIX
            + json.dumps(expected_payload, sort_keys=True).encode("utf-8")
        )
        assert blob == expected, (
            "Phase 3.x.11.x changed the multi-stage envelope's "
            "canonical bytes — this breaks pre-3.x.11.x receipt "
            "verification!"
        )

    def test_existing_decoder_still_works_round_trip(self):
        stages = [_stage(0), _stage(1)]
        blob = encode_multi_stage_attestation(stages)
        decoded = decode_multi_stage_attestation(blob)
        assert decoded is not None
        assert len(decoded) == 2
        assert decoded[0].stage_index == 0
        assert decoded[1].stage_index == 1
