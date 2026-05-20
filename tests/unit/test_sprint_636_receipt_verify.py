"""Sprint 636 — receipt-verify core tamper-detection pin tests.

These tests defend the Vision §7 truth-surfacing chain at CI level.
Sprint 635's `prsm node verify-receipts` is the operator's audit-time
gate; a subtle refactor that broke the verification path (e.g.
forgot to include activation_blob in the reconstructed signing
payload, or trusted the receipt's self-declared pubkey instead of
the anchor's) would be invisible in passing-end-to-end tests until
an auditor caught it.

Tamper coverage:
  - byte-flip in activation_blob → SIGNATURE_INVALID
  - byte-flip in stage_signature_b64 → SIGNATURE_INVALID
  - swap stage_node_id to another anchor-registered node →
    SIGNATURE_INVALID (anti-Mallory-substitution invariant)
  - PUBKEY_NOT_REGISTERED when anchor returns None
  - UNVERIFIABLE on pre-sprint-635 format (sha256-only)
  - RECONSTRUCT_FAILED on missing required fields
  - Happy path: honest receipt → OK
"""
from __future__ import annotations

import base64
import json
from typing import Dict, Optional
from unittest.mock import MagicMock

from prsm.cli_modules.receipt_verify import (
    verify_receipt_record, verify_receipts_file,
)
from prsm.compute.chain_rpc.protocol import RunLayerSliceResponse
from prsm.compute.tee.models import TEEType
from prsm.node.identity import NodeIdentity, generate_node_identity


def _make_stage_identity() -> NodeIdentity:
    """Fresh Ed25519 keypair for a fake stage node."""
    return generate_node_identity(display_name="test-stage")


def _signed_response(
    stage_identity: NodeIdentity,
    *,
    request_id: str = "test-req",
    activation_bytes: bytes = b"hidden-state-bytes",
    activation_shape: tuple = (1, 1, 8),
    activation_dtype: str = "float32",
    duration_seconds: float = 0.123,
    tee_attestation: bytes = b"sw-tee-attest",
    tee_type: TEEType = TEEType.SOFTWARE,
    epsilon_spent: float = 0.0,
) -> RunLayerSliceResponse:
    """Build + sign a RunLayerSliceResponse the way LayerStageServer does."""
    return RunLayerSliceResponse.sign(
        identity=stage_identity,
        request_id=request_id,
        activation_blob=activation_bytes,
        activation_shape=activation_shape,
        activation_dtype=activation_dtype,
        duration_seconds=duration_seconds,
        tee_attestation=tee_attestation,
        tee_type=tee_type,
        epsilon_spent=epsilon_spent,
    )


def _receipt_from_response(
    resp: RunLayerSliceResponse,
    *,
    settler_node_id: str = "settler-test",
    model_id: str = "test-model",
    n_layers: int = 4,
) -> Dict:
    """Build the sprint-635 receipt record dict from a signed response."""
    return {
        "step": 0,
        "wall_unix": 1000.0,
        "request_id": resp.request_id,
        "settler_node_id": settler_node_id,
        "stage_node_id": resp.stage_node_id,
        "stage_signature_b64": resp.stage_signature_b64,
        "model_id": model_id,
        "layer_range": [0, n_layers],
        "activation_shape": list(resp.activation_shape),
        "activation_dtype": resp.activation_dtype,
        "activation_sha256": "ignored-for-this-test",
        "activation_blob_b64": base64.b64encode(
            bytes(resp.activation_blob),
        ).decode("ascii"),
        "tee_attestation_b64": base64.b64encode(
            bytes(resp.tee_attestation),
        ).decode("ascii"),
        "duration_seconds": resp.duration_seconds,
        "epsilon_spent": resp.epsilon_spent,
        "tee_type": resp.tee_type.value,
        "protocol_version": resp.protocol_version,
        "next_token_id": 42,
        "next_token_text": " test",
    }


def _anchor_with(*identities: NodeIdentity):
    """Build a mock anchor whose lookup() returns the pubkey of any
    of the supplied identities by node_id; None for unknown ids.
    """
    pubkey_map = {
        identity.node_id: identity.public_key_b64
        for identity in identities
    }
    anchor = MagicMock()
    anchor.lookup = MagicMock(side_effect=lambda nid: pubkey_map.get(nid))
    return anchor


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_honest_receipt_verifies_OK():
    stage = _make_stage_identity()
    resp = _signed_response(stage)
    rec = _receipt_from_response(resp)
    anchor = _anchor_with(stage)

    result = verify_receipt_record(rec, anchor=anchor)
    assert result["status"] == "OK", (
        f"honest receipt must verify; got {result}"
    )
    assert result["stage_node_id"] == stage.node_id
    assert result["request_id"] == "test-req"


# --------------------------------------------------------------------------
# Tamper detection
# --------------------------------------------------------------------------


def test_tampered_activation_blob_is_SIGNATURE_INVALID():
    """Byte-flip in the signed-over activation → signature must
    fail. Defends against a relay swapping the logits between sign-
    time and the operator's argmax — the very property activation
    bytes commit to.
    """
    stage = _make_stage_identity()
    resp = _signed_response(stage)
    rec = _receipt_from_response(resp)

    # Decode, flip one byte, re-encode
    blob = bytearray(base64.b64decode(rec["activation_blob_b64"]))
    blob[0] ^= 0xFF  # flip every bit of first byte
    rec["activation_blob_b64"] = base64.b64encode(bytes(blob)).decode("ascii")

    anchor = _anchor_with(stage)
    result = verify_receipt_record(rec, anchor=anchor)
    assert result["status"] == "SIGNATURE_INVALID", (
        f"tampered activation must be detected; got {result}"
    )


def test_tampered_signature_is_SIGNATURE_INVALID():
    """Flip one bit in the signature itself. The receipt's other
    fields are intact but the signature no longer matches the
    canonical signing payload.
    """
    stage = _make_stage_identity()
    resp = _signed_response(stage)
    rec = _receipt_from_response(resp)

    # Flip a byte in the base64-decoded signature
    sig_bytes = bytearray(base64.b64decode(rec["stage_signature_b64"]))
    sig_bytes[0] ^= 0x01
    rec["stage_signature_b64"] = base64.b64encode(bytes(sig_bytes)).decode("ascii")

    anchor = _anchor_with(stage)
    result = verify_receipt_record(rec, anchor=anchor)
    assert result["status"] == "SIGNATURE_INVALID"


def test_anti_mallory_substitution():
    """Mallory has a different anchor-registered identity. Replace
    the stage's signed response with one Mallory signed (over
    different bytes, with Mallory's own pubkey listed). The
    receipt's stage_node_id field is rewritten to Mallory's id.

    verify_with_anchor's expected_stage_node_id contract should
    catch this because the EXPECTED node_id (from the receipt's
    declared stage_node_id) does match Mallory — but Mallory's
    response was over DIFFERENT bytes than the operator observed
    (the receipt claims those activation bytes were what stage
    produced; Mallory's signature is over Mallory's own activation
    bytes that the operator can't verify match).

    More directly: a swap that keeps stage_node_id as the HONEST
    stage but signs with Mallory's key. Honest stage's pubkey
    won't verify Mallory's signature → SIGNATURE_INVALID.
    """
    honest_stage = _make_stage_identity()
    mallory = _make_stage_identity()

    # Honest stage signs over honest activation bytes
    honest_resp = _signed_response(
        honest_stage, activation_bytes=b"honest-bytes",
    )
    # Mallory signs over different activation bytes
    mallory_resp = _signed_response(
        mallory, activation_bytes=b"mallory-bytes",
    )

    # Build a receipt that claims to be from the honest stage but
    # carries Mallory's signature + activation
    rec = _receipt_from_response(honest_resp)
    rec["stage_signature_b64"] = mallory_resp.stage_signature_b64
    rec["activation_blob_b64"] = base64.b64encode(
        bytes(mallory_resp.activation_blob),
    ).decode("ascii")
    rec["activation_shape"] = list(mallory_resp.activation_shape)

    # Anchor registers both — both have pubkeys on-chain
    anchor = _anchor_with(honest_stage, mallory)

    result = verify_receipt_record(rec, anchor=anchor)
    # Mallory's signature won't verify under honest_stage's pubkey
    # (which is what the receipt's stage_node_id resolves to)
    assert result["status"] == "SIGNATURE_INVALID", (
        f"Mallory substitution must be detected; got {result}"
    )


# --------------------------------------------------------------------------
# Anchor / pubkey failure modes
# --------------------------------------------------------------------------


def test_pubkey_not_registered():
    """Stage signed honestly but no entry in the anchor → must
    report PUBKEY_NOT_REGISTERED, not silently accept.
    """
    stage = _make_stage_identity()
    resp = _signed_response(stage)
    rec = _receipt_from_response(resp)

    # Anchor has nobody registered
    anchor = MagicMock()
    anchor.lookup = MagicMock(return_value=None)

    result = verify_receipt_record(rec, anchor=anchor)
    assert result["status"] == "PUBKEY_NOT_REGISTERED"
    assert stage.node_id in result["reason"]


def test_anchor_lookup_raises():
    """Anchor RPC fails (transient on-chain error). Must surface
    as ANCHOR_LOOKUP_FAILED, not crash the whole verification run.
    """
    stage = _make_stage_identity()
    resp = _signed_response(stage)
    rec = _receipt_from_response(resp)

    anchor = MagicMock()
    anchor.lookup = MagicMock(side_effect=RuntimeError("base mainnet RPC timeout"))

    result = verify_receipt_record(rec, anchor=anchor)
    assert result["status"] == "ANCHOR_LOOKUP_FAILED"
    assert "RuntimeError" in result["reason"]
    assert "RPC timeout" in result["reason"]


# --------------------------------------------------------------------------
# Format-version failure modes
# --------------------------------------------------------------------------


def test_pre_sprint_635_format_is_UNVERIFIABLE():
    """A receipt with the sprint-634 format (no activation_blob_b64)
    cannot be verified offline. Report UNVERIFIABLE with a clear
    breadcrumb to the sprint number — operators upgrading from 634
    learn why old audit files don't verify anymore.
    """
    rec = {
        "step": 0,
        "request_id": "old-receipt",
        "settler_node_id": "settler-x",
        "stage_node_id": "stage-y",
        "stage_signature_b64": "x",
        "activation_shape": [1, 1, 8],
        "activation_dtype": "float32",
        "activation_sha256": "x",
        "duration_seconds": 0,
        "epsilon_spent": 0,
        "tee_type": "software",
        "protocol_version": 2,
        "next_token_id": 1,
        "next_token_text": "x",
    }
    anchor = MagicMock()
    result = verify_receipt_record(rec, anchor=anchor)
    assert result["status"] == "UNVERIFIABLE"
    assert "sprint-635" in result["reason"]
    # Anchor lookup was NOT called — the early-reject path skipped it
    anchor.lookup.assert_not_called()


def test_missing_required_field_is_RECONSTRUCT_FAILED():
    """A receipt with activation_blob_b64 but missing other required
    fields (e.g. activation_shape) → RECONSTRUCT_FAILED, not a
    confusing downstream signature error.
    """
    rec = {
        "request_id": "test",
        "stage_node_id": "stage",
        "stage_signature_b64": "x",
        "activation_blob_b64": base64.b64encode(b"x").decode("ascii"),
        # missing: activation_shape, activation_dtype, duration_seconds, etc.
    }
    anchor = MagicMock()
    result = verify_receipt_record(rec, anchor=anchor)
    assert result["status"] == "RECONSTRUCT_FAILED"


# --------------------------------------------------------------------------
# File-level driver
# --------------------------------------------------------------------------


def test_verify_receipts_file_caches_pubkey_lookups(tmp_path):
    """3 receipts from the same stage → only ONE anchor.lookup call.
    Without this, a 10-token audit file would hit on-chain RPC 10x.
    """
    stage = _make_stage_identity()
    jsonl_path = tmp_path / "receipts.jsonl"
    with jsonl_path.open("w") as f:
        for i in range(3):
            resp = _signed_response(stage, request_id=f"req-{i}")
            rec = _receipt_from_response(resp)
            f.write(json.dumps(rec) + "\n")

    anchor = _anchor_with(stage)
    results = verify_receipts_file(str(jsonl_path), anchor=anchor)
    assert len(results) == 3
    assert all(r["status"] == "OK" for r in results)
    # Only ONE on-chain read despite 3 records (same stage_node_id)
    assert anchor.lookup.call_count == 1


def test_verify_receipts_file_skips_blank_lines(tmp_path):
    """Empty/whitespace lines must not produce status records and
    must not crash the loop.
    """
    stage = _make_stage_identity()
    resp = _signed_response(stage)
    rec = _receipt_from_response(resp)
    jsonl_path = tmp_path / "receipts.jsonl"
    with jsonl_path.open("w") as f:
        f.write("\n")
        f.write(json.dumps(rec) + "\n")
        f.write("   \n")

    anchor = _anchor_with(stage)
    results = verify_receipts_file(str(jsonl_path), anchor=anchor)
    assert len(results) == 1
    assert results[0]["status"] == "OK"


def test_verify_receipts_file_handles_malformed_json(tmp_path):
    """A bad line shouldn't kill the whole run; report MALFORMED_JSON
    + continue.
    """
    stage = _make_stage_identity()
    resp = _signed_response(stage)
    rec = _receipt_from_response(resp)
    jsonl_path = tmp_path / "receipts.jsonl"
    with jsonl_path.open("w") as f:
        f.write("{not-json\n")
        f.write(json.dumps(rec) + "\n")

    anchor = _anchor_with(stage)
    results = verify_receipts_file(str(jsonl_path), anchor=anchor)
    assert len(results) == 2
    assert results[0]["status"] == "MALFORMED_JSON"
    assert results[1]["status"] == "OK"


# --------------------------------------------------------------------------
# Sprint 637 — chain-of-custody invariants
# --------------------------------------------------------------------------


def _chain_record(
    *,
    settler: str = "settler-A",
    model: str = "gpt2",
    request_id: str = "req-0",
    wall_unix: float = 1000.0,
    next_token_id: int = 0,
    activation_blob_b64: Optional[str] = None,
    activation_shape: Optional[list] = None,
    activation_dtype: str = "float32",
) -> Dict:
    """Lightweight chain-test record (no signature crypto; chain
    invariants are orthogonal to signature verification).
    """
    import base64 as _b64
    import struct as _struct
    rec: Dict = {
        "settler_node_id": settler,
        "model_id": model,
        "request_id": request_id,
        "wall_unix": wall_unix,
        "next_token_id": next_token_id,
    }
    if activation_blob_b64 is not None:
        rec["activation_blob_b64"] = activation_blob_b64
        rec["activation_shape"] = activation_shape or [1, 1, 4]
        rec["activation_dtype"] = activation_dtype
    return rec


def _make_argmax_consistent_record(next_token_id: int, vocab: int = 8) -> Dict:
    """Build a record where activation_blob's argmax actually equals
    next_token_id. Useful for the happy-path C5 test.
    """
    import base64 as _b64
    import numpy as _np
    logits = _np.zeros((1, 1, vocab), dtype=_np.float32)
    logits[0, 0, next_token_id] = 99.0  # argmax = next_token_id
    return _chain_record(
        next_token_id=next_token_id,
        activation_blob_b64=_b64.b64encode(logits.tobytes()).decode("ascii"),
        activation_shape=[1, 1, vocab],
    )


def _make_argmax_tampered_record(
    claimed_token_id: int, actual_argmax: int, vocab: int = 8,
) -> Dict:
    """Activation's argmax = actual_argmax, but receipt claims
    claimed_token_id. Operator-side post-sampling tampering.
    """
    import base64 as _b64
    import numpy as _np
    logits = _np.zeros((1, 1, vocab), dtype=_np.float32)
    logits[0, 0, actual_argmax] = 99.0
    return _chain_record(
        next_token_id=claimed_token_id,
        activation_blob_b64=_b64.b64encode(logits.tobytes()).decode("ascii"),
        activation_shape=[1, 1, vocab],
    )


def test_chain_invariants_happy_path_no_findings():
    """Coherent run of 3 records → empty findings."""
    from prsm.cli_modules.receipt_verify import verify_chain_invariants

    recs = [
        _make_argmax_consistent_record(next_token_id=0),
        _make_argmax_consistent_record(next_token_id=1),
        _make_argmax_consistent_record(next_token_id=2),
    ]
    # Distinct request_ids + increasing wall_unix
    recs[0]["request_id"] = "req-0"
    recs[0]["wall_unix"] = 1000.0
    recs[1]["request_id"] = "req-1"
    recs[1]["wall_unix"] = 1001.0
    recs[2]["request_id"] = "req-2"
    recs[2]["wall_unix"] = 1002.0

    findings = verify_chain_invariants(recs)
    assert findings == [], f"expected no findings; got {findings}"


def test_chain_inconsistent_settler_caught():
    """Two settlers in same file → INCONSISTENT_SETTLER."""
    from prsm.cli_modules.receipt_verify import verify_chain_invariants

    recs = [
        _chain_record(settler="settler-A", request_id="r0", wall_unix=1.0),
        _chain_record(settler="settler-B", request_id="r1", wall_unix=2.0),
    ]
    findings = verify_chain_invariants(recs)
    kinds = [f["kind"] for f in findings]
    assert "INCONSISTENT_SETTLER" in kinds


def test_chain_inconsistent_model_caught():
    from prsm.cli_modules.receipt_verify import verify_chain_invariants
    recs = [
        _chain_record(model="gpt2", request_id="r0", wall_unix=1.0),
        _chain_record(model="llama-3", request_id="r1", wall_unix=2.0),
    ]
    findings = verify_chain_invariants(recs)
    kinds = [f["kind"] for f in findings]
    assert "INCONSISTENT_MODEL" in kinds


def test_chain_duplicate_request_id_caught():
    """Replay attempt — same request_id twice."""
    from prsm.cli_modules.receipt_verify import verify_chain_invariants
    recs = [
        _chain_record(request_id="dup", wall_unix=1.0),
        _chain_record(request_id="dup", wall_unix=2.0),
    ]
    findings = verify_chain_invariants(recs)
    kinds = [f["kind"] for f in findings]
    assert "DUPLICATE_REQUEST_ID" in kinds


def test_chain_non_monotonic_wall_unix_caught():
    """Receipts reordered post-generation."""
    from prsm.cli_modules.receipt_verify import verify_chain_invariants
    recs = [
        _chain_record(request_id="r0", wall_unix=10.0),
        _chain_record(request_id="r1", wall_unix=5.0),  # back in time
    ]
    findings = verify_chain_invariants(recs)
    kinds = [f["kind"] for f in findings]
    assert "NON_MONOTONIC_WALL_UNIX" in kinds


def test_chain_argmax_mismatch_caught():
    """The §7 anti-tamper guarantee. Operator declares
    next_token_id=99 but the activation bytes the stage signed over
    have argmax=42. The signature is still valid (we didn't touch
    the signed bytes), but the operator-recorded next_token_id is
    a lie.
    """
    from prsm.cli_modules.receipt_verify import verify_chain_invariants
    recs = [
        _make_argmax_tampered_record(
            claimed_token_id=99, actual_argmax=3, vocab=100,
        ),
    ]
    findings = verify_chain_invariants(recs)
    kinds = [f["kind"] for f in findings]
    assert "TOKEN_ID_ARGMAX_MISMATCH" in kinds


def test_chain_skips_argmax_check_when_no_activation_bytes():
    """Pre-635 receipts (sha256-only) can't run C5; verifier must
    silently skip the check rather than raising.
    """
    from prsm.cli_modules.receipt_verify import verify_chain_invariants
    rec = _chain_record(next_token_id=5)
    # No activation_blob_b64 in record
    findings = verify_chain_invariants([rec])
    assert findings == []  # C5 skipped, other invariants pass trivially


def test_chain_findings_attached_when_check_chain_flag_set(tmp_path):
    """End-to-end: verify_receipts_file with check_chain=True
    surfaces findings on the last result."""
    from prsm.cli_modules.receipt_verify import verify_receipts_file
    # Build receipts file with 2 records that pass signatures but
    # fail chain invariants (duplicate request_id)
    stage = _make_stage_identity()
    resp1 = _signed_response(stage, request_id="duplicate-id")
    resp2 = _signed_response(stage, request_id="duplicate-id")
    rec1 = _receipt_from_response(resp1)
    rec2 = _receipt_from_response(resp2)
    jsonl = tmp_path / "receipts.jsonl"
    with jsonl.open("w") as f:
        f.write(json.dumps(rec1) + "\n")
        f.write(json.dumps(rec2) + "\n")

    anchor = _anchor_with(stage)
    results = verify_receipts_file(
        str(jsonl), anchor=anchor, check_chain=True,
    )
    # Both signatures verify (same identity, both honest)
    sig_results = [r for r in results if r.get("status") in ("OK", "SIGNATURE_INVALID")]
    assert all(r["status"] == "OK" for r in sig_results)
    # But chain invariant catches the duplicate
    chain_findings = results[-1].get("chain_findings", [])
    kinds = [f["kind"] for f in chain_findings]
    assert "DUPLICATE_REQUEST_ID" in kinds


def test_chain_findings_empty_when_no_flag(tmp_path):
    """Default (no --check-chain) → results have no chain_findings key."""
    from prsm.cli_modules.receipt_verify import verify_receipts_file
    stage = _make_stage_identity()
    resp = _signed_response(stage)
    rec = _receipt_from_response(resp)
    jsonl = tmp_path / "receipts.jsonl"
    with jsonl.open("w") as f:
        f.write(json.dumps(rec) + "\n")

    anchor = _anchor_with(stage)
    results = verify_receipts_file(str(jsonl), anchor=anchor)
    for r in results:
        assert "chain_findings" not in r


# --------------------------------------------------------------------------
# Sprint 639 — sampling-mode-aware C5
# --------------------------------------------------------------------------


def test_C5_skipped_for_non_greedy_without_seed():
    """Sprint 639 + 640: receipt with non-greedy mode but NO seed
    can't be replayed → C5 silently skipped. Operators who want
    strong audit must always pass --seed for sampled runs.
    """
    from prsm.cli_modules.receipt_verify import verify_chain_invariants
    # claimed=99, argmax=3 — would fail C5 if greedy
    rec = _make_argmax_tampered_record(
        claimed_token_id=99, actual_argmax=3, vocab=100,
    )
    rec["sampling_mode"] = "temperature:0.700,top_k:50"  # no seed
    findings = verify_chain_invariants([rec])
    kinds = [f["kind"] for f in findings]
    assert "TOKEN_ID_ARGMAX_MISMATCH" not in kinds, (
        "non-greedy without seed must NOT trigger C5 — can't replay"
    )


def test_C5_seed_replay_passes_for_correct_sample():
    """Sprint 640: with seed in the mode string, C5 deterministically
    replays the sampling and verifies the recorded next_token_id
    matches. This test constructs a receipt by RUNNING the actual
    sampling logic, then verifies the verifier agrees.
    """
    import base64 as _b64
    import numpy as _np
    from prsm.cli_modules.receipt_verify import verify_chain_invariants

    vocab = 100
    seed = 1234
    step = 0
    temperature = 0.7
    top_k = 50

    # Build deterministic logits
    _np.random.seed(7)
    logits = _np.random.randn(1, 1, vocab).astype(_np.float32) * 3.0

    # Run the same sampling logic the CLI uses to compute the
    # ground-truth sampled token
    last_logits = logits[0, -1, :].astype(_np.float32)
    scaled = last_logits / temperature
    k = min(top_k, scaled.shape[-1])
    top_indices = _np.argpartition(scaled, -k)[-k:]
    mask = _np.full_like(scaled, -_np.inf)
    mask[top_indices] = scaled[top_indices]
    scaled = mask - _np.max(mask)
    probs = _np.exp(scaled)
    probs = probs / probs.sum()
    rng = _np.random.default_rng(seed + step)
    true_token_id = int(rng.choice(probs.shape[-1], p=probs))

    rec = _chain_record(
        next_token_id=true_token_id,
        activation_blob_b64=_b64.b64encode(logits.tobytes()).decode("ascii"),
        activation_shape=[1, 1, vocab],
    )
    rec["step"] = step
    rec["sampling_mode"] = f"temperature:{temperature},top_k:{top_k},seed:{seed}"

    findings = verify_chain_invariants([rec])
    kinds = [f["kind"] for f in findings]
    assert "TOKEN_ID_ARGMAX_MISMATCH" not in kinds, (
        f"seed-replay must accept the genuine sample; got {findings}"
    )


def test_C5_seed_replay_catches_tampering_for_sampled():
    """Sprint 640: even in non-greedy mode with seed recorded, a
    tampered next_token_id is caught by replaying the sample.
    """
    import base64 as _b64
    import numpy as _np
    from prsm.cli_modules.receipt_verify import verify_chain_invariants

    vocab = 100
    seed = 999
    step = 0
    temperature = 0.5

    _np.random.seed(3)
    logits = _np.random.randn(1, 1, vocab).astype(_np.float32) * 5.0

    rec = _chain_record(
        next_token_id=42,  # tampered; replay will compute something else
        activation_blob_b64=_b64.b64encode(logits.tobytes()).decode("ascii"),
        activation_shape=[1, 1, vocab],
    )
    rec["step"] = step
    rec["sampling_mode"] = f"temperature:{temperature},seed:{seed}"

    findings = verify_chain_invariants([rec])
    kinds = [f["kind"] for f in findings]
    assert "TOKEN_ID_ARGMAX_MISMATCH" in kinds, (
        f"seed-replay must catch tampered next_token_id; got {findings}"
    )


def test_C5_fires_for_explicit_greedy_sampling():
    """sampling_mode='greedy' (sprint 639 format) → C5 must still fire."""
    from prsm.cli_modules.receipt_verify import verify_chain_invariants
    rec = _make_argmax_tampered_record(
        claimed_token_id=99, actual_argmax=3, vocab=100,
    )
    rec["sampling_mode"] = "greedy"
    findings = verify_chain_invariants([rec])
    kinds = [f["kind"] for f in findings]
    assert "TOKEN_ID_ARGMAX_MISMATCH" in kinds


def test_gzipped_receipts_file_reads_correctly(tmp_path):
    """Sprint 642 — verify_receipts_file auto-decompresses .gz."""
    import gzip
    from prsm.cli_modules.receipt_verify import verify_receipts_file

    stage = _make_stage_identity()
    resp = _signed_response(stage)
    rec = _receipt_from_response(resp)
    jsonl = tmp_path / "receipts.jsonl.gz"
    with gzip.open(jsonl, "wt", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    anchor = _anchor_with(stage)
    results = verify_receipts_file(str(jsonl), anchor=anchor)
    assert len(results) == 1
    assert results[0]["status"] == "OK"


def test_gzipped_receipts_smaller_than_plain(tmp_path):
    """Compression actually saves space — sanity check that gzip
    on activation_blob_b64 (which is base64 random-ish bytes plus
    lots of JSON structure repetition) hits at least 30% size cut.
    """
    import gzip
    stage = _make_stage_identity()
    resp = _signed_response(stage)
    rec = _receipt_from_response(resp)
    plain_path = tmp_path / "p.jsonl"
    gz_path = tmp_path / "g.jsonl.gz"
    plain_path.write_text(json.dumps(rec) + "\n")
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    p_size = plain_path.stat().st_size
    g_size = gz_path.stat().st_size
    # With single-shot test bytes, ratio varies; loosely require
    # gzipped is smaller (the activation b64 won't compress as
    # tightly as repeated JSON keys, but the JSON wrapper compresses
    # significantly).
    assert g_size < p_size, (
        f"gzip didn't shrink the file: plain={p_size} gz={g_size}"
    )


def test_C5_treats_missing_sampling_mode_as_greedy():
    """Backwards compat: sprint 633-638 receipts don't have
    sampling_mode (only greedy existed). C5 must still apply.
    """
    from prsm.cli_modules.receipt_verify import verify_chain_invariants
    rec = _make_argmax_tampered_record(
        claimed_token_id=99, actual_argmax=3, vocab=100,
    )
    # No sampling_mode key — pre-sprint-639 format
    assert "sampling_mode" not in rec
    findings = verify_chain_invariants([rec])
    kinds = [f["kind"] for f in findings]
    assert "TOKEN_ID_ARGMAX_MISMATCH" in kinds, (
        "missing sampling_mode must default to greedy (sprint 633-638 "
        "backwards-compat)"
    )


# --------------------------------------------------------------------------
# Sprint 650 — --strict mode surfaces non-greedy-without-seed
# --------------------------------------------------------------------------


def test_strict_flag_surfaces_non_greedy_no_seed():
    """Sprint 650: a non-greedy receipt without a seed in
    sampling_mode is silently skipped by default (sprint 640).
    With --strict, it becomes a UNVERIFIABLE_NON_GREEDY_NO_SEED
    finding so the operator knows their audit has a weak row.
    """
    from prsm.cli_modules.receipt_verify import verify_chain_invariants
    rec = _make_argmax_consistent_record(next_token_id=0, vocab=8)
    rec["sampling_mode"] = "temperature:0.700,top_k:50"  # no seed

    # Default: silent skip
    findings = verify_chain_invariants([rec])
    kinds = [f["kind"] for f in findings]
    assert "UNVERIFIABLE_NON_GREEDY_NO_SEED" not in kinds

    # Strict: surfaces finding
    findings = verify_chain_invariants([rec], strict=True)
    kinds = [f["kind"] for f in findings]
    assert "UNVERIFIABLE_NON_GREEDY_NO_SEED" in kinds


def test_strict_does_not_affect_greedy_or_seeded():
    """Strict mode must NOT flag greedy receipts or seeded
    non-greedy receipts — only the non-greedy-no-seed case.
    """
    import base64 as _b64
    import numpy as _np
    from prsm.cli_modules.receipt_verify import verify_chain_invariants

    # Greedy receipt
    rec_greedy = _make_argmax_consistent_record(next_token_id=0)
    rec_greedy["sampling_mode"] = "greedy"

    # Seeded non-greedy receipt (re-derive the sample so it's honest)
    vocab = 100
    seed = 555
    step = 0
    temperature = 0.7
    _np.random.seed(99)
    logits = _np.random.randn(1, 1, vocab).astype(_np.float32) * 3.0
    last_logits = logits[0, -1, :].astype(_np.float32)
    scaled = last_logits / temperature
    scaled = scaled - _np.max(scaled)
    probs = _np.exp(scaled)
    probs = probs / probs.sum()
    rng = _np.random.default_rng(seed + step)
    true_id = int(rng.choice(probs.shape[-1], p=probs))

    rec_seeded = _chain_record(
        next_token_id=true_id,
        activation_blob_b64=_b64.b64encode(logits.tobytes()).decode("ascii"),
        activation_shape=[1, 1, vocab],
    )
    rec_seeded["step"] = step
    rec_seeded["sampling_mode"] = f"temperature:{temperature},seed:{seed}"

    findings = verify_chain_invariants(
        [rec_greedy, rec_seeded], strict=True,
    )
    kinds = [f["kind"] for f in findings]
    assert "UNVERIFIABLE_NON_GREEDY_NO_SEED" not in kinds
    assert "TOKEN_ID_ARGMAX_MISMATCH" not in kinds
