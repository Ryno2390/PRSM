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
