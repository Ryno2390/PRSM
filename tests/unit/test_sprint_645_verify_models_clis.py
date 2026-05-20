"""Sprint 645 — CliRunner-based unit tests for the operator
audit + discovery commands.

Sprint 636 unit-tested the verification CORE
(`receipt_verify.verify_receipts_file` etc.) but didn't cover the
Click command itself — exit codes, rendering, --check-chain wiring,
anchor-resolution failure handling. Same gap for `prsm node models`
(sprint 638): the core registry surface has list_models tests but
the CLI rendering doesn't.

These tests close both gaps so the operator-facing commands are
CI-defended end-to-end.
"""
from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import node
from prsm.compute.chain_rpc.protocol import RunLayerSliceResponse
from prsm.compute.tee.models import TEEType
from prsm.node.identity import generate_node_identity


@pytest.fixture
def runner():
    return CliRunner()


# --------------------------------------------------------------------------
# Helpers shared with sprint 644 — kept self-contained
# --------------------------------------------------------------------------


def _signed_receipt(stage_identity, *, request_id="r0"):
    """Return a receipt dict in sprint-635+ format."""
    import numpy as np
    logits = np.zeros((1, 1, 8), dtype=np.float32)
    logits[0, 0, 5] = 99.0
    resp = RunLayerSliceResponse.sign(
        identity=stage_identity,
        request_id=request_id,
        activation_blob=logits.tobytes(),
        activation_shape=(1, 1, 8),
        activation_dtype="float32",
        duration_seconds=0.1,
        tee_attestation=b"sw-attest",
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
    )
    return {
        "step": 0,
        "wall_unix": 1000.0,
        "request_id": resp.request_id,
        "settler_node_id": "settler-A",
        "stage_node_id": resp.stage_node_id,
        "stage_signature_b64": resp.stage_signature_b64,
        "model_id": "gpt2",
        "layer_range": [0, 12],
        "activation_shape": list(resp.activation_shape),
        "activation_dtype": resp.activation_dtype,
        "activation_sha256": "ignored",
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
        "next_token_id": 5,
        "next_token_text": " five",
        "sampling_mode": "greedy",
    }


def _anchor_with(identity):
    a = MagicMock()
    a.lookup = MagicMock(return_value=identity.public_key_b64)
    return a


# --------------------------------------------------------------------------
# verify-receipts CLI
# --------------------------------------------------------------------------


def test_verify_receipts_anchor_missing_exits_2(runner, tmp_path):
    """No anchor available → exit code 2 + breadcrumb."""
    receipts = tmp_path / "r.jsonl"
    receipts.write_text("{}\n")
    with patch(
        "prsm.node.inference_wiring._build_anchor_or_none",
        return_value=None,
    ):
        result = runner.invoke(
            node, ["verify-receipts", str(receipts)],
        )
    assert result.exit_code == 2, result.output
    assert "No anchor available" in result.output
    assert "PRSM_NETWORK" in result.output


def test_verify_receipts_happy_path_exit_0(runner, tmp_path):
    """All receipts verify → exit 0 + token-by-token render."""
    stage = generate_node_identity(display_name="stage")
    receipts = tmp_path / "r.jsonl"
    receipts.write_text(json.dumps(_signed_receipt(stage)) + "\n")
    with patch(
        "prsm.node.inference_wiring._build_anchor_or_none",
        return_value=_anchor_with(stage),
    ):
        result = runner.invoke(
            node, ["verify-receipts", str(receipts)],
        )
    assert result.exit_code == 0, result.output
    assert "1/1 receipts verified" in result.output


def test_verify_receipts_bad_signature_exit_1(runner, tmp_path):
    """Tampered signature → exit 1 + SIGNATURE_INVALID rendered."""
    stage = generate_node_identity(display_name="stage")
    rec = _signed_receipt(stage)
    # Flip a byte in the b64 signature → invalid
    sig_bytes = bytearray(base64.b64decode(rec["stage_signature_b64"]))
    sig_bytes[0] ^= 0x01
    rec["stage_signature_b64"] = base64.b64encode(
        bytes(sig_bytes),
    ).decode("ascii")
    receipts = tmp_path / "r.jsonl"
    receipts.write_text(json.dumps(rec) + "\n")
    with patch(
        "prsm.node.inference_wiring._build_anchor_or_none",
        return_value=_anchor_with(stage),
    ):
        result = runner.invoke(
            node, ["verify-receipts", str(receipts)],
        )
    assert result.exit_code == 1, result.output
    assert "SIGNATURE_INVALID" in result.output


def test_verify_receipts_json_format(runner, tmp_path):
    """--format json → parseable + canonical fields."""
    stage = generate_node_identity()
    receipts = tmp_path / "r.jsonl"
    receipts.write_text(json.dumps(_signed_receipt(stage)) + "\n")
    with patch(
        "prsm.node.inference_wiring._build_anchor_or_none",
        return_value=_anchor_with(stage),
    ):
        result = runner.invoke(
            node, ["verify-receipts", str(receipts), "--format", "json"],
        )
    assert result.exit_code == 0
    idx = result.output.find("{")
    payload = json.loads(result.output[idx:])
    for field in ("receipts_path", "total", "verified", "results"):
        assert field in payload


def test_verify_receipts_check_chain_flag_renders_chain_block(
    runner, tmp_path,
):
    """--check-chain → invariants block renders + exit reflects findings."""
    stage = generate_node_identity()
    receipts = tmp_path / "r.jsonl"
    receipts.write_text(json.dumps(_signed_receipt(stage)) + "\n")
    with patch(
        "prsm.node.inference_wiring._build_anchor_or_none",
        return_value=_anchor_with(stage),
    ):
        result = runner.invoke(
            node, [
                "verify-receipts", str(receipts), "--check-chain",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "chain-of-custody invariants OK" in result.output


def test_verify_receipts_check_chain_catches_duplicate(
    runner, tmp_path,
):
    """Two receipts with the same request_id → DUPLICATE_REQUEST_ID."""
    stage = generate_node_identity()
    rec1 = _signed_receipt(stage, request_id="dup")
    rec2 = _signed_receipt(stage, request_id="dup")
    receipts = tmp_path / "r.jsonl"
    receipts.write_text(
        json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n",
    )
    with patch(
        "prsm.node.inference_wiring._build_anchor_or_none",
        return_value=_anchor_with(stage),
    ):
        result = runner.invoke(
            node, [
                "verify-receipts", str(receipts), "--check-chain",
            ],
        )
    assert result.exit_code == 1, result.output
    assert "DUPLICATE_REQUEST_ID" in result.output


# --------------------------------------------------------------------------
# `prsm node models` CLI (sprint 638)
# --------------------------------------------------------------------------


def test_models_no_root_set_exits_1(runner, monkeypatch):
    """No PRSM_MODEL_REGISTRY_ROOT + no --registry-root → exit 1 +
    actionable breadcrumb."""
    monkeypatch.delenv("PRSM_MODEL_REGISTRY_ROOT", raising=False)
    result = runner.invoke(node, ["models"])
    assert result.exit_code == 1, result.output
    assert "Registry root not configured" in result.output


def test_models_empty_registry_exits_1(runner, tmp_path):
    """Empty registry root → exit 1 + "no models" message."""
    # Create an empty registry directory (FilesystemModelRegistry
    # requires the root exist).
    (tmp_path / "empty").mkdir()
    result = runner.invoke(
        node, ["models", "--registry-root", str(tmp_path / "empty")],
    )
    assert result.exit_code == 1
    assert "No models registered" in result.output


def test_models_lists_registered_model(runner, tmp_path):
    """Registry with one published model → exit 0 + model_id printed
    with publisher + shard count + layer range.
    """
    from prsm.compute.model_registry.registry import (
        FilesystemModelRegistry,
    )
    from prsm.compute.model_registry.models import (
        ManifestShardEntry, ModelManifest,
    )
    from prsm.node.identity import generate_node_identity

    pub = generate_node_identity(display_name="test-publisher")
    root = tmp_path / "registry"
    root.mkdir()
    registry = FilesystemModelRegistry(root=root)

    shard = ManifestShardEntry(
        shard_id="s0",
        shard_index=0,
        tensor_shape=(1, 8),
        sha256="0" * 64,
        size_bytes=8,
        layer_range=(0, 12),
    )
    manifest = ModelManifest(
        model_id="testmodel",
        model_name="test-model",
        publisher_node_id=pub.node_id,
        total_shards=1,
        shards=(shard,),
        published_at=1000.0,
    )
    # FilesystemModelRegistry.register signs the manifest with the
    # publisher's identity and writes manifest.json + the shard
    # subdir. Use the lower-level path so we don't need real
    # tensor bytes.
    import json as _json
    import base64 as _b64
    model_dir = root / "testmodel"
    model_dir.mkdir()
    # NodeIdentity.sign returns a base64-encoded signature str.
    # ModelManifest.publisher_signature is bytes, so decode.
    sig_b64 = pub.sign(manifest.signing_payload())
    signed_manifest = ModelManifest(
        model_id=manifest.model_id,
        model_name=manifest.model_name,
        publisher_node_id=manifest.publisher_node_id,
        total_shards=manifest.total_shards,
        shards=manifest.shards,
        published_at=manifest.published_at,
        publisher_signature=_b64.b64decode(sig_b64),
    )
    (model_dir / "manifest.json").write_text(
        _json.dumps(signed_manifest.to_dict()),
    )

    result = runner.invoke(
        node, ["models", "--registry-root", str(root)],
    )
    assert result.exit_code == 0, result.output
    assert "testmodel" in result.output
    assert pub.node_id[:16] in result.output  # truncated publisher
    assert "[0, 12)" in result.output  # layer range rendered


def test_models_json_format(runner, tmp_path):
    """--format json on empty root still produces parseable JSON."""
    (tmp_path / "empty").mkdir()
    result = runner.invoke(node, [
        "models",
        "--registry-root", str(tmp_path / "empty"),
        "--format", "json",
    ])
    # Exit 1 because empty; output still parseable
    idx = result.output.find("{")
    payload = json.loads(result.output[idx:])
    assert payload["total"] == 0
    assert payload["models"] == []
