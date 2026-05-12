"""Sprint 305a — TEE policy live dispatch enforcement.

When a /compute/inference or /compute/inference/stream
request carries an optional `tee_policy` field, the node
evaluates the policy against its OWN attestation blob
BEFORE doing any work. If the policy isn't satisfied, the
request is refused with 412 Precondition Failed.

This closes the §7 Enterprise Confidentiality Mode loop on
the compute side: layer-3 policy stops a non-attested node
from accepting an enterprise workload up front, rather than
relying solely on after-the-fact receipt verification.

The dispatch gate is composer-only on the operator side
(it refuses; it doesn't decrypt or execute anything risky)
— R-2026-05-08-1 preserved.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from prsm.compute.inference.attestation_backends import (
    SOFTWARE_TEE_ATTESTATION_PREFIX,
)
from prsm.node.api import create_api_app


def _client(node_attestation_blob=None):
    """Inference executor is intentionally left unwired so
    requests that pass the TEE-policy gate land on a 503
    or similar downstream status. The relevant assertion
    is 412-vs-not-412."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._tee_node_attestation_blob = node_attestation_blob
    node._content_filter_store = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node.inference_executor = None  # unwired → downstream 5xx
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _body(**extra):
    body = {
        "prompt": "hello",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    }
    body.update(extra)
    return body


def _software_blob() -> bytes:
    return SOFTWARE_TEE_ATTESTATION_PREFIX + b"\x00" * 32


def _hardware_blob() -> bytes:
    # SGX v3 magic — version=3 little-endian uint16.
    # Pad to the SGX minimum length (48 + 32 + 96 + 32 = 208).
    blob = bytearray(b"\x03\x00")
    blob.extend(b"\x00" * 206)
    return bytes(blob)


# ── /compute/inference dispatch gate ─────────────────


def test_inference_without_tee_policy_unaffected():
    """Backwards-compat: no tee_policy field → existing
    behavior preserved (the request flows past the gate
    and is refused by a downstream stage with 4xx/5xx —
    NOT 412)."""
    resp = _client().post("/compute/inference", json=_body())
    assert resp.status_code != 412


def test_inference_policy_none_passes_unattested_node():
    resp = _client(node_attestation_blob=None).post(
        "/compute/inference",
        json=_body(tee_policy={
            "min_attestation_tier": "none",
        }),
    )
    assert resp.status_code != 412


def test_inference_policy_software_refuses_unattested_node():
    resp = _client(node_attestation_blob=None).post(
        "/compute/inference",
        json=_body(tee_policy={
            "min_attestation_tier": "software",
        }),
    )
    assert resp.status_code == 412
    detail = resp.json()["detail"]
    # Surface the diagnostic so the requester knows WHY
    assert "TEE policy" in detail or "tee" in detail.lower()
    assert "effective_tier" in detail


def test_inference_policy_software_passes_software_node():
    resp = _client(node_attestation_blob=_software_blob()).post(
        "/compute/inference",
        json=_body(tee_policy={
            "min_attestation_tier": "software",
        }),
    )
    assert resp.status_code != 412


def test_inference_policy_hardware_refuses_software_node():
    resp = _client(node_attestation_blob=_software_blob()).post(
        "/compute/inference",
        json=_body(tee_policy={
            "min_attestation_tier": "hardware_unverified",
        }),
    )
    assert resp.status_code == 412
    detail = resp.json()["detail"]
    assert "hardware_unverified" in detail
    assert "software" in detail.lower()


def test_inference_policy_hardware_passes_hardware_node():
    resp = _client(node_attestation_blob=_hardware_blob()).post(
        "/compute/inference",
        json=_body(tee_policy={
            "min_attestation_tier": "hardware_unverified",
        }),
    )
    assert resp.status_code != 412


def test_inference_policy_vendor_allowlist_refuses():
    """SGX node, but the policy only allows AMD — refuse."""
    resp = _client(node_attestation_blob=_hardware_blob()).post(
        "/compute/inference",
        json=_body(tee_policy={
            "min_attestation_tier": "hardware_unverified",
            "allowed_vendors": ["amd-sev-snp"],
        }),
    )
    assert resp.status_code == 412


def test_inference_policy_invalid_tier_422():
    resp = _client().post(
        "/compute/inference",
        json=_body(tee_policy={
            "min_attestation_tier": "ultra-strong",
        }),
    )
    assert resp.status_code == 422


def test_inference_policy_not_a_dict_422():
    resp = _client().post(
        "/compute/inference",
        json=_body(tee_policy="not-a-dict"),
    )
    assert resp.status_code == 422


# ── /compute/inference/stream — same gate ─────────


def test_stream_inference_policy_software_refuses_unattested():
    resp = _client(node_attestation_blob=None).post(
        "/compute/inference/stream",
        json=_body(tee_policy={
            "min_attestation_tier": "software",
        }),
    )
    assert resp.status_code == 412


def test_stream_inference_policy_none_unaffected():
    resp = _client(node_attestation_blob=None).post(
        "/compute/inference/stream",
        json=_body(tee_policy={
            "min_attestation_tier": "none",
        }),
    )
    assert resp.status_code != 412


def test_stream_inference_without_policy_unaffected():
    resp = _client().post(
        "/compute/inference/stream", json=_body(),
    )
    assert resp.status_code != 412


# ── Diagnostic surfacing ─────────────────────────────


def test_412_detail_includes_effective_and_required_tier():
    resp = _client(node_attestation_blob=None).post(
        "/compute/inference",
        json=_body(tee_policy={
            "min_attestation_tier": "hardware_verified",
        }),
    )
    assert resp.status_code == 412
    detail = resp.json()["detail"]
    assert "effective_tier=none" in detail
    assert "hardware_verified" in detail
