"""Sprint 292 — HTTP + MCP surface for privacy-claim verification.

Wraps the sprint-292 verification primitive in:
  POST /compute/receipt/verify             body: {receipt, ...}
  prsm_verify_inference_privacy MCP tool

End-users running PRSM via Claude Code / Gemini CLI can ask
"did my prompt really run under hardware-attested TEE?" and
get a verdict.
"""
from __future__ import annotations

import base64
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.compute.inference.executor import (
    SOFTWARE_TEE_ATTESTATION_PREFIX,
)
from prsm.compute.inference.models import ContentTier
from prsm.compute.inference.receipt import (
    InferenceReceipt, sign_receipt,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_verify_inference_privacy,
)
from prsm.node.api import create_api_app
from prsm.node.identity import generate_node_identity


def _signed_receipt(
    *, use_dev_only: bool = True,
    privacy_tier: PrivacyLevel = PrivacyLevel.STANDARD,
):
    identity = generate_node_identity("verifier-endpoint-test")
    if use_dev_only:
        att = (
            SOFTWARE_TEE_ATTESTATION_PREFIX
            + hashlib.sha384(b"sw-tee:j").digest()
        )
    else:
        att = b"HW_VENDOR_QUOTE_" + b"\x00" * 48
    eps = (
        0.0 if privacy_tier == PrivacyLevel.NONE else 8.0
    )
    receipt = InferenceReceipt(
        job_id="infer-job-x",
        request_id="req-x",
        model_id="mock-llama-3-8b",
        content_tier=ContentTier.A,
        privacy_tier=privacy_tier,
        epsilon_spent=eps,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=att,
        output_hash=hashlib.sha256(b"out").digest(),
        duration_seconds=0.1,
        cost_ftns="0.01",
        settler_signature=b"\x00" * 64,
        settler_node_id="",
    )
    receipt = sign_receipt(receipt, identity)
    pub_b64 = base64.b64encode(
        identity.public_key_bytes
    ).decode("ascii")
    return receipt, identity, pub_b64


def _receipt_to_payload(receipt: InferenceReceipt) -> dict:
    """JSON-safe payload mirroring what an end-user gets back
    from POST /compute/inference."""
    return {
        "job_id": receipt.job_id,
        "request_id": receipt.request_id,
        "model_id": receipt.model_id,
        "content_tier": receipt.content_tier.value,
        "privacy_tier": receipt.privacy_tier.value,
        "epsilon_spent": receipt.epsilon_spent,
        "tee_type": receipt.tee_type.value,
        "tee_attestation_b64": base64.b64encode(
            receipt.tee_attestation
        ).decode("ascii"),
        "output_hash_b64": base64.b64encode(
            receipt.output_hash
        ).decode("ascii"),
        "duration_seconds": receipt.duration_seconds,
        "cost_ftns": receipt.cost_ftns,
        "settler_signature_b64": base64.b64encode(
            receipt.settler_signature
        ).decode("ascii"),
        "settler_node_id": receipt.settler_node_id,
    }


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── HTTP: happy path ─────────────────────────────────────


def test_endpoint_valid_dev_only_signed():
    receipt, identity, pub_b64 = _signed_receipt()
    payload = _receipt_to_payload(receipt)
    resp = _client().post(
        "/compute/receipt/verify",
        json={
            "receipt": payload,
            "public_key_b64": pub_b64,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["signature_valid"] is True
    assert body["dp_noise_applied"] is True
    assert body["hardware_attested"] is False
    # Default posture: ok=True even on dev-only
    assert body["ok"] is True


def test_endpoint_require_hardware_attestation_fails_on_dev_only():
    receipt, identity, pub_b64 = _signed_receipt()
    resp = _client().post(
        "/compute/receipt/verify",
        json={
            "receipt": _receipt_to_payload(receipt),
            "public_key_b64": pub_b64,
            "require_hardware_attestation": True,
        },
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["hardware_attested"] is False
    assert any(
        "dev-only" in r.lower() for r in body["reasons"]
    )


def test_endpoint_require_hardware_passes_on_real():
    receipt, identity, pub_b64 = _signed_receipt(
        use_dev_only=False,
    )
    resp = _client().post(
        "/compute/receipt/verify",
        json={
            "receipt": _receipt_to_payload(receipt),
            "public_key_b64": pub_b64,
            "require_hardware_attestation": True,
        },
    )
    body = resp.json()
    assert body["ok"] is True
    assert body["hardware_attested"] is True


def test_endpoint_422_missing_receipt():
    resp = _client().post(
        "/compute/receipt/verify",
        json={"public_key_b64": "abc"},
    )
    assert resp.status_code == 422


def test_endpoint_422_malformed_receipt():
    resp = _client().post(
        "/compute/receipt/verify",
        json={
            "receipt": {"not": "a receipt"},
            "public_key_b64": "abc",
        },
    )
    assert resp.status_code == 422


def test_endpoint_422_missing_verifier_key():
    receipt, _, _ = _signed_receipt()
    resp = _client().post(
        "/compute/receipt/verify",
        json={"receipt": _receipt_to_payload(receipt)},
    )
    # Either reject upfront (422) or run + surface "no
    # verifier key" reason. Current contract: missing key
    # → 422.
    assert resp.status_code == 422


def test_endpoint_bad_signature_returns_200_with_invalid():
    receipt, _, _ = _signed_receipt()
    # Wrong pubkey
    other = generate_node_identity("wrong-key")
    wrong_b64 = base64.b64encode(
        other.public_key_bytes
    ).decode("ascii")
    resp = _client().post(
        "/compute/receipt/verify",
        json={
            "receipt": _receipt_to_payload(receipt),
            "public_key_b64": wrong_b64,
        },
    )
    body = resp.json()
    assert body["signature_valid"] is False
    assert body["ok"] is False


# ── MCP tool ─────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_verify_inference_privacy" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_receipt_rejected_client_side():
    r = await handle_prsm_verify_inference_privacy(
        {"public_key_b64": "abc"},
    )
    assert "receipt" in r.lower()


@pytest.mark.asyncio
async def test_mcp_renders_verdict():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "ok": True,
            "reasons": [],
            "signature_valid": True,
            "dp_noise_applied": True,
            "hardware_attested": False,
            "multi_stage_envelope_present": False,
            "privacy_tier": "standard",
            "epsilon_spent": 8.0,
            "expected_epsilon": 8.0,
        }),
    ) as mock_call:
        r = await handle_prsm_verify_inference_privacy({
            "receipt": {"job_id": "x"},
            "public_key_b64": "abc",
        })
    args = mock_call.await_args[0]
    assert args[0] == "POST"
    assert args[1] == "/compute/receipt/verify"
    # Verdict rendered
    assert "VALID" in r.upper() or "ok" in r.lower()
    # Truth surfaces about hardware attestation
    assert "hardware" in r.lower() or "software" in r.lower()


@pytest.mark.asyncio
async def test_mcp_renders_warning_on_dev_only():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "ok": True,
            "reasons": [],
            "signature_valid": True,
            "dp_noise_applied": True,
            "hardware_attested": False,
            "multi_stage_envelope_present": False,
            "privacy_tier": "standard",
            "epsilon_spent": 8.0,
            "expected_epsilon": 8.0,
        }),
    ):
        r = await handle_prsm_verify_inference_privacy({
            "receipt": {"job_id": "x"},
            "public_key_b64": "abc",
        })
    # ⚠ or "software" marker visible for caller's trust
    # decision (default posture is permissive but the truth
    # is surfaced regardless)
    assert (
        "⚠" in r
        or "software" in r.lower()
        or "dev-only" in r.lower()
    )


@pytest.mark.asyncio
async def test_mcp_renders_require_hardware_failure():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "ok": False,
            "reasons": [
                "attestation is DEV-ONLY software fallback",
            ],
            "signature_valid": True,
            "dp_noise_applied": True,
            "hardware_attested": False,
            "multi_stage_envelope_present": False,
            "privacy_tier": "standard",
            "epsilon_spent": 8.0,
            "expected_epsilon": 8.0,
        }),
    ):
        r = await handle_prsm_verify_inference_privacy({
            "receipt": {"job_id": "x"},
            "public_key_b64": "abc",
            "require_hardware_attestation": True,
        })
    assert "REJECT" in r.upper() or "FAIL" in r.upper() or "❌" in r
    assert "dev-only" in r.lower() or "software" in r.lower()


@pytest.mark.asyncio
async def test_mcp_503_message():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "detail": "Receipt verifier surface not initialized",
        }),
    ):
        r = await handle_prsm_verify_inference_privacy({
            "receipt": {"job_id": "x"},
            "public_key_b64": "abc",
        })
    # Defensive — any non-verdict result surfaces detail
    assert (
        "not init" in r.lower()
        or "refused" in r.lower()
        or "error" in r.lower()
    )
