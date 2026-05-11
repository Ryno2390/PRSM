"""Sprint 253 — /compute/inference/stream persists signed receipt
to ReceiptStore on success.

Mirrors sprint 242's unary wiring. Pre-fix streaming inference
emitted the signed receipt on the wire but never persisted it,
so `prsm_receipt` + `prsm_receipts_list` saw only unary jobs.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.job_history import JobHistoryStore
from prsm.node.receipt_store import ReceiptStore


def _node():
    from prsm.node.identity import generate_node_identity
    node = MagicMock()
    # Real identity so sign_receipt works.
    node.identity = generate_node_identity()
    node.ftns_ledger = None
    node._payment_escrow = None
    node.privacy_budget = None
    node.inference_executor = MagicMock()
    node.inference_executor.supported_models = MagicMock(
        return_value=["mock-llama-3-8b"],
    )
    node.inference_executor.estimate_cost = AsyncMock(
        return_value=Decimal("0.05"),
    )
    node._job_history = JobHistoryStore()
    node._receipt_store = ReceiptStore()
    return node


def _client(node):
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _success_result():
    from prsm.compute.inference.models import (
        ContentTier, InferenceReceipt, InferenceResult,
    )
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    return InferenceResult(
        request_id="req-1",
        success=True,
        output="streamed output",
        error=None,
        receipt=InferenceReceipt(
            job_id="placeholder",
            request_id="req-1",
            model_id="mock-llama-3-8b",
            privacy_tier=PrivacyLevel.STANDARD,
            content_tier=ContentTier.A,
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=8.0,
            cost_ftns=Decimal("0.05"),
            duration_seconds=0.1,
            output_hash=b"\x00" * 32,
            tee_attestation=b"",
        ),
    )


async def _stream_success(*args, **kwargs):
    yield _success_result()


def test_streaming_success_persists_receipt():
    node = _node()
    node.inference_executor.execute_streaming = _stream_success
    resp = _client(node).post("/compute/inference/stream", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 200
    list(resp.iter_lines())
    # Exactly one receipt persisted.
    assert node._receipt_store.count() == 1
    receipts = node._receipt_store.list(offset=0, limit=10)
    r = receipts[0]
    # job_id rebound to the API-side streaming id.
    assert r["job_id"].startswith("infer-stream-")
    assert r["model_id"] == "mock-llama-3-8b"
    assert r["settler_node_id"] == node.identity.node_id


def test_streaming_signature_valid_on_persisted_receipt():
    """Signature on the persisted receipt must verify against
    the running node's identity — same invariant the wire-side
    receipt has."""
    from prsm.compute.inference.models import InferenceReceipt
    from prsm.compute.inference.receipt import verify_receipt
    node = _node()
    node.inference_executor.execute_streaming = _stream_success
    resp = _client(node).post("/compute/inference/stream", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 200
    list(resp.iter_lines())
    persisted = node._receipt_store.list(offset=0, limit=1)[0]
    rec = InferenceReceipt.from_dict(persisted)
    assert verify_receipt(
        rec, public_key_b64=node.identity.public_key_b64,
    )


def test_unwired_receipt_store_doesnt_block():
    node = _node()
    node._receipt_store = None  # force missing
    node.inference_executor.execute_streaming = _stream_success
    resp = _client(node).post("/compute/inference/stream", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 200
