"""Sprint 251 — /compute/inference writes JobHistoryStore.

Pre-fix inference jobs were invisible to prsm_jobs_list /
/compute/status (which read JobHistoryStore). Sprint 251 adds:
  - IN_PROGRESS put after job_id allocation
  - COMPLETED put after successful execution
  - FAILED put on exception path

route="inference" distinguishes from "qo_swarm" / "direct_llm".
Best-effort: history failures don't break the response.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.job_history import (
    JobHistoryStore, JobStatus,
)


def _client(executor_returns=None, executor_raises=None):
    from prsm.compute.inference.models import (
        ContentTier, InferenceReceipt, InferenceResult,
    )
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.identity.public_key_b64 = "x"
    node.identity.sign = MagicMock(return_value="sig")
    node.identity.verify = MagicMock(return_value=True)
    node.ftns_ledger = None
    node._payment_escrow = None
    node.privacy_budget = None
    node.inference_executor = MagicMock()
    node.inference_executor.supported_models = MagicMock(
        return_value=["mock-llama-3-8b"],
    )

    async def _exec(*a, **kw):
        if executor_raises:
            raise executor_raises
        return executor_returns

    node.inference_executor.execute = _exec
    node.inference_executor.estimate_cost = AsyncMock(
        return_value=Decimal("0.05"),
    )
    node._job_history = JobHistoryStore()
    node._receipt_store = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    ), node


def _make_result_success():
    from prsm.compute.inference.models import (
        ContentTier, InferenceReceipt, InferenceResult,
    )
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    return InferenceResult(
        request_id="req-1",
        success=True,
        output="hello world",
        error=None,
        receipt=InferenceReceipt(
            job_id="",  # rewritten by handler
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


def test_inference_writes_completed_to_history():
    client, node = _client(
        executor_returns=_make_result_success(),
    )
    resp = client.post("/compute/inference", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 200
    # JobHistoryStore now has a record for this job_id
    job_id = resp.json()["job_id"]
    record = node._job_history.get(job_id)
    assert record is not None
    assert record.status == JobStatus.COMPLETED
    assert record.route == "inference"
    assert record.response == "hello world"
    assert record.completed_at is not None


def test_inference_failure_writes_failed_to_history():
    client, node = _client(
        executor_raises=RuntimeError("executor exploded"),
    )
    resp = client.post("/compute/inference", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 500
    # Find the inference job in history
    failed = [
        r for r in node._job_history.list(
            status_filter=JobStatus.FAILED, limit=10, offset=0,
        )
    ]
    assert len(failed) == 1
    assert failed[0].route == "inference"
    assert "executor exploded" in failed[0].error


def test_inference_history_query_truncated_to_256_chars():
    """Long prompts get truncated in the history query field
    (preventing 10MB prompts from inflating history records)."""
    client, node = _client(
        executor_returns=_make_result_success(),
    )
    resp = client.post("/compute/inference", json={
        "prompt": "x" * 10_000,
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    record = node._job_history.get(job_id)
    assert len(record.query) <= 256


def test_inference_history_missing_doesnt_block_response():
    """node._job_history is None → handler still returns 200."""
    client, node = _client(
        executor_returns=_make_result_success(),
    )
    node._job_history = None  # force missing
    resp = client.post("/compute/inference", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 200
