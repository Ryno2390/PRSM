"""Sprint 252 — /compute/inference/stream writes JobHistoryStore.

Mirrors sprint 251 for the SSE-streaming sibling. Five terminal
sites in the event generator each record FAILED/COMPLETED to
JobHistoryStore; an IN_PROGRESS write fires before the generator
constructs.

route="inference_stream" distinguishes streaming inference from
unary inference + forge.
"""
from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.job_history import JobHistoryStore, JobStatus


def _node():
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
    node.inference_executor.estimate_cost = AsyncMock(
        return_value=Decimal("0.05"),
    )
    node._job_history = JobHistoryStore()
    node._receipt_store = None
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


async def _stream_success(*args, **kwargs):
    yield _success_result()


async def _stream_executor_error(*args, **kwargs):
    """Async generator that raises mid-iteration (the only way
    to surface as a generator-side exception in the catch-all)."""
    raise RuntimeError("executor blew up")
    yield  # unreachable but makes this an async generator


async def _stream_result_failure(*args, **kwargs):
    from prsm.compute.inference.models import InferenceResult
    yield InferenceResult(
        request_id="req-1",
        success=False,
        output="",
        error="bad input",
    )


async def _stream_empty(*args, **kwargs):
    # never yields a terminal
    if False:
        yield None


def test_stream_completed_writes_completed_to_history():
    node = _node()
    node.inference_executor.execute_streaming = _stream_success
    resp = _client(node).post("/compute/inference/stream", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 200
    # Drain the stream
    list(resp.iter_lines())
    # Find the streaming job (route="inference_stream")
    recs = node._job_history.list(
        status_filter=JobStatus.COMPLETED, limit=10, offset=0,
    )
    streaming = [r for r in recs if r.route == "inference_stream"]
    assert len(streaming) == 1
    assert streaming[0].response == "streamed output"


def test_stream_executor_exception_writes_failed():
    node = _node()
    node.inference_executor.execute_streaming = _stream_executor_error
    resp = _client(node).post("/compute/inference/stream", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 200
    list(resp.iter_lines())
    failed = [
        r for r in node._job_history.list(
            status_filter=JobStatus.FAILED, limit=10, offset=0,
        )
        if r.route == "inference_stream"
    ]
    assert len(failed) == 1
    assert "executor blew up" in failed[0].error


def test_stream_result_failure_writes_failed():
    node = _node()
    node.inference_executor.execute_streaming = _stream_result_failure
    resp = _client(node).post("/compute/inference/stream", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 200
    list(resp.iter_lines())
    failed = [
        r for r in node._job_history.list(
            status_filter=JobStatus.FAILED, limit=10, offset=0,
        )
        if r.route == "inference_stream"
    ]
    assert len(failed) == 1
    assert "bad input" in failed[0].error


def test_stream_exhausted_without_terminal_writes_failed():
    node = _node()
    node.inference_executor.execute_streaming = _stream_empty
    resp = _client(node).post("/compute/inference/stream", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 200
    list(resp.iter_lines())
    failed = [
        r for r in node._job_history.list(
            status_filter=JobStatus.FAILED, limit=10, offset=0,
        )
        if r.route == "inference_stream"
    ]
    assert len(failed) == 1
    assert "exhausted" in failed[0].error.lower()


def test_stream_in_progress_recorded_before_terminal():
    """IN_PROGRESS write fires before generator starts. With a
    successful generator that ALSO writes COMPLETED, only the
    final state is visible (LRU put replaces). So we test the
    record EXISTS with the latest state."""
    node = _node()
    node.inference_executor.execute_streaming = _stream_success
    resp = _client(node).post("/compute/inference/stream", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 200
    list(resp.iter_lines())
    # Total records: 1 (IN_PROGRESS overwritten by COMPLETED)
    assert node._job_history.count() == 1
