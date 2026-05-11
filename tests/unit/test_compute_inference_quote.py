"""Sprint 237 — POST /compute/inference/quote endpoint.

Pre-fix: prsm_inference was the only way to discover the FTNS
cost for an inference request — but submitting a request locks
escrow. End-users had no pre-flight cost-discovery surface.
InferenceExecutor.estimate_cost() was unreachable from HTTP.

Endpoint takes the same body shape as /compute/inference but
returns only {cost_ftns, model_id, ...} without executing.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client(estimate_return=Decimal("0.10"), supported=("mock-llama-3-8b",)):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.inference_executor = MagicMock()
    node.inference_executor.supported_models = MagicMock(
        return_value=list(supported),
    )
    node.inference_executor.estimate_cost = AsyncMock(
        return_value=estimate_return,
    )
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_quote_returns_cost():
    resp = _client(Decimal("0.25")).post(
        "/compute/inference/quote",
        json={"prompt": "hi", "model_id": "mock-llama-3-8b"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "mock-llama-3-8b"
    assert body["cost_ftns"] == "0.25"


def test_missing_prompt_rejected():
    resp = _client().post(
        "/compute/inference/quote",
        json={"model_id": "mock-llama-3-8b"},
    )
    assert resp.status_code == 400


def test_missing_model_id_rejected():
    resp = _client().post(
        "/compute/inference/quote",
        json={"prompt": "hi"},
    )
    assert resp.status_code == 400


def test_503_when_executor_unwired():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.inference_executor = None
    client = TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )
    resp = client.post(
        "/compute/inference/quote",
        json={"prompt": "hi", "model_id": "mock-llama-3-8b"},
    )
    assert resp.status_code == 503


def test_unsupported_model_400():
    """Estimate raises UnsupportedModelError → return 400."""
    from prsm.compute.inference.executor import UnsupportedModelError
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.inference_executor = MagicMock()
    node.inference_executor.supported_models = MagicMock(
        return_value=["mock-llama-3-8b"],
    )
    node.inference_executor.estimate_cost = AsyncMock(
        side_effect=UnsupportedModelError("Unknown model_id: bogus"),
    )
    client = TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )
    resp = client.post(
        "/compute/inference/quote",
        json={"prompt": "hi", "model_id": "bogus"},
    )
    assert resp.status_code == 400
    assert "unknown" in resp.json()["detail"].lower()


def test_oversized_prompt_413():
    """Inherits the sprint-198 prompt-size cap."""
    resp = _client().post(
        "/compute/inference/quote",
        json={
            "prompt": "x" * 1_000_000,
            "model_id": "mock-llama-3-8b",
        },
    )
    assert resp.status_code == 413
