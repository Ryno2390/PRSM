"""Sprint 235 — GET /compute/models endpoint.

End-users running prsm_inference need to know which model_id
values the node will accept BEFORE they submit a job. Pre-fix
the only path was reading the prsm_inference tool description
and hoping the operator hadn't customized the model registry.

Endpoint returns {"models": [...], "count": N} from
node.inference_executor.supported_models(). 503 when the
executor isn't initialized.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client(supported_models=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    if supported_models is None:
        node.inference_executor = None
    else:
        node.inference_executor = MagicMock()
        node.inference_executor.supported_models = MagicMock(
            return_value=supported_models,
        )
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_returns_supported_models():
    resp = _client([
        "mock-llama-3-8b", "mock-mistral-7b", "mock-phi-3",
    ]).get("/compute/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert "mock-llama-3-8b" in body["models"]
    assert "mock-mistral-7b" in body["models"]


def test_503_when_executor_unwired():
    resp = _client(None).get("/compute/models")
    assert resp.status_code == 503
    assert "executor" in resp.json()["detail"].lower()


def test_empty_models_list():
    resp = _client([]).get("/compute/models")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0
    assert resp.json()["models"] == []
