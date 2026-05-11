"""Sprint 198 — /compute/inference + /compute/inference/stream
prompt-size cap.

Pre-fix both endpoints accept arbitrary-size prompts. A single
multi-MB prompt is a DoS vector: memory blow-up plus amplified
downstream cost through the LLM token economy. Sibling endpoint
/compute/query has the same 100KB cap via PRSM_MAX_QUERY_BYTES
(see api.py:1764). Inference path was missed.

Cap default 100KB. Env override: PRSM_MAX_INFERENCE_PROMPT_BYTES.
Returns 413 with structured detail. Validation runs BEFORE
executor-availability 503 (consistent with sprints 153-156).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.inference_executor = MagicMock()
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


class TestInferenceUnaryPromptCap:
    def test_oversized_prompt_returns_413(self):
        big = "x" * 1_000_000
        resp = _client().post("/compute/inference", json={
            "prompt": big, "model_id": "mock-llama-3-8b",
        })
        assert resp.status_code == 413
        assert "exceeds" in resp.json()["detail"].lower()

    def test_typical_prompt_passes_validation(self):
        resp = _client().post("/compute/inference", json={
            "prompt": "hi", "model_id": "mock-llama-3-8b",
        })
        assert resp.status_code != 413

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PRSM_MAX_INFERENCE_PROMPT_BYTES", "10")
        resp = _client().post("/compute/inference", json={
            "prompt": "long enough prompt to exceed 10 bytes",
            "model_id": "mock-llama-3-8b",
        })
        assert resp.status_code == 413


class TestInferenceStreamPromptCap:
    def test_oversized_prompt_returns_413(self):
        big = "x" * 1_000_000
        resp = _client().post("/compute/inference/stream", json={
            "prompt": big, "model_id": "mock-llama-3-8b",
        })
        assert resp.status_code == 413
        assert "exceeds" in resp.json()["detail"].lower()

    def test_typical_prompt_passes_validation(self):
        resp = _client().post("/compute/inference/stream", json={
            "prompt": "hi", "model_id": "mock-llama-3-8b",
        })
        assert resp.status_code != 413

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PRSM_MAX_INFERENCE_PROMPT_BYTES", "10")
        resp = _client().post("/compute/inference/stream", json={
            "prompt": "long enough prompt to exceed 10 bytes",
            "model_id": "mock-llama-3-8b",
        })
        assert resp.status_code == 413


class TestInvalidEnvFallback:
    def test_garbage_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(
            "PRSM_MAX_INFERENCE_PROMPT_BYTES", "not_an_int",
        )
        resp = _client().post("/compute/inference", json={
            "prompt": "hi", "model_id": "mock-llama-3-8b",
        })
        assert resp.status_code != 413
