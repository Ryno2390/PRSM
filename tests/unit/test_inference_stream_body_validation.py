"""Sprint 155 — /compute/inference/stream body validation reordering.

Sibling endpoint to /compute/inference. Same bug class as sprints
153 + 154 fixed for /compute/forge + /compute/inference: validation
fired AFTER the executor 503 check, so any body-level error
(missing prompt/model_id, bad budget) leaked through to a 503
("Inference executor not initialized") instead of the right 4xx.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node_no_executor():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node.inference_executor = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _post(node, body):
    return _client(node).post("/compute/inference/stream", json=body)


class TestInferenceStreamBodyValidation:
    def test_missing_prompt_returns_400(self):
        resp = _post(_node_no_executor(), {
            "model_id": "x", "budget_ftns": 1.0,
        })
        assert resp.status_code == 400
        assert "prompt" in resp.json()["detail"].lower()

    def test_missing_model_id_returns_400(self):
        resp = _post(_node_no_executor(), {
            "prompt": "hi", "budget_ftns": 1.0,
        })
        assert resp.status_code == 400
        assert "model_id" in resp.json()["detail"].lower()

    def test_non_numeric_budget_returns_422(self):
        resp = _post(_node_no_executor(), {
            "prompt": "hi", "model_id": "x", "budget_ftns": "nope",
        })
        assert resp.status_code == 422
        assert "budget_ftns" in resp.json()["detail"].lower()

    def test_negative_budget_returns_422(self):
        resp = _post(_node_no_executor(), {
            "prompt": "hi", "model_id": "x", "budget_ftns": -3,
        })
        assert resp.status_code == 422
        assert "budget_ftns" in resp.json()["detail"].lower()

    def test_zero_budget_returns_422(self):
        resp = _post(_node_no_executor(), {
            "prompt": "hi", "model_id": "x", "budget_ftns": 0,
        })
        assert resp.status_code == 422
        assert "budget_ftns" in resp.json()["detail"].lower()

    def test_valid_body_reaches_executor_check_503(self):
        resp = _post(_node_no_executor(), {
            "prompt": "hi", "model_id": "x", "budget_ftns": 1.0,
        })
        assert resp.status_code == 503

    def test_bad_privacy_tier_returns_422(self):
        """Sprint 156 — bad PrivacyLevel enum → 422 (not 503)."""
        resp = _post(_node_no_executor(), {
            "prompt": "hi", "model_id": "x", "budget_ftns": 1.0,
            "privacy_tier": "INVALID",
        })
        assert resp.status_code == 422

    def test_bad_content_tier_returns_422(self):
        """Sprint 156 — bad ContentTier enum → 422 (not 503)."""
        resp = _post(_node_no_executor(), {
            "prompt": "hi", "model_id": "x", "budget_ftns": 1.0,
            "content_tier": "Z",
        })
        assert resp.status_code == 422
