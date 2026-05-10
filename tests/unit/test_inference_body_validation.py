"""Sprint 154 — /compute/inference must validate body fields BEFORE
the inference_executor availability check.

Same pattern as sprint 153 fixed for /compute/forge:
  - Pre-fix: missing prompt / missing model_id / bad budget_ftns
    type all leaked through to a 503 ("Inference executor not
    initialized") because the 503 fired first.
  - Post-fix: validation runs before availability check so
    operators get the right 4xx for bad input regardless of
    inference_executor state.

Live dogfood reproduced:
  curl -d '{"prompt":"hi","model_id":"x","budget_ftns":"not_a_num"}' \
    /compute/inference
  → 503 (wrong; should be 422 — body field bad type)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node_no_executor():
    """Node WITHOUT inference_executor so post-validation 503 wouldn't
    mask validation errors. Sprint 154 invariant: validation must
    fire BEFORE the executor availability check."""
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
    return _client(node).post("/compute/inference", json=body)


class TestInferenceBodyValidation:
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
            "prompt": "hi", "model_id": "x", "budget_ftns": -2,
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
        """Sprint 154 invariant — well-formed body reaches the
        503 (executor unwired), proving validation passed."""
        resp = _post(_node_no_executor(), {
            "prompt": "hi", "model_id": "x", "budget_ftns": 1.0,
        })
        assert resp.status_code == 503
