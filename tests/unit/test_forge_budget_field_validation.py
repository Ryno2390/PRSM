"""Sprint 153 — /compute/forge budget_ftns field type/value validation.

Pre-fix the endpoint had a latent bug: the PRSM_MAX_FTNS_PER_JOB
cap block parsed body["budget_ftns"] inside its own try/except
that was meant to catch ValueError on the cap env var. A
non-numeric body.budget_ftns silently disabled the cap, then
later float() at line ~1581 would also raise but only AFTER the
agent_forge availability check returned 503.

Net effect: a malformed `budget_ftns` request returned 503 (a
service-availability code) when it should have returned 422
(unprocessable entity / bad input).

Live dogfood reproduced:
  curl -d '{"query":"hi","budget_ftns":"not_a_number"}' /compute/forge
  → 503  (was: 503; expected after fix: 422)

Also: negative budgets bypassed validation (would default to
the malformed-string error at later float() call). Sprint 153
tightens this to explicit validation upfront.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node_no_forge():
    """Node WITHOUT agent_forge so post-validation 503 wouldn't
    mask validation errors. Sprint 153 invariant: validation
    must fire BEFORE the agent_forge availability check."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node.agent_forge = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _post(node, body):
    return _client(node).post("/compute/forge", json=body)


class TestBudgetFieldValidation:
    def test_non_numeric_budget_returns_422(self):
        """Sprint 153 — string-typed budget_ftns rejected with 422."""
        resp = _post(_node_no_forge(), {
            "query": "hi", "budget_ftns": "not_a_number",
        })
        assert resp.status_code == 422
        assert "budget_ftns" in resp.json()["detail"].lower()

    def test_negative_budget_returns_422(self):
        """Sprint 153 — negative budget rejected. Otherwise the
        downstream escrow lock would either accept negative-FTNS
        (impossible, ledger rejects) or behave inconsistently."""
        resp = _post(_node_no_forge(), {
            "query": "hi", "budget_ftns": -1.5,
        })
        assert resp.status_code == 422
        assert "budget_ftns" in resp.json()["detail"].lower()

    def test_zero_budget_returns_422(self):
        """Sprint 153 — zero budget is a degenerate case: no
        compute can run because cost > 0. Reject upfront."""
        resp = _post(_node_no_forge(), {
            "query": "hi", "budget_ftns": 0,
        })
        assert resp.status_code == 422
        assert "budget_ftns" in resp.json()["detail"].lower()

    def test_valid_budget_passes_validation(self):
        """Sprint 153 invariant — well-formed budget reaches the
        503 (agent_forge unwired), proving validation passed."""
        resp = _post(_node_no_forge(), {
            "query": "hi", "budget_ftns": 5.0,
        })
        # agent_forge is None in our node fixture → 503 expected
        assert resp.status_code == 503

    def test_default_budget_when_field_absent(self):
        """Sprint 153 invariant — body without budget_ftns reaches
        503 (validation passes, default applied internally)."""
        resp = _post(_node_no_forge(), {"query": "hi"})
        assert resp.status_code == 503

    def test_bad_privacy_level_returns_422(self):
        """Sprint 157 — bad privacy_level → 422. Pre-fix the
        endpoint silently fell back to the 'standard' epsilon
        for any unknown value, accepting invalid input as if
        well-formed."""
        resp = _post(_node_no_forge(), {
            "query": "hi", "budget_ftns": 1.0,
            "privacy_level": "INVALID",
        })
        assert resp.status_code == 422
        assert "privacy_level" in resp.json()["detail"].lower()
