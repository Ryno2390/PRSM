"""Sprint 200 — NaN/Infinity guards on remaining float-amount
endpoints.

After sprint 199 closed /ledger/transfer, the same gap exists on:
  - POST /agents/{agent_id}/allowance  (amount: float)
  - POST /settler/register             (bond_amount: float)
  - POST /settler/slash/propose        (slash_amount: float)

All use plain `amount <= 0` checks, which let NaN and Infinity
through (both comparisons evaluate False). Pydantic-protected
endpoints (bridge/staking with `Field(gt=0)`) reject NaN but
ACCEPT Infinity — confirmed via direct Pydantic v2 test. That
broader gap is out of scope for this sprint.

Post-fix: all three endpoints return 422 for NaN/Infinity.
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
    node.ledger = MagicMock()
    node.ledger.grant_agent_allowance = AsyncMock()
    node._settler_registry = MagicMock()
    node._settler_registry.register_settler = AsyncMock()
    node._settler_registry.propose_slash = AsyncMock()
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


class TestAgentAllowanceFinite:
    def test_nan_amount_rejected(self):
        resp = _client().post(
            "/agents/agent-1/allowance",
            params={"amount": "NaN"},
        )
        assert resp.status_code == 422

    def test_inf_amount_rejected(self):
        resp = _client().post(
            "/agents/agent-1/allowance",
            params={"amount": "Infinity"},
        )
        assert resp.status_code == 422

    def test_typical_passes_validation(self):
        resp = _client().post(
            "/agents/agent-1/allowance",
            params={"amount": 10.0},
        )
        assert resp.status_code != 422


class TestSettlerRegisterFinite:
    def test_nan_bond_rejected(self):
        resp = _client().post(
            "/settler/register",
            params={
                "settler_id": "s1", "address": "0xabc",
                "bond_amount": "NaN",
            },
        )
        assert resp.status_code == 422

    def test_inf_bond_rejected(self):
        resp = _client().post(
            "/settler/register",
            params={
                "settler_id": "s1", "address": "0xabc",
                "bond_amount": "Infinity",
            },
        )
        assert resp.status_code == 422


class TestSettlerSlashFinite:
    def test_nan_slash_rejected(self):
        resp = _client().post(
            "/settler/slash/propose",
            params={
                "settler_id": "s1", "slash_amount": "NaN",
                "reason": "bad", "proposer_id": "p1",
            },
        )
        assert resp.status_code == 422

    def test_inf_slash_rejected(self):
        resp = _client().post(
            "/settler/slash/propose",
            params={
                "settler_id": "s1", "slash_amount": "Infinity",
                "reason": "bad", "proposer_id": "p1",
            },
        )
        assert resp.status_code == 422
