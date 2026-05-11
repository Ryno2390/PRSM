"""Sprint 205 — dashboard Pydantic float fields gain upper bound +
``allow_inf_nan=False``.

After sprint 202's body-guard middleware closed the Infinity/NaN
wire path, dashboard Pydantic float fields still accepted any
finite value:

  - JobSubmitRequest.ftns_budget (default 1.0, only ge=0.01)
  - StakeRequest.amount          (only gt=0)
  - TransferRequest.amount       (only gt=0)

Sending amount=1e15 would silently submit a stake/transfer 11
orders of magnitude beyond any realistic supply — downstream
math could be exploited. Adding a sane absolute ceiling (1e12)
matches the api.py sprint-204 fix.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.dashboard.app import DashboardServer


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ledger_sync = MagicMock()
    node.ledger_sync.signed_transfer = AsyncMock()
    server = DashboardServer(node=node)
    return TestClient(server.app, raise_server_exceptions=False)


def test_jobs_submit_excessive_budget_rejected():
    resp = _client().post(
        "/api/jobs/submit",
        json={"job_type": "inference", "ftns_budget": 1e15},
    )
    assert resp.status_code == 422


def test_ftns_transfer_excessive_amount_rejected():
    resp = _client().post(
        "/api/ftns/transfer",
        json={"to_wallet": "w1", "amount": 1e15},
    )
    assert resp.status_code == 422


def test_jobs_submit_typical_budget_passes():
    resp = _client().post(
        "/api/jobs/submit",
        json={"job_type": "inference", "ftns_budget": 1.0},
    )
    assert resp.status_code != 422
