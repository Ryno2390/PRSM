"""Sprint 202 — dashboard sub-app also runs the Inf/NaN body guard.

When the dashboard is mounted under api.py (production path), the
sprint-201 middleware catches Infinity/NaN at the parent app
layer. When the dashboard is constructed standalone (tests, some
deployments), no such guard runs, so /api/transfer + /api/stake
silently accept Infinity through Pydantic `gt=0` and crash
downstream.

Post-fix: dashboard registers its own inf/nan body guard so it's
self-defended regardless of mount context.
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
    node.ledger_sync.signed_transfer = AsyncMock(return_value=None)
    server = DashboardServer(node=node)
    return TestClient(server.app, raise_server_exceptions=False)


def test_transfer_inf_rejected():
    resp = _client().post(
        "/api/ftns/transfer",
        content=(
            '{"to_wallet": "wallet_other", "amount": Infinity}'
        ),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422, resp.text


def test_jobs_submit_nan_rejected():
    resp = _client().post(
        "/api/jobs/submit",
        content='{"job_type": "inference", "ftns_budget": NaN}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422, resp.text


def test_typical_transfer_passes_guard():
    resp = _client().post(
        "/api/ftns/transfer",
        json={"to_wallet": "wallet_other", "amount": 1.0},
    )
    # Validation passes (or fails for other reasons like auth);
    # the guard itself does not trigger.
    assert resp.status_code != 422 or "NaN" not in resp.text
