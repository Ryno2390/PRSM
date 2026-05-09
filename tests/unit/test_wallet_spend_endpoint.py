"""GET /wallet/spend — operator-side FTNS spend aggregation.

Closes the operator cost-tracking gap: balance_check shows
current FTNS, escrow_summary shows what's locked, but neither
surfaces total FTNS spent on completed compute jobs over a
time window. /wallet/spend aggregates RELEASED escrows by
completed_at within the last N days.

Backs the ``prsm_spend_summary`` MCP tool.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.payment_escrow import EscrowEntry, EscrowStatus


def _entry(*, job_id, amount=1.0, status=EscrowStatus.RELEASED,
           completed_at=None, requester_id="0x" + "11" * 20):
    return EscrowEntry(
        escrow_id=f"esc-{job_id}", job_id=job_id,
        requester_id=requester_id, amount=amount, status=status,
        completed_at=completed_at,
    )


def _node(escrows=None, address="0x" + "11" * 20):
    node = MagicMock()
    node.identity.node_id = "test-node"

    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = address
    ftns_ledger._decimals = 18
    node.ftns_ledger = ftns_ledger

    if escrows is None:
        node._payment_escrow = None
    else:
        escrow_svc = MagicMock()
        escrow_svc.list_escrows_by_requester = MagicMock(
            return_value=escrows,
        )
        node._payment_escrow = escrow_svc
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestSpendAvailability:
    def test_503_when_escrow_not_wired(self):
        node = _node(escrows=None)
        resp = _client(node).get("/wallet/spend")
        assert resp.status_code == 503


class TestSpendHappyPath:
    def test_default_30_days_window(self):
        """Default days=30 returns spend over the last 30 days."""
        now = time.time()
        escrows = [
            _entry(job_id="j1", amount=2.0, completed_at=now - 1 * 86400),
            _entry(job_id="j2", amount=3.0, completed_at=now - 5 * 86400),
            _entry(job_id="j3", amount=4.0, completed_at=now - 31 * 86400),
        ]
        node = _node(escrows=escrows)
        resp = _client(node).get("/wallet/spend")
        body = resp.json()
        # j1 + j2 = 5.0 (j3 outside 30d window).
        assert body["total_spent_ftns"] == 5.0
        assert body["escrows_count"] == 2
        assert body["days"] == 30

    def test_explicit_days_param(self):
        now = time.time()
        escrows = [
            _entry(job_id="j1", amount=10.0, completed_at=now - 3 * 86400),
            _entry(job_id="j2", amount=20.0, completed_at=now - 10 * 86400),
        ]
        node = _node(escrows=escrows)
        resp = _client(node).get("/wallet/spend?days=7")
        body = resp.json()
        # Only j1 within 7-day window.
        assert body["total_spent_ftns"] == 10.0
        assert body["escrows_count"] == 1
        assert body["days"] == 7

    def test_only_released_escrows_counted(self):
        """REFUNDED + PENDING escrows do NOT count as spend."""
        now = time.time()
        escrows = [
            _entry(job_id="j1", amount=5.0,
                   status=EscrowStatus.RELEASED, completed_at=now - 1 * 86400),
            _entry(job_id="j2", amount=10.0,
                   status=EscrowStatus.REFUNDED, completed_at=now - 1 * 86400),
            _entry(job_id="j3", amount=20.0,
                   status=EscrowStatus.PENDING),
        ]
        node = _node(escrows=escrows)
        resp = _client(node).get("/wallet/spend")
        body = resp.json()
        assert body["total_spent_ftns"] == 5.0
        assert body["escrows_count"] == 1

    def test_no_escrows_returns_zero(self):
        node = _node(escrows=[])
        resp = _client(node).get("/wallet/spend")
        body = resp.json()
        assert body["total_spent_ftns"] == 0.0
        assert body["escrows_count"] == 0

    def test_address_override(self):
        captured = {}

        def list_for(requester, *, pending_only=True):
            captured["requester"] = requester
            captured["pending_only"] = pending_only
            return []

        node = _node(escrows=[])
        node._payment_escrow.list_escrows_by_requester = list_for
        target = "0x" + "ab" * 20
        _client(node).get(f"/wallet/spend?address={target}")
        assert captured["requester"] == target
        # /wallet/spend needs RELEASED escrows so include_terminal-equivalent.
        assert captured["pending_only"] is False


class TestSpendValidation:
    def test_zero_days_rejected(self):
        node = _node(escrows=[])
        resp = _client(node).get("/wallet/spend?days=0")
        assert resp.status_code == 422

    def test_negative_days_rejected(self):
        node = _node(escrows=[])
        resp = _client(node).get("/wallet/spend?days=-1")
        assert resp.status_code == 422

    def test_excessive_days_clamped_to_max(self):
        """Cap at 365 days to avoid scanning unbounded history."""
        node = _node(escrows=[])
        resp = _client(node).get("/wallet/spend?days=10000")
        assert resp.status_code == 422
