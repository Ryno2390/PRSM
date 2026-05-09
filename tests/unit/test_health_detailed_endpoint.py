"""GET /health/detailed — structured subsystem readiness probe.

Closes the ops-monitoring gap that the legacy GET /health left
open: load balancers want a fast 200 from /health, but operators
running production nodes want a deeper check that surfaces which
subsystems are healthy / degraded / unhealthy.

Top-level status:
  - healthy: all wired subsystems operational
  - degraded: optional subsystems unavailable but core (FTNS
    ledger + payment escrow) works
  - unhealthy: core subsystem missing or erroring

Per-subsystem fields: {available, status, error?}.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node_minimal():
    """Bare node — no FTNS, no escrow, no anything."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._royalty_distributor_client = None
    return node


def _node_full(*, ftns_balance=42.0, escrow_count=0):
    """Fully-wired node with FTNS ledger + escrow + history +
    royalty client."""
    node = MagicMock()
    node.identity.node_id = "test-node"

    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = "0x" + "11" * 20
    ftns_ledger._decimals = 18
    ftns_ledger.get_balance = AsyncMock(return_value=ftns_balance)
    node.ftns_ledger = ftns_ledger

    from prsm.node.payment_escrow import PaymentEscrow, EscrowEntry, EscrowStatus
    led = MagicMock()
    led.get_balance = AsyncMock(return_value=100.0)
    led.transfer = AsyncMock()
    led.create_wallet = AsyncMock()
    escrow = PaymentEscrow(ledger=led, node_id="test-node")
    for i in range(escrow_count):
        entry = EscrowEntry(
            escrow_id=f"e{i}", job_id=f"j{i}",
            requester_id="0x" + "11" * 20, amount=1.0,
            status=EscrowStatus.PENDING,
        )
        escrow._escrows[entry.escrow_id] = entry
    node._payment_escrow = escrow

    from prsm.node.job_history import JobHistoryStore
    node._job_history = JobHistoryStore()

    royalty = MagicMock()
    royalty.claimable = MagicMock(return_value=0)
    node._royalty_distributor_client = royalty
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Top-level status
# ──────────────────────────────────────────────────────────────────────


class TestHealthDetailedStatus:
    def test_unhealthy_when_no_subsystems(self):
        """No FTNS ledger + no escrow → unhealthy. The node is
        nominally up but can't actually serve any value-bearing
        requests."""
        node = _node_minimal()
        resp = _client(node).get("/health/detailed")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert body["node_id"] == "test-node"

    def test_healthy_when_all_subsystems_wired(self):
        node = _node_full()
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        assert body["status"] == "healthy"

    def test_degraded_when_optional_missing(self):
        """FTNS + escrow wired (core); royalty client missing
        (optional) → degraded, not unhealthy."""
        node = _node_full()
        node._royalty_distributor_client = None
        node._job_history = None
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        assert body["status"] == "degraded"


# ──────────────────────────────────────────────────────────────────────
# Per-subsystem reporting
# ──────────────────────────────────────────────────────────────────────


class TestHealthDetailedSubsystems:
    def test_subsystems_object_present(self):
        node = _node_full()
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        assert "subsystems" in body
        ss = body["subsystems"]
        assert "ftns_ledger" in ss
        assert "payment_escrow" in ss
        assert "job_history" in ss
        assert "royalty_distributor" in ss

    def test_unavailable_subsystem_marked(self):
        node = _node_full()
        node._royalty_distributor_client = None
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        assert body["subsystems"]["royalty_distributor"]["available"] is False

    def test_available_subsystem_marked(self):
        node = _node_full()
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        assert body["subsystems"]["ftns_ledger"]["available"] is True
        assert body["subsystems"]["payment_escrow"]["available"] is True

    def test_payment_escrow_reports_pending_count(self):
        node = _node_full(escrow_count=3)
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        ss = body["subsystems"]["payment_escrow"]
        assert ss["pending_count"] == 3


# ──────────────────────────────────────────────────────────────────────
# Subsystem fail-soft
# ──────────────────────────────────────────────────────────────────────


class TestHealthDetailedFailSoft:
    def test_subsystem_check_raising_does_not_500(self):
        """If a subsystem health probe raises (e.g., RPC down),
        the endpoint must NOT 500 — surface the error in the
        subsystem entry + flag overall as degraded/unhealthy."""
        node = _node_full()
        # Force claimable to raise.
        node._royalty_distributor_client.claimable = MagicMock(
            side_effect=RuntimeError("rpc down"),
        )
        resp = _client(node).get("/health/detailed")
        assert resp.status_code == 200
        body = resp.json()
        royalty = body["subsystems"]["royalty_distributor"]
        # Surface the error for ops debugging.
        assert "error" in royalty


# ──────────────────────────────────────────────────────────────────────
# Backwards-compat: /health unchanged
# ──────────────────────────────────────────────────────────────────────


class TestHealthLegacyPreserved:
    def test_simple_health_still_returns_minimal_response(self):
        """/health is the load-balancer probe. Must stay minimal +
        fast — no subsystem checks added."""
        node = _node_full()
        resp = _client(node).get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "subsystems" not in body
