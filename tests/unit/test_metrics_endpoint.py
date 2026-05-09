"""GET /metrics — Prometheus-format observability endpoint.

Operators running production nodes want Grafana dashboards +
alerting on PRSM-specific gauges. Standard Prometheus
text/plain exposition format makes the existing observability
stack just work.

Gauges emitted from live node state (no new tracking infra):
- prsm_pending_escrow_count
- prsm_total_locked_ftns
- prsm_job_history_size
- prsm_arbitration_pending_count
- prsm_claimable_royalties_wei
- prsm_node_health (1 healthy, 0.5 degraded, 0 unhealthy)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.payment_escrow import EscrowEntry, EscrowStatus, PaymentEscrow
from prsm.node.job_history import JobHistoryStore


def _ledger():
    led = MagicMock()
    led.get_balance = AsyncMock(return_value=100.0)
    led.transfer = AsyncMock()
    led.create_wallet = AsyncMock()
    return led


def _node(*, escrows=0, history_size=0, claimable_wei=0,
          arbitration_pending=0):
    node = MagicMock()
    node.identity.node_id = "test-node"

    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = "0x" + "11" * 20
    ftns_ledger._decimals = 18
    ftns_ledger.get_balance = AsyncMock(return_value=42.0)
    node.ftns_ledger = ftns_ledger

    escrow = PaymentEscrow(ledger=_ledger(), node_id="test-node")
    for i in range(escrows):
        e = EscrowEntry(
            escrow_id=f"e{i}", job_id=f"j{i}",
            requester_id="0x" + "11" * 20, amount=1.5,
            status=EscrowStatus.PENDING,
        )
        escrow._escrows[e.escrow_id] = e
    node._payment_escrow = escrow

    history = JobHistoryStore()
    from prsm.node.job_history import JobHistoryRecord, JobStatus
    import time
    for i in range(history_size):
        history.put(JobHistoryRecord(
            job_id=f"forge-{i}", query="q",
            status=JobStatus.COMPLETED,
            started_at=time.time(),
        ))
    node._job_history = history

    royalty = MagicMock()
    royalty.claimable = MagicMock(return_value=claimable_wei)
    node._royalty_distributor_client = royalty

    arb = MagicMock()
    arb.list_pending = AsyncMock(
        return_value=[MagicMock()] * arbitration_pending,
    )
    node._arbitration_queue = arb
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Format
# ──────────────────────────────────────────────────────────────────────


class TestMetricsFormat:
    def test_returns_prometheus_content_type(self):
        node = _node()
        resp = _client(node).get("/metrics")
        assert resp.status_code == 200
        ct = resp.headers["content-type"]
        # Prometheus exposition is text/plain, optionally with version.
        assert ct.startswith("text/plain")

    def test_emits_help_and_type_lines(self):
        node = _node()
        body = _client(node).get("/metrics").text
        # Each metric should have a # HELP and # TYPE line per format.
        assert "# HELP prsm_pending_escrow_count" in body
        assert "# TYPE prsm_pending_escrow_count gauge" in body


# ──────────────────────────────────────────────────────────────────────
# Gauges
# ──────────────────────────────────────────────────────────────────────


class TestMetricsGauges:
    def test_pending_escrow_count(self):
        node = _node(escrows=3)
        body = _client(node).get("/metrics").text
        assert "prsm_pending_escrow_count 3" in body

    def test_total_locked_ftns(self):
        node = _node(escrows=4)  # each amount=1.5 → 6.0 total
        body = _client(node).get("/metrics").text
        # Should emit "prsm_total_locked_ftns 6.0" or "6"
        assert "prsm_total_locked_ftns 6" in body

    def test_job_history_size(self):
        node = _node(history_size=7)
        body = _client(node).get("/metrics").text
        assert "prsm_job_history_size 7" in body

    def test_claimable_royalties_wei(self):
        node = _node(claimable_wei=5_000_000_000_000_000_000)
        body = _client(node).get("/metrics").text
        assert "prsm_claimable_royalties_wei 5000000000000000000" in body

    def test_arbitration_pending_count(self):
        node = _node(arbitration_pending=2)
        body = _client(node).get("/metrics").text
        assert "prsm_arbitration_pending_count 2" in body


# ──────────────────────────────────────────────────────────────────────
# Fail-soft: subsystem missing → metric absent or zero
# ──────────────────────────────────────────────────────────────────────


class TestMetricsFailSoft:
    def test_no_payment_escrow_emits_zero_or_absent(self):
        """Without PaymentEscrow wired, escrow gauges should still
        be safe — emit 0 or omit the metric, but never 500."""
        node = MagicMock()
        node.identity.node_id = "test-node"
        node.ftns_ledger = None
        node._payment_escrow = None
        node._job_history = None
        node._royalty_distributor_client = None
        node._arbitration_queue = None
        resp = _client(node).get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        # Either 0 or omitted — both acceptable. Test asserts no crash.
        assert "prsm_" in body  # at least some metric still present

    def test_royalty_rpc_error_does_not_500(self):
        node = _node()
        node._royalty_distributor_client.claimable = MagicMock(
            side_effect=RuntimeError("rpc down"),
        )
        resp = _client(node).get("/metrics")
        # Endpoint must NOT 500 — emit 0 or omit royalties gauge.
        assert resp.status_code == 200
