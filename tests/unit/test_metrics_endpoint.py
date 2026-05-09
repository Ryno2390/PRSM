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
    # Default no cleanup task wired — tests that need one set
    # _escrow_cleanup_task explicitly.
    node._escrow_cleanup_task = None
    # Default no daemon tasks wired — same rationale as cleanup_task.
    node._heartbeat_scheduler_task = None
    node._compensation_scheduler_task = None
    node._key_distribution_watcher_task = None
    node._storage_slashing_watcher_task = None
    node._compensation_distributor_watcher_task = None
    node._job_reaper_task = None
    node._daemon_watchdog_task = None
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


class TestCleanupTaskGauge:
    """prsm_escrow_cleanup_task_running gauge — companion to the
    cleanup_task_running field on /health/detailed. Lets operators
    alert via Prometheus when the periodic_cleanup task crashes
    silently."""

    def test_gauge_emitted_when_task_running(self):
        node = _node()
        fake_task = MagicMock()
        fake_task.done.return_value = False
        node._escrow_cleanup_task = fake_task
        body = _client(node).get("/metrics").text
        assert "prsm_escrow_cleanup_task_running 1" in body

    def test_gauge_emitted_zero_when_task_done(self):
        """Done == crashed (the task is an infinite loop). Gauge
        flips to 0 to alarm."""
        node = _node()
        fake_task = MagicMock()
        fake_task.done.return_value = True
        node._escrow_cleanup_task = fake_task
        body = _client(node).get("/metrics").text
        assert "prsm_escrow_cleanup_task_running 0" in body

    def test_gauge_omitted_when_task_not_wired(self):
        """No task attached → omit the gauge rather than emit
        a misleading 0/1 value. Differentiates "we don't know"
        from "definitely crashed"."""
        node = _node()  # no _escrow_cleanup_task attr
        body = _client(node).get("/metrics").text
        assert "prsm_escrow_cleanup_task_running" not in body


class TestDaemonGauges:
    """Same task_running gauge pattern applied to the remaining 4
    daemons so operators get full Prometheus alerting coverage."""

    def _setup(self, node, attr_pair, running=True):
        scheduler_attr, task_attr = attr_pair
        setattr(node, scheduler_attr, MagicMock())
        fake_task = MagicMock()
        fake_task.done.return_value = not running
        setattr(node, task_attr, fake_task)

    def test_heartbeat_scheduler_gauge(self):
        node = _node()
        self._setup(node, ("_heartbeat_scheduler", "_heartbeat_scheduler_task"))
        body = _client(node).get("/metrics").text
        assert "prsm_heartbeat_scheduler_running 1" in body

    def test_compensation_scheduler_gauge_crash(self):
        node = _node()
        self._setup(
            node,
            ("_compensation_scheduler", "_compensation_scheduler_task"),
            running=False,
        )
        body = _client(node).get("/metrics").text
        assert "prsm_compensation_scheduler_running 0" in body

    def test_key_distribution_watcher_gauge(self):
        node = _node()
        self._setup(
            node,
            ("_key_distribution_watcher", "_key_distribution_watcher_task"),
        )
        body = _client(node).get("/metrics").text
        assert "prsm_key_distribution_watcher_running 1" in body

    def test_storage_slashing_watcher_gauge(self):
        node = _node()
        self._setup(
            node,
            ("_storage_slashing_watcher", "_storage_slashing_watcher_task"),
        )
        body = _client(node).get("/metrics").text
        assert "prsm_storage_slashing_watcher_running 1" in body

    def test_compensation_distributor_watcher_gauge(self):
        node = _node()
        self._setup(
            node,
            ("_compensation_distributor_watcher",
             "_compensation_distributor_watcher_task"),
        )
        body = _client(node).get("/metrics").text
        assert (
            "prsm_compensation_distributor_watcher_running 1"
            in body
        )

    def test_job_reaper_gauge(self):
        node = _node()
        self._setup(node, ("_job_reaper", "_job_reaper_task"))
        body = _client(node).get("/metrics").text
        assert "prsm_job_reaper_running 1" in body

    def test_daemon_watchdog_gauge(self):
        node = _node()
        self._setup(node, ("_daemon_watchdog", "_daemon_watchdog_task"))
        body = _client(node).get("/metrics").text
        assert "prsm_daemon_watchdog_running 1" in body


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
