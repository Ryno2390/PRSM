"""Unit tests for prsm.node.p2p_metrics.

Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §6 Task 6a.
"""

from __future__ import annotations

import pytest

from prsm.node.p2p_metrics import (
    P2PMetrics,
    P2PMetricsCollector,
    prometheus_format,
)


# -----------------------------------------------------------------------------
# Gauges
# -----------------------------------------------------------------------------


def test_connected_peers_is_sum_of_directions():
    c = P2PMetricsCollector()
    c.set_connections(inbound=4, outbound=7)
    snap = c.snapshot()
    assert snap.inbound_connections == 4
    assert snap.outbound_connections == 7
    assert snap.connected_peers == 11


def test_set_connections_overwrites_not_accumulates():
    c = P2PMetricsCollector()
    c.set_connections(inbound=5, outbound=5)
    c.set_connections(inbound=2, outbound=3)
    snap = c.snapshot()
    assert snap.connected_peers == 5


def test_bootstrap_reachable_gauge():
    c = P2PMetricsCollector()
    c.set_bootstrap_reachable(3)
    assert c.snapshot().bootstrap_reachable_count == 3


def test_nat_type_is_reported():
    c = P2PMetricsCollector()
    c.set_nat_type("symmetric")
    assert c.snapshot().nat_type == "symmetric"


# -----------------------------------------------------------------------------
# Counters
# -----------------------------------------------------------------------------


def test_eviction_counter_increments():
    c = P2PMetricsCollector()
    c.record_peer_eviction()
    c.record_peer_eviction()
    c.record_peer_eviction()
    assert c.snapshot().peer_evictions_total == 3


def test_rate_limit_violations_are_category_separated():
    c = P2PMetricsCollector()
    c.record_rate_limit_violation("dht")
    c.record_rate_limit_violation("dht")
    c.record_rate_limit_violation("direct_message")
    snap = c.snapshot()
    assert snap.rate_limit_violations == {"dht": 2, "direct_message": 1}


def test_rate_limit_bans_counter():
    c = P2PMetricsCollector()
    c.record_rate_limit_ban()
    c.record_rate_limit_ban()
    assert c.snapshot().rate_limit_bans_total == 2


# -----------------------------------------------------------------------------
# Histograms + rolling rates
# -----------------------------------------------------------------------------


def test_dht_latency_percentiles_are_computed():
    c = P2PMetricsCollector()
    for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        c.record_dht_query(float(ms), success=True)
    snap = c.snapshot()
    # p50 nearest-rank on 10 evenly-spaced samples lands around the 5th
    # element (50ms); p95 lands at the last one (100ms).
    assert snap.dht_query_p50_ms in (50.0, 60.0)
    assert snap.dht_query_p95_ms == 100.0


def test_empty_histogram_yields_zero_percentiles():
    c = P2PMetricsCollector()
    snap = c.snapshot()
    assert snap.dht_query_p50_ms == 0.0
    assert snap.dht_query_p95_ms == 0.0


def test_nat_traversal_success_rate():
    c = P2PMetricsCollector()
    for _ in range(4):
        c.record_nat_traversal_attempt(success=True)
    for _ in range(1):
        c.record_nat_traversal_attempt(success=False)
    assert c.snapshot().nat_traversal_success_rate == pytest.approx(0.8)


def test_dht_lookups_per_sec_uses_span():
    clock = [1000.0]
    c = P2PMetricsCollector(clock=lambda: clock[0])
    c.record_dht_query(10.0, success=True)
    clock[0] += 10
    c.record_dht_query(10.0, success=True)
    # 2 successes across a 10-second span = 0.2/sec.
    assert c.snapshot().dht_successful_lookups_per_sec == pytest.approx(0.2)


def test_rolling_window_expires_old_entries():
    clock = [1000.0]
    c = P2PMetricsCollector(clock=lambda: clock[0])
    c.record_dht_query(10.0, success=True)
    clock[0] += 7200  # 2 hours later
    c.record_dht_query(10.0, success=True)
    snap = c.snapshot()
    # Only the recent entry remains; rate is 0/0 → 0.
    assert snap.dht_successful_lookups_per_sec == 0.0


def test_churn_events_rolling_hourly():
    clock = [2000.0]
    c = P2PMetricsCollector(clock=lambda: clock[0])
    for _ in range(3):
        c.record_churn_event()
    clock[0] += 3700  # past 1-hour window
    c.record_churn_event()
    snap = c.snapshot()
    # Only the fresh event remains.
    assert snap.peer_churn_events_per_hour == 1.0


# -----------------------------------------------------------------------------
# Prometheus exposition
# -----------------------------------------------------------------------------


def test_prometheus_format_includes_all_standard_series():
    c = P2PMetricsCollector()
    c.set_connections(inbound=2, outbound=3)
    c.set_nat_type("full_cone")
    c.record_peer_eviction()
    c.record_rate_limit_violation("dht")
    c.record_rate_limit_ban()

    out = prometheus_format(c)

    assert "p2p_connected_peers 5" in out
    assert "p2p_inbound_connections 2" in out
    assert "p2p_outbound_connections 3" in out
    assert "p2p_peer_evictions_total 1" in out
    assert "p2p_rate_limit_bans_total 1" in out
    assert 'p2p_rate_limit_violations_total{category="dht"} 1' in out


def test_prometheus_nat_type_one_hot():
    c = P2PMetricsCollector()
    c.set_nat_type("symmetric")
    out = prometheus_format(c)
    assert 'p2p_nat_type{type="symmetric"} 1' in out
    # All other types should be zero.
    for t in ("full_cone", "restricted_cone", "port_restricted", "unknown", "none"):
        assert f'p2p_nat_type{{type="{t}"}} 0' in out


def test_prometheus_includes_help_and_type_lines():
    c = P2PMetricsCollector()
    out = prometheus_format(c)
    # Every metric must have HELP + TYPE lines.
    for metric in (
        "p2p_connected_peers",
        "p2p_inbound_connections",
        "p2p_outbound_connections",
        "p2p_dht_query_latency_p50_ms",
        "p2p_dht_query_latency_p95_ms",
        "p2p_peer_evictions_total",
        "p2p_rate_limit_bans_total",
    ):
        assert f"# HELP {metric}" in out
        assert f"# TYPE {metric}" in out


def test_prometheus_format_ends_with_newline():
    """Prometheus exposition convention: trailing newline."""
    c = P2PMetricsCollector()
    out = prometheus_format(c)
    assert out.endswith("\n")


def test_snapshot_timestamp_reflects_clock():
    clock = [12345.678]
    c = P2PMetricsCollector(clock=lambda: clock[0])
    snap = c.snapshot()
    assert snap.timestamp_unix == 12345
