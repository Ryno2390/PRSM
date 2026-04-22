"""P2P observability metrics — Prometheus text exposition.

Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §4.2, §5.4, §6 Task 6.

Every PRSM node collects metrics locally; a Foundation-operated scraper
pulls the `/metrics` endpoint for the observability dashboard. Operators
running their own nodes see the same metrics locally without requiring
external telemetry upload — plan §5.4 commitment.

Metric surface:

  * p2p_connected_peers (gauge)
  * p2p_inbound_connections (gauge)
  * p2p_outbound_connections (gauge)
  * p2p_dht_query_latency_ms (histogram: p50, p95)
  * p2p_dht_successful_lookups_per_sec (gauge, rolling)
  * p2p_bootstrap_reachable_count (gauge)
  * p2p_nat_type (gauge with `type` label; value = 1 for active type)
  * p2p_nat_traversal_success_rate (gauge)
  * p2p_peer_churn_events_per_hour (gauge)
  * p2p_peer_evictions_total (counter)
  * p2p_rate_limit_violations_total (counter, labeled by category)
  * p2p_rate_limit_bans_total (counter)

Scope boundary: emits Prometheus text format. OpenTelemetry bridging is
plumbing over the same collector (a future helper in this module or
`prsm/observability/otel.py`), not required for Task 6a.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict


__all__ = [
    "P2PMetrics",
    "P2PMetricsCollector",
    "prometheus_format",
]


# -----------------------------------------------------------------------------
# Snapshot record
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class P2PMetrics:
    """Point-in-time P2P observability snapshot — matches plan §4.2."""

    connected_peers: int = 0
    inbound_connections: int = 0
    outbound_connections: int = 0
    dht_query_p50_ms: float = 0.0
    dht_query_p95_ms: float = 0.0
    dht_successful_lookups_per_sec: float = 0.0
    bootstrap_reachable_count: int = 0
    nat_type: str = "unknown"
    nat_traversal_success_rate: float = 0.0
    peer_churn_events_per_hour: float = 0.0
    peer_evictions_total: int = 0
    timestamp_unix: int = 0

    # Counter-style: category -> count.
    rate_limit_violations: Dict[str, int] = field(default_factory=dict)
    rate_limit_bans_total: int = 0


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------


class P2PMetricsCollector:
    """Accumulates per-node P2P metrics.

    Gauges overwrite; counters increment. Histograms (dht_query latency)
    hold a bounded window of samples and compute p50/p95 on demand.
    """

    _LATENCY_WINDOW = 1024  # samples kept for percentile computation

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock

        # Gauges.
        self._connected_peers = 0
        self._inbound_connections = 0
        self._outbound_connections = 0
        self._bootstrap_reachable_count = 0
        self._nat_type = "unknown"

        # Counters.
        self._peer_evictions_total = 0
        self._rate_limit_violations: Dict[str, int] = {}
        self._rate_limit_bans_total = 0

        # Success / churn rolling counters (both are rolling over an hour
        # window for plan §4.2 compatibility).
        self._nat_traversal_attempts: list[tuple[float, bool]] = []
        self._dht_lookups: list[tuple[float, bool]] = []
        self._churn_events: list[float] = []

        # Histogram samples for DHT query latency (ms).
        self._dht_latency_samples: list[float] = []

    # ---- gauges ----------------------------------------------------------

    def set_connections(
        self, *, inbound: int, outbound: int
    ) -> None:
        self._inbound_connections = inbound
        self._outbound_connections = outbound
        self._connected_peers = inbound + outbound

    def set_bootstrap_reachable(self, count: int) -> None:
        self._bootstrap_reachable_count = count

    def set_nat_type(self, nat_type: str) -> None:
        self._nat_type = nat_type

    # ---- counters --------------------------------------------------------

    def record_peer_eviction(self) -> None:
        self._peer_evictions_total += 1
        self._churn_events.append(self._clock())
        self._trim_churn()

    def record_rate_limit_violation(self, category: str) -> None:
        self._rate_limit_violations[category] = (
            self._rate_limit_violations.get(category, 0) + 1
        )

    def record_rate_limit_ban(self) -> None:
        self._rate_limit_bans_total += 1

    def record_churn_event(self) -> None:
        self._churn_events.append(self._clock())
        self._trim_churn()

    # ---- histograms + rolling ---------------------------------------------

    def record_dht_query(self, latency_ms: float, *, success: bool) -> None:
        self._dht_latency_samples.append(latency_ms)
        if len(self._dht_latency_samples) > self._LATENCY_WINDOW:
            # Drop oldest samples; histogram is intentionally approximate.
            del self._dht_latency_samples[0]
        self._dht_lookups.append((self._clock(), success))
        self._trim_lookups()

    def record_nat_traversal_attempt(self, *, success: bool) -> None:
        self._nat_traversal_attempts.append((self._clock(), success))
        self._trim_nat_attempts()

    # ---- snapshot --------------------------------------------------------

    def snapshot(self) -> P2PMetrics:
        now = self._clock()
        self._trim_lookups()
        self._trim_nat_attempts()
        self._trim_churn()

        return P2PMetrics(
            connected_peers=self._connected_peers,
            inbound_connections=self._inbound_connections,
            outbound_connections=self._outbound_connections,
            dht_query_p50_ms=self._percentile(
                self._dht_latency_samples, 0.50
            ),
            dht_query_p95_ms=self._percentile(
                self._dht_latency_samples, 0.95
            ),
            dht_successful_lookups_per_sec=self._successful_rate(
                self._dht_lookups
            ),
            bootstrap_reachable_count=self._bootstrap_reachable_count,
            nat_type=self._nat_type,
            nat_traversal_success_rate=self._success_rate(
                self._nat_traversal_attempts
            ),
            peer_churn_events_per_hour=float(len(self._churn_events)),
            peer_evictions_total=self._peer_evictions_total,
            timestamp_unix=int(now),
            rate_limit_violations=dict(self._rate_limit_violations),
            rate_limit_bans_total=self._rate_limit_bans_total,
        )

    # ---- internals -------------------------------------------------------

    @staticmethod
    def _percentile(samples: list[float], q: float) -> float:
        if not samples:
            return 0.0
        sorted_s = sorted(samples)
        # Nearest-rank percentile — simple + dependency-free; fine for
        # dashboarding at the single-node scale.
        idx = max(0, min(len(sorted_s) - 1, int(q * len(sorted_s))))
        return float(sorted_s[idx])

    @staticmethod
    def _success_rate(attempts: list[tuple[float, bool]]) -> float:
        if not attempts:
            return 0.0
        successes = sum(1 for _, ok in attempts if ok)
        return successes / len(attempts)

    @staticmethod
    def _successful_rate(attempts: list[tuple[float, bool]]) -> float:
        """Successful lookups per second, normalised over the current
        window span. Returns 0 for empty / single-sample input."""
        if len(attempts) < 2:
            return 0.0
        span = attempts[-1][0] - attempts[0][0]
        if span <= 0:
            return 0.0
        successes = sum(1 for _, ok in attempts if ok)
        return successes / span

    def _trim_lookups(self) -> None:
        # Keep a one-hour rolling window.
        cutoff = self._clock() - 3600.0
        self._dht_lookups = [
            (t, ok) for t, ok in self._dht_lookups if t >= cutoff
        ]

    def _trim_nat_attempts(self) -> None:
        cutoff = self._clock() - 3600.0
        self._nat_traversal_attempts = [
            (t, ok) for t, ok in self._nat_traversal_attempts if t >= cutoff
        ]

    def _trim_churn(self) -> None:
        cutoff = self._clock() - 3600.0
        self._churn_events = [t for t in self._churn_events if t >= cutoff]


# -----------------------------------------------------------------------------
# Prometheus text-format exposition
# -----------------------------------------------------------------------------


_NAT_TYPES = (
    "none",
    "full_cone",
    "restricted_cone",
    "port_restricted",
    "symmetric",
    "unknown",
)


def prometheus_format(collector: P2PMetricsCollector) -> str:
    """Return a Prometheus 0.0.4 text-format exposition of the collector's
    current state.

    Lines follow the TYPE / HELP / sample triplet convention. Labels on
    categorical metrics (nat_type, rate_limit category) use the standard
    label-value-selection pattern so a single time series carries the
    dimension.
    """
    m = collector.snapshot()
    lines: list[str] = []

    def emit_gauge(name: str, value: float, help_text: str) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {_fmt(value)}")

    def emit_counter(name: str, value: float, help_text: str) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {_fmt(value)}")

    emit_gauge(
        "p2p_connected_peers",
        m.connected_peers,
        "Current count of connected peers (inbound + outbound).",
    )
    emit_gauge(
        "p2p_inbound_connections",
        m.inbound_connections,
        "Current count of inbound connections.",
    )
    emit_gauge(
        "p2p_outbound_connections",
        m.outbound_connections,
        "Current count of outbound connections.",
    )
    emit_gauge(
        "p2p_dht_query_latency_p50_ms",
        m.dht_query_p50_ms,
        "p50 DHT query latency in milliseconds (rolling window).",
    )
    emit_gauge(
        "p2p_dht_query_latency_p95_ms",
        m.dht_query_p95_ms,
        "p95 DHT query latency in milliseconds (rolling window).",
    )
    emit_gauge(
        "p2p_dht_successful_lookups_per_sec",
        m.dht_successful_lookups_per_sec,
        "Successful DHT lookups per second (one-hour rolling window).",
    )
    emit_gauge(
        "p2p_bootstrap_reachable_count",
        m.bootstrap_reachable_count,
        "Count of bootstrap peers currently reachable.",
    )
    emit_gauge(
        "p2p_nat_traversal_success_rate",
        m.nat_traversal_success_rate,
        "Fraction of NAT traversal attempts succeeding (rolling).",
    )
    emit_gauge(
        "p2p_peer_churn_events_per_hour",
        m.peer_churn_events_per_hour,
        "Peer registration / de-registration events in the last hour.",
    )

    # nat_type: one-hot via labels, so dashboards can alert on
    # `nat_type="symmetric"` transitions.
    lines.append("# HELP p2p_nat_type Active local NAT type (one-hot).")
    lines.append("# TYPE p2p_nat_type gauge")
    for t in _NAT_TYPES:
        value = 1 if t == m.nat_type else 0
        lines.append(f'p2p_nat_type{{type="{t}"}} {value}')

    emit_counter(
        "p2p_peer_evictions_total",
        m.peer_evictions_total,
        "Cumulative peers evicted by the liveness monitor.",
    )
    emit_counter(
        "p2p_rate_limit_bans_total",
        m.rate_limit_bans_total,
        "Cumulative peers banned by the rate limiter.",
    )

    # Rate-limit violations by category.
    lines.append(
        "# HELP p2p_rate_limit_violations_total Rate-limit violations by category."
    )
    lines.append("# TYPE p2p_rate_limit_violations_total counter")
    for category, count in sorted(m.rate_limit_violations.items()):
        lines.append(
            f'p2p_rate_limit_violations_total{{category="{category}"}} {count}'
        )

    return "\n".join(lines) + "\n"


def _fmt(value: float) -> str:
    """Prometheus wants integers without decimals, floats in standard
    notation. Keep output stable for tests."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"
