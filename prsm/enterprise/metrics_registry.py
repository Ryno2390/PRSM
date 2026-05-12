"""Sprint 318d — minimal Prometheus-style metrics registry.

No new pip dep — implements the text exposition format
directly so operators can scrape with Prometheus / OTEL /
Grafana Agent / Datadog OpenMetrics receiver without
PRSM pulling the heavy `prometheus_client` dep.

Two metric types in v1: Counter (monotonic) and Gauge
(set/inc/dec). Histograms / Summaries deferred — most
production deploys want them but they add wire-format
complexity; sprint 318e can layer them on top of this
registry without breaking changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


def _clean_description(desc: str) -> str:
    """Prometheus exposition format treats `\n` as line
    separator; an embedded newline in a HELP description
    would break a scrape. Strip them."""
    return desc.replace("\n", " ").replace("\r", " ")


# ── Counter ────────────────────────────────────────


class Counter:
    """Monotonically-increasing counter."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = _clean_description(description)
        self._value: int = 0

    def value(self) -> int:
        return self._value

    def inc(self, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError(
                f"counter increment must be >= 0; "
                f"got {amount} (counters monotonically "
                f"increase)"
            )
        self._value += amount


# ── Gauge ──────────────────────────────────────────


class Gauge:
    """Gauge — can go up or down (queue depth, in-flight
    requests, current CPU, etc.)."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = _clean_description(description)
        self._value: int = 0

    def value(self) -> int:
        return self._value

    def set(self, value: int) -> None:
        self._value = int(value)

    def inc(self, amount: int = 1) -> None:
        self._value += int(amount)

    def dec(self, amount: int = 1) -> None:
        self._value -= int(amount)


# ── Registry ───────────────────────────────────────


class MetricsRegistry:
    """Collection of metrics + Prometheus text exposition.

    Re-registering the same metric name returns the
    existing instance (idempotent registration). Type
    mismatch on re-register raises ValueError so a module
    that registers a Counter can't be silently shadowed
    by another module trying to register a Gauge with the
    same name.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, object] = {}

    def counter(
        self, name: str, description: str,
    ) -> Counter:
        existing = self._metrics.get(name)
        if existing is not None:
            if not isinstance(existing, Counter):
                raise ValueError(
                    f"metric {name!r} already registered "
                    f"as {type(existing).__name__}; "
                    f"can't register as Counter"
                )
            return existing
        c = Counter(name=name, description=description)
        self._metrics[name] = c
        return c

    def gauge(
        self, name: str, description: str,
    ) -> Gauge:
        existing = self._metrics.get(name)
        if existing is not None:
            if not isinstance(existing, Gauge):
                raise ValueError(
                    f"metric {name!r} already registered "
                    f"as {type(existing).__name__}; "
                    f"can't register as Gauge"
                )
            return existing
        g = Gauge(name=name, description=description)
        self._metrics[name] = g
        return g

    def get(self, name: str) -> Optional[object]:
        return self._metrics.get(name)

    def all_metrics(self) -> List[object]:
        return list(self._metrics.values())

    def to_prometheus_text(self) -> str:
        """Render the registry in Prometheus exposition
        format (text/plain, version 0.0.4). Operators
        scrape this from /admin/enterprise/metrics."""
        lines: List[str] = []
        # Sort by name for stable output (handy for
        # tests + diffs in dashboard configs)
        for m in sorted(
            self._metrics.values(),
            key=lambda x: x.name,
        ):
            if isinstance(m, Counter):
                metric_type = "counter"
            elif isinstance(m, Gauge):
                metric_type = "gauge"
            else:
                # Future histograms / summaries handle
                # their own emit
                continue
            lines.append(
                f"# HELP {m.name} {m.description}"
            )
            lines.append(
                f"# TYPE {m.name} {metric_type}"
            )
            lines.append(f"{m.name} {m.value()}")
        return "\n".join(lines) + "\n"
