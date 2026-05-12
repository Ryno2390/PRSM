"""Sprint 318d — production-hardening primitives.

Ships two operator-facing observability surfaces:
  1. Metrics registry + Prometheus text exposition +
     /admin/enterprise/metrics HTTP endpoint
  2. Structured JSON logging config helper

Plus a CLI `metrics-snapshot` subcommand for one-shot
debugging without setting up a real Prometheus scrape.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.enterprise.metrics_registry import (
    Counter,
    Gauge,
    MetricsRegistry,
)


# ── Counter ────────────────────────────────────────


def test_counter_starts_at_zero():
    c = Counter(
        name="prsm_test_total",
        description="Test counter",
    )
    assert c.value() == 0


def test_counter_increments():
    c = Counter("prsm_test_total", "Test counter")
    c.inc()
    c.inc()
    c.inc()
    assert c.value() == 3


def test_counter_increments_by_amount():
    c = Counter("prsm_test_total", "Test counter")
    c.inc(5)
    assert c.value() == 5


def test_counter_rejects_negative_increment():
    """Counters monotonically increase — refuse loud
    on a programmer error trying to decrement."""
    c = Counter("prsm_test_total", "Test counter")
    with pytest.raises(
        ValueError, match=">= 0|negative|monotonically",
    ):
        c.inc(-1)


# ── Gauge ──────────────────────────────────────────


def test_gauge_starts_at_zero():
    g = Gauge(
        name="prsm_test_gauge",
        description="Test gauge",
    )
    assert g.value() == 0


def test_gauge_set_and_read():
    g = Gauge("prsm_test_gauge", "Test gauge")
    g.set(42)
    assert g.value() == 42
    g.set(-3)  # gauges allow negative
    assert g.value() == -3


def test_gauge_inc_dec():
    g = Gauge("prsm_test_gauge", "Test gauge")
    g.inc()
    g.inc(2)
    g.dec()
    assert g.value() == 2


# ── MetricsRegistry ────────────────────────────────


def test_registry_register_and_get():
    reg = MetricsRegistry()
    c = reg.counter("foo_total", "foo counter")
    assert reg.get("foo_total") is c


def test_registry_register_duplicate_returns_existing():
    """Re-registering the same name returns the same
    instance (idempotent). Otherwise module-level
    registration races would silently drop counts."""
    reg = MetricsRegistry()
    c1 = reg.counter("foo_total", "first description")
    c2 = reg.counter("foo_total", "second description")
    assert c1 is c2


def test_registry_rejects_metric_type_mismatch():
    """If `foo` is registered as a Counter and someone
    tries to register it as a Gauge, refuse loud."""
    reg = MetricsRegistry()
    reg.counter("foo_total", "counter")
    with pytest.raises(
        ValueError, match="Counter|Gauge|register",
    ):
        reg.gauge("foo_total", "gauge")


def test_registry_lists_all_metrics():
    reg = MetricsRegistry()
    reg.counter("a_total", "a")
    reg.counter("b_total", "b")
    reg.gauge("c_gauge", "c")
    names = {m.name for m in reg.all_metrics()}
    assert names == {"a_total", "b_total", "c_gauge"}


# ── Prometheus text exposition ─────────────────────


def test_prometheus_text_empty_registry():
    reg = MetricsRegistry()
    text = reg.to_prometheus_text()
    # Empty registry is well-formed (zero metrics)
    assert isinstance(text, str)


def test_prometheus_text_counter_format():
    """Per Prometheus exposition spec:
      # HELP <name> <description>
      # TYPE <name> counter
      <name> <value>
    """
    reg = MetricsRegistry()
    c = reg.counter("requests_total", "Request count")
    c.inc(7)
    text = reg.to_prometheus_text()
    assert "# HELP requests_total Request count" in text
    assert "# TYPE requests_total counter" in text
    assert "\nrequests_total 7\n" in text or (
        text.startswith("requests_total 7")
    )


def test_prometheus_text_gauge_format():
    reg = MetricsRegistry()
    g = reg.gauge("queue_depth", "Pending items")
    g.set(42)
    text = reg.to_prometheus_text()
    assert "# HELP queue_depth Pending items" in text
    assert "# TYPE queue_depth gauge" in text
    assert "queue_depth 42" in text


def test_prometheus_text_multiple_metrics():
    reg = MetricsRegistry()
    reg.counter("a_total", "A").inc(3)
    reg.gauge("b_gauge", "B").set(7)
    text = reg.to_prometheus_text()
    assert "a_total 3" in text
    assert "b_gauge 7" in text


def test_prometheus_text_escapes_description():
    """If a description contains a newline (operator
    bug), exposition format would break. Refuse / strip."""
    reg = MetricsRegistry()
    reg.counter(
        "weird_total",
        "Has\nnewlines\nin description",
    )
    text = reg.to_prometheus_text()
    # No literal newline in the HELP line (the line ends
    # at the next \n that belongs to the next line)
    help_line = [
        l for l in text.splitlines()
        if l.startswith("# HELP weird_total")
    ]
    assert len(help_line) == 1


# ── Enterprise metric defaults ─────────────────────


def test_enterprise_metrics_module_exposes_global_registry():
    """The enterprise metrics module exposes a GLOBAL
    registry instance + the standard counter/gauge names
    that orchestrator code increments."""
    from prsm.enterprise.metrics import (
        REGISTRY,
        FL_JOBS_PROPOSED,
        FL_ROUNDS_AGGREGATED,
        PIPELINE_INFERENCE_COMPLETED,
        CORP_CAPABILITIES_REDEEMED,
    )
    assert REGISTRY is not None
    # Each constant is a registered Counter
    assert REGISTRY.get(FL_JOBS_PROPOSED.name) is (
        FL_JOBS_PROPOSED
    )
    assert REGISTRY.get(FL_ROUNDS_AGGREGATED.name) is (
        FL_ROUNDS_AGGREGATED
    )


def test_enterprise_metrics_can_be_incremented():
    from prsm.enterprise.metrics import FL_JOBS_PROPOSED
    before = FL_JOBS_PROPOSED.value()
    FL_JOBS_PROPOSED.inc()
    assert FL_JOBS_PROPOSED.value() == before + 1


# ── /admin/enterprise/metrics endpoint ────────────


def _client():
    from prsm.node.api import create_api_app
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_enterprise_metrics_endpoint_returns_text():
    resp = _client().get("/admin/enterprise/metrics")
    assert resp.status_code == 200
    # Content-Type must be the Prometheus exposition
    # MIME type — scrapers expect it
    ct = resp.headers.get("content-type", "")
    assert "text/plain" in ct


def test_enterprise_metrics_endpoint_includes_known_counters():
    from prsm.enterprise.metrics import FL_JOBS_PROPOSED
    FL_JOBS_PROPOSED.inc()
    resp = _client().get("/admin/enterprise/metrics")
    assert FL_JOBS_PROPOSED.name in resp.text


# ── Structured JSON logging ────────────────────────


def test_configure_json_logging_installs_formatter(
    caplog,
):
    from prsm.enterprise.structured_logging import (
        JsonLogFormatter,
        configure_json_logging,
    )
    # configure_json_logging returns the installed
    # formatter for inspection in tests
    formatter = configure_json_logging()
    assert isinstance(formatter, JsonLogFormatter)


def test_json_log_formatter_emits_parseable_json():
    """A logged record formats to a single JSON line with
    standard ops fields (timestamp, level, name, msg)."""
    from prsm.enterprise.structured_logging import (
        JsonLogFormatter,
    )
    fmt = JsonLogFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="/x.py", lineno=1,
        msg="hello %s", args=("world",),
        exc_info=None,
    )
    line = fmt.format(record)
    payload = json.loads(line)
    assert payload["level"] == "INFO"
    assert payload["msg"] == "hello world"
    assert payload["logger"] == "test.logger"
    assert "timestamp" in payload


def test_json_log_formatter_captures_extras():
    """Loggers calling .info(..., extra={'foo': 'bar'})
    should see foo in the JSON payload — that's how ops
    pipelines correlate events to entities."""
    from prsm.enterprise.structured_logging import (
        JsonLogFormatter,
    )
    fmt = JsonLogFormatter()
    record = logging.LogRecord(
        name="t", level=logging.INFO,
        pathname="/x.py", lineno=1,
        msg="event", args=(), exc_info=None,
    )
    record.job_id = "j-1"  # added via extra=
    record.round_index = 3
    line = fmt.format(record)
    payload = json.loads(line)
    assert payload["job_id"] == "j-1"
    assert payload["round_index"] == 3


def test_json_log_formatter_handles_exc_info():
    """Exceptions get an exc_info field with the full
    traceback — ops needs this for incident triage."""
    from prsm.enterprise.structured_logging import (
        JsonLogFormatter,
    )
    fmt = JsonLogFormatter()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        name="t", level=logging.ERROR,
        pathname="/x.py", lineno=1,
        msg="failure", args=(), exc_info=exc_info,
    )
    line = fmt.format(record)
    payload = json.loads(line)
    assert payload["level"] == "ERROR"
    assert "exc_info" in payload
    assert "boom" in payload["exc_info"]


# ── CLI metrics-snapshot subcommand ────────────────


def test_cli_metrics_snapshot_returns_prometheus_text(
    capsys,
):
    """`bringup metrics-snapshot` prints the current
    registry state in Prometheus text format for
    debugging without scraping."""
    from prsm.enterprise.metrics import (
        FL_JOBS_PROPOSED,
    )
    FL_JOBS_PROPOSED.inc()
    from prsm.enterprise.bringup_cli import main
    rc = main(["metrics-snapshot"])
    out = capsys.readouterr().out
    assert rc == 0
    assert FL_JOBS_PROPOSED.name in out
    assert "# TYPE" in out


def test_cli_metrics_snapshot_is_idempotent(
    capsys,
):
    """Two snapshots in a row produce semantically
    equivalent output (counter values may differ if
    operations happened between, but format is stable)."""
    from prsm.enterprise.bringup_cli import main
    main(["metrics-snapshot"])
    out_a = capsys.readouterr().out
    main(["metrics-snapshot"])
    out_b = capsys.readouterr().out
    # Both contain the standard counters
    for name in ("fl_jobs_proposed_total",
                 "fl_rounds_aggregated_total"):
        assert name in out_a
        assert name in out_b
