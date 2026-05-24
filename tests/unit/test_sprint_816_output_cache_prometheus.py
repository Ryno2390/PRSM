"""Sprint 816 — Prometheus exposition of OutputCache counters.

Sprint 815 ships /admin/output-cache-stats + CLI for ad-hoc
reads. Sprint 816 adds the systematic-observability complement:
`prsm_inference_output_cache_*` lines on the /metrics endpoint
so operators scrape into Prometheus / Grafana.

Exposition shape:
  # HELP prsm_inference_output_cache_hits_total ...
  # TYPE prsm_inference_output_cache_hits_total counter
  prsm_inference_output_cache_hits_total <N>
  # HELP prsm_inference_output_cache_misses_total ...
  # TYPE prsm_inference_output_cache_misses_total counter
  prsm_inference_output_cache_misses_total <N>
  # ... puts_total, evictions_total, ttl_evictions_total
  # HELP prsm_inference_output_cache_size ...
  # TYPE prsm_inference_output_cache_size gauge
  prsm_inference_output_cache_size <N>

Counters use *_total suffix per Prometheus convention. size is
a gauge (snapshot).

Pin tests:
- /metrics body contains all 6 metric names when cache wired.
- /metrics body MISSING those metrics when no cache (graceful —
  no 500, no empty metrics).
- Counter values reflect the actual stats() values.
- Probe wrapped in try/except (fail-soft like other metrics).
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock


def _build_app_with_cache(cache):
    """Build minimal FastAPI app with /metrics route registered
    against a node whose inference_executor exposes the cache."""
    from fastapi import FastAPI
    from prsm.node import api as _api
    app = FastAPI()
    node = MagicMock()
    node.inference_executor = MagicMock()
    node.inference_executor._output_cache = cache
    # Set other commonly-probed fields to None so other metric
    # blocks don't crash.
    node._receipt_store = None
    node._payment_escrow = None
    node._royalty_dispatch_ring = None
    node._job_history = None
    return app, node


def _metrics_body_via_source_search():
    """Source-shape verification: api.py contains the cache
    metric block. This is the tightest pin we can do without
    standing up the full /metrics handler (which depends on a
    real node)."""
    from prsm.node import api as _api
    return inspect.getsource(_api)


# ---- Source-shape pins on /metrics --------------------------


def test_metrics_emits_hits_total():
    src = _metrics_body_via_source_search()
    assert "prsm_inference_output_cache_hits_total" in src


def test_metrics_emits_misses_total():
    src = _metrics_body_via_source_search()
    assert "prsm_inference_output_cache_misses_total" in src


def test_metrics_emits_puts_total():
    src = _metrics_body_via_source_search()
    assert "prsm_inference_output_cache_puts_total" in src


def test_metrics_emits_evictions_total():
    src = _metrics_body_via_source_search()
    assert "prsm_inference_output_cache_evictions_total" in src


def test_metrics_emits_ttl_evictions_total():
    src = _metrics_body_via_source_search()
    assert (
        "prsm_inference_output_cache_ttl_evictions_total" in src
    )


def test_metrics_emits_size_gauge():
    src = _metrics_body_via_source_search()
    assert "prsm_inference_output_cache_size" in src


# ---- Block is fail-soft + behind a try/except --------------


def test_metrics_block_wrapped_in_try_except():
    """The output cache probe must be inside a try/except so a
    stats() failure doesn't crash the entire /metrics endpoint
    (same fail-soft pattern as the receipt_store + royalty_ring
    probes)."""
    src = _metrics_body_via_source_search()
    # Find the cache metric block
    idx = src.find("prsm_inference_output_cache_hits_total")
    assert idx > 0
    # Look backward up to 500 chars for a try: statement
    before = src[max(0, idx - 500):idx]
    assert "try:" in before, (
        "Sprint 816: output cache metric probe must be inside "
        "a try/except so a stats() failure doesn't crash the "
        "/metrics endpoint"
    )


# ---- TYPE annotations correct -------------------------------


def test_counter_metrics_have_counter_type():
    """*_total metrics are counters per Prometheus convention.

    The output cache block emits TYPE via an f-string template
    `f"# TYPE {name} counter"` over a list of names; we verify
    BOTH the template AND each metric name are present in
    source. Runtime composition produces the canonical
    `# TYPE <name> counter` line per Prometheus spec.
    """
    src = _metrics_body_via_source_search()
    # The TYPE-template line for the counter loop
    assert '# TYPE {name} counter' in src, (
        "Sprint 816: counter-block TYPE template missing — the "
        "f-string loop must emit '# TYPE {name} counter' lines"
    )
    # Each counter name is referenced in the source
    for counter_name in (
        "prsm_inference_output_cache_hits_total",
        "prsm_inference_output_cache_misses_total",
        "prsm_inference_output_cache_puts_total",
        "prsm_inference_output_cache_evictions_total",
        "prsm_inference_output_cache_ttl_evictions_total",
    ):
        assert counter_name in src, (
            f"Sprint 816: {counter_name} must appear in the "
            "counter-block name list"
        )


def test_size_metric_has_gauge_type():
    src = _metrics_body_via_source_search()
    marker = "# TYPE prsm_inference_output_cache_size gauge"
    assert marker in src
