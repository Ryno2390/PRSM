"""Sprint 815 — /admin/output-cache-stats endpoint + CLI consumer.

Sprint 814 added counter attributes + stats() to OutputCache.
Operators with cache enabled need an HTTP/CLI surface to read
those counters — the alternative (Prometheus scrape) is overkill
for ad-hoc "is my cache working?" debugging.

  HTTP: GET /admin/output-cache-stats
        → {hits, misses, puts, evictions, ttl_evictions,
           size, max_entries, ttl_seconds}

  CLI:  prsm node output-cache-stats
        [--api-url URL] [--format text|json]

Endpoint behavior:
- 200 with stats() snapshot when cache wired
- 503 when cache not configured (PRSM_INFERENCE_OUTPUT_CACHE_ENABLED
  unset or executor missing _output_cache)
- Loopback-gated by sprint 734 admin middleware (inherits
  F65-F73 defenses)

Pin tests:
- register_output_cache_stats_endpoint helper exists
- Endpoint registered BEFORE dashboard catch-all mount (F30)
- Returns 503 when executor has no _output_cache
- Returns 200 + stats snapshot when cache wired
- CLI command registered under `node` group
- CLI text mode renders hit-rate + counters
- CLI json mode returns parseable payload
- CLI unreachable → exit 2
- CLI 503 → exit 1 with actionable hint
"""
from __future__ import annotations

import asyncio
import inspect
import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


# ---- Endpoint registration ------------------------------------


def test_register_helper_exists():
    from prsm.node.api import (
        register_output_cache_stats_endpoint,
    )
    assert callable(register_output_cache_stats_endpoint)


def test_endpoint_registered_before_dashboard_mount():
    """F30 lesson — admin endpoints must register BEFORE the
    dashboard catch-all SPA mount."""
    from prsm.node import api as _api
    src = inspect.getsource(_api)
    reg_idx = src.find(
        "register_output_cache_stats_endpoint(app, node)"
    )
    dash_idx = src.find("_dash_app = _create_dash_app")
    assert reg_idx > 0, (
        "Sprint 815: register_output_cache_stats_endpoint must "
        "be wired in create_api_app"
    )
    assert dash_idx > 0
    assert reg_idx < dash_idx, (
        "Endpoint must register BEFORE dashboard catch-all "
        "(F30 lesson)"
    )


def _route_for(app, path):
    for r in app.router.routes:
        if getattr(r, "path", "") == path:
            return r
    raise AssertionError(
        f"route {path!r} not registered on app"
    )


def _build_app_with_executor(executor):
    from fastapi import FastAPI
    from prsm.node import api as _api
    app = FastAPI()
    node = MagicMock()
    node.inference_executor = executor
    _api.register_output_cache_stats_endpoint(app, node)
    return app


def test_endpoint_503_when_no_cache():
    """Executor has no _output_cache → endpoint returns 503
    with actionable hint."""
    from fastapi import HTTPException
    import pytest as _pytest

    executor = MagicMock()
    executor._output_cache = None
    app = _build_app_with_executor(executor)
    route = _route_for(app, "/admin/output-cache-stats")
    with _pytest.raises(HTTPException) as exc_info:
        asyncio.run(route.endpoint())
    assert exc_info.value.status_code == 503


def test_endpoint_503_when_no_executor():
    """Defensive: no executor at all → 503 not 500."""
    from fastapi import HTTPException
    import pytest as _pytest

    app = _build_app_with_executor(None)
    route = _route_for(app, "/admin/output-cache-stats")
    with _pytest.raises(HTTPException) as exc_info:
        asyncio.run(route.endpoint())
    assert exc_info.value.status_code == 503


def test_endpoint_returns_stats_when_wired():
    from prsm.compute.inference.output_cache import OutputCache
    executor = MagicMock()
    executor._output_cache = OutputCache(
        max_entries=100, ttl_seconds=600,
    )
    executor._output_cache.put("k", "v")
    executor._output_cache.get("k")  # 1 hit
    executor._output_cache.get("nokey")  # 1 miss

    app = _build_app_with_executor(executor)
    route = _route_for(app, "/admin/output-cache-stats")
    result = asyncio.run(route.endpoint())
    assert result["hits"] == 1
    assert result["misses"] == 1
    assert result["puts"] == 1
    assert result["size"] == 1
    assert result["max_entries"] == 100
    assert result["ttl_seconds"] == 600.0


# ---- CLI consumer ---------------------------------------------


def _invoke(args=None):
    from prsm.cli import node as _node_group
    return CliRunner().invoke(
        _node_group, ["output-cache-stats"] + (args or []),
    )


def test_cli_command_registered():
    from prsm.cli import node as _node_group
    cmd_names = [c.name for c in _node_group.commands.values()]
    assert "output-cache-stats" in cmd_names


def test_cli_text_renders_stats():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "hits": 42,
        "misses": 8,
        "puts": 50,
        "evictions": 0,
        "ttl_evictions": 0,
        "size": 50,
        "max_entries": 1024,
        "ttl_seconds": 3600.0,
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0, result.output
    # Operators see the load-bearing numbers
    assert "42" in result.output
    assert "8" in result.output
    # And the hit-rate signal (42 / 50 = 84%)
    assert "84" in result.output or "0.84" in result.output


def test_cli_json_returns_payload():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "hits": 1, "misses": 0, "puts": 1, "evictions": 0,
        "ttl_evictions": 0, "size": 1, "max_entries": 10,
        "ttl_seconds": 60.0,
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke(["--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["hits"] == 1
    assert data["max_entries"] == 10


def test_cli_503_actionable():
    fake = MagicMock()
    fake.status_code = 503
    fake.text = '{"detail":"output cache not configured"}'
    with patch("httpx.get", return_value=fake):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 1
    out = result.output.lower()
    assert (
        "output_cache" in out
        or "503" in out
        or "prsm_inference_output_cache_enabled" in out
    )


def test_cli_unreachable_exits_2():
    with patch(
        "httpx.get",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()
