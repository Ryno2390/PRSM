"""Sprint 769 — /admin/auto-claim/status endpoint + --runtime CLI flag.

Sprint 767 shipped the auto-claim CLI as read-only config
inspection. Sprint 768's runbook flagged that runtime claim
counters (total claimed, attempts, failures) lived on the
worker but weren't exposed. Sprint 769 closes that carveout:

- New /admin/auto-claim/status endpoint (loopback-gated per
  sprint 734 admin middleware) returns config + worker counters.
- CLI gains `--runtime` flag that fetches + renders them.

Pin tests:
- Endpoint exists + registered before dashboard mount (F30 lesson)
- Returns 503 when worker is None (defensive)
- Returns 200 with full payload when worker present
- CLI --runtime flag wired
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
from decimal import Decimal
from unittest.mock import MagicMock

from click.testing import CliRunner


def setup_function():
    os.environ.pop("PRSM_AUTO_CLAIM_THRESHOLD_FTNS", None)
    os.environ.pop("PRSM_AUTO_CLAIM_INTERVAL_S", None)


def teardown_function():
    os.environ.pop("PRSM_AUTO_CLAIM_THRESHOLD_FTNS", None)
    os.environ.pop("PRSM_AUTO_CLAIM_INTERVAL_S", None)


# ---- Endpoint registration --------------------------------------


def test_register_auto_claim_status_endpoint_exists():
    """Function exported from api.py."""
    from prsm.node.api import register_auto_claim_status_endpoint
    assert callable(register_auto_claim_status_endpoint)


def test_endpoint_registered_before_dashboard_mount():
    """F30 lesson: dashboard mount is a catch-all. Endpoints
    registered AFTER silently 404. Pin source-shape ordering."""
    from prsm.node import api as _api
    src = inspect.getsource(_api)
    reg_idx = src.find(
        "register_auto_claim_status_endpoint(app, node)"
    )
    dash_idx = src.find("_dash_app = _create_dash_app")
    assert reg_idx > 0
    assert dash_idx > 0
    assert reg_idx < dash_idx, (
        "auto-claim status endpoint must be registered BEFORE "
        "the dashboard catch-all mount (F30 lesson)"
    )


def test_endpoint_returns_503_when_no_worker():
    """Defensive: daemon without staking_manager/identity (so
    no worker constructed) returns 503 with actionable error."""
    from fastapi import FastAPI, HTTPException
    from prsm.node.api import register_auto_claim_status_endpoint

    app = FastAPI()
    node = MagicMock()
    node._auto_claim_worker = None
    register_auto_claim_status_endpoint(app, node)

    import pytest
    route = next(
        r for r in app.router.routes
        if getattr(r, "path", "") == "/admin/auto-claim/status"
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(route.endpoint())
    assert exc_info.value.status_code == 503


def test_endpoint_returns_payload_when_worker_present():
    """Worker exists → 200 with config + counters."""
    from fastapi import FastAPI
    from prsm.node.api import register_auto_claim_status_endpoint
    from prsm.node.auto_claim import AutoClaimConfig

    app = FastAPI()
    node = MagicMock()
    worker = MagicMock()
    worker.config = AutoClaimConfig(Decimal("100"), 3600.0)
    worker.total_claimed_ftns = Decimal("250")
    worker.claim_attempts = 3
    worker.claim_failures = 0
    node._auto_claim_worker = worker

    register_auto_claim_status_endpoint(app, node)
    route = next(
        r for r in app.router.routes
        if getattr(r, "path", "") == "/admin/auto-claim/status"
    )
    result = asyncio.run(route.endpoint())
    assert result["enabled"] is True
    assert result["threshold_ftns"] == "100"
    assert result["interval_seconds"] == 3600.0
    assert result["total_claimed_ftns"] == "250"
    assert result["claim_attempts"] == 3
    assert result["claim_failures"] == 0


# ---- CLI integration --------------------------------------------


def test_cli_runtime_flag_exists():
    """--runtime flag accepted (doesn't error parse)."""
    from prsm.cli import node as _node_group
    runner = CliRunner()
    result = runner.invoke(
        _node_group, ["auto-claim", "--help"],
    )
    assert result.exit_code == 0
    assert "--runtime" in result.output


def test_cli_without_runtime_does_not_fetch():
    """Default CLI (no --runtime) does NOT make an HTTP call —
    matches sprint-767 behavior. Useful when daemon is offline."""
    from prsm.cli import node as _node_group
    runner = CliRunner()
    result = runner.invoke(
        _node_group, ["auto-claim", "--format", "json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    # No "runtime" field when --runtime not passed
    assert "runtime" not in data


def test_cli_runtime_flag_attempts_fetch():
    """With --runtime, CLI attempts an HTTP fetch. Mock the
    httpx call to verify the request was made."""
    from prsm.cli import node as _node_group
    from unittest.mock import patch
    runner = CliRunner()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "enabled": False,
        "threshold_ftns": "0",
        "interval_seconds": 3600.0,
        "total_claimed_ftns": "0",
        "claim_attempts": 0,
        "claim_failures": 0,
    }
    with patch("httpx.get", return_value=fake_resp) as mock_get:
        result = runner.invoke(
            _node_group,
            ["auto-claim", "--runtime", "--format", "json"],
        )
    assert result.exit_code == 0, result.output
    mock_get.assert_called_once()
    # The fetched runtime is surfaced in the JSON output
    data = json.loads(result.output)
    assert "runtime" in data
