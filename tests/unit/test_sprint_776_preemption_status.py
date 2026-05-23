"""Sprint 776 — /admin/preemption/status + `prsm node preemption-status` CLI.

Sprint 775 made the preemption arc operational. Sprint 776 closes
the operator-observability gap: operators can now confirm:
  1. Is the detector running?
  2. What backend? (aws / gcp)
  3. Has the flag been signaled?

Without sprint 776 operators would have to inspect daemon logs
or trust env vars + assume everything works. Mirrors sprint
769's /admin/auto-claim/status pattern.

Pin tests:
- register_preemption_status_endpoint exists.
- Endpoint registered BEFORE dashboard catch-all mount (F30).
- Returns 503 when no detector wired.
- Returns 200 with full payload when detector present.
- CLI command registered.
- CLI happy path (text + json).
- CLI 503 (detector unset) surfaces actionable message.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def setup_function():
    os.environ.pop("PRSM_PREEMPTION_DETECTOR", None)


def teardown_function():
    os.environ.pop("PRSM_PREEMPTION_DETECTOR", None)


# ---- Endpoint registration --------------------------------------


def test_register_preemption_status_endpoint_exists():
    from prsm.node.api import register_preemption_status_endpoint
    assert callable(register_preemption_status_endpoint)


def test_endpoint_registered_before_dashboard_mount():
    """F30 lesson — register before the dashboard catch-all."""
    from prsm.node import api as _api
    src = inspect.getsource(_api)
    reg_idx = src.find(
        "register_preemption_status_endpoint(app, node)"
    )
    dash_idx = src.find("_dash_app = _create_dash_app")
    assert reg_idx > 0
    assert dash_idx > 0
    assert reg_idx < dash_idx


def test_endpoint_returns_503_when_no_detector():
    from fastapi import FastAPI, HTTPException
    from prsm.node.api import register_preemption_status_endpoint

    app = FastAPI()
    node = MagicMock()
    node._preemption_detector = None
    register_preemption_status_endpoint(app, node)

    import pytest
    route = next(
        r for r in app.router.routes
        if getattr(r, "path", "") == "/admin/preemption/status"
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(route.endpoint())
    assert exc_info.value.status_code == 503


def test_endpoint_returns_payload_when_detector_present():
    from fastapi import FastAPI
    from prsm.node.api import register_preemption_status_endpoint
    from prsm.node.preemption import AWSPreemptionBackend

    app = FastAPI()
    node = MagicMock()
    det = MagicMock()
    det.backend = AWSPreemptionBackend()
    det.poll_interval_s = 10.0
    det.is_preempted.return_value = False
    node._preemption_detector = det

    register_preemption_status_endpoint(app, node)
    route = next(
        r for r in app.router.routes
        if getattr(r, "path", "") == "/admin/preemption/status"
    )
    result = asyncio.run(route.endpoint())
    assert result["enabled"] is True
    assert result["backend"] == "aws"
    assert result["preempted"] is False
    assert result["poll_interval_seconds"] == 10.0


def test_endpoint_reflects_preempted_true():
    from fastapi import FastAPI
    from prsm.node.api import register_preemption_status_endpoint
    from prsm.node.preemption import GCPPreemptionBackend

    app = FastAPI()
    node = MagicMock()
    det = MagicMock()
    det.backend = GCPPreemptionBackend()
    det.poll_interval_s = 5.0
    det.is_preempted.return_value = True
    node._preemption_detector = det

    register_preemption_status_endpoint(app, node)
    route = next(
        r for r in app.router.routes
        if getattr(r, "path", "") == "/admin/preemption/status"
    )
    result = asyncio.run(route.endpoint())
    assert result["backend"] == "gcp"
    assert result["preempted"] is True


# ---- CLI --------------------------------------------------------


def _invoke(args=None):
    from prsm.cli import node as _node_group
    return CliRunner().invoke(
        _node_group, ["preemption-status"] + (args or []),
    )


def test_cli_command_registered():
    from prsm.cli import node as _node_group
    cmd_names = [c.name for c in _node_group.commands.values()]
    assert "preemption-status" in cmd_names


def test_cli_happy_path_text():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "enabled": True,
        "backend": "aws",
        "preempted": False,
        "poll_interval_seconds": 10.0,
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "aws" in result.output
    # User-visible signals
    assert (
        "no" in result.output.lower()
        or "clear" in result.output.lower()
        or "false" in result.output.lower()
    )


def test_cli_happy_path_json():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "enabled": True,
        "backend": "gcp",
        "preempted": True,
        "poll_interval_seconds": 5.0,
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke(["--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["preempted"] is True
    assert data["backend"] == "gcp"


def test_cli_503_surfaces_actionable():
    fake = MagicMock()
    fake.status_code = 503
    fake.text = '{"detail":"detector not wired"}'
    with patch("httpx.get", return_value=fake):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 1
    # Operator should see what to do — set the env or check logs
    out = result.output.lower()
    assert (
        "preemption_detector" in out
        or "env" in out
        or "not configured" in out
        or "not wired" in out
    )
