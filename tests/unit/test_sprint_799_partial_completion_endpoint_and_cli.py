"""Sprint 799 — /admin/partial-completion-events endpoint + CLI.

Closes the sprint 798 carveout. Sprint 798 shipped the ring +
the settle-path integration; sprint 799 exposes:

  HTTP: GET /admin/partial-completion-events
        ?limit=50&offset=0&operator_node_id=<hex>
        → {entries, total, offset, limit}

  CLI:  prsm node partial-completion-history
        [--limit N] [--offset N] [--operator-node-id <hex>]
        [--format text|json] [--api-url <url>]

  Daemon-startup wire-up: PRSMNode.start constructs the ring +
  assigns to self._partial_completion_event_log so the sprint-
  784/785 append sites + this endpoint both see the same
  instance.

Pin tests:
- Endpoint registered BEFORE dashboard catch-all mount (F30).
- 503 when ring unwired.
- 200 + empty list when ring has no entries.
- 200 + entries when ring populated.
- Limit / offset pagination honored.
- operator_node_id filter honored.
- 422 on limit out of [1, 1000].
- 422 on offset < 0.
- CLI command registered under node group.
- CLI happy text path renders entries.
- CLI json path returns parseable payload.
- CLI --operator-node-id flag wired.
- CLI unreachable daemon → exit 2.
- PRSMNode.start source-shape: constructs the ring.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


# ---- Endpoint registration ------------------------------------


def test_endpoint_registered_before_dashboard_mount():
    """Source-shape: the partial-completion endpoint registrar
    is called BEFORE the dashboard catch-all mount (F30 lesson)."""
    from prsm.node import api as _api
    src = inspect.getsource(_api)
    reg_idx = src.find(
        "register_partial_completion_events_endpoint(app, node)"
    )
    dash_idx = src.find("_dash_app = _create_dash_app")
    assert reg_idx > 0, (
        "Sprint 799: register_partial_completion_events_endpoint "
        "must be wired in create_api_app"
    )
    assert dash_idx > 0
    assert reg_idx < dash_idx, (
        "Endpoint must register BEFORE dashboard catch-all "
        "(F30 lesson)"
    )


def _route_for(app, path):
    """Find the FastAPI route handler for `path`."""
    for r in app.router.routes:
        if getattr(r, "path", "") == path:
            return r
    raise AssertionError(
        f"route {path!r} not registered on app"
    )


def _build_app_with_ring(ring):
    from fastapi import FastAPI
    from prsm.node import api as _api
    app = FastAPI()
    node = MagicMock()
    node._partial_completion_event_log = ring
    _api.register_partial_completion_events_endpoint(app, node)
    return app


def test_endpoint_503_when_ring_unwired():
    """Call route directly (bypasses TestClient→httpx mock)."""
    from fastapi import HTTPException
    import pytest as _pytest
    app = _build_app_with_ring(None)
    route = _route_for(app, "/admin/partial-completion-events")
    with _pytest.raises(HTTPException) as exc_info:
        asyncio.run(route.endpoint())
    assert exc_info.value.status_code == 503


def test_endpoint_200_empty_when_ring_empty():
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    app = _build_app_with_ring(PartialCompletionEventRing())
    route = _route_for(app, "/admin/partial-completion-events")
    data = asyncio.run(route.endpoint())
    assert data["entries"] == []
    assert data["total"] == 0


def test_endpoint_returns_entries():
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    ring = PartialCompletionEventRing()
    ring.append(
        job_id="j1", operator_node_id="a" * 32,
        reason="error", tokens_completed=4, tokens_requested=10,
        timestamp=1000.0,
    )
    ring.append(
        job_id="j2", operator_node_id="a" * 32,
        reason="error", tokens_completed=2, tokens_requested=10,
        timestamp=2000.0,
    )
    app = _build_app_with_ring(ring)
    route = _route_for(app, "/admin/partial-completion-events")
    data = asyncio.run(route.endpoint())
    assert data["total"] == 2
    assert data["entries"][0]["job_id"] == "j2"
    assert data["entries"][1]["job_id"] == "j1"


def test_endpoint_limit_offset():
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    ring = PartialCompletionEventRing()
    for i in range(5):
        ring.append(
            job_id=f"j{i}", operator_node_id="a" * 32,
            reason="error", tokens_completed=i, tokens_requested=10,
            timestamp=1000.0 + i,
        )
    app = _build_app_with_ring(ring)
    route = _route_for(app, "/admin/partial-completion-events")
    data = asyncio.run(route.endpoint(limit=2, offset=1))
    assert [e["job_id"] for e in data["entries"]] == ["j3", "j2"]


def test_endpoint_operator_node_id_filter():
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    ring = PartialCompletionEventRing()
    ring.append(
        job_id="j-a", operator_node_id="a" * 32,
        reason="error", tokens_completed=1, tokens_requested=10,
    )
    ring.append(
        job_id="j-b", operator_node_id="b" * 32,
        reason="error", tokens_completed=2, tokens_requested=10,
    )
    app = _build_app_with_ring(ring)
    route = _route_for(app, "/admin/partial-completion-events")
    data = asyncio.run(route.endpoint(
        operator_node_id="a" * 32,
    ))
    assert len(data["entries"]) == 1
    assert data["entries"][0]["job_id"] == "j-a"


def test_endpoint_422_on_bad_limit():
    from fastapi import HTTPException
    import pytest as _pytest
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    app = _build_app_with_ring(PartialCompletionEventRing())
    route = _route_for(app, "/admin/partial-completion-events")
    with _pytest.raises(HTTPException) as exc_info:
        asyncio.run(route.endpoint(limit=5000))
    assert exc_info.value.status_code == 422


def test_endpoint_422_on_negative_offset():
    from fastapi import HTTPException
    import pytest as _pytest
    from prsm.node.partial_completion_event_log import (
        PartialCompletionEventRing,
    )
    app = _build_app_with_ring(PartialCompletionEventRing())
    route = _route_for(app, "/admin/partial-completion-events")
    with _pytest.raises(HTTPException) as exc_info:
        asyncio.run(route.endpoint(offset=-1))
    assert exc_info.value.status_code == 422


# ---- CLI consumer --------------------------------------------


def _invoke_cli(args=None, env=None):
    from prsm.cli import node as _node_group
    runner = CliRunner()
    return runner.invoke(
        _node_group,
        ["partial-completion-history"] + (args or []),
        env=env or {},
    )


def test_cli_command_registered():
    from prsm.cli import node as _node_group
    cmd_names = [c.name for c in _node_group.commands.values()]
    assert "partial-completion-history" in cmd_names


def test_cli_text_renders_entries():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "entries": [
            {
                "timestamp": 1000.0,
                "job_id": "job-1",
                "operator_node_id": "a" * 32,
                "reason": "error",
                "tokens_completed": 4,
                "tokens_requested": 10,
            },
        ],
        "total": 1,
        "offset": 0,
        "limit": 50,
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke_cli(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "job-1" in result.output
    assert "error" in result.output.lower()


def test_cli_json_returns_payload():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "entries": [],
        "total": 0,
        "offset": 0,
        "limit": 50,
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke_cli(["--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "entries" in data
    assert data["total"] == 0


def test_cli_operator_node_id_flag_threaded():
    """--operator-node-id passes through to the endpoint query
    string."""
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "entries": [],
        "total": 0,
        "offset": 0,
        "limit": 50,
    }
    with patch("httpx.get", return_value=fake) as mock_get:
        result = _invoke_cli([
            "--operator-node-id", "a" * 32,
            "--format", "json",
        ])
    assert result.exit_code == 0
    call_kwargs = mock_get.call_args.kwargs
    params = call_kwargs.get("params") or {}
    assert params.get("operator_node_id") == "a" * 32


def test_cli_unreachable_daemon_exits_2():
    with patch(
        "httpx.get",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke_cli(["--format", "text"])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()


def test_cli_empty_actionable():
    """No entries → operator-facing message, not silent."""
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "entries": [],
        "total": 0,
        "offset": 0,
        "limit": 50,
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke_cli(["--format", "text"])
    assert result.exit_code == 0
    out = result.output.lower()
    assert "no partial-completion" in out or "no slash" in out or "no events" in out


# ---- Daemon-startup wire-up ----------------------------------


def test_node_start_source_constructs_ring():
    """PRSMNode.start must construct the
    PartialCompletionEventRing + assign to
    self._partial_completion_event_log so sprint 784/785 append
    sites see a live ring."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "_partial_completion_event_log" in src
    assert "PartialCompletionEventRing" in src
