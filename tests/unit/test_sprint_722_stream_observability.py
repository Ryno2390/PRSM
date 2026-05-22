"""Sprint 722 — stream observability CLI + endpoint.

After sprints 711 (wire protocol), 713 (bounded queue), 719
(sender binding), 720 (disconnect cleanup), and 721 (request size
limit), operators had NO direct visibility into what their daemon
was doing with remote token-streams. Sprint 722 ships:

  - `/admin/parallax/streams` GET endpoint reading
    `node._chain_executor_pending_streams`
  - `prsm node streams` CLI wrapping that endpoint
  - Both reveal: active stream count, per-stream queue depth +
    maxsize (back-pressure signal), expected_sender prefix (hijack
    defense visibility), env values in effect.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_register_parallax_streams_endpoint_function_exists():
    """Pin: the registration function is exported from api.py so
    create_api_app can wire it (mirrors sprint-685 pool snapshot)."""
    from prsm.node.api import register_parallax_streams_endpoint
    assert callable(register_parallax_streams_endpoint)


def test_streams_endpoint_registered_before_dashboard_mount_in_source():
    """Pin: dashboard mount is a catch-all; any /admin/* endpoint
    registered AFTER the mount silently 404s in production despite
    appearing in openapi.json. Sprint 685 hit this (F30). Sprint
    722 must follow the same registration order."""
    import inspect
    from prsm.node import api as _api_mod
    src = inspect.getsource(_api_mod)
    stream_reg_idx = src.find("register_parallax_streams_endpoint(app, node)")
    dash_mount_idx = src.find("_dash_app = _create_dash_app")
    assert stream_reg_idx > 0, "stream endpoint registration not found"
    assert dash_mount_idx > 0, "dashboard mount call not found"
    assert stream_reg_idx < dash_mount_idx, (
        "sprint 722 endpoint must be registered BEFORE the dashboard "
        "mount (sprint 685 F30 lesson)"
    )


@pytest.mark.asyncio
async def test_streams_endpoint_returns_503_when_no_pending_state():
    """When node has no `_chain_executor_pending_streams` attr
    (daemon not started or wrong build), endpoint returns 503 with
    actionable error."""
    from fastapi import FastAPI, HTTPException
    from prsm.node.api import register_parallax_streams_endpoint
    app = FastAPI()
    node = MagicMock()
    # Force attr-absence: hasattr returns True for MagicMock auto-
    # children, so explicitly stub getattr default-None pattern.
    node._chain_executor_pending_streams = None
    register_parallax_streams_endpoint(app, node)
    # Call the handler directly to avoid TestClient cost.
    route = next(
        r for r in app.router.routes
        if getattr(r, "path", "") == "/admin/parallax/streams"
    )
    with pytest.raises(HTTPException) as exc_info:
        await route.endpoint()
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_streams_endpoint_reports_active_streams_with_queue_depth():
    """Happy path: one in-flight stream with depth=3 / maxsize=64
    + bound sender. Endpoint reports all fields."""
    import asyncio
    from fastapi import FastAPI
    from prsm.node.api import register_parallax_streams_endpoint

    app = FastAPI()
    # Real asyncio.Queue so qsize/maxsize/full() reflect reality.
    q = asyncio.Queue(maxsize=64)
    await q.put(b"f1")
    await q.put(b"f2")
    await q.put(b"f3")
    node = MagicMock()
    node._chain_executor_pending_streams = {
        "abcdef0123456789abcdef0123456789": (
            q, "expected-peer-id-1234567890abcdef",
        ),
    }
    register_parallax_streams_endpoint(app, node)
    route = next(
        r for r in app.router.routes
        if getattr(r, "path", "") == "/admin/parallax/streams"
    )
    result = await route.endpoint()
    assert result["count"] == 1
    assert "queue_maxsize" in result  # env-resolved (sprint 713)
    assert "request_max_bytes" in result  # env-resolved (sprint 721)
    s0 = result["streams"][0]
    assert s0["queue_depth"] == 3
    assert s0["queue_maxsize"] == 64
    assert s0["queue_full"] is False
    # Sender prefix is truncated to 16 chars
    assert s0["expected_sender_prefix"] == "expected-peer-id"
    assert s0["stream_id_prefix"] == "abcdef0123456789"


@pytest.mark.asyncio
async def test_streams_endpoint_handles_legacy_bare_queue_shape():
    """Defensive: if pending[] still holds the legacy bare queue
    (pre-sprint-719), endpoint must NOT crash. Reports
    expected_sender as empty string."""
    import asyncio
    from fastapi import FastAPI
    from prsm.node.api import register_parallax_streams_endpoint

    app = FastAPI()
    q = asyncio.Queue(maxsize=64)
    node = MagicMock()
    # Legacy: bare queue, no tuple
    node._chain_executor_pending_streams = {
        "legacy-id-1": q,
    }
    register_parallax_streams_endpoint(app, node)
    route = next(
        r for r in app.router.routes
        if getattr(r, "path", "") == "/admin/parallax/streams"
    )
    result = await route.endpoint()
    assert result["count"] == 1
    s0 = result["streams"][0]
    assert s0["expected_sender_prefix"] == ""
    assert s0["queue_depth"] == 0


def test_cli_streams_command_registered():
    """Pin: `prsm node streams` is a registered Click command."""
    from prsm.cli import node as _node_group
    cmd_names = [c.name for c in _node_group.commands.values()]
    assert "streams" in cmd_names, (
        "sprint 722 `prsm node streams` CLI command not registered"
    )
