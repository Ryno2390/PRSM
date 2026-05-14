"""Sprint 390 — bootstrap-server-side ops probe.

Operator-trifecta complement to:
  - sprint 380's `prsm node bootstrap` (operator NODE's
    bootstrap-registration state)
  - sprint 385's `prsm node bootstrap-test` (probes ALL
    canonical bootstraps from the operator's perspective)

This sprint adds the third corner: `prsm bootstrap-server
status` — for someone running a bootstrap *droplet*. SSH
in (or point at it remotely) and get a one-screen ops
summary: TCP connect to api_port + /health JSON +
/metrics JSON (the JSON view from sprint 389 content-neg)
rendered as TCP/HTTP/peer-state markers.

Sprint 389 added the observability surface. Sprint 390
makes consuming it ergonomic.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from prsm.cli_helpers.bootstrap_server_probe import (
    BootstrapServerProbe,
    ProbeStatus,
    fetch_server_status,
)


# ── fetch_server_status — happy path ─────────────────────


@pytest.mark.asyncio
async def test_fetch_server_status_success():
    health_payload = {
        "healthy": True,
        "uptime_seconds": 12345.6,
        "active_connections": 3,
    }
    metrics_payload = {
        "active_connections": 3,
        "total_connections": 42,
        "rejected_connections": 1,
        "messages_processed": 999,
        "total_peers_served": 17,
        "bytes_sent": 1024,
        "bytes_received": 2048,
        "uptime_seconds": 12345.6,
        "errors_count": 0,
        "peers_by_region": {"us-east-1": 2, "eu-west-1": 1},
        "peers_by_capability": {"compute": 3},
    }
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def fake_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if url.endswith("/health"):
            resp.json = lambda: health_payload
        elif url.endswith("/metrics"):
            resp.json = lambda: metrics_payload
        return resp

    mock_client.get = fake_get

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.httpx.AsyncClient",
        return_value=mock_client,
    ):
        probe = await fetch_server_status(
            host="bootstrap-eu.prsm-network.com",
            port=8000,
            timeout_seconds=5.0,
        )
    assert probe.status == ProbeStatus.OK
    assert probe.health == health_payload
    assert probe.metrics == metrics_payload
    assert probe.host == "bootstrap-eu.prsm-network.com"
    assert probe.port == 8000
    assert probe.error is None


@pytest.mark.asyncio
async def test_fetch_server_status_connect_refused():
    """When the bootstrap server isn't running, surface a
    clean error message instead of an unwrapped exception."""
    import httpx
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def fake_get(*args, **kwargs):
        raise httpx.ConnectError("All connection attempts failed")

    mock_client.get = fake_get

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.httpx.AsyncClient",
        return_value=mock_client,
    ):
        probe = await fetch_server_status(
            host="127.0.0.1", port=8000, timeout_seconds=2.0,
        )
    assert probe.status == ProbeStatus.CONNECT_FAIL
    assert "connection" in (probe.error or "").lower()
    assert probe.health is None
    assert probe.metrics is None


@pytest.mark.asyncio
async def test_fetch_server_status_timeout():
    import httpx
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def fake_get(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    mock_client.get = fake_get

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.httpx.AsyncClient",
        return_value=mock_client,
    ):
        probe = await fetch_server_status(
            host="127.0.0.1", port=8000, timeout_seconds=0.1,
        )
    assert probe.status == ProbeStatus.TIMEOUT
    assert probe.error is not None


@pytest.mark.asyncio
async def test_fetch_server_status_http_error():
    """A 500 from /health surfaces as HTTP_ERROR — not OK."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def fake_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        return resp

    mock_client.get = fake_get

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.httpx.AsyncClient",
        return_value=mock_client,
    ):
        probe = await fetch_server_status(
            host="127.0.0.1", port=8000, timeout_seconds=2.0,
        )
    assert probe.status == ProbeStatus.HTTP_ERROR
    assert "500" in (probe.error or "")


@pytest.mark.asyncio
async def test_fetch_server_status_partial_metrics_unavailable():
    """If /health succeeds but /metrics fails, return PARTIAL
    — health intel is the load-bearing liveness signal; metrics
    are observability gravy. Fail-soft on metrics."""
    health_payload = {"healthy": True, "uptime_seconds": 1.0}
    import httpx
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def fake_get(url, *args, **kwargs):
        if url.endswith("/health"):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = lambda: health_payload
            return resp
        raise httpx.ReadTimeout("metrics took too long")

    mock_client.get = fake_get

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.httpx.AsyncClient",
        return_value=mock_client,
    ):
        probe = await fetch_server_status(
            host="127.0.0.1", port=8000, timeout_seconds=2.0,
        )
    assert probe.status == ProbeStatus.PARTIAL
    assert probe.health == health_payload
    assert probe.metrics is None
    assert probe.error is not None  # documents which path failed


# ── to_dict + structured shape ───────────────────────────


@pytest.mark.asyncio
async def test_probe_to_dict_shape_is_stable():
    """Output schema is the public contract for the
    --format json CLI surface."""
    probe = BootstrapServerProbe(
        host="x", port=8000, status=ProbeStatus.OK,
        health={"a": 1}, metrics={"b": 2},
    )
    d = probe.to_dict()
    # Sprint 393 added health_detailed as an additive field;
    # defaults to None when caller doesn't set
    # include_subsystems=True.
    assert d == {
        "host": "x",
        "port": 8000,
        "status": "ok",
        "health": {"a": 1},
        "metrics": {"b": 2},
        "error": None,
        "health_detailed": None,
    }
