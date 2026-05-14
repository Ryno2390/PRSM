"""Sprint 391 — prsm_bootstrap_server_status MCP tool.

AI-assisted complement to sprint-390's
`prsm bootstrap-server status` CLI. Same probe core
(prsm.cli_helpers.bootstrap_server_probe.fetch_server_status),
MCP-rendered for AI-side-panel ops triage.

Completes the operator-trifecta MCP coverage:
  - prsm_bootstrap_status (sprint 266) — THIS node's
    registration state
  - prsm_bootstrap_test (sprint 387) — canonical fleet
    probe from this vantage
  - prsm_bootstrap_server_status (sprint 391) — MY OWN
    bootstrap droplet's HTTP control surface
"""
from __future__ import annotations

from unittest.mock import patch, AsyncMock

import pytest

from prsm.cli_helpers.bootstrap_server_probe import (
    BootstrapServerProbe,
    ProbeStatus,
)
from prsm.mcp_server import (
    TOOL_HANDLERS,
    handle_prsm_bootstrap_server_status,
)


def _ok_probe(host="bootstrap-eu.prsm-network.com", port=8000):
    return BootstrapServerProbe(
        host=host, port=port, status=ProbeStatus.OK,
        health={
            "healthy": True, "uptime_seconds": 7200.0,
            "active_connections": 4,
        },
        metrics={
            "active_connections": 4,
            "total_connections": 88,
            "rejected_connections": 3,
            "total_peers_served": 25,
            "messages_processed": 5000,
            "errors_count": 1,
            "uptime_seconds": 7200.0,
            "peers_by_region": {
                "us-east-1": 1, "eu-west-1": 3,
            },
            "peers_by_capability": {
                "compute": 4, "storage": 2,
            },
        },
    )


def _refused_probe():
    return BootstrapServerProbe(
        host="127.0.0.1", port=8000,
        status=ProbeStatus.CONNECT_FAIL,
        error="connection error: connection refused",
    )


# ── Tool registration ────────────────────────────────────


def test_tool_registered_in_handlers_dict():
    assert "prsm_bootstrap_server_status" in TOOL_HANDLERS


# ── Healthy rendering ────────────────────────────────────


@pytest.mark.asyncio
async def test_healthy_renders_marker_and_summary():
    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=AsyncMock(return_value=_ok_probe()),
    ):
        result = await handle_prsm_bootstrap_server_status({})
    assert "healthy" in result.lower() or "ok" in result.lower()
    assert "bootstrap-eu.prsm-network.com" in result
    # Key metrics surface
    assert "88" in result or "total_connections" in result.lower()


@pytest.mark.asyncio
async def test_healthy_renders_per_region_labeled_dict():
    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=AsyncMock(return_value=_ok_probe()),
    ):
        result = await handle_prsm_bootstrap_server_status({})
    # Region labels surface in the rendered metrics
    assert "us-east-1" in result
    assert "eu-west-1" in result


# ── Failure rendering ────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_refused_renders_error_and_status():
    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=AsyncMock(return_value=_refused_probe()),
    ):
        result = await handle_prsm_bootstrap_server_status({})
    assert (
        "connect" in result.lower()
        or "refused" in result.lower()
        or "✗" in result
        or "❌" in result
    )


@pytest.mark.asyncio
async def test_partial_status_marker_renders():
    partial = BootstrapServerProbe(
        host="x", port=8000, status=ProbeStatus.PARTIAL,
        health={"healthy": True, "uptime_seconds": 1.0},
        metrics=None,
        error="/metrics fetch failed",
    )
    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=AsyncMock(return_value=partial),
    ):
        result = await handle_prsm_bootstrap_server_status({})
    assert (
        "partial" in result.lower()
        or "degraded" in result.lower()
        or "⚠" in result
    )
    # Even on partial, the health intel surfaces
    assert "uptime_seconds" in result or "1.0" in result


# ── Input plumbing ───────────────────────────────────────


@pytest.mark.asyncio
async def test_defaults_to_localhost_8000():
    captured = {}

    async def fake_probe(host, port, **kw):
        captured["host"] = host
        captured["port"] = port
        return _ok_probe(host=host, port=port)

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=fake_probe,
    ):
        await handle_prsm_bootstrap_server_status({})
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000


@pytest.mark.asyncio
async def test_host_and_port_overrides_propagate():
    captured = {}

    async def fake_probe(host, port, **kw):
        captured["host"] = host
        captured["port"] = port
        return _ok_probe(host=host, port=port)

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=fake_probe,
    ):
        await handle_prsm_bootstrap_server_status(
            {"host": "remote.example", "port": 9001},
        )
    assert captured["host"] == "remote.example"
    assert captured["port"] == 9001


@pytest.mark.asyncio
async def test_timeout_param_propagated():
    captured = {}

    async def fake_probe(host, port, *, timeout_seconds, **kw):
        captured["timeout"] = timeout_seconds
        return _ok_probe()

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=fake_probe,
    ):
        await handle_prsm_bootstrap_server_status(
            {"timeout": 30},
        )
    assert captured["timeout"] == 30.0


@pytest.mark.asyncio
async def test_default_timeout_is_5s():
    captured = {}

    async def fake_probe(host, port, *, timeout_seconds, **kw):
        captured["timeout"] = timeout_seconds
        return _ok_probe()

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=fake_probe,
    ):
        await handle_prsm_bootstrap_server_status({})
    assert captured["timeout"] == 5.0


# ── Error handling ───────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_exception_surfaces_as_error_message():
    async def fake_probe(host, port, **kw):
        raise RuntimeError("network down")

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.fetch_server_status",
        new=fake_probe,
    ):
        result = await handle_prsm_bootstrap_server_status({})
    assert "failed" in result.lower() or "error" in result.lower()
    assert "network down" in result
