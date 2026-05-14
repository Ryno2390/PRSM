"""Sprint 393 — extend fetch_server_status to optionally
fetch /health/detailed (sprint 392 endpoint).

Closes the trifecta with subsystem awareness: when
operators set include_subsystems=True, the probe also
hits /health/detailed and stashes the result on
BootstrapServerProbe.health_detailed. Both the CLI
(--detailed flag) and the MCP tool (include_subsystems
boolean) consume this.

Default include_subsystems=False preserves the v1
contract — pre-sprint-393 callers see no change.
"""
from __future__ import annotations

from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from prsm.cli_helpers.bootstrap_server_probe import (
    BootstrapServerProbe,
    ProbeStatus,
    fetch_server_status,
)


def _mock_client_for(responses: dict):
    """Build an AsyncClient mock that routes by URL suffix.

    responses: maps `/path` → either:
      - dict (200 OK with json body)
      - int (raise httpx error with that "status")
      - Exception (raised)
    """
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def fake_get(url, *a, **kw):
        for suffix, action in responses.items():
            if url.endswith(suffix):
                if isinstance(action, Exception):
                    raise action
                resp = MagicMock()
                resp.status_code = 200
                resp.json = lambda payload=action: payload
                return resp
        # Default — 404
        resp = MagicMock()
        resp.status_code = 404
        resp.text = "not found"
        return resp

    mock_client.get = fake_get
    return mock_client


# ── Default behavior (include_subsystems=False) preserved ─


@pytest.mark.asyncio
async def test_default_does_not_fetch_health_detailed():
    """Pre-sprint-393 contract: only /health + /metrics
    are fetched. /health/detailed left alone."""
    captured_urls = []

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def fake_get(url, *a, **kw):
        captured_urls.append(url)
        resp = MagicMock()
        resp.status_code = 200
        resp.json = lambda: {"ok": True}
        return resp

    mock_client.get = fake_get

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.httpx.AsyncClient",
        return_value=mock_client,
    ):
        probe = await fetch_server_status(
            host="x", port=8000, timeout_seconds=2.0,
        )
    # /health/detailed NOT in the URLs hit by default
    assert not any("/health/detailed" in u for u in captured_urls)
    # probe object still has the field, defaulted to None
    assert probe.health_detailed is None


# ── Opt-in fetch ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_include_subsystems_fetches_health_detailed():
    detailed_payload = {
        "status": "healthy",
        "subsystems": {
            "peer_cleanup": {
                "alive": True, "status": "healthy",
                "last_heartbeat_age_seconds": 5.0,
                "expected_interval_seconds": 60.0,
            },
            "peer_backup": {
                "alive": True, "status": "healthy",
                "last_heartbeat_age_seconds": 30.0,
                "expected_interval_seconds": 300.0,
            },
        },
        "server_time": "2026-05-14T15:00:00Z",
    }
    mock_client = _mock_client_for({
        "/health": {"healthy": True},
        "/health/detailed": detailed_payload,
        "/metrics": {"total_connections": 1},
    })
    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.httpx.AsyncClient",
        return_value=mock_client,
    ):
        probe = await fetch_server_status(
            host="x", port=8000, timeout_seconds=2.0,
            include_subsystems=True,
        )
    assert probe.health_detailed == detailed_payload
    assert probe.status == ProbeStatus.OK


# ── Fail-soft on /health/detailed ────────────────────────


@pytest.mark.asyncio
async def test_health_detailed_failure_is_fail_soft():
    """When /health/detailed fetch fails but /health +
    /metrics succeed, probe is still OK — health_detailed
    is None, error notes the failure."""
    import httpx
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def fake_get(url, *a, **kw):
        if url.endswith("/health/detailed"):
            raise httpx.ReadTimeout("detailed timed out")
        if url.endswith("/health"):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = lambda: {"healthy": True}
            return resp
        if url.endswith("/metrics"):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = lambda: {"total_connections": 1}
            return resp
        resp = MagicMock()
        resp.status_code = 404
        return resp

    mock_client.get = fake_get

    with patch(
        "prsm.cli_helpers.bootstrap_server_probe.httpx.AsyncClient",
        return_value=mock_client,
    ):
        probe = await fetch_server_status(
            host="x", port=8000, timeout_seconds=2.0,
            include_subsystems=True,
        )
    # OK because liveness + metrics intact
    assert probe.status == ProbeStatus.OK
    assert probe.health_detailed is None
    assert probe.error is not None
    assert "detailed" in probe.error.lower()


# ── to_dict includes health_detailed ─────────────────────


def test_to_dict_includes_health_detailed_field():
    p = BootstrapServerProbe(
        host="x", port=8000, status=ProbeStatus.OK,
        health={"a": 1}, metrics={"b": 2},
        health_detailed={"status": "degraded"},
    )
    d = p.to_dict()
    assert d["health_detailed"] == {"status": "degraded"}


def test_to_dict_health_detailed_defaults_to_none():
    p = BootstrapServerProbe(
        host="x", port=8000, status=ProbeStatus.OK,
    )
    d = p.to_dict()
    assert "health_detailed" in d
    assert d["health_detailed"] is None
