"""Sprint 387 — prsm_bootstrap_test MCP tool.

AI-assisted complement to sprint-385's CLI subcommand.
Same probe surface (probe_fleet from
prsm.cli_helpers.bootstrap_probe), rendered for MCP.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.cli_helpers.bootstrap_probe import (
    FleetProbe,
    HostProbe,
    ProbeStatus,
)
from prsm.mcp_server import (
    TOOL_HANDLERS,
    handle_prsm_bootstrap_test,
)


def _fleet_all_ok():
    return FleetProbe(hosts=[
        HostProbe(
            url="wss://bootstrap1.prsm-network.com:8765",
            host="bootstrap1.prsm-network.com",
            port=8765,
            status=ProbeStatus.OK,
            tcp_ok=True, tls_ok=True, wss_ok=True,
            latency_ms=42.0,
            cert_subject="bootstrap1.prsm-network.com",
            cert_issuer="Let's Encrypt",
        ),
        HostProbe(
            url="wss://bootstrap-apac.prsm-network.com:8765",
            host="bootstrap-apac.prsm-network.com",
            port=8765,
            status=ProbeStatus.OK,
            tcp_ok=True, tls_ok=True, wss_ok=True,
            latency_ms=150.0,
            cert_subject="bootstrap-apac.prsm-network.com",
            cert_issuer="Let's Encrypt",
        ),
    ])


def _fleet_partial():
    return FleetProbe(hosts=[
        HostProbe(
            url="wss://up.example.com:8765",
            host="up.example.com", port=8765,
            status=ProbeStatus.OK,
            tcp_ok=True, tls_ok=True, wss_ok=True,
            latency_ms=50.0,
        ),
        HostProbe(
            url="wss://down.example.com:8765",
            host="down.example.com", port=8765,
            status=ProbeStatus.DNS_FAIL,
            error="gaierror: nodename not known",
        ),
    ])


# ── Tool registration ────────────────────────────────


def test_tool_registered_in_handlers_dict():
    assert "prsm_bootstrap_test" in TOOL_HANDLERS


# ── Happy path ───────────────────────────────────────


@pytest.mark.asyncio
async def test_all_healthy_renders_marker_and_summary():
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=_fleet_all_ok()),
    ):
        result = await handle_prsm_bootstrap_test({})
    assert "all healthy" in result.lower()
    assert "2/2 reachable" in result
    # Hostnames render
    assert "bootstrap1.prsm-network.com" in result
    assert "bootstrap-apac.prsm-network.com" in result


@pytest.mark.asyncio
async def test_per_layer_breakdown_rendered():
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=_fleet_all_ok()),
    ):
        result = await handle_prsm_bootstrap_test({})
    # All three layers shown per host
    assert "TCP" in result
    assert "TLS" in result
    assert "WSS" in result


@pytest.mark.asyncio
async def test_cert_subject_and_issuer_rendered():
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=_fleet_all_ok()),
    ):
        result = await handle_prsm_bootstrap_test({})
    assert "Let's Encrypt" in result
    assert "bootstrap1.prsm-network.com" in result


# ── Failure rendering ────────────────────────────────


@pytest.mark.asyncio
async def test_partial_failure_marker_and_error_surfaced():
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=_fleet_partial()),
    ):
        result = await handle_prsm_bootstrap_test({})
    assert "partial" in result.lower()
    assert "1/2 reachable" in result
    # Failed host's error renders
    assert "gaierror" in result


@pytest.mark.asyncio
async def test_all_degraded_marker():
    fleet = FleetProbe(hosts=[
        HostProbe(
            url="wss://a:8765", host="a", port=8765,
            status=ProbeStatus.TCP_FAIL,
            error="refused",
        ),
        HostProbe(
            url="wss://b:8765", host="b", port=8765,
            status=ProbeStatus.TIMEOUT,
            error="timed out",
        ),
    ])
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=fleet),
    ):
        result = await handle_prsm_bootstrap_test({})
    assert "all degraded" in result.lower()
    # Both errors surface
    assert "refused" in result
    assert "timed out" in result


# ── Input handling ───────────────────────────────────


@pytest.mark.asyncio
async def test_urls_param_overrides_canonical_fleet():
    """Operator-supplied URLs replace the canonical list."""
    captured = {}

    async def fake_probe_fleet(urls, *, timeout_seconds):
        captured["urls"] = list(urls)
        return _fleet_all_ok()

    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=fake_probe_fleet,
    ):
        await handle_prsm_bootstrap_test({
            "urls": ["wss://custom1:8765", "wss://custom2:8765"],
        })
    assert captured["urls"] == [
        "wss://custom1:8765",
        "wss://custom2:8765",
    ]


@pytest.mark.asyncio
async def test_timeout_param_propagated():
    captured = {}

    async def fake_probe_fleet(urls, *, timeout_seconds):
        captured["timeout"] = timeout_seconds
        return _fleet_all_ok()

    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=fake_probe_fleet,
    ):
        await handle_prsm_bootstrap_test({
            "urls": ["wss://a:8765"],
            "timeout": 30,
        })
    assert captured["timeout"] == 30.0


@pytest.mark.asyncio
async def test_default_timeout_is_10s():
    captured = {}

    async def fake_probe_fleet(urls, *, timeout_seconds):
        captured["timeout"] = timeout_seconds
        return _fleet_all_ok()

    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=fake_probe_fleet,
    ):
        await handle_prsm_bootstrap_test({
            "urls": ["wss://a:8765"],
        })
    assert captured["timeout"] == 10.0


@pytest.mark.asyncio
async def test_empty_urls_falls_back_to_canonical():
    """No urls param → uses canonical_bootstrap_urls()."""
    captured = {}

    async def fake_probe_fleet(urls, *, timeout_seconds):
        captured["urls"] = list(urls)
        return _fleet_all_ok()

    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=fake_probe_fleet,
    ):
        await handle_prsm_bootstrap_test({})
    # At least one canonical URL fired
    assert len(captured["urls"]) >= 1
    # And they're all wss:// URLs at canonical port
    assert all(
        "8765" in u for u in captured["urls"]
    )


# ── Error handling ───────────────────────────────────


@pytest.mark.asyncio
async def test_probe_exception_surfaces_as_error_message():
    async def fake_probe_fleet(urls, *, timeout_seconds):
        raise RuntimeError("network down")

    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=fake_probe_fleet,
    ):
        result = await handle_prsm_bootstrap_test({})
    assert "failed" in result.lower()
    assert "network down" in result


@pytest.mark.asyncio
async def test_no_urls_at_all_surfaces_clear_error():
    """Empty urls AND canonical_bootstrap_urls returns []
    (extreme edge — operator env-cleared everything)."""
    with patch(
        "prsm.cli_helpers.bootstrap_probe.canonical_bootstrap_urls",
        return_value=[],
    ):
        result = await handle_prsm_bootstrap_test({})
    assert "no URLs" in result or "no urls" in result.lower()
