"""Sprint 385 — `prsm node bootstrap-test` CLI tests.

Operator-trifecta complement to sprint-380's
`prsm node bootstrap` (which shows THIS node's bootstrap
state). The bootstrap-test subcommand probes ALL canonical
hosts from where the operator is standing.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import node
from prsm.cli_helpers.bootstrap_probe import (
    FleetProbe,
    HostProbe,
    ProbeStatus,
)


@pytest.fixture
def runner():
    return CliRunner()


def _fake_fleet_all_ok():
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


def _fake_fleet_partial():
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
            status=ProbeStatus.TCP_FAIL,
            tcp_ok=False, tls_ok=False, wss_ok=False,
            error="ConnectionRefused",
        ),
    ])


# ── Happy path ───────────────────────────────────────


def test_text_render_all_healthy(runner):
    fleet = _fake_fleet_all_ok()
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=fleet),
    ):
        result = runner.invoke(
            node, ["bootstrap-test"],
        )
    assert result.exit_code == 0
    assert "all healthy" in result.output
    assert "2/2 reachable" in result.output
    assert "bootstrap1.prsm-network.com" in result.output
    assert "bootstrap-apac.prsm-network.com" in result.output


def test_text_render_includes_per_layer_marks(runner):
    fleet = _fake_fleet_all_ok()
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=fleet),
    ):
        result = runner.invoke(node, ["bootstrap-test"])
    # Each host should show TCP / TLS / WSS layer marks
    assert "TCP" in result.output
    assert "TLS" in result.output
    assert "WSS" in result.output


def test_text_render_includes_cert_info_when_present(runner):
    fleet = _fake_fleet_all_ok()
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=fleet),
    ):
        result = runner.invoke(node, ["bootstrap-test"])
    assert "Let's Encrypt" in result.output


# ── Failure rendering ────────────────────────────────


def test_text_render_partial_failure_exits_1(runner):
    """Exit code 1 = some hosts degraded, others healthy."""
    fleet = _fake_fleet_partial()
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=fleet),
    ):
        result = runner.invoke(node, ["bootstrap-test"])
    assert result.exit_code == 1
    assert "partial" in result.output
    assert "1/2 reachable" in result.output
    # Failed host's error renders
    assert "ConnectionRefused" in result.output


def test_text_render_all_failed_exits_2(runner):
    """Exit code 2 = ALL hosts unreachable. Different
    severity than partial (1) — ops alerts can branch."""
    fleet = FleetProbe(hosts=[
        HostProbe(
            url="wss://a:8765", host="a", port=8765,
            status=ProbeStatus.TCP_FAIL,
            error="refused",
        ),
        HostProbe(
            url="wss://b:8765", host="b", port=8765,
            status=ProbeStatus.DNS_FAIL,
            error="nxdomain",
        ),
    ])
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=fleet),
    ):
        result = runner.invoke(node, ["bootstrap-test"])
    assert result.exit_code == 2
    assert "all degraded" in result.output


# ── JSON output ──────────────────────────────────────


def test_json_format_emits_full_payload(runner):
    fleet = _fake_fleet_all_ok()
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=fleet),
    ):
        result = runner.invoke(
            node,
            ["bootstrap-test", "--format", "json"],
        )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["all_healthy"] is True
    assert len(payload["hosts"]) == 2


def test_json_format_exit_code_matches_health(runner):
    fleet = _fake_fleet_partial()
    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=AsyncMock(return_value=fleet),
    ):
        result = runner.invoke(
            node,
            ["bootstrap-test", "--format", "json"],
        )
    assert result.exit_code == 1


# ── --url override ───────────────────────────────────


def test_url_flag_overrides_canonical_fleet(runner):
    """--url <wss://X> tests only that URL, not the
    canonical fleet."""
    captured = {}

    async def fake_probe_fleet(urls, *, timeout_seconds):
        captured["urls"] = list(urls)
        return _fake_fleet_all_ok()

    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=fake_probe_fleet,
    ):
        runner.invoke(
            node,
            ["bootstrap-test", "--url", "wss://custom:8765"],
        )
    assert captured["urls"] == ["wss://custom:8765"]


def test_multiple_url_flags_accumulated(runner):
    captured = {}

    async def fake_probe_fleet(urls, *, timeout_seconds):
        captured["urls"] = list(urls)
        return _fake_fleet_all_ok()

    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=fake_probe_fleet,
    ):
        runner.invoke(node, [
            "bootstrap-test",
            "--url", "wss://a:8765",
            "--url", "wss://b:8765",
        ])
    assert captured["urls"] == [
        "wss://a:8765",
        "wss://b:8765",
    ]


def test_timeout_flag_propagated_to_probe(runner):
    captured = {}

    async def fake_probe_fleet(urls, *, timeout_seconds):
        captured["timeout"] = timeout_seconds
        return _fake_fleet_all_ok()

    with patch(
        "prsm.cli_helpers.bootstrap_probe.probe_fleet",
        new=fake_probe_fleet,
    ):
        runner.invoke(node, [
            "bootstrap-test",
            "--url", "wss://a:8765",
            "--timeout", "30",
        ])
    assert captured["timeout"] == 30.0
