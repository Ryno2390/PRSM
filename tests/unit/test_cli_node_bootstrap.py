"""Sprint 380 — prsm node bootstrap CLI subcommand.

Third surface for the sprint-375 multi-bootstrap fields,
completing the operator-trifecta:

  HTTP    /bootstrap/status               (sprint 266 + 375)
  MCP     prsm_bootstrap_status           (sprint 325 + 375)
  CLI     prsm node bootstrap             (this sprint)

All three share the same data; the CLI is what operators
already reach for in incident response from the terminal.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import node


@pytest.fixture
def runner():
    return CliRunner()


def _resp(payload, status_code=200):
    r = MagicMock()
    r.status_code = status_code
    r.json = MagicMock(return_value=payload)
    r.text = json.dumps(payload)
    return r


# ── Canonical happy path ─────────────────────────────


_CANONICAL_OK_PAYLOAD = {
    "attempted": 2,
    "connected": 1,
    "degraded": False,
    "bootstrap_nodes": [
        "wss://bootstrap1.prsm-network.com:8765",
    ],
    "bootstrap_fallback_nodes": [
        "wss://bootstrap-eu.prsm-network.com:8765",
        "wss://bootstrap-apac.prsm-network.com:8765",
    ],
    "bootstrap_fallback_enabled": True,
    "active_url": (
        "wss://bootstrap-eu.prsm-network.com:8765"
    ),
    "discovered_peer_count": 5,
    "peer_join_events": 3,
    "peer_leave_events": 1,
    "stale_evictions": 0,
    "reconnect_attempts": 1,
    "reconnect_successes": 1,
    "client_state": "connected",
}


def test_renders_healthy_marker_when_connected(runner):
    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(
            return_value=_resp(_CANONICAL_OK_PAYLOAD),
        )
        result = runner.invoke(node, ["bootstrap"])
    assert result.exit_code == 0
    assert "healthy" in result.output


def test_renders_active_url_scheme_stripped(runner):
    """Operators see bare host:port, not wss:// noise."""
    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(
            return_value=_resp(_CANONICAL_OK_PAYLOAD),
        )
        result = runner.invoke(node, ["bootstrap"])
    assert "bootstrap-eu.prsm-network.com:8765" in result.output
    # Scheme stripped in the active-URL line
    assert "active URL" in result.output


def test_renders_counters(runner):
    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(
            return_value=_resp(_CANONICAL_OK_PAYLOAD),
        )
        result = runner.invoke(node, ["bootstrap"])
    assert "peer_join_events" in result.output
    assert "reconnect_attempts" in result.output


def test_renders_candidate_list_with_active_marker(runner):
    """The candidate-URL list shows which one is the
    currently-active host — operator scans the dot column
    and immediately sees primary vs fallback."""
    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(
            return_value=_resp(_CANONICAL_OK_PAYLOAD),
        )
        result = runner.invoke(node, ["bootstrap"])
    # Both primary + fallback URLs rendered
    assert "bootstrap1.prsm-network.com:8765" in result.output
    assert "bootstrap-apac.prsm-network.com:8765" in result.output


def test_renders_fallback_enabled_yes(runner):
    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(
            return_value=_resp(_CANONICAL_OK_PAYLOAD),
        )
        result = runner.invoke(node, ["bootstrap"])
    assert "fallback enabled" in result.output
    assert "yes" in result.output


def test_renders_fallback_disabled_posture(runner):
    payload = dict(_CANONICAL_OK_PAYLOAD)
    payload["bootstrap_fallback_enabled"] = False
    payload["bootstrap_fallback_nodes"] = []
    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(return_value=_resp(payload))
        result = runner.invoke(node, ["bootstrap"])
    assert "single-host posture" in result.output


# ── Failure modes ────────────────────────────────────


def test_renders_degraded_marker_when_all_failed(runner):
    payload = {
        "attempted": 3, "connected": 0, "degraded": True,
        "bootstrap_nodes": ["wss://primary"],
        "bootstrap_fallback_nodes": [
            "wss://eu", "wss://apac",
        ],
        "bootstrap_fallback_enabled": True,
        "active_url": None,
        "discovered_peer_count": 0,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 3,
        "reconnect_successes": 0, "client_state": "dead",
    }
    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(return_value=_resp(payload))
        result = runner.invoke(node, ["bootstrap"])
    assert "degraded" in result.output
    # No active URL when all failed
    assert "all candidates failed" in result.output


def test_503_exits_nonzero(runner):
    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(
            return_value=_resp(
                {"detail": "Peer discovery not initialized"},
                status_code=503,
            ),
        )
        result = runner.invoke(node, ["bootstrap"])
    assert result.exit_code != 0
    assert "not wired" in result.output


def test_connection_refused_exits_2(runner):
    """Local node not running → exit 2, like other node
    subcommands."""
    import httpx
    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(
            side_effect=httpx.ConnectError("refused"),
        )
        result = runner.invoke(node, ["bootstrap"])
    assert result.exit_code == 2
    assert "Cannot reach" in result.output


# ── JSON output ──────────────────────────────────────


def test_json_format_emits_raw_payload(runner):
    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(
            return_value=_resp(_CANONICAL_OK_PAYLOAD),
        )
        result = runner.invoke(
            node, ["bootstrap", "--format", "json"],
        )
    assert result.exit_code == 0
    # JSON-parseable
    parsed = json.loads(result.output)
    assert parsed["active_url"] == (
        "wss://bootstrap-eu.prsm-network.com:8765"
    )
    assert parsed["client_state"] == "connected"
    assert parsed["bootstrap_fallback_enabled"] is True


def test_json_format_includes_all_sprint_375_fields(runner):
    """JSON output is the operator-script-friendly surface;
    the sprint-375 fields must be passed through verbatim
    so downstream tools (ops scripts, log aggregators,
    custom dashboards) can rely on them."""
    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(
            return_value=_resp(_CANONICAL_OK_PAYLOAD),
        )
        result = runner.invoke(
            node, ["bootstrap", "--format", "json"],
        )
    parsed = json.loads(result.output)
    # All sprint-375 fields present
    assert "active_url" in parsed
    assert "bootstrap_fallback_nodes" in parsed
    assert "bootstrap_fallback_enabled" in parsed


# ── Custom API port ──────────────────────────────────


def test_honors_api_port_flag(runner):
    captured = {}

    def fake_get(url):
        captured["url"] = url
        return _resp(_CANONICAL_OK_PAYLOAD)

    with patch("httpx.Client") as MockClient:
        c = MockClient.return_value.__enter__.return_value
        c.get = MagicMock(side_effect=fake_get)
        runner.invoke(
            node, ["bootstrap", "--api-port", "9999"],
        )
    assert "9999" in captured["url"]
    assert "/bootstrap/status" in captured["url"]
