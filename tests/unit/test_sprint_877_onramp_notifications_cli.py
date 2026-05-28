"""Sprint 877 — `prsm node onramp-notifications` CLI pin tests."""
from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner


_real_Client = httpx.Client


@pytest.fixture(autouse=True)
def _restore_real_httpx(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _real_Client)
    yield


from prsm.cli import node_onramp_notifications  # noqa: E402


_BODY = {
    "count": 3, "success_count": 2, "failure_count": 1,
    "configured": True,
    "deliveries": [
        {
            "timestamp": 1700000300.0,
            "intent_id": "onramp_aa",
            "url": "https://example.com/wh",
            "status_code": 200, "success": True,
            "error": None, "signature_attached": True,
        },
        {
            "timestamp": 1700000200.0,
            "intent_id": "onramp_bb",
            "url": "https://example.com/wh",
            "status_code": 500, "success": False,
            "error": "internal",
            "signature_attached": True,
        },
        {
            "timestamp": 1700000100.0,
            "intent_id": "onramp_cc",
            "url": "https://example.com/wh",
            "status_code": 200, "success": True,
            "error": None, "signature_attached": False,
        },
    ],
}


@pytest.fixture
def patched_client():
    def _patch(handler):
        return patch(
            "httpx.Client",
            lambda *a, **kw: _real_Client(
                transport=httpx.MockTransport(handler),
            ),
        )
    return _patch


def _mock(body, status=200):
    def h(request):
        return httpx.Response(status, json=body)
    return h


# ── JSON pass-through ────────────────────────────────────────

def test_json_format_returns_body(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_BODY)):
        result = runner.invoke(
            node_onramp_notifications, ["--format", "json"],
        )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["count"] == 3
    assert len(parsed["deliveries"]) == 3


def test_json_format_applies_success_filter(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_BODY)):
        result = runner.invoke(
            node_onramp_notifications,
            ["--format", "json", "--success-only"],
        )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert len(parsed["deliveries"]) == 2
    assert all(d["success"] for d in parsed["deliveries"])


def test_json_format_applies_failure_filter(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_BODY)):
        result = runner.invoke(
            node_onramp_notifications,
            ["--format", "json", "--failures-only"],
        )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert len(parsed["deliveries"]) == 1
    assert parsed["deliveries"][0]["success"] is False


def test_success_and_failures_both_rejected(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_BODY)):
        result = runner.invoke(
            node_onramp_notifications,
            ["--success-only", "--failures-only"],
        )
    assert result.exit_code == 2
    assert "at most one" in result.output.lower()


# ── Text format ──────────────────────────────────────────────

def test_text_shows_header_and_counts(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_BODY)):
        result = runner.invoke(node_onramp_notifications, [])
    assert result.exit_code == 0
    assert "Onramp Completion Notifications" in result.output
    assert "2 successes" in result.output
    assert "1 failures" in result.output


def test_text_shows_success_rate(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_BODY)):
        result = runner.invoke(node_onramp_notifications, [])
    # 2/3 = 66.7%
    assert "66.7%" in result.output


def test_text_shows_intent_ids_in_rows(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_BODY)):
        result = runner.invoke(node_onramp_notifications, [])
    for intent in ["onramp_aa", "onramp_bb", "onramp_cc"]:
        assert intent in result.output


def test_text_shows_configured_warning_when_url_unset(
    patched_client,
):
    """Operator needs to know if the notifier is wired before
    interpreting an empty history as 'no traffic'."""
    body = dict(_BODY)
    body["configured"] = False
    body["count"] = 0
    body["success_count"] = 0
    body["failure_count"] = 0
    body["deliveries"] = []
    runner = CliRunner()
    with patched_client(_mock(body)):
        result = runner.invoke(node_onramp_notifications, [])
    assert "PRSM_ONRAMP_COMPLETION_WEBHOOK_URL" in result.output
    assert "no-op" in result.output.lower()


def test_text_empty_deliveries_with_filter(patched_client):
    """A success-only filter with no matching rows still renders
    cleanly (not blank), tells operator what was filtered."""
    body = dict(_BODY)
    body["deliveries"] = [
        d for d in body["deliveries"] if not d["success"]
    ]
    runner = CliRunner()
    with patched_client(_mock(body)):
        result = runner.invoke(
            node_onramp_notifications, ["--success-only"],
        )
    assert result.exit_code == 0
    assert (
        "No successful deliveries" in result.output
        or "match the filter" in result.output
    )


# ── Error paths ──────────────────────────────────────────────

def test_exit_2_on_connection_error(patched_client):
    def handler(request):
        raise httpx.ConnectError("refused")

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_onramp_notifications, [])
    assert result.exit_code == 2
    assert "Cannot reach PRSM node" in result.output


def test_exit_1_on_non_200(patched_client):
    runner = CliRunner()
    with patched_client(_mock({"detail": "?"}, status=500)):
        result = runner.invoke(node_onramp_notifications, [])
    assert result.exit_code == 1
    assert "500" in result.output


# ── --limit pass-through ─────────────────────────────────────

def test_limit_query_param_honored(patched_client):
    captured = []

    def handler(request):
        captured.append(str(request.url))
        return httpx.Response(200, json=_BODY)

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(
            node_onramp_notifications, ["--limit", "10"],
        )
    assert result.exit_code == 0
    assert any("limit=10" in u for u in captured)
