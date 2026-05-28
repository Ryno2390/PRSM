"""Sprint 866 — `prsm node onramp-funnel` CLI pin tests."""
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


from prsm.cli import node_onramp_funnel  # noqa: E402


_FUNNEL_BODY = {
    "summary": {
        "total_intents": 2,
        "status_counts": {
            "INTENT_RECORDED": 0,
            "PENDING_SETTLEMENT": 1,
            "CONFIRMED": 1,
            "EXPIRED": 0,
        },
        "total_expected_usd": 15.0,
        "total_confirmed_usdc": 4.92,
        "conversion_rate": 0.5,
    },
    "intents": [
        {
            "intent_id": "onramp_aaaa1111bbbb2222",
            "user_id": "alice",
            "destination_address": "0x" + "11" * 20,
            "expected_usd": 5.0,
            "session_token": "tk_alice",
            "created_at": 1779990000.0,
            "status": "CONFIRMED",
            "confirmed_at": 1779990100.0,
            "usdc_received": 4.92,
            "usdc_received_units": 4_920_000,
            "expired_at": 0.0,
        },
        {
            "intent_id": "onramp_cccc3333dddd4444",
            "user_id": "bob",
            "destination_address": "0x" + "22" * 20,
            "expected_usd": 10.0,
            "session_token": "tk_bob",
            "created_at": 1779991000.0,
            "status": "PENDING_SETTLEMENT",
            "confirmed_at": 0.0,
            "usdc_received": 0.0,
            "usdc_received_units": 0,
            "expired_at": 0.0,
        },
    ],
    "limit": 50,
    "filter_status": None,
}

_SWEEP_BODY = {
    "checked": 2, "confirmed_new": 1, "expired_new": 0,
}


def _handler_funnel_only():
    def h(request):
        return httpx.Response(200, json=_FUNNEL_BODY)
    return h


def _handler_funnel_and_sweep():
    def h(request):
        if request.url.path == "/wallet/onramp/sweep":
            return httpx.Response(200, json=_SWEEP_BODY)
        return httpx.Response(200, json=_FUNNEL_BODY)
    return h


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


# ── JSON format ──────────────────────────────────────────────

def test_json_format_pass_through(patched_client):
    runner = CliRunner()
    with patched_client(_handler_funnel_only()):
        result = runner.invoke(
            node_onramp_funnel, ["--format", "json"],
        )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["summary"]["total_intents"] == 2
    assert parsed["summary"]["conversion_rate"] == 0.5


def test_json_format_includes_sweep_when_requested(patched_client):
    runner = CliRunner()
    with patched_client(_handler_funnel_and_sweep()):
        result = runner.invoke(
            node_onramp_funnel,
            ["--format", "json", "--sweep"],
        )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "_sweep" in parsed
    assert parsed["_sweep"]["confirmed_new"] == 1


# ── Text format ──────────────────────────────────────────────

def test_text_format_shows_header_and_summary(patched_client):
    runner = CliRunner()
    with patched_client(_handler_funnel_only()):
        result = runner.invoke(node_onramp_funnel, [])
    assert result.exit_code == 0
    assert "Conversion Funnel" in result.output
    assert "50.0% conversion rate" in result.output
    assert "$15.00" in result.output  # total expected
    assert "$4.92" in result.output  # confirmed USDC


def test_text_format_shows_status_distribution(patched_client):
    runner = CliRunner()
    with patched_client(_handler_funnel_only()):
        result = runner.invoke(node_onramp_funnel, [])
    assert "Status Distribution" in result.output
    assert "INTENT_RECORDED" in result.output
    assert "PENDING_SETTLEMENT" in result.output
    assert "CONFIRMED" in result.output
    assert "EXPIRED" in result.output


def test_text_format_shows_per_intent_table(patched_client):
    runner = CliRunner()
    with patched_client(_handler_funnel_only()):
        result = runner.invoke(node_onramp_funnel, [])
    assert "alice" in result.output
    assert "bob" in result.output
    assert "$5.00" in result.output
    assert "$10.00" in result.output


def test_text_format_sweep_header_shown_with_flag(patched_client):
    runner = CliRunner()
    with patched_client(_handler_funnel_and_sweep()):
        result = runner.invoke(node_onramp_funnel, ["--sweep"])
    assert result.exit_code == 0
    assert "Sweep" in result.output
    assert "checked=2" in result.output
    assert "confirmed_new=" in result.output


# ── Status filter ────────────────────────────────────────────

def test_status_filter_passes_through_to_endpoint(patched_client):
    captured_urls = []

    def handler(request):
        captured_urls.append(str(request.url))
        return httpx.Response(200, json=_FUNNEL_BODY)

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(
            node_onramp_funnel, ["--status", "CONFIRMED"],
        )
    assert result.exit_code == 0
    assert any("status=CONFIRMED" in u for u in captured_urls)


def test_status_filter_invalid_value_rejected_by_click():
    """Click's Choice validator rejects invalid status values."""
    runner = CliRunner()
    result = runner.invoke(
        node_onramp_funnel, ["--status", "BOGUS"],
    )
    assert result.exit_code != 0  # non-zero from Click


# ── Limit option ─────────────────────────────────────────────

def test_limit_query_param_honored(patched_client):
    captured_urls = []

    def handler(request):
        captured_urls.append(str(request.url))
        return httpx.Response(200, json=_FUNNEL_BODY)

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(
            node_onramp_funnel, ["--limit", "25"],
        )
    assert result.exit_code == 0
    assert any("limit=25" in u for u in captured_urls)


# ── Empty funnel ─────────────────────────────────────────────

def test_empty_funnel_renders_clean(patched_client):
    empty = {
        "summary": {
            "total_intents": 0,
            "status_counts": {
                "INTENT_RECORDED": 0,
                "PENDING_SETTLEMENT": 0,
                "CONFIRMED": 0,
                "EXPIRED": 0,
            },
            "total_expected_usd": 0.0,
            "total_confirmed_usdc": 0.0,
            "conversion_rate": 0.0,
        },
        "intents": [],
        "limit": 50,
        "filter_status": None,
    }
    runner = CliRunner()
    with patched_client(_handler_funnel_only := (
        lambda: (lambda r: httpx.Response(200, json=empty))
    )()):
        result = runner.invoke(node_onramp_funnel, [])
    assert result.exit_code == 0
    assert "No intents to display" in result.output


# ── Error paths ──────────────────────────────────────────────

def test_exit_2_on_funnel_connection_error(patched_client):
    def handler(request):
        raise httpx.ConnectError("refused")

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_onramp_funnel, [])
    assert result.exit_code == 2
    assert "Cannot reach PRSM node" in result.output


def test_exit_2_on_sweep_connection_error(patched_client):
    """--sweep failures surface as exit 2 + 'Cannot reach' before
    the funnel call even fires."""
    def handler(request):
        raise httpx.ConnectError("refused")

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_onramp_funnel, ["--sweep"])
    assert result.exit_code == 2


def test_exit_1_on_funnel_non_200(patched_client):
    def handler(request):
        return httpx.Response(500, json={"detail": "internal"})

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_onramp_funnel, [])
    assert result.exit_code == 1
    assert "500" in result.output


def test_exit_1_on_sweep_non_200(patched_client):
    def handler(request):
        if request.url.path == "/wallet/onramp/sweep":
            return httpx.Response(503, json={"detail": "rpc down"})
        return httpx.Response(200, json=_FUNNEL_BODY)

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_onramp_funnel, ["--sweep"])
    assert result.exit_code == 1
    assert "503" in result.output
