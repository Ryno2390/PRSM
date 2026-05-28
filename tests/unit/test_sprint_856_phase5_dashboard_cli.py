"""Sprint 856 — `prsm node phase5-dashboard` unified view tests.

Defends the operator-facing dashboard that combines sp859/864/857
into one command. Key invariant: independent endpoint failures
must NOT blank the working sections — sibling fail-soft pattern.
"""
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


from prsm.cli import node_phase5_dashboard  # noqa: E402


_PHASE5_BODY = {
    "overall": "PARTIAL",
    "live_surface_count": 3, "total_surface_count": 5,
    "surfaces": {
        "kyc": {
            "commissioned": True, "adapter_wired": True,
            "live_exec": True, "notes": "ready",
        },
        "waas": {
            "commissioned": True, "adapter_wired": True,
            "live_exec": True, "notes": "ready",
        },
        "onramp": {
            "commissioned": True, "adapter_wired": True,
            "live_exec": True, "notes": "ready",
        },
        "paymaster": {
            "commissioned": True, "adapter_wired": True,
            "live_exec": False, "notes": "sp856 closes live",
        },
        "aerodrome": {
            "commissioned": False, "adapter_wired": False,
            "live_exec": False, "notes": "pool ceremony pending",
        },
    },
}

_TREASURY_BODY = {
    "overall": {
        "total_usdc": 0.0, "total_usdc_units": 0,
        "total_ftns": 0.0, "total_ftns_units": 0,
        "total_native_eth": 0.0, "total_native_eth_wei": 0,
        "wallet_count_total": 1,
        "wallet_count_with_address": 1,
        "wallet_count_funded": 0,
        "block_number": 46602212,
        "rpc_url": "https://mainnet.base.org",
    },
    "wallets": [],
}

_FUNNEL_BODY = {
    "summary": {
        "total_intents": 1,
        "status_counts": {
            "INTENT_RECORDED": 0,
            "PENDING_SETTLEMENT": 1,
            "CONFIRMED": 0,
            "EXPIRED": 0,
        },
        "total_expected_usd": 5.0,
        "total_confirmed_usdc": 0.0,
        "conversion_rate": 0.0,
    },
    "intents": [],
}


def _all_endpoints_handler():
    def h(request):
        path = request.url.path
        if path == "/wallet/phase5/status":
            return httpx.Response(200, json=_PHASE5_BODY)
        if path == "/wallet/treasury":
            return httpx.Response(200, json=_TREASURY_BODY)
        if path == "/wallet/onramp/funnel":
            return httpx.Response(200, json=_FUNNEL_BODY)
        return httpx.Response(404, json={"detail": "?"})
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


# ── Happy path ───────────────────────────────────────────────

def test_renders_all_three_sections(patched_client):
    runner = CliRunner()
    with patched_client(_all_endpoints_handler()):
        result = runner.invoke(node_phase5_dashboard, [])
    assert result.exit_code == 0
    assert "Phase 5 Dashboard" in result.output
    assert "§ Readiness" in result.output
    assert "§ Treasury" in result.output
    assert "§ Onramp" in result.output


def test_readiness_shows_surface_marks(patched_client):
    runner = CliRunner()
    with patched_client(_all_endpoints_handler()):
        result = runner.invoke(node_phase5_dashboard, [])
    # KYC live, paymaster not — different marks should appear
    assert "kyc" in result.output
    assert "paymaster" in result.output
    assert "✓" in result.output
    assert "✗" in result.output


def test_treasury_shows_wallet_counts_and_balances(patched_client):
    runner = CliRunner()
    with patched_client(_all_endpoints_handler()):
        result = runner.invoke(node_phase5_dashboard, [])
    assert "1 wallets" in result.output
    assert "1 provisioned" in result.output
    assert "0 funded" in result.output
    assert "46602212" in result.output  # block


def test_onramp_shows_intent_count_and_rate(patched_client):
    runner = CliRunner()
    with patched_client(_all_endpoints_handler()):
        result = runner.invoke(node_phase5_dashboard, [])
    assert "1 intents" in result.output
    assert "0.0% conv rate" in result.output
    assert "$5.00" in result.output
    assert "pending=1" in result.output


# ── JSON format ──────────────────────────────────────────────

def test_json_format_returns_all_three_sections(patched_client):
    runner = CliRunner()
    with patched_client(_all_endpoints_handler()):
        result = runner.invoke(
            node_phase5_dashboard, ["--format", "json"],
        )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "phase5_status" in parsed
    assert "treasury" in parsed
    assert "onramp_funnel" in parsed
    assert parsed["phase5_status"]["overall"] == "PARTIAL"


# ── Partial-failure resilience (load-bearing invariant) ──────

def test_phase5_endpoint_500_does_not_blank_others(patched_client):
    """Sibling fail-soft: phase5 status 500 must still render
    treasury + funnel sections — operator sees what IS available."""
    def handler(request):
        path = request.url.path
        if path == "/wallet/phase5/status":
            return httpx.Response(500, json={"detail": "boom"})
        if path == "/wallet/treasury":
            return httpx.Response(200, json=_TREASURY_BODY)
        if path == "/wallet/onramp/funnel":
            return httpx.Response(200, json=_FUNNEL_BODY)
        return httpx.Response(404, json={})

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_phase5_dashboard, [])
    assert result.exit_code == 0
    # Phase5 section shows error inline
    assert "Readiness ERR" in result.output
    # Other sections still render
    assert "§ Treasury" in result.output
    assert "1 wallets" in result.output
    assert "§ Onramp" in result.output


def test_treasury_500_does_not_blank_others(patched_client):
    def handler(request):
        path = request.url.path
        if path == "/wallet/phase5/status":
            return httpx.Response(200, json=_PHASE5_BODY)
        if path == "/wallet/treasury":
            return httpx.Response(503, json={"detail": "rpc down"})
        if path == "/wallet/onramp/funnel":
            return httpx.Response(200, json=_FUNNEL_BODY)
        return httpx.Response(404, json={})

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_phase5_dashboard, [])
    assert result.exit_code == 0
    assert "§ Readiness" in result.output
    assert "Treasury ERR" in result.output
    assert "§ Onramp" in result.output


def test_funnel_500_does_not_blank_others(patched_client):
    def handler(request):
        path = request.url.path
        if path == "/wallet/onramp/funnel":
            return httpx.Response(500, json={"detail": "?"})
        if path == "/wallet/phase5/status":
            return httpx.Response(200, json=_PHASE5_BODY)
        if path == "/wallet/treasury":
            return httpx.Response(200, json=_TREASURY_BODY)
        return httpx.Response(404, json={})

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_phase5_dashboard, [])
    assert result.exit_code == 0
    assert "§ Readiness" in result.output
    assert "§ Treasury" in result.output
    assert "Onramp ERR" in result.output


# ── All-3-down → exit 2 ──────────────────────────────────────

def test_all_three_unreachable_exits_2(patched_client):
    def handler(request):
        raise httpx.ConnectError("refused")

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_phase5_dashboard, [])
    assert result.exit_code == 2
    assert "All 3 endpoints unreachable" in result.output
    assert "prsm node start" in result.output


# ── --api-port ──────────────────────────────────────────────

def test_custom_api_port_honored(patched_client):
    captured_urls = []

    def handler(request):
        captured_urls.append(str(request.url))
        path = request.url.path
        if path == "/wallet/phase5/status":
            return httpx.Response(200, json=_PHASE5_BODY)
        if path == "/wallet/treasury":
            return httpx.Response(200, json=_TREASURY_BODY)
        if path == "/wallet/onramp/funnel":
            return httpx.Response(200, json=_FUNNEL_BODY)
        return httpx.Response(404, json={})

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(
            node_phase5_dashboard, ["--api-port", "9876"],
        )
    assert result.exit_code == 0
    assert any(":9876/" in u for u in captured_urls)
