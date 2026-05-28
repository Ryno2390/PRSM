"""Sprint 861 — `prsm node phase5-status` CLI pin tests.

Defends the terminal-friendly readiness grid rendered atop sp859's
aggregator endpoint. Click TestRunner exercises the command body
against mocked daemon responses; defends the load-bearing
behaviors:

  - --format json passes the body through unchanged
  - --format text renders the canonical surface order (kyc → waas
    → onramp → paymaster → aerodrome) — matches user-onboarding
    flow, NOT dict-iteration order
  - Overall status colored by READY / PARTIAL / NOT_READY
  - Exit code 2 when daemon unreachable
  - Exit code 1 when daemon returns non-200
"""
from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner


# Restore real httpx classes (same pattern as sp849/sp850/sp854)
_real_Client = httpx.Client


@pytest.fixture(autouse=True)
def _restore_real_httpx_classes(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _real_Client)
    yield


from prsm.cli import node_phase5_status  # noqa: E402


def _mock_response(body: dict, status: int = 200):
    """Build a mock daemon response."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)
    return handler


@pytest.fixture
def patched_client():
    """Patch httpx.Client at the module level — the CLI command
    body does `import httpx` inline + calls httpx.Client(), so
    we patch the httpx module's Client class itself."""
    def _patch(handler):
        return patch(
            "httpx.Client",
            lambda *a, **kw: _real_Client(
                transport=httpx.MockTransport(handler),
            ),
        )
    return _patch


# ── JSON format pass-through ─────────────────────────────────

def test_json_format_passes_body_through(patched_client):
    body = {
        "overall": "PARTIAL",
        "live_surface_count": 3,
        "total_surface_count": 5,
        "surfaces": {
            "kyc": {
                "commissioned": True, "adapter_wired": True,
                "live_exec": True, "notes": "ready",
            },
            "waas": {
                "commissioned": True, "adapter_wired": True,
                "live_exec": True, "wallet_count": 1,
                "notes": "ready",
            },
            "paymaster": {
                "commissioned": True, "adapter_wired": True,
                "live_exec": False, "sponsorships": 0,
                "notes": "sp856 closes",
            },
            "onramp": {
                "commissioned": True, "adapter_wired": True,
                "live_exec": True, "notes": "ready",
            },
            "aerodrome": {
                "commissioned": False, "adapter_wired": False,
                "live_exec": False, "notes": "pool ceremony",
            },
        },
    }
    runner = CliRunner()
    with patched_client(_mock_response(body)):
        result = runner.invoke(
            node_phase5_status, ["--format", "json"],
        )
    assert result.exit_code == 0
    # Strip any rich-console formatting from the output
    parsed = json.loads(result.output)
    assert parsed["overall"] == "PARTIAL"
    assert parsed["live_surface_count"] == 3
    assert parsed["surfaces"]["aerodrome"]["live_exec"] is False


# ── Text format rendering ────────────────────────────────────

def test_text_format_renders_header(patched_client):
    body = {
        "overall": "READY",
        "live_surface_count": 5,
        "total_surface_count": 5,
        "surfaces": {
            "kyc": {"commissioned": True, "adapter_wired": True,
                    "live_exec": True, "notes": "r"},
            "waas": {"commissioned": True, "adapter_wired": True,
                     "live_exec": True, "notes": "r"},
            "paymaster": {"commissioned": True,
                          "adapter_wired": True,
                          "live_exec": True, "notes": "r"},
            "onramp": {"commissioned": True, "adapter_wired": True,
                       "live_exec": True, "notes": "r"},
            "aerodrome": {"commissioned": True,
                          "adapter_wired": True,
                          "live_exec": True, "notes": "r"},
        },
    }
    runner = CliRunner()
    with patched_client(_mock_response(body)):
        result = runner.invoke(node_phase5_status, [])
    assert result.exit_code == 0
    assert "Phase 5 Readiness" in result.output
    assert "READY" in result.output
    assert "5/5 live" in result.output


def test_text_format_renders_surface_order(patched_client):
    """User-onboarding flow order: kyc → waas → onramp →
    paymaster → aerodrome. Pinned because dict-iteration order
    isn't guaranteed across the aggregator + future schema
    additions could break this."""
    body = {
        "overall": "PARTIAL",
        "live_surface_count": 3,
        "total_surface_count": 5,
        "surfaces": {
            "kyc": {"commissioned": True, "adapter_wired": True,
                    "live_exec": True, "notes": "n1"},
            "waas": {"commissioned": True, "adapter_wired": True,
                     "live_exec": True, "notes": "n2"},
            "paymaster": {"commissioned": True,
                          "adapter_wired": True,
                          "live_exec": False, "notes": "n3"},
            "onramp": {"commissioned": True, "adapter_wired": True,
                       "live_exec": True, "notes": "n4"},
            "aerodrome": {"commissioned": False,
                          "adapter_wired": False,
                          "live_exec": False, "notes": "n5"},
        },
    }
    runner = CliRunner()
    with patched_client(_mock_response(body)):
        result = runner.invoke(node_phase5_status, [])
    # Find the index of each surface in the output
    out = result.output
    idx_kyc = out.find("kyc")
    idx_waas = out.find("waas")
    idx_onramp = out.find("onramp")
    idx_paymaster = out.find("paymaster")
    idx_aerodrome = out.find("aerodrome")
    assert idx_kyc != -1
    assert idx_kyc < idx_waas < idx_onramp < idx_paymaster < (
        idx_aerodrome
    )


def test_text_format_shows_checkmarks(patched_client):
    body = {
        "overall": "PARTIAL", "live_surface_count": 1,
        "total_surface_count": 5,
        "surfaces": {
            "kyc": {"commissioned": True, "adapter_wired": True,
                    "live_exec": True, "notes": "n"},
            "waas": {"commissioned": False, "adapter_wired": False,
                     "live_exec": False, "notes": "n"},
            "paymaster": {"commissioned": False,
                          "adapter_wired": False,
                          "live_exec": False, "notes": "n"},
            "onramp": {"commissioned": False, "adapter_wired": False,
                       "live_exec": False, "notes": "n"},
            "aerodrome": {"commissioned": False,
                          "adapter_wired": False,
                          "live_exec": False, "notes": "n"},
        },
    }
    runner = CliRunner()
    with patched_client(_mock_response(body)):
        result = runner.invoke(node_phase5_status, [])
    # rich emits checkmarks
    assert "✓" in result.output
    assert "✗" in result.output


# ── Error paths ──────────────────────────────────────────────

def test_exit_code_2_on_connection_error(patched_client):
    """Daemon not running → exit 2 with actionable 'prsm node
    start' hint, NOT a Python traceback."""
    def handler(request):
        raise httpx.ConnectError("connection refused")

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_phase5_status, [])
    assert result.exit_code == 2
    assert "Cannot reach PRSM node" in result.output
    assert "prsm node start" in result.output


def test_exit_code_1_on_non_200(patched_client):
    """Daemon up but endpoint errored → exit 1 with status code +
    response body."""
    body = {"error": "internal"}
    runner = CliRunner()
    with patched_client(_mock_response(body, status=500)):
        result = runner.invoke(node_phase5_status, [])
    assert result.exit_code == 1
    assert "500" in result.output


# ── Custom api-port option ───────────────────────────────────

def test_custom_api_port_flag(patched_client):
    """--api-port flag is exposed for operators running on
    non-default ports."""
    body = {
        "overall": "READY", "live_surface_count": 5,
        "total_surface_count": 5,
        "surfaces": {n: {
            "commissioned": True, "adapter_wired": True,
            "live_exec": True, "notes": "r",
        } for n in [
            "kyc", "waas", "paymaster", "onramp", "aerodrome",
        ]},
    }
    captured_urls = []

    def handler(request):
        captured_urls.append(str(request.url))
        return httpx.Response(200, json=body)

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(
            node_phase5_status, ["--api-port", "9999"],
        )
    assert result.exit_code == 0
    assert any(":9999/" in u for u in captured_urls)
