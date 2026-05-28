"""Sprint 863 — `prsm node wallet-balance` CLI pin tests.

Defends the terminal-friendly USDC/FTNS/ETH readout backed by
sp862's /wallet/balance/* endpoints. Click TestRunner against
mocked daemon responses.

Pin tests:
  - 0x address routes to /wallet/balance/by-address/<addr>
  - non-0x identifier routes to /wallet/balance/<user_id>
  - --format json passes body through unchanged
  - Text format renders header + table with USDC + FTNS + ETH rows
  - User-id-resolved response renders user_id + wallet_id labels
  - Exit 2 on daemon-unreachable
  - Exit 1 on 404 + actionable "provision first" hint
  - Exit 1 on other non-200 with status code surfaced
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


from prsm.cli import node_wallet_balance  # noqa: E402


def _mock_response(body, status=200):
    def handler(request):
        return httpx.Response(status, json=body)
    return handler


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


_BALANCE_BODY = {
    "address": "0x01D1c152Ef261b1d74983EDC36C47D9cE3ba2fA5",
    "usdc": 0.0, "usdc_units": 0,
    "ftns": 0.0, "ftns_units": 0,
    "native_eth": 0.0, "native_eth_wei": 0,
    "block_number": 46601673,
    "rpc_url": "https://mainnet.base.org",
}


# ── Routing: 0x address vs user_id ───────────────────────────

def test_address_routes_to_by_address_endpoint(patched_client):
    captured_urls = []

    def handler(request):
        captured_urls.append(str(request.url))
        return httpx.Response(200, json=_BALANCE_BODY)

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(
            node_wallet_balance,
            ["0x01D1c152Ef261b1d74983EDC36C47D9cE3ba2fA5"],
        )
    assert result.exit_code == 0
    assert any(
        "/wallet/balance/by-address/" in u for u in captured_urls
    )


def test_user_id_routes_to_user_endpoint(patched_client):
    captured_urls = []

    def handler(request):
        captured_urls.append(str(request.url))
        body = dict(_BALANCE_BODY)
        body["user_id"] = "alice"
        body["wallet_id"] = "prsm-alice-abc"
        body["network"] = "base-mainnet"
        return httpx.Response(200, json=body)

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_wallet_balance, ["alice"])
    assert result.exit_code == 0
    assert any(
        u.endswith("/wallet/balance/alice") for u in captured_urls
    )


def test_short_0x_routed_as_user_id(patched_client):
    """A string starting with 0x but NOT 42 chars total is treated
    as a user_id (some user IDs could legitimately start 0x)."""
    captured_urls = []

    def handler(request):
        captured_urls.append(str(request.url))
        body = dict(_BALANCE_BODY)
        body["user_id"] = "0xshort"
        return httpx.Response(200, json=body)

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_wallet_balance, ["0xshort"])
    # Routed as user_id path, not by-address path
    assert any(
        u.endswith("/wallet/balance/0xshort")
        for u in captured_urls
    )
    assert not any("by-address" in u for u in captured_urls)


# ── JSON format pass-through ─────────────────────────────────

def test_json_format_pass_through(patched_client):
    runner = CliRunner()
    with patched_client(_mock_response(_BALANCE_BODY)):
        result = runner.invoke(
            node_wallet_balance,
            ["0x01D1c152Ef261b1d74983EDC36C47D9cE3ba2fA5",
             "--format", "json"],
        )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["usdc"] == 0.0
    assert parsed["block_number"] == 46601673


# ── Text format rendering ────────────────────────────────────

def test_text_format_shows_header_and_table(patched_client):
    runner = CliRunner()
    with patched_client(_mock_response(_BALANCE_BODY)):
        result = runner.invoke(
            node_wallet_balance,
            ["0x01D1c152Ef261b1d74983EDC36C47D9cE3ba2fA5"],
        )
    assert result.exit_code == 0
    assert "Wallet Balance" in result.output
    assert "USDC" in result.output
    assert "FTNS" in result.output
    assert "ETH" in result.output
    assert "0x01D1c152" in result.output  # address prefix
    assert "46601673" in result.output  # block number


def test_text_format_shows_user_id_when_present(patched_client):
    body = dict(_BALANCE_BODY)
    body["user_id"] = "alice"
    body["wallet_id"] = "prsm-alice-abc"
    runner = CliRunner()
    with patched_client(_mock_response(body)):
        result = runner.invoke(node_wallet_balance, ["alice"])
    assert result.exit_code == 0
    assert "alice" in result.output
    assert "prsm-alice-abc" in result.output


def test_text_format_shows_rpc_url(patched_client):
    """Block + RPC URL together are the audit-signal pair —
    operator can confirm WHICH RPC served the read."""
    runner = CliRunner()
    with patched_client(_mock_response(_BALANCE_BODY)):
        result = runner.invoke(
            node_wallet_balance,
            ["0x01D1c152Ef261b1d74983EDC36C47D9cE3ba2fA5"],
        )
    assert "mainnet.base.org" in result.output


def test_text_format_shows_positive_balance_colored(patched_client):
    """Non-zero balances render in green; tests that the value
    is present at least (color codes vary by terminal)."""
    body = dict(_BALANCE_BODY)
    body["usdc"] = 5.0
    body["usdc_units"] = 5_000_000
    runner = CliRunner()
    with patched_client(_mock_response(body)):
        result = runner.invoke(
            node_wallet_balance,
            ["0x01D1c152Ef261b1d74983EDC36C47D9cE3ba2fA5"],
        )
    assert result.exit_code == 0
    assert "5.000000 USDC" in result.output
    assert "5000000" in result.output  # base units shown


# ── Error paths ──────────────────────────────────────────────

def test_exit_2_on_connection_error(patched_client):
    def handler(request):
        raise httpx.ConnectError("connection refused")

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(
            node_wallet_balance,
            ["0x01D1c152Ef261b1d74983EDC36C47D9cE3ba2fA5"],
        )
    assert result.exit_code == 2
    assert "Cannot reach PRSM node" in result.output
    assert "prsm node start" in result.output


def test_exit_1_on_404_with_provision_hint(patched_client):
    """User_id not in WaaS store — surface actionable hint, not
    raw 404 traceback."""
    runner = CliRunner()
    with patched_client(_mock_response(
        {"detail": "no wallet for user_id='ghost'"}, status=404,
    )):
        result = runner.invoke(node_wallet_balance, ["ghost"])
    assert result.exit_code == 1
    assert "No wallet found" in result.output
    assert "provision first" in result.output


def test_exit_1_on_500(patched_client):
    runner = CliRunner()
    with patched_client(_mock_response(
        {"detail": "Base RPC failed"}, status=500,
    )):
        result = runner.invoke(
            node_wallet_balance,
            ["0x01D1c152Ef261b1d74983EDC36C47D9cE3ba2fA5"],
        )
    assert result.exit_code == 1
    assert "500" in result.output


# ── --api-port flag ──────────────────────────────────────────

def test_custom_api_port_honored(patched_client):
    captured = []

    def handler(request):
        captured.append(str(request.url))
        return httpx.Response(200, json=_BALANCE_BODY)

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(
            node_wallet_balance,
            ["0x01D1c152Ef261b1d74983EDC36C47D9cE3ba2fA5",
             "--api-port", "9999"],
        )
    assert result.exit_code == 0
    assert any(":9999/" in u for u in captured)
