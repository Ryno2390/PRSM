"""Sprint 865 — `prsm node treasury` CLI pin tests.

Defends the fleet-wide rollup readout. Same Click TestRunner +
httpx.MockTransport pattern as sp861/sp863.
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


from prsm.cli import node_treasury  # noqa: E402


_TREASURY_BODY = {
    "overall": {
        "total_usdc": 5.0, "total_usdc_units": 5_000_000,
        "total_ftns": 100.0, "total_ftns_units": int(100 * 10**18),
        "total_native_eth": 0.1,
        "total_native_eth_wei": int(0.1 * 10**18),
        "wallet_count_total": 2,
        "wallet_count_with_address": 2,
        "wallet_count_funded": 1,
        "block_number": 46601856,
        "rpc_url": "https://mainnet.base.org",
    },
    "wallets": [
        {
            "user_id": "alice",
            "wallet_id": "prsm-alice-abc",
            "address": "0x" + "11" * 20,
            "status": "PROVISIONED",
            "balances": {
                "address": "0x" + "11" * 20,
                "usdc": 5.0, "usdc_units": 5_000_000,
                "ftns": 100.0, "ftns_units": int(100 * 10**18),
                "native_eth": 0.1,
                "native_eth_wei": int(0.1 * 10**18),
                "block_number": 46601856,
                "rpc_url": "https://mainnet.base.org",
            },
        },
        {
            "user_id": "bob",
            "wallet_id": "prsm-bob-xyz",
            "address": "0x" + "22" * 20,
            "status": "PROVISIONED",
            "balances": {
                "address": "0x" + "22" * 20,
                "usdc": 0, "usdc_units": 0,
                "ftns": 0, "ftns_units": 0,
                "native_eth": 0, "native_eth_wei": 0,
                "block_number": 46601856,
                "rpc_url": "https://mainnet.base.org",
            },
        },
    ],
}


def _mock(body, status=200):
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


def test_json_format_pass_through(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_TREASURY_BODY)):
        result = runner.invoke(node_treasury, ["--format", "json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["overall"]["total_usdc"] == 5.0


def test_text_format_shows_aggregate_totals(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_TREASURY_BODY)):
        result = runner.invoke(node_treasury, [])
    assert result.exit_code == 0
    assert "Fleet Treasury" in result.output
    assert "Aggregate Holdings" in result.output
    assert "5.000000 USDC" in result.output
    assert "100.000000 FTNS" in result.output


def test_text_format_shows_per_wallet_breakdown(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_TREASURY_BODY)):
        result = runner.invoke(node_treasury, [])
    assert "Per-Wallet Breakdown" in result.output
    assert "alice" in result.output
    assert "bob" in result.output


def test_text_format_shows_wallet_counts(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_TREASURY_BODY)):
        result = runner.invoke(node_treasury, [])
    # 2 total · 2 provisioned · 1 funded
    assert "2 total" in result.output
    assert "2 provisioned" in result.output
    assert "1 funded" in result.output


def test_text_format_shows_block_and_rpc(patched_client):
    runner = CliRunner()
    with patched_client(_mock(_TREASURY_BODY)):
        result = runner.invoke(node_treasury, [])
    assert "46601856" in result.output
    assert "mainnet.base.org" in result.output


def test_empty_wallets_renders_clean(patched_client):
    body = {
        "overall": {
            "total_usdc": 0, "total_usdc_units": 0,
            "total_ftns": 0, "total_ftns_units": 0,
            "total_native_eth": 0, "total_native_eth_wei": 0,
            "wallet_count_total": 0,
            "wallet_count_with_address": 0,
            "wallet_count_funded": 0,
            "block_number": 0, "rpc_url": None,
        },
        "wallets": [],
        "note": "WaaS client not initialized.",
    }
    runner = CliRunner()
    with patched_client(_mock(body)):
        result = runner.invoke(node_treasury, [])
    assert result.exit_code == 0
    assert "No wallets to display" in result.output
    assert "WaaS client not initialized" in result.output


def test_failed_balance_wallet_renders_with_err_marker(
    patched_client,
):
    body = dict(_TREASURY_BODY)
    body["wallets"] = [{
        "user_id": "bad",
        "wallet_id": "prsm-bad",
        "address": "0x" + "ee" * 20,
        "status": "PROVISIONED",
        "balances": None,
        "error": "simulated RPC failure",
    }]
    runner = CliRunner()
    with patched_client(_mock(body)):
        result = runner.invoke(node_treasury, [])
    assert result.exit_code == 0
    assert "bad" in result.output
    assert "err" in result.output


def test_exit_2_on_connection_error(patched_client):
    def handler(request):
        raise httpx.ConnectError("refused")

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(node_treasury, [])
    assert result.exit_code == 2
    assert "Cannot reach PRSM node" in result.output


def test_exit_1_on_non_200(patched_client):
    runner = CliRunner()
    with patched_client(_mock({"detail": "err"}, status=500)):
        result = runner.invoke(node_treasury, [])
    assert result.exit_code == 1
    assert "500" in result.output


def test_max_wallets_query_param_honored(patched_client):
    captured_urls = []

    def handler(request):
        captured_urls.append(str(request.url))
        return httpx.Response(200, json=_TREASURY_BODY)

    runner = CliRunner()
    with patched_client(handler):
        result = runner.invoke(
            node_treasury, ["--max-wallets", "42"],
        )
    assert result.exit_code == 0
    assert any("max_wallets=42" in u for u in captured_urls)
