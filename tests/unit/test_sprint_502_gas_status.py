"""Sprint 502 — ETH gas balance monitoring surface.

The daemon broadcasts on-chain TX using FTNS_WALLET_PRIVATE_KEY's
ETH balance for gas. If that balance drops too low, broadcasts
fail mid-flight with cryptic "insufficient funds for gas" errors.

Operators need a pre-flight signal so they can top up before
things break. Sprint 502 ships:

  1. GET /wallet/gas-status → {address, eth_balance_wei,
     eth_balance, low_threshold_eth, critical_threshold_eth,
     status: "ok" | "low" | "critical" | "unavailable"}
  2. CLI: `prsm wallet info` extended to show ETH balance + a
     low-gas warning when below threshold

Thresholds (Base mainnet at observed ~0.0072 Gwei + ~60k gas/TX
≈ 4.3e-7 ETH/TX):
  critical: < 0.0001 ETH  (~230 TX runway — top up NOW)
  low:      < 0.0005 ETH  (~1150 TX runway — top up soon)
  ok:       ≥ 0.0005 ETH

Boundary: tests use a mocked OnChainFTNSLedger (no real RPC).
The endpoint reads from `ledger.w3.eth.get_balance()` which is
mocked to return controlled wei values.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def _build_app_with_ledger(eth_wei, has_w3=True):
    """Construct an app with a ledger whose w3.eth.get_balance
    returns the given wei amount."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    ledger = MagicMock()
    ledger._connected_address = (
        "0x4acdE458766C704B2511583572303e77109cFFE8"
    )
    if has_w3:
        ledger.w3 = MagicMock()
        ledger.w3.eth.get_balance.return_value = eth_wei
    else:
        ledger.w3 = None
    node.ftns_ledger = ledger
    return create_api_app(node, enable_security=False), node, ledger


def test_endpoint_503_when_ledger_missing():
    """Same 503 contract as sibling /wallet/* endpoints."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    node.ftns_ledger = None
    app = create_api_app(node, enable_security=False)
    client = TestClient(app)
    r = client.get("/wallet/gas-status")
    assert r.status_code == 503


def test_endpoint_returns_ok_status_when_balance_healthy():
    """Balance ≥ 0.0005 ETH (5e14 wei) must report status=ok."""
    app, _, _ = _build_app_with_ledger(eth_wei=10**15)  # 0.001 ETH
    client = TestClient(app)
    r = client.get("/wallet/gas-status")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["eth_balance"] == 0.001
    assert body["eth_balance_wei"] == 10**15
    assert body["address"] == (
        "0x4acdE458766C704B2511583572303e77109cFFE8"
    )


def test_endpoint_returns_low_status_below_threshold():
    """0.0001 ETH ≤ balance < 0.0005 ETH → status=low."""
    # 0.0003 ETH = 3 × 10^14 wei
    app, _, _ = _build_app_with_ledger(eth_wei=3 * 10**14)
    client = TestClient(app)
    body = client.get("/wallet/gas-status").json()
    assert body["status"] == "low"


def test_endpoint_returns_critical_status_below_critical():
    """Balance < 0.0001 ETH → status=critical."""
    # 0.00005 ETH = 5 × 10^13 wei
    app, _, _ = _build_app_with_ledger(eth_wei=5 * 10**13)
    client = TestClient(app)
    body = client.get("/wallet/gas-status").json()
    assert body["status"] == "critical"


def test_endpoint_handles_w3_not_initialized():
    """If ledger has no w3 (init failed or pre-init), endpoint
    must return status=unavailable, not 500."""
    app, _, _ = _build_app_with_ledger(eth_wei=0, has_w3=False)
    client = TestClient(app)
    r = client.get("/wallet/gas-status")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "unavailable"


def test_endpoint_includes_thresholds():
    """Response must surface the threshold constants so
    operators can correlate the status string with the
    actual numeric boundaries."""
    app, _, _ = _build_app_with_ledger(eth_wei=10**15)
    client = TestClient(app)
    body = client.get("/wallet/gas-status").json()
    assert "low_threshold_eth" in body
    assert "critical_threshold_eth" in body
    assert body["low_threshold_eth"] == 0.0005
    assert body["critical_threshold_eth"] == 0.0001


# ── CLI tests ───────────────────────────────────────────


def test_cli_gas_status_exists():
    """`prsm wallet gas-status` must be registered."""
    from click.testing import CliRunner
    from prsm.cli import main as cli
    runner = CliRunner()
    result = runner.invoke(
        cli, ["wallet", "gas-status", "--help"],
    )
    assert result.exit_code == 0
    assert "gas" in result.output.lower()


def test_cli_gas_status_surfaces_critical_warning():
    """CLI must show the actionable warning when daemon
    reports status=critical."""
    from unittest.mock import MagicMock, patch
    from click.testing import CliRunner
    from prsm.cli import main as cli

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "address": "0xAAAA",
        "eth_balance_wei": 5 * 10**13,
        "eth_balance": 0.00005,
        "low_threshold_eth": 0.0005,
        "critical_threshold_eth": 0.0001,
        "status": "critical",
    }
    with patch("httpx.get", return_value=mock_response):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "wallet", "gas-status",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "CRITICAL" in result.output
    assert "Top up" in result.output or "top up" in result.output
