"""Sprint 516 — `prsm ftns history --onchain --inbound --stats` CLI.

Symmetry with outbound: `--onchain --stats` renders aggregate
outbound summary (sprint 505). Sprint 516 adds the inbound
counterpart that wraps sprint-515's
/wallet/transactions/onchain/inbound/stats endpoint.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def test_inbound_stats_renders_summary():
    """--onchain --inbound --stats hits the stats endpoint
    and renders count + total + block range."""
    from prsm.cli import main as cli

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "recipient": (
            "0x4acdE458766C704B2511583572303e77109cFFE8"
        ),
        "from_block": 46155000,
        "to_block": 46165000,
        "count": 9,
        "total_inbound_ftns": 2.000008,
        "first_inbound_block": 46159960,
        "last_inbound_block": 46165077,
    }
    with patch("httpx.get", return_value=mock_resp):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ftns", "history",
                "--onchain", "--inbound", "--stats",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "9" in result.output
    assert "2.000008" in result.output
    assert "46159960" in result.output
    assert "46165077" in result.output


def test_inbound_stats_503_surfaces_detail():
    from prsm.cli import main as cli

    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.json.return_value = {
        "detail": (
            "On-chain FTNS ledger not initialized — daemon "
            "must be started with FTNS_WALLET_PRIVATE_KEY set."
        ),
    }
    with patch("httpx.get", return_value=mock_resp):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ftns", "history",
                "--onchain", "--inbound", "--stats",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    assert result.exit_code != 0
    assert "FTNS_WALLET_PRIVATE_KEY" in result.output


def test_inbound_stats_hits_stats_endpoint_not_list():
    """Must call the /stats sibling endpoint, not the
    list endpoint."""
    from prsm.cli import main as cli

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "recipient": "0xAAAA", "count": 0,
        "total_inbound_ftns": 0.0,
        "first_inbound_block": None,
        "last_inbound_block": None,
        "from_block": 0, "to_block": 100,
    }
    with patch("httpx.get", return_value=mock_resp) as p:
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "ftns", "history",
                "--onchain", "--inbound", "--stats",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    called_url = p.call_args.args[0]
    assert "inbound/stats" in called_url
