"""Sprint 513 — `prsm ftns history --onchain --inbound` CLI flag.

Sprint 512 shipped /wallet/transactions/onchain/inbound. Sprint
513 wraps it in the existing CLI: `prsm ftns history --onchain
--inbound` renders a Rich table of inbound transfers, matching
the outbound pattern from sprint 500.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def test_inbound_flag_exists():
    """`--inbound` must be an option on `ftns history`."""
    from prsm.cli import main as cli
    runner = CliRunner()
    result = runner.invoke(
        cli, ["ftns", "history", "--help"],
    )
    assert result.exit_code == 0
    assert "--inbound" in result.output
    assert "lookback-blocks" in result.output or (
        "lookback_blocks" in result.output
    )


def test_inbound_renders_table():
    """--onchain --inbound hits the endpoint and prints
    block + from + amount."""
    from prsm.cli import main as cli

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "recipient": (
            "0x4acdE458766C704B2511583572303e77109cFFE8"
        ),
        "from_block": 46159000,
        "to_block": 46160500,
        "count": 1,
        "transfers": [
            {
                "block_number": 46159960,
                "tx_hash": "0x" + "e8" * 32,
                "from_address": (
                    "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
                ),
                "to_address": (
                    "0x4acdE458766C704B2511583572303e77109cFFE8"
                ),
                "amount_ftns": 2.0,
            },
        ],
    }
    with patch("httpx.get", return_value=mock_resp):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ftns", "history",
                "--onchain", "--inbound",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "46159960" in result.output
    assert "2.000000" in result.output
    # Rich table truncates tx_hash per column width;
    # just check the recognizable prefix appears.
    assert "e8e8e8e8" in result.output


def test_inbound_empty_state_graceful():
    """No inbound transfers → clean message, not crash."""
    from prsm.cli import main as cli

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "recipient": (
            "0x4acdE458766C704B2511583572303e77109cFFE8"
        ),
        "from_block": 0,
        "to_block": 100,
        "count": 0,
        "transfers": [],
    }
    with patch("httpx.get", return_value=mock_resp):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ftns", "history",
                "--onchain", "--inbound",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    assert result.exit_code == 0
    assert "No inbound transfers" in result.output


def test_inbound_503_surfaces_detail():
    """503 from daemon must surface actionable detail."""
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
                "--onchain", "--inbound",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    assert result.exit_code != 0
    assert "FTNS_WALLET_PRIVATE_KEY" in result.output
