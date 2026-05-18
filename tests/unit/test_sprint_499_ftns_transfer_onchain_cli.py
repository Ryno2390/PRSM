"""Sprint 499 — ship `prsm ftns transfer-onchain` CLI.

Sprint 498's runbook documented:

    prsm wallet transfer-ftns \\
      --to 0x4acdE458… \\
      --amount 0.000001

but no `prsm wallet` group exists, and the existing
`prsm ftns transfer` uses **user IDs against the
off-chain DAG ledger** — fundamentally different from
the on-chain ERC-20 path.

Sprint 499 closes the documentation drift by shipping
`prsm ftns transfer-onchain` as a sibling to
`prsm ftns transfer`. It wraps the F38 endpoint
(`POST /wallet/transfer/onchain`) with the standard
PRSM CLI ergonomics (auth headers, error handling,
JSON-or-Rich output).

These pin tests use Click's CliRunner against a mocked
HTTP backend — they verify the CLI contract, NOT the
real broadcast (which is exercised by the live runbook
walk on Base mainnet).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_command_exists(runner):
    """The command must be exposed under `prsm ftns
    transfer-onchain`. If a future refactor moves it
    elsewhere, surface that here."""
    from prsm.cli import main as cli

    result = runner.invoke(
        cli, ["ftns", "transfer-onchain", "--help"],
    )
    assert result.exit_code == 0, result.output
    assert "on-chain" in result.output.lower() or (
        "on chain" in result.output.lower()
    )


def test_cli_requires_to_and_amount(runner):
    """Both --to and --amount must be required options.
    Forgetting either should fail before any HTTP call."""
    from prsm.cli import main as cli

    result = runner.invoke(cli, ["ftns", "transfer-onchain"])
    assert result.exit_code != 0
    assert (
        "--to" in result.output or "Missing" in result.output
    )


def test_cli_rejects_non_positive_amount(runner):
    """Amount must be > 0 before hitting the daemon —
    saves a round trip + matches /wallet/transfer/onchain's
    422 contract."""
    from prsm.cli import main as cli

    result = runner.invoke(
        cli,
        [
            "ftns", "transfer-onchain",
            "--to", "0x4acdE458766C704B2511583572303e77109cFFE8",
            "--amount", "0",
        ],
    )
    assert result.exit_code != 0
    assert "positive" in result.output.lower() or (
        "> 0" in result.output
    )


def test_cli_happy_path_surfaces_tx_hash(runner):
    """Successful 200 response from the daemon must
    surface tx_hash + block_number + status to operator.
    Empty tx_hash or status would mean the operator
    can't audit the on-chain side."""
    from prsm.cli import main as cli

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "tx_hash": "0x" + "ab" * 32,
        "status": "confirmed",
        "block_number": 46160224,
        "from_address": (
            "0x4acdE458766C704B2511583572303e77109cFFE8"
        ),
        "to_address": (
            "0x4acdE458766C704B2511583572303e77109cFFE8"
        ),
        "amount_ftns": 0.000001,
        "job_id": "manual-abc123",
    }

    with patch("httpx.post", return_value=mock_response):
        result = runner.invoke(
            cli,
            [
                "ftns", "transfer-onchain",
                "--to", (
                    "0x4acdE458766C704B2511583572303e77109cFFE8"
                ),
                "--amount", "0.000001",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "0x" + "ab" * 32 in result.output
    assert "46160224" in result.output
    assert "confirmed" in result.output


def test_cli_surfaces_503_with_actionable_hint(runner):
    """If daemon returns 503 (no FTNS_WALLET_PRIVATE_KEY),
    the CLI must surface the actionable detail string, not
    a generic 'transfer failed'."""
    from prsm.cli import main as cli

    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.json.return_value = {
        "detail": (
            "On-chain FTNS ledger not initialized — daemon "
            "must be started with FTNS_WALLET_PRIVATE_KEY "
            "set."
        ),
    }
    mock_response.text = '{"detail": "..."}'

    with patch("httpx.post", return_value=mock_response):
        result = runner.invoke(
            cli,
            [
                "ftns", "transfer-onchain",
                "--to", (
                    "0x4acdE458766C704B2511583572303e77109cFFE8"
                ),
                "--amount", "1.0",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    assert result.exit_code != 0
    assert "FTNS_WALLET_PRIVATE_KEY" in result.output
