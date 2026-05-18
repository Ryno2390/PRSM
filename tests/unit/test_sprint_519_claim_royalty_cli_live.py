"""Sprint 519 — `prsm node claim-royalty` live-verify on mainnet.

Sprint 471 shipped the CLI with status ⚠️ because the live path
returned 503 with an actionable hint about
PRSM_ROYALTY_DISTRIBUTOR_ADDRESS. Sprints 498-518 brought the
operator wallet fully on-chain with all env vars wired
(PRSM_ONCHAIN_FTNS=1, PRSM_ROYALTY_DISTRIBUTOR_ADDRESS, etc).

Sprint 519 confirms the CLI now works end-to-end on the live
mainnet RoyaltyDistributor v2 contract (0xfEa9aeB9…) without
trace of the sprint-471 "missing env hint" path. Both branches
exercised cleanly:

  DRY-RUN  → "Dry run — no on-chain action\n
              Claimable: 0.000000 FTNS (0 wei)\n
              Re-run with --execute to claim on-chain."
  --execute → "Nothing to claim — claimable balance is 0."

Live-verified Base mainnet operator wallet 0x4acdE458… (no
royalties accumulated; real-claim path requires multi-wallet
consumer bench).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def test_dry_run_returns_clean_empty_state():
    """When daemon reports claimable=0, dry-run output
    must surface the 0 balance + hint to use --execute."""
    from prsm.cli import main as cli

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "DRY_RUN",
        "claimable_wei": 0,
        "tx_hash": None,
    }
    fake_client = MagicMock()
    fake_client.__enter__.return_value.post.return_value = mock_resp
    fake_client.__exit__.return_value = None
    with patch("httpx.Client", return_value=fake_client):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["node", "claim-royalty", "--api-port", "8000"],
        )
    assert result.exit_code == 0, result.output
    assert "Claimable" in result.output
    assert "0.000000" in result.output
    assert "execute" in result.output.lower()


def test_execute_short_circuits_when_nothing_to_claim():
    """--execute path with claimable=0 must NOT broadcast
    a TX — wastes gas. Clean exit + clear message."""
    from prsm.cli import main as cli

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "SKIPPED_ZERO",
        "claimable_wei": 0,
        "tx_hash": None,
    }
    fake_client = MagicMock()
    fake_client.__enter__.return_value.post.return_value = mock_resp
    fake_client.__exit__.return_value = None
    with patch("httpx.Client", return_value=fake_client):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "node", "claim-royalty",
                "--api-port", "8000", "--execute",
            ],
        )
    assert result.exit_code == 0
    assert "Nothing to claim" in result.output or (
        "balance is 0" in result.output
    )


def test_dry_run_with_real_claimable_amount():
    """When claimable > 0, dry-run must surface the
    actual amount so operator can decide to --execute."""
    from prsm.cli import main as cli

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "DRY_RUN",
        "claimable_wei": int(5.5 * 10**18),
        "tx_hash": None,
    }
    fake_client = MagicMock()
    fake_client.__enter__.return_value.post.return_value = mock_resp
    fake_client.__exit__.return_value = None
    with patch("httpx.Client", return_value=fake_client):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["node", "claim-royalty", "--api-port", "8000"],
        )
    assert result.exit_code == 0
    assert "5.500000" in result.output or (
        "5.5" in result.output
    )
