"""prsm node claim-royalty CLI command (sprint 127)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import node


@pytest.fixture
def runner():
    return CliRunner()


def _ok(payload):
    r = MagicMock()
    r.status_code = 200
    r.json = MagicMock(return_value=payload)
    return r


class TestClaimRoyalty:
    def test_dry_run_default_shows_claimable(self, runner):
        # Default: --execute NOT set, expect dry-run path
        captured = {}

        def capture_post(url, json):
            captured["json"] = json
            return _ok({
                "status": "DRY_RUN",
                "claimable_wei": 1_500_000_000_000_000_000,
            })

        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.post = MagicMock(side_effect=capture_post)
            result = runner.invoke(node, ["claim-royalty"])
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "1.500000 FTNS" in result.output
        # dry_run=True passed in body
        assert captured["json"]["dry_run"] is True

    def test_execute_flag_sends_tx(self, runner):
        captured = {}

        def capture_post(url, json):
            captured["json"] = json
            return _ok({
                "status": "EXECUTED",
                "tx_hash": "0xCLAIMTX",
                "amount_claimed_wei": 1_500_000_000_000_000_000,
            })

        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.post = MagicMock(side_effect=capture_post)
            result = runner.invoke(
                node, ["claim-royalty", "--execute"],
            )
        assert result.exit_code == 0
        assert "0xCLAIMTX" in result.output
        assert "1.500000 FTNS" in result.output
        assert captured["json"]["dry_run"] is False

    def test_zero_claimable_friendly(self, runner):
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.post = MagicMock(
                return_value=_ok({"status": "SKIPPED_ZERO"}),
            )
            result = runner.invoke(node, ["claim-royalty"])
        assert result.exit_code == 0
        assert "Nothing to claim" in result.output

    def test_503_friendly(self, runner):
        bad = MagicMock()
        bad.status_code = 503
        bad.json = MagicMock(return_value={"detail": "not wired"})
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.post = MagicMock(return_value=bad)
            result = runner.invoke(node, ["claim-royalty"])
        assert result.exit_code == 1
        assert "not wired" in result.output.lower()

    def test_node_unreachable_exit_2(self, runner):
        import httpx
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.post = MagicMock(
                side_effect=httpx.RequestError("conn"),
            )
            result = runner.invoke(node, ["claim-royalty"])
        assert result.exit_code == 2
