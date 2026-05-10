"""prsm node trigger-heartbeat / trigger-distribution CLI."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import node


@pytest.fixture
def runner():
    return CliRunner()


def _ok():
    r = MagicMock()
    r.status_code = 200
    r.json = MagicMock(return_value={
        "tx_hash": "0xABC", "status": "CONFIRMED",
    })
    return r


def _503():
    r = MagicMock()
    r.status_code = 503
    r.json = MagicMock(return_value={"detail": "not wired"})
    return r


class TestTriggerHeartbeat:
    def test_renders_tx_hash_with_yes(self, runner):
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.post = MagicMock(return_value=_ok())
            result = runner.invoke(
                node, ["trigger-heartbeat", "-y"],
            )
        assert result.exit_code == 0
        assert "0xABC" in result.output
        assert "CONFIRMED" in result.output

    def test_aborts_without_confirmation(self, runner):
        # Without -y, click.confirm rejects on empty input
        with patch("httpx.Client") as MockClient:
            result = runner.invoke(
                node, ["trigger-heartbeat"], input="\n",
            )
        # Aborted = non-zero exit
        assert result.exit_code != 0
        # POST was NOT called because of abort
        ci = MockClient.return_value.__enter__.return_value
        assert not ci.post.called

    def test_503_friendly_exit_1(self, runner):
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.post = MagicMock(return_value=_503())
            result = runner.invoke(
                node, ["trigger-heartbeat", "-y"],
            )
        assert result.exit_code == 1
        assert "unavailable" in result.output.lower()

    def test_node_unreachable_exit_2(self, runner):
        import httpx
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.post = MagicMock(
                side_effect=httpx.RequestError("conn"),
            )
            result = runner.invoke(
                node, ["trigger-heartbeat", "-y"],
            )
        assert result.exit_code == 2


class TestTriggerDistribution:
    def test_renders_tx_hash_with_yes(self, runner):
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.post = MagicMock(return_value=_ok())
            result = runner.invoke(
                node, ["trigger-distribution", "-y"],
            )
        assert result.exit_code == 0
        assert "0xABC" in result.output

    def test_502_chain_error_friendly(self, runner):
        bad = MagicMock()
        bad.status_code = 502
        bad.json = MagicMock(
            return_value={"detail": "zero balance"},
        )
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.post = MagicMock(return_value=bad)
            result = runner.invoke(
                node, ["trigger-distribution", "-y"],
            )
        assert result.exit_code == 1
        assert "on-chain" in result.output.lower()
