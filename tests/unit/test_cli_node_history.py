"""prsm node slash-history / heartbeats / distributions CLI."""
from __future__ import annotations

import json
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


class TestSlashHistory:
    def test_renders_entries(self, runner):
        payload = {
            "entries": [
                {
                    "timestamp": 1700000000.0,
                    "kind": "proof_failure_slashed",
                    "provider": "0xPROV",
                    "challenger": "0xCHAL",
                    "slash_id": "0x" + "ab" * 32,
                    "extras": {},
                },
            ],
            "total": 1,
        }
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=_ok(payload))
            result = runner.invoke(node, ["slash-history"])
        assert result.exit_code == 0
        assert "Slash Events" in result.output

    def test_503_friendly_exit_0(self, runner):
        bad = MagicMock()
        bad.status_code = 503
        bad.json = MagicMock(return_value={"detail": "not wired"})
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=bad)
            result = runner.invoke(node, ["slash-history"])
        # 503 = not wired = exit 0 (informational, not error)
        assert result.exit_code == 0
        assert "not configured" in result.output.lower()

    def test_json_format(self, runner):
        payload = {"entries": [], "total": 0}
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=_ok(payload))
            result = runner.invoke(
                node, ["slash-history", "--format", "json"],
            )
        assert result.exit_code == 0
        assert json.loads(result.output) == payload

    def test_node_unreachable_exit_2(self, runner):
        import httpx
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(
                side_effect=httpx.RequestError("conn"),
            )
            result = runner.invoke(node, ["slash-history"])
        assert result.exit_code == 2

    def test_limit_passthrough(self, runner):
        payload = {"entries": [], "total": 0}
        captured_url = {}

        def capture_get(url):
            captured_url["url"] = url
            return _ok(payload)

        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(side_effect=capture_get)
            runner.invoke(node, ["slash-history", "--limit", "5"])
        assert "limit=5" in captured_url["url"]


class TestHeartbeats:
    def test_command_registered(self, runner):
        # Clean smoke test: command is callable end-to-end
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(
                return_value=_ok({"entries": [], "total": 0}),
            )
            result = runner.invoke(node, ["heartbeats"])
        assert result.exit_code == 0
        assert "Heartbeats" in result.output


class TestDistributions:
    def test_command_registered(self, runner):
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(
                return_value=_ok({"entries": [], "total": 0}),
            )
            result = runner.invoke(node, ["distributions"])
        assert result.exit_code == 0
        assert "Distributions" in result.output
