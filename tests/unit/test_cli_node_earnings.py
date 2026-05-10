"""prsm node earnings CLI command (sprint 98).

Hits /admin/earnings-summary on the running node daemon and
pretty-prints the 3-stream dashboard.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import node


@pytest.fixture
def runner():
    return CliRunner()


def _ok_response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value=payload)
    return resp


class TestEarnings:
    def test_renders_all_streams_wired(self, runner):
        payload = {
            "operator_address": "0xABC",
            "royalty": {
                "available": True,
                "claimable_wei": 1_500_000_000_000_000_000,
            },
            "heartbeat": {
                "available": True,
                "last_heartbeat": 1700000000,
                "grace_seconds": 3600,
                "grace_remaining": 3500,
                "expired": False,
                "at_risk": False,
            },
            "distribution": {
                "available": True,
                "last_distribution": 1700000000,
                "seconds_since": 7200,
            },
        }
        with patch("httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.get = MagicMock(return_value=_ok_response(payload))
            result = runner.invoke(node, ["earnings"])
        assert result.exit_code == 0
        assert "0xABC" in result.output
        assert "1.500000 FTNS" in result.output
        assert "ok" in result.output

    def test_at_risk_heartbeat_renders_warning(self, runner):
        payload = {
            "operator_address": "0xX",
            "royalty": {"available": False},
            "heartbeat": {
                "available": True,
                "last_heartbeat": 1700000000,
                "grace_seconds": 3600,
                "grace_remaining": 100,
                "expired": False,
                "at_risk": True,
            },
            "distribution": {"available": False},
        }
        with patch("httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.get = MagicMock(return_value=_ok_response(payload))
            result = runner.invoke(node, ["earnings"])
        assert result.exit_code == 0
        assert "at-risk" in result.output
        assert "100s grace remaining" in result.output

    def test_json_format_returns_raw_payload(self, runner):
        payload = {
            "operator_address": "0xX",
            "royalty": {"available": False},
            "heartbeat": {"available": False},
            "distribution": {"available": False},
        }
        with patch("httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.get = MagicMock(return_value=_ok_response(payload))
            result = runner.invoke(
                node, ["earnings", "--format", "json"],
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == payload

    def test_node_unreachable_exits_2(self, runner):
        import httpx
        with patch("httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.get = MagicMock(
                side_effect=httpx.RequestError("conn refused"),
            )
            result = runner.invoke(node, ["earnings"])
        assert result.exit_code == 2
        assert "Cannot reach PRSM node" in result.output

    def test_non_200_exits_1(self, runner):
        bad_resp = MagicMock()
        bad_resp.status_code = 503
        bad_resp.text = "boom"
        with patch("httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.get = MagicMock(return_value=bad_resp)
            result = runner.invoke(node, ["earnings"])
        assert result.exit_code == 1
        assert "503" in result.output
