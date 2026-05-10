"""prsm content mine CLI command (sprint 128)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import content


@pytest.fixture
def runner():
    return CliRunner()


def _ok(payload):
    r = MagicMock()
    r.status_code = 200
    r.json = MagicMock(return_value=payload)
    return r


class TestContentMine:
    def test_renders_uploads(self, runner):
        payload = {
            "entries": [
                {
                    "content_id": "cid-abcdef0123456789xyz",
                    "filename": "paper.pdf",
                    "size_bytes": 12345,
                    "royalty_rate": 0.05,
                    "access_count": 7,
                    "total_royalties": 0.42,
                    "provenance_tx_hash": "0xPROVTX",
                },
            ],
            "total": 1, "offset": 0, "limit": 20,
        }
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=_ok(payload))
            result = runner.invoke(content, ["mine"])
        assert result.exit_code == 0
        assert "paper.pdf" in result.output
        assert "0.420000 FTNS" in result.output
        assert "[chain]" in result.output  # provenance marker

    def test_empty_friendly(self, runner):
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(
                return_value=_ok({"entries": [], "total": 0}),
            )
            result = runner.invoke(content, ["mine"])
        assert result.exit_code == 0
        assert "No uploads" in result.output

    def test_json_format(self, runner):
        payload = {"entries": [], "total": 0, "offset": 0, "limit": 20}
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=_ok(payload))
            result = runner.invoke(content, ["mine", "--format", "json"])
        assert result.exit_code == 0
        assert json.loads(result.output) == payload

    def test_503_friendly_exit_0(self, runner):
        bad = MagicMock()
        bad.status_code = 503
        bad.json = MagicMock(
            return_value={"detail": "ContentUploader not initialized"},
        )
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=bad)
            result = runner.invoke(content, ["mine"])
        # 503 not configured = exit 0 (informational, not error)
        assert result.exit_code == 0
        assert "not configured" in result.output.lower()

    def test_node_unreachable_exit_2(self, runner):
        import httpx
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(
                side_effect=httpx.RequestError("conn"),
            )
            result = runner.invoke(content, ["mine"])
        assert result.exit_code == 2

    def test_off_chain_provenance_marker(self, runner):
        """Content uploaded without on-chain registration shows [off]."""
        payload = {
            "entries": [{
                "content_id": "cid1",
                "filename": "x.txt",
                "size_bytes": 100,
                "royalty_rate": 0.01,
                "access_count": 0,
                "total_royalties": 0.0,
                "provenance_tx_hash": None,  # no on-chain tx
            }],
            "total": 1,
        }
        with patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=_ok(payload))
            result = runner.invoke(content, ["mine"])
        assert "[off]" in result.output
