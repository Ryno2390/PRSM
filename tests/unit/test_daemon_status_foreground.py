"""prsm node status detects foreground-running nodes (sprint 123)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.cli_modules import daemon


class TestForegroundDetection:
    def test_no_pid_no_api_reports_stopped(self):
        with patch.object(daemon, "_get_daemon_pid", return_value=None), \
                patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(side_effect=Exception("no api"))
            # JSON output for assertable test
            with patch("prsm.cli._agent_output") as out:
                daemon.daemon_status(output_format="json")
            data = out.call_args[0][0]
        assert data["running"] is False
        assert data["foreground_running"] is False
        assert data["foreground_api_port"] is None

    def test_no_pid_but_api_responds_reports_foreground(self):
        ok = MagicMock()
        ok.status_code = 200
        with patch.object(daemon, "_get_daemon_pid", return_value=None), \
                patch("httpx.Client") as MockClient:
            ci = MockClient.return_value.__enter__.return_value
            ci.get = MagicMock(return_value=ok)
            with patch("prsm.cli._agent_output") as out:
                daemon.daemon_status(output_format="json")
            data = out.call_args[0][0]
        assert data["running"] is False
        assert data["foreground_running"] is True
        assert data["foreground_api_port"] == 8000

    def test_daemon_pid_skips_foreground_probe(self):
        """When daemon mode is detected, foreground probe is skipped
        (avoids redundant network call)."""
        with patch.object(daemon, "_get_daemon_pid", return_value=12345), \
                patch.object(daemon, "_is_daemon_running", return_value=True), \
                patch.object(daemon, "_get_daemon_uptime", return_value="1h"), \
                patch("httpx.Client") as MockClient:
            with patch("prsm.cli._agent_output") as out:
                daemon.daemon_status(output_format="json")
            data = out.call_args[0][0]
        assert data["running"] is True
        # foreground probe NOT made (running short-circuits it)
        ci = MockClient.return_value.__enter__.return_value
        ci.get.assert_not_called()
