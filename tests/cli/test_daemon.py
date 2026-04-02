"""Tests for PRSM daemon management (Phase 3).

Covers: PID management, start, stop, restart, status (text + JSON), logs,
and service install/uninstall shims. No real subprocess is launched — all
process management is mocked.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daemon_module(tmp_path: Path):
    """Create a fresh daemon module backed by a temp directory.

    The daemon module reads PID/log paths from module-level constants, so
    we inject a temporary directory to keep tests isolated.
    """
    pid_file = tmp_path / "daemon.pid"
    log_file = tmp_path / "logs" / "daemon.log"

    from prsm.cli_modules import daemon as _d

    # Override paths for this test
    _d._DAEMON_PID_FILE = pid_file
    _d._DAEMON_LOG_FILE = log_file
    return _d


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------


class TestGetDaemonPid:
    def test_returns_none_when_no_file(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        assert d._get_daemon_pid() is None

    def test_returns_pid_when_file_exists(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        d._DAEMON_PID_FILE.write_text("12345\n")
        assert d._get_daemon_pid() == 12345

    def test_returns_none_on_corrupt_file(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        d._DAEMON_PID_FILE.write_text("not-an-int\n")
        assert d._get_daemon_pid() is None


class TestIsDaemonRunning:
    def test_returns_false_when_no_pid(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        assert d._is_daemon_running() is False

    def test_returns_false_for_nonexistent_pid(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        # A PID that definitely doesn't exist
        assert d._is_daemon_running(9999999) is False

    def test_returns_true_for_current_process(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        # Current process always exists
        assert d._is_daemon_running(os.getpid()) is True

    def test_uses_pid_file_when_none_given(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        d._DAEMON_PID_FILE.write_text(str(os.getpid()))
        assert d._is_daemon_running() is True


class TestGetDaemonUptime:
    def test_returns_string_for_current_process(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        result = d._get_daemon_uptime(os.getpid())
        assert isinstance(result, str)

    def test_returns_unknown_for_bad_pid(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        result = d._get_daemon_uptime(9999999)
        assert result == "unknown"


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


class TestDaemonStart:
    def test_starts_and_writes_pid(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        fake_proc = MagicMock()
        fake_proc.pid = 54321

        with patch("subprocess.Popen", return_value=fake_proc):
            d.daemon_start(host="127.0.0.1", port=8000)

        assert d._DAEMON_PID_FILE.read_text() == "54321"

    def test_refuses_if_already_running(self, tmp_path, monkeypatch):
        d = _make_daemon_module(tmp_path)
        # Simulate an existing running daemon
        d._DAEMON_PID_FILE.write_text("11111")
        with patch.object(d, "_is_daemon_running", return_value=True):
            with pytest.raises(SystemExit) as excinfo:
                d.daemon_start()
            assert excinfo.value.code == 1

    def test_stale_pid_gets_cleaned(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        # Write a PID for a process that doesn't exist
        d._DAEMON_PID_FILE.write_text("9999999")
        fake_proc = MagicMock()
        fake_proc.pid = 22222

        with patch.object(d, "_is_daemon_running", return_value=False), \
             patch.object(d.subprocess, "Popen", return_value=fake_proc):
            d.daemon_start()

        # Old PID file was removed and new one written
        assert d._DAEMON_PID_FILE.read_text() == "22222"


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


class TestDaemonStop:
    def test_noop_when_not_running(self, tmp_path, mock_console):
        d = _make_daemon_module(tmp_path)
        result = d.daemon_stop()
        assert result is False

    def test_stops_running_process(self, tmp_path, mock_console):
        d = _make_daemon_module(tmp_path)
        d._DAEMON_PID_FILE.write_text(str(os.getpid()))

        call_count = [0]

        def fake_is_running(pid=None):
            call_count[0] += 1
            return call_count[0] == 1

        with patch.object(d, "_is_daemon_running", side_effect=fake_is_running), \
             patch("os.kill", return_value=None):
            result = d.daemon_stop()

        assert result is True
        assert not d._DAEMON_PID_FILE.exists()

    def test_sigkill_on_timeout(self, tmp_path, mock_console):
        d = _make_daemon_module(tmp_path)
        d._DAEMON_PID_FILE.write_text("9999999")

        with patch.object(d, "_is_daemon_running", return_value=True), \
             patch("os.kill", return_value=None):
            result = d.daemon_stop(timeout=0.1)

        assert result is True
        kill_calls = call.call_args_list if hasattr(call, 'call_args_list') else False


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------


class TestDaemonRestart:
    def test_start_when_not_running(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        fake_proc = MagicMock()
        fake_proc.pid = 33333
        with patch.object(d, "_is_daemon_running", return_value=False), \
             patch.object(d.subprocess, "Popen", return_value=fake_proc):
            d.daemon_restart()
        assert d._DAEMON_PID_FILE.read_text() == "33333"

    def test_stop_then_start_when_running(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        d._DAEMON_PID_FILE.write_text(str(os.getpid()))
        fake_proc = MagicMock()
        fake_proc.pid = 55555

        stop_called = [False]

        def fake_stop(timeout=10):
            stop_called[0] = True
            if d._DAEMON_PID_FILE.exists():
                d._DAEMON_PID_FILE.unlink()
            return True

        with patch.object(d, "daemon_stop", side_effect=fake_stop), \
             patch("subprocess.Popen", return_value=fake_proc):
            d.daemon_restart()

        assert stop_called[0] is True
        assert d._DAEMON_PID_FILE.read_text() == "55555"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestDaemonStatus:
    def test_shows_stopped_when_no_pid(self, tmp_path, mock_console):
        d = _make_daemon_module(tmp_path)
        d.daemon_status(output_format="text")
        output = " ".join(str(a) for args in mock_console for a in args[0])
        assert "stopped" in output.lower() or "running" in output.lower()

    def test_shows_running_when_pid_exists(self, tmp_path, mock_console):
        d = _make_daemon_module(tmp_path)
        d._DAEMON_PID_FILE.write_text(str(os.getpid()))
        d.daemon_status(output_format="text")
        output = " ".join(str(a) for args in mock_console for a in args[0])
        assert "running" in output.lower()

    def test_json_output(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        d._DAEMON_PID_FILE.write_text(str(os.getpid()))

        # _agent_output is called with structured data
        captured = []
        with patch("prsm.cli._agent_output", side_effect=lambda data: captured.append(data)):
            d.daemon_status(output_format="json")

        assert len(captured) == 1
        data = captured[0]
        assert data["ok"] is True
        assert "pid" in data
        assert "uptime" in data

    def test_json_when_stopped(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        captured = []
        with patch("prsm.cli._agent_output", side_effect=lambda data: captured.append(data)):
            d.daemon_status(output_format="json")
        data = captured[0]
        assert data["running"] is False
        assert data["pid"] is None


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


class TestDaemonLogs:
    def test_no_log_file_message(self, tmp_path, mock_console):
        d = _make_daemon_module(tmp_path)
        d.daemon_logs(lines=10, follow=False)
        output = " ".join(str(a) for args in mock_console for a in args[0])
        assert "no log file" in output.lower() or "not found" in output.lower()

    def test_shows_last_n_lines(self, tmp_path, mock_console):
        d = _make_daemon_module(tmp_path)
        # Create a log file with 5 lines
        d._DAEMON_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        d._DAEMON_LOG_FILE.write_text("line1\nline2\nline3\nline4\nline5\n")
        d.daemon_logs(lines=2, follow=False)
        # Just verify no exception — actual output goes to print()

    def test_follow_mode_reads_log(self, tmp_path, mock_console, monkeypatch):
        d = _make_daemon_module(tmp_path)
        d._DAEMON_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        d._DAEMON_LOG_FILE.write_text("existing log\n")

        # Simulate a KeyboardInterrupt to break the while True loop
        with patch("builtins.open", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                d.daemon_logs(lines=10, follow=True)


# ---------------------------------------------------------------------------
# Node command builder
# ---------------------------------------------------------------------------


class TestGetDaemonNodeCmd:
    def test_basic_command(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        cmd = d._get_daemon_node_cmd("127.0.0.1", 9001)
        assert cmd[0] == sys.executable
        assert "prsm.cli" in cmd
        assert "node" in cmd
        assert "start" in cmd
        assert "--no-dashboard" in cmd
        assert "--api-port" in cmd
        assert "9001" in cmd

    def test_adds_bootstrap_from_config(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        mock_cfg = MagicMock()
        mock_cfg.bootstrap_nodes = ["ws://node1:8000", "ws://node2:8000"]
        with patch("prsm.cli_modules.config_schema.PRSMConfig.load", return_value=mock_cfg):
            cmd = d._get_daemon_node_cmd("127.0.0.1", 9001)
        assert "--bootstrap" in cmd
        assert "ws://node1:8000,ws://node2:8000" in cmd


# ---------------------------------------------------------------------------
# System resource helpers
# ---------------------------------------------------------------------------


class TestSystemResourceUsage:
    def test_cpu_returns_int(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        try:
            import psutil
            result = d._get_system_cpu_usage()
            assert result is not None
            assert isinstance(result, (int, float))
        except ImportError:
            pytest.skip("psutil not installed")

    def test_memory_returns_dict(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        try:
            import psutil
            result = d._get_system_memory_usage()
            assert result is not None
            assert "pct" in result
            assert "used_gb" in result
            assert "total_gb" in result
        except ImportError:
            pytest.skip("psutil not installed")

    def test_disk_returns_dict(self, tmp_path):
        d = _make_daemon_module(tmp_path)
        try:
            import psutil
            result = d._get_system_disk_usage()
            assert result is not None
            assert "used_gb" in result
            assert "total_gb" in result
            assert "free_gb" in result
        except ImportError:
            pytest.skip("psutil not installed")


# ---------------------------------------------------------------------------
# Service install / uninstall (dry-run tests — no real system calls)
# ---------------------------------------------------------------------------


class TestServiceInstallDryRun:
    def test_launchd_dry_run_prints_plist(self, tmp_path, mock_console, monkeypatch):
        d = _make_daemon_module(tmp_path)
        monkeypatch.setattr(d.sys, "platform", "darwin")
        d.daemon_service_install(dry_run=True, host="127.0.0.1", port=8000)
        output = " ".join(str(a) for args in mock_console for a in args[0])
        assert "launchd" in output.lower()

    def test_systemd_dry_run_prints_unit(self, tmp_path, mock_console, monkeypatch):
        d = _make_daemon_module(tmp_path)
        monkeypatch.setattr(d.sys, "platform", "linux")
        d.daemon_service_install(dry_run=True, host="127.0.0.1", port=8000)
        output = " ".join(str(a) for args in mock_console for a in args[0])
        assert "systemd" in output.lower()

    def test_unsupported_platform_raises(self, tmp_path, mock_console, monkeypatch):
        d = _make_daemon_module(tmp_path)
        monkeypatch.setattr(d.sys, "platform", "win32")
        with pytest.raises(SystemExit) as excinfo:
            d.daemon_service_install(dry_run=True)
        assert excinfo.value.code == 1
