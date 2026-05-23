"""Sprint 757 — `prsm node schedule` CLI.

Read-only inspection of the operator's active-window schedule.
Reports:
- Configured window (or "always-active" if env unset)
- Whether we're currently inside the window
- Current local time in the schedule's timezone

Read-only. Operators change the schedule by editing their
systemd unit's `Environment=PRSM_ACTIVE_HOURS=...` line + restart.

Pin tests verify all three CLI states (unset / set + active /
set + inactive) plus the JSON output for scripting.
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

from click.testing import CliRunner


def setup_function():
    from prsm.node.schedule import reset_cache_for_testing
    reset_cache_for_testing()
    os.environ.pop("PRSM_ACTIVE_HOURS", None)
    os.environ.pop("PRSM_ACTIVE_TIMEZONE", None)


def teardown_function():
    from prsm.node.schedule import reset_cache_for_testing
    os.environ.pop("PRSM_ACTIVE_HOURS", None)
    os.environ.pop("PRSM_ACTIVE_TIMEZONE", None)
    reset_cache_for_testing()


def _invoke_schedule(args=None):
    """Invoke `prsm node schedule [args]` and return CliRunner Result."""
    from prsm.cli import node as _node_group
    runner = CliRunner()
    return runner.invoke(_node_group, ["schedule"] + (args or []))


def test_schedule_command_registered():
    """`prsm node schedule` is a registered Click command."""
    from prsm.cli import node as _node_group
    cmd_names = [c.name for c in _node_group.commands.values()]
    assert "schedule" in cmd_names


def test_schedule_text_unset_shows_always_active():
    """Env unset → 'always-active' status + helpful instruction
    pointing operator at PRSM_ACTIVE_HOURS."""
    result = _invoke_schedule(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "always-active" in result.output
    assert "PRSM_ACTIVE_HOURS" in result.output


def test_schedule_json_unset():
    """JSON output for unset env: configured=False, always-active,
    is_currently_active=True."""
    result = _invoke_schedule(["--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["configured"] is False
    assert data["mode"] == "always-active"
    assert data["is_currently_active"] is True


def test_schedule_text_active_window_inside():
    """Env set + currently inside window → ACTIVE status + green."""
    os.environ["PRSM_ACTIVE_HOURS"] = "00:00-23:59"  # ~always-active
    result = _invoke_schedule(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "ACTIVE" in result.output
    assert "00:00-23:59 UTC" in result.output


def test_schedule_json_active_window_inside():
    """JSON output for active window: full structured payload."""
    os.environ["PRSM_ACTIVE_HOURS"] = "00:00-23:59"
    result = _invoke_schedule(["--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["configured"] is True
    assert data["window"] == "00:00-23:59 UTC"
    assert data["start"] == "00:00"
    assert data["end"] == "23:59"
    assert data["timezone"] == "UTC"
    assert "now_local" in data
    assert data["is_currently_active"] is True


def test_schedule_text_active_window_outside():
    """When outside the window, status is INACTIVE + helpful
    message about 503 + announce-skip behavior."""
    # Set a narrow window we're definitely outside of by using
    # patch on is_active. The window is parseable; the
    # is_active() check is patched to return False.
    os.environ["PRSM_ACTIVE_HOURS"] = "22:00-08:00"
    with patch(
        "prsm.node.schedule.ActiveWindow.is_active",
        return_value=False,
    ):
        result = _invoke_schedule(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "INACTIVE" in result.output
    assert "503" in result.output  # mentions the 503 behavior


def test_schedule_malformed_env_exits_nonzero():
    """Malformed PRSM_ACTIVE_HOURS → CLI exits with error code
    matching what the daemon would crash with at startup."""
    os.environ["PRSM_ACTIVE_HOURS"] = "not-a-time-spec"
    result = _invoke_schedule(["--format", "text"])
    assert result.exit_code != 0
    assert "Schedule config error" in result.output


def test_schedule_malformed_env_json_exits_nonzero():
    """Same error path produces machine-readable JSON for CI."""
    os.environ["PRSM_ACTIVE_HOURS"] = "not-a-time-spec"
    result = _invoke_schedule(["--format", "json"])
    assert result.exit_code != 0
    data = json.loads(result.output)
    assert data["configured"] is False
    assert "error" in data
