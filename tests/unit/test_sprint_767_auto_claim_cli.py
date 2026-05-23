"""Sprint 767 — `prsm node auto-claim` CLI.

Read-only inspection of the auto-claim worker config. Mirrors
sprint-757's `prsm node schedule` pattern: shows enable status +
config values; operators change via systemd Environment= +
restart.

Pin tests cover:
- Command registered
- Text + JSON output for enabled / disabled states
- Threshold + interval values surface correctly
- Disabled state shows actionable instruction
"""
from __future__ import annotations

import json
import os

from click.testing import CliRunner


def setup_function():
    os.environ.pop("PRSM_AUTO_CLAIM_THRESHOLD_FTNS", None)
    os.environ.pop("PRSM_AUTO_CLAIM_INTERVAL_S", None)


def teardown_function():
    os.environ.pop("PRSM_AUTO_CLAIM_THRESHOLD_FTNS", None)
    os.environ.pop("PRSM_AUTO_CLAIM_INTERVAL_S", None)


def _invoke(args=None):
    from prsm.cli import node as _node_group
    return CliRunner().invoke(
        _node_group, ["auto-claim"] + (args or []),
    )


def test_auto_claim_command_registered():
    """Command exists in the Click group."""
    from prsm.cli import node as _node_group
    cmd_names = [c.name for c in _node_group.commands.values()]
    assert "auto-claim" in cmd_names


def test_text_unset_shows_disabled():
    """Env unset → 'disabled' + actionable instruction."""
    result = _invoke(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "disabled" in result.output.lower()
    assert "PRSM_AUTO_CLAIM_THRESHOLD_FTNS" in result.output


def test_json_unset_shows_disabled():
    """JSON shape: enabled=False, threshold=0."""
    result = _invoke(["--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["enabled"] is False
    assert data["threshold_ftns"] == "0"


def test_text_enabled_shows_threshold_and_interval():
    """Env set → enabled + threshold + interval shown."""
    os.environ["PRSM_AUTO_CLAIM_THRESHOLD_FTNS"] = "100"
    os.environ["PRSM_AUTO_CLAIM_INTERVAL_S"] = "1800"
    result = _invoke(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "enabled" in result.output.lower()
    assert "100" in result.output
    assert "1800" in result.output


def test_json_enabled_payload():
    """JSON contains the full config."""
    os.environ["PRSM_AUTO_CLAIM_THRESHOLD_FTNS"] = "250"
    os.environ["PRSM_AUTO_CLAIM_INTERVAL_S"] = "7200"
    result = _invoke(["--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["enabled"] is True
    assert data["threshold_ftns"] == "250"
    assert data["interval_seconds"] == 7200.0


def test_zero_threshold_treated_as_disabled():
    """Explicit threshold=0 → still disabled (consistent with
    sprint-765 config semantics)."""
    os.environ["PRSM_AUTO_CLAIM_THRESHOLD_FTNS"] = "0"
    result = _invoke(["--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["enabled"] is False
