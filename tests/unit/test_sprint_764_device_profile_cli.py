"""Sprint 764 — `prsm node device-profile` CLI.

Surfaces the 4 consumer-device knobs (scheduling, bandwidth,
CPU, battery) at a glance + can suggest a "polite-neighbor"
preset operators copy-paste into their systemd unit.

Pin tests cover:
- Command registered in Click group
- 3 default-unset knobs render as "unset" / "default"
- All 4 set → all 4 show with values
- JSON output for scripting
- --suggest emits a valid systemd Environment= block
- Malformed PRSM_ACTIVE_HOURS surfaces the parser error
"""
from __future__ import annotations

import json
import os

from click.testing import CliRunner


CONSUMER_ENVS = [
    "PRSM_ACTIVE_HOURS",
    "PRSM_ACTIVE_TIMEZONE",
    "PRSM_ACTIVE_ONLY_ON_AC",
    "PRSM_NODE_NICE",
    "PRSM_STORAGE_UPLOAD_MBPS",
    "PRSM_STORAGE_DOWNLOAD_MBPS",
]


def setup_function():
    from prsm.node.schedule import reset_cache_for_testing
    for env in CONSUMER_ENVS:
        os.environ.pop(env, None)
    reset_cache_for_testing()


def teardown_function():
    from prsm.node.schedule import reset_cache_for_testing
    for env in CONSUMER_ENVS:
        os.environ.pop(env, None)
    reset_cache_for_testing()


def _invoke(args=None):
    from prsm.cli import node as _node_group
    return CliRunner().invoke(
        _node_group, ["device-profile"] + (args or []),
    )


def test_device_profile_command_registered():
    """Command exists in the Click group."""
    from prsm.cli import node as _node_group
    cmd_names = [c.name for c in _node_group.commands.values()]
    assert "device-profile" in cmd_names


def test_text_output_unset_renders_all_as_default():
    """No knobs set → 4 'unset/default' lines."""
    result = _invoke(["--format", "text"])
    assert result.exit_code == 0, result.output
    # All 4 knobs mentioned even when unset
    assert "Active hours" in result.output
    assert "Battery awareness" in result.output
    assert "CPU politeness" in result.output
    assert "Bandwidth caps" in result.output
    # All show "unset" / "default" markers
    assert "unset" in result.output or "default" in result.output


def test_text_output_all_set_renders_values():
    """All 4 knobs set → values show in cyan."""
    os.environ["PRSM_ACTIVE_HOURS"] = "22:00-08:00"
    os.environ["PRSM_ACTIVE_TIMEZONE"] = "America/New_York"
    os.environ["PRSM_ACTIVE_ONLY_ON_AC"] = "1"
    os.environ["PRSM_NODE_NICE"] = "10"
    os.environ["PRSM_STORAGE_UPLOAD_MBPS"] = "10"
    os.environ["PRSM_STORAGE_DOWNLOAD_MBPS"] = "100"
    result = _invoke(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "22:00-08:00" in result.output
    assert "America/New_York" in result.output
    assert "only-on-AC" in result.output
    assert "+10" in result.output or "nice +10" in result.output
    assert "10 Mbps" in result.output
    assert "100 Mbps" in result.output


def test_json_output_shape():
    """JSON output: 5 distinct knobs + tolerable when unset."""
    result = _invoke(["--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "active_hours" in data
    assert "active_only_on_ac" in data
    assert "node_nice" in data
    assert "storage_upload_mbps" in data
    assert "storage_download_mbps" in data


def test_suggest_preset_emits_systemd_block():
    """--suggest emits an Environment= block operators can paste."""
    result = _invoke(["--suggest"])
    assert result.exit_code == 0, result.output
    # All 4 env families present
    assert "Environment=PRSM_ACTIVE_HOURS=" in result.output
    assert "Environment=PRSM_ACTIVE_ONLY_ON_AC=" in result.output
    assert "Environment=PRSM_NODE_NICE=" in result.output
    assert "Environment=PRSM_STORAGE_UPLOAD_MBPS=" in result.output
    # Polite-neighbor framing
    assert "polite-neighbor" in result.output.lower()


def test_malformed_active_hours_surfaces_error():
    """If PRSM_ACTIVE_HOURS is malformed, the CLI shows the parser
    error so the operator can fix it before daemon-start."""
    os.environ["PRSM_ACTIVE_HOURS"] = "not-a-time"
    result = _invoke(["--format", "text"])
    # The CLI itself shouldn't crash; should report the config error
    assert result.exit_code == 0
    assert "config error" in result.output.lower() or (
        "error" in result.output.lower()
    )


def test_json_includes_error_field_when_malformed():
    os.environ["PRSM_ACTIVE_HOURS"] = "not-a-time"
    result = _invoke(["--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data.get("active_hours_error") is not None
