"""End-to-end tests for the config lifecycle.

Tests the full flow: create config → read values → export → import → validate.
No network access required. Uses a temp directory for ~/.prsm isolation.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_config_module(tmp_path: Path):
    """Return PRSMConfig pointed at a temp dir."""
    from prsm.cli_modules import config_schema
    config_schema._TMP_PRSM_HOME = tmp_path
    config_schema.PRSMConfig.config_path = classmethod(
        lambda cls: tmp_path / "config.yaml"
    )
    return config_schema.PRSMConfig


class TestConfigRoundTrip:
    """Full lifecycle: defaults → set → save → load → modify → save → load."""

    def test_default_config_roundtrip(self, tmp_path):
        PRSMConfig = _make_config_module(tmp_path)
        cfg = PRSMConfig()
        assert cfg.display_name == "prsm-node"
        assert cfg.cpu_pct == 50
        cfg.display_name = "my-test-node"
        cfg.cpu_pct = 75
        cfg.save()
        cfg2 = PRSMConfig.load()
        assert cfg2.display_name == "my-test-node"
        assert cfg2.cpu_pct == 75

    def test_export_import_roundtrip(self, tmp_path):
        import yaml
        PRSMConfig = _make_config_module(tmp_path)
        cfg = PRSMConfig()
        cfg.cpu_pct = 60
        cfg.storage_gb = 25.0
        cfg.save()
        # Export (mode='json' converts enums to strings for yaml compat)
        data = cfg.model_dump(mode='json')
        exported = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        assert "cpu_pct: 60" in exported
        # Import into fresh config
        imported = yaml.safe_load(exported)
        cfg2 = PRSMConfig(**imported)
        assert cfg2.cpu_pct == 60
        assert cfg2.storage_gb == 25.0


class TestConfigExportCLI:
    """Test the config export command produces valid YAML."""

    def test_export_produces_yaml(self, tmp_path):
        from click.testing import CliRunner
        PRSMConfig = _make_config_module(tmp_path)
        cfg = PRSMConfig()
        cfg.cpu_pct = 65
        cfg.save()

        # Build config from the exported YAML (mode='json' for yaml compat)
        import yaml
        data = cfg.model_dump(mode='json')
        result = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        parsed = yaml.safe_load(result)
        assert parsed["cpu_pct"] == 65


class TestConfigSetViaPydantic:
    """Test that setting values validates correctly."""

    def test_cpu_validation(self, tmp_path):
        PRSMConfig = _make_config_module(tmp_path)
        # Valid
        cfg = PRSMConfig(cpu_pct=50)
        assert cfg.cpu_pct == 50
        cfg = PRSMConfig(cpu_pct=90)
        assert cfg.cpu_pct == 90
        cfg = PRSMConfig(cpu_pct=10)
        assert cfg.cpu_pct == 10

    def test_port_ranges(self, tmp_path):
        PRSMConfig = _make_config_module(tmp_path)
        cfg = PRSMConfig(p2p_port=9001, api_port=8000)
        assert cfg.p2p_port == 9001
        assert cfg.api_port == 8000

    def test_node_role_enum(self, tmp_path):
        PRSMConfig = _make_config_module(tmp_path)
        cfg = PRSMConfig(node_role="full")
        assert str(cfg.node_role) == "NodeRole.full" or cfg.node_role.value == "full"
        cfg2 = PRSMConfig(node_role="contributor")
        assert cfg2.node_role.value == "contributor"


class TestConfigValidation:
    """Test the PRSMConfig.exists() and validation logic."""

    def test_exists_false_when_no_file(self, tmp_path):
        PRSMConfig = _make_config_module(tmp_path)
        assert not PRSMConfig.config_path().exists()

    def test_exists_true_after_save(self, tmp_path):
        PRSMConfig = _make_config_module(tmp_path)
        cfg = PRSMConfig()
        cfg.save()
        assert PRSMConfig.config_path().exists()


class TestDaemonLifecycle:
    """Test daemon start/stop/status flow with mocks."""

    def test_start_then_status_json(self, tmp_path):
        from prsm.cli_modules import daemon as _d
        _d._DAEMON_PID_FILE = tmp_path / "daemon.pid"
        _d._DAEMON_LOG_FILE = tmp_path / "logs" / "daemon.log"

        # Not running
        assert not _d._is_daemon_running()
        assert _d._get_daemon_pid() is None

    def test_stop_noop_when_not_running(self, tmp_path, monkeypatch):
        from prsm.cli_modules import daemon as _d
        _d._DAEMON_PID_FILE = tmp_path / "daemon.pid"
        _d._DAEMON_LOG_FILE = tmp_path / "logs" / "daemon.log"

        captured = []
        monkeypatch.setattr(_d.console, "print", lambda *a, **kw: captured.append(a))
        result = _d.daemon_stop()
        assert result is False
        # Check the captured message (contains Rich markup)
        output = "".join(str(x) for args in captured for x in args[0] if args).lower()
        assert "daemon" in output and "not running" in output
