"""End-to-end tests for the PRSM setup wizard."""
from pathlib import Path
from unittest.mock import patch

import pytest


FAKE_SYS = {
    "os": "Darwin", "os_version": "23.0.0", "arch": "arm64",
    "python": "3.14.0", "cpu_model": "Apple M1", "cpu_cores": 8,
    "ram_gb": 16.0, "gpu": "Apple Silicon (Metal)",
    "disk_total_gb": 500.0, "disk_free_gb": 300.0,
}


@pytest.fixture
def wizard_env(tmp_path, monkeypatch):
    """Redirect all wizard I/O to a temp dir."""
    from prsm.cli_modules import config_schema as _cs
    from prsm.cli_modules import setup_wizard as _sw

    config_file = tmp_path / "config.yaml"
    _cs.PRSMConfig.config_path = classmethod(lambda cls: config_file)
    env_file = tmp_path / ".env"
    monkeypatch.setattr(_sw, "_ENV_FILE", env_file, raising=False)
    # Patch _detect_system globally so steps get real info
    monkeypatch.setattr(_sw, "_detect_system", lambda: dict(FAKE_SYS))
    yield {
        "config_file": config_file,
        "config_schema": _cs,
        "setup_wizard": _sw,
    }


class TestMinimalWizard:
    """Run the wizard with minimal=True (no prompts)."""

    def test_minimal_saves_config(self, wizard_env, monkeypatch):
        w = wizard_env["setup_wizard"]
        # Patch all prompts that might fire before/during minimal flow
        monkeypatch.setattr("click.confirm", lambda *a, **k: True)
        w.run_setup_wizard(minimal=True, dry_run=False)
        assert wizard_env["config_file"].exists()
        cfg = wizard_env["config_schema"].PRSMConfig.load()
        assert cfg.setup_completed is True
        assert cfg.display_name

    def test_minimal_dry_run_does_not_save(self, wizard_env, monkeypatch):
        w = wizard_env["setup_wizard"]
        monkeypatch.setattr("click.confirm", lambda *a, **k: True)
        w.run_setup_wizard(minimal=True, dry_run=True)
        assert not wizard_env["config_file"].exists()


class TestWizardSteps:
    """Test individual step functions with mocked input."""

    def _cfg(self, **kw):
        from prsm.cli_modules.config_schema import PRSMConfig
        return PRSMConfig(**kw)

    def test_step_welcome_returns_dict(self, wizard_env, monkeypatch):
        monkeypatch.setattr("click.prompt", lambda *a, **k: k.get("default", "test"))
        result = wizard_env["setup_wizard"]._step_welcome(
            self._cfg(), FAKE_SYS, minimal=False)
        assert isinstance(result, dict)

    def test_step_role_full(self, wizard_env, monkeypatch):
        monkeypatch.setattr("click.prompt", lambda *a, **k: "full")
        cfg = self._cfg()
        wizard_env["setup_wizard"]._step_role(cfg, minimal=False)
        assert cfg.node_role.value == "full"

    def test_step_role_contributor(self, wizard_env, monkeypatch):
        monkeypatch.setattr("click.prompt", lambda *a, **k: "contributor")
        cfg = self._cfg()
        wizard_env["setup_wizard"]._step_role(cfg, minimal=False)
        assert cfg.node_role.value == "contributor"

    def test_step_resources_contributor(self, wizard_env, monkeypatch):
        monkeypatch.setattr("click.prompt", lambda *a, **k: k.get("default", 50))
        cfg = self._cfg(node_role="contributor")
        wizard_env["setup_wizard"]._step_resources(cfg, FAKE_SYS, minimal=False)
        assert 10 <= cfg.cpu_pct <= 90
        assert 10 <= cfg.memory_pct <= 90

    def test_step_network_defaults(self, wizard_env, monkeypatch):
        # Port 49321 should be free so no confirm prompt fires
        monkeypatch.setattr("click.prompt", lambda *a, **k: k.get("default", ""))
        cfg = self._cfg()
        monkeypatch.setattr("click.confirm", lambda *a, **k: True)
        wizard_env["setup_wizard"]._step_network(cfg, minimal=False)
        assert cfg.p2p_port == 9001
        assert cfg.api_port == 8000

    def test_step_review_returns_bool(self, wizard_env, monkeypatch):
        monkeypatch.setattr("click.confirm", lambda *a, **k: True)
        result = wizard_env["setup_wizard"]._step_review(
            self._cfg(), dry_run=False)
        assert isinstance(result, bool)
        assert result is True  # confirmed → saves

    def test_step_review_dry_run_is_true(self, wizard_env):
        result = wizard_env["setup_wizard"]._step_review(
            self._cfg(), dry_run=True)
        # dry_run returns True to indicate wizard flow completed
        assert result is True


class TestPortChecking:
    def test_unlikely_port_is_free(self, wizard_env):
        assert wizard_env["setup_wizard"]._check_port(61234) is True
