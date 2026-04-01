"""Tests for prsm.cli_modules.config_schema — PRSMConfig model and NodeRole enum."""
import pytest
import yaml
from pathlib import Path

from prsm.cli_modules.config_schema import PRSMConfig, NodeRole


class TestNodeRole:
    def test_enum_values(self):
        assert NodeRole.CONTRIBUTOR.value == "contributor"
        assert NodeRole.CONSUMER.value == "consumer"
        assert NodeRole.FULL.value == "full"

    def test_enum_from_string(self):
        assert NodeRole("contributor") == NodeRole.CONTRIBUTOR
        assert NodeRole("full") == NodeRole.FULL

    def test_enum_is_str(self):
        # NodeRole inherits from str
        assert isinstance(NodeRole.FULL, str)


class TestPRSMConfigDefaults:
    def test_default_creation(self):
        cfg = PRSMConfig()
        assert cfg.display_name == "prsm-node"
        assert cfg.node_role == NodeRole.FULL

    def test_default_resource_values(self):
        cfg = PRSMConfig()
        assert cfg.cpu_pct == 50
        assert cfg.memory_pct == 50
        assert cfg.gpu_pct == 80
        assert cfg.storage_gb == 10.0
        assert cfg.max_concurrent_jobs == 3

    def test_default_network_values(self):
        cfg = PRSMConfig()
        assert cfg.p2p_port == 9001
        assert cfg.api_port == 8000
        assert cfg.bootstrap_nodes == []

    def test_default_api_keys(self):
        cfg = PRSMConfig()
        assert cfg.has_openai_key is False
        assert cfg.has_anthropic_key is False
        assert cfg.has_huggingface_token is False

    def test_default_meta(self):
        cfg = PRSMConfig()
        assert cfg.setup_completed is False
        assert cfg.setup_version == "1.0.0"
        assert cfg.mcp_server_enabled is True
        assert cfg.mcp_server_port == 9100


class TestPRSMConfigCustom:
    def test_custom_values(self):
        cfg = PRSMConfig(
            display_name="my-node",
            node_role=NodeRole.CONTRIBUTOR,
            cpu_pct=70,
            p2p_port=9002,
        )
        assert cfg.display_name == "my-node"
        assert cfg.node_role == NodeRole.CONTRIBUTOR
        assert cfg.cpu_pct == 70
        assert cfg.p2p_port == 9002

    def test_model_dump(self):
        cfg = PRSMConfig()
        data = cfg.model_dump()
        assert isinstance(data, dict)
        assert data["display_name"] == "prsm-node"
        assert data["cpu_pct"] == 50
        assert "node_role" in data


class TestPRSMConfigSaveLoad:
    def test_save_creates_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(PRSMConfig, "config_path", lambda cls=None: config_file)

        cfg = PRSMConfig(display_name="test-node", cpu_pct=60)
        path = cfg.save()
        assert path == config_file
        assert config_file.exists()

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(PRSMConfig, "config_path", lambda cls=None: config_file)

        original = PRSMConfig(
            display_name="roundtrip-node",
            node_role=NodeRole.CONTRIBUTOR,
            cpu_pct=75,
            memory_pct=60,
            p2p_port=9999,
        )
        original.save()

        loaded = PRSMConfig.load()
        assert loaded.display_name == "roundtrip-node"
        assert loaded.cpu_pct == 75
        assert loaded.memory_pct == 60
        assert loaded.p2p_port == 9999

    def test_load_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "nonexistent" / "config.yaml"
        monkeypatch.setattr(PRSMConfig, "config_path", lambda cls=None: config_file)

        cfg = PRSMConfig.load()
        assert cfg.display_name == "prsm-node"

    def test_saved_yaml_is_valid(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(PRSMConfig, "config_path", lambda cls=None: config_file)

        PRSMConfig(display_name="yaml-test").save()
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["display_name"] == "yaml-test"
        assert isinstance(data["cpu_pct"], int)


class TestPRSMConfigExists:
    def test_exists_false_when_no_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "nope.yaml"
        monkeypatch.setattr(PRSMConfig, "config_path", lambda cls=None: config_file)
        assert PRSMConfig.exists() is False

    def test_exists_true_after_save(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(PRSMConfig, "config_path", lambda cls=None: config_file)
        PRSMConfig().save()
        assert PRSMConfig.exists() is True


class TestPRSMConfigReset:
    def test_reset_deletes_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        monkeypatch.setattr(PRSMConfig, "config_path", lambda cls=None: config_file)

        cfg = PRSMConfig()
        cfg.save()
        assert config_file.exists()

        cfg.reset()
        assert not config_file.exists()

    def test_reset_noop_when_no_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "nope.yaml"
        monkeypatch.setattr(PRSMConfig, "config_path", lambda cls=None: config_file)
        PRSMConfig().reset()  # should not raise
