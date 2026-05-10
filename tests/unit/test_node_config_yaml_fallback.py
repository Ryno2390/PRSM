"""NodeConfig.load() falls back to config.yaml (sprint 134).

Pre-fix: NodeConfig.load() ONLY read node_config.json. After the
auto-migration in cli_modules/migration.py renames it to
node_config.json.bak and creates config.yaml, NodeConfig.load()
would silently return DEFAULTS — operator's wizard-configured
ports + bootstrap list never reached the runtime node.

Post-fix: NodeConfig.load() prefers node_config.json when present
(legacy compat), but falls back to reading config.yaml + mapping
PRSMConfig fields to NodeConfig fields.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from prsm.node.config import NodeConfig


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """Isolated HOME so the test doesn't read user's actual config."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "pathlib.Path.home", lambda: tmp_path,
    )
    (tmp_path / ".prsm").mkdir(exist_ok=True)
    return tmp_path


class TestLegacyJsonStillWorks:
    def test_node_config_json_takes_precedence(self, tmp_home):
        """Existing operators with node_config.json keep working."""
        json_data = {
            "display_name": "from-json",
            "p2p_port": 9999,
            "api_port": 8888,
            "bootstrap_nodes": ["json-bootstrap"],
            "roles": ["full"],
        }
        (tmp_home / ".prsm" / "node_config.json").write_text(
            json.dumps(json_data),
        )
        # Even if config.yaml exists, JSON wins
        yaml_data = {
            "display_name": "from-yaml",
            "p2p_port": 1111,
            "api_port": 2222,
            "bootstrap_nodes": ["yaml-bootstrap"],
        }
        (tmp_home / ".prsm" / "config.yaml").write_text(
            yaml.safe_dump(yaml_data),
        )

        cfg = NodeConfig.load()
        assert cfg.display_name == "from-json"
        assert cfg.p2p_port == 9999
        assert cfg.api_port == 8888
        assert cfg.bootstrap_nodes == ["json-bootstrap"]


class TestYamlFallback:
    def test_yaml_used_when_json_absent(self, tmp_home):
        """When migration renamed JSON → .bak, YAML should still work."""
        yaml_data = {
            "display_name": "post-migration",
            "p2p_port": 9011,
            "api_port": 8010,
            "bootstrap_nodes": [
                "/ip4/127.0.0.1/udp/9001/quic-v1/p2p/QmFakePeer",
            ],
        }
        (tmp_home / ".prsm" / "config.yaml").write_text(
            yaml.safe_dump(yaml_data),
        )

        cfg = NodeConfig.load()
        assert cfg.display_name == "post-migration"
        assert cfg.p2p_port == 9011
        assert cfg.api_port == 8010
        assert cfg.bootstrap_nodes == [
            "/ip4/127.0.0.1/udp/9001/quic-v1/p2p/QmFakePeer",
        ]

    def test_field_name_mapping(self, tmp_home):
        """PRSMConfig has cpu_pct; NodeConfig has cpu_allocation_pct.
        Mapping table must translate."""
        yaml_data = {
            "cpu_pct": 75,
            "memory_pct": 60,
            "gpu_pct": 50,
        }
        (tmp_home / ".prsm" / "config.yaml").write_text(
            yaml.safe_dump(yaml_data),
        )
        cfg = NodeConfig.load()
        assert cfg.cpu_allocation_pct == 75
        assert cfg.memory_allocation_pct == 60
        assert cfg.gpu_allocation_pct == 50

    def test_node_role_to_roles_list_mapping(self, tmp_home):
        """PRSMConfig: node_role: 'full' → NodeConfig: roles: [FULL]."""
        from prsm.node.config import NodeRole
        yaml_data = {"node_role": "full"}
        (tmp_home / ".prsm" / "config.yaml").write_text(
            yaml.safe_dump(yaml_data),
        )
        cfg = NodeConfig.load()
        assert cfg.roles == [NodeRole.FULL]


class TestNoConfigFiles:
    def test_returns_defaults_when_neither_exists(self, tmp_home):
        cfg = NodeConfig.load()
        assert cfg.display_name == "prsm-node"  # default
        assert cfg.p2p_port == 9001  # default


class TestUnknownFields:
    def test_yaml_with_unknown_fields_doesnt_crash(self, tmp_home):
        """PRSMConfig has fields like has_openai_key that don't
        exist on NodeConfig. Loader should ignore them, not raise."""
        yaml_data = {
            "p2p_port": 9011,
            "api_port": 8010,
            "has_openai_key": True,  # PRSMConfig-only field
            "wallet_address": "0xABC",  # PRSMConfig-only
            "setup_completed": True,  # PRSMConfig-only
            "setup_version": "1.0.0",  # PRSMConfig-only
            "node_role": "full",
            "mcp_server_enabled": True,  # PRSMConfig-only
            "mcp_server_port": 9100,  # PRSMConfig-only
        }
        (tmp_home / ".prsm" / "config.yaml").write_text(
            yaml.safe_dump(yaml_data),
        )
        cfg = NodeConfig.load()  # MUST NOT raise
        assert cfg.p2p_port == 9011
        assert cfg.api_port == 8010
