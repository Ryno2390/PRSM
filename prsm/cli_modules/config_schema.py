"""PRSM Configuration Schema — single source of truth for all settings.

Pydantic model that holds every setting from the setup wizard.
Saves to ~/.prsm/config.yaml, loads from same.
"""
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class NodeRole(str, Enum):
    CONTRIBUTOR = "contributor"
    CONSUMER = "consumer"
    FULL = "full"


class PRSMConfig(BaseModel):
    """Unified PRSM configuration — the single source of truth."""

    # Node identity
    display_name: str = "prsm-node"
    node_role: NodeRole = NodeRole.FULL

    # Resource allocation
    cpu_pct: int = Field(50, ge=10, le=90)
    memory_pct: int = Field(50, ge=10, le=90)
    gpu_pct: int = Field(80, ge=0, le=100)
    storage_gb: float = Field(10.0, ge=1.0)
    max_concurrent_jobs: int = Field(3, ge=1, le=20)
    upload_mbps_limit: float = Field(0.0, ge=0)  # 0 = unlimited
    active_hours_start: Optional[int] = None  # 0-23
    active_hours_end: Optional[int] = None
    active_days: list = Field(default_factory=list)

    # Network
    p2p_port: int = Field(9001, ge=1024, le=65535)
    api_port: int = Field(8000, ge=1024, le=65535)
    bootstrap_nodes: list = Field(default_factory=list)

    # API Keys (stored separately in .env but tracked here)
    has_openai_key: bool = False
    has_anthropic_key: bool = False
    has_openrouter_key: bool = False
    has_huggingface_token: bool = False

    # FTNS Wallet
    wallet_address: Optional[str] = None

    # AI Integration
    mcp_server_enabled: bool = True
    mcp_server_port: int = Field(9100, ge=1024, le=65535)

    # Meta
    setup_completed: bool = False
    setup_version: str = "1.0.0"

    @classmethod
    def config_path(cls) -> Path:
        return Path.home() / ".prsm" / "config.yaml"

    @classmethod
    def env_path(cls) -> Path:
        return Path.home() / ".prsm" / ".env"

    def save(self) -> Path:
        """Save config to ~/.prsm/config.yaml. Returns the path."""
        import yaml

        path = self.config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
        return path

    @classmethod
    def load(cls) -> "PRSMConfig":
        """Load config from ~/.prsm/config.yaml, or return defaults."""
        import yaml

        path = cls.config_path()
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()

    @classmethod
    def exists(cls) -> bool:
        """Check if a saved config exists."""
        return cls.config_path().exists()

    def reset(self) -> None:
        """Delete saved config."""
        path = self.config_path()
        if path.exists():
            path.unlink()
