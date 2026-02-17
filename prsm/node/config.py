"""
Node Configuration
==================

Dataclass-based configuration for a PRSM network node.
Defaults work out of the box for local development.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional
import json


class NodeRole(str, Enum):
    """Operating mode for the node."""
    FULL = "full"           # Compute + storage + routing
    COMPUTE = "compute"     # Compute jobs only
    STORAGE = "storage"     # Storage contribution only


@dataclass
class NodeConfig:
    """Configuration for a PRSM network node.

    Sensible defaults allow a node to start with zero configuration.
    """
    # Identity
    display_name: str = "prsm-node"
    roles: List[NodeRole] = field(default_factory=lambda: [NodeRole.FULL])

    # Network
    listen_host: str = "0.0.0.0"
    p2p_port: int = 9001
    api_port: int = 8000
    bootstrap_nodes: List[str] = field(default_factory=list)
    max_peers: int = 50

    # Resources
    storage_gb: float = 10.0
    cpu_allocation_pct: int = 50       # % of CPU to offer for jobs
    memory_allocation_pct: int = 50    # % of RAM to offer

    # Gossip protocol
    gossip_fanout: int = 3
    gossip_ttl: int = 5
    heartbeat_interval: float = 30.0   # seconds

    # Storage paths
    data_dir: str = field(default_factory=lambda: str(Path.home() / ".prsm"))

    # IPFS
    ipfs_api_url: str = "http://127.0.0.1:5001"

    # FTNS
    welcome_grant: float = 100.0

    @property
    def identity_path(self) -> Path:
        return Path(self.data_dir) / "identity.json"

    @property
    def ledger_path(self) -> Path:
        return Path(self.data_dir) / "ledger.db"

    @property
    def config_path(self) -> Path:
        return Path(self.data_dir) / "node_config.json"

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        """Persist config to disk."""
        self.ensure_dirs()
        data = {
            "display_name": self.display_name,
            "roles": [r.value for r in self.roles],
            "listen_host": self.listen_host,
            "p2p_port": self.p2p_port,
            "api_port": self.api_port,
            "bootstrap_nodes": self.bootstrap_nodes,
            "max_peers": self.max_peers,
            "storage_gb": self.storage_gb,
            "cpu_allocation_pct": self.cpu_allocation_pct,
            "memory_allocation_pct": self.memory_allocation_pct,
            "gossip_fanout": self.gossip_fanout,
            "gossip_ttl": self.gossip_ttl,
            "heartbeat_interval": self.heartbeat_interval,
            "data_dir": self.data_dir,
            "ipfs_api_url": self.ipfs_api_url,
            "welcome_grant": self.welcome_grant,
        }
        self.config_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NodeConfig":
        """Load config from disk, falling back to defaults."""
        if path is None:
            path = Path.home() / ".prsm" / "node_config.json"
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        roles = [NodeRole(r) for r in data.pop("roles", ["full"])]
        return cls(roles=roles, **data)
