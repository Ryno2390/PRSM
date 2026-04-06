"""
Node Configuration
==================

Dataclass-based configuration for a PRSM network node.
Defaults work out of the box for local development.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional
import json
import os


# Multi-region bootstrap server configuration
# Primary server (US region by default)
DEFAULT_BOOTSTRAP_NODES = [
    os.getenv("BOOTSTRAP_PRIMARY", "wss://bootstrap1.prsm-network.com:8765"),
]

# Multi-region fallback bootstrap servers for high availability
# Nodes try servers in order: Primary (US) → EU → APAC
FALLBACK_BOOTSTRAP_NODES = [
    os.getenv("BOOTSTRAP_FALLBACK_EU", "wss://bootstrap-eu.prsm-network.com:8765"),
    os.getenv("BOOTSTRAP_FALLBACK_APAC", "wss://bootstrap-apac.prsm-network.com:8765"),
]


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

    # Ledger type: "dag" for DAG-based (IOTA-style) or "legacy" for linear
    ledger_type: str = "dag"

    # Network
    listen_host: str = "0.0.0.0"
    p2p_port: int = 9001
    api_port: int = 8000
    bootstrap_nodes: List[str] = field(default_factory=lambda: list(DEFAULT_BOOTSTRAP_NODES))
    bootstrap_connect_timeout: float = 5.0
    bootstrap_retry_attempts: int = 2
    bootstrap_fallback_enabled: bool = True
    bootstrap_fallback_nodes: List[str] = field(
        default_factory=lambda: list(FALLBACK_BOOTSTRAP_NODES)
    )
    bootstrap_validate_addresses: bool = True
    bootstrap_backoff_base: float = 1.0
    bootstrap_backoff_max: float = 8.0
    max_peers: int = 50

    # Compute behavior
    allow_self_compute: bool = True        # Execute own jobs when no peers (single-node mode)

    # Resources
    storage_gb: float = 10.0
    cpu_allocation_pct: int = 50       # % of CPU to offer for jobs
    memory_allocation_pct: int = 50    # % of RAM to offer

    # Compute limits
    max_concurrent_jobs: int = 3          # Parallel job slots
    gpu_allocation_pct: int = 80          # % of GPU VRAM to offer (if GPU detected)

    # Network/bandwidth
    upload_mbps_limit: float = 0.0        # 0 = unlimited; non-zero = cap in Mbps
    download_mbps_limit: float = 0.0      # 0 = unlimited

    # Scheduling
    active_hours_start: Optional[int] = None  # Hour 0-23 (None = always on)
    active_hours_end: Optional[int] = None    # Hour 0-23 (None = always on)
    active_days: List[int] = field(          # 0=Mon ... 6=Sun (empty = every day)
        default_factory=list
    )

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

    # Content Economy (Phase 4)
    min_replicas: int = 3
    royalty_model: str = "phase4"  # "phase4" or "legacy"

    # WASM Runtime (Ring 1)
    wasm_enabled: bool = True
    wasm_max_memory_bytes: int = 256 * 1024 * 1024  # 256 MB default sandbox
    wasm_max_execution_seconds: int = 30
    wasm_max_module_size: int = 5 * 1024 * 1024  # 5 MB

    # Discovery tuning
    target_peers: int = 8
    announce_interval: float = 60.0
    maintenance_interval: float = 30.0
    peer_stale_timeout: float = 600.0          # 10 minutes

    # Transport tuning
    nonce_window: float = 300.0                # 5 minutes
    ws_ping_interval: float = 20.0
    ws_ping_timeout: float = 10.0
    handshake_timeout: float = 10.0
    nonce_cleanup_interval: float = 60.0

    # Collaboration tuning
    task_timeout: float = 3600.0               # 1 hour
    review_timeout: float = 3600.0             # 1 hour
    query_timeout: float = 1800.0              # 30 minutes
    max_completed_records: int = 500
    collab_cleanup_interval: float = 60.0

    # Bid selection tuning
    bid_strategy: str = "best_score"       # "lowest_cost", "fastest", "best_score"
    bid_window_seconds: float = 30.0
    min_bids: int = 1

    # Content index tuning
    max_indexed_cids: int = 10000

    # Ledger sync tuning
    reconciliation_interval: float = 300.0     # 5 minutes

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
            "bootstrap_connect_timeout": self.bootstrap_connect_timeout,
            "bootstrap_retry_attempts": self.bootstrap_retry_attempts,
            "bootstrap_fallback_enabled": self.bootstrap_fallback_enabled,
            "bootstrap_fallback_nodes": self.bootstrap_fallback_nodes,
            "bootstrap_validate_addresses": self.bootstrap_validate_addresses,
            "bootstrap_backoff_base": self.bootstrap_backoff_base,
            "bootstrap_backoff_max": self.bootstrap_backoff_max,
            "max_peers": self.max_peers,
            "allow_self_compute": self.allow_self_compute,
            "storage_gb": self.storage_gb,
            "cpu_allocation_pct": self.cpu_allocation_pct,
            "memory_allocation_pct": self.memory_allocation_pct,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "gpu_allocation_pct": self.gpu_allocation_pct,
            "upload_mbps_limit": self.upload_mbps_limit,
            "download_mbps_limit": self.download_mbps_limit,
            "active_hours_start": self.active_hours_start,
            "active_hours_end": self.active_hours_end,
            "active_days": self.active_days,
            "gossip_fanout": self.gossip_fanout,
            "gossip_ttl": self.gossip_ttl,
            "heartbeat_interval": self.heartbeat_interval,
            "data_dir": self.data_dir,
            "ipfs_api_url": self.ipfs_api_url,
            "welcome_grant": self.welcome_grant,
            "target_peers": self.target_peers,
            "announce_interval": self.announce_interval,
            "maintenance_interval": self.maintenance_interval,
            "peer_stale_timeout": self.peer_stale_timeout,
            "nonce_window": self.nonce_window,
            "ws_ping_interval": self.ws_ping_interval,
            "ws_ping_timeout": self.ws_ping_timeout,
            "handshake_timeout": self.handshake_timeout,
            "nonce_cleanup_interval": self.nonce_cleanup_interval,
            "task_timeout": self.task_timeout,
            "review_timeout": self.review_timeout,
            "query_timeout": self.query_timeout,
            "max_completed_records": self.max_completed_records,
            "collab_cleanup_interval": self.collab_cleanup_interval,
            "bid_strategy": self.bid_strategy,
            "bid_window_seconds": self.bid_window_seconds,
            "min_bids": self.min_bids,
            "max_indexed_cids": self.max_indexed_cids,
            "reconciliation_interval": self.reconciliation_interval,
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


def is_active_now(config: NodeConfig) -> bool:
    """Check if the node should be active based on configured schedule.
    
    Returns True if:
    - No schedule configured (always on), OR
    - Current time is within active hours AND current day is in active_days
    
    Args:
        config: NodeConfig with active_hours_start, active_hours_end, active_days
        
    Returns:
        True if node should accept work, False otherwise
    """
    # Always on if no schedule configured
    if config.active_hours_start is None or config.active_hours_end is None:
        return True
    
    now = datetime.now()
    current_hour = now.hour
    current_day = now.weekday()  # 0=Monday, 6=Sunday
    
    # Check if today is an active day
    if config.active_days and current_day not in config.active_days:
        return False
    
    start, end = config.active_hours_start, config.active_hours_end
    
    # Handle wrap-around (e.g., 22:00 - 06:00)
    if start <= end:
        # Normal range (e.g., 09:00 - 17:00)
        return start <= current_hour < end
    else:
        # Wraps midnight (e.g., 22:00 - 06:00)
        return current_hour >= start or current_hour < end
