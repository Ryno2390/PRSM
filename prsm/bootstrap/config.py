"""
Bootstrap Server Configuration

Configuration management for the PRSM bootstrap server infrastructure.
Supports environment variables, config files, and secure defaults.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class BootstrapConfig:
    """
    Configuration for bootstrap server.
    
    Provides all configuration options needed to run a PRSM bootstrap server,
    including network settings, security options, and operational parameters.
    
    Configuration can be loaded from:
    - Environment variables (highest priority)
    - Configuration files
    - Default values
    """
    
    # === Network Configuration ===
    
    domain: str = "prsm-network.com"
    """Domain name for the bootstrap server."""
    
    host: str = "0.0.0.0"
    """Host address to bind the server to."""
    
    port: int = 8765
    """Port for WebSocket connections."""
    
    api_port: int = 8000
    """Port for HTTP API and health checks."""
    
    external_ip: Optional[str] = None
    """External IP address for NAT traversal. Auto-detected if not set."""
    
    # === SSL/TLS Configuration ===
    
    ssl_enabled: bool = True
    """Enable SSL/TLS for secure connections."""
    
    ssl_cert_path: str = "/etc/ssl/certs/prsm.crt"
    """Path to SSL certificate file."""
    
    ssl_key_path: str = "/etc/ssl/private/prsm.key"
    """Path to SSL private key file."""
    
    ssl_ca_path: Optional[str] = None
    """Path to CA certificate for client verification."""
    
    require_client_cert: bool = False
    """Require client certificates for mutual TLS."""
    
    # === Peer Management ===
    
    max_peers: int = 1000
    """Maximum number of peers to track."""
    
    max_connections_per_ip: int = 5
    """Maximum connections allowed from a single IP address."""
    
    peer_timeout: int = 300
    """Seconds before a peer is considered stale."""
    
    heartbeat_interval: int = 30
    """Seconds between heartbeat messages."""
    
    peer_list_size: int = 50
    """Number of peers to return in peer list responses."""
    
    # === Security Settings ===
    
    auth_required: bool = False
    """Require authentication for peer connections."""
    
    auth_secret: Optional[str] = None
    """Shared secret for peer authentication."""
    
    rate_limit_requests: int = 100
    """Maximum requests per minute per peer."""
    
    rate_limit_window: int = 60
    """Rate limit window in seconds."""
    
    banned_peers: List[str] = field(default_factory=list)
    """List of banned peer IDs."""
    
    banned_ips: List[str] = field(default_factory=list)
    """List of banned IP addresses."""
    
    # === Performance Settings ===
    
    worker_count: int = 4
    """Number of worker processes for handling connections."""
    
    connection_timeout: int = 30
    """Timeout for establishing connections."""
    
    message_timeout: int = 10
    """Timeout for receiving messages."""
    
    max_message_size: int = 10 * 1024 * 1024  # 10 MB
    """Maximum message size in bytes."""
    
    buffer_size: int = 64 * 1024  # 64 KB
    """Buffer size for network operations."""
    
    # === Persistence Settings ===
    
    persist_peers: bool = True
    """Persist peer list to disk."""
    
    peer_db_path: str = "/app/data/bootstrap_peers.db"
    """Path to peer database file."""
    
    peer_db_backup_interval: int = 300
    """Seconds between peer database backups."""
    
    # === Monitoring Settings ===
    
    metrics_enabled: bool = True
    """Enable Prometheus metrics export."""
    
    metrics_port: int = 9090
    """Port for Prometheus metrics endpoint."""
    
    health_check_interval: int = 60
    """Seconds between health checks."""
    
    log_level: str = "INFO"
    """Logging level (DEBUG, INFO, WARNING, ERROR)."""
    
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """Log message format."""
    
    # === Federation Settings ===
    
    federation_enabled: bool = True
    """Enable federation with other bootstrap servers."""
    
    federation_peers: List[str] = field(default_factory=list)
    """List of other bootstrap servers to federate with."""
    
    federation_sync_interval: int = 300
    """Seconds between federation sync operations."""
    
    # === Region Settings ===
    
    region: str = "us-east-1"
    """Region identifier for this bootstrap server."""
    
    regions: Dict[str, str] = field(default_factory=lambda: {
        "us-east-1": "bootstrap-us-east.prsm-network.com:8765",
        "us-west-1": "bootstrap-us-west.prsm-network.com:8765",
        "eu-west-1": "bootstrap-eu-west.prsm-network.com:8765",
        "ap-southeast-1": "bootstrap-ap-southeast.prsm-network.com:8765",
    })
    """Mapping of region names to bootstrap server addresses."""
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        self._validate_paths()
        self._validate_values()
        self._load_from_environment()
    
    def _validate_paths(self) -> None:
        """Validate that required paths exist or can be created."""
        if self.ssl_enabled:
            if not os.path.exists(self.ssl_cert_path):
                logger.warning(f"SSL certificate not found at {self.ssl_cert_path}")
            if not os.path.exists(self.ssl_key_path):
                logger.warning(f"SSL key not found at {self.ssl_key_path}")
        
        # Ensure peer db directory exists
        peer_db_dir = os.path.dirname(self.peer_db_path)
        if peer_db_dir and not os.path.exists(peer_db_dir):
            try:
                os.makedirs(peer_db_dir, exist_ok=True)
                logger.info(f"Created peer database directory: {peer_db_dir}")
            except OSError as e:
                logger.warning(f"Could not create peer database directory: {e}")
    
    def _validate_values(self) -> None:
        """Validate configuration values."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port number: {self.port}")
        
        if self.max_peers < 1:
            raise ValueError(f"max_peers must be at least 1: {self.max_peers}")
        
        if self.heartbeat_interval < 1:
            raise ValueError(f"heartbeat_interval must be at least 1: {self.heartbeat_interval}")
        
        if self.peer_timeout < self.heartbeat_interval:
            logger.warning(
                f"peer_timeout ({self.peer_timeout}) is less than heartbeat_interval "
                f"({self.heartbeat_interval}), this may cause issues"
            )
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # Network settings
        if os.environ.get("PRSM_DOMAIN"):
            self.domain = os.environ["PRSM_DOMAIN"]
        if os.environ.get("PRSM_BOOTSTRAP_HOST"):
            self.host = os.environ["PRSM_BOOTSTRAP_HOST"]
        if os.environ.get("PRSM_BOOTSTRAP_PORT"):
            self.port = int(os.environ["PRSM_BOOTSTRAP_PORT"])
        if os.environ.get("PRSM_API_PORT"):
            self.api_port = int(os.environ["PRSM_API_PORT"])
        if os.environ.get("PRSM_EXTERNAL_IP"):
            self.external_ip = os.environ["PRSM_EXTERNAL_IP"]
        
        # SSL settings
        if os.environ.get("PRSM_SSL_ENABLED"):
            self.ssl_enabled = os.environ["PRSM_SSL_ENABLED"].lower() in ("true", "1", "yes")
        if os.environ.get("PRSM_SSL_CERT_PATH"):
            self.ssl_cert_path = os.environ["PRSM_SSL_CERT_PATH"]
        if os.environ.get("PRSM_SSL_KEY_PATH"):
            self.ssl_key_path = os.environ["PRSM_SSL_KEY_PATH"]
        
        # Peer settings
        if os.environ.get("PRSM_MAX_PEERS"):
            self.max_peers = int(os.environ["PRSM_MAX_PEERS"])
        if os.environ.get("PRSM_PEER_TIMEOUT"):
            self.peer_timeout = int(os.environ["PRSM_PEER_TIMEOUT"])
        if os.environ.get("PRSM_HEARTBEAT_INTERVAL"):
            self.heartbeat_interval = int(os.environ["PRSM_HEARTBEAT_INTERVAL"])
        
        # Security settings
        if os.environ.get("PRSM_AUTH_SECRET"):
            self.auth_secret = os.environ["PRSM_AUTH_SECRET"]
            self.auth_required = True
        
        # Performance settings
        if os.environ.get("PRSM_WORKER_COUNT"):
            self.worker_count = int(os.environ["PRSM_WORKER_COUNT"])
        
        # Monitoring settings
        if os.environ.get("PRSM_LOG_LEVEL"):
            self.log_level = os.environ["PRSM_LOG_LEVEL"]
        if os.environ.get("PRSM_METRICS_ENABLED"):
            self.metrics_enabled = os.environ["PRSM_METRICS_ENABLED"].lower() in ("true", "1", "yes")
        
        # Region settings
        if os.environ.get("PRSM_REGION"):
            self.region = os.environ["PRSM_REGION"]
        
        # Federation settings
        if os.environ.get("PRSM_FEDERATION_PEERS"):
            self.federation_peers = os.environ["PRSM_FEDERATION_PEERS"].split(",")
    
    @property
    def websocket_url(self) -> str:
        """Get the WebSocket URL for this bootstrap server."""
        protocol = "wss" if self.ssl_enabled else "ws"
        host = self.domain if self.domain else self.host
        return f"{protocol}://{host}:{self.port}"
    
    @property
    def api_url(self) -> str:
        """Get the HTTP API URL for this bootstrap server."""
        protocol = "https" if self.ssl_enabled else "http"
        host = self.domain if self.domain else self.host
        return f"{protocol}://{host}:{self.api_port}"
    
    @property
    def external_endpoint(self) -> str:
        """Get the external endpoint address for peer connections."""
        if self.external_ip:
            return f"{self.external_ip}:{self.port}"
        return f"{self.domain}:{self.port}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "domain": self.domain,
            "host": self.host,
            "port": self.port,
            "api_port": self.api_port,
            "external_ip": self.external_ip,
            "ssl_enabled": self.ssl_enabled,
            "ssl_cert_path": self.ssl_cert_path,
            "ssl_key_path": self.ssl_key_path,
            "max_peers": self.max_peers,
            "max_connections_per_ip": self.max_connections_per_ip,
            "peer_timeout": self.peer_timeout,
            "heartbeat_interval": self.heartbeat_interval,
            "peer_list_size": self.peer_list_size,
            "auth_required": self.auth_required,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window": self.rate_limit_window,
            "worker_count": self.worker_count,
            "connection_timeout": self.connection_timeout,
            "message_timeout": self.message_timeout,
            "max_message_size": self.max_message_size,
            "persist_peers": self.persist_peers,
            "peer_db_path": self.peer_db_path,
            "metrics_enabled": self.metrics_enabled,
            "metrics_port": self.metrics_port,
            "health_check_interval": self.health_check_interval,
            "log_level": self.log_level,
            "federation_enabled": self.federation_enabled,
            "federation_peers": self.federation_peers,
            "region": self.region,
            "websocket_url": self.websocket_url,
            "api_url": self.api_url,
            "external_endpoint": self.external_endpoint,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BootstrapConfig":
        """Create configuration from dictionary."""
        return cls(
            domain=data.get("domain", "prsm-network.com"),
            host=data.get("host", "0.0.0.0"),
            port=data.get("port", 8765),
            api_port=data.get("api_port", 8000),
            external_ip=data.get("external_ip"),
            ssl_enabled=data.get("ssl_enabled", True),
            ssl_cert_path=data.get("ssl_cert_path", "/etc/ssl/certs/prsm.crt"),
            ssl_key_path=data.get("ssl_key_path", "/etc/ssl/private/prsm.key"),
            max_peers=data.get("max_peers", 1000),
            max_connections_per_ip=data.get("max_connections_per_ip", 5),
            peer_timeout=data.get("peer_timeout", 300),
            heartbeat_interval=data.get("heartbeat_interval", 30),
            peer_list_size=data.get("peer_list_size", 50),
            auth_required=data.get("auth_required", False),
            auth_secret=data.get("auth_secret"),
            rate_limit_requests=data.get("rate_limit_requests", 100),
            rate_limit_window=data.get("rate_limit_window", 60),
            worker_count=data.get("worker_count", 4),
            connection_timeout=data.get("connection_timeout", 30),
            message_timeout=data.get("message_timeout", 10),
            max_message_size=data.get("max_message_size", 10 * 1024 * 1024),
            persist_peers=data.get("persist_peers", True),
            peer_db_path=data.get("peer_db_path", "/app/data/bootstrap_peers.db"),
            metrics_enabled=data.get("metrics_enabled", True),
            metrics_port=data.get("metrics_port", 9090),
            health_check_interval=data.get("health_check_interval", 60),
            log_level=data.get("log_level", "INFO"),
            federation_enabled=data.get("federation_enabled", True),
            federation_peers=data.get("federation_peers", []),
            region=data.get("region", "us-east-1"),
        )


# Global configuration instance
_config: Optional[BootstrapConfig] = None


def get_bootstrap_config() -> BootstrapConfig:
    """
    Get the global bootstrap configuration.
    
    Returns a singleton configuration instance, creating it on first access.
    Configuration is loaded from environment variables and defaults.
    """
    global _config
    if _config is None:
        _config = BootstrapConfig()
    return _config


def set_bootstrap_config(config: BootstrapConfig) -> None:
    """Set the global bootstrap configuration."""
    global _config
    _config = config


def reset_bootstrap_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
