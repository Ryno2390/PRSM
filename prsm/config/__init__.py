"""PRSM configuration module — per-network contract addresses + endpoints."""

from prsm.config.networks import (
    NETWORK_CONFIGS,
    NetworkConfig,
    get_network_config,
    DEFAULT_NETWORK,
)

__all__ = [
    "NETWORK_CONFIGS",
    "NetworkConfig",
    "get_network_config",
    "DEFAULT_NETWORK",
]
