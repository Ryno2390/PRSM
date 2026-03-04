"""
Network Configuration for PRSM Blockchain Deployment
=====================================================

Defines supported blockchain networks and their configurations for
FTNS token deployment and bridge operations.

Supported Networks:
- localhost: Local development (Hardhat/Ganache)
- sepolia: Ethereum Sepolia testnet
- polygon_mumbai: Polygon Mumbai testnet
- polygon: Polygon mainnet
- mainnet: Ethereum mainnet (future)

Features:
- Network-specific RPC endpoints
- Chain IDs and explorer URLs
- Gas configuration defaults
- Faucet information for testnets
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum
import os
import structlog

logger = structlog.get_logger(__name__)


class NetworkType(Enum):
    """Types of blockchain networks"""
    LOCAL = "local"
    TESTNET = "testnet"
    MAINNET = "mainnet"


class NetworkName(Enum):
    """Supported network names"""
    LOCALHOST = "localhost"
    SEPOLIA = "sepolia"
    POLYGON_MUMBAI = "polygon_mumbai"
    POLYGON = "polygon"
    MAINNET = "mainnet"


@dataclass
class NetworkConfig:
    """Configuration for a specific blockchain network"""
    name: str
    chain_id: int
    rpc_url: str
    network_type: NetworkType
    explorer_url: Optional[str] = None
    explorer_api_url: Optional[str] = None
    faucet_url: Optional[str] = None
    native_currency: str = "ETH"
    native_currency_decimals: int = 18
    block_time_seconds: int = 12
    confirmations_required: int = 1
    gas_multiplier: float = 1.0
    is_evm_compatible: bool = True
    
    def get_explorer_address_url(self, address: str) -> Optional[str]:
        """Get URL to view address on block explorer"""
        if self.explorer_url:
            return f"{self.explorer_url}/address/{address}"
        return None
    
    def get_explorer_tx_url(self, tx_hash: str) -> Optional[str]:
        """Get URL to view transaction on block explorer"""
        if self.explorer_url:
            return f"{self.explorer_url}/tx/{tx_hash}"
        return None


# Default network configurations
DEFAULT_NETWORKS: Dict[str, NetworkConfig] = {
    "localhost": NetworkConfig(
        name="localhost",
        chain_id=31337,
        rpc_url="http://localhost:8545",
        network_type=NetworkType.LOCAL,
        explorer_url=None,
        faucet_url=None,
        native_currency="ETH",
        block_time_seconds=1,
        confirmations_required=1,
        gas_multiplier=1.0,
    ),
    "sepolia": NetworkConfig(
        name="sepolia",
        chain_id=11155111,
        rpc_url=os.getenv("SEPOLIA_RPC_URL", "https://rpc.sepolia.org"),
        network_type=NetworkType.TESTNET,
        explorer_url="https://sepolia.etherscan.io",
        explorer_api_url="https://api-sepolia.etherscan.io/api",
        faucet_url="https://sepoliafaucet.com/",
        native_currency="ETH",
        block_time_seconds=12,
        confirmations_required=2,
        gas_multiplier=1.1,
    ),
    "polygon_mumbai": NetworkConfig(
        name="polygon_mumbai",
        chain_id=80001,
        rpc_url=os.getenv("POLYGON_MUMBAI_RPC_URL", "https://rpc-mumbai.maticvigil.com"),
        network_type=NetworkType.TESTNET,
        explorer_url="https://mumbai.polygonscan.com",
        explorer_api_url="https://api-testnet.polygonscan.com/api",
        faucet_url="https://faucet.polygon.technology/",
        native_currency="MATIC",
        block_time_seconds=2,
        confirmations_required=3,
        gas_multiplier=1.1,
    ),
    "polygon": NetworkConfig(
        name="polygon",
        chain_id=137,
        rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
        network_type=NetworkType.MAINNET,
        explorer_url="https://polygonscan.com",
        explorer_api_url="https://api.polygonscan.com/api",
        faucet_url=None,
        native_currency="MATIC",
        block_time_seconds=2,
        confirmations_required=5,
        gas_multiplier=1.0,
    ),
    "mainnet": NetworkConfig(
        name="mainnet",
        chain_id=1,
        rpc_url=os.getenv("MAINNET_RPC_URL", "https://eth.llamarpc.com"),
        network_type=NetworkType.MAINNET,
        explorer_url="https://etherscan.io",
        explorer_api_url="https://api.etherscan.io/api",
        faucet_url=None,
        native_currency="ETH",
        block_time_seconds=12,
        confirmations_required=12,
        gas_multiplier=1.0,
    ),
}


def get_network_config(network_name: str) -> NetworkConfig:
    """
    Get network configuration by name.
    
    Args:
        network_name: Name of the network (e.g., "sepolia", "polygon_mumbai")
        
    Returns:
        NetworkConfig for the specified network
        
    Raises:
        ValueError: If network is not supported
    """
    if network_name not in DEFAULT_NETWORKS:
        supported = ", ".join(DEFAULT_NETWORKS.keys())
        raise ValueError(f"Unsupported network: {network_name}. Supported: {supported}")
    
    return DEFAULT_NETWORKS[network_name]


def get_supported_networks() -> Dict[str, NetworkConfig]:
    """Get all supported network configurations"""
    return DEFAULT_NETWORKS.copy()


def get_testnet_networks() -> Dict[str, NetworkConfig]:
    """Get only testnet network configurations"""
    return {
        name: config for name, config in DEFAULT_NETWORKS.items()
        if config.network_type == NetworkType.TESTNET
    }


def get_mainnet_networks() -> Dict[str, NetworkConfig]:
    """Get only mainnet network configurations"""
    return {
        name: config for name, config in DEFAULT_NETWORKS.items()
        if config.network_type == NetworkType.MAINNET
    }


# Chain ID to network name mapping
CHAIN_ID_TO_NETWORK: Dict[int, str] = {
    config.chain_id: name for name, config in DEFAULT_NETWORKS.items()
}


def get_network_by_chain_id(chain_id: int) -> Optional[NetworkConfig]:
    """
    Get network configuration by chain ID.
    
    Args:
        chain_id: The chain ID to look up
        
    Returns:
        NetworkConfig if found, None otherwise
    """
    network_name = CHAIN_ID_TO_NETWORK.get(chain_id)
    if network_name:
        return DEFAULT_NETWORKS[network_name]
    return None


# Network-specific gas configurations
GAS_CONFIGS: Dict[str, Dict[str, Any]] = {
    "localhost": {
        "gas_limit_multiplier": 1.0,
        "gas_price_gwei": 1,
        "max_fee_per_gas_gwei": None,
        "max_priority_fee_per_gas_gwei": None,
    },
    "sepolia": {
        "gas_limit_multiplier": 1.2,
        "gas_price_gwei": None,  # Use EIP-1559
        "max_fee_per_gas_gwei": 30,
        "max_priority_fee_per_gas_gwei": 2,
    },
    "polygon_mumbai": {
        "gas_limit_multiplier": 1.2,
        "gas_price_gwei": 30,  # Polygon uses legacy gas
        "max_fee_per_gas_gwei": None,
        "max_priority_fee_per_gas_gwei": None,
    },
    "polygon": {
        "gas_limit_multiplier": 1.1,
        "gas_price_gwei": 30,
        "max_fee_per_gas_gwei": None,
        "max_priority_fee_per_gas_gwei": None,
    },
    "mainnet": {
        "gas_limit_multiplier": 1.2,
        "gas_price_gwei": None,  # Use EIP-1559
        "max_fee_per_gas_gwei": 50,
        "max_priority_fee_per_gas_gwei": 2,
    },
}


def get_gas_config(network_name: str) -> Dict[str, Any]:
    """
    Get gas configuration for a network.
    
    Args:
        network_name: Name of the network
        
    Returns:
        Gas configuration dictionary
    """
    return GAS_CONFIGS.get(network_name, GAS_CONFIGS["localhost"])


# Contract verification configurations
VERIFIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sepolia": {
        "verifier": "etherscan",
        "api_url": "https://api-sepolia.etherscan.io/api",
        "api_key_env": "ETHERSCAN_API_KEY",
    },
    "polygon_mumbai": {
        "verifier": "polygonscan",
        "api_url": "https://api-testnet.polygonscan.com/api",
        "api_key_env": "POLYGONSCAN_API_KEY",
    },
    "polygon": {
        "verifier": "polygonscan",
        "api_url": "https://api.polygonscan.com/api",
        "api_key_env": "POLYGONSCAN_API_KEY",
    },
    "mainnet": {
        "verifier": "etherscan",
        "api_url": "https://api.etherscan.io/api",
        "api_key_env": "ETHERSCAN_API_KEY",
    },
}


def get_verifier_config(network_name: str) -> Optional[Dict[str, Any]]:
    """
    Get contract verifier configuration for a network.
    
    Args:
        network_name: Name of the network
        
    Returns:
        Verifier configuration if available, None otherwise
    """
    return VERIFIER_CONFIGS.get(network_name)
