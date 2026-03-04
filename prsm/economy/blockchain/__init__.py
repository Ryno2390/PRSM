"""
PRSM Blockchain Module
======================

Provides blockchain integration for FTNS token system including:
- Smart contract deployment
- Contract management and interaction
- Cross-chain bridge functionality
- Network configuration

Components:
- networks: Network configuration for supported blockchains
- deployment: Smart contract deployment infrastructure
- contract_manager: Contract interaction management
- ftns_bridge: Local to on-chain token bridge
"""

from .networks import (
    NetworkConfig,
    NetworkType,
    NetworkName,
    get_network_config,
    get_supported_networks,
    get_testnet_networks,
    get_mainnet_networks,
    get_network_by_chain_id,
    get_gas_config,
    get_verifier_config,
    DEFAULT_NETWORKS,
    CHAIN_ID_TO_NETWORK,
)

from .deployment import (
    DeploymentConfig,
    DeployedContract,
    DeploymentResult,
    DeploymentStatus,
    ContractType,
    ContractDeployer,
    deploy_ftns_to_network,
    FTNS_TOKEN_ABI,
    BRIDGE_ABI,
    ERC20_ABI,
)

from .contract_manager import (
    ContractManager,
    TransactionStatus,
    RoleType,
    TokenBalance,
    TransferEvent,
    TransactionResult,
)

from .ftns_bridge import (
    FTNSBridge,
    BridgeDirection,
    BridgeStatus,
    BridgeTransaction,
    BridgeLimits,
    BridgeStats,
    BridgeError,
    InsufficientBalanceError,
    BridgeLimitError,
    ValidationError,
)

__all__ = [
    # Networks
    "NetworkConfig",
    "NetworkType",
    "NetworkName",
    "get_network_config",
    "get_supported_networks",
    "get_testnet_networks",
    "get_mainnet_networks",
    "get_network_by_chain_id",
    "get_gas_config",
    "get_verifier_config",
    "DEFAULT_NETWORKS",
    "CHAIN_ID_TO_NETWORK",
    
    # Deployment
    "DeploymentConfig",
    "DeployedContract",
    "DeploymentResult",
    "DeploymentStatus",
    "ContractType",
    "ContractDeployer",
    "deploy_ftns_to_network",
    "FTNS_TOKEN_ABI",
    "BRIDGE_ABI",
    "ERC20_ABI",
    
    # Contract Manager
    "ContractManager",
    "TransactionStatus",
    "RoleType",
    "TokenBalance",
    "TransferEvent",
    "TransactionResult",
    
    # Bridge
    "FTNSBridge",
    "BridgeDirection",
    "BridgeStatus",
    "BridgeTransaction",
    "BridgeLimits",
    "BridgeStats",
    "BridgeError",
    "InsufficientBalanceError",
    "BridgeLimitError",
    "ValidationError",
]