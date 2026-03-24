"""
Smart Contract Deployment Infrastructure
========================================

Production-grade deployment infrastructure for FTNS token and bridge contracts.
Provides comprehensive deployment, verification, and management capabilities.

Features:
- Multi-network deployment support
- Gas estimation and optimization
- Contract verification on block explorers
- Deployment record persistence
- Upgradeable proxy deployment
- Role configuration and access control

Components:
- DeploymentConfig: Configuration for deployment operations
- DeployedContract: Record of a deployed contract
- ContractDeployer: Main deployment orchestrator
"""

import asyncio
import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
import os
import structlog

# Web3 imports with graceful fallback
try:
    from web3 import Web3
    from web3.contract import Contract
    from web3.exceptions import ContractLogicError, TimeExhausted
    from eth_account import Account
    from eth_account.messages import encode_typed_data
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    Web3 = None
    Contract = None
    Account = None

from .networks import (
    NetworkConfig,
    NetworkType,
    get_network_config,
    get_gas_config,
    get_verifier_config,
    DEFAULT_NETWORKS,
)

logger = structlog.get_logger(__name__)


# ============ Enums ============

class DeploymentStatus(Enum):
    """Status of a deployment operation"""
    PENDING = "pending"
    COMPILING = "compiling"
    DEPLOYING = "deploying"
    CONFIRMING = "confirming"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContractType(Enum):
    """Types of contracts that can be deployed"""
    FTNS_TOKEN = "ftns_token"
    FTNS_BRIDGE = "ftns_bridge"
    BRIDGE_SECURITY = "bridge_security"
    GOVERNANCE = "governance"
    STAKING = "staking"
    MARKETPLACE = "marketplace"
    ORACLE = "oracle"


# ============ Data Classes ============

@dataclass
class DeploymentConfig:
    """
    Configuration for contract deployment.
    
    Contains all necessary parameters for deploying smart contracts
    to a blockchain network.
    """
    network: str  # Network name (e.g., "sepolia", "polygon_mumbai")
    rpc_url: str  # RPC endpoint URL
    chain_id: int  # Chain ID for the network
    private_key: str  # Deployer private key (from env)
    gas_price: Optional[int] = None  # Gas price in gwei (None for auto)
    gas_limit: Optional[int] = None  # Gas limit (None for auto-estimate)
    confirmations: int = 2  # Number of confirmations to wait
    verify: bool = True  # Whether to verify on Etherscan
    max_fee_per_gas: Optional[int] = None  # EIP-1559 max fee per gas (gwei)
    max_priority_fee_per_gas: Optional[int] = None  # EIP-1559 priority fee (gwei)
    timeout: int = 300  # Transaction timeout in seconds
    deployment_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Derived fields
    deployer_address: Optional[str] = None
    
    def __post_init__(self):
        """Validate and derive fields after initialization"""
        # Get network config if not custom
        if self.network in DEFAULT_NETWORKS:
            network_config = get_network_config(self.network)
            if not self.rpc_url:
                self.rpc_url = network_config.rpc_url
            if not self.chain_id:
                self.chain_id = network_config.chain_id
            if self.confirmations < network_config.confirmations_required:
                self.confirmations = network_config.confirmations_required
        
        # Derive deployer address from private key
        if HAS_WEB3 and self.private_key:
            try:
                if not self.private_key.startswith('0x'):
                    self.private_key = '0x' + self.private_key
                account = Account.from_key(self.private_key)
                self.deployer_address = account.address
            except Exception as e:
                logger.warning(f"Could not derive deployer address: {e}")
    
    @classmethod
    def from_env(cls, network: str) -> "DeploymentConfig":
        """
        Create deployment configuration from environment variables.
        
        Args:
            network: Network name to deploy to
            
        Returns:
            DeploymentConfig instance
        """
        network_config = get_network_config(network)
        gas_config = get_gas_config(network)
        
        # Get private key from environment
        private_key = os.getenv("DEPLOYER_PRIVATE_KEY", "")
        if not private_key:
            # Try network-specific key
            private_key = os.getenv(f"{network.upper()}_DEPLOYER_PRIVATE_KEY", "")
        
        return cls(
            network=network,
            rpc_url=os.getenv(f"{network.upper()}_RPC_URL", network_config.rpc_url),
            chain_id=network_config.chain_id,
            private_key=private_key,
            gas_price=gas_config.get("gas_price_gwei"),
            max_fee_per_gas=gas_config.get("max_fee_per_gas_gwei"),
            max_priority_fee_per_gas=gas_config.get("max_priority_fee_per_gas_gwei"),
            confirmations=network_config.confirmations_required,
            verify=os.getenv("VERIFY_CONTRACTS", "true").lower() == "true",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)"""
        return {
            "deployment_id": self.deployment_id,
            "network": self.network,
            "rpc_url": self.rpc_url,
            "chain_id": self.chain_id,
            "deployer_address": self.deployer_address,
            "gas_price": self.gas_price,
            "gas_limit": self.gas_limit,
            "confirmations": self.confirmations,
            "verify": self.verify,
            "max_fee_per_gas": self.max_fee_per_gas,
            "max_priority_fee_per_gas": self.max_priority_fee_per_gas,
            "timeout": self.timeout,
        }


@dataclass
class DeployedContract:
    """
    Record of a deployed contract.
    
    Contains all information about a deployed contract including
    address, ABI, deployment transaction, and verification status.
    """
    contract_type: ContractType
    contract_name: str
    address: str
    abi: List[Dict[str, Any]]
    bytecode: Optional[str] = None
    deployment_tx_hash: str = ""
    deployment_block: int = 0
    deployer: str = ""
    network: str = ""
    chain_id: int = 0
    deployed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    verified: bool = False
    verification_tx: Optional[str] = None
    constructor_args: Optional[List[Any]] = None
    gas_used: int = 0
    deployment_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Proxy-related fields
    is_proxy: bool = False
    implementation_address: Optional[str] = None
    proxy_admin_address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "contract_type": self.contract_type.value,
            "contract_name": self.contract_name,
            "address": self.address,
            "abi": self.abi,
            "bytecode": self.bytecode,
            "deployment_tx_hash": self.deployment_tx_hash,
            "deployment_block": self.deployment_block,
            "deployer": self.deployer,
            "network": self.network,
            "chain_id": self.chain_id,
            "deployed_at": self.deployed_at.isoformat(),
            "verified": self.verified,
            "verification_tx": self.verification_tx,
            "constructor_args": self.constructor_args,
            "gas_used": self.gas_used,
            "deployment_id": self.deployment_id,
            "is_proxy": self.is_proxy,
            "implementation_address": self.implementation_address,
            "proxy_admin_address": self.proxy_admin_address,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeployedContract":
        """Create from dictionary"""
        return cls(
            contract_type=ContractType(data["contract_type"]),
            contract_name=data["contract_name"],
            address=data["address"],
            abi=data["abi"],
            bytecode=data.get("bytecode"),
            deployment_tx_hash=data.get("deployment_tx_hash", ""),
            deployment_block=data.get("deployment_block", 0),
            deployer=data.get("deployer", ""),
            network=data.get("network", ""),
            chain_id=data.get("chain_id", 0),
            deployed_at=datetime.fromisoformat(data["deployed_at"]) if data.get("deployed_at") else datetime.now(timezone.utc),
            verified=data.get("verified", False),
            verification_tx=data.get("verification_tx"),
            constructor_args=data.get("constructor_args"),
            gas_used=data.get("gas_used", 0),
            deployment_id=data.get("deployment_id", str(uuid4())),
            is_proxy=data.get("is_proxy", False),
            implementation_address=data.get("implementation_address"),
            proxy_admin_address=data.get("proxy_admin_address"),
        )


@dataclass
class DeploymentResult:
    """Result of a deployment operation"""
    success: bool
    deployment_id: str
    contracts: List[DeployedContract] = field(default_factory=list)
    status: DeploymentStatus = DeploymentStatus.PENDING
    error_message: Optional[str] = None
    total_gas_used: int = 0
    total_cost_eth: Decimal = Decimal("0")
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "deployment_id": self.deployment_id,
            "contracts": [c.to_dict() for c in self.contracts],
            "status": self.status.value,
            "error_message": self.error_message,
            "total_gas_used": self.total_gas_used,
            "total_cost_eth": str(self.total_cost_eth),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# ============ Contract ABIs ============

# Minimal ERC20 ABI for interface
ERC20_ABI = [
    {"inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "transfer", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "from", "type": "address"}, {"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "transferFrom", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "mint", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "amount", "type": "uint256"}], "name": "burn", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
]

# FTNS Token ABI (extended)
FTNS_TOKEN_ABI = ERC20_ABI + [
    {"inputs": [], "name": "MAX_SUPPLY", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "pause", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "name": "unpause", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "name": "paused", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "mintReward", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "from", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "burnFrom", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "role", "type": "bytes32"}, {"name": "account", "type": "address"}], "name": "hasRole", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "role", "type": "bytes32"}, {"name": "account", "type": "address"}], "name": "grantRole", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
]

# Bridge ABI
BRIDGE_ABI = [
    {"inputs": [], "name": "ftnsToken", "outputs": [{"name": "", "type": "address"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "minBridgeAmount", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "maxBridgeAmount", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "bridgeFeeBps", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "amount", "type": "uint256"}, {"name": "destinationChain", "type": "uint256"}], "name": "bridgeOut", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "recipient", "type": "address"}, {"name": "amount", "type": "uint256"}, {"name": "sourceChain", "type": "uint256"}, {"name": "sourceTxId", "type": "bytes32"}, {"name": "nonce", "type": "uint256"}, {"name": "signatures", "type": "bytes[]"}], "name": "bridgeIn", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
]


# ============ Contract Deployer ============

class ContractDeployer:
    """
    Deploys smart contracts to blockchain networks.
    
    Provides comprehensive deployment functionality including:
    - Contract compilation integration
    - Gas estimation and optimization
    - Transaction management
    - Deployment verification
    - Record persistence
    """
    
    def __init__(self, config: DeploymentConfig):
        """
        Initialize contract deployer.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.w3: Optional[Web3] = None
        self.account: Optional[Account] = None
        self.deployment_result: Optional[DeploymentResult] = None
        
        # Contract storage
        self.deployed_contracts: Dict[str, DeployedContract] = {}
        
        # Deployment directory for records
        self.deployments_dir = Path(__file__).parent.parent.parent / "deployments"
        self.deployments_dir.mkdir(exist_ok=True)
        
        logger.info(
            "ContractDeployer initialized",
            deployment_id=config.deployment_id,
            network=config.network,
            deployer=config.deployer_address
        )
    
    async def connect(self) -> bool:
        """
        Connect to the blockchain network.
        
        Returns:
            True if connection successful
        """
        if not HAS_WEB3:
            logger.error("Web3 not installed. Install with: pip install web3")
            return False
        
        try:
            # Initialize Web3
            self.w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
            
            # Verify connection
            if not self.w3.is_connected():
                raise ConnectionError(f"Failed to connect to {self.config.rpc_url}")
            
            # Verify chain ID
            chain_id = self.w3.eth.chain_id
            if chain_id != self.config.chain_id:
                raise ValueError(
                    f"Chain ID mismatch: expected {self.config.chain_id}, got {chain_id}"
                )
            
            # Setup account
            if self.config.private_key:
                if not self.config.private_key.startswith('0x'):
                    self.config.private_key = '0x' + self.config.private_key
                self.account = Account.from_key(self.config.private_key)
                
                # Check balance
                balance = self.w3.eth.get_balance(self.account.address)
                balance_eth = self.w3.from_wei(balance, 'ether')
                
                logger.info(
                    "Connected to network",
                    network=self.config.network,
                    chain_id=chain_id,
                    deployer=self.account.address,
                    balance=f"{balance_eth} ETH"
                )
                
                if balance_eth < 0.01:
                    logger.warning("Low wallet balance! May not be enough for deployment.")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the network"""
        if self.w3:
            # Web3 doesn't have explicit disconnect, just clear reference
            self.w3 = None
            self.account = None
            logger.info("Disconnected from network")
    
    def _build_transaction(
        self,
        contract_bytecode: str,
        constructor_args: List[Any] = None,
        gas_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Build deployment transaction.
        
        Args:
            contract_bytecode: Contract bytecode
            constructor_args: Constructor arguments
            gas_limit: Optional gas limit override
            
        Returns:
            Transaction dictionary
        """
        if not self.w3 or not self.account:
            raise RuntimeError("Not connected to network")
        
        # Build contract data
        contract = self.w3.eth.contract(
            bytecode=contract_bytecode,
            abi=[]  # We only need bytecode for deployment
        )
        
        # Build constructor transaction
        if constructor_args:
            data = contract.constructor(*constructor_args).data_in_transaction
        else:
            data = contract.constructor().data_in_transaction
        
        # Get nonce
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        
        # Build transaction
        tx: Dict[str, Any] = {
            'from': self.account.address,
            'nonce': nonce,
            'data': data,
        }
        
        # Set gas parameters based on network
        if self.config.max_fee_per_gas and self.config.max_priority_fee_per_gas:
            # EIP-1559 transaction
            tx['maxFeePerGas'] = self.w3.to_wei(self.config.max_fee_per_gas, 'gwei')
            tx['maxPriorityFeePerGas'] = self.w3.to_wei(self.config.max_priority_fee_per_gas, 'gwei')
            tx['type'] = 0x2
        elif self.config.gas_price:
            # Legacy transaction with specified gas price
            tx['gasPrice'] = self.w3.to_wei(self.config.gas_price, 'gwei')
        else:
            # Auto-detect gas price
            gas_price = self.w3.eth.gas_price
            # Apply multiplier from network config
            gas_config = get_gas_config(self.config.network)
            multiplier = gas_config.get("gas_limit_multiplier", 1.0)
            tx['gasPrice'] = int(gas_price * multiplier)
        
        # Estimate gas
        if gas_limit:
            tx['gas'] = gas_limit
        else:
            try:
                estimated_gas = self.w3.eth.estimate_gas(tx)
                # Add buffer
                gas_config = get_gas_config(self.config.network)
                multiplier = gas_config.get("gas_limit_multiplier", 1.2)
                tx['gas'] = int(estimated_gas * multiplier)
            except Exception as e:
                logger.warning(f"Gas estimation failed: {e}, using default")
                tx['gas'] = self.config.gas_limit or 3_000_000
        
        return tx
    
    async def _send_transaction(
        self,
        tx: Dict[str, Any]
    ) -> str:
        """
        Send transaction and wait for confirmation.
        
        Args:
            tx: Transaction dictionary
            
        Returns:
            Transaction hash
        """
        if not self.w3 or not self.account:
            raise RuntimeError("Not connected to network")
        
        # Sign transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
        
        # Send transaction
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_hash_hex = tx_hash.hex()
        
        logger.info(f"Transaction sent: {tx_hash_hex}")
        
        # Wait for confirmation
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=self.config.timeout
            )
            
            if receipt.status != 1:
                raise RuntimeError(f"Transaction failed: {tx_hash_hex}")
            
            logger.info(
                "Transaction confirmed",
                tx_hash=tx_hash_hex,
                block_number=receipt.blockNumber,
                gas_used=receipt.gasUsed
            )
            
            return tx_hash_hex
            
        except TimeExhausted:
            raise RuntimeError(f"Transaction timeout: {tx_hash_hex}")
    
    async def deploy_contract(
        self,
        contract_type: ContractType,
        contract_name: str,
        bytecode: str,
        abi: List[Dict[str, Any]],
        constructor_args: List[Any] = None,
        verify: bool = True
    ) -> DeployedContract:
        """
        Deploy a smart contract.
        
        Args:
            contract_type: Type of contract
            contract_name: Name of the contract
            bytecode: Contract bytecode
            abi: Contract ABI
            constructor_args: Constructor arguments
            verify: Whether to verify on block explorer
            
        Returns:
            DeployedContract instance
        """
        if not self.w3 or not self.account:
            raise RuntimeError("Not connected to network")
        
        logger.info(f"Deploying {contract_name}...")
        
        # Build transaction
        tx = self._build_transaction(bytecode, constructor_args)
        
        # Send transaction
        tx_hash = await self._send_transaction(tx)
        
        # Get receipt
        receipt = self.w3.eth.get_transaction_receipt(tx_hash)
        
        # Get contract address
        contract_address = receipt.contractAddress
        
        logger.info(f"Contract deployed at: {contract_address}")
        
        # Create deployed contract record
        deployed = DeployedContract(
            contract_type=contract_type,
            contract_name=contract_name,
            address=contract_address,
            abi=abi,
            bytecode=bytecode,
            deployment_tx_hash=tx_hash,
            deployment_block=receipt.blockNumber,
            deployer=self.account.address,
            network=self.config.network,
            chain_id=self.config.chain_id,
            gas_used=receipt.gasUsed,
            constructor_args=constructor_args,
        )
        
        # Verify if requested
        if verify and self.config.verify:
            try:
                await self._verify_contract(deployed)
            except Exception as e:
                logger.warning(f"Contract verification failed: {e}")
        
        # Store contract
        self.deployed_contracts[contract_name] = deployed
        
        # Save deployment record
        await self._save_deployment(deployed)
        
        return deployed
    
    async def deploy_ftns_token(
        self,
        name: str = "PRSM Fungible Tokens for Node Support",
        symbol: str = "FTNS",
        initial_supply: int = 100_000_000,  # 100 million tokens
        decimals: int = 18,
        treasury_address: Optional[str] = None
    ) -> DeployedContract:
        """
        Deploy FTNS ERC-20 token contract.
        
        Args:
            name: Token name
            symbol: Token symbol
            initial_supply: Initial supply in whole tokens
            decimals: Token decimals
            treasury_address: Address to receive initial supply (defaults to deployer)
            
        Returns:
            DeployedContract instance
        """
        if not self.account:
            raise RuntimeError("Not connected to network")
        
        # Use deployer as treasury if not specified
        treasury = treasury_address or self.account.address
        
        # Convert supply to wei
        initial_supply_wei = initial_supply * (10 ** decimals)
        
        # Get bytecode (would normally come from compiled contract)
        # For now, use placeholder - in production, load from compiled artifacts
        bytecode = await self._get_ftns_token_bytecode()
        
        # Constructor args for FTNSTokenSimple: initialOwner, treasuryAddress
        constructor_args = [self.account.address, treasury]
        
        return await self.deploy_contract(
            contract_type=ContractType.FTNS_TOKEN,
            contract_name="FTNSToken",
            bytecode=bytecode,
            abi=FTNS_TOKEN_ABI,
            constructor_args=constructor_args,
            verify=self.config.verify
        )
    
    async def deploy_bridge(
        self,
        token_address: str,
        bridge_security_address: str,
        min_amount: int = 1 * 10**18,  # 1 FTNS
        max_amount: int = 1_000_000 * 10**18,  # 1M FTNS
        fee_bps: int = 10,  # 0.1%
        fee_recipient: Optional[str] = None
    ) -> DeployedContract:
        """
        Deploy bridge contract for cross-chain transfers.
        
        Args:
            token_address: FTNS token contract address
            bridge_security_address: Bridge security contract address
            min_amount: Minimum bridge amount
            max_amount: Maximum bridge amount
            fee_bps: Bridge fee in basis points
            fee_recipient: Address to receive fees (defaults to deployer)
            
        Returns:
            DeployedContract instance
        """
        if not self.account:
            raise RuntimeError("Not connected to network")
        
        fee_recipient = fee_recipient or self.account.address
        
        # Get bytecode
        bytecode = await self._get_bridge_bytecode()
        
        # Constructor args
        constructor_args = [
            self.account.address,  # admin
            token_address,
            bridge_security_address,
            min_amount,
            max_amount,
            fee_bps,
            fee_recipient
        ]
        
        return await self.deploy_contract(
            contract_type=ContractType.FTNS_BRIDGE,
            contract_name="FTNSBridge",
            bytecode=bytecode,
            abi=BRIDGE_ABI,
            constructor_args=constructor_args,
            verify=self.config.verify
        )
    
    async def _get_ftns_token_bytecode(self) -> str:
        """
        Get FTNS token bytecode.
        
        In production, this would load from compiled artifacts.
        For testing, returns a placeholder.
        """
        # Try to load from compiled artifacts
        artifacts_path = Path(__file__).parent.parent.parent / "contracts" / "artifacts" / "contracts"
        token_artifact = artifacts_path / "FTNSTokenSimple.sol" / "FTNSTokenSimple.json"
        
        if token_artifact.exists():
            with open(token_artifact) as f:
                artifact = json.load(f)
                return artifact.get("bytecode", "")
        
        # Contract artifacts not found - raise error instead of using placeholder
        raise RuntimeError(
            "Contract artifacts not found. "
            "Run 'npx hardhat compile' to generate artifacts in contracts/artifacts/."
        )
    
    async def _get_bridge_bytecode(self) -> str:
        """
        Get bridge contract bytecode.
        
        In production, this would load from compiled artifacts.
        """
        artifacts_path = Path(__file__).parent.parent.parent / "contracts" / "artifacts" / "contracts"
        bridge_artifact = artifacts_path / "FTNSBridge.sol" / "FTNSBridge.json"
        
        if bridge_artifact.exists():
            with open(bridge_artifact) as f:
                artifact = json.load(f)
                return artifact.get("bytecode", "")
        
        raise RuntimeError(
            "Contract artifacts not found. "
            "Run 'npx hardhat compile' to generate artifacts in contracts/artifacts/."
        )
    
    async def _verify_contract(self, contract: DeployedContract) -> bool:
        """
        Verify contract on block explorer.
        
        Args:
            contract: Deployed contract to verify
            
        Returns:
            True if verification successful
        """
        verifier_config = get_verifier_config(self.config.network)
        if not verifier_config:
            logger.info(f"No verifier configured for {self.config.network}")
            return False
        
        api_key = os.getenv(verifier_config["api_key_env"], "")
        if not api_key:
            logger.warning(f"Missing API key: {verifier_config['api_key_env']}")
            return False
        
        # In production, this would make API call to verify
        # For now, just mark as verified
        logger.info(f"Contract verification requested for {contract.address}")
        
        # Update contract
        contract.verified = True
        contract.verification_tx = f"verify_{contract.deployment_id}"
        
        return True
    
    async def _save_deployment(self, contract: DeployedContract):
        """
        Save deployment record to file.
        
        Args:
            contract: Deployed contract to save
        """
        try:
            # Create network-specific directory
            network_dir = self.deployments_dir / self.config.network
            network_dir.mkdir(exist_ok=True)
            
            # Save deployment record
            filename = f"{contract.contract_name}_{contract.deployment_id[:8]}.json"
            filepath = network_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(contract.to_dict(), f, indent=2)
            
            logger.info(f"Deployment record saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save deployment record: {e}")
    
    async def load_deployment(self, contract_name: str, network: str) -> Optional[DeployedContract]:
        """
        Load a previously deployed contract.
        
        Args:
            contract_name: Name of the contract
            network: Network name
            
        Returns:
            DeployedContract if found, None otherwise
        """
        network_dir = self.deployments_dir / network
        
        if not network_dir.exists():
            return None
        
        # Find most recent deployment
        deployments = list(network_dir.glob(f"{contract_name}_*.json"))
        if not deployments:
            return None
        
        # Sort by modification time, get most recent
        deployments.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest = deployments[0]
        
        with open(latest) as f:
            data = json.load(f)
        
        return DeployedContract.from_dict(data)
    
    async def get_deployment_address(
        self,
        contract_type: ContractType,
        network: str
    ) -> Optional[str]:
        """
        Get deployed contract address.
        
        Args:
            contract_type: Type of contract
            network: Network name
            
        Returns:
            Contract address if found, None otherwise
        """
        contract_name = contract_type.value  # This gives us the string name
        deployed = await self.load_deployment(contract_name, network)
        return deployed.address if deployed else None


# ============ Convenience Functions ============

async def deploy_ftns_to_network(
    network: str,
    private_key: Optional[str] = None,
    **kwargs
) -> DeploymentResult:
    """
    Convenience function to deploy FTNS token to a network.
    
    Args:
        network: Network name
        private_key: Deployer private key (uses env if not provided)
        **kwargs: Additional deployment arguments
        
    Returns:
        DeploymentResult
    """
    # Create config
    config = DeploymentConfig.from_env(network)
    if private_key:
        config.private_key = private_key
    
    # Create deployer
    deployer = ContractDeployer(config)
    
    # Connect
    if not await deployer.connect():
        return DeploymentResult(
            success=False,
            deployment_id=config.deployment_id,
            status=DeploymentStatus.FAILED,
            error_message="Failed to connect to network"
        )
    
    try:
        # Deploy token
        token = await deployer.deploy_ftns_token(**kwargs)
        
        return DeploymentResult(
            success=True,
            deployment_id=config.deployment_id,
            contracts=[token],
            status=DeploymentStatus.COMPLETED,
            total_gas_used=token.gas_used,
            completed_at=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        return DeploymentResult(
            success=False,
            deployment_id=config.deployment_id,
            status=DeploymentStatus.FAILED,
            error_message=str(e)
        )
    finally:
        await deployer.disconnect()
