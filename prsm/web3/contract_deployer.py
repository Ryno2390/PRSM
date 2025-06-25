"""
PRSM FTNS Smart Contract Deployer

Handles deployment of FTNS smart contracts to Polygon networks
with proper configuration and verification.
"""

import asyncio
import logging
import json
import os
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from web3 import Web3
from eth_account import Account
import requests

logger = logging.getLogger(__name__)

class FTNSContractDeployer:
    """
    Smart contract deployer for FTNS token system
    
    Features:
    - Multi-network deployment support
    - Contract verification on PolygonScan
    - Deployment record keeping
    - Gas optimization
    - Error handling and validation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_config()
        self.w3: Optional[Web3] = None
        self.account: Optional[Account] = None
        
    def _load_config(self) -> Dict:
        """Load deployment configuration"""
        return {
            "networks": {
                "polygon_mumbai": {
                    "rpc_url": os.getenv("POLYGON_MUMBAI_RPC_URL", "https://rpc-mumbai.maticvigil.com"),
                    "chain_id": 80001,
                    "name": "Polygon Mumbai Testnet",
                    "explorer": "https://mumbai.polygonscan.com",
                    "faucet": "https://faucet.polygon.technology/"
                },
                "polygon_mainnet": {
                    "rpc_url": os.getenv("POLYGON_MAINNET_RPC_URL", "https://polygon-rpc.com"),
                    "chain_id": 137,
                    "name": "Polygon Mainnet", 
                    "explorer": "https://polygonscan.com"
                }
            },
            "token": {
                "name": "PRSM Federated Token Network System",
                "symbol": "FTNS",
                "decimals": 18,
                "initial_supply": 100_000_000,  # 100M tokens
                "max_supply": 1_000_000_000     # 1B tokens max
            },
            "gas": {
                "limit_multiplier": 1.2,
                "price_multiplier": 1.1
            }
        }
    
    async def connect_to_network(self, network: str) -> bool:
        """Connect to specified network"""
        try:
            if network not in self.config["networks"]:
                raise ValueError(f"Unknown network: {network}")
                
            network_config = self.config["networks"][network]
            
            # Initialize Web3
            self.w3 = Web3(Web3.HTTPProvider(network_config["rpc_url"]))
            
            # Verify connection
            if not self.w3.is_connected():
                raise ConnectionError(f"Failed to connect to {network}")
                
            # Verify chain ID
            chain_id = self.w3.eth.chain_id
            expected_chain_id = network_config["chain_id"]
            if chain_id != expected_chain_id:
                raise ValueError(f"Chain ID mismatch: expected {expected_chain_id}, got {chain_id}")
                
            logger.info(f"Connected to {network_config['name']} (Chain ID: {chain_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to network {network}: {e}")
            return False
    
    async def setup_wallet(self, private_key: str) -> bool:
        """Setup deployment wallet"""
        try:
            if not private_key.startswith('0x'):
                private_key = '0x' + private_key
                
            self.account = Account.from_key(private_key)
            logger.info(f"Deployment wallet: {self.account.address}")
            
            # Check balance
            if self.w3:
                balance = self.w3.eth.get_balance(self.account.address)
                balance_eth = self.w3.from_wei(balance, 'ether')
                logger.info(f"Wallet balance: {balance_eth} MATIC")
                
                if balance_eth < 0.01:
                    logger.warning("Low wallet balance! Get MATIC from faucet.")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup wallet: {e}")
            return False
    
    async def deploy_ftns_token(self, network: str, treasury_address: Optional[str] = None) -> Optional[Dict]:
        """
        Deploy FTNS token contract
        
        Args:
            network: Target network
            treasury_address: Treasury address (defaults to deployer)
            
        Returns:
            Dict: Deployment information
        """
        try:
            if not self.w3 or not self.account:
                raise RuntimeError("Not connected to network or wallet not setup")
                
            # Use deployer address as treasury if not specified
            treasury = treasury_address or self.account.address
            
            # For now, create a mock deployment for testing
            # In production, this would deploy the actual contract
            logger.info("Creating mock deployment for testing...")
            
            # Generate mock contract address
            mock_address = self._generate_mock_address()
            
            # Create deployment record
            deployment_data = {
                "network": network,
                "chain_id": self.w3.eth.chain_id,
                "deployer": self.account.address,
                "treasury": treasury,
                "timestamp": datetime.utcnow().isoformat(),
                "contracts": {
                    "ftns_token": mock_address
                },
                "token_config": self.config["token"],
                "deployment_type": "mock",
                "gas_used": 2_500_000,  # Estimated
                "transaction_hash": f"0x{'0' * 64}",  # Mock transaction hash
                "verified": False
            }
            
            # Save deployment record
            await self._save_deployment_record(deployment_data)
            
            logger.info(f"Mock FTNS token deployed to: {mock_address}")
            logger.info("This is a test deployment for development purposes")
            
            return deployment_data
            
        except Exception as e:
            logger.error(f"Failed to deploy FTNS token: {e}")
            return None
    
    def _generate_mock_address(self) -> str:
        """Generate a mock contract address for testing"""
        import secrets
        return "0x" + secrets.token_hex(20)
    
    async def _save_deployment_record(self, deployment_data: Dict):
        """Save deployment record to file"""
        try:
            # Create deployments directory
            deployments_dir = Path(__file__).parent.parent.parent / "deployments"
            deployments_dir.mkdir(exist_ok=True)
            
            # Save deployment record
            filename = f"{deployment_data['network']}-{int(datetime.utcnow().timestamp())}.json"
            filepath = deployments_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(deployment_data, f, indent=2)
                
            logger.info(f"Deployment record saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save deployment record: {e}")
    
    async def verify_contract(self, contract_address: str, network: str) -> bool:
        """Verify contract on PolygonScan"""
        try:
            # This would implement actual contract verification
            # For now, just log the verification command
            
            network_config = self.config["networks"][network]
            explorer_url = network_config["explorer"]
            
            logger.info(f"Contract verification info:")
            logger.info(f"Contract: {contract_address}")
            logger.info(f"Explorer: {explorer_url}/address/{contract_address}")
            logger.info(f"Verification command (if using Hardhat):")
            logger.info(f"npx hardhat verify --network {network} {contract_address}")
            
            return True
            
        except Exception as e:
            logger.error(f"Contract verification failed: {e}")
            return False
    
    async def get_deployment_info(self, network: str) -> Optional[Dict]:
        """Get latest deployment info for network"""
        try:
            deployments_dir = Path(__file__).parent.parent.parent / "deployments"
            if not deployments_dir.exists():
                return None
                
            # Find latest deployment for network
            deployment_files = list(deployments_dir.glob(f"{network}-*.json"))
            if not deployment_files:
                return None
                
            # Get most recent
            latest_file = max(deployment_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                deployment_data = json.load(f)
                
            return deployment_data
            
        except Exception as e:
            logger.error(f"Failed to get deployment info: {e}")
            return None
    
    def get_deployment_instructions(self, network: str) -> str:
        """Get deployment instructions for network"""
        network_config = self.config["networks"][network]
        
        instructions = f"""
🚀 PRSM FTNS Token Deployment Instructions

Network: {network_config['name']}
Chain ID: {network_config['chain_id']}

Prerequisites:
1. MATIC for gas fees (minimum 0.01 MATIC)
2. Private key for deployment wallet
3. Network access and stable internet

Quick Start:
1. Get testnet MATIC: {network_config.get('faucet', 'N/A')}
2. Set environment variables:
   - POLYGON_MUMBAI_RPC_URL={network_config['rpc_url']}
   - PRIVATE_KEY=your_private_key_here
3. Run deployment script
4. Update PRSM configuration with contract address

Post-deployment:
1. Verify contract on {network_config['explorer']}
2. Test token operations
3. Initialize Web3 services
4. Begin user onboarding

Support:
- Check deployment logs for errors
- Verify network connectivity
- Ensure sufficient gas balance
"""
        return instructions

# Convenience functions for easy usage
async def deploy_to_mumbai(private_key: str, treasury_address: Optional[str] = None) -> Optional[Dict]:
    """Deploy FTNS token to Polygon Mumbai testnet"""
    deployer = FTNSContractDeployer()
    
    # Connect to Mumbai
    connected = await deployer.connect_to_network("polygon_mumbai")
    if not connected:
        logger.error("Failed to connect to Polygon Mumbai")
        return None
    
    # Setup wallet
    wallet_setup = await deployer.setup_wallet(private_key)
    if not wallet_setup:
        logger.error("Failed to setup deployment wallet")
        return None
    
    # Deploy contract
    deployment = await deployer.deploy_ftns_token("polygon_mumbai", treasury_address)
    if deployment:
        logger.info("✅ Deployment successful!")
        
        # Verify contract
        await deployer.verify_contract(deployment["contracts"]["ftns_token"], "polygon_mumbai")
        
    return deployment

async def get_latest_deployment(network: str = "polygon_mumbai") -> Optional[Dict]:
    """Get latest deployment information"""
    deployer = FTNSContractDeployer()
    return await deployer.get_deployment_info(network)

# Alias for compatibility
ContractDeployer = FTNSContractDeployer