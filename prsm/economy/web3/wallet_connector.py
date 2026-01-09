"""
Web3 Wallet Connector for PRSM FTNS Token Integration

Handles wallet connections, transaction signing, and Web3 provider management
for seamless integration with Polygon blockchain.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

from web3 import Web3, AsyncWeb3
from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware
from web3.exceptions import TransactionNotFound, BlockNotFound
from eth_account import Account
from eth_account.messages import encode_defunct
import json
import os

logger = logging.getLogger(__name__)

class NetworkType(Enum):
    MAINNET = "mainnet"
    TESTNET = "testnet"
    LOCAL = "local"

@dataclass
class WalletInfo:
    address: str
    balance_matic: Decimal
    balance_ftns: Decimal
    network: NetworkType
    connected: bool

@dataclass
class TransactionRequest:
    to: str
    value: int = 0
    gas_limit: int = 21000
    gas_price: Optional[int] = None
    data: str = "0x"
    nonce: Optional[int] = None

@dataclass
class TransactionResult:
    hash: str
    success: bool
    gas_used: int
    block_number: int
    error: Optional[str] = None

class Web3WalletConnector:
    """
    Manages Web3 wallet connections and blockchain interactions for PRSM FTNS system.
    
    Features:
    - Multi-network support (Polygon mainnet/testnet)
    - Wallet connection management
    - Transaction signing and broadcasting
    - Balance checking and monitoring
    - Gas estimation and optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_default_config()
        self.w3: Optional[Web3] = None
        self.connected_wallet: Optional[str] = None
        self.current_network: Optional[NetworkType] = None
        self._setup_logging()
        
    def _load_default_config(self) -> Dict:
        """Load default Web3 configuration"""
        return {
            "networks": {
                "polygon_mainnet": {
                    "rpc_url": "https://polygon-rpc.com",
                    "chain_id": 137,
                    "name": "Polygon Mainnet",
                    "explorer": "https://polygonscan.com"
                },
                "polygon_mumbai": {
                    "rpc_url": "https://rpc-mumbai.maticvigil.com",
                    "chain_id": 80001,
                    "name": "Polygon Mumbai Testnet",
                    "explorer": "https://mumbai.polygonscan.com"
                },
                "localhost": {
                    "rpc_url": "http://127.0.0.1:8545",
                    "chain_id": 31337,
                    "name": "Local Hardhat",
                    "explorer": "http://localhost:8545"
                }
            },
            "gas": {
                "limit_multiplier": 1.2,
                "price_multiplier": 1.1,
                "max_priority_fee": 30000000000,  # 30 gwei
                "max_fee": 100000000000  # 100 gwei
            },
            "timeouts": {
                "connection": 30,
                "transaction": 300
            }
        }
    
    def _setup_logging(self):
        """Setup logging for Web3 operations"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def connect_to_network(self, network: str = "polygon_mumbai") -> bool:
        """
        Connect to specified blockchain network
        
        Args:
            network: Network identifier (polygon_mainnet, polygon_mumbai, localhost)
            
        Returns:
            bool: True if connection successful
        """
        try:
            if network not in self.config["networks"]:
                raise ValueError(f"Unknown network: {network}")
                
            network_config = self.config["networks"][network]
            
            # Initialize Web3 provider
            self.w3 = Web3(Web3.HTTPProvider(
                network_config["rpc_url"],
                request_kwargs={"timeout": self.config["timeouts"]["connection"]}
            ))
            
            # Add PoA middleware for Polygon
            if "polygon" in network:
                self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            
            # Verify connection
            if not self.w3.is_connected():
                raise ConnectionError(f"Failed to connect to {network}")
                
            # Verify chain ID
            chain_id = self.w3.eth.chain_id
            expected_chain_id = network_config["chain_id"]
            if chain_id != expected_chain_id:
                raise ValueError(f"Chain ID mismatch: expected {expected_chain_id}, got {chain_id}")
                
            # Set network type
            if "mainnet" in network:
                self.current_network = NetworkType.MAINNET
            elif "mumbai" in network or "testnet" in network:
                self.current_network = NetworkType.TESTNET
            else:
                self.current_network = NetworkType.LOCAL
                
            self.logger.info(f"Connected to {network_config['name']} (Chain ID: {chain_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to network {network}: {e}")
            return False
    
    async def connect_wallet(self, private_key: Optional[str] = None, 
                           mnemonic: Optional[str] = None) -> Optional[str]:
        """
        Connect wallet using private key or mnemonic
        
        Args:
            private_key: Wallet private key (hex string)
            mnemonic: Wallet mnemonic phrase
            
        Returns:
            str: Wallet address if successful, None otherwise
        """
        try:
            if not self.w3:
                raise RuntimeError("Must connect to network first")
                
            if private_key:
                # Connect using private key
                if not private_key.startswith('0x'):
                    private_key = '0x' + private_key
                    
                account = Account.from_key(private_key)
                self.connected_wallet = account.address
                
                # Add account to Web3 instance
                self.w3.eth.default_account = self.connected_wallet
                
            elif mnemonic:
                # Connect using mnemonic (implement if needed)
                Account.enable_unaudited_hdwallet_features()
                account = Account.from_mnemonic(mnemonic)
                self.connected_wallet = account.address
                self.w3.eth.default_account = self.connected_wallet
                
            else:
                # Try to use environment variable
                env_key = os.getenv('WALLET_PRIVATE_KEY')
                if env_key:
                    return await self.connect_wallet(private_key=env_key)
                else:
                    raise ValueError("No wallet credentials provided")
            
            self.logger.info(f"Wallet connected: {self.connected_wallet}")
            return self.connected_wallet
            
        except Exception as e:
            self.logger.error(f"Failed to connect wallet: {e}")
            return None
    
    async def get_wallet_info(self, address: Optional[str] = None) -> Optional[WalletInfo]:
        """
        Get comprehensive wallet information
        
        Args:
            address: Wallet address (uses connected wallet if None)
            
        Returns:
            WalletInfo: Wallet information including balances
        """
        try:
            if not self.w3:
                raise RuntimeError("Not connected to network")
                
            wallet_address = address or self.connected_wallet
            if not wallet_address:
                raise ValueError("No wallet address provided")
                
            # Get MATIC balance
            balance_wei = self.w3.eth.get_balance(wallet_address)
            balance_matic = Decimal(self.w3.from_wei(balance_wei, 'ether'))
            
            # Get FTNS balance (will implement after contract integration)
            balance_ftns = Decimal('0')  # Placeholder
            
            return WalletInfo(
                address=wallet_address,
                balance_matic=balance_matic,
                balance_ftns=balance_ftns,
                network=self.current_network,
                connected=wallet_address == self.connected_wallet
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get wallet info: {e}")
            return None
    
    async def estimate_gas(self, transaction: TransactionRequest) -> int:
        """
        Estimate gas required for transaction
        
        Args:
            transaction: Transaction details
            
        Returns:
            int: Estimated gas amount
        """
        try:
            if not self.w3 or not self.connected_wallet:
                raise RuntimeError("Wallet not connected")
                
            # Build transaction for estimation
            tx_dict = {
                'from': self.connected_wallet,
                'to': transaction.to,
                'value': transaction.value,
                'data': transaction.data
            }
            
            # Estimate gas
            estimated_gas = self.w3.eth.estimate_gas(tx_dict)
            
            # Apply safety multiplier
            multiplier = self.config["gas"]["limit_multiplier"]
            final_gas = int(estimated_gas * multiplier)
            
            self.logger.debug(f"Gas estimation: {estimated_gas} -> {final_gas} (×{multiplier})")
            return final_gas
            
        except Exception as e:
            self.logger.error(f"Gas estimation failed: {e}")
            # Return default gas limit if estimation fails
            return transaction.gas_limit
    
    async def get_gas_price(self) -> Dict[str, int]:
        """
        Get current gas prices for different transaction speeds
        
        Returns:
            Dict: Gas prices (slow, standard, fast) in wei
        """
        try:
            if not self.w3:
                raise RuntimeError("Not connected to network")
                
            # Get current gas price
            current_gas_price = self.w3.eth.gas_price
            
            # Calculate different speed tiers
            multiplier = self.config["gas"]["price_multiplier"]
            
            prices = {
                "slow": current_gas_price,
                "standard": int(current_gas_price * multiplier),
                "fast": int(current_gas_price * multiplier * 1.5)
            }
            
            # Apply maximum limits
            max_fee = self.config["gas"]["max_fee"]
            for speed in prices:
                prices[speed] = min(prices[speed], max_fee)
                
            return prices
            
        except Exception as e:
            self.logger.error(f"Failed to get gas prices: {e}")
            return {
                "slow": 20000000000,    # 20 gwei
                "standard": 30000000000, # 30 gwei  
                "fast": 50000000000     # 50 gwei
            }
    
    async def send_transaction(self, transaction: TransactionRequest, 
                             private_key: Optional[str] = None) -> Optional[TransactionResult]:
        """
        Sign and send transaction to blockchain
        
        Args:
            transaction: Transaction details
            private_key: Private key for signing (uses connected wallet if None)
            
        Returns:
            TransactionResult: Transaction outcome
        """
        try:
            if not self.w3 or not self.connected_wallet:
                raise RuntimeError("Wallet not connected")
                
            # Get nonce if not provided
            nonce = transaction.nonce
            if nonce is None:
                nonce = self.w3.eth.get_transaction_count(self.connected_wallet)
                
            # Get gas price if not provided
            gas_price = transaction.gas_price
            if gas_price is None:
                gas_prices = await self.get_gas_price()
                gas_price = gas_prices["standard"]
                
            # Estimate gas if needed
            if transaction.gas_limit == 21000 and transaction.data != "0x":
                gas_limit = await self.estimate_gas(transaction)
            else:
                gas_limit = transaction.gas_limit
                
            # Build transaction
            tx_dict = {
                'nonce': nonce,
                'to': transaction.to,
                'value': transaction.value,
                'gas': gas_limit,
                'gasPrice': gas_price,
                'data': transaction.data,
                'chainId': self.w3.eth.chain_id
            }
            
            # Sign transaction
            if private_key:
                if not private_key.startswith('0x'):
                    private_key = '0x' + private_key
                signed_txn = self.w3.eth.account.sign_transaction(tx_dict, private_key)
            else:
                # Use connected wallet (implement signing mechanism)
                signed_txn = self.w3.eth.account.sign_transaction(
                    tx_dict, 
                    os.getenv('WALLET_PRIVATE_KEY')
                )
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            tx_hash_hex = tx_hash.hex()
            
            self.logger.info(f"Transaction sent: {tx_hash_hex}")
            
            # Wait for transaction receipt
            timeout = self.config["timeouts"]["transaction"]
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            
            # Check transaction status
            success = receipt['status'] == 1
            
            result = TransactionResult(
                hash=tx_hash_hex,
                success=success,
                gas_used=receipt['gasUsed'],
                block_number=receipt['blockNumber'],
                error=None if success else "Transaction reverted"
            )
            
            if success:
                self.logger.info(f"Transaction successful: {tx_hash_hex}")
            else:
                self.logger.error(f"Transaction failed: {tx_hash_hex}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Transaction failed: {e}")
            return TransactionResult(
                hash="",
                success=False,
                gas_used=0,
                block_number=0,
                error=str(e)
            )
    
    async def get_transaction_status(self, tx_hash: str) -> Optional[TransactionResult]:
        """
        Get status of transaction by hash
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            TransactionResult: Transaction status
        """
        try:
            if not self.w3:
                raise RuntimeError("Not connected to network")
                
            # Get transaction receipt
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            success = receipt['status'] == 1
            
            return TransactionResult(
                hash=tx_hash,
                success=success,
                gas_used=receipt['gasUsed'],
                block_number=receipt['blockNumber'],
                error=None if success else "Transaction reverted"
            )
            
        except TransactionNotFound:
            self.logger.warning(f"Transaction not found: {tx_hash}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to get transaction status: {e}")
            return None
    
    async def sign_message(self, message: str, private_key: Optional[str] = None) -> Optional[str]:
        """
        Sign message with wallet
        
        Args:
            message: Message to sign
            private_key: Private key for signing
            
        Returns:
            str: Signature hex string
        """
        try:
            if not private_key:
                private_key = os.getenv('WALLET_PRIVATE_KEY')
                
            if not private_key:
                raise ValueError("No private key available for signing")
                
            if not private_key.startswith('0x'):
                private_key = '0x' + private_key
                
            # Encode message
            encoded_msg = encode_defunct(text=message)
            
            # Sign message
            signed_message = Account.sign_message(encoded_msg, private_key)
            
            return signed_message.signature.hex()
            
        except Exception as e:
            self.logger.error(f"Message signing failed: {e}")
            return None
    
    async def verify_signature(self, message: str, signature: str, address: str) -> bool:
        """
        Verify message signature
        
        Args:
            message: Original message
            signature: Signature to verify
            address: Expected signer address
            
        Returns:
            bool: True if signature is valid
        """
        try:
            # Encode message
            encoded_msg = encode_defunct(text=message)
            
            # Recover address from signature
            recovered_address = Account.recover_message(encoded_msg, signature=signature)
            
            return recovered_address.lower() == address.lower()
            
        except Exception as e:
            self.logger.error(f"Signature verification failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Web3 provider and clear wallet connection"""
        self.w3 = None
        self.connected_wallet = None
        self.current_network = None
        self.logger.info("Disconnected from Web3 provider")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to blockchain network"""
        return self.w3 is not None and self.w3.is_connected()
    
    @property
    def has_wallet(self) -> bool:
        """Check if wallet is connected"""
        return self.connected_wallet is not None