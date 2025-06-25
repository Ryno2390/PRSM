"""
Smart Contract Interface for PRSM FTNS Token Operations

Provides high-level interface for interacting with FTNS smart contracts
including token operations, marketplace, and governance functions.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from dataclasses import dataclass
from pathlib import Path

from web3 import Web3
from web3.contract import Contract
from eth_utils import to_checksum_address

from .wallet_connector import Web3WalletConnector, TransactionRequest, TransactionResult

logger = logging.getLogger(__name__)

@dataclass
class TokenBalance:
    liquid: Decimal
    locked: Decimal
    staked: Decimal
    total: Decimal
    context_allocated: Decimal

@dataclass
class MarketplaceListing:
    id: int
    owner: str
    ipfs_hash: str
    title: str
    description: str
    price_per_hour: Decimal
    min_rental_hours: int
    max_rental_hours: int
    total_rentals: int
    average_rating: Decimal
    is_active: bool

@dataclass
class GovernanceProposal:
    id: int
    proposer: str
    title: str
    description: str
    category: int
    ipfs_hash: str
    is_emergency: bool
    creation_time: int
    voting_start: int
    voting_end: int
    votes_for: int
    votes_against: int
    votes_abstain: int
    state: int

class FTNSContractInterface:
    """
    High-level interface for FTNS smart contract interactions
    
    Features:
    - Token operations (transfer, mint, burn, stake)
    - Balance management (liquid, locked, staked)
    - Marketplace interactions (list models, rent, rate)
    - Governance participation (propose, vote, execute)
    - Event monitoring and logging
    """
    
    def __init__(self, wallet_connector: Web3WalletConnector, contract_addresses: Dict[str, str]):
        self.wallet = wallet_connector
        self.contract_addresses = contract_addresses
        self.contracts: Dict[str, Contract] = {}
        self._load_abis()
        
    def _load_abis(self):
        """Load contract ABIs from files"""
        try:
            # Load ABIs from contracts directory
            contracts_dir = Path(__file__).parent.parent.parent / "contracts" / "artifacts" / "contracts"
            
            self.abis = {
                "FTNSToken": self._load_abi_file(contracts_dir / "FTNSTokenSimple.sol" / "FTNSTokenSimple.json"),
                # Will add other contracts when they're ready
                # "FTNSMarketplace": self._load_abi_file(contracts_dir / "FTNSMarketplace.sol" / "FTNSMarketplace.json"),
                # "FTNSGovernance": self._load_abi_file(contracts_dir / "FTNSGovernance.sol" / "FTNSGovernance.json")
            }
            
        except Exception as e:
            logger.warning(f"Could not load ABIs from files: {e}")
            # Use minimal ABI for basic operations
            self.abis = {
                "FTNSToken": self._get_minimal_erc20_abi()
            }
    
    def _load_abi_file(self, file_path: Path) -> List[Dict]:
        """Load ABI from compiled contract JSON file"""
        try:
            with open(file_path, 'r') as f:
                contract_json = json.load(f)
                return contract_json['abi']
        except Exception as e:
            logger.error(f"Failed to load ABI from {file_path}: {e}")
            return []
    
    def _get_minimal_erc20_abi(self) -> List[Dict]:
        """Get minimal ERC20 ABI for basic token operations"""
        return [
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol", 
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "_to", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "transfer",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "_spender", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [
                    {"name": "_owner", "type": "address"},
                    {"name": "_spender", "type": "address"}
                ],
                "name": "allowance",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ]
    
    async def initialize_contracts(self) -> bool:
        """
        Initialize smart contract instances
        
        Returns:
            bool: True if initialization successful
        """
        try:
            if not self.wallet.is_connected:
                raise RuntimeError("Wallet not connected to network")
                
            w3 = self.wallet.w3
            
            # Initialize FTNS Token contract
            if "ftns_token" in self.contract_addresses:
                token_address = to_checksum_address(self.contract_addresses["ftns_token"])
                self.contracts["FTNSToken"] = w3.eth.contract(
                    address=token_address,
                    abi=self.abis["FTNSToken"]
                )
                logger.info(f"FTNS Token contract initialized at {token_address}")
            
            # Initialize other contracts when available
            # if "marketplace" in self.contract_addresses:
            #     marketplace_address = to_checksum_address(self.contract_addresses["marketplace"])
            #     self.contracts["FTNSMarketplace"] = w3.eth.contract(
            #         address=marketplace_address,
            #         abi=self.abis["FTNSMarketplace"]
            #     )
            
            return True
            
        except Exception as e:
            logger.error(f"Contract initialization failed: {e}")
            return False
    
    # === Token Operations ===
    
    async def get_token_info(self) -> Dict[str, Any]:
        """
        Get basic token information
        
        Returns:
            Dict: Token name, symbol, decimals, total supply
        """
        try:
            contract = self.contracts.get("FTNSToken")
            if not contract:
                raise RuntimeError("FTNS Token contract not initialized")
                
            # Get token details
            name = contract.functions.name().call()
            symbol = contract.functions.symbol().call()
            decimals = contract.functions.decimals().call()
            total_supply = contract.functions.totalSupply().call()
            
            return {
                "name": name,
                "symbol": symbol,
                "decimals": decimals,
                "total_supply": Decimal(total_supply) / (10 ** decimals),
                "address": contract.address
            }
            
        except Exception as e:
            logger.error(f"Failed to get token info: {e}")
            return {}
    
    async def get_balance(self, address: Optional[str] = None) -> Decimal:
        """
        Get FTNS token balance for address
        
        Args:
            address: Wallet address (uses connected wallet if None)
            
        Returns:
            Decimal: Token balance
        """
        try:
            contract = self.contracts.get("FTNSToken")
            if not contract:
                raise RuntimeError("FTNS Token contract not initialized")
                
            wallet_address = address or self.wallet.connected_wallet
            if not wallet_address:
                raise ValueError("No wallet address provided")
                
            # Get balance
            balance_wei = contract.functions.balanceOf(wallet_address).call()
            decimals = contract.functions.decimals().call()
            
            return Decimal(balance_wei) / (10 ** decimals)
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return Decimal('0')
    
    async def get_detailed_balance(self, address: Optional[str] = None) -> Optional[TokenBalance]:
        """
        Get detailed token balance information (liquid, locked, staked)
        
        Args:
            address: Wallet address (uses connected wallet if None)
            
        Returns:
            TokenBalance: Detailed balance information
        """
        try:
            contract = self.contracts.get("FTNSToken")
            if not contract:
                raise RuntimeError("FTNS Token contract not initialized")
                
            wallet_address = address or self.wallet.connected_wallet
            if not wallet_address:
                raise ValueError("No wallet address provided")
                
            # Try to get detailed balance (if contract supports it)
            try:
                account_info = contract.functions.getAccountInfo(wallet_address).call()
                decimals = contract.functions.decimals().call()
                divisor = 10 ** decimals
                
                return TokenBalance(
                    liquid=Decimal(account_info[0]) / divisor,
                    locked=Decimal(account_info[1]) / divisor,
                    staked=Decimal(account_info[2]) / divisor,
                    total=Decimal(account_info[3]) / divisor,
                    context_allocated=Decimal(account_info[4]) / divisor
                )
                
            except Exception:
                # Fallback to simple balance if detailed info not available
                balance = await self.get_balance(address)
                return TokenBalance(
                    liquid=balance,
                    locked=Decimal('0'),
                    staked=Decimal('0'),
                    total=balance,
                    context_allocated=Decimal('0')
                )
                
        except Exception as e:
            logger.error(f"Failed to get detailed balance: {e}")
            return None
    
    async def transfer_tokens(self, to_address: str, amount: Decimal) -> Optional[TransactionResult]:
        """
        Transfer FTNS tokens to another address
        
        Args:
            to_address: Recipient address
            amount: Amount to transfer
            
        Returns:
            TransactionResult: Transaction outcome
        """
        try:
            contract = self.contracts.get("FTNSToken")
            if not contract:
                raise RuntimeError("FTNS Token contract not initialized")
                
            if not self.wallet.connected_wallet:
                raise RuntimeError("No wallet connected")
                
            # Convert amount to wei
            decimals = contract.functions.decimals().call()
            amount_wei = int(amount * (10 ** decimals))
            
            # Build transaction data
            tx_data = contract.encodeABI(
                fn_name="transfer",
                args=[to_checksum_address(to_address), amount_wei]
            )
            
            # Create transaction request
            tx_request = TransactionRequest(
                to=contract.address,
                data=tx_data
            )
            
            # Send transaction
            result = await self.wallet.send_transaction(tx_request)
            
            if result and result.success:
                logger.info(f"Transferred {amount} FTNS to {to_address}")
            
            return result
            
        except Exception as e:
            logger.error(f"Token transfer failed: {e}")
            return None
    
    async def approve_spending(self, spender_address: str, amount: Decimal) -> Optional[TransactionResult]:
        """
        Approve another address to spend tokens on behalf of wallet
        
        Args:
            spender_address: Address to approve for spending
            amount: Amount to approve
            
        Returns:
            TransactionResult: Transaction outcome
        """
        try:
            contract = self.contracts.get("FTNSToken")
            if not contract:
                raise RuntimeError("FTNS Token contract not initialized")
                
            if not self.wallet.connected_wallet:
                raise RuntimeError("No wallet connected")
                
            # Convert amount to wei
            decimals = contract.functions.decimals().call()
            amount_wei = int(amount * (10 ** decimals))
            
            # Build transaction data
            tx_data = contract.encodeABI(
                fn_name="approve",
                args=[to_checksum_address(spender_address), amount_wei]
            )
            
            # Create transaction request
            tx_request = TransactionRequest(
                to=contract.address,
                data=tx_data
            )
            
            # Send transaction
            result = await self.wallet.send_transaction(tx_request)
            
            if result and result.success:
                logger.info(f"Approved {amount} FTNS spending for {spender_address}")
            
            return result
            
        except Exception as e:
            logger.error(f"Token approval failed: {e}")
            return None
    
    async def get_allowance(self, owner_address: str, spender_address: str) -> Decimal:
        """
        Get approved spending allowance
        
        Args:
            owner_address: Token owner address
            spender_address: Approved spender address
            
        Returns:
            Decimal: Approved amount
        """
        try:
            contract = self.contracts.get("FTNSToken")
            if not contract:
                raise RuntimeError("FTNS Token contract not initialized")
                
            # Get allowance
            allowance_wei = contract.functions.allowance(
                to_checksum_address(owner_address),
                to_checksum_address(spender_address)
            ).call()
            
            decimals = contract.functions.decimals().call()
            return Decimal(allowance_wei) / (10 ** decimals)
            
        except Exception as e:
            logger.error(f"Failed to get allowance: {e}")
            return Decimal('0')
    
    # === Event Monitoring ===
    
    async def get_transfer_events(self, from_block: int = 0, to_block: str = "latest", 
                                 address_filter: Optional[str] = None) -> List[Dict]:
        """
        Get token transfer events
        
        Args:
            from_block: Starting block number
            to_block: Ending block number or "latest"
            address_filter: Filter by specific address (from or to)
            
        Returns:
            List[Dict]: Transfer events
        """
        try:
            contract = self.contracts.get("FTNSToken")
            if not contract:
                raise RuntimeError("FTNS Token contract not initialized")
                
            # Create filter arguments
            filter_args = {
                "fromBlock": from_block,
                "toBlock": to_block
            }
            
            if address_filter:
                # Filter by address (either from or to)
                filter_args["argument_filters"] = {
                    "from": to_checksum_address(address_filter)
                }
            
            # Get events
            transfer_filter = contract.events.Transfer.create_filter(**filter_args)
            events = transfer_filter.get_all_entries()
            
            # Format events
            formatted_events = []
            for event in events:
                formatted_events.append({
                    "block_number": event["blockNumber"],
                    "transaction_hash": event["transactionHash"].hex(),
                    "from": event["args"]["from"],
                    "to": event["args"]["to"],
                    "value": event["args"]["value"],
                    "timestamp": await self._get_block_timestamp(event["blockNumber"])
                })
            
            return formatted_events
            
        except Exception as e:
            logger.error(f"Failed to get transfer events: {e}")
            return []
    
    async def _get_block_timestamp(self, block_number: int) -> int:
        """Get timestamp for block number"""
        try:
            block = self.wallet.w3.eth.get_block(block_number)
            return block["timestamp"]
        except Exception:
            return 0
    
    # === Utility Functions ===
    
    async def estimate_gas_for_transfer(self, to_address: str, amount: Decimal) -> int:
        """
        Estimate gas required for token transfer
        
        Args:
            to_address: Recipient address
            amount: Transfer amount
            
        Returns:
            int: Estimated gas amount
        """
        try:
            contract = self.contracts.get("FTNSToken")
            if not contract:
                raise RuntimeError("FTNS Token contract not initialized")
                
            if not self.wallet.connected_wallet:
                raise RuntimeError("No wallet connected")
                
            # Convert amount to wei
            decimals = contract.functions.decimals().call()
            amount_wei = int(amount * (10 ** decimals))
            
            # Estimate gas
            gas_estimate = contract.functions.transfer(
                to_checksum_address(to_address),
                amount_wei
            ).estimate_gas({"from": self.wallet.connected_wallet})
            
            return int(gas_estimate * 1.2)  # Add 20% safety margin
            
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}")
            return 100000  # Default gas limit for token transfers
    
    def get_contract_address(self, contract_name: str) -> Optional[str]:
        """Get contract address by name"""
        return self.contract_addresses.get(contract_name.lower())
    
    def is_contract_loaded(self, contract_name: str) -> bool:
        """Check if contract is loaded and ready"""
        return contract_name in self.contracts
    
    async def verify_contract_deployment(self, contract_name: str) -> bool:
        """
        Verify that contract is properly deployed
        
        Args:
            contract_name: Name of contract to verify
            
        Returns:
            bool: True if contract is deployed and accessible
        """
        try:
            contract = self.contracts.get(contract_name)
            if not contract:
                return False
                
            # Try to call a view function to verify deployment
            if contract_name == "FTNSToken":
                name = contract.functions.name().call()
                return len(name) > 0
                
            return True
            
        except Exception as e:
            logger.error(f"Contract verification failed for {contract_name}: {e}")
            return False