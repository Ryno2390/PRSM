"""
Contract Manager for PRSM Blockchain Integration
=================================================

Manages deployed smart contracts and provides a unified interface for
interacting with FTNS token and bridge contracts on-chain.

Features:
- Contract instance management
- Balance queries and transfers
- Role management
- Event monitoring
- Transaction tracking
- Gas estimation
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import structlog

# Web3 imports with graceful fallback
try:
    from web3 import Web3
    from web3.contract import Contract
    from web3.exceptions import ContractLogicError, TimeExhausted
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    Web3 = None
    Contract = None
    Account = None
    
    # Create a proper exception class for fallback
    class TimeExhausted(Exception):
        """Fallback TimeExhausted exception when web3 is not installed"""
        pass

from .networks import NetworkConfig, get_network_config, NetworkType
from .deployment import (
    DeploymentConfig,
    DeployedContract,
    ContractType,
    FTNS_TOKEN_ABI,
    BRIDGE_ABI,
    ERC20_ABI,
)

logger = structlog.get_logger(__name__)


# ============ Enums ============

class TransactionStatus(Enum):
    """Status of a blockchain transaction"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class RoleType(Enum):
    """Standard role types for FTNS token"""
    DEFAULT_ADMIN = "DEFAULT_ADMIN_ROLE"
    MINTER = "MINTER_ROLE"
    PAUSER = "PAUSER_ROLE"
    BURNER = "BURNER_ROLE"
    BRIDGE = "BRIDGE_ROLE"


# ============ Data Classes ============

@dataclass
class TransactionResult:
    """Result of a blockchain transaction"""
    success: bool
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    status: TransactionStatus = TransactionStatus.PENDING
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "gas_used": self.gas_used,
            "status": self.status.value,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TokenBalance:
    """Token balance information"""
    address: str
    balance: int  # In wei
    decimals: int = 18
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def formatted(self) -> Decimal:
        """Get formatted balance in human-readable form"""
        return Decimal(self.balance) / Decimal(10 ** self.decimals)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "address": self.address,
            "balance": str(self.balance),
            "balance_formatted": str(self.formatted),
            "decimals": self.decimals,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TransferEvent:
    """Token transfer event"""
    from_address: str
    to_address: str
    amount: int
    tx_hash: str
    block_number: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "from_address": self.from_address,
            "to_address": self.to_address,
            "amount": str(self.amount),
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "timestamp": self.timestamp.isoformat(),
        }


# ============ Contract Manager ============

class ContractManager:
    """
    Manages deployed contracts and provides interaction interface.
    
    Provides methods for:
    - Loading deployed contracts
    - Querying token balances
    - Executing transfers
    - Managing roles
    - Monitoring events
    """
    
    def __init__(
        self,
        web3_provider: Web3,
        config: DeploymentConfig,
        account: Optional[Account] = None
    ):
        """
        Initialize contract manager.
        
        Args:
            web3_provider: Web3 instance connected to network
            config: Deployment configuration
            account: Optional account for signing transactions
        """
        self.w3 = web3_provider
        self.config = config
        self.account = account
        
        # Contract instances
        self._contracts: Dict[str, Contract] = {}
        
        # Deployment records
        self._deployments: Dict[str, DeployedContract] = {}
        
        # Event listeners
        self._event_filters: Dict[str, Any] = {}
        self._event_callbacks: Dict[str, List[Callable]] = {}
        
        # Cache
        self._balance_cache: Dict[str, TokenBalance] = {}
        self._cache_ttl = 30  # seconds
        
        logger.info(
            "ContractManager initialized",
            network=config.network,
            chain_id=config.chain_id
        )
    
    async def load_contract(
        self,
        contract_type: ContractType,
        address: str,
        abi: Optional[List[Dict]] = None
    ) -> Contract:
        """
        Load a deployed contract.
        
        Args:
            contract_type: Type of contract
            address: Contract address
            abi: Optional ABI (uses default if not provided)
            
        Returns:
            Web3 Contract instance
        """
        # Get default ABI if not provided
        if abi is None:
            abi = self._get_default_abi(contract_type)
        
        # Create contract instance
        contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(address),
            abi=abi
        )
        
        # Store contract
        self._contracts[contract_type.value] = contract
        
        logger.info(
            "Contract loaded",
            contract_type=contract_type.value,
            address=address
        )
        
        return contract
    
    async def load_from_deployment(
        self,
        deployed: DeployedContract
    ) -> Contract:
        """
        Load contract from deployment record.
        
        Args:
            deployed: DeployedContract instance
            
        Returns:
            Web3 Contract instance
        """
        contract = await self.load_contract(
            contract_type=deployed.contract_type,
            address=deployed.address,
            abi=deployed.abi
        )
        
        # Store deployment record
        self._deployments[deployed.contract_type.value] = deployed
        
        return contract
    
    def _get_default_abi(self, contract_type: ContractType) -> List[Dict]:
        """Get default ABI for contract type"""
        if contract_type == ContractType.FTNS_TOKEN:
            return FTNS_TOKEN_ABI
        elif contract_type == ContractType.FTNS_BRIDGE:
            return BRIDGE_ABI
        else:
            return ERC20_ABI
    
    def get_token_contract(self, address: Optional[str] = None) -> Optional[Contract]:
        """
        Get FTNS token contract instance.
        
        Args:
            address: Optional address (uses loaded contract if not provided)
            
        Returns:
            Contract instance or None
        """
        key = ContractType.FTNS_TOKEN.value
        if address:
            return self._contracts.get(key)
        return self._contracts.get(key)
    
    def get_bridge_contract(self, address: Optional[str] = None) -> Optional[Contract]:
        """
        Get bridge contract instance.
        
        Args:
            address: Optional address (uses loaded contract if not provided)
            
        Returns:
            Contract instance or None
        """
        key = ContractType.FTNS_BRIDGE.value
        return self._contracts.get(key)
    
    # ============ Token Operations ============
    
    async def get_token_balance(
        self,
        address: str,
        use_cache: bool = True
    ) -> TokenBalance:
        """
        Get FTNS balance for an address.
        
        Args:
            address: Address to query
            use_cache: Whether to use cached balance
            
        Returns:
            TokenBalance instance
        """
        # Check cache
        cache_key = f"balance_{address}"
        if use_cache and cache_key in self._balance_cache:
            cached = self._balance_cache[cache_key]
            # Check if cache is still valid
            age = (datetime.now(timezone.utc) - cached.timestamp).total_seconds()
            if age < self._cache_ttl:
                return cached
        
        # Get contract
        contract = self.get_token_contract()
        if not contract:
            raise RuntimeError("Token contract not loaded")
        
        # Query balance
        checksum_address = Web3.to_checksum_address(address)
        balance = contract.functions.balanceOf(checksum_address).call()
        
        # Get decimals
        decimals = contract.functions.decimals().call()
        
        # Create balance object
        token_balance = TokenBalance(
            address=address,
            balance=balance,
            decimals=decimals
        )
        
        # Update cache
        self._balance_cache[cache_key] = token_balance
        
        return token_balance
    
    async def get_total_supply(self) -> int:
        """
        Get total FTNS token supply.
        
        Returns:
            Total supply in wei
        """
        contract = self.get_token_contract()
        if not contract:
            raise RuntimeError("Token contract not loaded")
        
        return contract.functions.totalSupply().call()
    
    async def transfer_tokens(
        self,
        to_address: str,
        amount: int,
        wait_for_confirmation: bool = True
    ) -> TransactionResult:
        """
        Transfer FTNS tokens.
        
        Args:
            to_address: Recipient address
            amount: Amount to transfer in wei
            wait_for_confirmation: Whether to wait for tx confirmation
            
        Returns:
            TransactionResult
        """
        if not self.account:
            raise RuntimeError("No account configured for transactions")
        
        contract = self.get_token_contract()
        if not contract:
            raise RuntimeError("Token contract not loaded")
        
        try:
            # Build transaction
            checksum_to = Web3.to_checksum_address(to_address)
            tx = contract.functions.transfer(checksum_to, amount).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
            })
            
            # Estimate gas
            gas_estimate = self.w3.eth.estimate_gas(tx)
            tx['gas'] = int(gas_estimate * 1.2)  # Add buffer
            
            # Get gas price
            gas_price = self.w3.eth.gas_price
            tx['gasPrice'] = gas_price
            
            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()
            
            logger.info(
                "Transfer transaction sent",
                to=to_address,
                amount=amount,
                tx_hash=tx_hash_hex
            )
            
            result = TransactionResult(
                success=True,
                tx_hash=tx_hash_hex,
                status=TransactionStatus.SUBMITTED
            )
            
            if wait_for_confirmation:
                # Wait for confirmation
                receipt = self.w3.eth.wait_for_transaction_receipt(
                    tx_hash,
                    timeout=self.config.timeout
                )
                
                if receipt.status == 1:
                    result.status = TransactionStatus.CONFIRMED
                    result.block_number = receipt.blockNumber
                    result.gas_used = receipt.gasUsed
                else:
                    result.success = False
                    result.status = TransactionStatus.FAILED
                    result.error_message = "Transaction execution failed"
            
            return result
            
        except TimeExhausted:
            return TransactionResult(
                success=False,
                status=TransactionStatus.TIMEOUT,
                error_message="Transaction confirmation timeout"
            )
        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            return TransactionResult(
                success=False,
                status=TransactionStatus.FAILED,
                error_message=str(e)
            )
    
    async def approve_tokens(
        self,
        spender_address: str,
        amount: int,
        wait_for_confirmation: bool = True
    ) -> TransactionResult:
        """
        Approve spender to transfer tokens.
        
        Args:
            spender_address: Address to approve
            amount: Amount to approve
            wait_for_confirmation: Whether to wait for confirmation
            
        Returns:
            TransactionResult
        """
        if not self.account:
            raise RuntimeError("No account configured for transactions")
        
        contract = self.get_token_contract()
        if not contract:
            raise RuntimeError("Token contract not loaded")
        
        try:
            checksum_spender = Web3.to_checksum_address(spender_address)
            tx = contract.functions.approve(checksum_spender, amount).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
            })
            
            # Estimate gas and add buffer
            gas_estimate = self.w3.eth.estimate_gas(tx)
            tx['gas'] = int(gas_estimate * 1.2)
            tx['gasPrice'] = self.w3.eth.gas_price
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()
            
            result = TransactionResult(
                success=True,
                tx_hash=tx_hash_hex,
                status=TransactionStatus.SUBMITTED
            )
            
            if wait_for_confirmation:
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.config.timeout)
                if receipt.status == 1:
                    result.status = TransactionStatus.CONFIRMED
                    result.block_number = receipt.blockNumber
                    result.gas_used = receipt.gasUsed
                else:
                    result.success = False
                    result.status = TransactionStatus.FAILED
            
            return result
            
        except Exception as e:
            return TransactionResult(
                success=False,
                status=TransactionStatus.FAILED,
                error_message=str(e)
            )
    
    async def get_allowance(
        self,
        owner_address: str,
        spender_address: str
    ) -> int:
        """
        Get approved allowance.
        
        Args:
            owner_address: Token owner address
            spender_address: Spender address
            
        Returns:
            Approved amount in wei
        """
        contract = self.get_token_contract()
        if not contract:
            raise RuntimeError("Token contract not loaded")
        
        checksum_owner = Web3.to_checksum_address(owner_address)
        checksum_spender = Web3.to_checksum_address(spender_address)
        
        return contract.functions.allowance(checksum_owner, checksum_spender).call()
    
    # ============ Role Management ============
    
    async def grant_role(
        self,
        role: RoleType,
        account_address: str,
        wait_for_confirmation: bool = True
    ) -> TransactionResult:
        """
        Grant role to account.
        
        Args:
            role: Role to grant
            account_address: Address to grant role to
            wait_for_confirmation: Whether to wait for confirmation
            
        Returns:
            TransactionResult
        """
        if not self.account:
            raise RuntimeError("No account configured for transactions")
        
        contract = self.get_token_contract()
        if not contract:
            raise RuntimeError("Token contract not loaded")
        
        try:
            # Get role hash
            role_hash = self._get_role_hash(role)
            checksum_account = Web3.to_checksum_address(account_address)
            
            tx = contract.functions.grantRole(role_hash, checksum_account).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
            })
            
            gas_estimate = self.w3.eth.estimate_gas(tx)
            tx['gas'] = int(gas_estimate * 1.2)
            tx['gasPrice'] = self.w3.eth.gas_price
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()
            
            result = TransactionResult(
                success=True,
                tx_hash=tx_hash_hex,
                status=TransactionStatus.SUBMITTED
            )
            
            if wait_for_confirmation:
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.config.timeout)
                if receipt.status == 1:
                    result.status = TransactionStatus.CONFIRMED
                    result.block_number = receipt.blockNumber
                    result.gas_used = receipt.gasUsed
                else:
                    result.success = False
                    result.status = TransactionStatus.FAILED
            
            return result
            
        except Exception as e:
            return TransactionResult(
                success=False,
                status=TransactionStatus.FAILED,
                error_message=str(e)
            )
    
    async def has_role(
        self,
        role: RoleType,
        account_address: str
    ) -> bool:
        """
        Check if account has role.
        
        Args:
            role: Role to check
            account_address: Address to check
            
        Returns:
            True if account has role
        """
        contract = self.get_token_contract()
        if not contract:
            raise RuntimeError("Token contract not loaded")
        
        role_hash = self._get_role_hash(role)
        checksum_account = Web3.to_checksum_address(account_address)
        
        return contract.functions.hasRole(role_hash, checksum_account).call()
    
    def _get_role_hash(self, role: RoleType) -> bytes:
        """Get role hash"""
        if role == RoleType.DEFAULT_ADMIN:
            # Default admin role is 0x00
            return b'\x00' * 32
        else:
            # Other roles are keccak256 of role name
            role_name = role.value
            return Web3.keccak(text=role_name)
    
    # ============ Bridge Operations ============
    
    async def bridge_out(
        self,
        amount: int,
        destination_chain: int,
        wait_for_confirmation: bool = True
    ) -> TransactionResult:
        """
        Bridge tokens out to another chain.
        
        Args:
            amount: Amount to bridge in wei
            destination_chain: Destination chain ID
            wait_for_confirmation: Whether to wait for confirmation
            
        Returns:
            TransactionResult
        """
        if not self.account:
            raise RuntimeError("No account configured for transactions")
        
        bridge_contract = self.get_bridge_contract()
        if not bridge_contract:
            raise RuntimeError("Bridge contract not loaded")
        
        token_contract = self.get_token_contract()
        if not token_contract:
            raise RuntimeError("Token contract not loaded")
        
        try:
            # First approve bridge to spend tokens
            approve_result = await self.approve_tokens(
                bridge_contract.address,
                amount
            )
            if not approve_result.success:
                return approve_result
            
            # Execute bridge out
            tx = bridge_contract.functions.bridgeOut(amount, destination_chain).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
            })
            
            gas_estimate = self.w3.eth.estimate_gas(tx)
            tx['gas'] = int(gas_estimate * 1.2)
            tx['gasPrice'] = self.w3.eth.gas_price
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()
            
            result = TransactionResult(
                success=True,
                tx_hash=tx_hash_hex,
                status=TransactionStatus.SUBMITTED
            )
            
            if wait_for_confirmation:
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.config.timeout)
                if receipt.status == 1:
                    result.status = TransactionStatus.CONFIRMED
                    result.block_number = receipt.blockNumber
                    result.gas_used = receipt.gas_used
                else:
                    result.success = False
                    result.status = TransactionStatus.FAILED
            
            return result
            
        except Exception as e:
            return TransactionResult(
                success=False,
                status=TransactionStatus.FAILED,
                error_message=str(e)
            )
    
    async def get_bridge_limits(self) -> Dict[str, int]:
        """
        Get bridge amount limits.
        
        Returns:
            Dictionary with min and max amounts
        """
        bridge_contract = self.get_bridge_contract()
        if not bridge_contract:
            raise RuntimeError("Bridge contract not loaded")
        
        min_amount = bridge_contract.functions.minBridgeAmount().call()
        max_amount = bridge_contract.functions.maxBridgeAmount().call()
        fee_bps = bridge_contract.functions.bridgeFeeBps().call()
        
        return {
            "min_amount": min_amount,
            "max_amount": max_amount,
            "fee_bps": fee_bps,
        }
    
    # ============ Event Monitoring ============
    
    async def watch_transfer_events(
        self,
        callback: Callable[[TransferEvent], None],
        from_block: Optional[int] = None,
        to_block: Optional[int] = None
    ) -> None:
        """
        Watch for transfer events.
        
        Args:
            callback: Callback function for transfer events
            from_block: Starting block (defaults to latest)
            to_block: Ending block (defaults to ongoing)
        """
        contract = self.get_token_contract()
        if not contract:
            raise RuntimeError("Token contract not loaded")
        
        # Create event filter
        if from_block is None:
            from_block = self.w3.eth.block_number
        
        event_filter = contract.events.Transfer.create_filter(
            from_block=from_block,
            to_block=to_block or 'latest'
        )
        
        # Store filter
        filter_id = f"transfer_{id(callback)}"
        self._event_filters[filter_id] = event_filter
        
        if filter_id not in self._event_callbacks:
            self._event_callbacks[filter_id] = []
        self._event_callbacks[filter_id].append(callback)
        
        # Start watching
        async def watch_loop():
            try:
                while True:
                    for event in event_filter.get_new_entries():
                        transfer_event = TransferEvent(
                            from_address=event['args']['from'],
                            to_address=event['args']['to'],
                            amount=event['args']['value'],
                            tx_hash=event['transactionHash'].hex(),
                            block_number=event['blockNumber'],
                            timestamp=datetime.now(timezone.utc)
                        )
                        await callback(transfer_event)
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
        
        asyncio.create_task(watch_loop())
    
    async def stop_watching(self, filter_id: Optional[str] = None):
        """
        Stop watching events.
        
        Args:
            filter_id: Specific filter to stop (stops all if None)
        """
        if filter_id:
            if filter_id in self._event_filters:
                self._event_filters[filter_id].uninstall()
                del self._event_filters[filter_id]
                self._event_callbacks.pop(filter_id, None)
        else:
            for fid, event_filter in self._event_filters.items():
                event_filter.uninstall()
            self._event_filters.clear()
            self._event_callbacks.clear()
    
    # ============ Utility Methods ============
    
    def clear_cache(self):
        """Clear balance cache"""
        self._balance_cache.clear()
    
    async def get_transaction_receipt(
        self,
        tx_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get transaction receipt.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Receipt dictionary or None
        """
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            return dict(receipt)
        except Exception:
            return None
    
    async def is_transaction_confirmed(
        self,
        tx_hash: str,
        required_confirmations: int = 1
    ) -> bool:
        """
        Check if transaction is confirmed.
        
        Args:
            tx_hash: Transaction hash
            required_confirmations: Number of confirmations required
            
        Returns:
            True if confirmed
        """
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            if receipt is None:
                return False
            
            if receipt.status != 1:
                return False
            
            current_block = self.w3.eth.block_number
            confirmations = current_block - receipt.blockNumber
            
            return confirmations >= required_confirmations
            
        except Exception:
            return False
    
    async def estimate_gas(
        self,
        to_address: str,
        amount: int
    ) -> int:
        """
        Estimate gas for transfer.
        
        Args:
            to_address: Recipient address
            amount: Amount to transfer
            
        Returns:
            Estimated gas
        """
        contract = self.get_token_contract()
        if not contract:
            raise RuntimeError("Token contract not loaded")
        
        checksum_to = Web3.to_checksum_address(to_address)
        tx = contract.functions.transfer(checksum_to, amount).build_transaction({
            'from': self.account.address if self.account else None,
        })
        
        return self.w3.eth.estimate_gas(tx)
    
    async def get_gas_price(self) -> int:
        """
        Get current gas price.
        
        Returns:
            Gas price in wei
        """
        return self.w3.eth.gas_price
    
    async def get_native_balance(self, address: str) -> int:
        """
        Get native token balance (ETH/MATIC).
        
        Args:
            address: Address to query
            
        Returns:
            Balance in wei
        """
        checksum_address = Web3.to_checksum_address(address)
        return self.w3.eth.get_balance(checksum_address)