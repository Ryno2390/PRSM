"""
CHRONOS Multi-Signature Wallet Manager

Handles secure custody of FTNS, Bitcoin, and USD reserves.
"""

import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime
import hashlib
import json
from dataclasses import dataclass

from .models import AssetType


logger = logging.getLogger(__name__)


@dataclass
class WalletAddress:
    """Represents a wallet address for an asset."""
    asset_type: AssetType
    address: str
    is_multi_sig: bool
    required_signatures: int
    total_signers: int
    created_at: datetime


@dataclass
class WalletBalance:
    """Current balance information for a wallet."""
    asset_type: AssetType
    address: str
    available_balance: Decimal
    reserved_balance: Decimal  # Locked in pending transactions
    total_balance: Decimal
    last_updated: datetime


@dataclass
class MultiSigTransaction:
    """Multi-signature transaction pending approval."""
    id: str
    asset_type: AssetType
    from_address: str
    to_address: str
    amount: Decimal
    purpose: str  # Description of transaction purpose
    signatures: List[str]  # List of signature hashes
    required_signatures: int
    created_at: datetime
    expires_at: datetime
    is_executed: bool = False


class MultiSigWalletManager:
    """Manages multi-signature wallets for CHRONOS reserves."""
    
    def __init__(self):
        # Initialize mock wallet addresses
        self.wallets = self._initialize_mock_wallets()
        
        # Track balances
        self.balances = self._initialize_mock_balances()
        
        # Pending multi-sig transactions
        self.pending_transactions: Dict[str, MultiSigTransaction] = {}
        
        # Authorized signers (in production, these would be hardware keys)
        self.authorized_signers = [
            "signer_1_pubkey_hash",
            "signer_2_pubkey_hash", 
            "signer_3_pubkey_hash",
            "signer_4_pubkey_hash",
            "signer_5_pubkey_hash"
        ]
    
    def _initialize_mock_wallets(self) -> Dict[AssetType, WalletAddress]:
        """Initialize mock wallet addresses for testing."""
        wallets = {}
        
        # FTNS multi-sig wallet (3-of-5)
        wallets[AssetType.FTNS] = WalletAddress(
            asset_type=AssetType.FTNS,
            address="ftns_multisig_3of5_chronos_main",
            is_multi_sig=True,
            required_signatures=3,
            total_signers=5,
            created_at=datetime.utcnow()
        )
        
        # Bitcoin multi-sig wallet (3-of-5)
        wallets[AssetType.BTC] = WalletAddress(
            asset_type=AssetType.BTC,
            address="bc1qchronos_multisig_3of5_reserve_wallet",
            is_multi_sig=True,
            required_signatures=3,
            total_signers=5,
            created_at=datetime.utcnow()
        )
        
        # USD custody account (requires 4-of-5 for large amounts)
        wallets[AssetType.USD] = WalletAddress(
            asset_type=AssetType.USD,
            address="USD_CUSTODY_TIER1_BANK_ACCOUNT_001",
            is_multi_sig=True,
            required_signatures=4,
            total_signers=5,
            created_at=datetime.utcnow()
        )

        # USDC stablecoin wallet (3-of-5)
        wallets[AssetType.USDC] = WalletAddress(
            asset_type=AssetType.USDC,
            address="usdc_multisig_3of5_chronos_reserve",
            is_multi_sig=True,
            required_signatures=3,
            total_signers=5,
            created_at=datetime.utcnow()
        )

        # USDT stablecoin wallet (3-of-5)
        wallets[AssetType.USDT] = WalletAddress(
            asset_type=AssetType.USDT,
            address="usdt_multisig_3of5_chronos_reserve",
            is_multi_sig=True,
            required_signatures=3,
            total_signers=5,
            created_at=datetime.utcnow()
        )

        # ETH wallet (3-of-5)
        wallets[AssetType.ETH] = WalletAddress(
            asset_type=AssetType.ETH,
            address="0xchronos_multisig_3of5_eth_reserve",
            is_multi_sig=True,
            required_signatures=3,
            total_signers=5,
            created_at=datetime.utcnow()
        )

        # ADA wallet (3-of-5)
        wallets[AssetType.ADA] = WalletAddress(
            asset_type=AssetType.ADA,
            address="addr_chronos_multisig_3of5_ada_reserve",
            is_multi_sig=True,
            required_signatures=3,
            total_signers=5,
            created_at=datetime.utcnow()
        )

        # SOL wallet (3-of-5)
        wallets[AssetType.SOL] = WalletAddress(
            asset_type=AssetType.SOL,
            address="sol_chronos_multisig_3of5_reserve",
            is_multi_sig=True,
            required_signatures=3,
            total_signers=5,
            created_at=datetime.utcnow()
        )

        # DOT wallet (3-of-5)
        wallets[AssetType.DOT] = WalletAddress(
            asset_type=AssetType.DOT,
            address="dot_chronos_multisig_3of5_reserve",
            is_multi_sig=True,
            required_signatures=3,
            total_signers=5,
            created_at=datetime.utcnow()
        )

        return wallets
    
    def _initialize_mock_balances(self) -> Dict[AssetType, WalletBalance]:
        """Initialize mock balances for testing."""
        balances = {}
        
        # FTNS reserves - 10M tokens for liquidity
        balances[AssetType.FTNS] = WalletBalance(
            asset_type=AssetType.FTNS,
            address=self.wallets[AssetType.FTNS].address,
            available_balance=Decimal("9500000"),  # 9.5M available
            reserved_balance=Decimal("500000"),    # 500K reserved
            total_balance=Decimal("10000000"),     # 10M total
            last_updated=datetime.utcnow()
        )
        
        # Bitcoin reserves - 100 BTC for liquidity
        balances[AssetType.BTC] = WalletBalance(
            asset_type=AssetType.BTC,
            address=self.wallets[AssetType.BTC].address,
            available_balance=Decimal("95.5"),    # 95.5 BTC available
            reserved_balance=Decimal("4.5"),      # 4.5 BTC reserved
            total_balance=Decimal("100"),         # 100 BTC total
            last_updated=datetime.utcnow()
        )
        
        # USD reserves - $5M for liquidity
        balances[AssetType.USD] = WalletBalance(
            asset_type=AssetType.USD,
            address=self.wallets[AssetType.USD].address,
            available_balance=Decimal("4750000"),  # $4.75M available
            reserved_balance=Decimal("250000"),    # $250K reserved
            total_balance=Decimal("5000000"),      # $5M total
            last_updated=datetime.utcnow()
        )

        # USDC reserves - $2M for stablecoin liquidity
        balances[AssetType.USDC] = WalletBalance(
            asset_type=AssetType.USDC,
            address=self.wallets[AssetType.USDC].address,
            available_balance=Decimal("1900000"),
            reserved_balance=Decimal("100000"),
            total_balance=Decimal("2000000"),
            last_updated=datetime.utcnow()
        )

        # USDT reserves - $1M for stablecoin liquidity
        balances[AssetType.USDT] = WalletBalance(
            asset_type=AssetType.USDT,
            address=self.wallets[AssetType.USDT].address,
            available_balance=Decimal("950000"),
            reserved_balance=Decimal("50000"),
            total_balance=Decimal("1000000"),
            last_updated=datetime.utcnow()
        )

        # ETH reserves - 500 ETH for liquidity
        balances[AssetType.ETH] = WalletBalance(
            asset_type=AssetType.ETH,
            address=self.wallets[AssetType.ETH].address,
            available_balance=Decimal("475"),
            reserved_balance=Decimal("25"),
            total_balance=Decimal("500"),
            last_updated=datetime.utcnow()
        )

        # ADA reserves - 500K ADA for liquidity
        balances[AssetType.ADA] = WalletBalance(
            asset_type=AssetType.ADA,
            address=self.wallets[AssetType.ADA].address,
            available_balance=Decimal("475000"),
            reserved_balance=Decimal("25000"),
            total_balance=Decimal("500000"),
            last_updated=datetime.utcnow()
        )

        # SOL reserves - 5000 SOL for liquidity
        balances[AssetType.SOL] = WalletBalance(
            asset_type=AssetType.SOL,
            address=self.wallets[AssetType.SOL].address,
            available_balance=Decimal("4750"),
            reserved_balance=Decimal("250"),
            total_balance=Decimal("5000"),
            last_updated=datetime.utcnow()
        )

        # DOT reserves - 50K DOT for liquidity
        balances[AssetType.DOT] = WalletBalance(
            asset_type=AssetType.DOT,
            address=self.wallets[AssetType.DOT].address,
            available_balance=Decimal("47500"),
            reserved_balance=Decimal("2500"),
            total_balance=Decimal("50000"),
            last_updated=datetime.utcnow()
        )

        return balances
    
    async def get_balance(self, asset_type: AssetType) -> WalletBalance:
        """Get current balance for an asset type."""
        if asset_type not in self.balances:
            raise ValueError(f"No wallet found for asset type: {asset_type}")
        
        return self.balances[asset_type]
    
    async def reserve_funds(self, asset_type: AssetType, amount: Decimal, purpose: str) -> bool:
        """Reserve funds for a pending transaction."""
        balance = self.balances.get(asset_type)
        if not balance:
            return False
        
        if balance.available_balance < amount:
            logger.warning(f"Insufficient funds to reserve {amount} {asset_type}")
            return False
        
        # Move funds from available to reserved
        balance.available_balance -= amount
        balance.reserved_balance += amount
        balance.last_updated = datetime.utcnow()
        
        logger.info(f"Reserved {amount} {asset_type} for: {purpose}")
        return True
    
    async def release_funds(self, asset_type: AssetType, amount: Decimal, purpose: str) -> bool:
        """Release reserved funds back to available."""
        balance = self.balances.get(asset_type)
        if not balance:
            return False
        
        if balance.reserved_balance < amount:
            logger.warning(f"Insufficient reserved funds to release {amount} {asset_type}")
            return False
        
        # Move funds from reserved back to available
        balance.reserved_balance -= amount
        balance.available_balance += amount
        balance.last_updated = datetime.utcnow()
        
        logger.info(f"Released {amount} {asset_type} from: {purpose}")
        return True
    
    async def transfer_funds(
        self, 
        asset_type: AssetType, 
        to_address: str, 
        amount: Decimal,
        purpose: str
    ) -> str:
        """Initiate a multi-sig transfer (returns transaction ID)."""
        wallet = self.wallets.get(asset_type)
        if not wallet:
            raise ValueError(f"No wallet configured for {asset_type}")
        
        balance = self.balances.get(asset_type)
        if not balance or balance.reserved_balance < amount:
            raise ValueError(f"Insufficient reserved funds for transfer")
        
        # Create multi-sig transaction
        tx_id = f"tx_{asset_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        multisig_tx = MultiSigTransaction(
            id=tx_id,
            asset_type=asset_type,
            from_address=wallet.address,
            to_address=to_address,
            amount=amount,
            purpose=purpose,
            signatures=[],
            required_signatures=wallet.required_signatures,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow().replace(hour=23, minute=59, second=59)  # End of day
        )
        
        self.pending_transactions[tx_id] = multisig_tx
        
        logger.info(f"Created multi-sig transaction {tx_id} for {amount} {asset_type}")
        return tx_id
    
    async def sign_transaction(self, tx_id: str, signer_id: str) -> bool:
        """Add signature to a pending multi-sig transaction."""
        if tx_id not in self.pending_transactions:
            logger.error(f"Transaction {tx_id} not found")
            return False
        
        if signer_id not in self.authorized_signers:
            logger.error(f"Signer {signer_id} not authorized")
            return False
        
        tx = self.pending_transactions[tx_id]
        
        if datetime.utcnow() > tx.expires_at:
            logger.error(f"Transaction {tx_id} has expired")
            return False
        
        if signer_id in tx.signatures:
            logger.warning(f"Signer {signer_id} already signed transaction {tx_id}")
            return True
        
        # Add signature
        tx.signatures.append(signer_id)
        
        logger.info(f"Transaction {tx_id} signed by {signer_id} ({len(tx.signatures)}/{tx.required_signatures})")
        
        # Check if we have enough signatures to execute
        if len(tx.signatures) >= tx.required_signatures:
            return await self._execute_multisig_transaction(tx_id)
        
        return True
    
    async def _execute_multisig_transaction(self, tx_id: str) -> bool:
        """Execute a multi-sig transaction once enough signatures are collected."""
        tx = self.pending_transactions.get(tx_id)
        if not tx or tx.is_executed:
            return False
        
        try:
            # In real implementation, this would broadcast to blockchain
            balance = self.balances[tx.asset_type]
            
            # Deduct from reserved balance
            balance.reserved_balance -= tx.amount
            balance.total_balance -= tx.amount
            balance.last_updated = datetime.utcnow()
            
            # Mark transaction as executed
            tx.is_executed = True
            
            logger.info(f"Executed multi-sig transaction {tx_id}: {tx.amount} {tx.asset_type} to {tx.to_address}")
            
            # Simulate blockchain transaction hash
            return f"blockchain_tx_{tx_id}_{hash(tx.to_address) % 1000000}"
            
        except Exception as e:
            logger.error(f"Failed to execute transaction {tx_id}: {e}")
            return False
    
    async def get_wallet_info(self, asset_type: AssetType) -> Dict:
        """Get comprehensive wallet information."""
        wallet = self.wallets.get(asset_type)
        balance = self.balances.get(asset_type)
        
        if not wallet or not balance:
            return {"error": f"No wallet configured for {asset_type}"}
        
        return {
            "asset_type": asset_type.value,
            "address": wallet.address,
            "is_multi_sig": wallet.is_multi_sig,
            "required_signatures": wallet.required_signatures,
            "total_signers": wallet.total_signers,
            "available_balance": str(balance.available_balance),
            "reserved_balance": str(balance.reserved_balance),
            "total_balance": str(balance.total_balance),
            "last_updated": balance.last_updated.isoformat(),
            "pending_transactions": len([
                tx for tx in self.pending_transactions.values() 
                if tx.asset_type == asset_type and not tx.is_executed
            ])
        }
    
    async def get_pending_transactions(self, asset_type: Optional[AssetType] = None) -> List[Dict]:
        """Get list of pending multi-sig transactions."""
        pending = []
        
        for tx in self.pending_transactions.values():
            if asset_type and tx.asset_type != asset_type:
                continue
                
            if tx.is_executed:
                continue
                
            pending.append({
                "id": tx.id,
                "asset_type": tx.asset_type.value,
                "from_address": tx.from_address,
                "to_address": tx.to_address,
                "amount": str(tx.amount),
                "purpose": tx.purpose,
                "signatures_count": len(tx.signatures),
                "required_signatures": tx.required_signatures,
                "created_at": tx.created_at.isoformat(),
                "expires_at": tx.expires_at.isoformat(),
                "is_ready": len(tx.signatures) >= tx.required_signatures
            })
        
        return pending