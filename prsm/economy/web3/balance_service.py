"""
FTNS Balance Checking and Transaction History Service

Provides comprehensive balance management and transaction history
with real-time updates, caching, and detailed analytics.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from prsm.core.database_service import DatabaseService
from prsm.core.models import FTNSWallet, FTNSTransaction
from .wallet_connector import Web3WalletConnector
from .contract_interface import FTNSContractInterface, TokenBalance

logger = logging.getLogger(__name__)

@dataclass
class BalanceSnapshot:
    address: str
    liquid_balance: Decimal
    locked_balance: Decimal
    staked_balance: Decimal
    total_balance: Decimal
    context_allocated: Decimal
    last_updated: datetime
    block_number: int

@dataclass
class TransactionSummary:
    total_transactions: int
    total_sent: Decimal
    total_received: Decimal
    total_fees_paid: Decimal
    first_transaction: Optional[datetime]
    last_transaction: Optional[datetime]
    unique_counterparties: int

@dataclass
class BalanceChange:
    address: str
    previous_balance: Decimal
    new_balance: Decimal
    change_amount: Decimal
    change_type: str
    timestamp: datetime
    transaction_hash: Optional[str] = None

class FTNSBalanceService:
    """
    Comprehensive FTNS balance and transaction management service
    
    Features:
    - Real-time balance tracking and updates
    - Historical balance snapshots
    - Transaction history with filtering and pagination
    - Balance change notifications
    - Analytics and reporting
    - Caching for performance optimization
    """
    
    def __init__(self,
                 wallet_connector: Web3WalletConnector,
                 contract_interface: FTNSContractInterface,
                 db_service: DatabaseService):
        self.wallet = wallet_connector
        self.contracts = contract_interface
        self.db_service = db_service
        
        # Caching
        self.balance_cache: Dict[str, BalanceSnapshot] = {}
        self.cache_ttl = 30  # 30 seconds cache TTL
        
        # Balance change tracking
        self.balance_change_listeners: List[callable] = []
        
    async def get_balance(self, address: str, force_refresh: bool = False) -> Optional[BalanceSnapshot]:
        """
        Get comprehensive balance information for address
        
        Args:
            address: Wallet address
            force_refresh: Force refresh from blockchain
            
        Returns:
            BalanceSnapshot: Current balance information
        """
        try:
            address = address.lower()
            
            # Check cache first (unless force refresh)
            if not force_refresh and address in self.balance_cache:
                cached_balance = self.balance_cache[address]
                cache_age = (datetime.utcnow() - cached_balance.last_updated).total_seconds()
                
                if cache_age < self.cache_ttl:
                    return cached_balance
            
            # Get balance from contract
            contract_balance = await self.contracts.get_detailed_balance(address)
            if not contract_balance:
                # Fallback to simple balance
                simple_balance = await self.contracts.get_balance(address)
                contract_balance = TokenBalance(
                    liquid=simple_balance,
                    locked=Decimal('0'),
                    staked=Decimal('0'),
                    total=simple_balance,
                    context_allocated=Decimal('0')
                )
            
            # Get current block number
            current_block = self.wallet.w3.eth.block_number
            
            # Create snapshot
            snapshot = BalanceSnapshot(
                address=address,
                liquid_balance=contract_balance.liquid,
                locked_balance=contract_balance.locked,
                staked_balance=contract_balance.staked,
                total_balance=contract_balance.total,
                context_allocated=contract_balance.context_allocated,
                last_updated=datetime.utcnow(),
                block_number=current_block
            )
            
            # Update cache
            old_snapshot = self.balance_cache.get(address)
            self.balance_cache[address] = snapshot
            
            # Check for balance changes and notify listeners
            if old_snapshot and old_snapshot.total_balance != snapshot.total_balance:
                change = BalanceChange(
                    address=address,
                    previous_balance=old_snapshot.total_balance,
                    new_balance=snapshot.total_balance,
                    change_amount=snapshot.total_balance - old_snapshot.total_balance,
                    change_type="update",
                    timestamp=datetime.utcnow()
                )
                await self._notify_balance_change(change)
            
            # Update database wallet record
            await self._update_wallet_record(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to get balance for {address}: {e}")
            return None
    
    async def get_multiple_balances(self, addresses: List[str]) -> Dict[str, BalanceSnapshot]:
        """
        Get balances for multiple addresses efficiently
        
        Args:
            addresses: List of wallet addresses
            
        Returns:
            Dict: Address -> BalanceSnapshot mapping
        """
        try:
            # Process addresses concurrently
            tasks = [self.get_balance(addr) for addr in addresses]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            balances = {}
            for addr, result in zip(addresses, results):
                if isinstance(result, BalanceSnapshot):
                    balances[addr.lower()] = result
                elif isinstance(result, Exception):
                    logger.error(f"Failed to get balance for {addr}: {result}")
                    
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get multiple balances: {e}")
            return {}
    
    async def get_transaction_history(self,
                                     address: str,
                                     limit: int = 50,
                                     offset: int = 0,
                                     transaction_type: Optional[str] = None,
                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None) -> Tuple[List[Dict], int]:
        """
        Get paginated transaction history for address
        
        Args:
            address: Wallet address
            limit: Maximum number of transactions to return
            offset: Number of transactions to skip
            transaction_type: Filter by transaction type
            start_date: Filter transactions after this date
            end_date: Filter transactions before this date
            
        Returns:
            Tuple: (transactions list, total count)
        """
        try:
            address = address.lower()
            
            # Get transactions from database
            transactions = await self.db_service.get_ftns_transactions_by_address(
                address=address,
                limit=limit,
                offset=offset,
                transaction_type=transaction_type,
                start_time=start_date,
                end_time=end_date
            )
            
            # Get total count
            total_count = await self.db_service.count_ftns_transactions_by_address(
                address=address,
                transaction_type=transaction_type,
                start_time=start_date,
                end_time=end_date
            )
            
            # Format transactions
            formatted_transactions = []
            for tx in transactions:
                formatted_tx = {
                    "hash": tx.transaction_hash,
                    "from_address": tx.from_address,
                    "to_address": tx.to_address,
                    "amount": float(tx.amount),
                    "type": tx.transaction_type,
                    "status": tx.status,
                    "timestamp": tx.timestamp.isoformat(),
                    "block_number": tx.block_number,
                    "gas_used": tx.gas_used,
                    "gas_price": tx.gas_price,
                    "direction": "sent" if tx.from_address.lower() == address else "received",
                    "metadata": tx.transaction_metadata
                }
                formatted_transactions.append(formatted_tx)
            
            return formatted_transactions, total_count
            
        except Exception as e:
            logger.error(f"Failed to get transaction history for {address}: {e}")
            return [], 0
    
    async def get_transaction_summary(self, address: str, days: int = 30) -> TransactionSummary:
        """
        Get transaction summary statistics for address
        
        Args:
            address: Wallet address
            days: Number of days to analyze
            
        Returns:
            TransactionSummary: Summary statistics
        """
        try:
            address = address.lower()
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get transactions in time range
            transactions, total_count = await self.get_transaction_history(
                address=address,
                limit=10000,  # Get all transactions in period
                start_date=start_date,
                end_date=end_date
            )
            
            if not transactions:
                return TransactionSummary(
                    total_transactions=0,
                    total_sent=Decimal('0'),
                    total_received=Decimal('0'),
                    total_fees_paid=Decimal('0'),
                    first_transaction=None,
                    last_transaction=None,
                    unique_counterparties=0
                )
            
            # Calculate statistics
            total_sent = Decimal('0')
            total_received = Decimal('0')
            total_fees = Decimal('0')
            counterparties = set()
            
            first_tx_time = None
            last_tx_time = None
            
            for tx in transactions:
                tx_time = datetime.fromisoformat(tx["timestamp"])
                
                if first_tx_time is None or tx_time < first_tx_time:
                    first_tx_time = tx_time
                if last_tx_time is None or tx_time > last_tx_time:
                    last_tx_time = tx_time
                
                amount = Decimal(str(tx["amount"]))
                
                if tx["direction"] == "sent":
                    total_sent += amount
                    counterparties.add(tx["to_address"])
                else:
                    total_received += amount
                    counterparties.add(tx["from_address"])
                
                # Add gas fees for sent transactions
                if tx["direction"] == "sent":
                    gas_fee = Decimal(tx["gas_used"] * tx["gas_price"]) / (10**18)
                    total_fees += gas_fee
            
            return TransactionSummary(
                total_transactions=len(transactions),
                total_sent=total_sent,
                total_received=total_received,
                total_fees_paid=total_fees,
                first_transaction=first_tx_time,
                last_transaction=last_tx_time,
                unique_counterparties=len(counterparties)
            )
            
        except Exception as e:
            logger.error(f"Failed to get transaction summary for {address}: {e}")
            return TransactionSummary(
                total_transactions=0,
                total_sent=Decimal('0'),
                total_received=Decimal('0'),
                total_fees_paid=Decimal('0'),
                first_transaction=None,
                last_transaction=None,
                unique_counterparties=0
            )
    
    async def get_balance_history(self, 
                                 address: str,
                                 days: int = 30,
                                 granularity: str = "daily") -> List[Dict]:
        """
        Get historical balance data
        
        Args:
            address: Wallet address
            days: Number of days to look back
            granularity: Time granularity (hourly, daily, weekly)
            
        Returns:
            List[Dict]: Historical balance data points
        """
        try:
            address = address.lower()
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get all transactions in time range
            transactions, _ = await self.get_transaction_history(
                address=address,
                limit=10000,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get current balance
            current_snapshot = await self.get_balance(address)
            if not current_snapshot:
                return []
            
            current_balance = current_snapshot.total_balance
            
            # Calculate historical balances by working backwards
            balance_history = []
            
            # Sort transactions by timestamp (newest first)
            transactions.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Set time intervals based on granularity
            if granularity == "hourly":
                interval = timedelta(hours=1)
            elif granularity == "weekly":
                interval = timedelta(weeks=1)
            else:  # daily
                interval = timedelta(days=1)
            
            # Generate time points
            time_points = []
            current_time = end_date
            while current_time >= start_date:
                time_points.append(current_time)
                current_time -= interval
            
            # Calculate balance at each time point
            running_balance = current_balance
            tx_index = 0
            
            for time_point in time_points:
                # Subtract transactions that happened after this time point
                while (tx_index < len(transactions) and 
                       datetime.fromisoformat(transactions[tx_index]["timestamp"]) > time_point):
                    
                    tx = transactions[tx_index]
                    amount = Decimal(str(tx["amount"]))
                    
                    if tx["direction"] == "sent":
                        running_balance += amount  # Add back sent amount
                    else:
                        running_balance -= amount  # Subtract received amount
                    
                    tx_index += 1
                
                balance_history.append({
                    "timestamp": time_point.isoformat(),
                    "balance": float(running_balance),
                    "date": time_point.strftime("%Y-%m-%d"),
                    "hour": time_point.hour if granularity == "hourly" else None
                })
            
            # Reverse to get chronological order
            balance_history.reverse()
            
            return balance_history
            
        except Exception as e:
            logger.error(f"Failed to get balance history for {address}: {e}")
            return []
    
    async def add_balance_change_listener(self, callback: callable):
        """Add listener for balance changes"""
        self.balance_change_listeners.append(callback)
    
    async def remove_balance_change_listener(self, callback: callable):
        """Remove balance change listener"""
        if callback in self.balance_change_listeners:
            self.balance_change_listeners.remove(callback)
    
    async def _notify_balance_change(self, change: BalanceChange):
        """Notify all listeners of balance change"""
        for listener in self.balance_change_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(change)
                else:
                    listener(change)
            except Exception as e:
                logger.error(f"Balance change listener error: {e}")
    
    async def _update_wallet_record(self, snapshot: BalanceSnapshot):
        """Update wallet record in database"""
        try:
            # Get existing wallet or create new one
            wallet = await self.db_service.get_ftns_wallet_by_address(snapshot.address)
            
            if not wallet:
                wallet = FTNSWallet(
                    address=snapshot.address,
                    balance=snapshot.total_balance,
                    locked_balance=snapshot.locked_balance,
                    staked_balance=snapshot.staked_balance,
                    last_activity=snapshot.last_updated
                )
                await self.db_service.create_ftns_wallet(wallet)
            else:
                # Update existing wallet
                wallet.balance = snapshot.total_balance
                wallet.locked_balance = snapshot.locked_balance
                wallet.staked_balance = snapshot.staked_balance
                wallet.last_activity = snapshot.last_updated
                await self.db_service.update_ftns_wallet(wallet)
                
        except Exception as e:
            logger.error(f"Failed to update wallet record: {e}")
    
    async def get_top_holders(self, limit: int = 100) -> List[Dict]:
        """
        Get top token holders by balance
        
        Args:
            limit: Maximum number of holders to return
            
        Returns:
            List[Dict]: Top holders information
        """
        try:
            wallets = await self.db_service.get_top_ftns_wallets(limit)
            
            holders = []
            for wallet in wallets:
                holders.append({
                    "address": wallet.address,
                    "balance": float(wallet.balance),
                    "locked_balance": float(wallet.locked_balance or 0),
                    "staked_balance": float(wallet.staked_balance or 0),
                    "last_activity": wallet.last_activity.isoformat() if wallet.last_activity else None
                })
            
            return holders
            
        except Exception as e:
            logger.error(f"Failed to get top holders: {e}")
            return []
    
    async def get_network_statistics(self) -> Dict[str, Any]:
        """
        Get network-wide token statistics
        
        Returns:
            Dict: Network statistics
        """
        try:
            # Get token info from contract
            token_info = await self.contracts.get_token_info()
            
            # Get wallet statistics from database
            wallet_stats = await self.db_service.get_ftns_wallet_statistics()
            
            # Get transaction statistics
            tx_stats = await self.db_service.get_ftns_transaction_statistics()
            
            return {
                "token_info": token_info,
                "total_wallets": wallet_stats.get("total_wallets", 0),
                "active_wallets_24h": wallet_stats.get("active_24h", 0),
                "total_transactions": tx_stats.get("total_transactions", 0),
                "transactions_24h": tx_stats.get("transactions_24h", 0),
                "total_volume_24h": float(tx_stats.get("volume_24h", 0)),
                "average_transaction_value": float(tx_stats.get("avg_transaction_value", 0)),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get network statistics: {e}")
            return {}
    
    def clear_cache(self, address: Optional[str] = None):
        """Clear balance cache"""
        if address:
            self.balance_cache.pop(address.lower(), None)
        else:
            self.balance_cache.clear()
        
        logger.info(f"Cleared balance cache for {address or 'all addresses'}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_cached = len(self.balance_cache)
        cache_hits = sum(1 for snapshot in self.balance_cache.values() 
                        if (datetime.utcnow() - snapshot.last_updated).total_seconds() < self.cache_ttl)
        
        return {
            "total_cached_addresses": total_cached,
            "valid_cache_entries": cache_hits,
            "cache_ttl_seconds": self.cache_ttl,
            "cache_hit_rate": cache_hits / total_cached if total_cached > 0 else 0
        }