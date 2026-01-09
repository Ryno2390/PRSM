"""
Smart Contract Event Monitoring and Logging System

Provides comprehensive monitoring of FTNS smart contract events
with real-time processing, logging, and database integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import json

from web3 import Web3
from web3.contract import Contract
from web3.exceptions import BlockNotFound
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database_service import DatabaseService
from ..core.models import FTNSTransaction, FTNSWallet, Session as PRSMSession
from .wallet_connector import Web3WalletConnector
from .contract_interface import FTNSContractInterface

logger = logging.getLogger(__name__)

@dataclass
class EventFilter:
    contract_name: str
    event_name: str
    from_block: int
    to_block: str = "latest"
    address_filter: Optional[str] = None
    active: bool = True

@dataclass
class ProcessedEvent:
    contract_address: str
    contract_name: str
    event_name: str
    block_number: int
    transaction_hash: str
    log_index: int
    args: Dict[str, Any]
    timestamp: datetime
    processed_at: datetime

class EventProcessor:
    """Base class for processing specific event types"""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        
    async def process(self, event: ProcessedEvent) -> bool:
        """
        Process a specific event
        
        Args:
            event: Event to process
            
        Returns:
            bool: True if processing successful
        """
        raise NotImplementedError

class TransferEventProcessor(EventProcessor):
    """Processes FTNS token transfer events"""
    
    async def process(self, event: ProcessedEvent) -> bool:
        try:
            args = event.args
            
            # Extract transfer details
            from_address = args.get("from", "").lower()
            to_address = args.get("to", "").lower()
            value = args.get("value", 0)
            
            # Convert value to FTNS (assuming 18 decimals)
            amount = Decimal(value) / (10 ** 18)
            
            # Create transaction record
            transaction = FTNSTransaction(
                transaction_hash=event.transaction_hash,
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                transaction_type="transfer",
                status="confirmed",
                block_number=event.block_number,
                timestamp=event.timestamp,
                gas_used=0,  # Will be filled by transaction receipt
                gas_price=0,
                transaction_metadata={
                    "event_log_index": event.log_index,
                    "contract_address": event.contract_address
                }
            )
            
            # Save to database
            await self.db_service.create_ftns_transaction(transaction)
            
            # Update wallet balances
            await self._update_wallet_balance(from_address, -amount)
            await self._update_wallet_balance(to_address, amount)
            
            logger.info(f"Processed transfer: {amount} FTNS from {from_address} to {to_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process transfer event: {e}")
            return False
    
    async def _update_wallet_balance(self, address: str, amount_change: Decimal):
        """Update wallet balance in database"""
        try:
            # Get or create wallet
            wallet = await self.db_service.get_ftns_wallet_by_address(address)
            if not wallet:
                wallet = FTNSWallet(
                    address=address,
                    balance=Decimal('0'),
                    locked_balance=Decimal('0'),
                    staked_balance=Decimal('0')
                )
                await self.db_service.create_ftns_wallet(wallet)
            
            # Update balance
            wallet.balance += amount_change
            if wallet.balance < 0:
                wallet.balance = Decimal('0')
                
            await self.db_service.update_ftns_wallet(wallet)
            
        except Exception as e:
            logger.error(f"Failed to update wallet balance for {address}: {e}")

class ApprovalEventProcessor(EventProcessor):
    """Processes FTNS token approval events"""
    
    async def process(self, event: ProcessedEvent) -> bool:
        try:
            args = event.args
            
            owner = args.get("owner", "").lower()
            spender = args.get("spender", "").lower()
            value = args.get("value", 0)
            amount = Decimal(value) / (10 ** 18)
            
            # Log approval for audit purposes
            transaction = FTNSTransaction(
                transaction_hash=event.transaction_hash,
                from_address=owner,
                to_address=spender,
                amount=amount,
                transaction_type="approval",
                status="confirmed",
                block_number=event.block_number,
                timestamp=event.timestamp,
                gas_used=0,
                gas_price=0,
                transaction_metadata={
                    "event_log_index": event.log_index,
                    "contract_address": event.contract_address,
                    "approval_amount": str(amount)
                }
            )
            
            await self.db_service.create_ftns_transaction(transaction)
            
            logger.info(f"Processed approval: {owner} approved {amount} FTNS for {spender}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process approval event: {e}")
            return False

class Web3EventMonitor:
    """
    Comprehensive Web3 event monitoring system
    
    Features:
    - Real-time event monitoring from multiple contracts
    - Automatic event processing and database storage
    - Configurable event filters and processors
    - Error handling and retry mechanisms
    - Performance monitoring and metrics
    """
    
    def __init__(self, 
                 wallet_connector: Web3WalletConnector,
                 contract_interface: FTNSContractInterface,
                 db_service: DatabaseService):
        self.wallet = wallet_connector
        self.contracts = contract_interface
        self.db_service = db_service
        
        # Event processing
        self.event_filters: List[EventFilter] = []
        self.event_processors: Dict[str, EventProcessor] = {}
        self.processed_events: Set[str] = set()  # Track processed events
        
        # Monitoring state
        self.is_running = False
        self.last_processed_block = 0
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Performance metrics
        self.events_processed = 0
        self.errors_encountered = 0
        self.last_activity = datetime.utcnow()
        
        self._setup_default_processors()
    
    def _setup_default_processors(self):
        """Setup default event processors"""
        self.event_processors["Transfer"] = TransferEventProcessor(self.db_service)
        self.event_processors["Approval"] = ApprovalEventProcessor(self.db_service)
    
    async def add_contract_monitor(self, 
                                  contract_name: str,
                                  events: List[str],
                                  from_block: Optional[int] = None) -> bool:
        """
        Add contract event monitoring
        
        Args:
            contract_name: Name of contract to monitor
            events: List of event names to monitor
            from_block: Starting block number (uses latest if None)
            
        Returns:
            bool: True if monitoring added successfully
        """
        try:
            # Get current block if from_block not specified
            if from_block is None:
                from_block = self.wallet.w3.eth.block_number
            
            # Add event filters for each event
            for event_name in events:
                event_filter = EventFilter(
                    contract_name=contract_name,
                    event_name=event_name,
                    from_block=from_block
                )
                self.event_filters.append(event_filter)
                
            logger.info(f"Added monitoring for {contract_name} events: {events}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add contract monitoring: {e}")
            return False
    
    async def start_monitoring(self):
        """Start event monitoring"""
        if self.is_running:
            logger.warning("Event monitoring already running")
            return
            
        self.is_running = True
        logger.info("Starting Web3 event monitoring")
        
        # Start monitoring tasks
        for event_filter in self.event_filters:
            if event_filter.active:
                task = asyncio.create_task(
                    self._monitor_events(event_filter)
                )
                self.monitoring_tasks.append(task)
        
        # Start metrics reporting task
        metrics_task = asyncio.create_task(self._report_metrics())
        self.monitoring_tasks.append(metrics_task)
    
    async def stop_monitoring(self):
        """Stop event monitoring"""
        if not self.is_running:
            return
            
        self.is_running = False
        logger.info("Stopping Web3 event monitoring")
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
    
    async def _monitor_events(self, event_filter: EventFilter):
        """Monitor events for specific filter"""
        try:
            contract = self.contracts.contracts.get(event_filter.contract_name)
            if not contract:
                logger.error(f"Contract {event_filter.contract_name} not found")
                return
                
            last_checked_block = event_filter.from_block
            
            while self.is_running and event_filter.active:
                try:
                    # Get current block
                    current_block = self.wallet.w3.eth.block_number
                    
                    if current_block > last_checked_block:
                        # Get events from last checked block to current
                        await self._process_block_range(
                            contract, 
                            event_filter,
                            last_checked_block + 1,
                            current_block
                        )
                        
                        last_checked_block = current_block
                        self.last_processed_block = current_block
                        self.last_activity = datetime.utcnow()
                    
                    # Wait before next check
                    await asyncio.sleep(2)  # Check every 2 seconds
                    
                except Exception as e:
                    logger.error(f"Error monitoring {event_filter.event_name}: {e}")
                    self.errors_encountered += 1
                    await asyncio.sleep(10)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Event monitoring failed for {event_filter.event_name}: {e}")
    
    async def _process_block_range(self, 
                                  contract: Contract,
                                  event_filter: EventFilter,
                                  from_block: int,
                                  to_block: int):
        """Process events in block range"""
        try:
            # Get event logs
            event_logs = await self._get_event_logs(
                contract,
                event_filter.event_name,
                from_block,
                to_block,
                event_filter.address_filter
            )
            
            # Process each event
            for log in event_logs:
                await self._process_event_log(log, event_filter.contract_name)
                
        except Exception as e:
            logger.error(f"Failed to process block range {from_block}-{to_block}: {e}")
    
    async def _get_event_logs(self,
                             contract: Contract,
                             event_name: str,
                             from_block: int,
                             to_block: int,
                             address_filter: Optional[str] = None) -> List:
        """Get event logs from contract"""
        try:
            # Get event signature
            event = getattr(contract.events, event_name, None)
            if not event:
                logger.error(f"Event {event_name} not found in contract")
                return []
            
            # Create filter arguments
            filter_args = {
                "fromBlock": from_block,
                "toBlock": to_block
            }
            
            if address_filter:
                # Add address filter if specified
                filter_args["argument_filters"] = {"from": address_filter}
            
            # Get event logs
            event_filter = event.create_filter(**filter_args)
            logs = event_filter.get_all_entries()
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get event logs for {event_name}: {e}")
            return []
    
    async def _process_event_log(self, log, contract_name: str):
        """Process individual event log"""
        try:
            # Create unique event ID
            event_id = f"{log['transactionHash'].hex()}_{log['logIndex']}"
            
            # Skip if already processed
            if event_id in self.processed_events:
                return
            
            # Get block timestamp
            block = self.wallet.w3.eth.get_block(log['blockNumber'])
            timestamp = datetime.fromtimestamp(block['timestamp'])
            
            # Create processed event
            processed_event = ProcessedEvent(
                contract_address=log['address'].lower(),
                contract_name=contract_name,
                event_name=log['event'],
                block_number=log['blockNumber'],
                transaction_hash=log['transactionHash'].hex(),
                log_index=log['logIndex'],
                args=dict(log['args']),
                timestamp=timestamp,
                processed_at=datetime.utcnow()
            )
            
            # Get appropriate processor
            processor = self.event_processors.get(log['event'])
            if processor:
                success = await processor.process(processed_event)
                if success:
                    self.processed_events.add(event_id)
                    self.events_processed += 1
                else:
                    self.errors_encountered += 1
            else:
                # Log unprocessed event for debugging
                logger.debug(f"No processor for event {log['event']}")
                
        except Exception as e:
            logger.error(f"Failed to process event log: {e}")
            self.errors_encountered += 1
    
    async def _report_metrics(self):
        """Report monitoring metrics periodically"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                logger.info(
                    f"Event Monitor Metrics - "
                    f"Processed: {self.events_processed}, "
                    f"Errors: {self.errors_encountered}, "
                    f"Last Block: {self.last_processed_block}, "
                    f"Active Filters: {len([f for f in self.event_filters if f.active])}"
                )
                
            except Exception as e:
                logger.error(f"Failed to report metrics: {e}")
    
    async def get_recent_events(self, 
                               contract_name: Optional[str] = None,
                               event_name: Optional[str] = None,
                               hours: int = 24) -> List[Dict]:
        """
        Get recent events from database
        
        Args:
            contract_name: Filter by contract name
            event_name: Filter by event name  
            hours: Look back this many hours
            
        Returns:
            List[Dict]: Recent events
        """
        try:
            # Calculate time range
            since = datetime.utcnow() - timedelta(hours=hours)
            
            # Get transactions from database
            transactions = await self.db_service.get_ftns_transactions_by_timerange(
                start_time=since,
                end_time=datetime.utcnow()
            )
            
            # Format for return
            events = []
            for tx in transactions:
                events.append({
                    "transaction_hash": tx.transaction_hash,
                    "from_address": tx.from_address,
                    "to_address": tx.to_address,
                    "amount": float(tx.amount),
                    "transaction_type": tx.transaction_type,
                    "timestamp": tx.timestamp.isoformat(),
                    "block_number": tx.block_number,
                    "metadata": tx.transaction_metadata
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []
    
    async def add_custom_processor(self, event_name: str, processor: EventProcessor):
        """Add custom event processor"""
        self.event_processors[event_name] = processor
        logger.info(f"Added custom processor for {event_name}")
    
    async def pause_monitoring(self, contract_name: Optional[str] = None, 
                              event_name: Optional[str] = None):
        """Pause monitoring for specific filters"""
        for event_filter in self.event_filters:
            if contract_name and event_filter.contract_name != contract_name:
                continue
            if event_name and event_filter.event_name != event_name:
                continue
                
            event_filter.active = False
            
        logger.info(f"Paused monitoring for {contract_name or 'all'}.{event_name or 'all'}")
    
    async def resume_monitoring(self, contract_name: Optional[str] = None,
                               event_name: Optional[str] = None):
        """Resume monitoring for specific filters"""
        for event_filter in self.event_filters:
            if contract_name and event_filter.contract_name != contract_name:
                continue
            if event_name and event_filter.event_name != event_name:
                continue
                
            event_filter.active = True
            
        logger.info(f"Resumed monitoring for {contract_name or 'all'}.{event_name or 'all'}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "is_running": self.is_running,
            "active_filters": len([f for f in self.event_filters if f.active]),
            "total_filters": len(self.event_filters),
            "events_processed": self.events_processed,
            "errors_encountered": self.errors_encountered,
            "last_processed_block": self.last_processed_block,
            "last_activity": self.last_activity.isoformat(),
            "filters": [
                {
                    "contract": f.contract_name,
                    "event": f.event_name,
                    "from_block": f.from_block,
                    "active": f.active
                }
                for f in self.event_filters
            ]
        }