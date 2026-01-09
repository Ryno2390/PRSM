"""
Comprehensive Web3 Service Integration for PRSM FTNS Token System

Main service layer that orchestrates all Web3 components including
wallet management, contract interactions, event monitoring, and balance tracking.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime
import os

from prsm.core.database_service import DatabaseService
from prsm.core.config import get_settings
from .wallet_connector import Web3WalletConnector, NetworkType, WalletInfo
from .contract_interface import FTNSContractInterface
from .event_monitor import Web3EventMonitor
from .balance_service import FTNSBalanceService, BalanceChange
from .faucet_integration import PolygonFaucetIntegration

logger = logging.getLogger(__name__)

class Web3ServiceManager:
    """
    Comprehensive Web3 service manager for PRSM FTNS integration
    
    Features:
    - Centralized Web3 component management
    - Automatic service initialization and cleanup
    - Configuration management
    - Health monitoring and diagnostics
    - Event coordination between services
    """
    
    def __init__(self, db_service: DatabaseService, config: Optional[Dict] = None):
        self.db_service = db_service
        self.config = config or self._load_config()
        
        # Core services
        self.wallet_connector: Optional[Web3WalletConnector] = None
        self.contract_interface: Optional[FTNSContractInterface] = None
        self.event_monitor: Optional[Web3EventMonitor] = None
        self.balance_service: Optional[FTNSBalanceService] = None
        self.faucet_integration: Optional[PolygonFaucetIntegration] = None
        
        # Service state
        self.is_initialized = False
        self.current_network = None
        self.contract_addresses = {}
        
        # Performance tracking
        self.service_metrics = {
            "initialization_time": None,
            "last_health_check": None,
            "total_transactions_processed": 0,
            "total_events_monitored": 0,
            "errors_encountered": 0
        }
    
    def _load_config(self) -> Dict:
        """Load Web3 service configuration"""
        settings = get_settings()
        
        return {
            "networks": {
                "polygon_mainnet": {
                    "rpc_url": os.getenv("POLYGON_MAINNET_RPC_URL", "https://polygon-rpc.com"),
                    "chain_id": 137,
                    "name": "Polygon Mainnet"
                },
                "polygon_mumbai": {
                    "rpc_url": os.getenv("POLYGON_MUMBAI_RPC_URL", "https://rpc-mumbai.maticvigil.com"),
                    "chain_id": 80001,
                    "name": "Polygon Mumbai Testnet"
                }
            },
            "contracts": {
                "ftns_token": os.getenv("FTNS_TOKEN_ADDRESS", ""),
                "marketplace": os.getenv("FTNS_MARKETPLACE_ADDRESS", ""),
                "governance": os.getenv("FTNS_GOVERNANCE_ADDRESS", ""),
                "timelock": os.getenv("FTNS_TIMELOCK_ADDRESS", "")
            },
            "monitoring": {
                "enabled": os.getenv("WEB3_MONITORING_ENABLED", "true").lower() == "true",
                "events": ["Transfer", "Approval", "Mint", "Burn"],
                "poll_interval": int(os.getenv("WEB3_POLL_INTERVAL", "5"))
            },
            "faucet": {
                "enabled": os.getenv("FAUCET_INTEGRATION_ENABLED", "true").lower() == "true",
                "auto_request": os.getenv("FAUCET_AUTO_REQUEST", "false").lower() == "true"
            }
        }
    
    async def initialize(self, network: str = "polygon_mumbai", 
                        private_key: Optional[str] = None) -> bool:
        """
        Initialize all Web3 services
        
        Args:
            network: Network to connect to
            private_key: Wallet private key for transactions
            
        Returns:
            bool: True if initialization successful
        """
        try:
            start_time = datetime.utcnow()
            logger.info(f"Initializing Web3 services for network: {network}")
            
            # Initialize wallet connector
            self.wallet_connector = Web3WalletConnector(self.config)
            network_connected = await self.wallet_connector.connect_to_network(network)
            if not network_connected:
                raise RuntimeError(f"Failed to connect to network: {network}")
            
            self.current_network = network
            
            # Connect wallet if private key provided
            if private_key:
                wallet_address = await self.wallet_connector.connect_wallet(private_key)
                if wallet_address:
                    logger.info(f"Wallet connected: {wallet_address}")
                else:
                    logger.warning("Failed to connect wallet")
            
            # Initialize contract addresses
            self.contract_addresses = {
                name: addr for name, addr in self.config["contracts"].items()
                if addr  # Only include non-empty addresses
            }
            
            if not self.contract_addresses:
                logger.warning("No contract addresses configured")
            
            # Initialize contract interface
            if self.contract_addresses:
                self.contract_interface = FTNSContractInterface(
                    self.wallet_connector,
                    self.contract_addresses
                )
                contracts_initialized = await self.contract_interface.initialize_contracts()
                if contracts_initialized:
                    logger.info("Smart contracts initialized successfully")
                else:
                    logger.warning("Some contracts failed to initialize")
            
            # Initialize balance service
            if self.contract_interface:
                self.balance_service = FTNSBalanceService(
                    self.wallet_connector,
                    self.contract_interface,
                    self.db_service
                )
                
                # Add balance change listener
                await self.balance_service.add_balance_change_listener(
                    self._handle_balance_change
                )
                
                logger.info("Balance service initialized")
            
            # Initialize event monitor
            if self.contract_interface and self.config["monitoring"]["enabled"]:
                self.event_monitor = Web3EventMonitor(
                    self.wallet_connector,
                    self.contract_interface,
                    self.db_service
                )
                
                # Add contract monitoring
                for contract_name in self.contract_addresses:
                    await self.event_monitor.add_contract_monitor(
                        contract_name,
                        self.config["monitoring"]["events"]
                    )
                
                # Start monitoring
                await self.event_monitor.start_monitoring()
                logger.info("Event monitoring started")
            
            # Initialize faucet integration
            if self.config["faucet"]["enabled"]:
                self.faucet_integration = PolygonFaucetIntegration()
                logger.info("Faucet integration initialized")
            
            # Mark as initialized
            self.is_initialized = True
            init_time = (datetime.utcnow() - start_time).total_seconds()
            self.service_metrics["initialization_time"] = init_time
            
            logger.info(f"Web3 services initialized successfully in {init_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Web3 service initialization failed: {e}")
            await self.cleanup()
            return False
    
    async def cleanup(self):
        """Cleanup all Web3 services"""
        try:
            logger.info("Cleaning up Web3 services")
            
            # Stop event monitoring
            if self.event_monitor:
                await self.event_monitor.stop_monitoring()
                self.event_monitor = None
            
            # Disconnect wallet
            if self.wallet_connector:
                self.wallet_connector.disconnect()
                self.wallet_connector = None
            
            # Clear other services
            self.contract_interface = None
            self.balance_service = None
            self.faucet_integration = None
            
            self.is_initialized = False
            logger.info("Web3 services cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Web3 services cleanup: {e}")
    
    async def get_wallet_info(self, address: Optional[str] = None) -> Optional[WalletInfo]:
        """Get wallet information"""
        if not self.wallet_connector:
            return None
        return await self.wallet_connector.get_wallet_info(address)
    
    async def get_balance(self, address: str, force_refresh: bool = False) -> Optional[Dict]:
        """Get comprehensive balance information"""
        if not self.balance_service:
            return None
        
        snapshot = await self.balance_service.get_balance(address, force_refresh)
        if not snapshot:
            return None
        
        return {
            "address": snapshot.address,
            "liquid_balance": float(snapshot.liquid_balance),
            "locked_balance": float(snapshot.locked_balance),
            "staked_balance": float(snapshot.staked_balance),
            "total_balance": float(snapshot.total_balance),
            "context_allocated": float(snapshot.context_allocated),
            "last_updated": snapshot.last_updated.isoformat(),
            "block_number": snapshot.block_number
        }
    
    async def transfer_tokens(self, to_address: str, amount: Decimal) -> Optional[Dict]:
        """Transfer FTNS tokens"""
        if not self.contract_interface:
            raise RuntimeError("Contract interface not initialized")
        
        result = await self.contract_interface.transfer_tokens(to_address, amount)
        if not result:
            return None
        
        self.service_metrics["total_transactions_processed"] += 1
        
        return {
            "hash": result.hash,
            "success": result.success,
            "gas_used": result.gas_used,
            "block_number": result.block_number,
            "error": result.error
        }
    
    async def get_transaction_history(self, address: str, limit: int = 50, offset: int = 0) -> Dict:
        """Get transaction history for address"""
        if not self.balance_service:
            return {"transactions": [], "total": 0}
        
        transactions, total = await self.balance_service.get_transaction_history(
            address, limit, offset
        )
        
        return {
            "transactions": transactions,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    
    async def request_testnet_matic(self, wallet_address: str) -> Optional[Dict]:
        """Request testnet MATIC from faucet"""
        if not self.faucet_integration:
            raise RuntimeError("Faucet integration not enabled")
        
        async with self.faucet_integration as faucet:
            result = await faucet.request_matic(wallet_address)
        
        return {
            "success": result.status.value in ["success", "pending"],
            "status": result.status.value,
            "transaction_hash": result.transaction_hash,
            "amount": result.amount,
            "message": result.message,
            "retry_after": result.retry_after
        }
    
    async def get_network_statistics(self) -> Dict:
        """Get comprehensive network statistics"""
        stats = {}
        
        # Get token statistics
        if self.balance_service:
            stats.update(await self.balance_service.get_network_statistics())
        
        # Get monitoring statistics
        if self.event_monitor:
            stats["monitoring"] = self.event_monitor.get_monitoring_status()
        
        # Add service metrics
        stats["service_metrics"] = self.service_metrics.copy()
        stats["service_metrics"]["last_health_check"] = datetime.utcnow().isoformat()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health = {
                "overall_status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {}
            }
            
            # Check wallet connector
            if self.wallet_connector:
                health["services"]["wallet_connector"] = {
                    "status": "healthy" if self.wallet_connector.is_connected else "unhealthy",
                    "network": self.current_network,
                    "has_wallet": self.wallet_connector.has_wallet
                }
            else:
                health["services"]["wallet_connector"] = {"status": "not_initialized"}
            
            # Check contract interface
            if self.contract_interface:
                # Verify contract deployment
                token_verified = await self.contract_interface.verify_contract_deployment("FTNSToken")
                health["services"]["contract_interface"] = {
                    "status": "healthy" if token_verified else "degraded",
                    "contracts_loaded": len(self.contract_interface.contracts),
                    "token_contract_verified": token_verified
                }
            else:
                health["services"]["contract_interface"] = {"status": "not_initialized"}
            
            # Check event monitor
            if self.event_monitor:
                monitor_status = self.event_monitor.get_monitoring_status()
                health["services"]["event_monitor"] = {
                    "status": "healthy" if monitor_status["is_running"] else "unhealthy",
                    "active_filters": monitor_status["active_filters"],
                    "events_processed": monitor_status["events_processed"],
                    "errors": monitor_status["errors_encountered"]
                }
            else:
                health["services"]["event_monitor"] = {"status": "not_initialized"}
            
            # Check balance service
            if self.balance_service:
                cache_stats = self.balance_service.get_cache_stats()
                health["services"]["balance_service"] = {
                    "status": "healthy",
                    "cache_stats": cache_stats
                }
            else:
                health["services"]["balance_service"] = {"status": "not_initialized"}
            
            # Determine overall status
            service_statuses = [svc["status"] for svc in health["services"].values()]
            if any(status == "unhealthy" for status in service_statuses):
                health["overall_status"] = "unhealthy"
            elif any(status == "degraded" for status in service_statuses):
                health["overall_status"] = "degraded"
            
            self.service_metrics["last_health_check"] = datetime.utcnow()
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _handle_balance_change(self, change: BalanceChange):
        """Handle balance change notifications"""
        try:
            logger.info(
                f"Balance change detected for {change.address}: "
                f"{change.previous_balance} -> {change.new_balance} "
                f"({change.change_amount:+})"
            )
            
            # Could emit WebSocket notifications here
            # Could trigger alerts for large changes
            # Could update analytics
            
        except Exception as e:
            logger.error(f"Error handling balance change: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            "initialized": self.is_initialized,
            "network": self.current_network,
            "contracts_configured": len(self.contract_addresses),
            "wallet_connected": self.wallet_connector.has_wallet if self.wallet_connector else False,
            "monitoring_active": self.event_monitor.is_running if self.event_monitor else False,
            "metrics": self.service_metrics
        }
    
    async def update_contract_addresses(self, new_addresses: Dict[str, str]):
        """Update contract addresses and reinitialize"""
        try:
            self.contract_addresses.update(new_addresses)
            
            if self.contract_interface and self.wallet_connector:
                # Reinitialize contract interface
                self.contract_interface = FTNSContractInterface(
                    self.wallet_connector,
                    self.contract_addresses
                )
                await self.contract_interface.initialize_contracts()
                
                # Update balance service
                if self.balance_service:
                    self.balance_service.contracts = self.contract_interface
                
                # Update event monitor
                if self.event_monitor:
                    await self.event_monitor.stop_monitoring()
                    self.event_monitor = Web3EventMonitor(
                        self.wallet_connector,
                        self.contract_interface,
                        self.db_service
                    )
                    
                    # Re-add contract monitoring
                    for contract_name in self.contract_addresses:
                        await self.event_monitor.add_contract_monitor(
                            contract_name,
                            self.config["monitoring"]["events"]
                        )
                    
                    await self.event_monitor.start_monitoring()
                
                logger.info("Contract addresses updated and services reinitialized")
                
        except Exception as e:
            logger.error(f"Failed to update contract addresses: {e}")
            raise

# Global service manager instance
_web3_service_manager: Optional[Web3ServiceManager] = None

async def get_web3_service_manager() -> Web3ServiceManager:
    """Get global Web3 service manager instance"""
    global _web3_service_manager
    
    if _web3_service_manager is None:
        db_service = DatabaseService()
        _web3_service_manager = Web3ServiceManager(db_service)
    
    return _web3_service_manager

async def initialize_web3_services(network: str = "polygon_mumbai", 
                                  private_key: Optional[str] = None) -> bool:
    """Initialize global Web3 services"""
    manager = await get_web3_service_manager()
    return await manager.initialize(network, private_key)

async def cleanup_web3_services():
    """Cleanup global Web3 services"""
    global _web3_service_manager
    
    if _web3_service_manager:
        await _web3_service_manager.cleanup()
        _web3_service_manager = None

async def get_web3_service() -> Web3ServiceManager:
    """Get the global Web3 service instance (alias for get_web3_service_manager)"""
    return await get_web3_service_manager()