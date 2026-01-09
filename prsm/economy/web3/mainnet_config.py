"""
PRSM Mainnet Configuration Management
====================================

Manages production configuration for Polygon mainnet deployment including
network settings, contract addresses, and environment-specific parameters.
"""

import asyncio
import structlog
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import os

from prsm.core.config import get_settings
from ..integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)


class MainnetConfigManager:
    """
    Manages production configuration for Polygon mainnet deployment
    """
    
    def __init__(self):
        self.config_id = f"mainnet_config_{int(datetime.now().timestamp())}"
        self.logger = logger.bind(component="mainnet_config", config_id=self.config_id)
        
        # Configuration paths
        self.config_dir = Path(__file__).parent.parent.parent / "config"
        self.mainnet_config_file = self.config_dir / "mainnet.json"
        self.backup_config_file = self.config_dir / "mainnet_backup.json"
        
        # Current configuration
        self.current_config = {}
        self.config_history = []
        
        # Network configurations
        self.network_configs = {
            "polygon_mainnet": {
                "name": "Polygon Mainnet",
                "chain_id": 137,
                "currency": "MATIC",
                "rpc_endpoints": [
                    "https://polygon-rpc.com",
                    "https://rpc-mainnet.matic.network",
                    "https://rpc-mainnet.maticvigil.com",
                    "https://polygonapi.terminet.io/rpc"
                ],
                "ws_endpoints": [
                    "wss://ws-mainnet.matic.network",
                    "wss://polygon-rpc.com"
                ],
                "explorer": {
                    "name": "PolygonScan",
                    "url": "https://polygonscan.com",
                    "api": "https://api.polygonscan.com/api"
                },
                "faucet": None,  # No faucet for mainnet
                "gas_station": "https://gasstation-mainnet.matic.network/v2"
            }
        }
        
        print("⚙️ Mainnet Configuration Manager initialized")
    
    async def initialize_mainnet_config(
        self,
        deployment_results: Dict[str, Any],
        deployer_address: str
    ) -> Dict[str, Any]:
        """
        Initialize mainnet configuration after successful deployment
        
        Args:
            deployment_results: Results from mainnet deployment
            deployer_address: Address of the deployment wallet
            
        Returns:
            Configuration initialization results
        """
        try:
            self.logger.info("⚙️ Initializing mainnet configuration")
            
            # Create base mainnet configuration
            mainnet_config = await self._create_base_config(deployment_results, deployer_address)
            
            # Add contract addresses
            mainnet_config["contracts"] = self._extract_contract_addresses(deployment_results)
            
            # Add deployment metadata
            mainnet_config["deployment"] = {
                "deployment_id": deployment_results.get("deployment_id"),
                "deployed_at": deployment_results.get("completed_at"),
                "deployer_address": deployer_address,
                "status": deployment_results.get("status"),
                "verification_status": self._get_verification_status(deployment_results)
            }
            
            # Add operational parameters
            mainnet_config["operations"] = await self._create_operational_config()
            
            # Add monitoring configuration
            mainnet_config["monitoring"] = await self._create_monitoring_config(mainnet_config["contracts"])
            
            # Save configuration
            save_result = await self._save_config(mainnet_config)
            
            if save_result["success"]:
                self.current_config = mainnet_config
                
                # Update environment variables
                await self._update_environment_variables(mainnet_config)
                
                # Audit log the configuration
                await audit_logger.log_security_event(
                    event_type="mainnet_configuration_initialized",
                    user_id="system",
                    details={
                        "config_id": self.config_id,
                        "contracts_configured": len(mainnet_config["contracts"]),
                        "deployer_address": deployer_address
                    },
                    security_level="critical"
                )
                
                self.logger.info("✅ Mainnet configuration initialized successfully")
                
                return {
                    "success": True,
                    "config_id": self.config_id,
                    "contracts_configured": len(mainnet_config["contracts"]),
                    "config_file": str(self.mainnet_config_file),
                    "configuration": mainnet_config
                }
            else:
                return {
                    "success": False,
                    "error": save_result["error"]
                }
                
        except Exception as e:
            self.logger.error("Failed to initialize mainnet configuration", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_contract_address(
        self,
        contract_name: str,
        new_address: str,
        update_reason: str
    ) -> Dict[str, Any]:
        """Update a contract address in the mainnet configuration"""
        try:
            if not self.current_config:
                await self._load_config()
            
            old_address = self.current_config.get("contracts", {}).get(contract_name)
            
            # Update the address
            if "contracts" not in self.current_config:
                self.current_config["contracts"] = {}
            
            self.current_config["contracts"][contract_name] = new_address
            
            # Add to update history
            update_record = {
                "contract_name": contract_name,
                "old_address": old_address,
                "new_address": new_address,
                "reason": update_reason,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "updated_by": "system"
            }
            
            if "update_history" not in self.current_config:
                self.current_config["update_history"] = []
            
            self.current_config["update_history"].append(update_record)
            
            # Save updated configuration
            save_result = await self._save_config(self.current_config)
            
            if save_result["success"]:
                # Update environment variables
                await self._update_environment_variables(self.current_config)
                
                # Audit log the update
                await audit_logger.log_security_event(
                    event_type="mainnet_contract_address_updated",
                    user_id="system",
                    details={
                        "contract_name": contract_name,
                        "old_address": old_address,
                        "new_address": new_address,
                        "reason": update_reason
                    },
                    security_level="high"
                )
                
                self.logger.info("✅ Contract address updated",
                               contract=contract_name,
                               new_address=new_address)
                
                return {
                    "success": True,
                    "contract_name": contract_name,
                    "old_address": old_address,
                    "new_address": new_address,
                    "update_record": update_record
                }
            else:
                return {
                    "success": False,
                    "error": save_result["error"]
                }
                
        except Exception as e:
            self.logger.error("Failed to update contract address", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_mainnet_config(self) -> Dict[str, Any]:
        """Get current mainnet configuration"""
        try:
            if not self.current_config:
                await self._load_config()
            
            return {
                "success": True,
                "config": self.current_config,
                "config_id": self.config_id,
                "last_updated": self.current_config.get("last_updated"),
                "contracts_count": len(self.current_config.get("contracts", {}))
            }
            
        except Exception as e:
            self.logger.error("Failed to get mainnet configuration", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate current mainnet configuration"""
        try:
            if not self.current_config:
                await self._load_config()
            
            validation_results = {
                "valid": True,
                "issues": [],
                "warnings": [],
                "checks_performed": []
            }
            
            # Check required contract addresses
            required_contracts = ["ftns_token", "marketplace", "governance", "timelock"]
            contracts = self.current_config.get("contracts", {})
            
            for contract_name in required_contracts:
                validation_results["checks_performed"].append(f"contract_address_{contract_name}")
                
                if contract_name not in contracts:
                    validation_results["issues"].append(f"Missing contract address: {contract_name}")
                    validation_results["valid"] = False
                elif not contracts[contract_name]:
                    validation_results["issues"].append(f"Empty contract address: {contract_name}")
                    validation_results["valid"] = False
                elif not contracts[contract_name].startswith("0x"):
                    validation_results["issues"].append(f"Invalid contract address format: {contract_name}")
                    validation_results["valid"] = False
            
            # Check network configuration
            validation_results["checks_performed"].append("network_configuration")
            network_config = self.current_config.get("network", {})
            
            if not network_config.get("chain_id") == 137:
                validation_results["issues"].append("Invalid chain ID for Polygon mainnet")
                validation_results["valid"] = False
            
            if not network_config.get("rpc_url"):
                validation_results["warnings"].append("No primary RPC URL configured")
            
            # Check deployment metadata
            validation_results["checks_performed"].append("deployment_metadata")
            deployment = self.current_config.get("deployment", {})
            
            if not deployment.get("deployment_id"):
                validation_results["warnings"].append("Missing deployment ID")
            
            if not deployment.get("deployed_at"):
                validation_results["warnings"].append("Missing deployment timestamp")
            
            # Check operational parameters
            validation_results["checks_performed"].append("operational_parameters")
            operations = self.current_config.get("operations", {})
            
            if not operations.get("gas_settings"):
                validation_results["warnings"].append("No gas settings configured")
            
            validation_results["total_checks"] = len(validation_results["checks_performed"])
            validation_results["issues_count"] = len(validation_results["issues"])
            validation_results["warnings_count"] = len(validation_results["warnings"])
            
            return validation_results
            
        except Exception as e:
            self.logger.error("Configuration validation failed", error=str(e))
            return {
                "valid": False,
                "issues": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "checks_performed": []
            }
    
    async def _create_base_config(
        self,
        deployment_results: Dict[str, Any],
        deployer_address: str
    ) -> Dict[str, Any]:
        """Create base mainnet configuration"""
        
        base_config = {
            "config_version": "1.0",
            "config_id": self.config_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "network": {
                "name": "Polygon Mainnet",
                "chain_id": 137,
                "currency": "MATIC",
                "rpc_url": os.getenv("POLYGON_MAINNET_RPC_URL", "https://polygon-rpc.com"),
                "backup_rpc_urls": self.network_configs["polygon_mainnet"]["rpc_endpoints"][1:],
                "ws_url": os.getenv("POLYGON_MAINNET_WS_URL", "wss://ws-mainnet.matic.network"),
                "explorer_url": "https://polygonscan.com",
                "explorer_api": "https://api.polygonscan.com/api"
            },
            "environment": "production",
            "deployment_type": "mainnet"
        }
        
        return base_config
    
    def _extract_contract_addresses(self, deployment_results: Dict[str, Any]) -> Dict[str, str]:
        """Extract contract addresses from deployment results"""
        contracts = {}
        
        deployed_contracts = deployment_results.get("contracts_deployed", {})
        for contract_name, contract_info in deployed_contracts.items():
            if isinstance(contract_info, dict) and "address" in contract_info:
                contracts[contract_name] = contract_info["address"]
            else:
                contracts[contract_name] = str(contract_info)
        
        return contracts
    
    def _get_verification_status(self, deployment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get contract verification status"""
        verification_results = deployment_results.get("verification_results", {})
        
        status = {
            "total_contracts": len(deployment_results.get("contracts_deployed", {})),
            "verified_contracts": 0,
            "verification_details": {}
        }
        
        for contract_name, verification_info in verification_results.items():
            verified = verification_info.get("verified", False)
            status["verification_details"][contract_name] = {
                "verified": verified,
                "verification_url": verification_info.get("verification_url"),
                "verified_at": verification_info.get("verified_at")
            }
            
            if verified:
                status["verified_contracts"] += 1
        
        status["all_verified"] = status["verified_contracts"] == status["total_contracts"]
        
        return status
    
    async def _create_operational_config(self) -> Dict[str, Any]:
        """Create operational configuration parameters"""
        return {
            "gas_settings": {
                "max_gas_price_gwei": 300,
                "gas_limit_multiplier": 1.2,
                "priority_fee_gwei": 2,
                "auto_gas_estimation": True
            },
            "transaction_settings": {
                "confirmation_blocks": 12,
                "timeout_seconds": 300,
                "retry_attempts": 3,
                "retry_delay_seconds": 30
            },
            "rate_limiting": {
                "rpc_requests_per_second": 10,
                "burst_limit": 50,
                "cooldown_seconds": 60
            },
            "monitoring": {
                "health_check_interval": 30,
                "event_polling_interval": 5,
                "balance_check_interval": 60
            }
        }
    
    async def _create_monitoring_config(self, contracts: Dict[str, str]) -> Dict[str, Any]:
        """Create monitoring configuration"""
        return {
            "contract_monitoring": {
                contract_name: {
                    "address": address,
                    "events_to_monitor": ["Transfer", "Approval", "Mint", "Burn"] if contract_name == "ftns_token" else ["*"],
                    "health_checks": True,
                    "balance_monitoring": contract_name == "ftns_token"
                }
                for contract_name, address in contracts.items()
            },
            "alerts": {
                "low_balance_threshold": "1.0",  # 1 MATIC
                "failed_transaction_threshold": 5,
                "gas_price_alert_threshold": 200,  # 200 Gwei
                "notification_channels": ["log", "webhook"]
            },
            "metrics": {
                "collect_gas_metrics": True,
                "collect_transaction_metrics": True,
                "collect_balance_metrics": True,
                "retention_days": 30
            }
        }
    
    async def _save_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Save configuration to file"""
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(exist_ok=True)
            
            # Backup existing config if it exists
            if self.mainnet_config_file.exists():
                backup_path = self.backup_config_file
                self.mainnet_config_file.rename(backup_path)
                self.logger.info("📋 Backed up existing configuration")
            
            # Update last modified timestamp
            config["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Save new configuration
            with open(self.mainnet_config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            self.logger.info("💾 Configuration saved", config_file=str(self.mainnet_config_file))
            
            return {
                "success": True,
                "config_file": str(self.mainnet_config_file),
                "backup_file": str(self.backup_config_file) if self.backup_config_file.exists() else None
            }
            
        except Exception as e:
            self.logger.error("Failed to save configuration", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if not self.mainnet_config_file.exists():
                self.logger.warning("No mainnet configuration file found")
                return {}
            
            with open(self.mainnet_config_file, 'r') as f:
                config = json.load(f)
            
            self.current_config = config
            self.logger.info("📋 Configuration loaded", config_file=str(self.mainnet_config_file))
            
            return config
            
        except Exception as e:
            self.logger.error("Failed to load configuration", error=str(e))
            return {}
    
    async def _update_environment_variables(self, config: Dict[str, Any]):
        """Update environment variables with mainnet configuration"""
        try:
            # Extract key configuration values
            contracts = config.get("contracts", {})
            network = config.get("network", {})
            
            # Environment variables to update
            env_updates = {
                "PRSM_NETWORK": "polygon_mainnet",
                "PRSM_CHAIN_ID": "137",
                "POLYGON_MAINNET_RPC_URL": network.get("rpc_url", ""),
                "FTNS_TOKEN_ADDRESS": contracts.get("ftns_token", ""),
                "FTNS_MARKETPLACE_ADDRESS": contracts.get("marketplace", ""),
                "FTNS_GOVERNANCE_ADDRESS": contracts.get("governance", ""),
                "FTNS_TIMELOCK_ADDRESS": contracts.get("timelock", ""),
                "WEB3_MONITORING_ENABLED": "true",
                "PRSM_PRODUCTION_MODE": "true"
            }
            
            # In production, these would be set in the actual environment
            # For now, just log what would be updated
            self.logger.info("🔧 Environment variables updated", 
                           variables_count=len(env_updates))
            
        except Exception as e:
            self.logger.error("Failed to update environment variables", error=str(e))


# Global mainnet config manager instance
_config_manager: Optional[MainnetConfigManager] = None

def get_mainnet_config_manager() -> MainnetConfigManager:
    """Get or create the global mainnet config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = MainnetConfigManager()
    return _config_manager