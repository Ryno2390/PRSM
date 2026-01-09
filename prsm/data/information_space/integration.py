"""
Information Space Integration Module

Integrates Information Space with existing PRSM systems including:
- IPFS client for content analysis
- FTNS service for tokenomics
- Federation network for distributed coordination
- Existing API endpoints
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from decimal import Decimal

from .service import InformationSpaceService
from .api import InformationSpaceAPI

logger = logging.getLogger(__name__)


class InformationSpaceIntegration:
    """Integration manager for Information Space with PRSM systems."""
    
    def __init__(self):
        self.service: Optional[InformationSpaceService] = None
        self.api: Optional[InformationSpaceAPI] = None
        self.ipfs_client = None
        self.ftns_service = None
        self.federation_client = None
        
        # Integration state
        self.is_integrated = False
        self.initialization_errors = []
        
    async def initialize_with_prsm_systems(
        self, 
        ipfs_client=None, 
        ftns_service=None, 
        federation_client=None
    ) -> bool:
        """Initialize Information Space with PRSM system integrations."""
        
        try:
            logger.info("Initializing Information Space integration with PRSM systems...")
            
            # Store system references
            self.ipfs_client = ipfs_client
            self.ftns_service = ftns_service
            self.federation_client = federation_client
            
            # Initialize Information Space service
            self.service = InformationSpaceService(
                ipfs_client=self.ipfs_client,
                ftns_service=self.ftns_service,
                federation_client=self.federation_client
            )
            
            # Initialize the service
            service_initialized = await self.service.initialize()
            if not service_initialized:
                self.initialization_errors.append("Failed to initialize Information Space service")
                logger.error("Failed to initialize Information Space service")
                
            # Create API endpoints
            self.api = InformationSpaceAPI(self.service)
            
            # Test integrations
            await self._test_integrations()
            
            self.is_integrated = True
            logger.info("Information Space integration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Information Space integration: {e}")
            self.initialization_errors.append(str(e))
            return False
            
    async def _test_integrations(self):
        """Test integration with PRSM systems."""
        
        # Test IPFS integration
        if self.ipfs_client:
            try:
                # Try to get IPFS status
                status = await self.ipfs_client.get_status()
                logger.info(f"IPFS integration test: {status}")
            except Exception as e:
                logger.warning(f"IPFS integration test failed: {e}")
                self.initialization_errors.append(f"IPFS integration: {e}")
                
        # Test FTNS service integration
        if self.ftns_service:
            try:
                # Try to get FTNS status
                balance = await self.ftns_service.get_balance("test_account")
                logger.info(f"FTNS integration test successful")
            except Exception as e:
                logger.warning(f"FTNS integration test failed: {e}")
                self.initialization_errors.append(f"FTNS integration: {e}")
                
        # Test federation client integration
        if self.federation_client:
            try:
                # Try to get network status
                status = await self.federation_client.get_network_status()
                logger.info(f"Federation integration test successful")
            except Exception as e:
                logger.warning(f"Federation integration test failed: {e}")
                self.initialization_errors.append(f"Federation integration: {e}")
                
    async def add_ipfs_content_to_information_space(self, ipfs_hash: str, priority: bool = False) -> bool:
        """Add IPFS content to Information Space for analysis."""
        
        if not self.service:
            logger.error("Information Space service not initialized")
            return False
            
        try:
            return await self.service.add_content(ipfs_hash, priority)
        except Exception as e:
            logger.error(f"Failed to add IPFS content to Information Space: {e}")
            return False
            
    async def reward_information_space_contribution(
        self, 
        node_id: str, 
        contributor: str, 
        contribution_value: Decimal
    ) -> bool:
        """Reward contribution to Information Space through FTNS tokens."""
        
        if not self.service:
            logger.error("Information Space service not initialized")
            return False
            
        try:
            return await self.service.update_node_contribution(
                node_id, contribution_value, contributor
            )
        except Exception as e:
            logger.error(f"Failed to reward Information Space contribution: {e}")
            return False
            
    async def federate_information_space_updates(self, update_data: Dict[str, Any]) -> bool:
        """Federate Information Space updates across the network."""
        
        if not self.federation_client:
            logger.warning("Federation client not available")
            return False
            
        try:
            # Send update to federation network
            result = await self.federation_client.broadcast_update(
                "information_space",
                update_data
            )
            logger.info(f"Federated Information Space update: {result}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to federate Information Space updates: {e}")
            return False
            
    def get_api_router(self):
        """Get FastAPI router for Information Space endpoints."""
        
        if not self.api:
            logger.error("Information Space API not initialized")
            return None
            
        return self.api.router
        
    async def shutdown(self):
        """Shutdown Information Space integration."""
        
        logger.info("Shutting down Information Space integration...")
        
        if self.service:
            await self.service.shutdown()
            
        self.is_integrated = False
        
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of Information Space integration."""
        
        return {
            "is_integrated": self.is_integrated,
            "service_initialized": self.service is not None and self.service.is_running,
            "api_initialized": self.api is not None,
            "ipfs_available": self.ipfs_client is not None,
            "ftns_available": self.ftns_service is not None,
            "federation_available": self.federation_client is not None,
            "initialization_errors": self.initialization_errors,
            "graph_size": {
                "nodes": len(self.service.graph.nodes) if self.service else 0,
                "edges": len(self.service.graph.edges) if self.service else 0,
                "opportunities": len(self.service.graph.opportunities) if self.service else 0
            } if self.service else {}
        }


# Global integration instance
information_space_integration = InformationSpaceIntegration()


async def initialize_information_space_integration(
    ipfs_client=None, 
    ftns_service=None, 
    federation_client=None
) -> bool:
    """Initialize global Information Space integration."""
    
    return await information_space_integration.initialize_with_prsm_systems(
        ipfs_client, ftns_service, federation_client
    )


def get_information_space_router():
    """Get Information Space API router for FastAPI integration."""
    
    return information_space_integration.get_api_router()


async def add_content_to_information_space(ipfs_hash: str, priority: bool = False) -> bool:
    """Add content to Information Space (convenience function)."""
    
    return await information_space_integration.add_ipfs_content_to_information_space(
        ipfs_hash, priority
    )


async def reward_information_space_contribution(
    node_id: str, 
    contributor: str, 
    contribution_value: Decimal
) -> bool:
    """Reward Information Space contribution (convenience function)."""
    
    return await information_space_integration.reward_information_space_contribution(
        node_id, contributor, contribution_value
    )


def get_information_space_status() -> Dict[str, Any]:
    """Get Information Space integration status (convenience function)."""
    
    return information_space_integration.get_integration_status()