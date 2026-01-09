"""
IPFS Fallback Storage System for PRSM P2P Collaboration

This module provides IPFS-based fallback storage for when P2P nodes
are offline or unavailable. It ensures data availability and redundancy
by leveraging the IPFS distributed file system as a backup layer.

Key Features:
- Automatic IPFS integration for data redundancy
- Smart fallback when P2P nodes are unavailable
- Content addressing and verification
- Pinning services integration
- Graceful degradation from P2P to IPFS
- Hybrid storage strategies
"""

import asyncio
import json
import logging
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import aiohttp
import base64
import os

logger = logging.getLogger(__name__)


class StorageStrategy(Enum):
    """Storage strategies for fallback management"""
    P2P_ONLY = "p2p_only"                    # Use only P2P network
    IPFS_ONLY = "ipfs_only"                  # Use only IPFS
    P2P_WITH_IPFS_BACKUP = "p2p_backup"     # P2P primary, IPFS backup
    HYBRID = "hybrid"                        # Use both simultaneously
    ADAPTIVE = "adaptive"                    # Choose based on conditions


class StorageStatus(Enum):
    """Status of stored data"""
    AVAILABLE = "available"
    PARTIALLY_AVAILABLE = "partially_available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class IPFSNode:
    """IPFS node configuration"""
    node_id: str
    api_url: str
    gateway_url: str
    is_local: bool = False
    is_pinning_service: bool = False
    api_key: Optional[str] = None
    max_pin_size: Optional[int] = None  # bytes
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers


@dataclass
class StoredContent:
    """Information about stored content"""
    content_id: str
    ipfs_hash: Optional[str] = None
    p2p_locations: List[str] = None
    size_bytes: int = 0
    content_type: str = 'application/octet-stream'
    created_at: float = 0.0
    last_verified: float = 0.0
    storage_strategy: StorageStrategy = StorageStrategy.HYBRID
    pinned_nodes: List[str] = None
    
    def __post_init__(self):
        if self.p2p_locations is None:
            self.p2p_locations = []
        if self.pinned_nodes is None:
            self.pinned_nodes = []
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    @property
    def is_available_via_p2p(self) -> bool:
        """Check if content is available via P2P"""
        return len(self.p2p_locations) > 0
    
    @property
    def is_available_via_ipfs(self) -> bool:
        """Check if content is available via IPFS"""
        return self.ipfs_hash is not None
    
    @property
    def storage_status(self) -> StorageStatus:
        """Get overall storage status"""
        if self.is_available_via_p2p and self.is_available_via_ipfs:
            return StorageStatus.AVAILABLE
        elif self.is_available_via_p2p or self.is_available_via_ipfs:
            return StorageStatus.PARTIALLY_AVAILABLE
        else:
            return StorageStatus.UNAVAILABLE


class IPFSClient:
    """Client for interacting with IPFS nodes"""
    
    def __init__(self, nodes: List[IPFSNode]):
        self.nodes = {node.node_id: node for node in nodes}
        self.session: Optional[aiohttp.ClientSession] = None
        self.default_timeout = 30
    
    async def start(self):
        """Initialize the IPFS client"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.default_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info(f"IPFS client started with {len(self.nodes)} nodes")
    
    async def stop(self):
        """Stop the IPFS client"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("IPFS client stopped")
    
    async def add_content(self, content: bytes, 
                         node_id: Optional[str] = None) -> Optional[str]:
        """Add content to IPFS and return hash"""
        if not self.session:
            await self.start()
        
        # Use specified node or try all nodes
        nodes_to_try = [self.nodes[node_id]] if node_id and node_id in self.nodes else list(self.nodes.values())
        
        for node in nodes_to_try:
            try:
                # Prepare multipart form data
                form_data = aiohttp.FormData()
                form_data.add_field('file', content, content_type='application/octet-stream')
                
                url = f"{node.api_url}/api/v0/add"
                
                async with self.session.post(url, data=form_data, headers=node.headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        ipfs_hash = result.get('Hash')
                        
                        if ipfs_hash:
                            logger.info(f"Added content to IPFS: {ipfs_hash} ({len(content)} bytes)")
                            return ipfs_hash
                    else:
                        logger.warning(f"IPFS add failed on {node.node_id}: {response.status}")
            
            except Exception as e:
                logger.warning(f"Error adding content to IPFS node {node.node_id}: {e}")
        
        return None
    
    async def get_content(self, ipfs_hash: str, 
                         node_id: Optional[str] = None) -> Optional[bytes]:
        """Retrieve content from IPFS by hash"""
        if not self.session:
            await self.start()
        
        # Use specified node or try all nodes
        nodes_to_try = [self.nodes[node_id]] if node_id and node_id in self.nodes else list(self.nodes.values())
        
        for node in nodes_to_try:
            try:
                # Try API first, then gateway
                urls_to_try = [
                    f"{node.api_url}/api/v0/cat?arg={ipfs_hash}",
                    f"{node.gateway_url}/ipfs/{ipfs_hash}"
                ]
                
                for url in urls_to_try:
                    try:
                        async with self.session.get(url, headers=node.headers) as response:
                            if response.status == 200:
                                content = await response.read()
                                logger.debug(f"Retrieved content from IPFS: {ipfs_hash} ({len(content)} bytes)")
                                return content
                    except Exception as e:
                        logger.debug(f"Failed to retrieve from {url}: {e}")
                        continue
            
            except Exception as e:
                logger.warning(f"Error retrieving content from IPFS node {node.node_id}: {e}")
        
        return None
    
    async def pin_content(self, ipfs_hash: str, 
                         node_id: Optional[str] = None) -> bool:
        """Pin content to prevent garbage collection"""
        if not self.session:
            await self.start()
        
        # Use specified node or try all nodes
        nodes_to_try = [self.nodes[node_id]] if node_id and node_id in self.nodes else list(self.nodes.values())
        
        success_count = 0
        
        for node in nodes_to_try:
            try:
                url = f"{node.api_url}/api/v0/pin/add?arg={ipfs_hash}"
                
                async with self.session.post(url, headers=node.headers) as response:
                    if response.status == 200:
                        success_count += 1
                        logger.debug(f"Pinned {ipfs_hash} on {node.node_id}")
                    else:
                        logger.warning(f"Pin failed on {node.node_id}: {response.status}")
            
            except Exception as e:
                logger.warning(f"Error pinning content on {node.node_id}: {e}")
        
        return success_count > 0
    
    async def unpin_content(self, ipfs_hash: str, 
                           node_id: Optional[str] = None) -> bool:
        """Unpin content to allow garbage collection"""
        if not self.session:
            await self.start()
        
        # Use specified node or try all nodes
        nodes_to_try = [self.nodes[node_id]] if node_id and node_id in self.nodes else list(self.nodes.values())
        
        success_count = 0
        
        for node in nodes_to_try:
            try:
                url = f"{node.api_url}/api/v0/pin/rm?arg={ipfs_hash}"
                
                async with self.session.post(url, headers=node.headers) as response:
                    if response.status == 200:
                        success_count += 1
                        logger.debug(f"Unpinned {ipfs_hash} on {node.node_id}")
                    else:
                        logger.warning(f"Unpin failed on {node.node_id}: {response.status}")
            
            except Exception as e:
                logger.warning(f"Error unpinning content on {node.node_id}: {e}")
        
        return success_count > 0
    
    async def check_content_availability(self, ipfs_hash: str) -> Dict[str, bool]:
        """Check content availability across IPFS nodes"""
        if not self.session:
            await self.start()
        
        availability = {}
        
        for node_id, node in self.nodes.items():
            try:
                # Try to stat the content (faster than full retrieval)
                url = f"{node.api_url}/api/v0/object/stat?arg={ipfs_hash}"
                
                async with self.session.post(url, headers=node.headers) as response:
                    availability[node_id] = response.status == 200
            
            except Exception as e:
                logger.debug(f"Availability check failed for {node_id}: {e}")
                availability[node_id] = False
        
        return availability
    
    async def get_node_stats(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics from an IPFS node"""
        if node_id not in self.nodes or not self.session:
            return None
        
        node = self.nodes[node_id]
        
        try:
            url = f"{node.api_url}/api/v0/stats/repo"
            
            async with self.session.post(url, headers=node.headers) as response:
                if response.status == 200:
                    return await response.json()
        
        except Exception as e:
            logger.warning(f"Error getting stats from {node_id}: {e}")
        
        return None


class FallbackStorageManager:
    """
    Manages fallback storage strategies and coordinates between P2P and IPFS
    """
    
    def __init__(self, ipfs_client: IPFSClient, config: Optional[Dict[str, Any]] = None):
        self.ipfs_client = ipfs_client
        self.config = config or {}
        
        # Storage tracking
        self.stored_content: Dict[str, StoredContent] = {}
        
        # Configuration
        self.default_strategy = StorageStrategy(
            self.config.get('default_strategy', 'hybrid')
        )
        self.auto_pin_threshold = self.config.get('auto_pin_threshold', 10 * 1024 * 1024)  # 10MB
        self.verification_interval = self.config.get('verification_interval', 3600)  # 1 hour
        self.max_fallback_delay = self.config.get('max_fallback_delay', 30)  # seconds
        
        # Background tasks
        self.verification_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info("Fallback storage manager initialized")
    
    async def start(self):
        """Start the fallback storage manager"""
        if self.running:
            return
        
        self.running = True
        await self.ipfs_client.start()
        
        # Start background verification
        self.verification_task = asyncio.create_task(self._verification_loop())
        
        logger.info("Fallback storage manager started")
    
    async def stop(self):
        """Stop the fallback storage manager"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop background tasks
        if self.verification_task and not self.verification_task.done():
            self.verification_task.cancel()
            try:
                await self.verification_task
            except asyncio.CancelledError:
                pass
        
        await self.ipfs_client.stop()
        
        logger.info("Fallback storage manager stopped")
    
    async def store_content(self, content_id: str, content: bytes,
                           strategy: Optional[StorageStrategy] = None,
                           p2p_nodes: Optional[List[str]] = None) -> StoredContent:
        """Store content using specified strategy"""
        strategy = strategy or self.default_strategy
        
        stored_content = StoredContent(
            content_id=content_id,
            size_bytes=len(content),
            storage_strategy=strategy,
            p2p_locations=p2p_nodes or []
        )
        
        # Store based on strategy
        if strategy in [StorageStrategy.IPFS_ONLY, StorageStrategy.P2P_WITH_IPFS_BACKUP, 
                       StorageStrategy.HYBRID]:
            # Add to IPFS
            ipfs_hash = await self.ipfs_client.add_content(content)
            if ipfs_hash:
                stored_content.ipfs_hash = ipfs_hash
                
                # Pin if content is large enough or strategy requires it
                if (len(content) >= self.auto_pin_threshold or 
                    strategy == StorageStrategy.IPFS_ONLY):
                    await self.ipfs_client.pin_content(ipfs_hash)
                    stored_content.pinned_nodes = list(self.ipfs_client.nodes.keys())
                
                logger.info(f"Stored content {content_id} in IPFS: {ipfs_hash}")
            else:
                logger.error(f"Failed to store content {content_id} in IPFS")
        
        if strategy in [StorageStrategy.P2P_ONLY, StorageStrategy.HYBRID]:
            # P2P storage would be handled by the shard distribution system
            # This is a placeholder for integration
            logger.debug(f"P2P storage for {content_id} handled by shard distributor")
        
        # Track the stored content
        self.stored_content[content_id] = stored_content
        
        return stored_content
    
    async def retrieve_content(self, content_id: str, 
                              prefer_p2p: bool = True) -> Optional[bytes]:
        """Retrieve content using best available method"""
        if content_id not in self.stored_content:
            logger.error(f"Content {content_id} not found in storage registry")
            return None
        
        stored = self.stored_content[content_id]
        
        # Try P2P first if preferred and available
        if prefer_p2p and stored.is_available_via_p2p:
            # This would integrate with the P2P retrieval system
            # For now, fall back to IPFS
            logger.debug(f"P2P retrieval for {content_id} not implemented, falling back to IPFS")
        
        # Try IPFS
        if stored.is_available_via_ipfs:
            content = await self.ipfs_client.get_content(stored.ipfs_hash)
            if content:
                logger.info(f"Retrieved content {content_id} from IPFS ({len(content)} bytes)")
                return content
        
        logger.error(f"Failed to retrieve content {content_id} from any source")
        return None
    
    async def check_content_availability(self, content_id: str) -> StorageStatus:
        """Check availability of stored content"""
        if content_id not in self.stored_content:
            return StorageStatus.UNKNOWN
        
        stored = self.stored_content[content_id]
        
        # Check IPFS availability if hash exists
        ipfs_available = False
        if stored.ipfs_hash:
            availability = await self.ipfs_client.check_content_availability(stored.ipfs_hash)
            ipfs_available = any(availability.values())
        
        # Check P2P availability (placeholder)
        p2p_available = len(stored.p2p_locations) > 0
        
        # Update stored content status
        if ipfs_available and p2p_available:
            return StorageStatus.AVAILABLE
        elif ipfs_available or p2p_available:
            return StorageStatus.PARTIALLY_AVAILABLE
        else:
            return StorageStatus.UNAVAILABLE
    
    async def migrate_to_strategy(self, content_id: str, 
                                 new_strategy: StorageStrategy) -> bool:
        """Migrate content to a new storage strategy"""
        if content_id not in self.stored_content:
            return False
        
        stored = self.stored_content[content_id]
        current_strategy = stored.storage_strategy
        
        if current_strategy == new_strategy:
            return True  # Already using target strategy
        
        # Retrieve content for migration
        content = await self.retrieve_content(content_id)
        if not content:
            logger.error(f"Cannot migrate {content_id}: content not retrievable")
            return False
        
        # Apply new strategy
        if new_strategy == StorageStrategy.IPFS_ONLY:
            # Ensure content is in IPFS and pinned
            if not stored.ipfs_hash:
                ipfs_hash = await self.ipfs_client.add_content(content)
                if ipfs_hash:
                    stored.ipfs_hash = ipfs_hash
                else:
                    return False
            
            # Pin the content
            if await self.ipfs_client.pin_content(stored.ipfs_hash):
                stored.pinned_nodes = list(self.ipfs_client.nodes.keys())
            
            # Remove P2P locations (in real implementation, would notify P2P nodes)
            stored.p2p_locations = []
        
        elif new_strategy == StorageStrategy.P2P_ONLY:
            # Remove from IPFS (unpin)
            if stored.ipfs_hash:
                await self.ipfs_client.unpin_content(stored.ipfs_hash)
                stored.ipfs_hash = None
                stored.pinned_nodes = []
            
            # Ensure P2P storage (placeholder)
            logger.debug(f"P2P migration for {content_id} would be handled by shard distributor")
        
        elif new_strategy == StorageStrategy.HYBRID:
            # Ensure content is available in both systems
            if not stored.ipfs_hash:
                ipfs_hash = await self.ipfs_client.add_content(content)
                if ipfs_hash:
                    stored.ipfs_hash = ipfs_hash
            
            # Ensure P2P availability (placeholder)
            logger.debug(f"P2P migration for {content_id} would be handled by shard distributor")
        
        # Update strategy
        stored.storage_strategy = new_strategy
        
        logger.info(f"Migrated {content_id} from {current_strategy.value} to {new_strategy.value}")
        return True
    
    async def optimize_storage(self, content_id: str) -> bool:
        """Optimize storage strategy based on usage patterns"""
        if content_id not in self.stored_content:
            return False
        
        stored = self.stored_content[content_id]
        
        # Simple optimization rules
        current_time = time.time()
        age_days = (current_time - stored.created_at) / 86400
        
        # If content is old and large, prefer IPFS for cost efficiency
        if age_days > 30 and stored.size_bytes > 100 * 1024 * 1024:  # 30 days, 100MB
            if stored.storage_strategy != StorageStrategy.IPFS_ONLY:
                return await self.migrate_to_strategy(content_id, StorageStrategy.IPFS_ONLY)
        
        # If content is new and small, prefer P2P for speed
        elif age_days < 1 and stored.size_bytes < 10 * 1024 * 1024:  # 1 day, 10MB
            if stored.storage_strategy != StorageStrategy.P2P_ONLY:
                return await self.migrate_to_strategy(content_id, StorageStrategy.HYBRID)
        
        return True
    
    async def _verification_loop(self):
        """Background loop to verify content availability"""
        while self.running:
            try:
                await self._verify_stored_content()
                await asyncio.sleep(self.verification_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in verification loop: {e}")
    
    async def _verify_stored_content(self):
        """Verify availability of all stored content"""
        for content_id, stored in list(self.stored_content.items()):
            try:
                status = await self.check_content_availability(content_id)
                
                if status == StorageStatus.UNAVAILABLE:
                    logger.warning(f"Content {content_id} is unavailable")
                    # Could trigger recovery procedures here
                
                elif status == StorageStatus.PARTIALLY_AVAILABLE:
                    logger.info(f"Content {content_id} is partially available")
                    # Could trigger replication to improve availability
                
                stored.last_verified = time.time()
                
            except Exception as e:
                logger.error(f"Error verifying content {content_id}: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage system statistics"""
        total_content = len(self.stored_content)
        total_size = sum(content.size_bytes for content in self.stored_content.values())
        
        strategy_counts = {}
        for strategy in StorageStrategy:
            strategy_counts[strategy.value] = sum(
                1 for content in self.stored_content.values()
                if content.storage_strategy == strategy
            )
        
        availability_counts = {
            'available': 0,
            'partially_available': 0,
            'unavailable': 0,
            'unknown': 0
        }
        
        for content in self.stored_content.values():
            status = content.storage_status
            if status == StorageStatus.AVAILABLE:
                availability_counts['available'] += 1
            elif status == StorageStatus.PARTIALLY_AVAILABLE:
                availability_counts['partially_available'] += 1
            elif status == StorageStatus.UNAVAILABLE:
                availability_counts['unavailable'] += 1
            else:
                availability_counts['unknown'] += 1
        
        return {
            'total_content_items': total_content,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'strategy_distribution': strategy_counts,
            'availability_distribution': availability_counts,
            'ipfs_nodes': len(self.ipfs_client.nodes),
            'verification_interval': self.verification_interval
        }
    
    def export_content_registry(self) -> Dict[str, Any]:
        """Export the content registry"""
        return {
            'stored_content': {
                content_id: asdict(content)
                for content_id, content in self.stored_content.items()
            },
            'exported_at': time.time()
        }
    
    def import_content_registry(self, registry_data: Dict[str, Any]):
        """Import a content registry"""
        if 'stored_content' in registry_data:
            for content_id, content_data in registry_data['stored_content'].items():
                self.stored_content[content_id] = StoredContent(**content_data)
        
        logger.info(f"Imported {len(self.stored_content)} content items")


# Example usage and configuration
async def example_fallback_storage():
    """Example of fallback storage system usage"""
    
    # Configure IPFS nodes
    ipfs_nodes = [
        IPFSNode(
            node_id="local_node",
            api_url="http://127.0.0.1:5001",
            gateway_url="http://127.0.0.1:8080",
            is_local=True
        ),
        IPFSNode(
            node_id="pinata",
            api_url="https://api.pinata.cloud",
            gateway_url="https://gateway.pinata.cloud",
            is_pinning_service=True,
            api_key="your-pinata-api-key"
        )
    ]
    
    # Initialize components
    ipfs_client = IPFSClient(ipfs_nodes)
    
    config = {
        'default_strategy': 'hybrid',
        'auto_pin_threshold': 5 * 1024 * 1024,  # 5MB
        'verification_interval': 1800  # 30 minutes
    }
    
    fallback_manager = FallbackStorageManager(ipfs_client, config)
    
    try:
        await fallback_manager.start()
        
        # Store some example content
        test_content = b"This is test content for the fallback storage system" * 1000
        
        stored = await fallback_manager.store_content(
            "test_content_1",
            test_content,
            strategy=StorageStrategy.HYBRID
        )
        
        print(f"Stored content: {stored.content_id}")
        print(f"IPFS hash: {stored.ipfs_hash}")
        print(f"Size: {stored.size_bytes} bytes")
        print(f"Strategy: {stored.storage_strategy.value}")
        
        # Check availability
        status = await fallback_manager.check_content_availability("test_content_1")
        print(f"Availability status: {status.value}")
        
        # Retrieve content
        retrieved = await fallback_manager.retrieve_content("test_content_1")
        if retrieved:
            print(f"Retrieved {len(retrieved)} bytes")
            print(f"Content matches: {retrieved == test_content}")
        
        # Get statistics
        stats = fallback_manager.get_storage_stats()
        print(f"Storage stats: {json.dumps(stats, indent=2)}")
        
        # Test migration
        success = await fallback_manager.migrate_to_strategy(
            "test_content_1", StorageStrategy.IPFS_ONLY
        )
        print(f"Migration to IPFS-only: {'Success' if success else 'Failed'}")
        
    finally:
        await fallback_manager.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_fallback_storage())