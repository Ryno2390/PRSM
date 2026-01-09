"""
PRSM IPFS Integration Client

Production-ready IPFS client for content storage and retrieval.
Provides content addressing, pinning, and distributed storage for PRSM.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, BinaryIO
import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class IPFSConfig:
    """Configuration for IPFS client"""
    api_url: str = "http://localhost:5001"
    gateway_url: str = "http://localhost:8080"
    timeout: float = 30.0
    chunk_size: int = 1024 * 1024  # 1MB chunks for large files
    pin_content: bool = True
    verify_content: bool = True
    
    # Cluster settings for production
    cluster_peers: List[str] = None
    replication_factor: int = 3
    
    # Content policies
    max_file_size: int = 100 * 1024 * 1024  # 100MB max
    allowed_content_types: List[str] = None


@dataclass
class IPFSContent:
    """IPFS content metadata"""
    cid: str  # Content Identifier
    size: int
    content_type: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = None
    pinned: bool = False
    added_at: Optional[float] = None
    
    # Provenance tracking
    creator_id: Optional[str] = None
    creator_signature: Optional[str] = None
    license: Optional[str] = None


@dataclass
class IPFSStats:
    """IPFS node statistics"""
    peer_id: str
    version: str
    connected_peers: int
    total_storage: int
    available_storage: int
    pinned_objects: int
    bandwidth_in: int
    bandwidth_out: int


class IPFSError(Exception):
    """Base exception for IPFS operations"""
    pass


class IPFSTimeoutError(IPFSError):
    """IPFS operation timeout"""
    pass


class IPFSClient:
    """
    Production IPFS client for PRSM content storage
    
    Features:
    - Async content upload and retrieval
    - Automatic content pinning and verification
    - Content metadata tracking
    - Cluster support for distributed storage
    - Performance monitoring and statistics
    - Content deduplication via CID
    """
    
    def __init__(self, config: IPFSConfig = None):
        self.config = config or IPFSConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._stats = {
            'uploads': 0,
            'downloads': 0,
            'bytes_uploaded': 0,
            'bytes_downloaded': 0,
            'errors': 0
        }
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Initialize IPFS client connection"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'PRSM-IPFS-Client/1.0'}
        )
        
        # Test connection
        try:
            await self.get_node_info()
            logger.info(f"✅ Connected to IPFS node at {self.config.api_url}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to IPFS node: {e}")
            raise IPFSError(f"IPFS connection failed: {e}")
    
    async def disconnect(self):
        """Close IPFS client connection"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_node_info(self) -> IPFSStats:
        """Get IPFS node information and statistics"""
        try:
            # Get basic node info
            async with self.session.post(f"{self.config.api_url}/api/v0/id") as resp:
                if resp.status != 200:
                    raise IPFSError(f"Failed to get node info: {resp.status}")
                node_info = await resp.json()
            
            # Get repository stats
            async with self.session.post(f"{self.config.api_url}/api/v0/repo/stat") as resp:
                if resp.status != 200:
                    logger.warning("Could not get repo stats")
                    repo_stats = {'RepoSize': 0, 'NumObjects': 0}
                else:
                    repo_stats = await resp.json()
            
            # Get bandwidth stats
            async with self.session.post(f"{self.config.api_url}/api/v0/stats/bw") as resp:
                if resp.status != 200:
                    logger.warning("Could not get bandwidth stats")
                    bw_stats = {'TotalIn': 0, 'TotalOut': 0}
                else:
                    bw_stats = await resp.json()
            
            # Get swarm peers
            async with self.session.post(f"{self.config.api_url}/api/v0/swarm/peers") as resp:
                if resp.status != 200:
                    logger.warning("Could not get peer count")
                    peer_count = 0
                else:
                    peers_data = await resp.json()
                    peer_count = len(peers_data.get('Peers', []))
            
            return IPFSStats(
                peer_id=node_info.get('ID', 'unknown'),
                version=node_info.get('AgentVersion', 'unknown'),
                connected_peers=peer_count,
                total_storage=repo_stats.get('RepoSize', 0),
                available_storage=0,  # Would need additional API calls
                pinned_objects=repo_stats.get('NumObjects', 0),
                bandwidth_in=bw_stats.get('TotalIn', 0),
                bandwidth_out=bw_stats.get('TotalOut', 0)
            )
        
        except Exception as e:
            logger.error(f"Failed to get IPFS node info: {e}")
            raise IPFSError(f"Node info retrieval failed: {e}")
    
    async def add_content(self, 
                         content: Union[str, bytes, BinaryIO], 
                         filename: str = None,
                         metadata: Dict[str, Any] = None,
                         pin: bool = None) -> IPFSContent:
        """
        Add content to IPFS and return content information
        
        Args:
            content: Content to add (text, bytes, or file-like object)
            filename: Optional filename for the content
            metadata: Additional metadata to store with content
            pin: Whether to pin the content (defaults to config setting)
        
        Returns:
            IPFSContent object with CID and metadata
        """
        start_time = time.time()
        pin = pin if pin is not None else self.config.pin_content
        
        try:
            # Prepare content for upload
            if isinstance(content, str):
                content_bytes = content.encode('utf-8')
                content_type = 'text/plain'
            elif isinstance(content, bytes):
                content_bytes = content
                content_type = 'application/octet-stream'
            else:
                # File-like object
                content_bytes = content.read()
                content_type = 'application/octet-stream'
            
            # Check file size limits
            if len(content_bytes) > self.config.max_file_size:
                raise IPFSError(f"Content size {len(content_bytes)} exceeds limit {self.config.max_file_size}")
            
            # Create form data for upload
            data = aiohttp.FormData()
            data.add_field('file', 
                          content_bytes, 
                          filename=filename or 'content',
                          content_type=content_type)
            
            # Add to IPFS
            params = {'pin': 'true' if pin else 'false'}
            async with self.session.post(
                f"{self.config.api_url}/api/v0/add",
                data=data,
                params=params
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise IPFSError(f"Failed to add content: {resp.status} - {error_text}")
                
                # Parse response
                response_lines = (await resp.text()).strip().split('\n')
                # Take the last line which contains the final hash
                result = json.loads(response_lines[-1])
                cid = result['Hash']
            
            # Verify content if configured
            if self.config.verify_content:
                await self._verify_content(cid, content_bytes)
            
            # Update statistics
            self._stats['uploads'] += 1
            self._stats['bytes_uploaded'] += len(content_bytes)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Added content to IPFS: {cid} ({len(content_bytes)} bytes, {processing_time:.2f}s)")
            
            return IPFSContent(
                cid=cid,
                size=len(content_bytes),
                content_type=content_type,
                filename=filename,
                metadata=metadata or {},
                pinned=pin,
                added_at=time.time()
            )
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"❌ Failed to add content to IPFS: {e}")
            raise IPFSError(f"Content upload failed: {e}")
    
    async def get_content(self, cid: str, 
                         timeout: float = None) -> bytes:
        """
        Retrieve content from IPFS by CID
        
        Args:
            cid: Content Identifier
            timeout: Optional timeout override
        
        Returns:
            Content bytes
        """
        start_time = time.time()
        
        try:
            # Use custom timeout if provided
            session_timeout = timeout or self.config.timeout
            timeout_obj = aiohttp.ClientTimeout(total=session_timeout)
            
            async with self.session.post(
                f"{self.config.api_url}/api/v0/cat",
                params={'arg': cid},
                timeout=timeout_obj
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise IPFSError(f"Failed to get content {cid}: {resp.status} - {error_text}")
                
                content = await resp.read()
            
            # Update statistics
            self._stats['downloads'] += 1
            self._stats['bytes_downloaded'] += len(content)
            
            processing_time = time.time() - start_time
            logger.debug(f"Retrieved content {cid} ({len(content)} bytes, {processing_time:.2f}s)")
            
            return content
        
        except asyncio.TimeoutError:
            self._stats['errors'] += 1
            raise IPFSTimeoutError(f"Timeout retrieving content {cid}")
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"❌ Failed to get content {cid}: {e}")
            raise IPFSError(f"Content retrieval failed: {e}")
    
    async def get_content_info(self, cid: str) -> Dict[str, Any]:
        """Get information about content without downloading it"""
        try:
            async with self.session.post(
                f"{self.config.api_url}/api/v0/object/stat",
                params={'arg': cid}
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise IPFSError(f"Failed to get content info {cid}: {resp.status} - {error_text}")
                
                return await resp.json()
        
        except Exception as e:
            logger.error(f"❌ Failed to get content info {cid}: {e}")
            raise IPFSError(f"Content info retrieval failed: {e}")
    
    async def pin_content(self, cid: str) -> bool:
        """Pin content to prevent garbage collection"""
        try:
            async with self.session.post(
                f"{self.config.api_url}/api/v0/pin/add",
                params={'arg': cid}
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.warning(f"Failed to pin content {cid}: {resp.status} - {error_text}")
                    return False
                
                logger.debug(f"Pinned content {cid}")
                return True
        
        except Exception as e:
            logger.error(f"❌ Failed to pin content {cid}: {e}")
            return False
    
    async def unpin_content(self, cid: str) -> bool:
        """Unpin content to allow garbage collection"""
        try:
            async with self.session.post(
                f"{self.config.api_url}/api/v0/pin/rm",
                params={'arg': cid}
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.warning(f"Failed to unpin content {cid}: {resp.status} - {error_text}")
                    return False
                
                logger.debug(f"Unpinned content {cid}")
                return True
        
        except Exception as e:
            logger.error(f"❌ Failed to unpin content {cid}: {e}")
            return False
    
    async def list_pinned_content(self) -> List[str]:
        """List all pinned content CIDs"""
        try:
            async with self.session.post(f"{self.config.api_url}/api/v0/pin/ls") as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise IPFSError(f"Failed to list pinned content: {resp.status} - {error_text}")
                
                result = await resp.json()
                return list(result.get('Keys', {}).keys())
        
        except Exception as e:
            logger.error(f"❌ Failed to list pinned content: {e}")
            raise IPFSError(f"Pin list retrieval failed: {e}")
    
    async def _verify_content(self, cid: str, original_content: bytes):
        """Verify that uploaded content matches original"""
        try:
            retrieved_content = await self.get_content(cid)
            if retrieved_content != original_content:
                raise IPFSError(f"Content verification failed for {cid}")
            logger.debug(f"Content verification successful for {cid}")
        except Exception as e:
            logger.error(f"Content verification failed for {cid}: {e}")
            raise
    
    async def search_content(self, 
                           content_type: str = None,
                           size_range: tuple = None,
                           metadata_filter: Dict[str, Any] = None) -> List[str]:
        """
        Search for content by attributes (requires custom indexing)
        Note: This is a placeholder for future advanced search capabilities
        """
        # This would require additional IPFS indexing infrastructure
        # For now, return pinned content that matches basic criteria
        pinned_cids = await self.list_pinned_content()
        
        # Basic filtering would be implemented here
        # In production, this would integrate with a separate search index
        
        return pinned_cids
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client operation statistics"""
        return {
            'operations': self._stats.copy(),
            'config': {
                'api_url': self.config.api_url,
                'gateway_url': self.config.gateway_url,
                'pin_content': self.config.pin_content,
                'verify_content': self.config.verify_content,
                'max_file_size': self.config.max_file_size
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on IPFS connection"""
        try:
            start_time = time.time()
            node_info = await self.get_node_info()
            response_time = time.time() - start_time
            
            return {
                'healthy': True,
                'response_time': response_time,
                'node_info': {
                    'peer_id': node_info.peer_id,
                    'version': node_info.version,
                    'connected_peers': node_info.connected_peers,
                    'total_storage': node_info.total_storage
                },
                'stats': self._stats
            }
        
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'stats': self._stats
            }


# Utility functions

async def create_ipfs_client(api_url: str = "http://localhost:5001",
                           gateway_url: str = "http://localhost:8080") -> IPFSClient:
    """Create and connect IPFS client"""
    config = IPFSConfig(api_url=api_url, gateway_url=gateway_url)
    client = IPFSClient(config)
    await client.connect()
    return client


async def add_text_to_ipfs(client: IPFSClient, 
                          text: str, 
                          filename: str = None,
                          metadata: Dict[str, Any] = None) -> IPFSContent:
    """Convenience function to add text content to IPFS"""
    return await client.add_content(
        content=text,
        filename=filename or "text_content.txt",
        metadata=metadata
    )


async def add_json_to_ipfs(client: IPFSClient,
                          data: Dict[str, Any],
                          filename: str = None,
                          metadata: Dict[str, Any] = None) -> IPFSContent:
    """Convenience function to add JSON data to IPFS"""
    json_content = json.dumps(data, indent=2, ensure_ascii=False)
    return await client.add_content(
        content=json_content,
        filename=filename or "data.json",
        metadata=metadata
    )


async def get_text_from_ipfs(client: IPFSClient, cid: str) -> str:
    """Convenience function to get text content from IPFS"""
    content_bytes = await client.get_content(cid)
    return content_bytes.decode('utf-8')


async def get_json_from_ipfs(client: IPFSClient, cid: str) -> Dict[str, Any]:
    """Convenience function to get JSON data from IPFS"""
    content_bytes = await client.get_content(cid)
    return json.loads(content_bytes.decode('utf-8'))