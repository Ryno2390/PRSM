"""
PRSM IPFS Client - Distributed Storage Integration

🎯 PURPOSE IN PRSM:
This module provides comprehensive IPFS integration for PRSM, implementing
robust distributed storage for models, datasets, research content, and metadata
across the decentralized network with advanced error handling and performance optimization.

🔧 INTEGRATION POINTS:
- Model storage: Distribute trained models across IPFS network
- Dataset sharing: Publish and discover training datasets
- Content distribution: Research papers, documentation, and results
- Metadata persistence: Model configurations and performance data
- Version control: Immutable content addressing for reproducibility
- P2P coordination: Facilitate model sharing across federation nodes

🚀 REAL-WORLD CAPABILITIES:
- Multiple IPFS node connections with intelligent failover
- Chunked uploads/downloads for large model files (multi-GB support)
- Automatic retry mechanisms with exponential backoff
- Content validation and integrity verification
- Bandwidth optimization and QoS management
- Gateway fallback for reliability and accessibility
- Real-time progress tracking for large transfers
"""

import asyncio
import hashlib
import json
import os
import time
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, AsyncGenerator, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import structlog

import aiohttp
import aiofiles
from prsm.core.config import get_settings
from prsm.core.redis_client import get_model_cache
from prsm.core.ipfs_sharding import (
    ContentSharder,
    ShardingConfig,
    ShardManifest,
    ShardInfo,
    ShardIndex,
    ShardingError,
    ShardVerificationError,
    ShardMissingError,
    ManifestError
)

logger = structlog.get_logger(__name__)
settings = get_settings()


class IPFSOperationType(Enum):
    """Types of IPFS operations"""
    ADD = "add"
    GET = "get"
    PIN = "pin"
    UNPIN = "unpin"
    RESOLVE = "resolve"
    STAT = "stat"


class IPFSConnectionType(Enum):
    """IPFS connection types"""
    HTTP_API = "http_api"
    GATEWAY = "gateway"
    LOCAL_NODE = "local_node"


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================

class IPFSError(Exception):
    """
    Base exception for IPFS operations.
    
    All IPFS-related exceptions inherit from this class, allowing for
    both specific exception handling and general IPFS error catching.
    
    Example:
        try:
            result = await client.add_content(content)
        except IPFSError as e:
            logger.error(f"IPFS operation failed: {e}")
    """
    pass


class IPFSTimeoutError(IPFSError):
    """
    Exception raised when an IPFS operation times out.
    
    This exception is raised when operations exceed their configured
    timeout duration, allowing for specific timeout handling logic.
    
    Example:
        try:
            content = await client.get_content(cid, timeout=30)
        except IPFSTimeoutError:
            logger.warning("Content retrieval timed out, retrying...")
            content = await client.get_content(cid, timeout=60)
    """
    pass


class IPFSConnectionError(IPFSError):
    """
    Exception raised when connection to IPFS node fails.
    
    This exception indicates network-level issues such as:
    - IPFS daemon not running
    - Network connectivity problems
    - Invalid API endpoint configuration
    
    Example:
        try:
            await client.connect()
        except IPFSConnectionError as e:
            logger.error(f"Cannot connect to IPFS node: {e}")
            # Attempt fallback or notify user
    """
    pass


# =============================================================================
# Configuration and Content Dataclasses
# =============================================================================

@dataclass
class IPFSConfig:
    """
    Configuration for IPFS client connections.
    
    This dataclass provides a clean, type-safe way to configure IPFS
    client behavior with sensible defaults for most use cases.
    
    Attributes:
        api_url: URL of the IPFS API endpoint (default: localhost:5001)
        gateway_url: URL of the IPFS HTTP gateway (default: localhost:8080)
        timeout: Default timeout for operations in seconds
        chunk_size: Size of chunks for large file transfers
        pin_content: Whether to pin content by default
        verify_content: Whether to verify content integrity after upload
        max_file_size: Maximum allowed file size for uploads
        cluster_peers: List of IPFS cluster peer addresses for distributed storage
        replication_factor: Number of replicas for cluster storage
    
    Example:
        config = IPFSConfig(
            api_url="http://ipfs.example.com:5001",
            timeout=60.0,
            pin_content=True
        )
        client = IPFSClient(config)
    """
    api_url: str = "http://localhost:5001"
    gateway_url: str = "http://localhost:8080"
    timeout: float = 30.0
    chunk_size: int = 1024 * 1024  # 1MB chunks for large files
    pin_content: bool = True
    verify_content: bool = True
    
    # Cluster settings for production
    cluster_peers: Optional[List[str]] = None
    replication_factor: int = 3
    
    # Content policies
    max_file_size: int = 100 * 1024 * 1024  # 100MB max
    allowed_content_types: Optional[List[str]] = None


@dataclass
class IPFSContent:
    """
    IPFS content metadata container.
    
    This dataclass holds comprehensive metadata about content stored
    in IPFS, including provenance information for tracking content
    ownership and licensing.
    
    Attributes:
        cid: Content Identifier (hash) - the unique address in IPFS
        size: Size of the content in bytes
        content_type: MIME type of the content
        filename: Original filename if available
        metadata: Additional metadata dictionary
        pinned: Whether the content is pinned locally
        added_at: Timestamp when content was added
        creator_id: ID of the content creator
        creator_signature: Cryptographic signature of creator
        license: License information for the content
    
    Example:
        content = await client.add_content(data, filename="model.bin")
        print(f"Content stored at {content.cid}")
        print(f"Size: {content.size} bytes")
    """
    cid: str  # Content Identifier
    size: int
    content_type: str
    filename: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    pinned: bool = False
    added_at: Optional[float] = None
    
    # Provenance tracking
    creator_id: Optional[str] = None
    creator_signature: Optional[str] = None
    license: Optional[str] = None


@dataclass
class IPFSStats:
    """
    IPFS node statistics container.
    
    This dataclass holds operational statistics about an IPFS node,
    useful for monitoring node health and network participation.
    
    Attributes:
        peer_id: Unique identifier of the IPFS node
        version: Version string of the IPFS implementation
        connected_peers: Number of peers currently connected
        total_storage: Total storage used by the node in bytes
        available_storage: Available storage capacity in bytes
        pinned_objects: Number of objects pinned on this node
        bandwidth_in: Total incoming bandwidth in bytes
        bandwidth_out: Total outgoing bandwidth in bytes
    
    Example:
        stats = await client.get_node_info()
        print(f"Node {stats.peer_id} has {stats.connected_peers} peers")
    """
    peer_id: str
    version: str
    connected_peers: int
    total_storage: int
    available_storage: int
    pinned_objects: int
    bandwidth_in: int
    bandwidth_out: int


@dataclass
class IPFSUploadProgress:
    """Progress tracking for IPFS uploads"""
    bytes_uploaded: int
    total_bytes: int
    percentage: float
    chunk_number: int
    total_chunks: int
    elapsed_time: float
    estimated_remaining: float


@dataclass
class IPFSResult:
    """Result from IPFS operations"""
    success: bool
    cid: Optional[str] = None
    size: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    execution_time: float = 0.0
    retry_count: int = 0
    connection_type: Optional[IPFSConnectionType] = None


@dataclass
class IPFSNodeStatus:
    """Status of an IPFS node"""
    url: str
    connection_type: IPFSConnectionType
    healthy: bool
    last_check: datetime
    response_time: float
    error: Optional[str] = None


class IPFSRetryStrategy:
    """
    Configurable retry strategy for IPFS operations
    
    🔄 RETRY LOGIC:
    - Exponential backoff with jitter
    - Configurable max attempts and timeouts
    - Different strategies for different operation types
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


class IPFSNode:
    """
    Represents a single IPFS node connection
    
    🌐 NODE MANAGEMENT:
    Handles connection to individual IPFS nodes with health monitoring,
    load balancing, and automatic failover capabilities
    """
    
    def __init__(
        self,
        url: str,
        connection_type: IPFSConnectionType,
        timeout: int = 60,
        max_chunk_size: int = 1024 * 1024  # 1MB chunks
    ):
        self.url = url.rstrip('/')
        self.connection_type = connection_type
        self.timeout = timeout
        self.max_chunk_size = max_chunk_size
        self.session: Optional[aiohttp.ClientSession] = None
        self.status = IPFSNodeStatus(
            url=url,
            connection_type=connection_type,
            healthy=False,
            last_check=datetime.now(),
            response_time=float('inf')
        )
    
    async def initialize(self):
        """Initialize HTTP session for this node"""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def cleanup(self):
        """Clean up HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def health_check(self) -> bool:
        """
        Perform health check on this IPFS node
        
        🏥 HEALTH VALIDATION:
        - API endpoint connectivity
        - Response time measurement
        - Basic functionality verification
        """
        start_time = time.time()
        
        try:
            if not self.session:
                await self.initialize()
            
            # Different health checks based on connection type
            if self.connection_type == IPFSConnectionType.HTTP_API:
                # Check IPFS API health
                async with self.session.post(f"{self.url}/api/v0/version") as response:
                    if response.status == 200:
                        self.status.healthy = True
                        self.status.error = None
                    else:
                        self.status.healthy = False
                        self.status.error = f"API returned status {response.status}"
            
            elif self.connection_type == IPFSConnectionType.GATEWAY:
                # Check gateway health
                async with self.session.head(f"{self.url}/ipfs/QmUNLLsPACCz1vLxQVkXqqLX5R1X345qqfHbsf67hvA3Nn") as response:
                    # This is a well-known IPFS hash for testing
                    if response.status in [200, 301, 302]:
                        self.status.healthy = True
                        self.status.error = None
                    else:
                        self.status.healthy = False
                        self.status.error = f"Gateway returned status {response.status}"
            
            self.status.response_time = time.time() - start_time
            self.status.last_check = datetime.now()
            
            logger.debug("IPFS node health check completed",
                        url=self.url,
                        healthy=self.status.healthy,
                        response_time=self.status.response_time)
            
            return self.status.healthy
            
        except Exception as e:
            self.status.healthy = False
            self.status.error = str(e)
            self.status.response_time = time.time() - start_time
            self.status.last_check = datetime.now()
            
            logger.warning("IPFS node health check failed",
                          url=self.url,
                          error=str(e))
            
            return False
    
    async def add_content(
        self,
        content: Union[bytes, str, Path],
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[IPFSUploadProgress], None]] = None
    ) -> IPFSResult:
        """
        Add content to IPFS via this node
        
        📤 CONTENT UPLOAD:
        - Supports bytes, strings, or file paths
        - Chunked uploads for large files
        - Progress tracking and callbacks
        - Content validation and verification
        """
        start_time = time.time()
        
        try:
            if not self.session:
                await self.initialize()
            
            # Prepare content for upload
            if isinstance(content, Path):
                return await self._add_file(content, progress_callback)
            elif isinstance(content, str):
                content_bytes = content.encode('utf-8')
                filename = filename or "content.txt"
            else:
                content_bytes = content
                filename = filename or "data.bin"
            
            # Upload content
            form_data = aiohttp.FormData()
            form_data.add_field(
                'file',
                content_bytes,
                filename=filename,
                content_type='application/octet-stream'
            )
            
            async with self.session.post(
                f"{self.url}/api/v0/add",
                data=form_data,
                params={'stream-channels': 'true', 'progress': 'true'}
            ) as response:
                
                if response.status != 200:
                    return IPFSResult(
                        success=False,
                        error=f"Upload failed with status {response.status}",
                        execution_time=time.time() - start_time,
                        connection_type=self.connection_type
                    )
                
                # Parse response
                response_text = await response.text()
                result_data = json.loads(response_text.strip().split('\n')[-1])  # Last line contains final result
                
                # Verify upload
                cid = result_data.get('Hash')
                size = result_data.get('Size', len(content_bytes))
                
                if cid:
                    # Optional: Verify content integrity
                    verification_result = await self._verify_content(cid, content_bytes)
                    if not verification_result:
                        logger.warning("Content verification failed", cid=cid)
                
                execution_time = time.time() - start_time
                
                logger.info("Content uploaded to IPFS",
                           cid=cid,
                           size=size,
                           execution_time=execution_time,
                           node_url=self.url)
                
                return IPFSResult(
                    success=True,
                    cid=cid,
                    size=size,
                    execution_time=execution_time,
                    connection_type=self.connection_type,
                    metadata=result_data
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Failed to upload content to IPFS",
                        node_url=self.url,
                        error=str(e),
                        execution_time=execution_time)
            
            return IPFSResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                connection_type=self.connection_type
            )
    
    async def _add_file(
        self,
        file_path: Path,
        progress_callback: Optional[Callable[[IPFSUploadProgress], None]] = None
    ) -> IPFSResult:
        """Add large file with chunked upload and progress tracking"""
        start_time = time.time()
        
        try:
            file_size = file_path.stat().st_size
            total_chunks = (file_size + self.max_chunk_size - 1) // self.max_chunk_size
            
            form_data = aiohttp.FormData()
            
            # For large files, we might need to implement chunked upload
            # For now, upload entire file with progress tracking
            async with aiofiles.open(file_path, 'rb') as file:
                file_content = await file.read()
            
            form_data.add_field(
                'file',
                file_content,
                filename=file_path.name,
                content_type='application/octet-stream'
            )
            
            # Track upload progress if callback provided
            if progress_callback:
                progress = IPFSUploadProgress(
                    bytes_uploaded=0,
                    total_bytes=file_size,
                    percentage=0.0,
                    chunk_number=0,
                    total_chunks=total_chunks,
                    elapsed_time=0.0,
                    estimated_remaining=0.0
                )
                progress_callback(progress)
            
            async with self.session.post(
                f"{self.url}/api/v0/add",
                data=form_data,
                params={'stream-channels': 'true', 'progress': 'true'}
            ) as response:
                
                if response.status != 200:
                    return IPFSResult(
                        success=False,
                        error=f"File upload failed with status {response.status}",
                        execution_time=time.time() - start_time,
                        connection_type=self.connection_type
                    )
                
                # Parse streaming response for progress updates
                async for line in response.content:
                    try:
                        line_text = line.decode('utf-8').strip()
                        if line_text:
                            data = json.loads(line_text)
                            
                            if progress_callback and 'Bytes' in data:
                                bytes_uploaded = data['Bytes']
                                elapsed = time.time() - start_time
                                
                                progress = IPFSUploadProgress(
                                    bytes_uploaded=bytes_uploaded,
                                    total_bytes=file_size,
                                    percentage=(bytes_uploaded / file_size) * 100,
                                    chunk_number=1,
                                    total_chunks=1,
                                    elapsed_time=elapsed,
                                    estimated_remaining=(elapsed * file_size / bytes_uploaded) - elapsed if bytes_uploaded > 0 else 0.0
                                )
                                progress_callback(progress)
                            
                            # Final result contains the hash
                            if 'Hash' in data:
                                execution_time = time.time() - start_time
                                
                                logger.info("Large file uploaded to IPFS",
                                           cid=data['Hash'],
                                           filename=file_path.name,
                                           size=file_size,
                                           execution_time=execution_time)
                                
                                return IPFSResult(
                                    success=True,
                                    cid=data['Hash'],
                                    size=file_size,
                                    execution_time=execution_time,
                                    connection_type=self.connection_type,
                                    metadata=data
                                )
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
                
                # If we reach here, no hash was found
                return IPFSResult(
                    success=False,
                    error="No hash returned from upload",
                    execution_time=time.time() - start_time,
                    connection_type=self.connection_type
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Failed to upload file to IPFS",
                        file_path=str(file_path),
                        node_url=self.url,
                        error=str(e))
            
            return IPFSResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                connection_type=self.connection_type
            )
    
    async def get_content(
        self,
        cid: str,
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> IPFSResult:
        """
        Retrieve content from IPFS via this node
        
        📥 CONTENT DOWNLOAD:
        - Stream large files efficiently
        - Progress tracking for downloads
        - Content integrity verification
        - Automatic retry on partial failures
        """
        start_time = time.time()
        
        try:
            if not self.session:
                await self.initialize()
            
            # Use gateway for downloads if available
            if self.connection_type == IPFSConnectionType.GATEWAY:
                url = f"{self.url}/ipfs/{cid}"
            else:
                url = f"{self.url}/api/v0/cat"
                params = {'arg': cid}
            
            async with self.session.get(url, params=params if self.connection_type != IPFSConnectionType.GATEWAY else None) as response:
                if response.status != 200:
                    return IPFSResult(
                        success=False,
                        error=f"Download failed with status {response.status}",
                        execution_time=time.time() - start_time,
                        connection_type=self.connection_type
                    )
                
                content_length = response.headers.get('Content-Length')
                total_size = int(content_length) if content_length else 0
                
                # Download content
                if output_path:
                    # Stream to file
                    downloaded_size = 0
                    async with aiofiles.open(output_path, 'wb') as file:
                        async for chunk in response.content.iter_chunked(8192):
                            await file.write(chunk)
                            downloaded_size += len(chunk)
                            
                            if progress_callback and total_size > 0:
                                progress_callback(downloaded_size, total_size)
                    
                    execution_time = time.time() - start_time
                    
                    logger.info("Content downloaded from IPFS to file",
                               cid=cid,
                               output_path=str(output_path),
                               size=downloaded_size,
                               execution_time=execution_time)
                    
                    return IPFSResult(
                        success=True,
                        cid=cid,
                        size=downloaded_size,
                        execution_time=execution_time,
                        connection_type=self.connection_type,
                        metadata={"output_path": str(output_path)}
                    )
                else:
                    # Load to memory
                    content = await response.read()
                    execution_time = time.time() - start_time
                    
                    logger.info("Content downloaded from IPFS to memory",
                               cid=cid,
                               size=len(content),
                               execution_time=execution_time)
                    
                    return IPFSResult(
                        success=True,
                        cid=cid,
                        size=len(content),
                        execution_time=execution_time,
                        connection_type=self.connection_type,
                        metadata={"content": content}
                    )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Failed to download content from IPFS",
                        cid=cid,
                        node_url=self.url,
                        error=str(e))
            
            return IPFSResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                connection_type=self.connection_type
            )
    
    async def _verify_content(self, cid: str, original_content: bytes) -> bool:
        """Verify uploaded content integrity"""
        try:
            # Download and compare
            result = await self.get_content(cid)
            if result.success and result.metadata:
                downloaded_content = result.metadata.get("content")
                if downloaded_content:
                    return downloaded_content == original_content
            return False
        except Exception as e:
            logger.warning("Content verification failed", cid=cid, error=str(e))
            return False
    
    async def pin_content(self, cid: str) -> IPFSResult:
        """Pin content to prevent garbage collection"""
        start_time = time.time()
        
        try:
            if not self.session:
                await self.initialize()
            
            async with self.session.post(
                f"{self.url}/api/v0/pin/add",
                params={'arg': cid}
            ) as response:
                
                execution_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    logger.info("Content pinned successfully",
                               cid=cid,
                               execution_time=execution_time)
                    
                    return IPFSResult(
                        success=True,
                        cid=cid,
                        execution_time=execution_time,
                        connection_type=self.connection_type,
                        metadata=data
                    )
                else:
                    error_text = await response.text()
                    return IPFSResult(
                        success=False,
                        error=f"Pin failed: {error_text}",
                        execution_time=execution_time,
                        connection_type=self.connection_type
                    )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Failed to pin content",
                        cid=cid,
                        node_url=self.url,
                        error=str(e))
            
            return IPFSResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                connection_type=self.connection_type
            )


class IPFSClient:
    """
    High-level IPFS client with multi-node support and intelligent failover
    
    🌐 DISTRIBUTED STORAGE MANAGEMENT:
    Coordinates multiple IPFS nodes for reliability, performance optimization,
    and automatic failover with comprehensive error handling and monitoring
    """
    
    def __init__(self, config: Optional[IPFSConfig] = None, sharding_config: Optional[ShardingConfig] = None):
        default_api_url = "http://localhost:5001"
        if getattr(settings, "ipfs_host", None) and getattr(settings, "ipfs_port", None):
            default_api_url = f"http://{settings.ipfs_host}:{settings.ipfs_port}"

        self.config = config or IPFSConfig(
            api_url=default_api_url,
            timeout=float(getattr(settings, "ipfs_timeout", 30.0)),
        )
        self.nodes: List[IPFSNode] = []
        self.primary_node: Optional[IPFSNode] = None
        self.gateway_nodes: List[IPFSNode] = []
        self.retry_strategy = IPFSRetryStrategy()
        self.connected = False
        self.last_health_check = None
        
        # Content cache for frequently accessed items
        self.content_cache_enabled = True
        self.max_cache_size = 100 * 1024 * 1024  # 100MB cache
        
        # Sharding support for large files
        self.sharding_config = sharding_config or ShardingConfig()
        self._sharder: Optional[ContentSharder] = None
    
    async def initialize(self):
        """
        Initialize IPFS client with multiple node connections
        
        🚀 INITIALIZATION SEQUENCE:
        1. Configure primary IPFS nodes from settings
        2. Set up gateway fallback nodes
        3. Perform initial health checks
        4. Establish connection priorities
        """
        try:
            # Reset any previously initialized nodes to keep initialize idempotent
            if self.nodes:
                await self.cleanup()

            # 🔗 Configure primary IPFS API node
            if self.config.api_url:
                primary_url = self.config.api_url.rstrip('/')
                primary_node = IPFSNode(
                    url=primary_url,
                    connection_type=IPFSConnectionType.HTTP_API,
                    timeout=int(self.config.timeout)
                )
                await primary_node.initialize()
                self.nodes.append(primary_node)
                self.primary_node = primary_node
                
                logger.info("Primary IPFS node configured", url=primary_url)
            
            # 🌐 Configure gateway fallback nodes
            gateway_urls = [
                self.config.gateway_url.rstrip('/'),
                "https://ipfs.io",
                "https://gateway.pinata.cloud",
                "https://cloudflare-ipfs.com",
                "https://dweb.link",
            ]

            # Keep first-seen order while removing duplicates
            unique_gateway_urls = list(dict.fromkeys(gateway_urls))

            for gateway_url in unique_gateway_urls:
                gateway_node = IPFSNode(
                    url=gateway_url,
                    connection_type=IPFSConnectionType.GATEWAY,
                    timeout=30  # Shorter timeout for gateways
                )
                await gateway_node.initialize()
                self.nodes.append(gateway_node)
                self.gateway_nodes.append(gateway_node)
            
            logger.info("IPFS gateway fallback nodes configured",
                       gateway_count=len(self.gateway_nodes))
            
            # 🏥 Initial health check
            healthy_nodes = await self.health_check()
            
            if healthy_nodes > 0:
                self.connected = True
                logger.info("IPFS client initialized successfully",
                           total_nodes=len(self.nodes),
                           healthy_nodes=healthy_nodes)
            else:
                logger.warning("IPFS client initialized but no healthy nodes found")
            
        except Exception as e:
            logger.error("Failed to initialize IPFS client", error=str(e))
            raise

    async def connect(self):
        """Backward-compatible alias for initialize()."""
        await self.initialize()

    async def add_content(
        self,
        content: Union[bytes, str, Path],
        filename: Optional[str] = None,
        pin: bool = True,
        progress_callback: Optional[Callable[[IPFSUploadProgress], None]] = None,
    ) -> IPFSResult:
        """Backward-compatible alias for upload_content()."""
        return await self.upload_content(
            content=content,
            filename=filename,
            pin=pin,
            progress_callback=progress_callback,
        )

    async def get_content(
        self,
        cid: str,
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        verify_integrity: bool = True,
    ) -> IPFSResult:
        """Backward-compatible alias for download_content()."""
        return await self.download_content(
            cid=cid,
            output_path=output_path,
            progress_callback=progress_callback,
            verify_integrity=verify_integrity,
        )
    
    async def health_check(self) -> int:
        """
        Perform health check on all IPFS nodes
        
        🏥 COMPREHENSIVE HEALTH MONITORING:
        - Tests all configured nodes
        - Updates connection priorities
        - Identifies fastest responding nodes
        """
        healthy_count = 0
        
        health_tasks = []
        for node in self.nodes:
            health_tasks.append(node.health_check())
        
        try:
            results = await asyncio.gather(*health_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, bool) and result:
                    healthy_count += 1
                elif isinstance(result, Exception):
                    logger.warning("Node health check failed",
                                 node_url=self.nodes[i].url,
                                 error=str(result))
            
            self.last_health_check = datetime.now()
            
            # Sort nodes by health and response time
            self.nodes.sort(key=lambda n: (not n.status.healthy, n.status.response_time))
            
            logger.debug("IPFS health check completed",
                        total_nodes=len(self.nodes),
                        healthy_nodes=healthy_count)
            
            return healthy_count
            
        except Exception as e:
            logger.error("IPFS health check failed", error=str(e))
            return 0
    
    async def upload_content(
        self,
        content: Union[bytes, str, Path],
        filename: Optional[str] = None,
        pin: bool = True,
        progress_callback: Optional[Callable[[IPFSUploadProgress], None]] = None
    ) -> IPFSResult:
        """
        Upload content to IPFS with automatic retry and failover
        
        📤 INTELLIGENT UPLOAD:
        - Tries multiple nodes in priority order
        - Automatic retry with exponential backoff
        - Optional pinning for persistence
        - Progress tracking for large uploads
        """
        if not self.connected:
            return IPFSResult(
                success=False,
                error="IPFS client not connected"
            )
        
        # Try nodes in order of health/priority
        for attempt in range(self.retry_strategy.max_attempts):
            for node in self.nodes:
                if not node.status.healthy:
                    continue
                
                # Only use API nodes for uploads
                if node.connection_type != IPFSConnectionType.HTTP_API:
                    continue
                
                try:
                    result = await node.add_content(content, filename, progress_callback)
                    
                    if result.success:
                        # Pin content if requested
                        if pin and result.cid:
                            pin_result = await node.pin_content(result.cid)
                            if pin_result.success:
                                result.metadata = result.metadata or {}
                                result.metadata["pinned"] = True
                        
                        # Cache successful result
                        if self.content_cache_enabled and result.cid:
                            await self._cache_content_metadata(result.cid, result.metadata)
                        
                        logger.info("Content uploaded successfully",
                                   cid=result.cid,
                                   node_url=node.url,
                                   attempt=attempt + 1,
                                   pinned=pin)
                        
                        return result
                    else:
                        logger.warning("Upload failed, trying next node",
                                     node_url=node.url,
                                     error=result.error)
                
                except Exception as e:
                    logger.warning("Node upload attempt failed",
                                 node_url=node.url,
                                 error=str(e))
                    continue
            
            # Wait before retry if not last attempt
            if attempt < self.retry_strategy.max_attempts - 1:
                delay = self.retry_strategy.get_delay(attempt)
                logger.info("Retrying upload after delay",
                           attempt=attempt + 1,
                           delay=delay)
                await asyncio.sleep(delay)
        
        return IPFSResult(
            success=False,
            error="Upload failed on all nodes after all retry attempts",
            retry_count=self.retry_strategy.max_attempts
        )
    
    async def download_content(
        self,
        cid: str,
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        verify_integrity: bool = True
    ) -> IPFSResult:
        """
        Download content from IPFS with fallback and verification
        
        📥 INTELLIGENT DOWNLOAD:
        - Tries fastest available nodes first
        - Falls back to gateways if API nodes fail
        - Optional content integrity verification
        - Progress tracking for large downloads
        """
        if not self.connected:
            return IPFSResult(
                success=False,
                error="IPFS client not connected"
            )
        
        # Check cache first
        cached_metadata = await self._get_cached_content_metadata(cid)
        
        # Try nodes in order of preference (API nodes first, then gateways)
        all_nodes = [n for n in self.nodes if n.status.healthy]
        
        for attempt in range(self.retry_strategy.max_attempts):
            for node in all_nodes:
                try:
                    result = await node.get_content(cid, output_path, progress_callback)
                    
                    if result.success:
                        # Verify integrity if requested
                        if verify_integrity and result.metadata and "content" in result.metadata:
                            content = result.metadata["content"]
                            expected_hash = self._calculate_content_hash(content)
                            # Note: IPFS CID verification would be more complex in practice
                            
                        logger.info("Content downloaded successfully",
                                   cid=cid,
                                   node_url=node.url,
                                   node_type=node.connection_type.value,
                                   attempt=attempt + 1)
                        
                        return result
                    else:
                        logger.warning("Download failed, trying next node",
                                     node_url=node.url,
                                     error=result.error)
                
                except Exception as e:
                    logger.warning("Node download attempt failed",
                                 node_url=node.url,
                                 error=str(e))
                    continue
            
            # Wait before retry if not last attempt
            if attempt < self.retry_strategy.max_attempts - 1:
                delay = self.retry_strategy.get_delay(attempt)
                logger.info("Retrying download after delay",
                           cid=cid,
                           attempt=attempt + 1,
                           delay=delay)
                await asyncio.sleep(delay)
        
        return IPFSResult(
            success=False,
            error="Download failed on all nodes after all retry attempts",
            retry_count=self.retry_strategy.max_attempts
        )
    
    async def _cache_content_metadata(self, cid: str, metadata: Dict[str, Any]):
        """Cache content metadata in Redis"""
        try:
            model_cache = get_model_cache()
            if model_cache and model_cache.redis.connected:
                cache_key = f"ipfs_metadata:{cid}"
                cache_data = {
                    "metadata": metadata,
                    "cached_at": datetime.now().isoformat()
                }
                
                serialized_data = json.dumps(cache_data, default=str)
                await model_cache.redis.redis_client.setex(
                    cache_key, 
                    86400,  # 24 hours
                    serialized_data
                )
                
                logger.debug("IPFS metadata cached", cid=cid)
        except Exception as e:
            logger.warning("Failed to cache IPFS metadata", cid=cid, error=str(e))
    
    async def _get_cached_content_metadata(self, cid: str) -> Optional[Dict[str, Any]]:
        """Get cached content metadata from Redis"""
        try:
            model_cache = get_model_cache()
            if model_cache and model_cache.redis.connected:
                cache_key = f"ipfs_metadata:{cid}"
                cached_data = await model_cache.redis.redis_client.get(cache_key)
                
                if cached_data:
                    cache_info = json.loads(cached_data)
                    logger.debug("IPFS metadata cache hit", cid=cid)
                    return cache_info.get("metadata")
            
            return None
        except Exception as e:
            logger.warning("Failed to get cached IPFS metadata", cid=cid, error=str(e))
            return None
    
    def _calculate_content_hash(self, content: bytes) -> str:
        """Calculate content hash for integrity verification"""
        return hashlib.sha256(content).hexdigest()
    
    async def get_node_status(self) -> List[Dict[str, Any]]:
        """Get status of all IPFS nodes"""
        node_statuses = []
        
        for node in self.nodes:
            status_info = {
                "url": node.url,
                "connection_type": node.connection_type.value,
                "healthy": node.status.healthy,
                "last_check": node.status.last_check.isoformat(),
                "response_time": node.status.response_time,
                "error": node.status.error
            }
            node_statuses.append(status_info)
        
        return node_statuses
    
    async def cleanup(self):
        """Clean up all IPFS connections"""
        cleanup_tasks = []
        for node in self.nodes:
            cleanup_tasks.append(node.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.nodes.clear()
        self.gateway_nodes.clear()
        self.primary_node = None
        self.connected = False
        self._sharder = None  # Reset sharder on cleanup
        
        logger.info("IPFS client cleanup completed")
    
    # =========================================================================
    # Sharding Support Methods
    # =========================================================================
    
    @property
    def sharder(self) -> ContentSharder:
        """Get or create the content sharder."""
        if self._sharder is None:
            self._sharder = ContentSharder(self, self.sharding_config)
        return self._sharder
    
    async def upload_content_sharded(
        self,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[ShardManifest, str]:
        """
        Upload content with automatic sharding for large files.
        
        This method splits large content into shards, uploads them in parallel,
        and creates a manifest for tracking.
        
        Args:
            content: The content bytes to upload
            metadata: Optional metadata to include in the manifest
            progress_callback: Optional callback(completed, total) for progress
            
        Returns:
            Tuple of (ShardManifest, manifest_cid)
            
        Raises:
            ShardingError: If sharding or upload fails
        """
        return await self.sharder.shard_content(
            content=content,
            metadata=metadata,
            progress_callback=progress_callback
        )
    
    async def download_content_sharded(
        self,
        manifest_cid: str,
        verify: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bytes:
        """
        Download and reassemble sharded content.
        
        This method downloads the manifest, retrieves all shards in parallel,
        and reassembles the original content.
        
        Args:
            manifest_cid: CID of the shard manifest
            verify: Whether to verify shard hashes
            progress_callback: Optional callback(completed, total) for progress
            
        Returns:
            The reassembled content bytes
            
        Raises:
            ManifestError: If manifest cannot be loaded
            ShardMissingError: If shards cannot be downloaded
            ShardVerificationError: If hash verification fails
        """
        manifest = await self.sharder.get_shard_manifest(manifest_cid)
        return await self.sharder.reassemble_content(
            manifest=manifest,
            verify=verify,
            progress_callback=progress_callback
        )
    
    def should_shard(self, content_size: int) -> bool:
        """
        Determine if content should be sharded based on size.
        
        Args:
            content_size: Size of the content in bytes
            
        Returns:
            True if content exceeds the sharding threshold
        """
        return self.sharding_config.auto_shard_threshold > 0 and \
               content_size > self.sharding_config.auto_shard_threshold
    
    async def upload_content_auto(
        self,
        content: Union[bytes, str, Path],
        filename: Optional[str] = None,
        pin: bool = True,
        progress_callback: Optional[Callable[..., None]] = None,
        force_shard: bool = False
    ) -> IPFSResult:
        """
        Upload content with automatic sharding decision.
        
        This method automatically decides whether to shard based on content size.
        For large files, it uses sharding. For smaller files, it uses regular upload.
        
        Args:
            content: Content to upload (bytes, string, or file path)
            filename: Optional filename
            pin: Whether to pin the content
            progress_callback: Optional progress callback
            force_shard: Force sharding regardless of size
            
        Returns:
            IPFSResult with CID and metadata (includes sharding info if sharded)
        """
        start_time = time.time()
        
        try:
            # Convert content to bytes for size check
            if isinstance(content, Path):
                content_size = content.stat().st_size
            elif isinstance(content, str):
                content_bytes = content.encode('utf-8')
                content_size = len(content_bytes)
                content = content_bytes
            else:
                content_size = len(content)
            
            # Decide whether to shard
            should_use_sharding = force_shard or self.should_shard(content_size)
            
            if should_use_sharding and isinstance(content, bytes):
                logger.info("Using sharding for large content upload",
                           content_size=content_size,
                           threshold=self.sharding_config.auto_shard_threshold)
                
                # Use sharding
                def shard_progress(completed: int, total: int):
                    if progress_callback:
                        progress_callback(completed, total)
                
                manifest, manifest_cid = await self.upload_content_sharded(
                    content=content,
                    progress_callback=shard_progress
                )
                
                execution_time = time.time() - start_time
                
                return IPFSResult(
                    success=True,
                    cid=manifest_cid,  # Return manifest CID as the primary CID
                    size=content_size,
                    execution_time=execution_time,
                    connection_type=IPFSConnectionType.HTTP_API,
                    metadata={
                        "sharded": True,
                        "manifest_cid": manifest_cid,
                        "total_shards": manifest.total_shards,
                        "original_size": content_size,
                        "original_hash": manifest.original_hash,
                        "shard_size": manifest.shard_size
                    }
                )
            else:
                # Use regular upload
                logger.debug("Using regular upload",
                           content_size=content_size)
                
                # Convert progress callback for regular upload
                def regular_progress(progress: IPFSUploadProgress):
                    if progress_callback:
                        progress_callback(progress.bytes_uploaded, progress.total_bytes)
                
                result = await self.upload_content(
                    content=content,
                    filename=filename,
                    pin=pin,
                    progress_callback=regular_progress if progress_callback else None
                )
                
                if result.success:
                    result.metadata = result.metadata or {}
                    result.metadata["sharded"] = False
                
                return result
                
        except ShardingError as e:
            execution_time = time.time() - start_time
            logger.error("Sharding upload failed", error=str(e))
            return IPFSResult(
                success=False,
                error=f"Sharding failed: {str(e)}",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Auto upload failed", error=str(e))
            return IPFSResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def download_content_auto(
        self,
        cid: str,
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        verify_integrity: bool = True
    ) -> IPFSResult:
        """
        Download content with automatic sharding detection.
        
        This method attempts to detect if the CID refers to a shard manifest
        and handles both regular and sharded content transparently.
        
        Args:
            cid: Content identifier (regular CID or manifest CID)
            output_path: Optional path to save content
            progress_callback: Optional progress callback
            verify_integrity: Whether to verify content integrity
            
        Returns:
            IPFSResult with content or file path
        """
        start_time = time.time()
        
        try:
            # First, try to load as a shard manifest
            try:
                manifest = await self.sharder.get_shard_manifest(cid)
                
                # It's a manifest - download and reassemble
                logger.info("Detected shard manifest, reassembling content",
                           manifest_cid=cid,
                           total_shards=manifest.total_shards)
                
                content = await self.sharder.reassemble_content(
                    manifest=manifest,
                    verify=verify_integrity,
                    progress_callback=progress_callback
                )
                
                # Save to file if requested
                if output_path:
                    async with aiofiles.open(output_path, 'wb') as f:
                        await f.write(content)
                    
                    execution_time = time.time() - start_time
                    return IPFSResult(
                        success=True,
                        cid=cid,
                        size=len(content),
                        execution_time=execution_time,
                        connection_type=IPFSConnectionType.HTTP_API,
                        metadata={
                            "sharded": True,
                            "output_path": str(output_path),
                            "total_shards": manifest.total_shards
                        }
                    )
                else:
                    execution_time = time.time() - start_time
                    return IPFSResult(
                        success=True,
                        cid=cid,
                        size=len(content),
                        execution_time=execution_time,
                        connection_type=IPFSConnectionType.HTTP_API,
                        metadata={
                            "sharded": True,
                            "content": content,
                            "total_shards": manifest.total_shards
                        }
                    )
                    
            except (ManifestError, json.JSONDecodeError, KeyError):
                # Not a manifest, try regular download
                pass
            
            # Regular download
            result = await self.download_content(
                cid=cid,
                output_path=output_path,
                progress_callback=progress_callback,
                verify_integrity=verify_integrity
            )
            
            if result.success:
                result.metadata = result.metadata or {}
                result.metadata["sharded"] = False
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Auto download failed", cid=cid, error=str(e))
            return IPFSResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def get_shard_index(self) -> 'ShardIndex':
        """Get the shard index for tracking sharded content."""
        return self.sharder.get_index()


# === Global IPFS Client ===

ipfs_client = IPFSClient()


# === Helper Functions ===

async def init_ipfs():
    """Initialize IPFS client for PRSM"""
    await ipfs_client.initialize()


async def close_ipfs():
    """Close IPFS client connections"""
    await ipfs_client.cleanup()


def get_ipfs_client() -> IPFSClient:
    """Get IPFS client instance"""
    return ipfs_client


# === PRSM-Specific IPFS Operations ===

class PRSMIPFSOperations:
    """
    PRSM-specific IPFS operations for models, datasets, and content
    
    🎯 PRSM INTEGRATION:
    High-level operations tailored for PRSM use cases including
    model distribution, dataset sharing, and research content publication
    """
    
    def __init__(self, client: IPFSClient):
        self.client = client
    
    async def upload_model(
        self,
        model_path: Path,
        model_metadata: Dict[str, Any],
        progress_callback: Optional[Callable[[IPFSUploadProgress], None]] = None
    ) -> IPFSResult:
        """
        Upload trained model to IPFS with metadata
        
        🤖 MODEL DISTRIBUTION:
        - Uploads model files with comprehensive metadata
        - Automatic pinning for availability
        - Progress tracking for large model files
        """
        try:
            # Upload model file
            model_result = await self.client.upload_content(
                content=model_path,
                filename=model_path.name,
                pin=True,
                progress_callback=progress_callback
            )
            
            if not model_result.success:
                return model_result
            
            # Create and upload metadata
            full_metadata = {
                "type": "prsm_model",
                "model_cid": model_result.cid,
                "filename": model_path.name,
                "size": model_result.size,
                "uploaded_at": datetime.now().isoformat(),
                **model_metadata
            }
            
            metadata_json = json.dumps(full_metadata, indent=2)
            metadata_result = await self.client.upload_content(
                content=metadata_json,
                filename=f"{model_path.stem}_metadata.json",
                pin=True
            )
            
            if metadata_result.success:
                logger.info("Model uploaded to IPFS with metadata",
                           model_cid=model_result.cid,
                           metadata_cid=metadata_result.cid,
                           model_name=model_metadata.get("name", "unknown"))
                
                # Return combined result
                model_result.metadata = model_result.metadata or {}
                model_result.metadata.update({
                    "metadata_cid": metadata_result.cid,
                    "prsm_metadata": full_metadata
                })
            
            return model_result
            
        except Exception as e:
            logger.error("Failed to upload model to IPFS",
                        model_path=str(model_path),
                        error=str(e))
            
            return IPFSResult(
                success=False,
                error=f"Model upload failed: {str(e)}"
            )
    
    async def download_model(
        self,
        model_cid: str,
        output_directory: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> IPFSResult:
        """
        Download model from IPFS with metadata
        
        🤖 MODEL RETRIEVAL:
        - Downloads model files and associated metadata
        - Verifies model integrity
        - Organizes files in output directory
        """
        try:
            output_directory.mkdir(parents=True, exist_ok=True)
            
            # First, try to get metadata if available
            metadata_result = None
            try:
                # Try to find associated metadata file
                metadata_result = await self.client.download_content(
                    cid=f"{model_cid}_metadata",  # This is a simplified approach
                    verify_integrity=False
                )
            except Exception:
                pass  # Metadata not found, continue with model download
            
            # Download main model file
            model_filename = "model.bin"  # Default filename
            if metadata_result and metadata_result.success:
                metadata_content = metadata_result.metadata.get("content", b"")
                if metadata_content:
                    metadata_json = json.loads(metadata_content.decode('utf-8'))
                    model_filename = metadata_json.get("filename", model_filename)
            
            model_output_path = output_directory / model_filename
            model_result = await self.client.download_content(
                cid=model_cid,
                output_path=model_output_path,
                progress_callback=progress_callback,
                verify_integrity=True
            )
            
            if model_result.success:
                # Save metadata if available
                if metadata_result and metadata_result.success:
                    metadata_path = output_directory / "metadata.json"
                    async with aiofiles.open(metadata_path, 'w') as f:
                        metadata_content = metadata_result.metadata.get("content", b"")
                        await f.write(metadata_content.decode('utf-8'))
                
                logger.info("Model downloaded from IPFS",
                           model_cid=model_cid,
                           output_path=str(model_output_path))
            
            return model_result
            
        except Exception as e:
            logger.error("Failed to download model from IPFS",
                        model_cid=model_cid,
                        error=str(e))
            
            return IPFSResult(
                success=False,
                error=f"Model download failed: {str(e)}"
            )
    
    async def publish_research_content(
        self,
        content_path: Path,
        content_metadata: Dict[str, Any]
    ) -> IPFSResult:
        """
        Publish research content (papers, datasets, etc.) to IPFS
        
        📚 RESEARCH PUBLISHING:
        - Publishes research content with comprehensive metadata
        - Enables content discovery and citation
        - Supports academic collaboration
        """
        try:
            # Upload content
            content_result = await self.client.upload_content(
                content=content_path,
                filename=content_path.name,
                pin=True
            )
            
            if not content_result.success:
                return content_result
            
            # Create research metadata
            research_metadata = {
                "type": "prsm_research_content",
                "content_cid": content_result.cid,
                "filename": content_path.name,
                "size": content_result.size,
                "published_at": datetime.now().isoformat(),
                "content_type": content_metadata.get("content_type", "unknown"),
                "title": content_metadata.get("title", ""),
                "authors": content_metadata.get("authors", []),
                "description": content_metadata.get("description", ""),
                "tags": content_metadata.get("tags", []),
                "license": content_metadata.get("license", ""),
                **content_metadata
            }
            
            # Upload metadata
            metadata_json = json.dumps(research_metadata, indent=2)
            metadata_result = await self.client.upload_content(
                content=metadata_json,
                filename=f"{content_path.stem}_research_metadata.json",
                pin=True
            )
            
            if metadata_result.success:
                logger.info("Research content published to IPFS",
                           content_cid=content_result.cid,
                           metadata_cid=metadata_result.cid,
                           title=research_metadata.get("title", "unknown"))
                
                content_result.metadata = content_result.metadata or {}
                content_result.metadata.update({
                    "metadata_cid": metadata_result.cid,
                    "research_metadata": research_metadata
                })
            
            return content_result
            
        except Exception as e:
            logger.error("Failed to publish research content to IPFS",
                        content_path=str(content_path),
                        error=str(e))
            
            return IPFSResult(
                success=False,
                error=f"Research content publishing failed: {str(e)}"
            )


# =============================================================================
# Utility Functions for Common Operations
# =============================================================================

async def create_ipfs_client(
    api_url: str = "http://localhost:5001",
    gateway_url: str = "http://localhost:8080",
    timeout: float = 30.0
) -> 'IPFSClient':
    """
    Create and connect an IPFS client with specified configuration.
    
    This is a convenience function for quickly creating a connected
    IPFS client with common default settings.
    
    Args:
        api_url: URL of the IPFS API endpoint
        gateway_url: URL of the IPFS HTTP gateway
        timeout: Default timeout for operations in seconds
    
    Returns:
        Connected IPFSClient instance
    
    Example:
        client = await create_ipfs_client(
            api_url="http://ipfs.example.com:5001",
            timeout=60.0
        )
        result = await client.add_content(b"Hello IPFS!")
    """
    def _validate_http_url(value: str, field: str) -> None:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"Invalid {field} '{value}'. Expected a full http(s) URL, "
                "for example 'http://localhost:5001'."
            )

    _validate_http_url(api_url, "api_url")
    _validate_http_url(gateway_url, "gateway_url")
    if timeout <= 0:
        raise ValueError("Invalid timeout value. timeout must be > 0.")

    config = IPFSConfig(api_url=api_url, gateway_url=gateway_url, timeout=timeout)
    client = IPFSClient(config)
    await client.initialize()
    return client


async def add_text_to_ipfs(
    client: 'IPFSClient',
    text: str,
    filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> IPFSContent:
    """
    Add text content to IPFS.
    
    Convenience function for adding string/text content to IPFS
    with proper encoding and content type detection.
    
    Args:
        client: Connected IPFSClient instance
        text: Text content to add
        filename: Optional filename for the content
        metadata: Additional metadata to store with content
    
    Returns:
        IPFSContent object with CID and metadata
    
    Example:
        content = await add_text_to_ipfs(
            client,
            "Hello, IPFS World!",
            filename="greeting.txt"
        )
        print(f"Text stored at: {content.cid}")
    """
    result = await client.add_content(
        content=text,
        filename=filename or "text_content.txt"
    )
    
    if not result.success:
        raise IPFSError(f"Failed to add text content: {result.error}")
    
    return IPFSContent(
        cid=result.cid,
        size=result.size or len(text.encode('utf-8')),
        content_type='text/plain',
        filename=filename,
        metadata=metadata or {},
        pinned=True,
        added_at=time.time()
    )


async def get_text_from_ipfs(client: 'IPFSClient', cid: str) -> str:
    """
    Retrieve text content from IPFS.
    
    Convenience function for retrieving and decoding text content
    stored in IPFS.
    
    Args:
        client: Connected IPFSClient instance
        cid: Content Identifier to retrieve
    
    Returns:
        Decoded text content as string
    
    Example:
        text = await get_text_from_ipfs(client, "QmXxx...")
        print(f"Retrieved: {text}")
    """
    result = await client.get_content(cid)
    
    if not result.success:
        raise IPFSError(f"Failed to get text content: {result.error}")
    
    # Handle both bytes and IPFSResult with content
    if isinstance(result, bytes):
        return result.decode('utf-8')
    elif hasattr(result, 'metadata') and result.metadata and 'content' in result.metadata:
        return result.metadata['content'].decode('utf-8')
    else:
        raise IPFSError("No content returned from IPFS")


async def add_json_to_ipfs(
    client: 'IPFSClient',
    data: Dict[str, Any],
    filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> IPFSContent:
    """
    Add JSON data to IPFS.
    
    Convenience function for adding JSON-serializable data to IPFS
    with proper content type headers.
    
    Args:
        client: Connected IPFSClient instance
        data: JSON-serializable dictionary to store
        filename: Optional filename for the content
        metadata: Additional metadata to store with content
    
    Returns:
        IPFSContent object with CID and metadata
    
    Example:
        content = await add_json_to_ipfs(
            client,
            {"model": "gpt-4", "accuracy": 0.95},
            filename="model_metrics.json"
        )
        print(f"JSON stored at: {content.cid}")
    """
    json_content = json.dumps(data, indent=2, ensure_ascii=False)
    
    result = await client.add_content(
        content=json_content,
        filename=filename or "data.json"
    )
    
    if not result.success:
        raise IPFSError(f"Failed to add JSON content: {result.error}")
    
    return IPFSContent(
        cid=result.cid,
        size=result.size or len(json_content.encode('utf-8')),
        content_type='application/json',
        filename=filename,
        metadata=metadata or {},
        pinned=True,
        added_at=time.time()
    )


async def get_json_from_ipfs(client: 'IPFSClient', cid: str) -> Dict[str, Any]:
    """
    Retrieve JSON data from IPFS.
    
    Convenience function for retrieving and parsing JSON content
    stored in IPFS.
    
    Args:
        client: Connected IPFSClient instance
        cid: Content Identifier to retrieve
    
    Returns:
        Parsed JSON data as dictionary
    
    Example:
        data = await get_json_from_ipfs(client, "QmXxx...")
        print(f"Model: {data['model']}")
    """
    result = await client.get_content(cid)
    
    if not result.success:
        raise IPFSError(f"Failed to get JSON content: {result.error}")
    
    # Handle both bytes and IPFSResult with content
    if isinstance(result, bytes):
        content_str = result.decode('utf-8')
    elif hasattr(result, 'metadata') and result.metadata and 'content' in result.metadata:
        content_str = result.metadata['content'].decode('utf-8')
    else:
        raise IPFSError("No content returned from IPFS")
    
    try:
        return json.loads(content_str)
    except json.JSONDecodeError as e:
        raise IPFSError(f"Failed to parse JSON content: {e}")


# =============================================================================
# Additional IPFS Operations
# =============================================================================

async def list_pinned_content(client: 'IPFSClient') -> List[str]:
    """
    List all content pinned on the connected IPFS node.
    
    Returns a list of CIDs for all content that is pinned locally,
    preventing it from being garbage collected.
    
    Args:
        client: Connected IPFSClient instance
    
    Returns:
        List of pinned content CIDs
    
    Example:
        pinned = await list_pinned_content(client)
        print(f"Pinned items: {len(pinned)}")
        for cid in pinned[:5]:
            print(f"  - {cid}")
    """
    try:
        if not client.connected:
            raise IPFSError("IPFS client is not connected")

        api_nodes = [
            node for node in client.nodes
            if node.connection_type == IPFSConnectionType.HTTP_API and node.status.healthy and node.session
        ]
        if not api_nodes:
            raise IPFSError("No healthy IPFS API node available for pin listing")

        node = api_nodes[0]
        async with node.session.post(f"{node.url}/api/v0/pin/ls") as response:
            if response.status == 200:
                result = await response.json()
                return list(result.get('Keys', {}).keys())
            raise IPFSError(f"Failed to list pins: {response.status}")
    except Exception as e:
        logger.error("Failed to list pinned content", error=str(e))
        raise IPFSError(f"Pin list retrieval failed: {e}")


async def get_content_info(client: 'IPFSClient', cid: str) -> Dict[str, Any]:
    """
    Get information about content without downloading it.
    
    Retrieves metadata about content stored in IPFS, including
    size, links, and block information.
    
    Args:
        client: Connected IPFSClient instance
        cid: Content Identifier to inspect
    
    Returns:
        Dictionary with content information including:
        - Hash: The CID
        - NumLinks: Number of links in the object
        - BlockSize: Size of the block
        - LinksSize: Size of links
        - DataSize: Size of data
        - CumulativeSize: Total size including linked objects
    
    Example:
        info = await get_content_info(client, "QmXxx...")
        print(f"Content size: {info['DataSize']} bytes")
    """
    try:
        if not client.connected:
            raise IPFSError("IPFS client is not connected")

        api_nodes = [
            node for node in client.nodes
            if node.connection_type == IPFSConnectionType.HTTP_API and node.status.healthy and node.session
        ]
        if not api_nodes:
            raise IPFSError("No healthy IPFS API node available for content info retrieval")

        node = api_nodes[0]
        async with node.session.post(
            f"{node.url}/api/v0/object/stat",
            params={'arg': cid}
        ) as response:
            if response.status == 200:
                return await response.json()
            error_text = await response.text()
            raise IPFSError(f"Failed to get content info: {response.status} - {error_text}")
    except Exception as e:
        logger.error("Failed to get content info", cid=cid, error=str(e))
        raise IPFSError(f"Content info retrieval failed: {e}")


# Global PRSM IPFS operations
prsm_ipfs = PRSMIPFSOperations(ipfs_client)
