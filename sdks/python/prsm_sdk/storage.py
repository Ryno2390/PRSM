"""
PRSM SDK Storage Client
IPFS storage operations for decentralized data management
"""

import structlog
from typing import Optional, List, Dict, Any, BinaryIO
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from .exceptions import PRSMError, NetworkError

logger = structlog.get_logger(__name__)


class StorageStatus(str, Enum):
    """Status of stored content"""
    UPLOADING = "uploading"
    AVAILABLE = "available"
    PINNED = "pinned"
    UNAVAILABLE = "unavailable"
    EXPIRED = "expired"


class ContentType(str, Enum):
    """Types of content in storage"""
    FILE = "file"
    DATASET = "dataset"
    MODEL = "model"
    DOCUMENT = "document"
    CODE = "code"
    OTHER = "other"


class StorageUploadRequest(BaseModel):
    """Request to upload content to storage"""
    content_type: ContentType = Field(ContentType.FILE, description="Type of content")
    filename: Optional[str] = Field(None, description="Original filename")
    description: Optional[str] = Field(None, description="Content description")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    is_public: bool = Field(False, description="Make content publicly accessible")
    pin: bool = Field(True, description="Pin content for persistence")
    replication: int = Field(3, ge=1, le=10, description="Replication factor")


class StorageUploadResult(BaseModel):
    """Result of content upload"""
    cid: str = Field(..., description="IPFS content identifier")
    size: int = Field(..., description="Content size in bytes")
    content_type: ContentType = Field(..., description="Type of content")
    filename: Optional[str] = Field(None, description="Original filename")
    upload_time: datetime = Field(..., description="Upload timestamp")
    ftns_cost: float = Field(..., description="FTNS cost for upload")
    gateway_url: str = Field(..., description="Gateway URL for access")
    is_pinned: bool = Field(..., description="Whether content is pinned")


class StorageInfo(BaseModel):
    """Information about stored content"""
    cid: str = Field(..., description="IPFS content identifier")
    content_type: ContentType = Field(..., description="Type of content")
    size: int = Field(..., description="Content size in bytes")
    filename: Optional[str] = Field(None, description="Original filename")
    description: Optional[str] = Field(None, description="Content description")
    tags: List[str] = Field(default_factory=list, description="Tags")
    status: StorageStatus = Field(..., description="Current status")
    is_public: bool = Field(..., description="Public accessibility")
    is_pinned: bool = Field(..., description="Pinned status")
    replication: int = Field(..., description="Replication factor")
    created_at: datetime = Field(..., description="Upload timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    owner: str = Field(..., description="Owner address")
    access_count: int = Field(0, description="Number of accesses")


class StorageSearchRequest(BaseModel):
    """Search request for stored content"""
    query: Optional[str] = Field(None, description="Search query")
    content_type: Optional[ContentType] = Field(None, description="Filter by type")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    owner: Optional[str] = Field(None, description="Filter by owner")
    is_public: Optional[bool] = Field(None, description="Filter by public status")
    min_size: Optional[int] = Field(None, description="Minimum size in bytes")
    max_size: Optional[int] = Field(None, description="Maximum size in bytes")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class StorageSearchResult(BaseModel):
    """Search result for stored content"""
    items: List[StorageInfo] = Field(default_factory=list, description="Found items")
    total: int = Field(..., description="Total matching items")
    offset: int = Field(..., description="Current offset")
    limit: int = Field(..., description="Current limit")


class PinInfo(BaseModel):
    """Information about a pinned content"""
    cid: str = Field(..., description="Content identifier")
    pinned_at: datetime = Field(..., description="When pinned")
    size: int = Field(..., description="Content size")
    replication: int = Field(..., description="Replication factor")
    monthly_cost: float = Field(..., description="Monthly FTNS cost")


class StorageClient:
    """
    Client for IPFS storage operations
    
    Provides methods for:
    - Uploading content to IPFS
    - Downloading content
    - Managing pins
    - Searching stored content
    """
    
    def __init__(self, client):
        """
        Initialize storage client
        
        Args:
            client: PRSMClient instance for making API requests
        """
        self._client = client
    
    async def upload(
        self,
        data: BinaryIO,
        content_type: ContentType = ContentType.FILE,
        filename: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: bool = False,
        pin: bool = True,
        replication: int = 3
    ) -> StorageUploadResult:
        """
        Upload content to IPFS storage
        
        Args:
            data: Binary file-like object to upload
            content_type: Type of content
            filename: Original filename
            description: Content description
            tags: Tags for categorization
            is_public: Make content publicly accessible
            pin: Pin content for persistence
            replication: Replication factor
            
        Returns:
            StorageUploadResult with CID and metadata
            
        Example:
            with open("model_weights.bin", "rb") as f:
                result = await client.storage.upload(
                    f,
                    content_type=ContentType.MODEL,
                    description="Trained model weights",
                    tags=["neural-network", "weights"]
                )
            print(f"Uploaded to: {result.cid}")
        """
        request = StorageUploadRequest(
            content_type=content_type,
            filename=filename,
            description=description,
            tags=tags or [],
            is_public=is_public,
            pin=pin,
            replication=replication
        )
        
        # Upload with multipart form data
        response = await self._client._request(
            "POST",
            "/storage/upload",
            json_data=request.model_dump(exclude_none=True),
            files={"file": (filename or "file", data)}
        )
        
        return StorageUploadResult(**response)
    
    async def upload_bytes(
        self,
        data: bytes,
        content_type: ContentType = ContentType.FILE,
        filename: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: bool = False,
        pin: bool = True
    ) -> StorageUploadResult:
        """
        Upload bytes to IPFS storage
        
        Args:
            data: Bytes to upload
            content_type: Type of content
            filename: Original filename
            description: Content description
            tags: Tags for categorization
            is_public: Make content publicly accessible
            pin: Pin content for persistence
            
        Returns:
            StorageUploadResult with CID and metadata
        """
        from io import BytesIO
        
        return await self.upload(
            BytesIO(data),
            content_type=content_type,
            filename=filename,
            description=description,
            tags=tags,
            is_public=is_public,
            pin=pin
        )
    
    async def upload_string(
        self,
        content: str,
        content_type: ContentType = ContentType.DOCUMENT,
        filename: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: bool = False,
        pin: bool = True
    ) -> StorageUploadResult:
        """
        Upload string content to IPFS storage
        
        Args:
            content: String content to upload
            content_type: Type of content
            filename: Original filename
            description: Content description
            tags: Tags for categorization
            is_public: Make content publicly accessible
            pin: Pin content for persistence
            
        Returns:
            StorageUploadResult with CID and metadata
        """
        return await self.upload_bytes(
            content.encode('utf-8'),
            content_type=content_type,
            filename=filename,
            description=description,
            tags=tags,
            is_public=is_public,
            pin=pin
        )
    
    async def download(self, cid: str) -> bytes:
        """
        Download content from IPFS
        
        Args:
            cid: IPFS content identifier
            
        Returns:
            Content as bytes
            
        Example:
            data = await client.storage.download("QmXxx...")
            with open("downloaded_file.bin", "wb") as f:
                f.write(data)
        """
        response = await self._client._request(
            "GET",
            f"/storage/{cid}/download",
            raw_response=True
        )
        
        return response
    
    async def get_string(self, cid: str) -> str:
        """
        Download content as string
        
        Args:
            cid: IPFS content identifier
            
        Returns:
            Content as string
        """
        data = await self.download(cid)
        return data.decode('utf-8')
    
    async def get_info(self, cid: str) -> StorageInfo:
        """
        Get information about stored content
        
        Args:
            cid: IPFS content identifier
            
        Returns:
            StorageInfo with content details
        """
        response = await self._client._request(
            "GET",
            f"/storage/{cid}"
        )
        
        return StorageInfo(**response)
    
    async def pin(self, cid: str, replication: int = 3) -> PinInfo:
        """
        Pin content for persistent storage
        
        Args:
            cid: IPFS content identifier
            replication: Replication factor
            
        Returns:
            PinInfo with pin details
            
        Example:
            pin_info = await client.storage.pin("QmXxx...", replication=5)
            print(f"Pinned at {pin_info.pinned_at}")
        """
        response = await self._client._request(
            "POST",
            f"/storage/{cid}/pin",
            json_data={"replication": replication}
        )
        
        return PinInfo(**response)
    
    async def unpin(self, cid: str) -> bool:
        """
        Unpin content from storage
        
        Args:
            cid: IPFS content identifier
            
        Returns:
            True if unpinned successfully
        """
        response = await self._client._request(
            "POST",
            f"/storage/{cid}/unpin"
        )
        
        return response.get("unpinned", False)
    
    async def list_pins(self, limit: int = 50) -> List[PinInfo]:
        """
        List all pinned content
        
        Args:
            limit: Maximum results
            
        Returns:
            List of PinInfo objects
        """
        response = await self._client._request(
            "GET",
            "/storage/pins",
            params={"limit": limit}
        )
        
        return [PinInfo(**p) for p in response.get("pins", [])]
    
    async def search(
        self,
        query: Optional[str] = None,
        content_type: Optional[ContentType] = None,
        tags: Optional[List[str]] = None,
        owner: Optional[str] = None,
        is_public: Optional[bool] = None,
        limit: int = 20,
        offset: int = 0
    ) -> StorageSearchResult:
        """
        Search for stored content
        
        Args:
            query: Search query string
            content_type: Filter by content type
            tags: Filter by tags
            owner: Filter by owner address
            is_public: Filter by public status
            limit: Maximum results
            offset: Offset for pagination
            
        Returns:
            StorageSearchResult with matching items
        """
        request = StorageSearchRequest(
            query=query,
            content_type=content_type,
            tags=tags,
            owner=owner,
            is_public=is_public,
            limit=limit,
            offset=offset
        )
        
        response = await self._client._request(
            "POST",
            "/storage/search",
            json_data=request.model_dump(exclude_none=True)
        )
        
        return StorageSearchResult(**response)
    
    async def delete(self, cid: str) -> bool:
        """
        Delete content from storage
        
        Args:
            cid: IPFS content identifier
            
        Returns:
            True if deleted successfully
        """
        response = await self._client._request(
            "DELETE",
            f"/storage/{cid}"
        )
        
        return response.get("deleted", False)
    
    async def update_metadata(
        self,
        cid: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: Optional[bool] = None
    ) -> StorageInfo:
        """
        Update content metadata
        
        Args:
            cid: IPFS content identifier
            description: New description
            tags: New tags
            is_public: New public status
            
        Returns:
            Updated StorageInfo
        """
        data = {}
        if description is not None:
            data["description"] = description
        if tags is not None:
            data["tags"] = tags
        if is_public is not None:
            data["is_public"] = is_public
        
        response = await self._client._request(
            "PATCH",
            f"/storage/{cid}",
            json_data=data
        )
        
        return StorageInfo(**response)
    
    async def get_gateway_url(self, cid: str) -> str:
        """
        Get HTTP gateway URL for content
        
        Args:
            cid: IPFS content identifier
            
        Returns:
            Gateway URL string
        """
        info = await self.get_info(cid)
        return info.gateway_url if hasattr(info, 'gateway_url') else f"https://ipfs.io/ipfs/{cid}"
    
    async def estimate_upload_cost(
        self,
        size_bytes: int,
        replication: int = 3,
        duration_days: int = 30
    ) -> float:
        """
        Estimate FTNS cost for uploading content
        
        Args:
            size_bytes: Size of content in bytes
            replication: Replication factor
            duration_days: Storage duration in days
            
        Returns:
            Estimated FTNS cost
        """
        response = await self._client._request(
            "POST",
            "/storage/estimate-cost",
            json_data={
                "size": size_bytes,
                "replication": replication,
                "duration_days": duration_days
            }
        )
        
        return response.get("estimated_cost", 0.0)
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage usage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        response = await self._client._request(
            "GET",
            "/storage/stats"
        )
        
        return response