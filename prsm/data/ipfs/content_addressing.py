"""
PRSM Content Addressing System

Advanced content-based addressing using IPFS CIDs for scientific content.
Provides content provenance, versioning, and deduplication for PRSM.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

from .ipfs_client import IPFSClient, IPFSContent

logger = logging.getLogger(__name__)


class ContentStatus(str, Enum):
    """Status of content in the PRSM system"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    FLAGGED = "flagged"
    DELETED = "deleted"


class ContentCategory(str, Enum):
    """Categories of scientific content"""
    RESEARCH_PAPER = "research_paper"
    DATASET = "dataset"
    CODE_REPOSITORY = "code_repository"
    PROTOCOL = "protocol"
    REVIEW = "review"
    PREPRINT = "preprint"
    SUPPLEMENT = "supplement"
    PRESENTATION = "presentation"


@dataclass
class ContentVersion:
    """Version information for content"""
    version: str
    cid: str
    created_at: datetime
    changes: str
    parent_version: Optional[str] = None


@dataclass
class ContentLicense:
    """License information for content"""
    license_type: str  # e.g., "CC-BY-4.0", "MIT", "Apache-2.0"
    license_url: str
    commercial_use: bool = True
    modification_allowed: bool = True
    attribution_required: bool = True


@dataclass
class ContentProvenance:
    """Provenance tracking for content"""
    creator_id: str
    creator_name: str
    creator_signature: Optional[str] = None
    institution: Optional[str] = None
    funding_source: Optional[str] = None
    ethics_approval: Optional[str] = None
    conflict_of_interest: Optional[str] = None


@dataclass
class ContentMetrics:
    """Usage and quality metrics for content"""
    view_count: int = 0
    download_count: int = 0
    citation_count: int = 0
    replication_count: int = 0
    quality_score: float = 0.0
    peer_review_score: float = 0.0
    community_rating: float = 0.0
    last_accessed: Optional[datetime] = None


@dataclass
class AddressedContent:
    """Complete content object with IPFS addressing"""
    # Core content
    cid: str  # Primary IPFS CID
    title: str
    description: str
    content_type: str
    category: ContentCategory
    status: ContentStatus
    
    # Versioning
    version: str
    versions: List[ContentVersion]
    
    # Provenance and licensing
    provenance: ContentProvenance
    license: ContentLicense
    
    # Metadata
    keywords: List[str]
    tags: List[str]
    
    # Metrics and tracking
    metrics: ContentMetrics
    
    # Technical metadata
    size_bytes: int
    checksum: str
    created_at: datetime
    updated_at: datetime
    
    # Optional fields with defaults
    doi: Optional[str] = None
    related_content: Optional[List[str]] = None  # CIDs of related content
    
    # FTNS economics
    royalty_rate: float = 0.08  # 8% default creator royalty
    price_per_access: float = 0.0  # Free by default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO format
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        
        # Convert version datetimes
        for version in data['versions']:
            version['created_at'] = version['created_at'].isoformat()
        
        # Convert metrics datetime
        if data['metrics']['last_accessed']:
            data['metrics']['last_accessed'] = self.metrics.last_accessed.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AddressedContent':
        """Create from dictionary"""
        # Convert datetime strings back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert version datetimes
        for version_data in data['versions']:
            version_data['created_at'] = datetime.fromisoformat(version_data['created_at'])
        
        # Convert metrics datetime
        if data['metrics']['last_accessed']:
            data['metrics']['last_accessed'] = datetime.fromisoformat(data['metrics']['last_accessed'])
        
        # Create objects from nested dictionaries
        data['provenance'] = ContentProvenance(**data['provenance'])
        data['license'] = ContentLicense(**data['license'])
        data['metrics'] = ContentMetrics(**data['metrics'])
        data['versions'] = [ContentVersion(**v) for v in data['versions']]
        
        return cls(**data)


class ContentAddressingSystem:
    """
    Advanced content addressing system for PRSM scientific content
    
    Features:
    - Content-based addressing using IPFS CIDs
    - Version control and content history
    - Provenance tracking and digital signatures
    - Automated metadata extraction and indexing
    - Deduplication and content verification
    - License management and royalty tracking
    """
    
    def __init__(self, ipfs_client: IPFSClient):
        self.ipfs_client = ipfs_client
        self.content_registry: Dict[str, AddressedContent] = {}
        self.cid_to_metadata: Dict[str, str] = {}  # CID -> metadata CID mapping
        
        # Performance tracking
        self.stats = {
            'content_added': 0,
            'content_retrieved': 0,
            'versions_created': 0,
            'duplicates_detected': 0,
            'total_storage_bytes': 0
        }
    
    async def add_content(self,
                         content: Union[str, bytes],
                         title: str,
                         description: str,
                         content_type: str,
                         category: ContentCategory,
                         provenance: ContentProvenance,
                         license: ContentLicense,
                         keywords: List[str] = None,
                         tags: List[str] = None,
                         filename: str = None) -> AddressedContent:
        """
        Add new content to the addressing system
        
        Args:
            content: The actual content (text, bytes, etc.)
            title: Human-readable title
            description: Content description
            content_type: MIME type or content format
            category: Content category enum
            provenance: Creator and source information
            license: License information
            keywords: Search keywords
            tags: Classification tags
            filename: Optional filename
        
        Returns:
            AddressedContent object with CID and metadata
        """
        start_time = time.time()
        
        try:
            # Add content to IPFS
            ipfs_content = await self.ipfs_client.add_content(
                content=content,
                filename=filename,
                metadata={
                    'title': title,
                    'content_type': content_type,
                    'category': category.value
                }
            )
            
            # Check for duplicates
            existing_content = await self._check_for_duplicate(ipfs_content.cid)
            if existing_content:
                logger.info(f"Duplicate content detected: {ipfs_content.cid}")
                self.stats['duplicates_detected'] += 1
                return existing_content
            
            # Create content metadata
            now = datetime.now()
            initial_version = ContentVersion(
                version="1.0.0",
                cid=ipfs_content.cid,
                created_at=now,
                changes="Initial version"
            )
            
            addressed_content = AddressedContent(
                cid=ipfs_content.cid,
                title=title,
                description=description,
                content_type=content_type,
                category=category,
                status=ContentStatus.PUBLISHED,
                version="1.0.0",
                versions=[initial_version],
                provenance=provenance,
                license=license,
                keywords=keywords or [],
                tags=tags or [],
                related_content=None,
                metrics=ContentMetrics(),
                size_bytes=ipfs_content.size,
                checksum=self._compute_checksum(content),
                created_at=now,
                updated_at=now
            )
            
            # Store metadata in IPFS
            metadata_cid = await self._store_metadata(addressed_content)
            
            # Register in local cache
            self.content_registry[ipfs_content.cid] = addressed_content
            self.cid_to_metadata[ipfs_content.cid] = metadata_cid
            
            # Update statistics
            self.stats['content_added'] += 1
            self.stats['total_storage_bytes'] += ipfs_content.size
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Added content {ipfs_content.cid}: '{title}' ({processing_time:.2f}s)")
            
            return addressed_content
        
        except Exception as e:
            logger.error(f"❌ Failed to add content: {e}")
            raise
    
    async def get_content(self, cid: str, 
                         include_metadata: bool = True) -> Tuple[Union[str, bytes], Optional[AddressedContent]]:
        """
        Retrieve content by CID
        
        Args:
            cid: Content Identifier
            include_metadata: Whether to fetch and return metadata
            
        Returns:
            Tuple of (content, metadata) or (content, None)
        """
        start_time = time.time()
        
        try:
            # Get content from IPFS
            content_bytes = await self.ipfs_client.get_content(cid)
            
            metadata = None
            if include_metadata:
                # Check local cache first
                if cid in self.content_registry:
                    metadata = self.content_registry[cid]
                    # Update access metrics
                    metadata.metrics.view_count += 1
                    metadata.metrics.last_accessed = datetime.now()
                else:
                    # Try to load metadata from IPFS
                    metadata = await self._load_metadata(cid)
            
            self.stats['content_retrieved'] += 1
            
            processing_time = time.time() - start_time
            logger.debug(f"Retrieved content {cid} ({len(content_bytes)} bytes, {processing_time:.2f}s)")
            
            return content_bytes, metadata
        
        except Exception as e:
            logger.error(f"❌ Failed to get content {cid}: {e}")
            raise
    
    async def create_new_version(self,
                               original_cid: str,
                               new_content: Union[str, bytes],
                               version: str,
                               changes: str,
                               filename: str = None) -> AddressedContent:
        """
        Create a new version of existing content
        
        Args:
            original_cid: CID of the original content
            new_content: Updated content
            version: Version string (e.g., "1.1.0")
            changes: Description of changes
            filename: Optional filename
            
        Returns:
            Updated AddressedContent with new version
        """
        try:
            # Get original metadata
            original_metadata = await self._load_metadata(original_cid)
            if not original_metadata:
                raise ValueError(f"Original content {original_cid} not found")
            
            # Add new content to IPFS
            ipfs_content = await self.ipfs_client.add_content(
                content=new_content,
                filename=filename,
                metadata={'version': version, 'parent': original_cid}
            )
            
            # Create new version entry
            new_version = ContentVersion(
                version=version,
                cid=ipfs_content.cid,
                created_at=datetime.now(),
                changes=changes,
                parent_version=original_metadata.version
            )
            
            # Update metadata
            original_metadata.cid = ipfs_content.cid  # Point to latest version
            original_metadata.version = version
            original_metadata.versions.append(new_version)
            original_metadata.size_bytes = ipfs_content.size
            original_metadata.checksum = self._compute_checksum(new_content)
            original_metadata.updated_at = datetime.now()
            
            # Store updated metadata
            metadata_cid = await self._store_metadata(original_metadata)
            
            # Update registries
            self.content_registry[ipfs_content.cid] = original_metadata
            self.cid_to_metadata[ipfs_content.cid] = metadata_cid
            
            self.stats['versions_created'] += 1
            
            logger.info(f"✅ Created version {version} for content: {ipfs_content.cid}")
            
            return original_metadata
        
        except Exception as e:
            logger.error(f"❌ Failed to create new version: {e}")
            raise
    
    async def search_content(self,
                           query: str = None,
                           category: ContentCategory = None,
                           keywords: List[str] = None,
                           creator_id: str = None,
                           license_type: str = None,
                           content_type: str = None) -> List[AddressedContent]:
        """
        Search for content based on various criteria
        
        Args:
            query: Text search in title/description
            category: Content category filter
            keywords: Keyword filters
            creator_id: Creator filter
            license_type: License filter
            content_type: Content type filter
            
        Returns:
            List of matching AddressedContent objects
        """
        results = []
        
        for cid, content in self.content_registry.items():
            # Apply filters
            if category and content.category != category:
                continue
            
            if creator_id and content.provenance.creator_id != creator_id:
                continue
            
            if license_type and content.license.license_type != license_type:
                continue
            
            if content_type and content.content_type != content_type:
                continue
            
            if keywords:
                content_keywords = set(content.keywords + content.tags)
                if not any(kw.lower() in content_keywords for kw in keywords):
                    continue
            
            if query:
                query_lower = query.lower()
                if (query_lower not in content.title.lower() and 
                    query_lower not in content.description.lower()):
                    continue
            
            results.append(content)
        
        # Sort by relevance (for now, by creation date)
        results.sort(key=lambda x: x.created_at, reverse=True)
        
        return results
    
    async def get_content_history(self, cid: str) -> List[ContentVersion]:
        """Get version history for content"""
        metadata = await self._load_metadata(cid)
        if metadata:
            return metadata.versions
        return []
    
    async def get_related_content(self, cid: str) -> List[AddressedContent]:
        """Get content related to the specified CID"""
        metadata = await self._load_metadata(cid)
        if not metadata or not metadata.related_content:
            return []
        
        related = []
        for related_cid in metadata.related_content:
            related_metadata = await self._load_metadata(related_cid)
            if related_metadata:
                related.append(related_metadata)
        
        return related
    
    async def link_related_content(self, cid1: str, cid2: str):
        """Create bidirectional link between two pieces of content"""
        try:
            # Load both metadata objects
            metadata1 = await self._load_metadata(cid1)
            metadata2 = await self._load_metadata(cid2)
            
            if not metadata1 or not metadata2:
                raise ValueError("One or both content items not found")
            
            # Add bidirectional links
            if metadata1.related_content is None:
                metadata1.related_content = []
            if cid2 not in metadata1.related_content:
                metadata1.related_content.append(cid2)
                await self._store_metadata(metadata1)
            
            if metadata2.related_content is None:
                metadata2.related_content = []
            if cid1 not in metadata2.related_content:
                metadata2.related_content.append(cid1)
                await self._store_metadata(metadata2)
            
            logger.info(f"✅ Linked related content: {cid1} <-> {cid2}")
        
        except Exception as e:
            logger.error(f"❌ Failed to link related content: {e}")
            raise
    
    async def _check_for_duplicate(self, cid: str) -> Optional[AddressedContent]:
        """Check if content with this CID already exists"""
        return self.content_registry.get(cid)
    
    async def _store_metadata(self, content: AddressedContent) -> str:
        """Store content metadata in IPFS"""
        metadata_dict = content.to_dict()
        metadata_json = json.dumps(metadata_dict, indent=2, ensure_ascii=False, default=str)
        
        ipfs_content = await self.ipfs_client.add_content(
            content=metadata_json,
            filename=f"metadata_{content.cid}.json",
            metadata={'type': 'prsm_content_metadata', 'content_cid': content.cid}
        )
        
        return ipfs_content.cid
    
    async def _load_metadata(self, cid: str) -> Optional[AddressedContent]:
        """Load content metadata from IPFS"""
        try:
            # Check if we have metadata CID mapping
            if cid in self.cid_to_metadata:
                metadata_cid = self.cid_to_metadata[cid]
                metadata_bytes = await self.ipfs_client.get_content(metadata_cid)
                metadata_dict = json.loads(metadata_bytes.decode('utf-8'))
                return AddressedContent.from_dict(metadata_dict)
        
        except Exception as e:
            logger.debug(f"Could not load metadata for {cid}: {e}")
        
        return None
    
    def _compute_checksum(self, content: Union[str, bytes]) -> str:
        """Compute SHA256 checksum for content"""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'addressing_stats': self.stats.copy(),
            'content_count': len(self.content_registry),
            'categories': self._get_category_distribution(),
            'total_versions': sum(len(c.versions) for c in self.content_registry.values()),
            'ipfs_stats': self.ipfs_client.get_stats()
        }
    
    def _get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of content by category"""
        distribution = {}
        for content in self.content_registry.values():
            category = content.category.value
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the addressing system"""
        try:
            # Check IPFS client health
            ipfs_health = await self.ipfs_client.health_check()
            
            # Verify a few random content items
            sample_cids = list(self.content_registry.keys())[:3]
            verification_results = []
            
            for cid in sample_cids:
                try:
                    content, metadata = await self.get_content(cid, include_metadata=True)
                    verification_results.append({
                        'cid': cid,
                        'accessible': True,
                        'metadata_found': metadata is not None
                    })
                except Exception as e:
                    verification_results.append({
                        'cid': cid,
                        'accessible': False,
                        'error': str(e)
                    })
            
            return {
                'healthy': ipfs_health['healthy'],
                'ipfs_health': ipfs_health,
                'content_verification': verification_results,
                'stats': self.get_stats()
            }
        
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'stats': self.get_stats()
            }


# Utility functions

def create_addressing_system(ipfs_client: IPFSClient) -> ContentAddressingSystem:
    """Create a new content addressing system"""
    return ContentAddressingSystem(ipfs_client)


def create_basic_provenance(creator_id: str, 
                          creator_name: str,
                          institution: str = None) -> ContentProvenance:
    """Create basic provenance information"""
    return ContentProvenance(
        creator_id=creator_id,
        creator_name=creator_name,
        institution=institution
    )


def create_open_license() -> ContentLicense:
    """Create an open CC-BY-4.0 license"""
    return ContentLicense(
        license_type="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        commercial_use=True,
        modification_allowed=True,
        attribution_required=True
    )