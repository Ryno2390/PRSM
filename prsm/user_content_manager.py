#!/usr/bin/env python3
"""
PRSM User Content Management System
User-friendly interface for content uploads leveraging the existing pipeline

This module provides a streamlined interface for users to upload content
while utilizing the same robust verification, CID marking, and NWTN
integration pipeline used for public sources.

Key Features:
1. User-friendly upload interface
2. Automatic metadata extraction and enhancement
3. Real-time quality feedback
4. Batch upload support
5. User dashboard and content management
6. Integration with existing FTNS tokenomics
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from datetime import datetime, timezone
import mimetypes
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from .knowledge_system import UnifiedKnowledgeSystem
from .ipfs.content_addressing import ContentCategory, ContentProvenance, ContentLicense
from .ingestion.public_source_porter import IngestionCandidate, IngestionResult, LicenseCompatibility
from .tokenomics.ftns_economics import FTNSEconomicsEngine

logger = structlog.get_logger(__name__)


class UserContentType(str, Enum):
    """Types of user content uploads"""
    RESEARCH_PAPER = "research_paper"
    DATASET = "dataset"
    CODE_REPOSITORY = "code_repository"
    DOCUMENTATION = "documentation"
    EXPERIMENTAL_RESULTS = "experimental_results"
    PROTOCOL = "protocol"
    REVIEW = "review"
    PRESENTATION = "presentation"
    NOTEBOOK = "notebook"
    OTHER = "other"


class UploadStatus(str, Enum):
    """Status of user uploads"""
    PENDING = "pending"
    PROCESSING = "processing"
    QUALITY_CHECK = "quality_check"
    VERIFICATION = "verification"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class UserProfile:
    """User profile for content management"""
    
    user_id: str
    username: str
    email: str
    
    # Credentials and verification
    institution: Optional[str] = None
    orcid_id: Optional[str] = None
    verified_researcher: bool = False
    
    # Content statistics
    uploads_count: int = 0
    total_size_bytes: int = 0
    quality_score_average: float = 0.0
    
    # FTNS economics
    ftns_balance: float = 0.0
    earnings_total: float = 0.0
    royalties_earned: float = 0.0
    
    # Reputation and trust
    trust_score: float = 0.5  # 0-1 scale
    peer_review_score: float = 0.0
    community_rating: float = 0.0
    
    # Settings
    auto_publish: bool = True
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UserUpload:
    """User content upload"""
    
    upload_id: str
    user_id: str
    
    # Content metadata
    title: str
    description: str
    content_type: UserContentType
    file_name: Optional[str] = None
    file_size: int = 0
    mime_type: str = "text/plain"
    
    # Content data
    content_text: Optional[str] = None
    content_binary: Optional[bytes] = None
    content_url: Optional[str] = None
    
    # Classification and metadata
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    language: str = "en"
    
    # Licensing and rights
    license_type: str = "CC-BY-4.0"  # Default to open
    copyright_holder: Optional[str] = None
    attribution_required: bool = True
    commercial_use_allowed: bool = True
    
    # Quality and review
    quality_score: float = 0.0
    peer_reviewed: bool = False
    review_comments: List[str] = field(default_factory=list)
    
    # Processing status
    status: UploadStatus = UploadStatus.PENDING
    processing_messages: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    
    # Results after processing
    content_cid: Optional[str] = None
    ipfs_url: Optional[str] = None
    
    # FTNS economics
    royalty_rate: float = 0.08  # 8% default
    access_price: float = 0.0   # Free by default
    earnings_to_date: float = 0.0
    
    # Timestamps
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None


class UserUploadResult(BaseModel):
    """Result of user upload processing"""
    
    upload_id: str
    success: bool
    
    # Processing results
    content_cid: Optional[str] = None
    ipfs_url: Optional[str] = None
    quality_score: float = 0.0
    
    # Economic results
    ftns_earned: float = 0.0
    estimated_annual_royalties: float = 0.0
    
    # Integration results
    indexed_in_corpus: bool = False
    available_to_nwtn: bool = False
    
    # Feedback
    quality_feedback: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    
    # Processing details
    processing_time_seconds: float = 0.0
    verification_passed: bool = False
    
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserContentManager:
    """
    User Content Management System leveraging existing PRSM pipeline
    
    This system provides a user-friendly interface for content uploads
    while utilizing the robust verification and integration infrastructure.
    """
    
    def __init__(self, 
                 knowledge_system: UnifiedKnowledgeSystem,
                 ftns_engine: Optional[Any] = None):  # FTNSEconomicsEngine
        
        self.knowledge_system = knowledge_system
        self.ftns_engine = ftns_engine
        
        # User management
        self.user_profiles: Dict[str, UserProfile] = {}
        self.user_uploads: Dict[str, UserUpload] = {}
        self.upload_queue: List[str] = []  # Upload IDs in processing queue
        
        # Content type mappings
        self.content_type_mapping = {
            UserContentType.RESEARCH_PAPER: ContentCategory.RESEARCH_PAPER,
            UserContentType.DATASET: ContentCategory.DATASET,
            UserContentType.CODE_REPOSITORY: ContentCategory.CODE_REPOSITORY,
            UserContentType.DOCUMENTATION: ContentCategory.SUPPLEMENT,
            UserContentType.EXPERIMENTAL_RESULTS: ContentCategory.DATASET,
            UserContentType.PROTOCOL: ContentCategory.PROTOCOL,
            UserContentType.REVIEW: ContentCategory.REVIEW,
            UserContentType.PRESENTATION: ContentCategory.PRESENTATION,
            UserContentType.NOTEBOOK: ContentCategory.RESEARCH_PAPER,
            UserContentType.OTHER: ContentCategory.RESEARCH_PAPER
        }
        
        # Processing configuration
        self.quality_thresholds = {
            UserContentType.RESEARCH_PAPER: 0.7,
            UserContentType.DATASET: 0.6,
            UserContentType.CODE_REPOSITORY: 0.6,
            UserContentType.DOCUMENTATION: 0.5,
            UserContentType.OTHER: 0.5
        }
        
        # Statistics
        self.stats = {
            'total_users': 0,
            'total_uploads': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'total_content_size': 0,
            'ftns_distributed': 0.0,
            'royalties_paid': 0.0
        }
        
        logger.info("User Content Manager initialized")
    
    async def register_user(self, 
                          user_id: str,
                          username: str, 
                          email: str,
                          institution: str = None,
                          orcid_id: str = None) -> UserProfile:
        """Register a new user"""
        
        if user_id in self.user_profiles:
            raise ValueError(f"User {user_id} already registered")
        
        profile = UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            institution=institution,
            orcid_id=orcid_id,
            verified_researcher=bool(orcid_id and institution)  # Simple verification
        )
        
        self.user_profiles[user_id] = profile
        self.stats['total_users'] += 1
        
        logger.info("User registered",
                   user_id=user_id,
                   username=username,
                   verified=profile.verified_researcher)
        
        return profile
    
    async def upload_content(self,
                           user_id: str,
                           title: str,
                           description: str,
                           content_type: UserContentType,
                           content: Union[str, bytes] = None,
                           file_path: str = None,
                           file_obj: BinaryIO = None,
                           metadata: Dict[str, Any] = None) -> UserUpload:
        """
        Upload content from user
        
        Args:
            user_id: User identifier
            title: Content title
            description: Content description
            content_type: Type of content being uploaded
            content: Direct content string/bytes
            file_path: Path to file to upload
            file_obj: File object to upload
            metadata: Additional metadata
            
        Returns:
            UserUpload object tracking the upload
        """
        
        if user_id not in self.user_profiles:
            raise ValueError(f"User {user_id} not registered")
        
        upload_id = str(uuid4())
        metadata = metadata or {}
        
        logger.info("Starting user content upload",
                   user_id=user_id,
                   upload_id=upload_id,
                   title=title[:50],
                   content_type=content_type.value)
        
        try:
            # Create upload record
            upload = UserUpload(
                upload_id=upload_id,
                user_id=user_id,
                title=title,
                description=description,
                content_type=content_type,
                domain=metadata.get('domain'),
                keywords=metadata.get('keywords', []),
                tags=metadata.get('tags', []),
                language=metadata.get('language', 'en'),
                license_type=metadata.get('license_type', 'CC-BY-4.0'),
                royalty_rate=metadata.get('royalty_rate', 0.08)
            )
            
            # Handle different content input methods
            if content:
                if isinstance(content, str):
                    upload.content_text = content
                    upload.file_size = len(content.encode())
                else:
                    upload.content_binary = content
                    upload.file_size = len(content)
                    
            elif file_path:
                file_path_obj = Path(file_path)
                upload.file_name = file_path_obj.name
                upload.mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
                
                # Read file content
                if upload.mime_type.startswith('text/'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        upload.content_text = f.read()
                else:
                    with open(file_path, 'rb') as f:
                        upload.content_binary = f.read()
                
                upload.file_size = file_path_obj.stat().st_size
                
            elif file_obj:
                upload.file_name = getattr(file_obj, 'name', 'uploaded_file')
                content_data = file_obj.read()
                
                if isinstance(content_data, str):
                    upload.content_text = content_data
                else:
                    upload.content_binary = content_data
                
                upload.file_size = len(content_data)
                upload.mime_type = mimetypes.guess_type(upload.file_name)[0] or "application/octet-stream"
            
            else:
                raise ValueError("Must provide content, file_path, or file_obj")
            
            # Store upload and add to queue
            self.user_uploads[upload_id] = upload
            self.upload_queue.append(upload_id)
            
            # Update user statistics
            user_profile = self.user_profiles[user_id]
            user_profile.uploads_count += 1
            user_profile.total_size_bytes += upload.file_size
            user_profile.last_active = datetime.now(timezone.utc)
            
            # Update system statistics
            self.stats['total_uploads'] += 1
            self.stats['total_content_size'] += upload.file_size
            
            logger.info("User upload queued",
                       upload_id=upload_id,
                       file_size=upload.file_size,
                       content_type=content_type.value)
            
            # Start processing asynchronously
            asyncio.create_task(self._process_upload(upload_id))
            
            return upload
            
        except Exception as e:
            logger.error("User upload failed",
                        user_id=user_id,
                        title=title[:50],
                        error=str(e))
            raise
    
    async def _process_upload(self, upload_id: str) -> UserUploadResult:
        """Process a user upload through the existing pipeline"""
        
        upload = self.user_uploads[upload_id]
        user_profile = self.user_profiles[upload.user_id]
        
        start_time = datetime.now()
        upload.status = UploadStatus.PROCESSING
        
        logger.info("Processing user upload",
                   upload_id=upload_id,
                   user_id=upload.user_id,
                   title=upload.title[:50])
        
        try:
            # Step 1: Quality assessment
            upload.status = UploadStatus.QUALITY_CHECK
            quality_ok = await self._assess_upload_quality(upload)
            
            if not quality_ok:
                upload.status = UploadStatus.REJECTED
                upload.error_messages.append("Content quality below threshold")
                
                return UserUploadResult(
                    upload_id=upload_id,
                    success=False,
                    error_message="Content quality insufficient",
                    quality_score=upload.quality_score,
                    quality_feedback=upload.processing_messages
                )
            
            # Step 2: Create content for knowledge system
            content_text = upload.content_text
            if upload.content_binary and not content_text:
                # Try to extract text from binary content
                content_text = await self._extract_text_from_binary(upload)
            
            if not content_text:
                raise ValueError("Could not extract text content for processing")
            
            # Step 3: Use knowledge system for ingestion
            upload.status = UploadStatus.VERIFICATION
            
            content_category = self.content_type_mapping[upload.content_type]
            
            ingestion_result = await self.knowledge_system.ingest_content(
                content=content_text,
                title=upload.title,
                description=upload.description,
                category=content_category,
                source_info={
                    'creator_id': upload.user_id,
                    'creator_name': user_profile.username,
                    'institution': user_profile.institution,
                    'keywords': upload.keywords,
                    'tags': upload.tags + ['user_upload'],
                    'license_type': upload.license_type,
                    'royalty_rate': upload.royalty_rate
                }
            )
            
            if not ingestion_result['success']:
                raise RuntimeError("Knowledge system ingestion failed")
            
            # Step 4: Update upload with results
            upload.status = UploadStatus.INDEXING
            upload.content_cid = ingestion_result['cid']
            upload.ipfs_url = f"ipfs://{upload.content_cid}"
            upload.processed_at = datetime.now(timezone.utc)
            
            # Step 5: Handle FTNS economics if available
            ftns_earned = 0.0
            estimated_royalties = 0.0
            
            if self.ftns_engine:
                # Award FTNS for quality content
                base_reward = min(100.0, upload.file_size / 1000)  # Size-based reward
                quality_multiplier = upload.quality_score
                ftns_earned = base_reward * quality_multiplier
                
                # Estimate annual royalties
                estimated_royalties = ftns_earned * upload.royalty_rate * 12  # Rough estimate
                
                # Update user balance
                user_profile.ftns_balance += ftns_earned
                user_profile.earnings_total += ftns_earned
                upload.earnings_to_date = ftns_earned
                
                self.stats['ftns_distributed'] += ftns_earned
            
            # Step 6: Complete processing
            upload.status = UploadStatus.COMPLETED
            
            # Update user profile
            user_profile.quality_score_average = (
                (user_profile.quality_score_average * (user_profile.uploads_count - 1) + upload.quality_score) 
                / user_profile.uploads_count
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = UserUploadResult(
                upload_id=upload_id,
                success=True,
                content_cid=upload.content_cid,
                ipfs_url=upload.ipfs_url,
                quality_score=upload.quality_score,
                ftns_earned=ftns_earned,
                estimated_annual_royalties=estimated_royalties,
                indexed_in_corpus=ingestion_result.get('indexed_in_corpus', False),
                available_to_nwtn=True,
                quality_feedback=upload.processing_messages,
                processing_time_seconds=processing_time,
                verification_passed=True
            )
            
            # Update statistics
            self.stats['successful_uploads'] += 1
            
            logger.info("User upload processed successfully",
                       upload_id=upload_id,
                       cid=upload.content_cid,
                       quality_score=upload.quality_score,
                       ftns_earned=ftns_earned)
            
            return result
            
        except Exception as e:
            upload.status = UploadStatus.FAILED
            upload.error_messages.append(str(e))
            
            self.stats['failed_uploads'] += 1
            
            logger.error("User upload processing failed",
                        upload_id=upload_id,
                        error=str(e))
            
            return UserUploadResult(
                upload_id=upload_id,
                success=False,
                error_message=str(e),
                processing_time_seconds=(datetime.now() - start_time).total_seconds()
            )
        
        finally:
            # Remove from processing queue
            if upload_id in self.upload_queue:
                self.upload_queue.remove(upload_id)
    
    async def _assess_upload_quality(self, upload: UserUpload) -> bool:
        """Assess quality of user upload"""
        
        content = upload.content_text or ""
        quality_factors = []
        
        # Content length check
        word_count = len(content.split()) if content else 0
        if word_count < 50:
            upload.processing_messages.append("Content too short (minimum 50 words)")
            return False
        
        length_score = min(1.0, word_count / 1000)
        quality_factors.append(length_score * 0.4)
        upload.processing_messages.append(f"Content length: {word_count} words")
        
        # Title and description quality
        title_score = min(1.0, len(upload.title.split()) / 10)  # Prefer descriptive titles
        desc_score = min(1.0, len(upload.description.split()) / 20)
        
        quality_factors.append(title_score * 0.2)
        quality_factors.append(desc_score * 0.2)
        
        # User trust factor
        user_profile = self.user_profiles[upload.user_id]
        trust_factor = user_profile.trust_score
        quality_factors.append(trust_factor * 0.2)
        
        # Calculate overall quality
        upload.quality_score = sum(quality_factors)
        
        # Check against threshold for this content type
        threshold = self.quality_thresholds.get(upload.content_type, 0.6)
        
        if upload.quality_score >= threshold:
            upload.processing_messages.append(f"Quality score: {upload.quality_score:.2f} (passed)")
            return True
        else:
            upload.processing_messages.append(f"Quality score: {upload.quality_score:.2f} (below threshold {threshold})")
            return False
    
    async def _extract_text_from_binary(self, upload: UserUpload) -> Optional[str]:
        """Extract text from binary content"""
        
        # This would implement text extraction for various file types
        # For now, return None for binary content
        upload.processing_messages.append("Binary content detected - text extraction not implemented")
        return None
    
    async def get_user_uploads(self, user_id: str) -> List[UserUpload]:
        """Get all uploads for a user"""
        
        return [upload for upload in self.user_uploads.values() if upload.user_id == user_id]
    
    async def get_upload_status(self, upload_id: str) -> Optional[UserUpload]:
        """Get status of a specific upload"""
        
        return self.user_uploads.get(upload_id)
    
    async def get_user_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user dashboard data"""
        
        if user_id not in self.user_profiles:
            raise ValueError(f"User {user_id} not found")
        
        user_profile = self.user_profiles[user_id]
        user_uploads = await self.get_user_uploads(user_id)
        
        # Calculate statistics
        successful_uploads = [u for u in user_uploads if u.status == UploadStatus.COMPLETED]
        failed_uploads = [u for u in user_uploads if u.status == UploadStatus.FAILED]
        pending_uploads = [u for u in user_uploads if u.status in [UploadStatus.PENDING, UploadStatus.PROCESSING]]
        
        return {
            'user_profile': user_profile,
            'upload_statistics': {
                'total_uploads': len(user_uploads),
                'successful_uploads': len(successful_uploads),
                'failed_uploads': len(failed_uploads),
                'pending_uploads': len(pending_uploads),
                'total_size_mb': user_profile.total_size_bytes / (1024 * 1024),
                'average_quality_score': user_profile.quality_score_average
            },
            'economic_summary': {
                'ftns_balance': user_profile.ftns_balance,
                'total_earnings': user_profile.earnings_total,
                'total_royalties': user_profile.royalties_earned,
                'content_value': sum(u.earnings_to_date for u in successful_uploads)
            },
            'recent_uploads': sorted(user_uploads, key=lambda x: x.uploaded_at, reverse=True)[:10],
            'content_breakdown': {
                content_type.value: len([u for u in user_uploads if u.content_type == content_type])
                for content_type in UserContentType
            }
        }
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get content manager statistics"""
        
        return {
            'manager_stats': self.stats.copy(),
            'user_count': len(self.user_profiles),
            'total_uploads': len(self.user_uploads),
            'queue_length': len(self.upload_queue),
            'upload_status_distribution': {
                status.value: len([u for u in self.user_uploads.values() if u.status == status])
                for status in UploadStatus
            },
            'content_type_distribution': {
                content_type.value: len([u for u in self.user_uploads.values() if u.content_type == content_type])
                for content_type in UserContentType
            }
        }


# Utility functions

def create_user_content_manager(knowledge_system: UnifiedKnowledgeSystem,
                               ftns_engine: Any = None) -> UserContentManager:
    """Create a new user content manager"""
    return UserContentManager(knowledge_system, ftns_engine)


async def quick_user_upload(manager: UserContentManager,
                          user_id: str,
                          title: str,
                          content: str,
                          content_type: UserContentType = UserContentType.OTHER) -> UserUploadResult:
    """Quick utility for user uploads"""
    
    upload = await manager.upload_content(
        user_id=user_id,
        title=title,
        description=f"User upload: {title}",
        content_type=content_type,
        content=content
    )
    
    # Wait for processing to complete
    while upload.status not in [UploadStatus.COMPLETED, UploadStatus.FAILED, UploadStatus.REJECTED]:
        await asyncio.sleep(0.1)
        upload = await manager.get_upload_status(upload.upload_id)
    
    # Get final result
    return upload