#!/usr/bin/env python3
"""
Public Source Content Porting Pipeline
Automated system for importing, processing, and cryptographically marking public content

This pipeline handles the ingestion of public domain content, research papers,
open datasets, and other publicly available information sources into PRSM's
IPFS-based knowledge corpus with full provenance tracking and verification.

Key Features:
1. Automated content discovery and ingestion
2. License compatibility verification
3. Cryptographic content marking and CID generation
4. Metadata extraction and enrichment
5. Quality assessment and filtering
6. Duplicate detection and deduplication
7. Integration with NWTN knowledge corpus
"""

import asyncio
import json
import hashlib
import aiohttp
import feedparser
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from datetime import datetime, timezone
import re

import structlog
from pydantic import BaseModel, Field

from ..ipfs.content_addressing import (
    ContentAddressingSystem, AddressedContent, ContentCategory, 
    ContentProvenance, ContentLicense, create_basic_provenance, create_open_license
)
from ..ipfs.content_verification import ContentVerificationSystem
from ..nwtn.knowledge_corpus_interface import NWTNKnowledgeCorpusInterface

logger = structlog.get_logger(__name__)


class SourceType(str, Enum):
    """Types of content sources"""
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    GITHUB = "github"
    WIKIPEDIA = "wikipedia"
    GUTENBERG = "gutenberg"
    CREATIVE_COMMONS = "creative_commons"
    GOVERNMENT_DATA = "government_data"
    OPEN_DATASET = "open_dataset"
    RSS_FEED = "rss_feed"
    WEB_SCRAPE = "web_scrape"
    USER_UPLOAD = "user_upload"  # Added for user content


class LicenseCompatibility(str, Enum):
    """License compatibility levels"""
    FULLY_COMPATIBLE = "fully_compatible"
    COMPATIBLE_WITH_ATTRIBUTION = "compatible_with_attribution"
    RESTRICTED_USE = "restricted_use"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass
class ContentSource:
    """Configuration for a public content source"""
    
    source_id: str
    source_type: SourceType
    name: str
    base_url: str
    
    # Ingestion parameters
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 60
    max_items_per_batch: int = 100
    
    # Content filters
    allowed_categories: List[ContentCategory] = field(default_factory=list)
    quality_threshold: float = 0.6
    min_content_length: int = 100
    
    # License requirements
    required_license_compatibility: LicenseCompatibility = LicenseCompatibility.COMPATIBLE_WITH_ATTRIBUTION
    
    # Processing options
    extract_metadata: bool = True
    generate_summary: bool = True
    auto_categorize: bool = True
    
    # Status tracking
    active: bool = True
    last_sync: Optional[datetime] = None
    items_processed: int = 0
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class IngestionCandidate:
    """Content candidate for ingestion"""
    
    candidate_id: str
    source_id: str
    source_url: str
    
    # Content metadata
    title: str
    description: str
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    
    # Content data
    content_text: Optional[str] = None
    content_url: Optional[str] = None
    content_type: str = "text/plain"
    estimated_size: int = 0
    
    # Classification
    suggested_category: Optional[ContentCategory] = None
    keywords: List[str] = field(default_factory=list)
    language: str = "en"
    
    # License information
    license_info: Optional[str] = None
    license_url: Optional[str] = None
    license_compatibility: LicenseCompatibility = LicenseCompatibility.UNKNOWN
    
    # Quality metrics
    quality_score: float = 0.0
    completeness_score: float = 0.0
    
    # Processing status
    processed: bool = False
    ingested: bool = False
    error_message: Optional[str] = None
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class IngestionResult(BaseModel):
    """Result of content ingestion process"""
    
    candidate_id: str
    ingestion_successful: bool
    
    # Generated content
    content_cid: Optional[str] = None
    addressed_content: Optional[AddressedContent] = None
    
    # Processing details
    processing_time_seconds: float = 0.0
    content_size_bytes: int = 0
    generated_metadata_count: int = 0
    
    # Quality metrics
    final_quality_score: float = 0.0
    license_verification_passed: bool = False
    duplicate_detected: bool = False
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PublicSourcePorter:
    """
    Automated pipeline for porting public source content into PRSM
    
    This system handles the complete workflow from content discovery
    to IPFS integration with full provenance tracking.
    """
    
    def __init__(self,
                 content_addressing: ContentAddressingSystem,
                 content_verification: ContentVerificationSystem,
                 corpus_interface: NWTNKnowledgeCorpusInterface):
        
        self.content_addressing = content_addressing
        self.content_verification = content_verification
        self.corpus_interface = corpus_interface
        
        # Content sources
        self.content_sources: Dict[str, ContentSource] = {}
        self.active_sources: List[str] = []
        
        # Ingestion tracking
        self.ingestion_queue: List[IngestionCandidate] = []
        self.processed_candidates: Dict[str, IngestionResult] = {}
        
        # Duplicate detection
        self.content_hashes: Dict[str, str] = {}  # hash -> CID mapping
        
        # Rate limiting
        self.rate_limiters: Dict[str, asyncio.Semaphore] = {}
        
        # Quality filters
        self.quality_filters = {
            'min_word_count': 50,
            'max_word_count': 1000000,
            'prohibited_patterns': [r'(?i)\b(spam|advertisement|click here)\b'],
            'required_patterns': [r'\w+'],  # At least some words
        }
        
        # Statistics
        self.stats = {
            'sources_configured': 0,
            'candidates_discovered': 0,
            'content_ingested': 0,
            'duplicates_detected': 0,
            'license_rejections': 0,
            'quality_rejections': 0,
            'total_bytes_processed': 0
        }
        
        logger.info("Public Source Porter initialized")
    
    async def configure_source(self, source: ContentSource):
        """Configure a new public content source"""
        
        self.content_sources[source.source_id] = source
        
        if source.active:
            self.active_sources.append(source.source_id)
            
            # Set up rate limiter
            self.rate_limiters[source.source_id] = asyncio.Semaphore(source.rate_limit_per_minute)
        
        self.stats['sources_configured'] += 1
        
        logger.info("Content source configured",
                   source_id=source.source_id,
                   source_type=source.source_type.value,
                   active=source.active)
    
    async def discover_content(self, source_id: str, max_items: int = None) -> List[IngestionCandidate]:
        """Discover new content from a specific source"""
        
        if source_id not in self.content_sources:
            raise ValueError(f"Unknown source: {source_id}")
        
        source = self.content_sources[source_id]
        
        logger.info("Starting content discovery",
                   source_id=source_id,
                   source_type=source.source_type.value,
                   max_items=max_items)
        
        # Route to appropriate discovery method
        if source.source_type == SourceType.ARXIV:
            candidates = await self._discover_arxiv_content(source, max_items)
        elif source.source_type == SourceType.GITHUB:
            candidates = await self._discover_github_content(source, max_items)
        elif source.source_type == SourceType.RSS_FEED:
            candidates = await self._discover_rss_content(source, max_items)
        elif source.source_type == SourceType.WIKIPEDIA:
            candidates = await self._discover_wikipedia_content(source, max_items)
        else:
            candidates = await self._discover_generic_content(source, max_items)
        
        # Add to ingestion queue
        self.ingestion_queue.extend(candidates)
        self.stats['candidates_discovered'] += len(candidates)
        
        logger.info("Content discovery completed",
                   source_id=source_id,
                   candidates_found=len(candidates))
        
        return candidates
    
    async def ingest_candidate(self, candidate: IngestionCandidate) -> IngestionResult:
        """Ingest a single content candidate"""
        
        start_time = datetime.now()
        
        logger.info("Starting content ingestion",
                   candidate_id=candidate.candidate_id,
                   title=candidate.title)
        
        try:
            # Step 1: Download content if needed
            if not candidate.content_text and candidate.content_url:
                candidate.content_text = await self._download_content(candidate)
            
            if not candidate.content_text:
                return IngestionResult(
                    candidate_id=candidate.candidate_id,
                    ingestion_successful=False,
                    error_message="No content available for ingestion"
                )
            
            # Step 2: Verify license compatibility
            license_ok = await self._verify_license_compatibility(candidate)
            if not license_ok:
                self.stats['license_rejections'] += 1
                return IngestionResult(
                    candidate_id=candidate.candidate_id,
                    ingestion_successful=False,
                    error_message="License not compatible with PRSM requirements"
                )
            
            # Step 3: Quality assessment
            quality_ok = await self._assess_content_quality(candidate)
            if not quality_ok:
                self.stats['quality_rejections'] += 1
                return IngestionResult(
                    candidate_id=candidate.candidate_id,
                    ingestion_successful=False,
                    error_message="Content quality below threshold"
                )
            
            # Step 4: Duplicate detection
            duplicate_cid = await self._check_for_duplicates(candidate)
            if duplicate_cid:
                self.stats['duplicates_detected'] += 1
                return IngestionResult(
                    candidate_id=candidate.candidate_id,
                    ingestion_successful=False,
                    duplicate_detected=True,
                    content_cid=duplicate_cid,
                    error_message=f"Duplicate content detected: {duplicate_cid}"
                )
            
            # Step 5: Create provenance and license information
            provenance = await self._create_provenance(candidate)
            license_obj = await self._create_license(candidate)
            
            # Step 6: Add to IPFS with addressing system
            addressed_content = await self.content_addressing.add_content(
                content=candidate.content_text,
                title=candidate.title,
                description=candidate.description,
                content_type=candidate.content_type,
                category=candidate.suggested_category or ContentCategory.RESEARCH_PAPER,
                provenance=provenance,
                license=license_obj,
                keywords=candidate.keywords,
                tags=[candidate.source_id, candidate.language]
            )
            
            # Step 7: Update corpus index
            await self.corpus_interface.index_new_content(addressed_content.cid)
            
            # Step 8: Store duplicate detection hash
            content_hash = hashlib.sha256(candidate.content_text.encode()).hexdigest()
            self.content_hashes[content_hash] = addressed_content.cid
            
            # Step 9: Mark as processed
            candidate.processed = True
            candidate.ingested = True
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.stats['content_ingested'] += 1
            self.stats['total_bytes_processed'] += len(candidate.content_text.encode())
            
            result = IngestionResult(
                candidate_id=candidate.candidate_id,
                ingestion_successful=True,
                content_cid=addressed_content.cid,
                addressed_content=addressed_content,
                processing_time_seconds=processing_time,
                content_size_bytes=len(candidate.content_text.encode()),
                final_quality_score=candidate.quality_score,
                license_verification_passed=True
            )
            
            self.processed_candidates[candidate.candidate_id] = result
            
            logger.info("Content ingestion completed",
                       candidate_id=candidate.candidate_id,
                       cid=addressed_content.cid,
                       processing_time=processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Content ingestion failed",
                        candidate_id=candidate.candidate_id,
                        error=str(e))
            
            return IngestionResult(
                candidate_id=candidate.candidate_id,
                ingestion_successful=False,
                error_message=str(e),
                processing_time_seconds=(datetime.now() - start_time).total_seconds()
            )
    
    async def batch_ingest(self, source_id: str = None, max_concurrent: int = 5) -> List[IngestionResult]:
        """Batch ingest all queued candidates"""
        
        # Filter by source if specified
        if source_id:
            candidates = [c for c in self.ingestion_queue if c.source_id == source_id]
        else:
            candidates = self.ingestion_queue
        
        logger.info("Starting batch ingestion",
                   candidates=len(candidates),
                   max_concurrent=max_concurrent)
        
        # Process with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def ingest_with_semaphore(candidate):
            async with semaphore:
                return await self.ingest_candidate(candidate)
        
        tasks = [ingest_with_semaphore(candidate) for candidate in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = IngestionResult(
                    candidate_id=candidates[i].candidate_id,
                    ingestion_successful=False,
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        # Clear processed candidates from queue
        if source_id:
            self.ingestion_queue = [c for c in self.ingestion_queue if c.source_id != source_id]
        else:
            self.ingestion_queue.clear()
        
        successful = sum(1 for r in processed_results if r.ingestion_successful)
        
        logger.info("Batch ingestion completed",
                   total_processed=len(processed_results),
                   successful=successful,
                   failed=len(processed_results) - successful)
        
        return processed_results
    
    # === Content Discovery Methods ===
    
    async def _discover_arxiv_content(self, source: ContentSource, max_items: int = None) -> List[IngestionCandidate]:
        """Discover content from arXiv"""
        
        candidates = []
        
        # arXiv API query
        api_url = "http://export.arxiv.org/api/query"
        search_query = "cat:cs.AI OR cat:stat.ML"  # AI/ML papers
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': min(max_items or 100, source.max_items_per_batch)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse arXiv XML response
                        candidates = await self._parse_arxiv_response(content, source)
                    
        except Exception as e:
            logger.error("arXiv discovery failed", error=str(e))
        
        return candidates
    
    async def _discover_github_content(self, source: ContentSource, max_items: int = None) -> List[IngestionCandidate]:
        """Discover content from GitHub"""
        
        candidates = []
        
        # GitHub API search for AI/ML repositories
        api_url = "https://api.github.com/search/repositories"
        
        params = {
            'q': 'machine learning OR artificial intelligence language:Python',
            'sort': 'updated',
            'per_page': min(max_items or 30, source.max_items_per_batch)
        }
        
        headers = {}
        if source.api_key:
            headers['Authorization'] = f'token {source.api_key}'
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        candidates = await self._parse_github_response(data, source)
                    
        except Exception as e:
            logger.error("GitHub discovery failed", error=str(e))
        
        return candidates
    
    async def _discover_rss_content(self, source: ContentSource, max_items: int = None) -> List[IngestionCandidate]:
        """Discover content from RSS feeds"""
        
        candidates = []
        
        try:
            # Use feedparser to parse RSS
            feed = feedparser.parse(source.base_url)
            
            for entry in feed.entries[:max_items or source.max_items_per_batch]:
                candidate = IngestionCandidate(
                    candidate_id=str(uuid4()),
                    source_id=source.source_id,
                    source_url=entry.link,
                    title=entry.title,
                    description=entry.get('summary', ''),
                    publication_date=self._parse_rss_date(entry.get('published')),
                    content_url=entry.link,
                    suggested_category=ContentCategory.RESEARCH_PAPER,
                    license_compatibility=LicenseCompatibility.UNKNOWN
                )
                candidates.append(candidate)
                
        except Exception as e:
            logger.error("RSS discovery failed", error=str(e))
        
        return candidates
    
    async def _discover_wikipedia_content(self, source: ContentSource, max_items: int = None) -> List[IngestionCandidate]:
        """Discover content from Wikipedia"""
        
        candidates = []
        
        # Wikipedia API for recent changes in AI/CS categories
        api_url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
        
        try:
            async with aiohttp.ClientSession() as session:
                for _ in range(min(max_items or 10, source.max_items_per_batch)):
                    async with session.get(api_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            candidate = IngestionCandidate(
                                candidate_id=str(uuid4()),
                                source_id=source.source_id,
                                source_url=data['content_urls']['desktop']['page'],
                                title=data['title'],
                                description=data.get('extract', ''),
                                content_url=data['content_urls']['desktop']['page'],
                                suggested_category=ContentCategory.RESEARCH_PAPER,
                                license_compatibility=LicenseCompatibility.FULLY_COMPATIBLE,  # Wikipedia is CC-BY-SA
                                language=data.get('lang', 'en')
                            )
                            candidates.append(candidate)
                            
        except Exception as e:
            logger.error("Wikipedia discovery failed", error=str(e))
        
        return candidates
    
    async def _discover_generic_content(self, source: ContentSource, max_items: int = None) -> List[IngestionCandidate]:
        """Generic content discovery for web sources"""
        
        candidates = []
        
        # Basic web scraping approach
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source.base_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Extract basic metadata
                        title = self._extract_title_from_html(content)
                        description = self._extract_description_from_html(content)
                        
                        if title and description:
                            candidate = IngestionCandidate(
                                candidate_id=str(uuid4()),
                                source_id=source.source_id,
                                source_url=source.base_url,
                                title=title,
                                description=description,
                                content_text=content,
                                content_type="text/html",
                                suggested_category=ContentCategory.RESEARCH_PAPER,
                                license_compatibility=LicenseCompatibility.UNKNOWN
                            )
                            candidates.append(candidate)
                            
        except Exception as e:
            logger.error("Generic discovery failed", error=str(e))
        
        return candidates
    
    # === Content Processing Methods ===
    
    async def _download_content(self, candidate: IngestionCandidate) -> str:
        """Download content from URL"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(candidate.content_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        candidate.estimated_size = len(content.encode())
                        return content
                    
        except Exception as e:
            logger.error("Content download failed",
                        url=candidate.content_url,
                        error=str(e))
        
        return ""
    
    async def _verify_license_compatibility(self, candidate: IngestionCandidate) -> bool:
        """Verify license compatibility"""
        
        source = self.content_sources[candidate.source_id]
        
        # Check if license compatibility meets source requirements
        required_level = source.required_license_compatibility
        candidate_level = candidate.license_compatibility
        
        # Define compatibility hierarchy
        compatibility_levels = [
            LicenseCompatibility.INCOMPATIBLE,
            LicenseCompatibility.UNKNOWN,
            LicenseCompatibility.RESTRICTED_USE,
            LicenseCompatibility.COMPATIBLE_WITH_ATTRIBUTION,
            LicenseCompatibility.FULLY_COMPATIBLE
        ]
        
        required_index = compatibility_levels.index(required_level)
        candidate_index = compatibility_levels.index(candidate_level)
        
        return candidate_index >= required_index
    
    async def _assess_content_quality(self, candidate: IngestionCandidate) -> bool:
        """Assess content quality"""
        
        if not candidate.content_text:
            return False
        
        content = candidate.content_text
        quality_factors = []
        
        # Length check
        word_count = len(content.split())
        if word_count < self.quality_filters['min_word_count']:
            return False
        if word_count > self.quality_filters['max_word_count']:
            return False
        
        length_score = min(1.0, word_count / 1000)  # Normalize to reasonable length
        quality_factors.append(length_score * 0.3)
        
        # Pattern checks
        prohibited_found = any(re.search(pattern, content) for pattern in self.quality_filters['prohibited_patterns'])
        if prohibited_found:
            return False
        
        required_found = all(re.search(pattern, content) for pattern in self.quality_filters['required_patterns'])
        if not required_found:
            return False
        
        quality_factors.append(0.4)  # Pattern check passed
        
        # Structure score (presence of paragraphs, sentences)
        paragraph_count = content.count('\n\n')
        sentence_count = content.count('.')
        
        structure_score = min(1.0, (paragraph_count * 0.1 + sentence_count * 0.01))
        quality_factors.append(structure_score * 0.3)
        
        # Calculate overall quality
        quality_score = sum(quality_factors)
        candidate.quality_score = quality_score
        
        source = self.content_sources[candidate.source_id]
        return quality_score >= source.quality_threshold
    
    async def _check_for_duplicates(self, candidate: IngestionCandidate) -> Optional[str]:
        """Check for duplicate content"""

        content_hash = hashlib.sha256(candidate.content_text.encode()).hexdigest()
        return self.content_hashes.get(content_hash)

    # === API Connector Methods ===

    async def _fetch_content(self, source: ContentSource) -> Optional[Dict[str, Any]]:
        """
        Fetch content from a source using the appropriate API connector.

        Dispatcher method that routes to the correct connector based on SourceType.

        Returns:
            Dict with content metadata and data, or None on failure
        """
        dispatcher = {
            SourceType.ARXIV: self._fetch_from_arxiv,
            SourceType.PUBMED: self._fetch_from_pubmed,
            SourceType.GITHUB: self._fetch_from_github,
            SourceType.WIKIPEDIA: self._fetch_from_wikipedia,
            SourceType.RSS_FEED: self._fetch_from_rss,
            SourceType.USER_UPLOAD: self._fetch_from_user_upload,
        }

        handler = dispatcher.get(source.source_type)
        if not handler:
            logger.warning("No connector for source type", source_type=source.source_type.value)
            return None

        try:
            return await handler(source)
        except Exception as e:
            logger.error("Failed to fetch content from source",
                        source_id=source.source_id,
                        source_type=source.source_type.value,
                        error=str(e))
            return None

    async def _fetch_from_arxiv(self, source: ContentSource) -> Optional[Dict[str, Any]]:
        """
        Fetch content from arXiv API.

        Uses arXiv API v2 (https://export.arxiv.org/api/query)

        Returns:
            Dict with papers list and metadata
        """
        api_url = source.api_endpoint or "http://export.arxiv.org/api/query"

        # Build search query - default to AI/ML if not specified
        search_query = source.base_url if source.base_url else "cat:cs.AI OR cat:stat.ML"

        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': min(source.max_items_per_batch, 100),
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        try:
            # Rate limiting: 3 requests/second per arXiv ToS
            await asyncio.sleep(0.34)

            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params) as response:
                    if response.status != 200:
                        logger.error("arXiv API error", status=response.status)
                        return None

                    content = await response.text()

            # Parse Atom XML feed
            import xml.etree.ElementTree as ET

            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            root = ET.fromstring(content)
            papers = []

            for entry in root.findall('atom:entry', namespaces):
                try:
                    # Extract arXiv ID from the URL
                    id_url = entry.find('atom:id', namespaces)
                    arxiv_id = id_url.text.split('/')[-1] if id_url is not None else None

                    # Extract title
                    title_elem = entry.find('atom:title', namespaces)
                    title = title_elem.text.strip() if title_elem is not None else ""

                    # Extract authors
                    authors = []
                    for author in entry.findall('atom:author', namespaces):
                        name = author.find('atom:name', namespaces)
                        if name is not None:
                            authors.append(name.text)

                    # Extract abstract/summary
                    summary_elem = entry.find('atom:summary', namespaces)
                    abstract = summary_elem.text.strip() if summary_elem is not None else ""

                    # Extract categories
                    categories = []
                    for cat in entry.findall('atom:category', namespaces):
                        term = cat.get('term')
                        if term:
                            categories.append(term)

                    # Extract dates
                    published_elem = entry.find('atom:published', namespaces)
                    published = published_elem.text if published_elem is not None else None

                    updated_elem = entry.find('atom:updated', namespaces)
                    updated = updated_elem.text if updated_elem is not None else None

                    # Extract DOI if present
                    doi_elem = entry.find('arxiv:doi', namespaces)
                    doi = doi_elem.text if doi_elem is not None else None

                    # Build PDF URL
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None

                    papers.append({
                        'arxiv_id': arxiv_id,
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'categories': categories,
                        'submitted_date': published,
                        'updated_date': updated,
                        'pdf_url': pdf_url,
                        'doi': doi,
                        'source_url': f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
                        'license': 'arXiv-1.0',  # arXiv default license
                        'license_compatibility': LicenseCompatibility.COMPATIBLE_WITH_ATTRIBUTION
                    })

                except Exception as e:
                    logger.debug("Failed to parse arXiv entry", error=str(e))
                    continue

            logger.info("Fetched arXiv papers", count=len(papers), source_id=source.source_id)

            return {
                'source_type': 'arxiv',
                'source_id': source.source_id,
                'papers': papers,
                'total_results': len(papers),
                'fetched_at': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error("arXiv fetch failed", error=str(e))
            return None

    async def _fetch_from_pubmed(self, source: ContentSource) -> Optional[Dict[str, Any]]:
        """
        Fetch content from PubMed via NCBI E-utilities API.

        Two-step process: esearch (get IDs) → efetch (get records)

        Returns:
            Dict with articles list and metadata
        """
        # NCBI E-utilities base URLs
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        # Build search query
        search_term = source.base_url if source.base_url else "artificial intelligence[mh] OR machine learning[mh]"

        # API key increases rate limit from 3/sec to 10/sec
        api_key = source.api_key or None

        try:
            # Step 1: Search for PMIDs
            search_params = {
                'db': 'pubmed',
                'term': search_term,
                'retmax': min(source.max_items_per_batch, 100),
                'retmode': 'json',
                'sort': 'pub_date'
            }
            if api_key:
                search_params['api_key'] = api_key

            # Rate limiting: 3/sec without key, 10/sec with key
            await asyncio.sleep(0.34 if not api_key else 0.1)

            async with aiohttp.ClientSession() as session:
                async with session.get(esearch_url, params=search_params) as response:
                    if response.status != 200:
                        logger.error("PubMed esearch error", status=response.status)
                        return None

                    search_data = await response.json()

            pmids = search_data.get('esearchresult', {}).get('idlist', [])

            if not pmids:
                logger.info("No PubMed results found", source_id=source.source_id)
                return {'source_type': 'pubmed', 'articles': [], 'total_results': 0}

            # Step 2: Fetch article details
            await asyncio.sleep(0.34 if not api_key else 0.1)

            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }
            if api_key:
                fetch_params['api_key'] = api_key

            async with aiohttp.ClientSession() as session:
                async with session.get(efetch_url, params=fetch_params) as response:
                    if response.status != 200:
                        logger.error("PubMed efetch error", status=response.status)
                        return None

                    xml_content = await response.text()

            # Parse XML response
            import xml.etree.ElementTree as ET

            root = ET.fromstring(xml_content)
            articles = []

            for article in root.findall('.//PubmedArticle'):
                try:
                    medline = article.find('MedlineCitation')
                    pmid_elem = medline.find('PMID') if medline else None
                    pmid = pmid_elem.text if pmid_elem is not None else None

                    # Extract article element
                    article_elem = medline.find('Article') if medline else None

                    # Title
                    title_elem = article_elem.find('ArticleTitle') if article_elem else None
                    title = title_elem.text if title_elem is not None else ""

                    # Authors
                    authors = []
                    author_list = article_elem.find('AuthorList') if article_elem else None
                    if author_list is not None:
                        for author in author_list.findall('Author'):
                            lastname = author.find('LastName')
                            forename = author.find('ForeName')
                            if lastname is not None:
                                name = lastname.text
                                if forename is not None:
                                    name = f"{name} {forename.text}"
                                authors.append(name)

                    # Journal
                    journal_elem = article_elem.find('Journal/Title') if article_elem else None
                    journal = journal_elem.text if journal_elem is not None else ""

                    # Publication date
                    pub_date = None
                    date_elem = article_elem.find('ArticleDate') if article_elem else None
                    if date_elem is not None:
                        year = date_elem.find('Year')
                        month = date_elem.find('Month')
                        day = date_elem.find('Day')
                        if year is not None:
                            pub_date = year.text
                            if month is not None:
                                pub_date = f"{pub_date}-{month.text}"
                            if day is not None:
                                pub_date = f"{pub_date}-{day.text}"

                    # Abstract
                    abstract_elem = article_elem.find('Abstract/AbstractText') if article_elem else None
                    abstract = abstract_elem.text if abstract_elem is not None else ""

                    # MeSH terms
                    mesh_terms = []
                    mesh_list = medline.findall('MeshHeadingList/MeshHeading') if medline else []
                    for mesh in mesh_list:
                        descriptor = mesh.find('DescriptorName')
                        if descriptor is not None:
                            mesh_terms.append(descriptor.text)

                    # DOI
                    doi = None
                    for eloc in article_elem.findall('ELocationID') if article_elem else []:
                        if eloc.get('EIdType') == 'doi':
                            doi = eloc.text
                            break

                    # PMC ID (for open access)
                    pmc_id = None
                    for id_elem in article.findall('.//ArticleId'):
                        if id_elem.get('IdType') == 'pmc':
                            pmc_id = id_elem.text

                    articles.append({
                        'pmid': pmid,
                        'title': title,
                        'authors': authors,
                        'journal': journal,
                        'pub_date': pub_date,
                        'abstract': abstract,
                        'mesh_terms': mesh_terms,
                        'doi': doi,
                        'pmc_id': pmc_id,
                        'is_open_access': pmc_id is not None,
                        'source_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                        'full_text_url': f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/" if pmc_id else None,
                        'license': 'Open Access' if pmc_id else 'Restricted',
                        'license_compatibility': LicenseCompatibility.FULLY_COMPATIBLE if pmc_id else LicenseCompatibility.RESTRICTED_USE
                    })

                except Exception as e:
                    logger.debug("Failed to parse PubMed article", error=str(e))
                    continue

            logger.info("Fetched PubMed articles", count=len(articles), source_id=source.source_id)

            return {
                'source_type': 'pubmed',
                'source_id': source.source_id,
                'articles': articles,
                'total_results': len(articles),
                'fetched_at': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error("PubMed fetch failed", error=str(e))
            return None

    async def _fetch_from_github(self, source: ContentSource) -> Optional[Dict[str, Any]]:
        """
        Fetch content from GitHub REST API.

        Supports searching repositories by topic, language, stars, and license.

        Returns:
            Dict with repositories list and metadata
        """
        api_url = "https://api.github.com/search/repositories"

        # Build search query
        query_parts = []
        if source.base_url:
            query_parts.append(source.base_url)
        else:
            query_parts.append("machine learning OR artificial intelligence")

        # Restrict to open-source licenses
        query_parts.append("license:mit OR license:apache-2.0 OR license:bsd-3-clause OR license:gpl-3.0")

        # Add language filter if specified
        if hasattr(source, 'language') and source.language:
            query_parts.append(f"language:{source.language}")

        params = {
            'q': ' '.join(query_parts),
            'sort': 'stars',
            'order': 'desc',
            'per_page': min(source.max_items_per_batch, 100)
        }

        headers = {
            'Accept': 'application/vnd.github.v3+json'
        }

        # Use API token if available (increases rate limit from 60/hr to 5000/hr)
        if source.api_key:
            headers['Authorization'] = f'token {source.api_key}'

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params, headers=headers) as response:
                    if response.status == 403:
                        logger.warning("GitHub API rate limit exceeded")
                        return None
                    if response.status != 200:
                        logger.error("GitHub API error", status=response.status)
                        return None

                    data = await response.json()

                    # Check rate limit headers
                    remaining = response.headers.get('X-RateLimit-Remaining')
                    if remaining and int(remaining) < 10:
                        logger.warning("GitHub API rate limit low", remaining=remaining)

            repositories = []

            for repo in data.get('items', []):
                try:
                    # Fetch README content for ML repos
                    readme_content = None
                    if repo.get('description') and 'ml' in repo.get('description', '').lower():
                        try:
                            readme_url = f"https://api.github.com/repos/{repo['full_name']}/readme"
                            async with aiohttp.ClientSession() as session:
                                async with session.get(readme_url, headers=headers) as readme_resp:
                                    if readme_resp.status == 200:
                                        readme_data = await readme_resp.json()
                                        import base64
                                        readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
                        except Exception:
                            pass

                    repositories.append({
                        'repo_name': repo['name'],
                        'full_name': repo['full_name'],
                        'description': repo.get('description', ''),
                        'topics': repo.get('topics', []),
                        'language': repo.get('language'),
                        'stars': repo.get('stargazers_count', 0),
                        'forks': repo.get('forks_count', 0),
                        'license': repo.get('license', {}).get('spdx_id') if repo.get('license') else None,
                        'html_url': repo['html_url'],
                        'clone_url': repo.get('clone_url'),
                        'default_branch': repo.get('default_branch', 'main'),
                        'readme_content': readme_content[:10000] if readme_content else None,  # Truncate
                        'created_at': repo.get('created_at'),
                        'updated_at': repo.get('updated_at'),
                        'pushed_at': repo.get('pushed_at'),
                        'license_compatibility': LicenseCompatibility.COMPATIBLE_WITH_ATTRIBUTION
                    })

                except Exception as e:
                    logger.debug("Failed to parse GitHub repo", error=str(e))
                    continue

            logger.info("Fetched GitHub repositories", count=len(repositories), source_id=source.source_id)

            return {
                'source_type': 'github',
                'source_id': source.source_id,
                'repositories': repositories,
                'total_count': data.get('total_count', len(repositories)),
                'fetched_at': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error("GitHub fetch failed", error=str(e))
            return None

    async def _fetch_from_wikipedia(self, source: ContentSource) -> Optional[Dict[str, Any]]:
        """
        Fetch content from Wikipedia REST API.

        Only CC-BY-SA licensed content.

        Returns:
            Dict with pages list and metadata
        """
        base_url = source.base_url or "https://en.wikipedia.org"
        api_url = f"{base_url}/api/rest_v1"

        pages = []

        try:
            # Fetch random pages or specific category
            if source.base_url and source.base_url.startswith('Category:'):
                # Category-based fetching
                category = source.base_url.replace('Category:', '')
                list_url = f"{base_url}/w/api.php"
                params = {
                    'action': 'query',
                    'list': 'categorymembers',
                    'cmtitle': f'Category:{category}',
                    'cmlimit': min(source.max_items_per_batch, 50),
                    'format': 'json'
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(list_url, params=params) as response:
                        if response.status != 200:
                            logger.error("Wikipedia category API error", status=response.status)
                            return None
                        data = await response.json()

                page_titles = [item['title'] for item in data.get('query', {}).get('categorymembers', [])]
            else:
                # Fetch random pages
                random_url = f"{api_url}/page/random/summary"
                page_titles = None  # Will fetch random pages

            async with aiohttp.ClientSession() as session:
                if page_titles:
                    # Fetch specified pages
                    for title in page_titles[:source.max_items_per_batch]:
                        await asyncio.sleep(0.1)  # Rate limiting

                        summary_url = f"{api_url}/page/summary/{title}"
                        async with session.get(summary_url) as response:
                            if response.status == 200:
                                page_data = await response.json()
                                pages.append(self._parse_wikipedia_page(page_data))
                else:
                    # Fetch random pages
                    for _ in range(min(source.max_items_per_batch, 20)):
                        await asyncio.sleep(0.1)  # Rate limiting

                        random_url = f"{api_url}/page/random/summary"
                        async with session.get(random_url) as response:
                            if response.status == 200:
                                page_data = await response.json()
                                pages.append(self._parse_wikipedia_page(page_data))

            logger.info("Fetched Wikipedia pages", count=len(pages), source_id=source.source_id)

            return {
                'source_type': 'wikipedia',
                'source_id': source.source_id,
                'pages': pages,
                'total_results': len(pages),
                'fetched_at': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error("Wikipedia fetch failed", error=str(e))
            return None

    def _parse_wikipedia_page(self, page_data: Dict) -> Dict[str, Any]:
        """Parse Wikipedia page summary data."""
        return {
            'title': page_data.get('title', ''),
            'extract': page_data.get('extract', ''),
            'full_url': page_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
            'api_url': page_data.get('content_urls', {}).get('desktop', {}).get('api', ''),
            'last_modified': page_data.get('timestamp'),
            'language': page_data.get('lang', 'en'),
            'thumbnail': page_data.get('thumbnail', {}).get('source') if page_data.get('thumbnail') else None,
            'coordinates': page_data.get('coordinates'),
            'license': 'CC-BY-SA-4.0',
            'license_compatibility': LicenseCompatibility.FULLY_COMPATIBLE
        }

    async def _fetch_from_rss(self, source: ContentSource) -> Optional[Dict[str, Any]]:
        """
        Fetch content from RSS/Atom feed.

        Returns:
            Dict with feed items and metadata
        """
        try:
            feed = feedparser.parse(source.base_url)

            items = []
            for entry in feed.entries[:source.max_items_per_batch]:
                items.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published'),
                    'updated': entry.get('updated'),
                    'author': entry.get('author', ''),
                    'tags': [tag.term for tag in entry.get('tags', [])] if entry.get('tags') else [],
                    'license_compatibility': LicenseCompatibility.UNKNOWN
                })

            logger.info("Fetched RSS feed items", count=len(items), source_id=source.source_id)

            return {
                'source_type': 'rss_feed',
                'source_id': source.source_id,
                'feed_title': feed.feed.get('title', ''),
                'feed_link': feed.feed.get('link', ''),
                'items': items,
                'total_results': len(items),
                'fetched_at': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error("RSS fetch failed", error=str(e))
            return None

    async def _fetch_from_user_upload(self, source: ContentSource) -> Optional[Dict[str, Any]]:
        """Handle user-uploaded content (placeholder for direct uploads)."""
        logger.warning("User upload source type requires direct content, not fetching")
        return None

    # === Content Storage Methods ===

    async def _store_content(self, content: Dict[str, Any], source: ContentSource) -> Optional[str]:
        """
        Store content to IPFS with provenance tracking.

        Serializes content to JSON, stores via DataSpineProxy/IPFS,
        and creates a ProvenanceRecord.

        Args:
            content: Content dict from _fetch_content()
            source: Source configuration

        Returns:
            CID string or None on failure
        """
        try:
            # Import DataSpineProxy for IPFS storage
            from prsm.compute.spine.data_spine_proxy import PRSMDataSpineProxy

            # Serialize content to JSON
            content_json = json.dumps(content, default=str, ensure_ascii=False)
            content_bytes = content_json.encode('utf-8')

            # Calculate content hash
            content_hash = hashlib.sha256(content_bytes).hexdigest()

            # Create filename based on source type and ID
            filename = f"{source.source_type.value}_{source.source_id}_{content_hash[:8]}.json"

            # Initialize data spine proxy
            data_spine = PRSMDataSpineProxy()
            await data_spine._phase1_initialize_infrastructure()

            # Store to IPFS via the IPFS client
            cid = await data_spine.ipfs_client.add_content(content_bytes)

            if cid:
                # Pin the content for persistence
                await data_spine.ipfs_client.pin_content(cid)

                # Create provenance record
                provenance = await self._create_storage_provenance(
                    cid=cid,
                    content_hash=content_hash,
                    source=source,
                    size_bytes=len(content_bytes)
                )

                logger.info("Content stored to IPFS",
                           cid=cid,
                           source_id=source.source_id,
                           size_bytes=len(content_bytes))

                return cid
            else:
                logger.error("Failed to store content to IPFS", source_id=source.source_id)
                return None

        except Exception as e:
            logger.error("Content storage failed", error=str(e), source_id=source.source_id)
            return None

    async def _create_storage_provenance(
        self,
        cid: str,
        content_hash: str,
        source: ContentSource,
        size_bytes: int
    ) -> Optional[str]:
        """Create a provenance record for stored content."""
        try:
            # Import provenance system
            from prsm.data.provenance.enhanced_provenance_system import (
                EnhancedProvenanceSystem,
                ProvenanceType,
                TrustLevel
            )

            provenance_system = EnhancedProvenanceSystem()

            record = provenance_system.create_provenance_record(
                provenance_type=ProvenanceType.DATA_SOURCE,
                source_entity=f"ingestion:{source.source_id}",
                target_entity=f"ipfs:{cid}",
                operation="ingest_and_store",
                inputs=[],
                outputs=[cid],
                metadata={
                    'source_type': source.source_type.value,
                    'source_name': source.name,
                    'content_hash': content_hash,
                    'size_bytes': size_bytes,
                    'license_compatibility': source.required_license_compatibility.value
                },
                trust_level=TrustLevel.CREDIBLE
            )

            return record.record_id

        except Exception as e:
            logger.warning("Failed to create storage provenance", error=str(e))
            return None
    
    async def _create_provenance(self, candidate: IngestionCandidate) -> ContentProvenance:
        """Create provenance information"""
        
        # Extract creator information
        creator_id = f"public_source_{candidate.source_id}"
        creator_name = candidate.authors[0] if candidate.authors else f"Public Source: {candidate.source_id}"
        
        return create_basic_provenance(
            creator_id=creator_id,
            creator_name=creator_name,
            institution=f"Public Source: {self.content_sources[candidate.source_id].name}"
        )
    
    async def _create_license(self, candidate: IngestionCandidate) -> ContentLicense:
        """Create license information"""
        
        if candidate.license_compatibility == LicenseCompatibility.FULLY_COMPATIBLE:
            return create_open_license()
        else:
            # Create custom license based on candidate info
            return ContentLicense(
                license_type=candidate.license_info or "Custom",
                license_url=candidate.license_url or "",
                commercial_use=candidate.license_compatibility != LicenseCompatibility.RESTRICTED_USE,
                modification_allowed=True,
                attribution_required=True
            )
    
    # === Utility Methods ===
    
    async def _parse_arxiv_response(self, xml_content: str, source: ContentSource) -> List[IngestionCandidate]:
        """Parse arXiv XML response"""
        
        candidates = []
        
        # Simple XML parsing (would use proper XML parser in production)
        entries = xml_content.split('<entry>')
        
        for entry in entries[1:]:  # Skip first empty split
            try:
                # Extract basic fields using regex
                title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                id_match = re.search(r'<id>(.*?)</id>', entry)
                
                if title_match and summary_match and id_match:
                    candidate = IngestionCandidate(
                        candidate_id=str(uuid4()),
                        source_id=source.source_id,
                        source_url=id_match.group(1),
                        title=title_match.group(1).strip(),
                        description=summary_match.group(1).strip(),
                        content_url=id_match.group(1),
                        suggested_category=ContentCategory.PREPRINT,
                        license_compatibility=LicenseCompatibility.COMPATIBLE_WITH_ATTRIBUTION
                    )
                    candidates.append(candidate)
                    
            except Exception as e:
                logger.debug("Failed to parse arXiv entry", error=str(e))
                continue
        
        return candidates
    
    async def _parse_github_response(self, data: Dict[str, Any], source: ContentSource) -> List[IngestionCandidate]:
        """Parse GitHub API response"""
        
        candidates = []
        
        for repo in data.get('items', []):
            try:
                candidate = IngestionCandidate(
                    candidate_id=str(uuid4()),
                    source_id=source.source_id,
                    source_url=repo['html_url'],
                    title=repo['name'],
                    description=repo.get('description', ''),
                    content_url=f"{repo['html_url']}/blob/main/README.md",  # Default to README
                    suggested_category=ContentCategory.CODE_REPOSITORY,
                    license_compatibility=LicenseCompatibility.COMPATIBLE_WITH_ATTRIBUTION,  # Most repos are open source
                    language=repo.get('language', 'unknown')
                )
                candidates.append(candidate)
                
            except Exception as e:
                logger.debug("Failed to parse GitHub entry", error=str(e))
                continue
        
        return candidates
    
    def _parse_rss_date(self, date_str: str) -> Optional[datetime]:
        """Parse RSS date string"""
        
        if not date_str:
            return None
        
        # Try common RSS date formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_title_from_html(self, html: str) -> str:
        """Extract title from HTML"""
        
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        
        # Try h1 tag
        h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.IGNORECASE | re.DOTALL)
        if h1_match:
            return re.sub(r'<[^>]+>', '', h1_match.group(1)).strip()
        
        return "Untitled"
    
    def _extract_description_from_html(self, html: str) -> str:
        """Extract description from HTML"""
        
        # Try meta description
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', html, re.IGNORECASE)
        if desc_match:
            return desc_match.group(1).strip()
        
        # Try first paragraph
        p_match = re.search(r'<p[^>]*>(.*?)</p>', html, re.IGNORECASE | re.DOTALL)
        if p_match:
            # Remove HTML tags and truncate
            text = re.sub(r'<[^>]+>', '', p_match.group(1)).strip()
            return text[:500] + '...' if len(text) > 500 else text
        
        return "No description available"
    
    def get_porter_stats(self) -> Dict[str, Any]:
        """Get porter statistics"""
        
        return {
            'porter_stats': self.stats.copy(),
            'configured_sources': len(self.content_sources),
            'active_sources': len(self.active_sources),
            'queued_candidates': len(self.ingestion_queue),
            'processed_candidates': len(self.processed_candidates),
            'content_hashes': len(self.content_hashes),
            'source_details': {
                source_id: {
                    'type': source.source_type.value,
                    'active': source.active,
                    'last_sync': source.last_sync.isoformat() if source.last_sync else None,
                    'items_processed': source.items_processed
                }
                for source_id, source in self.content_sources.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on porter system"""
        
        try:
            # Check underlying systems
            addressing_health = await self.content_addressing.health_check()
            verification_health = await self.content_verification.health_check()
            corpus_health = await self.corpus_interface.health_check()
            
            # Test ingestion pipeline
            test_candidate = IngestionCandidate(
                candidate_id="health_check",
                source_id="test",
                source_url="https://test.example.com",
                title="Test Content",
                description="Health check test content",
                content_text="This is a test content for health check validation.",
                license_compatibility=LicenseCompatibility.FULLY_COMPATIBLE
            )
            
            try:
                # Test quality assessment only (don't actually ingest)
                quality_ok = await self._assess_content_quality(test_candidate)
                pipeline_functional = True
            except Exception:
                pipeline_functional = False
            
            return {
                'healthy': (addressing_health['healthy'] and 
                           verification_health['healthy'] and 
                           corpus_health['healthy'] and 
                           pipeline_functional),
                'addressing_health': addressing_health,
                'verification_health': verification_health,
                'corpus_health': corpus_health,
                'pipeline_functional': pipeline_functional,
                'stats': self.get_porter_stats()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'stats': self.get_porter_stats()
            }


# Utility functions

def create_porter_system(content_addressing: ContentAddressingSystem,
                        content_verification: ContentVerificationSystem,
                        corpus_interface: NWTNKnowledgeCorpusInterface) -> PublicSourcePorter:
    """Create a new public source porter"""
    return PublicSourcePorter(content_addressing, content_verification, corpus_interface)


async def configure_default_sources(porter: PublicSourcePorter):
    """Configure default public content sources"""
    
    # arXiv configuration
    arxiv_source = ContentSource(
        source_id="arxiv_ai_ml",
        source_type=SourceType.ARXIV,
        name="arXiv AI/ML Papers",
        base_url="https://arxiv.org",
        api_endpoint="http://export.arxiv.org/api/query",
        rate_limit_per_minute=20,  # arXiv rate limit
        allowed_categories=[ContentCategory.PREPRINT, ContentCategory.RESEARCH_PAPER],
        quality_threshold=0.7,
        required_license_compatibility=LicenseCompatibility.COMPATIBLE_WITH_ATTRIBUTION
    )
    
    # Wikipedia configuration
    wikipedia_source = ContentSource(
        source_id="wikipedia_en",
        source_type=SourceType.WIKIPEDIA,
        name="Wikipedia (English)",
        base_url="https://en.wikipedia.org",
        rate_limit_per_minute=60,
        allowed_categories=[ContentCategory.RESEARCH_PAPER],
        quality_threshold=0.6,
        required_license_compatibility=LicenseCompatibility.FULLY_COMPATIBLE
    )
    
    # Configure sources
    await porter.configure_source(arxiv_source)
    await porter.configure_source(wikipedia_source)