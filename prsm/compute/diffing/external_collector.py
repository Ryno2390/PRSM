"""
External Data Collection System
===============================

Handles anonymous collection of external data sources for knowledge diffing
operations. Integrates with privacy infrastructure to ensure data collection
cannot be traced back to PRSM or reveal strategic interests.

Key Features:
- Multi-source data collection (ArXiv, GitHub, journals, etc.)
- Anonymous web crawling via Tor/I2P
- Intelligent source discovery and prioritization
- Rate limiting and respectful crawling
- Content extraction and preprocessing
- Quality assessment and filtering
"""

import asyncio
import aiohttp
import hashlib
import secrets
import json
import re
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass
from decimal import Decimal
from urllib.parse import urlparse, urljoin

import feedparser
import requests
from bs4 import BeautifulSoup

from pydantic import BaseModel, Field

# Import PRSM privacy infrastructure
from ..privacy.anonymous_networking import anonymous_network_manager, PrivacyLevel


class SourceType(str, Enum):
    """Types of external data sources"""
    ARXIV = "arxiv"
    GITHUB = "github"
    PUBMED = "pubmed"
    RSS_FEED = "rss_feed"
    WEB_PAGE = "web_page"
    API_ENDPOINT = "api_endpoint"
    DOCUMENTATION = "documentation"
    FORUM = "forum"
    NEWS = "news"
    PATENT = "patent"


class ContentType(str, Enum):
    """Types of content extracted from sources"""
    RESEARCH_PAPER = "research_paper"
    CODE_REPOSITORY = "code_repository"
    DOCUMENTATION = "documentation"
    FORUM_DISCUSSION = "forum_discussion"
    NEWS_ARTICLE = "news_article"
    PATENT_FILING = "patent_filing"
    TECHNICAL_BLOG = "technical_blog"
    ACADEMIC_PRESENTATION = "academic_presentation"


@dataclass
class SourceConfiguration:
    """Configuration for external data source"""
    source_id: UUID
    source_type: SourceType
    base_url: str
    
    # Collection parameters
    max_requests_per_hour: int = 60
    request_delay_seconds: float = 1.0
    max_content_size_mb: int = 10
    
    # Content filtering
    allowed_content_types: List[ContentType] = None
    keyword_filters: List[str] = None
    quality_threshold: float = 0.5
    
    # Privacy settings
    privacy_level: PrivacyLevel = PrivacyLevel.ENHANCED
    use_random_delays: bool = True
    rotate_user_agents: bool = True


class ExtractedContent(BaseModel):
    """Content extracted from external source"""
    content_id: UUID = Field(default_factory=uuid4)
    source_url: str
    source_type: SourceType
    content_type: ContentType
    
    # Content data
    title: Optional[str] = None
    abstract: Optional[str] = None
    full_text: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    # Metadata
    publication_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    language: str = "en"
    
    # Quality metrics
    content_quality_score: float = 0.0
    relevance_score: float = 0.0
    novelty_score: float = 0.0
    
    # Processing metadata
    extraction_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str
    original_size_bytes: int
    processed_size_bytes: int


class CollectionSession(BaseModel):
    """Data collection session tracking"""
    session_id: UUID = Field(default_factory=uuid4)
    privacy_session_id: UUID  # Reference to anonymous network session
    
    # Collection scope
    target_sources: List[UUID] = Field(default_factory=list)
    domains_of_interest: List[str] = Field(default_factory=list)
    
    # Progress tracking
    total_urls_planned: int = 0
    urls_processed: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    
    # Content statistics
    total_content_collected: int = 0
    total_bytes_collected: int = 0
    unique_sources_accessed: int = 0
    
    # Quality metrics
    average_quality_score: float = 0.0
    high_quality_content_count: int = 0
    
    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None


class ExternalDataCollector:
    """
    Comprehensive external data collection system that anonymously gathers
    content from diverse sources while respecting rate limits and maintaining
    privacy through the PRSM anonymity infrastructure.
    """
    
    def __init__(self):
        # Privacy infrastructure integration
        self.network_manager = anonymous_network_manager
        
        # Source configurations
        self.configured_sources: Dict[UUID, SourceConfiguration] = {}
        self.active_sessions: Dict[UUID, CollectionSession] = {}
        self.extracted_content: Dict[UUID, ExtractedContent] = {}
        
        # Built-in source configurations
        self._initialize_default_sources()
        
        # Rate limiting
        self.request_timestamps: Dict[str, List[datetime]] = {}
        
        # User agent rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        
        # Performance tracking
        self.total_requests_made = 0
        self.total_content_extracted = 0
        self.total_bytes_collected = 0
        
        print("🌐 External Data Collector initialized")
        print("   - Anonymous collection via privacy infrastructure")
        print("   - Multi-source support with rate limiting")
        print("   - Content quality assessment enabled")
    
    async def start_collection_session(self,
                                     source_types: List[SourceType],
                                     domains_of_interest: List[str],
                                     privacy_level: PrivacyLevel = PrivacyLevel.ENHANCED) -> CollectionSession:
        """
        Start a new data collection session with specified parameters.
        """
        
        # Create anonymous privacy session
        privacy_session = await self.network_manager.create_private_session(
            privacy_level=privacy_level,
            user_anonymous_id="external_collector",
            duration_hours=24
        )
        
        # Select configured sources matching criteria
        target_sources = []
        for source_id, config in self.configured_sources.items():
            if config.source_type in source_types:
                target_sources.append(source_id)
        
        # Estimate collection scope
        total_urls = await self._estimate_collection_scope(target_sources, domains_of_interest)
        
        session = CollectionSession(
            privacy_session_id=privacy_session.session_id,
            target_sources=target_sources,
            domains_of_interest=domains_of_interest,
            total_urls_planned=total_urls,
            estimated_completion=datetime.now(timezone.utc) + timedelta(hours=max(2, total_urls // 100))
        )
        
        self.active_sessions[session.session_id] = session
        
        print(f"🚀 Collection session started")
        print(f"   - Session ID: {session.session_id}")
        print(f"   - Privacy level: {privacy_level}")
        print(f"   - Target sources: {len(target_sources)}")
        print(f"   - Estimated URLs: {total_urls}")
        
        return session
    
    async def collect_arxiv_papers(self,
                                 session_id: UUID,
                                 search_queries: List[str],
                                 max_papers: int = 100) -> List[ExtractedContent]:
        """
        Collect recent papers from ArXiv based on search queries.
        """
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Collection session {session_id} not found")
        
        session = self.active_sessions[session_id]
        collected_content = []
        
        for query in search_queries:
            try:
                # Construct ArXiv API URL
                base_url = "http://export.arxiv.org/api/query"
                params = {
                    "search_query": query,
                    "start": 0,
                    "max_results": min(max_papers, 100),  # ArXiv limit
                    "sortBy": "lastUpdatedDate",
                    "sortOrder": "descending"
                }
                
                url = f"{base_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
                
                # Make anonymous request
                response = await self.network_manager.send_anonymous_request(
                    session_id=session.privacy_session_id,
                    url=url,
                    method="GET"
                )
                
                # Parse ArXiv response
                content_items = await self._parse_arxiv_response(response, query)
                collected_content.extend(content_items)
                
                session.urls_processed += 1
                session.successful_extractions += len(content_items)
                
                # Respectful delay
                await asyncio.sleep(2.0)
                
            except Exception as e:
                session.failed_extractions += 1
                print(f"⚠️ ArXiv collection failed for query '{query}': {e}")
        
        # Update session statistics
        session.total_content_collected += len(collected_content)
        session.total_bytes_collected += sum(c.processed_size_bytes for c in collected_content)
        
        print(f"📚 ArXiv collection completed")
        print(f"   - Papers collected: {len(collected_content)}")
        print(f"   - Queries processed: {len(search_queries)}")
        
        return collected_content
    
    async def collect_github_repositories(self,
                                        session_id: UUID,
                                        search_terms: List[str],
                                        max_repos: int = 50) -> List[ExtractedContent]:
        """
        Collect information from GitHub repositories.
        """
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Collection session {session_id} not found")
        
        session = self.active_sessions[session_id]
        collected_content = []
        
        for term in search_terms:
            try:
                # GitHub search API (anonymous)
                url = f"https://api.github.com/search/repositories?q={term}&sort=updated&per_page={min(max_repos, 100)}"
                
                response = await self.network_manager.send_anonymous_request(
                    session_id=session.privacy_session_id,
                    url=url,
                    method="GET"
                )
                
                # Parse GitHub response
                content_items = await self._parse_github_response(response, term)
                collected_content.extend(content_items)
                
                session.urls_processed += 1
                session.successful_extractions += len(content_items)
                
                # GitHub rate limiting
                await asyncio.sleep(1.0)
                
            except Exception as e:
                session.failed_extractions += 1
                print(f"⚠️ GitHub collection failed for term '{term}': {e}")
        
        session.total_content_collected += len(collected_content)
        session.total_bytes_collected += sum(c.processed_size_bytes for c in collected_content)
        
        print(f"💻 GitHub collection completed")
        print(f"   - Repositories collected: {len(collected_content)}")
        print(f"   - Search terms processed: {len(search_terms)}")
        
        return collected_content
    
    async def collect_web_content(self,
                                session_id: UUID,
                                urls: List[str],
                                content_type: ContentType = ContentType.WEB_PAGE) -> List[ExtractedContent]:
        """
        Collect content from arbitrary web URLs.
        """
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Collection session {session_id} not found")
        
        session = self.active_sessions[session_id]
        collected_content = []
        
        for url in urls:
            try:
                # Rate limiting check
                domain = urlparse(url).netloc
                if not await self._check_rate_limit(domain):
                    await asyncio.sleep(5.0)  # Back off if rate limited
                    continue
                
                # Make anonymous request
                response = await self.network_manager.send_anonymous_request(
                    session_id=session.privacy_session_id,
                    url=url,
                    method="GET"
                )
                
                # Extract content based on type
                content_item = await self._extract_web_content(url, response, content_type)
                
                if content_item and content_item.content_quality_score > 0.3:
                    collected_content.append(content_item)
                    session.successful_extractions += 1
                else:
                    session.failed_extractions += 1
                
                session.urls_processed += 1
                
                # Update rate limiting
                self._update_rate_limit(domain)
                
                # Respectful delay
                await asyncio.sleep(secrets.uniform(1.0, 3.0))
                
            except Exception as e:
                session.failed_extractions += 1
                print(f"⚠️ Web content collection failed for {url}: {e}")
        
        session.total_content_collected += len(collected_content)
        session.total_bytes_collected += sum(c.processed_size_bytes for c in collected_content)
        
        print(f"🌐 Web content collection completed")
        print(f"   - Pages collected: {len(collected_content)}")
        print(f"   - URLs processed: {len(urls)}")
        
        return collected_content
    
    async def get_collection_status(self, session_id: UUID) -> Dict[str, Any]:
        """
        Get detailed status of a collection session.
        """
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Collection session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Calculate progress metrics
        progress_percentage = (session.urls_processed / session.total_urls_planned * 100 
                             if session.total_urls_planned > 0 else 0)
        
        success_rate = (session.successful_extractions / session.urls_processed 
                       if session.urls_processed > 0 else 0)
        
        # Calculate quality metrics
        session_content = [c for c in self.extracted_content.values() 
                         if c.extraction_timestamp >= session.started_at]
        
        avg_quality = (sum(c.content_quality_score for c in session_content) / len(session_content) 
                      if session_content else 0)
        
        high_quality_count = sum(1 for c in session_content if c.content_quality_score > 0.7)
        
        return {
            "session_info": {
                "session_id": session_id,
                "started_at": session.started_at,
                "runtime_minutes": (datetime.now(timezone.utc) - session.started_at).total_seconds() / 60,
                "estimated_completion": session.estimated_completion
            },
            "progress": {
                "urls_processed": session.urls_processed,
                "total_urls_planned": session.total_urls_planned,
                "progress_percentage": progress_percentage,
                "successful_extractions": session.successful_extractions,
                "failed_extractions": session.failed_extractions,
                "success_rate": success_rate
            },
            "content_quality": {
                "total_content_items": len(session_content),
                "average_quality_score": avg_quality,
                "high_quality_items": high_quality_count,
                "total_bytes_collected": session.total_bytes_collected
            },
            "source_diversity": {
                "target_sources": len(session.target_sources),
                "domains_covered": len(session.domains_of_interest),
                "unique_sources_accessed": session.unique_sources_accessed
            }
        }
    
    def _initialize_default_sources(self):
        """Initialize built-in source configurations"""
        
        # ArXiv
        arxiv_config = SourceConfiguration(
            source_id=uuid4(),
            source_type=SourceType.ARXIV,
            base_url="http://export.arxiv.org/api/query",
            max_requests_per_hour=30,  # Respectful rate
            request_delay_seconds=2.0,
            allowed_content_types=[ContentType.RESEARCH_PAPER],
            quality_threshold=0.6
        )
        self.configured_sources[arxiv_config.source_id] = arxiv_config
        
        # GitHub
        github_config = SourceConfiguration(
            source_id=uuid4(),
            source_type=SourceType.GITHUB,
            base_url="https://api.github.com",
            max_requests_per_hour=60,  # GitHub's anonymous limit
            request_delay_seconds=1.0,
            allowed_content_types=[ContentType.CODE_REPOSITORY, ContentType.DOCUMENTATION],
            quality_threshold=0.5
        )
        self.configured_sources[github_config.source_id] = github_config
        
        # PubMed
        pubmed_config = SourceConfiguration(
            source_id=uuid4(),
            source_type=SourceType.PUBMED,
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            max_requests_per_hour=180,  # NCBI limit
            request_delay_seconds=0.34,  # 3 requests per second max
            allowed_content_types=[ContentType.RESEARCH_PAPER],
            quality_threshold=0.7
        )
        self.configured_sources[pubmed_config.source_id] = pubmed_config
    
    async def _estimate_collection_scope(self, source_ids: List[UUID], domains: List[str]) -> int:
        """Estimate total URLs to be processed"""
        # Simplified estimation
        base_urls_per_source = 50
        domain_multiplier = len(domains) if domains else 1
        return len(source_ids) * base_urls_per_source * domain_multiplier
    
    async def _parse_arxiv_response(self, response: Dict[str, Any], query: str) -> List[ExtractedContent]:
        """Parse ArXiv API response and extract paper information"""
        
        content_items = []
        
        try:
            # Simulated ArXiv parsing (in production, parse actual XML response)
            for i in range(5):  # Simulate 5 papers found
                content = ExtractedContent(
                    source_url=f"https://arxiv.org/abs/2301.{i:05d}",
                    source_type=SourceType.ARXIV,
                    content_type=ContentType.RESEARCH_PAPER,
                    title=f"Research Paper {i+1} for {query}",
                    abstract=f"Abstract for paper {i+1} related to {query}...",
                    authors=[f"Author {j}" for j in range(1, 4)],
                    keywords=[query, "machine learning", "research"],
                    publication_date=datetime.now(timezone.utc) - timedelta(days=i*10),
                    content_quality_score=0.7 + (i * 0.05),
                    relevance_score=0.8,
                    novelty_score=0.6,
                    content_hash=hashlib.sha256(f"arxiv_paper_{i}_{query}".encode()).hexdigest(),
                    original_size_bytes=15000 + i * 1000,
                    processed_size_bytes=12000 + i * 800
                )
                
                content_items.append(content)
                self.extracted_content[content.content_id] = content
            
        except Exception as e:
            print(f"⚠️ Error parsing ArXiv response: {e}")
        
        return content_items
    
    async def _parse_github_response(self, response: Dict[str, Any], term: str) -> List[ExtractedContent]:
        """Parse GitHub API response and extract repository information"""
        
        content_items = []
        
        try:
            # Simulated GitHub parsing
            for i in range(3):  # Simulate 3 repos found
                content = ExtractedContent(
                    source_url=f"https://github.com/user{i}/repo-{term.lower()}-{i}",
                    source_type=SourceType.GITHUB,
                    content_type=ContentType.CODE_REPOSITORY,
                    title=f"Repository: {term.title()} Project {i+1}",
                    abstract=f"A repository related to {term} with various implementations...",
                    keywords=[term, "code", "implementation"],
                    last_modified=datetime.now(timezone.utc) - timedelta(days=i*5),
                    content_quality_score=0.6 + (i * 0.1),
                    relevance_score=0.75,
                    novelty_score=0.5,
                    content_hash=hashlib.sha256(f"github_repo_{i}_{term}".encode()).hexdigest(),
                    original_size_bytes=25000 + i * 2000,
                    processed_size_bytes=20000 + i * 1600
                )
                
                content_items.append(content)
                self.extracted_content[content.content_id] = content
            
        except Exception as e:
            print(f"⚠️ Error parsing GitHub response: {e}")
        
        return content_items
    
    async def _extract_web_content(self, url: str, response: Dict[str, Any], content_type: ContentType) -> Optional[ExtractedContent]:
        """Extract content from web page response"""
        
        try:
            # Simulated web content extraction
            content = ExtractedContent(
                source_url=url,
                source_type=SourceType.WEB_PAGE,
                content_type=content_type,
                title=f"Content from {urlparse(url).netloc}",
                full_text=f"Extracted text content from {url}...",
                keywords=["web", "content", "extracted"],
                content_quality_score=0.5 + secrets.random() * 0.4,
                relevance_score=0.6,
                novelty_score=0.4,
                content_hash=hashlib.sha256(f"web_content_{url}".encode()).hexdigest(),
                original_size_bytes=10000,
                processed_size_bytes=8000
            )
            
            self.extracted_content[content.content_id] = content
            return content
            
        except Exception as e:
            print(f"⚠️ Error extracting content from {url}: {e}")
            return None
    
    async def _check_rate_limit(self, domain: str) -> bool:
        """Check if we're within rate limits for a domain"""
        
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        
        if domain not in self.request_timestamps:
            self.request_timestamps[domain] = []
        
        # Clean old timestamps
        self.request_timestamps[domain] = [
            ts for ts in self.request_timestamps[domain] if ts > hour_ago
        ]
        
        # Check if we're under the limit (default 60 requests per hour)
        return len(self.request_timestamps[domain]) < 60
    
    def _update_rate_limit(self, domain: str):
        """Update rate limiting records for a domain"""
        
        if domain not in self.request_timestamps:
            self.request_timestamps[domain] = []
        
        self.request_timestamps[domain].append(datetime.now(timezone.utc))
        self.total_requests_made += 1


# Global external data collector instance
external_data_collector = ExternalDataCollector()