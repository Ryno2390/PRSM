#!/usr/bin/env python3
"""
Breadth-Optimized Content Sources Configuration
===============================================

This module defines public content sources optimized for maximum domain breadth
to enhance NWTN's analogical reasoning capabilities.

Key Features:
1. Diverse domain coverage for maximum analogical potential
2. Public, freely accessible content sources
3. High-quality academic and research content
4. Cross-domain connection opportunities
5. Structured metadata for quality assessment

Sources prioritized for breadth over depth to maximize analogical reasoning.
"""

import asyncio
import json
import logging
import aiohttp
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ContentSource:
    """Content source configuration"""
    name: str
    description: str
    domain: str
    base_url: str
    api_endpoint: str
    rate_limit: int  # requests per minute
    max_items: int
    query_params: Dict[str, Any]
    enabled: bool = True


class BreadthOptimizedContentSources:
    """
    Breadth-Optimized Content Sources Manager
    
    Manages diverse content sources optimized for maximum domain breadth
    to enhance analogical reasoning capabilities.
    """
    
    def __init__(self):
        self.sources = self._initialize_sources()
        self.session = None
        
        logger.info("Breadth-Optimized Content Sources initialized")
    
    def _initialize_sources(self) -> Dict[str, ContentSource]:
        """Initialize all content sources with breadth optimization"""
        
        sources = {
            # ArXiv - Multiple domains for breadth
            "arxiv_cs": ContentSource(
                name="ArXiv Computer Science",
                description="Computer Science papers from ArXiv",
                domain="computer_science",
                base_url="http://export.arxiv.org/api/query",
                api_endpoint="",
                rate_limit=10,  # ArXiv allows ~1 request per 3 seconds
                max_items=20000,
                query_params={
                    "search_query": "cat:cs.*",
                    "start": 0,
                    "max_results": 100,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending"
                }
            ),
            
            "arxiv_physics": ContentSource(
                name="ArXiv Physics",
                description="Physics papers from ArXiv",
                domain="physics",
                base_url="http://export.arxiv.org/api/query",
                api_endpoint="",
                rate_limit=10,
                max_items=20000,
                query_params={
                    "search_query": "cat:physics.*",
                    "start": 0,
                    "max_results": 100,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending"
                }
            ),
            
            "arxiv_math": ContentSource(
                name="ArXiv Mathematics",
                description="Mathematics papers from ArXiv",
                domain="mathematics",
                base_url="http://export.arxiv.org/api/query",
                api_endpoint="",
                rate_limit=10,
                max_items=20000,
                query_params={
                    "search_query": "cat:math.*",
                    "start": 0,
                    "max_results": 100,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending"
                }
            ),
            
            "arxiv_bio": ContentSource(
                name="ArXiv Biology",
                description="Biology papers from ArXiv",
                domain="biology",
                base_url="http://export.arxiv.org/api/query",
                api_endpoint="",
                rate_limit=10,
                max_items=15000,
                query_params={
                    "search_query": "cat:q-bio.*",
                    "start": 0,
                    "max_results": 100,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending"
                }
            ),
            
            # PubMed - Biomedical literature
            "pubmed": ContentSource(
                name="PubMed",
                description="Biomedical literature from PubMed",
                domain="biomedical",
                base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                api_endpoint="esearch.fcgi",
                rate_limit=180,  # 3 requests per second
                max_items=25000,
                query_params={
                    "db": "pubmed",
                    "term": "science[MeSH] OR research[MeSH]",
                    "retmax": 100,
                    "retmode": "xml",
                    "sort": "pub_date",
                    "retstart": 0
                }
            ),
            
            # Semantic Scholar - Cross-domain academic papers
            "semantic_scholar": ContentSource(
                name="Semantic Scholar",
                description="Cross-domain academic papers from Semantic Scholar",
                domain="multidisciplinary",
                base_url="https://api.semanticscholar.org/graph/v1/",
                api_endpoint="paper/search",
                rate_limit=60,  # 1 request per second
                max_items=30000,
                query_params={
                    "query": "research AND (method OR methodology OR approach)",
                    "limit": 100,
                    "offset": 0,
                    "fields": "title,abstract,authors,year,venue,citationCount,fieldsOfStudy"
                }
            ),
            
            # bioRxiv - Biology preprints
            "biorxiv": ContentSource(
                name="bioRxiv",
                description="Biology preprints from bioRxiv",
                domain="biology",
                base_url="https://api.biorxiv.org/",
                api_endpoint="details/biorxiv",
                rate_limit=60,
                max_items=15000,
                query_params={
                    "server": "biorxiv",
                    "count": 100,
                    "format": "json"
                }
            ),
            
            # medRxiv - Medical preprints
            "medrxiv": ContentSource(
                name="medRxiv",
                description="Medical preprints from medRxiv",
                domain="medicine",
                base_url="https://api.biorxiv.org/",
                api_endpoint="details/medrxiv",
                rate_limit=60,
                max_items=15000,
                query_params={
                    "server": "medrxiv",
                    "count": 100,
                    "format": "json"
                }
            ),
            
            # OpenReview - ML/AI reviews and papers
            "openreview": ContentSource(
                name="OpenReview",
                description="ML/AI reviews and papers from OpenReview",
                domain="ai_ml",
                base_url="https://api.openreview.net/",
                api_endpoint="notes",
                rate_limit=30,
                max_items=10000,
                query_params={
                    "details": "replyCount,invitation,original",
                    "limit": 100,
                    "offset": 0
                }
            ),
            
            # DOAJ - Open access journals (multidisciplinary)
            "doaj": ContentSource(
                name="DOAJ",
                description="Open access journals from DOAJ",
                domain="multidisciplinary",
                base_url="https://doaj.org/api/v2/",
                api_endpoint="search/articles",
                rate_limit=60,
                max_items=20000,
                query_params={
                    "pageSize": 100,
                    "page": 1,
                    "sort": "created_date:desc"
                }
            ),
            
            # CORE - Open access research papers
            "core": ContentSource(
                name="CORE",
                description="Open access research papers from CORE",
                domain="multidisciplinary",
                base_url="https://api.core.ac.uk/v3/",
                api_endpoint="search/works",
                rate_limit=60,
                max_items=25000,
                query_params={
                    "q": "research methodology",
                    "limit": 100,
                    "offset": 0
                }
            )
        }
        
        # Log source configuration
        total_max_items = sum(source.max_items for source in sources.values() if source.enabled)
        enabled_domains = {source.domain for source in sources.values() if source.enabled}
        
        logger.info(f"📚 Configured {len(sources)} content sources")
        logger.info(f"🎯 Target maximum items: {total_max_items:,}")
        logger.info(f"🌍 Domains covered: {len(enabled_domains)}")
        logger.info(f"📖 Domains: {sorted(enabled_domains)}")
        
        return sources
    
    async def initialize(self):
        """Initialize HTTP session and connections"""
        
        logger.info("🔗 Initializing content source connections...")
        
        # Create HTTP session with appropriate headers
        self.session = aiohttp.ClientSession(
            headers={
                "User-Agent": "PRSM-NWTN/1.0 (Educational Research; contact@prsm.ai)",
                "Accept": "application/json, application/xml, text/xml"
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Test connections to all enabled sources
        connection_results = {}
        for source_name, source in self.sources.items():
            if source.enabled:
                try:
                    await self._test_source_connection(source)
                    connection_results[source_name] = "✅ Connected"
                except Exception as e:
                    connection_results[source_name] = f"❌ Failed: {e}"
                    logger.warning(f"Source connection failed: {source_name}: {e}")
        
        # Log connection results
        logger.info("📡 Source connection results:")
        for source_name, result in connection_results.items():
            logger.info(f"   {source_name}: {result}")
        
        logger.info("✅ Content source connections initialized")
    
    async def _test_source_connection(self, source: ContentSource):
        """Test connection to a content source"""
        
        # Build test URL
        if source.api_endpoint:
            test_url = f"{source.base_url}{source.api_endpoint}"
        else:
            test_url = source.base_url
        
        # Create minimal test params
        test_params = source.query_params.copy()
        if "max_results" in test_params:
            test_params["max_results"] = 1
        if "limit" in test_params:
            test_params["limit"] = 1
        if "retmax" in test_params:
            test_params["retmax"] = 1
        if "pageSize" in test_params:
            test_params["pageSize"] = 1
        
        async with self.session.get(test_url, params=test_params) as response:
            if response.status == 200:
                return True
            else:
                raise Exception(f"HTTP {response.status}")
    
    async def get_content_iterator(self, source_name: str) -> AsyncIterator[Dict[str, Any]]:
        """Get content iterator for a specific source"""
        
        if source_name not in self.sources:
            raise ValueError(f"Unknown source: {source_name}")
        
        source = self.sources[source_name]
        if not source.enabled:
            raise ValueError(f"Source disabled: {source_name}")
        
        logger.info(f"📥 Starting content stream from {source_name}")
        
        # Get appropriate iterator based on source type
        if "arxiv" in source_name:
            async for item in self._get_arxiv_content(source):
                yield item
        elif source_name == "pubmed":
            async for item in self._get_pubmed_content(source):
                yield item
        elif source_name == "semantic_scholar":
            async for item in self._get_semantic_scholar_content(source):
                yield item
        elif source_name in ["biorxiv", "medrxiv"]:
            async for item in self._get_biorxiv_content(source):
                yield item
        elif source_name == "openreview":
            async for item in self._get_openreview_content(source):
                yield item
        elif source_name == "doaj":
            async for item in self._get_doaj_content(source):
                yield item
        elif source_name == "core":
            async for item in self._get_core_content(source):
                yield item
        else:
            logger.warning(f"No iterator implemented for source: {source_name}")
    
    async def _get_arxiv_content(self, source: ContentSource) -> AsyncIterator[Dict[str, Any]]:
        """Get content from ArXiv API"""
        
        start = 0
        max_results = source.query_params["max_results"]
        rate_limit_delay = 60 / source.rate_limit  # Convert to seconds between requests
        
        while start < source.max_items:
            try:
                # Update pagination
                params = source.query_params.copy()
                params["start"] = start
                
                # Make API request
                async with self.session.get(source.base_url, params=params) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        
                        # Parse ArXiv XML response
                        root = ET.fromstring(xml_content)
                        
                        # Extract entries
                        entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
                        
                        if not entries:
                            logger.info(f"No more entries from {source.name}")
                            break
                        
                        for entry in entries:
                            # Extract paper information
                            paper_data = await self._parse_arxiv_entry(entry, source)
                            if paper_data:
                                yield paper_data
                        
                        start += max_results
                        
                        # Rate limiting
                        await asyncio.sleep(rate_limit_delay)
                        
                    else:
                        logger.error(f"ArXiv API error: {response.status}")
                        break
                        
            except Exception as e:
                logger.error(f"Error fetching ArXiv content: {e}")
                break
    
    async def _parse_arxiv_entry(self, entry: ET.Element, source: ContentSource) -> Optional[Dict[str, Any]]:
        """Parse ArXiv XML entry"""
        
        try:
            # Extract basic information
            title = entry.find('.//{http://www.w3.org/2005/Atom}title')
            abstract = entry.find('.//{http://www.w3.org/2005/Atom}summary')
            published = entry.find('.//{http://www.w3.org/2005/Atom}published')
            id_elem = entry.find('.//{http://www.w3.org/2005/Atom}id')
            
            # Extract categories
            categories = []
            for category in entry.findall('.//{http://www.w3.org/2005/Atom}category'):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Extract authors
            authors = []
            for author in entry.findall('.//{http://www.w3.org/2005/Atom}author'):
                name = author.find('.//{http://www.w3.org/2005/Atom}name')
                if name is not None:
                    authors.append(name.text)
            
            # Build paper data
            paper_data = {
                "id": id_elem.text.split('/')[-1] if id_elem is not None else None,
                "title": title.text.strip() if title is not None else "",
                "abstract": abstract.text.strip() if abstract is not None else "",
                "authors": authors,
                "published_date": published.text if published is not None else None,
                "categories": categories,
                "keywords": categories,  # Use categories as keywords
                "domain": source.domain,
                "type": "preprint" if "arxiv" in source.name else "research_paper",
                "source": source.name,
                "url": id_elem.text if id_elem is not None else None,
                "venue": "ArXiv",
                "year": int(published.text[:4]) if published is not None else None
            }
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Error parsing ArXiv entry: {e}")
            return None
    
    async def _get_semantic_scholar_content(self, source: ContentSource) -> AsyncIterator[Dict[str, Any]]:
        """Get content from Semantic Scholar API"""
        
        offset = 0
        limit = source.query_params["limit"]
        rate_limit_delay = 60 / source.rate_limit
        
        while offset < source.max_items:
            try:
                # Update pagination
                params = source.query_params.copy()
                params["offset"] = offset
                
                # Build URL
                url = f"{source.base_url}{source.api_endpoint}"
                
                # Make API request
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        papers = data.get("data", [])
                        if not papers:
                            logger.info(f"No more papers from {source.name}")
                            break
                        
                        for paper in papers:
                            paper_data = await self._parse_semantic_scholar_paper(paper, source)
                            if paper_data:
                                yield paper_data
                        
                        offset += limit
                        
                        # Rate limiting
                        await asyncio.sleep(rate_limit_delay)
                        
                    else:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        break
                        
            except Exception as e:
                logger.error(f"Error fetching Semantic Scholar content: {e}")
                break
    
    async def _parse_semantic_scholar_paper(self, paper: Dict[str, Any], source: ContentSource) -> Optional[Dict[str, Any]]:
        """Parse Semantic Scholar paper"""
        
        try:
            # Extract author names
            authors = []
            for author in paper.get("authors", []):
                if "name" in author:
                    authors.append(author["name"])
            
            # Extract fields of study
            fields = paper.get("fieldsOfStudy", [])
            
            paper_data = {
                "id": paper.get("paperId"),
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": authors,
                "published_date": paper.get("publicationDate"),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "citation_count": paper.get("citationCount", 0),
                "fields_of_study": fields,
                "keywords": fields,  # Use fields of study as keywords
                "domain": source.domain,
                "type": "research_paper",
                "source": source.name,
                "url": paper.get("url")
            }
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Error parsing Semantic Scholar paper: {e}")
            return None
    
    # Placeholder implementations for other sources
    async def _get_pubmed_content(self, source: ContentSource) -> AsyncIterator[Dict[str, Any]]:
        """Get content from PubMed API"""
        # Implementation would go here
        logger.warning(f"PubMed iterator not fully implemented")
        return
        yield  # Make it a generator
    
    async def _get_biorxiv_content(self, source: ContentSource) -> AsyncIterator[Dict[str, Any]]:
        """Get content from bioRxiv/medRxiv API"""
        # Implementation would go here
        logger.warning(f"bioRxiv/medRxiv iterator not fully implemented")
        return
        yield  # Make it a generator
    
    async def _get_openreview_content(self, source: ContentSource) -> AsyncIterator[Dict[str, Any]]:
        """Get content from OpenReview API"""
        # Implementation would go here
        logger.warning(f"OpenReview iterator not fully implemented")
        return
        yield  # Make it a generator
    
    async def _get_doaj_content(self, source: ContentSource) -> AsyncIterator[Dict[str, Any]]:
        """Get content from DOAJ API"""
        # Implementation would go here
        logger.warning(f"DOAJ iterator not fully implemented")
        return
        yield  # Make it a generator
    
    async def _get_core_content(self, source: ContentSource) -> AsyncIterator[Dict[str, Any]]:
        """Get content from CORE API"""
        # Implementation would go here
        logger.warning(f"CORE iterator not fully implemented")
        return
        yield  # Make it a generator
    
    def get_enabled_sources(self) -> List[str]:
        """Get list of enabled source names"""
        return [name for name, source in self.sources.items() if source.enabled]
    
    def get_domain_coverage(self) -> Dict[str, int]:
        """Get domain coverage statistics"""
        domain_counts = {}
        for source in self.sources.values():
            if source.enabled:
                domain_counts[source.domain] = domain_counts.get(source.domain, 0) + source.max_items
        return domain_counts
    
    def get_breadth_optimization_score(self) -> float:
        """Calculate breadth optimization score"""
        enabled_domains = {source.domain for source in self.sources.values() if source.enabled}
        total_possible_domains = len(set(source.domain for source in self.sources.values()))
        return len(enabled_domains) / total_possible_domains
    
    async def shutdown(self):
        """Shutdown content sources"""
        
        logger.info("🔄 Shutting down content sources...")
        
        if self.session:
            await self.session.close()
        
        logger.info("✅ Content sources shutdown complete")


# Test function
async def test_content_sources():
    """Test content sources functionality"""
    
    print("📚 BREADTH-OPTIMIZED CONTENT SOURCES TEST")
    print("=" * 60)
    
    sources = BreadthOptimizedContentSources()
    
    try:
        # Initialize
        await sources.initialize()
        
        # Test domain coverage
        domain_coverage = sources.get_domain_coverage()
        print(f"📊 Domain Coverage:")
        for domain, count in domain_coverage.items():
            print(f"   {domain}: {count:,} items")
        
        # Test breadth score
        breadth_score = sources.get_breadth_optimization_score()
        print(f"🎯 Breadth Optimization Score: {breadth_score:.2f}")
        
        # Test source listing
        enabled_sources = sources.get_enabled_sources()
        print(f"✅ Enabled Sources: {len(enabled_sources)}")
        for source in enabled_sources:
            print(f"   - {source}")
        
        # Test content iterator (limited)
        print(f"\n📥 Testing ArXiv content iterator...")
        content_iterator = sources.get_content_iterator("arxiv_cs")
        count = 0
        async for paper in content_iterator:
            count += 1
            print(f"   Paper {count}: {paper['title'][:50]}...")
            if count >= 3:  # Just test first 3 papers
                break
        
        print("✅ Content sources test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await sources.shutdown()


if __name__ == "__main__":
    asyncio.run(test_content_sources())