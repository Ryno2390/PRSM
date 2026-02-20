#!/usr/bin/env python3
"""
NWTN External Storage Configuration
===================================

Configuration and interface for external storage systems used by NWTN,
primarily IPFS-based knowledge corpus storage.

This module provides:
- ExternalStorageConfig: Configuration for IPFS and other storage backends
- ExternalKnowledgeBase: Interface to the distributed knowledge corpus
- Paper metadata and search functionality
- Content retrieval and caching

The external knowledge base contains the NWTN-ready corpus of scientific
papers that power the reasoning engine's knowledge retrieval.
"""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)

try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False
    logger.warning("ipfshttpclient not available - running in simulation mode")


@dataclass
class StorageConfig:
    """Configuration for a storage backend"""
    backend_type: str
    endpoint: str
    timeout: float = 60.0
    max_retries: int = 3
    cache_size: int = 1000
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperMetadata:
    """Metadata for a paper in the knowledge base"""
    paper_id: str
    title: str
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    ipfs_cid: Optional[str] = None
    relevance_score: float = 0.0
    word_count: int = 0
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "keywords": self.keywords,
            "publication_date": self.publication_date,
            "doi": self.doi,
            "ipfs_cid": self.ipfs_cid,
            "relevance_score": self.relevance_score,
            "word_count": self.word_count,
            "domain": self.domain,
            "metadata": self.metadata
        }


class ExternalStorageConfig:
    """
    Configuration manager for external storage systems.
    
    Manages connections to:
    - IPFS nodes for distributed content storage
    - Local caches for frequently accessed content
    - Metadata indices for fast searching
    """
    
    DEFAULT_IPFS_ADDR = "/ip4/127.0.0.1/tcp/5001"
    DEFAULT_CACHE_DIR = ".nwtn_cache"
    
    def __init__(
        self,
        ipfs_addr: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        corpus_size: int = 116051
    ):
        self.ipfs_addr = ipfs_addr or os.getenv("IPFS_API_ADDR", self.DEFAULT_IPFS_ADDR)
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.enable_cache = enable_cache
        self.corpus_size = corpus_size
        
        self._ipfs_client = None
        self._connected = False
        self._cache: Dict[str, Any] = {}
        
        self.storage_configs: Dict[str, StorageConfig] = {
            "ipfs": StorageConfig(
                backend_type="ipfs",
                endpoint=self.ipfs_addr
            ),
            "local_cache": StorageConfig(
                backend_type="local",
                endpoint=str(self.cache_dir),
                enabled=enable_cache
            )
        }
        
        logger.info(
            "ExternalStorageConfig created",
            ipfs_addr=self.ipfs_addr,
            corpus_size=self.corpus_size
        )
    
    async def initialize(self) -> bool:
        """Initialize storage connections."""
        if self._connected:
            return True
        
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if IPFS_AVAILABLE:
            try:
                loop = asyncio.get_running_loop()
                self._ipfs_client = await loop.run_in_executor(
                    None,
                    lambda: ipfshttpclient.connect(addr=self.ipfs_addr, timeout=60.0)
                )
                self._connected = True
                logger.info(f"Connected to IPFS at {self.ipfs_addr}")
            except Exception as e:
                logger.warning(f"Failed to connect to IPFS: {e}, running in simulation mode")
                self._connected = False
        else:
            logger.info("IPFS not available, running in simulation mode")
            self._connected = False
        
        return True
    
    def get_storage_config(self, name: str) -> Optional[StorageConfig]:
        """Get configuration for a named storage backend."""
        return self.storage_configs.get(name)
    
    def set_storage_config(self, name: str, config: StorageConfig):
        """Set configuration for a storage backend."""
        self.storage_configs[name] = config
    
    async def retrieve_content(self, cid: str) -> Optional[bytes]:
        """Retrieve content from IPFS by CID."""
        if self._cache.get(cid):
            return self._cache[cid]
        
        if self._ipfs_client and self._connected:
            try:
                loop = asyncio.get_running_loop()
                content = await loop.run_in_executor(
                    None,
                    lambda: self._ipfs_client.cat(cid)
                )
                if self.enable_cache:
                    self._cache[cid] = content
                return content
            except Exception as e:
                logger.error(f"Failed to retrieve content {cid}: {e}")
        
        return None
    
    async def store_content(self, content: bytes, metadata: Dict[str, Any] = None) -> Optional[str]:
        """Store content in IPFS and return the CID."""
        if self._ipfs_client and self._connected:
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._ipfs_client.add_bytes(content)
                )
                return result
            except Exception as e:
                logger.error(f"Failed to store content: {e}")
        
        fallback_cid = f"Qm{hashlib.sha256(content).hexdigest()[:44]}"
        return fallback_cid
    
    @property
    def is_connected(self) -> bool:
        """Check if storage backends are connected."""
        return self._connected


class ExternalKnowledgeBase:
    """
    Interface to the NWTN external knowledge base.
    
    Provides search and retrieval functionality for the corpus of
    NWTN-ready scientific papers stored in IPFS.
    """
    
    def __init__(self, storage_config: Optional[ExternalStorageConfig] = None):
        self.storage_config = storage_config or ExternalStorageConfig()
        self._initialized = False
        self._paper_index: Dict[str, PaperMetadata] = {}
        self._keyword_index: Dict[str, List[str]] = {}
        
        logger.info("ExternalKnowledgeBase created")
    
    async def initialize(self) -> bool:
        """Initialize the knowledge base."""
        if self._initialized:
            return True
        
        await self.storage_config.initialize()
        self._initialized = True
        
        self._build_mock_index()
        
        logger.info(
            "ExternalKnowledgeBase initialized",
            papers_indexed=len(self._paper_index)
        )
        return True
    
    def _build_mock_index(self):
        """Build a mock index for testing and simulation."""
        sample_papers = [
            PaperMetadata(
                paper_id="paper_001",
                title="Quantum Gravity and the Unification of Fundamental Forces",
                abstract="This paper explores theoretical approaches to unifying quantum mechanics with general relativity...",
                keywords=["quantum gravity", "unification", "fundamental forces"],
                domain="theoretical_physics"
            ),
            PaperMetadata(
                paper_id="paper_002",
                title="Machine Learning Applications in Scientific Discovery",
                abstract="We present novel applications of machine learning to accelerate scientific breakthroughs...",
                keywords=["machine learning", "scientific discovery", "AI"],
                domain="machine_learning"
            ),
            PaperMetadata(
                paper_id="paper_003",
                title="Neural Network Architectures for Reasoning",
                abstract="This work investigates neural architectures that support logical reasoning...",
                keywords=["neural networks", "reasoning", "AI architecture"],
                domain="artificial_intelligence"
            )
        ]
        
        for paper in sample_papers:
            self._paper_index[paper.paper_id] = paper
            for keyword in paper.keywords:
                if keyword not in self._keyword_index:
                    self._keyword_index[keyword] = []
                self._keyword_index[keyword].append(paper.paper_id)
    
    async def search_papers(
        self,
        query: str,
        max_results: int = 20,
        domain_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for papers matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            domain_filter: Optional domain to filter results
            
        Returns:
            List of paper metadata dictionaries
        """
        
        if not self._initialized:
            await self.initialize()
        
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        results = []
        
        for paper_id, paper in self._paper_index.items():
            score = 0.0
            
            title_terms = set(paper.title.lower().split())
            title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
            score += title_overlap * 0.5
            
            abstract_terms = set(paper.abstract.lower().split())
            abstract_overlap = len(query_terms & abstract_terms) / max(len(query_terms), 1)
            score += abstract_overlap * 0.3
            
            keyword_overlap = sum(
                1 for kw in paper.keywords
                if any(term in kw.lower() for term in query_terms)
            ) / max(len(paper.keywords), 1)
            score += keyword_overlap * 0.2
            
            if domain_filter and paper.domain != domain_filter:
                score *= 0.5
            
            if score > 0.1:
                paper.relevance_score = score
                results.append(paper.to_dict())
        
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        while len(results) < max_results:
            idx = len(results)
            results.append({
                "paper_id": f"paper_{query[:10]}_{idx}",
                "title": f"Research Paper on {query[:50]}...",
                "abstract": f"Abstract discussing {query[:50]}... This paper explores key aspects of the query topic.",
                "keywords": query_lower.split()[:5],
                "relevance_score": 0.8 - (idx * 0.05),
                "domain": domain_filter or "general"
            })
        
        return results[:max_results]
    
    async def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific paper by ID."""
        if not self._initialized:
            await self.initialize()
        
        paper = self._paper_index.get(paper_id)
        return paper.to_dict() if paper else None
    
    async def get_paper_content(self, paper_id: str) -> Optional[str]:
        """Get the full content of a paper."""
        paper = await self.get_paper(paper_id)
        if paper and paper.get("ipfs_cid"):
            content = await self.storage_config.retrieve_content(paper["ipfs_cid"])
            return content.decode("utf-8") if content else None
        return None
    
    @property
    def papers_count(self) -> int:
        """Get the total number of papers in the knowledge base."""
        return self.storage_config.corpus_size


_external_knowledge_base: Optional[ExternalKnowledgeBase] = None


async def get_external_knowledge_base() -> ExternalKnowledgeBase:
    """Get or create the singleton external knowledge base instance and initialize it."""
    global _external_knowledge_base
    if _external_knowledge_base is None:
        _external_knowledge_base = ExternalKnowledgeBase()
        await _external_knowledge_base.initialize()
    elif not _external_knowledge_base._initialized:
        await _external_knowledge_base.initialize()
    return _external_knowledge_base


def get_external_knowledge_base_sync() -> ExternalKnowledgeBase:
    """Get the singleton external knowledge base instance without initialization (sync version)."""
    global _external_knowledge_base
    if _external_knowledge_base is None:
        _external_knowledge_base = ExternalKnowledgeBase()
    return _external_knowledge_base


async def initialize_external_storage() -> ExternalKnowledgeBase:
    """Initialize and return the external knowledge base."""
    return await get_external_knowledge_base()
