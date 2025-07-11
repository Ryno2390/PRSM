#!/usr/bin/env python3
"""
NWTN Knowledge Corpus Interface
Unified interface between NWTN and IPFS-based knowledge corpus

This module provides NWTN with direct access to the cryptographically marked
content corpus stored in IPFS, enabling the hybrid reasoning system to leverage
verified public knowledge sources alongside its internal knowledge base.

Key Features:
1. Direct IPFS corpus access for NWTN reasoning
2. Content verification and provenance tracking
3. SOC generation from corpus content
4. Cross-domain knowledge retrieval
5. Real-time corpus indexing and search
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from datetime import datetime, timezone
import hashlib

import structlog
from pydantic import BaseModel, Field

from ..ipfs.content_addressing import ContentAddressingSystem, AddressedContent, ContentCategory
from ..ipfs.content_verification import ContentVerificationSystem, VerificationResult
from .hybrid_architecture import SOC, SOCType, ConfidenceLevel, HybridNWTNEngine
from .world_model_engine import WorldModelEngine
from ..embeddings.semantic_embedding_engine import SemanticEmbeddingEngine, EmbeddingSearchQuery, EmbeddingSpace

logger = structlog.get_logger(__name__)


class CorpusSearchType(str, Enum):
    """Types of corpus searches"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    CITATION = "citation"
    DOMAIN_SPECIFIC = "domain_specific"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


class ContentRelevance(str, Enum):
    """Content relevance levels"""
    HIGHLY_RELEVANT = "highly_relevant"
    MODERATELY_RELEVANT = "moderately_relevant"
    TANGENTIALLY_RELEVANT = "tangentially_relevant"
    NOT_RELEVANT = "not_relevant"


@dataclass
class CorpusSearchQuery:
    """Query for searching the knowledge corpus"""
    
    query_id: str
    query_text: str
    search_type: CorpusSearchType
    domain: Optional[str] = None
    
    # Search parameters
    max_results: int = 20
    min_relevance: ContentRelevance = ContentRelevance.MODERATELY_RELEVANT
    include_related: bool = True
    
    # Content filters
    content_categories: List[ContentCategory] = field(default_factory=list)
    min_quality_score: float = 0.7
    max_age_days: Optional[int] = None
    
    # SOC context
    context_socs: List[SOC] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CorpusSearchResult:
    """Result from corpus search"""
    
    content_cid: str
    addressed_content: AddressedContent
    relevance_score: float
    relevance_level: ContentRelevance
    
    # Content analysis
    generated_socs: List[SOC] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    domain_tags: List[str] = field(default_factory=list)
    
    # Verification status
    verification_result: Optional[VerificationResult] = None
    trust_score: float = 0.0
    
    # Reasoning integration
    analogical_mappings: List[str] = field(default_factory=list)
    causal_relationships: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CorpusQueryResult(BaseModel):
    """Complete result of corpus query"""
    
    query_id: str
    search_results: List[CorpusSearchResult] = Field(default_factory=list)
    
    # Query metrics
    total_results_found: int = 0
    search_time_seconds: float = 0.0
    verification_time_seconds: float = 0.0
    
    # Generated knowledge
    synthesized_socs: List[SOC] = Field(default_factory=list)
    domain_insights: List[str] = Field(default_factory=list)
    analogical_discoveries: List[str] = Field(default_factory=list)
    
    # Quality metrics
    average_relevance: float = 0.0
    average_trust_score: float = 0.0
    verification_success_rate: float = 0.0
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NWTNKnowledgeCorpusInterface:
    """
    Unified interface between NWTN and IPFS knowledge corpus
    
    This system enables NWTN to:
    1. Query the IPFS corpus for relevant content
    2. Verify content authenticity and provenance
    3. Generate SOCs from corpus content
    4. Integrate corpus knowledge into reasoning processes
    5. Maintain real-time corpus indexing
    """
    
    def __init__(self, 
                 content_addressing: ContentAddressingSystem,
                 content_verification: ContentVerificationSystem,
                 world_model: WorldModelEngine,
                 embedding_engine: SemanticEmbeddingEngine = None):
        
        self.content_addressing = content_addressing
        self.content_verification = content_verification
        self.world_model = world_model
        self.embedding_engine = embedding_engine
        
        # Corpus indexing
        self.content_index: Dict[str, Dict[str, Any]] = {}
        self.domain_index: Dict[str, List[str]] = {}
        self.concept_index: Dict[str, List[str]] = {}
        
        # SOC generation cache
        self.soc_cache: Dict[str, List[SOC]] = {}
        
        # Performance tracking
        self.stats = {
            'queries_processed': 0,
            'content_items_indexed': 0,
            'socs_generated': 0,
            'verifications_performed': 0,
            'cache_hits': 0
        }
        
        logger.info("NWTN Knowledge Corpus Interface initialized")
    
    async def query_corpus(self, query: CorpusSearchQuery) -> CorpusQueryResult:
        """
        Query the knowledge corpus for relevant content
        
        Args:
            query: Search query specification
            
        Returns:
            CorpusQueryResult with relevant content and generated knowledge
        """
        start_time = datetime.now()
        
        logger.info("Processing corpus query",
                   query_id=query.query_id,
                   search_type=query.search_type.value,
                   domain=query.domain)
        
        try:
            # Step 1: Search for relevant content
            search_results = await self._search_content(query)
            
            # Step 2: Verify content authenticity
            verified_results = await self._verify_search_results(search_results)
            
            # Step 3: Generate SOCs from content
            enriched_results = await self._enrich_with_socs(verified_results, query)
            
            # Step 4: Synthesize knowledge insights
            synthesized_knowledge = await self._synthesize_corpus_knowledge(enriched_results, query)
            
            # Step 5: Calculate metrics
            search_time = (datetime.now() - start_time).total_seconds()
            
            result = CorpusQueryResult(
                query_id=query.query_id,
                search_results=enriched_results,
                total_results_found=len(enriched_results),
                search_time_seconds=search_time,
                verification_time_seconds=sum(r.verification_result.verification_time 
                                            for r in enriched_results 
                                            if r.verification_result),
                synthesized_socs=synthesized_knowledge['socs'],
                domain_insights=synthesized_knowledge['insights'],
                analogical_discoveries=synthesized_knowledge['analogies'],
                average_relevance=sum(r.relevance_score for r in enriched_results) / max(1, len(enriched_results)),
                average_trust_score=sum(r.trust_score for r in enriched_results) / max(1, len(enriched_results)),
                verification_success_rate=sum(1 for r in enriched_results 
                                            if r.verification_result and r.verification_result.status.value == "verified") / max(1, len(enriched_results))
            )
            
            # Update statistics
            self.stats['queries_processed'] += 1
            
            logger.info("Corpus query completed",
                       query_id=query.query_id,
                       results_found=len(enriched_results),
                       search_time=search_time)
            
            return result
            
        except Exception as e:
            logger.error("Corpus query failed",
                        query_id=query.query_id,
                        error=str(e))
            raise
    
    async def _search_content(self, query: CorpusSearchQuery) -> List[CorpusSearchResult]:
        """Search for relevant content in the corpus"""
        
        # Use embedding engine for semantic search if available
        if self.embedding_engine and query.search_type == CorpusSearchType.SEMANTIC:
            return await self._semantic_search_content(query)
        
        # Fallback to traditional search
        return await self._traditional_search_content(query)
    
    async def _semantic_search_content(self, query: CorpusSearchQuery) -> List[CorpusSearchResult]:
        """Perform semantic search using embeddings"""
        
        # Create embedding search query
        embedding_query = EmbeddingSearchQuery(
            query_text=query.query_text,
            embedding_space=EmbeddingSpace.CONTENT_SEMANTIC,
            max_results=query.max_results,
            min_similarity=0.3,
            content_types=query.content_categories,
            domains=[query.domain] if query.domain else []
        )
        
        # Perform semantic search
        similarity_results = await self.embedding_engine.semantic_search(embedding_query)
        
        # Convert to corpus search results
        search_results = []
        for sim_result in similarity_results:
            try:
                # Get content from addressing system
                content_bytes, metadata = await self.content_addressing.get_content(sim_result.content_cid, include_metadata=True)
                
                if metadata:
                    relevance_level = self._classify_relevance(sim_result.similarity_score)
                    
                    # Filter by minimum relevance
                    if relevance_level.value in [r.value for r in ContentRelevance 
                                                if list(ContentRelevance).index(r) <= list(ContentRelevance).index(query.min_relevance)]:
                        
                        result = CorpusSearchResult(
                            content_cid=sim_result.content_cid,
                            addressed_content=metadata,
                            relevance_score=sim_result.similarity_score,
                            relevance_level=relevance_level
                        )
                        search_results.append(result)
                        
            except Exception as e:
                logger.warning("Failed to retrieve content for similarity result",
                             cid=sim_result.content_cid,
                             error=str(e))
        
        return search_results
    
    async def _traditional_search_content(self, query: CorpusSearchQuery) -> List[CorpusSearchResult]:
        """Traditional search using content addressing system"""
        
        # Use content addressing system's search capabilities
        search_kwargs = {
            'query': query.query_text,
            'keywords': query.query_text.split() if query.search_type == CorpusSearchType.KEYWORD else None
        }
        
        # Add domain filter if specified
        if query.domain:
            search_kwargs['keywords'] = (search_kwargs.get('keywords', []) + [query.domain])
        
        # Add category filters
        if query.content_categories:
            # Search each category separately and combine results
            all_content = []
            for category in query.content_categories:
                search_kwargs['category'] = category
                category_content = await self.content_addressing.search_content(**search_kwargs)
                all_content.extend(category_content)
        else:
            all_content = await self.content_addressing.search_content(**search_kwargs)
        
        # Convert to search results with relevance scoring
        search_results = []
        for content in all_content[:query.max_results]:
            relevance_score = await self._calculate_relevance(content, query)
            relevance_level = self._classify_relevance(relevance_score)
            
            # Filter by minimum relevance
            if relevance_level.value in [r.value for r in ContentRelevance 
                                        if list(ContentRelevance).index(r) <= list(ContentRelevance).index(query.min_relevance)]:
                
                result = CorpusSearchResult(
                    content_cid=content.cid,
                    addressed_content=content,
                    relevance_score=relevance_score,
                    relevance_level=relevance_level
                )
                search_results.append(result)
        
        return search_results
    
    async def _verify_search_results(self, search_results: List[CorpusSearchResult]) -> List[CorpusSearchResult]:
        """Verify authenticity of search results"""
        
        for result in search_results:
            try:
                # Verify content using verification system
                verification = await self.content_verification.verify_content(
                    cid=result.content_cid,
                    expected_checksum=result.addressed_content.checksum
                )
                
                result.verification_result = verification
                
                # Calculate trust score based on verification
                result.trust_score = await self._calculate_trust_score(verification, result.addressed_content)
                
                self.stats['verifications_performed'] += 1
                
            except Exception as e:
                logger.warning("Content verification failed",
                             cid=result.content_cid,
                             error=str(e))
                result.trust_score = 0.0
        
        return search_results
    
    async def _enrich_with_socs(self, search_results: List[CorpusSearchResult], query: CorpusSearchQuery) -> List[CorpusSearchResult]:
        """Enrich search results with generated SOCs"""
        
        for result in search_results:
            try:
                # Check cache first
                if result.content_cid in self.soc_cache:
                    result.generated_socs = self.soc_cache[result.content_cid]
                    self.stats['cache_hits'] += 1
                else:
                    # Generate SOCs from content
                    content_bytes, _ = await self.content_addressing.get_content(result.content_cid)
                    content_text = content_bytes.decode('utf-8') if isinstance(content_bytes, bytes) else str(content_bytes)
                    
                    socs = await self._generate_socs_from_content(content_text, result.addressed_content)
                    result.generated_socs = socs
                    
                    # Cache the results
                    self.soc_cache[result.content_cid] = socs
                    self.stats['socs_generated'] += len(socs)
                
                # Extract key concepts and domain tags
                result.key_concepts = await self._extract_key_concepts(result.generated_socs)
                result.domain_tags = await self._extract_domain_tags(result.addressed_content, result.generated_socs)
                
                # Generate analogical mappings if requested
                if query.search_type == CorpusSearchType.ANALOGICAL:
                    result.analogical_mappings = await self._generate_analogical_mappings(result.generated_socs, query.context_socs)
                
                # Generate causal relationships if requested
                if query.search_type == CorpusSearchType.CAUSAL:
                    result.causal_relationships = await self._generate_causal_relationships(result.generated_socs)
                
            except Exception as e:
                logger.warning("SOC enrichment failed",
                             cid=result.content_cid,
                             error=str(e))
        
        return search_results
    
    async def _synthesize_corpus_knowledge(self, search_results: List[CorpusSearchResult], query: CorpusSearchQuery) -> Dict[str, Any]:
        """Synthesize knowledge insights from search results"""
        
        # Collect all SOCs
        all_socs = []
        for result in search_results:
            all_socs.extend(result.generated_socs)
        
        # Generate synthesized SOCs (high-level concepts)
        synthesized_socs = await self._synthesize_socs(all_socs, query.domain)
        
        # Generate domain insights
        domain_insights = await self._generate_domain_insights(search_results, query.domain)
        
        # Generate analogical discoveries
        analogical_discoveries = await self._discover_analogies(search_results, query.context_socs)
        
        return {
            'socs': synthesized_socs,
            'insights': domain_insights,
            'analogies': analogical_discoveries
        }
    
    async def _calculate_relevance(self, content: AddressedContent, query: CorpusSearchQuery) -> float:
        """Calculate relevance score for content"""
        
        relevance_factors = []
        
        # Text similarity (simplified)
        query_lower = query.query_text.lower()
        title_lower = content.title.lower()
        desc_lower = content.description.lower()
        
        # Title match
        title_score = len(set(query_lower.split()) & set(title_lower.split())) / max(1, len(query_lower.split()))
        relevance_factors.append(title_score * 0.4)
        
        # Description match
        desc_score = len(set(query_lower.split()) & set(desc_lower.split())) / max(1, len(query_lower.split()))
        relevance_factors.append(desc_score * 0.3)
        
        # Keyword match
        keyword_score = len(set(query_lower.split()) & set([kw.lower() for kw in content.keywords])) / max(1, len(query_lower.split()))
        relevance_factors.append(keyword_score * 0.2)
        
        # Quality score
        relevance_factors.append(content.metrics.quality_score * 0.1)
        
        return sum(relevance_factors)
    
    def _classify_relevance(self, score: float) -> ContentRelevance:
        """Classify relevance score into levels"""
        if score >= 0.8:
            return ContentRelevance.HIGHLY_RELEVANT
        elif score >= 0.6:
            return ContentRelevance.MODERATELY_RELEVANT
        elif score >= 0.4:
            return ContentRelevance.TANGENTIALLY_RELEVANT
        else:
            return ContentRelevance.NOT_RELEVANT
    
    async def _calculate_trust_score(self, verification: VerificationResult, content: AddressedContent) -> float:
        """Calculate trust score based on verification and metadata"""
        
        trust_factors = []
        
        # Verification status
        if verification.status.value == "verified":
            trust_factors.append(0.5)
        elif verification.status.value == "failed":
            trust_factors.append(0.0)
        else:
            trust_factors.append(0.3)
        
        # Signature validity
        trust_factors.append(0.2 if verification.signature_valid else 0.0)
        
        # Content quality
        trust_factors.append(content.metrics.quality_score * 0.2)
        
        # Peer review score
        trust_factors.append(content.metrics.peer_review_score * 0.1)
        
        return sum(trust_factors)
    
    async def _generate_socs_from_content(self, content_text: str, metadata: AddressedContent) -> List[SOC]:
        """Generate SOCs from content text"""
        
        socs = []
        
        # Extract entities and concepts (simplified)
        words = content_text.lower().split()
        
        # Generate SOCs for key terms
        for keyword in metadata.keywords:
            soc = SOC(
                name=keyword,
                soc_type=SOCType.CONCEPT,
                confidence=0.8,
                confidence_level=ConfidenceLevel.INTERMEDIATE,
                relationships={},
                properties={
                    'source': 'corpus',
                    'cid': metadata.cid,
                    'domain': metadata.category.value
                },
                evidence_count=1,
                last_updated=datetime.now(timezone.utc),
                domain=metadata.category.value
            )
            socs.append(soc)
        
        # Generate SOCs for domain-specific concepts
        domain_concepts = await self._extract_domain_concepts(content_text, metadata.category.value)
        for concept in domain_concepts:
            soc = SOC(
                name=concept,
                soc_type=SOCType.CONCEPT,
                confidence=0.7,
                confidence_level=ConfidenceLevel.TENABLE,
                relationships={},
                properties={
                    'source': 'corpus_extracted',
                    'cid': metadata.cid,
                    'domain': metadata.category.value
                },
                evidence_count=1,
                last_updated=datetime.now(timezone.utc),
                domain=metadata.category.value
            )
            socs.append(soc)
        
        return socs
    
    async def _extract_domain_concepts(self, content_text: str, domain: str) -> List[str]:
        """Extract domain-specific concepts from content"""
        
        # Domain-specific concept patterns
        domain_patterns = {
            'research_paper': ['hypothesis', 'methodology', 'results', 'conclusion', 'analysis'],
            'dataset': ['variables', 'measurements', 'observations', 'samples', 'features'],
            'code_repository': ['functions', 'classes', 'modules', 'algorithms', 'implementation'],
            'protocol': ['procedure', 'steps', 'requirements', 'specifications', 'guidelines']
        }
        
        patterns = domain_patterns.get(domain, [])
        content_lower = content_text.lower()
        
        found_concepts = []
        for pattern in patterns:
            if pattern in content_lower:
                found_concepts.append(pattern)
        
        return found_concepts
    
    async def _extract_key_concepts(self, socs: List[SOC]) -> List[str]:
        """Extract key concepts from SOCs"""
        return [soc.name for soc in socs if soc.confidence > 0.7]
    
    async def _extract_domain_tags(self, content: AddressedContent, socs: List[SOC]) -> List[str]:
        """Extract domain tags"""
        tags = [content.category.value]
        tags.extend(content.tags)
        
        # Add SOC domains
        for soc in socs:
            if soc.domain and soc.domain not in tags:
                tags.append(soc.domain)
        
        return list(set(tags))
    
    async def _generate_analogical_mappings(self, content_socs: List[SOC], context_socs: List[SOC]) -> List[str]:
        """Generate analogical mappings between content and context SOCs"""
        
        mappings = []
        
        for content_soc in content_socs:
            for context_soc in context_socs:
                # Simple analogical mapping based on name similarity
                if content_soc.name != context_soc.name:
                    # Check for conceptual similarity (simplified)
                    content_words = set(content_soc.name.lower().split())
                    context_words = set(context_soc.name.lower().split())
                    
                    overlap = len(content_words & context_words)
                    if overlap > 0:
                        mapping = f"{content_soc.name} ↔ {context_soc.name} (overlap: {overlap})"
                        mappings.append(mapping)
        
        return mappings
    
    async def _generate_causal_relationships(self, socs: List[SOC]) -> List[str]:
        """Generate causal relationships between SOCs"""
        
        relationships = []
        
        # Simple causal pattern detection
        causal_indicators = ['causes', 'results in', 'leads to', 'produces', 'affects']
        
        for i, soc1 in enumerate(socs):
            for j, soc2 in enumerate(socs[i+1:], i+1):
                # Check properties for causal indicators
                props1 = str(soc1.properties)
                props2 = str(soc2.properties)
                
                for indicator in causal_indicators:
                    if indicator in props1.lower() or indicator in props2.lower():
                        relationship = f"{soc1.name} → {soc2.name} ({indicator})"
                        relationships.append(relationship)
                        break
        
        return relationships
    
    async def _synthesize_socs(self, all_socs: List[SOC], domain: Optional[str]) -> List[SOC]:
        """Synthesize high-level SOCs from multiple sources"""
        
        # Group SOCs by name
        soc_groups = {}
        for soc in all_socs:
            if soc.name not in soc_groups:
                soc_groups[soc.name] = []
            soc_groups[soc.name].append(soc)
        
        synthesized = []
        
        for name, group in soc_groups.items():
            if len(group) > 1:  # Only synthesize if multiple sources
                # Calculate synthesized confidence
                avg_confidence = sum(soc.confidence for soc in group) / len(group)
                
                # Determine highest confidence level
                levels = [soc.confidence_level for soc in group]
                highest_level = max(levels, key=lambda x: list(ConfidenceLevel).index(x))
                
                # Merge properties
                merged_properties = {}
                for soc in group:
                    merged_properties.update(soc.properties)
                merged_properties['synthesis_source_count'] = len(group)
                
                synthesized_soc = SOC(
                    name=name,
                    soc_type=group[0].soc_type,
                    confidence=min(1.0, avg_confidence * 1.1),  # Boost for multiple sources
                    confidence_level=highest_level,
                    relationships={},
                    properties=merged_properties,
                    evidence_count=sum(soc.evidence_count for soc in group),
                    last_updated=datetime.now(timezone.utc),
                    domain=domain or group[0].domain
                )
                synthesized.append(synthesized_soc)
        
        return synthesized
    
    async def _generate_domain_insights(self, search_results: List[CorpusSearchResult], domain: Optional[str]) -> List[str]:
        """Generate domain-specific insights from search results"""
        
        insights = []
        
        # Collect all concepts
        all_concepts = []
        for result in search_results:
            all_concepts.extend(result.key_concepts)
        
        # Find most common concepts
        concept_counts = {}
        for concept in all_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        if concept_counts:
            most_common = max(concept_counts.keys(), key=lambda k: concept_counts[k])
            insights.append(f"Most prevalent concept in {domain or 'corpus'}: {most_common} (appears {concept_counts[most_common]} times)")
        
        # Quality insights
        high_quality_results = [r for r in search_results if r.trust_score > 0.8]
        if high_quality_results:
            insights.append(f"Found {len(high_quality_results)} high-trust sources with average relevance {sum(r.relevance_score for r in high_quality_results) / len(high_quality_results):.2f}")
        
        return insights
    
    async def _discover_analogies(self, search_results: List[CorpusSearchResult], context_socs: List[SOC]) -> List[str]:
        """Discover analogical patterns across search results"""
        
        analogies = []
        
        # Find patterns across multiple results
        common_mappings = {}
        for result in search_results:
            for mapping in result.analogical_mappings:
                common_mappings[mapping] = common_mappings.get(mapping, 0) + 1
        
        # Identify recurring analogical patterns
        for mapping, count in common_mappings.items():
            if count > 1:
                analogies.append(f"Recurring pattern: {mapping} (found {count} times)")
        
        return analogies
    
    async def index_new_content(self, content_cid: str):
        """Index newly added content for future searches"""
        
        try:
            # Get content metadata
            _, metadata = await self.content_addressing.get_content(content_cid, include_metadata=True)
            
            if metadata:
                # Add to content index
                self.content_index[content_cid] = {
                    'title': metadata.title,
                    'description': metadata.description,
                    'keywords': metadata.keywords,
                    'category': metadata.category.value,
                    'created_at': metadata.created_at,
                    'quality_score': metadata.metrics.quality_score
                }
                
                # Add to domain index
                domain = metadata.category.value
                if domain not in self.domain_index:
                    self.domain_index[domain] = []
                if content_cid not in self.domain_index[domain]:
                    self.domain_index[domain].append(content_cid)
                
                # Add to concept index
                for keyword in metadata.keywords:
                    if keyword not in self.concept_index:
                        self.concept_index[keyword] = []
                    if content_cid not in self.concept_index[keyword]:
                        self.concept_index[keyword].append(content_cid)
                
                self.stats['content_items_indexed'] += 1
                
                logger.debug("Content indexed",
                           cid=content_cid,
                           domain=domain,
                           keywords=len(metadata.keywords))
            
        except Exception as e:
            logger.error("Content indexing failed",
                        cid=content_cid,
                        error=str(e))
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get corpus interface statistics"""
        
        return {
            'interface_stats': self.stats.copy(),
            'indexed_content': len(self.content_index),
            'domains': len(self.domain_index),
            'concepts': len(self.concept_index),
            'cached_socs': len(self.soc_cache),
            'domain_distribution': {domain: len(cids) for domain, cids in self.domain_index.items()}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on corpus interface"""
        
        try:
            # Check underlying systems
            addressing_health = await self.content_addressing.health_check()
            verification_health = await self.content_verification.health_check()
            
            # Test query functionality
            test_query = CorpusSearchQuery(
                query_id="health_check",
                query_text="test",
                search_type=CorpusSearchType.KEYWORD,
                max_results=1
            )
            
            try:
                await self.query_corpus(test_query)
                query_healthy = True
            except Exception:
                query_healthy = False
            
            return {
                'healthy': (addressing_health['healthy'] and 
                           verification_health['healthy'] and 
                           query_healthy),
                'addressing_health': addressing_health,
                'verification_health': verification_health,
                'query_functional': query_healthy,
                'stats': self.get_corpus_stats()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'stats': self.get_corpus_stats()
            }


# Utility functions

def create_corpus_interface(content_addressing: ContentAddressingSystem,
                          content_verification: ContentVerificationSystem,
                          world_model: WorldModelEngine,
                          embedding_engine: SemanticEmbeddingEngine = None) -> NWTNKnowledgeCorpusInterface:
    """Create a new NWTN knowledge corpus interface"""
    return NWTNKnowledgeCorpusInterface(content_addressing, content_verification, world_model, embedding_engine)


async def query_corpus_for_socs(interface: NWTNKnowledgeCorpusInterface,
                               query_text: str,
                               domain: str = None,
                               context_socs: List[SOC] = None) -> List[SOC]:
    """Quick utility to query corpus and get SOCs"""
    
    query = CorpusSearchQuery(
        query_id=str(uuid4()),
        query_text=query_text,
        search_type=CorpusSearchType.SEMANTIC,
        domain=domain,
        context_socs=context_socs or []
    )
    
    result = await interface.query_corpus(query)
    
    # Return all generated SOCs
    all_socs = []
    for search_result in result.search_results:
        all_socs.extend(search_result.generated_socs)
    all_socs.extend(result.synthesized_socs)
    
    return all_socs