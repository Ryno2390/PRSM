#!/usr/bin/env python3
"""
PRSM Unified Knowledge System
Complete integration of IPFS content corpus with NWTN reasoning

This module provides the unified interface that connects all knowledge
management components: IPFS storage, cryptographic verification,
public source ingestion, and NWTN reasoning capabilities.

Key Features:
1. Unified knowledge corpus with IPFS backend
2. Automated public source content ingestion
3. NWTN integration for hybrid reasoning
4. Cryptographic content verification
5. Cross-domain knowledge transfer
6. Real-time corpus indexing and search
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

import structlog
from pydantic import BaseModel

# IPFS and content systems
from .ipfs.ipfs_client import IPFSClient
from .ipfs.content_addressing import ContentAddressingSystem, ContentCategory
from .ipfs.content_verification import ContentVerificationSystem

# NWTN systems
from .nwtn.hybrid_architecture import HybridNWTNEngine, SOC
from .nwtn.knowledge_corpus_interface import (
    NWTNKnowledgeCorpusInterface, CorpusSearchQuery, CorpusQueryResult, CorpusSearchType
)
from .nwtn.world_model_engine import WorldModelEngine

# Ingestion and federation
from .ingestion.public_source_porter import (
    PublicSourcePorter, ContentSource, SourceType, configure_default_sources
)
from .federation.knowledge_transfer import CrossDomainKnowledgeTransferSystem
from .distillation.knowledge_extractor import KnowledgeExtractor
from .embeddings.semantic_embedding_engine import SemanticEmbeddingEngine, EmbeddingSearchQuery

logger = structlog.get_logger(__name__)


class KnowledgeSystemMode(str, Enum):
    """Operating modes for the knowledge system"""
    RESEARCH = "research"           # Focus on research and discovery
    PRODUCTION = "production"       # Optimized for production workloads
    INGESTION = "ingestion"        # Focus on content ingestion
    REASONING = "reasoning"        # Focus on NWTN reasoning
    FULL = "full"                  # All capabilities enabled


@dataclass
class KnowledgeSystemConfig:
    """Configuration for the unified knowledge system"""
    
    # System mode
    mode: KnowledgeSystemMode = KnowledgeSystemMode.FULL
    
    # IPFS configuration
    ipfs_gateway_url: str = "http://localhost:8080"
    ipfs_api_url: str = "http://localhost:5001"
    
    # Content processing
    enable_auto_ingestion: bool = True
    ingestion_interval_hours: int = 24
    max_concurrent_ingestions: int = 10
    
    # NWTN integration
    enable_hybrid_reasoning: bool = True
    enable_analogical_discovery: bool = True
    enable_meaning_grounding: bool = True
    
    # Performance tuning
    corpus_cache_size: int = 10000
    soc_cache_size: int = 50000
    verification_cache_ttl: int = 3600  # seconds
    
    # Quality thresholds
    min_content_quality: float = 0.7
    min_verification_trust: float = 0.8
    min_relevance_score: float = 0.6


class UnifiedKnowledgeSystem:
    """
    Unified Knowledge Management System for PRSM
    
    This system integrates all knowledge-related capabilities:
    - IPFS-based distributed storage
    - Cryptographic content verification
    - Automated public source ingestion
    - NWTN hybrid reasoning integration
    - Cross-domain knowledge transfer
    """
    
    def __init__(self, config: KnowledgeSystemConfig = None):
        self.config = config or KnowledgeSystemConfig()
        
        # Core systems (initialized in setup)
        self.ipfs_client: Optional[IPFSClient] = None
        self.content_addressing: Optional[ContentAddressingSystem] = None
        self.content_verification: Optional[ContentVerificationSystem] = None
        self.world_model: Optional[WorldModelEngine] = None
        self.nwtn_engine: Optional[HybridNWTNEngine] = None
        self.corpus_interface: Optional[NWTNKnowledgeCorpusInterface] = None
        self.source_porter: Optional[PublicSourcePorter] = None
        self.knowledge_transfer: Optional[CrossDomainKnowledgeTransferSystem] = None
        self.knowledge_extractor: Optional[KnowledgeExtractor] = None
        self.embedding_engine: Optional[SemanticEmbeddingEngine] = None
        
        # System state
        self.initialized = False
        self.auto_ingestion_running = False
        
        # Performance tracking
        self.stats = {
            'system_start_time': datetime.now(timezone.utc),
            'queries_processed': 0,
            'content_items_indexed': 0,
            'reasoning_sessions': 0,
            'ingestion_runs': 0,
            'cache_hits': 0,
            'total_response_time': 0.0
        }
        
        logger.info("Unified Knowledge System created", mode=self.config.mode.value)
    
    async def initialize(self):
        """Initialize all knowledge system components"""
        
        if self.initialized:
            logger.warning("Knowledge system already initialized")
            return
        
        logger.info("Initializing unified knowledge system",
                   mode=self.config.mode.value)
        
        try:
            # Step 1: Initialize IPFS client
            self.ipfs_client = IPFSClient(
                api_url=self.config.ipfs_api_url,
                gateway_url=self.config.ipfs_gateway_url
            )
            
            # Test IPFS connectivity
            ipfs_healthy = await self.ipfs_client.health_check()
            if not ipfs_healthy['healthy']:
                raise RuntimeError("IPFS client health check failed")
            
            # Step 2: Initialize content systems
            self.content_addressing = ContentAddressingSystem(self.ipfs_client)
            self.content_verification = ContentVerificationSystem(self.ipfs_client)
            
            # Step 3: Initialize NWTN components (if enabled)
            if self.config.mode in [KnowledgeSystemMode.REASONING, KnowledgeSystemMode.FULL]:
                self.world_model = WorldModelEngine()
                self.nwtn_engine = HybridNWTNEngine(agent_id="unified_knowledge_system")
                self.embedding_engine = SemanticEmbeddingEngine()
                await self.embedding_engine.initialize()
                self.corpus_interface = NWTNKnowledgeCorpusInterface(
                    self.content_addressing,
                    self.content_verification,
                    self.world_model,
                    self.embedding_engine
                )
                self.knowledge_extractor = KnowledgeExtractor()
            
            # Step 4: Initialize ingestion system (if enabled)
            if self.config.mode in [KnowledgeSystemMode.INGESTION, KnowledgeSystemMode.FULL]:
                if self.corpus_interface:
                    self.source_porter = PublicSourcePorter(
                        self.content_addressing,
                        self.content_verification,
                        self.corpus_interface
                    )
                    
                    # Configure default sources
                    await configure_default_sources(self.source_porter)
            
            # Step 5: Initialize federation system
            if self.config.mode == KnowledgeSystemMode.FULL:
                # Note: Would need federation system here in full implementation
                pass
            
            # Step 6: Start auto-ingestion if enabled
            if self.config.enable_auto_ingestion and self.source_porter:
                asyncio.create_task(self._auto_ingestion_loop())
            
            self.initialized = True
            
            logger.info("Unified knowledge system initialized successfully",
                       components_active=self._get_active_components())
            
        except Exception as e:
            logger.error("Knowledge system initialization failed", error=str(e))
            raise
    
    async def query_knowledge(self,
                            query: str,
                            domain: str = None,
                            search_type: CorpusSearchType = CorpusSearchType.SEMANTIC,
                            enable_reasoning: bool = True,
                            max_results: int = 20) -> Dict[str, Any]:
        """
        Query the unified knowledge system
        
        Args:
            query: Natural language query
            domain: Optional domain filter
            search_type: Type of search to perform
            enable_reasoning: Whether to use NWTN reasoning
            max_results: Maximum results to return
            
        Returns:
            Comprehensive query result with content and reasoning
        """
        
        if not self.initialized:
            raise RuntimeError("Knowledge system not initialized")
        
        start_time = datetime.now()
        
        logger.info("Processing knowledge query",
                   query=query[:100],
                   domain=domain,
                   search_type=search_type.value,
                   enable_reasoning=enable_reasoning)
        
        try:
            result = {
                'query': query,
                'domain': domain,
                'timestamp': start_time,
                'corpus_results': None,
                'reasoning_results': None,
                'synthesized_knowledge': None,
                'performance_metrics': {}
            }
            
            # Step 1: Query corpus if available
            if self.corpus_interface:
                corpus_query = CorpusSearchQuery(
                    query_id=f"unified_{int(start_time.timestamp())}",
                    query_text=query,
                    search_type=search_type,
                    domain=domain,
                    max_results=max_results
                )
                
                corpus_result = await self.corpus_interface.query_corpus(corpus_query)
                result['corpus_results'] = corpus_result
                
                # Extract SOCs for reasoning
                context_socs = []
                for search_result in corpus_result.search_results:
                    context_socs.extend(search_result.generated_socs)
                context_socs.extend(corpus_result.synthesized_socs)
                
            else:
                context_socs = []
            
            # Step 2: Apply NWTN reasoning if enabled and available
            if enable_reasoning and self.nwtn_engine:
                reasoning_context = {
                    'domain': domain,
                    'corpus_socs': context_socs,
                    'search_results': result.get('corpus_results', {}).get('search_results', [])
                }
                
                reasoning_result = await self.nwtn_engine.process_query(query, reasoning_context)
                result['reasoning_results'] = reasoning_result
                
                # Update stats
                self.stats['reasoning_sessions'] += 1
            
            # Step 3: Synthesize knowledge from all sources
            synthesized = await self._synthesize_knowledge(result)
            result['synthesized_knowledge'] = synthesized
            
            # Step 4: Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            result['performance_metrics'] = {
                'total_processing_time': processing_time,
                'corpus_search_time': result.get('corpus_results', {}).get('search_time_seconds', 0),
                'reasoning_time': 0,  # Would extract from reasoning results
                'synthesis_time': 0   # Would measure synthesis time
            }
            
            # Update system stats
            self.stats['queries_processed'] += 1
            self.stats['total_response_time'] += processing_time
            
            logger.info("Knowledge query completed",
                       query=query[:50],
                       processing_time=processing_time,
                       corpus_results=len(result.get('corpus_results', {}).get('search_results', [])),
                       reasoning_enabled=enable_reasoning)
            
            return result
            
        except Exception as e:
            logger.error("Knowledge query failed",
                        query=query[:100],
                        error=str(e))
            raise
    
    async def ingest_content(self,
                           content: str,
                           title: str,
                           description: str,
                           category: ContentCategory,
                           source_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Manually ingest content into the system
        
        Args:
            content: Content text
            title: Content title
            description: Content description
            category: Content category
            source_info: Optional source metadata
            
        Returns:
            Ingestion result with CID and metadata
        """
        
        if not self.initialized or not self.content_addressing:
            raise RuntimeError("Content addressing system not available")
        
        logger.info("Manual content ingestion",
                   title=title[:50],
                   category=category.value)
        
        try:
            # Create provenance and license
            from .ipfs.content_addressing import create_basic_provenance, create_open_license
            
            provenance = create_basic_provenance(
                creator_id=source_info.get('creator_id', 'manual_upload'),
                creator_name=source_info.get('creator_name', 'Manual Upload'),
                institution=source_info.get('institution', 'PRSM User')
            )
            
            license_obj = create_open_license()
            
            # Add to IPFS
            addressed_content = await self.content_addressing.add_content(
                content=content,
                title=title,
                description=description,
                content_type="text/plain",
                category=category,
                provenance=provenance,
                license=license_obj,
                keywords=source_info.get('keywords', []),
                tags=source_info.get('tags', [])
            )
            
            # Generate embeddings if available
            if self.embedding_engine:
                await self.embedding_engine.embed_content(addressed_content, content_text=content)
            
            # Index in corpus interface if available
            if self.corpus_interface:
                await self.corpus_interface.index_new_content(addressed_content.cid)
            
            # Update stats
            self.stats['content_items_indexed'] += 1
            
            result = {
                'success': True,
                'cid': addressed_content.cid,
                'addressed_content': addressed_content,
                'indexed_in_corpus': self.corpus_interface is not None
            }
            
            logger.info("Content ingestion completed",
                       cid=addressed_content.cid,
                       title=title[:50])
            
            return result
            
        except Exception as e:
            logger.error("Manual content ingestion failed",
                        title=title[:50],
                        error=str(e))
            raise
    
    async def discover_and_ingest_sources(self, source_ids: List[str] = None) -> Dict[str, Any]:
        """
        Discover and ingest content from configured sources
        
        Args:
            source_ids: Optional list of specific source IDs to process
            
        Returns:
            Ingestion summary with results and statistics
        """
        
        if not self.source_porter:
            raise RuntimeError("Source porter not available")
        
        logger.info("Starting source discovery and ingestion",
                   source_ids=source_ids)
        
        try:
            # Get active sources
            sources_to_process = source_ids or self.source_porter.active_sources
            
            ingestion_results = {}
            total_discovered = 0
            total_ingested = 0
            
            for source_id in sources_to_process:
                # Discover content
                candidates = await self.source_porter.discover_content(source_id, max_items=50)
                total_discovered += len(candidates)
                
                # Batch ingest
                results = await self.source_porter.batch_ingest(source_id)
                successful_ingestions = sum(1 for r in results if r.ingestion_successful)
                total_ingested += successful_ingestions
                
                ingestion_results[source_id] = {
                    'candidates_discovered': len(candidates),
                    'successful_ingestions': successful_ingestions,
                    'failed_ingestions': len(results) - successful_ingestions,
                    'results': results
                }
            
            # Update stats
            self.stats['ingestion_runs'] += 1
            self.stats['content_items_indexed'] += total_ingested
            
            summary = {
                'ingestion_successful': True,
                'sources_processed': len(sources_to_process),
                'total_candidates_discovered': total_discovered,
                'total_content_ingested': total_ingested,
                'source_results': ingestion_results,
                'timestamp': datetime.now(timezone.utc)
            }
            
            logger.info("Source ingestion completed",
                       sources_processed=len(sources_to_process),
                       total_ingested=total_ingested)
            
            return summary
            
        except Exception as e:
            logger.error("Source ingestion failed", error=str(e))
            raise
    
    async def _synthesize_knowledge(self, query_result: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize knowledge from corpus and reasoning results"""
        
        synthesis = {
            'key_concepts': [],
            'domain_insights': [],
            'analogical_discoveries': [],
            'confidence_score': 0.0,
            'knowledge_sources': []
        }
        
        # Extract from corpus results
        corpus_results = query_result.get('corpus_results')
        if corpus_results:
            synthesis['key_concepts'].extend(corpus_results.domain_insights)
            synthesis['analogical_discoveries'].extend(corpus_results.analogical_discoveries)
            synthesis['knowledge_sources'].extend([
                f"Corpus: {len(corpus_results.search_results)} sources"
            ])
        
        # Extract from reasoning results
        reasoning_results = query_result.get('reasoning_results')
        if reasoning_results:
            # Extract insights from reasoning
            synthesis['domain_insights'].append("NWTN hybrid reasoning applied")
            synthesis['knowledge_sources'].append("NWTN hybrid reasoning engine")
        
        # Calculate confidence based on available sources
        source_count = len(synthesis['knowledge_sources'])
        synthesis['confidence_score'] = min(1.0, source_count * 0.3)
        
        return synthesis
    
    async def _auto_ingestion_loop(self):
        """Background loop for automatic content ingestion"""
        
        self.auto_ingestion_running = True
        
        logger.info("Starting auto-ingestion loop",
                   interval_hours=self.config.ingestion_interval_hours)
        
        while self.auto_ingestion_running:
            try:
                # Wait for the specified interval
                await asyncio.sleep(self.config.ingestion_interval_hours * 3600)
                
                if not self.auto_ingestion_running:
                    break
                
                # Run ingestion
                logger.info("Running scheduled content ingestion")
                await self.discover_and_ingest_sources()
                
            except Exception as e:
                logger.error("Auto-ingestion loop error", error=str(e))
                # Continue running even if one iteration fails
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    def _get_active_components(self) -> List[str]:
        """Get list of active system components"""
        
        components = []
        
        if self.ipfs_client:
            components.append("IPFS Client")
        if self.content_addressing:
            components.append("Content Addressing")
        if self.content_verification:
            components.append("Content Verification")
        if self.nwtn_engine:
            components.append("NWTN Engine")
        if self.corpus_interface:
            components.append("Corpus Interface")
        if self.source_porter:
            components.append("Source Porter")
        if self.knowledge_transfer:
            components.append("Knowledge Transfer")
        if self.embedding_engine:
            components.append("Embedding Engine")
        
        return components
    
    async def shutdown(self):
        """Graceful shutdown of the knowledge system"""
        
        logger.info("Shutting down unified knowledge system")
        
        # Stop auto-ingestion
        self.auto_ingestion_running = False
        
        # Close connections and cleanup
        if self.ipfs_client:
            # IPFS client cleanup would go here
            pass
        
        self.initialized = False
        
        logger.info("Knowledge system shutdown completed")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        stats = {
            'system_stats': self.stats.copy(),
            'config': {
                'mode': self.config.mode.value,
                'auto_ingestion_enabled': self.config.enable_auto_ingestion,
                'hybrid_reasoning_enabled': self.config.enable_hybrid_reasoning
            },
            'component_stats': {},
            'active_components': self._get_active_components(),
            'system_uptime_seconds': (datetime.now(timezone.utc) - self.stats['system_start_time']).total_seconds()
        }
        
        # Add component-specific stats
        if self.content_addressing:
            stats['component_stats']['content_addressing'] = self.content_addressing.get_stats()
        
        if self.content_verification:
            stats['component_stats']['content_verification'] = self.content_verification.get_verification_stats()
        
        if self.corpus_interface:
            stats['component_stats']['corpus_interface'] = self.corpus_interface.get_corpus_stats()
        
        if self.source_porter:
            stats['component_stats']['source_porter'] = self.source_porter.get_porter_stats()
        
        if self.embedding_engine:
            stats['component_stats']['embedding_engine'] = self.embedding_engine.get_embedding_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all components"""
        
        health = {
            'overall_healthy': True,
            'component_health': {},
            'system_initialized': self.initialized,
            'auto_ingestion_running': self.auto_ingestion_running
        }
        
        if not self.initialized:
            health['overall_healthy'] = False
            health['error'] = "System not initialized"
            return health
        
        # Check each component
        try:
            if self.ipfs_client:
                ipfs_health = await self.ipfs_client.health_check()
                health['component_health']['ipfs'] = ipfs_health
                if not ipfs_health['healthy']:
                    health['overall_healthy'] = False
            
            if self.content_addressing:
                addressing_health = await self.content_addressing.health_check()
                health['component_health']['content_addressing'] = addressing_health
                if not addressing_health['healthy']:
                    health['overall_healthy'] = False
            
            if self.content_verification:
                verification_health = await self.content_verification.health_check()
                health['component_health']['content_verification'] = verification_health
                if not verification_health['healthy']:
                    health['overall_healthy'] = False
            
            if self.corpus_interface:
                corpus_health = await self.corpus_interface.health_check()
                health['component_health']['corpus_interface'] = corpus_health
                if not corpus_health['healthy']:
                    health['overall_healthy'] = False
            
            if self.source_porter:
                porter_health = await self.source_porter.health_check()
                health['component_health']['source_porter'] = porter_health
                if not porter_health['healthy']:
                    health['overall_healthy'] = False
            
            if self.embedding_engine:
                embedding_health = await self.embedding_engine.health_check()
                health['component_health']['embedding_engine'] = embedding_health
                if not embedding_health['healthy']:
                    health['overall_healthy'] = False
            
        except Exception as e:
            health['overall_healthy'] = False
            health['error'] = str(e)
        
        health['stats'] = self.get_system_stats()
        
        return health


# Utility functions

async def create_knowledge_system(config: KnowledgeSystemConfig = None) -> UnifiedKnowledgeSystem:
    """Create and initialize a unified knowledge system"""
    
    system = UnifiedKnowledgeSystem(config)
    await system.initialize()
    return system


async def quick_knowledge_query(system: UnifiedKnowledgeSystem, 
                              query: str, 
                              domain: str = None) -> Dict[str, Any]:
    """Quick utility for knowledge queries"""
    
    return await system.query_knowledge(
        query=query,
        domain=domain,
        search_type=CorpusSearchType.SEMANTIC,
        enable_reasoning=True
    )