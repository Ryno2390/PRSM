"""
PRSM Complete Embedding Pipeline

Integrated pipeline combining content processing, embedding generation,
caching, and vector storage for production-ready PRSM deployment.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..content_processing.text_processor import (
    ContentTextProcessor, ProcessedContent, ProcessingConfig, ContentType
)
from .embedding_cache import EmbeddingCache, create_optimized_cache
from .real_embedding_api import RealEmbeddingAPI, get_embedding_api
from ..vector_store.base import PRSMVectorStore

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingPipelineConfig:
    """Configuration for the complete embedding pipeline"""
    
    # Content processing configuration
    max_chunk_size: int = 512
    chunk_overlap: int = 50
    preserve_structure: bool = True
    extract_metadata: bool = True
    
    # Embedding configuration
    preferred_embedding_provider: str = 'openai'
    embedding_model: str = 'text-embedding-ada-002'
    batch_size: int = 100
    
    # Cache configuration
    cache_dir: str = "prsm_embedding_cache"
    max_cache_size_mb: int = 500
    cache_max_age_days: int = 30
    
    # Performance configuration
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0
    retry_attempts: int = 3
    
    # Storage configuration
    store_intermediate_results: bool = True
    enable_deduplication: bool = True


@dataclass
class PipelineResult:
    """Result from the embedding pipeline"""
    content_id: str
    processed_content: ProcessedContent
    embeddings: List[np.ndarray]
    storage_ids: List[str]
    processing_stats: Dict[str, Any]
    total_processing_time: float
    success: bool
    errors: List[str] = None


class EmbeddingPipeline:
    """
    Complete embedding pipeline for PRSM content processing
    
    Features:
    - End-to-end content processing from raw text to vector storage
    - Intelligent caching and batch processing
    - Multiple embedding provider support with fallbacks
    - Performance monitoring and optimization
    - Error handling and retry logic
    - Integration with PRSM vector stores
    """
    
    def __init__(self, 
                 vector_store: PRSMVectorStore,
                 config: EmbeddingPipelineConfig = None):
        
        self.vector_store = vector_store
        self.config = config or EmbeddingPipelineConfig()
        
        # Initialize components
        self.text_processor = None
        self.embedding_cache = None
        self.embedding_api = None
        
        # Performance tracking
        self.total_processed = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        # Semaphore for controlling concurrent requests
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
    
    async def initialize(self):
        """Initialize pipeline components"""
        logger.info("Initializing PRSM Embedding Pipeline...")
        
        # Initialize embedding cache
        self.embedding_cache = await create_optimized_cache(
            cache_dir=self.config.cache_dir,
            max_size_mb=self.config.max_cache_size_mb
        )
        
        # Initialize embedding API
        self.embedding_api = get_embedding_api()
        
        # Test embedding providers
        provider_tests = await self.embedding_api.test_all_providers()
        working_providers = [
            name for name, result in provider_tests.items() 
            if result.get('success', False)
        ]
        
        if not working_providers:
            logger.warning("No working embedding providers found - using mock only")
        else:
            logger.info(f"Working embedding providers: {working_providers}")
        
        logger.info("Embedding Pipeline initialized successfully")
    
    async def process_content(self, 
                            text: str, 
                            content_id: str,
                            content_type: ContentType = ContentType.TEXT,
                            metadata: Dict[str, Any] = None) -> PipelineResult:
        """Process single content through complete pipeline"""
        
        start_time = time.time()
        errors = []
        
        try:
            # Step 1: Text processing and chunking
            processing_config = ProcessingConfig(
                max_chunk_size=self.config.max_chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                preserve_structure=self.config.preserve_structure,
                extract_metadata=self.config.extract_metadata,
                content_type=content_type
            )
            
            self.text_processor = ContentTextProcessor(processing_config)
            processed_content = self.text_processor.process_content(
                text, content_id, metadata or {}
            )
            
            logger.info(f"Processed content {content_id} into {len(processed_content.processed_chunks)} chunks")
            
            # Step 2: Generate embeddings with caching
            chunk_texts = [chunk.text for chunk in processed_content.processed_chunks]
            chunk_metadata = [chunk.metadata for chunk in processed_content.processed_chunks]
            
            embeddings = await self._generate_embeddings_with_cache(
                chunk_texts, chunk_metadata
            )
            
            # Step 3: Store in vector database
            storage_ids = await self._store_embeddings(
                processed_content, embeddings
            )
            
            # Step 4: Compile results
            total_time = time.time() - start_time
            
            processing_stats = {
                'chunks_processed': len(processed_content.processed_chunks),
                'embeddings_generated': len(embeddings),
                'cache_hit_rate': self.embedding_cache.stats.cache_hits / max(1, 
                    self.embedding_cache.stats.cache_hits + self.embedding_cache.stats.cache_misses),
                'total_processing_time': total_time,
                'content_processing_time': processed_content.processing_stats['processing_time'],
                'embedding_generation_time': total_time - processed_content.processing_stats['processing_time'],
                'average_chunk_size': processed_content.processing_stats['average_chunk_size'],
                'embedding_provider': self.config.preferred_embedding_provider,
            }
            
            # Update global stats
            self.total_processed += 1
            self.total_processing_time += total_time
            
            return PipelineResult(
                content_id=content_id,
                processed_content=processed_content,
                embeddings=embeddings,
                storage_ids=storage_ids,
                processing_stats=processing_stats,
                total_processing_time=total_time,
                success=True,
                errors=errors if errors else None
            )
            
        except Exception as e:
            self.error_count += 1
            error_msg = f"Pipeline failed for content {content_id}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return PipelineResult(
                content_id=content_id,
                processed_content=None,
                embeddings=[],
                storage_ids=[],
                processing_stats={},
                total_processing_time=time.time() - start_time,
                success=False,
                errors=errors
            )
    
    async def process_content_batch(self, 
                                  content_items: List[Tuple[str, str, ContentType, Dict[str, Any]]],
                                  max_concurrent: int = None) -> List[PipelineResult]:
        """Process multiple content items concurrently"""
        
        if max_concurrent:
            semaphore = asyncio.Semaphore(max_concurrent)
        else:
            semaphore = self.semaphore
        
        async def process_single(item):
            async with semaphore:
                text, content_id, content_type, metadata = item
                return await self.process_content(text, content_id, content_type, metadata)
        
        logger.info(f"Processing batch of {len(content_items)} content items")
        
        # Process all items concurrently
        tasks = [process_single(item) for item in content_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {result}")
                content_id = content_items[i][1] if len(content_items[i]) > 1 else f"item_{i}"
                processed_results.append(PipelineResult(
                    content_id=content_id,
                    processed_content=None,
                    embeddings=[],
                    storage_ids=[],
                    processing_stats={},
                    total_processing_time=0.0,
                    success=False,
                    errors=[str(result)]
                ))
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.success)
        logger.info(f"Batch processing complete: {successful}/{len(processed_results)} successful")
        
        return processed_results
    
    async def _generate_embeddings_with_cache(self, 
                                            texts: List[str],
                                            metadata_list: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings using cache and API"""
        
        model_name = f"{self.config.preferred_embedding_provider}/{self.config.embedding_model}"
        
        # Create embedding function for cache integration
        async def embedding_function(input_texts):
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            
            return await self.embedding_api.generate_embeddings(
                input_texts, self.config.preferred_embedding_provider
            )
        
        # Use cache for batch generation
        embeddings = await self.embedding_cache.batch_get_or_generate_embeddings(
            texts, model_name, embedding_function, metadata_list
        )
        
        return embeddings
    
    async def _store_embeddings(self, 
                              processed_content: ProcessedContent,
                              embeddings: List[np.ndarray]) -> List[str]:
        """Store embeddings in vector database"""
        
        storage_ids = []
        
        for chunk, embedding in zip(processed_content.processed_chunks, embeddings):
            # Prepare metadata for storage
            storage_metadata = {
                **chunk.metadata,
                'parent_content_id': processed_content.content_id,
                'processing_stats': processed_content.processing_stats,
                'pipeline_version': '1.0',
                'embedding_model': f"{self.config.preferred_embedding_provider}/{self.config.embedding_model}",
            }
            
            # Store in vector database
            storage_id = await self.vector_store.add_item(
                text=chunk.text,
                embeddings=embedding,
                metadata=storage_metadata,
                item_id=chunk.chunk_id
            )
            
            storage_ids.append(storage_id)
        
        logger.debug(f"Stored {len(storage_ids)} embeddings in vector database")
        return storage_ids
    
    async def search_similar_content(self, 
                                   query: str,
                                   top_k: int = 10,
                                   similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar content using the pipeline"""
        
        # Generate query embedding
        query_embedding = await self._generate_embeddings_with_cache([query], [{}])
        
        # Search in vector store
        results = await self.vector_store.search_similar(
            query_embeddings=query_embedding[0],
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        
        base_stats = {
            'pipeline_performance': {
                'total_processed': self.total_processed,
                'total_processing_time': self.total_processing_time,
                'average_processing_time': (
                    self.total_processing_time / self.total_processed 
                    if self.total_processed > 0 else 0.0
                ),
                'error_count': self.error_count,
                'success_rate': (
                    (self.total_processed - self.error_count) / self.total_processed
                    if self.total_processed > 0 else 0.0
                ),
            },
            'configuration': {
                'max_chunk_size': self.config.max_chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'preferred_provider': self.config.preferred_embedding_provider,
                'embedding_model': self.config.embedding_model,
                'batch_size': self.config.batch_size,
                'max_concurrent': self.config.max_concurrent_requests,
            }
        }
        
        # Add cache stats if available
        if self.embedding_cache:
            base_stats['cache_stats'] = self.embedding_cache.get_cache_stats()
        
        # Add API stats if available
        if self.embedding_api:
            base_stats['api_stats'] = self.embedding_api.get_provider_stats()
        
        return base_stats
    
    async def cleanup(self):
        """Cleanup pipeline resources"""
        if self.embedding_cache:
            await self.embedding_cache.cleanup_cache()
        
        logger.info("Pipeline cleanup complete")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all pipeline components"""
        
        health_status = {
            'overall_healthy': True,
            'components': {},
            'timestamp': time.time()
        }
        
        # Check embedding API
        try:
            if self.embedding_api:
                provider_tests = await self.embedding_api.test_all_providers()
                working_providers = [
                    name for name, result in provider_tests.items() 
                    if result.get('success', False)
                ]
                
                health_status['components']['embedding_api'] = {
                    'healthy': len(working_providers) > 0,
                    'working_providers': working_providers,
                    'provider_tests': provider_tests
                }
                
                if len(working_providers) == 0:
                    health_status['overall_healthy'] = False
            else:
                health_status['components']['embedding_api'] = {
                    'healthy': False,
                    'error': 'API not initialized'
                }
                health_status['overall_healthy'] = False
        
        except Exception as e:
            health_status['components']['embedding_api'] = {
                'healthy': False,
                'error': str(e)
            }
            health_status['overall_healthy'] = False
        
        # Check vector store
        try:
            # Simple test query
            test_results = await self.vector_store.search_similar(
                query_embeddings=np.random.rand(384),  # Mock embedding
                top_k=1
            )
            
            health_status['components']['vector_store'] = {
                'healthy': True,
                'test_query_results': len(test_results)
            }
        
        except Exception as e:
            health_status['components']['vector_store'] = {
                'healthy': False,
                'error': str(e)
            }
            health_status['overall_healthy'] = False
        
        # Check cache
        try:
            if self.embedding_cache:
                cache_stats = self.embedding_cache.get_cache_stats()
                health_status['components']['embedding_cache'] = {
                    'healthy': True,
                    'cache_entries': cache_stats['cache_storage']['total_entries'],
                    'cache_size_mb': cache_stats['cache_storage']['total_size_mb']
                }
            else:
                health_status['components']['embedding_cache'] = {
                    'healthy': False,
                    'error': 'Cache not initialized'
                }
                health_status['overall_healthy'] = False
        
        except Exception as e:
            health_status['components']['embedding_cache'] = {
                'healthy': False,
                'error': str(e)
            }
            health_status['overall_healthy'] = False
        
        return health_status


# Utility functions for easy pipeline usage

async def create_pipeline(vector_store: PRSMVectorStore, 
                        config: EmbeddingPipelineConfig = None) -> EmbeddingPipeline:
    """Create and initialize an embedding pipeline"""
    pipeline = EmbeddingPipeline(vector_store, config)
    await pipeline.initialize()
    return pipeline


async def process_research_paper(pipeline: EmbeddingPipeline,
                               paper_text: str,
                               paper_id: str,
                               metadata: Dict[str, Any] = None) -> PipelineResult:
    """Process a research paper through the pipeline"""
    return await pipeline.process_content(
        text=paper_text,
        content_id=paper_id,
        content_type=ContentType.RESEARCH_PAPER,
        metadata=metadata
    )


async def process_code_repository(pipeline: EmbeddingPipeline,
                                code_content: str,
                                repo_id: str,
                                metadata: Dict[str, Any] = None) -> PipelineResult:
    """Process code repository content through the pipeline"""
    return await pipeline.process_content(
        text=code_content,
        content_id=repo_id,
        content_type=ContentType.CODE,
        metadata=metadata
    )