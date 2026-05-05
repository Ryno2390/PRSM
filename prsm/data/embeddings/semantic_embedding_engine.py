#!/usr/bin/env python3
"""
PRSM Semantic Embedding Engine
High-dimensional embedding spaces for content and knowledge representation

This module implements sophisticated embedding systems that enable:
1. Semantic similarity search across IPFS content
2. Cross-domain knowledge mapping
3. Efficient content retrieval and clustering
4. Multi-modal embedding support
5. Dynamic embedding adaptation and learning

Key Features:
- Multiple embedding models (sentence transformers, domain-specific)
- Hierarchical embedding spaces for different granularities
- Efficient vector similarity search with approximate nearest neighbors
- Embedding persistence and caching for performance
- Integration with IPFS content addressing and NWTN reasoning
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from datetime import datetime, timezone
import pickle
import hashlib
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

# Vector similarity and embeddings
try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from ..ipfs.content_addressing import AddressedContent
from ..nwtn.hybrid_architecture import SOC

logger = structlog.get_logger(__name__)


class EmbeddingModelType(str, Enum):
    """Types of embedding models"""
    SENTENCE_TRANSFORMER = "sentence_transformer"
    DOMAIN_SPECIFIC = "domain_specific"
    MULTILINGUAL = "multilingual"
    CODE_EMBEDDING = "code_embedding"
    SCIENTIFIC_PAPER = "scientific_paper"
    CUSTOM = "custom"


class EmbeddingSpace(str, Enum):
    """Different embedding spaces for different purposes"""
    CONTENT_SEMANTIC = "content_semantic"      # General content semantics
    DOMAIN_SPECIFIC = "domain_specific"        # Domain-specific embeddings
    CROSS_DOMAIN = "cross_domain"             # Cross-domain analogical mappings
    SOC_CONCEPTUAL = "soc_conceptual"         # SOC concept embeddings
    CITATION_NETWORK = "citation_network"      # Citation and reference embeddings
    TEMPORAL_CONTEXT = "temporal_context"      # Time-aware embeddings


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    
    model_type: EmbeddingModelType
    model_name: str
    embedding_dimension: int
    
    # Model parameters
    max_sequence_length: int = 512
    batch_size: int = 32
    normalize_embeddings: bool = True
    
    # Caching and performance
    cache_embeddings: bool = True
    cache_size_limit: int = 100000
    use_gpu: bool = False
    
    # Index configuration
    use_approximate_search: bool = True
    index_type: str = "IVF"  # FAISS index type
    num_clusters: int = 1000
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ContentEmbedding:
    """Embedding representation of content"""
    
    content_cid: str
    embedding_id: str
    embedding_space: EmbeddingSpace
    
    # Embedding data
    vector: np.ndarray
    dimension: int
    model_name: str
    
    # Content metadata
    content_title: str
    content_type: str
    domain: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # Embedding metadata
    confidence_score: float = 1.0
    quality_score: float = 0.0
    
    # Hierarchical embeddings
    chunk_embeddings: List[np.ndarray] = field(default_factory=list)
    paragraph_embeddings: List[np.ndarray] = field(default_factory=list)
    
    # Temporal and contextual
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0


@dataclass
class SimilarityResult:
    """Result of similarity search"""
    
    content_cid: str
    similarity_score: float
    embedding_distance: float
    
    # Content metadata
    title: str
    content_type: str
    domain: Optional[str] = None
    
    # Match details
    matching_chunks: List[int] = field(default_factory=list)
    relevance_explanation: str = ""
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EmbeddingSearchQuery(BaseModel):
    """Query for embedding-based search"""
    
    query_text: str
    embedding_space: EmbeddingSpace = EmbeddingSpace.CONTENT_SEMANTIC
    
    # Search parameters
    max_results: int = 20
    min_similarity: float = 0.3
    include_metadata: bool = True
    
    # Filtering
    content_types: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    date_range: Optional[Tuple[datetime, datetime]] = None
    
    # Advanced options
    use_reranking: bool = True
    explain_relevance: bool = False
    boost_recent_content: bool = False


class SemanticEmbeddingEngine:
    """
    High-dimensional embedding engine for semantic content representation
    
    This system provides sophisticated embedding capabilities for:
    - Semantic similarity search across content
    - Cross-domain knowledge mapping
    - Efficient content clustering and organization
    - Multi-granularity content representation
    """
    
    def __init__(self, embedding_configs: Dict[EmbeddingSpace, EmbeddingConfig] = None):
        
        # Embedding models for different spaces
        self.embedding_models: Dict[EmbeddingSpace, Any] = {}
        self.embedding_configs = embedding_configs or self._create_default_configs()
        
        # Vector indices for fast similarity search
        self.vector_indices: Dict[EmbeddingSpace, Any] = {}
        
        # Embedding storage
        self.content_embeddings: Dict[str, ContentEmbedding] = {}  # CID -> embedding
        self.embedding_cache: Dict[str, np.ndarray] = {}  # text_hash -> vector
        
        # Hierarchical embedding maps
        self.domain_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.soc_embeddings: Dict[str, np.ndarray] = {}
        
        # Performance tracking
        self.stats = {
            'embeddings_created': 0,
            'similarity_searches': 0,
            'cache_hits': 0,
            'total_embedding_time': 0.0,
            'total_search_time': 0.0
        }
        
        # Initialize models
        self.initialized = False
        
        logger.info("Semantic Embedding Engine created")
    
    async def initialize(self):
        """Initialize embedding models and indices"""
        
        if self.initialized:
            return
        
        logger.info("Initializing semantic embedding engine",
                   spaces=list(self.embedding_configs.keys()))
        
        # Initialize each embedding space
        for space, config in self.embedding_configs.items():
            try:
                await self._initialize_embedding_space(space, config)
            except Exception as e:
                logger.error("Failed to initialize embedding space",
                           space=space.value,
                           error=str(e))
        
        self.initialized = True
        logger.info("Semantic embedding engine initialized")
    
    async def _initialize_embedding_space(self, space: EmbeddingSpace, config: EmbeddingConfig):
        """Initialize a specific embedding space"""
        
        if config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMER:
            if not HAS_SENTENCE_TRANSFORMERS:
                logger.warning("sentence-transformers not available",
                             space=space.value)
                return
            
            # Load sentence transformer model
            model = SentenceTransformer(config.model_name)
            if config.use_gpu and model.device.type != 'cuda':
                logger.warning("GPU requested but not available for",
                             space=space.value)
            
            self.embedding_models[space] = model
            
        elif config.model_type == EmbeddingModelType.DOMAIN_SPECIFIC:
            # Load domain-specific model
            model = await self._load_domain_specific_model(config)
            self.embedding_models[space] = model
            
        elif config.model_type == EmbeddingModelType.SCIENTIFIC_PAPER:
            # Use SciBERT or similar scientific paper model
            if HAS_SENTENCE_TRANSFORMERS:
                model = SentenceTransformer('allenai-specter')  # Scientific paper embeddings
                self.embedding_models[space] = model
        
        # Initialize vector index
        if HAS_FAISS and config.use_approximate_search:
            index = await self._create_faiss_index(config)
            self.vector_indices[space] = index
        
        logger.debug("Embedding space initialized",
                    space=space.value,
                    model=config.model_name,
                    dimension=config.embedding_dimension)
    
    async def _create_faiss_index(self, config: EmbeddingConfig):
        """Create FAISS index for efficient similarity search"""
        
        if config.index_type == "IVF":
            # Create IVF (Inverted File) index
            quantizer = faiss.IndexFlatL2(config.embedding_dimension)
            index = faiss.IndexIVFFlat(quantizer, config.embedding_dimension, config.num_clusters)
        elif config.index_type == "HNSW":
            # Create HNSW (Hierarchical Navigable Small World) index
            index = faiss.IndexHNSWFlat(config.embedding_dimension, 32)
        else:
            # Default to flat L2 index
            index = faiss.IndexFlatL2(config.embedding_dimension)
        
        return index
    
    async def embed_content(self,
                          content: AddressedContent,
                          embedding_space: EmbeddingSpace = EmbeddingSpace.CONTENT_SEMANTIC,
                          content_text: str = None) -> ContentEmbedding:
        """
        Create embeddings for content
        
        Args:
            content: AddressedContent object
            embedding_space: Which embedding space to use
            content_text: Optional content text (will be retrieved if not provided)
            
        Returns:
            ContentEmbedding with vector representations
        """
        
        if not self.initialized:
            await self.initialize()
        
        if embedding_space not in self.embedding_models:
            raise ValueError(f"Embedding space {embedding_space.value} not initialized")
        
        start_time = datetime.now()
        
        logger.debug("Creating content embedding",
                    cid=content.cid,
                    space=embedding_space.value,
                    title=content.title[:50])
        
        try:
            # Get content text if not provided
            if not content_text:
                # Would retrieve from IPFS in real implementation
                content_text = f"{content.title}\n\n{content.description}"
            
            # Check cache first
            text_hash = hashlib.md5(content_text.encode()).hexdigest()
            cache_key = f"{embedding_space.value}_{text_hash}"
            
            if cache_key in self.embedding_cache:
                main_vector = self.embedding_cache[cache_key]
                self.stats['cache_hits'] += 1
            else:
                # Generate embedding
                model = self.embedding_models[embedding_space]
                main_vector = await self._generate_embedding(model, content_text)
                
                # Cache the result
                self.embedding_cache[cache_key] = main_vector
            
            # Create hierarchical embeddings
            chunk_embeddings = await self._create_chunk_embeddings(
                model, content_text, chunk_size=500
            )
            paragraph_embeddings = await self._create_paragraph_embeddings(
                model, content_text
            )
            
            # Calculate quality score
            quality_score = await self._calculate_embedding_quality(
                main_vector, content_text, content
            )
            
            # Create embedding object
            embedding = ContentEmbedding(
                content_cid=content.cid,
                embedding_id=str(uuid4()),
                embedding_space=embedding_space,
                vector=main_vector,
                dimension=len(main_vector),
                model_name=self.embedding_configs[embedding_space].model_name,
                content_title=content.title,
                content_type=content.content_type,
                domain=content.category.value,
                keywords=content.keywords,
                confidence_score=1.0,
                quality_score=quality_score,
                chunk_embeddings=chunk_embeddings,
                paragraph_embeddings=paragraph_embeddings
            )
            
            # Store embedding
            self.content_embeddings[content.cid] = embedding
            
            # Add to vector index
            await self._add_to_vector_index(embedding_space, main_vector, content.cid)
            
            # Update statistics
            embedding_time = (datetime.now() - start_time).total_seconds()
            self.stats['embeddings_created'] += 1
            self.stats['total_embedding_time'] += embedding_time
            
            logger.debug("Content embedding created",
                        cid=content.cid,
                        dimension=len(main_vector),
                        quality_score=quality_score,
                        embedding_time=embedding_time)
            
            return embedding
            
        except Exception as e:
            logger.error("Content embedding failed",
                        cid=content.cid,
                        error=str(e))
            raise
    
    async def _generate_embedding(self, model: Any, text: str) -> np.ndarray:
        """Generate embedding vector for text"""
        
        if isinstance(model, SentenceTransformer):
            # Use sentence transformers
            embedding = model.encode(text, convert_to_numpy=True)
            
            # Normalize if configured
            space_config = None
            for space, config in self.embedding_configs.items():
                if self.embedding_models.get(space) == model:
                    space_config = config
                    break
            
            if space_config and getattr(space_config, 'normalize_embeddings', True):
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        else:
            # Custom model - implement based on model type
            return await self._custom_embedding_generation(model, text)
    
    async def _create_chunk_embeddings(self, model: Any, text: str, chunk_size: int = 500) -> List[np.ndarray]:
        """Create embeddings for text chunks"""
        
        # Split text into chunks
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        # Generate embeddings for each chunk
        chunk_embeddings = []
        for chunk in chunks:
            if len(chunk.strip()) > 10:  # Only embed substantial chunks
                embedding = await self._generate_embedding(model, chunk)
                chunk_embeddings.append(embedding)
        
        return chunk_embeddings
    
    async def _create_paragraph_embeddings(self, model: Any, text: str) -> List[np.ndarray]:
        """Create embeddings for paragraphs"""
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        
        paragraph_embeddings = []
        for paragraph in paragraphs:
            embedding = await self._generate_embedding(model, paragraph)
            paragraph_embeddings.append(embedding)
        
        return paragraph_embeddings
    
    async def _calculate_embedding_quality(self, 
                                         vector: np.ndarray, 
                                         text: str, 
                                         content: AddressedContent) -> float:
        """Calculate quality score for embedding"""
        
        quality_factors = []
        
        # Vector quality (non-zero, reasonable magnitude)
        vector_magnitude = np.linalg.norm(vector)
        if vector_magnitude > 0.1 and vector_magnitude < 10.0:
            quality_factors.append(0.3)
        
        # Text quality
        word_count = len(text.split())
        if word_count > 50:
            text_quality = min(1.0, word_count / 1000)
            quality_factors.append(text_quality * 0.3)
        
        # Content metadata quality
        if content.keywords:
            quality_factors.append(min(0.2, len(content.keywords) * 0.05))
        
        # Domain specificity
        if content.category:
            quality_factors.append(0.2)
        
        return sum(quality_factors)
    
    async def _add_to_vector_index(self, space: EmbeddingSpace, vector: np.ndarray, cid: str):
        """Add vector to appropriate index"""
        
        if space in self.vector_indices:
            index = self.vector_indices[space]
            
            if HAS_FAISS:
                # Add to FAISS index
                vector_2d = vector.reshape(1, -1).astype('float32')
                
                # Get current index size before adding
                current_size = index.ntotal
                
                # Add vector to index
                index.add(vector_2d)
                
                # Store CID mapping
                if not hasattr(index, 'index_to_cid'):
                    index.index_to_cid = {}
                index.index_to_cid[current_size] = cid
                
                # Also store reverse mapping
                if not hasattr(index, 'cid_to_index'):
                    index.cid_to_index = {}
                index.cid_to_index[cid] = current_size
    
    async def semantic_search(self, query: EmbeddingSearchQuery) -> List[SimilarityResult]:
        """
        Perform semantic similarity search
        
        Args:
            query: Search query with parameters
            
        Returns:
            List of similar content ranked by relevance
        """
        
        if not self.initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        logger.info("Performing semantic search",
                   query=query.query_text[:100],
                   space=query.embedding_space.value,
                   max_results=query.max_results)
        
        try:
            # Generate query embedding
            if query.embedding_space not in self.embedding_models:
                raise ValueError(f"Embedding space {query.embedding_space.value} not available")
            
            model = self.embedding_models[query.embedding_space]
            query_vector = await self._generate_embedding(model, query.query_text)
            
            # Perform similarity search
            if query.embedding_space in self.vector_indices and HAS_FAISS:
                # Use FAISS for efficient search
                results = await self._faiss_similarity_search(
                    query.embedding_space, query_vector, query
                )
            else:
                # Fallback to brute force search
                results = await self._brute_force_similarity_search(
                    query.embedding_space, query_vector, query
                )
            
            # Post-process results
            if query.use_reranking:
                results = await self._rerank_results(results, query)
            
            if query.boost_recent_content:
                results = await self._boost_recent_content(results)
            
            # Update statistics
            search_time = (datetime.now() - start_time).total_seconds()
            self.stats['similarity_searches'] += 1
            self.stats['total_search_time'] += search_time
            
            logger.info("Semantic search completed",
                       query=query.query_text[:50],
                       results_found=len(results),
                       search_time=search_time)
            
            return results[:query.max_results]
            
        except Exception as e:
            logger.error("Semantic search failed",
                        query=query.query_text[:100],
                        error=str(e))
            raise
    
    async def _faiss_similarity_search(self,
                                     space: EmbeddingSpace,
                                     query_vector: np.ndarray,
                                     query: EmbeddingSearchQuery) -> List[SimilarityResult]:
        """Perform similarity search using FAISS index"""
        
        index = self.vector_indices[space]
        
        # Search for similar vectors
        query_2d = query_vector.reshape(1, -1).astype('float32')
        distances, indices = index.search(query_2d, query.max_results * 2)  # Get extra for filtering
        
        results = []
        
        # Map indices to actual content CIDs
        index_to_cid = getattr(index, 'index_to_cid', {})
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
            
            # Convert distance to similarity score
            similarity_score = 1.0 / (1.0 + distance)
            
            if similarity_score < query.min_similarity:
                continue
            
            # Get actual content CID if available
            content_cid = index_to_cid.get(idx, f"indexed_content_{idx}")
            
            # Try to get embedding from storage
            embedding = self.content_embeddings.get(content_cid)
            if embedding:
                result = SimilarityResult(
                    content_cid=content_cid,
                    similarity_score=similarity_score,
                    embedding_distance=distance,
                    title=embedding.content_title,
                    content_type=embedding.content_type,
                    domain=embedding.domain,
                    relevance_explanation=f"Semantic similarity: {similarity_score:.3f}"
                )
            else:
                # Fallback result
                result = SimilarityResult(
                    content_cid=content_cid,
                    similarity_score=similarity_score,
                    embedding_distance=distance,
                    title=f"Similar Content {idx}",
                    content_type="research_paper",
                    relevance_explanation=f"Semantic similarity: {similarity_score:.3f}"
                )
            
            results.append(result)
        
        return results
    
    async def _brute_force_similarity_search(self,
                                           space: EmbeddingSpace,
                                           query_vector: np.ndarray,
                                           query: EmbeddingSearchQuery) -> List[SimilarityResult]:
        """Fallback brute force similarity search"""
        
        results = []
        
        # Search through all content embeddings
        for cid, embedding in self.content_embeddings.items():
            if embedding.embedding_space != space:
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector, embedding.vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(embedding.vector)
            )
            
            if similarity < query.min_similarity:
                continue
            
            # Apply filters
            if query.content_types and embedding.content_type not in query.content_types:
                continue
            
            if query.domains and embedding.domain not in query.domains:
                continue
            
            result = SimilarityResult(
                content_cid=cid,
                similarity_score=similarity,
                embedding_distance=1.0 - similarity,
                title=embedding.content_title,
                content_type=embedding.content_type,
                domain=embedding.domain,
                relevance_explanation=f"Cosine similarity: {similarity:.3f}"
            )
            results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results
    
    async def _rerank_results(self, results: List[SimilarityResult], query: EmbeddingSearchQuery) -> List[SimilarityResult]:
        """Rerank results using additional criteria"""
        
        # Simple reranking based on multiple factors
        for result in results:
            rerank_score = result.similarity_score
            
            # Boost based on content type relevance
            if "research" in query.query_text.lower() and result.content_type == "research_paper":
                rerank_score *= 1.1
            
            # Boost based on domain match
            if query.domains and result.domain in query.domains:
                rerank_score *= 1.05
            
            # Update similarity score with reranking
            result.similarity_score = min(1.0, rerank_score)
        
        # Resort
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results
    
    async def _boost_recent_content(self, results: List[SimilarityResult]) -> List[SimilarityResult]:
        """Boost recent content in results"""
        
        # Would implement temporal boosting based on content timestamps
        return results
    
    async def embed_soc(self, soc: SOC, embedding_space: EmbeddingSpace = EmbeddingSpace.SOC_CONCEPTUAL) -> np.ndarray:
        """Create embedding for SOC (Subject-Object-Concept)"""
        
        if embedding_space not in self.embedding_models:
            raise ValueError(f"Embedding space {embedding_space.value} not available")
        
        # Create text representation of SOC
        soc_text = f"{soc.name}\nType: {soc.soc_type.value}\nDomain: {soc.domain}\nProperties: {soc.properties}"
        
        # Generate embedding
        model = self.embedding_models[embedding_space]
        soc_vector = await self._generate_embedding(model, soc_text)
        
        # Store SOC embedding
        self.soc_embeddings[soc.name] = soc_vector
        
        return soc_vector
    
    async def find_similar_socs(self, soc: SOC, max_results: int = 10) -> List[Tuple[str, float]]:
        """Find SOCs similar to the given SOC"""
        
        # Get embedding for query SOC
        query_vector = await self.embed_soc(soc)
        
        # Search through stored SOC embeddings
        similarities = []
        
        for soc_name, soc_vector in self.soc_embeddings.items():
            if soc_name == soc.name:
                continue
            
            similarity = np.dot(query_vector, soc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(soc_vector)
            )
            
            similarities.append((soc_name, similarity))
        
        # Sort and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:max_results]
    
    async def _load_domain_specific_model(self, config: EmbeddingConfig):
        """Load domain-specific embedding model"""
        
        # Would load specialized models for different domains
        # For now, fallback to sentence transformer
        if HAS_SENTENCE_TRANSFORMERS:
            return SentenceTransformer('all-MiniLM-L6-v2')
        else:
            raise ValueError("No embedding model available")
    
    async def _custom_embedding_generation(self, model: Any, text: str) -> np.ndarray:
        """Generate embeddings with custom model"""
        
        # Placeholder for custom embedding generation
        # Would implement based on specific model type
        return np.random.random(384)  # Dummy embedding
    
    def _create_default_configs(self) -> Dict[EmbeddingSpace, EmbeddingConfig]:
        """Create default embedding configurations"""
        
        configs = {}
        
        # General content semantics
        configs[EmbeddingSpace.CONTENT_SEMANTIC] = EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            model_name='all-MiniLM-L6-v2',
            embedding_dimension=384,
            max_sequence_length=512,
            use_approximate_search=True
        )
        
        # Scientific papers
        configs[EmbeddingSpace.DOMAIN_SPECIFIC] = EmbeddingConfig(
            model_type=EmbeddingModelType.SCIENTIFIC_PAPER,
            model_name='allenai-specter',
            embedding_dimension=768,
            max_sequence_length=512,
            use_approximate_search=True
        )
        
        # SOC conceptual embeddings
        configs[EmbeddingSpace.SOC_CONCEPTUAL] = EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            model_name='all-MiniLM-L6-v2',
            embedding_dimension=384,
            max_sequence_length=256,
            use_approximate_search=False  # Smaller space, exact search OK
        )
        
        return configs
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding system statistics"""
        
        return {
            'embedding_stats': self.stats.copy(),
            'initialized_spaces': list(self.embedding_models.keys()),
            'content_embeddings': len(self.content_embeddings),
            'soc_embeddings': len(self.soc_embeddings),
            'cache_size': len(self.embedding_cache),
            'vector_indices': list(self.vector_indices.keys()),
            'average_embedding_time': (
                self.stats['total_embedding_time'] / max(1, self.stats['embeddings_created'])
            ),
            'average_search_time': (
                self.stats['total_search_time'] / max(1, self.stats['similarity_searches'])
            ),
            'dependencies': {
                'sentence_transformers': HAS_SENTENCE_TRANSFORMERS,
                'faiss': HAS_FAISS,
                'umap': HAS_UMAP
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on embedding system"""
        
        try:
            health = {
                'healthy': True,
                'initialized': self.initialized,
                'models_loaded': len(self.embedding_models),
                'indices_ready': len(self.vector_indices)
            }
            
            if self.initialized:
                # Test embedding generation
                try:
                    if self.embedding_models:
                        first_space = next(iter(self.embedding_models.keys()))
                        model = self.embedding_models[first_space]
                        test_embedding = await self._generate_embedding(model, "test text")
                        health['embedding_generation_working'] = len(test_embedding) > 0
                    else:
                        health['embedding_generation_working'] = False
                        health['healthy'] = False
                except Exception as e:
                    health['embedding_generation_working'] = False
                    health['embedding_error'] = str(e)
                    health['healthy'] = False
            else:
                health['healthy'] = False
                health['error'] = "Not initialized"
            
            health['stats'] = self.get_embedding_stats()
            
            return health
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'stats': self.get_embedding_stats()
            }


# Utility functions

def create_embedding_engine(configs: Dict[EmbeddingSpace, EmbeddingConfig] = None) -> SemanticEmbeddingEngine:
    """Create a new semantic embedding engine"""
    return SemanticEmbeddingEngine(configs)


async def embed_and_search(engine: SemanticEmbeddingEngine,
                          content: AddressedContent,
                          search_query: str,
                          content_text: str = None) -> Tuple[ContentEmbedding, List[SimilarityResult]]:
    """Utility to embed content and perform search"""
    
    # Embed the content
    embedding = await engine.embed_content(content, content_text=content_text)
    
    # Search for similar content
    search_query_obj = EmbeddingSearchQuery(
        query_text=search_query,
        max_results=10
    )
    
    similar_content = await engine.semantic_search(search_query_obj)
    
    return embedding, similar_content