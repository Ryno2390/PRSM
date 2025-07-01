"""
PRSM Embedding Cache and Optimization System

High-performance caching layer for embedding generation and storage.
Provides intelligent caching, batch processing, and optimization strategies
for cost-effective and fast embedding operations.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingCacheEntry:
    """Cache entry for storing embedding with metadata"""
    content_hash: str
    text: str
    embedding: np.ndarray
    model_name: str
    created_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['embedding'] = self.embedding.tolist()
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat() if self.last_accessed else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingCacheEntry':
        """Create from dictionary"""
        data['embedding'] = np.array(data['embedding'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data['last_accessed']:
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


@dataclass
class BatchEmbeddingRequest:
    """Request for batch embedding generation"""
    texts: List[str]
    model_name: str
    request_id: str
    priority: int = 0
    callback: Optional[callable] = None
    metadata: Dict[str, Any] = None


@dataclass
class EmbeddingStats:
    """Statistics for embedding operations"""
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    batch_efficiency: float = 0.0


class EmbeddingCache:
    """
    High-performance embedding cache with intelligent optimization
    
    Features:
    - SQLite-based persistent cache with fast retrieval
    - Content-based hashing for deduplication
    - Batch processing for cost optimization
    - LRU eviction and cache size management
    - Provider-specific optimization strategies
    - Real-time performance metrics
    """
    
    def __init__(self, 
                 cache_dir: str = "prsm_embedding_cache",
                 max_cache_size_mb: int = 500,
                 max_age_days: int = 30,
                 batch_size: int = 100,
                 batch_timeout: float = 2.0):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_cache_size_mb = max_cache_size_mb
        self.max_age_days = max_age_days
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Initialize SQLite database
        self.db_path = self.cache_dir / "embeddings.db"
        self._init_database()
        
        # Batch processing queue
        self.batch_queue: List[BatchEmbeddingRequest] = []
        self.batch_processing = False
        self.batch_lock = asyncio.Lock()
        
        # Statistics
        self.stats = EmbeddingStats()
        
        # Model-specific configurations
        self.model_configs = {
            'openai': {
                'batch_size': 2048,  # OpenAI supports large batches
                'rate_limit': 3000,  # Requests per minute
                'cost_per_1k_tokens': 0.0001,
            },
            'anthropic': {
                'batch_size': 100,   # More conservative batching
                'rate_limit': 1000,
                'cost_per_1k_tokens': 0.0002,
            },
            'local': {
                'batch_size': 50,
                'rate_limit': float('inf'),
                'cost_per_1k_tokens': 0.0,
            }
        }
    
    def _init_database(self):
        """Initialize SQLite database for persistent caching"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    content_hash TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    metadata TEXT,
                    text_length INTEGER,
                    embedding_dimension INTEGER
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON embeddings(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON embeddings(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON embeddings(last_accessed)")
            
            conn.commit()
    
    def _compute_content_hash(self, text: str, model_name: str) -> str:
        """Compute hash for content and model combination"""
        content = f"{text}|{model_name}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def get_embedding(self, text: str, model_name: str, 
                           metadata: Dict[str, Any] = None) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        content_hash = self._compute_content_hash(text, model_name)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT embedding, access_count, last_accessed 
                FROM embeddings 
                WHERE content_hash = ?
            """, (content_hash,))
            
            row = cursor.fetchone()
            if row:
                # Update access statistics
                access_count = row[1] + 1
                current_time = datetime.now()
                
                conn.execute("""
                    UPDATE embeddings 
                    SET access_count = ?, last_accessed = ?
                    WHERE content_hash = ?
                """, (access_count, current_time, content_hash))
                
                conn.commit()
                
                # Deserialize embedding
                embedding = pickle.loads(row[0])
                self.stats.cache_hits += 1
                
                logger.debug(f"Cache hit for content hash {content_hash[:8]}...")
                return embedding
        
        self.stats.cache_misses += 1
        logger.debug(f"Cache miss for content hash {content_hash[:8]}...")
        return None
    
    async def store_embedding(self, text: str, embedding: np.ndarray, 
                             model_name: str, metadata: Dict[str, Any] = None):
        """Store embedding in cache"""
        content_hash = self._compute_content_hash(text, model_name)
        current_time = datetime.now()
        
        # Serialize embedding
        embedding_blob = pickle.dumps(embedding)
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO embeddings (
                    content_hash, text, embedding, model_name, created_at,
                    access_count, last_accessed, metadata, text_length, embedding_dimension
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content_hash, text, embedding_blob, model_name, current_time,
                1, current_time, metadata_json, len(text), len(embedding)
            ))
            conn.commit()
        
        logger.debug(f"Stored embedding for content hash {content_hash[:8]}...")
    
    async def get_or_generate_embedding(self, text: str, model_name: str,
                                       embedding_function: callable,
                                       metadata: Dict[str, Any] = None) -> np.ndarray:
        """Get embedding from cache or generate if not found"""
        # Try cache first
        embedding = await self.get_embedding(text, model_name, metadata)
        if embedding is not None:
            return embedding
        
        # Generate new embedding
        start_time = time.time()
        embedding = await embedding_function(text)
        
        # Update statistics
        self.stats.api_calls += 1
        self.stats.total_tokens += len(text.split())  # Rough estimate
        response_time = time.time() - start_time
        
        # Update average response time
        if self.stats.api_calls == 1:
            self.stats.average_response_time = response_time
        else:
            self.stats.average_response_time = (
                (self.stats.average_response_time * (self.stats.api_calls - 1) + response_time) 
                / self.stats.api_calls
            )
        
        # Store in cache
        await self.store_embedding(text, embedding, model_name, metadata)
        
        return embedding
    
    async def batch_get_or_generate_embeddings(self, 
                                              texts: List[str], 
                                              model_name: str,
                                              embedding_function: callable,
                                              metadata: List[Dict[str, Any]] = None) -> List[np.ndarray]:
        """Batch processing for multiple embeddings with optimization"""
        if not texts:
            return []
        
        if metadata is None:
            metadata = [{}] * len(texts)
        
        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []
        missing_metadata = []
        
        # Check cache for all texts
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            embedding = await self.get_embedding(text, model_name, meta)
            if embedding is not None:
                results[i] = embedding
            else:
                missing_indices.append(i)
                missing_texts.append(text)
                missing_metadata.append(meta)
        
        # Generate missing embeddings in batches
        if missing_texts:
            logger.info(f"Generating {len(missing_texts)} missing embeddings for model {model_name}")
            
            # Get model configuration
            provider = model_name.split('/')[0] if '/' in model_name else 'local'
            config = self.model_configs.get(provider, self.model_configs['local'])
            
            batch_size = min(config['batch_size'], self.batch_size)
            
            for batch_start in range(0, len(missing_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(missing_texts))
                batch_texts = missing_texts[batch_start:batch_end]
                batch_meta = missing_metadata[batch_start:batch_end]
                
                # Generate embeddings for batch
                start_time = time.time()
                batch_embeddings = await embedding_function(batch_texts)
                batch_time = time.time() - start_time
                
                # Update batch efficiency statistics
                expected_time = len(batch_texts) * self.stats.average_response_time
                if expected_time > 0:
                    efficiency = expected_time / batch_time
                    self.stats.batch_efficiency = (
                        (self.stats.batch_efficiency + efficiency) / 2
                        if self.stats.batch_efficiency > 0 else efficiency
                    )
                
                # Store results
                for i, (embedding, meta) in enumerate(zip(batch_embeddings, batch_meta)):
                    original_index = missing_indices[batch_start + i]
                    results[original_index] = embedding
                    
                    # Store in cache
                    await self.store_embedding(
                        batch_texts[i], embedding, model_name, meta
                    )
                
                # Update statistics
                self.stats.api_calls += 1
                self.stats.total_tokens += sum(len(text.split()) for text in batch_texts)
                
                # Rate limiting
                if batch_end < len(missing_texts) and config['rate_limit'] < float('inf'):
                    sleep_time = 60 / config['rate_limit']  # Seconds between requests
                    await asyncio.sleep(sleep_time)
        
        return results
    
    async def cleanup_cache(self):
        """Remove old and least-used cache entries"""
        cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Remove old entries
            cursor = conn.execute("DELETE FROM embeddings WHERE created_at < ?", (cutoff_date,))
            deleted_old = cursor.rowcount
            
            # Check cache size
            cursor = conn.execute("""
                SELECT COUNT(*), 
                       SUM(LENGTH(embedding) + LENGTH(text)) as total_size 
                FROM embeddings
            """)
            count, total_size = cursor.fetchone()
            
            # Convert to MB
            total_size_mb = (total_size or 0) / (1024 * 1024)
            
            if total_size_mb > self.max_cache_size_mb:
                # Remove least recently accessed entries
                excess_mb = total_size_mb - self.max_cache_size_mb
                excess_bytes = excess_mb * 1024 * 1024
                
                cursor = conn.execute("""
                    SELECT content_hash, LENGTH(embedding) + LENGTH(text) as entry_size
                    FROM embeddings 
                    ORDER BY last_accessed ASC, access_count ASC
                """)
                
                removed_size = 0
                to_remove = []
                
                for content_hash, entry_size in cursor:
                    to_remove.append(content_hash)
                    removed_size += entry_size
                    if removed_size >= excess_bytes:
                        break
                
                if to_remove:
                    placeholders = ','.join('?' for _ in to_remove)
                    conn.execute(f"DELETE FROM embeddings WHERE content_hash IN ({placeholders})", to_remove)
                    deleted_lru = len(to_remove)
                else:
                    deleted_lru = 0
            else:
                deleted_lru = 0
            
            conn.commit()
        
        logger.info(f"Cache cleanup: removed {deleted_old} old entries, {deleted_lru} LRU entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT model_name) as unique_models,
                    SUM(LENGTH(embedding) + LENGTH(text)) as total_size_bytes,
                    AVG(access_count) as avg_access_count,
                    MIN(created_at) as oldest_entry,
                    MAX(created_at) as newest_entry
                FROM embeddings
            """)
            
            cache_info = cursor.fetchone()
            
            # Model distribution
            cursor = conn.execute("""
                SELECT model_name, COUNT(*) as count
                FROM embeddings 
                GROUP BY model_name
                ORDER BY count DESC
            """)
            
            model_distribution = dict(cursor.fetchall())
        
        return {
            'cache_performance': {
                'hit_rate': self.stats.cache_hits / max(1, self.stats.cache_hits + self.stats.cache_misses),
                'total_hits': self.stats.cache_hits,
                'total_misses': self.stats.cache_misses,
                'api_calls': self.stats.api_calls,
                'average_response_time': self.stats.average_response_time,
                'batch_efficiency': self.stats.batch_efficiency,
            },
            'cache_storage': {
                'total_entries': cache_info[0],
                'unique_models': cache_info[1],
                'total_size_mb': (cache_info[2] or 0) / (1024 * 1024),
                'avg_access_count': cache_info[3] or 0,
                'oldest_entry': cache_info[4],
                'newest_entry': cache_info[5],
            },
            'model_distribution': model_distribution,
            'cost_estimation': {
                'total_tokens': self.stats.total_tokens,
                'estimated_cost': self.stats.total_cost,
            }
        }
    
    async def preload_embeddings(self, content_list: List[Tuple[str, str]], 
                               embedding_function: callable):
        """Preload embeddings for a list of (text, model_name) pairs"""
        logger.info(f"Preloading {len(content_list)} embeddings")
        
        # Group by model for efficient batch processing
        model_groups = {}
        for text, model_name in content_list:
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(text)
        
        # Process each model group
        for model_name, texts in model_groups.items():
            logger.info(f"Preloading {len(texts)} embeddings for model {model_name}")
            await self.batch_get_or_generate_embeddings(
                texts, model_name, embedding_function
            )
        
        logger.info("Preloading complete")


# Utility functions for easy integration

async def create_optimized_cache(cache_dir: str = "prsm_embedding_cache",
                               max_size_mb: int = 500) -> EmbeddingCache:
    """Create an optimized embedding cache"""
    cache = EmbeddingCache(
        cache_dir=cache_dir,
        max_cache_size_mb=max_size_mb,
        batch_size=100,
        batch_timeout=2.0
    )
    
    # Initial cleanup
    await cache.cleanup_cache()
    
    return cache


async def warm_cache_from_content(cache: EmbeddingCache, 
                                processed_content_list: List[Any],
                                embedding_function: callable):
    """Warm cache from processed content objects"""
    content_pairs = []
    
    for content in processed_content_list:
        model_name = content.metadata.get('preferred_model', 'openai/text-embedding-ada-002')
        for chunk in content.processed_chunks:
            content_pairs.append((chunk.text, model_name))
    
    await cache.preload_embeddings(content_pairs, embedding_function)