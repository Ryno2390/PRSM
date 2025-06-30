"""
PostgreSQL + pgvector implementation for PRSM vector store

Production-grade implementation using PostgreSQL with the pgvector extension.
Perfect for:
- Development and prototyping with Docker Compose  
- Small to medium scale deployments (up to millions of vectors)
- Integration with existing PostgreSQL infrastructure
- Migration bridge to larger vector databases in Phase 1B

Features:
- Async operations using asyncpg for high performance
- HNSW indexing for fast similarity search
- JSON metadata storage with GIN indexing for complex queries
- Built-in provenance tracking for FTNS royalty calculations
- Connection pooling and automatic reconnection
- Comprehensive error handling and logging

Installation requirements:
- PostgreSQL 12+ with pgvector extension
- pip install asyncpg
- Optional: pip install psycopg2-binary (for sync operations)

Quick Start with Docker:
    docker-compose -f docker-compose.vector.yml up postgres-vector
    python test_pgvector_implementation.py
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import asyncpg
except ImportError:
    raise ImportError(
        "PgVectorStore requires asyncpg. Install with:\n"
        "pip install asyncpg"
    )

# psycopg2 is optional - only needed for sync operations and some advanced features
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    # This is fine - asyncpg handles all async operations

from ..base import (
    PRSMVectorStore, ContentMatch, SearchFilters, VectorStoreConfig,
    ContentType, VectorStoreType
)

logger = logging.getLogger(__name__)


class PgVectorStore(PRSMVectorStore):
    """
    PostgreSQL + pgvector implementation of PRSM vector store
    
    Features:
    - Async operations using asyncpg
    - JSON metadata storage with GIN indexing
    - Cosine similarity search with HNSW indexing
    - IPFS content addressing integration
    - Automatic provenance tracking for royalties
    - Built-in performance monitoring
    """
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.connection_pool: Optional[asyncpg.Pool] = None
        
        # Use schema-qualified table name for production setup
        self.schema_name = "prsm_vector"
        self.table_name = f"{self.schema_name}.content_vectors"
        
        # Validate config for pgvector
        if config.store_type != VectorStoreType.PGVECTOR:
            raise ValueError(f"Expected pgvector config, got {config.store_type}")
    
    async def connect(self) -> bool:
        """Establish connection pool to PostgreSQL with pgvector"""
        try:
            # Build connection string with proper defaults
            username = self.config.username or "postgres"
            password = self.config.password or "postgres"
            host = self.config.host or "localhost"
            port = self.config.port or 5432
            database = self.config.database or "prsm_vector_dev"
            
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            
            logger.info(f"Connecting to PostgreSQL: {host}:{port}/{database}")
            
            # Create connection pool with proper settings
            self.connection_pool = await asyncpg.create_pool(
                connection_string,
                min_size=2,
                max_size=min(self.config.max_connections, 20),
                command_timeout=self.config.query_timeout,
                # Set application name for monitoring
                server_settings={"application_name": "prsm_vector_store"}
            )
            
            # Test connection and verify setup
            async with self.connection_pool.acquire() as conn:
                # Check PostgreSQL version
                version_result = await conn.fetchrow("SELECT version()")
                logger.info(f"Connected to: {version_result['version']}")
                
                # Check if pgvector extension is available
                vector_check = await conn.fetchrow(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
                
                if not vector_check[0]:
                    logger.warning("pgvector extension not found - this is expected for fresh databases")
                    logger.info("Database initialization will be handled by init scripts")
                else:
                    logger.info("pgvector extension found and ready")
                
                # Check if our schema exists
                schema_check = await conn.fetchrow(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = $1)",
                    self.schema_name
                )
                
                if schema_check[0]:
                    logger.info(f"Schema '{self.schema_name}' found")
                    
                    # Check if our table exists
                    table_check = await conn.fetchrow(
                        """SELECT EXISTS(SELECT 1 FROM information_schema.tables 
                           WHERE table_schema = $1 AND table_name = 'content_vectors')""",
                        self.schema_name
                    )
                    
                    if table_check[0]:
                        logger.info("PRSM vector store table found and ready")
                        
                        # Get table stats for confirmation
                        stats = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {self.table_name}")
                        logger.info(f"Found {stats['count']} existing vectors in database")
                    else:
                        logger.warning("PRSM vector store table not found - will be created by init scripts")
                else:
                    logger.warning(f"Schema '{self.schema_name}' not found - will be created by init scripts")
            
            self.is_connected = True
            logger.info("✅ PostgreSQL connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            logger.error("Make sure PostgreSQL is running and accessible")
            logger.error("For Docker setup: docker-compose -f docker-compose.vector.yml up postgres-vector")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Close connection pool"""
        try:
            if self.connection_pool:
                await self.connection_pool.close()
                self.connection_pool = None
            self.is_connected = False
            logger.info("Disconnected from PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            return False
    
    async def create_collection(self, collection_name: str, 
                              vector_dimension: int,
                              metadata_schema: Dict[str, Any]) -> bool:
        """Create a new table for storing vectors"""
        try:
            async with self.connection_pool.acquire() as conn:
                # Create table with vector column
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {collection_name} (
                    id SERIAL PRIMARY KEY,
                    content_cid TEXT UNIQUE NOT NULL,
                    vector vector({vector_dimension}),
                    metadata JSONB,
                    content_type TEXT,
                    creator_id TEXT,
                    royalty_rate FLOAT DEFAULT 0.08,
                    quality_score FLOAT,
                    peer_review_score FLOAT,
                    citation_count INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_accessed TIMESTAMP WITH TIME ZONE,
                    INDEX_created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                -- Create indexes for performance
                CREATE INDEX IF NOT EXISTS idx_{collection_name}_content_cid 
                    ON {collection_name} (content_cid);
                    
                CREATE INDEX IF NOT EXISTS idx_{collection_name}_creator_id 
                    ON {collection_name} (creator_id);
                    
                CREATE INDEX IF NOT EXISTS idx_{collection_name}_content_type 
                    ON {collection_name} (content_type);
                    
                CREATE INDEX IF NOT EXISTS idx_{collection_name}_metadata 
                    ON {collection_name} USING GIN (metadata);
                    
                CREATE INDEX IF NOT EXISTS idx_{collection_name}_created_at 
                    ON {collection_name} (created_at);
                
                -- Create HNSW index for vector similarity search
                CREATE INDEX IF NOT EXISTS idx_{collection_name}_vector_cosine
                    ON {collection_name} USING hnsw (vector vector_cosine_ops);
                """
                
                await conn.execute(create_table_sql)
                logger.info(f"Created collection table: {collection_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    async def store_content_with_embeddings(self,
                                          content_cid: str,
                                          embeddings: np.ndarray,
                                          metadata: Dict[str, Any]) -> str:
        """Store content embeddings with metadata in PostgreSQL"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.connection_pool.acquire() as conn:
                # Convert numpy array to pgvector format
                # pgvector expects a string representation like '[1.0,2.0,3.0]'
                vector_str = '[' + ','.join(map(str, embeddings.tolist())) + ']'
                
                # Extract structured fields from metadata
                title = metadata.get('title', 'Untitled')
                description = metadata.get('description', '')
                content_type = metadata.get('content_type', ContentType.TEXT.value)
                creator_id = metadata.get('creator_id')
                royalty_rate = float(metadata.get('royalty_rate', 0.08))
                license_info = metadata.get('license', '')
                quality_score = metadata.get('quality_score')
                peer_review_score = metadata.get('peer_review_score')
                citation_count = int(metadata.get('citation_count', 0))
                
                # Convert quality scores to proper format
                if quality_score is not None:
                    quality_score = float(quality_score)
                if peer_review_score is not None:
                    peer_review_score = float(peer_review_score)
                
                # Insert or update the content using our schema
                insert_sql = f"""
                INSERT INTO {self.table_name} 
                (content_cid, embedding, title, description, content_type, creator_id, 
                 royalty_rate, license, quality_score, peer_review_score, citation_count, metadata)
                VALUES ($1, $2::vector, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (content_cid) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    content_type = EXCLUDED.content_type,
                    creator_id = EXCLUDED.creator_id,
                    royalty_rate = EXCLUDED.royalty_rate,
                    license = EXCLUDED.license,
                    quality_score = EXCLUDED.quality_score,
                    peer_review_score = EXCLUDED.peer_review_score,
                    citation_count = EXCLUDED.citation_count,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING id
                """
                
                result = await conn.fetchrow(
                    insert_sql,
                    content_cid,
                    vector_str,
                    title,
                    description,
                    content_type,
                    creator_id,
                    royalty_rate,
                    license_info,
                    quality_score,
                    peer_review_score,
                    citation_count,
                    json.dumps(metadata)
                )
                
                vector_id = str(result['id'])
                
                # Update performance metrics
                duration = asyncio.get_event_loop().time() - start_time
                self._update_performance_metrics("storage", duration, True)
                
                logger.debug(f"Stored content {content_cid} with vector ID {vector_id}")
                return vector_id
                
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            self._update_performance_metrics("storage", duration, False)
            logger.error(f"Failed to store content {content_cid}: {e}")
            raise
    
    async def search_similar_content(self,
                                   query_embedding: np.ndarray,
                                   filters: Optional[SearchFilters] = None,
                                   top_k: int = 10) -> List[ContentMatch]:
        """Search for similar content using pgvector cosine similarity"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.connection_pool.acquire() as conn:
                # Convert query embedding to pgvector format
                query_vector_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
                
                # Build WHERE clause based on filters
                where_conditions = ["active = TRUE"]  # Only active content
                params = [query_vector_str]
                param_count = 1
                
                if filters:
                    if filters.content_types:
                        param_count += 1
                        # Handle enum comparison properly
                        content_type_values = [ct.value for ct in filters.content_types]
                        where_conditions.append(f"content_type = ANY(${param_count})")
                        params.append(content_type_values)
                    
                    if filters.creator_ids:
                        param_count += 1
                        where_conditions.append(f"creator_id = ANY(${param_count})")
                        params.append(filters.creator_ids)
                    
                    if filters.min_quality_score is not None:
                        param_count += 1
                        where_conditions.append(f"quality_score >= ${param_count}")
                        params.append(filters.min_quality_score)
                    
                    if filters.max_royalty_rate is not None:
                        param_count += 1
                        where_conditions.append(f"royalty_rate <= ${param_count}")
                        params.append(filters.max_royalty_rate)
                    
                    if filters.exclude_content_cids:
                        param_count += 1
                        where_conditions.append(f"content_cid != ALL(${param_count})")
                        params.append(filters.exclude_content_cids)
                    
                    if filters.require_open_license:
                        where_conditions.append(
                            "(license ILIKE '%open%' OR license ILIKE '%creative commons%' OR "
                            "metadata->>'license' ILIKE '%open%' OR metadata->>'license' ILIKE '%creative commons%')"
                        )
                
                # Build the search query using our schema
                where_clause = " AND ".join(where_conditions)
                
                search_sql = f"""
                SELECT 
                    content_cid,
                    (1 - (embedding <=> $1::vector)) as similarity_score,
                    title,
                    description,
                    content_type,
                    creator_id,
                    royalty_rate,
                    license,
                    quality_score,
                    peer_review_score,
                    citation_count,
                    access_count,
                    created_at,
                    last_accessed,
                    metadata
                FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY embedding <=> $1::vector
                LIMIT ${param_count + 1}
                """
                
                # Add top_k as final parameter
                params.append(top_k)
                
                results = await conn.fetch(search_sql, *params)
                
                # Convert results to ContentMatch objects
                content_matches = []
                for row in results:
                    # Parse metadata JSON if present
                    metadata_dict = json.loads(row['metadata']) if row['metadata'] else {}
                    
                    # Add structured fields to metadata for consistency
                    metadata_dict.update({
                        'title': row['title'],
                        'description': row['description'],
                        'license': row['license']
                    })
                    
                    # Ensure similarity score is within valid range
                    similarity = float(row['similarity_score']) if row['similarity_score'] is not None else 0.0
                    similarity = min(1.0, max(0.0, similarity))
                    
                    match = ContentMatch(
                        content_cid=row['content_cid'],
                        similarity_score=similarity,
                        metadata=metadata_dict,
                        creator_id=row['creator_id'],
                        royalty_rate=float(row['royalty_rate']) if row['royalty_rate'] else 0.08,
                        content_type=ContentType(row['content_type']) if row['content_type'] else ContentType.TEXT,
                        access_count=row['access_count'] or 0,
                        last_accessed=row['last_accessed'],
                        quality_score=float(row['quality_score']) if row['quality_score'] else None,
                        peer_review_score=float(row['peer_review_score']) if row['peer_review_score'] else None,
                        citation_count=row['citation_count'] or 0
                    )
                    content_matches.append(match)
                
                # Update access counts for returned results
                if content_matches:
                    content_cids = [match.content_cid for match in content_matches]
                    await self._track_content_access_batch(conn, content_cids)
                
                # Update performance metrics
                duration = asyncio.get_event_loop().time() - start_time
                self._update_performance_metrics("query", duration, True)
                
                logger.debug(f"Found {len(content_matches)} similar content items")
                return content_matches
                
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            self._update_performance_metrics("query", duration, False)
            logger.error(f"Failed to search content: {e}")
            raise
    
    async def update_content_metadata(self,
                                    content_cid: str,
                                    metadata_updates: Dict[str, Any]) -> bool:
        """Update metadata for existing content"""
        try:
            async with self.connection_pool.acquire() as conn:
                # Get current metadata
                current_row = await conn.fetchrow(
                    f"SELECT metadata FROM {self.table_name} WHERE content_cid = $1",
                    content_cid
                )
                
                if not current_row:
                    logger.warning(f"Content {content_cid} not found for metadata update")
                    return False
                
                # Merge with existing metadata
                current_metadata = json.loads(current_row['metadata']) if current_row['metadata'] else {}
                current_metadata.update(metadata_updates)
                
                # Update the record
                await conn.execute(
                    f"UPDATE {self.table_name} SET metadata = $1, updated_at = NOW() WHERE content_cid = $2",
                    json.dumps(current_metadata),
                    content_cid
                )
                
                logger.debug(f"Updated metadata for content {content_cid}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update metadata for {content_cid}: {e}")
            return False
    
    async def delete_content(self, content_cid: str) -> bool:
        """Delete content from vector store (for DMCA compliance)"""
        try:
            async with self.connection_pool.acquire() as conn:
                result = await conn.execute(
                    f"DELETE FROM {self.table_name} WHERE content_cid = $1",
                    content_cid
                )
                
                deleted_count = int(result.split()[-1])
                logger.info(f"Deleted {deleted_count} records for content {content_cid}")
                return deleted_count > 0
                
        except Exception as e:
            logger.error(f"Failed to delete content {content_cid}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            async with self.connection_pool.acquire() as conn:
                stats_sql = f"""
                SELECT 
                    COUNT(*) as total_vectors,
                    COUNT(DISTINCT creator_id) as unique_creators,
                    COUNT(DISTINCT content_type) as content_types,
                    AVG(citation_count) as avg_citations,
                    AVG(quality_score) as avg_quality,
                    MAX(created_at) as latest_content,
                    MIN(created_at) as earliest_content,
                    SUM(access_count) as total_accesses
                FROM {self.table_name}
                WHERE active = TRUE
                """
                
                stats_row = await conn.fetchrow(stats_sql)
                
                # Get table size
                table_size_mb = await self._get_table_size_mb(conn)
                
                return {
                    "total_vectors": stats_row['total_vectors'] or 0,
                    "unique_creators": stats_row['unique_creators'] or 0,
                    "content_types": stats_row['content_types'] or 0,
                    "average_citations": float(stats_row['avg_citations']) if stats_row['avg_citations'] else 0.0,
                    "average_quality": float(stats_row['avg_quality']) if stats_row['avg_quality'] else 0.0,
                    "latest_content": stats_row['latest_content'],
                    "earliest_content": stats_row['earliest_content'],
                    "total_accesses": stats_row['total_accesses'] or 0,
                    "table_size_mb": table_size_mb
                }
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def _ensure_table_exists(self):
        """Ensure the main table exists - handled by init scripts"""
        # Table creation is handled by the database initialization scripts
        # This method is a placeholder for compatibility
        pass
    
    async def _track_content_access_batch(self, conn, content_cids: List[str]):
        """Update access counts for content that was retrieved"""
        try:
            # Use the PostgreSQL function we created in the init script
            for content_cid in content_cids:
                await conn.execute(
                    "SELECT prsm_vector.track_content_access($1, $2)",
                    content_cid,
                    "system"  # Default user for system access
                )
        except Exception as e:
            # Don't fail the search if access tracking fails
            logger.warning(f"Failed to update access counts: {e}")
    
    async def _get_table_size_mb(self, conn) -> float:
        """Get approximate table size in MB"""
        try:
            # Query the full table name with schema
            size_result = await conn.fetchrow(
                "SELECT pg_total_relation_size($1) as size_bytes",
                self.table_name
            )
            return float(size_result['size_bytes']) / (1024 * 1024) if size_result and size_result['size_bytes'] else 0.0
        except Exception as e:
            logger.debug(f"Could not get table size: {e}")
            return 0.0


# Utility functions for easy testing and deployment

async def create_development_pgvector_store(host: str = "localhost", port: int = 5433) -> PgVectorStore:
    """
    Create a pgvector store for development using our Docker setup
    
    Args:
        host: Database host (default: localhost)  
        port: Database port (default: 5433 for docker-compose.vector.yml)
        
    Returns:
        Connected PgVectorStore instance ready for use
    """
    config = VectorStoreConfig(
        store_type=VectorStoreType.PGVECTOR,
        host=host,
        port=port,
        database="prsm_vector_dev",
        username="postgres",
        password="postgres",
        collection_name="prsm_vectors",
        vector_dimension=384,
        max_connections=10
    )
    
    store = PgVectorStore(config)
    connected = await store.connect()
    
    if not connected:
        raise ConnectionError(
            f"Failed to connect to PostgreSQL at {host}:{port}\n"
            "Make sure the database is running:\n"
            "docker-compose -f docker-compose.vector.yml up postgres-vector"
        )
    
    return store


async def create_production_pgvector_store(database_url: str) -> PgVectorStore:
    """
    Create a pgvector store for production deployment
    
    Args:
        database_url: Full PostgreSQL connection URL 
                     (postgresql://user:pass@host:port/database)
        
    Returns:
        Connected PgVectorStore instance ready for production use
    """
    from urllib.parse import urlparse
    parsed = urlparse(database_url)
    
    config = VectorStoreConfig(
        store_type=VectorStoreType.PGVECTOR,
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path[1:] if parsed.path else "prsm_vector",
        username=parsed.username,
        password=parsed.password,
        collection_name="prsm_vectors",
        vector_dimension=384,
        max_connections=50  # Higher for production
    )
    
    store = PgVectorStore(config)
    connected = await store.connect()
    
    if not connected:
        raise ConnectionError(f"Failed to connect to production database: {database_url}")
    
    return store


def get_docker_connection_info() -> dict:
    """Get connection information for Docker setup"""
    return {
        "host": "localhost",
        "port": 5433,  # Our docker-compose.vector.yml port
        "database": "prsm_vector_dev",
        "username": "postgres", 
        "password": "postgres",
        "docker_command": "docker-compose -f docker-compose.vector.yml up postgres-vector",
        "pgadmin_url": "http://localhost:8081",
        "pgadmin_credentials": "admin@prsm.ai / admin"
    }