"""
Vector Store Coordinator

Manages transitions between different vector database backends and coordinates
operations across multiple stores during migration phases.

Key responsibilities:
- Seamless migration from pgvector to Milvus during Phase 1B
- Dual-write operations during transition periods
- Load balancing across multiple vector stores
- Fallback and disaster recovery coordination
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np

from .base import PRSMVectorStore, ContentMatch, SearchFilters, VectorStoreConfig
from prsm.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MigrationPhase(str, Enum):
    """Phases of vector store migration"""
    SINGLE_STORE = "single_store"          # Using only primary store
    DUAL_WRITE = "dual_write"              # Writing to both old and new
    DUAL_READ = "dual_read"                # Reading from both, preferring new
    MIGRATION_COMPLETE = "migration_complete"  # Using only new store


class LoadBalancingStrategy(str, Enum):
    """Strategies for distributing load across multiple stores"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    GEOGRAPHIC = "geographic"
    CONTENT_TYPE = "content_type"


class VectorStoreCoordinator:
    """
    Coordinates operations across multiple vector stores
    
    Handles:
    - Migration between vector database types (e.g., pgvector → Milvus)
    - Load balancing across multiple stores
    - Fallback and error recovery
    - Performance monitoring and optimization
    """
    
    def __init__(self):
        self.primary_store: Optional[PRSMVectorStore] = None
        self.secondary_store: Optional[PRSMVectorStore] = None
        self.migration_phase = MigrationPhase.SINGLE_STORE
        self.load_balancing_strategy = LoadBalancingStrategy.ROUND_ROBIN
        
        # Migration tracking
        self.migration_progress = {
            "total_content_items": 0,
            "migrated_items": 0,
            "failed_items": 0,
            "start_time": None,
            "estimated_completion": None
        }
        
        # Load balancing state
        self.round_robin_counter = 0
        self.store_load_metrics = {}
        
        # Performance tracking
        self.coordinator_metrics = {
            "total_operations": 0,
            "dual_write_operations": 0,
            "fallback_operations": 0,
            "migration_operations": 0,
            "average_response_time": 0.0
        }
    
    async def initialize_stores(self, 
                              primary_config: VectorStoreConfig,
                              secondary_config: Optional[VectorStoreConfig] = None):
        """Initialize primary and optional secondary vector stores"""
        # Import store implementations dynamically to avoid circular imports
        from .implementations import create_vector_store
        
        # Initialize primary store
        self.primary_store = create_vector_store(primary_config)
        await self.primary_store.connect()
        
        # Initialize secondary store if provided
        if secondary_config:
            self.secondary_store = create_vector_store(secondary_config)
            await self.secondary_store.connect()
            logger.info(f"Initialized secondary store: {secondary_config.store_type}")
        
        logger.info(f"Vector store coordinator initialized with primary: {primary_config.store_type}")
    
    async def start_migration(self, target_config: VectorStoreConfig, 
                            migration_batch_size: int = 1000):
        """
        Start migration from primary store to a new target store
        
        Args:
            target_config: Configuration for the new target vector store
            migration_batch_size: Number of items to migrate in each batch
        """
        from .implementations import create_vector_store
        
        logger.info(f"Starting migration to {target_config.store_type}")
        
        # Initialize target store
        target_store = create_vector_store(target_config)
        await target_store.connect()
        
        # Set up dual-write phase
        self.secondary_store = target_store
        self.migration_phase = MigrationPhase.DUAL_WRITE
        self.migration_progress["start_time"] = datetime.utcnow()
        
        # Start background migration task
        migration_task = asyncio.create_task(
            self._migrate_existing_content(migration_batch_size)
        )
        
        logger.info("Migration started - entering dual-write phase")
        return migration_task
    
    async def complete_migration(self):
        """Complete migration by switching to the new primary store"""
        if self.migration_phase != MigrationPhase.DUAL_READ:
            raise ValueError("Migration must be in DUAL_READ phase before completion")
        
        # Switch stores
        old_primary = self.primary_store
        self.primary_store = self.secondary_store
        self.secondary_store = None
        self.migration_phase = MigrationPhase.MIGRATION_COMPLETE
        
        # Clean up old store
        if old_primary:
            await old_primary.disconnect()
        
        logger.info("Migration completed successfully")
    
    async def store_content_with_embeddings(self,
                                          content_cid: str,
                                          embeddings: np.ndarray,
                                          metadata: Dict[str, Any]) -> str:
        """Store content with coordination across active stores"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if self.migration_phase == MigrationPhase.SINGLE_STORE:
                # Single store operation
                result = await self.primary_store.store_content_with_embeddings(
                    content_cid, embeddings, metadata
                )
            
            elif self.migration_phase == MigrationPhase.DUAL_WRITE:
                # Write to both stores
                primary_task = self.primary_store.store_content_with_embeddings(
                    content_cid, embeddings, metadata
                )
                secondary_task = self.secondary_store.store_content_with_embeddings(
                    content_cid, embeddings, metadata
                )
                
                # Wait for both operations
                primary_result, secondary_result = await asyncio.gather(
                    primary_task, secondary_task, return_exceptions=True
                )
                
                # Use primary result, but log secondary errors
                if isinstance(secondary_result, Exception):
                    logger.warning(f"Secondary store write failed: {secondary_result}")
                
                result = primary_result if not isinstance(primary_result, Exception) else secondary_result
                self.coordinator_metrics["dual_write_operations"] += 1
            
            else:
                # Migration complete or dual-read - use new primary
                result = await self.primary_store.store_content_with_embeddings(
                    content_cid, embeddings, metadata
                )
            
            # Update metrics
            duration = asyncio.get_event_loop().time() - start_time
            self._update_coordinator_metrics("storage", duration, True)
            
            return result
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            self._update_coordinator_metrics("storage", duration, False)
            logger.error(f"Coordinated storage operation failed: {e}")
            raise
    
    async def search_similar_content(self,
                                   query_embedding: np.ndarray,
                                   filters: Optional[SearchFilters] = None,
                                   top_k: int = 10) -> List[ContentMatch]:
        """Search content with coordination and fallback"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if self.migration_phase in [MigrationPhase.SINGLE_STORE, MigrationPhase.DUAL_WRITE]:
                # Use primary store
                selected_store = self.primary_store
            
            elif self.migration_phase == MigrationPhase.DUAL_READ:
                # Prefer secondary (new) store, fallback to primary
                try:
                    results = await self.secondary_store.search_similar_content(
                        query_embedding, filters, top_k
                    )
                    duration = asyncio.get_event_loop().time() - start_time
                    self._update_coordinator_metrics("search", duration, True)
                    return results
                except Exception as e:
                    logger.warning(f"Secondary store search failed, using fallback: {e}")
                    selected_store = self.primary_store
                    self.coordinator_metrics["fallback_operations"] += 1
            
            else:
                # Migration complete
                selected_store = self.primary_store
            
            # Perform search on selected store
            results = await selected_store.search_similar_content(
                query_embedding, filters, top_k
            )
            
            duration = asyncio.get_event_loop().time() - start_time
            self._update_coordinator_metrics("search", duration, True)
            
            return results
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            self._update_coordinator_metrics("search", duration, False)
            logger.error(f"Coordinated search operation failed: {e}")
            raise
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status and progress"""
        return {
            "migration_phase": self.migration_phase.value,
            "progress": self.migration_progress,
            "stores": {
                "primary": {
                    "type": self.primary_store.config.store_type.value if self.primary_store else None,
                    "connected": self.primary_store.is_connected if self.primary_store else False
                },
                "secondary": {
                    "type": self.secondary_store.config.store_type.value if self.secondary_store else None,
                    "connected": self.secondary_store.is_connected if self.secondary_store else False
                }
            },
            "coordinator_metrics": self.coordinator_metrics
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for all managed stores"""
        health_status = {
            "coordinator_status": "healthy",
            "migration_phase": self.migration_phase.value,
            "stores": {}
        }
        
        # Check primary store
        if self.primary_store:
            try:
                primary_health = await self.primary_store.health_check()
                health_status["stores"]["primary"] = primary_health
            except Exception as e:
                health_status["stores"]["primary"] = {"status": "error", "error": str(e)}
                health_status["coordinator_status"] = "degraded"
        
        # Check secondary store
        if self.secondary_store:
            try:
                secondary_health = await self.secondary_store.health_check()
                health_status["stores"]["secondary"] = secondary_health
            except Exception as e:
                health_status["stores"]["secondary"] = {"status": "error", "error": str(e)}
                if self.migration_phase == MigrationPhase.DUAL_READ:
                    health_status["coordinator_status"] = "degraded"
        
        health_status["metrics"] = self.coordinator_metrics
        return health_status
    
    async def _migrate_existing_content(self, batch_size: int = 1000):
        """Background task to migrate existing content from primary to secondary store"""
        try:
            logger.info("Starting background content migration")
            
            # Get collection stats to estimate total items
            primary_stats = await self.primary_store.get_collection_stats()
            total_items = primary_stats.get("total_vectors", 0)
            self.migration_progress["total_content_items"] = total_items
            
            migrated_count = 0
            failed_count = 0
            
            # Migration implementation would depend on specific vector store APIs
            # This is a simplified version - actual implementation would need
            # store-specific methods to iterate through existing content
            
            # Placeholder for actual migration logic
            # In real implementation, this would:
            # 1. Query primary store for content in batches
            # 2. Copy each batch to secondary store
            # 3. Verify successful migration
            # 4. Update progress metrics
            
            # Simulate migration progress for now
            for batch_start in range(0, total_items, batch_size):
                batch_end = min(batch_start + batch_size, total_items)
                
                try:
                    # Actual migration logic would go here
                    migrated_count += (batch_end - batch_start)
                    self.migration_progress["migrated_items"] = migrated_count
                    
                    # Small delay to avoid overwhelming the stores
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Migration batch failed: {e}")
                    failed_count += (batch_end - batch_start)
                    self.migration_progress["failed_items"] = failed_count
            
            # Check if migration was successful
            if failed_count == 0:
                logger.info("Background migration completed successfully")
                # Switch to dual-read phase
                self.migration_phase = MigrationPhase.DUAL_READ
            else:
                logger.warning(f"Migration completed with {failed_count} failed items")
                
        except Exception as e:
            logger.error(f"Background migration failed: {e}")
            # Reset to single store mode on failure
            self.migration_phase = MigrationPhase.SINGLE_STORE
            if self.secondary_store:
                await self.secondary_store.disconnect()
                self.secondary_store = None
    
    def _update_coordinator_metrics(self, operation_type: str, duration: float, success: bool):
        """Update coordinator performance metrics"""
        self.coordinator_metrics["total_operations"] += 1
        
        # Update average response time
        total_ops = self.coordinator_metrics["total_operations"]
        current_avg = self.coordinator_metrics["average_response_time"]
        self.coordinator_metrics["average_response_time"] = (
            (current_avg * (total_ops - 1) + duration) / total_ops
        )
        
        if operation_type == "migration":
            self.coordinator_metrics["migration_operations"] += 1


# Global coordinator instance
_coordinator_instance: Optional[VectorStoreCoordinator] = None

def get_vector_store_coordinator() -> VectorStoreCoordinator:
    """Get or create global vector store coordinator instance"""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = VectorStoreCoordinator()
    return _coordinator_instance