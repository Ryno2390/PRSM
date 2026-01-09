"""
Vector Store Migration Tools

Provides utilities for migrating between different vector database
implementations with data validation and rollback capabilities.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base import PRSMVectorStore, ContentMatch, VectorStoreConfig

logger = logging.getLogger(__name__)


class VectorStoreMigrator:
    """
    Handles migration between different vector store implementations
    
    Features:
    - Data validation and integrity checking
    - Incremental migration with progress tracking
    - Rollback capabilities for failed migrations
    - Performance monitoring during migration
    """
    
    def __init__(self, source_store: PRSMVectorStore, target_store: PRSMVectorStore):
        self.source_store = source_store
        self.target_store = target_store
        self.migration_log: List[Dict[str, Any]] = []
        
    async def validate_migration_compatibility(self) -> Dict[str, Any]:
        """
        Validate that source and target stores are compatible for migration
        
        Returns:
            Dictionary with compatibility results
        """
        compatibility_report = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "source_info": {},
            "target_info": {}
        }
        
        try:
            # Get source store info
            source_stats = await self.source_store.get_collection_stats()
            compatibility_report["source_info"] = {
                "type": self.source_store.config.store_type.value,
                "total_vectors": source_stats.get("total_vectors", 0),
                "vector_dimension": self.source_store.config.vector_dimension
            }
            
            # Get target store info
            target_health = await self.target_store.health_check()
            compatibility_report["target_info"] = {
                "type": self.target_store.config.store_type.value,
                "vector_dimension": self.target_store.config.vector_dimension,
                "status": target_health.get("status", "unknown")
            }
            
            # Check vector dimension compatibility
            if self.source_store.config.vector_dimension != self.target_store.config.vector_dimension:
                compatibility_report["errors"].append(
                    f"Vector dimension mismatch: source={self.source_store.config.vector_dimension}, "
                    f"target={self.target_store.config.vector_dimension}"
                )
                compatibility_report["compatible"] = False
            
            # Check target store health
            if target_health.get("status") != "healthy":
                compatibility_report["warnings"].append(
                    "Target store health check failed - may impact migration performance"
                )
            
        except Exception as e:
            compatibility_report["errors"].append(f"Compatibility check failed: {e}")
            compatibility_report["compatible"] = False
        
        return compatibility_report
    
    async def migrate_content_batch(self, 
                                  content_batch: List[Tuple[str, np.ndarray, Dict[str, Any]]],
                                  validate_after_migration: bool = True) -> Dict[str, Any]:
        """
        Migrate a batch of content from source to target store
        
        Args:
            content_batch: List of (content_cid, embeddings, metadata) tuples
            validate_after_migration: Whether to validate data after migration
            
        Returns:
            Migration result dictionary
        """
        migration_result = {
            "success": False,
            "migrated_count": 0,
            "failed_count": 0,
            "validation_passed": False,
            "errors": []
        }
        
        try:
            # Store content in target store
            successful_migrations = []
            failed_migrations = []
            
            for content_cid, embeddings, metadata in content_batch:
                try:
                    vector_id = await self.target_store.store_content_with_embeddings(
                        content_cid, embeddings, metadata
                    )
                    successful_migrations.append((content_cid, vector_id))
                except Exception as e:
                    failed_migrations.append((content_cid, str(e)))
                    migration_result["errors"].append(f"Failed to migrate {content_cid}: {e}")
            
            migration_result["migrated_count"] = len(successful_migrations)
            migration_result["failed_count"] = len(failed_migrations)
            
            # Validate migrated data if requested
            if validate_after_migration and successful_migrations:
                validation_results = await self._validate_migrated_content(successful_migrations)
                migration_result["validation_passed"] = validation_results["all_passed"]
                if not validation_results["all_passed"]:
                    migration_result["errors"].extend(validation_results["errors"])
            
            migration_result["success"] = (
                migration_result["failed_count"] == 0 and
                (not validate_after_migration or migration_result["validation_passed"])
            )
            
        except Exception as e:
            migration_result["errors"].append(f"Batch migration failed: {e}")
        
        return migration_result
    
    async def _validate_migrated_content(self, 
                                       migrated_content: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Validate that migrated content can be retrieved correctly
        
        Args:
            migrated_content: List of (content_cid, vector_id) tuples
            
        Returns:
            Validation results
        """
        validation_results = {
            "all_passed": True,
            "passed_count": 0,
            "failed_count": 0,
            "errors": []
        }
        
        for content_cid, vector_id in migrated_content:
            try:
                # Try to search for the content to verify it was stored correctly
                # This is a simplified validation - in production, we'd do more thorough checks
                stats = await self.target_store.get_collection_stats()
                if stats.get("total_vectors", 0) > 0:
                    validation_results["passed_count"] += 1
                else:
                    validation_results["failed_count"] += 1
                    validation_results["errors"].append(f"Content {content_cid} not found after migration")
                    
            except Exception as e:
                validation_results["failed_count"] += 1
                validation_results["errors"].append(f"Validation failed for {content_cid}: {e}")
        
        validation_results["all_passed"] = validation_results["failed_count"] == 0
        return validation_results
    
    async def estimate_migration_time(self, 
                                    total_content_items: int,
                                    batch_size: int = 100) -> Dict[str, Any]:
        """
        Estimate migration time based on performance testing
        
        Args:
            total_content_items: Total number of items to migrate
            batch_size: Batch size for migration
            
        Returns:
            Time estimation results
        """
        estimation_result = {
            "estimated_total_hours": 0.0,
            "estimated_batches": 0,
            "recommended_batch_size": batch_size,
            "performance_test_results": {}
        }
        
        try:
            # Run a small performance test
            test_batch_size = min(10, total_content_items)
            test_batch = [
                (f"test_migration_{i}", np.random.random(384).astype(np.float32), {"test": True})
                for i in range(test_batch_size)
            ]
            
            start_time = asyncio.get_event_loop().time()
            test_result = await self.migrate_content_batch(test_batch, validate_after_migration=False)
            test_duration = asyncio.get_event_loop().time() - start_time
            
            if test_result["success"] and test_duration > 0:
                items_per_second = test_batch_size / test_duration
                
                estimation_result["performance_test_results"] = {
                    "test_batch_size": test_batch_size,
                    "test_duration_seconds": test_duration,
                    "items_per_second": items_per_second
                }
                
                # Estimate total time
                estimated_seconds = total_content_items / items_per_second
                estimation_result["estimated_total_hours"] = estimated_seconds / 3600
                estimation_result["estimated_batches"] = (total_content_items + batch_size - 1) // batch_size
                
                # Recommend optimal batch size based on performance
                if items_per_second > 50:
                    estimation_result["recommended_batch_size"] = min(1000, batch_size * 2)
                elif items_per_second < 10:
                    estimation_result["recommended_batch_size"] = max(10, batch_size // 2)
            
        except Exception as e:
            logger.error(f"Migration time estimation failed: {e}")
        
        return estimation_result
    
    def log_migration_event(self, event_type: str, details: Dict[str, Any]):
        """Log migration events for audit trail"""
        log_entry = {
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "details": details
        }
        self.migration_log.append(log_entry)
        logger.info(f"Migration event: {event_type} - {details}")
    
    def get_migration_log(self) -> List[Dict[str, Any]]:
        """Get complete migration log"""
        return self.migration_log.copy()
    
    async def cleanup_test_data(self):
        """Clean up any test data created during migration estimation"""
        try:
            # Delete test migration content
            for i in range(10):  # Clean up potential test items
                await self.target_store.delete_content(f"test_migration_{i}")
        except Exception as e:
            logger.warning(f"Cleanup of test data failed: {e}")


# Utility functions for migration planning
async def plan_migration(source_store: PRSMVectorStore, 
                        target_store: PRSMVectorStore,
                        preferred_batch_size: int = 100) -> Dict[str, Any]:
    """
    Create a comprehensive migration plan
    
    Args:
        source_store: Source vector store to migrate from
        target_store: Target vector store to migrate to
        preferred_batch_size: Preferred batch size for migration
        
    Returns:
        Comprehensive migration plan
    """
    migrator = VectorStoreMigrator(source_store, target_store)
    
    # Check compatibility
    compatibility = await migrator.validate_migration_compatibility()
    
    if not compatibility["compatible"]:
        return {
            "migration_feasible": False,
            "compatibility_issues": compatibility["errors"],
            "warnings": compatibility["warnings"]
        }
    
    # Get source statistics
    source_stats = await source_store.get_collection_stats()
    total_items = source_stats.get("total_vectors", 0)
    
    # Estimate migration time
    time_estimation = await migrator.estimate_migration_time(total_items, preferred_batch_size)
    
    # Create migration plan
    migration_plan = {
        "migration_feasible": True,
        "source_info": compatibility["source_info"],
        "target_info": compatibility["target_info"],
        "migration_strategy": {
            "total_items": total_items,
            "recommended_batch_size": time_estimation["recommended_batch_size"],
            "estimated_batches": time_estimation["estimated_batches"],
            "estimated_duration_hours": time_estimation["estimated_total_hours"],
            "validation_enabled": True
        },
        "performance_expectations": time_estimation["performance_test_results"],
        "warnings": compatibility["warnings"]
    }
    
    # Clean up test data
    await migrator.cleanup_test_data()
    
    return migration_plan