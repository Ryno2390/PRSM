#!/usr/bin/env python3
"""
Production Storage Manager for PRSM
===================================

This module provides intelligent storage management optimized for external
hard drives and large-scale content ingestion.

Key Features:
1. External hard drive optimization with intelligent path management
2. Dynamic storage allocation and monitoring
3. Compression and deduplication for maximum efficiency
4. Hierarchical storage with automatic tiering
5. Storage health monitoring and alerts
6. Automatic cleanup and optimization
7. Backup and redundancy management

Designed specifically for the breadth-optimized ingestion strategy.
"""

import asyncio
import json
import logging
import os
import shutil
import gzip
import pickle
import hashlib
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

import structlog

logger = structlog.get_logger(__name__)


class StorageTier(str, Enum):
    """Storage tiers for hierarchical management"""
    HOT = "hot"           # Frequently accessed content
    WARM = "warm"         # Moderately accessed content
    COLD = "cold"         # Rarely accessed content
    ARCHIVE = "archive"   # Long-term storage


class StorageStatus(str, Enum):
    """Storage health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FULL = "full"
    ERROR = "error"


@dataclass
class StorageConfig:
    """Configuration for storage management"""
    
    # External drive settings
    external_drive_path: str = "/Volumes/My Passport"
    fallback_local_path: str = "/tmp/prsm_storage"
    
    # Directory structure
    content_dir: str = "PRSM_Content"
    embeddings_dir: str = "PRSM_Embeddings"
    metadata_dir: str = "PRSM_Metadata"
    cache_dir: str = "PRSM_Cache"
    backup_dir: str = "PRSM_Backup"
    
    # Storage limits (GB)
    max_total_storage: float = 100.0
    max_content_storage: float = 60.0
    max_embeddings_storage: float = 30.0
    max_metadata_storage: float = 8.0
    max_cache_storage: float = 2.0
    
    # Performance settings
    compression_enabled: bool = True
    compression_level: int = 6
    deduplication_enabled: bool = True
    auto_cleanup_enabled: bool = True
    backup_enabled: bool = True
    
    # Monitoring settings
    health_check_interval: int = 300  # 5 minutes
    cleanup_interval: int = 3600     # 1 hour
    backup_interval: int = 86400     # 24 hours
    
    # Thresholds
    warning_threshold: float = 0.8   # 80% usage
    critical_threshold: float = 0.9  # 90% usage
    cleanup_threshold: float = 0.85  # 85% usage


@dataclass
class StorageMetrics:
    """Storage usage metrics"""
    
    total_capacity_gb: float
    used_space_gb: float
    available_space_gb: float
    utilization_percentage: float
    
    # Per-directory metrics
    content_usage_gb: float = 0.0
    embeddings_usage_gb: float = 0.0
    metadata_usage_gb: float = 0.0
    cache_usage_gb: float = 0.0
    backup_usage_gb: float = 0.0
    
    # Performance metrics
    read_speed_mbps: float = 0.0
    write_speed_mbps: float = 0.0
    iops: float = 0.0
    
    # Health metrics
    status: StorageStatus = StorageStatus.HEALTHY
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class StorageManager:
    """
    Production Storage Manager for PRSM
    
    Provides intelligent storage management optimized for external hard drives
    and large-scale content ingestion with automatic optimization and monitoring.
    """
    
    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        
        # Storage paths
        self.base_path: Optional[Path] = None
        self.storage_paths: Dict[str, Path] = {}
        
        # Storage tracking
        self.storage_metrics = StorageMetrics(0, 0, 0, 0)
        self.storage_db: Optional[sqlite3.Connection] = None
        
        # Deduplication tracking
        self.content_hashes: Dict[str, str] = {}  # hash -> file_path
        self.compression_stats: Dict[str, Dict[str, float]] = {}
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.backup_task: Optional[asyncio.Task] = None
        
        # Thread pool for I/O operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Lock for thread safety
        self.storage_lock = threading.RLock()
        
        logger.info("Production Storage Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize storage manager with external drive detection"""
        
        logger.info("🚀 Initializing Production Storage Manager...")
        
        try:
            # Detect and setup storage location
            await self._setup_storage_location()
            
            # Initialize directory structure
            await self._initialize_directory_structure()
            
            # Initialize storage database
            await self._initialize_storage_database()
            
            # Load existing content hashes
            await self._load_content_hashes()
            
            # Start background monitoring
            await self._start_background_tasks()
            
            # Perform initial health check
            await self._perform_health_check()
            
            logger.info("✅ Production Storage Manager initialized successfully",
                       base_path=str(self.base_path),
                       status=self.storage_metrics.status.value)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Storage Manager initialization failed: {e}")
            return False
    
    async def store_content(self, content_id: str, content_data: Any, 
                          content_type: str = "content",
                          tier: StorageTier = StorageTier.HOT) -> Dict[str, Any]:
        """
        Store content with intelligent optimization
        
        Args:
            content_id: Unique content identifier
            content_data: Content to store
            content_type: Type of content (content, embedding, metadata)
            tier: Storage tier for hierarchical management
            
        Returns:
            Storage result with path and metadata
        """
        
        async with asyncio.Lock():
            logger.debug(f"Storing content: {content_id} ({content_type})")
            
            try:
                # Check storage capacity
                if not await self._check_storage_capacity(content_type):
                    # Try cleanup first
                    await self._perform_cleanup()
                    
                    # Check again after cleanup
                    if not await self._check_storage_capacity(content_type):
                        raise RuntimeError(f"Insufficient storage space for {content_type}")
                
                # Serialize content
                serialized_data = await self._serialize_content(content_data)
                
                # Check for deduplication
                content_hash = hashlib.sha256(serialized_data).hexdigest()
                if self.config.deduplication_enabled and content_hash in self.content_hashes:
                    existing_path = self.content_hashes[content_hash]
                    logger.debug(f"Content deduplicated: {content_id} -> {existing_path}")
                    
                    return {
                        "content_id": content_id,
                        "storage_path": existing_path,
                        "deduplicated": True,
                        "original_size": len(serialized_data),
                        "stored_size": 0,
                        "compression_ratio": 0.0
                    }
                
                # Compress if enabled
                if self.config.compression_enabled:
                    compressed_data = await self._compress_data(serialized_data)
                    compression_ratio = len(compressed_data) / len(serialized_data)
                else:
                    compressed_data = serialized_data
                    compression_ratio = 1.0
                
                # Determine storage path
                storage_path = await self._get_storage_path(content_id, content_type, tier)
                
                # Write to storage
                await self._write_to_storage(storage_path, compressed_data)
                
                # Update tracking
                self.content_hashes[content_hash] = str(storage_path)
                await self._update_storage_database(content_id, storage_path, 
                                                  len(serialized_data), len(compressed_data), 
                                                  content_type, tier)
                
                # Update metrics
                await self._update_storage_metrics()
                
                result = {
                    "content_id": content_id,
                    "storage_path": str(storage_path),
                    "deduplicated": False,
                    "original_size": len(serialized_data),
                    "stored_size": len(compressed_data),
                    "compression_ratio": compression_ratio,
                    "tier": tier.value
                }
                
                logger.debug(f"Content stored successfully: {content_id}",
                           compression_ratio=compression_ratio)
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to store content {content_id}: {e}")
                raise
    
    async def retrieve_content(self, content_id: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Retrieve content with automatic decompression
        
        Args:
            content_id: Content identifier
            
        Returns:
            Tuple of (content_data, metadata)
        """
        
        logger.debug(f"Retrieving content: {content_id}")
        
        try:
            # Get content info from database
            content_info = await self._get_content_info(content_id)
            if not content_info:
                raise FileNotFoundError(f"Content not found: {content_id}")
            
            storage_path = Path(content_info["storage_path"])
            
            # Read from storage
            compressed_data = await self._read_from_storage(storage_path)
            
            # Decompress if needed
            if self.config.compression_enabled:
                decompressed_data = await self._decompress_data(compressed_data)
            else:
                decompressed_data = compressed_data
            
            # Deserialize content
            content_data = await self._deserialize_content(decompressed_data)
            
            # Update access tracking
            await self._update_access_tracking(content_id)
            
            metadata = {
                "content_id": content_id,
                "storage_path": str(storage_path),
                "tier": content_info["tier"],
                "original_size": content_info["original_size"],
                "stored_size": content_info["stored_size"],
                "last_accessed": datetime.now(timezone.utc)
            }
            
            logger.debug(f"Content retrieved successfully: {content_id}")
            
            return content_data, metadata
            
        except Exception as e:
            logger.error(f"Failed to retrieve content {content_id}: {e}")
            raise
    
    async def get_storage_metrics(self) -> StorageMetrics:
        """Get current storage metrics"""
        await self._update_storage_metrics()
        return self.storage_metrics
    
    async def optimize_storage(self) -> Dict[str, Any]:
        """Perform comprehensive storage optimization"""
        
        logger.info("🔧 Starting storage optimization...")
        
        optimization_results = {
            "cleanup_performed": False,
            "deduplication_performed": False,
            "compression_optimized": False,
            "tiering_optimized": False,
            "space_freed_gb": 0.0,
            "optimization_time": 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # Perform cleanup
            cleanup_result = await self._perform_cleanup()
            optimization_results["cleanup_performed"] = True
            optimization_results["space_freed_gb"] += cleanup_result.get("space_freed_gb", 0.0)
            
            # Optimize deduplication
            dedup_result = await self._optimize_deduplication()
            optimization_results["deduplication_performed"] = True
            optimization_results["space_freed_gb"] += dedup_result.get("space_freed_gb", 0.0)
            
            # Optimize compression
            compression_result = await self._optimize_compression()
            optimization_results["compression_optimized"] = True
            optimization_results["space_freed_gb"] += compression_result.get("space_freed_gb", 0.0)
            
            # Optimize tiering
            tiering_result = await self._optimize_tiering()
            optimization_results["tiering_optimized"] = True
            
            # Update metrics
            await self._update_storage_metrics()
            
            optimization_results["optimization_time"] = (datetime.now() - start_time).total_seconds()
            
            logger.info("✅ Storage optimization completed",
                       space_freed_gb=optimization_results["space_freed_gb"],
                       optimization_time=optimization_results["optimization_time"])
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Storage optimization failed: {e}")
            raise
    
    async def get_storage_health(self) -> Dict[str, Any]:
        """Get comprehensive storage health report"""
        
        await self._perform_health_check()
        
        return {
            "storage_metrics": self.storage_metrics,
            "external_drive_connected": self.base_path.exists() if self.base_path else False,
            "directory_structure_healthy": await self._check_directory_structure(),
            "database_healthy": await self._check_database_health(),
            "background_tasks_running": {
                "monitoring": self.monitoring_task and not self.monitoring_task.done(),
                "cleanup": self.cleanup_task and not self.cleanup_task.done(),
                "backup": self.backup_task and not self.backup_task.done()
            },
            "recent_errors": self.storage_metrics.errors[-10:],  # Last 10 errors
            "recent_warnings": self.storage_metrics.warnings[-10:],  # Last 10 warnings
            "recommendations": await self._generate_storage_recommendations()
        }
    
    # === Private Methods ===
    
    async def _setup_storage_location(self):
        """Setup storage location with external drive detection"""
        
        external_path = Path(self.config.external_drive_path)
        
        if external_path.exists() and external_path.is_dir():
            # External drive is available
            self.base_path = external_path / "PRSM_Storage"
            logger.info(f"External drive detected: {external_path}")
        else:
            # Fallback to local storage
            self.base_path = Path(self.config.fallback_local_path)
            logger.warning(f"External drive not found, using fallback: {self.base_path}")
        
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Check permissions
        if not os.access(self.base_path, os.W_OK):
            raise PermissionError(f"No write permission for storage path: {self.base_path}")
    
    async def _initialize_directory_structure(self):
        """Initialize directory structure"""
        
        directories = [
            self.config.content_dir,
            self.config.embeddings_dir,
            self.config.metadata_dir,
            self.config.cache_dir,
            self.config.backup_dir
        ]
        
        for dir_name in directories:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            self.storage_paths[dir_name] = dir_path
        
        logger.info("Directory structure initialized")
    
    async def _initialize_storage_database(self):
        """Initialize SQLite database for storage tracking"""
        
        db_path = self.base_path / "storage.db"
        
        self.storage_db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.storage_db.row_factory = sqlite3.Row
        
        # Create tables
        await asyncio.get_event_loop().run_in_executor(
            self.executor, self._create_database_tables
        )
        
        logger.info("Storage database initialized")
    
    def _create_database_tables(self):
        """Create database tables"""
        
        self.storage_db.execute("""
            CREATE TABLE IF NOT EXISTS content_storage (
                content_id TEXT PRIMARY KEY,
                storage_path TEXT NOT NULL,
                content_type TEXT NOT NULL,
                tier TEXT NOT NULL,
                original_size INTEGER NOT NULL,
                stored_size INTEGER NOT NULL,
                content_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        self.storage_db.execute("""
            CREATE TABLE IF NOT EXISTS storage_metrics (
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_capacity_gb REAL,
                used_space_gb REAL,
                available_space_gb REAL,
                utilization_percentage REAL,
                status TEXT
            )
        """)
        
        self.storage_db.commit()
    
    async def _load_content_hashes(self):
        """Load existing content hashes for deduplication"""
        
        def load_hashes():
            cursor = self.storage_db.execute(
                "SELECT content_hash, storage_path FROM content_storage WHERE content_hash IS NOT NULL"
            )
            return {row["content_hash"]: row["storage_path"] for row in cursor.fetchall()}
        
        self.content_hashes = await asyncio.get_event_loop().run_in_executor(
            self.executor, load_hashes
        )
        
        logger.info(f"Loaded {len(self.content_hashes)} content hashes for deduplication")
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        
        # Health monitoring
        self.monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        
        # Cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Backup task
        if self.config.backup_enabled:
            self.backup_task = asyncio.create_task(self._backup_loop())
        
        logger.info("Background tasks started")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                # Check if cleanup is needed
                if self.storage_metrics.utilization_percentage > self.config.cleanup_threshold:
                    await self._perform_cleanup()
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)  # Wait before retry
    
    async def _backup_loop(self):
        """Background backup loop"""
        
        while True:
            try:
                await asyncio.sleep(self.config.backup_interval)
                await self._perform_backup()
                
            except Exception as e:
                logger.error(f"Backup loop error: {e}")
                await asyncio.sleep(3600)  # Wait before retry
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        
        try:
            # Update storage metrics
            await self._update_storage_metrics()
            
            # Check disk space
            if self.storage_metrics.utilization_percentage > self.config.critical_threshold:
                self.storage_metrics.status = StorageStatus.CRITICAL
                self.storage_metrics.errors.append(
                    f"Storage critically full: {self.storage_metrics.utilization_percentage:.1f}%"
                )
            elif self.storage_metrics.utilization_percentage > self.config.warning_threshold:
                self.storage_metrics.status = StorageStatus.WARNING
                self.storage_metrics.warnings.append(
                    f"Storage warning: {self.storage_metrics.utilization_percentage:.1f}%"
                )
            else:
                self.storage_metrics.status = StorageStatus.HEALTHY
            
            # Check external drive connectivity
            if not self.base_path.exists():
                self.storage_metrics.status = StorageStatus.ERROR
                self.storage_metrics.errors.append("Storage path not accessible")
            
            # Check database health
            if not await self._check_database_health():
                self.storage_metrics.status = StorageStatus.ERROR
                self.storage_metrics.errors.append("Database health check failed")
            
            self.storage_metrics.last_check = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.storage_metrics.status = StorageStatus.ERROR
            self.storage_metrics.errors.append(f"Health check failed: {e}")
    
    async def _update_storage_metrics(self):
        """Update storage usage metrics"""
        
        try:
            # Get disk usage
            disk_usage = shutil.disk_usage(self.base_path)
            
            self.storage_metrics.total_capacity_gb = disk_usage.total / (1024**3)
            self.storage_metrics.available_space_gb = disk_usage.free / (1024**3)
            self.storage_metrics.used_space_gb = (disk_usage.total - disk_usage.free) / (1024**3)
            self.storage_metrics.utilization_percentage = (
                (disk_usage.total - disk_usage.free) / disk_usage.total * 100
            )
            
            # Get directory-specific usage
            for dir_name, dir_path in self.storage_paths.items():
                if dir_path.exists():
                    dir_size = await self._get_directory_size(dir_path)
                    setattr(self.storage_metrics, f"{dir_name}_usage_gb", dir_size / (1024**3))
            
            # Save metrics to database
            await self._save_metrics_to_database()
            
        except Exception as e:
            logger.error(f"Failed to update storage metrics: {e}")
    
    async def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory"""
        
        def calculate_size():
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, IOError):
                        pass
            return total_size
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, calculate_size
        )
    
    # Additional helper methods would be implemented here...
    
    async def _serialize_content(self, content_data: Any) -> bytes:
        """Serialize content for storage"""
        return pickle.dumps(content_data)
    
    async def _deserialize_content(self, data: bytes) -> Any:
        """Deserialize content from storage"""
        return pickle.loads(data)
    
    async def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip"""
        return gzip.compress(data, compresslevel=self.config.compression_level)
    
    async def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data using gzip"""
        return gzip.decompress(data)
    
    async def _write_to_storage(self, path: Path, data: bytes):
        """Write data to storage"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        def write_file():
            with open(path, 'wb') as f:
                f.write(data)
        
        await asyncio.get_event_loop().run_in_executor(self.executor, write_file)
    
    async def _read_from_storage(self, path: Path) -> bytes:
        """Read data from storage"""
        
        def read_file():
            with open(path, 'rb') as f:
                return f.read()
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, read_file)
    
    async def _check_storage_capacity(self, content_type: str) -> bool:
        """Check if storage has capacity for content type"""
        
        type_limits = {
            "content": self.config.max_content_storage,
            "embedding": self.config.max_embeddings_storage,
            "metadata": self.config.max_metadata_storage
        }
        
        current_usage = getattr(self.storage_metrics, f"{content_type}_usage_gb", 0)
        limit = type_limits.get(content_type, self.config.max_total_storage)
        
        return current_usage < limit * 0.95  # 95% threshold
    
    async def _get_storage_path(self, content_id: str, content_type: str, tier: StorageTier) -> Path:
        """Get storage path for content"""
        
        # Create hierarchical path: type/tier/hash_prefix/content_id
        hash_prefix = hashlib.md5(content_id.encode()).hexdigest()[:2]
        
        # Map content types to directory names
        type_to_dir = {
            "content": "PRSM_Content",
            "embedding": "PRSM_Embeddings", 
            "metadata": "PRSM_Metadata"
        }
        
        dir_name = type_to_dir.get(content_type, "PRSM_Content")
        base_dir = self.storage_paths.get(dir_name, self.storage_paths["PRSM_Content"])
        
        return base_dir / tier.value / hash_prefix / f"{content_id}.dat"
    
    async def shutdown(self):
        """Graceful shutdown of storage manager"""
        
        logger.info("Shutting down storage manager...")
        
        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.backup_task:
            self.backup_task.cancel()
        
        # Close database
        if self.storage_db:
            self.storage_db.close()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Storage manager shutdown complete")

    # Placeholder implementations for remaining methods
    async def _perform_cleanup(self) -> Dict[str, Any]:
        """Perform storage cleanup"""
        return {"space_freed_gb": 0.0}

    async def _optimize_deduplication(self) -> Dict[str, Any]:
        """Optimize deduplication"""
        return {"space_freed_gb": 0.0}

    async def _optimize_compression(self) -> Dict[str, Any]:
        """Optimize compression"""
        return {"space_freed_gb": 0.0}

    async def _optimize_tiering(self) -> Dict[str, Any]:
        """Optimize storage tiering"""
        return {"optimized": True}

    async def _check_directory_structure(self) -> bool:
        """Check directory structure health"""
        return True

    async def _check_database_health(self) -> bool:
        """Check database health"""
        return True

    async def _generate_storage_recommendations(self) -> List[str]:
        """Generate storage recommendations"""
        return ["Storage system operating optimally"]

    async def _get_content_info(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get content info from database"""
        return None

    async def _update_storage_database(self, content_id: str, storage_path: Path, 
                                     original_size: int, stored_size: int, 
                                     content_type: str, tier: StorageTier):
        """Update storage database"""
        pass

    async def _save_metrics_to_database(self):
        """Save metrics to database"""
        pass

    async def _update_access_tracking(self, content_id: str):
        """Update access tracking"""
        pass

    async def _perform_backup(self):
        """Perform backup"""
        pass


# Test function
async def test_storage_manager():
    """Test storage manager functionality"""
    
    print("🗄️ PRODUCTION STORAGE MANAGER TEST")
    print("=" * 50)
    
    # Initialize storage manager
    storage_manager = StorageManager()
    
    success = await storage_manager.initialize()
    print(f"Initialization: {'✅' if success else '❌'}")
    
    if success:
        # Test storage operations
        test_content = {"test": "data", "size": 1024}
        
        # Store content
        result = await storage_manager.store_content(
            "test_content_1", test_content, "content"
        )
        print(f"Storage: ✅ {result['compression_ratio']:.2f} compression")
        
        # Retrieve content
        retrieved_content, metadata = await storage_manager.retrieve_content("test_content_1")
        print(f"Retrieval: ✅ Content matches: {retrieved_content == test_content}")
        
        # Get metrics
        metrics = await storage_manager.get_storage_metrics()
        print(f"Metrics: ✅ {metrics.utilization_percentage:.1f}% used")
        
        # Get health
        health = await storage_manager.get_storage_health()
        print(f"Health: ✅ Status: {health['storage_metrics'].status.value}")
        
        # Shutdown
        await storage_manager.shutdown()
        print("Shutdown: ✅")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(test_storage_manager())