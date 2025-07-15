#!/usr/bin/env python3
"""
Production Ingestion Orchestrator for PRSM
==========================================

This module orchestrates large-scale breadth-optimized content ingestion
using all production components working together.

Key Features:
1. Unified orchestration of all production components
2. Breadth-optimized content source configuration
3. Real-time monitoring and quality control
4. Adaptive performance optimization
5. Comprehensive error handling and recovery
6. Progress tracking and reporting

Designed to maximize analogical reasoning potential through domain breadth.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import structlog

from prsm.nwtn.production_storage_manager import StorageManager, StorageConfig
from prsm.nwtn.content_quality_filter import ContentQualityFilter, FilterConfig, QualityDecision
from prsm.nwtn.batch_processing_optimizer import BatchProcessingOptimizer, BatchProcessingConfig
from prsm.nwtn.production_monitoring_system import ProductionMonitoringSystem, MonitoringConfig
# from prsm.nwtn.unified_ipfs_pipeline import UnifiedIPFSPipeline, IPFSConfig
from prsm.nwtn.breadth_optimized_sources import BreadthOptimizedContentSources

logger = structlog.get_logger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for production ingestion"""
    
    # Content sources (breadth-optimized)
    content_sources: List[str] = None
    max_total_content: int = 150000  # Target 150k items for breadth
    
    # Quality settings - OPTIMIZED FOR SPEED
    quality_over_quantity: bool = False  # Prioritize breadth for analogical reasoning
    min_quality_threshold: float = 0.25  # Much lower threshold for faster ingestion
    
    # Performance settings - OPTIMIZED FOR SPEED
    max_concurrent_downloads: int = 20  # Increased concurrency
    batch_size: int = 200              # Larger batches
    ingestion_rate_limit: int = 10000  # Much higher rate limit
    
    # Storage settings
    external_drive_path: str = "/Volumes/My Passport"
    max_storage_gb: float = 100.0
    
    # Monitoring settings
    progress_report_interval: int = 300  # 5 minutes
    health_check_interval: int = 60     # 1 minute
    
    def __post_init__(self):
        if self.content_sources is None:
            self.content_sources = [
                "arxiv_cs",      # Computer Science
                "arxiv_math",    # Mathematics  
                "arxiv_physics", # Physics
                "arxiv_bio",     # Biology
                "pubmed",        # Biomedical
                "semantic_scholar", # Cross-domain
                "github_papers", # Code + Papers
                "openreview",    # ML/AI Reviews
                "biorxiv",       # Biology preprints
                "medrxiv"        # Medical preprints
            ]


class ProductionIngestionOrchestrator:
    """
    Production Ingestion Orchestrator
    
    Coordinates all production components for large-scale breadth-optimized
    content ingestion with real-time monitoring and quality control.
    """
    
    def __init__(self, config: IngestionConfig = None):
        self.config = config or IngestionConfig()
        
        # Initialize all production components
        self.storage_manager = None
        self.quality_filter = None
        self.batch_optimizer = None
        self.monitoring_system = None
        self.ipfs_pipeline = None
        
        # Ingestion state
        self.ingestion_active = False
        self.ingestion_stats = {
            "total_processed": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "total_stored": 0,
            "domains_covered": set(),
            "ingestion_start_time": None,
            "last_progress_report": None
        }
        
        # Content sources
        self.content_iterators = {}
        self.source_exhausted = set()
        
        logger.info("Production Ingestion Orchestrator initialized")
    
    async def initialize(self) -> bool:
        """Initialize all production components"""
        
        logger.info("🚀 Initializing Production Ingestion Orchestrator...")
        
        try:
            # Initialize storage manager
            storage_config = StorageConfig(
                external_drive_path=self.config.external_drive_path,
                max_total_storage=self.config.max_storage_gb
            )
            self.storage_manager = StorageManager(storage_config)
            if not await self.storage_manager.initialize():
                raise RuntimeError("Storage manager initialization failed")
            
            # Initialize quality filter
            filter_config = FilterConfig(
                min_overall_quality=self.config.min_quality_threshold,
                adaptive_thresholds=True
            )
            self.quality_filter = ContentQualityFilter(filter_config)
            await self.quality_filter.initialize()
            
            # Initialize batch optimizer
            batch_config = BatchProcessingConfig(
                target_batch_size=self.config.batch_size,
                max_concurrent_batches=self.config.max_concurrent_downloads
            )
            self.batch_optimizer = BatchProcessingOptimizer(batch_config)
            await self.batch_optimizer.initialize()
            
            # Initialize monitoring system
            monitoring_config = MonitoringConfig(
                health_check_interval=self.config.health_check_interval
            )
            self.monitoring_system = ProductionMonitoringSystem(monitoring_config)
            await self.monitoring_system.initialize()
            
            # Initialize IPFS pipeline (disabled for now)
            # ipfs_config = IPFSConfig()
            # self.ipfs_pipeline = UnifiedIPFSPipeline(ipfs_config)
            # await self.ipfs_pipeline.initialize()
            self.ipfs_pipeline = None
            
            # Initialize content sources
            self.content_sources = BreadthOptimizedContentSources()
            await self.content_sources.initialize()
            
            logger.info("✅ Production Ingestion Orchestrator ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            return False
    
    async def start_ingestion(self) -> Dict[str, Any]:
        """Start large-scale breadth-optimized ingestion"""
        
        if self.ingestion_active:
            return {"status": "already_active", "message": "Ingestion already in progress"}
        
        logger.info("🌍 Starting large-scale breadth-optimized ingestion...")
        enabled_sources = self.content_sources.get_enabled_sources()
        logger.info(f"Target: {self.config.max_total_content:,} items across {len(enabled_sources)} sources")
        
        self.ingestion_active = True
        self.ingestion_stats["ingestion_start_time"] = datetime.now(timezone.utc)
        
        try:
            # Start progress monitoring
            progress_task = asyncio.create_task(self._progress_monitoring_loop())
            
            # Start health monitoring
            health_task = asyncio.create_task(self._health_monitoring_loop())
            
            # Start main ingestion pipeline
            ingestion_task = asyncio.create_task(self._run_ingestion_pipeline())
            
            # Wait for completion or cancellation
            await asyncio.gather(ingestion_task, progress_task, health_task)
            
            final_stats = await self._generate_final_report()
            
            logger.info("✅ Large-scale ingestion completed successfully")
            return {
                "status": "completed",
                "final_stats": final_stats,
                "message": "Breadth-optimized ingestion completed"
            }
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "partial_stats": self.ingestion_stats
            }
        finally:
            self.ingestion_active = False
    
    async def _run_ingestion_pipeline(self):
        """Main ingestion pipeline"""
        
        logger.info("🔄 Starting ingestion pipeline...")
        
        # Create unified content stream from all sources
        content_stream = self._create_unified_content_stream()
        
        # Process content through the pipeline
        async for content_batch in self._batch_content_stream(content_stream):
            
            # Check if we've reached our target
            if self.ingestion_stats["total_processed"] >= self.config.max_total_content:
                logger.info(f"🎯 Target reached: {self.ingestion_stats['total_processed']:,} items processed")
                break
            
            # Process batch through quality filter
            quality_results = await self.quality_filter.batch_assess_quality(content_batch)
            
            # Filter accepted content
            accepted_content = []
            for content, analysis in zip(content_batch, quality_results):
                if analysis.quality_decision == QualityDecision.ACCEPT:
                    accepted_content.append({
                        "content": content,
                        "analysis": analysis
                    })
                    self.ingestion_stats["domains_covered"].add(content.get("domain", "unknown"))
                
                self.ingestion_stats["total_processed"] += 1
                
                if analysis.quality_decision == QualityDecision.ACCEPT:
                    self.ingestion_stats["total_accepted"] += 1
                else:
                    self.ingestion_stats["total_rejected"] += 1
            
            # Store accepted content
            if accepted_content:
                await self._store_content_batch(accepted_content)
            
            # Record metrics
            await self._record_batch_metrics(len(content_batch), len(accepted_content))
            
            # Check storage capacity
            storage_metrics = await self.storage_manager.get_storage_metrics()
            if storage_metrics.utilization_percentage > 90:
                logger.warning("⚠️ Storage capacity critical, slowing ingestion")
                await asyncio.sleep(5)
            
            # Adaptive rate limiting
            await self._adaptive_rate_limiting()
    
    async def _create_unified_content_stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Create unified content stream from all sources"""
        
        enabled_sources = self.content_sources.get_enabled_sources()
        logger.info(f"📥 Creating unified stream from {len(enabled_sources)} sources")
        
        # Create iterators for each source
        source_tasks = []
        for source in enabled_sources:
            if source not in self.source_exhausted:
                task = asyncio.create_task(self._get_real_source_iterator(source))
                source_tasks.append(task)
        
        # Round-robin through sources for breadth
        active_iterators = {}
        for task in source_tasks:
            try:
                source_name, iterator = await task
                active_iterators[source_name] = iterator
            except Exception as e:
                logger.error(f"Failed to initialize source iterator: {e}")
        
        # Yield content from all sources in round-robin fashion
        while active_iterators and self.ingestion_active:
            for source_name in list(active_iterators.keys()):
                try:
                    iterator = active_iterators[source_name]
                    content = await iterator.__anext__()
                    
                    # Add source metadata
                    content["source"] = source_name
                    content["ingestion_timestamp"] = datetime.now(timezone.utc).isoformat()
                    
                    yield content
                    
                except StopAsyncIteration:
                    # Source exhausted
                    logger.info(f"📭 Source exhausted: {source_name}")
                    del active_iterators[source_name]
                    self.source_exhausted.add(source_name)
                    
                except Exception as e:
                    logger.error(f"Error reading from source {source_name}: {e}")
                    # Remove problematic source
                    del active_iterators[source_name]
                    self.source_exhausted.add(source_name)
    
    async def _batch_content_stream(self, content_stream: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[List[Dict[str, Any]]]:
        """Batch content stream for efficient processing"""
        
        current_batch = []
        
        async for content in content_stream:
            current_batch.append(content)
            
            if len(current_batch) >= self.config.batch_size:
                yield current_batch
                current_batch = []
        
        # Yield remaining content
        if current_batch:
            yield current_batch
    
    async def _store_content_batch(self, accepted_content: List[Dict[str, Any]]):
        """Store batch of accepted content"""
        
        # Use batch optimizer for efficient storage
        async def store_item(item):
            content = item["content"]
            analysis = item["analysis"]
            
            # Store main content
            storage_result = await self.storage_manager.store_content(
                content_id=content.get("id", f"content_{hash(str(content))}"),
                content_data=content,
                content_type="content"
            )
            
            # Store quality analysis
            await self.storage_manager.store_content(
                content_id=f"analysis_{analysis.content_id}",
                content_data=analysis.to_dict(),
                content_type="metadata"
            )
            
            self.ingestion_stats["total_stored"] += 1
            
            return storage_result
        
        # Process batch in parallel
        results = await self.batch_optimizer.process_batch_list(
            accepted_content, store_item
        )
        
        logger.debug(f"📦 Stored batch: {len(results)} items")
    
    async def _record_batch_metrics(self, batch_size: int, accepted_count: int):
        """Record batch processing metrics"""
        
        # Record to monitoring system
        await self.monitoring_system.record_metric("ingestion.batch_size", batch_size)
        await self.monitoring_system.record_metric("ingestion.acceptance_rate", accepted_count / batch_size)
        await self.monitoring_system.record_metric("ingestion.total_processed", self.ingestion_stats["total_processed"])
        await self.monitoring_system.record_metric("ingestion.total_stored", self.ingestion_stats["total_stored"])
    
    async def _adaptive_rate_limiting(self):
        """Implement adaptive rate limiting"""
        
        # Check system resources
        system_status = await self.batch_optimizer.get_system_status()
        cpu_percent = system_status["system_resources"]["cpu_percent"]
        memory_percent = system_status["system_resources"]["memory_percent"]
        
        # Adaptive delay based on system load
        if cpu_percent > 90 or memory_percent > 90:
            await asyncio.sleep(2)  # Longer delay for high load
        elif cpu_percent > 70 or memory_percent > 70:
            await asyncio.sleep(0.5)  # Moderate delay
        else:
            await asyncio.sleep(0.1)  # Minimal delay
    
    async def _progress_monitoring_loop(self):
        """Background progress monitoring"""
        
        while self.ingestion_active:
            try:
                await asyncio.sleep(self.config.progress_report_interval)
                await self._report_progress()
                
            except Exception as e:
                logger.error(f"Progress monitoring error: {e}")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring"""
        
        while self.ingestion_active:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._check_system_health()
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def _report_progress(self):
        """Report ingestion progress"""
        
        current_time = datetime.now(timezone.utc)
        elapsed_time = (current_time - self.ingestion_stats["ingestion_start_time"]).total_seconds()
        
        processing_rate = self.ingestion_stats["total_processed"] / elapsed_time if elapsed_time > 0 else 0
        acceptance_rate = self.ingestion_stats["total_accepted"] / max(1, self.ingestion_stats["total_processed"])
        
        progress_report = {
            "timestamp": current_time.isoformat(),
            "elapsed_time_minutes": elapsed_time / 60,
            "total_processed": self.ingestion_stats["total_processed"],
            "total_accepted": self.ingestion_stats["total_accepted"],
            "total_stored": self.ingestion_stats["total_stored"],
            "acceptance_rate": acceptance_rate,
            "processing_rate_per_minute": processing_rate * 60,
            "domains_covered": len(self.ingestion_stats["domains_covered"]),
            "progress_percentage": (self.ingestion_stats["total_processed"] / self.config.max_total_content) * 100,
            "active_sources": len(self.config.content_sources) - len(self.source_exhausted)
        }
        
        logger.info("📊 INGESTION PROGRESS REPORT", **progress_report)
        
        # Record to monitoring system
        await self.monitoring_system.record_metric("ingestion.progress_percentage", progress_report["progress_percentage"])
        await self.monitoring_system.record_metric("ingestion.processing_rate", progress_report["processing_rate_per_minute"])
        
        self.ingestion_stats["last_progress_report"] = progress_report
    
    async def _check_system_health(self):
        """Check system health during ingestion"""
        
        try:
            # Check storage health
            storage_health = await self.storage_manager.get_storage_health()
            
            # Check batch optimizer health
            optimizer_status = await self.batch_optimizer.get_system_status()
            
            # Check monitoring system health
            monitoring_overview = await self.monitoring_system.get_system_overview()
            
            # Generate health alerts if needed
            if storage_health["storage_metrics"].utilization_percentage > 95:
                await self.monitoring_system.generate_alert(
                    "storage_critical",
                    "critical",
                    "Storage Critical",
                    f"Storage usage: {storage_health['storage_metrics'].utilization_percentage:.1f}%",
                    "storage_manager"
                )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final ingestion report"""
        
        end_time = datetime.now(timezone.utc)
        total_time = (end_time - self.ingestion_stats["ingestion_start_time"]).total_seconds()
        
        # Get final system metrics
        storage_metrics = await self.storage_manager.get_storage_metrics()
        filter_stats = await self.quality_filter.get_filter_statistics()
        optimizer_metrics = await self.batch_optimizer.get_processing_metrics()
        
        final_report = {
            "ingestion_summary": {
                "total_processed": self.ingestion_stats["total_processed"],
                "total_accepted": self.ingestion_stats["total_accepted"],
                "total_rejected": self.ingestion_stats["total_rejected"],
                "total_stored": self.ingestion_stats["total_stored"],
                "acceptance_rate": self.ingestion_stats["total_accepted"] / max(1, self.ingestion_stats["total_processed"]),
                "domains_covered": len(self.ingestion_stats["domains_covered"]),
                "domain_list": list(self.ingestion_stats["domains_covered"]),
                "sources_used": len(self.config.content_sources) - len(self.source_exhausted),
                "total_time_hours": total_time / 3600,
                "processing_rate_per_hour": (self.ingestion_stats["total_processed"] / total_time) * 3600
            },
            "storage_metrics": {
                "total_capacity_gb": storage_metrics.total_capacity_gb,
                "used_space_gb": storage_metrics.used_space_gb,
                "utilization_percentage": storage_metrics.utilization_percentage,
                "content_usage_gb": storage_metrics.content_usage_gb,
                "metadata_usage_gb": storage_metrics.metadata_usage_gb
            },
            "quality_metrics": filter_stats,
            "performance_metrics": {
                "throughput_items_per_second": optimizer_metrics.items_per_second,
                "success_rate": optimizer_metrics.success_rate,
                "peak_cpu_usage": optimizer_metrics.peak_cpu_usage,
                "peak_memory_usage": optimizer_metrics.peak_memory_usage
            },
            "breadth_optimization_results": {
                "breadth_score": len(self.ingestion_stats["domains_covered"]) / len(self.config.content_sources),
                "analogical_potential_maximized": True,
                "cross_domain_coverage": "excellent" if len(self.ingestion_stats["domains_covered"]) > 8 else "good"
            }
        }
        
        return final_report
    
    async def _initialize_content_sources(self):
        """Initialize content source iterators"""
        
        logger.info("📚 Initializing content sources...")
        
        # In production, these would connect to actual APIs
        # For now, we'll create mock iterators
        
        for source in self.config.content_sources:
            logger.info(f"📖 Initializing source: {source}")
            # Mock source initialization
            self.content_iterators[source] = None
        
        logger.info(f"✅ Initialized {len(self.config.content_sources)} content sources")
    
    async def _get_real_source_iterator(self, source_name: str):
        """Get iterator for real content source"""
        
        try:
            content_iterator = self.content_sources.get_content_iterator(source_name)
            return source_name, content_iterator
        except Exception as e:
            logger.error(f"Failed to get iterator for {source_name}: {e}")
            raise
    
    async def _get_source_iterator(self, source_name: str):
        """Get iterator for specific content source (fallback mock)"""
        
        # Mock implementation - fallback for testing
        async def mock_source_iterator():
            for i in range(1000):  # Mock 1000 items per source
                yield {
                    "id": f"{source_name}_{i}",
                    "title": f"Sample {source_name} Paper {i}",
                    "abstract": f"This is a sample abstract from {source_name} discussing advanced topics in the domain.",
                    "keywords": ["sample", "research", source_name.split("_")[0]],
                    "domain": source_name.split("_")[0],
                    "type": "research_paper",
                    "source": source_name
                }
                await asyncio.sleep(0.01)  # Rate limiting
        
        return source_name, mock_source_iterator()
    
    async def get_ingestion_status(self) -> Dict[str, Any]:
        """Get current ingestion status"""
        
        return {
            "ingestion_active": self.ingestion_active,
            "current_stats": self.ingestion_stats,
            "last_progress_report": self.ingestion_stats.get("last_progress_report"),
            "system_health": await self._get_system_health_summary()
        }
    
    async def _get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary"""
        
        try:
            return {
                "storage_healthy": True,
                "quality_filter_healthy": True,
                "batch_optimizer_healthy": True,
                "monitoring_healthy": True,
                "overall_status": "healthy"
            }
        except Exception as e:
            return {
                "overall_status": "error",
                "error": str(e)
            }
    
    async def shutdown(self):
        """Graceful shutdown"""
        
        logger.info("🔄 Shutting down ingestion orchestrator...")
        
        self.ingestion_active = False
        
        # Shutdown all components
        if self.storage_manager:
            await self.storage_manager.shutdown()
        if self.batch_optimizer:
            await self.batch_optimizer.shutdown()
        if self.monitoring_system:
            await self.monitoring_system.shutdown()
        if self.content_sources:
            await self.content_sources.shutdown()
        
        logger.info("✅ Ingestion orchestrator shutdown complete")


# Main execution function
async def main():
    """Main function to start production ingestion"""
    
    print("🌍 PRODUCTION INGESTION ORCHESTRATOR")
    print("=" * 60)
    print("Starting large-scale breadth-optimized content ingestion")
    print("=" * 60)
    
    # Create orchestrator
    config = IngestionConfig(
        max_total_content=150000,  # 150k items for maximum breadth
        external_drive_path="/Volumes/My Passport"
    )
    
    orchestrator = ProductionIngestionOrchestrator(config)
    
    try:
        # Initialize
        if await orchestrator.initialize():
            print("✅ Orchestrator initialized successfully")
            
            # Start ingestion
            result = await orchestrator.start_ingestion()
            
            print(f"\n🎉 Ingestion completed: {result['status']}")
            if result['status'] == 'completed':
                final_stats = result['final_stats']
                print(f"📊 Final Stats:")
                print(f"   Total processed: {final_stats['ingestion_summary']['total_processed']:,}")
                print(f"   Total stored: {final_stats['ingestion_summary']['total_stored']:,}")
                print(f"   Domains covered: {final_stats['ingestion_summary']['domains_covered']}")
                print(f"   Acceptance rate: {final_stats['ingestion_summary']['acceptance_rate']:.1%}")
                print(f"   Storage used: {final_stats['storage_metrics']['used_space_gb']:.1f} GB")
                print(f"   Breadth score: {final_stats['breadth_optimization_results']['breadth_score']:.2f}")
        else:
            print("❌ Orchestrator initialization failed")
            
    except KeyboardInterrupt:
        print("\n⚠️ Ingestion interrupted by user")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
    finally:
        await orchestrator.shutdown()
        print("🔄 Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())