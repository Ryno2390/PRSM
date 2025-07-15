#!/usr/bin/env python3
"""
Speed-Optimized Ingestion for PRSM
==================================

This is a speed-optimized version of the ingestion system that prioritizes
quantity and breadth over perfectionism to build the knowledge base faster.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import structlog

from prsm.nwtn.production_ingestion_orchestrator import ProductionIngestionOrchestrator, IngestionConfig

logger = structlog.get_logger(__name__)


class SpeedOptimizedConfig(IngestionConfig):
    """Speed-optimized configuration"""
    
    def __init__(self):
        super().__init__()
        
        # SPEED OPTIMIZATIONS
        self.max_total_content = 150000
        self.min_quality_threshold = 0.15  # Very low threshold
        self.max_concurrent_downloads = 30  # High concurrency
        self.batch_size = 500              # Large batches
        self.ingestion_rate_limit = 50000  # Very high rate limit
        
        # Progress reporting
        self.progress_report_interval = 60  # Report every minute
        self.health_check_interval = 30     # Check every 30 seconds


async def main():
    """Main function for speed-optimized ingestion"""
    
    print("🚀 SPEED-OPTIMIZED PRSM NWTN INGESTION")
    print("=" * 60)
    print("⚡ OPTIMIZED FOR MAXIMUM SPEED AND BREADTH")
    print("📊 Target: 150,000 papers with relaxed quality filters")
    print("🎯 Focus: Quantity and domain breadth over perfectionism")
    print("=" * 60)
    
    # Create speed-optimized config
    config = SpeedOptimizedConfig()
    
    # Display optimizations
    print("⚡ SPEED OPTIMIZATIONS APPLIED:")
    print(f"   🔽 Quality Threshold: {config.min_quality_threshold}")
    print(f"   🔄 Concurrent Downloads: {config.max_concurrent_downloads}")
    print(f"   📦 Batch Size: {config.batch_size}")
    print(f"   🚀 Rate Limit: {config.ingestion_rate_limit:,} items/hour")
    print()
    
    # Create orchestrator
    orchestrator = ProductionIngestionOrchestrator(config)
    
    try:
        # Initialize
        print("🔧 Initializing speed-optimized system...")
        if await orchestrator.initialize():
            print("✅ Speed-optimized system ready!")
            print("🚀 Starting high-speed ingestion...")
            print("📝 Monitor progress with: tail -f /tmp/ingestion.log")
            print("📊 Check estimates with: python prsm/nwtn/ingestion_progress_estimator.py")
            print()
            
            # Start high-speed ingestion
            result = await orchestrator.start_ingestion()
            
            print(f"\n🎉 Ingestion completed: {result['status']}")
            if result['status'] == 'completed':
                final_stats = result['final_stats']
                print(f"📊 Final Results:")
                print(f"   📚 Total Processed: {final_stats['ingestion_summary']['total_processed']:,}")
                print(f"   💾 Total Stored: {final_stats['ingestion_summary']['total_stored']:,}")
                print(f"   🌍 Domains Covered: {final_stats['ingestion_summary']['domains_covered']}")
                print(f"   ✅ Acceptance Rate: {final_stats['ingestion_summary']['acceptance_rate']:.1%}")
                print(f"   💿 Storage Used: {final_stats['storage_metrics']['used_space_gb']:.1f} GB")
                print(f"   🎯 Breadth Score: {final_stats['breadth_optimization_results']['breadth_score']:.2f}")
                print(f"   ⏱️ Processing Rate: {final_stats['ingestion_summary']['processing_rate_per_hour']:.0f} items/hour")
        else:
            print("❌ Speed-optimized system initialization failed")
            
    except KeyboardInterrupt:
        print("\n⚠️ Ingestion interrupted by user")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
    finally:
        await orchestrator.shutdown()
        print("🔄 Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())