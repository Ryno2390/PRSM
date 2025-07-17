#!/usr/bin/env python3
"""
Production FAISS Index Builder
==============================

This script monitors embedding completion and builds production-ready
FAISS indices incrementally as more embeddings become available.
"""

import time
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from build_faiss_index import FAISSIndexBuilder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionIndexBuilder:
    """Build production FAISS indices incrementally"""
    
    def __init__(self):
        self.embeddings_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Embeddings")
        self.production_indices_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Indices_Production")
        self.production_indices_dir.mkdir(parents=True, exist_ok=True)
        
        # Track building progress
        self.last_batch_count = 0
        self.build_thresholds = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]  # Batch milestones
        self.completed_builds = set()
        
    def get_current_batch_count(self) -> int:
        """Get current number of embedding batches"""
        batch_files = list(self.embeddings_dir.glob("embeddings_batch_*.pkl"))
        return len(batch_files)
    
    def should_build_index(self, current_batches: int) -> Optional[int]:
        """Check if we should build an index at current batch count"""
        for threshold in self.build_thresholds:
            if current_batches >= threshold and threshold not in self.completed_builds:
                return threshold
        return None
    
    def build_incremental_index(self, batch_count: int, index_type: str = "IVF") -> bool:
        """Build incremental index with specified batch count"""
        logger.info(f"🏗️  Building {index_type} index with {batch_count} batches...")
        
        try:
            # Create index builder
            builder = FAISSIndexBuilder(
                embeddings_dir=str(self.embeddings_dir),
                index_dir=str(self.production_indices_dir),
                index_type=index_type
            )
            
            # Build index with limited batches
            success = builder.build_complete_index(max_batches=batch_count)
            
            if success:
                logger.info(f"✅ {index_type} index built successfully with {batch_count} batches")
                
                # Rename index files to include batch count
                old_index = self.production_indices_dir / f"faiss_index_{index_type.lower()}.index"
                new_index = self.production_indices_dir / f"faiss_index_{index_type.lower()}_{batch_count}_batches.index"
                
                old_metadata = self.production_indices_dir / f"paper_metadata_{index_type.lower()}.pkl"
                new_metadata = self.production_indices_dir / f"paper_metadata_{index_type.lower()}_{batch_count}_batches.pkl"
                
                old_info = self.production_indices_dir / f"index_metadata_{index_type.lower()}.json"
                new_info = self.production_indices_dir / f"index_metadata_{index_type.lower()}_{batch_count}_batches.json"
                
                # Move files
                if old_index.exists():
                    old_index.rename(new_index)
                if old_metadata.exists():
                    old_metadata.rename(new_metadata)
                if old_info.exists():
                    old_info.rename(new_info)
                
                return True
            else:
                logger.error(f"❌ Failed to build {index_type} index")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error building index: {e}")
            return False
    
    def monitor_and_build(self, duration_minutes: int = 60):
        """Monitor embedding progress and build indices incrementally"""
        logger.info("🚀 Starting production index building monitor")
        logger.info(f"⏱️  Monitoring for {duration_minutes} minutes")
        logger.info(f"📊 Build thresholds: {self.build_thresholds}")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                current_batches = self.get_current_batch_count()
                
                # Check if we should build an index
                threshold = self.should_build_index(current_batches)
                
                if threshold:
                    logger.info(f"🎯 Reached threshold: {threshold} batches (current: {current_batches})")
                    
                    # Build both IVF and Flat indices
                    for index_type in ["IVF", "Flat"]:
                        success = self.build_incremental_index(threshold, index_type)
                        if success:
                            logger.info(f"✅ {index_type} index built for {threshold} batches")
                        else:
                            logger.error(f"❌ Failed to build {index_type} index for {threshold} batches")
                    
                    # Mark threshold as completed
                    self.completed_builds.add(threshold)
                    logger.info(f"📈 Completed builds: {sorted(self.completed_builds)}")
                
                # Log current progress
                if current_batches != self.last_batch_count:
                    papers_processed = current_batches * 32
                    logger.info(f"📊 Progress: {current_batches:,} batches, {papers_processed:,} papers")
                    self.last_batch_count = current_batches
                
                # Wait before next check
                time.sleep(120)  # Check every 2 minutes
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error during monitoring: {e}")
                time.sleep(60)
        
        # Final status
        final_batches = self.get_current_batch_count()
        logger.info(f"📊 Final Status:")
        logger.info(f"  Total batches: {final_batches:,}")
        logger.info(f"  Completed builds: {sorted(self.completed_builds)}")
        
        # Build final index if we have enough batches
        if final_batches >= 1000 and final_batches not in self.completed_builds:
            logger.info("🏗️  Building final production index...")
            for index_type in ["IVF", "Flat"]:
                self.build_incremental_index(final_batches, index_type)
        
        logger.info("✅ Production index building completed")

def main():
    """Main function"""
    builder = ProductionIndexBuilder()
    
    # Get monitoring duration from command line
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    
    # Check initial state
    current_batches = builder.get_current_batch_count()
    logger.info(f"📊 Initial state: {current_batches:,} batches available")
    
    # Start monitoring and building
    builder.monitor_and_build(duration)

if __name__ == "__main__":
    main()