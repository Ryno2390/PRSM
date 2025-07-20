#!/usr/bin/env python3
"""
Background arXiv Processing for NWTN Testing
============================================

This script runs the arXiv bulk dataset processing in the background
to generate embeddings for testing NWTN's meta-reasoning capabilities.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import logging
import signal
import os

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.bulk_dataset_processor import BulkDatasetProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/arxiv_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackgroundArxivProcessor:
    """Background processor for arXiv dataset"""
    
    def __init__(self):
        self.processor = BulkDatasetProcessor()
        self.progress_file = Path('/Users/ryneschultz/Documents/GitHub/PRSM/arxiv_progress.json')
        self.target_papers = 150000
        self.running = True
        
    def save_progress(self, processed: int, accepted: int, elapsed: float):
        """Save processing progress"""
        progress = {
            'processed': processed,
            'accepted': accepted,
            'target': self.target_papers,
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat(),
            'rate_per_second': processed / elapsed if elapsed > 0 else 0,
            'acceptance_rate': (accepted / processed) * 100 if processed > 0 else 0,
            'estimated_completion': (self.target_papers - processed) / (processed / elapsed) if processed > 0 and elapsed > 0 else 0
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def process_arxiv_background(self):
        """Process arXiv dataset in background"""
        
        logger.info("🚀 Starting background arXiv processing...")
        logger.info(f"📄 Target: {self.target_papers:,} papers")
        logger.info(f"📊 Progress will be saved to: {self.progress_file}")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        start_time = time.time()
        
        try:
            # Initialize processor
            await self.processor.initialize()
            logger.info("✅ Processor initialized successfully")
            
            # Start processing
            success = await self.processor.download_arxiv_dataset(target_papers=self.target_papers)
            
            elapsed = time.time() - start_time
            
            if success:
                logger.info(f"🎉 Processing completed successfully in {elapsed:.2f} seconds")
                
                # Save final progress
                self.save_progress(
                    processed=self.processor.stats.get('total_processed', 0),
                    accepted=self.processor.stats.get('total_accepted', 0),
                    elapsed=elapsed
                )
                
                # Generate final report
                await self.generate_final_report()
                
            else:
                logger.error("❌ Processing failed")
                
        except Exception as e:
            logger.error(f"💥 Error during processing: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("🏁 Background processing completed")
    
    async def generate_final_report(self):
        """Generate final processing report"""
        
        report_file = Path('/Users/ryneschultz/Documents/GitHub/PRSM/arxiv_final_report.json')
        
        # Check database for final counts
        try:
            import sqlite3
            db_path = '/Volumes/My Passport/PRSM_Storage/storage.db'
            
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get final counts
                cursor.execute('SELECT COUNT(*) FROM content_storage WHERE content_type = "content"')
                final_count = cursor.fetchone()[0]
                
                # Get domains
                cursor.execute('SELECT DISTINCT json_extract(content_data, "$.domain") FROM content_storage WHERE content_type = "content"')
                domains = [row[0] for row in cursor.fetchall() if row[0]]
                
                conn.close()
                
                report = {
                    'completion_time': datetime.now().isoformat(),
                    'final_paper_count': final_count,
                    'target_papers': self.target_papers,
                    'completion_percentage': (final_count / self.target_papers) * 100,
                    'domains_processed': domains,
                    'stats': self.processor.stats,
                    'ready_for_nwtn_testing': final_count > 0
                }
                
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"📊 Final report saved to: {report_file}")
                logger.info(f"🎯 Papers processed: {final_count:,}")
                logger.info(f"📈 Completion: {(final_count / self.target_papers) * 100:.1f}%")
                logger.info(f"🌍 Domains: {', '.join(domains)}")
                
                if final_count > 0:
                    logger.info("✅ READY FOR NWTN TESTING!")
                    
        except Exception as e:
            logger.error(f"Error generating final report: {e}")

def main():
    """Main entry point"""
    
    print("🚀 Background arXiv Processing Starting...")
    print("=" * 60)
    print("📄 Target: 150,000 papers from arXiv")
    print("💾 Storage: External drive + high-dimensional embeddings")
    print("⚡ Processing: Background mode with progress tracking")
    print("🌍 Domains: Physics, CS, Math, Biology, etc.")
    print()
    print("📊 Progress will be logged to: arxiv_processing.log")
    print("📈 Status updates saved to: arxiv_progress.json")
    print()
    print("🔄 Processing started... This will take 30-60 minutes")
    print("🛑 Use Ctrl+C to stop gracefully")
    print("=" * 60)
    
    processor = BackgroundArxivProcessor()
    
    # Run the async processing
    try:
        asyncio.run(processor.process_arxiv_background())
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()