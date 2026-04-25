#!/usr/bin/env python3
"""
Safe PDF Download Process Optimizer
===================================

Safely optimizes the running PDF download process by creating an additional
optimized worker that coordinates with the existing process to maximize
throughput without conflicts.
"""

import asyncio
import sqlite3
import sys
import time
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import psutil

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.external_storage_config import ExternalKnowledgeBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/pdf_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SafePDFOptimizer:
    """Safely optimizes PDF download process without interrupting current work"""
    
    def __init__(self):
        self.db_path = Path("/Volumes/My Passport/PRSM_Storage/storage.db")
        self.running = True
        self.kb = None
        self.processed_count = 0
        self.optimization_stats = {
            'start_time': None,
            'papers_processed': 0,
            'optimization_rate': 0.0,
            'resource_usage': {}
        }
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down optimizer...")
        self.running = False

    def check_existing_process(self) -> Optional[Dict[str, Any]]:
        """Check if the main PDF download process is still running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['cmdline'] and any('download_full_pdfs.py' in cmd for cmd in proc.info['cmdline']):
                    return {
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'process': proc
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource utilization"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'available_memory_gb': memory.available / (1024**3),
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        }

    def calculate_safe_parameters(self, system_resources: Dict[str, float]) -> Dict[str, int]:
        """Calculate safe optimization parameters based on available resources"""
        
        # Conservative approach - use available resources without overwhelming system
        cpu_usage = system_resources['cpu_percent']
        memory_usage = system_resources['memory_percent']
        
        # Base parameters (more aggressive than current process)
        base_concurrent = 15  # Increased from 10
        base_batch_size = 75  # Increased from 50
        base_delay = 1.0     # Reduced from 2.0 seconds
        
        # Adjust based on available resources
        if cpu_usage < 30 and memory_usage < 70:
            # System has plenty of capacity
            max_concurrent = 20
            batch_size = 100
            batch_delay = 0.8
        elif cpu_usage < 50 and memory_usage < 80:
            # Moderate capacity available
            max_concurrent = 18
            batch_size = 85
            batch_delay = 1.0
        else:
            # Conservative settings
            max_concurrent = base_concurrent
            batch_size = base_batch_size
            batch_delay = base_delay
        
        return {
            'max_concurrent': max_concurrent,
            'batch_size': batch_size,
            'batch_delay': batch_delay
        }

    async def get_unprocessed_papers(self, limit: int = 50000) -> List[tuple]:
        """Get papers that need processing, avoiding conflicts with main process"""
        
        if not self.db_path.exists():
            logger.error("Database not found")
            return []
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get papers without full content, ordered differently than main process
            # Main process orders by publish_date DESC, we'll use a different strategy
            cursor.execute("""
                SELECT arxiv_id, title FROM arxiv_papers 
                WHERE (has_full_content = 0 OR has_full_content IS NULL)
                AND arxiv_id NOT IN (
                    -- Exclude papers that might be currently being processed
                    SELECT arxiv_id FROM arxiv_papers 
                    WHERE has_full_content = 0 OR has_full_content IS NULL
                    ORDER BY publish_date DESC 
                    LIMIT 1000
                )
                ORDER BY RANDOM()  -- Different ordering to avoid conflicts
                LIMIT ?
            """, (limit,))
            
            papers = cursor.fetchall()
            conn.close()
            
            logger.info(f"Found {len(papers)} papers available for optimization processing")
            return papers
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []

    async def process_papers_optimized(self, papers: List[tuple], params: Dict[str, int]) -> Dict[str, Any]:
        """Process papers with optimized parameters"""
        
        logger.info(f"🚀 Starting optimized processing with {len(papers)} papers")
        logger.info(f"⚙️ Parameters: concurrent={params['max_concurrent']}, "
                   f"batch_size={params['batch_size']}, delay={params['batch_delay']}s")
        
        # Initialize external knowledge base with optimized settings
        self.kb = ExternalKnowledgeBase()
        await self.kb.initialize()
        
        # Track statistics
        stats = {
            'total_papers': len(papers),
            'processed': 0,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': time.time(),
            'batches_completed': 0
        }
        
        # Process in optimized batches
        batch_size = params['batch_size']
        semaphore = asyncio.Semaphore(params['max_concurrent'])
        
        for batch_start in range(0, len(papers), batch_size):
            if not self.running:
                break
                
            batch_end = min(batch_start + batch_size, len(papers))
            current_batch = papers[batch_start:batch_end]
            
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(papers) + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing optimization batch {batch_num}/{total_batches} "
                       f"({len(current_batch)} papers)")
            
            # Process batch concurrently
            batch_tasks = [
                self._process_single_paper(semaphore, paper[0], paper[1])
                for paper in current_batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Update statistics
            for result in batch_results:
                if isinstance(result, Exception):
                    stats['failed'] += 1
                elif result is None:
                    stats['skipped'] += 1
                else:
                    if result.get('downloaded'):
                        stats['downloaded'] += 1
                    if result.get('processed'):
                        stats['processed'] += 1
            
            stats['batches_completed'] += 1
            
            # Progress update
            progress = (batch_end / len(papers)) * 100
            elapsed = time.time() - stats['start_time']
            rate = stats['processed'] / elapsed if elapsed > 0 else 0
            
            logger.info(f"✅ Optimization batch {batch_num} completed: "
                       f"{progress:.1f}% | {stats['processed']} processed | "
                       f"{rate:.1f} papers/sec")
            
            # Monitor system resources and adjust if needed
            resources = self.get_system_resources()
            if resources['cpu_percent'] > 80 or resources['memory_percent'] > 90:
                logger.warning("High resource usage detected, increasing batch delay")
                await asyncio.sleep(params['batch_delay'] * 2)
            else:
                await asyncio.sleep(params['batch_delay'])
        
        # Final statistics
        total_time = time.time() - stats['start_time']
        stats['total_time'] = total_time
        stats['avg_rate'] = stats['processed'] / total_time if total_time > 0 else 0
        
        return stats

    async def _process_single_paper(self, semaphore: asyncio.Semaphore, arxiv_id: str, title: str) -> Optional[Dict[str, Any]]:
        """Process a single paper with semaphore control"""
        async with semaphore:
            try:
                # Double-check if paper is already processed (avoid race conditions)
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT has_full_content FROM arxiv_papers WHERE arxiv_id = ?", (arxiv_id,))
                row = cursor.fetchone()
                conn.close()
                
                if row and row[0] == 1:
                    return {'skipped': True, 'reason': 'already_processed'}
                
                # Use the existing KB method for downloading and processing
                result = await self.kb.storage_manager._download_and_process_paper_with_semaphore(
                    semaphore, arxiv_id, title
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to process {arxiv_id}: {e}")
                return {'downloaded': False, 'processed': False, 'error': str(e)}

    async def run_optimization(self):
        """Main optimization function"""
        
        logger.info("🚀 PDF DOWNLOAD OPTIMIZER STARTING")
        logger.info("=" * 60)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.optimization_stats['start_time'] = time.time()
        
        try:
            # Check if main process is still running
            main_process = self.check_existing_process()
            if main_process:
                logger.info(f"✅ Main PDF process detected (PID: {main_process['pid']})")
                logger.info("🤝 Running optimization in coordination mode")
            else:
                logger.info("ℹ️ No main PDF process detected, running in standalone mode")
            
            # Get system resources
            resources = self.get_system_resources()
            logger.info(f"💻 System Resources: CPU {resources['cpu_percent']:.1f}%, "
                       f"Memory {resources['memory_percent']:.1f}%, "
                       f"Available {resources['available_memory_gb']:.1f}GB")
            
            # Calculate safe parameters
            params = self.calculate_safe_parameters(resources)
            logger.info(f"⚙️ Optimization Parameters: {params}")
            
            # Get papers to process
            papers = await self.get_unprocessed_papers(limit=20000)  # Process up to 20K papers
            
            if not papers:
                logger.info("ℹ️ No papers available for optimization processing")
                return
            
            # Run optimized processing
            stats = await self.process_papers_optimized(papers, params)
            
            # Final report
            logger.info("\n" + "=" * 60)
            logger.info("🎉 PDF DOWNLOAD OPTIMIZATION COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"📊 Papers Processed: {stats['processed']:,}")
            logger.info(f"📥 Papers Downloaded: {stats['downloaded']:,}")
            logger.info(f"❌ Failed: {stats['failed']:,}")
            logger.info(f"⏭️ Skipped: {stats['skipped']:,}")
            logger.info(f"⏱️ Total Time: {stats['total_time']/3600:.2f} hours")
            logger.info(f"⚡ Average Rate: {stats['avg_rate']:.1f} papers/second")
            logger.info(f"🚀 Rate Improvement: ~{((stats['avg_rate'] * 3600) / 7200 - 1) * 100:.0f}%")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            logger.info("🏁 PDF optimization process completed")

def main():
    """Main entry point"""
    print("🚀 PDF DOWNLOAD SAFE OPTIMIZER")
    print("=" * 50)
    print("This optimizer will:")
    print("✓ Work alongside the existing PDF download process")
    print("✓ Use optimized parameters based on available resources")
    print("✓ Process different papers to avoid conflicts")
    print("✓ Monitor system resources and adjust automatically")
    print("✓ Provide real-time progress updates")
    print()
    print("Starting optimization in 5 seconds... (Ctrl+C to cancel)")
    
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n❌ Optimization cancelled by user")
        return
    
    optimizer = SafePDFOptimizer()
    
    try:
        asyncio.run(optimizer.run_optimization())
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()