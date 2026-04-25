#!/usr/bin/env python3
"""
Simple High-Performance PDF Downloader
=====================================

A complete rewrite that replicates the 2.42 seconds/paper performance
without the complex database infrastructure that's causing issues.

This script:
- Downloads PDFs directly from arXiv at maximum speed
- Uses simple file-based tracking instead of complex database queries
- Focuses purely on speed and reliability
- Matches the proven 70,319 downloads performance
"""

import asyncio
import aiohttp
import sqlite3
import time
import json
import os
from pathlib import Path
import logging

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class HighPerformanceDownloader:
    def __init__(self):
        self.db_path = "/Volumes/My Passport/PRSM_Storage/01_RAW_PAPERS/storage.db"
        self.progress_file = "/Users/ryneschultz/Documents/GitHub/PRSM/download_progress.json"
        self.session = None
        self.stats = {
            'downloaded': 0,
            'failed': 0,
            'start_time': time.time()
        }
        
    async def initialize(self):
        """Initialize HTTP session with optimal settings"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; PRSM-Research-Bot/1.0)'}
        )
        
        logger.info("🚀 High-performance downloader initialized")
        
    async def close(self):
        """Clean shutdown"""
        if self.session:
            await self.session.close()
            
    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.stats, f)
            
    def load_progress(self):
        """Load previous progress"""
        try:
            with open(self.progress_file, 'r') as f:
                self.stats.update(json.load(f))
        except FileNotFoundError:
            pass
            
    async def download_pdf(self, arxiv_id: str) -> bytes:
        """Download PDF with maximum efficiency"""
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    return content
                elif response.status == 404:
                    return None  # Paper doesn't exist
                else:
                    logger.warning(f"HTTP {response.status} for {arxiv_id}")
                    return None
        except Exception as e:
            logger.debug(f"Download error {arxiv_id}: {e}")
            return None
            
    def extract_text_fast(self, pdf_content: bytes) -> str:
        """Fast text extraction optimized for speed"""
        try:
            import PyPDF2
            import io
            
            pdf_file = io.BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            # Process maximum 10 pages for speed
            max_pages = min(10, len(reader.pages))
            
            for i in range(max_pages):
                try:
                    page_text = reader.pages[i].extract_text()
                    if page_text:
                        text += page_text + " "
                        # Stop if we have enough content
                        if len(text) > 10000:
                            break
                except:
                    continue
                    
            return text.strip()
        except:
            return ""
            
    async def store_paper_fast(self, arxiv_id: str, full_text: str):
        """Store paper with minimal database operations"""
        try:
            # Use WAL mode for better concurrency
            conn = sqlite3.connect(self.db_path, timeout=2)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            cursor = conn.cursor()
            
            # Simple, fast update
            cursor.execute("""
                UPDATE arxiv_papers 
                SET full_text = ?, has_full_content = 1, processed_date = datetime('now')
                WHERE arxiv_id = ?
            """, (full_text[:20000], arxiv_id))  # Limit text size for speed
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.debug(f"Storage error {arxiv_id}: {e}")
            return False
            
    async def process_paper(self, arxiv_id: str) -> bool:
        """Process single paper with maximum speed"""
        try:
            # Download
            pdf_content = await self.download_pdf(arxiv_id)
            if not pdf_content:
                self.stats['failed'] += 1
                return False
                
            # Extract text
            text = self.extract_text_fast(pdf_content)
            if not text:
                self.stats['failed'] += 1
                return False
                
            # Store
            success = await self.store_paper_fast(arxiv_id, text)
            if success:
                self.stats['downloaded'] += 1
                return True
            else:
                self.stats['failed'] += 1
                return False
                
        except Exception as e:
            logger.debug(f"Process error {arxiv_id}: {e}")
            self.stats['failed'] += 1
            return False
            
    async def get_papers_to_process(self) -> list:
        """Get papers that need processing (simplified query)"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            cursor = conn.cursor()
            
            # Simple query with limit to avoid timeouts
            cursor.execute("""
                SELECT arxiv_id FROM arxiv_papers 
                WHERE (has_full_content IS NULL OR has_full_content = 0) 
                AND arxiv_id IS NOT NULL
                ORDER BY publish_date DESC
                LIMIT 10000
            """)
            
            papers = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            logger.info(f"📊 Found {len(papers)} papers to process")
            return papers
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []
            
    async def run_high_speed_download(self):
        """Run download at maximum speed"""
        logger.info("🚀 Starting high-speed PDF download")
        
        # Load previous progress
        self.load_progress()
        
        # Get papers to process
        papers = await self.get_papers_to_process()
        if not papers:
            logger.info("✅ No papers to process")
            return
            
        logger.info(f"📦 Processing {len(papers)} papers")
        
        # Process with high concurrency
        semaphore = asyncio.Semaphore(25)  # High concurrency for speed
        
        async def process_with_semaphore(arxiv_id):
            async with semaphore:
                return await self.process_paper(arxiv_id)
                
        # Process in batches for memory management
        batch_size = 100
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            
            batch_start = time.time()
            tasks = [process_with_semaphore(arxiv_id) for arxiv_id in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_time = time.time() - batch_start
            
            # Calculate and log performance
            batch_successes = sum(1 for r in results if r is True)
            total_time = time.time() - self.stats['start_time']
            rate = self.stats['downloaded'] / total_time if total_time > 0 else 0
            
            logger.info(f"📊 Batch {i//batch_size + 1}: {batch_successes}/{len(batch)} successful, "
                       f"{rate:.2f} papers/sec, {rate*3600:.0f} papers/hour")
            
            # Save progress
            self.save_progress()
            
            # Brief pause for system stability
            await asyncio.sleep(0.5)
            
        # Final stats
        total_time = time.time() - self.stats['start_time']
        final_rate = self.stats['downloaded'] / total_time if total_time > 0 else 0
        
        logger.info("🎉 High-speed download completed!")
        logger.info(f"📊 Downloaded: {self.stats['downloaded']}, Failed: {self.stats['failed']}")
        logger.info(f"⚡ Final rate: {final_rate:.2f} papers/sec ({final_rate*3600:.0f} papers/hour)")
        logger.info(f"🎯 Target was 1500 papers/hour (0.42 papers/sec)")

async def main():
    """Main execution"""
    downloader = HighPerformanceDownloader()
    
    try:
        await downloader.initialize()
        await downloader.run_high_speed_download()
    finally:
        await downloader.close()

if __name__ == "__main__":
    # Check dependencies
    try:
        import PyPDF2
        import aiohttp
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install PyPDF2 aiohttp")
        exit(1)
        
    print("🔥 SIMPLE HIGH-PERFORMANCE PDF DOWNLOADER")
    print("Targeting 1500+ papers/hour (matching proven performance)")
    print("=" * 60)
    
    asyncio.run(main())