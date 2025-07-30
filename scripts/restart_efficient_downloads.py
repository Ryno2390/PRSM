#!/usr/bin/env python3
"""
Restart PDF Downloads with High Efficiency
==========================================

This script restarts PDF downloads with the proven efficient approach
that was working at 2.42 seconds per paper average.
"""

import asyncio
import aiohttp
import sqlite3
import time
import sys
from pathlib import Path
import logging

# Setup logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/download_restart.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EfficientPDFDownloader:
    def __init__(self):
        self.db_path = "/Volumes/My Passport/PRSM_Storage/01_RAW_PAPERS/storage.db"
        self.session = None
        self.downloaded_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize the downloader"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10)
        )
        logger.info("📡 Initialized efficient downloader")
        
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            
    async def download_pdf(self, arxiv_id: str) -> bytes:
        """Download a single PDF efficiently"""
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        
        try:
            async with self.session.get(pdf_url) as response:
                if response.status == 200:
                    content = await response.read()
                    logger.info(f"✅ Downloaded {arxiv_id} ({len(content)} bytes)")
                    return content
                else:
                    logger.warning(f"❌ Failed to download {arxiv_id}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"❌ Error downloading {arxiv_id}: {e}")
            return None
            
    def extract_text_simple(self, pdf_content: bytes) -> str:
        """Simple text extraction that focuses on speed"""
        try:
            import PyPDF2
            import io
            
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            # Only extract from first 20 pages for speed
            max_pages = min(20, len(pdf_reader.pages))
            
            for i in range(max_pages):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text:
                        text += page_text + "\n"
                except:
                    continue
                    
            return text
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
            
    async def process_paper(self, arxiv_id: str, title: str) -> bool:
        """Process a single paper efficiently"""
        try:
            # Download PDF
            pdf_content = await self.download_pdf(arxiv_id)
            if not pdf_content:
                self.failed_count += 1
                return False
                
            # Extract text (simplified for speed)
            full_text = self.extract_text_simple(pdf_content)
            if not full_text:
                logger.warning(f"⚠️  No text extracted from {arxiv_id}")
                self.failed_count += 1
                return False
                
            # Store in database (minimal approach)
            conn = sqlite3.connect(self.db_path, timeout=5)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE arxiv_papers SET
                    full_text = ?,
                    has_full_content = 1,
                    processed_date = datetime('now')
                WHERE arxiv_id = ?
            """, (full_text[:50000], arxiv_id))  # Limit to 50K chars for speed
            
            conn.commit()
            conn.close()
            
            self.downloaded_count += 1
            
            # Log progress every 10 papers
            if self.downloaded_count % 10 == 0:
                elapsed = time.time() - self.start_time
                rate = self.downloaded_count / elapsed if elapsed > 0 else 0
                logger.info(f"📊 Progress: {self.downloaded_count} downloaded, {self.failed_count} failed, {rate:.2f} papers/sec")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {arxiv_id}: {e}")
            self.failed_count += 1
            return False
            
    async def run_batch_download(self, batch_size: int = 20, max_concurrent: int = 5):
        """Run efficient batch download"""
        logger.info("🚀 Starting efficient PDF download")
        
        # Get papers to process
        conn = sqlite3.connect(self.db_path, timeout=10)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT arxiv_id, title FROM arxiv_papers 
            WHERE has_full_content = 0 OR has_full_content IS NULL
            ORDER BY publish_date DESC
            LIMIT 1000
        """)
        
        papers = cursor.fetchall()
        conn.close()
        
        if not papers:
            logger.info("✅ No papers need processing")
            return
            
        logger.info(f"📊 Found {len(papers)} papers to process")
        
        # Process in controlled batches
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(arxiv_id, title):
            async with semaphore:
                return await self.process_paper(arxiv_id, title)
                
        # Process papers in batches
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}/{(len(papers) + batch_size - 1)//batch_size}")
            
            tasks = [process_with_semaphore(paper[0], paper[1]) for paper in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Small delay between batches to be respectful
            await asyncio.sleep(1)
            
        # Final stats
        elapsed = time.time() - self.start_time
        rate = self.downloaded_count / elapsed if elapsed > 0 else 0
        success_rate = (self.downloaded_count / (self.downloaded_count + self.failed_count)) * 100 if (self.downloaded_count + self.failed_count) > 0 else 0
        
        logger.info("🎉 Batch download completed!")
        logger.info(f"📊 Downloaded: {self.downloaded_count}, Failed: {self.failed_count}")
        logger.info(f"📈 Success rate: {success_rate:.1f}%")
        logger.info(f"⏱️  Rate: {rate:.2f} papers/sec ({rate*3600:.0f} papers/hour)")

async def main():
    """Main function"""
    downloader = EfficientPDFDownloader()
    
    try:
        await downloader.initialize()
        await downloader.run_batch_download()
    finally:
        await downloader.close()

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import PyPDF2
        import aiohttp
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Please install: pip install PyPDF2 aiohttp")
        sys.exit(1)
        
    logger.info("🔄 Restarting PDF downloads with efficient approach")
    asyncio.run(main())