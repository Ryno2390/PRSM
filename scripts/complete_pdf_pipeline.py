#!/usr/bin/env python3
"""
Complete PDF Processing Pipeline for NWTN
=========================================

This script completely bypasses the corrupted database and implements
the full pipeline that NWTN needs:

1. Download full PDFs from arXiv at high speed
2. Extract complete text content 
3. Generate high-dimensional embeddings
4. Store embeddings in NWTN-searchable format

Target: 1500+ papers/hour (proven performance level)
"""

import asyncio
import aiohttp
import json
import time
import pickle
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompletePDFPipeline:
    def __init__(self):
        self.storage_path = Path("/Volumes/My Passport/PRSM_Storage")
        self.embeddings_path = self.storage_path / "03_EMBEDDINGS_NWTN_SEARCH"
        self.content_path = self.storage_path / "02_PROCESSED_CONTENT"
        self.progress_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/pipeline_progress.json")
        self.processed_set_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/processed_papers.json")
        
        # Create directories
        self.embeddings_path.mkdir(exist_ok=True)
        self.content_path.mkdir(exist_ok=True)
        
        self.session = None
        self.embedding_model = None
        self.processed_papers = set()  # Track already processed papers
        self.stats = {
            'downloaded': 0,
            'processed': 0,
            'embedded': 0,
            'failed': 0,
            'start_time': time.time()
        }
        
    async def initialize(self):
        """Initialize HTTP session and embedding model"""
        # HTTP session optimized for speed
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; PRSM-NWTN-Pipeline/1.0)'}
        )
        
        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("⚠️ sentence-transformers not available, embeddings disabled")
            self.embedding_model = None
            
        logger.info("🚀 Complete PDF pipeline initialized")
        
    async def close(self):
        """Clean shutdown"""
        if self.session:
            await self.session.close()
            
    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
    def load_progress(self):
        """Load previous progress"""
        try:
            with open(self.progress_file, 'r') as f:
                saved_stats = json.load(f)
                self.stats.update(saved_stats)
                # Reset start time for current session
                self.stats['start_time'] = time.time()
        except FileNotFoundError:
            pass
            
        # Load processed papers set
        try:
            with open(self.processed_set_file, 'r') as f:
                processed_list = json.load(f)
                self.processed_papers = set(processed_list)
                logger.info(f"📚 Loaded {len(self.processed_papers)} previously processed papers")
        except FileNotFoundError:
            # Build from existing files if no tracking file exists
            self.build_processed_set_from_files()
            
    def build_processed_set_from_files(self):
        """Build processed papers set from existing files"""
        logger.info("🔍 Building processed papers set from existing files...")
        
        # Check content files
        for content_file in self.content_path.glob("*.json"):
            arxiv_id = content_file.stem
            if arxiv_id and not arxiv_id.endswith("_full"):
                self.processed_papers.add(arxiv_id)
                
        logger.info(f"📚 Found {len(self.processed_papers)} previously processed papers")
        self.save_processed_set()
        
    def save_processed_set(self):
        """Save processed papers set"""
        try:
            with open(self.processed_set_file, 'w') as f:
                json.dump(list(self.processed_papers), f)
        except Exception as e:
            logger.debug(f"Error saving processed set: {e}")
            
    async def download_pdf(self, arxiv_id: str) -> bytes:
        """Download PDF with retry logic"""
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        
        for attempt in range(3):  # Retry up to 3 times
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        if len(content) > 1000:  # Basic sanity check
                            return content
                    elif response.status == 404:
                        return None  # Paper doesn't exist
                    
            except Exception as e:
                if attempt == 2:  # Last attempt
                    logger.debug(f"Download failed {arxiv_id}: {e}")
                await asyncio.sleep(0.1 * (attempt + 1))  # Brief retry delay
                
        return None
        
    def extract_complete_content(self, pdf_content: bytes) -> Dict[str, str]:
        """Extract complete text content for NWTN processing"""
        try:
            import PyPDF2
            import io
            
            pdf_file = io.BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract all text from all pages
            full_text = ""
            sections = {
                'title': '',
                'abstract': '',
                'introduction': '',
                'methods': '',
                'results': '',
                'conclusion': '',
                'references': ''
            }
            
            # Process all pages for complete content
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                        
                        # Simple section detection for first few pages
                        if page_num < 5:
                            page_lower = page_text.lower()
                            
                            # Extract title (usually first substantial text)
                            if not sections['title'] and page_num == 0:
                                lines = page_text.split('\n')
                                for line in lines:
                                    if len(line.strip()) > 10 and len(line.strip()) < 200:
                                        sections['title'] = line.strip()
                                        break
                                        
                            # Detect abstract
                            if 'abstract' in page_lower and not sections['abstract']:
                                abstract_start = page_lower.find('abstract')
                                if abstract_start >= 0:
                                    abstract_text = page_text[abstract_start:abstract_start+1500]
                                    sections['abstract'] = abstract_text
                                    
                            # Detect introduction
                            if 'introduction' in page_lower and not sections['introduction']:
                                intro_start = page_lower.find('introduction')
                                if intro_start >= 0:
                                    sections['introduction'] = page_text[intro_start:intro_start+2000]
                                    
                except Exception as e:
                    logger.debug(f"Page {page_num} extraction error: {e}")
                    continue
                    
            # Structure the content
            structured_content = {
                'full_text': full_text,
                'title': sections['title'],
                'abstract': sections['abstract'], 
                'introduction': sections['introduction'],
                'content_length': len(full_text),
                'page_count': len(reader.pages)
            }
            
            return structured_content
            
        except Exception as e:
            logger.debug(f"Content extraction error: {e}")
            return {}
            
    def generate_nwtn_embeddings(self, content: Dict[str, str], arxiv_id: str) -> Dict[str, np.ndarray]:
        """Generate high-dimensional embeddings for NWTN search"""
        if not self.embedding_model:
            return {}
            
        try:
            embeddings = {}
            
            # 1. Full paper embedding (for comprehensive search)
            if content.get('full_text'):
                # Use first 8000 chars to stay within model limits
                full_text_sample = content['full_text'][:8000]
                if len(full_text_sample) > 100:
                    full_embedding = self.embedding_model.encode(full_text_sample)
                    embeddings['full_paper'] = full_embedding
                    
            # 2. Title embedding (for precise matching)
            if content.get('title'):
                title_embedding = self.embedding_model.encode(content['title'])
                embeddings['title'] = title_embedding
                
            # 3. Abstract embedding (for quick relevance)
            if content.get('abstract'):
                abstract_embedding = self.embedding_model.encode(content['abstract'])
                embeddings['abstract'] = abstract_embedding
                
            # 4. Introduction embedding (for context understanding)
            if content.get('introduction'):
                intro_embedding = self.embedding_model.encode(content['introduction'][:2000])
                embeddings['introduction'] = intro_embedding
                
            # 5. Composite embedding (title + abstract + intro)
            composite_text = ""
            if content.get('title'):
                composite_text += content['title'] + " "
            if content.get('abstract'):
                composite_text += content['abstract'][:500] + " "
            if content.get('introduction'):
                composite_text += content['introduction'][:1000]
                
            if composite_text.strip():
                composite_embedding = self.embedding_model.encode(composite_text)
                embeddings['composite'] = composite_embedding
                
            return embeddings
            
        except Exception as e:
            logger.debug(f"Embedding generation error for {arxiv_id}: {e}")
            return {}
            
    def store_for_nwtn(self, arxiv_id: str, content: Dict[str, str], embeddings: Dict[str, np.ndarray]):
        """Store content and embeddings in NWTN-searchable format"""
        try:
            # Store processed content
            content_file = self.content_path / f"{arxiv_id}.json"
            with open(content_file, 'w', encoding='utf-8') as f:
                # Prepare content for JSON serialization
                json_content = {
                    'arxiv_id': arxiv_id,
                    'title': content.get('title', ''),
                    'abstract': content.get('abstract', ''),
                    'introduction': content.get('introduction', ''),
                    'full_text_preview': content.get('full_text', '')[:5000],  # Preview for JSON
                    'content_length': content.get('content_length', 0),
                    'page_count': content.get('page_count', 0),
                    'processed_at': time.time()
                }
                json.dump(json_content, f, indent=2, ensure_ascii=False)
                
            # Store full text separately (for memory efficiency)
            full_text_file = self.content_path / f"{arxiv_id}_full.txt"
            with open(full_text_file, 'w', encoding='utf-8') as f:
                f.write(content.get('full_text', ''))
                
            # Store embeddings in NWTN format
            if embeddings:
                embedding_file = self.embeddings_path / f"{arxiv_id}_embeddings.pkl"
                embedding_data = {
                    'arxiv_id': arxiv_id,
                    'embeddings': embeddings,
                    'metadata': {
                        'title': content.get('title', ''),
                        'content_length': content.get('content_length', 0),
                        'embedding_model': 'all-MiniLM-L6-v2',
                        'created_at': time.time()
                    }
                }
                
                with open(embedding_file, 'wb') as f:
                    pickle.dump(embedding_data, f)
            
            # Track as processed
            self.processed_papers.add(arxiv_id)
            if len(self.processed_papers) % 100 == 0:  # Save periodically
                self.save_processed_set()
                    
            return True
            
        except Exception as e:
            logger.debug(f"Storage error for {arxiv_id}: {e}")
            return False
            
    async def process_paper_complete(self, arxiv_id: str) -> bool:
        """Complete processing pipeline for one paper"""
        try:
            # 1. Download PDF
            pdf_content = await self.download_pdf(arxiv_id)
            if not pdf_content:
                self.stats['failed'] += 1
                return False
                
            self.stats['downloaded'] += 1
            
            # 2. Extract complete content
            content = self.extract_complete_content(pdf_content)
            if not content or not content.get('full_text'):
                self.stats['failed'] += 1
                return False
                
            self.stats['processed'] += 1
            
            # 3. Generate embeddings
            embeddings = self.generate_nwtn_embeddings(content, arxiv_id)
            
            # 4. Store for NWTN
            success = self.store_for_nwtn(arxiv_id, content, embeddings)
            if success and embeddings:
                self.stats['embedded'] += 1
                
            return success
            
        except Exception as e:
            logger.debug(f"Complete processing error for {arxiv_id}: {e}")
            self.stats['failed'] += 1
            return False
            
    def get_arxiv_ids_to_process(self) -> List[str]:
        """Generate arXiv IDs to process - targeting exactly 150K papers with broad coverage"""
        arxiv_ids = []
        target_total = 150000
        
        # Distribute exactly 150K papers across years for broad subject matter and temporal coverage
        year_distributions = {
            2024: 22000,  # Recent research - 22K papers
            2023: 20000,  # Recent research - 20K papers  
            2022: 18000,  # Recent research - 18K papers
            2021: 16000,  # COVID/ML boom - 16K papers
            2020: 14000,  # COVID/ML boom - 14K papers
            2019: 12000,  # Pre-COVID ML - 12K papers
            2018: 10000,  # Deep learning maturity - 10K papers
            2017: 9000,   # Transformer era - 9K papers
            2016: 8000,   # ResNet/deep learning - 8K papers
            2015: 7000,   # Early deep learning - 7K papers
            2014: 6000,   # Classical ML transition - 6K papers
            2013: 4000,   # Classical papers - 4K papers
            2012: 3000,   # Historical papers - 3K papers
            2011: 1000,   # Earlier historical - 1K papers
        }
        
        # Generate papers distributed across months for subject diversity
        for year, year_target in year_distributions.items():
            year_short = str(year)[2:]
            papers_per_month = year_target // 12  # Distribute evenly across months
            remainder = year_target % 12
            
            for month in range(1, 13):
                month_str = f"{month:02d}"
                
                # Add extra papers to early months for remainder
                month_target = papers_per_month + (1 if month <= remainder else 0)
                
                # Sample across paper numbers for diverse subject coverage
                # ArXiv typically has thousands of papers per month, we sample strategically
                if year >= 2020:
                    max_paper_range = 10000
                elif year >= 2016:
                    max_paper_range = 6000
                elif year >= 2012:
                    max_paper_range = 3000
                else:  # 2011 and earlier
                    max_paper_range = 1500
                    
                step_size = max(1, max_paper_range // month_target)
                
                paper_count = 0
                for paper_num in range(1, max_paper_range + 1, step_size):
                    if paper_count >= month_target:
                        break
                    arxiv_id = f"{year_short}{month_str}.{paper_num:04d}"
                    arxiv_ids.append(arxiv_id)
                    paper_count += 1
        
        # Shuffle the year-distributed papers for better processing distribution
        import random
        random.shuffle(arxiv_ids)
        
        # Ensure we have exactly 150K by adding or removing papers if needed
        current_count = len(arxiv_ids)
        if current_count < target_total:
            # Add some foundational/classic papers to reach exactly 150K
            needed = target_total - current_count
            classic_papers = []
            
            # Generate classic papers
            classic_ranges = [
                ("1412.", 1, 1000, 5),   # 2014 - Adam optimizer, GANs
                ("1506.", 1, 800, 4),    # 2015 - ResNet era
                ("1706.", 1, 1200, 6),   # 2017 - Attention Is All You Need
                ("1810.", 1, 1000, 5),   # 2018 - BERT era
                ("0704.", 1, 500, 3),    # 2007 - Historical ML
                ("1301.", 1, 600, 4),    # 2013 - Word2Vec era
                ("1409.", 1, 800, 4),    # 2014 - Important ML year
                ("1312.", 1, 600, 3),    # 2013 - VAE era
            ]
            
            for pattern, start, end, step in classic_ranges:
                for i in range(start, end + 1, step):
                    classic_papers.append(f"{pattern}{i:04d}")
                    if len(classic_papers) >= needed:
                        break
                if len(classic_papers) >= needed:
                    break
            
            # Add exactly what we need
            arxiv_ids.extend(classic_papers[:needed])
            
        elif current_count > target_total:
            # Trim to exactly 150K
            arxiv_ids = arxiv_ids[:target_total]
            
        # Final shuffle for optimal processing distribution
        random.shuffle(arxiv_ids)
        
        logger.info(f"📋 Generated {len(arxiv_ids)} arXiv IDs - targeting exactly 150K papers")
        logger.info(f"📚 Broad coverage: 2012-2024 with diverse subject sampling")
        return arxiv_ids
        
    async def run_complete_pipeline(self):
        """Run the complete PDF processing pipeline"""
        logger.info("🚀 Starting complete PDF processing pipeline")
        logger.info("Pipeline: Download → Extract → Embed → Store for NWTN")
        
        # Load previous progress
        self.load_progress()
        
        # Get papers to process and filter out already processed
        all_arxiv_ids = self.get_arxiv_ids_to_process()
        arxiv_ids = [arxiv_id for arxiv_id in all_arxiv_ids if arxiv_id not in self.processed_papers]
        
        logger.info(f"📦 Total papers in dataset: {len(all_arxiv_ids)}")
        logger.info(f"📚 Already processed: {len(self.processed_papers)}")
        logger.info(f"🎯 Remaining to process: {len(arxiv_ids)}")
        
        if not arxiv_ids:
            logger.info("🎉 All papers in dataset have been processed!")
            return
        
        # Process with high concurrency for maximum speed
        semaphore = asyncio.Semaphore(20)  # Balanced for speed vs stability
        
        async def process_with_semaphore(arxiv_id):
            async with semaphore:
                return await self.process_paper_complete(arxiv_id)
                
        # Process in batches for progress tracking
        batch_size = 100
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i:i+batch_size]
            
            batch_start = time.time()
            tasks = [process_with_semaphore(arxiv_id) for arxiv_id in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_time = time.time() - batch_start
            
            # Calculate performance metrics
            batch_successes = sum(1 for r in results if r is True)
            total_time = time.time() - self.stats['start_time']
            
            if total_time > 0:
                download_rate = self.stats['downloaded'] / total_time
                process_rate = self.stats['processed'] / total_time
                embed_rate = self.stats['embedded'] / total_time
                
                logger.info(f"📊 Batch {i//batch_size + 1}: {batch_successes}/{len(batch)} successful")
                logger.info(f"⚡ Rates: {download_rate:.1f} downloads/sec, {process_rate:.1f} processed/sec, {embed_rate:.1f} embedded/sec")
                logger.info(f"🎯 Hourly: {download_rate*3600:.0f} downloads, {process_rate*3600:.0f} processed, {embed_rate*3600:.0f} embedded")
                
            # Save progress
            self.save_progress()
            
            # Brief pause for system stability
            await asyncio.sleep(0.2)
            
        # Final statistics
        total_time = time.time() - self.stats['start_time']
        
        # Save final processed set
        self.save_processed_set()
        
        logger.info("🎉 Complete pipeline finished!")
        logger.info(f"📊 Final stats:")
        logger.info(f"   Downloaded: {self.stats['downloaded']}")
        logger.info(f"   Processed: {self.stats['processed']}")
        logger.info(f"   Embedded: {self.stats['embedded']}")
        logger.info(f"   Failed: {self.stats['failed']}")
        logger.info(f"   Total processed: {len(self.processed_papers)}")
        
        if total_time > 0:
            final_rate = self.stats['processed'] / total_time
            logger.info(f"⚡ Final rate: {final_rate:.2f} papers/sec ({final_rate*3600:.0f} papers/hour)")
            logger.info(f"🎯 Target was 1500+ papers/hour - {'✅ ACHIEVED' if final_rate*3600 >= 1500 else '⚠️ BELOW TARGET'}")
            
        # Check if more papers remain in the full dataset
        current_dataset_size = len(self.get_arxiv_ids_to_process())
        remaining_in_dataset = current_dataset_size - len(self.processed_papers)
        if remaining_in_dataset > 0:
            logger.info(f"📋 {remaining_in_dataset} papers remain in full dataset")
        else:
            logger.info("🎯 Complete dataset processed!")

async def main():
    """Main execution"""
    pipeline = CompletePDFPipeline()
    
    try:
        await pipeline.initialize()
        await pipeline.run_complete_pipeline()
    finally:
        await pipeline.close()

if __name__ == "__main__":
    # Check dependencies
    missing_deps = []
    
    try:
        import PyPDF2
    except ImportError:
        missing_deps.append("PyPDF2")
        
    try:
        import aiohttp
    except ImportError:
        missing_deps.append("aiohttp")
        
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
        
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
        
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        exit(1)
        
    print("🔥 COMPLETE PDF PROCESSING PIPELINE FOR NWTN")
    print("Features: Download → Extract → Embed → Store")
    print("Target: 1500+ papers/hour with full embeddings")
    print("=" * 60)
    
    asyncio.run(main())