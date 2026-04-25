#!/usr/bin/env python3
"""
NWTN Direct Pipeline - Fresh Start
==================================

Fresh local NWTN pipeline that processes papers directly from ArXiv,
downloading and processing them locally on the MacBook Pro for maximum performance.
"""

import asyncio
import json
import time
import hashlib
import os
import logging
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_direct.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ContentHash:
    """Post-quantum resistant content hash"""
    algorithm: str
    hash_value: str
    content_length: int
    generated_at: float

@dataclass
class ProvenanceRecord:
    """Complete provenance tracking for NWTN"""
    arxiv_id: str
    source_url: str
    content_hash: ContentHash
    processing_timestamp: float

@dataclass
class NWTNEmbedding:
    """NWTN-compatible embedding format"""
    arxiv_id: str
    embedding_vector: List[float]
    embedding_model: str
    content_sections: Dict[str, str]
    provenance_hash: str
    generated_at: float

class NWTNDirectPipeline:
    def __init__(self):
        # Local paths (all on MacBook Pro SSD)
        self.nwtn_dir = Path("/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY")
        self.processed_papers_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/processed_papers.json")
        self.progress_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_direct_progress.json")
        
        # Statistics
        self.stats = {
            'total_papers': 0,
            'processed': 0,
            'skipped_existing': 0,
            'embeddings_generated': 0,
            'content_hashes_created': 0,
            'provenance_records_created': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # Load papers list
        self.paper_ids = self.load_paper_ids()
        
        # Load progress if exists
        self.load_progress()
        
        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.embedding_model = None
            
        # Ensure output directories exist
        for subdir in ["embeddings", "content_hashes", "provenance"]:
            (self.nwtn_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def load_paper_ids(self) -> List[str]:
        """Load the list of processed paper IDs"""
        try:
            with open(self.processed_papers_file, 'r') as f:
                paper_ids = json.load(f)
                logger.info(f"📊 Loaded {len(paper_ids):,} paper IDs")
                return paper_ids
        except Exception as e:
            logger.error(f"❌ Failed to load paper IDs: {e}")
            return []
    
    def load_progress(self):
        """Load previous progress if available"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    saved_stats = json.load(f)
                    self.stats.update(saved_stats)
                    logger.info(f"🔄 Resuming from {self.stats['processed']:,} processed papers")
            except Exception as e:
                logger.warning(f"⚠️ Could not load progress: {e}")
    
    def save_progress(self):
        """Save current progress"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save progress: {e}")
    
    def already_processed(self, arxiv_id: str) -> bool:
        """Check if paper already has NWTN outputs"""
        required_files = [
            self.nwtn_dir / "embeddings" / f"{arxiv_id}.json",
            self.nwtn_dir / "content_hashes" / f"{arxiv_id}.json", 
            self.nwtn_dir / "provenance" / f"{arxiv_id}.json"
        ]
        return all(f.exists() for f in required_files)
    
    async def fetch_arxiv_metadata(self, arxiv_id: str) -> Optional[Dict[str, str]]:
        """Fetch paper metadata directly from ArXiv API"""
        try:
            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Extract entry
            entry = root.find('.//{http://www.w3.org/2005/Atom}entry')
            if entry is None:
                return None
            
            # Extract fields
            title = entry.find('.//{http://www.w3.org/2005/Atom}title')
            summary = entry.find('.//{http://www.w3.org/2005/Atom}summary')
            
            content_sections = {
                'title': title.text.strip() if title is not None else '',
                'abstract': summary.text.strip() if summary is not None else '',
                'full_text': ''  # We don't have full text from API, but that's ok for embeddings
            }
            
            return content_sections
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch metadata for {arxiv_id}: {e}")
            return None
    
    async def process_paper(self, arxiv_id: str) -> bool:
        """Process single paper directly from ArXiv"""
        try:
            # Skip if already processed
            if self.already_processed(arxiv_id):
                self.stats['skipped_existing'] += 1
                return True
            
            # Fetch metadata from ArXiv
            content_sections = await self.fetch_arxiv_metadata(arxiv_id)
            if not content_sections:
                logger.warning(f"⚠️ No content found for {arxiv_id}")
                return False
            
            # Validate content
            if not any(content_sections.values()):
                logger.warning(f"⚠️ Empty content for {arxiv_id}")
                return False
            
            full_content = f"{content_sections['title']} {content_sections['abstract']} {content_sections['full_text']}"
            
            # Generate content hash
            content_hash = ContentHash(
                algorithm="SHA3-256",
                hash_value=hashlib.sha3_256(full_content.encode('utf-8')).hexdigest(),
                content_length=len(full_content),
                generated_at=time.time()
            )
            
            # Create provenance record
            provenance = ProvenanceRecord(
                arxiv_id=arxiv_id,
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
                content_hash=content_hash,
                processing_timestamp=time.time()
            )
            
            # Generate embeddings
            if self.embedding_model:
                full_embedding = self.embedding_model.encode(full_content[:1000]).tolist()
                
                nwtn_embedding = NWTNEmbedding(
                    arxiv_id=arxiv_id,
                    embedding_vector=full_embedding,
                    embedding_model="all-MiniLM-L6-v2",
                    content_sections=content_sections,
                    provenance_hash=content_hash.hash_value,
                    generated_at=time.time()
                )
            else:
                return False
            
            # Save to NWTN directories (fast local I/O)
            try:
                # 1. Content hash
                hash_file = self.nwtn_dir / "content_hashes" / f"{arxiv_id}.json"
                with open(hash_file, 'w') as f:
                    json.dump(asdict(content_hash), f, indent=2)
                
                # 2. Provenance record
                provenance_file = self.nwtn_dir / "provenance" / f"{arxiv_id}.json"
                with open(provenance_file, 'w') as f:
                    json.dump(asdict(provenance), f, indent=2)
                
                # 3. NWTN embedding
                embedding_file = self.nwtn_dir / "embeddings" / f"{arxiv_id}.json"
                with open(embedding_file, 'w') as f:
                    json.dump(asdict(nwtn_embedding), f, indent=2)
                
            except Exception as e:
                logger.error(f"❌ Failed to save NWTN files for {arxiv_id}: {e}")
                return False
            
            # Update statistics
            self.stats['processed'] += 1
            self.stats['embeddings_generated'] += 1
            self.stats['content_hashes_created'] += 1
            self.stats['provenance_records_created'] += 1
            
            if self.stats['processed'] % 100 == 0:
                logger.info(f"✅ Processed {self.stats['processed']:,} papers for NWTN")
                self.save_progress()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {arxiv_id}: {str(e)}")
            self.stats['errors'] += 1
            return False
    
    async def run_pipeline(self):
        """Run direct NWTN pipeline"""
        logger.info("🚀 Starting NWTN direct pipeline (fresh processing from ArXiv)")
        
        if not self.paper_ids:
            logger.error("❌ No paper IDs loaded")
            return
        
        self.stats['total_papers'] = len(self.paper_ids)
        logger.info(f"📊 Processing {len(self.paper_ids):,} papers from ArXiv API")
        
        # Start from where we left off
        start_index = self.stats.get('processed', 0)
        remaining_papers = self.paper_ids[start_index:]
        
        if start_index > 0:
            logger.info(f"🔄 Resuming from paper {start_index:,}")
        
        # Process in efficient batches
        batch_size = 50  # Reasonable for API calls
        
        for i in range(0, len(remaining_papers), batch_size):
            batch = remaining_papers[i:i+batch_size]
            
            # Process batch with rate limiting
            tasks = [self.process_paper(arxiv_id) for arxiv_id in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log batch completion
            successful = sum(1 for r in results if r is True)
            skipped = sum(1 for r in results if r is True and hasattr(r, 'skipped'))
            batch_num = (i + start_index) // batch_size + 1
            logger.info(f"📦 Batch {batch_num}: {successful}/{len(batch)} papers processed")
            
            # Small delay to be respectful to ArXiv API
            await asyncio.sleep(1)
        
        # Final statistics
        elapsed_time = time.time() - self.stats['start_time']
        logger.info(f"""
🎉 NWTN DIRECT PIPELINE COMPLETE!
================================
📊 Total papers: {self.stats['total_papers']:,}
✅ Successfully processed: {self.stats['processed']:,}
⏭️  Skipped (existing): {self.stats['skipped_existing']:,}
🔗 Embeddings generated: {self.stats['embeddings_generated']:,}
🔐 Content hashes created: {self.stats['content_hashes_created']:,}
📝 Provenance records: {self.stats['provenance_records_created']:,}
❌ Errors: {self.stats['errors']:,}
⏱️  Processing time: {elapsed_time:.1f} seconds
📈 Rate: {self.stats['processed'] / elapsed_time:.1f} papers/second
        """)
        
        self.save_progress()

async def main():
    """Main execution function"""
    pipeline = NWTNDirectPipeline()
    await pipeline.run_pipeline()

if __name__ == "__main__":
    asyncio.run(main())