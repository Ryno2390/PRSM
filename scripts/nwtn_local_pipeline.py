#!/usr/bin/env python3
"""
NWTN Local Pipeline - Direct Processing
======================================

Fresh start pipeline that processes from local downloaded papers,
bypassing the problematic external drive entirely.
"""

import asyncio
import json
import time
import hashlib
import os
import logging
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
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_local.log'),
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

class NWTNLocalPipeline:
    def __init__(self):
        # Local paths (all on MacBook Pro SSD)
        self.input_dir = Path("/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/02_PROCESSED_CONTENT")
        self.nwtn_dir = Path("/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY")
        self.progress_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_local_progress.json")
        
        # Statistics
        self.stats = {
            'total_papers': 0,
            'processed': 0,
            'embeddings_generated': 0,
            'content_hashes_created': 0,
            'provenance_records_created': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
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
    
    async def process_paper(self, json_file: Path) -> bool:
        """Process single paper with local file system"""
        try:
            arxiv_id = json_file.stem
            
            # Skip if already processed
            if self.already_processed(arxiv_id):
                return True
            
            # Load content (should be fast on local SSD)
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Could not read {arxiv_id}: {e}")
                return False
            
            # Extract content sections
            content_sections = {
                'title': content.get('title', ''),
                'abstract': content.get('abstract', ''),
                'full_text': content.get('content', '')
            }
            
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
            logger.error(f"❌ Error processing {json_file}: {str(e)}")
            self.stats['errors'] += 1
            return False
    
    async def run_pipeline(self):
        """Run local NWTN pipeline"""
        logger.info("🚀 Starting NWTN local pipeline (MacBook Pro SSD)")
        
        # Check if we have any source files
        if not self.input_dir.exists():
            logger.error(f"❌ Input directory does not exist: {self.input_dir}")
            logger.info("📝 Need to populate source files first")
            return
        
        # Get JSON files (should be fast on local SSD)
        json_files = list(self.input_dir.glob("*.json"))
        if not json_files:
            logger.error("❌ No JSON files found in input directory")
            logger.info("📝 Need to copy/download processed papers first")
            return
        
        self.stats['total_papers'] = len(json_files)
        logger.info(f"📊 Found {len(json_files):,} papers to process locally")
        
        # Process in efficient batches for local I/O
        batch_size = 100  # Larger batches since we're on fast SSD
        
        for i in range(0, len(json_files), batch_size):
            batch = json_files[i:i+batch_size]
            
            # Process batch
            tasks = [self.process_paper(json_file) for json_file in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log batch completion
            successful = sum(1 for r in results if r is True)
            batch_num = i // batch_size + 1
            logger.info(f"📦 Batch {batch_num}: {successful}/{len(batch)} papers processed")
        
        # Final statistics
        elapsed_time = time.time() - self.stats['start_time']
        logger.info(f"""
🎉 NWTN LOCAL PIPELINE COMPLETE!
===============================
📊 Total papers processed: {self.stats['processed']:,}
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
    pipeline = NWTNLocalPipeline()
    await pipeline.run_pipeline()

if __name__ == "__main__":
    asyncio.run(main())