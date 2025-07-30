#!/usr/bin/env python3
"""
NWTN Ingestion Recovery Pipeline
===============================

Enhanced version with:
- Resume capability from last successful batch
- Better error handling for missing files
- Drive performance monitoring
- Corrupted file detection and skipping
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
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_recovery.log'),
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

class NWTNRecoveryPipeline:
    def __init__(self):
        # Paths
        self.input_dir = Path("/Volumes/My Passport/PRSM_Storage/02_PROCESSED_CONTENT")
        self.nwtn_dir = Path("/Volumes/My Passport/PRSM_Storage/03_NWTN_READY")
        self.progress_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_recovery_progress.json")
        
        # Statistics
        self.stats = {
            'total_papers': 0,
            'processed': 0,
            'skipped_missing': 0,
            'skipped_existing': 0,
            'embeddings_generated': 0,
            'content_hashes_created': 0,
            'provenance_records_created': 0,
            'errors': 0,
            'start_time': time.time(),
            'last_processed_file': None
        }
        
        # Resume point
        self.resume_from = 0
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
                    self.resume_from = self.stats.get('processed', 0)
                    logger.info(f"🔄 Resuming from paper {self.resume_from:,}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load progress: {e}")
    
    def save_progress(self):
        """Save current progress"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save progress: {e}")
    
    def file_exists_and_accessible(self, file_path: Path, timeout: float = 2.0) -> bool:
        """Check if file exists and is accessible within timeout"""
        try:
            start_time = time.time()
            exists = file_path.exists()
            if time.time() - start_time > timeout:
                logger.warning(f"⚠️ Slow file access detected: {file_path}")
                return False
            return exists
        except Exception:
            return False
    
    def already_processed(self, arxiv_id: str) -> bool:
        """Check if paper already has NWTN outputs"""
        required_files = [
            self.nwtn_dir / "embeddings" / f"{arxiv_id}.json",
            self.nwtn_dir / "content_hashes" / f"{arxiv_id}.json", 
            self.nwtn_dir / "provenance" / f"{arxiv_id}.json"
        ]
        return all(f.exists() for f in required_files)
    
    async def process_paper(self, json_file: Path) -> bool:
        """Process single paper with enhanced error handling"""
        try:
            arxiv_id = json_file.stem
            
            # Skip if already processed
            if self.already_processed(arxiv_id):
                self.stats['skipped_existing'] += 1
                return True
            
            # Check file accessibility
            if not self.file_exists_and_accessible(json_file):
                logger.warning(f"⚠️ Skipping inaccessible file: {arxiv_id}")
                self.stats['skipped_missing'] += 1
                return False
            
            # Load and validate content
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"⚠️ Corrupted file {arxiv_id}: {e}")
                self.stats['skipped_missing'] += 1
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
                self.stats['skipped_missing'] += 1
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
                title_embedding = self.embedding_model.encode(content_sections['title']).tolist()
                abstract_embedding = self.embedding_model.encode(content_sections['abstract']).tolist()
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
            
            # Save to NWTN directories atomically
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
                # Clean up partial files
                for file_path in [hash_file, provenance_file, embedding_file]:
                    if file_path.exists():
                        file_path.unlink()
                return False
            
            # Update statistics
            self.stats['processed'] += 1
            self.stats['embeddings_generated'] += 1
            self.stats['content_hashes_created'] += 1
            self.stats['provenance_records_created'] += 1
            self.stats['last_processed_file'] = arxiv_id
            
            if self.stats['processed'] % 50 == 0:
                logger.info(f"✅ Processed {self.stats['processed']:,} papers for NWTN")
                self.save_progress()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {json_file}: {str(e)}")
            self.stats['errors'] += 1
            return False
    
    async def run_pipeline(self):
        """Run recovery pipeline"""
        logger.info("🚀 Starting NWTN recovery pipeline")
        
        # Get all JSON files with timeout protection
        try:
            logger.info("📁 Scanning for JSON files...")
            json_files = []
            scan_start = time.time()
            
            # Use os.listdir for better performance than Path.glob
            try:
                filenames = os.listdir(self.input_dir)
                json_files = [
                    self.input_dir / f for f in filenames 
                    if f.endswith('.json')
                ]
                json_files.sort()  # Ensure consistent ordering
            except Exception as e:
                logger.error(f"❌ Failed to scan directory: {e}")
                return
            
            scan_time = time.time() - scan_start
            logger.info(f"📊 Found {len(json_files):,} files in {scan_time:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to enumerate files: {e}")
            return
        
        self.stats['total_papers'] = len(json_files)
        
        # Resume from last position
        if self.resume_from > 0:
            json_files = json_files[self.resume_from:]
            logger.info(f"🔄 Resuming from position {self.resume_from:,}")
        
        # Process in smaller batches for better error handling
        batch_size = 25  # Reduced from 50 for better recovery
        
        for i in range(0, len(json_files), batch_size):
            batch = json_files[i:i+batch_size]
            
            # Process batch with timeout protection
            try:
                tasks = [self.process_paper(json_file) for json_file in batch]
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=300  # 5 minute timeout per batch
                )
                
                # Log batch completion
                successful = sum(1 for r in results if r is True)
                batch_num = (i + self.resume_from) // batch_size + 1
                logger.info(f"📦 Batch {batch_num}: {successful}/{len(batch)} papers processed")
                
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Batch {batch_num} timed out, skipping...")
                self.stats['errors'] += len(batch)
                continue
            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                self.stats['errors'] += len(batch)
                continue
        
        # Final statistics
        elapsed_time = time.time() - self.stats['start_time']
        logger.info(f"""
🎉 NWTN RECOVERY COMPLETE!
=========================
📊 Total papers: {self.stats['total_papers']:,}
✅ Successfully processed: {self.stats['processed']:,}
⏭️  Skipped (existing): {self.stats['skipped_existing']:,}
⚠️  Skipped (missing/corrupt): {self.stats['skipped_missing']:,}
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
    pipeline = NWTNRecoveryPipeline()
    await pipeline.run_pipeline()

if __name__ == "__main__":
    asyncio.run(main())