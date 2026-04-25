#!/usr/bin/env python3
"""
NWTN Ingestion Pipeline for Existing Processed Content
=====================================================

Ingests the 189,872 verified unique papers into NWTN with:
- Post-quantum content hashing (SHA3-256)
- Full provenance tracking 
- High-dimensional embeddings for search
- Source traceability and attribution
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
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_ingest.log'),
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
    nwtn_version: str = "1.0.0"
    pipeline_version: str = "2.1.0"
    creator_attribution: str = "arXiv.org"
    license_info: str = "arXiv non-exclusive license"

@dataclass 
class NWTNEmbedding:
    """High-dimensional embedding for NWTN search"""
    arxiv_id: str
    embedding_vector: List[float]
    embedding_model: str
    content_sections: Dict[str, str]  # title, abstract, full_text
    provenance_hash: str
    generated_at: float

class PostQuantumContentHashGenerator:
    """Post-quantum resistant content hashing"""
    
    def __init__(self, algorithm: str = "sha3_256"):
        # Post-quantum resistant algorithms
        pq_algorithms = ["sha3_256", "sha3_512", "blake2b", "blake2s"]
        if algorithm not in pq_algorithms:
            raise ValueError(f"Algorithm must be post-quantum resistant: {pq_algorithms}")
        self.algorithm = algorithm
        
    def generate_hash(self, content: str) -> ContentHash:
        """Generate post-quantum resistant content hash"""
        content_bytes = content.encode('utf-8')
        
        if self.algorithm == "sha3_256":
            hash_obj = hashlib.sha3_256()
        elif self.algorithm == "sha3_512":
            hash_obj = hashlib.sha3_512()
        elif self.algorithm == "blake2b":
            hash_obj = hashlib.blake2b()
        elif self.algorithm == "blake2s":
            hash_obj = hashlib.blake2s()
            
        hash_obj.update(content_bytes)
        content_hash = hash_obj.hexdigest()
        
        return ContentHash(
            algorithm=self.algorithm,
            hash_value=content_hash,
            content_length=len(content_bytes),
            generated_at=time.time()
        )

class NWTNIngestionPipeline:
    """Complete NWTN ingestion pipeline for existing processed content"""
    
    def __init__(self):
        self.input_dir = Path("/Volumes/My Passport/PRSM_Storage/02_PROCESSED_CONTENT")
        self.nwtn_dir = Path("/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY")
        self.progress_file = "/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_ingest_progress.json"
        
        # Create NWTN directories
        self.nwtn_dir.mkdir(exist_ok=True)
        (self.nwtn_dir / "embeddings").mkdir(exist_ok=True)
        (self.nwtn_dir / "provenance").mkdir(exist_ok=True)
        (self.nwtn_dir / "content_hashes").mkdir(exist_ok=True)
        
        self.hash_generator = PostQuantumContentHashGenerator("sha3_256")
        self.embedding_model = None
        
        # Statistics
        self.stats = {
            "total_papers": 0,
            "processed": 0,
            "embeddings_generated": 0,
            "content_hashes_created": 0,
            "provenance_records_created": 0,
            "errors": 0,
            "start_time": time.time()
        }
        
        self.load_progress()
        
    async def initialize(self):
        """Initialize embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded: all-MiniLM-L6-v2")
        except ImportError:
            logger.error("❌ sentence-transformers not available - installing...")
            os.system("pip install sentence-transformers")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model installed and loaded")
            
    def load_progress(self):
        """Load previous progress"""
        try:
            with open(self.progress_file, 'r') as f:
                saved_stats = json.load(f)
                self.stats.update(saved_stats)
                self.stats['start_time'] = time.time()  # Reset for current session
        except FileNotFoundError:
            pass
            
    def save_progress(self):
        """Save progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
    def get_processed_files(self) -> set:
        """Get already processed arXiv IDs"""
        processed = set()
        
        # Check embeddings directory
        embeddings_dir = self.nwtn_dir / "embeddings"
        if embeddings_dir.exists():
            for file_path in embeddings_dir.glob("*.json"):
                arxiv_id = file_path.stem
                processed.add(arxiv_id)
                
        return processed
        
    async def process_paper(self, json_file: Path) -> bool:
        """Process a single paper into NWTN format"""
        try:
            # Load existing processed content
            with open(json_file, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
                
            arxiv_id = paper_data.get('arxiv_id')
            if not arxiv_id:
                logger.error(f"❌ No arXiv ID found in {json_file}")
                return False
                
            # Skip if already processed
            if arxiv_id in self.get_processed_files():
                return True
                
            # Extract content sections
            content_sections = {
                'title': paper_data.get('title', ''),
                'abstract': paper_data.get('abstract', ''),
                'introduction': paper_data.get('introduction', ''),
                'full_text_preview': paper_data.get('full_text_preview', '')
            }
            
            # Combine content for hashing and embedding
            full_content = f"{content_sections['title']}\n\n{content_sections['abstract']}\n\n{content_sections['full_text_preview']}"
            
            # Generate post-quantum content hash
            content_hash = self.hash_generator.generate_hash(full_content)
            
            # Create provenance record
            provenance = ProvenanceRecord(
                arxiv_id=arxiv_id,
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
                content_hash=content_hash,
                processing_timestamp=time.time()
            )
            
            # Generate embeddings
            if self.embedding_model:
                # Create embeddings for different content sections
                title_embedding = self.embedding_model.encode(content_sections['title']).tolist()
                abstract_embedding = self.embedding_model.encode(content_sections['abstract']).tolist()
                full_embedding = self.embedding_model.encode(full_content[:1000]).tolist()  # Limit for model
                
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
                
            # Save to NWTN directories
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
                
            # Update statistics
            self.stats['processed'] += 1
            self.stats['embeddings_generated'] += 1
            self.stats['content_hashes_created'] += 1
            self.stats['provenance_records_created'] += 1
            
            if self.stats['processed'] % 100 == 0:
                logger.info(f"✅ Processed {self.stats['processed']:,} papers for NWTN ingestion")
                self.save_progress()
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {json_file}: {str(e)}")
            self.stats['errors'] += 1
            return False
            
    async def run_pipeline(self):
        """Run complete NWTN ingestion pipeline"""
        logger.info("🚀 Starting NWTN ingestion pipeline for existing processed content")
        
        # Get all JSON files
        json_files = list(self.input_dir.glob("*.json"))
        self.stats['total_papers'] = len(json_files)
        
        logger.info(f"📊 Found {len(json_files):,} papers to ingest into NWTN")
        
        # Process in batches for efficiency
        batch_size = 50
        for i in range(0, len(json_files), batch_size):
            batch = json_files[i:i+batch_size]
            
            # Process batch
            tasks = [self.process_paper(json_file) for json_file in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log batch completion
            successful = sum(1 for r in results if r is True)
            logger.info(f"📦 Batch {i//batch_size + 1}: {successful}/{len(batch)} papers processed")
            
        # Final statistics
        elapsed_time = time.time() - self.stats['start_time']
        logger.info(f"""
🎉 NWTN INGESTION COMPLETE!
==========================
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
    pipeline = NWTNIngestionPipeline()
    await pipeline.initialize()
    await pipeline.run_pipeline()

if __name__ == "__main__":
    asyncio.run(main())