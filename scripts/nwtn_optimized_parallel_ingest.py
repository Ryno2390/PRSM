#!/usr/bin/env python3
"""
NWTN Optimized Parallel Ingestion Pipeline
==========================================

Uses incremental scanning and batched processing to avoid
the slow full directory scan of 189K+ files.
"""

import asyncio
import json
import time
import hashlib
import os
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import argparse
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_optimized_ingestion.log'),
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
    content_sections: Dict[str, str]
    provenance_hash: str
    generated_at: float

class PostQuantumContentHashGenerator:
    """Post-quantum resistant content hashing"""
    
    def __init__(self, algorithm: str = "sha3_256"):
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

def process_paper_worker(args: Tuple[str, str, str]) -> Tuple[bool, str]:
    """Worker function to process a single paper - runs in separate process"""
    json_file_path, nwtn_dir, worker_id = args
    
    try:
        # Initialize embedding model in worker process
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        hash_generator = PostQuantumContentHashGenerator("sha3_256")
        
        # Load paper data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
            
        arxiv_id = paper_data.get('arxiv_id')
        if not arxiv_id:
            return False, f"No arXiv ID in {json_file_path}"
            
        # Check if already processed
        embedding_file = Path(nwtn_dir) / "embeddings" / f"{arxiv_id}.json"
        if embedding_file.exists():
            return True, f"Already processed: {arxiv_id}"
            
        # Extract content sections
        content_sections = {
            'title': paper_data.get('title', ''),
            'abstract': paper_data.get('abstract', ''),
            'introduction': paper_data.get('introduction', ''),
            'full_text_preview': paper_data.get('full_text_preview', '')
        }
        
        # Combine content for hashing and embedding
        full_content = f"{content_sections['title']}\\n\\n{content_sections['abstract']}\\n\\n{content_sections['full_text_preview']}"
        
        # Generate post-quantum content hash
        content_hash = hash_generator.generate_hash(full_content)
        
        # Create provenance record
        provenance = ProvenanceRecord(
            arxiv_id=arxiv_id,
            source_url=f"https://arxiv.org/abs/{arxiv_id}",
            content_hash=content_hash,
            processing_timestamp=time.time()
        )
        
        # Generate embeddings
        full_embedding = embedding_model.encode(full_content[:1000]).tolist()
        
        nwtn_embedding = NWTNEmbedding(
            arxiv_id=arxiv_id,
            embedding_vector=full_embedding,
            embedding_model="all-MiniLM-L6-v2",
            content_sections=content_sections,
            provenance_hash=content_hash.hash_value,
            generated_at=time.time()
        )
        
        # Ensure directories exist
        nwtn_path = Path(nwtn_dir)
        (nwtn_path / "content_hashes").mkdir(parents=True, exist_ok=True)
        (nwtn_path / "provenance").mkdir(parents=True, exist_ok=True)
        (nwtn_path / "embeddings").mkdir(parents=True, exist_ok=True)
        
        # Save to NWTN directories
        hash_file = nwtn_path / "content_hashes" / f"{arxiv_id}.json"
        with open(hash_file, 'w') as f:
            json.dump(asdict(content_hash), f, indent=2)
            
        provenance_file = nwtn_path / "provenance" / f"{arxiv_id}.json"
        with open(provenance_file, 'w') as f:
            json.dump(asdict(provenance), f, indent=2)
            
        embedding_file = nwtn_path / "embeddings" / f"{arxiv_id}.json"
        with open(embedding_file, 'w') as f:
            json.dump(asdict(nwtn_embedding), f, indent=2)
            
        return True, f"Processed: {arxiv_id}"
        
    except Exception as e:
        return False, f"Error processing {json_file_path}: {str(e)}"

class NWTNOptimizedIngestionPipeline:
    """Optimized NWTN ingestion with incremental scanning"""
    
    def __init__(self, num_workers: int = None, batch_size: int = 1000):
        self.input_dir = Path("/Volumes/My Passport/PRSM_Storage/02_PROCESSED_CONTENT")
        self.nwtn_dir = Path("/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY")
        self.progress_file = "/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_optimized_progress.json"
        
        # Determine optimal number of workers
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.scan_batch_size = batch_size  # Files to scan per batch
        
        # Create NWTN directories
        self.nwtn_dir.mkdir(exist_ok=True)
        (self.nwtn_dir / "embeddings").mkdir(exist_ok=True)
        (self.nwtn_dir / "provenance").mkdir(exist_ok=True)
        (self.nwtn_dir / "content_hashes").mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_papers": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "batches_scanned": 0,
            "start_time": time.time(),
            "workers": self.num_workers,
            "scan_batch_size": self.scan_batch_size
        }
        
        self.load_progress()
        
    def load_progress(self):
        """Load previous progress"""
        try:
            with open(self.progress_file, 'r') as f:
                saved_stats = json.load(f)
                self.stats.update(saved_stats)
                self.stats['start_time'] = time.time()
        except FileNotFoundError:
            pass
            
    def save_progress(self):
        """Save progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
    def get_processed_arxiv_ids(self) -> set:
        """Get set of already processed arXiv IDs"""
        processed = set()
        embeddings_dir = self.nwtn_dir / "embeddings"
        if embeddings_dir.exists():
            for file_path in embeddings_dir.glob("*.json"):
                processed.add(file_path.stem)
        return processed
        
    def scan_files_incrementally(self):
        """Generator that yields batches of files to process"""
        processed_ids = self.get_processed_arxiv_ids()
        logger.info(f"📋 Found {len(processed_ids):,} already processed papers")
        
        # Use glob to scan in smaller chunks by pattern
        patterns = [
            "07*.json", "08*.json", "09*.json", "10*.json", "11*.json", "12*.json",
            "13*.json", "14*.json", "15*.json", "16*.json", "17*.json", "18*.json", 
            "19*.json", "20*.json", "21*.json", "22*.json", "23*.json", "24*.json"
        ]
        
        total_found = 0
        for pattern in patterns:
            logger.info(f"🔍 Scanning pattern: {pattern}")
            pattern_path = str(self.input_dir / pattern)
            
            try:
                files = glob.glob(pattern_path)
                unprocessed_files = []
                
                for file_path in files:
                    arxiv_id = Path(file_path).stem
                    if arxiv_id not in processed_ids:
                        unprocessed_files.append(file_path)
                
                if unprocessed_files:
                    total_found += len(unprocessed_files)
                    logger.info(f"📦 Pattern {pattern}: {len(unprocessed_files)} new files to process")
                    
                    # Yield files in batches
                    for i in range(0, len(unprocessed_files), self.scan_batch_size):
                        batch = unprocessed_files[i:i + self.scan_batch_size]
                        yield batch
                        self.stats["batches_scanned"] += 1
                else:
                    logger.info(f"✅ Pattern {pattern}: All files already processed")
                    
            except Exception as e:
                logger.error(f"❌ Error scanning pattern {pattern}: {e}")
                continue
        
        logger.info(f"🎯 Total new files found: {total_found:,}")
        self.stats["total_papers"] = total_found
        
    def run_optimized_pipeline(self):
        """Run the optimized ingestion pipeline"""
        logger.info(f"🚀 Starting OPTIMIZED NWTN ingestion with {self.num_workers} workers")
        logger.info(f"📊 Scanning in batches of {self.scan_batch_size} files")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for file_batch in self.scan_files_incrementally():
                batch_start = time.time()
                
                # Prepare worker arguments
                worker_args = []
                for json_file_path in file_batch:
                    worker_id = f"worker_{len(worker_args) % self.num_workers}"
                    worker_args.append((json_file_path, str(self.nwtn_dir), worker_id))
                
                # Submit batch to workers
                future_to_arg = {executor.submit(process_paper_worker, arg): arg for arg in worker_args}
                
                # Collect results
                batch_results = {"success": 0, "skipped": 0, "errors": 0}
                for future in as_completed(future_to_arg):
                    try:
                        success, message = future.result()
                        if success:
                            if "Already processed" in message:
                                batch_results["skipped"] += 1
                            else:
                                batch_results["success"] += 1
                        else:
                            batch_results["errors"] += 1
                            logger.error(message)
                    except Exception as e:
                        batch_results["errors"] += 1
                        logger.error(f"Worker exception: {e}")
                
                # Update statistics
                self.stats["processed"] += batch_results["success"]
                self.stats["skipped"] += batch_results["skipped"] 
                self.stats["errors"] += batch_results["errors"]
                
                # Log progress
                batch_time = time.time() - batch_start
                total_processed = self.stats["processed"] + self.stats["skipped"]
                
                logger.info(f"📦 Batch {self.stats['batches_scanned']}: "
                          f"{batch_results['success']} processed, "
                          f"{batch_results['skipped']} skipped, "
                          f"{batch_results['errors']} errors "
                          f"({batch_time:.1f}s) "
                          f"[Total: {total_processed:,}]")
                
                # Save progress periodically
                if self.stats['batches_scanned'] % 5 == 0:
                    elapsed = time.time() - self.stats["start_time"]
                    rate = self.stats["processed"] / elapsed if elapsed > 0 else 0
                    
                    logger.info(f"🎯 Progress: {self.stats['processed']:,} processed "
                              f"Rate: {rate:.1f} papers/sec "
                              f"Batches scanned: {self.stats['batches_scanned']}")
                    
                    self.save_progress()
        
        # Final statistics
        elapsed_time = time.time() - self.stats['start_time']
        logger.info(f"""
🎉 OPTIMIZED NWTN INGESTION COMPLETE!
====================================
📊 Papers processed: {self.stats['processed']:,}
⏭️  Papers skipped: {self.stats['skipped']:,}
❌ Errors: {self.stats['errors']:,}
📦 Batches scanned: {self.stats['batches_scanned']}
⚡ Workers used: {self.stats['workers']}
⏱️  Total time: {elapsed_time:.1f} seconds
📈 Rate: {self.stats['processed'] / elapsed_time:.1f} papers/second
        """)
        
        self.save_progress()

def main():
    parser = argparse.ArgumentParser(description="NWTN Optimized Parallel Ingestion Pipeline")
    parser.add_argument("--workers", type=int, default=None, 
                       help="Number of worker processes (default: CPU count - 1)")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Files to scan per batch (default: 1000)")
    args = parser.parse_args()
    
    pipeline = NWTNOptimizedIngestionPipeline(num_workers=args.workers, batch_size=args.batch_size)
    pipeline.run_optimized_pipeline()

if __name__ == "__main__":
    main()