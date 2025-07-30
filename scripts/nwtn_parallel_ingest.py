#!/usr/bin/env python3
"""
NWTN Parallel Ingestion Pipeline for Existing Processed Content
==============================================================

High-performance parallel ingestion with multiple worker processes.
Dramatically accelerates processing of 189,882 papers.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_parallel_ingestion.log'),
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

class NWTNParallelIngestionPipeline:
    """Parallel NWTN ingestion pipeline with multiple worker processes"""
    
    def __init__(self, num_workers: int = None):
        self.input_dir = Path("/Volumes/My Passport/PRSM_Storage/02_PROCESSED_CONTENT")
        self.nwtn_dir = Path("/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY")
        self.progress_file = "/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_parallel_progress.json"
        
        # Determine optimal number of workers
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)  # Leave one core free
        
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
            "start_time": time.time(),
            "workers": self.num_workers,
            "batch_size": self.num_workers * 10  # 10 papers per worker per batch
        }
        
        self.load_progress()
        
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
            
    def get_remaining_files(self) -> List[Path]:
        """Get files that haven't been processed yet"""
        all_files = list(self.input_dir.glob("*.json"))
        processed_files = set()
        
        # Check what's already been processed
        embeddings_dir = self.nwtn_dir / "embeddings"
        if embeddings_dir.exists():
            for file_path in embeddings_dir.glob("*.json"):
                processed_files.add(file_path.stem)
        
        remaining = [f for f in all_files if f.stem not in processed_files]
        return remaining
        
    def run_parallel_pipeline(self):
        """Run the parallel ingestion pipeline"""
        logger.info(f"🚀 Starting PARALLEL NWTN ingestion with {self.num_workers} workers")
        
        # Get remaining files to process
        remaining_files = self.get_remaining_files()
        self.stats['total_papers'] = len(remaining_files)
        
        if not remaining_files:
            logger.info("✅ All papers already processed!")
            return
            
        logger.info(f"📊 Found {len(remaining_files):,} papers remaining to process")
        logger.info(f"⚡ Using {self.num_workers} parallel workers")
        
        # Prepare worker arguments
        worker_args = []
        for i, json_file in enumerate(remaining_files):
            worker_id = f"worker_{i % self.num_workers}"
            worker_args.append((str(json_file), str(self.nwtn_dir), worker_id))
        
        # Process with parallel workers
        batch_size = self.stats["batch_size"]
        total_batches = (len(worker_args) + batch_size - 1) // batch_size
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for batch_num in range(0, len(worker_args), batch_size):
                batch_start = time.time()
                batch_args = worker_args[batch_num:batch_num + batch_size]
                
                # Submit batch to workers
                future_to_arg = {executor.submit(process_paper_worker, arg): arg for arg in batch_args}
                
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
                batch_idx = batch_num // batch_size + 1
                total_processed = self.stats["processed"] + self.stats["skipped"]
                
                logger.info(f"📦 Batch {batch_idx}/{total_batches}: "
                          f"{batch_results['success']} processed, "
                          f"{batch_results['skipped']} skipped, "
                          f"{batch_results['errors']} errors "
                          f"({batch_time:.1f}s)")
                
                # Progress summary every 10 batches
                if batch_idx % 10 == 0:
                    elapsed = time.time() - self.stats["start_time"]
                    rate = self.stats["processed"] / elapsed if elapsed > 0 else 0
                    remaining = len(remaining_files) - total_processed
                    eta = remaining / rate / 3600 if rate > 0 else 0
                    
                    logger.info(f"🎯 Progress: {total_processed:,}/{len(remaining_files):,} "
                              f"({100*total_processed/len(remaining_files):.1f}%) "
                              f"Rate: {rate:.1f} papers/sec "
                              f"ETA: {eta:.1f} hours")
                    
                    self.save_progress()
        
        # Final statistics
        elapsed_time = time.time() - self.stats['start_time']
        logger.info(f"""
🎉 PARALLEL NWTN INGESTION COMPLETE!
===================================
📊 Papers processed: {self.stats['processed']:,}
⏭️  Papers skipped: {self.stats['skipped']:,}
❌ Errors: {self.stats['errors']:,}
⚡ Workers used: {self.stats['workers']}
⏱️  Total time: {elapsed_time:.1f} seconds
📈 Rate: {self.stats['processed'] / elapsed_time:.1f} papers/second
🚀 Speedup: ~{self.num_workers}x faster than single-threaded
        """)
        
        self.save_progress()

def main():
    parser = argparse.ArgumentParser(description="NWTN Parallel Ingestion Pipeline")
    parser.add_argument("--workers", type=int, default=None, 
                       help="Number of worker processes (default: CPU count - 1)")
    args = parser.parse_args()
    
    pipeline = NWTNParallelIngestionPipeline(num_workers=args.workers)
    pipeline.run_parallel_pipeline()

if __name__ == "__main__":
    main()