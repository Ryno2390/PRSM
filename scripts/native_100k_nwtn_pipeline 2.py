#!/usr/bin/env python3
"""
Native 100K NWTN Pipeline
=========================

Selects the best 100,000 papers from local database and processes them
natively with parallel ingestion for optimal NWTN performance.
"""

import sqlite3
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
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/native_100k_pipeline.log'),
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
    pipeline_version: str = "3.0.0"
    creator_attribution: str = "arXiv.org"
    license_info: str = "arXiv non-exclusive license"
    source_database: str = "native_local_sqlite"

@dataclass 
class NWTNEmbedding:
    """High-dimensional embedding for NWTN search"""
    arxiv_id: str
    embedding_vector: List[float]
    embedding_model: str
    content_sections: Dict[str, str]
    provenance_hash: str
    generated_at: float
    quality_score: float
    paper_metadata: Dict[str, Any]

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

def calculate_paper_quality_score(paper: Dict[str, Any]) -> float:
    """Calculate quality score for paper prioritization"""
    score = 0.0
    
    # Recency bonus (higher for recent papers)
    if paper.get('publish_date'):
        try:
            year = int(paper['publish_date'][:4])
            if year >= 2020:
                score += 10.0
            elif year >= 2015:
                score += 7.0
            elif year >= 2010:
                score += 4.0
            else:
                score += 1.0
        except:
            score += 0.5
    
    # Content completeness
    if paper.get('has_full_content'):
        score += 15.0
    if paper.get('full_text') and len(str(paper['full_text'])) > 1000:
        score += 10.0
    if paper.get('introduction') and len(str(paper['introduction'])) > 200:
        score += 5.0
    if paper.get('abstract') and len(str(paper['abstract'])) > 100:
        score += 3.0
    
    # Existing embeddings
    if paper.get('enhanced_embedding_generated'):
        score += 8.0
    if paper.get('full_paper_embedding'):
        score += 5.0
    
    # Content length (optimal papers have substantial content)
    content_length = paper.get('content_length', 0) or 0
    if content_length > 10000:
        score += 5.0
    elif content_length > 5000:
        score += 3.0
    elif content_length > 1000:
        score += 1.0
    
    return score

def process_paper_worker(args: Tuple[Dict[str, Any], str, str]) -> Tuple[bool, str, Dict[str, Any]]:
    """Worker function to process a single paper - runs in separate process"""
    paper_data, processed_dir, nwtn_dir = args
    
    try:
        # Initialize embedding model in worker process (offline mode to avoid 429 errors)
        import os
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', use_auth_token=False)
        hash_generator = PostQuantumContentHashGenerator("sha3_256")
        
        arxiv_id = paper_data.get('arxiv_id') or paper_data.get('id', '').replace('arxiv:', '')
        if not arxiv_id:
            return False, f"No arXiv ID found", {}
            
        # Check if already processed (check both processed content and embeddings)
        processed_file = Path(processed_dir) / f"{arxiv_id}.json"
        embedding_file = Path(nwtn_dir) / "embeddings" / f"{arxiv_id}.json"
        if processed_file.exists() and embedding_file.exists():
            return True, f"Already processed: {arxiv_id}", {"skipped": True}
            
        # Extract and prepare content sections
        content_sections = {
            'title': paper_data.get('title', ''),
            'abstract': paper_data.get('abstract', ''),
            'introduction': paper_data.get('introduction', ''),
            'methodology': paper_data.get('methodology', ''),
            'results': paper_data.get('results', ''),
            'discussion': paper_data.get('discussion', ''),
            'conclusion': paper_data.get('conclusion', ''),
            'full_text': paper_data.get('full_text', '')
        }
        
        # Create comprehensive content for embedding and hashing
        content_parts = []
        for section, text in content_sections.items():
            if text and len(str(text).strip()) > 10:
                content_parts.append(f"[{section.upper()}]\\n{text}")
        
        full_content = "\\n\\n".join(content_parts)
        
        if len(full_content) < 100:
            return False, f"Insufficient content for {arxiv_id}", {}
        
        # Generate post-quantum content hash
        content_hash = hash_generator.generate_hash(full_content)
        
        # Calculate quality score
        quality_score = calculate_paper_quality_score(paper_data)
        
        # Prepare metadata
        paper_metadata = {
            'domain': paper_data.get('domain', ''),
            'categories': paper_data.get('categories', ''),
            'publish_date': paper_data.get('publish_date', ''),
            'authors': paper_data.get('authors', ''),
            'journal_ref': paper_data.get('journal_ref', ''),
            'content_length': paper_data.get('content_length', len(full_content)),
            'has_full_content': paper_data.get('has_full_content', False),
            'quality_score': quality_score
        }
        
        # Create provenance record
        provenance = ProvenanceRecord(
            arxiv_id=arxiv_id,
            source_url=f"https://arxiv.org/abs/{arxiv_id}",
            content_hash=content_hash,
            processing_timestamp=time.time()
        )
        
        # Generate embeddings (limit content to avoid model limits)
        embedding_content = full_content[:2000]  # First 2000 chars for embedding
        embedding_vector = embedding_model.encode(embedding_content).tolist()
        
        nwtn_embedding = NWTNEmbedding(
            arxiv_id=arxiv_id,
            embedding_vector=embedding_vector,
            embedding_model="all-MiniLM-L6-v2",
            content_sections=content_sections,
            provenance_hash=content_hash.hash_value,
            generated_at=time.time(),
            quality_score=quality_score,
            paper_metadata=paper_metadata
        )
        
        # Ensure directories exist
        processed_path = Path(processed_dir)
        nwtn_path = Path(nwtn_dir)
        processed_path.mkdir(parents=True, exist_ok=True)
        (nwtn_path / "content_hashes").mkdir(parents=True, exist_ok=True)
        (nwtn_path / "provenance").mkdir(parents=True, exist_ok=True)
        (nwtn_path / "embeddings").mkdir(parents=True, exist_ok=True)
        
        # Step 1: Save full processed content to 02_PROCESSED_CONTENT
        processed_content = {
            "arxiv_id": arxiv_id,
            "title": content_sections['title'],
            "abstract": content_sections['abstract'],
            "introduction": content_sections['introduction'],
            "methodology": content_sections['methodology'],
            "results": content_sections['results'],
            "discussion": content_sections['discussion'],
            "conclusion": content_sections['conclusion'],
            "full_text_preview": content_sections['full_text'][:3000],  # Preview for compatibility
            "full_text": content_sections['full_text'],  # Complete full text
            "content_length": len(full_content),
            "page_count": paper_data.get('content_length', 0) // 2000,  # Estimate
            "processed_at": time.time(),
            "metadata": paper_metadata,
            "source_database": "native_sqlite"
        }
        
        processed_file = processed_path / f"{arxiv_id}.json"
        with open(processed_file, 'w') as f:
            json.dump(processed_content, f, indent=2)
        
        # Step 2: Save to NWTN directories (embeddings reference processed content)
        hash_file = nwtn_path / "content_hashes" / f"{arxiv_id}.json"
        with open(hash_file, 'w') as f:
            json.dump(asdict(content_hash), f, indent=2)
            
        provenance_file = nwtn_path / "provenance" / f"{arxiv_id}.json"
        with open(provenance_file, 'w') as f:
            json.dump(asdict(provenance), f, indent=2)
            
        # Enhanced embedding with source reference
        nwtn_embedding.paper_metadata['source_file'] = str(processed_file)
        embedding_file = nwtn_path / "embeddings" / f"{arxiv_id}.json"
        with open(embedding_file, 'w') as f:
            json.dump(asdict(nwtn_embedding), f, indent=2)
            
        return True, f"Processed: {arxiv_id} (quality: {quality_score:.1f})", {
            "quality_score": quality_score,
            "content_length": len(full_content),
            "domain": paper_metadata['domain']
        }
        
    except Exception as e:
        return False, f"Error processing {arxiv_id}: {str(e)}", {}

class Native100KNWTNPipeline:
    """Native processing pipeline for 100K best papers"""
    
    def __init__(self, num_workers: int = None, target_papers: int = 100000):
        self.db_path = "/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/01_RAW_PAPERS/storage.db"
        self.processed_dir = Path("/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/02_PROCESSED_CONTENT")
        self.nwtn_dir = Path("/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY")
        self.progress_file = "/Users/ryneschultz/Documents/GitHub/PRSM/native_100k_progress.json"
        
        # Configuration
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.target_papers = target_papers
        
        # Create directories for both processed content and NWTN ready
        self.processed_dir.mkdir(exist_ok=True)
        self.nwtn_dir.mkdir(exist_ok=True)
        (self.nwtn_dir / "embeddings").mkdir(exist_ok=True)
        (self.nwtn_dir / "provenance").mkdir(exist_ok=True)
        (self.nwtn_dir / "content_hashes").mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            "target_papers": target_papers,
            "selected_papers": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "start_time": time.time(),
            "workers": self.num_workers,
            "quality_distribution": {},
            "domain_distribution": {}
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
    
    def select_best_papers(self) -> List[Dict[str, Any]]:
        """Select the best 100K papers from database"""
        logger.info(f"🔍 Selecting best {self.target_papers:,} papers from database...")
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        # Complex query to select diverse, high-quality papers
        query = """
        WITH ranked_papers AS (
            SELECT *,
                   CASE 
                       WHEN publish_date >= '2020-01-01' THEN 10
                       WHEN publish_date >= '2015-01-01' THEN 7
                       WHEN publish_date >= '2010-01-01' THEN 4
                       ELSE 1
                   END as recency_score,
                   CASE 
                       WHEN has_full_content = 1 THEN 15 ELSE 0
                   END as content_score,
                   CASE 
                       WHEN enhanced_embedding_generated = 1 THEN 8 ELSE 0
                   END as embedding_score,
                   CASE 
                       WHEN content_length > 10000 THEN 5
                       WHEN content_length > 5000 THEN 3
                       WHEN content_length > 1000 THEN 1
                       ELSE 0
                   END as length_score
            FROM arxiv_papers
            WHERE title IS NOT NULL 
              AND abstract IS NOT NULL
              AND LENGTH(abstract) > 100
        )
        SELECT *,
               (recency_score + content_score + embedding_score + length_score) as total_score
        FROM ranked_papers
        ORDER BY total_score DESC, publish_date DESC
        LIMIT ?
        """
        
        cursor.execute(query, (self.target_papers,))
        papers = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        self.stats['selected_papers'] = len(papers)
        
        # Analyze selection
        domain_counts = {}
        quality_scores = []
        
        for paper in papers:
            domain = paper.get('domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Calculate quality score for stats
            quality_score = calculate_paper_quality_score(paper)
            quality_scores.append(quality_score)
        
        self.stats['domain_distribution'] = domain_counts
        self.stats['quality_distribution'] = {
            'mean': np.mean(quality_scores) if quality_scores else 0,
            'median': np.median(quality_scores) if quality_scores else 0,
            'min': np.min(quality_scores) if quality_scores else 0,
            'max': np.max(quality_scores) if quality_scores else 0
        }
        
        logger.info(f"✅ Selected {len(papers):,} papers")
        logger.info(f"📊 Domain distribution: {domain_counts}")
        logger.info(f"🎯 Quality scores - Mean: {self.stats['quality_distribution']['mean']:.1f}, "
                   f"Range: {self.stats['quality_distribution']['min']:.1f}-{self.stats['quality_distribution']['max']:.1f}")
        
        return papers
    
    def run_native_pipeline(self):
        """Run the complete native 100K pipeline"""
        logger.info(f"🚀 Starting NATIVE 100K NWTN Pipeline with {self.num_workers} workers")
        
        # Step 1: Select best papers
        papers = self.select_best_papers()
        
        if not papers:
            logger.error("❌ No papers selected from database")
            return
        
        logger.info(f"📋 Processing {len(papers):,} selected papers...")
        
        # Step 2: Process with parallel workers
        batch_size = self.num_workers * 10  # 10 papers per worker per batch
        total_batches = (len(papers) + batch_size - 1) // batch_size
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for batch_num in range(0, len(papers), batch_size):
                batch_start = time.time()
                batch_papers = papers[batch_num:batch_num + batch_size]
                
                # Prepare worker arguments
                worker_args = [(paper, str(self.processed_dir), str(self.nwtn_dir)) for paper in batch_papers]
                
                # Submit batch to workers
                future_to_paper = {executor.submit(process_paper_worker, arg): arg[0] for arg in worker_args}
                
                # Collect results
                batch_results = {"success": 0, "skipped": 0, "errors": 0}
                batch_quality_scores = []
                batch_domains = []
                
                for future in as_completed(future_to_paper):
                    try:
                        success, message, metadata = future.result()
                        if success:
                            if metadata.get("skipped"):
                                batch_results["skipped"] += 1
                            else:
                                batch_results["success"] += 1
                                batch_quality_scores.append(metadata.get("quality_score", 0))
                                batch_domains.append(metadata.get("domain", "unknown"))
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
                total_done = self.stats["processed"] + self.stats["skipped"]
                
                avg_quality = np.mean(batch_quality_scores) if batch_quality_scores else 0
                
                logger.info(f"📦 Batch {batch_idx}/{total_batches}: "
                          f"{batch_results['success']} processed, "
                          f"{batch_results['skipped']} skipped, "
                          f"{batch_results['errors']} errors "
                          f"({batch_time:.1f}s) "
                          f"[Quality: {avg_quality:.1f}] "
                          f"[Total: {total_done:,}/{len(papers):,}]")
                
                # Progress summary every 10 batches
                if batch_idx % 10 == 0:
                    elapsed = time.time() - self.stats["start_time"]
                    rate = self.stats["processed"] / elapsed if elapsed > 0 else 0
                    remaining = len(papers) - total_done
                    eta = remaining / rate / 3600 if rate > 0 else 0
                    
                    progress_pct = (total_done / len(papers)) * 100
                    
                    logger.info(f"🎯 Progress: {total_done:,}/{len(papers):,} "
                              f"({progress_pct:.1f}%) "
                              f"Rate: {rate:.1f} papers/sec "
                              f"ETA: {eta:.1f} hours")
                    
                    self.save_progress()
        
        # Final statistics
        elapsed_time = time.time() - self.stats['start_time']
        logger.info(f"""
🎉 NATIVE 100K NWTN PIPELINE COMPLETE!
=====================================
🎯 Target papers: {self.stats['target_papers']:,}
📊 Papers selected: {self.stats['selected_papers']:,}
✅ Papers processed: {self.stats['processed']:,}
⏭️  Papers skipped: {self.stats['skipped']:,}
❌ Errors: {self.stats['errors']:,}
⚡ Workers used: {self.stats['workers']}
⏱️  Total time: {elapsed_time:.1f} seconds ({elapsed_time/3600:.1f} hours)
📈 Rate: {self.stats['processed'] / elapsed_time:.1f} papers/second
🔥 SUCCESS RATE: {(self.stats['processed']/(self.stats['processed']+self.stats['errors'])*100):.1f}%

📊 DOMAIN DISTRIBUTION:
{json.dumps(self.stats['domain_distribution'], indent=2)}

🎯 QUALITY METRICS:
Mean Quality Score: {self.stats['quality_distribution']['mean']:.1f}
Quality Range: {self.stats['quality_distribution']['min']:.1f} - {self.stats['quality_distribution']['max']:.1f}
        """)
        
        self.save_progress()
        
        # Create summary file
        summary_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/NATIVE_100K_NWTN_SUMMARY.md")
        with open(summary_file, 'w') as f:
            f.write(f"""# Native 100K NWTN Pipeline Results

## Processing Summary
- **Target Papers**: {self.stats['target_papers']:,}
- **Papers Selected**: {self.stats['selected_papers']:,}
- **Papers Processed**: {self.stats['processed']:,}
- **Processing Rate**: {self.stats['processed'] / elapsed_time:.1f} papers/second
- **Success Rate**: {(self.stats['processed']/(self.stats['processed']+self.stats['errors'])*100):.1f}%

## Quality Metrics
- **Mean Quality Score**: {self.stats['quality_distribution']['mean']:.1f}
- **Quality Range**: {self.stats['quality_distribution']['min']:.1f} - {self.stats['quality_distribution']['max']:.1f}

## Domain Distribution
{json.dumps(self.stats['domain_distribution'], indent=2)}

## NWTN Ready
The processed papers are now available in:
- **Embeddings**: `/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY/embeddings/`
- **Provenance**: `/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY/provenance/`
- **Content Hashes**: `/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY/content_hashes/`

Ready for quantum gravity analysis and other NWTN operations!
""")
        
        logger.info(f"📋 Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Native 100K NWTN Pipeline")
    parser.add_argument("--workers", type=int, default=None, 
                       help="Number of worker processes (default: CPU count - 1)")
    parser.add_argument("--target", type=int, default=100000,
                       help="Target number of papers to process (default: 100,000)")
    args = parser.parse_args()
    
    pipeline = Native100KNWTNPipeline(num_workers=args.workers, target_papers=args.target)
    pipeline.run_native_pipeline()

if __name__ == "__main__":
    main()