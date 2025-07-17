#!/usr/bin/env python3
"""
NWTN Paper Embedding Pipeline
============================

This script processes 150k+ arXiv papers and builds semantic embeddings
for NWTN's advanced reasoning capabilities.

Pipeline:
1. Process compressed paper files (.dat)
2. Extract text content (title + abstract)
3. Generate semantic embeddings using sentence-transformers
4. Build FAISS indices for efficient similarity search
5. Store embeddings for NWTN semantic search
"""

import asyncio
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import logging

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PaperData:
    """Structured paper data for embedding"""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    domain: str
    categories: List[str]
    published_date: str
    file_path: str
    
    def get_text_content(self) -> str:
        """Get combined text content for embedding"""
        return f"{self.title}\n\n{self.abstract}"

@dataclass
class PaperEmbedding:
    """Paper with its embedding vector"""
    paper_data: PaperData
    embedding: np.ndarray
    embedding_model: str
    created_at: str

class PaperEmbeddingPipeline:
    """Complete pipeline for processing arXiv papers into embeddings"""
    
    def __init__(self, 
                 papers_dir: str = "/Volumes/My Passport/PRSM_Storage/PRSM_Content/hot",
                 output_dir: str = "/Volumes/My Passport/PRSM_Storage/PRSM_Embeddings",
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 max_workers: int = 4):
        
        self.papers_dir = Path(papers_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = None
        self.embedding_dimension = None
        
        # Statistics
        self.processed_count = 0
        self.failed_count = 0
        self.total_papers = 0
        
        logger.info(f"🚀 Initializing Paper Embedding Pipeline")
        logger.info(f"📁 Papers directory: {self.papers_dir}")
        logger.info(f"💾 Output directory: {self.output_dir}")
        logger.info(f"🤖 Model: {self.model_name}")
        logger.info(f"📦 Batch size: {self.batch_size}")
        
    def initialize_embedding_model(self):
        """Initialize the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"🔧 Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # Test embedding to get dimension
            test_embedding = self.embedding_model.encode(["test"])
            self.embedding_dimension = test_embedding.shape[1]
            logger.info(f"✅ Model loaded successfully (dimension: {self.embedding_dimension})")
            
        except ImportError:
            logger.error("❌ sentence-transformers not available. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def discover_paper_files(self) -> List[Path]:
        """Discover all paper files in the directory"""
        logger.info("🔍 Discovering paper files...")
        
        paper_files = list(self.papers_dir.rglob("*.dat"))
        self.total_papers = len(paper_files)
        
        logger.info(f"📄 Found {self.total_papers:,} paper files")
        return paper_files
    
    def load_paper_data(self, file_path: Path) -> Optional[PaperData]:
        """Load and parse a single paper file"""
        try:
            # Papers are stored in pickle format
            import pickle
            with gzip.open(file_path, 'rb') as f:
                paper_json = pickle.load(f)
            
            paper_data = PaperData(
                paper_id=paper_json.get('id', 'unknown'),
                title=paper_json.get('title', 'Unknown Title'),
                abstract=paper_json.get('abstract', 'No abstract available'),
                authors=paper_json.get('authors', []),
                domain=paper_json.get('domain', 'unknown'),
                categories=paper_json.get('categories', []),
                published_date=paper_json.get('published_date', 'unknown'),
                file_path=str(file_path)
            )
            
            return paper_data
            
        except Exception as e:
            logger.debug(f"Failed to load paper {file_path}: {e}")
            return None
    
    def process_paper_batch(self, paper_files: List[Path]) -> List[PaperEmbedding]:
        """Process a batch of papers into embeddings"""
        papers_data = []
        
        # Load paper data
        for file_path in paper_files:
            paper_data = self.load_paper_data(file_path)
            if paper_data:
                papers_data.append(paper_data)
        
        if not papers_data:
            return []
        
        # Extract text content for embedding
        text_contents = [paper.get_text_content() for paper in papers_data]
        
        # Generate embeddings
        try:
            embeddings = self.embedding_model.encode(
                text_contents,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            # Create PaperEmbedding objects
            paper_embeddings = []
            for paper_data, embedding in zip(papers_data, embeddings):
                paper_embedding = PaperEmbedding(
                    paper_data=paper_data,
                    embedding=embedding,
                    embedding_model=self.model_name,
                    created_at=time.strftime('%Y-%m-%d %H:%M:%S')
                )
                paper_embeddings.append(paper_embedding)
            
            self.processed_count += len(paper_embeddings)
            return paper_embeddings
            
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            self.failed_count += len(papers_data)
            return []
    
    def save_embeddings_batch(self, embeddings: List[PaperEmbedding], batch_id: int):
        """Save a batch of embeddings to disk"""
        if not embeddings:
            return
        
        batch_file = self.output_dir / f"embeddings_batch_{batch_id:06d}.pkl"
        
        try:
            with open(batch_file, 'wb') as f:
                pickle.dump(embeddings, f)
            
            logger.info(f"💾 Saved batch {batch_id}: {len(embeddings)} embeddings -> {batch_file}")
            
        except Exception as e:
            logger.error(f"Failed to save batch {batch_id}: {e}")
    
    async def process_all_papers(self):
        """Process all papers through the embedding pipeline"""
        logger.info("🚀 Starting paper embedding pipeline...")
        
        # Initialize model
        self.initialize_embedding_model()
        
        # Discover papers
        paper_files = self.discover_paper_files()
        
        if not paper_files:
            logger.warning("No paper files found!")
            return
        
        # Process in batches
        batch_id = 0
        start_time = time.time()
        
        # Create batches of files
        for i in range(0, len(paper_files), self.batch_size):
            batch_files = paper_files[i:i + self.batch_size]
            
            # Process batch
            embeddings = self.process_paper_batch(batch_files)
            
            # Save batch
            if embeddings:
                self.save_embeddings_batch(embeddings, batch_id)
            
            batch_id += 1
            
            # Progress reporting
            if batch_id % 50 == 0:
                elapsed = time.time() - start_time
                progress = (self.processed_count + self.failed_count) / self.total_papers * 100
                rate = (self.processed_count + self.failed_count) / elapsed if elapsed > 0 else 0
                
                logger.info(f"📊 Progress: {progress:.1f}% | "
                           f"Processed: {self.processed_count:,} | "
                           f"Failed: {self.failed_count:,} | "
                           f"Rate: {rate:.1f} papers/sec")
        
        # Final statistics
        total_time = time.time() - start_time
        logger.info(f"✅ Pipeline completed in {total_time:.1f}s")
        logger.info(f"📊 Final stats: {self.processed_count:,} processed, {self.failed_count:,} failed")
        
        # Save metadata
        self.save_pipeline_metadata()
    
    def save_pipeline_metadata(self):
        """Save pipeline metadata and statistics"""
        metadata = {
            'total_papers': self.total_papers,
            'processed_count': self.processed_count,
            'failed_count': self.failed_count,
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'batch_size': self.batch_size,
            'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = self.output_dir / "pipeline_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Pipeline metadata saved to {metadata_file}")

async def main():
    """Main execution function"""
    pipeline = PaperEmbeddingPipeline()
    await pipeline.process_all_papers()

if __name__ == "__main__":
    asyncio.run(main())