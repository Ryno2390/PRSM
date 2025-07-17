#!/usr/bin/env python3
"""
FAISS Index Builder for NWTN Paper Embeddings
=============================================

This script builds FAISS indices from the processed paper embeddings
for efficient semantic search in NWTN.

Features:
1. Load processed paper embeddings
2. Build optimized FAISS indices
3. Support for multiple index types (IVF, HNSW, etc.)
4. Index metadata and statistics
5. Validation and testing
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pickle
import logging
from dataclasses import dataclass

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

# Import the PaperEmbedding class
from build_paper_embeddings import PaperEmbedding, PaperData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IndexMetadata:
    """Metadata for FAISS index"""
    index_type: str
    total_embeddings: int
    embedding_dimension: int
    model_name: str
    created_at: str
    index_parameters: Dict[str, Any]
    build_time_seconds: float

class FAISSIndexBuilder:
    """Builder for FAISS indices from paper embeddings"""
    
    def __init__(self, 
                 embeddings_dir: str = "/Volumes/My Passport/PRSM_Storage/PRSM_Embeddings",
                 index_dir: str = "/Volumes/My Passport/PRSM_Storage/PRSM_Indices",
                 index_type: str = "IVF"):
        
        self.embeddings_dir = Path(embeddings_dir)
        self.index_dir = Path(index_dir)
        self.index_type = index_type
        
        # Create index directory
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS
        self.faiss = None
        self.index = None
        self.paper_metadata = []
        
        logger.info(f"🏗️  Initializing FAISS Index Builder")
        logger.info(f"📁 Embeddings directory: {self.embeddings_dir}")
        logger.info(f"💾 Index directory: {self.index_dir}")
        logger.info(f"🔧 Index type: {self.index_type}")
        
    def initialize_faiss(self):
        """Initialize FAISS library"""
        try:
            import faiss
            self.faiss = faiss
            logger.info("✅ FAISS initialized successfully")
            return True
        except ImportError:
            logger.error("❌ FAISS not available. Install with: pip install faiss-cpu")
            return False
    
    def load_embeddings(self, max_batches: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load all embeddings from batch files"""
        logger.info("📥 Loading embeddings from batch files...")
        
        batch_files = list(self.embeddings_dir.glob("embeddings_batch_*.pkl"))
        batch_files.sort()  # Ensure consistent order
        
        if not batch_files:
            raise ValueError(f"No embedding batch files found in {self.embeddings_dir}")
        
        # Limit batches for testing/partial builds
        if max_batches:
            batch_files = batch_files[:max_batches]
        
        logger.info(f"📦 Found {len(batch_files)} batch files to process")
        
        all_embeddings = []
        all_metadata = []
        failed_batches = 0
        
        for i, batch_file in enumerate(batch_files):
            try:
                with open(batch_file, 'rb') as f:
                    batch_embeddings = pickle.load(f)
                
                batch_size = len(batch_embeddings)
                
                for paper_embedding in batch_embeddings:
                    all_embeddings.append(paper_embedding.embedding)
                    
                    # Store metadata for later lookup
                    metadata = {
                        'paper_id': paper_embedding.paper_data.paper_id,
                        'title': paper_embedding.paper_data.title,
                        'abstract': paper_embedding.paper_data.abstract,
                        'authors': paper_embedding.paper_data.authors,
                        'domain': paper_embedding.paper_data.domain,
                        'categories': paper_embedding.paper_data.categories,
                        'published_date': paper_embedding.paper_data.published_date,
                        'file_path': paper_embedding.paper_data.file_path,
                        'embedding_model': paper_embedding.embedding_model
                    }
                    all_metadata.append(metadata)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"📊 Loaded {i + 1}/{len(batch_files)} batches, {len(all_embeddings):,} embeddings")
                    
            except Exception as e:
                logger.error(f"Failed to load batch {batch_file}: {e}")
                failed_batches += 1
                continue
        
        if not all_embeddings:
            raise ValueError("No embeddings loaded successfully")
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        logger.info(f"✅ Loaded {len(all_embeddings):,} embeddings from {len(batch_files)} batches")
        logger.info(f"📐 Embedding shape: {embeddings_array.shape}")
        if failed_batches > 0:
            logger.warning(f"⚠️  {failed_batches} batches failed to load")
        
        return embeddings_array, all_metadata
    
    def build_flat_index(self, embeddings: np.ndarray) -> object:
        """Build a flat (brute-force) index for exact search"""
        logger.info("🔧 Building Flat index...")
        
        dimension = embeddings.shape[1]
        index = self.faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Add embeddings to index
        index.add(embeddings)
        
        logger.info(f"✅ Flat index built with {index.ntotal:,} vectors")
        return index
    
    def build_ivf_index(self, embeddings: np.ndarray, nlist: int = 1000) -> object:
        """Build an IVF index for fast approximate search"""
        logger.info(f"🔧 Building IVF index with {nlist} clusters...")
        
        dimension = embeddings.shape[1]
        n_embeddings = embeddings.shape[0]
        
        # Create quantizer and index
        quantizer = self.faiss.IndexFlatIP(dimension)
        index = self.faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        # Train the index
        logger.info("🎯 Training IVF index...")
        training_sample_size = min(n_embeddings, 100000)  # Use subset for training
        training_indices = np.random.choice(n_embeddings, training_sample_size, replace=False)
        training_data = embeddings[training_indices]
        
        index.train(training_data)
        
        # Add embeddings to index
        logger.info("➕ Adding embeddings to index...")
        index.add(embeddings)
        
        logger.info(f"✅ IVF index built with {index.ntotal:,} vectors")
        return index
    
    def build_hnsw_index(self, embeddings: np.ndarray, M: int = 16) -> object:
        """Build an HNSW index for very fast approximate search"""
        logger.info(f"🔧 Building HNSW index with M={M}...")
        
        dimension = embeddings.shape[1]
        index = self.faiss.IndexHNSWFlat(dimension, M)
        
        # Set construction parameters
        index.hnsw.efConstruction = 40
        
        # Add embeddings to index
        logger.info("➕ Adding embeddings to index...")
        index.add(embeddings)
        
        logger.info(f"✅ HNSW index built with {index.ntotal:,} vectors")
        return index
    
    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> IndexMetadata:
        """Build the specified index type"""
        logger.info(f"🏗️  Building {self.index_type} index...")
        start_time = time.time()
        
        # Store metadata for later lookup
        self.paper_metadata = metadata
        
        # Build appropriate index type
        if self.index_type == "Flat":
            self.index = self.build_flat_index(embeddings)
            index_params = {"type": "Flat"}
        elif self.index_type == "IVF":
            nlist = min(int(np.sqrt(embeddings.shape[0])), 4000)  # Adaptive nlist
            self.index = self.build_ivf_index(embeddings, nlist)
            index_params = {"type": "IVF", "nlist": nlist}
        elif self.index_type == "HNSW":
            self.index = self.build_hnsw_index(embeddings)
            index_params = {"type": "HNSW", "M": 16}
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        build_time = time.time() - start_time
        
        # Create metadata
        index_metadata = IndexMetadata(
            index_type=self.index_type,
            total_embeddings=embeddings.shape[0],
            embedding_dimension=embeddings.shape[1],
            model_name=metadata[0]['embedding_model'] if metadata else "unknown",
            created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            index_parameters=index_params,
            build_time_seconds=build_time
        )
        
        logger.info(f"✅ Index built in {build_time:.1f}s")
        return index_metadata
    
    def save_index(self, index_metadata: IndexMetadata):
        """Save the index and metadata to disk"""
        logger.info("💾 Saving index to disk...")
        
        # Save FAISS index
        index_file = self.index_dir / f"faiss_index_{self.index_type.lower()}.index"
        self.faiss.write_index(self.index, str(index_file))
        
        # Save paper metadata
        metadata_file = self.index_dir / f"paper_metadata_{self.index_type.lower()}.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.paper_metadata, f)
        
        # Save index metadata
        index_metadata_file = self.index_dir / f"index_metadata_{self.index_type.lower()}.json"
        with open(index_metadata_file, 'w') as f:
            json.dump({
                'index_type': index_metadata.index_type,
                'total_embeddings': index_metadata.total_embeddings,
                'embedding_dimension': index_metadata.embedding_dimension,
                'model_name': index_metadata.model_name,
                'created_at': index_metadata.created_at,
                'index_parameters': index_metadata.index_parameters,
                'build_time_seconds': index_metadata.build_time_seconds
            }, f, indent=2)
        
        logger.info(f"✅ Index saved:")
        logger.info(f"  📁 Index: {index_file}")
        logger.info(f"  📄 Metadata: {metadata_file}")
        logger.info(f"  ℹ️  Info: {index_metadata_file}")
    
    def test_index(self, test_queries: List[str] = None, k: int = 5):
        """Test the index with sample queries"""
        if not self.index or not self.paper_metadata:
            logger.warning("No index loaded for testing")
            return
        
        logger.info(f"🧪 Testing index with {k} results per query...")
        
        # Default test queries
        if test_queries is None:
            test_queries = [
                "quantum mechanics and relativity",
                "machine learning algorithms",
                "neural networks deep learning",
                "computer vision object detection",
                "natural language processing transformers"
            ]
        
        # Initialize embedding model for query encoding
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            for query in test_queries:
                query_embedding = model.encode([query])[0].reshape(1, -1)
                
                # Search the index
                scores, indices = self.index.search(query_embedding, k)
                
                logger.info(f"🔍 Query: '{query}'")
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(self.paper_metadata):
                        paper = self.paper_metadata[idx]
                        logger.info(f"  {i+1}. ({score:.3f}) {paper['title'][:80]}...")
                logger.info("")
                
        except Exception as e:
            logger.error(f"Test failed: {e}")
    
    def build_complete_index(self, max_batches: Optional[int] = None):
        """Build the complete index pipeline"""
        logger.info("🚀 Starting FAISS index building pipeline...")
        
        # Initialize FAISS
        if not self.initialize_faiss():
            return False
        
        # Load embeddings
        embeddings, metadata = self.load_embeddings(max_batches)
        
        # Build index
        index_metadata = self.build_index(embeddings, metadata)
        
        # Save index
        self.save_index(index_metadata)
        
        # Test index
        self.test_index()
        
        logger.info("✅ FAISS index building completed successfully!")
        return True
        
    def build_incremental_index(self, batch_size: int = 1000):
        """Build index incrementally for very large datasets"""
        logger.info("🚀 Starting incremental FAISS index building...")
        
        # Initialize FAISS
        if not self.initialize_faiss():
            return False
        
        batch_files = list(self.embeddings_dir.glob("embeddings_batch_*.pkl"))
        batch_files.sort()
        
        if not batch_files:
            raise ValueError(f"No embedding batch files found in {self.embeddings_dir}")
        
        logger.info(f"📦 Processing {len(batch_files)} batch files incrementally")
        
        # Initialize index with first batch to get dimensions
        first_batch_embeddings, first_batch_metadata = self.load_embeddings(max_batches=1)
        dimension = first_batch_embeddings.shape[1]
        
        # Create index
        if self.index_type == "Flat":
            self.index = self.faiss.IndexFlatIP(dimension)
        elif self.index_type == "IVF":
            quantizer = self.faiss.IndexFlatIP(dimension)
            nlist = min(int(np.sqrt(len(batch_files) * 32)), 4000)  # Estimate based on batch count
            self.index = self.faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif self.index_type == "HNSW":
            self.index = self.faiss.IndexHNSWFlat(dimension, 16)
        
        # Add first batch
        self.index.add(first_batch_embeddings)
        self.paper_metadata = first_batch_metadata
        
        # Process remaining batches
        for i, batch_file in enumerate(batch_files[1:], 1):
            try:
                with open(batch_file, 'rb') as f:
                    batch_embeddings = pickle.load(f)
                
                embeddings_array = np.array([pe.embedding for pe in batch_embeddings], dtype=np.float32)
                
                # Add to index
                self.index.add(embeddings_array)
                
                # Add metadata
                for paper_embedding in batch_embeddings:
                    metadata = {
                        'paper_id': paper_embedding.paper_data.paper_id,
                        'title': paper_embedding.paper_data.title,
                        'abstract': paper_embedding.paper_data.abstract,
                        'authors': paper_embedding.paper_data.authors,
                        'domain': paper_embedding.paper_data.domain,
                        'categories': paper_embedding.paper_data.categories,
                        'published_date': paper_embedding.paper_data.published_date,
                        'file_path': paper_embedding.paper_data.file_path,
                        'embedding_model': paper_embedding.embedding_model
                    }
                    self.paper_metadata.append(metadata)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(batch_files)} batches, {self.index.ntotal:,} embeddings")
                    
            except Exception as e:
                logger.error(f"Failed to process batch {batch_file}: {e}")
                continue
        
        # Create metadata
        index_metadata = IndexMetadata(
            index_type=self.index_type,
            total_embeddings=self.index.ntotal,
            embedding_dimension=dimension,
            model_name=self.paper_metadata[0]['embedding_model'] if self.paper_metadata else "unknown",
            created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            index_parameters={"type": self.index_type},
            build_time_seconds=0.0  # Not tracking time for incremental
        )
        
        # Save index
        self.save_index(index_metadata)
        
        # Test index
        self.test_index()
        
        logger.info("✅ Incremental FAISS index building completed successfully!")
        return True

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index for paper embeddings")
    parser.add_argument("--index-type", choices=["Flat", "IVF", "HNSW"], default="IVF",
                       help="Type of FAISS index to build")
    parser.add_argument("--embeddings-dir", default="/Volumes/My Passport/PRSM_Storage/PRSM_Embeddings",
                       help="Directory containing embedding batch files")
    parser.add_argument("--index-dir", default="/Volumes/My Passport/PRSM_Storage/PRSM_Indices",
                       help="Directory to save FAISS indices")
    parser.add_argument("--max-batches", type=int, default=None,
                       help="Maximum number of batches to process (for testing)")
    parser.add_argument("--incremental", action="store_true",
                       help="Build index incrementally for large datasets")
    
    args = parser.parse_args()
    
    builder = FAISSIndexBuilder(
        embeddings_dir=args.embeddings_dir,
        index_dir=args.index_dir,
        index_type=args.index_type
    )
    
    if args.incremental:
        success = builder.build_incremental_index()
    else:
        success = builder.build_complete_index(max_batches=args.max_batches)
    
    if success:
        logger.info("🎉 Index building completed successfully!")
    else:
        logger.error("❌ Index building failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()