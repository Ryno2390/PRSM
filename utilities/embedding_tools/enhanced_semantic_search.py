#!/usr/bin/env python3
"""
Enhanced Semantic Search Engine for NWTN
========================================

This module provides production-ready semantic search capabilities
for NWTN using pre-built FAISS indices and paper embeddings.

Features:
1. Fast semantic search using FAISS indices
2. Multiple search strategies (exact, approximate, hybrid)
3. Result ranking and filtering
4. Integration with NWTN reasoning pipeline
5. Comprehensive result metadata
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
from datetime import datetime

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result from semantic search"""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    domain: str
    categories: List[str]
    published_date: str
    similarity_score: float
    embedding_model: str
    file_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.paper_id,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'domain': self.domain,
            'categories': self.categories,
            'published_date': self.published_date,
            'relevance_score': self.similarity_score,
            'embedding_model': self.embedding_model,
            'file_path': self.file_path
        }

@dataclass
class SearchQuery:
    """Search query with parameters"""
    query_text: str
    max_results: int = 10
    similarity_threshold: float = 0.7
    domains: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None
    
class EnhancedSemanticSearchEngine:
    """Production-ready semantic search engine for NWTN"""
    
    def __init__(self, 
                 index_dir: str = "/Volumes/My Passport/PRSM_Storage/PRSM_Indices",
                 index_type: str = "IVF",
                 model_name: str = "all-MiniLM-L6-v2"):
        
        self.index_dir = Path(index_dir)
        self.index_type = index_type
        self.model_name = model_name
        
        # Components
        self.faiss = None
        self.index = None
        self.paper_metadata = None
        self.embedding_model = None
        self.index_metadata = None
        
        # Statistics
        self.total_searches = 0
        self.total_search_time = 0.0
        
        logger.info(f"🔍 Initializing Enhanced Semantic Search Engine")
        logger.info(f"📁 Index directory: {self.index_dir}")
        logger.info(f"🔧 Index type: {self.index_type}")
        logger.info(f"🤖 Model: {self.model_name}")
        
    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize FAISS
            if not self._initialize_faiss():
                return False
            
            # Load index
            if not self._load_index():
                return False
            
            # Load paper metadata
            if not self._load_paper_metadata():
                return False
            
            # Initialize embedding model
            if not self._initialize_embedding_model():
                return False
            
            # Load index metadata
            self._load_index_metadata()
            
            logger.info("✅ Enhanced Semantic Search Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def _initialize_faiss(self) -> bool:
        """Initialize FAISS library"""
        try:
            import faiss
            self.faiss = faiss
            logger.info("✅ FAISS initialized")
            return True
        except ImportError:
            logger.error("❌ FAISS not available. Install with: pip install faiss-cpu")
            return False
    
    def _load_index(self) -> bool:
        """Load FAISS index from disk"""
        index_file = self.index_dir / f"faiss_index_{self.index_type.lower()}.index"
        
        if not index_file.exists():
            logger.error(f"❌ Index file not found: {index_file}")
            return False
        
        try:
            self.index = self.faiss.read_index(str(index_file))
            logger.info(f"✅ FAISS index loaded: {self.index.ntotal:,} vectors")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            return False
    
    def _load_paper_metadata(self) -> bool:
        """Load paper metadata from disk"""
        metadata_file = self.index_dir / f"paper_metadata_{self.index_type.lower()}.pkl"
        
        if not metadata_file.exists():
            logger.error(f"❌ Metadata file not found: {metadata_file}")
            return False
        
        try:
            with open(metadata_file, 'rb') as f:
                self.paper_metadata = pickle.load(f)
            logger.info(f"✅ Paper metadata loaded: {len(self.paper_metadata):,} papers")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {e}")
            return False
    
    def _initialize_embedding_model(self) -> bool:
        """Initialize sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Embedding model loaded: {self.model_name}")
            return True
        except ImportError:
            logger.error("❌ sentence-transformers not available")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            return False
    
    def _load_index_metadata(self):
        """Load index metadata"""
        metadata_file = self.index_dir / f"index_metadata_{self.index_type.lower()}.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.index_metadata = json.load(f)
                logger.info("✅ Index metadata loaded")
            except Exception as e:
                logger.warning(f"Failed to load index metadata: {e}")
    
    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode query text to embedding vector"""
        try:
            embedding = self.embedding_model.encode([query_text])[0]
            return embedding.reshape(1, -1).astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise
    
    def search_raw(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform raw FAISS search"""
        try:
            # Configure search parameters for IVF index
            if self.index_type == "IVF":
                # Set search parameters for better recall
                original_nprobe = self.index.nprobe
                self.index.nprobe = min(50, max(1, int(np.sqrt(k))))
            
            # Perform search
            scores, indices = self.index.search(query_embedding, k)
            
            # Restore original parameters
            if self.index_type == "IVF":
                self.index.nprobe = original_nprobe
            
            return scores, indices
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def filter_results(self, 
                      indices: np.ndarray, 
                      scores: np.ndarray,
                      query: SearchQuery) -> List[SearchResult]:
        """Filter and convert raw results to SearchResult objects"""
        results = []
        
        for score, idx in zip(scores[0], indices[0]):
            # Skip invalid indices
            if idx < 0 or idx >= len(self.paper_metadata):
                continue
            
            # Skip low similarity scores
            if score < query.similarity_threshold:
                continue
            
            # Get paper metadata
            paper = self.paper_metadata[idx]
            
            # Apply domain filter
            if query.domains and paper['domain'] not in query.domains:
                continue
            
            # Apply category filter
            if query.categories:
                paper_categories = paper.get('categories', [])
                if not any(cat in paper_categories for cat in query.categories):
                    continue
            
            # Apply date range filter
            if query.date_range:
                # This would need proper date parsing
                pass
            
            # Create search result
            result = SearchResult(
                paper_id=paper['paper_id'],
                title=paper['title'],
                abstract=paper['abstract'],
                authors=paper['authors'],
                domain=paper['domain'],
                categories=paper['categories'],
                published_date=paper['published_date'],
                similarity_score=float(score),
                embedding_model=paper['embedding_model'],
                file_path=paper['file_path']
            )
            
            results.append(result)
            
            # Limit results
            if len(results) >= query.max_results:
                break
        
        return results
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform semantic search with full pipeline"""
        start_time = time.time()
        
        try:
            # Encode query
            query_embedding = self.encode_query(query.query_text)
            
            # Perform search with extra results for filtering
            search_k = min(query.max_results * 3, 1000)
            scores, indices = self.search_raw(query_embedding, search_k)
            
            # Filter and convert results
            results = self.filter_results(indices, scores, query)
            
            # Update statistics
            search_time = time.time() - start_time
            self.total_searches += 1
            self.total_search_time += search_time
            
            logger.info(f"🔍 Search completed: '{query.query_text[:50]}...' -> {len(results)} results in {search_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        avg_search_time = self.total_search_time / self.total_searches if self.total_searches > 0 else 0
        
        stats = {
            'total_papers': len(self.paper_metadata) if self.paper_metadata else 0,
            'total_searches': self.total_searches,
            'average_search_time': avg_search_time,
            'index_type': self.index_type,
            'model_name': self.model_name
        }
        
        if self.index_metadata:
            stats.update({
                'index_build_time': self.index_metadata.get('build_time_seconds', 0),
                'embedding_dimension': self.index_metadata.get('embedding_dimension', 0),
                'index_created_at': self.index_metadata.get('created_at', 'unknown')
            })
        
        return stats

# Compatibility class for NWTN integration
class NWTNSemanticSearchEngine:
    """NWTN-compatible semantic search engine"""
    
    def __init__(self):
        self.engine = EnhancedSemanticSearchEngine()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the search engine"""
        if not self.initialized:
            self.initialized = self.engine.initialize()
        return self.initialized
    
    async def search_content(self, 
                           query: str, 
                           limit: int = 10, 
                           similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search interface compatible with NWTN"""
        if not self.initialized:
            await self.initialize()
        
        if not self.initialized:
            return []
        
        search_query = SearchQuery(
            query_text=query,
            max_results=limit,
            similarity_threshold=similarity_threshold
        )
        
        results = await self.engine.search(search_query)
        return [result.to_dict() for result in results]

async def main():
    """Test the enhanced semantic search engine"""
    engine = EnhancedSemanticSearchEngine()
    
    if not engine.initialize():
        logger.error("Failed to initialize search engine")
        return
    
    # Test queries
    test_queries = [
        "quantum mechanics and relativity theory",
        "machine learning neural networks",
        "computer vision object detection",
        "natural language processing transformers",
        "reinforcement learning algorithms"
    ]
    
    for query_text in test_queries:
        query = SearchQuery(
            query_text=query_text,
            max_results=5,
            similarity_threshold=0.5
        )
        
        results = await engine.search(query)
        
        print(f"\n🔍 Query: '{query_text}'")
        print(f"📊 Results: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. ({result.similarity_score:.3f}) {result.title}")
            print(f"     Domain: {result.domain} | Authors: {', '.join(result.authors[:2])}")
    
    # Print statistics
    stats = engine.get_statistics()
    print(f"\n📊 Search Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())