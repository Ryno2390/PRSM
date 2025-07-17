#!/usr/bin/env python3
"""
Test Production FAISS Indices
=============================

This script tests different FAISS index types for performance and accuracy
with the full corpus of 151,120 papers.
"""

import time
import json
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from enhanced_semantic_search import EnhancedSemanticSearchEngine, SearchQuery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionIndexTester:
    """Test production FAISS indices for performance and accuracy"""
    
    def __init__(self):
        self.indices_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Indices")
        self.test_queries = [
            "quantum mechanics and relativity theory",
            "machine learning neural networks deep learning",
            "computer vision object detection algorithms",
            "natural language processing transformers",
            "reinforcement learning optimization",
            "black holes and general relativity",
            "protein folding molecular dynamics",
            "climate change environmental modeling",
            "cryptography quantum computing",
            "robotics autonomous systems",
            "CRISPR gene editing biotechnology",
            "blockchain distributed systems",
            "photonic quantum computing",
            "metabolic pathway biochemistry",
            "topological quantum materials"
        ]
    
    def get_available_indices(self) -> List[Dict]:
        """Get information about available indices"""
        indices = []
        
        # Look for index files
        for index_file in self.indices_dir.glob("*.index"):
            index_name = index_file.stem
            
            # Determine index type from filename
            if "flat" in index_name.lower():
                index_type = "Flat"
            elif "ivf" in index_name.lower():
                index_type = "IVF"
            elif "hnsw" in index_name.lower():
                index_type = "HNSW"
            else:
                index_type = "Unknown"
            
            # Look for metadata file
            metadata_file = None
            for pattern in [f"index_metadata_{index_type.lower()}.json", f"*{index_type.lower()}*.json"]:
                matches = list(self.indices_dir.glob(pattern))
                if matches:
                    metadata_file = matches[0]
                    break
            
            index_info = {
                "name": index_name,
                "type": index_type,
                "file_path": str(index_file),
                "file_size_mb": index_file.stat().st_size / (1024**2),
                "metadata_file": str(metadata_file) if metadata_file else None,
                "total_embeddings": 0
            }
            
            # Load metadata if available
            if metadata_file and metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    index_info.update({
                        "total_embeddings": metadata.get("total_embeddings", 0),
                        "embedding_dimension": metadata.get("embedding_dimension", 0),
                        "build_time_seconds": metadata.get("build_time_seconds", 0),
                        "created_at": metadata.get("created_at", "unknown")
                    })
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {index_name}: {e}")
            
            indices.append(index_info)
        
        return indices
    
    async def test_index_performance(self, index_info: Dict) -> Dict:
        """Test performance of a specific index"""
        logger.info(f"🧪 Testing {index_info['type']} index: {index_info['name']}")
        
        try:
            # Initialize search engine
            search_engine = EnhancedSemanticSearchEngine(
                index_dir=str(self.indices_dir),
                index_type=index_info['type']
            )
            
            if not search_engine.initialize():
                logger.error(f"Failed to initialize search engine for {index_info['type']}")
                return {"error": "Failed to initialize"}
            
            # Test queries
            results = []
            total_search_time = 0
            
            for query_text in self.test_queries:
                start_time = time.time()
                
                query = SearchQuery(
                    query_text=query_text,
                    max_results=10,
                    similarity_threshold=0.1
                )
                
                search_results = await search_engine.search(query)
                
                search_time = time.time() - start_time
                total_search_time += search_time
                
                results.append({
                    "query": query_text,
                    "results_count": len(search_results),
                    "search_time": search_time,
                    "top_similarity": search_results[0].similarity_score if search_results else 0.0,
                    "avg_similarity": sum(r.similarity_score for r in search_results) / len(search_results) if search_results else 0.0
                })
                
                logger.info(f"  Query: '{query_text[:50]}...' -> {len(search_results)} results in {search_time:.3f}s")
            
            # Calculate statistics
            avg_search_time = total_search_time / len(self.test_queries)
            avg_results_count = sum(r["results_count"] for r in results) / len(results)
            avg_top_similarity = sum(r["top_similarity"] for r in results) / len(results)
            
            return {
                "index_type": index_info['type'],
                "index_name": index_info['name'],
                "total_embeddings": index_info.get('total_embeddings', 0),
                "file_size_mb": index_info['file_size_mb'],
                "test_results": results,
                "performance_metrics": {
                    "avg_search_time": avg_search_time,
                    "avg_results_count": avg_results_count,
                    "avg_top_similarity": avg_top_similarity,
                    "total_test_time": total_search_time,
                    "queries_tested": len(self.test_queries)
                }
            }
            
        except Exception as e:
            logger.error(f"Error testing {index_info['type']} index: {e}")
            return {"error": str(e)}
    
    async def run_comprehensive_test(self) -> Dict:
        """Run comprehensive test of all available indices"""
        logger.info("🚀 Starting comprehensive production index testing")
        
        available_indices = self.get_available_indices()
        
        if not available_indices:
            logger.warning("No indices found to test")
            return {"error": "No indices available"}
        
        logger.info(f"📊 Found {len(available_indices)} indices to test:")
        for idx in available_indices:
            logger.info(f"  {idx['type']}: {idx['name']} ({idx['total_embeddings']:,} embeddings, {idx['file_size_mb']:.1f}MB)")
        
        # Test each index
        test_results = []
        
        for index_info in available_indices:
            result = await self.test_index_performance(index_info)
            test_results.append(result)
            
            # Log performance summary
            if "performance_metrics" in result:
                metrics = result["performance_metrics"]
                logger.info(f"✅ {index_info['type']} Performance:")
                logger.info(f"  Avg search time: {metrics['avg_search_time']:.3f}s")
                logger.info(f"  Avg results: {metrics['avg_results_count']:.1f}")
                logger.info(f"  Avg top similarity: {metrics['avg_top_similarity']:.3f}")
        
        # Generate comparison report
        comparison_report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_indices_tested": len(available_indices),
            "test_queries_count": len(self.test_queries),
            "index_results": test_results
        }
        
        # Save report
        with open("production_index_test_report.json", "w") as f:
            json.dump(comparison_report, f, indent=2)
        
        logger.info("✅ Comprehensive index testing completed")
        return comparison_report
    
    def print_performance_comparison(self, report: Dict):
        """Print performance comparison between indices"""
        logger.info("📊 Performance Comparison:")
        logger.info("=" * 80)
        
        # Extract performance metrics
        performances = []
        for result in report["index_results"]:
            if "performance_metrics" in result:
                performances.append({
                    "type": result["index_type"],
                    "name": result["index_name"],
                    "search_time": result["performance_metrics"]["avg_search_time"],
                    "results_count": result["performance_metrics"]["avg_results_count"],
                    "similarity": result["performance_metrics"]["avg_top_similarity"],
                    "embeddings": result["total_embeddings"]
                })
        
        # Sort by search time (fastest first)
        performances.sort(key=lambda x: x["search_time"])
        
        logger.info(f"{'Type':<8} {'Name':<20} {'Search Time':<12} {'Results':<8} {'Similarity':<10} {'Embeddings':<12}")
        logger.info("-" * 80)
        
        for perf in performances:
            logger.info(f"{perf['type']:<8} {perf['name'][:20]:<20} {perf['search_time']:.3f}s      {perf['results_count']:.1f}    {perf['similarity']:.3f}     {perf['embeddings']:,}")

async def main():
    """Main function"""
    tester = ProductionIndexTester()
    
    # Check available indices
    available_indices = tester.get_available_indices()
    
    if not available_indices:
        logger.warning("No indices found. Please build indices first.")
        return
    
    logger.info(f"📊 Found {len(available_indices)} indices to test")
    
    # Run comprehensive test
    report = await tester.run_comprehensive_test()
    
    # Print comparison
    if "index_results" in report:
        tester.print_performance_comparison(report)
    
    logger.info("🎉 Production index testing completed!")

if __name__ == "__main__":
    asyncio.run(main())