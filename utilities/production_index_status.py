#!/usr/bin/env python3
"""
Production Index Status Report
==============================

Generate a comprehensive status report of all production FAISS indices.
"""

import json
import time
from pathlib import Path
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_production_index_status() -> Dict:
    """Get comprehensive status of production indices"""
    indices_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Indices")
    
    if not indices_dir.exists():
        return {"error": "Indices directory not found"}
    
    # Get all index files
    index_files = list(indices_dir.glob("*.index"))
    
    indices_info = []
    total_size_mb = 0
    
    for index_file in index_files:
        file_size_mb = index_file.stat().st_size / (1024**2)
        total_size_mb += file_size_mb
        
        # Determine index type
        index_name = index_file.stem
        if "flat" in index_name.lower():
            index_type = "Flat"
        elif "hnsw" in index_name.lower():
            index_type = "HNSW"  
        elif "ivf" in index_name.lower():
            index_type = "IVF"
        else:
            index_type = "Unknown"
        
        # Look for metadata
        metadata_file = indices_dir / f"index_metadata_{index_type.lower()}.json"
        metadata = {}
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Look for paper metadata
        paper_metadata_file = indices_dir / f"paper_metadata_{index_type.lower()}.pkl"
        has_paper_metadata = paper_metadata_file.exists()
        paper_metadata_size_mb = 0
        if has_paper_metadata:
            paper_metadata_size_mb = paper_metadata_file.stat().st_size / (1024**2)
        
        index_info = {
            "name": index_name,
            "type": index_type,
            "file_size_mb": file_size_mb,
            "created_at": metadata.get("created_at", "unknown"),
            "total_embeddings": metadata.get("total_embeddings", 0),
            "embedding_dimension": metadata.get("embedding_dimension", 0),
            "model_name": metadata.get("model_name", "unknown"),
            "build_time_seconds": metadata.get("build_time_seconds", 0),
            "has_paper_metadata": has_paper_metadata,
            "paper_metadata_size_mb": paper_metadata_size_mb
        }
        
        indices_info.append(index_info)
        total_size_mb += paper_metadata_size_mb
    
    # Sort by total embeddings (descending)
    indices_info.sort(key=lambda x: x["total_embeddings"], reverse=True)
    
    return {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "total_indices": len(indices_info),
        "total_size_mb": total_size_mb,
        "indices": indices_info
    }

def print_status_report(status: Dict):
    """Print formatted status report"""
    print("🏗️  PRODUCTION FAISS INDICES STATUS REPORT")
    print("=" * 80)
    print(f"📅 Generated: {status['timestamp']}")
    print(f"📊 Total Indices: {status['total_indices']}")
    print(f"💾 Total Size: {status['total_size_mb']:.1f} MB")
    print()
    
    if status['total_indices'] == 0:
        print("⚠️  No indices found")
        return
    
    print(f"{'Type':<8} {'Name':<25} {'Embeddings':<12} {'Size (MB)':<10} {'Created':<20}")
    print("-" * 80)
    
    for idx in status['indices']:
        print(f"{idx['type']:<8} {idx['name'][:25]:<25} {idx['total_embeddings']:,}      {idx['file_size_mb']:.1f}      {idx['created_at'][:19]}")
    
    print()
    print("📋 Index Details:")
    print("-" * 40)
    
    for idx in status['indices']:
        print(f"🔍 {idx['type']} Index: {idx['name']}")
        print(f"   📊 Embeddings: {idx['total_embeddings']:,}")
        print(f"   📐 Dimension: {idx['embedding_dimension']}")
        print(f"   🤖 Model: {idx['model_name']}")
        print(f"   💾 Index Size: {idx['file_size_mb']:.1f} MB")
        print(f"   📄 Metadata: {idx['paper_metadata_size_mb']:.1f} MB")
        print(f"   ⏱️  Build Time: {idx['build_time_seconds']:.1f}s")
        print(f"   📅 Created: {idx['created_at']}")
        print()

def main():
    """Main function"""
    print("📊 Gathering production index status...")
    
    status = get_production_index_status()
    
    if "error" in status:
        print(f"❌ Error: {status['error']}")
        return
    
    print_status_report(status)
    
    # Save report
    with open("production_index_status_report.json", "w") as f:
        json.dump(status, f, indent=2)
    
    print("✅ Status report saved to production_index_status_report.json")

if __name__ == "__main__":
    main()