#!/usr/bin/env python3
"""
Monitor Embedding Pipeline Progress
==================================

This script monitors the progress of the background embedding pipeline
and provides real-time statistics on processing speed, completion rate,
and estimated time remaining.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

def get_embedding_progress() -> Dict:
    """Get current embedding pipeline progress"""
    embeddings_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Embeddings")
    
    # Count completed batches
    batch_files = list(embeddings_dir.glob("embeddings_batch_*.pkl"))
    completed_batches = len(batch_files)
    
    # Estimate total papers processed (32 papers per batch)
    papers_per_batch = 32
    papers_processed = completed_batches * papers_per_batch
    
    # Count total papers to process
    papers_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Content/hot")
    if papers_dir.exists():
        total_papers = len(list(papers_dir.rglob("*.dat")))
    else:
        total_papers = 151120  # Known count
    
    # Calculate progress
    progress_percentage = (papers_processed / total_papers) * 100
    
    return {
        "completed_batches": completed_batches,
        "papers_processed": papers_processed,
        "total_papers": total_papers,
        "progress_percentage": progress_percentage,
        "remaining_papers": total_papers - papers_processed,
        "estimated_remaining_batches": (total_papers - papers_processed) // papers_per_batch
    }

def estimate_completion_time(progress_history: List[Dict]) -> Optional[str]:
    """Estimate completion time based on progress history"""
    if len(progress_history) < 2:
        return None
    
    # Calculate processing rate from last two measurements
    recent = progress_history[-1]
    previous = progress_history[-2]
    
    time_diff = recent["timestamp"] - previous["timestamp"]
    papers_diff = recent["papers_processed"] - previous["papers_processed"]
    
    if time_diff <= 0 or papers_diff <= 0:
        return None
    
    # Papers per second
    processing_rate = papers_diff / time_diff
    
    # Time to complete remaining papers
    remaining_seconds = recent["remaining_papers"] / processing_rate
    
    # Convert to human readable format
    hours = int(remaining_seconds // 3600)
    minutes = int((remaining_seconds % 3600) // 60)
    seconds = int(remaining_seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def monitor_progress(duration_minutes: int = 5):
    """Monitor embedding progress for specified duration"""
    
    print("📊 Embedding Pipeline Progress Monitor")
    print("=" * 50)
    print(f"⏱️  Monitoring for {duration_minutes} minutes")
    print()
    
    progress_history = []
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    while time.time() < end_time:
        try:
            current_time = time.time()
            progress = get_embedding_progress()
            progress["timestamp"] = current_time
            progress_history.append(progress)
            
            # Print current status
            print(f"⏰ {time.strftime('%H:%M:%S')}")
            print(f"📦 Batches: {progress['completed_batches']:,}")
            print(f"📄 Papers: {progress['papers_processed']:,} / {progress['total_papers']:,}")
            print(f"📊 Progress: {progress['progress_percentage']:.1f}%")
            print(f"⏳ Remaining: {progress['remaining_papers']:,} papers")
            
            # Estimate completion time
            eta = estimate_completion_time(progress_history)
            if eta:
                print(f"🎯 ETA: {eta}")
            
            # Calculate current processing rate
            if len(progress_history) >= 2:
                recent = progress_history[-1]
                previous = progress_history[-2]
                time_diff = recent["timestamp"] - previous["timestamp"]
                papers_diff = recent["papers_processed"] - previous["papers_processed"]
                
                if time_diff > 0:
                    rate = papers_diff / time_diff
                    print(f"🚀 Rate: {rate:.1f} papers/sec")
            
            print("-" * 30)
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)
    
    print("\n📊 Final Progress Summary:")
    if progress_history:
        final_progress = progress_history[-1]
        print(f"✅ Completed: {final_progress['papers_processed']:,} papers")
        print(f"📈 Progress: {final_progress['progress_percentage']:.1f}%")
        print(f"⏳ Remaining: {final_progress['remaining_papers']:,} papers")

def get_faiss_index_status():
    """Check status of FAISS indices"""
    indices_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Indices")
    
    if not indices_dir.exists():
        return {"status": "not_found", "indices": []}
    
    indices = []
    for index_file in indices_dir.glob("*.index"):
        metadata_file = indices_dir / f"index_metadata_{index_file.stem.split('_')[-1]}.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                indices.append({
                    "name": index_file.name,
                    "type": metadata.get("index_type", "unknown"),
                    "total_embeddings": metadata.get("total_embeddings", 0),
                    "created_at": metadata.get("created_at", "unknown")
                })
            except:
                indices.append({
                    "name": index_file.name,
                    "type": "unknown",
                    "total_embeddings": 0,
                    "created_at": "unknown"
                })
    
    return {"status": "found", "indices": indices}

def main():
    """Main function"""
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            duration = 5
    else:
        duration = 5
    
    # Show current status
    progress = get_embedding_progress()
    print(f"📊 Current Status:")
    print(f"  Papers processed: {progress['papers_processed']:,}")
    print(f"  Progress: {progress['progress_percentage']:.1f}%")
    print(f"  Remaining: {progress['remaining_papers']:,}")
    print()
    
    # Show FAISS index status
    index_status = get_faiss_index_status()
    print(f"🔍 FAISS Index Status:")
    if index_status["status"] == "found":
        for idx in index_status["indices"]:
            print(f"  {idx['name']}: {idx['total_embeddings']:,} embeddings ({idx['type']})")
    else:
        print("  No indices found")
    print()
    
    # Start monitoring
    monitor_progress(duration)

if __name__ == "__main__":
    main()