#!/usr/bin/env python3
"""
Monitor FAISS Index Building Progress
====================================

This script monitors the progress of FAISS index building for the full corpus.
"""

import time
import json
import psutil
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexBuildingMonitor:
    """Monitor FAISS index building progress"""
    
    def __init__(self):
        self.indices_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Indices")
        self.embeddings_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Embeddings")
        
    def get_embedding_stats(self) -> Dict:
        """Get statistics about available embeddings"""
        batch_files = list(self.embeddings_dir.glob("embeddings_batch_*.pkl"))
        total_batches = len(batch_files)
        total_papers = total_batches * 32  # 32 papers per batch
        
        return {
            "total_batches": total_batches,
            "total_papers": total_papers,
            "total_embeddings": total_papers
        }
    
    def get_index_building_processes(self) -> List[Dict]:
        """Get information about running index building processes"""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'build_faiss_index.py' in cmdline:
                        # Extract index type from command line
                        index_type = "Unknown"
                        if "--index-type IVF" in cmdline:
                            index_type = "IVF"
                        elif "--index-type HNSW" in cmdline:
                            index_type = "HNSW"
                        elif "--index-type Flat" in cmdline:
                            index_type = "Flat"
                        
                        processes.append({
                            "pid": proc.info['pid'],
                            "index_type": index_type,
                            "cpu_percent": proc.info['cpu_percent'],
                            "memory_percent": proc.info['memory_percent'],
                            "cmdline": cmdline
                        })
            except:
                continue
        
        return processes
    
    def get_completed_indices(self) -> List[Dict]:
        """Get information about completed indices"""
        indices = []
        
        # Check for index files
        for index_file in self.indices_dir.glob("*.index"):
            index_name = index_file.stem
            
            # Look for corresponding metadata
            metadata_file = None
            for pattern in ["index_metadata_*.json", f"*{index_name.split('_')[-1]}*.json"]:
                matches = list(self.indices_dir.glob(pattern))
                if matches:
                    metadata_file = matches[0]
                    break
            
            index_info = {
                "name": index_name,
                "file_size_mb": index_file.stat().st_size / (1024**2),
                "created_at": time.ctime(index_file.stat().st_mtime),
                "type": "Unknown",
                "total_embeddings": 0
            }
            
            if metadata_file:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    index_info.update({
                        "type": metadata.get("index_type", "Unknown"),
                        "total_embeddings": metadata.get("total_embeddings", 0),
                        "embedding_dimension": metadata.get("embedding_dimension", 0),
                        "build_time_seconds": metadata.get("build_time_seconds", 0)
                    })
                except:
                    pass
            
            indices.append(index_info)
        
        return indices
    
    def get_system_resources(self) -> Dict:
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/Volumes/My Passport")
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "disk_percent": (disk.used / disk.total) * 100
        }
    
    def generate_status_report(self) -> Dict:
        """Generate comprehensive status report"""
        embedding_stats = self.get_embedding_stats()
        building_processes = self.get_index_building_processes()
        completed_indices = self.get_completed_indices()
        system_resources = self.get_system_resources()
        
        return {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "embedding_stats": embedding_stats,
            "building_processes": building_processes,
            "completed_indices": completed_indices,
            "system_resources": system_resources
        }
    
    def monitor_progress(self, duration_minutes: int = 30):
        """Monitor index building progress"""
        logger.info("🏗️  Starting FAISS Index Building Monitor")
        logger.info(f"⏱️  Monitoring for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                report = self.generate_status_report()
                
                # Log embedding statistics
                embedding_stats = report["embedding_stats"]
                logger.info(f"📊 Embeddings: {embedding_stats['total_papers']:,} papers, {embedding_stats['total_batches']:,} batches")
                
                # Log building processes
                building_processes = report["building_processes"]
                if building_processes:
                    logger.info(f"🔧 Building processes: {len(building_processes)}")
                    for proc in building_processes:
                        logger.info(f"  {proc['index_type']} (PID {proc['pid']}): CPU {proc['cpu_percent']:.1f}%, Memory {proc['memory_percent']:.1f}%")
                else:
                    logger.info("🔧 No building processes currently running")
                
                # Log completed indices
                completed_indices = report["completed_indices"]
                if completed_indices:
                    logger.info(f"✅ Completed indices: {len(completed_indices)}")
                    for idx in completed_indices:
                        logger.info(f"  {idx['name']} ({idx['type']}): {idx['total_embeddings']:,} embeddings, {idx['file_size_mb']:.1f}MB")
                else:
                    logger.info("✅ No completed indices yet")
                
                # Log system resources
                resources = report["system_resources"]
                logger.info(f"💻 System: CPU {resources['cpu_percent']:.1f}%, Memory {resources['memory_percent']:.1f}%, Disk {resources['disk_percent']:.1f}%")
                
                logger.info("-" * 60)
                
                # Wait before next check
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error during monitoring: {e}")
                time.sleep(30)
        
        # Final report
        final_report = self.generate_status_report()
        logger.info("📊 Final Status Report:")
        logger.info(f"  Building processes: {len(final_report['building_processes'])}")
        logger.info(f"  Completed indices: {len(final_report['completed_indices'])}")
        
        # Save final report
        with open("index_building_report.json", "w") as f:
            json.dump(final_report, f, indent=2)
        
        logger.info("✅ Index building monitoring completed")

def main():
    """Main function"""
    monitor = IndexBuildingMonitor()
    
    # Get monitoring duration from command line
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    # Generate initial report
    initial_report = monitor.generate_status_report()
    
    logger.info("📊 Initial Status:")
    logger.info(f"  Total papers: {initial_report['embedding_stats']['total_papers']:,}")
    logger.info(f"  Building processes: {len(initial_report['building_processes'])}")
    logger.info(f"  Completed indices: {len(initial_report['completed_indices'])}")
    
    # Start monitoring
    monitor.monitor_progress(duration)

if __name__ == "__main__":
    main()