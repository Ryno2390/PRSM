#!/usr/bin/env python3
"""
Embedding Pipeline Performance Optimizer
========================================

This script analyzes and optimizes the embedding pipeline performance
by implementing various optimization strategies.
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

class EmbeddingPipelineOptimizer:
    """Optimize embedding pipeline performance"""
    
    def __init__(self):
        self.embeddings_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Embeddings")
        self.papers_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Content/hot")
        
    def analyze_system_resources(self) -> Dict:
        """Analyze current system resource usage"""
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
    
    def get_processing_stats(self) -> Dict:
        """Get current processing statistics"""
        batch_files = list(self.embeddings_dir.glob("embeddings_batch_*.pkl"))
        completed_batches = len(batch_files)
        
        # Get file sizes to analyze batch efficiency
        batch_sizes = []
        for batch_file in batch_files[-10:]:  # Last 10 batches
            try:
                batch_sizes.append(batch_file.stat().st_size)
            except:
                continue
        
        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
        
        return {
            "completed_batches": completed_batches,
            "papers_processed": completed_batches * 32,
            "avg_batch_size_mb": avg_batch_size / (1024**2),
            "estimated_total_size_gb": (avg_batch_size * 4724) / (1024**3)  # 4724 total batches
        }
    
    def detect_bottlenecks(self) -> List[str]:
        """Detect performance bottlenecks"""
        bottlenecks = []
        
        resources = self.analyze_system_resources()
        
        # CPU bottleneck
        if resources["cpu_percent"] > 90:
            bottlenecks.append("HIGH_CPU_USAGE")
        
        # Memory bottleneck
        if resources["memory_percent"] > 85:
            bottlenecks.append("HIGH_MEMORY_USAGE")
        
        # Disk space bottleneck
        if resources["disk_percent"] > 90:
            bottlenecks.append("LOW_DISK_SPACE")
        
        # Check for multiple embedding processes
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'build_paper_embeddings.py' in cmdline:
                        python_processes.append(proc.info['pid'])
            except:
                continue
        
        if len(python_processes) > 1:
            bottlenecks.append("MULTIPLE_EMBEDDING_PROCESSES")
        
        return bottlenecks
    
    def optimize_batch_processing(self) -> Dict:
        """Optimize batch processing parameters"""
        resources = self.analyze_system_resources()
        
        # Recommend optimal batch size based on available memory
        available_memory_gb = resources["memory_available_gb"]
        
        if available_memory_gb > 8:
            recommended_batch_size = 64
        elif available_memory_gb > 4:
            recommended_batch_size = 32
        else:
            recommended_batch_size = 16
        
        # Recommend number of parallel processes
        cpu_cores = psutil.cpu_count()
        if resources["cpu_percent"] < 70:
            recommended_processes = min(2, cpu_cores // 2)
        else:
            recommended_processes = 1
        
        return {
            "recommended_batch_size": recommended_batch_size,
            "recommended_processes": recommended_processes,
            "current_batch_size": 32,  # Current default
            "optimization_potential": "HIGH" if available_memory_gb > 8 else "MEDIUM"
        }
    
    def generate_optimization_report(self) -> Dict:
        """Generate comprehensive optimization report"""
        logger.info("🔍 Analyzing embedding pipeline performance...")
        
        resources = self.analyze_system_resources()
        stats = self.get_processing_stats()
        bottlenecks = self.detect_bottlenecks()
        optimizations = self.optimize_batch_processing()
        
        report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "system_resources": resources,
            "processing_stats": stats,
            "bottlenecks": bottlenecks,
            "optimizations": optimizations,
            "recommendations": []
        }
        
        # Generate recommendations
        if "HIGH_CPU_USAGE" in bottlenecks:
            report["recommendations"].append("Reduce batch size or number of parallel processes")
        
        if "HIGH_MEMORY_USAGE" in bottlenecks:
            report["recommendations"].append("Implement memory-efficient batching")
        
        if "LOW_DISK_SPACE" in bottlenecks:
            report["recommendations"].append("Clean up temporary files or expand storage")
        
        if "MULTIPLE_EMBEDDING_PROCESSES" in bottlenecks:
            report["recommendations"].append("Consolidate to single optimized process")
        
        if not bottlenecks:
            report["recommendations"].append("Consider increasing batch size for faster processing")
        
        return report
    
    def save_optimization_report(self, report: Dict):
        """Save optimization report to file"""
        report_file = Path("embedding_optimization_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Optimization report saved to {report_file}")
    
    def monitor_and_optimize(self, duration_minutes: int = 10):
        """Monitor pipeline and suggest optimizations"""
        logger.info(f"🚀 Starting embedding pipeline optimization monitor")
        logger.info(f"⏱️  Monitoring for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        reports = []
        
        while time.time() < end_time:
            try:
                report = self.generate_optimization_report()
                reports.append(report)
                
                # Log key metrics
                logger.info(f"📊 System: CPU {report['system_resources']['cpu_percent']:.1f}%, "
                           f"Memory {report['system_resources']['memory_percent']:.1f}%")
                logger.info(f"📈 Progress: {report['processing_stats']['papers_processed']:,} papers processed")
                
                if report['bottlenecks']:
                    logger.warning(f"⚠️  Bottlenecks detected: {', '.join(report['bottlenecks'])}")
                
                for rec in report['recommendations']:
                    logger.info(f"💡 Recommendation: {rec}")
                
                # Wait before next check
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error during monitoring: {e}")
                time.sleep(30)
        
        # Save final report
        if reports:
            final_report = reports[-1]
            final_report["monitoring_duration_minutes"] = duration_minutes
            final_report["total_reports"] = len(reports)
            self.save_optimization_report(final_report)
        
        logger.info("✅ Optimization monitoring completed")

def main():
    """Main function"""
    optimizer = EmbeddingPipelineOptimizer()
    
    # Get monitoring duration from command line
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    # Generate initial report
    initial_report = optimizer.generate_optimization_report()
    
    logger.info("📊 Initial Performance Analysis:")
    logger.info(f"  CPU Usage: {initial_report['system_resources']['cpu_percent']:.1f}%")
    logger.info(f"  Memory Usage: {initial_report['system_resources']['memory_percent']:.1f}%")
    logger.info(f"  Papers Processed: {initial_report['processing_stats']['papers_processed']:,}")
    logger.info(f"  Bottlenecks: {initial_report['bottlenecks']}")
    
    # Start monitoring
    optimizer.monitor_and_optimize(duration)

if __name__ == "__main__":
    main()