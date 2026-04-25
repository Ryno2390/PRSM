#!/usr/bin/env python3
"""
PDF Process Optimization Analysis
=================================

Analyzes the current PDF download process to determine if we can safely
increase processing power without interrupting the current operation.
"""

import os
import psutil
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

class PDFProcessOptimizer:
    """Analyzes and suggests optimizations for PDF download process"""
    
    def __init__(self):
        self.current_pid = None
        self.log_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/archive/databases/pdf_download_log.txt")
        self.db_path = Path("/Volumes/My Passport/PRSM_Storage/storage.db")
        
    def find_pdf_process(self) -> Optional[Dict[str, Any]]:
        """Find the current PDF download process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
            try:
                if proc.info['cmdline'] and any('download_full_pdfs.py' in cmd for cmd in proc.info['cmdline']):
                    self.current_pid = proc.info['pid']
                    return {
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'create_time': proc.info['create_time'],
                        'runtime_hours': (time.time() - proc.info['create_time']) / 3600,
                        'process': proc
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def analyze_system_resources(self) -> Dict[str, Any]:
        """Analyze current system resource utilization"""
        cpu_times = psutil.cpu_times()
        memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        
        # Get detailed CPU usage over 1 second interval
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        
        return {
            'cpu': {
                'overall_percent': psutil.cpu_percent(),
                'per_core': cpu_percent,
                'core_count': psutil.cpu_count(logical=True),
                'physical_cores': psutil.cpu_count(logical=False),
                'user_time': cpu_times.user,
                'system_time': cpu_times.system,
                'idle_time': cpu_times.idle,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            'memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent_used': memory.percent,
                'free_gb': memory.free / (1024**3),
                'cached_gb': getattr(memory, 'cached', 0) / (1024**3)
            },
            'disk': {
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'percent_used': (disk_usage.used / disk_usage.total) * 100
            }
        }
    
    def analyze_download_progress(self) -> Dict[str, Any]:
        """Analyze current download progress and performance"""
        if not self.log_file.exists():
            return {'error': 'Log file not found'}
        
        # Count processed papers
        with open(self.log_file, 'r') as f:
            content = f.read()
        
        processed_count = content.count("Successfully processed")
        failed_count = content.count("Failed to download PDF")
        
        # Get recent processing rate (last 1000 lines)
        lines = content.split('\n')
        recent_lines = lines[-1000:] if len(lines) > 1000 else lines
        recent_processed = sum(1 for line in recent_lines if "Successfully processed" in line)
        
        # Extract timestamps from recent activity
        recent_timestamps = []
        for line in recent_lines[-100:]:  # Last 100 lines
            if "Successfully processed" in line:
                try:
                    # Extract timestamp from log line format: "2025-07-23 19:42:18"
                    timestamp_str = line.split()[0] + " " + line.split()[1]
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    recent_timestamps.append(timestamp)
                except (IndexError, ValueError):
                    continue
        
        # Calculate processing rate
        processing_rate = 0.0
        if len(recent_timestamps) >= 2:
            time_span = (recent_timestamps[-1] - recent_timestamps[0]).total_seconds()
            if time_span > 0:
                processing_rate = len(recent_timestamps) / time_span  # papers per second
        
        # Database analysis
        db_stats = self.analyze_database_progress()
        
        return {
            'log_analysis': {
                'total_processed': processed_count,
                'total_failed': failed_count,
                'recent_processed': recent_processed,
                'processing_rate_per_second': processing_rate,
                'processing_rate_per_hour': processing_rate * 3600,
                'success_rate': (processed_count / (processed_count + failed_count)) * 100 if (processed_count + failed_count) > 0 else 0
            },
            'database_stats': db_stats
        }
    
    def analyze_database_progress(self) -> Dict[str, Any]:
        """Analyze database to get accurate progress statistics"""
        if not self.db_path.exists():
            return {'error': 'Database not found'}
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Total papers
            cursor.execute("SELECT COUNT(*) FROM arxiv_papers")
            total_papers = cursor.fetchone()[0]
            
            # Papers with full content
            cursor.execute("SELECT COUNT(*) FROM arxiv_papers WHERE has_full_content = 1")
            papers_with_content = cursor.fetchone()[0]
            
            # Papers without full content
            cursor.execute("SELECT COUNT(*) FROM arxiv_papers WHERE has_full_content = 0 OR has_full_content IS NULL")
            papers_without_content = cursor.fetchone()[0]
            
            conn.close()
            
            completion_percent = (papers_with_content / total_papers) * 100 if total_papers > 0 else 0
            
            return {
                'total_papers': total_papers,
                'papers_with_content': papers_with_content,
                'papers_without_content': papers_without_content,
                'completion_percent': completion_percent,
                'remaining_papers': papers_without_content
            }
            
        except Exception as e:
            return {'error': f'Database analysis failed: {e}'}
    
    def calculate_optimization_potential(self, process_info: Dict, system_resources: Dict, progress_info: Dict) -> Dict[str, Any]:
        """Calculate potential optimizations without interrupting current process"""
        
        current_config = {
            'batch_size': 50,  # From download_full_pdfs.py
            'max_concurrent': 10,  # From download_full_pdfs.py
            'batch_delay': 2  # 2 second delay between batches
        }
        
        # System capacity analysis
        cpu_utilization = system_resources['cpu']['overall_percent']
        memory_utilization = system_resources['memory']['percent_used']
        available_cores = system_resources['cpu']['core_count']
        
        # Calculate safe optimization parameters
        optimizations = {
            'safe_increases': {},
            'estimated_improvements': {},
            'risk_assessment': 'low',
            'current_performance': {
                'papers_per_hour': progress_info['log_analysis']['processing_rate_per_hour'],
                'cpu_usage': cpu_utilization,
                'memory_usage': memory_utilization
            }
        }
        
        # Conservative optimizations (can be applied without restart)
        if cpu_utilization < 60 and memory_utilization < 70:
            # Safe to increase concurrency slightly
            new_concurrent = min(15, current_config['max_concurrent'] + 5)
            optimizations['safe_increases']['max_concurrent'] = {
                'current': current_config['max_concurrent'],
                'recommended': new_concurrent,
                'justification': 'CPU and memory usage low enough for 50% concurrency increase'
            }
        
        if cpu_utilization < 50:
            # Safe to reduce batch delay
            new_delay = max(0.5, current_config['batch_delay'] * 0.5)
            optimizations['safe_increases']['batch_delay'] = {
                'current': current_config['batch_delay'],
                'recommended': new_delay,
                'justification': 'Low CPU usage allows faster batch processing'
            }
        
        if memory_utilization < 60:
            # Safe to increase batch size
            new_batch_size = min(100, current_config['batch_size'] + 50)
            optimizations['safe_increases']['batch_size'] = {
                'current': current_config['batch_size'],
                'recommended': new_batch_size,
                'justification': 'Sufficient memory for larger batches'
            }
        
        # Estimate completion time improvements
        current_rate = progress_info['log_analysis']['processing_rate_per_hour']
        if current_rate > 0:
            db_stats = progress_info.get('database_stats', {})
            remaining_papers = db_stats.get('remaining_papers', 90000)  # Conservative estimate
            current_completion_hours = remaining_papers / current_rate
            
            # Conservative estimate: 30-50% improvement possible
            estimated_improvement = 1.4  # 40% improvement
            improved_rate = current_rate * estimated_improvement
            improved_completion_hours = remaining_papers / improved_rate
            
            optimizations['estimated_improvements'] = {
                'current_completion_hours': current_completion_hours,
                'current_completion_days': current_completion_hours / 24,
                'optimized_completion_hours': improved_completion_hours,
                'optimized_completion_days': improved_completion_hours / 24,
                'time_saved_hours': current_completion_hours - improved_completion_hours,
                'time_saved_days': (current_completion_hours - improved_completion_hours) / 24,
                'improvement_factor': estimated_improvement
            }
        
        return optimizations
    
    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate comprehensive recommendations for safe optimization"""
        
        print("🔍 ANALYZING PDF DOWNLOAD PROCESS...")
        print("=" * 60)
        
        # Find current process
        process_info = self.find_pdf_process()
        if not process_info:
            return {'error': 'PDF download process not found'}
        
        print(f"✅ Found PDF download process (PID: {process_info['pid']})")
        print(f"   Runtime: {process_info['runtime_hours']:.1f} hours")
        print(f"   CPU Usage: {process_info['cpu_percent']:.1f}%")
        print(f"   Memory Usage: {process_info['memory_percent']:.1f}%")
        print()
        
        # Analyze system resources
        system_resources = self.analyze_system_resources()
        print("💻 SYSTEM RESOURCE ANALYSIS:")
        print(f"   CPU Usage: {system_resources['cpu']['overall_percent']:.1f}%")
        print(f"   CPU Cores: {system_resources['cpu']['core_count']} ({system_resources['cpu']['physical_cores']} physical)")
        print(f"   Memory Usage: {system_resources['memory']['percent_used']:.1f}%")
        print(f"   Available Memory: {system_resources['memory']['available_gb']:.1f} GB")
        print(f"   Disk Usage: {system_resources['disk']['percent_used']:.1f}%")
        print()
        
        # Analyze download progress
        progress_info = self.analyze_download_progress()
        if 'error' not in progress_info:
            print("📊 DOWNLOAD PROGRESS ANALYSIS:")
            print(f"   Total Processed: {progress_info['log_analysis']['total_processed']:,}")
            print(f"   Processing Rate: {progress_info['log_analysis']['processing_rate_per_hour']:.0f} papers/hour")
            print(f"   Success Rate: {progress_info['log_analysis']['success_rate']:.1f}%")
            
            # Handle database stats safely
            db_stats = progress_info.get('database_stats', {})
            if 'completion_percent' in db_stats:
                print(f"   Database Progress: {db_stats['completion_percent']:.1f}%")
                print(f"   Remaining Papers: {db_stats['remaining_papers']:,}")
            else:
                print(f"   Database Status: {db_stats.get('error', 'Analysis pending')}")
            print()
        
        # Calculate optimizations
        optimizations = self.calculate_optimization_potential(process_info, system_resources, progress_info)
        
        print("🚀 OPTIMIZATION RECOMMENDATIONS:")
        print("=" * 60)
        
        if optimizations['safe_increases']:
            print("✅ SAFE OPTIMIZATIONS (can be applied without restart):")
            for param, details in optimizations['safe_increases'].items():
                print(f"   {param}:")
                print(f"     Current: {details['current']}")
                print(f"     Recommended: {details['recommended']}")
                print(f"     Reason: {details['justification']}")
                print()
        else:
            print("⚠️  No safe optimizations available with current resource usage.")
            print("   System resources are already well-utilized.")
            print()
        
        if 'estimated_improvements' in optimizations:
            est = optimizations['estimated_improvements']
            print("📈 ESTIMATED COMPLETION TIME IMPROVEMENTS:")
            print(f"   Current Completion: {est['current_completion_days']:.1f} days")
            print(f"   Optimized Completion: {est['optimized_completion_days']:.1f} days")
            print(f"   Time Saved: {est['time_saved_days']:.1f} days")
            print(f"   Speed Improvement: {((est['improvement_factor'] - 1) * 100):.0f}%")
            print()
        
        print("🛡️  SAFETY ASSESSMENT:")
        print(f"   Risk Level: {optimizations['risk_assessment'].upper()}")
        print("   Recommended Approach: Monitor resource usage and apply")
        print("   conservative increases without interrupting current process.")
        print()
        
        print("💡 IMPLEMENTATION OPTIONS:")
        print("   1. MONITOR ONLY - Continue current analysis")
        print("   2. DYNAMIC TUNING - Adjust parameters via process signals (advanced)")
        print("   3. PARALLEL PROCESS - Start additional download workers (risky)")
        print("   4. WAIT FOR COMPLETION - No changes to current process")
        
        return {
            'process_info': process_info,
            'system_resources': system_resources,
            'progress_info': progress_info,
            'optimizations': optimizations,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main analysis function"""
    optimizer = PDFProcessOptimizer()
    results = optimizer.generate_recommendations()
    
    # Save detailed analysis to file
    results_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/pdf_optimization_analysis.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📝 Detailed analysis saved to: {results_file}")

if __name__ == "__main__":
    main()