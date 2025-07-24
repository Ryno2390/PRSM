#!/usr/bin/env python3
"""
Real-time Resource Monitor for PDF Download Process
==================================================

Monitors the PDF download process and system resources in real-time,
providing insights and suggestions for optimization.
"""

import time
import psutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

class PDFProcessMonitor:
    """Real-time monitor for PDF download process"""
    
    def __init__(self):
        self.log_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/archive/databases/pdf_download_log.txt")
        self.monitor_log = Path("/Users/ryneschultz/Documents/GitHub/PRSM/pdf_monitor.log")
        self.last_processed_count = 0
        self.start_time = time.time()
        
    def find_pdf_process(self) -> Optional[Dict[str, Any]]:
        """Find the current PDF download process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
            try:
                if proc.info['cmdline'] and any('download_full_pdfs.py' in cmd for cmd in proc.info['cmdline']):
                    return {
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'create_time': proc.info['create_time'],
                        'runtime_hours': (time.time() - proc.info['create_time']) / 3600
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network stats
        net_io = psutil.net_io_counters()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'overall_percent': sum(cpu_percent) / len(cpu_percent),
                'per_core': cpu_percent,
                'core_count': len(cpu_percent),
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            },
            'memory': {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent_used': memory.percent,
                'cached_gb': getattr(memory, 'cached', 0) / (1024**3)
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent_used': (disk.used / disk.total) * 100
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        }
    
    def analyze_progress(self) -> Dict[str, Any]:
        """Analyze current download progress"""
        if not self.log_file.exists():
            return {'error': 'Log file not found'}
        
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
            
            # Count processed papers
            processed_count = content.count("Successfully processed")
            failed_count = content.count("Failed to download PDF")
            
            # Calculate rate since last check
            rate_since_last = 0
            if hasattr(self, 'last_check_time'):
                time_diff = time.time() - self.last_check_time
                count_diff = processed_count - self.last_processed_count
                if time_diff > 0:
                    rate_since_last = count_diff / time_diff
            
            self.last_processed_count = processed_count
            self.last_check_time = time.time()
            
            # Get recent activity
            lines = content.split('\\n')
            recent_lines = lines[-50:] if len(lines) > 50 else lines
            recent_activity = [line for line in recent_lines if line.strip()]
            
            return {
                'total_processed': processed_count,
                'total_failed': failed_count,
                'success_rate': (processed_count / (processed_count + failed_count)) * 100 if (processed_count + failed_count) > 0 else 0,
                'rate_since_last_check': rate_since_last,
                'rate_per_hour': rate_since_last * 3600,
                'recent_activity': recent_activity[-5:],  # Last 5 lines
                'log_size_mb': len(content) / (1024 * 1024)
            }
            
        except Exception as e:
            return {'error': f'Progress analysis failed: {e}'}
    
    def calculate_completion_estimate(self, progress: Dict[str, Any], target_papers: int = 149726) -> Dict[str, Any]:
        """Calculate completion estimates"""
        
        if 'error' in progress or progress['rate_per_hour'] <= 0:
            return {'error': 'Cannot estimate completion'}
        
        processed = progress['total_processed']
        remaining = target_papers - processed
        rate_per_hour = progress['rate_per_hour']
        
        hours_remaining = remaining / rate_per_hour if rate_per_hour > 0 else float('inf')
        completion_time = datetime.now() + timedelta(hours=hours_remaining)
        
        return {
            'papers_remaining': remaining,
            'progress_percent': (processed / target_papers) * 100,
            'hours_remaining': hours_remaining,
            'days_remaining': hours_remaining / 24,
            'estimated_completion': completion_time.strftime('%Y-%m-%d %H:%M:%S'),
            'current_rate_per_hour': rate_per_hour
        }
    
    def generate_optimization_suggestions(self, system_metrics: Dict[str, Any], pdf_process: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on current state"""
        
        suggestions = []
        
        cpu_usage = system_metrics['cpu']['overall_percent']
        memory_usage = system_metrics['memory']['percent_used']
        available_memory = system_metrics['memory']['available_gb']
        
        # CPU optimization suggestions
        if cpu_usage < 30:
            suggestions.append(f"💡 LOW CPU USAGE ({cpu_usage:.1f}%) - Could increase concurrent downloads")
        elif cpu_usage > 80:
            suggestions.append(f"⚠️ HIGH CPU USAGE ({cpu_usage:.1f}%) - Consider reducing concurrent operations")
        
        # Memory optimization suggestions
        if memory_usage < 60:
            suggestions.append(f"💡 MEMORY AVAILABLE ({available_memory:.1f}GB free) - Could increase batch sizes")
        elif memory_usage > 85:
            suggestions.append(f"⚠️ HIGH MEMORY USAGE ({memory_usage:.1f}%) - Consider reducing batch sizes")
        
        # Process-specific suggestions
        if pdf_process:
            process_cpu = pdf_process['cpu_percent']
            process_memory = pdf_process['memory_percent']
            
            if process_cpu < 5:
                suggestions.append("💡 PDF process using minimal CPU - likely I/O bound (network/disk)")
            
            if process_memory < 5:
                suggestions.append("💡 PDF process using minimal memory - could handle larger batches")
        
        # System capacity suggestions
        core_count = system_metrics['cpu']['core_count']
        if cpu_usage < 50 and core_count > 4:
            suggestions.append(f"💡 UNDERUTILIZED CORES ({core_count} available) - Could run parallel processes")
        
        return suggestions
    
    def save_monitoring_data(self, data: Dict[str, Any]):
        """Save monitoring data to log file"""
        try:
            with open(self.monitor_log, 'a') as f:
                f.write(json.dumps(data) + '\\n')
        except Exception as e:
            print(f"Warning: Could not save monitoring data: {e}")
    
    def run_continuous_monitor(self, duration_minutes: int = 30, check_interval: int = 10):
        """Run continuous monitoring for specified duration"""
        
        print("🔍 PDF DOWNLOAD PROCESS MONITOR")
        print("=" * 60)
        print(f"Monitoring for {duration_minutes} minutes with {check_interval}s intervals")
        print("Press Ctrl+C to stop monitoring")
        print()
        
        end_time = time.time() + (duration_minutes * 60)
        check_count = 0
        
        try:
            while time.time() < end_time:
                check_count += 1
                
                print(f"\\n📊 CHECK #{check_count} - {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 40)
                
                # Find PDF process
                pdf_process = self.find_pdf_process()
                if pdf_process:
                    print(f"✅ PDF Process: PID {pdf_process['pid']} | "
                          f"Runtime: {pdf_process['runtime_hours']:.1f}h | "
                          f"CPU: {pdf_process['cpu_percent']:.1f}% | "
                          f"Memory: {pdf_process['memory_percent']:.1f}%")
                else:
                    print("❌ PDF process not found")
                
                # Get system metrics
                system_metrics = self.get_system_metrics()
                print(f"💻 System: CPU {system_metrics['cpu']['overall_percent']:.1f}% | "
                      f"Memory {system_metrics['memory']['percent_used']:.1f}% | "
                      f"Available {system_metrics['memory']['available_gb']:.1f}GB")
                
                # Analyze progress
                progress = self.analyze_progress()
                if 'error' not in progress:
                    print(f"📈 Progress: {progress['total_processed']:,} processed | "
                          f"Rate: {progress['rate_per_hour']:.0f}/hour | "
                          f"Success: {progress['success_rate']:.1f}%")
                    
                    # Completion estimate
                    completion = self.calculate_completion_estimate(progress)
                    if 'error' not in completion:
                        print(f"⏱️ Estimate: {completion['progress_percent']:.1f}% complete | "
                              f"{completion['days_remaining']:.1f} days remaining | "
                              f"Done: {completion['estimated_completion']}")
                
                # Generate suggestions
                if pdf_process:
                    suggestions = self.generate_optimization_suggestions(system_metrics, pdf_process)
                    if suggestions:
                        print("💡 Suggestions:")
                        for suggestion in suggestions[:3]:  # Show top 3
                            print(f"   {suggestion}")
                
                # Save monitoring data
                monitoring_data = {
                    'check_number': check_count,
                    'system_metrics': system_metrics,
                    'pdf_process': pdf_process,
                    'progress': progress,
                    'completion_estimate': self.calculate_completion_estimate(progress) if 'error' not in progress else None
                }
                self.save_monitoring_data(monitoring_data)
                
                # Wait for next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\\n🛑 Monitoring stopped by user")
        
        print(f"\\n✅ Monitoring completed after {check_count} checks")
        print(f"📝 Detailed data saved to: {self.monitor_log}")

def main():
    """Main monitoring function"""
    
    monitor = PDFProcessMonitor()
    
    print("Choose monitoring option:")
    print("1. Single check (quick status)")
    print("2. Continuous monitoring (30 minutes)")
    print("3. Extended monitoring (2 hours)")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            # Single check
            pdf_process = monitor.find_pdf_process()
            system_metrics = monitor.get_system_metrics()
            progress = monitor.analyze_progress()
            
            print("\\n📊 CURRENT STATUS:")
            print("=" * 40)
            
            if pdf_process:
                print(f"PDF Process: Running (PID {pdf_process['pid']})")
                print(f"Runtime: {pdf_process['runtime_hours']:.1f} hours")
            else:
                print("PDF Process: Not found")
            
            print(f"CPU Usage: {system_metrics['cpu']['overall_percent']:.1f}%")
            print(f"Memory Usage: {system_metrics['memory']['percent_used']:.1f}%")
            
            if 'error' not in progress:
                print(f"Papers Processed: {progress['total_processed']:,}")
                print(f"Processing Rate: {progress['rate_per_hour']:.0f} papers/hour")
                
                completion = monitor.calculate_completion_estimate(progress)
                if 'error' not in completion:
                    print(f"Progress: {completion['progress_percent']:.1f}%")
                    print(f"Estimated Completion: {completion['estimated_completion']}")
        
        elif choice == '2':
            monitor.run_continuous_monitor(duration_minutes=30, check_interval=10)
        
        elif choice == '3':
            monitor.run_continuous_monitor(duration_minutes=120, check_interval=30)
        
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\\n🛑 Monitoring cancelled")

if __name__ == "__main__":
    main()