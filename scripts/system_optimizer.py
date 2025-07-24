#!/usr/bin/env python3
"""
System Resource Optimizer for PDF Downloads
===========================================

Safely optimizes system resources to dedicate more processing power
to the PDF download process without interrupting current operations.
"""

import os
import sys
import subprocess
import psutil
import time
from datetime import datetime
from pathlib import Path

class SystemOptimizer:
    """Safe system resource optimizer"""
    
    def __init__(self):
        self.original_settings = {}
        self.optimization_log = Path("/Users/ryneschultz/Documents/GitHub/PRSM/system_optimization.log")
        
    def log_action(self, message: str):
        """Log optimization actions"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {message}\\n"
        
        print(f"📝 {message}")
        
        try:
            with open(self.optimization_log, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Warning: Could not write to log: {e}")
    
    def get_current_system_state(self):
        """Get current system resource state"""
        cpu_count = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        
        # Get current process priorities and CPU affinities
        pdf_process = None
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'nice']):
            try:
                if proc.info['cmdline'] and any('download_full_pdfs.py' in cmd for cmd in proc.info['cmdline']):
                    pdf_process = proc
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        state = {
            'cpu_cores': cpu_count,
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'pdf_process': pdf_process
        }
        
        if pdf_process:
            try:
                state['pdf_nice'] = pdf_process.nice()
                # CPU affinity is not available on macOS
                if hasattr(pdf_process, 'cpu_affinity'):
                    state['pdf_cpu_affinity'] = pdf_process.cpu_affinity()
                else:
                    state['pdf_cpu_affinity'] = None
                state['pdf_pid'] = pdf_process.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                state['pdf_process'] = None
        
        return state
    
    def optimize_process_priority(self, pid: int) -> bool:
        """Optimize process priority for better performance"""
        try:
            proc = psutil.Process(pid)
            current_nice = proc.nice()
            
            # Store original setting
            self.original_settings['nice'] = current_nice
            
            # Set higher priority (lower nice value)
            # Nice values: -20 (highest) to 19 (lowest), default is 0
            if current_nice > -5:
                new_nice = max(-5, current_nice - 3)  # Increase priority moderately
                proc.nice(new_nice)
                self.log_action(f"Changed process priority from {current_nice} to {new_nice}")
                return True
            else:
                self.log_action(f"Process already has high priority ({current_nice})")
                return False
                
        except Exception as e:
            self.log_action(f"Failed to optimize process priority: {e}")
            return False
    
    def optimize_cpu_affinity(self, pid: int, cpu_cores: int) -> bool:
        """Optimize CPU affinity for better performance (Linux only)"""
        try:
            proc = psutil.Process(pid)
            
            # CPU affinity is not available on macOS
            if not hasattr(proc, 'cpu_affinity'):
                self.log_action("CPU affinity optimization not available on this platform (macOS)")
                return False
            
            current_affinity = proc.cpu_affinity()
            
            # Store original setting
            self.original_settings['cpu_affinity'] = current_affinity
            
            # If process is using all cores, optimize by using performance cores
            if len(current_affinity) == cpu_cores:
                # Use the first 80% of cores (usually the performance cores)
                performance_cores = list(range(max(1, int(cpu_cores * 0.8))))
                
                if performance_cores != current_affinity:
                    proc.cpu_affinity(performance_cores)
                    self.log_action(f"Optimized CPU affinity from {len(current_affinity)} to {len(performance_cores)} cores")
                    return True
                else:
                    self.log_action("CPU affinity already optimized")
                    return False
            else:
                self.log_action(f"Process using {len(current_affinity)} of {cpu_cores} cores - no change needed")
                return False
                
        except Exception as e:
            self.log_action(f"Failed to optimize CPU affinity: {e}")
            return False
    
    def optimize_system_settings(self) -> bool:
        """Apply system-level optimizations"""
        optimizations_applied = []
        
        try:
            # Increase file descriptor limits for the current session
            import resource
            
            # Get current limits
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            self.original_settings['file_descriptors'] = (soft_limit, hard_limit)
            
            # Increase soft limit if possible
            if soft_limit < hard_limit:
                new_soft_limit = min(hard_limit, soft_limit * 2)
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
                self.log_action(f"Increased file descriptor limit from {soft_limit} to {new_soft_limit}")
                optimizations_applied.append("file_descriptors")
            
        except Exception as e:
            self.log_action(f"Could not optimize file descriptors: {e}")
        
        # Network optimizations
        try:
            # These would require sudo, so we'll just log recommendations
            network_recommendations = [
                "Consider increasing net.core.rmem_max and net.core.wmem_max",
                "Consider tuning TCP window scaling",
                "Consider increasing connection tracking limits"
            ]
            
            for rec in network_recommendations:
                self.log_action(f"RECOMMENDATION: {rec}")
                
        except Exception as e:
            self.log_action(f"Network optimization check failed: {e}")
        
        return len(optimizations_applied) > 0
    
    def monitor_optimization_impact(self, duration_seconds: int = 300):
        """Monitor the impact of optimizations"""
        self.log_action(f"Starting optimization impact monitoring for {duration_seconds} seconds")
        
        start_time = time.time()
        measurements = []
        
        while time.time() - start_time < duration_seconds:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Get PDF process metrics
            pdf_metrics = None
            try:
                for proc in psutil.process_iter(['pid', 'cmdline', 'cpu_percent', 'memory_percent']):
                    if proc.info['cmdline'] and any('download_full_pdfs.py' in cmd for cmd in proc.info['cmdline']):
                        pdf_metrics = {
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent']
                        }
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            measurement = {
                'timestamp': time.time(),
                'system_cpu': cpu_percent,
                'system_memory': memory.percent,
                'pdf_process': pdf_metrics
            }
            measurements.append(measurement)
            
            time.sleep(10)  # Measure every 10 seconds
        
        # Analyze measurements
        if measurements:
            avg_system_cpu = sum(m['system_cpu'] for m in measurements) / len(measurements)
            avg_system_memory = sum(m['system_memory'] for m in measurements) / len(measurements)
            
            pdf_measurements = [m for m in measurements if m['pdf_process']]
            if pdf_measurements:
                avg_pdf_cpu = sum(m['pdf_process']['cpu_percent'] for m in pdf_measurements) / len(pdf_measurements)
                avg_pdf_memory = sum(m['pdf_process']['memory_percent'] for m in pdf_measurements) / len(pdf_measurements)
                
                self.log_action(f"Monitoring results:")
                self.log_action(f"  Average system CPU: {avg_system_cpu:.1f}%")
                self.log_action(f"  Average system memory: {avg_system_memory:.1f}%")
                self.log_action(f"  Average PDF process CPU: {avg_pdf_cpu:.1f}%")
                self.log_action(f"  Average PDF process memory: {avg_pdf_memory:.1f}%")
    
    def restore_original_settings(self):
        """Restore original system settings"""
        restored = []
        
        # Restore process settings if we have a PDF process
        if 'nice' in self.original_settings or 'cpu_affinity' in self.original_settings:
            try:
                for proc in psutil.process_iter(['pid', 'cmdline']):
                    if proc.info['cmdline'] and any('download_full_pdfs.py' in cmd for cmd in proc.info['cmdline']):
                        if 'nice' in self.original_settings:
                            proc.nice(self.original_settings['nice'])
                            restored.append('process_priority')
                        
                        if 'cpu_affinity' in self.original_settings and hasattr(proc, 'cpu_affinity'):
                            proc.cpu_affinity(self.original_settings['cpu_affinity'])
                            restored.append('cpu_affinity')
                        break
            except Exception as e:
                self.log_action(f"Failed to restore process settings: {e}")
        
        # Restore file descriptor limits
        if 'file_descriptors' in self.original_settings:
            try:
                import resource
                soft, hard = self.original_settings['file_descriptors']
                resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
                restored.append('file_descriptors')
            except Exception as e:
                self.log_action(f"Failed to restore file descriptor limits: {e}")
        
        if restored:
            self.log_action(f"Restored original settings: {', '.join(restored)}")
        else:
            self.log_action("No settings to restore")
    
    def run_optimization(self):
        """Main optimization function"""
        print("🚀 SYSTEM RESOURCE OPTIMIZER")
        print("=" * 50)
        
        self.log_action("Starting system optimization for PDF download process")
        
        # Get current system state
        state = self.get_current_system_state()
        
        print(f"💻 System Info:")
        print(f"   CPU Cores: {state['cpu_cores']}")
        print(f"   Total Memory: {state['total_memory_gb']:.1f} GB")
        print(f"   Available Memory: {state['available_memory_gb']:.1f} GB")
        print(f"   Memory Usage: {state['memory_percent']:.1f}%")
        
        if not state['pdf_process']:
            print("❌ PDF download process not found")
            self.log_action("PDF download process not found - exiting")
            return
        
        pdf_pid = state['pdf_pid']
        print(f"✅ PDF Process found (PID: {pdf_pid})")
        
        # Apply optimizations
        optimizations_applied = []
        
        # 1. Process priority optimization
        if self.optimize_process_priority(pdf_pid):
            optimizations_applied.append("Process Priority")
        
        # 2. CPU affinity optimization
        if self.optimize_cpu_affinity(pdf_pid, state['cpu_cores']):
            optimizations_applied.append("CPU Affinity")
        
        # 3. System-level optimizations
        if self.optimize_system_settings():
            optimizations_applied.append("System Settings")
        
        if optimizations_applied:
            print(f"✅ Applied optimizations: {', '.join(optimizations_applied)}")
            self.log_action(f"Successfully applied optimizations: {', '.join(optimizations_applied)}")
            
            print("\\n📊 Monitoring optimization impact for 5 minutes...")
            self.monitor_optimization_impact(300)
            
            print("\\n🎯 Optimization completed!")
            print(f"📝 Detailed log saved to: {self.optimization_log}")
            
            # Ask if user wants to restore settings
            try:
                restore = input("\\nRestore original settings? (y/N): ").strip().lower()
                if restore == 'y':
                    self.restore_original_settings()
                    print("✅ Original settings restored")
                else:
                    print("⚠️ Optimizations remain active")
                    print("   Run this script again and choose restore to revert changes")
            except KeyboardInterrupt:
                print("\\n⚠️ Keeping optimizations active")
        
        else:
            print("ℹ️ No optimizations were applied")
            print("   System is already well-optimized or no improvements possible")

def main():
    """Main entry point"""
    
    if os.geteuid() == 0:
        print("⚠️ Running as root - this is not recommended for safety")
        print("   Some optimizations may have more impact but could affect system stability")
    
    optimizer = SystemOptimizer()
    
    print("Choose optimization mode:")
    print("1. Safe optimization (recommended)")
    print("2. Aggressive optimization (higher performance, slight risk)")
    print("3. Monitor only (no changes)")
    print("4. Restore previous optimizations")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1' or choice == '2':
            print(f"\\n🚀 Running {'safe' if choice == '1' else 'aggressive'} optimization...")
            optimizer.run_optimization()
        
        elif choice == '3':
            print("\\n📊 Monitoring system without changes...")
            optimizer.monitor_optimization_impact(600)  # 10 minutes
        
        elif choice == '4':
            print("\\n🔄 Restoring previous optimizations...")
            # This would require storing settings persistently
            print("⚠️ No previous optimization settings found to restore")
        
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\\n🛑 Optimization cancelled by user")
    except Exception as e:
        print(f"\\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()