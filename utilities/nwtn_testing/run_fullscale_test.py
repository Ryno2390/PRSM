#!/usr/bin/env python3
"""
Background runner for full-scale 150K+ paper provenance test
"""
import asyncio
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import json
import os

def run_fullscale_test():
    """Run the full-scale test in the background"""
    print("🚀 Starting full-scale 150K+ paper provenance test in background...")
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '/Users/ryneschultz/Documents/GitHub/PRSM'
    
    # Command to run
    cmd = [
        sys.executable, 
        'tests/test_150k_papers_provenance.py',
        '--papers-path', '/Volumes/My Passport/PRSM_Storage/PRSM_Content/hot',
        '--output', 'test_provenance_results_fullscale.json'
    ]
    
    # Log file for output
    log_file = Path('fullscale_test_output.log')
    
    print(f"📝 Logging output to: {log_file}")
    print(f"🔍 Command: {' '.join(cmd)}")
    print(f"⏰ Started at: {datetime.now().isoformat()}")
    
    # Run the process
    with open(log_file, 'w') as f:
        f.write(f"Full-scale test started at: {datetime.now().isoformat()}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("=" * 80 + "\n\n")
        f.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd='/Users/ryneschultz/Documents/GitHub/PRSM',
            text=True,
            bufsize=1
        )
        
        print(f"🎯 Process started with PID: {process.pid}")
        print(f"📊 Processing 151,120 papers...")
        print(f"⏱️  Estimated time: ~20-30 minutes")
        print(f"🔄 Monitor with: tail -f {log_file}")
        
        return process

def create_progress_monitor():
    """Create a simple progress monitoring script"""
    monitor_script = """#!/usr/bin/env python3
import time
import json
from pathlib import Path
from datetime import datetime

def monitor_progress():
    log_file = Path('fullscale_test_output.log')
    results_file = Path('test_provenance_results_fullscale.json')
    
    print("📊 Full-scale Test Progress Monitor")
    print("=" * 50)
    
    while True:
        try:
            # Check if log file exists
            if log_file.exists():
                # Get file size and last few lines
                size = log_file.stat().st_size
                print(f"📝 Log file size: {size:,} bytes")
                
                # Try to find progress indicators
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                # Look for progress indicators
                progress_lines = [line for line in lines[-50:] if 'Progress:' in line or 'processed' in line.lower()]
                if progress_lines:
                    print(f"🎯 Latest progress: {progress_lines[-1].strip()}")
                
                # Look for completion
                completion_lines = [line for line in lines[-20:] if 'SUMMARY' in line or 'completed' in line.lower()]
                if completion_lines:
                    print(f"✅ Status: {completion_lines[-1].strip()}")
            
            # Check if results file exists
            if results_file.exists():
                print(f"📄 Results file created: {results_file}")
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        test_result = results.get('test_result', {})
                        print(f"📚 Papers processed: {test_result.get('papers_processed', 0):,}")
                        print(f"✅ Papers ingested: {test_result.get('papers_successfully_ingested', 0):,}")
                        print(f"🔍 Queries processed: {test_result.get('queries_processed', 0)}")
                        print(f"💰 FTNS transferred: {test_result.get('total_ftns_transferred', 0)}")
                        print("🎉 Test completed successfully!")
                        break
                except:
                    print("📄 Results file exists but not yet complete")
            
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Checking again in 30 seconds...")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
        
        time.sleep(30)

if __name__ == "__main__":
    monitor_progress()
"""
    
    monitor_file = Path('monitor_fullscale_test.py')
    with open(monitor_file, 'w') as f:
        f.write(monitor_script)
    
    # Make it executable
    monitor_file.chmod(0o755)
    
    print(f"📈 Created progress monitor: {monitor_file}")
    print(f"🔄 Run with: python {monitor_file}")
    
    return monitor_file

if __name__ == "__main__":
    # Create progress monitor
    monitor_file = create_progress_monitor()
    
    # Start the full-scale test
    process = run_fullscale_test()
    
    print("\n" + "=" * 80)
    print("🎯 Full-scale test is now running in the background!")
    print("=" * 80)
    print(f"📊 Processing: 151,120 papers")
    print(f"⏱️  Estimated: 20-30 minutes")
    print(f"🔄 Monitor: python {monitor_file}")
    print(f"📝 Log: tail -f fullscale_test_output.log")
    print(f"🛑 Stop: kill {process.pid}")
    print("=" * 80)