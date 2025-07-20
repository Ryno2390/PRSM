#!/usr/bin/env python3
"""
Check NWTN Interface Test Status
===============================

Quick status check for the running NWTN interface test.
"""

import subprocess
import sys
import time
from datetime import datetime

def check_test_status():
    """Check if the NWTN interface test is still running"""
    print("🔍 NWTN Interface Test Status Check")
    print("=" * 50)
    print(f"📅 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Check if test is running
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        
        running_tests = []
        for line in result.stdout.split('\n'):
            if 'test_fixed_nwtn_interface.py' in line and 'grep' not in line:
                running_tests.append(line.strip())
        
        if running_tests:
            print("✅ NWTN Interface Test: RUNNING")
            print(f"📊 Active processes: {len(running_tests)}")
            for test in running_tests:
                parts = test.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    print(f"   🔹 PID: {pid}, CPU: {cpu}%, Memory: {mem}")
        else:
            print("⏹️  NWTN Interface Test: NOT RUNNING")
            print("🔍 Test may have completed or stopped")
        
        print()
        
        # Check for any output files or logs
        import os
        current_dir = "/Users/ryneschultz/Documents/GitHub/PRSM"
        
        # Look for any new log files or output
        potential_outputs = [
            "nwtn_test_output.log",
            "test_results.json",
            "nwtn_interface_results.txt"
        ]
        
        outputs_found = []
        for output_file in potential_outputs:
            full_path = os.path.join(current_dir, output_file)
            if os.path.exists(full_path):
                stat = os.stat(full_path)
                outputs_found.append({
                    'file': output_file,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })
        
        if outputs_found:
            print("📄 Output Files Found:")
            for output in outputs_found:
                print(f"   📝 {output['file']}: {output['size']} bytes, modified {output['modified']}")
        else:
            print("📄 No output files detected yet")
        
        print()
        
        # Memory usage check
        try:
            memory_result = subprocess.run(['ps', '-o', 'pid,rss,comm', '-p', str(running_tests[0].split()[1]) if running_tests else '0'], 
                                         capture_output=True, text=True)
            if "test_fixed_nwtn_interface" in memory_result.stdout:
                lines = memory_result.stdout.strip().split('\n')
                if len(lines) > 1:
                    data = lines[1].split()
                    memory_mb = int(data[1]) / 1024 if len(data) > 1 else 0
                    print(f"💾 Memory Usage: {memory_mb:.1f} MB")
        except:
            pass
        
        print("🎯 Test Status: In Progress")
        print("⏳ Estimated completion: 2-5 minutes for full reasoning test")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")

if __name__ == "__main__":
    check_test_status()