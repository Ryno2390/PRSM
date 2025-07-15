#!/usr/bin/env python3
"""
Speed Monitor for Optimized Ingestion
=====================================

Simple monitoring script to track the speed improvements.
"""

import time
import os
from pathlib import Path
from datetime import datetime

def monitor_speed():
    """Monitor ingestion speed"""
    
    storage_path = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Content")
    
    print("⚡ SPEED-OPTIMIZED INGESTION MONITOR")
    print("=" * 50)
    print("📊 Monitoring file creation rate...")
    print("🔄 Press Ctrl+C to stop")
    print()
    
    # Get initial count
    if storage_path.exists():
        initial_count = len(list(storage_path.rglob("*.dat")))
    else:
        initial_count = 0
    
    start_time = time.time()
    last_count = initial_count
    last_time = start_time
    
    print(f"📁 Starting count: {initial_count:,} files")
    print()
    
    try:
        while True:
            time.sleep(60)  # Check every minute
            
            # Get current count
            if storage_path.exists():
                current_count = len(list(storage_path.rglob("*.dat")))
            else:
                current_count = 0
            
            current_time = time.time()
            
            # Calculate rates
            total_elapsed = current_time - start_time
            interval_elapsed = current_time - last_time
            
            total_added = current_count - initial_count
            interval_added = current_count - last_count
            
            overall_rate = (total_added / total_elapsed) * 3600 if total_elapsed > 0 else 0
            interval_rate = (interval_added / interval_elapsed) * 3600 if interval_elapsed > 0 else 0
            
            # Display status
            print(f"📊 {datetime.now().strftime('%H:%M:%S')} - Files: {current_count:,} (+{interval_added:,} in last min)")
            print(f"   ⚡ Current Rate: {interval_rate:.0f} files/hour")
            print(f"   📈 Overall Rate: {overall_rate:.0f} files/hour")
            print(f"   ⏱️ Runtime: {total_elapsed/3600:.1f} hours")
            
            # Estimate completion
            if overall_rate > 0:
                remaining = 150000 - current_count
                hours_remaining = remaining / overall_rate
                print(f"   🎯 Est. Completion: {hours_remaining:.1f} hours remaining")
            
            print()
            
            # Update for next iteration
            last_count = current_count
            last_time = current_time
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")
        final_count = len(list(storage_path.rglob("*.dat"))) if storage_path.exists() else 0
        final_elapsed = time.time() - start_time
        final_rate = ((final_count - initial_count) / final_elapsed) * 3600 if final_elapsed > 0 else 0
        
        print(f"📊 Final Stats:")
        print(f"   📁 Total Files: {final_count:,}")
        print(f"   ➕ Files Added: {final_count - initial_count:,}")
        print(f"   ⏱️ Runtime: {final_elapsed/3600:.1f} hours")
        print(f"   📈 Average Rate: {final_rate:.0f} files/hour")

if __name__ == "__main__":
    monitor_speed()