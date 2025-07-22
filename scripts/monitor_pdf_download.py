#!/usr/bin/env python3
"""
PDF Download Monitor
===================

Monitor the progress of the background PDF download process.
"""

import os
import sys
import time
from datetime import datetime

def monitor_pdf_download():
    """Monitor the PDF download progress"""
    
    print("📊 NWTN PDF DOWNLOAD MONITOR")
    print("=" * 50)
    print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if process is running
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    pdf_processes = [line for line in result.stdout.split('\n') if 'download_full_pdfs.py' in line and 'grep' not in line]
    
    if pdf_processes:
        print("✅ PDF DOWNLOAD PROCESS STATUS: RUNNING")
        for proc in pdf_processes:
            parts = proc.split()
            if len(parts) >= 11:
                cpu_usage = parts[2]
                memory_usage = parts[3]
                start_time = parts[8]
                runtime = parts[9]
                print(f"   PID: {parts[1]}")
                print(f"   CPU Usage: {cpu_usage}%")
                print(f"   Memory Usage: {memory_usage}%")
                print(f"   Start Time: {start_time}")
                print(f"   Runtime: {runtime}")
        print()
    else:
        print("❌ PDF DOWNLOAD PROCESS STATUS: NOT RUNNING")
        print()
    
    # Check log file
    log_file = "pdf_download_log.txt"
    if os.path.exists(log_file):
        print("📄 LOG FILE ANALYSIS:")
        
        with open(log_file, 'r') as f:
            content = f.read()
            
        lines = content.split('\n')
        total_lines = len(lines)
        
        print(f"   Log file size: {len(content):,} characters")
        print(f"   Total log lines: {total_lines:,}")
        
        # Look for progress indicators
        processed_count = content.count("Successfully processed")
        failed_count = content.count("Failed to download PDF")
        batch_completed = content.count("Batch") and content.count("completed")
        
        print(f"   Papers processed: {processed_count}")
        print(f"   Download failures: {failed_count}")
        print(f"   Batches completed: {batch_completed}")
        
        # Show recent activity (last 10 lines)
        print("\n📝 RECENT ACTIVITY (Last 10 log lines):")
        recent_lines = lines[-10:] if len(lines) > 10 else lines
        for i, line in enumerate(recent_lines):
            if line.strip():
                print(f"   {line}")
        print()
        
        # Check for completion
        if "Batch PDF download completed!" in content:
            print("🎉 DOWNLOAD COMPLETED!")
            print("All 149,726 papers have been processed with full PDF content.")
        elif "Starting batch PDF download" in content:
            print("🚀 DOWNLOAD IN PROGRESS")
            print("The system is actively downloading and processing PDFs.")
        
    else:
        print("❌ LOG FILE NOT FOUND")
        print("The download process may not have started yet.")
    
    print()
    print("🔄 MONITORING COMMANDS:")
    print("   python monitor_pdf_download.py    - Check current status")
    print("   tail -f pdf_download_log.txt      - Real-time log monitoring")
    print("   ps aux | grep download_full_pdfs  - Check if process is running")
    print()
    
    if pdf_processes:
        print("💡 The download is running in the background and will continue")
        print("   even if you close this terminal. Expected completion: 60-80 hours.")
    else:
        print("⚠️  The download process is not running. You may need to restart it.")

if __name__ == "__main__":
    monitor_pdf_download()