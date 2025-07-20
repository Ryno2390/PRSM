#!/usr/bin/env python3
"""
Comprehensive status report for full-scale test
"""
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import re

def get_test_status():
    """Get comprehensive test status"""
    print("🎯 Full-Scale NWTN Provenance Test Status")
    print("=" * 60)
    
    # Check if process is running
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        process_running = 'test_150k_papers_provenance.py' in result.stdout
        
        if process_running:
            # Extract PID and resource usage
            lines = result.stdout.split('\n')
            for line in lines:
                if 'test_150k_papers_provenance.py' in line:
                    parts = line.split()
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    time_used = parts[9]
                    print(f"🔄 Status: RUNNING (PID: {pid})")
                    print(f"💻 CPU Usage: {cpu}%")
                    print(f"🧠 Memory Usage: {mem}%")
                    print(f"⏱️  Process Time: {time_used}")
                    break
        else:
            print("❌ Status: NOT RUNNING")
            
    except Exception as e:
        print(f"❓ Process Status: Unknown ({e})")
    
    # Check log file progress
    log_file = Path('fullscale_test_output.log')
    if log_file.exists():
        print(f"\n📝 Log File: {log_file}")
        print(f"📊 Size: {log_file.stat().st_size:,} bytes")
        
        # Get latest progress
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Find latest progress line
            progress_lines = [line for line in lines if 'Progress:' in line]
            if progress_lines:
                latest_progress = progress_lines[-1].strip()
                print(f"🎯 Latest Progress: {latest_progress}")
                
                # Extract percentage and count
                match = re.search(r'(\d+\.\d+)%.*?(\d+)', latest_progress)
                if match:
                    percentage = float(match.group(1))
                    count = int(match.group(2))
                    
                    # Calculate remaining time estimate
                    total_papers = 151120
                    if count > 0:
                        papers_per_second = count / (time.time() - get_start_time())
                        remaining_papers = total_papers - count
                        remaining_seconds = remaining_papers / papers_per_second
                        remaining_time = timedelta(seconds=remaining_seconds)
                        
                        print(f"📊 Papers Processed: {count:,} / {total_papers:,}")
                        print(f"⚡ Processing Rate: {papers_per_second:.1f} papers/second")
                        print(f"⏳ Estimated Remaining: {remaining_time}")
            
            # Check for errors
            error_lines = [line for line in lines if 'ERROR' in line or 'Failed' in line]
            if error_lines:
                print(f"\n⚠️  Errors Found: {len(error_lines)}")
                print("📝 Recent Errors:")
                for error in error_lines[-5:]:
                    print(f"   {error.strip()}")
            
            # Check for completion
            completion_lines = [line for line in lines if 'SUMMARY' in line or 'Test completed' in line]
            if completion_lines:
                print(f"\n🎉 Test Status: COMPLETED")
                print(f"✅ Completion Message: {completion_lines[-1].strip()}")
                
        except Exception as e:
            print(f"❌ Error reading log: {e}")
    else:
        print("❌ Log file not found")
    
    # Check results file
    results_file = Path('test_provenance_results_fullscale.json')
    if results_file.exists():
        print(f"\n📄 Results File: {results_file}")
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
                
            test_result = results.get('test_result', {})
            print(f"📚 Papers Processed: {test_result.get('papers_processed', 0):,}")
            print(f"✅ Papers Ingested: {test_result.get('papers_successfully_ingested', 0):,}")
            print(f"🔍 Queries Processed: {test_result.get('queries_processed', 0)}")
            print(f"✅ Successful Queries: {test_result.get('successful_queries', 0)}")
            print(f"💰 FTNS Transferred: {test_result.get('total_ftns_transferred', 0)}")
            print(f"👤 Provenance User Earnings: {test_result.get('provenance_user_earnings', 0)}")
            print(f"👤 Prompt User Spending: {test_result.get('prompt_user_spending', 0)}")
            print(f"⏱️  Execution Time: {test_result.get('execution_time_seconds', 0):.1f}s")
            
            if test_result.get('errors'):
                print(f"⚠️  Errors: {len(test_result['errors'])}")
                
        except Exception as e:
            print(f"❌ Error reading results: {e}")
    else:
        print("📄 Results file not yet created")
    
    print("\n" + "=" * 60)

def get_start_time():
    """Get test start time from log file"""
    try:
        log_file = Path('fullscale_test_output.log')
        if log_file.exists():
            with open(log_file, 'r') as f:
                first_line = f.readline()
                if 'started at:' in first_line:
                    time_str = first_line.split('started at: ')[1].strip()
                    return datetime.fromisoformat(time_str).timestamp()
        return time.time() - 300  # Fallback: 5 minutes ago
    except:
        return time.time() - 300

if __name__ == "__main__":
    get_test_status()