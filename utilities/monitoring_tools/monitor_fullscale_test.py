#!/usr/bin/env python3
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
