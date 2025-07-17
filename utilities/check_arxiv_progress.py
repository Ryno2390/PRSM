#!/usr/bin/env python3
"""
Check arXiv Processing Progress
==============================

Quick status checker for background arXiv processing.
"""

import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

def check_progress():
    """Check current processing progress"""
    
    print("📊 ARXIV PROCESSING STATUS")
    print("=" * 50)
    
    # Check progress file
    progress_file = Path('/Users/ryneschultz/Documents/GitHub/PRSM/arxiv_progress.json')
    
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        print(f"📄 Papers processed: {progress['processed']:,}")
        print(f"✅ Papers accepted: {progress['accepted']:,}")
        print(f"🎯 Target: {progress['target']:,}")
        print(f"📈 Progress: {(progress['processed'] / progress['target']) * 100:.1f}%")
        print(f"📊 Acceptance rate: {progress['acceptance_rate']:.1f}%")
        print(f"⚡ Processing rate: {progress['rate_per_second']:.1f} papers/sec")
        
        if progress['estimated_completion'] > 0:
            completion_time = datetime.now() + timedelta(seconds=progress['estimated_completion'])
            print(f"🕒 Estimated completion: {completion_time.strftime('%H:%M:%S')}")
        
        # Check if process is still running
        last_update = datetime.fromisoformat(progress['timestamp'])
        time_since_update = (datetime.now() - last_update).total_seconds()
        
        if time_since_update < 120:  # Less than 2 minutes ago
            print("🟢 Status: RUNNING")
        else:
            print("🟡 Status: STALLED or COMPLETED")
    else:
        print("❌ No progress file found")
    
    print()
    
    # Check database
    db_path = '/Volumes/My Passport/PRSM_Storage/storage.db'
    
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get current count
            cursor.execute('SELECT COUNT(*) FROM content_storage WHERE content_type = "content"')
            current_count = cursor.fetchone()[0]
            
            # Get domains
            cursor.execute('SELECT DISTINCT json_extract(content_data, "$.domain") FROM content_storage WHERE content_type = "content" LIMIT 10')
            domains = [row[0] for row in cursor.fetchall() if row[0]]
            
            conn.close()
            
            print("💾 DATABASE STATUS")
            print("=" * 50)
            print(f"📚 Papers stored: {current_count:,}")
            print(f"🌍 Domains found: {', '.join(domains[:5])}")
            if len(domains) > 5:
                print(f"    ... and {len(domains) - 5} more")
            
            if current_count > 0:
                print("✅ READY FOR NWTN TESTING!")
            else:
                print("⏳ Waiting for papers to be processed...")
                
        except Exception as e:
            print(f"❌ Database error: {e}")
    else:
        print("❌ Storage database not found")
    
    print()
    
    # Check final report
    report_file = Path('/Users/ryneschultz/Documents/GitHub/PRSM/arxiv_final_report.json')
    
    if report_file.exists():
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        print("📋 FINAL REPORT")
        print("=" * 50)
        print(f"🎉 Processing completed: {report['completion_time']}")
        print(f"📄 Final paper count: {report['final_paper_count']:,}")
        print(f"📈 Completion rate: {report['completion_percentage']:.1f}%")
        print(f"🌍 Domains: {', '.join(report['domains_processed'])}")
        
        if report['ready_for_nwtn_testing']:
            print("🚀 READY FOR NWTN TESTING!")
        else:
            print("⚠️  Not ready for testing yet")

if __name__ == "__main__":
    check_progress()