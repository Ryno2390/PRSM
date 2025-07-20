#!/usr/bin/env python3
"""
Monitor arXiv Ingestion Progress
===============================

Monitor the progress of the 150K arXiv papers ingestion process.
"""

import sys
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, '.')

from prsm.nwtn.external_storage_config import ExternalStorageConfig, ExternalStorageManager

async def check_ingestion_status():
    """Check the current status of arXiv ingestion"""
    
    print("📊 ARXIV INGESTION STATUS MONITOR")
    print("=" * 50)
    
    # Check progress file
    progress_file = "arxiv_ingestion_progress.json"
    progress_data = None
    
    if Path(progress_file).exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read progress file: {e}")
    
    if progress_data:
        processed = progress_data.get("processed_count", 0)
        timestamp = progress_data.get("timestamp", "")
        status = progress_data.get("status", "unknown")
        
        print(f"📋 Progress File Status:")
        print(f"   Papers processed: {processed:,} / 150,000")
        print(f"   Progress: {(processed/150000)*100:.1f}%")
        print(f"   Last update: {timestamp}")
        print(f"   Status: {status}")
        print()
    else:
        print("📋 No progress file found - ingestion not started")
        print()
    
    # Check database status
    try:
        print("🗄️ Checking database status...")
        external_config = ExternalStorageConfig()
        storage_manager = ExternalStorageManager(external_config)
        await storage_manager.initialize()
        
        if storage_manager.storage_db:
            cursor = storage_manager.storage_db.cursor()
            
            # Check if arxiv_papers table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='arxiv_papers'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Count papers in database
                cursor.execute("SELECT COUNT(*) FROM arxiv_papers")
                db_count = cursor.fetchone()[0]
                
                # Get domain breakdown
                cursor.execute("SELECT domain, COUNT(*) FROM arxiv_papers GROUP BY domain ORDER BY COUNT(*) DESC LIMIT 10")
                domain_stats = cursor.fetchall()
                
                print(f"📊 Database Status:")
                print(f"   Papers in database: {db_count:,}")
                print(f"   Progress: {(db_count/150000)*100:.1f}%")
                print()
                
                if domain_stats:
                    print("🎯 Top Domains:")
                    for domain, count in domain_stats:
                        print(f"   {domain}: {count:,} papers")
                    print()
                
                # Get recent papers
                cursor.execute("SELECT title, domain, publish_date FROM arxiv_papers ORDER BY created_at DESC LIMIT 5")
                recent_papers = cursor.fetchall()
                
                if recent_papers:
                    print("📄 Recent Papers Added:")
                    for title, domain, pub_date in recent_papers:
                        print(f"   {domain}: {title[:60]}...")
                    print()
            else:
                print("❌ arxiv_papers table not found - ingestion not started")
                print()
        
        storage_manager.close()
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        print()
    
    # Check storage usage
    try:
        storage_path = Path("/Volumes/My Passport/PRSM_Storage")
        if storage_path.exists():
            import shutil
            usage = shutil.disk_usage(storage_path)
            used_gb = (usage.total - usage.free) / (1024**3)
            free_gb = usage.free / (1024**3)
            
            print(f"💾 Storage Usage:")
            print(f"   Used: {used_gb:.1f} GB")
            print(f"   Free: {free_gb:.1f} GB")
            print()
        else:
            print("❌ External storage not found")
            print()
    except Exception as e:
        print(f"⚠️ Storage check failed: {e}")
        print()
    
    # Recommendations
    if progress_data:
        processed = progress_data.get("processed_count", 0)
        if processed == 0:
            print("💡 Recommendations:")
            print("   Run: python ingest_150k_arxiv_papers_background.py")
        elif processed < 150000:
            print("💡 Recommendations:")
            print("   Continue ingestion: python ingest_150k_arxiv_papers_background.py")
            print("   Or wait for current process to complete")
        else:
            print("🎉 Ingestion Complete! Ready to test NWTN pipeline")
            print("💡 Next step: python test_complete_nwtn_pipeline_with_real_arxiv.py")
    
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(check_ingestion_status())