#!/usr/bin/env python3
"""
Debug PDF Download Performance Issues
====================================

This script investigates why PDF downloads went from thousands/day to 2/night.
"""

import sys
import sqlite3
import time
import asyncio
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_database_status():
    """Check the current database status"""
    db_path = "/Volumes/My Passport/PRSM_Storage/01_RAW_PAPERS/storage.db"
    
    print("📊 DATABASE STATUS CHECK")
    print("=" * 50)
    
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()
        
        # Basic counts
        cursor.execute("SELECT COUNT(*) FROM arxiv_papers")
        total_papers = cursor.fetchone()[0]
        print(f"Total papers in database: {total_papers:,}")
        
        cursor.execute("SELECT COUNT(*) FROM arxiv_papers WHERE has_full_content = 1")
        processed_papers = cursor.fetchone()[0]
        print(f"Papers with full content: {processed_papers:,}")
        
        remaining = total_papers - processed_papers
        print(f"Papers remaining to process: {remaining:,}")
        
        # Check recent processing activity
        cursor.execute("""
            SELECT COUNT(*) FROM arxiv_papers 
            WHERE processed_date IS NOT NULL 
            AND date(processed_date) = date('now')
        """)
        today_processed = cursor.fetchone()[0]
        print(f"Papers processed today: {today_processed:,}")
        
        cursor.execute("""
            SELECT COUNT(*) FROM arxiv_papers 
            WHERE processed_date IS NOT NULL 
            AND date(processed_date) = date('now', '-1 day')
        """)
        yesterday_processed = cursor.fetchone()[0]
        print(f"Papers processed yesterday: {yesterday_processed:,}")
        
        # Check for recent errors or patterns
        cursor.execute("""
            SELECT arxiv_id, processed_date FROM arxiv_papers 
            WHERE processed_date IS NOT NULL 
            ORDER BY processed_date DESC 
            LIMIT 10
        """)
        recent_papers = cursor.fetchall()
        print(f"\nRecent successful processing:")
        for paper_id, proc_date in recent_papers:
            print(f"  {paper_id}: {proc_date}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

def test_single_download():
    """Test downloading a single paper to identify bottlenecks"""
    print("\n🧪 SINGLE DOWNLOAD TEST")
    print("=" * 50)
    
    # Try to import the external storage module
    try:
        from prsm.nwtn.external_storage_config import ExternalKnowledgeBase
        
        async def test_download():
            kb = ExternalKnowledgeBase()
            await kb.initialize()
            
            # Get a paper that needs processing
            cursor = kb.storage_manager.storage_db.cursor()
            cursor.execute("""
                SELECT arxiv_id, title FROM arxiv_papers 
                WHERE has_full_content = 0 OR has_full_content IS NULL
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if not row:
                print("No papers need processing")
                return
                
            arxiv_id, title = row
            print(f"Testing download for: {arxiv_id}")
            print(f"Title: {title[:100]}...")
            
            start_time = time.time()
            
            # Test the download and processing pipeline
            result = await kb._download_and_process_paper_with_semaphore(
                asyncio.Semaphore(1), arxiv_id, title
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"Result: {result}")
            print(f"Processing time: {processing_time:.2f} seconds")
            
            if result and result.get('processed'):
                print("✅ Single download test PASSED")
            else:
                print("❌ Single download test FAILED")
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        # Run the async test
        asyncio.run(test_download())
        
    except Exception as e:
        print(f"❌ Single download test failed: {e}")
        import traceback
        traceback.print_exc()

def analyze_performance_difference():
    """Analyze what changed between high-performance and current versions"""
    print("\n📈 PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    print("Previous high-performance stats:")
    print("  - 70,319 successful downloads")
    print("  - 2.42 seconds per paper average")
    print("  - 47% success rate") 
    print("  - ~695 papers per hour")
    print("  - ~16,680 papers per day")
    
    print("\nCurrent performance (last night):")
    print("  - 2 successful downloads out of 113 attempts")
    print("  - 1.8% success rate")
    print("  - ~0.2 papers per hour")
    print("  - ~5 papers per day")
    
    print("\n🔍 POTENTIAL ISSUES:")
    print("1. Network connectivity issues")
    print("2. arXiv server blocking/rate limiting")
    print("3. Database corruption or locking")
    print("4. External drive performance issues")
    print("5. Python environment or dependency changes")
    print("6. Script logic changes")

def check_network_connectivity():
    """Test network connectivity to arXiv"""
    print("\n🌐 NETWORK CONNECTIVITY TEST")
    print("=" * 50)
    
    import subprocess
    
    # Test basic connectivity
    try:
        result = subprocess.run(['ping', '-c', '3', 'arxiv.org'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ arxiv.org is reachable")
        else:
            print("❌ arxiv.org is not reachable")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Ping test failed: {e}")
    
    # Test HTTP connectivity to arXiv PDF endpoint
    try:
        import aiohttp
        import asyncio
        
        async def test_arxiv_access():
            # Try to access a known paper
            test_url = "https://arxiv.org/pdf/2301.00001"
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(test_url) as response:
                        print(f"arXiv PDF access test: {response.status}")
                        if response.status == 200:
                            print("✅ Can access arXiv PDFs")
                        elif response.status == 404:
                            print("🔍 Test paper not found (expected)")
                        else:
                            print(f"⚠️  Unexpected status: {response.status}")
                except Exception as e:
                    print(f"❌ arXiv access failed: {e}")
        
        asyncio.run(test_arxiv_access())
        
    except Exception as e:
        print(f"❌ HTTP test failed: {e}")

if __name__ == "__main__":
    print("🔍 PDF DOWNLOAD PERFORMANCE DIAGNOSTIC")
    print("=" * 60)
    print()
    
    check_database_status()
    check_network_connectivity()
    test_single_download()
    analyze_performance_difference()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS COMPLETE")
    print("Review the output above to identify the root cause.")