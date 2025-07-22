#!/usr/bin/env python3
"""
Test Batch PDF Download System
==============================

Tests the batch PDF download functionality with a small sample
before running on the full 149,726 paper corpus.
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from prsm.nwtn.external_storage_config import ExternalKnowledgeBase

async def test_batch_download():
    """Test batch PDF download with a small sample"""
    
    print("🧪 TESTING BATCH PDF DOWNLOAD SYSTEM")
    print("=" * 50)
    print("Testing with small sample before full corpus download")
    print()
    
    # Initialize knowledge base
    print("🔧 Initializing external knowledge base...")
    kb = ExternalKnowledgeBase()
    await kb.initialize()
    
    # Check what papers are available
    cursor = kb.storage_manager.storage_db.cursor()
    cursor.execute("""
        SELECT arxiv_id, title, has_full_content 
        FROM arxiv_papers 
        WHERE has_full_content = 0 OR has_full_content IS NULL
        ORDER BY publish_date DESC 
        LIMIT 10
    """)
    
    sample_papers = cursor.fetchall()
    
    print(f"📋 Found {len(sample_papers)} papers without full content")
    print("Sample papers to test:")
    for i, (arxiv_id, title, has_full) in enumerate(sample_papers[:3], 1):
        print(f"   {i}. {arxiv_id} - {title[:60]}...")
    print()
    
    if not sample_papers:
        print("✅ All papers already have full content!")
        return
    
    print("🚀 Testing batch download with 3 papers...")
    print("Configuration: batch_size=3, max_concurrent=2")
    print()
    
    try:
        # Test with small batch
        stats = await kb.download_all_pdfs_batch(
            batch_size=3,
            max_concurrent=2
        )
        
        print("✅ BATCH DOWNLOAD TEST COMPLETED!")
        print("=" * 50)
        print(f"📊 Results:")
        print(f"   Papers processed: {stats['total_papers']}")
        print(f"   Downloaded: {stats['downloaded']}")
        print(f"   Successfully processed: {stats['processed']}")
        print(f"   Failed: {stats['failed']}")
        print(f"   Skipped: {stats['skipped']}")
        
        if stats['processed'] > 0:
            print(f"   Average content size: {stats.get('average_content_size', 0):,} chars")
            print(f"   Success rate: {stats.get('success_rate', 0):.1f}%")
        
        print()
        
        if stats['processed'] > 0:
            print("🎯 TEST SUCCESSFUL! The batch download system is working correctly.")
            print("📝 You can now run the full corpus download with:")
            print("   python download_full_pdfs.py")
        else:
            print("⚠️  No papers were successfully processed. Check logs for issues.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_batch_download())