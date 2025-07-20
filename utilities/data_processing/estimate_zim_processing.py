#!/usr/bin/env python3
"""
Estimate ZIM file processing time
"""

import os
import time
from zimply.zimply import ZIMFile

def estimate_processing_time():
    """Estimate how long it would take to process the physics ZIM file"""
    
    print("⏱️ ZIM File Processing Time Estimation")
    print("=" * 50)
    
    zim_path = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/raw_sources/zim_files/physics_wikipedia_no_pic.zim"
    
    if not os.path.exists(zim_path):
        print(f"❌ ZIM file not found: {zim_path}")
        return
    
    file_size_mb = os.path.getsize(zim_path) / (1024 * 1024)
    print(f"📁 File size: {file_size_mb:.1f} MB")
    
    try:
        print("🔍 Opening ZIM file...")
        start_time = time.time()
        
        zim = ZIMFile(zim_path, 'utf-8')
        
        open_time = time.time() - start_time
        print(f"⏳ Time to open: {open_time:.2f} seconds")
        
        total_articles = len(zim)
        print(f"📚 Total articles: {total_articles:,}")
        
        # Test processing a small sample
        print("\n🧪 Testing sample processing...")
        sample_size = 10
        sample_articles = []
        
        start_sample = time.time()
        count = 0
        
        for article_name in zim:
            if count >= sample_size:
                break
            
            try:
                content = zim[article_name]
                if content and len(content.strip()) > 100:
                    sample_articles.append((article_name, len(content)))
                    count += 1
            except Exception as e:
                continue
        
        sample_time = time.time() - start_sample
        print(f"⏳ Time to process {len(sample_articles)} articles: {sample_time:.2f} seconds")
        
        if sample_articles:
            avg_time_per_article = sample_time / len(sample_articles)
            print(f"📊 Average time per article: {avg_time_per_article:.3f} seconds")
            
            # Estimate total time
            estimated_total_seconds = avg_time_per_article * total_articles
            estimated_minutes = estimated_total_seconds / 60
            estimated_hours = estimated_minutes / 60
            
            print(f"\n📈 Estimates for full processing:")
            print(f"   Total time: {estimated_total_seconds:.0f} seconds")
            print(f"   Total time: {estimated_minutes:.1f} minutes")
            print(f"   Total time: {estimated_hours:.1f} hours")
            
            # Show sample articles
            print(f"\n📄 Sample articles processed:")
            for name, length in sample_articles[:5]:
                print(f"   - {name} ({length:,} chars)")
            
            # More realistic estimates
            print(f"\n🎯 More realistic estimates:")
            print(f"   If we only process key physics topics (~500 articles):")
            key_topics_time = avg_time_per_article * 500
            print(f"   Time: {key_topics_time:.0f} seconds ({key_topics_time/60:.1f} minutes)")
            
            print(f"\n   If we process 50 articles per minute:")
            articles_per_minute = 50
            realistic_minutes = total_articles / articles_per_minute
            print(f"   Time: {realistic_minutes:.1f} minutes ({realistic_minutes/60:.1f} hours)")
            
        else:
            print("❌ No sample articles processed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    estimate_processing_time()