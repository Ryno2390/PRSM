#!/usr/bin/env python3
"""
Direct ZIM Article Lookup (No Iteration)
========================================

This script avoids the iteration bottleneck by directly looking up
specific articles in the ZIM file without iterating through all entries.
"""

import os
import json
import time
from typing import Dict, List, Optional
from zimply.zimply import ZIMFile

def direct_zim_lookup(zim_path: str, target_topics: List[str]) -> Dict[str, str]:
    """
    Look up specific articles directly without iterating through the ZIM file
    
    Args:
        zim_path: Path to ZIM file
        target_topics: List of topics to look up directly
        
    Returns:
        Dictionary of found articles
    """
    
    print(f"🔍 Opening ZIM file for direct lookup: {zim_path}")
    
    try:
        zim = ZIMFile(zim_path, 'utf-8')
        
        print(f"🎯 Attempting direct lookup of {len(target_topics)} topics...")
        
        found_articles = {}
        
        for i, topic in enumerate(target_topics):
            # Progress indicator
            if i % 5 == 0:
                print(f"   Progress: {i}/{len(target_topics)} topics checked")
            
            # Try direct lookup first
            try:
                if topic in zim:
                    content = zim[topic]
                    if content and len(content.strip()) > 200:
                        found_articles[topic] = content
                        print(f"   ✅ Direct hit: {topic} ({len(content):,} chars)")
                        continue
            except Exception as e:
                pass
            
            # Try common variations without iteration
            variations = [
                topic.replace(" ", "_"),              # Space to underscore
                topic.replace("_", " "),              # Underscore to space
                topic.replace("'", "'"),              # Different apostrophe
                topic.replace("'", "'"),              # Different apostrophe
                topic.lower(),                        # Lowercase
                topic.upper(),                        # Uppercase
                topic.title(),                        # Title case
                topic.replace(" ", "_").lower(),      # URL format lowercase
                topic.replace(" ", "_").title(),      # URL format title case
                topic.replace("'s", "s"),             # Remove possessive
                topic.replace("'s", "s"),             # Remove possessive (different apostrophe)
                topic.replace(" law", ""),            # Remove "law" suffix
                topic.replace(" constant", ""),       # Remove "constant" suffix
                topic.replace("First ", "1st "),      # Number variations
                topic.replace("Second ", "2nd "),     # Number variations
                topic.replace("Third ", "3rd "),      # Number variations
                topic.replace("Newton's", "Newtons"), # Possessive variations
                topic.replace("Maxwell's", "Maxwells"), # Possessive variations
                topic.replace("Planck's", "Plancks"), # Possessive variations
                topic.replace("Coulomb's", "Coulombs"), # Possessive variations
                topic.replace("Hooke's", "Hookes"),   # Possessive variations
                topic.replace("Ohm's", "Ohms"),       # Possessive variations
                topic.replace("Faraday's", "Faradays"), # Possessive variations
                topic.replace("Boyle's", "Boyles"),   # Possessive variations
                topic.replace("Charles's", "Charles"), # Possessive variations
            ]
            
            found_variation = False
            for variation in variations:
                try:
                    if variation in zim:
                        content = zim[variation]
                        if content and len(content.strip()) > 200:
                            found_articles[topic] = content
                            print(f"   ✅ Found via variation '{variation}': {topic} ({len(content):,} chars)")
                            found_variation = True
                            break
                except Exception as e:
                    continue
            
            if not found_variation:
                print(f"   ❌ Not found: {topic}")
        
        print(f"\n📝 Direct lookup completed: {len(found_articles)} articles found")
        return found_articles
        
    except Exception as e:
        print(f"❌ Error with ZIM file: {e}")
        return {}

def test_direct_lookup():
    """Test the direct lookup approach with a small sample"""
    
    print("🧪 Testing Direct ZIM Lookup")
    print("=" * 50)
    
    zim_path = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/raw_sources/zim_files/physics_wikipedia_no_pic.zim"
    
    if not os.path.exists(zim_path):
        print(f"❌ ZIM file not found: {zim_path}")
        return
    
    # Test with a small sample of high-priority topics
    test_topics = [
        "Force",
        "Energy",
        "Mass",
        "Newton's laws of motion",
        "Conservation of energy",
        "Speed of light",
        "Gravity",
        "Momentum",
        "Acceleration",
        "Velocity"
    ]
    
    print(f"🎯 Testing with {len(test_topics)} sample topics...")
    
    start_time = time.time()
    
    found_articles = direct_zim_lookup(zim_path, test_topics)
    
    end_time = time.time()
    
    print(f"\n📊 Test Results:")
    print(f"   Topics tested: {len(test_topics)}")
    print(f"   Articles found: {len(found_articles)}")
    print(f"   Success rate: {len(found_articles)/len(test_topics)*100:.1f}%")
    print(f"   Processing time: {end_time - start_time:.2f} seconds")
    print(f"   Speed: {len(test_topics)/(end_time - start_time):.1f} topics/second")
    
    # Show found articles
    print(f"\n📄 Found Articles:")
    for topic, content in found_articles.items():
        print(f"   - {topic}: {len(content):,} characters")
        
        # Show first few sentences
        sentences = content.split('.')[:2]
        preview = '. '.join(sentences) + '...'
        print(f"     Preview: {preview[:150]}...")
        print()
    
    # Save test results
    if found_articles:
        output_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/physics/direct_lookup_test.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(found_articles, f, indent=2, ensure_ascii=False)
        print(f"💾 Test results saved to: {output_file}")
    
    # Estimate full extraction time
    if len(found_articles) > 0:
        topics_per_second = len(test_topics) / (end_time - start_time)
        
        # Load full topic list
        try:
            with open("/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/physics/top_tier_topics.json", 'r') as f:
                full_topics = json.load(f)
            
            estimated_time = len(full_topics) / topics_per_second
            print(f"\n⏱️  Estimated time for full top-tier extraction:")
            print(f"   Full topics: {len(full_topics)}")
            print(f"   Estimated time: {estimated_time:.1f} seconds ({estimated_time/60:.1f} minutes)")
            
        except Exception as e:
            print(f"Could not load full topics list: {e}")
    
    return len(found_articles) > 0

if __name__ == "__main__":
    success = test_direct_lookup()
    
    if success:
        print("\n✅ Direct lookup method works! Ready for full extraction.")
    else:
        print("\n❌ Direct lookup method failed.")