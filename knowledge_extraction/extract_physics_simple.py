#!/usr/bin/env python3
"""
Simple Physics Knowledge Extraction Test
"""

import os
import json
from zimply.zimply import ZIMFile

def test_zim_extraction():
    """Test basic ZIM file extraction"""
    
    print("🧠 Simple Physics Knowledge Extraction Test")
    print("=" * 50)
    
    zim_path = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/raw_sources/zim_files/physics_wikipedia_no_pic.zim"
    
    if not os.path.exists(zim_path):
        print(f"❌ ZIM file not found: {zim_path}")
        return
    
    try:
        print("🔍 Opening ZIM file...")
        zim = ZIMFile(zim_path, 'utf-8')
        
        print(f"📚 ZIM file contains {len(zim)} articles")
        
        # Test with a few key physics topics
        test_topics = [
            "Force",
            "Newton's laws of motion",
            "newtons_laws_of_motion",
            "Newton's_laws_of_motion",
            "Mass",
            "Acceleration",
            "Energy",
            "Momentum",
            "Gravity",
            "Thermodynamics"
        ]
        
        found_articles = {}
        
        for topic in test_topics:
            try:
                if topic in zim:
                    content = zim[topic]
                    if content and len(content.strip()) > 100:
                        found_articles[topic] = content[:1000]  # First 1000 chars
                        print(f"✅ Found: {topic} ({len(content)} chars)")
                    else:
                        print(f"❌ Too short: {topic}")
                else:
                    print(f"❌ Not found: {topic}")
            except Exception as e:
                print(f"❌ Error with {topic}: {e}")
        
        print(f"\n📝 Successfully found {len(found_articles)} articles")
        
        # Save sample results
        output_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/physics/sample_extraction.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(found_articles, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Sample results saved to: {output_file}")
        
        # Show one example
        if found_articles:
            first_topic = list(found_articles.keys())[0]
            first_content = found_articles[first_topic]
            print(f"\n📄 Sample content from '{first_topic}':")
            print("-" * 40)
            print(first_content[:500] + "..." if len(first_content) > 500 else first_content)
            print("-" * 40)
        
        # Try to extract some basic physics principles
        physics_principles = []
        
        for topic, content in found_articles.items():
            # Look for basic equations
            if "F = ma" in content or "F=ma" in content:
                physics_principles.append({
                    'content': 'Force equals mass times acceleration',
                    'certainty': 0.9999,
                    'domain': 'physics',
                    'category': 'classical_mechanics',
                    'mathematical_form': 'F = m*a',
                    'source': f'Wikipedia: {topic}'
                })
            
            if "E = mc" in content:
                physics_principles.append({
                    'content': 'Energy equals mass times speed of light squared',
                    'certainty': 0.9999,
                    'domain': 'physics',
                    'category': 'relativity',
                    'mathematical_form': 'E = mc²',
                    'source': f'Wikipedia: {topic}'
                })
        
        if physics_principles:
            print(f"\n🔬 Extracted {len(physics_principles)} physics principles:")
            for principle in physics_principles:
                print(f"   - {principle['content']}")
                print(f"     Mathematical form: {principle['mathematical_form']}")
                print(f"     Certainty: {principle['certainty']}")
                print()
        
        print("🎉 Simple extraction test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_zim_extraction()