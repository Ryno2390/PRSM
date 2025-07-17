#!/usr/bin/env python3
"""
Efficient ZIM Extraction with Minimal Operations
===============================================

This script uses a more efficient approach to extract knowledge from ZIM files
by minimizing expensive operations and using caching.
"""

import os
import json
import time
import re
from typing import Dict, List, Optional, Tuple
from zimply.zimply import ZIMFile

class EfficientZIMExtractor:
    """Efficient ZIM file knowledge extractor"""
    
    def __init__(self, zim_path: str):
        self.zim_path = zim_path
        self.zim = None
        self.cache = {}
        
    def __enter__(self):
        print(f"🔍 Opening ZIM file: {self.zim_path}")
        self.zim = ZIMFile(self.zim_path, 'utf-8')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.zim:
            # Clean up if needed
            pass
    
    def extract_specific_topics(self, topics: List[str], max_articles: int = 20) -> Dict[str, str]:
        """
        Extract specific topics with early termination
        
        Args:
            topics: List of topics to find
            max_articles: Maximum articles to extract (for speed)
            
        Returns:
            Dictionary of found articles
        """
        
        found_articles = {}
        topics_found = 0
        
        print(f"🎯 Extracting maximum {max_articles} articles from {len(topics)} topics...")
        
        for i, topic in enumerate(topics):
            if topics_found >= max_articles:
                print(f"   ⏹️  Early termination: reached {max_articles} articles")
                break
            
            # Progress every 10 topics
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(topics)} topics, {topics_found} articles found")
            
            try:
                # Try direct lookup with minimal variations
                content = self._try_topic_variations(topic)
                
                if content:
                    found_articles[topic] = content
                    topics_found += 1
                    print(f"   ✅ Found: {topic} ({len(content):,} chars)")
                
            except Exception as e:
                continue
        
        print(f"📝 Extraction complete: {len(found_articles)} articles found")
        return found_articles
    
    def _try_topic_variations(self, topic: str) -> Optional[str]:
        """Try a limited set of topic variations efficiently"""
        
        # Only try the most common variations
        variations = [
            topic,                                    # Original
            topic.replace(" ", "_"),                  # Space to underscore
            topic.replace("'", "'"),                  # Apostrophe variation
            topic.lower(),                            # Lowercase
            topic.replace(" ", "_").lower(),          # URL format
        ]
        
        for variation in variations:
            try:
                if variation in self.zim:
                    content = self.zim[variation]
                    if content and len(content.strip()) > 200:
                        return content
            except:
                continue
        
        return None
    
    def extract_knowledge_from_content(self, content: str, topic: str) -> List[Dict[str, any]]:
        """
        Extract knowledge items from content efficiently
        
        Args:
            content: Article content
            topic: Topic name
            
        Returns:
            List of knowledge items
        """
        
        knowledge_items = []
        
        # Clean content quickly
        clean_content = re.sub(r'<[^>]+>', '', content)  # Remove HTML
        clean_content = re.sub(r'\[\d+\]', '', clean_content)  # Remove references
        clean_content = re.sub(r'\s+', ' ', clean_content)  # Normalize whitespace
        clean_content = clean_content.strip()
        
        # Extract only the most important patterns
        
        # 1. Simple equations (most valuable)
        equation_pattern = r'([A-Za-z]+)\s*=\s*([A-Za-z0-9\s\*\+\-\/\(\)]+?)(?:[\.\;\,\n]|$)'
        equations = re.findall(equation_pattern, clean_content)
        
        for eq in equations[:3]:  # Limit to 3 equations per article
            if len(eq) == 2 and len(eq[0]) < 20 and len(eq[1]) < 50:
                knowledge_items.append({
                    'type': 'equation',
                    'content': f"{eq[0]} = {eq[1]}",
                    'mathematical_form': f"{eq[0]} = {eq[1]}",
                    'topic': topic,
                    'certainty': 0.99
                })
        
        # 2. Key statements (laws, principles)
        law_pattern = r'([A-Z][^.]*?(?:law|principle|states that)[^.]*?)\.(?:\s|$)'
        laws = re.findall(law_pattern, clean_content)
        
        for law in laws[:2]:  # Limit to 2 laws per article
            if 20 < len(law) < 200:
                knowledge_items.append({
                    'type': 'principle',
                    'content': law.strip(),
                    'topic': topic,
                    'certainty': 0.98
                })
        
        # 3. Simple definitions
        def_pattern = r'([A-Z][^.]*?)\s+is\s+([^.]+?)\.(?:\s|$)'
        definitions = re.findall(def_pattern, clean_content)
        
        for definition in definitions[:2]:  # Limit to 2 definitions per article
            if len(definition) == 2 and 5 < len(definition[0]) < 50 and 10 < len(definition[1]) < 150:
                knowledge_items.append({
                    'type': 'definition',
                    'content': f"{definition[0]}: {definition[1]}",
                    'topic': topic,
                    'certainty': 0.95
                })
        
        return knowledge_items

def quick_physics_extraction():
    """Quick physics knowledge extraction with minimal processing"""
    
    print("⚡ Quick Physics Knowledge Extraction")
    print("=" * 50)
    
    zim_path = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/raw_sources/zim_files/physics_wikipedia_no_pic.zim"
    
    if not os.path.exists(zim_path):
        print(f"❌ ZIM file not found: {zim_path}")
        return
    
    # Load high-priority topics
    priority_topics = [
        "Force",
        "Energy", 
        "Mass",
        "Momentum",
        "Acceleration",
        "Velocity",
        "Gravity",
        "Newton's laws of motion",
        "Conservation of energy",
        "Conservation of momentum",
        "Speed of light",
        "Planck constant",
        "Gravitational constant",
        "First law of thermodynamics",
        "Second law of thermodynamics",
        "Coulomb's law",
        "Ohm's law",
        "Hooke's law",
        "Ideal gas law",
        "Maxwell's equations"
    ]
    
    print(f"🎯 Target: {len(priority_topics)} priority topics")
    
    start_time = time.time()
    
    try:
        with EfficientZIMExtractor(zim_path) as extractor:
            # Extract articles (limit to 15 for speed)
            articles = extractor.extract_specific_topics(priority_topics, max_articles=15)
            
            if not articles:
                print("❌ No articles found")
                return
            
            # Extract knowledge from found articles
            all_knowledge = []
            
            print(f"\n🔬 Processing {len(articles)} articles for knowledge extraction...")
            
            for topic, content in articles.items():
                knowledge_items = extractor.extract_knowledge_from_content(content, topic)
                all_knowledge.extend(knowledge_items)
                print(f"   {topic}: {len(knowledge_items)} knowledge items")
            
            end_time = time.time()
            
            print(f"\n📊 Extraction Results:")
            print(f"   Articles found: {len(articles)}")
            print(f"   Knowledge items: {len(all_knowledge)}")
            print(f"   Processing time: {end_time - start_time:.1f} seconds")
            print(f"   Speed: {len(articles)/(end_time - start_time):.1f} articles/second")
            
            # Convert to WorldModelCore format
            world_model_items = []
            for item in all_knowledge:
                world_model_item = {
                    'content': item['content'],
                    'certainty': item['certainty'],
                    'domain': 'physics',
                    'category': 'extracted_knowledge',
                    'mathematical_form': item.get('mathematical_form', None),
                    'references': [f"Wikipedia: {item['topic']}"],
                    'extraction_date': '2025-07-16',
                    'extraction_method': 'efficient_zim_extraction',
                    'source_type': item['type']
                }
                world_model_items.append(world_model_item)
            
            # Save results
            output_dir = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/physics"
            
            # Save articles
            articles_file = os.path.join(output_dir, "quick_extraction_articles.json")
            with open(articles_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            
            # Save knowledge items
            knowledge_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/processed_knowledge/physics_quick_extraction_v1.json"
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(world_model_items, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved:")
            print(f"   Articles: {articles_file}")
            print(f"   Knowledge: {knowledge_file}")
            
            # Show samples
            print(f"\n📄 Sample Knowledge Items:")
            for i, item in enumerate(world_model_items[:5]):
                print(f"   {i+1}. {item['content']}")
                print(f"      Certainty: {item['certainty']}, Type: {item['source_type']}")
                if item['mathematical_form']:
                    print(f"      Mathematical form: {item['mathematical_form']}")
                print()
            
            print(f"🎉 Quick extraction complete!")
            print(f"💡 WorldModelCore can be expanded with {len(world_model_items)} new physics items!")
            
            return True
            
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_physics_extraction()
    
    if success:
        print("\n✅ Efficient extraction method successful!")
    else:
        print("\n❌ Extraction failed.")