#!/usr/bin/env python3
"""
Top Tier Physics Knowledge Extraction
====================================

This script extracts the most essential physics topics (Top Tier) from the ZIM file
using the systematically identified high-importance, high-certainty topics.
"""

import os
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from zimply.zimply import ZIMFile

def load_top_tier_topics() -> List[str]:
    """Load the top tier physics topics list"""
    
    topics_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/physics/top_tier_topics.json"
    
    if os.path.exists(topics_file):
        with open(topics_file, 'r') as f:
            return json.load(f)
    else:
        # Fallback to hardcoded list if file doesn't exist
        return [
            "Newton's laws of motion", "Newtons laws of motion", "Newton's_laws_of_motion",
            "Conservation of energy", "Energy conservation", "Conservation_of_energy",
            "Conservation of momentum", "Momentum conservation", "Conservation_of_momentum",
            "Speed of light", "Speed_of_light", "Light speed",
            "First law of thermodynamics", "First_law_of_thermodynamics",
            "Second law of thermodynamics", "Second_law_of_thermodynamics",
            "Planck constant", "Planck's constant", "Planck_constant",
            "Force", "Force (physics)",
            "Energy", "Energy (physics)",
            "Coulomb's law", "Coulombs law", "Coulomb's_law",
            "Gravitational constant", "Gravitational_constant", "Newton's gravitational constant",
            "Momentum", "Linear momentum",
            "Mass", "Mass (physics)",
            "Gravity", "Gravitation", "Gravitational force",
            "Special relativity", "Special_relativity",
            "Quantum mechanics", "Quantum_mechanics",
            "Maxwell's equations", "Maxwells equations", "Maxwell's_equations"
        ]

def extract_targeted_articles(zim_path: str, target_topics: List[str]) -> Dict[str, str]:
    """
    Extract specific articles from ZIM file based on target topics
    
    Args:
        zim_path: Path to the ZIM file
        target_topics: List of topic variations to search for
        
    Returns:
        Dictionary mapping found topics to their content
    """
    
    print(f"🔍 Opening ZIM file: {zim_path}")
    
    try:
        zim = ZIMFile(zim_path, 'utf-8')
        
        print(f"📚 ZIM file contains {len(zim):,} articles")
        print(f"🎯 Searching for {len(target_topics)} target topics...")
        
        found_articles = {}
        search_progress = 0
        
        for topic in target_topics:
            search_progress += 1
            
            # Show progress
            if search_progress % 10 == 0 or search_progress == len(target_topics):
                print(f"   Progress: {search_progress}/{len(target_topics)} topics searched")
            
            try:
                if topic in zim:
                    content = zim[topic]
                    if content and len(content.strip()) > 200:  # Minimum meaningful content
                        found_articles[topic] = content
                        print(f"   ✅ Found: {topic} ({len(content):,} chars)")
                    else:
                        print(f"   ⚠️  Too short: {topic} ({len(content) if content else 0} chars)")
                else:
                    # Try some common variations
                    variations = [
                        topic.replace("'", "'"),  # Different apostrophe
                        topic.replace(" ", "_"),  # URL format
                        topic.replace("_", " "),  # Space format
                        topic.lower(),
                        topic.upper(),
                        topic.title()
                    ]
                    
                    found_variation = False
                    for variation in variations:
                        if variation in zim:
                            content = zim[variation]
                            if content and len(content.strip()) > 200:
                                found_articles[topic] = content
                                print(f"   ✅ Found via variation: {variation} ({len(content):,} chars)")
                                found_variation = True
                                break
                    
                    if not found_variation:
                        print(f"   ❌ Not found: {topic}")
                        
            except Exception as e:
                print(f"   ❌ Error with {topic}: {e}")
                continue
        
        print(f"\n📝 Successfully extracted {len(found_articles)} articles")
        return found_articles
        
    except Exception as e:
        print(f"❌ Error opening ZIM file: {e}")
        return {}

def extract_physics_knowledge(content: str, topic: str) -> List[Dict[str, Any]]:
    """
    Extract structured physics knowledge from article content
    
    Args:
        content: Article content
        topic: Topic name
        
    Returns:
        List of extracted knowledge items
    """
    
    knowledge_items = []
    
    # Clean content
    content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
    content = re.sub(r'\[\d+\]', '', content)  # Remove reference numbers
    content = re.sub(r'\s+', ' ', content)     # Normalize whitespace
    content = content.strip()
    
    # Extract mathematical equations and relationships
    equation_patterns = [
        r'([A-Za-z]+)\s*=\s*([^.;,\n]+?)(?:[\.\;\,\n]|$)',  # Basic equations
        r'([A-Za-z]+)\s*∝\s*([^.;,\n]+?)(?:[\.\;\,\n]|$)',  # Proportional relationships
        r'([A-Za-z]+)\s*→\s*([^.;,\n]+?)(?:[\.\;\,\n]|$)',  # Transformations
        r'([A-Za-z][^=]*?)\s*=\s*([^.;,\n]+?)(?:[\.\;\,\n]|$)',  # Complex left side
    ]
    
    for pattern in equation_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if len(match) == 2:
                left, right = match[0].strip(), match[1].strip()
                
                # Filter for reasonable equations
                if (2 < len(left) < 50 and 2 < len(right) < 100 and
                    not any(word in left.lower() for word in ['the', 'this', 'that', 'where', 'when']) and
                    not any(word in right.lower() for word in ['the', 'this', 'that', 'where', 'when'])):
                    
                    knowledge_items.append({
                        'type': 'mathematical_equation',
                        'content': f"{left} equals {right}",
                        'mathematical_form': f"{left} = {right}",
                        'source_topic': topic,
                        'certainty': 0.999 if 'law' in topic.lower() else 0.99
                    })
    
    # Extract physics laws and principles
    law_patterns = [
        r'([A-Z][^.]*?(?:law|principle|theorem|constant|equation)[^.]*?)\.(?:\s|$)',
        r'([A-Z][^.]*?states that[^.]*?)\.(?:\s|$)',
        r'([A-Z][^.]*?is defined as[^.]*?)\.(?:\s|$)',
        r'([A-Z][^.]*?can be written as[^.]*?)\.(?:\s|$)',
    ]
    
    for pattern in law_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            statement = match.strip()
            
            # Filter for reasonable length and content
            if (30 < len(statement) < 300 and 
                not statement.startswith('The following') and
                not statement.startswith('See also') and
                'wikipedia' not in statement.lower()):
                
                knowledge_items.append({
                    'type': 'physics_principle',
                    'content': statement,
                    'source_topic': topic,
                    'certainty': 0.999 if 'law' in topic.lower() else 0.98
                })
    
    # Extract key definitions
    definition_patterns = [
        r'([A-Z][^.]*?)\s+is\s+([^.]+?)\.(?:\s|$)',
        r'([A-Z][^.]*?)\s+refers to\s+([^.]+?)\.(?:\s|$)',
        r'([A-Z][^.]*?)\s+means\s+([^.]+?)\.(?:\s|$)',
        r'([A-Z][^.]*?)\s+represents\s+([^.]+?)\.(?:\s|$)',
    ]
    
    for pattern in definition_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if len(match) == 2:
                term, definition = match[0].strip(), match[1].strip()
                
                # Filter for reasonable definitions
                if (5 < len(term) < 100 and 10 < len(definition) < 200 and
                    not any(word in term.lower() for word in ['this', 'that', 'these', 'those']) and
                    not any(word in definition.lower() for word in ['this article', 'wikipedia', 'see also'])):
                    
                    knowledge_items.append({
                        'type': 'definition',
                        'content': f"{term}: {definition}",
                        'term': term,
                        'definition': definition,
                        'source_topic': topic,
                        'certainty': 0.98
                    })
    
    # Extract physical constants
    constant_patterns = [
        r'([A-Za-z\s]+)\s*=\s*([0-9]+\.?[0-9]*\s*×\s*10[⁻\-]?[0-9]+|[0-9]+\.?[0-9]*)\s*([A-Za-z/\s\-²³⁴⁵⁶⁷⁸⁹⁰]+)',
        r'([A-Za-z\s]+)\s*≈\s*([0-9]+\.?[0-9]*\s*×\s*10[⁻\-]?[0-9]+|[0-9]+\.?[0-9]*)\s*([A-Za-z/\s\-²³⁴⁵⁶⁷⁸⁹⁰]+)',
    ]
    
    for pattern in constant_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if len(match) == 3:
                name, value, unit = match[0].strip(), match[1].strip(), match[2].strip()
                
                if (3 < len(name) < 50 and len(value) > 0 and len(unit) > 0):
                    knowledge_items.append({
                        'type': 'physical_constant',
                        'content': f"{name} = {value} {unit}",
                        'constant_name': name,
                        'value': value,
                        'unit': unit,
                        'source_topic': topic,
                        'certainty': 0.9999
                    })
    
    return knowledge_items

def create_world_model_items(knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert extracted knowledge to WorldModelCore format
    
    Args:
        knowledge_items: List of extracted knowledge items
        
    Returns:
        List of WorldModelCore KnowledgeItem dictionaries
    """
    
    world_model_items = []
    
    for item in knowledge_items:
        # Determine category based on source topic
        category = "physics_general"
        if any(word in item['source_topic'].lower() for word in ['newton', 'force', 'momentum', 'mass']):
            category = "classical_mechanics"
        elif any(word in item['source_topic'].lower() for word in ['energy', 'thermodynamics', 'heat', 'temperature']):
            category = "thermodynamics"
        elif any(word in item['source_topic'].lower() for word in ['electric', 'magnetic', 'maxwell', 'coulomb']):
            category = "electromagnetism"
        elif any(word in item['source_topic'].lower() for word in ['quantum', 'planck', 'uncertainty']):
            category = "quantum_physics"
        elif any(word in item['source_topic'].lower() for word in ['relativity', 'light', 'speed']):
            category = "relativity"
        elif any(word in item['source_topic'].lower() for word in ['constant', 'gravitational']):
            category = "fundamental_constants"
        
        # Create knowledge item
        world_model_item = {
            'content': item['content'],
            'certainty': item['certainty'],
            'domain': 'physics',
            'category': category,
            'dependencies': [],
            'references': [f"Wikipedia: {item['source_topic']}"],
            'applicable_conditions': [],
            'extraction_date': '2025-07-16',
            'extraction_method': 'targeted_zim_extraction',
            'source_type': item['type']
        }
        
        # Add type-specific fields
        if item['type'] == 'mathematical_equation':
            world_model_item['mathematical_form'] = item.get('mathematical_form', None)
        elif item['type'] == 'physical_constant':
            world_model_item['mathematical_form'] = f"{item.get('constant_name', '')} = {item.get('value', '')} {item.get('unit', '')}"
            world_model_item['constant_value'] = item.get('value', '')
            world_model_item['unit'] = item.get('unit', '')
        else:
            world_model_item['mathematical_form'] = None
        
        world_model_items.append(world_model_item)
    
    return world_model_items

def main():
    """Main extraction process"""
    
    print("🚀 Top Tier Physics Knowledge Extraction")
    print("=" * 60)
    
    # Paths
    zim_path = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/raw_sources/zim_files/physics_wikipedia_no_pic.zim"
    output_dir = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/physics"
    
    # Check ZIM file
    if not os.path.exists(zim_path):
        print(f"❌ ZIM file not found: {zim_path}")
        return
    
    # Load target topics
    print("📋 Loading top tier topics...")
    target_topics = load_top_tier_topics()
    print(f"🎯 Target topics: {len(target_topics)}")
    
    # Extract articles
    print(f"\n1. Extracting articles from ZIM file...")
    start_time = time.time()
    
    articles = extract_targeted_articles(zim_path, target_topics)
    
    extraction_time = time.time() - start_time
    print(f"⏱️  Extraction completed in {extraction_time:.1f} seconds")
    
    if not articles:
        print("❌ No articles extracted")
        return
    
    # Process articles to extract knowledge
    print(f"\n2. Processing articles for knowledge extraction...")
    all_knowledge_items = []
    
    for topic, content in articles.items():
        print(f"   Processing: {topic}")
        
        knowledge_items = extract_physics_knowledge(content, topic)
        all_knowledge_items.extend(knowledge_items)
        
        print(f"     Extracted {len(knowledge_items)} knowledge items")
    
    print(f"📊 Total knowledge items extracted: {len(all_knowledge_items)}")
    
    # Convert to WorldModelCore format
    print(f"\n3. Converting to WorldModelCore format...")
    world_model_items = create_world_model_items(all_knowledge_items)
    
    # Remove duplicates (simple content-based)
    unique_items = []
    seen_content = set()
    
    for item in world_model_items:
        if item['content'] not in seen_content:
            unique_items.append(item)
            seen_content.add(item['content'])
    
    print(f"📝 Unique knowledge items: {len(unique_items)}")
    
    # Save results
    print(f"\n4. Saving results...")
    
    # Save extracted articles
    articles_file = os.path.join(output_dir, "top_tier_articles.json")
    with open(articles_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    
    # Save raw knowledge items
    raw_knowledge_file = os.path.join(output_dir, "top_tier_raw_knowledge.json")
    with open(raw_knowledge_file, 'w', encoding='utf-8') as f:
        json.dump(all_knowledge_items, f, indent=2, ensure_ascii=False)
    
    # Save WorldModelCore items
    world_model_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/processed_knowledge/physics_top_tier_v1.json"
    with open(world_model_file, 'w', encoding='utf-8') as f:
        json.dump(unique_items, f, indent=2, ensure_ascii=False)
    
    # Create summary report
    summary = {
        'extraction_date': '2025-07-16',
        'extraction_time_seconds': extraction_time,
        'target_topics_searched': len(target_topics),
        'articles_found': len(articles),
        'raw_knowledge_items': len(all_knowledge_items),
        'unique_knowledge_items': len(unique_items),
        'knowledge_by_type': {},
        'knowledge_by_category': {},
        'average_certainty': sum(item['certainty'] for item in unique_items) / len(unique_items) if unique_items else 0
    }
    
    # Count by type and category
    for item in unique_items:
        item_type = item['source_type']
        category = item['category']
        
        summary['knowledge_by_type'][item_type] = summary['knowledge_by_type'].get(item_type, 0) + 1
        summary['knowledge_by_category'][category] = summary['knowledge_by_category'].get(category, 0) + 1
    
    summary_file = os.path.join(output_dir, "top_tier_extraction_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Files saved:")
    print(f"   📄 Articles: {articles_file}")
    print(f"   🔬 Raw knowledge: {raw_knowledge_file}")
    print(f"   🧠 WorldModel items: {world_model_file}")
    print(f"   📊 Summary: {summary_file}")
    
    # Show extraction summary
    print(f"\n5. Extraction Summary:")
    print(f"   🎯 Topics searched: {len(target_topics)}")
    print(f"   📄 Articles found: {len(articles)}")
    print(f"   🔬 Knowledge items: {len(all_knowledge_items)}")
    print(f"   🧠 Unique items: {len(unique_items)}")
    print(f"   ⏱️  Processing time: {extraction_time:.1f} seconds")
    print(f"   📊 Average certainty: {summary['average_certainty']:.3f}")
    
    print(f"\n   Knowledge by type:")
    for item_type, count in summary['knowledge_by_type'].items():
        print(f"     {item_type}: {count}")
    
    print(f"\n   Knowledge by category:")
    for category, count in summary['knowledge_by_category'].items():
        print(f"     {category}: {count}")
    
    # Show sample items
    print(f"\n6. Sample Knowledge Items:")
    for i, item in enumerate(unique_items[:5]):
        print(f"   {i+1}. {item['content'][:100]}...")
        print(f"      Category: {item['category']}")
        print(f"      Certainty: {item['certainty']}")
        if item['mathematical_form']:
            print(f"      Mathematical form: {item['mathematical_form']}")
        print()
    
    print(f"🎉 Top Tier Physics Knowledge Extraction Complete!")
    print(f"💡 WorldModelCore expanded from 28 to {28 + len(unique_items)} items!")

if __name__ == "__main__":
    main()