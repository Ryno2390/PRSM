#!/usr/bin/env python3
"""
Physics Knowledge Extraction from ZIM File
==========================================

This script extracts core physics knowledge from a Wikipedia ZIM file
and structures it for the NWTN WorldModelCore system.
"""

import os
import sys
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import zipfile
from zimply.zimply import ZIMFile

def extract_physics_articles(zim_path: str) -> Dict[str, str]:
    """
    Extract key physics articles from ZIM file
    
    Args:
        zim_path: Path to the physics ZIM file
        
    Returns:
        Dictionary mapping article titles to content
    """
    
    # Key physics topics to extract
    key_topics = [
        "Newton's laws of motion",
        "Conservation of energy",
        "Conservation of momentum",
        "Thermodynamics",
        "First law of thermodynamics",
        "Second law of thermodynamics",
        "Maxwell's equations",
        "Special relativity",
        "General relativity",
        "Quantum mechanics",
        "Schrödinger equation",
        "Uncertainty principle",
        "Speed of light",
        "Gravitational constant",
        "Planck constant",
        "Elementary charge",
        "Coulomb's law",
        "Ohm's law",
        "Faraday's law",
        "Lenz's law",
        "Hooke's law",
        "Ideal gas law",
        "Boyle's law",
        "Charles's law",
        "Wave equation",
        "Doppler effect",
        "Photoelectric effect",
        "Blackbody radiation",
        "Stefan-Boltzmann law",
        "Kirchhoff's laws",
        "Bernoulli's principle",
        "Archimedes' principle",
        "Pascal's principle",
        "Centripetal force",
        "Angular momentum",
        "Moment of inertia",
        "Torque",
        "Work (physics)",
        "Energy",
        "Power (physics)",
        "Pressure",
        "Temperature",
        "Heat",
        "Entropy",
        "Enthalpy",
        "Free energy",
        "Electromagnetic radiation",
        "Atomic structure",
        "Periodic table",
        "Radioactive decay",
        "Nuclear fission",
        "Nuclear fusion",
        "Standard Model"
    ]
    
    print(f"🔍 Opening ZIM file: {zim_path}")
    
    try:
        zim = ZIMFile(zim_path, 'utf-8')
        articles = {}
        
        print(f"📚 ZIM file contains {len(zim)} articles")
        
        # Search for each key topic
        for topic in key_topics:
            print(f"   Searching for: {topic}")
            
            # Try different variations of the topic name
            variations = [
                topic,
                topic.replace("'", "'"),  # Different apostrophe
                topic.replace(" ", "_"),   # Wikipedia URL format
                topic.lower(),
                topic.replace(" ", "_").lower(),
                topic.replace("'s", "s"),  # Possessive variations
            ]
            
            found = False
            for variation in variations:
                try:
                    if variation in zim:
                        content = zim[variation]
                        if content and len(content.strip()) > 100:  # Minimum content length
                            articles[topic] = content
                            print(f"     ✅ Found: {variation}")
                            found = True
                            break
                except Exception as e:
                    continue
            
            if not found:
                print(f"     ❌ Not found: {topic}")
        
        print(f"📝 Successfully extracted {len(articles)} articles")
        return articles
        
    except Exception as e:
        print(f"❌ Error reading ZIM file: {e}")
        return {}

def extract_physics_principles(content: str, title: str) -> List[Dict[str, Any]]:
    """
    Extract physics principles from article content
    
    Args:
        content: Wikipedia article content
        title: Article title
        
    Returns:
        List of extracted physics principles
    """
    
    principles = []
    
    # Clean up the content
    content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
    content = re.sub(r'\[\d+\]', '', content)  # Remove reference numbers
    content = re.sub(r'\s+', ' ', content)     # Normalize whitespace
    
    # Extract mathematical equations
    math_patterns = [
        r'([A-Za-z]+)\s*=\s*([^.]+?)(?:\.|$)',  # Basic equations like F = ma
        r'([A-Za-z]+)\s*∝\s*([^.]+?)(?:\.|$)',  # Proportionality
        r'([A-Za-z]+)\s*→\s*([^.]+?)(?:\.|$)',  # Transformations
    ]
    
    for pattern in math_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if len(match) == 2:
                principles.append({
                    'type': 'mathematical_relation',
                    'title': title,
                    'left_side': match[0].strip(),
                    'right_side': match[1].strip(),
                    'full_equation': f"{match[0].strip()} = {match[1].strip()}"
                })
    
    # Extract laws and principles
    law_patterns = [
        r'([A-Z][^.]*?law[^.]*?)\.(?:\s|$)',     # Laws
        r'([A-Z][^.]*?principle[^.]*?)\.(?:\s|$)', # Principles
        r'([A-Z][^.]*?theorem[^.]*?)\.(?:\s|$)',  # Theorems
        r'([A-Z][^.]*?constant[^.]*?)\.(?:\s|$)', # Constants
    ]
    
    for pattern in law_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if 20 < len(match) < 200:  # Reasonable length
                principles.append({
                    'type': 'law_or_principle',
                    'title': title,
                    'statement': match.strip(),
                    'source': 'wikipedia'
                })
    
    # Extract definitions
    definition_patterns = [
        r'([A-Z][^.]*?)\s+is\s+([^.]+?)\.(?:\s|$)',
        r'([A-Z][^.]*?)\s+refers to\s+([^.]+?)\.(?:\s|$)',
        r'([A-Z][^.]*?)\s+means\s+([^.]+?)\.(?:\s|$)',
    ]
    
    for pattern in definition_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if len(match) == 2 and 10 < len(match[1]) < 150:
                principles.append({
                    'type': 'definition',
                    'title': title,
                    'term': match[0].strip(),
                    'definition': match[1].strip(),
                    'source': 'wikipedia'
                })
    
    return principles

def create_knowledge_items(principles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert extracted principles to WorldModelCore KnowledgeItem format
    
    Args:
        principles: List of extracted principles
        
    Returns:
        List of KnowledgeItem dictionaries
    """
    
    knowledge_items = []
    
    for principle in principles:
        # Determine certainty based on type and content
        certainty = 0.95  # Default high certainty for physics
        
        if principle['type'] == 'mathematical_relation':
            certainty = 0.9999  # Very high for mathematical relations
        elif principle['type'] == 'law_or_principle':
            certainty = 0.999   # High for established laws
        elif principle['type'] == 'definition':
            certainty = 0.98    # Slightly lower for definitions
        
        # Create knowledge item
        if principle['type'] == 'mathematical_relation':
            content = f"{principle['left_side']} equals {principle['right_side']}"
            mathematical_form = principle['full_equation']
        elif principle['type'] == 'law_or_principle':
            content = principle['statement']
            mathematical_form = None
        elif principle['type'] == 'definition':
            content = f"{principle['term']}: {principle['definition']}"
            mathematical_form = None
        else:
            content = str(principle)
            mathematical_form = None
        
        # Skip if content is too short or too long
        if len(content) < 10 or len(content) > 300:
            continue
        
        knowledge_item = {
            'content': content,
            'certainty': certainty,
            'domain': 'physics',
            'category': 'extracted_knowledge',
            'mathematical_form': mathematical_form,
            'applicable_conditions': [],
            'references': [f"Wikipedia: {principle['title']}"],
            'source_type': principle['type'],
            'extraction_date': '2025-07-16'
        }
        
        knowledge_items.append(knowledge_item)
    
    return knowledge_items

def main():
    """Main extraction process"""
    
    print("🧠 Physics Knowledge Extraction from ZIM File")
    print("=" * 60)
    
    # Paths
    zim_path = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/raw_sources/zim_files/physics_wikipedia_no_pic.zim"
    output_dir = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/physics"
    
    # Check if ZIM file exists
    if not os.path.exists(zim_path):
        print(f"❌ ZIM file not found: {zim_path}")
        return
    
    # Extract articles
    print("\n1. Extracting physics articles...")
    articles = extract_physics_articles(zim_path)
    
    if not articles:
        print("❌ No articles extracted")
        return
    
    # Extract principles from articles
    print("\n2. Extracting physics principles...")
    all_principles = []
    
    for title, content in articles.items():
        print(f"   Processing: {title}")
        principles = extract_physics_principles(content, title)
        all_principles.extend(principles)
        print(f"     Extracted {len(principles)} principles")
    
    print(f"📊 Total principles extracted: {len(all_principles)}")
    
    # Convert to knowledge items
    print("\n3. Converting to knowledge items...")
    knowledge_items = create_knowledge_items(all_principles)
    
    print(f"📝 Created {len(knowledge_items)} knowledge items")
    
    # Save results
    print("\n4. Saving results...")
    
    # Save raw articles
    articles_file = os.path.join(output_dir, "extracted_articles.json")
    with open(articles_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    
    # Save extracted principles
    principles_file = os.path.join(output_dir, "extracted_principles.json")
    with open(principles_file, 'w', encoding='utf-8') as f:
        json.dump(all_principles, f, indent=2, ensure_ascii=False)
    
    # Save knowledge items
    knowledge_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/processed_knowledge/physics_knowledge_v1.json"
    with open(knowledge_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_items, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved:")
    print(f"   Articles: {articles_file}")
    print(f"   Principles: {principles_file}")
    print(f"   Knowledge items: {knowledge_file}")
    
    # Show sample knowledge items
    print("\n5. Sample extracted knowledge items:")
    for i, item in enumerate(knowledge_items[:5]):
        print(f"   {i+1}. {item['content'][:100]}...")
        print(f"      Certainty: {item['certainty']}")
        if item['mathematical_form']:
            print(f"      Mathematical form: {item['mathematical_form']}")
        print()
    
    print("🎉 Physics knowledge extraction complete!")

if __name__ == "__main__":
    main()