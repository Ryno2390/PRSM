#!/usr/bin/env python3
"""
Simple Physics Integration Test
==============================

Test just the WorldModelCore physics integration without the full MetaReasoningEngine
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

# Suppress warnings
logging.getLogger().setLevel(logging.ERROR)

def test_worldmodel_loading():
    """Test WorldModelCore loading directly"""
    
    print("🧪 Testing WorldModelCore Physics Loading")
    print("=" * 50)
    
    try:
        # Import just the classes we need
        from prsm.nwtn.meta_reasoning_engine import WorldModelCore, KnowledgeItem
        
        # Initialize WorldModelCore directly
        world_model = WorldModelCore()
        
        # Check physics laws loading
        physics_laws = world_model.physical_laws
        
        print(f"✅ Physics laws loaded: {len(physics_laws)} items")
        
        # Show sample physics items
        print(f"\n📚 Sample Physics Knowledge:")
        for i, (key, item) in enumerate(list(physics_laws.items())[:3]):
            print(f"   {i+1}. {key}")
            print(f"      Content: {item.content}")
            print(f"      Certainty: {item.certainty}")
            print(f"      Math form: {item.mathematical_form}")
            print(f"      Category: {item.category}")
            print()
        
        # Test knowledge search
        print(f"📊 Testing Knowledge Search:")
        
        # Search for Newton's laws
        newton_items = world_model.search_knowledge("newton")
        print(f"   Newton-related items: {len(newton_items)}")
        
        # Search for conservation laws
        conservation_items = world_model.search_knowledge("conservation")
        print(f"   Conservation-related items: {len(conservation_items)}")
        
        # Test total knowledge index
        total_items = len(world_model.knowledge_index)
        print(f"\n📈 Total knowledge items: {total_items}")
        
        # Check if our extracted physics knowledge is present
        expected_physics_items = [
            "newtons_first_law_law_of_inertia",
            "newtons_second_law", 
            "conservation_of_energy",
            "conservation_of_momentum",
            "speed_of_light"
        ]
        
        found_items = []
        for expected in expected_physics_items:
            if expected in physics_laws:
                found_items.append(expected)
        
        print(f"\n🔍 Expected physics items found: {len(found_items)}/{len(expected_physics_items)}")
        for item in found_items:
            print(f"   ✅ {item}")
        
        missing = set(expected_physics_items) - set(found_items)
        if missing:
            print(f"\n❌ Missing expected items:")
            for item in missing:
                print(f"   - {item}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading WorldModelCore: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_physics_validation():
    """Test physics knowledge validation"""
    
    print("\n🔍 Testing Physics Knowledge Validation")
    print("=" * 50)
    
    try:
        from prsm.nwtn.meta_reasoning_engine import WorldModelCore
        
        # Initialize WorldModelCore
        world_model = WorldModelCore()
        
        # Test validation system
        validator = world_model.validator
        
        # Test by creating a mock reasoning result
        from prsm.nwtn.meta_reasoning_engine import ReasoningResult
        
        # Create a mock reasoning result for testing
        mock_result = ReasoningResult(
            reasoning_type='deductive',
            conclusion='Energy can be created from nothing',
            confidence=0.8,
            reasoning_trace='This contradicts conservation of energy',
            evidence=[],
            assumptions=[],
            alternative_conclusions=[]
        )
        
        print(f"Testing reasoning result: '{mock_result.conclusion}'")
        
        # Test validation
        validation_result = validator.validate_reasoning(mock_result)
        
        print(f"Validation result: {validation_result.is_valid}")
        print(f"Conflicts detected: {len(validation_result.conflicts)}")
        print(f"Supporting knowledge count: {len(validation_result.supporting_knowledge)}")
        
        if validation_result.conflicts:
            print("Conflict details:")
            for i, conflict in enumerate(validation_result.conflicts[:2]):  # Show first 2
                print(f"   {i+1}. Type: {conflict.get('type', 'N/A')}")
                print(f"      Severity: {conflict.get('severity', 'N/A')}")
        
        # Test supporting statement
        mock_result2 = ReasoningResult(
            reasoning_type='deductive',
            conclusion='Force equals mass times acceleration',
            confidence=0.9,
            reasoning_trace='This is Newton\'s second law',
            evidence=[],
            assumptions=[],
            alternative_conclusions=[]
        )
        
        validation_result2 = validator.validate_reasoning(mock_result2)
        
        print(f"\nTesting reasoning result: '{mock_result2.conclusion}'")
        print(f"Validation result: {validation_result2.is_valid}")
        print(f"Supporting knowledge count: {len(validation_result2.supporting_knowledge)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_file_direct():
    """Test if the physics JSON file can be loaded directly"""
    
    print("\n📁 Testing Direct JSON File Loading")
    print("=" * 50)
    
    physics_file = "/Volumes/My Passport/PRSM_Storage/WorldModel_Knowledge/processed_knowledge/physics_essential_manual_v1.json"
    
    try:
        with open(physics_file, 'r', encoding='utf-8') as f:
            physics_data = json.load(f)
        
        print(f"✅ Physics file loaded successfully")
        print(f"   Items: {len(physics_data)}")
        
        # Show first few items
        print(f"\n📄 Sample Items:")
        for i, item in enumerate(physics_data[:3]):
            print(f"   {i+1}. {item.get('principle_name', 'N/A')}")
            print(f"      Content: {item['content'][:60]}...")
            print(f"      Certainty: {item['certainty']}")
            print(f"      Category: {item['category']}")
            print()
        
        # Check categories
        categories = {}
        for item in physics_data:
            cat = item['category']
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        print(f"📊 Categories found:")
        for cat, count in categories.items():
            print(f"   {cat}: {count} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simplified physics integration tests"""
    
    print("🚀 Simple Physics Integration Test")
    print("=" * 60)
    
    tests = [
        ("JSON File Loading", test_json_file_direct),
        ("WorldModel Loading", test_worldmodel_loading),
        ("Physics Validation", test_physics_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 Test Results Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic physics integration successful!")
        return True
    else:
        print("⚠️  Some tests failed. Check integration issues.")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Ready to proceed with full integration!")
    else:
        print("\n❌ Basic integration needs fixes.")