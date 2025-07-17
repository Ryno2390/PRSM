#!/usr/bin/env python3
"""
Simple Mathematics Knowledge Test
=================================

Quick test to verify mathematics knowledge integration works at basic level
"""

import sys
import os
import json
import logging

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

# Suppress warnings for cleaner output
logging.getLogger().setLevel(logging.ERROR)

def test_mathematics_knowledge_loading():
    """Test that mathematics knowledge loads correctly"""
    
    print("🧮 Simple Mathematics Knowledge Integration Test")
    print("=" * 60)
    
    try:
        # Import the WorldModelCore directly
        from prsm.nwtn.meta_reasoning_engine import WorldModelCore
        
        print("✅ Successfully imported WorldModelCore")
        
        # Initialize the WorldModelCore
        print("\n🔧 Initializing WorldModelCore...")
        world_model = WorldModelCore()
        
        print("✅ WorldModelCore initialized successfully")
        
        # Check knowledge counts
        total_knowledge = len(world_model.knowledge_index)
        physics_count = len(world_model.physical_laws)
        math_count = len(world_model.mathematical_truths)
        
        print(f"\n📊 Knowledge Statistics:")
        print(f"   - Total Knowledge Items: {total_knowledge}")
        print(f"   - Physics Laws: {physics_count}")
        print(f"   - Mathematical Truths: {math_count}")
        
        # Show some mathematics knowledge items
        print(f"\n🔍 Sample Mathematics Knowledge:")
        math_items = list(world_model.mathematical_truths.items())[:5]
        for key, item in math_items:
            print(f"   - {key}: {item.content}")
            print(f"     Mathematical form: {item.mathematical_form}")
            print(f"     Certainty: {item.certainty}")
            print(f"     Category: {item.category}")
            print()
        
        # Test mathematics knowledge searching
        print(f"🔍 Testing Mathematics Knowledge Search:")
        
        # Test searching for mathematical knowledge
        math_query = "area of circle"
        supporting_knowledge = world_model.get_supporting_knowledge(math_query)
        
        print(f"   Query: {math_query}")
        print(f"   Supporting Knowledge Found: {len(supporting_knowledge)}")
        
        for knowledge in supporting_knowledge[:3]:  # Show first 3 results
            print(f"      - {knowledge.content}")
            print(f"        Certainty: {knowledge.certainty}")
            print(f"        Mathematical form: {knowledge.mathematical_form}")
        
        # Test another search
        math_query2 = "commutative property"
        supporting_knowledge2 = world_model.get_supporting_knowledge(math_query2)
        
        print(f"\n   Query: {math_query2}")
        print(f"   Supporting Knowledge Found: {len(supporting_knowledge2)}")
        
        for knowledge in supporting_knowledge2[:2]:  # Show first 2 results
            print(f"      - {knowledge.content}")
            print(f"        Certainty: {knowledge.certainty}")
            print(f"        Mathematical form: {knowledge.mathematical_form}")
        
        # Test combined physics and mathematics
        print(f"\n🔬 Testing Combined Physics + Mathematics Knowledge:")
        
        # Show physics knowledge count
        physics_items = list(world_model.physical_laws.items())[:3]
        for key, item in physics_items:
            print(f"   - {key}: {item.content}")
            print(f"     Mathematical form: {item.mathematical_form}")
            print(f"     Certainty: {item.certainty}")
            print()
        
        # Test validation success
        total_tests = 4
        passed_tests = 0
        
        if total_knowledge > 45:  # Should have physics + math
            passed_tests += 1
            print("✅ Combined knowledge count test passed")
        else:
            print("❌ Combined knowledge count test failed")
            
        if math_count >= 20:  # Should have substantial mathematics
            passed_tests += 1
            print("✅ Mathematics knowledge count test passed")
        else:
            print("❌ Mathematics knowledge count test failed")
            
        if len(supporting_knowledge) > 0:
            passed_tests += 1
            print("✅ Mathematics search test passed")
        else:
            print("❌ Mathematics search test failed")
            
        if len(supporting_knowledge2) > 0:
            passed_tests += 1
            print("✅ Algebra search test passed")
        else:
            print("❌ Algebra search test failed")
        
        success_rate = passed_tests / total_tests
        print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1%})")
        
        if success_rate >= 0.75:
            print("🎉 MATHEMATICS INTEGRATION SUCCESSFUL!")
            print("✅ Mathematics knowledge successfully integrated with physics")
            print("🚀 WorldModelCore ready for full meta-reasoning integration")
            return True
        else:
            print("❌ Mathematics integration needs improvement")
            return False
        
    except Exception as e:
        print(f"❌ Error during mathematics integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mathematics_knowledge_loading()
    
    if success:
        print("\n🎉 SIMPLE MATHEMATICS TEST PASSED!")
        print("✅ Mathematics knowledge integration verified")
        print("🔧 Ready to proceed with full system testing")
    else:
        print("\n❌ Mathematics integration test failed")
        print("🔧 Check WorldModelCore mathematics loading")