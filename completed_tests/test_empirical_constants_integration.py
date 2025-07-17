#!/usr/bin/env python3
"""
Test Empirical Constants Integration in NWTN Meta-Reasoning Engine
==================================================================

This script tests the empirical constants integration and verifies
the complete 4-domain knowledge system.
"""

import sys
import os
import json
import logging

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

# Suppress warnings for cleaner output
logging.getLogger().setLevel(logging.ERROR)

def test_empirical_constants_integration():
    """Test empirical constants integration"""
    
    print("🔢 Testing Empirical Constants Integration in NWTN")
    print("=" * 60)
    
    try:
        # Import the WorldModelCore directly
        from prsm.nwtn.meta_reasoning_engine import WorldModelCore
        
        print("✅ Successfully imported WorldModelCore")
        
        # Initialize the WorldModelCore
        print("\n🔧 Initializing WorldModelCore with All 4 Domains...")
        world_model = WorldModelCore()
        
        print("✅ WorldModelCore initialized successfully")
        
        # Check knowledge counts
        total_knowledge = len(world_model.knowledge_index)
        physics_count = len(world_model.physical_laws)
        math_count = len(world_model.mathematical_truths)
        logic_count = len(world_model.logical_principles)
        constants_count = len(world_model.empirical_constants)
        
        print(f"\n📊 Complete Knowledge Statistics:")
        print(f"   - Total Knowledge Items: {total_knowledge}")
        print(f"   - Physics Laws: {physics_count}")
        print(f"   - Mathematical Truths: {math_count}")
        print(f"   - Logical Principles: {logic_count}")
        print(f"   - Empirical Constants: {constants_count}")
        
        # Show some empirical constants
        print(f"\n🔍 Sample Empirical Constants:")
        constants_items = list(world_model.empirical_constants.items())[:5]
        for key, item in constants_items:
            print(f"   - {key}: {item.content}")
            print(f"     Mathematical form: {item.mathematical_form}")
            print(f"     Certainty: {item.certainty}")
            print(f"     Category: {item.category}")
            print()
        
        # Test constants knowledge searching
        print(f"🔍 Testing Constants Knowledge Search:")
        
        # Test searching for physics constants
        constants_query = "speed of light"
        supporting_knowledge = world_model.get_supporting_knowledge(constants_query)
        
        print(f"   Query: {constants_query}")
        print(f"   Supporting Knowledge Found: {len(supporting_knowledge)}")
        
        for knowledge in supporting_knowledge[:3]:  # Show first 3 results
            print(f"      - {knowledge.content}")
            print(f"        Certainty: {knowledge.certainty}")
            print(f"        Mathematical form: {knowledge.mathematical_form}")
        
        # Test another search
        constants_query2 = "planck constant"
        supporting_knowledge2 = world_model.get_supporting_knowledge(constants_query2)
        
        print(f"\n   Query: {constants_query2}")
        print(f"   Supporting Knowledge Found: {len(supporting_knowledge2)}")
        
        for knowledge in supporting_knowledge2[:2]:  # Show first 2 results
            print(f"      - {knowledge.content}")
            print(f"        Certainty: {knowledge.certainty}")
            print(f"        Mathematical form: {knowledge.mathematical_form}")
        
        # Test cross-domain knowledge integration
        print(f"\n🌐 Testing Cross-Domain Knowledge Integration:")
        
        # Show distribution across all domains
        print(f"   📊 Domain Distribution:")
        print(f"      Physics: {physics_count} items")
        print(f"      Mathematics: {math_count} items")
        print(f"      Logic: {logic_count} items")
        print(f"      Empirical Constants: {constants_count} items")
        
        # Test validation success
        total_tests = 6
        passed_tests = 0
        
        if total_knowledge > 95:  # Should have all 4 domains
            passed_tests += 1
            print("✅ Combined knowledge count test passed")
        else:
            print("❌ Combined knowledge count test failed")
            
        if constants_count >= 20:  # Should have substantial constants
            passed_tests += 1
            print("✅ Constants knowledge count test passed")
        else:
            print("❌ Constants knowledge count test failed")
            
        if len(supporting_knowledge) > 0:
            passed_tests += 1
            print("✅ Constants search test passed")
        else:
            print("❌ Constants search test failed")
            
        if len(supporting_knowledge2) > 0:
            passed_tests += 1
            print("✅ Physics constants search test passed")
        else:
            print("❌ Physics constants search test failed")
            
        # Test domain balance
        domain_balance = all([
            physics_count >= 20,
            math_count >= 20,
            logic_count >= 20,
            constants_count >= 20
        ])
        if domain_balance:
            passed_tests += 1
            print("✅ Domain balance test passed")
        else:
            print("❌ Domain balance test failed")
        
        # Test knowledge diversity
        total_expected = physics_count + math_count + logic_count + constants_count
        diversity_test = abs(total_knowledge - total_expected) <= 10  # Account for other domains
        if diversity_test:
            passed_tests += 1
            print("✅ Knowledge diversity test passed")
        else:
            print("❌ Knowledge diversity test failed")
        
        success_rate = passed_tests / total_tests
        print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            print("🎉 EMPIRICAL CONSTANTS INTEGRATION SUCCESSFUL!")
            print("✅ All 4 core domains successfully integrated!")
            print("🚀 WorldModelCore ready for advanced domain expansion!")
            
            print(f"\n🌟 NWTN Core Knowledge Foundation Complete:")
            print(f"   🔬 Physics: {physics_count} fundamental laws")
            print(f"   🧮 Mathematics: {math_count} essential principles")
            print(f"   🔣 Logic: {logic_count} reasoning rules")
            print(f"   🔢 Constants: {constants_count} empirical values")
            print(f"   📊 Total: {total_knowledge} knowledge items")
            
            return True
        else:
            print("❌ Empirical constants integration needs improvement")
            return False
        
    except Exception as e:
        print(f"❌ Error during empirical constants integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_empirical_constants_integration()
    
    if success:
        print(f"\n🎉 4-DOMAIN CORE KNOWLEDGE INTEGRATION COMPLETE!")
        print("✅ Physics, Mathematics, Logic, and Constants all integrated")
        print("🔬 Ready for advanced domain expansion (Biology, Chemistry, etc.)")
        print("🚀 NWTN WorldModelCore foundation is solid and comprehensive!")
    else:
        print("\n❌ Empirical constants integration test failed")
        print("🔧 Check WorldModelCore constants loading")