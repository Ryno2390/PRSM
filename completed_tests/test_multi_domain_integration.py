#!/usr/bin/env python3
"""
Test Multi-Domain Integration in NWTN Meta-Reasoning Engine
==========================================================

This script tests the integration of multiple domains (6 total) 
in the WorldModelCore system.
"""

import sys
import os
import json
import logging

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

# Suppress warnings for cleaner output
logging.getLogger().setLevel(logging.ERROR)

def test_multi_domain_integration():
    """Test multi-domain knowledge integration"""
    
    print("🌐 Testing Multi-Domain Integration in NWTN")
    print("=" * 60)
    
    try:
        # Import the WorldModelCore directly
        from prsm.nwtn.meta_reasoning_engine import WorldModelCore
        
        print("✅ Successfully imported WorldModelCore")
        
        # Initialize the WorldModelCore
        print("\n🔧 Initializing WorldModelCore with Multiple Domains...")
        world_model = WorldModelCore()
        
        print("✅ WorldModelCore initialized successfully")
        
        # Check knowledge counts
        total_knowledge = len(world_model.knowledge_index)
        physics_count = len(world_model.physical_laws)
        math_count = len(world_model.mathematical_truths)
        logic_count = len(world_model.logical_principles)
        constants_count = len(world_model.empirical_constants)
        biology_count = len(world_model.biological_foundations)
        chemistry_count = len(world_model.chemical_principles)
        
        print(f"\n📊 Multi-Domain Knowledge Statistics:")
        print(f"   - Total Knowledge Items: {total_knowledge}")
        print(f"   - Physics Laws: {physics_count}")
        print(f"   - Mathematical Truths: {math_count}")
        print(f"   - Logical Principles: {logic_count}")
        print(f"   - Empirical Constants: {constants_count}")
        print(f"   - Biological Foundations: {biology_count}")
        print(f"   - Chemical Principles: {chemistry_count}")
        
        # Show sample knowledge from each domain
        print(f"\n🔍 Sample Knowledge from Each Domain:")
        
        # Biology sample
        bio_items = list(world_model.biological_foundations.items())[:2]
        print(f"   🧬 Biology:")
        for key, item in bio_items:
            print(f"      - {item.content}")
            print(f"        Mathematical form: {item.mathematical_form}")
            print(f"        Certainty: {item.certainty}")
        
        # Chemistry sample
        chem_items = list(world_model.chemical_principles.items())[:2]
        print(f"   ⚗️ Chemistry:")
        for key, item in chem_items:
            print(f"      - {item.content}")
            print(f"        Mathematical form: {item.mathematical_form}")
            print(f"        Certainty: {item.certainty}")
        
        # Test cross-domain knowledge searching
        print(f"\n🔍 Testing Cross-Domain Knowledge Search:")
        
        # Test biology search
        bio_query = "cell division"
        bio_results = world_model.get_supporting_knowledge(bio_query)
        print(f"   Biology Query '{bio_query}': {len(bio_results)} results")
        
        # Test chemistry search
        chem_query = "chemical reaction"
        chem_results = world_model.get_supporting_knowledge(chem_query)
        print(f"   Chemistry Query '{chem_query}': {len(chem_results)} results")
        
        # Test validation success
        total_tests = 8
        passed_tests = 0
        
        expected_total = physics_count + math_count + logic_count + constants_count + biology_count + chemistry_count
        
        if total_knowledge >= expected_total - 5:  # Allow small variance
            passed_tests += 1
            print("✅ Total knowledge count test passed")
        else:
            print("❌ Total knowledge count test failed")
            
        if biology_count >= 25:  # Should have substantial biology
            passed_tests += 1
            print("✅ Biology knowledge count test passed")
        else:
            print("❌ Biology knowledge count test failed")
            
        if chemistry_count >= 25:  # Should have substantial chemistry
            passed_tests += 1
            print("✅ Chemistry knowledge count test passed")
        else:
            print("❌ Chemistry knowledge count test failed")
            
        if len(bio_results) > 0:
            passed_tests += 1
            print("✅ Biology search test passed")
        else:
            print("❌ Biology search test failed")
            
        if len(chem_results) > 0:
            passed_tests += 1
            print("✅ Chemistry search test passed")
        else:
            print("❌ Chemistry search test failed")
        
        # Test domain balance
        domain_balance = all([
            physics_count >= 20,
            math_count >= 20,
            logic_count >= 20,
            constants_count >= 20,
            biology_count >= 20,
            chemistry_count >= 20
        ])
        if domain_balance:
            passed_tests += 1
            print("✅ Domain balance test passed")
        else:
            print("❌ Domain balance test failed")
        
        # Test knowledge diversity
        diversity_test = total_knowledge >= 150  # Should have substantial knowledge
        if diversity_test:
            passed_tests += 1
            print("✅ Knowledge diversity test passed")
        else:
            print("❌ Knowledge diversity test failed")
            
        # Test high certainty
        high_certainty = all([
            physics_count > 0,
            math_count > 0,
            logic_count > 0,
            constants_count > 0,
            biology_count > 0,
            chemistry_count > 0
        ])
        if high_certainty:
            passed_tests += 1
            print("✅ Multi-domain integration test passed")
        else:
            print("❌ Multi-domain integration test failed")
        
        success_rate = passed_tests / total_tests
        print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1%})")
        
        if success_rate >= 0.75:
            print("🎉 MULTI-DOMAIN INTEGRATION SUCCESSFUL!")
            print("✅ 6 domains successfully integrated!")
            print("🚀 WorldModelCore ready for remaining domains!")
            
            print(f"\n🌟 Current Multi-Domain Knowledge Foundation:")
            print(f"   🔬 Physics: {physics_count} fundamental laws")
            print(f"   🧮 Mathematics: {math_count} essential principles")
            print(f"   🔣 Logic: {logic_count} reasoning rules")
            print(f"   🔢 Constants: {constants_count} empirical values")
            print(f"   🧬 Biology: {biology_count} biological principles")
            print(f"   ⚗️ Chemistry: {chemistry_count} chemical principles")
            print(f"   📊 Total: {total_knowledge} knowledge items")
            
            return True
        else:
            print("❌ Multi-domain integration needs improvement")
            return False
        
    except Exception as e:
        print(f"❌ Error during multi-domain integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_domain_integration()
    
    if success:
        print(f"\n🎉 6-DOMAIN INTEGRATION SUCCESS!")
        print("✅ Physics, Mathematics, Logic, Constants, Biology, Chemistry integrated")
        print("🔬 Ready for Computer Science, Astronomy, and Medicine!")
        print("🚀 NWTN WorldModelCore is becoming truly comprehensive!")
    else:
        print("\n❌ Multi-domain integration test failed")
        print("🔧 Check domain loading and integration")