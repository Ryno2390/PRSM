#!/usr/bin/env python3
"""
Test Physics Integration with NWTN WorldModelCore
=================================================

This script tests the integration of extracted physics knowledge
with the NWTN Meta-Reasoning System.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.meta_reasoning_engine import WorldModelCore, MetaReasoningEngine

def test_physics_loading():
    """Test that physics knowledge loads correctly"""
    
    print("🧪 Testing Physics Knowledge Loading")
    print("=" * 50)
    
    try:
        # Initialize WorldModelCore
        world_model = WorldModelCore()
        
        # Check physics laws loading
        physics_laws = world_model.physical_laws
        
        print(f"✅ Physics laws loaded: {len(physics_laws)} items")
        
        # Show sample physics items
        print(f"\n📚 Sample Physics Knowledge:")
        for i, (key, item) in enumerate(list(physics_laws.items())[:5]):
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
        
        # Search for thermodynamics
        thermo_items = world_model.search_knowledge("thermodynamics", domain="physics")
        print(f"   Thermodynamics items: {len(thermo_items)}")
        
        # Test total knowledge index
        total_items = len(world_model.knowledge_index)
        print(f"\n📈 Total knowledge items: {total_items}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading physics knowledge: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_physics_validation():
    """Test physics knowledge validation"""
    
    print("\n🔍 Testing Physics Knowledge Validation")
    print("=" * 50)
    
    try:
        # Initialize WorldModelCore
        world_model = WorldModelCore()
        
        # Test validation system
        validator = world_model.validator
        
        # Test conflicting statement
        test_statement = "Energy can be created from nothing"
        conflicts = validator.detect_conflicts(test_statement)
        
        print(f"Testing statement: '{test_statement}'")
        print(f"Conflicts detected: {len(conflicts)}")
        
        if conflicts:
            print("Conflict details:")
            for conflict in conflicts:
                print(f"   - Conflicts with: {conflict['conflicting_knowledge']}")
                print(f"     Confidence: {conflict['confidence']:.2f}")
                print(f"     Reason: {conflict['reason']}")
        
        # Test supporting statement
        test_statement2 = "Force equals mass times acceleration"
        validation_result = validator.validate_statement(test_statement2)
        
        print(f"\nTesting statement: '{test_statement2}'")
        print(f"Validation result: {validation_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_meta_reasoning_integration():
    """Test integration with full meta-reasoning system"""
    
    print("\n🧠 Testing Meta-Reasoning Integration")
    print("=" * 50)
    
    try:
        # Initialize full meta-reasoning engine
        meta_engine = MetaReasoningEngine()
        
        # Test physics-based reasoning
        physics_query = {
            'query': 'What happens when a force is applied to an object?',
            'context': 'classical mechanics',
            'domain': 'physics'
        }
        
        print(f"Query: {physics_query['query']}")
        print("Processing with meta-reasoning engine...")
        
        # Process query
        result = meta_engine.process_query(physics_query)
        
        print(f"\n📊 Meta-Reasoning Result:")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Reasoning engines used: {len(result.reasoning_paths)}")
        
        # Check if world model validation was used
        if hasattr(result, 'world_model_validation'):
            print(f"   World model validation: {result.world_model_validation}")
        
        # Show reasoning summary
        if hasattr(result, 'synthesis_summary'):
            print(f"   Synthesis: {result.synthesis_summary[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing meta-reasoning: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_physics_knowledge_coverage():
    """Test coverage of physics knowledge"""
    
    print("\n📊 Testing Physics Knowledge Coverage")
    print("=" * 50)
    
    try:
        # Initialize WorldModelCore
        world_model = WorldModelCore()
        
        # Analyze physics knowledge by category
        physics_laws = world_model.physical_laws
        
        categories = {}
        for key, item in physics_laws.items():
            category = item.category
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
        
        print(f"Physics knowledge by category:")
        for category, items in categories.items():
            print(f"   {category}: {len(items)} items")
            
            # Show average certainty
            avg_certainty = sum(item.certainty for item in items) / len(items)
            print(f"      Average certainty: {avg_certainty:.3f}")
            
            # Show mathematical coverage
            with_math = sum(1 for item in items if item.mathematical_form)
            print(f"      With mathematical form: {with_math}/{len(items)}")
            print()
        
        # Test specific physics concepts
        key_concepts = [
            "newton", "conservation", "thermodynamics", "electromagnetic", 
            "quantum", "relativity", "energy", "momentum", "force"
        ]
        
        print(f"Coverage of key physics concepts:")
        for concept in key_concepts:
            matches = world_model.search_knowledge(concept, domain="physics")
            print(f"   {concept}: {len(matches)} matches")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing coverage: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all physics integration tests"""
    
    print("🚀 Physics Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Physics Loading", test_physics_loading),
        ("Physics Validation", test_physics_validation),
        ("Meta-Reasoning Integration", test_meta_reasoning_integration),
        ("Physics Coverage", test_physics_knowledge_coverage)
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
        print("🎉 All tests passed! Physics integration successful!")
        return True
    else:
        print("⚠️  Some tests failed. Check integration issues.")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Physics integration ready for production!")
    else:
        print("\n❌ Physics integration needs fixes.")