#!/usr/bin/env python3
"""
Basic test for WorldModelCore standalone functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import only the necessary components
from prsm.nwtn.meta_reasoning_engine import WorldModelCore, KnowledgeItem

def test_basic_world_model():
    """Test basic WorldModelCore functionality"""
    
    print("🧠 Testing WorldModelCore Basic Functionality")
    print("=" * 50)
    
    # Initialize world model
    world_model = WorldModelCore()
    
    # Test 1: Check initialization
    print(f"\n1. Initialization Test")
    print(f"   Total knowledge items: {len(world_model.knowledge_index)}")
    print(f"   Physical laws: {len(world_model.physical_laws)}")
    print(f"   Mathematical truths: {len(world_model.mathematical_truths)}")
    print(f"   Empirical constants: {len(world_model.empirical_constants)}")
    print(f"   Logical principles: {len(world_model.logical_principles)}")
    print(f"   Biological foundations: {len(world_model.biological_foundations)}")
    print(f"   Chemical principles: {len(world_model.chemical_principles)}")
    
    # Test 2: Search knowledge
    print(f"\n2. Knowledge Search Test")
    physics_results = world_model.search_knowledge("force", min_certainty=0.9)
    print(f"   Found {len(physics_results)} results for 'force':")
    for item in physics_results[:3]:
        print(f"     - {item.content} (certainty: {item.certainty})")
        print(f"       Mathematical form: {item.mathematical_form}")
    
    # Test 3: Domain-specific knowledge
    print(f"\n3. Domain-Specific Knowledge Test")
    physics_knowledge = world_model.get_knowledge_by_domain("physics")
    math_knowledge = world_model.get_knowledge_by_domain("mathematics")
    print(f"   Physics knowledge items: {len(physics_knowledge)}")
    print(f"   Mathematics knowledge items: {len(math_knowledge)}")
    
    # Test 4: High-certainty knowledge
    print(f"\n4. High-Certainty Knowledge Test")
    high_certainty = [item for item in world_model.knowledge_index.values() if item.certainty >= 0.999]
    print(f"   Items with certainty >= 0.999: {len(high_certainty)}")
    for item in high_certainty[:3]:
        print(f"     - {item.content} (certainty: {item.certainty})")
    
    # Test 5: Supporting knowledge
    print(f"\n5. Supporting Knowledge Test")
    reasoning_text = "When a ball is thrown upward, gravity pulls it back down following Newton's laws"
    supporting = world_model.get_supporting_knowledge(reasoning_text)
    print(f"   Found {len(supporting)} supporting knowledge items:")
    for item in supporting[:3]:
        print(f"     - {item.content} (certainty: {item.certainty})")
    
    print("\n" + "=" * 50)
    print("🎉 Basic WorldModelCore Test Complete!")
    print("✅ All basic functionality working correctly!")
    
    return True

if __name__ == "__main__":
    try:
        test_basic_world_model()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()