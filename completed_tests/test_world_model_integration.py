#!/usr/bin/env python3
"""
Test script for WorldModelCore integration with NWTN Meta-Reasoning System
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prsm.nwtn.meta_reasoning_engine import (
    MetaReasoningEngine, 
    ThinkingMode, 
    ReasoningResult,
    ReasoningEngine
)

async def test_world_model_integration():
    """Test the WorldModelCore integration with meta-reasoning"""
    
    print("🧠 Testing NWTN WorldModelCore Integration")
    print("=" * 50)
    
    # Initialize meta-reasoning engine
    meta_engine = MetaReasoningEngine()
    
    # Test 1: Search world model knowledge
    print("\n1. Testing World Model Knowledge Search")
    physics_knowledge = meta_engine.search_world_knowledge("force", min_certainty=0.9)
    print(f"Found {len(physics_knowledge)} physics knowledge items:")
    for item in physics_knowledge[:3]:  # Show first 3
        print(f"  - {item['content']} (certainty: {item['certainty']})")
        print(f"    Mathematical form: {item['mathematical_form']}")
        print(f"    Domain: {item['domain']}, Category: {item['category']}")
    
    # Test 2: Get relevant knowledge for a query
    print("\n2. Testing Relevant Knowledge Retrieval")
    query = "What happens when a ball is thrown upward?"
    relevant_knowledge = meta_engine.get_world_model_knowledge(query)
    print(f"Relevant knowledge for '{query}':")
    for item in relevant_knowledge:
        print(f"  - {item['content']} (certainty: {item['certainty']})")
    
    # Test 3: Test reasoning with world model validation
    print("\n3. Testing Reasoning with World Model Validation")
    
    # Create a test reasoning result
    test_result = ReasoningResult(
        reasoning_type=ReasoningEngine.DEDUCTIVE,
        conclusion="When a ball is thrown upward, it will eventually fall back down due to gravity",
        confidence=0.8,
        reasoning_trace="Applied Newton's laws of motion and gravitational force analysis",
        evidence=["Gravitational force", "Newton's laws"],
        assumptions=["Air resistance is negligible", "Near Earth's surface"],
        alternative_conclusions=["Ball might escape Earth's gravity if thrown fast enough"],
        metadata={"test": True}
    )
    
    # Validate against world model
    validation_result = meta_engine.validate_reasoning_against_world_model(test_result)
    print(f"Validation result:")
    print(f"  - Is valid: {validation_result['is_valid']}")
    print(f"  - Conflicts: {len(validation_result['conflicts'])}")
    print(f"  - Supporting knowledge: {len(validation_result['supporting_knowledge'])}")
    print(f"  - Confidence adjustment: {validation_result['confidence_adjustment']}")
    print(f"  - Recommendations: {validation_result['recommendations']}")
    
    # Test 4: Meta-reasoning with world model integration
    print("\n4. Testing Meta-Reasoning with World Model")
    
    context = {
        "domain": "physics",
        "level": "undergraduate"
    }
    
    try:
        # Test with world model integration enabled
        result = await meta_engine.meta_reason(
            query="Why does an object fall when dropped?",
            context=context,
            thinking_mode=ThinkingMode.QUICK,
            include_world_model=True
        )
        
        print(f"Meta-reasoning completed successfully!")
        print(f"  - Reasoning depth: {result.reasoning_depth}")
        print(f"  - Meta-confidence: {result.meta_confidence}")
        print(f"  - Parallel results: {len(result.parallel_results) if result.parallel_results else 0}")
        
        # Check if world model validation was applied
        if result.final_synthesis and 'world_model_validation' in result.final_synthesis:
            wm_validation = result.final_synthesis['world_model_validation']
            print(f"  - World model validation applied:")
            print(f"    * Valid: {wm_validation['is_valid']}")
            print(f"    * Conflicts: {wm_validation['conflicts']}")
            print(f"    * Supporting knowledge: {wm_validation['supporting_knowledge_count']}")
            print(f"    * Confidence adjustment: {wm_validation['confidence_adjustment']}")
        
    except Exception as e:
        print(f"Error during meta-reasoning: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Compare reasoning with and without world model
    print("\n5. Testing Comparison: With vs Without World Model")
    
    try:
        # Without world model
        result_without = await meta_engine.meta_reason(
            query="What is the relationship between force and acceleration?",
            context=context,
            thinking_mode=ThinkingMode.QUICK,
            include_world_model=False
        )
        
        # With world model
        result_with = await meta_engine.meta_reason(
            query="What is the relationship between force and acceleration?",
            context=context,
            thinking_mode=ThinkingMode.QUICK,
            include_world_model=True
        )
        
        print(f"Results comparison:")
        print(f"  - Without world model:")
        print(f"    * Meta-confidence: {result_without.meta_confidence}")
        print(f"    * Final synthesis confidence: {result_without.final_synthesis.get('confidence', 'N/A') if result_without.final_synthesis else 'N/A'}")
        
        print(f"  - With world model:")
        print(f"    * Meta-confidence: {result_with.meta_confidence}")
        print(f"    * Final synthesis confidence: {result_with.final_synthesis.get('confidence', 'N/A') if result_with.final_synthesis else 'N/A'}")
        
        if result_with.final_synthesis and 'world_model_confidence_adjustment' in result_with.final_synthesis:
            print(f"    * World model confidence adjustment: {result_with.final_synthesis['world_model_confidence_adjustment']}")
        
    except Exception as e:
        print(f"Error during comparison test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎉 WorldModelCore Integration Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_world_model_integration())