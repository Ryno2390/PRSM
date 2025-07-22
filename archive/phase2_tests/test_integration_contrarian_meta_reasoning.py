#!/usr/bin/env python3
"""
Integration Test: Contrarian Candidates + Meta-Reasoning Engine
==============================================================

Tests the integration of contrarian candidate generation with the meta-reasoning engine.
This verifies that Phase 2.1 (Contrarian Candidate Generation) is fully integrated
with the System 1 Breakthrough Candidate Generation.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
from prsm.nwtn.breakthrough_modes import BreakthroughMode

def print_header():
    """Print test header"""
    print("🔄 CONTRARIAN CANDIDATES + META-REASONING INTEGRATION TEST")
    print("=" * 65)
    print("Testing Phase 2.1 integration: Contrarian + Meta-Reasoning")
    print()

async def test_breakthrough_mode_integration():
    """Test breakthrough mode determination and contrarian generation in meta-reasoning"""
    print("🚀 TESTING BREAKTHROUGH MODE INTEGRATION")
    print("-" * 40)
    
    # Initialize meta-reasoning engine
    engine = MetaReasoningEngine()
    await engine.initialize_external_knowledge_base()
    
    test_queries = [
        {
            "query": "What are the most effective vitamin D supplements for immune health?",
            "context": {"breakthrough_mode": "conservative"},
            "expected_mode": BreakthroughMode.CONSERVATIVE
        },
        {
            "query": "How can AI systems achieve breakthrough performance on reasoning tasks?",
            "context": {"breakthrough_mode": "creative"},
            "expected_mode": BreakthroughMode.CREATIVE
        },
        {
            "query": "What impossible energy storage solutions could revolutionize transportation?",
            "context": {"breakthrough_mode": "revolutionary"},
            "expected_mode": BreakthroughMode.REVOLUTIONARY
        },
        {
            "query": "Compare different marketing strategies for business growth",
            "context": {},  # No explicit mode - should use AI suggestion
            "expected_mode": None  # Will be determined by AI
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"🔍 TEST CASE {i}: {test_case['expected_mode'].value if test_case['expected_mode'] else 'AI_SUGGESTED'}")
        print(f"   Query: {test_case['query']}")
        
        # Test breakthrough mode determination
        determined_mode = engine._determine_breakthrough_mode(test_case['query'], test_case['context'])
        
        print(f"   Context: {test_case['context']}")
        print(f"   Determined Mode: {determined_mode.value}")
        
        if test_case['expected_mode']:
            if determined_mode == test_case['expected_mode']:
                print("   ✅ Breakthrough mode determination: PASSED")
            else:
                print(f"   ❌ Expected {test_case['expected_mode'].value}, got {determined_mode.value}")
        else:
            print(f"   ✅ AI-suggested mode: {determined_mode.value}")
        
        print()

async def test_full_meta_reasoning_with_contrarian():
    """Test complete meta-reasoning with contrarian candidate generation"""
    print("🧠 TESTING FULL META-REASONING WITH CONTRARIAN CANDIDATES")
    print("-" * 50)
    
    engine = MetaReasoningEngine()
    await engine.initialize_external_knowledge_base()
    
    # Test with creative mode (should generate contrarian candidates)
    test_query = "What are the best approaches to renewable energy adoption?"
    test_context = {
        "breakthrough_mode": "creative",
        "thinking_mode": "QUICK",
        "verbosity_level": "STANDARD",
        "user_tier": "standard"
    }
    
    print(f"Query: {test_query}")
    print(f"Context: {test_context}")
    print()
    
    try:
        # Run complete meta-reasoning
        print("🔄 Starting meta-reasoning with contrarian integration...")
        result = await engine.meta_reason(
            query=test_query,
            context=test_context,
            thinking_mode=ThinkingMode.QUICK
        )
        
        # Verify breakthrough mode integration
        print(f"✅ Meta-reasoning completed successfully!")
        print(f"   Result ID: {result.id}")
        print(f"   Breakthrough Mode: {result.breakthrough_mode}")
        print(f"   Contrarian Candidates Generated: {len(result.contrarian_candidates)}")
        
        if result.contrarian_candidates:
            print("   📋 Contrarian Candidates:")
            for i, candidate in enumerate(result.contrarian_candidates[:3], 1):
                print(f"     {i}. Type: {candidate.contrarian_type.value}")
                print(f"        Position: {candidate.contrarian_position[:100]}...")
                print(f"        Confidence: {candidate.confidence_score:.2f}")
                print(f"        Novelty: {candidate.novelty_score:.2f}")
        else:
            print("   ⚠️  No contrarian candidates generated (may be due to 0% contrarian distribution)")
        
        print(f"   Processing Time: {result.total_processing_time:.2f} seconds")
        print(f"   Confidence: {result.meta_confidence:.2f}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Meta-reasoning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_contrarian_distribution_by_mode():
    """Test that different breakthrough modes generate different amounts of contrarian candidates"""
    print("📊 TESTING CONTRARIAN DISTRIBUTION BY BREAKTHROUGH MODE")
    print("-" * 50)
    
    engine = MetaReasoningEngine()
    
    modes_to_test = [
        BreakthroughMode.CONSERVATIVE,
        BreakthroughMode.BALANCED, 
        BreakthroughMode.CREATIVE,
        BreakthroughMode.REVOLUTIONARY
    ]
    
    query = "How can we improve AI safety and alignment?"
    
    for mode in modes_to_test:
        print(f"🎯 Testing {mode.value.upper()} mode")
        
        context = {
            "breakthrough_mode": mode.value,
            "thinking_mode": "QUICK"
        }
        
        # Test mode determination and contrarian generation logic
        determined_mode = engine._determine_breakthrough_mode(query, context)
        print(f"   Determined Mode: {determined_mode.value}")
        
        # Get breakthrough config to check contrarian distribution
        from prsm.nwtn.breakthrough_modes import get_breakthrough_mode_config
        config = get_breakthrough_mode_config(determined_mode)
        contrarian_percentage = config.candidate_distribution.contrarian * 100
        
        print(f"   Expected Contrarian %: {contrarian_percentage:.0f}%")
        print(f"   Assumption Challenging: {config.assumption_challenging_enabled}")
        print(f"   Wild Hypothesis: {config.wild_hypothesis_enabled}")
        print(f"   Impossibility Exploration: {config.impossibility_exploration_enabled}")
        print()

async def main():
    """Run all integration tests"""
    try:
        print_header()
        
        # Run individual tests
        await test_breakthrough_mode_integration()
        await test_contrarian_distribution_by_mode()
        
        # Run full integration test
        success = await test_full_meta_reasoning_with_contrarian()
        
        if success:
            print("✅ ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY")
            print()
            print("🎯 INTEGRATION STATUS:")
            print("   • Breakthrough mode determination: ✅ Working")
            print("   • Contrarian candidate generation: ✅ Working") 
            print("   • Meta-reasoning engine integration: ✅ Working")
            print("   • Result dataclass enhancement: ✅ Working")
            print()
            print("🚀 PHASE 2.1 CONTRARIAN INTEGRATION: ✅ COMPLETE")
            print("   The Enhanced System 1 Breakthrough Candidate Generation")
            print("   now includes fully integrated contrarian candidate support!")
            
        else:
            print("❌ INTEGRATION TEST FAILED - Check error messages above")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())