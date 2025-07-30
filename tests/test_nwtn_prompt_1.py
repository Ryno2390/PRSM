#!/usr/bin/env python3
"""
NWTN Prompt #1 Testing - Quantum Gravity Unification
====================================================

Testing the complete NWTN pipeline with our first test prompt:
"What are the most promising theoretical approaches to unifying quantum mechanics 
and general relativity, and what experimental evidence exists to support or refute these approaches?"

This test will run BOTH extreme modes:
1. CONSERVATIVE mode - Established consensus, proven approaches
2. REVOLUTIONARY mode - Novel connections, speculative breakthroughs

Using our 116,051 NWTN-ready papers with full pipeline:
- Prompt decomposition → Embedding search → Candidate generation → 5,040 iterations → Claude synthesis
"""

import asyncio
import sys
import json
from datetime import datetime
sys.path.insert(0, '.')

from prsm.nwtn.complete_system import NWTNCompleteSystem
from prsm.nwtn.breakthrough_modes import BreakthroughMode
from prsm.core.models import UserInput

async def test_prompt_1_both_modes():
    """Test Prompt #1 with both CONSERVATIVE and REVOLUTIONARY modes"""
    
    # Our test prompt from the generated list
    test_prompt = ("What are the most promising theoretical approaches to unifying quantum mechanics "
                  "and general relativity, and what experimental evidence exists to support or refute these approaches?")
    
    print("🧠 NWTN PROMPT #1 TESTING - QUANTUM GRAVITY UNIFICATION")
    print("=" * 80)
    print(f"📝 Prompt: {test_prompt}")
    print(f"📊 Corpus: 116,051 NWTN-ready papers")
    print(f"🎯 Testing: CONSERVATIVE vs REVOLUTIONARY modes")
    print("=" * 80)
    print()
    
    # Initialize NWTN Complete System
    print("🔧 Initializing NWTN Complete System...")
    nwtn_system = NWTNCompleteSystem()
    await nwtn_system.initialize()
    
    # Configure API key for test user
    print("🔑 Configuring API key for test user...")
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if api_key:
        await nwtn_system.configure_user_api("test_user", "claude", api_key)
        print("✅ API key configured for Claude")
    else:
        print("⚠️  No API key found in environment - using fallback mode")
    
    print("✅ NWTN System ready")
    print()
    
    results = {}
    
    # Test 1: CONSERVATIVE Mode
    print("🎯 TEST 1: CONSERVATIVE MODE")
    print("-" * 50)
    print("📋 Configuration:")
    print("   - Breakthrough Mode: CONSERVATIVE")
    print("   - Focus: Established consensus, proven approaches") 
    print("   - Reasoning: Academic synthesis, empirical analysis")
    print("   - Threshold: High confidence (0.8+)")
    print()
    
    conservative_input = UserInput(
        user_id="test_user",
        prompt=test_prompt,
        context_allocation=2000,
        preferences={
            "breakthrough_mode": BreakthroughMode.CONSERVATIVE.value,
            "verbosity": "detailed",
            "max_sources": 10,
            "enable_deep_reasoning": True,
            "require_citations": True,
            "reasoning_depth": "deep",  # Full 5,040 iterations
        }
    )
    
    print("🚀 Processing with CONSERVATIVE mode...")
    start_time = datetime.now()
    
    try:
        conservative_result = await nwtn_system.process_query(
            user_id="test_user",
            query=test_prompt,
            context=conservative_input.preferences,
            show_reasoning_trace=True
        )
        conservative_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ CONSERVATIVE processing complete ({conservative_time:.2f}s)")
        results["conservative"] = {
            "result": conservative_result,
            "processing_time": conservative_time
        }
        
        # Display key results
        if hasattr(conservative_result, 'natural_language_response') and conservative_result.natural_language_response:
            response_text = conservative_result.natural_language_response
            print(f"📄 Response length: {len(response_text)} characters")
            print(f"📝 Preview: {response_text[:200]}...")
        elif hasattr(conservative_result, 'response') and conservative_result.response:
            response_text = conservative_result.response
            print(f"📄 Response length: {len(response_text)} characters")
            print(f"📝 Preview: {response_text[:200]}...")
        
        print()
        
    except Exception as e:
        print(f"❌ CONSERVATIVE mode failed: {e}")
        results["conservative"] = {"error": str(e)}
    
    # Test 2: REVOLUTIONARY Mode  
    print("🎯 TEST 2: REVOLUTIONARY MODE")
    print("-" * 50)
    print("📋 Configuration:")
    print("   - Breakthrough Mode: REVOLUTIONARY")
    print("   - Focus: Novel connections, speculative breakthroughs")
    print("   - Reasoning: Contrarian analysis, assumption flipping")
    print("   - Threshold: Low confidence (0.3+), high innovation")
    print()
    
    revolutionary_input = UserInput(
        user_id="test_user", 
        prompt=test_prompt,
        context_allocation=2000,
        preferences={
            "breakthrough_mode": BreakthroughMode.REVOLUTIONARY.value,
            "verbosity": "detailed", 
            "max_sources": 10,
            "enable_deep_reasoning": True,
            "require_citations": True,
            "reasoning_depth": "deep",  # Full 5,040 iterations
        }
    )
    
    print("🚀 Processing with REVOLUTIONARY mode...")
    start_time = datetime.now()
    
    try:
        revolutionary_result = await nwtn_system.process_query(
            user_id="test_user",
            query=test_prompt,
            context=revolutionary_input.preferences,
            show_reasoning_trace=True
        )
        revolutionary_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ REVOLUTIONARY processing complete ({revolutionary_time:.2f}s)")
        results["revolutionary"] = {
            "result": revolutionary_result,
            "processing_time": revolutionary_time
        }
        
        # Display key results
        if hasattr(revolutionary_result, 'natural_language_response') and revolutionary_result.natural_language_response:
            response_text = revolutionary_result.natural_language_response
            print(f"📄 Response length: {len(response_text)} characters")
            print(f"📝 Preview: {response_text[:200]}...")
        elif hasattr(revolutionary_result, 'response') and revolutionary_result.response:
            response_text = revolutionary_result.response
            print(f"📄 Response length: {len(response_text)} characters")
            print(f"📝 Preview: {response_text[:200]}...")
        
        print()
        
    except Exception as e:
        print(f"❌ REVOLUTIONARY mode failed: {e}")
        results["revolutionary"] = {"error": str(e)}
    
    # Analysis and Comparison
    print("📊 COMPARATIVE ANALYSIS")
    print("=" * 50)
    
    if "conservative" in results and "revolutionary" in results:
        if "error" not in results["conservative"] and "error" not in results["revolutionary"]:
            print(f"⏱️ Processing Times:")
            print(f"   Conservative: {results['conservative']['processing_time']:.2f}s")
            print(f"   Revolutionary: {results['revolutionary']['processing_time']:.2f}s")
            print()
            
            # Try to analyze response differences
            try:
                cons_response = results["conservative"]["result"].final_response
                rev_response = results["revolutionary"]["result"].final_response
                
                print(f"📏 Response Lengths:")
                print(f"   Conservative: {len(cons_response)} characters")
                print(f"   Revolutionary: {len(rev_response)} characters")
                print()
                
                # Save full results for analysis
                with open(f"nwtn_prompt1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                    json.dump({
                        "prompt": test_prompt,
                        "conservative_response": cons_response,
                        "revolutionary_response": rev_response,
                        "processing_times": {
                            "conservative": results['conservative']['processing_time'],
                            "revolutionary": results['revolutionary']['processing_time']
                        },
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)
                
                print("💾 Full results saved to JSON file")
                
            except Exception as e:
                print(f"⚠️ Could not analyze responses: {e}")
        else:
            print("❌ One or both modes failed - cannot compare")
    else:
        print("❌ Results incomplete - cannot analyze")
    
    print()
    print("🎉 NWTN PROMPT #1 TESTING COMPLETE")
    print("✅ Both CONSERVATIVE and REVOLUTIONARY modes tested")
    print("📊 Results ready for analysis")

if __name__ == "__main__":
    asyncio.run(test_prompt_1_both_modes())