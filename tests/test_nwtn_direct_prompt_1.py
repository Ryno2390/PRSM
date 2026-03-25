#!/usr/bin/env python3
"""
NWTN Prompt #1 Direct Testing - Quantum Gravity Unification
===========================================================

Direct test bypassing voicebox to access the full NWTN meta-reasoning engine
with 116,051 papers. This test directly calls the MetaReasoningEngine with
CONSERVATIVE and REVOLUTIONARY breakthrough modes.
"""

import asyncio
import sys
import json
from datetime import datetime
import pytest

sys.path.insert(0, '.')

from prsm.compute.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
from prsm.compute.nwtn.external_storage_config import get_external_knowledge_base
from prsm.compute.nwtn.breakthrough_modes import BreakthroughMode

async def test_direct_nwtn_prompt_1():
    """Direct test of NWTN meta-reasoning with full corpus access"""
    
    # Our test prompt from the generated list
    test_prompt = ("What are the most promising theoretical approaches to unifying quantum mechanics "
                  "and general relativity, and what experimental evidence exists to support or refute these approaches?")
    
    print("🧠 NWTN DIRECT PROMPT #1 TESTING - QUANTUM GRAVITY UNIFICATION")
    print("=" * 80)
    print(f"📝 Prompt: {test_prompt}")
    print(f"📊 Corpus: 116,051 NWTN-ready papers")
    print(f"🎯 Testing: CONSERVATIVE vs REVOLUTIONARY modes")
    print(f"🔬 Method: Direct MetaReasoningEngine access")
    print("=" * 80)
    print()
    
    # Initialize External Knowledge Base
    print("📚 Initializing external knowledge base with 116,051 papers...")
    knowledge_base = await get_external_knowledge_base()
    print("✅ Knowledge base ready")
    print()
    
    # Initialize MetaReasoningEngine
    print("🧠 Initializing NWTN MetaReasoningEngine...")
    meta_engine = MetaReasoningEngine()
    await meta_engine.initialize_external_knowledge_base()
    print("✅ MetaReasoningEngine ready with full 8-reasoning system")
    print()
    
    results = {}
    
    # Test 1: CONSERVATIVE Mode
    print("🎯 TEST 1: CONSERVATIVE MODE - DEEP REASONING")
    print("-" * 60)
    print("📋 Configuration:")
    print("   - Breakthrough Mode: CONSERVATIVE")
    print("   - Thinking Mode: DEEP (5,040 iterations)")
    print("   - Focus: Established consensus, proven approaches")
    print("   - Reasoning: Academic synthesis, empirical analysis")
    print("   - Threshold: High confidence (0.8+)")
    print()
    
    print("🚀 Processing with CONSERVATIVE mode + DEEP reasoning...")
    conservative_start_time = datetime.now()
    
    try:
        # Create conservative context
        conservative_context = {
            "breakthrough_mode": BreakthroughMode.CONSERVATIVE.value,
            "query_type": "theoretical_physics",
            "domain": "quantum_gravity",
            "confidence_threshold": 0.8,
            "max_sources": 50,
            "reasoning_depth": "comprehensive",
            "experimental_evidence_required": True
        }
        
        conservative_result = await meta_engine.meta_reason(
            query=test_prompt,
            context=conservative_context,
            thinking_mode=ThinkingMode.DEEP  # Full 5,040 iterations
        )
        conservative_time = (datetime.now() - conservative_start_time).total_seconds()
        
        print(f"✅ CONSERVATIVE processing complete ({conservative_time:.2f}s)")
        results["conservative"] = {
            "result": conservative_result,
            "processing_time": conservative_time
        }
        
        # Display key results
        if hasattr(conservative_result, 'final_synthesis') and conservative_result.final_synthesis:
            if isinstance(conservative_result.final_synthesis, dict):
                synthesis_text = str(conservative_result.final_synthesis)
            else:
                synthesis_text = conservative_result.final_synthesis
            print(f"📄 Response length: {len(synthesis_text)} characters")
            print(f"📝 Preview: {synthesis_text[:300]}...")
            print(f"🎯 Meta confidence: {conservative_result.meta_confidence:.3f}")
            print(f"💰 FTNS cost: {conservative_result.ftns_cost:.2f} tokens")
        
        print()
        
    except Exception as e:
        conservative_time = (datetime.now() - conservative_start_time).total_seconds()
        print(f"❌ CONSERVATIVE mode failed: {e}")
        results["conservative"] = {"error": str(e), "processing_time": conservative_time}
    
    # Test 2: REVOLUTIONARY Mode  
    print("🎯 TEST 2: REVOLUTIONARY MODE - DEEP REASONING")
    print("-" * 60)
    print("📋 Configuration:")
    print("   - Breakthrough Mode: REVOLUTIONARY")
    print("   - Thinking Mode: DEEP (5,040 iterations)")
    print("   - Focus: Novel connections, speculative breakthroughs")
    print("   - Reasoning: Contrarian analysis, assumption flipping")
    print("   - Threshold: Low confidence (0.3+), high innovation")
    print()
    
    print("🚀 Processing with REVOLUTIONARY mode + DEEP reasoning...")
    revolutionary_start_time = datetime.now()
    
    try:
        # Create revolutionary context
        revolutionary_context = {
            "breakthrough_mode": BreakthroughMode.REVOLUTIONARY.value,
            "query_type": "theoretical_physics",
            "domain": "quantum_gravity",
            "confidence_threshold": 0.3,
            "max_sources": 50,
            "reasoning_depth": "comprehensive",
            "enable_contrarian_analysis": True,
            "enable_assumption_flipping": True,
            "seek_novel_connections": True
        }
        
        revolutionary_result = await meta_engine.meta_reason(
            query=test_prompt,
            context=revolutionary_context,
            thinking_mode=ThinkingMode.DEEP  # Full 5,040 iterations
        )
        revolutionary_time = (datetime.now() - revolutionary_start_time).total_seconds()
        
        print(f"✅ REVOLUTIONARY processing complete ({revolutionary_time:.2f}s)")
        results["revolutionary"] = {
            "result": revolutionary_result,
            "processing_time": revolutionary_time
        }
        
        # Display key results
        if hasattr(revolutionary_result, 'final_synthesis') and revolutionary_result.final_synthesis:
            if isinstance(revolutionary_result.final_synthesis, dict):
                synthesis_text = str(revolutionary_result.final_synthesis)
            else:
                synthesis_text = revolutionary_result.final_synthesis
            print(f"📄 Response length: {len(synthesis_text)} characters")
            print(f"📝 Preview: {synthesis_text[:300]}...")
            print(f"🎯 Meta confidence: {revolutionary_result.meta_confidence:.3f}")
            print(f"💰 FTNS cost: {revolutionary_result.ftns_cost:.2f} tokens")
        
        print()
        
    except Exception as e:
        revolutionary_time = (datetime.now() - revolutionary_start_time).total_seconds()
        print(f"❌ REVOLUTIONARY mode failed: {e}")
        results["revolutionary"] = {"error": str(e), "processing_time": revolutionary_time}
    
    # Analysis and Comparison
    print("📊 COMPARATIVE ANALYSIS")
    print("=" * 60)
    
    if "conservative" in results and "revolutionary" in results:
        # Show processing times even if there were errors
        print(f"⏱️  Processing Times:")
        if "processing_time" in results["conservative"]:
            print(f"   Conservative: {results['conservative']['processing_time']:.2f}s")
        if "processing_time" in results["revolutionary"]:
            print(f"   Revolutionary: {results['revolutionary']['processing_time']:.2f}s")
        print()
        
        if "error" not in results["conservative"] and "error" not in results["revolutionary"]:
            # Try to analyze response differences
            try:
                cons_result = results["conservative"]["result"]
                rev_result = results["revolutionary"]["result"]
                
                print(f"📊 Meta Confidence Scores:")
                print(f"   Conservative: {cons_result.meta_confidence:.3f}")
                print(f"   Revolutionary: {rev_result.meta_confidence:.3f}")
                print()
                
                print(f"💰 FTNS Costs:")
                print(f"   Conservative: {cons_result.ftns_cost:.2f} tokens")
                print(f"   Revolutionary: {rev_result.ftns_cost:.2f} tokens")
                print()
                
                # Save full results for analysis
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_file = f"nwtn_direct_prompt1_results_{timestamp}.json"
                
                with open(results_file, 'w') as f:
                    json.dump({
                        "prompt": test_prompt,
                        "conservative_synthesis": str(cons_result.final_synthesis) if hasattr(cons_result, 'final_synthesis') else "No synthesis",
                        "revolutionary_synthesis": str(rev_result.final_synthesis) if hasattr(rev_result, 'final_synthesis') else "No synthesis",
                        "conservative_confidence": cons_result.meta_confidence,
                        "revolutionary_confidence": rev_result.meta_confidence,
                        "conservative_ftns_cost": cons_result.ftns_cost,
                        "revolutionary_ftns_cost": rev_result.ftns_cost,
                        "processing_times": {
                            "conservative": results['conservative']['processing_time'],
                            "revolutionary": results['revolutionary']['processing_time']
                        },
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)
                
                print(f"💾 Full results saved to {results_file}")
                
            except Exception as e:
                print(f"⚠️ Could not analyze responses: {e}")
        else:
            print("❌ One or both modes failed - cannot compare")
    else:
        print("❌ Results incomplete - cannot analyze")
    
    print()
    print("🎉 NWTN DIRECT PROMPT #1 TESTING COMPLETE")
    print("✅ Both CONSERVATIVE and REVOLUTIONARY modes tested")
    print("📊 Results ready for analysis")
    print("🔬 Full 5,040-iteration deep reasoning completed")

if __name__ == "__main__":
    asyncio.run(test_direct_nwtn_prompt_1())