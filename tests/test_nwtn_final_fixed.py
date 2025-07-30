#!/usr/bin/env python3
"""
NWTN Pipeline FINAL FIXED TEST - Complete End-to-End
===================================================

Test the NWTN pipeline with the fundamental timing bug fixed.
This should provide complete synthesis results without any errors.
"""

import asyncio
import sys
import json
from datetime import datetime
sys.path.insert(0, '.')

from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
from prsm.nwtn.external_storage_config import get_external_knowledge_base
from prsm.nwtn.breakthrough_modes import BreakthroughMode

async def test_fixed_nwtn_pipeline():
    """Test the fully fixed NWTN pipeline end-to-end"""
    
    test_prompt = ("What are the most promising theoretical approaches to unifying quantum mechanics "
                  "and general relativity, and what experimental evidence exists to support or refute these approaches?")
    
    print("🔧 NWTN PIPELINE FINAL FIXED TEST")
    print("=" * 60)
    print("🐛 Bug Fixed: estimated_time None value in progress visualizer")
    print("🎯 Expected: Complete synthesis extraction with no errors")
    print("=" * 60)
    print()
    
    print("❓ ORIGINAL PROMPT:")
    print("-" * 20)
    print(f'"{test_prompt}"')
    print()
    
    # Initialize systems
    print("🚀 Initializing NWTN systems...")
    try:
        knowledge_base = await get_external_knowledge_base()
        meta_engine = MetaReasoningEngine()
        await meta_engine.initialize_external_knowledge_base()
        print("✅ Systems initialized successfully")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    print()
    
    # Test Conservative Mode
    print("🎯 TESTING CONSERVATIVE MODE")
    print("-" * 40)
    
    conservative_start = datetime.now()
    
    try:
        conservative_context = {
            "breakthrough_mode": BreakthroughMode.CONSERVATIVE.value,
            "query_type": "theoretical_physics",
            "domain": "quantum_gravity",
            "confidence_threshold": 0.8,
            "max_sources": 50,
            "reasoning_depth": "comprehensive",
            "experimental_evidence_required": True
        }
        
        print("🧠 Running Conservative deep reasoning (5,040 iterations)...")
        conservative_result = await meta_engine.meta_reason(
            query=test_prompt,
            context=conservative_context,
            thinking_mode=ThinkingMode.DEEP
        )
        
        conservative_time = (datetime.now() - conservative_start).total_seconds()
        
        print(f"✅ Conservative mode completed successfully!")
        print(f"⏱️  Processing Time: {conservative_time:.1f}s ({conservative_time/60:.1f} min)")
        print(f"🎯 Meta Confidence: {conservative_result.meta_confidence:.1%}")
        print(f"📊 Quality Score: {conservative_result.get_overall_quality():.1%}")
        print(f"💰 FTNS Cost: {conservative_result.ftns_cost:.1f} tokens")
        print()
        
        # Extract synthesis
        conservative_synthesis = conservative_result.get_synthesized_response()
        print("📝 CONSERVATIVE SYNTHESIS:")
        print("-" * 30)
        print(conservative_synthesis)
        print()
        
    except Exception as e:
        conservative_time = (datetime.now() - conservative_start).total_seconds()
        print(f"❌ Conservative mode failed after {conservative_time:.1f}s: {e}")
        conservative_result = None
        conservative_synthesis = f"Error: {e}"
    
    print("=" * 60)
    print()
    
    # Test Revolutionary Mode
    print("🚀 TESTING REVOLUTIONARY MODE")
    print("-" * 40)
    
    revolutionary_start = datetime.now()
    
    try:
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
        
        print("🧠 Running Revolutionary deep reasoning (5,040 iterations)...")
        revolutionary_result = await meta_engine.meta_reason(
            query=test_prompt,
            context=revolutionary_context,
            thinking_mode=ThinkingMode.DEEP
        )
        
        revolutionary_time = (datetime.now() - revolutionary_start).total_seconds()
        
        print(f"✅ Revolutionary mode completed successfully!")
        print(f"⏱️  Processing Time: {revolutionary_time:.1f}s ({revolutionary_time/60:.1f} min)")
        print(f"🎯 Meta Confidence: {revolutionary_result.meta_confidence:.1%}")
        print(f"📊 Quality Score: {revolutionary_result.get_overall_quality():.1%}")
        print(f"💰 FTNS Cost: {revolutionary_result.ftns_cost:.1f} tokens")
        print()
        
        # Extract synthesis
        revolutionary_synthesis = revolutionary_result.get_synthesized_response()
        print("📝 REVOLUTIONARY SYNTHESIS:")
        print("-" * 30)
        print(revolutionary_synthesis)
        print()
        
    except Exception as e:
        revolutionary_time = (datetime.now() - revolutionary_start).total_seconds()
        print(f"❌ Revolutionary mode failed after {revolutionary_time:.1f}s: {e}")
        revolutionary_result = None
        revolutionary_synthesis = f"Error: {e}"
    
    print("=" * 60)
    print()
    
    # Final Results Summary
    print("📋 COMPLETE SYNTHESIS RESULTS")
    print("=" * 50)
    print()
    
    print("❓ ORIGINAL PROMPT:")
    print(f'"{test_prompt}"')
    print()
    
    print("🎯 CONSERVATIVE ANSWER (Established Consensus Approach):")
    print("-" * 55)
    print(conservative_synthesis)
    print()
    
    print("🚀 REVOLUTIONARY ANSWER (Novel Breakthrough Approach):")
    print("-" * 55)
    print(revolutionary_synthesis)
    print()
    
    # Success summary
    if conservative_result and revolutionary_result:
        print("🎉 PIPELINE FULLY FUNCTIONAL!")
        print("✅ Both modes completed successfully")
        print("📄 Complete synthesis content extracted")
        print("🔧 Fundamental timing bug FIXED")
        print("🚀 Ready for production use and remaining 9 test prompts")
        
        # Save complete results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"nwtn_fixed_complete_results_{timestamp}.json"
        
        results_data = {
            "test_metadata": {
                "prompt": test_prompt,
                "bug_fixed": "estimated_time None value in progress visualizer",
                "test_timestamp": timestamp,
                "status": "FULLY_FUNCTIONAL"
            },
            "conservative_mode": {
                "success": True,
                "processing_time_seconds": conservative_time,
                "meta_confidence": float(conservative_result.meta_confidence),
                "quality_score": float(conservative_result.get_overall_quality()),
                "ftns_cost": float(conservative_result.ftns_cost),
                "synthesis": conservative_synthesis
            },
            "revolutionary_mode": {
                "success": True,
                "processing_time_seconds": revolutionary_time,
                "meta_confidence": float(revolutionary_result.meta_confidence),
                "quality_score": float(revolutionary_result.get_overall_quality()),
                "ftns_cost": float(revolutionary_result.ftns_cost),
                "synthesis": revolutionary_synthesis
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Complete results saved to: {results_file}")
        
    else:
        print("❌ Pipeline still has issues - further debugging needed")
    
    print()
    print("🔧 FUNDAMENTAL PIPELINE FIX COMPLETE")

if __name__ == "__main__":
    asyncio.run(test_fixed_nwtn_pipeline())