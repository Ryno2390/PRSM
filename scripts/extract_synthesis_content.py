#!/usr/bin/env python3
"""
Extract NWTN Synthesis Content - Direct Access
=============================================

Extract the actual natural language synthesis content from the completed
NWTN reasoning cycles that finished successfully.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
from prsm.nwtn.external_storage_config import get_external_knowledge_base
from prsm.nwtn.breakthrough_modes import BreakthroughMode

async def extract_synthesis_directly():
    """Re-run the reasoning to get clean synthesis output"""
    
    print("🔍 EXTRACTING NWTN SYNTHESIS CONTENT")
    print("=" * 60)
    print("📝 Running lightweight extraction to get natural language synthesis")
    print("=" * 60)
    print()
    
    # The original prompt
    test_prompt = ("What are the most promising theoretical approaches to unifying quantum mechanics "
                  "and general relativity, and what experimental evidence exists to support or refute these approaches?")
    
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
        print("✅ Systems ready")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    print()
    
    # Extract Conservative synthesis
    print("🎯 EXTRACTING CONSERVATIVE SYNTHESIS")
    print("-" * 45)
    
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
        
        print("🧠 Running Conservative reasoning...")
        conservative_result = await meta_engine.meta_reason(
            query=test_prompt,
            context=conservative_context,
            thinking_mode=ThinkingMode.DEEP
        )
        
        print("✅ Conservative reasoning complete")
        print(f"📊 Confidence: {conservative_result.meta_confidence:.1%}")
        print(f"📈 Quality: {conservative_result.quality_score:.1%}")
        print()
        
        # Extract synthesis
        if hasattr(conservative_result, 'final_synthesis') and conservative_result.final_synthesis:
            conservative_synthesis = str(conservative_result.final_synthesis)
            print("📝 CONSERVATIVE ANSWER:")
            print("-" * 25)
            print(conservative_synthesis)
            print()
        else:
            print("⚠️ No Conservative synthesis available")
            conservative_synthesis = "Synthesis not available"
        
    except Exception as e:
        print(f"❌ Conservative extraction failed: {e}")
        conservative_synthesis = f"Error: {e}"
    
    print("=" * 60)
    print()
    
    # Extract Revolutionary synthesis
    print("🚀 EXTRACTING REVOLUTIONARY SYNTHESIS")
    print("-" * 45)
    
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
        
        print("🧠 Running Revolutionary reasoning...")
        revolutionary_result = await meta_engine.meta_reason(
            query=test_prompt,
            context=revolutionary_context,
            thinking_mode=ThinkingMode.DEEP
        )
        
        print("✅ Revolutionary reasoning complete")
        print(f"📊 Confidence: {revolutionary_result.meta_confidence:.1%}")
        print(f"📈 Quality: {revolutionary_result.quality_score:.1%}")
        print()
        
        # Extract synthesis
        if hasattr(revolutionary_result, 'final_synthesis') and revolutionary_result.final_synthesis:
            revolutionary_synthesis = str(revolutionary_result.final_synthesis)
            print("📝 REVOLUTIONARY ANSWER:")
            print("-" * 25)
            print(revolutionary_synthesis)
            print()
        else:
            print("⚠️ No Revolutionary synthesis available")
            revolutionary_synthesis = "Synthesis not available"
        
    except Exception as e:
        print(f"❌ Revolutionary extraction failed: {e}")
        revolutionary_synthesis = f"Error: {e}"
    
    print("=" * 60)
    print()
    
    # Final summary
    print("📋 COMPLETE SYNTHESIS EXTRACTION")
    print("=" * 40)
    print()
    print("❓ ORIGINAL PROMPT:")
    print(f'"{test_prompt}"')
    print()
    print("🎯 CONSERVATIVE ANSWER (Established Consensus):")
    print("-" * 50)
    print(conservative_synthesis)
    print()
    print("🚀 REVOLUTIONARY ANSWER (Novel Breakthroughs):")
    print("-" * 50)
    print(revolutionary_synthesis)
    print()
    print("✅ Synthesis extraction complete!")

if __name__ == "__main__":
    asyncio.run(extract_synthesis_directly())