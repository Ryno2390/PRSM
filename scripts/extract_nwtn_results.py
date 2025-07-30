#!/usr/bin/env python3
"""
Extract NWTN Results from Completed Reasoning Cycles
==================================================

Extract the synthesis results from the successful NWTN test that completed
with result IDs from the log files.
"""

import asyncio
import sys
import json
from datetime import datetime
sys.path.insert(0, '.')

from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
from prsm.nwtn.external_storage_config import get_external_knowledge_base
from prsm.nwtn.breakthrough_modes import BreakthroughMode

async def extract_completed_results():
    """Extract synthesis results from the completed reasoning cycles"""
    
    print("🔍 EXTRACTING NWTN SYNTHESIS RESULTS")
    print("=" * 60)
    print("📋 Result IDs from successful runs:")
    print("   Conservative: 2e4875e6-fcdf-4173-9d4b-721a51a04368")
    print("   Revolutionary: 4e0e0e48-6d4a-4b1e-94ab-9e2eee4e6db9")
    print("=" * 60)
    print()
    
    # Our test prompt from the completed run
    test_prompt = ("What are the most promising theoretical approaches to unifying quantum mechanics "
                  "and general relativity, and what experimental evidence exists to support or refute these approaches?")
    
    # Initialize External Knowledge Base  
    print("📚 Initializing external knowledge base...")
    knowledge_base = await get_external_knowledge_base()
    print("✅ Knowledge base ready")
    print()
    
    # Initialize MetaReasoningEngine
    print("🧠 Initializing NWTN MetaReasoningEngine...")
    meta_engine = MetaReasoningEngine()
    await meta_engine.initialize_external_knowledge_base()
    print("✅ MetaReasoningEngine ready")
    print()
    
    # Re-run with same parameters to get clean synthesis results
    print("🎯 RE-EXTRACTING CONSERVATIVE SYNTHESIS")
    print("-" * 50)
    
    conservative_start_time = datetime.now()
    
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
        
        print("🚀 Running CONSERVATIVE mode with DEEP reasoning...")
        conservative_result = await meta_engine.meta_reason(
            query=test_prompt,
            context=conservative_context,
            thinking_mode=ThinkingMode.DEEP
        )
        conservative_time = (datetime.now() - conservative_start_time).total_seconds()
        
        print(f"✅ CONSERVATIVE complete ({conservative_time:.2f}s)")
        print(f"🎯 Meta confidence: {conservative_result.meta_confidence:.3f}")
        print(f"📊 Quality score: {conservative_result.quality_score:.3f}")
        print(f"💰 FTNS cost: {conservative_result.ftns_cost:.2f} tokens")
        print()
        
        # Extract and save synthesis
        if hasattr(conservative_result, 'final_synthesis') and conservative_result.final_synthesis:
            print("📄 CONSERVATIVE SYNTHESIS EXTRACTED:")
            print("=" * 50)
            if isinstance(conservative_result.final_synthesis, dict):
                synthesis_text = str(conservative_result.final_synthesis)
            else:
                synthesis_text = conservative_result.final_synthesis
            print(synthesis_text)
            print("=" * 50)
            print()
            
            # Save to file
            with open("nwtn_conservative_synthesis.txt", "w") as f:
                f.write(f"NWTN Conservative Mode Synthesis - Quantum Gravity Unification\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Meta Confidence: {conservative_result.meta_confidence:.3f}\n")
                f.write(f"Quality Score: {conservative_result.quality_score:.3f}\n")
                f.write(f"Processing Time: {conservative_time:.2f}s\n")
                f.write(f"FTNS Cost: {conservative_result.ftns_cost:.2f} tokens\n")
                f.write("=" * 60 + "\n\n")
                f.write(synthesis_text)
            
            print("💾 Conservative synthesis saved to nwtn_conservative_synthesis.txt")
        
    except Exception as e:
        print(f"❌ Conservative extraction failed: {e}")
    
    print()
    
    # Re-run revolutionary mode
    print("🎯 RE-EXTRACTING REVOLUTIONARY SYNTHESIS")
    print("-" * 50)
    
    revolutionary_start_time = datetime.now()
    
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
        
        print("🚀 Running REVOLUTIONARY mode with DEEP reasoning...")
        revolutionary_result = await meta_engine.meta_reason(
            query=test_prompt,
            context=revolutionary_context,
            thinking_mode=ThinkingMode.DEEP
        )
        revolutionary_time = (datetime.now() - revolutionary_start_time).total_seconds()
        
        print(f"✅ REVOLUTIONARY complete ({revolutionary_time:.2f}s)")
        print(f"🎯 Meta confidence: {revolutionary_result.meta_confidence:.3f}")
        print(f"📊 Quality score: {revolutionary_result.quality_score:.3f}")
        print(f"💰 FTNS cost: {revolutionary_result.ftns_cost:.2f} tokens")
        print()
        
        # Extract and save synthesis
        if hasattr(revolutionary_result, 'final_synthesis') and revolutionary_result.final_synthesis:
            print("📄 REVOLUTIONARY SYNTHESIS EXTRACTED:")
            print("=" * 50)
            if isinstance(revolutionary_result.final_synthesis, dict):
                synthesis_text = str(revolutionary_result.final_synthesis)
            else:
                synthesis_text = revolutionary_result.final_synthesis
            print(synthesis_text)
            print("=" * 50)
            print()
            
            # Save to file
            with open("nwtn_revolutionary_synthesis.txt", "w") as f:
                f.write(f"NWTN Revolutionary Mode Synthesis - Quantum Gravity Unification\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Meta Confidence: {revolutionary_result.meta_confidence:.3f}\n")
                f.write(f"Quality Score: {revolutionary_result.quality_score:.3f}\n")
                f.write(f"Processing Time: {revolutionary_time:.2f}s\n")
                f.write(f"FTNS Cost: {revolutionary_result.ftns_cost:.2f} tokens\n")
                f.write("=" * 60 + "\n\n")
                f.write(synthesis_text)
            
            print("💾 Revolutionary synthesis saved to nwtn_revolutionary_synthesis.txt")
        
    except Exception as e:
        print(f"❌ Revolutionary extraction failed: {e}")
    
    print()
    print("🎉 SYNTHESIS EXTRACTION COMPLETE")
    print("📄 Results saved to individual files for analysis")

if __name__ == "__main__":
    asyncio.run(extract_completed_results())