#!/usr/bin/env python3
"""
Test the voicebox integration with MetaReasoningResult
"""

import asyncio
import sys
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode

async def main():
    print("🧪 Testing voicebox integration with MetaReasoningResult")
    
    # Initialize engine
    engine = MetaReasoningEngine()
    
    # Simple test query
    query = "What are the key benefits of quantum computing for drug discovery?"
    
    print(f"📝 Query: {query}")
    print("🔄 Running NWTN meta-reasoning...")
    
    # Run reasoning
    result = await engine.meta_reason(
        query=query,
        context={'test': True},
        thinking_mode=ThinkingMode.QUICK
    )
    
    print(f"✅ Reasoning completed with confidence: {result.meta_confidence:.3f}")
    
    # Test voicebox response
    print("🎤 Testing voicebox response generation...")
    
    try:
        voicebox_response = await result.get_voicebox_response(query)
        print("✅ Voicebox response generated successfully!")
        print("📋 Response preview:")
        print("-" * 40)
        print(voicebox_response[:500] + "..." if len(voicebox_response) > 500 else voicebox_response)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Voicebox response failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🧪 Test complete!")

if __name__ == "__main__":
    asyncio.run(main())