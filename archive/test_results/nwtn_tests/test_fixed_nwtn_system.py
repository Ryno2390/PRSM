#!/usr/bin/env python3
"""
Test the fixed NWTN system with all error corrections
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_fixed_nwtn_system():
    """Test the fixed NWTN system"""
    print("🧪 Testing fixed NWTN system...")
    
    # Set Claude API key
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    
    # Override default NWTN model to use Claude instead of GPT-4
    os.environ["PRSM_NWTN_MODEL"] = "claude-3-5-sonnet-20241022"
    
    try:
        # Test 1: Import and initialize voicebox
        print("🔧 Testing voicebox initialization...")
        from prsm.nwtn.voicebox import get_voicebox_service, LLMProvider
        
        voicebox = await get_voicebox_service()
        await voicebox.initialize()
        
        # Configure API key
        await voicebox.configure_api_key(
            user_id="test_user_001",
            provider=LLMProvider.CLAUDE,
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        
        # Add sufficient FTNS balance
        from prsm.tokenomics.ftns_service import get_ftns_service
        ftns_service = await get_ftns_service()
        await ftns_service.reward_contribution("test_user_001", "data", 2000.0)
        
        print("✅ Voicebox initialized successfully")
        
        # Test 2: Test simple query processing
        print("🔍 Testing simple query processing...")
        
        test_query = "What is the fundamental principle of quantum mechanics?"
        
        # Create simple content sources
        content_sources = [
            {
                "content_id": "quantum_001",
                "title": "Quantum Mechanics Basics",
                "content": "Quantum mechanics is based on the principle of superposition, where particles can exist in multiple states simultaneously until measured.",
                "source": "test",
                "creator_id": "test_creator_001"
            },
            {
                "content_id": "quantum_002",
                "title": "Wave-Particle Duality",
                "content": "Fundamental to quantum mechanics is wave-particle duality, where matter and energy exhibit both wave and particle properties.",
                "source": "test",
                "creator_id": "test_creator_002"
            }
        ]
        
        # Process query through voicebox
        response = await voicebox.process_query(
            user_id="test_user_001",
            query=test_query,
            context={"content_sources": content_sources}
        )
        
        print("\n" + "="*80)
        print("🎯 NWTN SYSTEM RESPONSE:")
        print("="*80)
        print(f"📝 Natural Language Response: {response.natural_language_response}")
        print(f"🧠 Reasoning Engines Used: {response.used_reasoning_modes}")
        print(f"🎯 Confidence Score: {response.confidence_score}")
        print(f"⏱️  Processing Time: {response.processing_time_seconds} seconds")
        print(f"🔗 Source Links: {response.source_links}")
        print(f"📊 Attribution Summary: {response.attribution_summary}")
        print("="*80)
        
        # Check if we got a real response (not fallback)
        if (response.natural_language_response and 
            len(response.natural_language_response) > 100 and
            response.confidence_score > 0.0 and
            response.processing_time_seconds > 0.5):
            print("✅ Successfully received full NWTN system response with Claude API!")
            return True
        else:
            print("❌ Response appears to be fallback or incomplete")
            print(f"   - Response length: {len(response.natural_language_response)}")
            print(f"   - Confidence: {response.confidence_score}")
            print(f"   - Processing time: {response.processing_time_seconds}")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fixed_nwtn_system())
    if success:
        print("\n🎉 Fixed NWTN system test successful!")
    else:
        print("\n🚨 Fixed NWTN system test failed!")