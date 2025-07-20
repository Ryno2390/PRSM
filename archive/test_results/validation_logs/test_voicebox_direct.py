#!/usr/bin/env python3
"""
Direct test of NWTN Voicebox with Claude API
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_voicebox_direct():
    """Test voicebox directly with Claude API"""
    print("🔧 Testing NWTN Voicebox with Claude API...")
    
    # Set Claude API key
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    
    try:
        # Import and initialize voicebox
        from prsm.nwtn.voicebox import get_voicebox_service, LLMProvider
        
        voicebox = await get_voicebox_service()
        await voicebox.initialize()
        
        # Configure API key for test user
        await voicebox.configure_api_key(
            user_id="test_user_001",
            provider=LLMProvider.CLAUDE,
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        
        # Add FTNS balance for test user (increased to account for fees/taxes)
        from prsm.tokenomics.ftns_service import get_ftns_service
        ftns_service = await get_ftns_service()
        await ftns_service.reward_contribution("test_user_001", "data", 2000.0)
        
        print("✅ Voicebox initialized successfully with FTNS balance")
        
        # Test query
        test_query = "What are the key principles of quantum mechanics?"
        
        # Create mock content sources
        content_sources = [
            {
                "content_id": "test_content_001",
                "title": "Introduction to Quantum Mechanics",
                "content": "Quantum mechanics is the fundamental theory in physics that describes the behavior of matter and energy at the atomic and subatomic scale. Key principles include wave-particle duality, uncertainty principle, and superposition.",
                "source": "test",
                "creator_id": "test_creator_001"
            },
            {
                "content_id": "test_content_002", 
                "title": "Quantum Entanglement",
                "content": "Quantum entanglement is a phenomenon where particles become interconnected and the state of one particle instantly affects the state of another, regardless of distance.",
                "source": "test",
                "creator_id": "test_creator_002"
            }
        ]
        
        print(f"🔍 Processing query: {test_query}")
        
        # Process query through voicebox
        response = await voicebox.process_query(
            user_id="test_user_001",
            query=test_query,
            context={"content_sources": content_sources}
        )
        
        print("\n" + "="*80)
        print("🎯 NWTN VOICEBOX RESPONSE:")
        print("="*80)
        print(f"📝 Natural Language Response:\n{response.natural_language_response}")
        print(f"\n🧠 Reasoning Engines Used: {response.used_reasoning_modes}")
        print(f"🎯 Confidence Score: {response.confidence_score}")
        print(f"⏱️  Processing Time: {response.processing_time_seconds} seconds")
        print(f"🔗 Source Links: {response.source_links}")
        print(f"📊 Attribution Summary: {response.attribution_summary}")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_voicebox_direct())
    if success:
        print("\n🎉 Direct voicebox test completed successfully!")
    else:
        print("\n🚨 Direct voicebox test failed!")