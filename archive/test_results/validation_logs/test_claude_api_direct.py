#!/usr/bin/env python3
"""
Direct test of Claude API integration
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_claude_api_direct():
    """Test Claude API directly with voicebox"""
    print("🔧 Testing Claude API integration directly...")
    
    # Set Claude API key
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    
    try:
        # Initialize voicebox
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
        
        # Check balance
        balance = await ftns_service.get_balance("test_user_001")
        print(f"✅ FTNS balance: {balance} FTNS")
        
        # Test simple query without complex reasoning
        test_query = "What is 2+2?"
        
        # Create minimal content sources
        content_sources = [
            {
                "content_id": "test_content_001",
                "title": "Basic Math",
                "content": "Basic arithmetic: 2+2=4. This is fundamental mathematics.",
                "source": "test",
                "creator_id": "test_creator_001"
            }
        ]
        
        print(f"🔍 Testing simple query: {test_query}")
        
        # Process query through voicebox
        response = await voicebox.process_query(
            user_id="test_user_001",
            query=test_query,
            context={"content_sources": content_sources}
        )
        
        print("\n" + "="*80)
        print("🎯 CLAUDE API RESPONSE:")
        print("="*80)
        print(f"📝 Natural Language Response: {response.natural_language_response}")
        print(f"🧠 Reasoning Engines Used: {response.used_reasoning_modes}")
        print(f"🎯 Confidence Score: {response.confidence_score}")
        print(f"⏱️  Processing Time: {response.processing_time_seconds} seconds")
        print("="*80)
        
        # Check if we got a real response (not fallback)
        if response.natural_language_response and len(response.natural_language_response) > 50:
            print("✅ Successfully received natural language response from Claude API!")
            return True
        else:
            print("❌ Response seems to be fallback, not real Claude API response")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_claude_api_direct())
    if success:
        print("\n🎉 Claude API direct test completed successfully!")
    else:
        print("\n🚨 Claude API direct test failed!")