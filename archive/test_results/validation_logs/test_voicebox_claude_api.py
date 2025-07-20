#!/usr/bin/env python3
"""
Test voicebox Claude API method directly
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_voicebox_claude_api():
    """Test voicebox Claude API method directly"""
    print("🧪 Testing voicebox Claude API method directly...")
    
    # Set Claude API key
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    
    try:
        # Import voicebox
        from prsm.nwtn.voicebox import get_voicebox_service, LLMProvider, APIConfiguration
        
        # Initialize voicebox
        voicebox = await get_voicebox_service()
        await voicebox.initialize()
        
        # Create API configuration
        api_config = APIConfiguration(
            provider=LLMProvider.CLAUDE,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_name="claude-3-5-sonnet-20241022",
            max_tokens=100,
            temperature=0.7,
            timeout=30.0
        )
        
        print("✅ API configuration created")
        
        # Test simple prompt
        prompt = "What is 2+2? Answer in one sentence."
        
        print(f"🔍 Testing prompt: {prompt}")
        
        # Call Claude API directly
        response = await voicebox._call_claude_api(api_config, prompt)
        
        print("\n" + "="*80)
        print("🎯 CLAUDE API RESPONSE:")
        print("="*80)
        print(f"📝 Response: {response}")
        print("="*80)
        
        # Check if we got a real response
        if response and len(response) > 10:
            print("✅ Successfully received response from Claude API!")
            return True
        else:
            print("❌ No response or empty response from Claude API")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_voicebox_claude_api())
    if success:
        print("\n🎉 Voicebox Claude API test successful!")
    else:
        print("\n🚨 Voicebox Claude API test failed!")