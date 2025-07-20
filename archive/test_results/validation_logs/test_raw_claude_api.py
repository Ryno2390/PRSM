#!/usr/bin/env python3
"""
Test raw Claude API directly without NWTN
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_raw_claude_api():
    """Test raw Claude API directly"""
    print("🧪 Testing raw Claude API directly...")
    
    # Set Claude API key
    api_key = "your-api-key-here"
    
    try:
        # Test with anthropic library directly
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "What is 2+2? Answer in one sentence."}
            ]
        )
        
        print(f"✅ Raw Claude API Response: {message.content[0].text}")
        print(f"📊 Usage: {message.usage}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_raw_claude_api())
    if success:
        print("\n🎉 Raw Claude API test successful!")
    else:
        print("\n🚨 Raw Claude API test failed!")