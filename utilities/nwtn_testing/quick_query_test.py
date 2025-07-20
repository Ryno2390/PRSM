#!/usr/bin/env python3
"""
Quick test to verify Claude API integration with corrected API key configuration
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from prsm.nwtn.voicebox import get_voicebox_service, LLMProvider

async def test_claude_api_with_fixed_config():
    """Test Claude API integration with proper user configuration"""
    
    # Set Claude API key in environment
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    
    # Initialize voicebox
    voicebox = await get_voicebox_service()
    await voicebox.initialize()
    
    # Give user sufficient FTNS balance
    from prsm.tokenomics.ftns_service import get_ftns_service
    ftns_service = await get_ftns_service()
    await ftns_service.reward_contribution("test_user", "data", 1000.0)  # Give 1000 FTNS
    
    # Check balance
    balance = await ftns_service.get_balance("test_user")
    print(f"User FTNS balance: {balance}")
    
    # Configure API key for user
    await voicebox.configure_api_key(
        user_id="test_user",
        provider=LLMProvider.CLAUDE,
        api_key="your-api-key-here"
    )
    
    # Test query
    query = "What are the latest advances in machine learning and neural networks?"
    
    print("🔮 Testing Claude API integration with fixed configuration...")
    print(f"Query: {query}")
    
    try:
        # Generate response using voicebox
        response = await voicebox.process_query(
            user_id="test_user",
            query=query
        )
        
        print("\n✅ Claude API integration successful!")
        print(f"Response: {response.response}")
        print(f"Source links: {len(response.source_links)}")
        print(f"Attribution: {response.attribution_summary}")
        print(f"Confidence: {response.confidence_score}")
        print(f"Processing time: {response.processing_time}s")
        print(f"Reasoning engines used: {response.reasoning_engines_used}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Claude API integration failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_claude_api_with_fixed_config())