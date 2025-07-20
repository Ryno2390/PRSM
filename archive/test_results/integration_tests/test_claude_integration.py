#!/usr/bin/env python3
"""
Test script to verify Claude API integration with NWTN voicebox
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from prsm.nwtn.voicebox import NWTNVoicebox, get_voicebox_service
from prsm.nwtn.voicebox import QueryComplexity

async def test_claude_integration():
    """Test Claude API integration with voicebox"""
    
    # Set Claude API key
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    
    # Initialize voicebox
    voicebox = await get_voicebox_service()
    await voicebox.initialize()
    
    # Configure API key for user
    from prsm.nwtn.voicebox import LLMProvider
    await voicebox.configure_api_key(
        user_id="test_user",
        provider=LLMProvider.CLAUDE,
        api_key="your-api-key-here"
    )
    
    # Initialize FTNS balance for test user
    from prsm.tokenomics.ftns_service import get_ftns_service
    ftns_service = await get_ftns_service()
    await ftns_service._update_balance("test_user", 100.0)  # Give user 100 FTNS for testing
    
    # Test query
    query = "What are the latest advances in machine learning and neural networks?"
    
    print("🔮 Testing Claude API integration with NWTN voicebox...")
    print(f"Query: {query}")
    
    try:
        # Generate response using voicebox
        response = await voicebox.process_query(
            user_id="test_user",
            query=query
        )
        
        print("\n✅ Claude API integration successful!")
        print(f"Response: {response.natural_language_response}")
        print(f"Source links: {len(response.source_links)}")
        print(f"Attribution: {response.attribution_summary}")
        print(f"Confidence: {response.confidence_score}")
        print(f"Processing time: {response.processing_time_seconds}s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Claude API integration failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_claude_integration())