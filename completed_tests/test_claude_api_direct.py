#!/usr/bin/env python3
"""
Direct Claude API Integration Test
===============================

Tests Claude API integration directly without full NWTN initialization.
"""

import asyncio
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Simple test setup
class LLMProvider(str, Enum):
    CLAUDE = "claude"

@dataclass
class APIConfiguration:
    provider: LLMProvider
    api_key: str
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30

async def test_claude_api_direct():
    """Test Claude API integration directly"""
    print("🎯 DIRECT CLAUDE API INTEGRATION TEST")
    print("=" * 50)
    
    # Check environment variable
    claude_api_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY')
    
    if not claude_api_key:
        print("❌ No Claude API key found in environment variables")
        return
    
    print(f"✅ Claude API key found: {claude_api_key[:20]}...")
    
    # Create API configuration
    api_config = APIConfiguration(
        provider=LLMProvider.CLAUDE,
        api_key=claude_api_key,
        model_name="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        temperature=0.7,
        timeout=30
    )
    
    print(f"🤖 Using model: {api_config.model_name}")
    
    # Test prompt
    prompt = """You are the natural language interface for NWTN, the world's most advanced multi-modal reasoning AI system. NWTN has just completed sophisticated analysis of a user's query using all 7 fundamental forms of reasoning.

Your task is to translate NWTN's structured insights into a clear, helpful natural language response that directly addresses the user's question.

ORIGINAL USER QUERY:
What is quantum computing?

NWTN'S STRUCTURED ANALYSIS:
- Confidence Score: 0.8
- Reasoning Modes Used: deductive, analogical
- Primary Insights: Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information.
- Supporting Evidence: Quantum bits (qubits) can exist in multiple states simultaneously, Quantum algorithms can solve certain problems exponentially faster
- Uncertainty Factors: Hardware limitations, Error rates

REASONING TRACE:
[{"engine": "test", "confidence": 0.8}]

INSTRUCTIONS:
1. Provide a clear, direct answer to the user's question
2. Explain the key insights in accessible language
3. Mention the reasoning approach used (but don't over-explain the technical details)
4. If there are important uncertainties or limitations, mention them
5. Be conversational but authoritative
6. Focus on practical value and actionable insights

Generate a natural language response that makes NWTN's sophisticated reasoning accessible to the user:"""

    # Call Claude API
    try:
        import aiohttp
        import json
        
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_config.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": api_config.model_name,
            "max_tokens": api_config.max_tokens,
            "temperature": api_config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        timeout = aiohttp.ClientTimeout(total=api_config.timeout)
        
        print("🚀 Making Claude API call...")
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    natural_response = result["content"][0]["text"]
                    
                    print("✨ CLAUDE API NATURAL LANGUAGE RESPONSE:")
                    print("=" * 60)
                    print(natural_response)
                    print("=" * 60)
                    print("🎉 SUCCESS: Claude API integration is working perfectly!")
                    
                else:
                    error_text = await response.text()
                    print(f"❌ Claude API error {response.status}: {error_text}")
                    
    except ImportError:
        print("❌ aiohttp not available for Claude API calls")
        print("📦 Install with: pip install aiohttp")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_claude_api_direct())