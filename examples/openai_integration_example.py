#!/usr/bin/env python3
"""
OpenAI Integration Example
=========================

Simple example showing how to use the enhanced OpenAI client
in your PRSM applications.
"""

import asyncio
import os
from pathlib import Path
import sys

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.agents.executors.api_clients import (
    OpenAIClient, 
    ModelExecutionRequest, 
    ModelProvider
)


async def basic_example():
    """Basic usage example"""
    
    # Get API key (you'd set this in your environment)
    api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
    
    # Create and initialize client
    client = OpenAIClient(api_key)
    await client.initialize()
    
    try:
        # Create a request
        request = ModelExecutionRequest(
            prompt="Explain quantum computing in simple terms",
            model_id="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
            max_tokens=150,
            temperature=0.7,
            system_prompt="You are a helpful science teacher"
        )
        
        # Execute the request
        response = await client.execute(request)
        
        # Handle the response
        if response.success:
            print(f"✅ Response: {response.content}")
            print(f"💰 Tokens used: {response.token_usage}")
            print(f"⏱️ Execution time: {response.execution_time:.2f}s")
        else:
            print(f"❌ Error: {response.error}")
    
    finally:
        await client.close()


async def enhanced_example():
    """Example using the enhanced client with cost tracking"""
    
    try:
        # This would use the enhanced client (requires tenacity package)
        from prsm.agents.executors.enhanced_openai_client import create_enhanced_openai_client
        
        api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
        
        # Create enhanced client with budget limit
        client = await create_enhanced_openai_client(
            api_key=api_key,
            budget_limit_usd=1.00,  # $1 budget limit
            enable_cost_tracking=True,
            enable_rate_limiting=True
        )
        
        try:
            # Make multiple requests to demonstrate cost tracking
            prompts = [
                "What is machine learning?",
                "Explain neural networks briefly",
                "What is deep learning?"
            ]
            
            total_cost = 0
            
            for i, prompt in enumerate(prompts, 1):
                request = ModelExecutionRequest(
                    prompt=prompt,
                    model_id="gpt-3.5-turbo",
                    provider=ModelProvider.OPENAI,
                    max_tokens=100,
                    temperature=0.5
                )
                
                response = await client.execute(request)
                
                if response.success:
                    cost = response.metadata.get('cost_usd', 0)
                    total_cost += cost
                    
                    print(f"✅ Request {i}: ${cost:.4f}")
                    print(f"   Response: {response.content[:100]}...")
                else:
                    print(f"❌ Request {i} failed: {response.error}")
            
            # Get usage summary
            summary = await client.get_usage_summary()
            print(f"\n📊 Total cost: ${summary['total_cost_usd']:.4f}")
            print(f"📊 Budget remaining: ${summary['budget_remaining_usd']:.4f}")
            print(f"📊 Requests made: {summary['total_requests']}")
        
        finally:
            await client.close()
    
    except ImportError:
        print("⚠️ Enhanced client requires 'tenacity' package")
        print("Install with: pip install tenacity")
        await basic_example()


async def prsm_integration_example():
    """Example showing integration with PRSM's agent architecture"""
    
    print("🏗️ PRSM Agent Integration Example")
    print("=" * 40)
    
    # This shows how the OpenAI client integrates with PRSM's
    # multi-agent architecture
    
    # In a real PRSM agent, you'd have:
    # 1. Architect Agent - breaks down complex tasks
    # 2. Prompter Agent - optimizes prompts for specific models
    # 3. Router Agent - routes to appropriate model/provider
    # 4. Executor Agent - uses our OpenAI client to execute
    # 5. Compiler Agent - combines and formats results
    
    print("Agent Pipeline Flow:")
    print("  Architect → Prompter → Router → [OpenAI Client] → Compiler")
    print("\nOpenAI Client Features:")
    print("  ✅ Async execution for non-blocking pipeline")
    print("  ✅ Cost tracking integrated with FTNS tokens")
    print("  ✅ Error handling with retry logic")
    print("  ✅ Performance monitoring for optimization")
    print("  ✅ Multi-model support (GPT-4, GPT-3.5-turbo)")


def code_structure_demo():
    """Show the code structure and key components"""
    
    print("\n📁 OpenAI Integration Structure")
    print("=" * 40)
    
    structure = """
prsm/agents/executors/
├── api_clients.py              # Base OpenAI client (production-ready)
├── enhanced_openai_client.py   # Enhanced client with cost/retry features
└── [future: anthropic_client.py, local_client.py]

scripts/
├── setup_openai_api.py         # Easy API key setup
├── test_openai_integration.py  # Basic integration tests
├── test_enhanced_openai_client.py # Comprehensive test suite
└── demo_openai_integration.py  # This demo

examples/
└── openai_integration_example.py # Usage examples

docs/
└── OPENAI_INTEGRATION_SUMMARY.md # Complete documentation
"""
    
    print(structure)
    
    print("\n🔧 Key Components:")
    print("  • ModelExecutionRequest - Standard PRSM interface")
    print("  • ModelExecutionResponse - Rich response with metadata")
    print("  • CostTracker - Real-time cost monitoring")
    print("  • RateLimiter - Prevents API quota exhaustion")
    print("  • Retry Logic - Exponential backoff for reliability")


async def main():
    """Run all examples"""
    
    print("🎭 OpenAI Integration Examples for PRSM")
    print("=" * 50)
    
    print("\n📝 Note: These examples use mock responses since no API key is configured")
    print("To test with real OpenAI API, set OPENAI_API_KEY environment variable\n")
    
    # Show code structure
    code_structure_demo()
    
    # Show PRSM integration
    await prsm_integration_example()
    
    # Run basic example (with mock client if no API key)
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != 'your-api-key-here':
        print("\n🚀 Running with real OpenAI API...")
        await enhanced_example()
    else:
        print("\n🎭 Would run with real OpenAI API if OPENAI_API_KEY was set")
        print("Example request structure:")
        print("""
        request = ModelExecutionRequest(
            prompt="Explain quantum computing in simple terms",
            model_id="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
            max_tokens=150,
            temperature=0.7,
            system_prompt="You are a helpful science teacher"
        )
        
        response = await client.execute(request)
        # → Gets real response from OpenAI API
        # → Tracks cost automatically
        # → Retries on failures
        # → Monitors performance
        """)


if __name__ == '__main__':
    asyncio.run(main())