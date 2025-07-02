#!/usr/bin/env python3
"""
PRSM Tutorial 1: Setup and First Query

This tutorial walks you through setting up the PRSM Python SDK
and making your first AI query. Perfect for beginners!

📚 What you'll learn:
- How to install and configure the PRSM SDK
- Basic authentication and API key setup
- Making your first query to an AI model
- Understanding responses and costs
- Basic error handling
"""

import asyncio
import os
from pathlib import Path

# Check if PRSM SDK is installed
try:
    from prsm_sdk import PRSMClient, PRSMError
    print("✅ PRSM SDK is installed and ready!")
except ImportError:
    print("❌ PRSM SDK not found. Please install it first:")
    print("   pip install prsm-python-sdk")
    exit(1)


def check_environment_setup():
    """Check if environment is properly configured"""
    print("\n🔧 Checking environment setup...")
    
    # Check for API key
    api_key = os.getenv("PRSM_API_KEY")
    if not api_key:
        print("❌ PRSM_API_KEY not found in environment variables")
        print("\n📋 To get started:")
        print("1. Sign up at https://prsm.ai")
        print("2. Get your API key from the dashboard")
        print("3. Set environment variable:")
        print("   export PRSM_API_KEY='your_api_key_here'")
        print("   # Or on Windows: set PRSM_API_KEY=your_api_key_here")
        return False
    
    print(f"✅ API key found (starts with: {api_key[:8]}...)")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        print(f"⚠️ Python {sys.version} detected. Python 3.8+ recommended")
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    
    return True


async def first_query_example():
    """Your first PRSM query - Simple text generation"""
    print("\n🚀 Making your first PRSM query...")
    
    # Initialize the PRSM client
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        # Make a simple query
        print("📤 Sending query: 'Explain what PRSM is in simple terms'")
        
        result = await client.models.infer(
            model="gpt-4",  # Using GPT-4 for high quality
            prompt="Explain what PRSM (Protocol for Recursive Scientific Modeling) is in simple terms",
            max_tokens=150,  # Limit response length
            temperature=0.7  # Some creativity, but not too much
        )
        
        # Display the results
        print("\n📥 Response received!")
        print(f"🤖 Model used: {result.model}")
        print(f"💬 Response: {result.content}")
        print(f"📊 Tokens used: {result.usage.total_tokens}")
        print(f"💰 Cost: ${result.cost:.4f}")
        
        return result
        
    except PRSMError as e:
        print(f"❌ PRSM Error: {e.message}")
        print("💡 This might be due to:")
        print("   - Invalid API key")
        print("   - Insufficient credits")
        print("   - Network connectivity issues")
        return None
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None
    
    finally:
        # Always close the client to clean up resources
        await client.close()


async def explore_different_models():
    """Try different AI models to see their characteristics"""
    print("\n🔬 Exploring different AI models...")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    # Different models to try
    models_to_test = [
        ("gpt-3.5-turbo", "Fast and cost-effective"),
        ("gpt-4", "Most capable, higher cost"),
        ("claude-3", "Great for analysis and reasoning")
    ]
    
    prompt = "Write a haiku about artificial intelligence"
    
    results = []
    
    for model_name, description in models_to_test:
        try:
            print(f"\n🤖 Testing {model_name} ({description})...")
            
            result = await client.models.infer(
                model=model_name,
                prompt=prompt,
                max_tokens=50,
                temperature=0.8  # Higher creativity for poetry
            )
            
            print(f"📝 Response:\n{result.content}")
            print(f"💰 Cost: ${result.cost:.4f}")
            print(f"⚡ Tokens: {result.usage.total_tokens}")
            
            results.append({
                "model": model_name,
                "response": result.content,
                "cost": result.cost,
                "tokens": result.usage.total_tokens
            })
            
        except PRSMError as e:
            print(f"❌ Error with {model_name}: {e.message}")
            continue
    
    # Compare results
    if results:
        print("\n📊 Model Comparison Summary:")
        total_cost = sum(r["cost"] for r in results)
        print(f"💰 Total cost for comparison: ${total_cost:.4f}")
        
        cheapest = min(results, key=lambda x: x["cost"])
        print(f"💸 Most cost-effective: {cheapest['model']} (${cheapest['cost']:.4f})")
    
    await client.close()
    return results


async def understanding_parameters():
    """Learn how different parameters affect AI responses"""
    print("\n🎛️ Understanding query parameters...")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    base_prompt = "Describe the benefits of renewable energy"
    
    # Test different temperature settings
    temperatures = [0.1, 0.5, 1.0]
    
    print("🌡️ Testing different temperature settings:")
    print("(Temperature controls creativity: 0.0 = deterministic, 1.0+ = very creative)")
    
    for temp in temperatures:
        try:
            print(f"\n🔥 Temperature: {temp}")
            
            result = await client.models.infer(
                model="gpt-3.5-turbo",
                prompt=base_prompt,
                max_tokens=100,
                temperature=temp
            )
            
            print(f"📄 Response: {result.content[:150]}...")
            print(f"💰 Cost: ${result.cost:.4f}")
            
        except PRSMError as e:
            print(f"❌ Error at temperature {temp}: {e.message}")
            continue
    
    # Test different max_tokens settings
    print("\n📏 Testing different max_tokens settings:")
    print("(max_tokens controls response length)")
    
    token_limits = [50, 150, 300]
    
    for max_tokens in token_limits:
        try:
            print(f"\n📊 Max tokens: {max_tokens}")
            
            result = await client.models.infer(
                model="gpt-3.5-turbo",
                prompt="Write a comprehensive guide to machine learning",
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            print(f"📄 Response length: {len(result.content)} characters")
            print(f"🔢 Actual tokens used: {result.usage.total_tokens}")
            print(f"💰 Cost: ${result.cost:.4f}")
            
        except PRSMError as e:
            print(f"❌ Error with {max_tokens} tokens: {e.message}")
            continue
    
    await client.close()


def save_tutorial_progress():
    """Save progress and provide next steps"""
    progress_file = Path.home() / ".prsm_tutorial_progress.txt"
    
    with open(progress_file, "w") as f:
        f.write("PRSM Tutorial Progress\n")
        f.write("====================\n")
        f.write("✅ Tutorial 1: Setup and First Query - COMPLETED\n")
        f.write("⏳ Tutorial 2: Understanding FTNS Tokens - NEXT\n")
        f.write("⏳ Tutorial 3: Basic Error Handling - PENDING\n")
        f.write("⏳ Tutorial 4: Cost Optimization - PENDING\n")
    
    print(f"\n💾 Progress saved to: {progress_file}")


async def main():
    """Main tutorial function"""
    print("🎓 PRSM Tutorial 1: Setup and First Query")
    print("=" * 50)
    
    # Step 1: Check environment
    if not check_environment_setup():
        print("\n❌ Environment setup incomplete. Please fix the issues above and try again.")
        return
    
    # Step 2: First query
    result = await first_query_example()
    if not result:
        print("\n❌ First query failed. Please check your setup and try again.")
        return
    
    # Step 3: Explore models
    await explore_different_models()
    
    # Step 4: Learn parameters
    await understanding_parameters()
    
    # Step 5: Save progress
    save_tutorial_progress()
    
    print("\n" + "=" * 50)
    print("🎉 Tutorial 1 completed successfully!")
    print("\n📚 What you learned:")
    print("✅ How to set up and configure PRSM SDK")
    print("✅ Making basic queries to AI models")
    print("✅ Understanding response structure and costs")
    print("✅ Comparing different AI models")
    print("✅ How temperature and max_tokens affect responses")
    
    print("\n⏭️ Next steps:")
    print("1. Run Tutorial 2: Understanding FTNS Tokens")
    print("2. Experiment with different prompts and models")
    print("3. Check out the cost optimization features")
    
    print("\n💡 Pro tips:")
    print("• Start with gpt-3.5-turbo for cost-effective testing")
    print("• Use lower temperatures (0.1-0.3) for factual tasks")
    print("• Use higher temperatures (0.7-1.0) for creative tasks")
    print("• Monitor your costs with budget management features")


if __name__ == "__main__":
    # Run the tutorial
    asyncio.run(main())