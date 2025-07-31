#!/usr/bin/env python3
"""
Direct Claude API Test
======================
Tests the model executor directly to see if Claude response is being extracted correctly
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from prsm.agents.executors.model_executor import ModelExecutor

print("🔬 DIRECT CLAUDE API TEST")
print("=" * 40)

async def test_direct_claude():
    """Test Claude API directly through model executor"""
    
    # Simple test prompt
    test_prompt = "What is the capital of France? Answer in one sentence."
    
    print("📝 TEST PROMPT:")
    print(f"   {test_prompt}")
    print()
    
    try:
        # Load Claude API key
        api_key_path = "/Users/ryneschultz/Documents/GitHub/Anthropic_API_Key.txt"
        with open(api_key_path, 'r') as f:
            claude_api_key = f.read().strip()
        
        print("✅ Claude API key loaded")
        
        # Initialize model executor directly
        print("🔧 Initializing ModelExecutor...")
        executor = ModelExecutor()
        print("✅ ModelExecutor initialized")
        
        # Prepare input data for Claude
        input_data = {
            "task": test_prompt,
            "models": ["claude-3-5-sonnet-20241022"],
            "parallel": False,
            "temperature": 0.7,
            "api_key": claude_api_key
        }
        
        print("🚀 Executing Claude API call...")
        print("   Model: claude-3-5-sonnet-20241022")
        print("   Temperature: 0.7")
        print()
        
        # Execute directly
        results = await executor.process(input_data)
        
        print("🎉 CLAUDE API EXECUTION COMPLETE!")
        print("=" * 40)
        
        if results and len(results) > 0:
            result = results[0]
            print("📋 EXECUTION RESULT:")
            print("-" * 20)
            print(f"✅ Success: {result.success}")
            print(f"✅ Model ID: {result.model_id}")
            print(f"✅ Execution Time: {result.execution_time:.3f}s")
            print()
            
            if result.success and result.result:
                print("📄 CLAUDE RESPONSE DATA:")
                print("-" * 25)
                print(f"Result type: {type(result.result)}")
                print(f"Result keys: {result.result.keys() if isinstance(result.result, dict) else 'Not a dict'}")
                print()
                
                if isinstance(result.result, dict):
                    content = result.result.get('content', '')
                    if content:
                        print("🎯 CLAUDE NATURAL LANGUAGE RESPONSE:")
                        print("-" * 38)
                        print(content)
                        print()
                        print("✅ SUCCESS: Claude response extracted successfully!")
                        print(f"   Response length: {len(content)} characters")
                        return True
                    else:
                        print("❌ FAILURE: No 'content' key in result")
                        print(f"   Available keys: {list(result.result.keys())}")
                        return False
                else:
                    print("❌ FAILURE: Result is not a dictionary")
                    print(f"   Result value: {result.result}")
                    return False
            else:
                print("❌ FAILURE: Execution unsuccessful or no result")
                if result.error:
                    print(f"   Error: {result.error}")
                return False
        else:
            print("❌ CRITICAL FAILURE: No results returned from executor")
            return False
        
    except Exception as e:
        print(f"❌ Direct Claude test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_direct_claude())
    if success:
        print("\n🎉 DIRECT CLAUDE TEST: PASSED")
    else:
        print("\n💥 DIRECT CLAUDE TEST: FAILED")