#!/usr/bin/env python3
"""
Simple Claude Response Verification Test
========================================
Bypasses NWTN pipeline completely and uses ModelExecutor directly with known working input data structure
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from prsm.agents.executors.model_executor import ModelExecutor

print("⚡ SIMPLE CLAUDE RESPONSE TEST")
print("=" * 40)

async def test_simple_claude():
    """Test direct Claude API call using correct input structure"""
    
    test_prompt = "What is 2+2? Answer in one sentence."
    
    print(f"📝 Test prompt: {test_prompt}")
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
        
        # Prepare input data for Claude using EXACT format from working direct test
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
                        
                        # Check if this looks like a valid response to 2+2
                        if "4" in content or "four" in content.lower():
                            print("🎯 VALIDATION: Response contains correct answer!")
                            return True, content
                        else:
                            print("⚠️  VALIDATION: Response may not contain expected answer")
                            return True, content
                    else:
                        print("❌ FAILURE: No 'content' key in result")
                        print(f"   Available keys: {list(result.result.keys())}")
                        return False, None
                else:
                    print("❌ FAILURE: Result is not a dictionary")
                    print(f"   Result value: {result.result}")
                    return False, None
            else:
                print("❌ FAILURE: Execution unsuccessful or no result")
                if result.error:
                    print(f"   Error: {result.error}")
                return False, None
        else:
            print("❌ CRITICAL FAILURE: No results returned from executor")
            return False, None
        
    except Exception as e:
        print(f"❌ Simple Claude test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, response = asyncio.run(test_simple_claude())
    if success:
        print("\n🎉 SIMPLE CLAUDE TEST: PASSED ✅")
        print(f"📝 RESPONSE: '{response}'")
    else:
        print("\n💥 SIMPLE CLAUDE TEST: FAILED ❌")