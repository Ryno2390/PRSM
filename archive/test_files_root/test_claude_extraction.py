#!/usr/bin/env python3
"""
Direct Claude Response Extraction Test
=====================================
Tests just the natural language response extraction from Claude without the full breakthrough pipeline
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
from prsm.core.models import UserInput

print("🧪 CLAUDE RESPONSE EXTRACTION TEST")
print("=" * 50)

async def test_claude_extraction():
    """Test direct Claude response extraction"""
    
    # Simple prompt to test Claude response
    test_prompt = "What is quantum computing in simple terms?"
    
    print("🔍 TEST QUERY:")
    print("-" * 20)
    print(f"Prompt: {test_prompt}")
    print()
    
    try:
        # Load Claude API key
        api_key_path = "/Users/ryneschultz/Documents/GitHub/Anthropic_API_Key.txt"
        with open(api_key_path, 'r') as f:
            claude_api_key = f.read().strip()
        
        print("✅ Claude API key loaded")
        
        # Create simple UserInput - no breakthrough mode
        user_input = UserInput(
            user_id="claude_test_user",
            prompt=test_prompt,
            context_allocation=200,  # Small allocation
            preferences={
                "reasoning_depth": "shallow",  # Skip complex reasoning
                "response_length": "brief",  # Short response
                "enable_cross_domain": False,  # No cross-domain analysis
                "api_key": claude_api_key,
                "disable_database_persistence": True,  # Test mode
                "bypass_breakthrough_pipeline": True  # Go straight to standard pipeline
            }
        )
        
        print("✅ UserInput created for simple test")
        print(f"   • Reasoning Depth: shallow")
        print(f"   • Response Length: brief")
        print(f"   • Bypass Breakthrough: True")
        print()
        
        # Initialize Enhanced NWTN Orchestrator
        print("🔧 Initializing Enhanced NWTN Orchestrator...")
        orchestrator = EnhancedNWTNOrchestrator()
        print("✅ Enhanced orchestrator initialized")
        
        # Execute simple query (should use standard pipeline)
        print("🚀 EXECUTING SIMPLE CLAUDE TEST...")
        print("=" * 40)
        
        response = await orchestrator.process_query(user_input=user_input)
        
        print("🎉 CLAUDE RESPONSE EXTRACTION TEST COMPLETE!")
        print("=" * 50)
        
        # Display results - focus on the actual response
        if response and hasattr(response, 'final_answer'):
            print("📋 CLAUDE NATURAL LANGUAGE RESPONSE:")
            print("-" * 40)
            print(response.final_answer)
            print()
            
            print("📊 RESPONSE METRICS:")
            print("-" * 20)
            print(f"✅ Response Length: {len(response.final_answer)} characters")
            print(f"✅ Has Content: {bool(response.final_answer.strip())}")
            print(f"✅ Confidence Score: {response.confidence_score:.2f}" if response.confidence_score else "✅ Confidence Score: Not available")
            print(f"✅ Context Used: {response.context_used} tokens")
            print(f"✅ Safety Validated: {response.safety_validated}")
            print()
            
            # Check if it's a real Claude response or fallback
            if len(response.final_answer.strip()) > 50 and "processed by the PRSM system" not in response.final_answer:
                print("🎯 SUCCESS: Actual Claude response detected!")
                return True
            else:
                print("⚠️  WARNING: Response appears to be fallback text, not actual Claude response")
                return False
                
        else:
            print("❌ CRITICAL FAILURE: No response generated")
            return False
        
    except Exception as e:
        print(f"❌ Claude extraction test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_claude_extraction())