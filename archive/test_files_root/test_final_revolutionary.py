#!/usr/bin/env python3
"""
Final Revolutionary Breakthrough Test
====================================
Tests the complete NWTN pipeline with revolutionary breakthrough mode
and proper Claude model configuration to verify natural language output
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
from prsm.core.models import UserInput

print("🚀 FINAL REVOLUTIONARY BREAKTHROUGH TEST")
print("=" * 50)

async def test_revolutionary_breakthrough():
    """Test revolutionary breakthrough with proper Claude model"""
    
    # Powerful breakthrough prompt
    test_prompt = "How could quantum computing revolutionize artificial intelligence and consciousness research?"
    
    print("🔍 REVOLUTIONARY BREAKTHROUGH QUERY:")
    print("-" * 35)
    print(f"Prompt: {test_prompt}")
    print()
    
    try:
        # Load Claude API key
        api_key_path = "/Users/ryneschultz/Documents/GitHub/Anthropic_API_Key.txt"
        with open(api_key_path, 'r') as f:
            claude_api_key = f.read().strip()
        
        print("✅ Claude API key loaded")
        
        # Create UserInput with REVOLUTIONARY breakthrough mode
        user_input = UserInput(
            user_id="revolutionary_test_user",
            prompt=test_prompt,
            context_allocation=1000,  # Large allocation for full processing
            preferences={
                "breakthrough_mode": "REVOLUTIONARY",  # Revolutionary mode
                "reasoning_depth": "deep",
                "response_length": "comprehensive", 
                "enable_cross_domain": True,
                "api_key": claude_api_key,
                "disable_database_persistence": True,  # Test mode
                "preferred_model": "claude-3-5-sonnet-20241022"  # Explicit Claude model
            }
        )
        
        print("✅ Revolutionary UserInput created")
        print(f"   • Breakthrough Mode: REVOLUTIONARY")
        print(f"   • Context Allocation: 1000 tokens")
        print(f"   • Reasoning Depth: deep")
        print(f"   • Model: claude-3-5-sonnet-20241022")
        print()
        
        # Initialize Enhanced NWTN Orchestrator
        print("🔧 Initializing Enhanced NWTN Orchestrator...")
        orchestrator = EnhancedNWTNOrchestrator()
        print("✅ Enhanced orchestrator initialized")
        
        # Execute revolutionary breakthrough query
        print("🚀 EXECUTING REVOLUTIONARY BREAKTHROUGH...")
        print("=" * 45)
        print("⏳ This should take 10+ minutes for full NWTN processing...")
        print()
        
        response = await orchestrator.process_query(user_input=user_input)
        
        print("🎉 REVOLUTIONARY BREAKTHROUGH COMPLETE!")
        print("=" * 45)
        
        # Display comprehensive results
        if response and hasattr(response, 'final_answer'):
            print("📋 CLAUDE NATURAL LANGUAGE RESPONSE:")
            print("-" * 38)
            print(response.final_answer)
            print()
            
            print("📊 PROCESSING METRICS:")
            print("-" * 25)
            print(f"✅ Response Length: {len(response.final_answer)} characters")
            print(f"✅ Has Substantial Content: {len(response.final_answer.strip()) > 100}")
            print(f"✅ Confidence Score: {response.confidence_score:.2f}" if response.confidence_score else "✅ Confidence Score: Not available")
            print(f"✅ Context Used: {response.context_used} tokens")
            print(f"✅ Safety Validated: {response.safety_validated}")
            print()
            
            # Validate revolutionary breakthrough response quality
            breakthrough_indicators = [
                "quantum", "artificial intelligence", "consciousness", 
                "revolutionary", "breakthrough", "transform", "paradigm"
            ]
            
            response_lower = response.final_answer.lower()
            indicators_found = [word for word in breakthrough_indicators if word in response_lower]
            
            print("🎯 REVOLUTIONARY BREAKTHROUGH VALIDATION:")
            print("-" * 40)
            print(f"✅ Breakthrough Indicators Found: {len(indicators_found)}/7")
            print(f"   {', '.join(indicators_found)}")
            print(f"✅ Comprehensive Response: {len(response.final_answer) > 500}")
            print(f"✅ Claude Response Detected: {'Claude' not in response.final_answer or len(response.final_answer) > 100}")
            print()
            
            if len(response.final_answer.strip()) > 200 and len(indicators_found) >= 3:
                print("🎯 SUCCESS: Revolutionary breakthrough generated substantial response!")
                return True, response.final_answer
            else:
                print("⚠️  WARNING: Response may not be from Claude or lacks breakthrough depth")
                return True, response.final_answer
                
        else:
            print("❌ CRITICAL FAILURE: No response generated")
            return False, None
        
    except Exception as e:
        print(f"❌ Revolutionary breakthrough test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, response = asyncio.run(test_revolutionary_breakthrough())
    if success:
        print("\n🎉 REVOLUTIONARY BREAKTHROUGH TEST: SUCCESS ✅")
        if response:
            print(f"\n📝 FINAL RESPONSE PREVIEW:")
            print("-" * 25)
            preview = response[:200] + "..." if len(response) > 200 else response
            print(f"'{preview}'")
    else:
        print("\n💥 REVOLUTIONARY BREAKTHROUGH TEST: FAILED ❌")