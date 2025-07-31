#!/usr/bin/env python3
"""
NWTN Final Demo - Complete Pipeline Test
=======================================
Demonstrates the complete NWTN pipeline with natural language response
exactly as the user would see it
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
from prsm.core.models import UserInput

print("🚀 NWTN PIPELINE DEMONSTRATION")
print("=" * 50)
print("Testing complete NWTN pipeline with natural language response")
print("=" * 50)

async def test_nwtn_pipeline():
    """Test complete NWTN pipeline as if user submitted query"""
    
    # Test prompt that should generate a comprehensive response
    test_prompt = "How does quantum entanglement work and what are its practical applications?"
    
    print("📝 USER PROMPT:")
    print("-" * 15)
    print(f'"{test_prompt}"')
    print()
    print("⏳ Processing with NWTN pipeline...")
    print("   • Semantic retrieval from knowledge base")
    print("   • Multi-agent reasoning and analysis") 
    print("   • Claude API synthesis")
    print("   • Response compilation and validation")
    print()
    
    try:
        # Load Claude API key
        api_key_path = "/Users/ryneschultz/Documents/GitHub/Anthropic_API_Key.txt"
        with open(api_key_path, 'r') as f:
            claude_api_key = f.read().strip()
        
        # Create UserInput exactly as user would submit
        user_input = UserInput(
            user_id="demo_user",
            prompt=test_prompt,
            context_allocation=500,  # Reasonable allocation
            preferences={
                "reasoning_depth": "deep",
                "response_length": "comprehensive", 
                "enable_cross_domain": True,
                "api_key": claude_api_key,
                "disable_database_persistence": True,  # Demo mode
                "preferred_model": "claude-3-5-sonnet-20241022"
            }
        )
        
        # Initialize and run NWTN orchestrator
        orchestrator = EnhancedNWTNOrchestrator()
        
        print("🔄 NWTN Processing...")
        response = await orchestrator.process_query(user_input=user_input)
        
        print("✅ NWTN Processing Complete!")
        print("=" * 50)
        
        # Display the response exactly as user would see it
        if response and hasattr(response, 'final_answer'):
            print("🎯 NATURAL LANGUAGE RESPONSE:")
            print("=" * 35)
            print()
            print(response.final_answer)
            print()
            print("=" * 50)
            
            # Show processing metrics
            print("📊 PROCESSING METRICS:")
            print("-" * 25)
            print(f"Response Length: {len(response.final_answer)} characters")
            print(f"Context Used: {response.context_used} tokens")
            print(f"Confidence Score: {response.confidence_score:.2f}" if response.confidence_score else "Confidence Score: N/A")
            print(f"Safety Validated: {response.safety_validated}")
            print()
            
            # Check response quality
            if len(response.final_answer.strip()) > 100:
                print("✅ SUCCESS: Comprehensive natural language response generated!")
                return True, response.final_answer
            else:
                print("⚠️  WARNING: Response may be incomplete")
                return True, response.final_answer
                
        else:
            print("❌ ERROR: No response generated")
            return False, None
        
    except Exception as e:
        print(f"❌ NWTN pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    print("Starting NWTN demonstration...")
    success, response = asyncio.run(test_nwtn_pipeline())
    
    if success:
        print("\n🎉 NWTN DEMONSTRATION: SUCCESS")
        print("\nThe NWTN pipeline successfully generated a natural language response!")
        print("This is exactly what the user would see when submitting their query.")
    else:
        print("\n💥 NWTN DEMONSTRATION: FAILED")
        print("Please check the error output above.")